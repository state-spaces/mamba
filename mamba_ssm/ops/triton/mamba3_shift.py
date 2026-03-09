# Copyright (c) 2024, Tri Dao, Albert Gu.
# Mamba-3 shift-by-1 Triton kernels with seq_idx masking.
#
# Simple memory-bound kernels for shifting tensors by 1 position along the
# sequence dimension, used for the trapezoidal lookback term in Mamba-3.
# Supports seq_idx masking to zero out shifted values at document boundaries.
#
# Works for any tensor with seqlen as dim 1 (e.g. B: (b,l,h,n) or x: (b,l,h,p)).
# Everything after dim 1 is treated as a flat dimension.

import torch
import triton
import triton.language as tl


# =============================================================================
# Forward kernel: shift by 1 position
# =============================================================================

@triton.jit
def _mamba3_shift_fwd_kernel(
    # Input/output pointers
    x_ptr, out_ptr, seq_idx_ptr, initial_ptr,
    # Dimensions
    batch, seqlen, flat_dim,
    # x strides
    stride_x_batch, stride_x_seqlen, stride_x_flat,
    # out strides
    stride_out_batch, stride_out_seqlen, stride_out_flat,
    # seq_idx strides
    stride_si_batch, stride_si_seqlen,
    # initial strides (batch, flat_dim) or scalar 0
    stride_init_batch, stride_init_flat,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Shift tensor by 1 along seqlen dimension.

    Grid: (batch, cdiv(seqlen, BLOCK_L), cdiv(flat_dim, BLOCK_D))

    out[:,0,:] = initial (or 0)
    out[:,t,:] = x[:,t-1,:] for t > 0
    if seq_idx: out[:,t,:] = 0 where seq_idx[:,t] != seq_idx[:,t-1]
    """
    pid_b = tl.program_id(axis=0)
    pid_l = tl.program_id(axis=1)
    pid_d = tl.program_id(axis=2)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    l_mask = offs_l < seqlen
    d_mask = offs_d < flat_dim
    mask = l_mask[:, None] & d_mask[None, :]

    # For t > 0: load from t-1
    # For t == 0: load initial or zero
    src_l = offs_l - 1  # source positions (t-1)
    src_valid = src_l >= 0

    # Load source values (from position t-1)
    x_base = pid_b * stride_x_batch
    src_ptrs = x_ptr + x_base + src_l[:, None] * stride_x_seqlen + offs_d[None, :] * stride_x_flat
    src_mask = mask & src_valid[:, None]
    vals = tl.load(src_ptrs, mask=src_mask, other=0.0)

    # Handle position 0: use initial if provided
    if HAS_INITIAL:
        init_ptrs = initial_ptr + pid_b * stride_init_batch + offs_d * stride_init_flat
        init_vals = tl.load(init_ptrs, mask=d_mask, other=0.0)
        # Broadcast init_vals to (1, BLOCK_D) and apply where src_l < 0
        is_pos_zero = (offs_l == 0)
        vals = tl.where(is_pos_zero[:, None] & d_mask[None, :], init_vals[None, :], vals)

    # seq_idx masking: zero out where document boundary crossed
    if HAS_SEQ_IDX:
        si_base = pid_b * stride_si_batch
        si_cur = tl.load(seq_idx_ptr + si_base + offs_l * stride_si_seqlen, mask=l_mask, other=-1)
        si_prev = tl.load(seq_idx_ptr + si_base + src_l * stride_si_seqlen, mask=l_mask & src_valid, other=-2)
        # Zero out where current != previous (document boundary) or position 0
        boundary = ~src_valid | (si_cur != si_prev)
        vals = tl.where(boundary[:, None], 0.0, vals)

    # Store output
    out_base = pid_b * stride_out_batch
    out_ptrs = out_ptr + out_base + offs_l[:, None] * stride_out_seqlen + offs_d[None, :] * stride_out_flat
    tl.store(out_ptrs, vals, mask=mask)


def _mamba3_shift_fwd(x, seq_idx=None, initial=None):
    """Shift tensor by 1 position along seqlen dim, with seq_idx masking.

    Args:
        x: (batch, seqlen, ...) -- any shape with seqlen as dim 1
        seq_idx: (batch, seqlen) -- document boundaries, or None
        initial: (batch, ...) value for position 0 (default: zero), or None

    Returns:
        x_shifted: same shape as x, shifted by 1
    """
    batch = x.shape[0]
    seqlen = x.shape[1]
    # Flatten everything after dim 1
    orig_shape = x.shape
    flat_dim = 1
    for s in x.shape[2:]:
        flat_dim *= s
    x_flat = x.reshape(batch, seqlen, flat_dim)

    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()

    out = torch.empty_like(x_flat)

    has_initial = initial is not None
    if has_initial:
        initial_flat = initial.reshape(batch, flat_dim)
        if not initial_flat.is_contiguous():
            initial_flat = initial_flat.contiguous()
    else:
        initial_flat = None

    BLOCK_L = min(triton.next_power_of_2(seqlen), 256)
    BLOCK_D = min(triton.next_power_of_2(flat_dim), 256)

    grid = (batch, triton.cdiv(seqlen, BLOCK_L), triton.cdiv(flat_dim, BLOCK_D))

    with torch.cuda.device(x.device.index):
        _mamba3_shift_fwd_kernel[grid](
            x_flat, out, seq_idx, initial_flat,
            batch, seqlen, flat_dim,
            x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            *((initial_flat.stride(0), initial_flat.stride(1)) if has_initial else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
            HAS_INITIAL=has_initial,
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D,
        )

    return out.reshape(orig_shape)


# =============================================================================
# Backward kernel: reverse shift
# =============================================================================

@triton.jit
def _mamba3_shift_bwd_kernel(
    # Input/output pointers
    dout_ptr, dx_ptr, seq_idx_ptr,
    # Dimensions
    batch, seqlen, flat_dim,
    # dout strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_flat,
    # dx strides
    stride_dx_batch, stride_dx_seqlen, stride_dx_flat,
    # seq_idx strides
    stride_si_batch, stride_si_seqlen,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Backward of shift: reverse the shift.

    Grid: (batch, cdiv(seqlen, BLOCK_L), cdiv(flat_dim, BLOCK_D))

    dx[:,t,:] = dout[:,t+1,:] for t < seqlen-1
    dx[:,seqlen-1,:] = 0
    if seq_idx: dx[:,t,:] = 0 where seq_idx[:,t+1] != seq_idx[:,t]
    """
    pid_b = tl.program_id(axis=0)
    pid_l = tl.program_id(axis=1)
    pid_d = tl.program_id(axis=2)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    l_mask = offs_l < seqlen
    d_mask = offs_d < flat_dim
    mask = l_mask[:, None] & d_mask[None, :]

    # For position t: gradient comes from dout at position t+1
    # (because forward: out[t+1] = x[t])
    dst_l = offs_l + 1  # where this position's value went to in forward
    dst_valid = dst_l < seqlen

    # Load gradient from position t+1
    dout_base = pid_b * stride_dout_batch
    dout_ptrs = dout_ptr + dout_base + dst_l[:, None] * stride_dout_seqlen + offs_d[None, :] * stride_dout_flat
    src_mask = mask & dst_valid[:, None]
    vals = tl.load(dout_ptrs, mask=src_mask, other=0.0)

    # seq_idx masking: zero out where forward would have zeroed
    if HAS_SEQ_IDX:
        si_base = pid_b * stride_si_batch
        si_cur = tl.load(seq_idx_ptr + si_base + offs_l * stride_si_seqlen, mask=l_mask, other=-1)
        si_next = tl.load(seq_idx_ptr + si_base + dst_l * stride_si_seqlen, mask=l_mask & dst_valid, other=-2)
        # In forward: out[t+1] = x[t] only if seq_idx[t+1] == seq_idx[t]
        # So gradient flows back only when seq_idx matches
        boundary = ~dst_valid | (si_next != si_cur)
        vals = tl.where(boundary[:, None], 0.0, vals)

    # Store dx
    dx_base = pid_b * stride_dx_batch
    dx_ptrs = dx_ptr + dx_base + offs_l[:, None] * stride_dx_seqlen + offs_d[None, :] * stride_dx_flat
    tl.store(dx_ptrs, vals, mask=mask)


def _mamba3_shift_bwd(dx_shifted, seq_idx=None):
    """Backward through shift: reverse the shift to get gradient w.r.t. input.

    Args:
        dx_shifted: (batch, seqlen, ...) -- gradient of shifted output
        seq_idx: (batch, seqlen) -- document boundaries, or None

    Returns:
        dx: same shape as dx_shifted, unshifted gradient
    """
    batch = dx_shifted.shape[0]
    seqlen = dx_shifted.shape[1]
    orig_shape = dx_shifted.shape
    flat_dim = 1
    for s in dx_shifted.shape[2:]:
        flat_dim *= s
    dout_flat = dx_shifted.reshape(batch, seqlen, flat_dim)

    if not dout_flat.is_contiguous():
        dout_flat = dout_flat.contiguous()

    dx = torch.empty_like(dout_flat)

    BLOCK_L = min(triton.next_power_of_2(seqlen), 256)
    BLOCK_D = min(triton.next_power_of_2(flat_dim), 256)

    grid = (batch, triton.cdiv(seqlen, BLOCK_L), triton.cdiv(flat_dim, BLOCK_D))

    with torch.cuda.device(dx_shifted.device.index):
        _mamba3_shift_bwd_kernel[grid](
            dout_flat, dx, seq_idx,
            batch, seqlen, flat_dim,
            dout_flat.stride(0), dout_flat.stride(1), dout_flat.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D,
        )

    return dx.reshape(orig_shape)
