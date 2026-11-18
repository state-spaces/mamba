import os
from typing import Tuple, Optional

import torch
from torch import Tensor

import triton
import triton.language as tl


# Rotary phases are a running sum over the whole sequence, so a per-token error in tanh accumulates: with the
# hardware tanh.approx.f32 (rel. error ~5e-4) the angle is off by ~6e-4 rad after 32k tokens, 10x the fp32
# summation error. An accurate tanh keeps training, prefill and the decode step (same formula, fp32 state
# wrapped mod 2*pi) within ~5e-5 rad of the fp64 phase.
@triton.jit
def _tanh(x):
    return tl.sigmoid(2.0 * x) * 2.0 - 1.0


@triton.jit
def _sech2(x):
    t = _tanh(x)
    return 1.0 - t * t


# -----------------------------------------------------------------------------
# Forward kernel
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [2, 4, 8]
    ],
    key=["CHUNK_SIZE", "BLOCK_D", "HAS_INIT_STATE", "RETURN_OUTPUT_STATE", "IS_VARLEN"],
)
@triton.jit
def angle_dt_fwd_kernel(
    # Outputs
    OUT, OUTPUT_STATE,
    # Inputs
    ANGLE, DT, INIT_STATE, CU_SEQLENS,
    # Strides for OUT (batch, seqlen, nheads, dim)
    stride_out_batch, stride_out_seq, stride_out_head, stride_out_dim,
    # Strides for OUTPUT_STATE (num_sequences, nheads, dim)
    stride_output_state_seq, stride_output_state_head, stride_output_state_dim,
    # Strides for ANGLE (batch, seqlen, nheads, dim)
    stride_angle_batch, stride_angle_seq, stride_angle_head, stride_angle_dim,
    # Strides for DT (batch, nheads, seqlen)
    stride_dt_batch, stride_dt_head, stride_dt_seq,
    # Strides for INIT_STATE (num_sequences, nheads, dim)
    stride_init_seq, stride_init_head, stride_init_dim,
    # Stride for CU_SEQLENS
    stride_cu_seqlen,
    # Dimensions
    seqlen, dim,
    # Meta-parameters
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    RETURN_OUTPUT_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Handle varlen mode
    if IS_VARLEN:
        pid_seq = tl.program_id(2)
        seq_idx = pid_seq
        cu_seqlen_start = tl.load(CU_SEQLENS + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_end = tl.load(CU_SEQLENS + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seq_len = cu_seqlen_end - cu_seqlen_start
        seq_offset = cu_seqlen_start
    else:
        seq_idx = pid_b
        seq_len = seqlen
        seq_offset = 0

    nchunks = tl.cdiv(seq_len, CHUNK_SIZE)

    # Offset base pointers by batch and head
    ANGLE += pid_b * stride_angle_batch + pid_h * stride_angle_head + seq_offset * stride_angle_seq
    DT += pid_b * stride_dt_batch + pid_h * stride_dt_head + seq_offset * stride_dt_seq
    OUT += pid_b * stride_out_batch + pid_h * stride_out_head + seq_offset * stride_out_seq

    dim_range = tl.arange(0, BLOCK_D)
    dim_mask = dim_range < dim

    # Initialize state from init_state or zeros
    if HAS_INIT_STATE:
        init_ptrs = INIT_STATE + seq_idx * stride_init_seq + pid_h * stride_init_head + dim_range * stride_init_dim
        state = tl.load(init_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_D,), dtype=tl.float32)

    PI = 3.141592653589793
    TWO_PI = 2 * PI

    for chunk_idx in range(nchunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        seq_range = tl.arange(0, CHUNK_SIZE)
        seq_mask = (chunk_start + seq_range) < seq_len

        # Load angle (CHUNK_SIZE, BLOCK_D)
        angle_ptrs = ANGLE + (chunk_start + seq_range[:, None]) * stride_angle_seq + dim_range[None, :] * stride_angle_dim
        angle_vals = tl.load(angle_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
        angle_vals = _tanh(angle_vals) * PI

        # Load dt (CHUNK_SIZE,)
        dt_ptrs = DT + (chunk_start + seq_range) * stride_dt_seq
        dt_vals = tl.load(dt_ptrs, mask=seq_mask, other=0.0).to(tl.float32)

        # Compute vals = angle * dt
        vals = angle_vals * dt_vals[:, None]

        # Cumsum within chunk + add state from previous chunks
        chunk_cumsum = tl.cumsum(vals, axis=0)
        out_vals = chunk_cumsum + state[None, :]

        # Apply mod 2*pi for rotary angle normalization
        out_vals = out_vals - TWO_PI * tl.floor(out_vals / TWO_PI)

        # Store output
        out_ptrs = OUT + (chunk_start + seq_range[:, None]) * stride_out_seq + dim_range[None, :] * stride_out_dim
        tl.store(out_ptrs, out_vals, mask=seq_mask[:, None] & dim_mask[None, :])

        # Update state: add chunk sum and apply mod 2*pi
        chunk_sum = tl.sum(vals, axis=0)
        state = state + chunk_sum
        state = state - TWO_PI * tl.floor(state / TWO_PI)

    # Store final state if requested
    if RETURN_OUTPUT_STATE:
        output_state_ptrs = OUTPUT_STATE + seq_idx * stride_output_state_seq + pid_h * stride_output_state_head + dim_range * stride_output_state_dim
        tl.store(output_state_ptrs, state, mask=dim_mask)


def angle_dt_fwd(
    angle: Tensor,
    dt: Tensor,
    init_state: Optional[Tensor] = None,
    chunk_size: int = 64,
    return_output_state: bool = False,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Forward pass for angle * dt cumsum.

    Args:
        angle: Angle tensor             (batch, seqlen, nheads, dim)
        dt: Time delta tensor           (batch, nheads, seqlen)
        init_state: Initial state       (num_sequences, nheads, dim) or None
        chunk_size: Chunk size for chunked computation
        return_output_state: Whether to return final state
        cu_seqlens: Cumulative sequence lengths (num_sequences + 1,) for varlen mode

    Returns:
        If return_output_state=False:
            out: Cumulative output      (batch, seqlen, nheads, dim)
        If return_output_state=True:
            Tuple of:
                out: Cumulative output      (batch, seqlen, nheads, dim)
                output_state: Final state   (num_sequences, nheads, dim)
    """
    batch, seqlen, nheads, dim = angle.shape
    is_varlen = cu_seqlens is not None
    if is_varlen and _SEQ_PARALLEL:
        return _angle_dt_fwd_seq_parallel(angle, dt, init_state=init_state, chunk_size=chunk_size,
                                          return_output_state=return_output_state, cu_seqlens=cu_seqlens)

    # Determine number of sequences
    if is_varlen:
        assert batch == 1, "Varlen mode requires batch=1"
        num_sequences = cu_seqlens.shape[0] - 1
    else:
        num_sequences = batch

    assert dt.shape == (batch, nheads, seqlen), f"dt shape mismatch: {dt.shape}"
    if init_state is not None:
        assert init_state.shape == (num_sequences, nheads, dim), f"init_state shape mismatch: {init_state.shape}"

    out = torch.empty_like(angle)
    BLOCK_D = triton.next_power_of_2(dim)

    # Handle None init_state for kernel
    HAS_INIT_STATE = init_state is not None
    if not HAS_INIT_STATE:
        init_state = angle  # dummy, won't be accessed
        stride_init = (0, 0, 0)
    else:
        stride_init = init_state.stride()

    # Handle output_state
    if return_output_state:
        output_state = torch.empty(num_sequences, nheads, dim, device=angle.device, dtype=angle.dtype)
        stride_output_state = output_state.stride()
    else:
        output_state = out  # dummy, won't be accessed
        stride_output_state = (0, 0, 0)

    # Handle cu_seqlens
    if cu_seqlens is not None:
        stride_cu_seqlen = cu_seqlens.stride(0)
    else:
        cu_seqlens = angle  # dummy, won't be accessed
        stride_cu_seqlen = 0

    # Grid setup
    if is_varlen:
        grid = (nheads, batch, num_sequences)
    else:
        grid = (nheads, batch)

    angle_dt_fwd_kernel[grid](
        out, output_state,
        angle, dt, init_state, cu_seqlens,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        stride_output_state[0], stride_output_state[1], stride_output_state[2],
        angle.stride(0), angle.stride(1), angle.stride(2), angle.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2),
        stride_init[0], stride_init[1], stride_init[2],
        stride_cu_seqlen,
        seqlen, dim,
        CHUNK_SIZE=chunk_size,
        BLOCK_D=BLOCK_D,
        HAS_INIT_STATE=HAS_INIT_STATE,
        RETURN_OUTPUT_STATE=return_output_state,
        IS_VARLEN=is_varlen,
    )

    if return_output_state:
        return out, output_state
    return out


# -----------------------------------------------------------------------------
# Backward kernel
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [2, 4, 8]
    ],
    key=["CHUNK_SIZE", "BLOCK_D", "HAS_INIT_STATE", "HAS_GRAD_OUTPUT_STATE", "IS_VARLEN"],
)
@triton.jit
def angle_dt_bwd_kernel(
    # Outputs
    GRAD_ANGLE, GRAD_DT, GRAD_INIT_STATE,
    # Inputs
    GRAD_OUT, GRAD_OUTPUT_STATE, ANGLE, DT, CU_SEQLENS,
    # Strides for GRAD_ANGLE (batch, seqlen, nheads, dim)
    stride_grad_angle_batch, stride_grad_angle_seq, stride_grad_angle_head, stride_grad_angle_dim,
    # Strides for GRAD_DT (batch, nheads, seqlen)
    stride_grad_dt_batch, stride_grad_dt_head, stride_grad_dt_seq,
    # Strides for GRAD_INIT_STATE (num_sequences, nheads, dim)
    stride_grad_init_seq, stride_grad_init_head, stride_grad_init_dim,
    # Strides for GRAD_OUT (batch, seqlen, nheads, dim)
    stride_grad_out_batch, stride_grad_out_seq, stride_grad_out_head, stride_grad_out_dim,
    # Strides for GRAD_OUTPUT_STATE (num_sequences, nheads, dim)
    stride_grad_output_state_seq, stride_grad_output_state_head, stride_grad_output_state_dim,
    # Strides for ANGLE (batch, seqlen, nheads, dim)
    stride_angle_batch, stride_angle_seq, stride_angle_head, stride_angle_dim,
    # Strides for DT (batch, nheads, seqlen)
    stride_dt_batch, stride_dt_head, stride_dt_seq,
    # Stride for CU_SEQLENS
    stride_cu_seqlen,
    # Dimensions
    seqlen, dim,
    # Meta-parameters
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    HAS_GRAD_OUTPUT_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Handle varlen mode
    if IS_VARLEN:
        pid_seq = tl.program_id(2)
        seq_idx = pid_seq
        cu_seqlen_start = tl.load(CU_SEQLENS + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_end = tl.load(CU_SEQLENS + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seq_len = cu_seqlen_end - cu_seqlen_start
        seq_offset = cu_seqlen_start
    else:
        seq_idx = pid_b
        seq_len = seqlen
        seq_offset = 0

    nchunks = tl.cdiv(seq_len, CHUNK_SIZE)

    # Offset base pointers by batch and head
    GRAD_ANGLE += pid_b * stride_grad_angle_batch + pid_h * stride_grad_angle_head + seq_offset * stride_grad_angle_seq
    GRAD_DT += pid_b * stride_grad_dt_batch + pid_h * stride_grad_dt_head + seq_offset * stride_grad_dt_seq
    GRAD_OUT += pid_b * stride_grad_out_batch + pid_h * stride_grad_out_head + seq_offset * stride_grad_out_seq
    ANGLE += pid_b * stride_angle_batch + pid_h * stride_angle_head + seq_offset * stride_angle_seq
    DT += pid_b * stride_dt_batch + pid_h * stride_dt_head + seq_offset * stride_dt_seq

    dim_range = tl.arange(0, BLOCK_D)
    dim_mask = dim_range < dim
    PI = 3.141592653589793

    # Initialize gradient state from grad_output_state or zeros
    if HAS_GRAD_OUTPUT_STATE:
        grad_output_state_ptrs = GRAD_OUTPUT_STATE + seq_idx * stride_grad_output_state_seq + pid_h * stride_grad_output_state_head + dim_range * stride_grad_output_state_dim
        grad_state = tl.load(grad_output_state_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
    else:
        grad_state = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Loop in reverse: derivative of cumsum is reverse cumsum
    for chunk_idx in range(nchunks - 1, -1, -1):
        chunk_start = chunk_idx * CHUNK_SIZE
        seq_range = tl.arange(0, CHUNK_SIZE)
        seq_mask = (chunk_start + seq_range) < seq_len

        # Load grad_out (CHUNK_SIZE, BLOCK_D)
        grad_out_ptrs = GRAD_OUT + (chunk_start + seq_range[:, None]) * stride_grad_out_seq + dim_range[None, :] * stride_grad_out_dim
        grad_out_vals = tl.load(grad_out_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)

        # Reverse cumsum within chunk: rev_cumsum = total - cumsum + x
        # But we need to handle the mask properly for partial chunks
        chunk_sum = tl.sum(grad_out_vals, axis=0)
        fwd_cumsum = tl.cumsum(grad_out_vals, axis=0)
        rev_cumsum = chunk_sum[None, :] - fwd_cumsum + grad_out_vals

        # Add gradient from future chunks
        grad_vals = rev_cumsum + grad_state[None, :]

        # Load angle and dt
        angle_ptrs = ANGLE + (chunk_start + seq_range[:, None]) * stride_angle_seq + dim_range[None, :] * stride_angle_dim
        pretanh_angle_vals = tl.load(angle_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
        angle_vals = _tanh(pretanh_angle_vals) * PI

        dt_ptrs = DT + (chunk_start + seq_range) * stride_dt_seq
        dt_vals = tl.load(dt_ptrs, mask=seq_mask, other=0.0).to(tl.float32)

        # Compute gradients: out = angle * dt
        grad_angle_vals = grad_vals * dt_vals[:, None] * PI * _sech2(pretanh_angle_vals)
        grad_dt_vals = tl.sum(grad_vals * angle_vals, axis=1)

        # Store gradients
        grad_angle_ptrs = GRAD_ANGLE + (chunk_start + seq_range[:, None]) * stride_grad_angle_seq + dim_range[None, :] * stride_grad_angle_dim
        tl.store(grad_angle_ptrs, grad_angle_vals, mask=seq_mask[:, None] & dim_mask[None, :])

        grad_dt_ptrs = GRAD_DT + (chunk_start + seq_range) * stride_grad_dt_seq
        tl.store(grad_dt_ptrs, grad_dt_vals, mask=seq_mask)

        # Update state for previous chunk
        grad_state = grad_state + chunk_sum

    # Store gradient for init_state if provided
    if HAS_INIT_STATE:
        grad_init_ptrs = GRAD_INIT_STATE + seq_idx * stride_grad_init_seq + pid_h * stride_grad_init_head + dim_range * stride_grad_init_dim
        tl.store(grad_init_ptrs, grad_state, mask=dim_mask)


def angle_dt_bwd(
    grad_out: Tensor,
    angle: Tensor,
    dt: Tensor,
    has_init_state: bool = False,
    chunk_size: int = 64,
    grad_output_state: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Backward pass for angle * dt cumsum.

    Args:
        grad_out: Gradient of output         (batch, seqlen, nheads, dim)
        angle: Angle tensor                  (batch, seqlen, nheads, dim)
        dt: Time delta tensor                (batch, nheads, seqlen)
        has_init_state: Whether init_state was provided in forward
        chunk_size: Chunk size for chunked computation
        grad_output_state: Gradient of output state (num_sequences, nheads, dim) or None
        cu_seqlens: Cumulative sequence lengths (num_sequences + 1,) for varlen mode

    Returns:
        grad_angle: Gradient for angle       (batch, seqlen, nheads, dim)
        grad_dt: Gradient for dt             (batch, nheads, seqlen)
        grad_init_state: Gradient for init_state (num_sequences, nheads, dim) or None
    """
    batch, seqlen, nheads, dim = angle.shape
    is_varlen = cu_seqlens is not None
    if is_varlen and _SEQ_PARALLEL:
        return _angle_dt_bwd_seq_parallel(grad_out, angle, dt, has_init_state=has_init_state, chunk_size=chunk_size,
                                          grad_output_state=grad_output_state, cu_seqlens=cu_seqlens)

    # Determine number of sequences
    if is_varlen:
        assert batch == 1, "Varlen mode requires batch=1"
        num_sequences = cu_seqlens.shape[0] - 1
    else:
        num_sequences = batch

    grad_angle = torch.empty_like(angle)
    grad_dt = torch.empty_like(dt)
    BLOCK_D = triton.next_power_of_2(dim)

    # Handle init_state gradient
    if has_init_state:
        grad_init_state = torch.empty(num_sequences, nheads, dim, device=angle.device, dtype=torch.float32)
        stride_grad_init = grad_init_state.stride()
    else:
        grad_init_state = None
        stride_grad_init = (0, 0, 0)
        grad_init_dummy = grad_angle  # dummy pointer

    # Handle grad_output_state
    HAS_GRAD_OUTPUT_STATE = grad_output_state is not None
    if not HAS_GRAD_OUTPUT_STATE:
        grad_output_state = grad_angle  # dummy, won't be accessed
        stride_grad_output_state = (0, 0, 0)
    else:
        stride_grad_output_state = grad_output_state.stride()

    # Handle cu_seqlens
    if cu_seqlens is not None:
        stride_cu_seqlen = cu_seqlens.stride(0)
    else:
        cu_seqlens = angle  # dummy, won't be accessed
        stride_cu_seqlen = 0

    # Grid setup
    if is_varlen:
        grid = (nheads, batch, num_sequences)
    else:
        grid = (nheads, batch)

    angle_dt_bwd_kernel[grid](
        grad_angle, grad_dt, grad_init_state if has_init_state else grad_init_dummy,
        grad_out, grad_output_state, angle, dt, cu_seqlens,
        grad_angle.stride(0), grad_angle.stride(1), grad_angle.stride(2), grad_angle.stride(3),
        grad_dt.stride(0), grad_dt.stride(1), grad_dt.stride(2),
        stride_grad_init[0], stride_grad_init[1], stride_grad_init[2],
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
        stride_grad_output_state[0], stride_grad_output_state[1], stride_grad_output_state[2],
        angle.stride(0), angle.stride(1), angle.stride(2), angle.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2),
        stride_cu_seqlen,
        seqlen, dim,
        CHUNK_SIZE=chunk_size,
        BLOCK_D=BLOCK_D,
        HAS_INIT_STATE=has_init_state,
        HAS_GRAD_OUTPUT_STATE=HAS_GRAD_OUTPUT_STATE,
        IS_VARLEN=is_varlen,
    )
    return grad_angle, grad_dt, grad_init_state

# -----------------------------------------------------------------------------
# Sequence-parallel varlen path
# -----------------------------------------------------------------------------
# The kernels above run one program per (head, segment) and walk the segment in
# CHUNK_SIZE steps with a loop-carried state: a long segment is a long serial
# chain on a handful of SMs (H=12 per TP rank, 2x32k segments -> 24 programs
# walking 2048 chunks each). The varlen path below cuts every segment into
# blocks of BLK_CHUNKS chunks: pass 1 reduces each block in parallel, pass 2
# scans the few block totals per (head, segment), pass 3 replays each block in
# parallel from its offset with the same per-chunk loop. Same numerics up to
# fp32 reassociation of the phase sum (both paths sit at the same distance from
# an fp64 reference, which tanh.approx dominates). GH200, H=12, S=65536:
# 2x32768 fwd 1.89 -> 0.32 ms, bwd 2.87 -> 0.32 ms.
# M3_ANGLE_SERIAL=1 forces the serial kernels.
PI_C = tl.constexpr(3.141592653589793)
_SEQ_PARALLEL = os.environ.get("M3_ANGLE_SERIAL", "0") != "1"


@triton.jit
def _block_owner(CU, b, ns, BLK_TOKENS: tl.constexpr):
    """Owner of global block slot b: sequence i owns slots
    [cu[i]//T + i, cu[i]//T + i + cdiv(len_i, T)) (strictly increasing starts,
    so binary search). Returns (seq or -1 for a padding slot, local block)."""
    lo = 0
    hi = ns - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tl.load(CU + mid) // BLK_TOKENS + mid <= b:
            lo = mid
        else:
            hi = mid - 1
    s0 = tl.load(CU + lo)
    local = b - (s0 // BLK_TOKENS + lo)
    n = tl.cdiv(tl.load(CU + lo + 1) - s0, BLK_TOKENS)
    return tl.where(local < n, lo, -1), local


@triton.jit
def _fwd_block_kernel(OUT, BLOCK_TOT, OFFSET, ANGLE, DT, CU, ns,
                      s_ang_seq, s_ang_head, s_dt_head, s_dt_seq, nblk,
                      dim, PASS: tl.constexpr, CHUNK: tl.constexpr, BLK_CHUNKS: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    b = tl.program_id(1)
    seq, local = _block_owner(CU, b, ns, CHUNK * BLK_CHUNKS)
    if seq < 0:
        return
    s0 = tl.load(CU + seq)
    seq_len = tl.load(CU + seq + 1) - s0
    blk_start = local * (CHUNK * BLK_CHUNKS)
    d = tl.arange(0, BLOCK_D)
    dm = d < dim
    TWO_PI_C = 2 * PI_C
    if PASS == 1:
        state = tl.zeros((BLOCK_D,), dtype=tl.float32)
    else:
        state = tl.load(OFFSET + (h * nblk + b) * BLOCK_D + d)
    r = tl.arange(0, CHUNK)
    for c in range(BLK_CHUNKS):
        t0 = blk_start + c * CHUNK
        sm = (t0 + r) < seq_len
        tok = s0 + t0 + r
        a = tl.load(ANGLE + tok[:, None] * s_ang_seq + h * s_ang_head + d[None, :], mask=sm[:, None] & dm[None, :], other=0.0)
        a = _tanh(a.to(tl.float32)) * PI_C
        dtv = tl.load(DT + h * s_dt_head + tok * s_dt_seq, mask=sm, other=0.0).to(tl.float32)
        v = a * dtv[:, None]
        if PASS == 3:
            o = tl.cumsum(v, axis=0) + state[None, :]
            o = o - TWO_PI_C * tl.floor(o / TWO_PI_C)
            tl.store(OUT + tok[:, None] * s_ang_seq + h * s_ang_head + d[None, :], o, mask=sm[:, None] & dm[None, :])
        state = state + tl.sum(v, axis=0)
        state = state - TWO_PI_C * tl.floor(state / TWO_PI_C)
    if PASS == 1:
        tl.store(BLOCK_TOT + (h * nblk + b) * BLOCK_D + d, state)


@triton.jit
def _scan_blocks_kernel(OFFSET, FINAL, BLOCK_TOT, INIT, CU, nblk, nheads, dim,
                        HAS_INIT: tl.constexpr, REVERSE: tl.constexpr, MOD: tl.constexpr, BLOCK_D: tl.constexpr,
                        BLK_TOKENS: tl.constexpr):
    """Per (head, segment): exclusive scan of the block totals (forward with
    mod 2*pi, or reverse plain sum for the backward)."""
    h = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dm = d < dim
    TWO_PI_C = 2 * PI_C
    if HAS_INIT:
        state = tl.load(INIT + (s * nheads + h) * dim + d, mask=dm, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_D,), dtype=tl.float32)
    s0 = tl.load(CU + s)
    st = s0 // BLK_TOKENS + s
    n = tl.cdiv(tl.load(CU + s + 1) - s0, BLK_TOKENS)
    for i in range(n):
        j = st + (n - 1 - i) if REVERSE else st + i
        tl.store(OFFSET + (h * nblk + j) * BLOCK_D + d, state)
        state = state + tl.load(BLOCK_TOT + (h * nblk + j) * BLOCK_D + d)
        if MOD:
            state = state - TWO_PI_C * tl.floor(state / TWO_PI_C)
    tl.store(FINAL + (s * nheads + h) * dim + d, state, mask=dm)


@triton.jit
def _bwd_block_kernel(GRAD_ANGLE, GRAD_DT, BLOCK_TOT, OFFSET, GRAD_OUT, ANGLE, DT, CU, ns,
                      s_ang_seq, s_ang_head, s_dt_head, s_dt_seq, nblk,
                      dim, PASS: tl.constexpr, CHUNK: tl.constexpr, BLK_CHUNKS: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    b = tl.program_id(1)
    seq, local = _block_owner(CU, b, ns, CHUNK * BLK_CHUNKS)
    if seq < 0:
        return
    s0 = tl.load(CU + seq)
    seq_len = tl.load(CU + seq + 1) - s0
    blk_start = local * (CHUNK * BLK_CHUNKS)
    d = tl.arange(0, BLOCK_D)
    dm = d < dim
    r = tl.arange(0, CHUNK)
    if PASS == 1:
        tot = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for c in range(BLK_CHUNKS):
            t0 = blk_start + c * CHUNK
            sm = (t0 + r) < seq_len
            tok = s0 + t0 + r
            g = tl.load(GRAD_OUT + tok[:, None] * s_ang_seq + h * s_ang_head + d[None, :], mask=sm[:, None] & dm[None, :], other=0.0)
            tot = tot + tl.sum(g.to(tl.float32), axis=0)
        tl.store(BLOCK_TOT + (h * nblk + b) * BLOCK_D + d, tot)
    else:
        gstate = tl.load(OFFSET + (h * nblk + b) * BLOCK_D + d)
        for ci in range(BLK_CHUNKS):
            c = BLK_CHUNKS - 1 - ci
            t0 = blk_start + c * CHUNK
            sm = (t0 + r) < seq_len
            tok = s0 + t0 + r
            m2 = sm[:, None] & dm[None, :]
            g = tl.load(GRAD_OUT + tok[:, None] * s_ang_seq + h * s_ang_head + d[None, :], mask=m2, other=0.0).to(tl.float32)
            csum = tl.sum(g, axis=0)
            rev = csum[None, :] - tl.cumsum(g, axis=0) + g
            gv = rev + gstate[None, :]
            pre = tl.load(ANGLE + tok[:, None] * s_ang_seq + h * s_ang_head + d[None, :], mask=m2, other=0.0).to(tl.float32)
            a = _tanh(pre) * PI_C
            dtv = tl.load(DT + h * s_dt_head + tok * s_dt_seq, mask=sm, other=0.0).to(tl.float32)
            tl.store(GRAD_ANGLE + tok[:, None] * s_ang_seq + h * s_ang_head + d[None, :], gv * dtv[:, None] * PI_C * _sech2(pre), mask=m2)
            tl.store(GRAD_DT + h * s_dt_head + tok * s_dt_seq, tl.sum(gv * a, axis=1), mask=sm)
            gstate = gstate + csum


def _prep(angle, dt, cu_seqlens, chunk_size, blk_chunks):
    assert angle.shape[0] == 1 and cu_seqlens is not None, "varlen layout only"
    angle = angle.contiguous(); dt = dt.contiguous()
    _, seqlen, nheads, dim = angle.shape
    cu = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    nblk = seqlen // (chunk_size * blk_chunks) + cu.numel() - 1
    return angle, dt, cu, seqlen, nheads, dim, nblk


def _angle_dt_fwd_seq_parallel(angle: Tensor, dt: Tensor, init_state: Optional[Tensor] = None, chunk_size: int = 16,
                     return_output_state: bool = False, cu_seqlens: Optional[Tensor] = None, blk_chunks: int = 64):
    angle, dt, cu, seqlen, nheads, dim, nblk = _prep(angle, dt, cu_seqlens, chunk_size, blk_chunks)
    ns = cu.numel() - 1
    BD = triton.next_power_of_2(dim)
    out = torch.empty_like(angle)
    tot = torch.empty(nheads, nblk, BD, device=angle.device, dtype=torch.float32)
    off = torch.empty_like(tot)
    final = torch.empty(ns, nheads, dim, device=angle.device, dtype=angle.dtype)
    common = dict(s_ang_seq=angle.stride(1), s_ang_head=angle.stride(2), s_dt_head=dt.stride(1), s_dt_seq=dt.stride(2),
                  nblk=nblk, dim=dim, CHUNK=chunk_size, BLK_CHUNKS=blk_chunks, BLOCK_D=BD)
    _fwd_block_kernel[(nheads, nblk)](out, tot, off, angle, dt, cu, ns, PASS=1, **common)
    init = init_state.contiguous().float() if init_state is not None else final
    _scan_blocks_kernel[(nheads, ns)](off, final, tot, init, cu, nblk, nheads, dim,
                                      HAS_INIT=init_state is not None, REVERSE=False, MOD=True, BLOCK_D=BD,
                                      BLK_TOKENS=chunk_size * blk_chunks)
    _fwd_block_kernel[(nheads, nblk)](out, tot, off, angle, dt, cu, ns, PASS=3, **common)
    return (out, final) if return_output_state else out


def _angle_dt_bwd_seq_parallel(grad_out: Tensor, angle: Tensor, dt: Tensor, has_init_state: bool = False, chunk_size: int = 16,
                     grad_output_state: Optional[Tensor] = None, cu_seqlens: Optional[Tensor] = None, blk_chunks: int = 64):
    angle, dt, cu, seqlen, nheads, dim, nblk = _prep(angle, dt, cu_seqlens, chunk_size, blk_chunks)
    grad_out = grad_out.contiguous()
    ns = cu.numel() - 1
    BD = triton.next_power_of_2(dim)
    grad_angle = torch.empty_like(angle)
    grad_dt = torch.empty_like(dt)
    tot = torch.empty(nheads, nblk, BD, device=angle.device, dtype=torch.float32)
    off = torch.empty_like(tot)
    ginit = torch.empty(ns, nheads, dim, device=angle.device, dtype=torch.float32)
    common = dict(s_ang_seq=angle.stride(1), s_ang_head=angle.stride(2), s_dt_head=dt.stride(1), s_dt_seq=dt.stride(2),
                  nblk=nblk, dim=dim, CHUNK=chunk_size, BLK_CHUNKS=blk_chunks, BLOCK_D=BD)
    _bwd_block_kernel[(nheads, nblk)](grad_angle, grad_dt, tot, off, grad_out, angle, dt, cu, ns, PASS=1, **common)
    gos = grad_output_state.contiguous().float() if grad_output_state is not None else ginit
    _scan_blocks_kernel[(nheads, ns)](off, ginit, tot, gos, cu, nblk, nheads, dim,
                                      HAS_INIT=grad_output_state is not None, REVERSE=True, MOD=False, BLOCK_D=BD,
                                      BLK_TOKENS=chunk_size * blk_chunks)
    _bwd_block_kernel[(nheads, nblk)](grad_angle, grad_dt, tot, off, grad_out, angle, dt, cu, ns, PASS=3, **common)
    return grad_angle, grad_dt, (ginit if has_init_state else None)
