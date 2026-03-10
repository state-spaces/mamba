# Copyright (c) 2024, Tri Dao, Albert Gu.
# Mamba-3 chunk scan Triton kernel (forward pass).
#
# Extends Mamba-2's _chunk_scan_fwd_kernel to support the trapezoidal
# discretization used in Mamba-3:
#
#   Y_off[m] = C[m] @ prev_states * exp(dA_cs[m])          (inter-chunk, same as Mamba-2)
#
#   Y_diag[m] = sum_k L[m,k] * gamma_k * CB[m,k] * x[k]   (intra-chunk current term)
#             + sum_k L[m,k] * beta_k  * CB_s[m,k] * x_s[k]  (intra-chunk lookback term)
#
# Where L[m,k] = exp(dA_cs[m] - dA_cs[k]) is the causal decay matrix.
# gamma replaces dt in Mamba-2's diagonal accumulation.
# The lookback term is optional (controlled by HAS_LOOKBACK constexpr).

import math
from packaging import version

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from mamba_ssm.utils.determinism import autotune_configs

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


@triton.autotune(
    configs=autotune_configs([
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=2),
    ]),
    key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'],
)
@triton.jit
def _mamba3_chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr,
    dA_cumsum_ptr, gamma_ptr, seq_idx_ptr,
    C_ptr, prev_states_ptr, D_ptr,
    # Lookback pointers (may be null)
    beta_ptr, cb_shifted_ptr, x_shifted_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # cb strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    # x strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    # z strides
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    # out strides
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    # dA_cumsum strides (chunked layout: batch, chunk, head, csize in memory)
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # gamma strides (same chunked layout)
    stride_gamma_batch, stride_gamma_chunk, stride_gamma_head, stride_gamma_csize,
    # seq_idx strides
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # C strides
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    # prev_states strides
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    # D stride
    stride_D_head,
    # beta strides (same chunked layout)
    stride_beta_batch, stride_beta_chunk, stride_beta_head, stride_beta_csize,
    # cb_shifted strides (same as cb)
    stride_cbs_batch, stride_cbs_chunk, stride_cbs_head, stride_cbs_csize_m, stride_cbs_csize_k,
    # x_shifted strides (same as x)
    stride_xs_batch, stride_xs_seqlen, stride_xs_head, stride_xs_hdim,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    HAS_LOOKBACK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    # Advance base pointers to this batch, chunk, head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    gamma_ptr += pid_b * stride_gamma_batch + pid_c * stride_gamma_chunk + pid_h * stride_gamma_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
    if HAS_LOOKBACK:
        beta_ptr += pid_b * stride_beta_batch + pid_c * stride_beta_chunk + pid_h * stride_beta_head
        cb_shifted_ptr += pid_b * stride_cbs_batch + pid_c * stride_cbs_chunk + (pid_h // nheads_ngroups_ratio) * stride_cbs_head
        x_shifted_ptr += pid_b * stride_xs_batch + pid_c * chunk_size * stride_xs_seqlen + pid_h * stride_xs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # === Phase 1: Off-diagonal (inter-chunk) contribution ===
    # Y_off[m] = C[m] @ prev_states * exp(dA_cs[m])
    # This is identical to Mamba-2.
    if IS_TRITON_22 or pid_c > -1:
        offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
        prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        if BLOCK_SIZE_DSTATE <= 128:
            C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
            prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate - k), other=0.0)
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
                acc += tl.dot(C, prev_states)
                C_ptrs += BLOCK_SIZE_K
                prev_states_ptrs += BLOCK_SIZE_K
            acc *= scale_m[:, None]

    # === Phase 2: Diagonal (intra-chunk) contribution ===
    # Current term: sum_k L[m,k] * gamma_k * CB[m,k] * x[k]
    # Lookback term: sum_k L[m,k] * beta_k * CB_shifted[m,k] * x_shifted[k]
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    gamma_ptrs = gamma_ptr + offs_k * stride_gamma_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_LOOKBACK:
        cbs_ptrs = cb_shifted_ptr + (offs_m[:, None] * stride_cbs_csize_m + offs_k[None, :] * stride_cbs_csize_k)
        xs_ptrs = x_shifted_ptr + (offs_k[:, None] * stride_xs_seqlen + offs_n[None, :] * stride_xs_hdim)
        beta_ptrs = beta_ptr + offs_k * stride_beta_csize

    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        # Load CB values and compute decay
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        # If there's seq_idx, CB was already zeroed for cross-doc pairs by _bmm_chunk_fwd.
        # L[m,k] = exp(dA_cs_m - dA_cs_k), clamped to avoid overflow
        cb *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_k[None, :]), 0.0))

        # Scale by gamma (Mamba-3) instead of dt (Mamba-2)
        gamma_k = tl.load(gamma_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= gamma_k

        # Causal mask
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)

        x = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim), other=0.0)
        acc += tl.dot(cb, x)

        # Lookback term
        if HAS_LOOKBACK:
            cbs = tl.load(cbs_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.float32)
            cbs *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_k[None, :]), 0.0))

            beta_k = tl.load(beta_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
            cbs *= beta_k

            if IS_CAUSAL:
                cbs = tl.where(mask, cbs, 0.0)
            cbs = cbs.to(x_ptr.dtype.element_ty)

            xs = tl.load(xs_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim), other=0.0)
            acc += tl.dot(cbs, xs)

            # Advance lookback pointers
            cbs_ptrs += BLOCK_SIZE_K * stride_cbs_csize_k
            xs_ptrs += BLOCK_SIZE_K * stride_xs_seqlen
            beta_ptrs += BLOCK_SIZE_K * stride_beta_csize

        # Advance pointers
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        gamma_ptrs += BLOCK_SIZE_K * stride_gamma_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # === D skip connection ===
    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(
            x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0
        ).to(tl.float32)
        acc += x_residual * D

    # === Z gating ===
    if HAS_Z:
        out_x_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
        tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

        z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    # Store output
    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))


def _mamba3_chunk_scan_fwd(CB, x, dt, dA_cumsum, gamma, C, prev_states,
                            D=None, z=None, beta=None, CB_shifted=None,
                            x_shifted=None, seq_idx=None):
    """
    Compute chunked scan output for Mamba-3 SSD.

    Arguments:
        CB: (batch, nchunks, ngroups, chunk_size, chunk_size) -- C^T @ B per chunk
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size) -- NOT directly used by kernel (kept for API compat)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        gamma: (batch, nheads, nchunks, chunk_size) -- current term weight (replaces dt in Mamba-2)
        C: (batch, seqlen, ngroups, dstate) -- for off-diagonal computation
        prev_states: (batch, nchunks, nheads, headdim, dstate) -- boundary states
        D: (nheads,) or (nheads, headdim) or None -- skip connection
        z: (batch, seqlen, nheads, headdim) or None -- gating
        beta: (batch, nheads, nchunks, chunk_size) or None -- lookback weight
        CB_shifted: (batch, nchunks, ngroups, chunk_size, chunk_size) or None -- C^T @ B_shifted
        x_shifted: (batch, seqlen, nheads, headdim) or None -- shifted x
        seq_idx: (batch, seqlen) or None -- document boundaries

    Returns:
        out: (batch, seqlen, nheads, headdim)
        out_x: (batch, seqlen, nheads, headdim) or None -- pre-gating output if z is present
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dA_cumsum.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert gamma.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

    has_lookback = beta is not None and CB_shifted is not None and x_shifted is not None
    if has_lookback:
        assert beta.shape == (batch, nheads, nchunks, chunk_size)
        assert CB_shifted.shape == CB.shape
        assert x_shifted.shape == x.shape

    # Allocate output
    out = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
    if z is not None:
        out_x = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
        assert out_x.stride() == out.stride()
    else:
        out_x = None

    grid = lambda META: (
        triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
        batch * nchunks,
        nheads,
    )

    z_strides = (z.stride(0), z.stride(1), z.stride(2), z.stride(3)) if z is not None else (0, 0, 0, 0)

    with torch.cuda.device(x.device.index):
        _mamba3_chunk_scan_fwd_kernel[grid](
            # Core data pointers
            CB, x, z, out, out_x,
            dA_cumsum, gamma, seq_idx,
            C, prev_states, D,
            # Lookback pointers (None if not used)
            beta, CB_shifted, x_shifted,
            # Dimensions
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            # CB strides
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(3), CB.stride(4),
            # x strides
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            # z strides
            z_strides[0], z_strides[1], z_strides[2], z_strides[3],
            # out strides
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            # dA_cumsum strides (note: tensor is (b, nheads, nchunks, chunk_size), kernel expects batch, chunk, head, csize)
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            # gamma strides
            gamma.stride(0), gamma.stride(2), gamma.stride(1), gamma.stride(3),
            # seq_idx strides
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            # C strides
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            # prev_states strides
            prev_states.stride(0), prev_states.stride(1), prev_states.stride(2), prev_states.stride(3), prev_states.stride(4),
            # D stride
            D.stride(0) if D is not None else 0,
            # beta strides
            *((beta.stride(0), beta.stride(2), beta.stride(1), beta.stride(3))
              if has_lookback else (0, 0, 0, 0)),
            # CB_shifted strides
            *((CB_shifted.stride(0), CB_shifted.stride(1), CB_shifted.stride(2), CB_shifted.stride(3), CB_shifted.stride(4))
              if has_lookback else (0, 0, 0, 0, 0)),
            # x_shifted strides
            *((x_shifted.stride(0), x_shifted.stride(1), x_shifted.stride(2), x_shifted.stride(3))
              if has_lookback else (0, 0, 0, 0)),
            # Constexpr meta-parameters
            True,  # IS_CAUSAL
            D is not None,  # HAS_D
            D.dim() == 2 if D is not None else True,  # D_HAS_HDIM
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            HAS_Z=z is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            HAS_LOOKBACK=has_lookback,
            IS_TRITON_22=TRITON_22,
        )

    return out, out_x
