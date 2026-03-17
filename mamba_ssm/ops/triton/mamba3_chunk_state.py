# Copyright (c) 2024, Tri Dao, Albert Gu.
# Mamba-3 chunk state Triton kernel (forward pass).
#
# Extends Mamba-2's _chunk_state_fwd_kernel to support the trapezoidal
# discretization used in Mamba-3:
#
#   states[c] = sum_t exp(dA_last - dA_t) * gamma_t * B_t outer x_t        (current term)
#             + sum_t exp(dA_last - dA_t) * beta_t * B_shifted_t outer x_shifted_t  (lookback term)
#
# The lookback term is optional (controlled by HAS_LOOKBACK constexpr).
# When gamma is None, falls back to dt scaling (Mamba-2 compatibility mode).

import torch
import triton
import triton.language as tl

from mamba_ssm.utils.determinism import autotune_configs


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
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _mamba3_chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, states_ptr,
    dt_ptr, dA_cumsum_ptr, gamma_ptr, seq_idx_ptr,
    # Lookback pointers (may be null)
    beta_ptr, b_shifted_ptr, x_shifted_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio,
    # x strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    # b strides
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    # states strides
    stride_states_batch, stride_states_chunk, stride_states_head,
    stride_states_hdim, stride_states_dstate,
    # dt strides (chunked layout: b, nheads, nchunks, chunk_size)
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    # dA_cumsum strides (same layout as dt)
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # gamma strides (same layout as dt)
    stride_gamma_batch, stride_gamma_chunk, stride_gamma_head, stride_gamma_csize,
    # seq_idx strides
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # beta strides (same layout as dt)
    stride_beta_batch, stride_beta_chunk, stride_beta_head, stride_beta_csize,
    # b_shifted strides (same layout as b)
    stride_bs_batch, stride_bs_seqlen, stride_bs_head, stride_bs_dstate,
    # x_shifted strides (same layout as x)
    stride_xs_batch, stride_xs_seqlen, stride_xs_head, stride_xs_hdim,
    # Meta-parameters
    HAS_GAMMA: tl.constexpr,
    HAS_LOOKBACK: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    # Advance base pointers to this batch, chunk, head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_GAMMA:
        gamma_ptr += pid_b * stride_gamma_batch + pid_c * stride_gamma_chunk + pid_h * stride_gamma_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
    if HAS_LOOKBACK:
        beta_ptr += pid_b * stride_beta_batch + pid_c * stride_beta_chunk + pid_h * stride_beta_head
        b_shifted_ptr += pid_b * stride_bs_batch + pid_c * chunk_size * stride_bs_seqlen + (pid_h // nheads_ngroups_ratio) * stride_bs_head
        x_shifted_ptr += pid_b * stride_xs_batch + pid_c * chunk_size * stride_xs_seqlen + pid_h * stride_xs_head

    # Tile offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for the inner loop
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_GAMMA:
        gamma_ptrs = gamma_ptr + offs_k * stride_gamma_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen
    if HAS_LOOKBACK:
        beta_ptrs = beta_ptr + offs_k * stride_beta_csize
        bs_ptrs = b_shifted_ptr + (offs_n[None, :] * stride_bs_dstate + offs_k[:, None] * stride_bs_seqlen)
        xs_ptrs = x_shifted_ptr + (offs_m[:, None] * stride_xs_hdim + offs_k[None, :] * stride_xs_seqlen)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        # Load x and B tiles
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)

        # Load dA cumsum and compute decay from position to end of chunk
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)

        # Seq_idx masking: zero out positions from different documents
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)

        # Compute scale = exp(dA_last - dA_k) * weight_k
        # weight_k is gamma_k for Mamba-3, dt_k for Mamba-2 fallback
        if HAS_GAMMA:
            weight_k = tl.load(gamma_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        else:
            weight_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)

        if not HAS_SEQ_IDX:
            scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * weight_k
        else:
            scale = tl.where(
                (seq_idx_last >= 0) & (seq_idx_k == seq_idx_last),
                tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * weight_k,
                0.0
            )

        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)

        # Lookback term: beta_k * B_shifted_k outer x_shifted_k
        if HAS_LOOKBACK:
            xs = tl.load(xs_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
            bs = tl.load(bs_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)

            beta_k = tl.load(beta_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)

            if not HAS_SEQ_IDX:
                scale_lb = tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * beta_k
            else:
                scale_lb = tl.where(
                    (seq_idx_last >= 0) & (seq_idx_k == seq_idx_last),
                    tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * beta_k,
                    0.0
                )

            bs *= scale_lb[:, None]
            bs = bs.to(x_ptr.dtype.element_ty)
            acc += tl.dot(xs, bs)

            # Advance lookback pointers
            xs_ptrs += BLOCK_SIZE_K * stride_xs_seqlen
            bs_ptrs += BLOCK_SIZE_K * stride_bs_seqlen
            beta_ptrs += BLOCK_SIZE_K * stride_beta_csize

        # Advance pointers
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_GAMMA:
            gamma_ptrs += BLOCK_SIZE_K * stride_gamma_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen

    # Store the output tile
    states = acc.to(states_ptr.dtype.element_ty)
    states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def _mamba3_chunk_state_fwd(B, x, dt, dA_cumsum, gamma=None, beta=None,
                             B_shifted=None, x_shifted=None, seq_idx=None,
                             states_in_fp32=True):
    """
    Compute per-chunk states for Mamba-3 SSD.

    If gamma is None, falls back to dt scaling (Mamba-2 mode).
    If beta/B_shifted/x_shifted are None, only current term (Euler mode).

    Arguments:
        B: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)  -- chunked layout
        dA_cumsum: (batch, nheads, nchunks, chunk_size) -- cumulative dA within chunks
        gamma: (batch, nheads, nchunks, chunk_size) or None -- current term weight
        beta: (batch, nheads, nchunks, chunk_size) or None -- lookback term weight
        B_shifted: (batch, seqlen, ngroups, dstate) or None -- B shifted by 1
        x_shifted: (batch, seqlen, nheads, headdim) or None -- x shifted by 1
        seq_idx: (batch, seqlen) or None -- document boundary indices

    Returns:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if gamma is not None:
        assert gamma.shape == (batch, nheads, nchunks, chunk_size)
    if beta is not None:
        assert beta.shape == (batch, nheads, nchunks, chunk_size)
        assert B_shifted is not None and x_shifted is not None
        assert B_shifted.shape == B.shape
        assert x_shifted.shape == x.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

    has_lookback = beta is not None and B_shifted is not None and x_shifted is not None
    has_gamma = gamma is not None

    states_dtype = torch.float32 if states_in_fp32 else B.dtype
    states = torch.empty(
        (batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype
    )

    grid = lambda META: (
        triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
        batch * nchunks,
        nheads,
    )

    with torch.cuda.device(x.device.index):
        _mamba3_chunk_state_fwd_kernel[grid](
            # Core data pointers
            x, B, states,
            dt, dA_cumsum, gamma, seq_idx,
            # Lookback pointers (None if not used)
            beta, B_shifted, x_shifted,
            # Dimensions
            headdim, dstate, chunk_size,
            batch, seqlen, nheads // ngroups,
            # x strides
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            # B strides
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            # states strides
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            # dt strides (note: dt is (b, nheads, nchunks, chunk_size), kernel expects batch, chunk, head, csize)
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            # dA_cumsum strides
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            # gamma strides
            *((gamma.stride(0), gamma.stride(2), gamma.stride(1), gamma.stride(3))
              if has_gamma else (0, 0, 0, 0)),
            # seq_idx strides
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            # beta strides
            *((beta.stride(0), beta.stride(2), beta.stride(1), beta.stride(3))
              if has_lookback else (0, 0, 0, 0)),
            # B_shifted strides
            *((B_shifted.stride(0), B_shifted.stride(1), B_shifted.stride(2), B_shifted.stride(-1))
              if has_lookback else (0, 0, 0, 0)),
            # x_shifted strides
            *((x_shifted.stride(0), x_shifted.stride(1), x_shifted.stride(2), x_shifted.stride(3))
              if has_lookback else (0, 0, 0, 0)),
            # Constexpr flags
            HAS_GAMMA=has_gamma,
            HAS_LOOKBACK=has_lookback,
            HAS_SEQ_IDX=seq_idx is not None,
        )

    return states
