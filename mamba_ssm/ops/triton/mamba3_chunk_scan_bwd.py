# Copyright (c) 2024, Tri Dao, Albert Gu.
# Mamba-3 chunk scan Triton backward kernels.
#
# Extends Mamba-2's backward kernels from ssd_chunk_scan.py and ssd_combined.py
# to support the trapezoidal discretization used in Mamba-3:
#
#   Y_diag[m] = sum_k L[m,k] * gamma[k] * CB[m,k] * x[k]          (current term)
#             + sum_k L[m,k] * beta[k]  * CB_s[m,k] * x_s[k]       (lookback term)
#
# Where L[m,k] = exp(dA_cs[m] - dA_cs[k]) is the causal decay matrix.
# gamma replaces dt in Mamba-2's intra-chunk, beta scales the lookback term.
#
# Three kernels:
#   1. _mamba3_chunk_scan_chunk_state_bwd_dx_kernel  -- dx, dgamma, dbeta, dD from both paths
#   2. _mamba3_chunk_scan_bwd_dcb_kernel             -- dCB and dCB_shifted
#   3. _mamba3_chunk_scan_bwd_ddAcs_stable_kernel    -- ddA_cumsum (stable)

import math
from packaging import version

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from mamba_ssm.utils.determinism import (
    alloc_tile_workspace,
    finalize_tile_workspace,
    use_deterministic_mode,
    autotune_configs,
)

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


# =============================================================================
# Kernel 1: Combined backward dx from intra-chunk (CB path) + inter-chunk (states path)
# =============================================================================

@triton.autotune(
    configs=autotune_configs([
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr", "dgamma_ptr", "dbeta_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr", "dgamma_ptr", "dbeta_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr", "dgamma_ptr", "dbeta_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr", "dgamma_ptr", "dbeta_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr", "dgamma_ptr", "dbeta_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr", "dgamma_ptr", "dbeta_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr", "dgamma_ptr", "dbeta_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr", "dgamma_ptr", "dbeta_ptr"])),
    ]),
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _mamba3_chunk_scan_chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr, cb_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, D_ptr,
    b_ptr, dstates_ptr,
    # Mamba-3 specific pointers
    gamma_ptr, beta_ptr,
    cb_shifted_ptr, x_shifted_ptr, b_shifted_ptr,
    # Output pointers
    dx_ptr, ddt_ptr, dD_ptr,
    dgamma_ptr, dbeta_ptr, dx_shifted_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # x strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    # cb strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    # dout strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    # dt strides (chunked layout: batch, chunk, head, csize)
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    # dA_cumsum strides
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # seq_idx strides
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # D stride
    stride_D_head,
    # B strides
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    # dstates strides
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim, stride_dstates_dstate,
    # gamma strides (chunked layout)
    stride_gamma_batch, stride_gamma_chunk, stride_gamma_head, stride_gamma_csize,
    # beta strides
    stride_beta_batch, stride_beta_chunk, stride_beta_head, stride_beta_csize,
    # cb_shifted strides
    stride_cbs_batch, stride_cbs_chunk, stride_cbs_head, stride_cbs_csize_m, stride_cbs_csize_k,
    # x_shifted strides
    stride_xs_batch, stride_xs_seqlen, stride_xs_head, stride_xs_hdim,
    # b_shifted strides
    stride_bs_batch, stride_bs_seqlen, stride_bs_head, stride_bs_dstate,
    # dx strides
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    # ddt strides
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize, stride_ddt_tile,
    # dD strides
    stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
    # dgamma strides
    stride_dgamma_batch, stride_dgamma_chunk, stride_dgamma_head, stride_dgamma_csize, stride_dgamma_tile,
    # dbeta strides
    stride_dbeta_batch, stride_dbeta_chunk, stride_dbeta_head, stride_dbeta_csize, stride_dbeta_tile,
    # dx_shifted strides
    stride_dxs_batch, stride_dxs_seqlen, stride_dxs_head, stride_dxs_hdim,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    HAS_LOOKBACK: tl.constexpr,
    HAS_GAMMA: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    DETERMINISTIC_REDUCTION: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    # Advance base pointers
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head + pid_n * stride_ddt_tile
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
    if HAS_GAMMA:
        gamma_ptr += pid_b * stride_gamma_batch + pid_c * stride_gamma_chunk + pid_h * stride_gamma_head
        dgamma_ptr += pid_b * stride_dgamma_batch + pid_c * stride_dgamma_chunk + pid_h * stride_dgamma_head + pid_n * stride_dgamma_tile
    if HAS_LOOKBACK:
        beta_ptr += pid_b * stride_beta_batch + pid_c * stride_beta_chunk + pid_h * stride_beta_head
        dbeta_ptr += pid_b * stride_dbeta_batch + pid_c * stride_dbeta_chunk + pid_h * stride_dbeta_head + pid_n * stride_dbeta_tile
        cb_shifted_ptr += pid_b * stride_cbs_batch + pid_c * stride_cbs_chunk + (pid_h // nheads_ngroups_ratio) * stride_cbs_head
        x_shifted_ptr += pid_b * stride_xs_batch + pid_c * chunk_size * stride_xs_seqlen + pid_h * stride_xs_head
        b_shifted_ptr += pid_b * stride_bs_batch + pid_c * chunk_size * stride_bs_seqlen + (pid_h // nheads_ngroups_ratio) * stride_bs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # ========================================================================
    # Phase 1: Inter-chunk contribution (from dstates)
    # dx_curr[m] += B[m]^T @ dstates * exp(dA_last - dA_m) * gamma[m]   (current)
    # dx_shift[m] += B_shifted[m]^T @ dstates * exp(dA_last - dA_m) * beta[m]  (lookback)
    # ========================================================================
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)

    if not HAS_SEQ_IDX:
        scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0)), 0.0)

    # Compute B[m] @ dstates for current inter-chunk term
    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_dstates_hdim + offs_dstate[:, None] * stride_dstates_dstate)
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate), other=0.0)
        dstates_val = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates_val = dstates_val.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates_val) * scale[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate - k), other=0.0)
            dstates_val = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates_val = dstates_val.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates_val)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= scale[:, None]

    # acc now holds B[m] @ dstates * scale for inter-chunk.
    # For lookback inter-chunk: B_shifted[m] @ dstates * scale * beta[m]
    if HAS_LOOKBACK:
        acc_shift = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        offs_dstate_lb = tl.arange(0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
        bs_ptrs = b_shifted_ptr + (offs_m[:, None] * stride_bs_seqlen + offs_dstate_lb[None, :] * stride_bs_dstate)
        dstates_ptrs2 = dstates_ptr + (offs_n[None, :] * stride_dstates_hdim + offs_dstate_lb[:, None] * stride_dstates_dstate)
        if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
            bs = tl.load(bs_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate_lb[None, :] < dstate), other=0.0)
            dstates_val2 = tl.load(dstates_ptrs2, mask=(offs_dstate_lb[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
            dstates_val2 = dstates_val2.to(bs_ptrs.dtype.element_ty)
            acc_shift = tl.dot(bs, dstates_val2) * scale[:, None]
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                bs = tl.load(bs_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate_lb[None, :] < dstate - k), other=0.0)
                dstates_val2 = tl.load(dstates_ptrs2, mask=(offs_dstate_lb[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
                dstates_val2 = dstates_val2.to(bs_ptrs.dtype.element_ty)
                acc_shift += tl.dot(bs, dstates_val2)
                bs_ptrs += BLOCK_SIZE_K * stride_bs_dstate
                dstates_ptrs2 += BLOCK_SIZE_K * stride_dstates_dstate
            acc_shift *= scale[:, None]

    # ========================================================================
    # Phase 2: Intra-chunk contribution (from CB path, transposed causal)
    # dx_curr[m] += sum_k CB^T[k,m] * L^T[k,m] * gamma[m] * dout[k]     (k >= m)
    # dx_shift[m] += sum_k CB_shifted^T[k,m] * L^T[k,m] * beta[m] * dout[k]
    #
    # In the backward, we iterate over k >= m (upper triangle of CB, transposed).
    # The CB matrix is stored as CB[row, col] with row=k, col=m in the transposed view.
    # In the stored layout, cb_ptr[m, k] is accessed as cb_ptr + m * stride_csize_m + k * stride_csize_k.
    # For the backward, we need CB^T[k, m] which is cb[m, k] since CB is stored with
    # stride_cb_csize_m for row and stride_cb_csize_k for col.
    # ========================================================================
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    K_MAX = chunk_size_limit
    K_MIN = pid_m * BLOCK_SIZE_M
    cb_ptrs += K_MIN * stride_cb_csize_k
    dout_ptrs += K_MIN * stride_dout_seqlen
    dA_cumsum_ptrs += K_MIN * stride_dA_cs_csize

    if HAS_LOOKBACK:
        cbs_ptrs = cb_shifted_ptr + (offs_m[:, None] * stride_cbs_csize_m + offs_k[None, :] * stride_cbs_csize_k)
        cbs_ptrs += K_MIN * stride_cbs_csize_k

    for k in range(K_MIN, K_MAX, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
        # Load CB values and compute transposed causal decay
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k), other=0.0)
        dout = tl.load(dout_ptrs, mask=(offs_k[:, None] < K_MAX - k) & (offs_n[None, :] < hdim), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < K_MAX - k, other=0.0).to(tl.float32)
        # L^T[k,m] = exp(dA_cs_k - dA_cs_m), but transposed: for backward k >= m
        cb *= tl.exp(tl.minimum((dA_cs_k[None, :] - dA_cs_m[:, None]), 0.0))
        mask = (k + offs_k[None, :] >= offs_m[:, None]) & (k + offs_k[None, :] < K_MAX)
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout)

        # Lookback intra-chunk contribution
        if HAS_LOOKBACK:
            cbs = tl.load(cbs_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k), other=0.0)
            cbs *= tl.exp(tl.minimum((dA_cs_k[None, :] - dA_cs_m[:, None]), 0.0))
            cbs = tl.where(mask, cbs, 0.0)
            cbs = cbs.to(dout_ptr.dtype.element_ty)
            acc_shift += tl.dot(cbs, dout)
            cbs_ptrs += BLOCK_SIZE_K * stride_cbs_csize_k

        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # ========================================================================
    # Phase 3: Scale by gamma/beta, compute outputs
    # ========================================================================
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_GAMMA:
        gamma_ptrs = gamma_ptr + offs_m * stride_gamma_csize
        gamma_m = tl.load(gamma_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
        dx = acc * gamma_m[:, None]
    else:
        # Fallback to dt (Mamba-2 compatible)
        dt_ptrs = dt_ptr + offs_m * stride_dt_csize
        dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
        dx = acc * dt_m[:, None]

    # D skip connection gradient
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    if HAS_D:
        dout_res_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
        dout_res = tl.load(dout_res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * D
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

    # dD computation
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    if HAS_D:
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize
        if D_HAS_HDIM:
            dD_ptrs_local = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs_local, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            if DETERMINISTIC_REDUCTION:
                tl.store(dD_ptr + pid_n * stride_dD_hdim, dD)
            else:
                tl.atomic_add(dD_ptr, dD)

    # When HAS_GAMMA: scaling factor is gamma (not dt), so ddt=0 from this kernel.
    # The dt gradient only comes from the ddA_cumsum path.
    # When !HAS_GAMMA: dt IS the scaling factor (Mamba-2 fallback), so ddt = sum(acc*x).
    if HAS_GAMMA:
        # dgamma = sum(acc * x) — gradient w.r.t. the gamma scaling factor
        dgamma = tl.sum(acc * x, axis=1)
        dgamma_ptrs = dgamma_ptr + offs_m * stride_dgamma_csize
        if DETERMINISTIC_REDUCTION:
            tl.store(dgamma_ptrs, dgamma, mask=offs_m < chunk_size)
        else:
            tl.atomic_add(dgamma_ptrs, dgamma, mask=offs_m < chunk_size)
        # ddt stays zero (from zero_init) — dt gradient only comes from ddA path
    else:
        # Mamba-2 fallback: dt IS the scaling factor, so ddt = sum(acc * x)
        ddt = tl.sum(acc * x, axis=1)
        ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
        if DETERMINISTIC_REDUCTION:
            tl.store(ddt_ptrs, ddt, mask=offs_m < chunk_size)
        else:
            tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)

    # Lookback: store dx_shifted and compute dbeta
    if HAS_LOOKBACK:
        beta_ptrs = beta_ptr + offs_m * stride_beta_csize
        beta_m = tl.load(beta_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
        dx_shift = acc_shift * beta_m[:, None]
        dx_shifted_ptr += pid_b * stride_dxs_batch + pid_c * chunk_size * stride_dxs_seqlen + pid_h * stride_dxs_head
        dxs_ptrs = dx_shifted_ptr + (offs_m[:, None] * stride_dxs_seqlen + offs_n[None, :] * stride_dxs_hdim)
        tl.store(dxs_ptrs, dx_shift, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

        # dbeta[m] = sum_n acc_shift_before_beta[m,n] * x_shifted[m,n]
        xs_ptrs = x_shifted_ptr + (offs_m[:, None] * stride_xs_seqlen + offs_n[None, :] * stride_xs_hdim)
        xs = tl.load(xs_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        dbeta = tl.sum(acc_shift * xs, axis=1)
        dbeta_ptrs = dbeta_ptr + offs_m * stride_dbeta_csize
        if DETERMINISTIC_REDUCTION:
            tl.store(dbeta_ptrs, dbeta, mask=offs_m < chunk_size)
        else:
            tl.atomic_add(dbeta_ptrs, dbeta, mask=offs_m < chunk_size)


_MAMBA3_CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N = min(
    cfg.kwargs['BLOCK_SIZE_N'] for cfg in _mamba3_chunk_scan_chunk_state_bwd_dx_kernel.configs
)


def _mamba3_chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates,
                                           D=None, seq_idx=None,
                                           gamma=None, beta=None,
                                           CB_shifted=None, x_shifted=None, B_shifted=None):
    """
    Combined backward for dx from intra-chunk (CB) and inter-chunk (states).

    Arguments:
        x: (batch, seqlen, nheads, headdim) -- input
        dt: (batch, nheads, nchunks, chunk_size) -- timestep
        dA_cumsum: (batch, nheads, nchunks, chunk_size) -- cumulative dA
        B: (batch, seqlen, ngroups, dstate) -- input projection
        CB: (batch, nchunks, ngroups, chunk_size, chunk_size) -- C^T B product
        dout: (batch, seqlen, nheads, headdim) -- output gradient
        dstates: (batch, nchunks, nheads, headdim, dstate) -- state gradient
        D: (nheads,) or (nheads, headdim) or None -- skip connection
        seq_idx: (batch, seqlen) or None
        gamma: (batch, nheads, nchunks, chunk_size) or None -- current term weight
        beta: (batch, nheads, nchunks, chunk_size) or None -- lookback weight
        CB_shifted: same shape as CB or None
        x_shifted: same shape as x or None
        B_shifted: same shape as B or None

    Returns: dx, ddt, dD, dgamma, dbeta, dx_shifted
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

    has_gamma = gamma is not None
    has_lookback = (beta is not None and CB_shifted is not None
                    and x_shifted is not None and B_shifted is not None)
    if has_gamma:
        assert gamma.shape == (batch, nheads, nchunks, chunk_size)
    if has_lookback:
        assert beta.shape == (batch, nheads, nchunks, chunk_size)
        assert CB_shifted.shape == CB.shape
        assert x_shifted.shape == x.shape
        assert B_shifted.shape == B.shape

    deterministic = use_deterministic_mode()

    # Allocate outputs
    dx = torch.empty_like(x)

    tile_count = math.ceil(headdim / _MAMBA3_CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N)

    ddt, stride_ddt_tile = alloc_tile_workspace(
        (batch, nheads, nchunks, chunk_size),
        tile_count,
        torch.float32,
        dout.device,
        deterministic,
        zero_init=True,
    )

    # dD allocation
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        pid_m_tiles = triton.cdiv(chunk_size, BLOCK_SIZE_min)
        pid_n_tiles = math.ceil(headdim / _MAMBA3_CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N)
        if D.dim() == 2:
            dD_hdim = headdim
        elif deterministic:
            dD_hdim = pid_n_tiles
        else:
            dD_hdim = 1
        dD = torch.zeros(pid_m_tiles, batch, nchunks, nheads, dD_hdim, device=D.device, dtype=torch.float32)
        dD_strides = (dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
    else:
        dD = None
        dD_strides = (0, 0, 0, 0, 0)

    # dgamma, dbeta allocation
    if has_gamma:
        dgamma, stride_dgamma_tile = alloc_tile_workspace(
            (batch, nheads, nchunks, chunk_size),
            tile_count,
            torch.float32,
            dout.device,
            deterministic,
            zero_init=True,
        )
    else:
        dgamma = None
        stride_dgamma_tile = 0

    if has_lookback:
        dbeta, stride_dbeta_tile = alloc_tile_workspace(
            (batch, nheads, nchunks, chunk_size),
            tile_count,
            torch.float32,
            dout.device,
            deterministic,
            zero_init=True,
        )
        dx_shifted = torch.empty_like(x)
    else:
        dbeta = None
        stride_dbeta_tile = 0
        dx_shifted = None

    # Strides for optional tensors
    gamma_strides = (gamma.stride(0), gamma.stride(2), gamma.stride(1), gamma.stride(3)) if has_gamma else (0, 0, 0, 0)
    beta_strides = (beta.stride(0), beta.stride(2), beta.stride(1), beta.stride(3)) if has_lookback else (0, 0, 0, 0)
    cbs_strides = (CB_shifted.stride(0), CB_shifted.stride(1), CB_shifted.stride(2), CB_shifted.stride(-1), CB_shifted.stride(-2)) if has_lookback else (0, 0, 0, 0, 0)
    xs_strides = (x_shifted.stride(0), x_shifted.stride(1), x_shifted.stride(2), x_shifted.stride(3)) if has_lookback else (0, 0, 0, 0)
    bs_strides = (B_shifted.stride(0), B_shifted.stride(1), B_shifted.stride(2), B_shifted.stride(3)) if has_lookback else (0, 0, 0, 0)
    dxs_strides = (dx_shifted.stride(0), dx_shifted.stride(1), dx_shifted.stride(2), dx_shifted.stride(3)) if has_lookback else (0, 0, 0, 0)

    # dgamma/dbeta have shape (batch, nheads, nchunks, chunk_size) from alloc_tile_workspace
    # Stride order: batch, chunk, head, csize (matching ddt convention)
    dgamma_strides = (dgamma.stride(0), dgamma.stride(2), dgamma.stride(1), dgamma.stride(3)) if has_gamma else (0, 0, 0, 0)
    dbeta_strides = (dbeta.stride(0), dbeta.stride(2), dbeta.stride(1), dbeta.stride(3)) if has_lookback else (0, 0, 0, 0)

    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                        batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _mamba3_chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x, CB, dout, dt, dA_cumsum, seq_idx, D,
            B, dstates,
            gamma, beta,
            CB_shifted, x_shifted, B_shifted,
            dx, ddt, dD,
            dgamma, dbeta, dx_shifted,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            # x strides
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            # CB strides
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(-1), CB.stride(-2),
            # dout strides
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            # dt strides
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            # dA_cumsum strides
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            # seq_idx strides
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            # D stride
            D.stride(0) if D is not None else 0,
            # B strides
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            # dstates strides
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            # gamma strides
            *gamma_strides,
            # beta strides
            *beta_strides,
            # CB_shifted strides
            *cbs_strides,
            # x_shifted strides
            *xs_strides,
            # B_shifted strides
            *bs_strides,
            # dx strides
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            # ddt strides
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3), stride_ddt_tile,
            # dD strides
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            # dgamma strides
            *dgamma_strides, stride_dgamma_tile,
            # dbeta strides
            *dbeta_strides, stride_dbeta_tile,
            # dx_shifted strides
            *dxs_strides,
            # constexpr
            D is not None,
            D.dim() == 2 if D is not None else True,
            HAS_SEQ_IDX=seq_idx is not None,
            HAS_LOOKBACK=has_lookback,
            HAS_GAMMA=has_gamma,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            IS_TRITON_22=TRITON_22,
            DETERMINISTIC_REDUCTION=deterministic,
        )

    # Finalize reductions
    ddt = finalize_tile_workspace(ddt, deterministic)
    if D is not None:
        BLOCK_SIZE_actual = _mamba3_chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2))
        if D.dim() == 1:
            dD = dD.sum(dim=-1)
        dD = dD.to(dtype=D.dtype)
    if has_gamma:
        dgamma = finalize_tile_workspace(dgamma, deterministic)
    if has_lookback:
        dbeta = finalize_tile_workspace(dbeta, deterministic)

    return dx, ddt, dD, dgamma, dbeta, dx_shifted


# =============================================================================
# Kernel 2: Backward dCB (and dCB_shifted)
# =============================================================================

@triton.autotune(
    configs=autotune_configs([
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
    ]),
    key=['chunk_size', 'hdim'],
)
@triton.jit
def _mamba3_chunk_scan_bwd_dcb_kernel(
    # Pointers to matrices
    x_ptr, dout_ptr, dA_cumsum_ptr, seq_idx_ptr,
    gamma_ptr, beta_ptr, x_shifted_ptr,
    # Output pointers
    dcb_ptr, dcb_shifted_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen, nheads, nheads_per_program, ngroups,
    # x strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    # dout strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    # dA_cumsum strides
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # seq_idx strides
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # gamma strides
    stride_gamma_batch, stride_gamma_chunk, stride_gamma_head, stride_gamma_csize,
    # beta strides
    stride_beta_batch, stride_beta_chunk, stride_beta_head, stride_beta_csize,
    # x_shifted strides
    stride_xs_batch, stride_xs_seqlen, stride_xs_head, stride_xs_hdim,
    # dcb strides
    stride_dcb_batch, stride_dcb_chunk, stride_dcb_split, stride_dcb_group, stride_dcb_csize_m, stride_dcb_csize_n,
    # dcb_shifted strides
    stride_dcbs_batch, stride_dcbs_chunk, stride_dcbs_split, stride_dcbs_group, stride_dcbs_csize_m, stride_dcbs_csize_n,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    HAS_LOOKBACK: tl.constexpr,
    HAS_GAMMA: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(axis=2)
    pid_s = pid_sg // ngroups
    pid_g = pid_sg - pid_s * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_x_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dout_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dA_cs_head
    if HAS_GAMMA:
        gamma_ptr += pid_b * stride_gamma_batch + pid_c * stride_gamma_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_gamma_head
    if HAS_LOOKBACK:
        beta_ptr += pid_b * stride_beta_batch + pid_c * stride_beta_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_beta_head
        x_shifted_ptr += pid_b * stride_xs_batch + pid_c * chunk_size * stride_xs_seqlen + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_xs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] * stride_x_hdim)
    if HAS_GAMMA:
        gamma_ptrs = gamma_ptr + offs_n * stride_gamma_csize
    if HAS_LOOKBACK:
        xs_ptrs = x_shifted_ptr + (offs_n[None, :] * stride_xs_seqlen + offs_k[:, None] * stride_xs_hdim)
        beta_ptrs = beta_ptr + offs_n * stride_beta_csize

    # Early exit for blocks entirely above the causal diagonal
    if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
        dcb_ptr += pid_b * stride_dcb_batch + pid_c * stride_dcb_chunk + pid_g * stride_dcb_group + pid_s * stride_dcb_split
        dcb_ptrs_out = dcb_ptr + (offs_m[:, None] * stride_dcb_csize_m + offs_n[None, :] * stride_dcb_csize_n)
        tl.store(dcb_ptrs_out, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dcb_ptr.dtype.element_ty), mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))
        if HAS_LOOKBACK:
            dcbs_ptr = dcb_shifted_ptr + pid_b * stride_dcbs_batch + pid_c * stride_dcbs_chunk + pid_g * stride_dcbs_group + pid_s * stride_dcbs_split
            dcbs_ptrs_out = dcbs_ptr + (offs_m[:, None] * stride_dcbs_csize_m + offs_n[None, :] * stride_dcbs_csize_n)
            tl.store(dcbs_ptrs_out, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dcbs_ptr.dtype.element_ty), mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))
        return

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    chunk_size_limit_n = min(chunk_size_limit, (pid_m + 1) * BLOCK_SIZE_M)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_LOOKBACK:
        acc_shift = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s * nheads_per_program)
    for h in range(nheads_iter):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit_n), other=0.0)
        dcb = tl.dot(dout, x)

        # Scale by gamma[n] (replaces dt[n] in Mamba-2)
        if HAS_GAMMA:
            gamma_n = tl.load(gamma_ptrs, mask=offs_n < chunk_size, other=0.0).to(tl.float32)
            dcb *= gamma_n
        else:
            # Mamba-2 fallback: would use dt, but in Mamba-3 context this shouldn't happen
            pass

        dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
        dA_cs_n = tl.load(dA_cumsum_ptr + offs_n * stride_dA_cs_csize, mask=offs_n < chunk_size_limit, other=0.0).to(tl.float32)
        dcb *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_n[None, :]), 0.0))
        acc += dcb

        # Lookback term: dCB_shifted[m,n] = sum_k dout[m,k] * x_shifted[n,k] * beta[n] * L[m,n]
        if HAS_LOOKBACK:
            xs = tl.load(xs_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit_n), other=0.0)
            dcbs = tl.dot(dout, xs)
            beta_n = tl.load(beta_ptrs, mask=offs_n < chunk_size, other=0.0).to(tl.float32)
            dcbs *= beta_n
            dcbs *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_n[None, :]), 0.0))
            acc_shift += dcbs
            xs_ptrs += stride_xs_head
            beta_ptrs += stride_beta_head

        dout_ptrs += stride_dout_head
        x_ptrs += stride_x_head
        dA_cumsum_ptr += stride_dA_cs_head
        if HAS_GAMMA:
            gamma_ptrs += stride_gamma_head

    # Apply causal mask and seq_idx mask
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
        if HAS_LOOKBACK:
            acc_shift = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc_shift, 0.0)
    mask = offs_m[:, None] >= offs_n[None, :]
    acc = tl.where(mask, acc, 0.0)

    dcb_ptr += pid_b * stride_dcb_batch + pid_c * stride_dcb_chunk + pid_g * stride_dcb_group + pid_s * stride_dcb_split
    dcb_ptrs_out = dcb_ptr + (offs_m[:, None] * stride_dcb_csize_m + offs_n[None, :] * stride_dcb_csize_n)
    tl.store(dcb_ptrs_out, acc, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))

    if HAS_LOOKBACK:
        acc_shift = tl.where(mask, acc_shift, 0.0)
        dcbs_ptr = dcb_shifted_ptr + pid_b * stride_dcbs_batch + pid_c * stride_dcbs_chunk + pid_g * stride_dcbs_group + pid_s * stride_dcbs_split
        dcbs_ptrs_out = dcbs_ptr + (offs_m[:, None] * stride_dcbs_csize_m + offs_n[None, :] * stride_dcbs_csize_n)
        tl.store(dcbs_ptrs_out, acc_shift, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))


def _mamba3_chunk_scan_bwd_dcb(x, dA_cumsum, dout, seq_idx=None,
                                gamma=None, beta=None, x_shifted=None, ngroups=1):
    """
    Backward for dCB (and dCB_shifted if lookback is enabled).

    Arguments:
        x: (batch, seqlen, nheads, headdim)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        dout: (batch, seqlen, nheads, headdim)
        seq_idx: (batch, seqlen) or None
        gamma: (batch, nheads, nchunks, chunk_size) or None
        beta: (batch, nheads, nchunks, chunk_size) or None
        x_shifted: (batch, seqlen, nheads, headdim) or None
        ngroups: int

    Returns: dCB, dCB_shifted (or dCB_shifted=None if no lookback)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dA_cumsum.shape
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == x.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

    has_gamma = gamma is not None
    has_lookback = beta is not None and x_shifted is not None

    if has_gamma:
        assert gamma.shape == (batch, nheads, nchunks, chunk_size)
    if has_lookback:
        assert beta.shape == (batch, nheads, nchunks, chunk_size)
        assert x_shifted.shape == x.shape

    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)

    dcb = torch.empty(batch, nchunks, nsplits, ngroups, chunk_size, chunk_size, device=x.device, dtype=torch.float32)
    if has_lookback:
        dcb_shifted = torch.empty_like(dcb)
    else:
        dcb_shifted = None

    # Strides for optional tensors
    gamma_strides = (gamma.stride(0), gamma.stride(2), gamma.stride(1), gamma.stride(3)) if has_gamma else (0, 0, 0, 0)
    beta_strides = (beta.stride(0), beta.stride(2), beta.stride(1), beta.stride(3)) if has_lookback else (0, 0, 0, 0)
    xs_strides = (x_shifted.stride(0), x_shifted.stride(1), x_shifted.stride(2), x_shifted.stride(3)) if has_lookback else (0, 0, 0, 0)
    dcbs_strides = (dcb_shifted.stride(0), dcb_shifted.stride(1), dcb_shifted.stride(2), dcb_shifted.stride(3), dcb_shifted.stride(4), dcb_shifted.stride(5)) if has_lookback else (0, 0, 0, 0, 0, 0)

    grid_dcb = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(chunk_size, META['BLOCK_SIZE_N']),
                        batch * nchunks, nsplits * ngroups)
    with torch.cuda.device(x.device.index):
        _mamba3_chunk_scan_bwd_dcb_kernel[grid_dcb](
            x, dout, dA_cumsum, seq_idx,
            gamma, beta, x_shifted,
            dcb, dcb_shifted,
            chunk_size, headdim,
            batch, seqlen, nheads, nheads_per_program, ngroups,
            # x strides
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            # dout strides
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            # dA_cumsum strides
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            # seq_idx strides
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            # gamma strides
            *gamma_strides,
            # beta strides
            *beta_strides,
            # x_shifted strides
            *xs_strides,
            # dcb strides
            dcb.stride(0), dcb.stride(1), dcb.stride(2), dcb.stride(3), dcb.stride(4), dcb.stride(5),
            # dcb_shifted strides
            *dcbs_strides,
            # constexpr
            HAS_SEQ_IDX=seq_idx is not None,
            HAS_LOOKBACK=has_lookback,
            HAS_GAMMA=has_gamma,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )

    dcb = dcb.sum(2)
    if has_lookback:
        dcb_shifted = dcb_shifted.sum(2)

    return dcb, dcb_shifted


# =============================================================================
# Kernel 3: Backward ddA_cumsum (stable version)
# =============================================================================

@triton.autotune(
    configs=autotune_configs([
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
    ]),
    key=['chunk_size', 'hdim'],
)
@triton.jit
def _mamba3_chunk_scan_bwd_ddAcs_stable_kernel(
    # Pointers to matrices
    x_ptr, dout_ptr, dA_cumsum_ptr, cb_ptr,
    gamma_ptr, beta_ptr, x_shifted_ptr, cb_shifted_ptr,
    # Output pointer
    ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen, nheads_ngroups_ratio,
    # x strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    # dout strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    # dA_cumsum strides
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # cb strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_n,
    # gamma strides
    stride_gamma_batch, stride_gamma_chunk, stride_gamma_head, stride_gamma_csize,
    # beta strides
    stride_beta_batch, stride_beta_chunk, stride_beta_head, stride_beta_csize,
    # x_shifted strides
    stride_xs_batch, stride_xs_seqlen, stride_xs_head, stride_xs_hdim,
    # cb_shifted strides
    stride_cbs_batch, stride_cbs_chunk, stride_cbs_head, stride_cbs_csize_m, stride_cbs_csize_n,
    # ddA_cumsum strides
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize_m, stride_ddA_cs_csize_n,
    # Meta-parameters
    HAS_LOOKBACK: tl.constexpr,
    HAS_GAMMA: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head + pid_m * stride_ddA_cs_csize_m
    if HAS_GAMMA:
        gamma_ptr += pid_b * stride_gamma_batch + pid_c * stride_gamma_chunk + pid_h * stride_gamma_head
    if HAS_LOOKBACK:
        beta_ptr += pid_b * stride_beta_batch + pid_c * stride_beta_chunk + pid_h * stride_beta_head
        x_shifted_ptr += pid_b * stride_xs_batch + pid_c * chunk_size * stride_xs_seqlen + pid_h * stride_xs_head
        cb_shifted_ptr += pid_b * stride_cbs_batch + pid_c * stride_cbs_chunk + (pid_h // nheads_ngroups_ratio) * stride_cbs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] * stride_x_hdim)
    if HAS_GAMMA:
        gamma_ptrs = gamma_ptr + offs_n * stride_gamma_csize
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None, :] * stride_cb_csize_n)
    ddAcs_ptrs = ddA_cumsum_ptr + offs_n * stride_ddA_cs_csize_n
    tl.store(ddA_cumsum_ptr, 0.0)

    if HAS_LOOKBACK:
        xs_ptrs = x_shifted_ptr + (offs_n[None, :] * stride_xs_seqlen + offs_k[:, None] * stride_xs_hdim)
        beta_ptrs = beta_ptr + offs_n * stride_beta_csize
        cbs_ptrs = cb_shifted_ptr + (offs_m[:, None] * stride_cbs_csize_m + offs_n[None, :] * stride_cbs_csize_n)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    rowsum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    lo, hi = 0, (pid_m + 1) * BLOCK_SIZE_M

    for start_n in range(lo, hi, BLOCK_SIZE_N):
        start_n = tl.multiple_of(start_n, BLOCK_SIZE_N)

        # Current term: dout[m] @ x[n]^T * gamma[n] * CB[m,n] * L[m,n]
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit - start_n), other=0.0)
        acc = tl.dot(dout, x)

        if HAS_GAMMA:
            gamma_n = tl.load(gamma_ptrs, mask=offs_n < chunk_size - start_n, other=0.0).to(tl.float32)
            acc *= gamma_n
        # If there's seq_idx, CB was already zeroed for cross-doc pairs
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size - start_n), other=0.0).to(tl.float32)
        acc *= cb
        dA_cs_n = tl.load(dA_cumsum_ptr + (start_n + offs_n) * stride_dA_cs_csize, mask=offs_n < chunk_size - start_n, other=0.0).to(tl.float32)
        acc *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_n[None, :]), 0.0))

        # Lookback term: dout[m] @ x_shifted[n]^T * beta[n] * CB_shifted[m,n] * L[m,n]
        if HAS_LOOKBACK:
            xs = tl.load(xs_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit - start_n), other=0.0)
            acc_lb = tl.dot(dout, xs)
            beta_n = tl.load(beta_ptrs, mask=offs_n < chunk_size - start_n, other=0.0).to(tl.float32)
            acc_lb *= beta_n
            cbs = tl.load(cbs_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size - start_n), other=0.0).to(tl.float32)
            acc_lb *= cbs
            acc_lb *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_n[None, :]), 0.0))
            acc += acc_lb

        # Apply causal mask and cumsum (same structure as Mamba-2)
        mask = offs_m[:, None] >= start_n + offs_n[None, :] + 1
        acc = tl.where(mask, acc, 0.0)
        rowsum_new = rowsum + tl.sum(acc, axis=1)
        acc = rowsum[:, None] + tl.cumsum(acc, axis=1)
        rowsum = rowsum_new
        acc = tl.where(mask, acc, 0.0)
        ddA_cs = tl.sum(acc, axis=0)
        tl.store(ddAcs_ptrs + stride_ddA_cs_csize_n, ddA_cs, mask=offs_n < chunk_size - start_n - 1)

        # Advance pointers
        x_ptrs += BLOCK_SIZE_N * stride_x_seqlen
        cb_ptrs += BLOCK_SIZE_N * stride_cb_csize_n
        ddAcs_ptrs += BLOCK_SIZE_N * stride_ddA_cs_csize_n
        if HAS_GAMMA:
            gamma_ptrs += BLOCK_SIZE_N * stride_gamma_csize
        if HAS_LOOKBACK:
            xs_ptrs += BLOCK_SIZE_N * stride_xs_seqlen
            beta_ptrs += BLOCK_SIZE_N * stride_beta_csize
            cbs_ptrs += BLOCK_SIZE_N * stride_cbs_csize_n

    # Zero out the rest (since we sum rows together later)
    for start_n in range(hi, chunk_size, BLOCK_SIZE_N):
        tl.store(ddAcs_ptrs + stride_ddA_cs_csize_n, tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32), mask=offs_n < chunk_size - start_n - 1)
        ddAcs_ptrs += BLOCK_SIZE_N * stride_ddA_cs_csize_n


def _mamba3_chunk_scan_bwd_ddAcs_stable(x, dA_cumsum, dout, CB, seq_idx=None,
                                         gamma=None, beta=None,
                                         x_shifted=None, CB_shifted=None, ngroups=1):
    """
    Backward for ddA_cumsum (numerically stable version).

    Arguments:
        x: (batch, seqlen, nheads, headdim)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        dout: (batch, seqlen, nheads, headdim)
        CB: (batch, nchunks, ngroups, chunk_size, chunk_size)
        seq_idx: (batch, seqlen) or None
        gamma: (batch, nheads, nchunks, chunk_size) or None
        beta: (batch, nheads, nchunks, chunk_size) or None
        x_shifted: (batch, seqlen, nheads, headdim) or None
        CB_shifted: (batch, nchunks, ngroups, chunk_size, chunk_size) or None
        ngroups: int

    Returns: ddA_cumsum
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dA_cumsum.shape
    assert dout.shape == x.shape
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert nheads % ngroups == 0
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

    has_gamma = gamma is not None
    has_lookback = (beta is not None and x_shifted is not None and CB_shifted is not None)

    if has_gamma:
        assert gamma.shape == (batch, nheads, nchunks, chunk_size)
    if has_lookback:
        assert beta.shape == (batch, nheads, nchunks, chunk_size)
        assert x_shifted.shape == x.shape
        assert CB_shifted.shape == CB.shape

    # Strides for optional tensors
    gamma_strides = (gamma.stride(0), gamma.stride(2), gamma.stride(1), gamma.stride(3)) if has_gamma else (0, 0, 0, 0)
    beta_strides = (beta.stride(0), beta.stride(2), beta.stride(1), beta.stride(3)) if has_lookback else (0, 0, 0, 0)
    xs_strides = (x_shifted.stride(0), x_shifted.stride(1), x_shifted.stride(2), x_shifted.stride(3)) if has_lookback else (0, 0, 0, 0)
    cbs_strides = (CB_shifted.stride(0), CB_shifted.stride(1), CB_shifted.stride(2), CB_shifted.stride(3), CB_shifted.stride(4)) if has_lookback else (0, 0, 0, 0, 0)

    BLOCK_SIZE_M_min = 32
    ddA_cumsum = torch.empty(batch, nheads, nchunks, triton.cdiv(chunk_size, BLOCK_SIZE_M_min),
                             chunk_size, device=x.device, dtype=torch.float32)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _mamba3_chunk_scan_bwd_ddAcs_stable_kernel[grid_ddtcs](
            x, dout, dA_cumsum, CB,
            gamma, beta, x_shifted, CB_shifted,
            ddA_cumsum,
            chunk_size, headdim,
            batch, seqlen, nheads // ngroups,
            # x strides
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            # dout strides
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            # dA_cumsum strides
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            # cb strides
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(3), CB.stride(4),
            # gamma strides
            *gamma_strides,
            # beta strides
            *beta_strides,
            # x_shifted strides
            *xs_strides,
            # cb_shifted strides
            *cbs_strides,
            # ddA_cumsum strides
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3), ddA_cumsum.stride(4),
            # constexpr
            HAS_LOOKBACK=has_lookback,
            HAS_GAMMA=has_gamma,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    BLOCK_SIZE_M_actual = _mamba3_chunk_scan_bwd_ddAcs_stable_kernel.best_config.kwargs["BLOCK_SIZE_M"]
    n_valid_blocks = (chunk_size + BLOCK_SIZE_M_actual - 1) // BLOCK_SIZE_M_actual
    ddA_cumsum = ddA_cumsum[:, :, :, :n_valid_blocks].sum(dim=3)
    return ddA_cumsum
