"""
Triton kernels for Mamba3 backward-pass dA/dt/trap computations.

This module is self-contained: it provides both dense (fixed-length) and varlen
(variable-length sequences packed into a single tensor) versions of every kernel.
Varlen variants follow the same naming convention as their dense counterparts but
carry a ``_varlen`` suffix.

Kernel groups
-------------
1. bwd_dadt_cumsum_fused  – Fused reverse-cumsum + forward-exclusive-cumsum that
                             back-propagates through the per-chunk cumulative-sum
                             terms ddA_cs and ddA_cs_rev into ddt.
2. bwd_segsum_dadt        – Accumulates the 2-D inter-token segsum contribution of
                             dSSdA into ddt, one [C×C] chunk block at a time.
3. bwd_dtrap_ddt          – Back-propagates through the trapezoidal-rule gamma
                             parameterisation to produce ddt and d(trap_presigmoid).
4. dacs_segsum            – Forward helper: computes the per-chunk prefix sums
                             da_cs, da_cs_rev, and the lower-triangular segsum
                             matrix from the raw decay increments da.

Public API
----------
Dense wrappers (fixed-length sequences):
    bwd_dadt_fused_triton(dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size)
    bwd_dtrap_ddt_triton(trap, dt, dfactor, dgamma_diag, chunk_size)
    compute_dacs_segsum_triton(da, chunk_size)

Varlen wrappers (variable-length sequences packed end-to-end):
    bwd_dadt_fused_triton_varlen(dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size, cu_seqlens)
    bwd_dtrap_ddt_triton_varlen(trap, dt, dfactor, dgamma_diag, chunk_size, cu_seqlens)
    compute_dacs_segsum_triton_varlen(da, chunk_size, cu_seqlens)

Varlen chunk layout
-------------------
For a packed tensor of total length S containing NS sequences, the global chunk
index for the first chunk of sequence i is::

    global_chunk_start_i = (cu_seqlens[i] // chunk_size) + i

giving nchunks_global = (S // chunk_size) + NS total chunk slots.  The additive
NS term reserves one extra slot per sequence to hold its (possibly partial) final
chunk without overwriting the next sequence's slots.  Two int32 mapping tensors
(state_seq_mapping, state_chunk_in_seq) route each kernel program to the correct
position in the packed tensor; see _build_varlen_chunk_mapping for details.
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple



# ============================================================================
# Kernel group 1 (dense): fused reverse-cumsum + forward-exclusive-cumsum
#   bwd_dadt_cumsum_fused_kernel        – dense (fixed-length)
#   bwd_dadt_cumsum_fused_kernel_varlen – varlen (see bottom of file)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
    restore_value=["ddt_out_ptr"],
)
@triton.jit
def bwd_dadt_cumsum_fused_kernel(
    ddA_cs_ptr,         # [B, H, S]
    ddA_cs_rev_ptr,     # [B, H, S]
    dA_cs_ptr,          # [B, H, S]
    dA_cs_rev_ptr,      # [B, H, S]
    ddt_out_ptr,        # [B, H, S] - output
    stride_batch,
    stride_head,
    stride_seq,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Compute ddt contributions from ddA_cs (reverse-cumsum) and ddA_cs_rev
    (forward-exclusive-cumsum) within each chunk.

    Each program handles one CHUNK_SIZE-length slice of the sequence for one
    (batch, head) pair.  Grid: (B, H, nchunks).

    dA_cs / dA_cs_rev are the forward cumulative sums of the decay increments
    used as exponent arguments (clipped to ≤ 0 for numerical stability).
    ddA_cs / ddA_cs_rev are the upstream gradients w.r.t. those cumsums.
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    chunk_start = pid_chunk * CHUNK_SIZE
    offs_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
    mask = offs_seq < S

    base_offset = pid_batch * stride_batch + pid_head * stride_head

    ddA_cs     = tl.load(ddA_cs_ptr     + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)
    ddA_cs_rev = tl.load(ddA_cs_rev_ptr + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)
    dA_cs      = tl.load(dA_cs_ptr      + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)
    dA_cs_rev  = tl.load(dA_cs_rev_ptr  + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)

    # Reverse-cumsum contribution from ddA_cs.
    scaled_ddA_cs = tl.exp(dA_cs) * ddA_cs
    ddt_cs = tl.cumsum(scaled_ddA_cs, axis=0, reverse=True)

    # Forward-exclusive-cumsum contribution from ddA_cs_rev.
    # Compute inclusive cumsum first, then shift right by one position so that
    # output[i] = sum(scaled_ddA_cs_rev[0 : i])  (exclusive).
    scaled_ddA_cs_rev = tl.exp(dA_cs_rev) * ddA_cs_rev
    ddt_cs_rev_inclusive = tl.cumsum(scaled_ddA_cs_rev, axis=0)

    # Roll one to the right:
    i = tl.arange(0, CHUNK_SIZE)[:, None]          # [N,1]
    j = tl.arange(0, CHUNK_SIZE)[None, :]          # [1,N]
    S = (i == j + 1)                      # strictly lower diagonal (one below main)
    ddt_cs_rev_exclusive = tl.sum(tl.where(S, ddt_cs_rev_inclusive, 0), axis=1)

    ddt_total = ddt_cs + ddt_cs_rev_exclusive
    tl.store(ddt_out_ptr + base_offset + offs_seq * stride_seq, ddt_total, mask=mask)


# ============================================================================
# Kernel group 2 (dense): 2-D segsum accumulation into ddt
#   bwd_segsum_dadt_kernel        – dense (fixed-length)
#   bwd_segsum_dadt_kernel_varlen – varlen (see bottom of file)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
    restore_value=["ddt_out_ptr"],

)
@triton.jit
def bwd_segsum_dadt_kernel(
    dSSdA_ptr,          # [B, H, nchunks, C, C]
    SSdA_cs_ptr,          # [B, H, S]
    ddt_out_ptr,        # [B, H, S] - accumulated output
    stride_dSSdA_batch,
    stride_dSSdA_head,
    stride_dSSdA_chunk,
    stride_dSSdA_row,
    stride_dSSdA_col,
    stride_SSdA_batch,
    stride_SSdA_head,
    stride_SSdA_chunk,
    stride_SSdA_row,
    stride_SSdA_col,
    stride_ddt_batch,
    stride_ddt_head,
    stride_ddt_seq,
    B: tl.constexpr,
    H: tl.constexpr,
    nchunks: tl.constexpr,
    C: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Accumulate the inter-token segsum contribution of dSSdA into ddt.

    For each [C×C] chunk block the operation is:
        1. Load dSSdA[chunk] (stored transposed as seq_k × seq_q).
        2. Element-wise multiply by exp(SSdA[chunk]), where SSdA[i,j] =
           dA_cs[i] - dA_cs[j] is the log-decay from token j to token i.
        3. Reverse-cumsum along axis 0 (accumulate contributions from later
           query positions back toward earlier key positions).
        4. Zero the upper triangle (i ≤ j) so only causal pairs contribute.
        5. Row-sum and atomically add into ddt_out[chunk_start : chunk_start+C].

    Grid: (B, H, nchunks).
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    chunk_start = pid_chunk * CHUNK_SIZE
    offs_c = tl.arange(0, CHUNK_SIZE)
    offs_seq = chunk_start + offs_c

    # dSSdA is stored transposed (seq_k × seq_q), so row/col strides are swapped.
    dSSdA_offset = dSSdA_ptr + (pid_batch * stride_dSSdA_batch +
                    pid_head * stride_dSSdA_head +
                    pid_chunk * stride_dSSdA_chunk)
    SSdA_offset = SSdA_cs_ptr + (pid_batch * stride_SSdA_batch +
                    pid_head * stride_SSdA_head +
                    pid_chunk * stride_SSdA_chunk)
    ddt_ptrs = ddt_out_ptr + (pid_batch * stride_ddt_batch +
                    pid_head * stride_ddt_head +
                    offs_seq * stride_ddt_seq)

    # dSSdA is stored as (seq_k × seq_q); swap row/col strides to transpose on load.
    dSSdA_block = tl.load(dSSdA_offset + offs_c[:, None]*stride_dSSdA_col + offs_c[None, :]*stride_dSSdA_row)
    SSdA_block  = tl.load(SSdA_offset  + offs_c[:, None]*stride_SSdA_row  + offs_c[None, :]*stride_SSdA_col)

    dSSdA_block = dSSdA_block * tl.exp(SSdA_block)
    dSSdA_block = tl.cumsum(dSSdA_block, axis=0, reverse=True)

    offs_i = tl.arange(0, CHUNK_SIZE)[:, None]
    offs_j = tl.arange(0, CHUNK_SIZE)[None, :]
    SS_mask = offs_i > offs_j
    dSSdA = tl.where(SS_mask, dSSdA_block, 0.0)

    ddt_chunk = tl.load(ddt_ptrs)
    ddt_chunk += tl.sum(dSSdA, axis=1)
    tl.store(ddt_ptrs, ddt_chunk)


# ============================================================================
# Kernel group 3 (dense): backward through trapezoidal-rule gamma → ddt, dtrap
#   bwd_dtrap_ddt_kernel        – dense (fixed-length)
#   bwd_dtrap_ddt_kernel_varlen – varlen (see bottom of file)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
)
@triton.jit
def bwd_dtrap_ddt_kernel(
    trap_ptr, dt_ptr, dfactor_ptr, dgamma_diag_ptr,
    ddt_ptr, dtrap_ptr, 
    stride_trap_batch, stride_trap_head, stride_trap_seq,
    stride_dt_batch, stride_dt_head, stride_dt_seq,
    stride_dfactor_batch, stride_dfactor_head, stride_dfactor_seq,
    stride_dgamma_diag_batch, stride_dgamma_diag_head, stride_dgamma_diag_seq,
    stride_ddt_batch, stride_ddt_head, stride_ddt_seq,
    stride_dtrap_batch, stride_dtrap_head, stride_dtrap_seq,

    SEQLEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Back-propagate through the trapezoidal-rule gamma parameterisation.

    The forward pass computes:
        gamma[t]  = sigmoid(trap[t]) * dt[t]           (current-token weight)
        sgamma[t] = sigmoid(trap[t+1]) * dt[t+1]       (next-token weight, shifted)
        factor[t] = gamma[t] + sgamma[t]

    This kernel computes ddt and d(trap_presigmoid) given dfactor (= d(factor))
    and dgamma_diag (= d(gamma) from the diagonal path) by unrolling the chain
    rule and applying a right-shift for the cross-token sgamma contribution.

    Grid: (B, H, nchunks).  Each program writes one contiguous CHUNK_SIZE slice.
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    chunk_start = pid_chunk * CHUNK_SIZE
    offs_c = tl.arange(0, CHUNK_SIZE)
    offs_seq = chunk_start + offs_c

    trap_offset = pid_batch*stride_trap_batch + pid_head*stride_trap_head
    dt_offset = pid_batch*stride_dt_batch + pid_head*stride_dt_head
    dfactor_offset = pid_batch*stride_dfactor_batch + pid_head*stride_dfactor_head
    dgamma_diag_offset = pid_batch*stride_dgamma_diag_batch + pid_head*stride_dgamma_diag_head

    strap_block = tl.load(
        trap_ptr + trap_offset + (offs_seq + 1)*stride_trap_seq,
        mask=(offs_seq + 1) < SEQLEN, other=0.0
        )
    sdt_block = tl.load(
        dt_ptr + dt_offset + (offs_seq + 1)*stride_dt_seq,
        mask=(offs_seq + 1) < SEQLEN, other=0.0
    )
    trap_block = tl.load(
        trap_ptr + trap_offset + offs_seq * stride_trap_seq,
        mask=offs_seq < SEQLEN, other=0.0
    )
    dt_block = tl.load(
        dt_ptr + dt_offset + offs_seq * stride_dt_seq,
        mask=offs_seq < SEQLEN, other=0.0
    )
    dfactor_block = tl.load(
        dfactor_ptr + dfactor_offset + offs_seq * stride_dfactor_seq,
        mask=offs_seq < SEQLEN, other=0.0
    )
    dgamma_diag_input_block = tl.load(
        dgamma_diag_ptr + dgamma_diag_offset + offs_seq * stride_dgamma_diag_seq,
        mask=offs_seq < SEQLEN, other=0.0
    )

    dgamma_block  = dfactor_block + dgamma_diag_input_block
    dsgamma_block = dfactor_block

    # Gradients w.r.t. the shifted (next-token) trap/dt values.
    dsdt_block  = tl.sigmoid(-strap_block.to(tl.float32)) * dsgamma_block
    dstrap_block = -sdt_block * dsgamma_block

    # Fetch the last token of the previous chunk to fill position 0 of this chunk
    # after the right-shift (cross-chunk boundary contribution).
    prev_seq  = chunk_start - 1
    prev_mask = prev_seq >= 0
    prev_dgamma  = tl.load(
        dfactor_ptr + dfactor_offset + prev_seq * stride_dfactor_seq,
        mask=prev_mask, other=0.0
    )
    prev_dsgamma = prev_dgamma
    prev_strap = tl.load(
        trap_ptr + trap_offset + chunk_start * stride_trap_seq,
        mask=chunk_start < SEQLEN, other=0.0
    )
    prev_sdt = tl.load(
        dt_ptr + dt_offset + chunk_start * stride_dt_seq,
        mask=chunk_start < SEQLEN, other=0.0
    )
    prev_dsdt = tl.sigmoid(-prev_strap.to(tl.float32)) * prev_dsgamma
    prev_dstrap = -prev_sdt * prev_dsgamma

    # Right-shift within chunk: shifted[i] = original[i-1].
    # Position 0 takes its value from the previous chunk's last token.
    offs_i = tl.arange(0, CHUNK_SIZE)[:, None]
    offs_j = tl.arange(0, CHUNK_SIZE)[None, :]
    shift_mask = offs_i == (offs_j + 1)
    dsdt_shift = tl.sum(tl.where(shift_mask, dsdt_block[None, :], 0.0), axis=1)
    dstrap_shift = tl.sum(tl.where(shift_mask, dstrap_block[None, :], 0.0), axis=1)

    offs = tl.arange(0, CHUNK_SIZE)
    dsdt_shift = tl.where(offs == 0, prev_dsdt, dsdt_shift)
    dstrap_shift = tl.where(offs == 0, prev_dstrap, dstrap_shift)

    # Add dgamma path
    ddt_out = dsdt_shift + dgamma_block * tl.sigmoid(trap_block.to(tl.float32))
    dtrap_out = dstrap_shift + dgamma_block * dt_block 
    dtrap_out *= tl.sigmoid(trap_block.to(tl.float32)) * tl.sigmoid(-trap_block.to(tl.float32)) 

    ddt_ptrs = ddt_ptr + (pid_batch * stride_ddt_batch +
                          pid_head * stride_ddt_head +
                          offs_seq * stride_ddt_seq)
    dtrap_ptrs = dtrap_ptr + (pid_batch * stride_dtrap_batch +
                              pid_head * stride_dtrap_head +
                              offs_seq * stride_dtrap_seq)

    tl.store(ddt_ptrs, ddt_out, mask=offs_seq < SEQLEN)
    tl.store(dtrap_ptrs, dtrap_out, mask=offs_seq < SEQLEN)


# ============================================================================
# Kernel group 4 (varlen): compute da_cs, da_cs_rev, and segsum from da
#   dacs_segsum_kernel        – dense (fixed-length, below)
#   dacs_segsum_kernel_varlen – varlen (this kernel)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
)
@triton.jit
def dacs_segsum_kernel_varlen(
    da_ptr,
    da_cs_ptr,
    da_cs_rev_ptr,
    segsum_ptr,
    cu_seqlens_ptr,
    state_seq_mapping_ptr,
    state_chunk_in_seq_ptr,
    stride_da_batch, stride_da_head, stride_da_seq,
    stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seq,
    stride_da_cs_rev_batch, stride_da_cs_rev_head, stride_da_cs_rev_seq,
    stride_segsum_batch, stride_segsum_head, stride_segsum_chunk,
    stride_segsum_row, stride_segsum_col,
    stride_cu_seqlen, stride_state_seq_mapping, stride_state_chunk_in_seq,
    SEQLEN,
    NUM_SEQUENCES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Varlen version of dacs_segsum_kernel.

    Computes da_cs (forward prefix-sum of da, clipped to ≤ 0), da_cs_rev
    (exclusive reverse prefix-sum, clipped to ≤ 0), and the lower-triangular
    segsum matrix for each chunk, respecting sequence boundaries so that sums
    do not cross from one sequence into another.

    pid_chunk indexes into the global chunk array (length nchunks_global).
    state_seq_mapping and state_chunk_in_seq decode which sequence and local
    chunk position the program is responsible for.  Inactive (padding) slots
    produce an all-False mask and write nothing.

    Grid: (B, H, nchunks_global).
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    curr_seq_ind = tl.load(state_seq_mapping_ptr + pid_chunk * stride_state_seq_mapping)
    local_chunk_ind = tl.load(state_chunk_in_seq_ptr + pid_chunk * stride_state_chunk_in_seq)
    seq_start = tl.load(cu_seqlens_ptr + curr_seq_ind * stride_cu_seqlen)
    seq_end = tl.load(cu_seqlens_ptr + (curr_seq_ind + 1) * stride_cu_seqlen)
    offs = tl.arange(0, CHUNK_SIZE)
    offs_seq = seq_start + local_chunk_ind * CHUNK_SIZE + offs
    mask = (offs_seq < seq_end) & (offs_seq < SEQLEN)

    base_da = pid_batch * stride_da_batch + pid_head * stride_da_head
    da_chunk = tl.load(da_ptr + base_da + offs_seq * stride_da_seq, mask=mask, other=0.0)

    da_cs = tl.cumsum(da_chunk, axis=0)
    da_cs = tl.minimum(da_cs, 0.0)

    da_cs_rev_inclusive = tl.cumsum(da_chunk, axis=0, reverse=True)
    da_cs_rev = da_cs_rev_inclusive - da_chunk
    da_cs_rev = tl.minimum(da_cs_rev, 0.0)

    base_da_cs = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head
    base_da_cs_rev = pid_batch * stride_da_cs_rev_batch + pid_head * stride_da_cs_rev_head
    tl.store(da_cs_ptr + base_da_cs + offs_seq * stride_da_cs_seq, da_cs, mask=mask)
    tl.store(da_cs_rev_ptr + base_da_cs_rev + offs_seq * stride_da_cs_rev_seq, da_cs_rev, mask=mask)

    offs_i = offs[:, None]
    offs_j = offs[None, :]
    segsum = tl.where(offs_i > offs_j, da_chunk[:, None], 0.0)
    segsum = tl.cumsum(segsum, axis=0)
    segsum = tl.minimum(segsum, 0.0)

    base_segsum = (pid_batch * stride_segsum_batch +
                   pid_head * stride_segsum_head +
                   pid_chunk * stride_segsum_chunk)
    tl.store(segsum_ptr + base_segsum + offs_i * stride_segsum_row + offs_j * stride_segsum_col, segsum)


# ============================================================================
# Kernel group 4 (dense): compute da_cs, da_cs_rev, and segsum from da
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
)
@triton.jit
def dacs_segsum_kernel(
    da_ptr,
    da_cs_ptr,
    da_cs_rev_ptr,
    segsum_ptr,
    stride_da_batch, stride_da_head, stride_da_seq,
    stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seq,
    stride_da_cs_rev_batch, stride_da_cs_rev_head, stride_da_cs_rev_seq,
    stride_segsum_batch, stride_segsum_head, stride_segsum_chunk,
    stride_segsum_row, stride_segsum_col,
    SEQLEN: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Dense (fixed-length) version of dacs_segsum_kernel_varlen.

    Computes da_cs, da_cs_rev, and the lower-triangular segsum matrix for each
    contiguous chunk.  Sequences are assumed to be fixed-length and aligned to
    chunk boundaries; no varlen mapping tensors are needed.

    Grid: (B, H, nchunks).
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    chunk_start = pid_chunk * CHUNK_SIZE
    offs = tl.arange(0, CHUNK_SIZE)
    offs_seq = chunk_start + offs
    mask = offs_seq < SEQLEN

    base_da = pid_batch * stride_da_batch + pid_head * stride_da_head
    da_chunk = tl.load(da_ptr + base_da + offs_seq * stride_da_seq, mask=mask, other=0.0)

    da_cs = tl.cumsum(da_chunk, axis=0)
    da_cs = tl.minimum(da_cs, 0.0)

    da_cs_rev = tl.cumsum(da_chunk, axis=0, reverse=True)
    # Roll one to the left:
    i = tl.arange(0, CHUNK_SIZE)[:, None]          # [N,1]
    j = tl.arange(0, CHUNK_SIZE)[None, :]          # [1,N]
    S_mat = (i == j - 1)                      # strictly upper diagonal (one above main)
    da_cs_rev = tl.sum(tl.where(S_mat, da_cs_rev, 0), axis=1)
    da_cs_rev = tl.minimum(da_cs_rev, 0.0)

    base_da_cs = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head
    base_da_cs_rev = pid_batch * stride_da_cs_rev_batch + pid_head * stride_da_cs_rev_head
    tl.store(da_cs_ptr + base_da_cs + offs_seq * stride_da_cs_seq, da_cs, mask=mask)
    tl.store(da_cs_rev_ptr + base_da_cs_rev + offs_seq * stride_da_cs_rev_seq, da_cs_rev, mask=mask)

    broadcasted_indices = tl.zeros_like(offs)
    segsum = tl.load(da_ptr + base_da + offs_seq[:, None] * stride_da_seq + broadcasted_indices[None, :])
    offs_i = offs[:, None]
    offs_j = offs[None, :]
    segsum = tl.where(offs_i > offs_j, segsum, 0.0)
    segsum = tl.cumsum(segsum, axis=0)
    segsum = tl.minimum(segsum, 0.0)

    base_segsum = (pid_batch * stride_segsum_batch +
                   pid_head * stride_segsum_head +
                   pid_chunk * stride_segsum_chunk)
    tl.store(segsum_ptr + base_segsum + offs_i * stride_segsum_row + offs_j * stride_segsum_col, segsum)


# ============================================================================
# Public wrappers – dense (fixed-length sequences)
# ============================================================================

def bwd_dadt_fused_triton(
    dSSdA: torch.Tensor,              # [B, H, nchunks, C, C]
    SSdA: torch.Tensor,               # [B, H, nchunks, C, C]
    ddA_cs: torch.Tensor,             # [B, H, S]
    ddA_cs_rev: torch.Tensor,         # [B, H, S]
    dA_cs: torch.Tensor,              # [B, H, S]
    dA_cs_rev: torch.Tensor,          # [B, H, S]
    chunk_size: int,
) -> torch.Tensor:
    """Fuse the three ddt backward contributions for fixed-length sequences.

    Runs bwd_dadt_cumsum_fused_kernel (cumsum terms) and bwd_segsum_dadt_kernel
    (segsum term) back-to-back and returns their sum.  The result must be
    multiplied by NEG_LOG2E = -log2(e) ≈ -1.4427 by the caller to obtain the
    final ddt in log2-decay space.

    S must be a multiple of chunk_size.

    Returns
    -------
    dadt_out : float32 tensor of shape [B, H, S]
    """
    B, H, S = ddA_cs.shape
    nchunks = S // chunk_size
    assert S % chunk_size == 0, f"Sequence length {S} must be divisible by chunk_size {chunk_size}"
    assert dSSdA.shape == (B, H, nchunks, chunk_size, chunk_size), \
        f"dSSdA shape mismatch: expected {(B, H, nchunks, chunk_size, chunk_size)}, got {dSSdA.shape}"

    # Initialize output tensor
    dadt_out = torch.zeros(B, H, S, device=ddA_cs.device, dtype=torch.float32)

    # Kernel 1: Fused ddA_cs and ddA_cs_rev contributions
    grid1 = (B, H, nchunks)
    bwd_dadt_cumsum_fused_kernel[grid1](
        ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, dadt_out,
        ddA_cs.stride(0), ddA_cs.stride(1), ddA_cs.stride(2),
        B, H, S,
        CHUNK_SIZE=chunk_size,
    )

    # Kernel 2: dSSdA segsum contribution
    grid2 = (B, H, nchunks)
    bwd_segsum_dadt_kernel[grid2](
        dSSdA, SSdA, dadt_out,
        dSSdA.stride(0), dSSdA.stride(1), dSSdA.stride(2),
        dSSdA.stride(3), dSSdA.stride(4),
        SSdA.stride(0), SSdA.stride(1), SSdA.stride(2),
        SSdA.stride(3), SSdA.stride(4),
        dadt_out.stride(0), dadt_out.stride(1), dadt_out.stride(2),
        B, H, nchunks, chunk_size,
        CHUNK_SIZE=chunk_size,
    )

    return dadt_out

def bwd_dtrap_ddt_triton(
    trap: torch.Tensor,           # [B, H, S]  – trap_presigmoid values
    dt: torch.Tensor,             # [B, H, S]
    dfactor: torch.Tensor,        # [B, H, S]  – upstream gradient of factor
    dgamma_diag: torch.Tensor,    # [B, H, S]  – upstream gradient of gamma (diagonal)
    chunk_size: int,              # may differ from the chunk_size used by bwd_dadt_fused_triton
):
    """Back-propagate through the trapezoidal gamma parameterisation (dense).

    Returns
    -------
    ddt   : tensor same shape and dtype as dt
    dtrap : tensor same shape and dtype as trap  (gradient w.r.t. trap_presigmoid)
    """
    B, H, S = dt.shape
    nchunks = S // chunk_size

    ddt = torch.zeros_like(dt)
    dtrap = torch.zeros_like(trap)

    grid = (B, H, nchunks)
    bwd_dtrap_ddt_kernel[grid](
        trap, dt, dfactor, dgamma_diag,
        ddt, dtrap,
        trap.stride(0), trap.stride(1), trap.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        dfactor.stride(0), dfactor.stride(1), dfactor.stride(2),
        dgamma_diag.stride(0), dgamma_diag.stride(1), dgamma_diag.stride(2),
        ddt.stride(0), ddt.stride(1), ddt.stride(2),
        dtrap.stride(0), dtrap.stride(1), dtrap.stride(2),
        S,
        chunk_size,
    )
    return ddt, dtrap

def compute_dacs_segsum_triton_varlen(
    da: torch.Tensor,              # [B, H, S]
    chunk_size: int,
    cu_seqlens: torch.Tensor = None,   # [NS+1], int32
):
    """Compute da_cs, da_cs_rev, and segsum for variable-length packed sequences.

    Falls back to compute_dacs_segsum_triton (dense) when cu_seqlens is None.

    Parameters
    ----------
    da           : decay increments, shape [B, H, S].
    chunk_size   : tokens per chunk (must be a power of two).
    cu_seqlens   : cumulative sequence lengths, shape [NS+1], starting at 0.

    Returns
    -------
    da_cs        : forward inclusive prefix-sum of da, clipped to ≤ 0. [B, H, S]
    da_cs_rev    : exclusive reverse prefix-sum of da, clipped to ≤ 0. [B, H, S]
    segsum       : lower-triangular intra-chunk segsum. [B, H, nchunks_global, C, C]
                   nchunks_global = (S // chunk_size) + num_sequences.
    """
    if cu_seqlens is None:
        return compute_dacs_segsum_triton(da, chunk_size)

    B, H, S = da.shape
    assert cu_seqlens.ndim == 1, f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens.shape)}"
    num_sequences = max(int(cu_seqlens.numel()) - 1, 0)
    if num_sequences == 0:
        return compute_dacs_segsum_triton(da, chunk_size)

    # Global chunk count: each sequence contributes ceil(len/chunk_size) chunks,
    # which equals (len // chunk_size) + 1 under the "always +1" convention used
    # throughout this module.  Summing over all sequences gives:
    #     nchunks = (S // chunk_size) + num_sequences
    nchunks = (S // chunk_size) + num_sequences

    da_cs = torch.empty_like(da)
    da_cs_rev = torch.empty_like(da)
    segsum = torch.zeros(B, H, nchunks, chunk_size, chunk_size, device=da.device, dtype=da.dtype)

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks_per_seq = (seq_lens // chunk_size) + 1

    # Build mapping tensors: both have length nchunks_global.
    # Inactive (padding) slots are given a sentinel local-chunk index that
    # places chunk_start >= seq_end, making the kernel mask all-False.
    state_seq_mapping  = torch.zeros(nchunks, dtype=torch.int32, device=da.device)
    state_chunk_in_seq = torch.zeros(nchunks, dtype=torch.int32, device=da.device)
    default_seq_len = int(seq_lens[0].item()) if num_sequences > 0 else 0
    default_inactive_local_chunk = (default_seq_len + chunk_size - 1) // chunk_size
    state_chunk_in_seq.fill_(default_inactive_local_chunk)

    for i in range(num_sequences):
        start = int(cu_seqlens[i].item())
        n = int(chunks_per_seq[i].item())
        chunk_start = (start // chunk_size) + i
        chunk_end = chunk_start + n
        assert chunk_end <= nchunks, (
            f"Chunk mapping overflow for seq {i}: [{chunk_start}, {chunk_end}) vs nchunks={nchunks}"
        )
        state_seq_mapping[chunk_start:chunk_end] = i
        state_chunk_in_seq[chunk_start:chunk_end] = torch.arange(n, dtype=torch.int32, device=da.device)

    grid = (B, H, nchunks)
    dacs_segsum_kernel_varlen[grid](
        da, da_cs, da_cs_rev, segsum, cu_seqlens, state_seq_mapping, state_chunk_in_seq,
        da.stride(0), da.stride(1), da.stride(2),
        da_cs.stride(0), da_cs.stride(1), da_cs.stride(2),
        da_cs_rev.stride(0), da_cs_rev.stride(1), da_cs_rev.stride(2),
        segsum.stride(0), segsum.stride(1), segsum.stride(2),
        segsum.stride(3), segsum.stride(4),
        cu_seqlens.stride(0), state_seq_mapping.stride(0), state_chunk_in_seq.stride(0),
        S,
        num_sequences,
        chunk_size,
    )

    return da_cs, da_cs_rev, segsum


def compute_dacs_segsum_triton(
    da: torch.Tensor,  # [B, H, S]
    chunk_size: int,
):
    """Compute da_cs, da_cs_rev, and segsum for fixed-length sequences.

    Returns
    -------
    da_cs    : forward inclusive prefix-sum of da, clipped to ≤ 0. [B, H, S]
    da_cs_rev: exclusive reverse prefix-sum of da, clipped to ≤ 0. [B, H, S]
    segsum   : lower-triangular intra-chunk segsum. [B, H, nchunks, C, C]
    """
    B, H, S = da.shape
    nchunks = (S + chunk_size - 1) // chunk_size

    da_cs = torch.empty_like(da)
    da_cs_rev = torch.empty_like(da)
    segsum = torch.empty(B, H, nchunks, chunk_size, chunk_size, device=da.device, dtype=da.dtype)

    grid = (B, H, nchunks)
    dacs_segsum_kernel[grid](
        da, da_cs, da_cs_rev, segsum,
        da.stride(0), da.stride(1), da.stride(2),
        da_cs.stride(0), da_cs.stride(1), da_cs.stride(2),
        da_cs_rev.stride(0), da_cs_rev.stride(1), da_cs_rev.stride(2),
        segsum.stride(0), segsum.stride(1), segsum.stride(2),
        segsum.stride(3), segsum.stride(4),
        S,
        chunk_size,
    )

    return da_cs, da_cs_rev, segsum


# ============================================================================
# Reference implementations (pure PyTorch, used for correctness tests)
# Dense versions: bwd_segsum_ddt_from_dSSdA_ref, bwd_ddt_from_ddA_cs_rev_ref,
#                 bwd_ddt_from_ddA_cs_ref, compute_dtrap_ddt_ref,
#                 compute_dacs_segsum_ref
# Varlen versions: bwd_dadt_fused_varlen_ref, compute_dtrap_ddt_varlen_ref,
#                  compute_dacs_segsum_ref_varlen
# ============================================================================

def bwd_segsum_ddt_from_dSSdA_ref(
    dSSdA: torch.Tensor,   # [B, H, nchunks, C, C]
    dA_cs: torch.Tensor,   # [B, H, S]
    chunk_size: int,
) -> torch.Tensor:         # [B, H, S]
    """Dense reference for the segsum contribution to ddt.

    Computes the row-sum of exp(dA_cs[i]-dA_cs[j]) * dSSdA[j,i] for i > j
    within each chunk, then applies a reverse cumsum so that each token i
    aggregates contributions from all later tokens j > i in the chunk.
    Result is scaled by -log2(e) to match the log2-decay convention.
    """
    B, H, nchunks, C, C_ = dSSdA.shape
    assert C == chunk_size == C_
    dA_cs_chunk = dA_cs.view(B, H, nchunks, C)
    dSSdA = dSSdA.permute([0, 1, 2, 4, 3])
    seg = dA_cs_chunk[..., :, None] - dA_cs_chunk[..., None, :]
    dSSdA = dSSdA * torch.exp(seg)
    ddA = torch.flip(torch.cumsum(torch.flip(dSSdA, dims=[-2]), dim=-2), dims=[-2])
    mask = torch.tril(torch.ones(C, C, device=dSSdA.device, dtype=dSSdA.dtype), -1)
    ddA = ddA * mask
    ddA = ddA.sum(-1)
    ddt = ddA * (-math.log2(math.e))
    return ddt.reshape(B, H, nchunks*C)


def bwd_ddt_from_ddA_cs_rev_ref(
    ddA_cs_rev: torch.Tensor,  # [B, H, S]
    dA_cs_rev: torch.Tensor,   # [B, H, S]
    chunk_size: int,
) -> torch.Tensor:             # [B, H, S]
    """Dense reference for the forward-exclusive-cumsum contribution to ddt.

    Propagates ddA_cs_rev through exp(dA_cs_rev) and a forward exclusive
    cumsum within each chunk.  Result is scaled by -log2(e).
    """
    B, H, S = ddA_cs_rev.shape
    nchunks = S // chunk_size
    ddA_cs_rev = torch.exp(dA_cs_rev) * ddA_cs_rev
    dA_cs_rev = dA_cs_rev.view(B, H, nchunks, chunk_size)
    ddA_cs_rev = ddA_cs_rev.view(B, H, nchunks, chunk_size)
    ddA = torch.cumsum(ddA_cs_rev, dim=-1)
    ddA = torch.cat([torch.zeros_like(ddA[..., :1]), ddA[..., :-1]], dim=-1)
    ddt = ddA * (-math.log2(math.e))
    return ddt.reshape(B, H, nchunks*chunk_size)

 
def bwd_ddt_from_ddA_cs_ref(
    ddA_cs: torch.Tensor,  # [B, H, S]
    dA_cs: torch.Tensor,   # [B, H, S]
    chunk_size: int,
) -> torch.Tensor:         # [B, H, S]
    """Dense reference for the reverse-cumsum contribution to ddt.

    Propagates ddA_cs through exp(dA_cs) and a reverse cumsum within each
    chunk.  Result is scaled by -log2(e).
    """
    B, H, S = ddA_cs.shape
    nchunks = S // chunk_size
    ddA_cs =  torch.exp(dA_cs) * ddA_cs
    dA_cs = dA_cs.view(B, H, nchunks, chunk_size)
    ddA_cs = ddA_cs.view(B, H, nchunks, chunk_size)
    ddA = torch.flip(torch.cumsum(torch.flip(ddA_cs, dims=[-1]), dim=-1), dims=[-1])
    ddt = ddA * (-math.log2(math.e))
    return ddt.reshape(B, H, nchunks*chunk_size)

def compute_dtrap_ddt_ref(
    dfactor: torch.Tensor,          # [B, H, S]
    dgamma_diag_input: torch.Tensor, # [B, H, S]
    trap_presigmoid: torch.Tensor,  # [B, H, S]
    dt: torch.Tensor,               # [B, H, S]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense reference for bwd_dtrap_ddt_triton.

    Returns (ddt, dtrap_presigmoid), both shaped [B, H, S].
    """
    trap   = torch.nn.functional.sigmoid(trap_presigmoid)
    strap  = torch.nn.functional.pad(trap[:, :, 1:], (0, 1), value=0.0)
    sdt    = torch.nn.functional.pad(dt[:, :, 1:],   (0, 1), value=0.0)
    dgamma  = dfactor.detach().clone() + dgamma_diag_input.detach().clone()
    dsgamma = dfactor.detach().clone()
    dsdt   = (1 - strap) * dsgamma
    dstrap = -sdt * dsgamma
    # Right-shift by one: gradient from sgamma[t] lands on position t-1.
    ddt = torch.nn.functional.pad(dsdt[:, :, :-1], (1, 0), value=0.0)
    dtrap = torch.nn.functional.pad(dstrap[:, :, :-1], (1, 0), value=0.0)
    # Add the dgamma path:
    dtrap += dgamma*dt
    # grad of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    dtrap *= trap * torch.nn.functional.sigmoid(-trap_presigmoid)
    ddt += dgamma*trap
    return ddt, dtrap

def compute_dacs_segsum_ref(
    da: torch.Tensor,  # [B, H, S]
    chunk_size: int,
):
    """Dense reference for compute_dacs_segsum_triton.

    Requires S to be a multiple of chunk_size.  Returns (da_cs, da_cs_rev, segsum).
    """
    from einops import repeat
    B, H, S = da.shape
    nchunks = S // chunk_size

    da_reshaped = da.view(B, H, nchunks, chunk_size)
    da_cs = torch.cumsum(da_reshaped, dim=-1)
    da_cs_sum = torch.sum(da_reshaped, dim=-1)
    da_cs_rev = da_cs_sum[..., None] - da_cs

    segsum = repeat(da_reshaped, "... d -> ... d e", e=chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=da_cs.device, dtype=bool), diagonal=-1)
    segsum = segsum.masked_fill(~mask, 0)
    segsum = torch.cumsum(segsum, dim=-2)

    return da_cs.view(B, H, S), da_cs_rev.view(B, H, S), segsum


def compute_dacs_segsum_ref_varlen(
    da: torch.Tensor,              # [B, H, S]
    chunk_size: int,
    cu_seqlens: torch.Tensor,      # [NS+1]
    num_sequences: int,
):
    """Varlen reference for compute_dacs_segsum_triton_varlen.

    Calls compute_dacs_segsum_ref as a subroutine per sequence, zero-padding
    each sequence to the nearest chunk_size multiple.  Uses the same global
    chunk layout as _build_varlen_chunk_mapping so the segsum output is directly
    comparable to the Triton kernel output.
    """
    from einops import repeat

    B, H, S = da.shape
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks_per_seq = (seq_lens // chunk_size) + 1
    nchunks_global = (S // chunk_size) + num_sequences

    da_cs = torch.empty(B, H, S, device=da.device, dtype=da.dtype)
    da_cs_rev = torch.empty(B, H, S, device=da.device, dtype=da.dtype)
    segsum = torch.zeros(B, H, nchunks_global, chunk_size, chunk_size, device=da.device, dtype=da.dtype)

    for i in range(num_sequences):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        curr_seqlen = end - start
        curr_nchunks = int(chunks_per_seq[i].item())
        curr_padded_len = curr_nchunks * chunk_size
        # Global chunk slot for sequence i (matches _build_varlen_chunk_mapping)
        global_chunk_start = (start // chunk_size) + i

        da_padded = torch.zeros(B, H, curr_padded_len, device=da.device, dtype=da.dtype)
        if curr_seqlen > 0:
            da_padded[:, :, :curr_seqlen] = da[:, :, start:end]

        # Inline of compute_dacs_segsum_ref (dense, non-varlen).
        da_reshaped = da_padded.view(B, H, curr_nchunks, chunk_size)
        da_cs_seq = torch.cumsum(da_reshaped, dim=-1)
        da_cs_sum = torch.sum(da_reshaped, dim=-1)
        da_cs_rev_seq = da_cs_sum[..., None] - da_cs_seq
        segsum_seq = repeat(da_reshaped, "... d -> ... d e", e=chunk_size)
        tril_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=da.device, dtype=torch.bool), diagonal=-1)
        segsum_seq = segsum_seq.masked_fill(~tril_mask, 0)
        segsum_seq = torch.cumsum(segsum_seq, dim=-2)

        if curr_seqlen > 0:
            da_cs    [:, :, start:end] = da_cs_seq    .view(B, H, curr_padded_len)[:, :, :curr_seqlen]
            da_cs_rev[:, :, start:end] = da_cs_rev_seq.view(B, H, curr_padded_len)[:, :, :curr_seqlen]
        segsum[:, :, global_chunk_start:global_chunk_start + curr_nchunks, :, :] = segsum_seq

    return da_cs, da_cs_rev, segsum


def bwd_dadt_fused_varlen_ref(
    dSSdA: torch.Tensor,       # [B, H, nchunks_global, C, C]
    ddA_cs: torch.Tensor,      # [B, H, S]
    ddA_cs_rev: torch.Tensor,  # [B, H, S]
    dA_cs: torch.Tensor,       # [B, H, S]
    dA_cs_rev: torch.Tensor,   # [B, H, S]
    chunk_size: int,
    cu_seqlens: torch.Tensor,  # [NS+1]
) -> torch.Tensor:
    """Varlen reference for bwd_dadt_fused_triton_varlen.

    Calls the non-varlen reference functions as subroutines, one sequence at a time.
    Each sequence is zero-padded to the nearest chunk_size multiple before the call,
    and only the valid (non-padded) positions are written to the output.
    """
    B, H, S = ddA_cs.shape
    num_sequences = int(cu_seqlens.numel()) - 1
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks_per_seq = (seq_lens // chunk_size) + 1

    dadt_out = torch.zeros(B, H, S, device=ddA_cs.device, dtype=torch.float32)

    for i in range(num_sequences):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        curr_seqlen = end - start
        if curr_seqlen == 0:
            continue
        curr_nchunks = int(chunks_per_seq[i].item())
        curr_padded_len = curr_nchunks * chunk_size
        # Global chunk slot for the first chunk of this sequence (matches _build_varlen_chunk_mapping)
        global_chunk_start = (start // chunk_size) + i

        def _pad(x):
            padded = torch.zeros(B, H, curr_padded_len, device=x.device, dtype=x.dtype)
            padded[:, :, :curr_seqlen] = x[:, :, start:end]
            return padded

        ddA_cs_seq    = _pad(ddA_cs)
        ddA_cs_rev_seq = _pad(ddA_cs_rev)
        dA_cs_seq     = _pad(dA_cs)
        dA_cs_rev_seq  = _pad(dA_cs_rev)
        dSSdA_seq = dSSdA[:, :, global_chunk_start:global_chunk_start + curr_nchunks, :, :]

        ddt1 = bwd_segsum_ddt_from_dSSdA_ref(dSSdA_seq, dA_cs_seq, chunk_size)
        ddt2 = bwd_ddt_from_ddA_cs_rev_ref(ddA_cs_rev_seq, dA_cs_rev_seq, chunk_size)
        ddt3 = bwd_ddt_from_ddA_cs_ref(ddA_cs_seq, dA_cs_seq, chunk_size)
        dadt_out[:, :, start:end] = (ddt1 + ddt2 + ddt3)[:, :, :curr_seqlen]

    return dadt_out


def compute_dtrap_ddt_varlen_ref(
    dfactor: torch.Tensor,          # [B, H, S]
    dgamma_diag_input: torch.Tensor, # [B, H, S]
    trap_presigmoid: torch.Tensor,  # [B, H, S]
    dt: torch.Tensor,               # [B, H, S]
    chunk_size: int,
    cu_seqlens: torch.Tensor,       # [NS+1]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Varlen reference for bwd_dtrap_ddt_triton_varlen.

    Calls compute_dtrap_ddt_ref as subroutine per sequence, zero-padding each
    sequence to a chunk_size multiple and zeroing the cross-sequence boundary
    shift (first position of each sequence gets zero incoming gradient, matching
    the varlen kernel's is_first_chunk_in_seq guard).
    """
    B, H, S = dt.shape
    num_sequences = int(cu_seqlens.numel()) - 1
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks_per_seq = (seq_lens // chunk_size) + 1

    ddt_out   = torch.zeros(B, H, S, device=dt.device, dtype=dt.dtype)
    dtrap_out = torch.zeros(B, H, S, device=dt.device, dtype=dt.dtype)

    for i in range(num_sequences):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        curr_seqlen = end - start
        if curr_seqlen == 0:
            continue
        curr_nchunks = int(chunks_per_seq[i].item())
        curr_padded_len = curr_nchunks * chunk_size

        def _pad(x):
            padded = torch.zeros(B, H, curr_padded_len, device=x.device, dtype=x.dtype)
            padded[:, :, :curr_seqlen] = x[:, :, start:end]
            return padded

        ddt_seq, dtrap_seq = compute_dtrap_ddt_ref(
            _pad(dfactor), _pad(dgamma_diag_input), _pad(trap_presigmoid), _pad(dt)
        )
        ddt_out  [:, :, start:end] = ddt_seq  [:, :, :curr_seqlen]
        dtrap_out[:, :, start:end] = dtrap_seq[:, :, :curr_seqlen]

    return ddt_out, dtrap_out


# ============================================================================
# Public wrappers – varlen (variable-length sequences packed end-to-end)
# Kernel groups 1-3 varlen implementations.  Group 4 varlen kernel is above.
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
    restore_value=["ddt_out_ptr"],
)
@triton.jit
def bwd_dadt_cumsum_fused_kernel_varlen(
    ddA_cs_ptr,         # [B, H, S]
    ddA_cs_rev_ptr,     # [B, H, S]
    dA_cs_ptr,          # [B, H, S]
    dA_cs_rev_ptr,      # [B, H, S]
    ddt_out_ptr,        # [B, H, S] - output (accumulate)
    cu_seqlens_ptr,     # [NS+1]
    state_seq_mapping_ptr,    # [nchunks_global]
    state_chunk_in_seq_ptr,   # [nchunks_global]
    stride_batch,
    stride_head,
    stride_seq,
    stride_cu,
    stride_ssm,
    stride_scis,
    S: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Varlen version of bwd_dadt_cumsum_fused_kernel.
    Grid: (B, H, nchunks_global).  Each program handles one global chunk slot.
    """
    pid_batch = tl.program_id(0)
    pid_head  = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    # Decode global chunk → sequence start and local chunk index.
    curr_seq_ind    = tl.load(state_seq_mapping_ptr  + pid_chunk * stride_ssm)
    local_chunk_ind = tl.load(state_chunk_in_seq_ptr + pid_chunk * stride_scis)
    seq_start = tl.load(cu_seqlens_ptr + curr_seq_ind       * stride_cu)
    seq_end   = tl.load(cu_seqlens_ptr + (curr_seq_ind + 1) * stride_cu)

    chunk_start = seq_start + local_chunk_ind * CHUNK_SIZE
    offs        = tl.arange(0, CHUNK_SIZE)
    offs_seq    = chunk_start + offs
    mask        = (offs_seq < seq_end) & (offs_seq < S)

    base = pid_batch * stride_batch + pid_head * stride_head

    ddA_cs     = tl.load(ddA_cs_ptr     + base + offs_seq * stride_seq, mask=mask, other=0.0)
    ddA_cs_rev = tl.load(ddA_cs_rev_ptr + base + offs_seq * stride_seq, mask=mask, other=0.0)
    dA_cs      = tl.load(dA_cs_ptr      + base + offs_seq * stride_seq, mask=mask, other=0.0)
    dA_cs_rev  = tl.load(dA_cs_rev_ptr  + base + offs_seq * stride_seq, mask=mask, other=0.0)

    # Reverse cumsum contribution from ddA_cs
    scaled_ddA_cs = tl.exp(dA_cs) * ddA_cs
    ddt_cs = tl.cumsum(scaled_ddA_cs, axis=0, reverse=True)

    # Forward exclusive cumsum contribution from ddA_cs_rev
    scaled_ddA_cs_rev = tl.exp(dA_cs_rev) * ddA_cs_rev
    ddt_cs_rev_inclusive = tl.cumsum(scaled_ddA_cs_rev, axis=0)
    i = tl.arange(0, CHUNK_SIZE)[:, None]
    j = tl.arange(0, CHUNK_SIZE)[None, :]
    S_mat = (i == j + 1)
    ddt_cs_rev_exclusive = tl.sum(tl.where(S_mat, ddt_cs_rev_inclusive, 0), axis=1)

    ddt_total = ddt_cs + ddt_cs_rev_exclusive
    tl.store(ddt_out_ptr + base + offs_seq * stride_seq, ddt_total, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
    restore_value=["ddt_out_ptr"],
)
@triton.jit
def bwd_segsum_dadt_kernel_varlen(
    dSSdA_ptr,          # [B, H, nchunks_global, C, C]
    SSdA_cs_ptr,        # [B, H, nchunks_global, C, C]
    ddt_out_ptr,        # [B, H, S] - accumulated output
    cu_seqlens_ptr,     # [NS+1]
    state_seq_mapping_ptr,    # [nchunks_global]
    state_chunk_in_seq_ptr,   # [nchunks_global]
    stride_dSSdA_batch, stride_dSSdA_head, stride_dSSdA_chunk,
    stride_dSSdA_row,   stride_dSSdA_col,
    stride_SSdA_batch,  stride_SSdA_head,  stride_SSdA_chunk,
    stride_SSdA_row,    stride_SSdA_col,
    stride_ddt_batch,   stride_ddt_head,   stride_ddt_seq,
    stride_cu, stride_ssm, stride_scis,
    S: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Varlen version of bwd_segsum_dadt_kernel.
    Grid: (B, H, nchunks_global).
    """
    pid_batch = tl.program_id(0)
    pid_head  = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    curr_seq_ind    = tl.load(state_seq_mapping_ptr  + pid_chunk * stride_ssm)
    local_chunk_ind = tl.load(state_chunk_in_seq_ptr + pid_chunk * stride_scis)
    seq_start = tl.load(cu_seqlens_ptr + curr_seq_ind       * stride_cu)
    seq_end   = tl.load(cu_seqlens_ptr + (curr_seq_ind + 1) * stride_cu)

    chunk_start = seq_start + local_chunk_ind * CHUNK_SIZE
    offs_c   = tl.arange(0, CHUNK_SIZE)
    offs_seq = chunk_start + offs_c
    mask_seq = (offs_seq < seq_end) & (offs_seq < S)

    dSSdA_off = (pid_batch * stride_dSSdA_batch + pid_head * stride_dSSdA_head +
                 pid_chunk * stride_dSSdA_chunk)
    SSdA_off  = (pid_batch * stride_SSdA_batch  + pid_head * stride_SSdA_head  +
                 pid_chunk * stride_SSdA_chunk)
    ddt_ptrs = ddt_out_ptr + (pid_batch * stride_ddt_batch +
                               pid_head  * stride_ddt_head  +
                               offs_seq  * stride_ddt_seq)

    # NOTE: dSSdA stored as (seq_k x seq_q) – transpose indices match non-varlen version.
    dSSdA_block = tl.load(dSSdA_off + dSSdA_ptr +
                          offs_c[:, None] * stride_dSSdA_col +
                          offs_c[None, :] * stride_dSSdA_row)
    SSdA_block  = tl.load(SSdA_off  + SSdA_cs_ptr  +
                           offs_c[:, None] * stride_SSdA_row +
                           offs_c[None, :] * stride_SSdA_col)

    dSSdA_block = dSSdA_block * tl.exp(SSdA_block)
    dSSdA_block = tl.cumsum(dSSdA_block, axis=0, reverse=True)

    offs_i = tl.arange(0, CHUNK_SIZE)[:, None]
    offs_j = tl.arange(0, CHUNK_SIZE)[None, :]
    SS_mask = offs_i > offs_j
    dSSdA   = tl.where(SS_mask, dSSdA_block, 0.0)

    ddt_chunk = tl.load(ddt_ptrs, mask=mask_seq, other=0.0)
    ddt_chunk += tl.sum(dSSdA, axis=1)
    tl.store(ddt_ptrs, ddt_chunk, mask=mask_seq)


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
)
@triton.jit
def bwd_dtrap_ddt_kernel_varlen(
    trap_ptr, dt_ptr, dfactor_ptr, dgamma_diag_ptr,
    ddt_ptr, dtrap_ptr,
    cu_seqlens_ptr,
    state_seq_mapping_ptr,
    state_chunk_in_seq_ptr,
    stride_trap_batch, stride_trap_head, stride_trap_seq,
    stride_dt_batch,   stride_dt_head,   stride_dt_seq,
    stride_dfactor_batch, stride_dfactor_head, stride_dfactor_seq,
    stride_dgamma_batch,  stride_dgamma_head,  stride_dgamma_seq,
    stride_ddt_batch,  stride_ddt_head,  stride_ddt_seq,
    stride_dtrap_batch, stride_dtrap_head, stride_dtrap_seq,
    stride_cu, stride_ssm, stride_scis,
    S: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Varlen version of bwd_dtrap_ddt_kernel.
    Grid: (B, H, nchunks_global).
    """
    pid_batch = tl.program_id(0)
    pid_head  = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    curr_seq_ind    = tl.load(state_seq_mapping_ptr  + pid_chunk * stride_ssm)
    local_chunk_ind = tl.load(state_chunk_in_seq_ptr + pid_chunk * stride_scis)
    seq_start = tl.load(cu_seqlens_ptr + curr_seq_ind       * stride_cu)
    seq_end   = tl.load(cu_seqlens_ptr + (curr_seq_ind + 1) * stride_cu)
    is_first_chunk_in_seq = (local_chunk_ind == 0)

    chunk_start = seq_start + local_chunk_ind * CHUNK_SIZE
    offs_c   = tl.arange(0, CHUNK_SIZE)
    offs_seq = chunk_start + offs_c
    mask_seq = (offs_seq < seq_end) & (offs_seq < S)

    trap_off = pid_batch * stride_trap_batch + pid_head * stride_trap_head
    dt_off   = pid_batch * stride_dt_batch   + pid_head * stride_dt_head
    dfactor_off  = pid_batch * stride_dfactor_batch  + pid_head * stride_dfactor_head
    dgamma_off   = pid_batch * stride_dgamma_batch   + pid_head * stride_dgamma_head

    # Shifted (next-token) trap and dt – zero out at seq boundary.
    shift_mask = (offs_seq + 1 < seq_end) & (offs_seq + 1 < S)
    strap_block = tl.load(trap_ptr + trap_off + (offs_seq + 1) * stride_trap_seq,
                          mask=shift_mask, other=0.0)
    sdt_block   = tl.load(dt_ptr   + dt_off   + (offs_seq + 1) * stride_dt_seq,
                          mask=shift_mask, other=0.0)
    trap_block  = tl.load(trap_ptr + trap_off + offs_seq * stride_trap_seq,
                          mask=mask_seq, other=0.0)
    dt_block    = tl.load(dt_ptr   + dt_off   + offs_seq * stride_dt_seq,
                          mask=mask_seq, other=0.0)
    dfactor_block     = tl.load(dfactor_ptr  + dfactor_off  + offs_seq * stride_dfactor_seq,
                                mask=mask_seq, other=0.0)
    dgamma_diag_block = tl.load(dgamma_diag_ptr + dgamma_off + offs_seq * stride_dgamma_seq,
                                mask=mask_seq, other=0.0)

    dgamma_block  = dfactor_block + dgamma_diag_block
    dsgamma_block = dfactor_block

    dsdt_block  = tl.sigmoid(-strap_block.to(tl.float32)) * dsgamma_block
    dstrap_block = -sdt_block * dsgamma_block

    # Cross-chunk: the first position of this chunk gets its shifted gradient from
    # the previous chunk's last position – UNLESS this is the first chunk of a sequence.
    prev_seq_valid = chunk_start > 0 and not is_first_chunk_in_seq
    prev_dgamma = tl.load(
        dfactor_ptr + dfactor_off + (chunk_start - 1) * stride_dfactor_seq,
        mask=prev_seq_valid, other=0.0,
    )
    prev_dsgamma = prev_dgamma
    prev_strap = tl.load(trap_ptr + trap_off + chunk_start * stride_trap_seq,
                         mask=(chunk_start < S) and not is_first_chunk_in_seq, other=0.0)
    prev_sdt   = tl.load(dt_ptr   + dt_off   + chunk_start * stride_dt_seq,
                         mask=(chunk_start < S) and not is_first_chunk_in_seq, other=0.0)
    prev_dsdt  = tl.sigmoid(-prev_strap.to(tl.float32)) * prev_dsgamma
    prev_dstrap = -prev_sdt * prev_dsgamma

    offs_i = tl.arange(0, CHUNK_SIZE)[:, None]
    offs_j = tl.arange(0, CHUNK_SIZE)[None, :]
    shift_mask_mat = offs_i == (offs_j + 1)
    dsdt_shift  = tl.sum(tl.where(shift_mask_mat, dsdt_block [None, :], 0.0), axis=1)
    dstrap_shift = tl.sum(tl.where(shift_mask_mat, dstrap_block[None, :], 0.0), axis=1)

    offs = tl.arange(0, CHUNK_SIZE)
    dsdt_shift  = tl.where(offs == 0, prev_dsdt,  dsdt_shift)
    dstrap_shift = tl.where(offs == 0, prev_dstrap, dstrap_shift)

    ddt_out  = dsdt_shift + dgamma_block * tl.sigmoid(trap_block.to(tl.float32))
    dtrap_out = dstrap_shift + dgamma_block * dt_block
    dtrap_out *= tl.sigmoid(trap_block.to(tl.float32)) * tl.sigmoid(-trap_block.to(tl.float32))

    ddt_ptrs  = ddt_ptr  + (pid_batch * stride_ddt_batch  + pid_head * stride_ddt_head  + offs_seq * stride_ddt_seq)
    dtrap_ptrs = dtrap_ptr + (pid_batch * stride_dtrap_batch + pid_head * stride_dtrap_head + offs_seq * stride_dtrap_seq)
    tl.store(ddt_ptrs,  ddt_out,  mask=mask_seq)
    tl.store(dtrap_ptrs, dtrap_out, mask=mask_seq)


def _build_varlen_chunk_mapping(cu_seqlens: torch.Tensor, chunk_size: int):
    """Build the three varlen indexing primitives consumed by every varlen kernel.

    Because sequences have different lengths, chunks do not cross sequence
    boundaries — the last chunk of each sequence may be shorter than
    chunk_size.  All chunks across all sequences are laid out in a single flat
    "global" array of length nchunks_global, and the two mapping arrays let
    each kernel thread locate the sequence and position it belongs to.

    Returns
    -------
    nchunks_global : int
        Total number of chunk slots across all sequences:
            nchunks_global = (S // chunk_size) + num_sequences
        The additive num_sequences term accounts for the partial last chunk of
        each sequence.  Kernel grids are launched with nchunks_global programs
        along the chunk dimension.

    state_seq_mapping : int32 tensor, shape [nchunks_global]
        Maps each global chunk index to the sequence it belongs to.
        state_seq_mapping[pid_chunk] == i  means chunk pid_chunk is part of
        sequence i.  Inactive (padding) slots are left at 0 (zero-init), so
        they appear to belong to sequence 0.

    state_chunk_in_seq : int32 tensor, shape [nchunks_global]
        Maps each global chunk index to its 0-based position within its
        sequence.  Active slots hold values in [0, chunks_per_seq[i]).
        Inactive slots are filled with the sentinel ceil(seq_lens[0] /
        chunk_size), which is co-designed with the zero-initialisation of
        state_seq_mapping: an inactive slot "belongs" to sequence 0, and the
        offset is large enough that chunk_start = cu_seqlens[0] + offset *
        chunk_size >= cu_seqlens[1], placing every element beyond seq_end and
        making the kernel's mask all-False (a no-op).
    """
    num_sequences = int(cu_seqlens.numel()) - 1
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks_per_seq = (seq_lens // chunk_size) + 1  # same convention as compute_dacs_segsum_triton
    S = int(cu_seqlens[-1].item())
    nchunks_global = (S // chunk_size) + num_sequences

    state_seq_mapping  = torch.zeros(nchunks_global, dtype=torch.int32, device=cu_seqlens.device)
    state_chunk_in_seq = torch.zeros(nchunks_global, dtype=torch.int32, device=cu_seqlens.device)
    # Fill with out-of-range sentinel so inactive slots are masked out.
    default_seq_len = int(seq_lens[0].item()) if num_sequences > 0 else 0
    state_chunk_in_seq.fill_((default_seq_len + chunk_size - 1) // chunk_size)

    for i in range(num_sequences):
        start = int(cu_seqlens[i].item())
        n     = int(chunks_per_seq[i].item())
        chunk_start = (start // chunk_size) + i
        state_seq_mapping [chunk_start:chunk_start + n] = i
        state_chunk_in_seq[chunk_start:chunk_start + n] = torch.arange(n, dtype=torch.int32,
                                                                         device=cu_seqlens.device)
    return nchunks_global, state_seq_mapping, state_chunk_in_seq


def bwd_dadt_fused_triton_varlen(
    dSSdA: torch.Tensor,       # [B, H, nchunks_global, C, C]
    SSdA: torch.Tensor,        # [B, H, nchunks_global, C, C]
    ddA_cs: torch.Tensor,      # [B, H, S]
    ddA_cs_rev: torch.Tensor,  # [B, H, S]
    dA_cs: torch.Tensor,       # [B, H, S]
    dA_cs_rev: torch.Tensor,   # [B, H, S]
    chunk_size: int,
    cu_seqlens: torch.Tensor,  # [NS+1]
) -> torch.Tensor:
    """Varlen version of bwd_dadt_fused_triton.

    dSSdA and SSdA use the global chunk layout defined by _build_varlen_chunk_mapping;
    nchunks_global = (S // chunk_size) + num_sequences.  The result must be
    multiplied by NEG_LOG2E = -log2(e) ≈ -1.4427 by the caller.

    Returns float32 tensor of shape [B, H, S].
    """
    B, H, S = ddA_cs.shape
    nchunks_global, state_seq_mapping, state_chunk_in_seq = _build_varlen_chunk_mapping(
        cu_seqlens, chunk_size
    )
    assert dSSdA.shape == (B, H, nchunks_global, chunk_size, chunk_size), \
        f"dSSdA shape mismatch: got {dSSdA.shape}, expected {(B, H, nchunks_global, chunk_size, chunk_size)}"

    dadt_out = torch.zeros(B, H, S, device=ddA_cs.device, dtype=torch.float32)
    grid = (B, H, nchunks_global)

    bwd_dadt_cumsum_fused_kernel_varlen[grid](
        ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, dadt_out,
        cu_seqlens, state_seq_mapping, state_chunk_in_seq,
        ddA_cs.stride(0), ddA_cs.stride(1), ddA_cs.stride(2),
        cu_seqlens.stride(0),
        state_seq_mapping.stride(0),
        state_chunk_in_seq.stride(0),
        S=S,
        CHUNK_SIZE=chunk_size,
    )

    bwd_segsum_dadt_kernel_varlen[grid](
        dSSdA, SSdA, dadt_out,
        cu_seqlens, state_seq_mapping, state_chunk_in_seq,
        dSSdA.stride(0), dSSdA.stride(1), dSSdA.stride(2),
        dSSdA.stride(3), dSSdA.stride(4),
        SSdA.stride(0),  SSdA.stride(1),  SSdA.stride(2),
        SSdA.stride(3),  SSdA.stride(4),
        dadt_out.stride(0), dadt_out.stride(1), dadt_out.stride(2),
        cu_seqlens.stride(0),
        state_seq_mapping.stride(0),
        state_chunk_in_seq.stride(0),
        S=S,
        CHUNK_SIZE=chunk_size,
    )
    return dadt_out


def bwd_dtrap_ddt_triton_varlen(
    trap: torch.Tensor,           # [B, H, S]  – trap_presigmoid values
    dt: torch.Tensor,             # [B, H, S]
    dfactor: torch.Tensor,        # [B, H, S]
    dgamma_diag: torch.Tensor,    # [B, H, S]
    chunk_size: int,
    cu_seqlens: torch.Tensor,     # [NS+1]
):
    """Varlen version of bwd_dtrap_ddt_triton.

    Cross-sequence boundary shifts are suppressed: the first chunk of each
    sequence receives zero incoming gradient from the shifted sgamma path,
    matching the is_first_chunk_in_seq guard in bwd_dtrap_ddt_kernel_varlen.

    Returns (ddt, dtrap_presigmoid), both shaped [B, H, S].
    """
    B, H, S = dt.shape
    nchunks_global, state_seq_mapping, state_chunk_in_seq = _build_varlen_chunk_mapping(
        cu_seqlens, chunk_size
    )
    ddt   = torch.zeros_like(dt)
    dtrap = torch.zeros_like(trap)
    grid  = (B, H, nchunks_global)

    bwd_dtrap_ddt_kernel_varlen[grid](
        trap, dt, dfactor, dgamma_diag, ddt, dtrap,
        cu_seqlens, state_seq_mapping, state_chunk_in_seq,
        trap.stride(0),    trap.stride(1),    trap.stride(2),
        dt.stride(0),      dt.stride(1),      dt.stride(2),
        dfactor.stride(0), dfactor.stride(1), dfactor.stride(2),
        dgamma_diag.stride(0), dgamma_diag.stride(1), dgamma_diag.stride(2),
        ddt.stride(0),   ddt.stride(1),   ddt.stride(2),
        dtrap.stride(0), dtrap.stride(1), dtrap.stride(2),
        cu_seqlens.stride(0),
        state_seq_mapping.stride(0),
        state_chunk_in_seq.stride(0),
        S=S,
        CHUNK_SIZE=chunk_size,
    )
    return ddt, dtrap


# ============================================================================
# Correctness tests (run via __main__)
# ============================================================================

def test_bwd_ddt_fused_correctness():
    """Test bwd_dadt_fused_triton (dense) against the reference implementations."""
    print("=" * 70)
    print("Test: bwd_ddt_fused_correctness (dense)")
    print("=" * 70)

    B, H, S = 16, 32, 2048
    chunk_size = 16
    nchunks = S // chunk_size
    C = chunk_size

    # Generate random inputs
    torch.manual_seed(42)
    dSSdA = torch.randn(B, H, nchunks, C, C, device='cuda', dtype=torch.float32)
    ddA_cs = torch.randn(B, H, S, device='cuda', dtype=torch.float32)
    ddA_cs_rev = torch.randn(B, H, S, device='cuda', dtype=torch.float32)
    dA_cs = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1  # Scale to avoid overflow
    dA_cs_rev = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1
    
    dA_cs_reshape = dA_cs.view(B, H, nchunks, chunk_size)
    SSdA = dA_cs_reshape[:, :, :, :, None] - dA_cs_reshape[:, :, :, None, :]

    # Reference implementation (separate functions)
    ddt_ref1 = bwd_segsum_ddt_from_dSSdA_ref(dSSdA.clone(), dA_cs.clone(), chunk_size)
    ddt_ref2 = bwd_ddt_from_ddA_cs_rev_ref(ddA_cs_rev.clone(), dA_cs_rev.clone(), chunk_size)
    ddt_ref3 = bwd_ddt_from_ddA_cs_ref(ddA_cs.clone(), dA_cs.clone(), chunk_size)
    ddt_ref = ddt_ref1 + ddt_ref2 + ddt_ref3 # TODO: 

    # Fused Triton implementation
    ddt_triton = bwd_dadt_fused_triton(
        dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size
    ) * -1.4426950408889634 # i.e., -log2(e)

    # Compare
    max_diff = (ddt_ref - ddt_triton).abs().max().item()
    mean_diff = (ddt_ref - ddt_triton).abs().mean().item()

    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    passed = max_diff < 1e-4
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()

    return passed

def test_dtrap_ddt_correctness():
    """Test bwd_dtrap_ddt_triton (dense) against compute_dtrap_ddt_ref."""
    import torch.nn.functional as F

    print("=" * 70)
    print("Test: dtrap_ddt_correctness (dense)")
    print("=" * 70)

    B, H, S = 16, 32, 2048
    chunk_size = 16
    nchunks = S // chunk_size
    C = chunk_size

    # Generate random inputs
    torch.manual_seed(42)

    trap = torch.rand(B, H, S, device='cuda', dtype=torch.float16)
    dt = F.softplus(-3.0 + torch.randn(B, H, S, device='cuda', dtype=torch.float))
    dfactor = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1
    dgamma_diag = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1

    # Reference implementation
    ddt_ref, dtrap_ref = compute_dtrap_ddt_ref(dfactor, dgamma_diag, trap, dt)

    # Triton implementation
    ddt_triton, dtrap_triton = bwd_dtrap_ddt_triton(
        trap, dt, dfactor, dgamma_diag, chunk_size
    )

    # Compare
    max_diff_ddt = (ddt_ref - ddt_triton).abs().max().item()
    mean_diff_ddt = (ddt_ref - ddt_triton).abs().mean().item()
    max_diff_dtrap = (dtrap_ref - dtrap_triton).abs().max().item()
    mean_diff_dtrap = (dtrap_ref - dtrap_triton).abs().mean().item()

    print(f"  ddt max difference:   {max_diff_ddt:.2e}")
    print(f"  ddt mean difference:  {mean_diff_ddt:.2e}")
    print(f"  dtrap max difference: {max_diff_dtrap:.2e}")
    print(f"  dtrap mean difference:{mean_diff_dtrap:.2e}")
    passed = max(max_diff_ddt, max_diff_dtrap) < 1e-3
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()

    return passed

def test_dacs_segsum_correctness_varlen():
    import torch.nn.functional as F
    B, H = 1, 8
    chunk_size = 16
    test_cases = [
        ("single_non_multiple", [37]),
        ("single_exact_multiple", [32]),
        ("mixed_with_zero_len", [0, 7, 32, 19, 0, 16]),
    ]

    all_passed = True
    for name, seq_lengths in test_cases:
        S = sum(seq_lengths)
        if S == 0:
            continue
        da = -F.softplus(-3.0 + torch.randn(B, H, S, device='cuda', dtype=torch.float))
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lengths), dim=0).tolist()),
            device='cuda',
            dtype=torch.int32,
        )
        num_sequences = len(seq_lengths)

        da_cs_ref, da_cs_rev_ref, segsum_ref = compute_dacs_segsum_ref_varlen(
            da, chunk_size, cu_seqlens, num_sequences
        )
        da_cs_triton, da_cs_rev_triton, segsum_triton = compute_dacs_segsum_triton_varlen(
            da, chunk_size, cu_seqlens
        )

        max_diff_cs = (da_cs_ref - da_cs_triton).abs().max().item()
        mean_diff_cs = (da_cs_ref - da_cs_triton).abs().mean().item()
        max_diff_cs_rev = (da_cs_rev_ref - da_cs_rev_triton).abs().max().item()
        mean_diff_cs_rev = (da_cs_rev_ref - da_cs_rev_triton).abs().mean().item()
        max_diff_segsum = (segsum_ref - segsum_triton).abs().max().item()
        mean_diff_segsum = (segsum_ref - segsum_triton).abs().mean().item()

        print(f"[{name}]")
        print(f"  da_cs max difference:     {max_diff_cs:.2e}")
        print(f"  da_cs mean difference:    {mean_diff_cs:.2e}")
        print(f"  da_cs_rev max difference: {max_diff_cs_rev:.2e}")
        print(f"  da_cs_rev mean difference:{mean_diff_cs_rev:.2e}")
        print(f"  segsum max difference:    {max_diff_segsum:.2e}")
        print(f"  segsum mean difference:   {mean_diff_segsum:.2e}")
        passed = max(max_diff_cs, max_diff_cs_rev, max_diff_segsum) < 1e-4
        print(f"  Status: {'PASS' if passed else 'FAIL'}")
        print()
        all_passed = all_passed and passed

    return all_passed


def test_dacs_segsum_correctness():
    """Test the dense dacs+segsum Triton kernel against the dense reference."""
    import torch.nn.functional as F
    B, H, S = 16, 32, 2048
    chunk_size = 16
    da = -F.softplus(-3.0 + torch.randn(B, H, S, device='cuda', dtype=torch.float))

    da_cs_ref, da_cs_rev_ref, segsum_ref = compute_dacs_segsum_ref(da, chunk_size)
    da_cs_triton, da_cs_rev_triton, segsum_triton = compute_dacs_segsum_triton(da, chunk_size)

    max_diff_cs = (da_cs_ref - da_cs_triton).abs().max().item()
    mean_diff_cs = (da_cs_ref - da_cs_triton).abs().mean().item()
    max_diff_cs_rev = (da_cs_rev_ref - da_cs_rev_triton).abs().max().item()
    mean_diff_cs_rev = (da_cs_rev_ref - da_cs_rev_triton).abs().mean().item()
    max_diff_segsum = (segsum_ref - segsum_triton).abs().max().item()
    mean_diff_segsum = (segsum_ref - segsum_triton).abs().mean().item()

    print("=" * 70)
    print("Test: dacs_segsum_correctness")
    print("=" * 70)
    print(f"  da_cs max difference:     {max_diff_cs:.2e}")
    print(f"  da_cs mean difference:    {mean_diff_cs:.2e}")
    print(f"  da_cs_rev max difference: {max_diff_cs_rev:.2e}")
    print(f"  da_cs_rev mean difference:{mean_diff_cs_rev:.2e}")
    print(f"  segsum max difference:    {max_diff_segsum:.2e}")
    print(f"  segsum mean difference:   {mean_diff_segsum:.2e}")
    passed = max(max_diff_cs, max_diff_cs_rev, max_diff_segsum) < 1e-4
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()

    return passed


def test_bwd_dadt_fused_varlen_correctness():
    """Test bwd_dadt_fused_triton_varlen against varlen reference."""
    print("=" * 70)
    print("Test: bwd_dadt_fused_varlen_correctness")
    print("=" * 70)

    import torch.nn.functional as F
    B, H, chunk_size = 1, 8, 16
    # Three sequences of lengths 37, 32, 28; total S = 97
    seq_lengths = [37, 32, 28]
    S = sum(seq_lengths)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lengths), dim=0).tolist()),
        device='cuda', dtype=torch.int32,
    )
    num_sequences = len(seq_lengths)
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks_per_seq = (seq_lens // chunk_size) + 1
    nchunks_global = (S // chunk_size) + num_sequences
    C = chunk_size

    torch.manual_seed(42)
    dSSdA    = torch.randn(B, H, nchunks_global, C, C, device='cuda', dtype=torch.float32)
    ddA_cs   = torch.randn(B, H, S, device='cuda', dtype=torch.float32)
    ddA_cs_rev = torch.randn(B, H, S, device='cuda', dtype=torch.float32)
    dA_cs    = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1
    dA_cs_rev = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1

    # Build SSdA in the varlen global chunk layout.
    # For each sequence i, the global chunk slots start at (cu_seqlens[i] // chunk_size) + i.
    SSdA = torch.zeros(B, H, nchunks_global, C, C, device='cuda', dtype=torch.float32)
    for i in range(num_sequences):
        start = int(cu_seqlens[i].item())
        end   = int(cu_seqlens[i + 1].item())
        curr_seqlen  = end - start
        curr_nchunks = int(chunks_per_seq[i].item())
        curr_padded_len = curr_nchunks * chunk_size
        global_chunk_start = (start // chunk_size) + i

        dA_cs_padded = torch.zeros(B, H, curr_padded_len, device='cuda', dtype=torch.float32)
        if curr_seqlen > 0:
            dA_cs_padded[:, :, :curr_seqlen] = dA_cs[:, :, start:end]
        dA_cs_seq = dA_cs_padded.view(B, H, curr_nchunks, C)
        SSdA[:, :, global_chunk_start:global_chunk_start + curr_nchunks, :, :] = (
            dA_cs_seq[:, :, :, :, None] - dA_cs_seq[:, :, :, None, :]
        )

    # Varlen reference
    ddt_ref = bwd_dadt_fused_varlen_ref(
        dSSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size, cu_seqlens
    )

    # Varlen Triton kernel (result scaled by -log2(e) to match reference sign convention)
    ddt_triton = bwd_dadt_fused_triton_varlen(
        dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size, cu_seqlens
    ) * -1.4426950408889634

    max_diff  = (ddt_ref - ddt_triton).abs().max().item()
    mean_diff = (ddt_ref - ddt_triton).abs().mean().item()
    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    passed = max_diff < 1e-4
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_dtrap_ddt_varlen_correctness():
    """Test bwd_dtrap_ddt_triton_varlen against varlen reference."""
    print("=" * 70)
    print("Test: dtrap_ddt_varlen_correctness")
    print("=" * 70)

    import torch.nn.functional as F
    B, H, chunk_size = 1, 8, 16
    seq_lengths = [37, 32, 28]
    S = sum(seq_lengths)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lengths), dim=0).tolist()),
        device='cuda', dtype=torch.int32,
    )

    torch.manual_seed(42)
    trap        = torch.rand(B, H, S, device='cuda', dtype=torch.float16)
    dt          = F.softplus(-3.0 + torch.randn(B, H, S, device='cuda', dtype=torch.float32))
    dfactor     = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1
    dgamma_diag = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1

    # Varlen reference
    ddt_ref, dtrap_ref = compute_dtrap_ddt_varlen_ref(
        dfactor, dgamma_diag, trap, dt, chunk_size, cu_seqlens
    )

    # Varlen Triton kernel
    ddt_triton, dtrap_triton = bwd_dtrap_ddt_triton_varlen(
        trap, dt, dfactor, dgamma_diag, chunk_size, cu_seqlens
    )

    max_diff_ddt   = (ddt_ref   - ddt_triton  ).abs().max().item()
    mean_diff_ddt  = (ddt_ref   - ddt_triton  ).abs().mean().item()
    max_diff_dtrap = (dtrap_ref - dtrap_triton).abs().max().item()
    mean_diff_dtrap = (dtrap_ref - dtrap_triton).abs().mean().item()
    print(f"  ddt  max difference:  {max_diff_ddt:.2e}")
    print(f"  ddt  mean difference: {mean_diff_ddt:.2e}")
    print(f"  dtrap max difference:  {max_diff_dtrap:.2e}")
    print(f"  dtrap mean difference: {mean_diff_dtrap:.2e}")
    passed = max(max_diff_ddt, max_diff_dtrap) < 1e-3
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================================
# Benchmarks (run individually as needed)
# ============================================================================

def benchmark_bwd_ddt():
    """Benchmark fused kernel against unfused baseline."""
    from triton.testing import do_bench


    print("=" * 70)
    print("Benchmark: bwd_ddt_fused")
    print("=" * 70)

    B, H, S = 16, 32, 2048
    chunk_size = 16
    nchunks = S // chunk_size
    C = chunk_size

    print(f"Configuration: B={B}, H={H}, S={S}, chunk_size={chunk_size}")
    print()

    # Setup inputs
    torch.manual_seed(42)
    dSSdA = torch.randn(B, H, nchunks, C, C, device='cuda', dtype=torch.float32)
    ddA_cs = torch.randn(B, H, S, device='cuda', dtype=torch.float32)
    ddA_cs_rev = torch.randn(B, H, S, device='cuda', dtype=torch.float32)
    dA_cs = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1
    dA_cs_rev = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1

    dA_cs_reshape = dA_cs.view(B, H, nchunks, chunk_size)
    SSdA = dA_cs_reshape[:, :, :, :, None] - dA_cs_reshape[:, :, :, None, :]

    # Benchmark reference (unfused)
    def ref_impl():
        ddt1 = bwd_segsum_ddt_from_dSSdA_ref(dSSdA, dA_cs, chunk_size)
        ddt2 = bwd_ddt_from_ddA_cs_rev_ref(ddA_cs_rev, dA_cs_rev, chunk_size)
        ddt3 = bwd_ddt_from_ddA_cs_ref(ddA_cs, dA_cs, chunk_size)
        return ddt1 + ddt2 + ddt3

    # Benchmark individual components
    ref1_time = do_bench(lambda: bwd_segsum_ddt_from_dSSdA_ref(dSSdA, dA_cs, chunk_size), warmup=25, rep=100)
    ref2_time = do_bench(lambda: bwd_ddt_from_ddA_cs_rev_ref(ddA_cs_rev, dA_cs_rev, chunk_size), warmup=25, rep=100)
    ref3_time = do_bench(lambda: bwd_ddt_from_ddA_cs_ref(ddA_cs, dA_cs, chunk_size), warmup=25, rep=100)
    ref_time = do_bench(ref_impl, warmup=25, rep=100)

    # Benchmark fused
    def fused_impl():
        return bwd_dadt_fused_triton(
            dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size
        )

    fused_time = do_bench(fused_impl, warmup=25, rep=100)

    print("Reference (unfused):")
    print(f"  Function 1 (segsum): {ref1_time:.3f} ms")
    print(f"  Function 2 (cs_rev): {ref2_time:.3f} ms")
    print(f"  Function 3 (cs):     {ref3_time:.3f} ms")
    print(f"  Total:               {ref_time:.3f} ms")
    print()
    print("Fused Triton:")
    print(f"  Total:               {fused_time:.3f} ms")
    print(f"  Speedup:             {ref_time / fused_time:.2f}x")
    print()

    return ref_time, fused_time


def benchmark_dacs_segsum():
    """Benchmark dacs+segsum Triton against reference implementation."""
    from triton.testing import do_bench
    import torch.nn.functional as F

    print("=" * 70)
    print("Benchmark: dacs_segsum")
    print("=" * 70)

    B, H, S = 16, 32, 2048
    chunk_size = 16

    print(f"Configuration: B={B}, H={H}, S={S}, chunk_size={chunk_size}")
    print()

    torch.manual_seed(42)
    da = F.softplus(-3.0 + torch.randn(B, H, S, device='cuda', dtype=torch.float))

    def ref_impl():
        return compute_dacs_segsum_ref(da, chunk_size)

    def triton_impl():
        return compute_dacs_segsum_triton(da, chunk_size)

    ref_time = do_bench(ref_impl, warmup=25, rep=100)
    triton_time = do_bench(triton_impl, warmup=25, rep=100)

    print("Reference:")
    print(f"  Total: {ref_time:.3f} ms")
    print("Triton:")
    print(f"  Total: {triton_time:.3f} ms")
    print(f"  Speedup: {ref_time / triton_time:.2f}x")
    print()

    return ref_time, triton_time


def benchmark_dtrap_ddt():
    """Benchmark dtrap/ddt kernel against reference implementation."""
    from triton.testing import do_bench
    import torch.nn.functional as F

    print("=" * 70)
    print("Benchmark: bwd_dtrap_ddt")
    print("=" * 70)

    B, H, S = 16, 32, 2048
    chunk_size = 16

    print(f"Configuration: B={B}, H={H}, S={S}, chunk_size={chunk_size}")
    print()

    torch.manual_seed(42)
    trap = torch.ones(B, H, S, device='cuda', dtype=torch.float16) * 0.5
    dt = F.softplus(-3.0 + torch.randn(B, H, S, device='cuda', dtype=torch.float))
    dfactor = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1
    dgamma_diag = torch.randn(B, H, S, device='cuda', dtype=torch.float32) * 0.1

    def ref_impl():
        return compute_dtrap_ddt_ref(dfactor, dgamma_diag, trap, dt)

    def triton_impl():
        return bwd_dtrap_ddt_triton(trap, dt, dfactor, dgamma_diag, chunk_size)

    ref_time = do_bench(ref_impl, warmup=25, rep=100)
    triton_time = do_bench(triton_impl, warmup=25, rep=100)

    print("Reference:")
    print(f"  Total: {ref_time:.3f} ms")
    print("Triton:")
    print(f"  Total: {triton_time:.3f} ms")
    print(f"  Speedup: {ref_time / triton_time:.2f}x")
    print()

    return ref_time, triton_time


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    # Non-varlen (dense) tests
    test_bwd_ddt_fused_correctness()
    test_dtrap_ddt_correctness()
    test_dacs_segsum_correctness()
    # benchmark_bwd_ddt()
    # benchmark_dtrap_ddt()
    # benchmark_dacs_segsum()
    # Varlen tests
    test_dacs_segsum_correctness_varlen()
    test_bwd_dadt_fused_varlen_correctness()
    test_dtrap_ddt_varlen_correctness()
