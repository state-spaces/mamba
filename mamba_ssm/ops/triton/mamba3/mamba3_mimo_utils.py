"""
Fused Triton kernels for Mamba3 backward pass ddt computation.

This module implements fused kernels that combine three separate backward operations:
1. bwd_segsum_ddt_from_dSSdA - Complex 2D segsum operation
2. bwd_ddt_from_ddA_cs_rev - Forward exclusive cumsum operation
3. bwd_ddt_from_ddA_cs - Reverse cumsum operation

The fusion reduces memory traffic and kernel launch overhead.
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple

# Constants
LOG2 = math.log(2.0)
NEG_LOG2E = -math.log2(math.e)


# ============================================================================
# Kernel 1: Fused Cumsum Operations (forward exclusive + reverse)
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
    """
    Fused kernel that computes contributions from:
    - bwd_ddt_from_ddA_cs: reverse cumsum operation
    - bwd_ddt_from_ddA_cs_rev: forward exclusive cumsum operation

    Each program handles one chunk for one (batch, head) pair.
    Grid: (B, H, nchunks)
    """
    # Get program indices
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    # Calculate chunk boundaries
    chunk_start = pid_chunk * CHUNK_SIZE
    offs_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
    mask = offs_seq < S

    # Compute base offset for this (batch, head) pair
    base_offset = pid_batch * stride_batch + pid_head * stride_head

    # Load chunk data for all four input tensors
    ddA_cs = tl.load(ddA_cs_ptr + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)
    ddA_cs_rev = tl.load(ddA_cs_rev_ptr + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)
    dA_cs = tl.load(dA_cs_ptr + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)
    dA_cs_rev = tl.load(dA_cs_rev_ptr + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)

    # ========================================================================
    # Operation 1: bwd_ddt_from_ddA_cs (reverse cumsum)
    # ========================================================================
    # Scale by log(2) * exp2(dA_cs)
    # Use literal constants instead of globals
    scaled_ddA_cs =  tl.exp(dA_cs) * ddA_cs  # LOG2
    # Apply reverse cumsum within chunk
    ddt_cs = tl.cumsum(scaled_ddA_cs, axis=0, reverse=True)

    # ========================================================================
    # Operation 2: bwd_ddt_from_ddA_cs_rev (forward exclusive cumsum)
    # ========================================================================
    # Scale by log(2) * exp2(dA_cs_rev)
    # Use literal constants instead of globals
    scaled_ddA_cs_rev = tl.exp(dA_cs_rev) * ddA_cs_rev  # LOG2
    # Apply forward cumsum within chunk (inclusive)
    ddt_cs_rev_inclusive = tl.cumsum(scaled_ddA_cs_rev, axis=0)

    # Roll one to the right:
    i = tl.arange(0, CHUNK_SIZE)[:, None]          # [N,1]
    j = tl.arange(0, CHUNK_SIZE)[None, :]          # [1,N]
    S = (i == j + 1)                      # strictly lower diagonal (one below main)
    ddt_cs_rev_exclusive = tl.sum(tl.where(S, ddt_cs_rev_inclusive, 0), axis=1)

    # # Convert to exclusive cumsum
    # # Exclusive cumsum: output[i] = sum(input[0:i])
    # # Inclusive cumsum: cumsum[i] = sum(input[0:i+1])
    # # Therefore: exclusive[i] = inclusive[i] - input[i]
    # # Which is: exclusive[i] = cumsum[i] - scaled_ddA_cs_rev[i]
    # ddt_cs_rev_shifted = ddt_cs_rev_inclusive - scaled_ddA_cs_rev

    # ========================================================================
    # Combine contributions and apply final scaling
    # ========================================================================
    # Use literal constant instead of global
    ddt_total = ddt_cs + ddt_cs_rev_exclusive 

    # Store result
    tl.store(ddt_out_ptr + base_offset + offs_seq * stride_seq, ddt_total, mask=mask)


# ============================================================================
# Kernel 2: Segsum Operation with 2D Matrix Processing
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
    """
    Kernel for bwd_segsum_ddt_from_dSSdA operation.
    Matches the reference implementation:
    1. Permute dSSdA last two dims
    2. Compute seg = dA_cs[i] - dA_cs[j]
    3. Scale by log(2) * exp2(seg)
    4. Reverse cumsum along dim -2 (column-wise for each row)
    5. Apply lower triangular mask (i > j)
    6. Sum along dim -1 (sum over j for each i)

    Each program handles one chunk for one (batch, head) pair.
    Grid: (B, H, nchunks)
    """
    # Get program indices
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    # Calculate chunk boundaries
    chunk_start = pid_chunk * CHUNK_SIZE
    offs_c = tl.arange(0, CHUNK_SIZE)
    offs_seq = chunk_start + offs_c

    # Load dA_cs for this chunk [C]
    # dA_cs_offset = pid_batch * stride_dA_batch + pid_head * stride_dA_head
    # dA_cs_chunk = tl.load(dA_cs_ptr + dA_cs_offset + offs_seq * stride_dA_seq)

    # Base offset for dSSdA matrix [nchunks, C, C]
    dSSdA_offset = dSSdA_ptr + (pid_batch * stride_dSSdA_batch +
                    pid_head * stride_dSSdA_head +
                    pid_chunk * stride_dSSdA_chunk)
    SSdA_offset = SSdA_cs_ptr + (pid_batch * stride_SSdA_batch +
                    pid_head * stride_SSdA_head +
                    pid_chunk * stride_SSdA_chunk)
    ddt_ptrs = ddt_out_ptr + (pid_batch * stride_ddt_batch +
                    pid_head * stride_ddt_head +
                    offs_seq * stride_ddt_seq)

    # NOTE: dSSdA is actually the transpose corresponding to seq_k \time seq_q
    dSSdA_block = tl.load(dSSdA_offset + offs_c[:, None]*stride_dSSdA_col + offs_c[None, :]*stride_dSSdA_row)
    SSdA_block = tl.load(SSdA_offset + offs_c[:, None]*stride_SSdA_row + offs_c[None, :]*stride_SSdA_col)

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
# Kernel 3:  backwards from gamma terms to trap 
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
    # Get program indices
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    # Calculate chunk boundaries
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

    # dgamma and dsgamma for current positions
    dgamma_block = dfactor_block + dgamma_diag_input_block
    dsgamma_block = dfactor_block #+ dsgamma_input_block

    # dsdt and dstrap for current positions (using shifted strap/sdt)
    dsdt_block = tl.sigmoid(-strap_block.to(tl.float32)) * dsgamma_block
    dstrap_block = -sdt_block * dsgamma_block

    # Compute dsdt/dstrap at previous position for cross-chunk shift
    prev_seq = chunk_start - 1
    prev_mask = prev_seq >= 0
    prev_dgamma = tl.load(
        dfactor_ptr + dfactor_offset + prev_seq * stride_dfactor_seq,
        mask=prev_mask, other=0.0
    )
    # prev_dsgamma_input = tl.load(
    #     dsgamma_ptr + dsgamma_offset + prev_seq * stride_dsgamma_seq,
    #     mask=prev_mask, other=0.0
    # )
    prev_dsgamma = prev_dgamma  #+ prev_dsgamma_input
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

    # Shift right by one within chunk: out[i] = in[i-1], with cross-chunk value at i=0
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
# Kernel 4:  compute da_cs, da_cs_rev, segsum from da
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
    S = (i == j - 1)                      # strictly upper diagonal (one above main)
    da_cs_rev = tl.sum(tl.where(S, da_cs_rev, 0), axis=1)
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
# Wrapper Function
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
    # Validate inputs
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
    trap: torch.Tensor,      # [B, H, S]
    dt: torch.Tensor,        # [B, H, S]
    dfactor: torch.Tensor,   # [B, H, S]
    dgamma_diag: torch.Tensor,   # [B, H, S]
    chunk_size: int, # NOTE: the chunk_size does not have to be the same as the other kernels
):
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

def compute_dacs_segsum_triton(
    da: torch.Tensor,  # (B, H, S)
    chunk_size: int,
):
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
# Reference Implementations (for testing)
# ============================================================================

def bwd_segsum_ddt_from_dSSdA_ref(
    dSSdA: torch.Tensor,
    dA_cs: torch.Tensor,
    chunk_size: int,
):
    """Reference implementation of bwd_segsum_ddt_from_dSSdA."""
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
    ddA_cs_rev: torch.Tensor,
    dA_cs_rev: torch.Tensor,
    chunk_size: int,
):
    """Reference implementation of bwd_ddt_from_ddA_cs_rev."""
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
    ddA_cs: torch.Tensor,
    dA_cs: torch.Tensor,
    chunk_size: int,
):
    """Reference implementation of bwd_ddt_from_ddA_cs."""
    B, H, S = ddA_cs.shape
    nchunks = S // chunk_size
    ddA_cs =  torch.exp(dA_cs) * ddA_cs
    dA_cs = dA_cs.view(B, H, nchunks, chunk_size)
    ddA_cs = ddA_cs.view(B, H, nchunks, chunk_size)
    ddA = torch.flip(torch.cumsum(torch.flip(ddA_cs, dims=[-1]), dim=-1), dims=[-1])
    ddt = ddA * (-math.log2(math.e))
    return ddt.reshape(B, H, nchunks*chunk_size)

def compute_dtrap_ddt_ref(dfactor: torch.Tensor,
                          dgamma_diag_input: torch.Tensor,
                          trap_presigmoid,
                          dt,
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    trap = torch.nn.functional.sigmoid(trap_presigmoid)
    strap = torch.nn.functional.pad(trap[:, :, 1:], (0, 1), value=0.0)
    sdt = torch.nn.functional.pad(dt[:, :, 1:], (0, 1), value=0.0)
    dgamma = dfactor.detach().clone() + dgamma_diag_input.detach().clone()
    dsgamma = dfactor.detach().clone() # + dsgamma_input.detach().clone()
    dsdt = (1 - strap) * dsgamma
    dstrap = -sdt * dsgamma
    # shift rightward:
    ddt = torch.nn.functional.pad(dsdt[:, :, :-1], (1, 0), value=0.0)
    dtrap = torch.nn.functional.pad(dstrap[:, :, :-1], (1, 0), value=0.0)
    # Add the dgamma path:
    dtrap += dgamma*dt
    # grad of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    dtrap *= trap * torch.nn.functional.sigmoid(-trap_presigmoid)
    ddt += dgamma*trap
    return ddt, dtrap

def compute_dacs_segsum_ref(da: torch.Tensor, # (B, H, S)
                        chunk_size: int,
                        ):
    B, H, S = da.shape
    nchunks = S // chunk_size

    da_reshaped = da.view(B, H, nchunks, chunk_size)
    da_cs = torch.cumsum(da_reshaped, dim=-1)
    da_cs_sum = torch.sum(da_reshaped, dim=-1)
    da_cs_rev = da_cs_sum[..., None] - da_cs #torch.flip(torch.cumsum(torch.flip(da_reshaped, dims=[-1]), dim=-1), dims=[-1])

    from einops import repeat
    segsum = repeat(da_reshaped, "... d -> ... d e", e=chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=da_cs.device, dtype=bool), diagonal=-1)
    segsum = segsum.masked_fill(~mask, 0)
    segsum = torch.cumsum(segsum, dim=-2)

    return da_cs.view(B, H, S), da_cs_rev.view(B, H, S), segsum


# ============================================================================
# Testing Functions
# ============================================================================

def test_bwd_ddt_fused_correctness():
    """Test the fused kernel against reference implementation."""
    print("=" * 70)
    print("Test: basic_correctness")
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
    """Test the fused kernel against reference implementation."""
    import torch.nn.functional as F

    print("=" * 70)
    print("Test: basic_correctness")
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

def test_dacs_segsum_correctness():
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


# ============================================================================
# Benchmarking Functions
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
    test_bwd_ddt_fused_correctness()
    # benchmark_bwd_ddt()
    test_dtrap_ddt_correctness()
    # benchmark_dtrap_ddt()
    # benchmark_dacs_segsum()
    test_dacs_segsum_correctness()
    # benchmark_dacs_segsum()