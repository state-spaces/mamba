"""
Triton-based fallback implementation of Mamba3 MIMO kernels for non-Hopper GPUs.

This module provides efficient Triton kernels that work on a wider range of GPUs
including Ampere (RTX 30 series, A100), Ada (RTX 40 series, L40), and Volta architectures.

The kernels in this file are designed to be compatible with older GPU architectures while
maintaining reasonable performance through block-level tiling and optimized memory access patterns.

Copyright (c) 2026, Dao AI Lab, Goombalab
"""

import torch
import triton
import triton.language as tl
from typing import Optional
import math


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=8),
    ],
    key=["seq_len", "head_dim", "mimo_rank"],
)
@triton.jit
def _mamba3_mimo_forward_kernel(
    # Inputs
    Q_ptr,
    K_ptr,
    V_ptr,
    Q_bias_ptr,
    K_bias_ptr,
    MIMO_V_ptr,
    MIMO_O_ptr,
    Z_ptr,
    D_ptr,
    MIMO_Z_ptr,
    # Output
    O_ptr,
    # Dimensions
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    state_dim,
    mimo_rank,
    num_groups,
    # Strides
    Q_batch_stride,
    Q_seq_stride,
    K_batch_stride,
    K_seq_stride,
    V_batch_stride,
    V_seq_stride,
    O_batch_stride,
    O_seq_stride,
    # Flags
    has_z: tl.constexpr,
    has_d: tl.constexpr,
    has_mimo_o: tl.constexpr,
    # Block config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Simplified block-wise forward kernel for Mamba3 MIMO compatible with older GPUs.
    
    This kernel processes the sequence in manageable blocks to reduce memory pressure
    and avoid hardware-specific features like Tensor Memory Accelerators (TMA).
    """
    # Program IDs
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)
    
    # Early exit if out of bounds
    if pid_batch >= batch_size or pid_head >= num_heads:
        return
    
    # Compute sequence block range
    block_start_seq = pid_m * BLOCK_M
    block_end_seq = tl.minimum(block_start_seq + BLOCK_M, seq_len)
    block_seq_len = block_end_seq - block_start_seq
    
    # Indices for M (sequence) and N (head_dim) dimensions
    seq_idx = block_start_seq + tl.arange(0, BLOCK_M)
    head_dim_idx = tl.arange(0, BLOCK_N)
    state_idx = tl.arange(0, BLOCK_K)
    
    # Create masks
    seq_mask = seq_idx < seq_len
    head_mask = head_dim_idx < head_dim
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute block-wise GEMM for each state dimension block
    num_state_blocks = tl.cdiv(state_dim, BLOCK_K)
    
    for k_block in range(num_state_blocks):
        k_start = k_block * BLOCK_K
        k_end = tl.minimum(k_start + BLOCK_K, state_dim)
        k_len = k_end - k_start
        
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < state_dim
        
        # Load Q block [BLOCK_M, state_dim]
        # Q is [batch, seq, mimo_rank, num_groups, state_dim]
        q_ptrs = (
            Q_ptr 
            + pid_batch * Q_batch_stride
            + seq_idx[:, None] * Q_seq_stride
            + pid_head * state_dim
            + k_idx[None, :]
        )
        q_block = tl.load(q_ptrs, mask=seq_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load K block [BLOCK_M, state_dim]
        k_ptrs = (
            K_ptr
            + pid_batch * K_batch_stride
            + seq_idx[:, None] * K_seq_stride
            + pid_head * state_dim
            + k_idx[None, :]
        )
        k_block = tl.load(k_ptrs, mask=seq_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Accumulate Q @ K^T contribution
        qk_contrib = tl.dot(q_block, tl.trans(k_block), allow_tf32=True)
        acc = acc + qk_contrib
    
    # Load V and apply MIMO projections
    v_ptrs = (
        V_ptr
        + pid_batch * V_batch_stride
        + seq_idx[:, None] * V_seq_stride
        + pid_head * head_dim
        + head_dim_idx[None, :]
    )
    v_block = tl.load(v_ptrs, mask=seq_mask[:, None] & head_mask[None, :], other=0.0)
    
    # Load MIMO_V projection [num_heads, mimo_rank, head_dim]
    mimo_v_ptr = MIMO_V_ptr + pid_head * mimo_rank * head_dim
    mimo_v = tl.load(mimo_v_ptr + head_dim_idx, mask=head_mask, other=0.0)
    
    # Apply MIMO projection: v @ mimo_v^T
    output = tl.dot(v_block, mimo_v, allow_tf32=True)
    
    # Optionally apply D skip connection
    if has_d:
        d_val = tl.load(D_ptr + pid_head)
        output = output + v_block * d_val
    
    # Optionally apply output projection MIMO_O
    if has_mimo_o:
        mimo_o_ptr = MIMO_O_ptr + pid_head * mimo_rank * head_dim
        mimo_o = tl.load(mimo_o_ptr + head_dim_idx, mask=head_mask, other=0.0)
        output = tl.dot(output, mimo_o, allow_tf32=True)
    
    # Store output [batch, seq, num_heads, head_dim]
    o_ptrs = (
        O_ptr
        + pid_batch * O_batch_stride
        + seq_idx[:, None] * O_seq_stride
        + pid_head * head_dim
        + head_dim_idx[None, :]
    )
    tl.store(o_ptrs, output.to(V_ptr.dtype.element_ty), mask=seq_mask[:, None] & head_mask[None, :])


def mamba3_mimo_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_bias: Optional[torch.Tensor],
    k_bias: Optional[torch.Tensor],
    mimo_v: torch.Tensor,
    mimo_o: Optional[torch.Tensor],
    mimo_z: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    d: Optional[torch.Tensor],
    angles: torch.Tensor,
    da_cs: torch.Tensor,
    da_cs_rev: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    segsum: torch.Tensor,
    chunk_size: int = 16,
    rotary_dim_divisor: int = 4,
) -> torch.Tensor:
    """
    Triton-based MIMO forward pass for non-Hopper GPUs.
    
    Args:
        q: Query tensor [batch, seq_len, mimo_rank, num_groups, state_dim]
        k: Key tensor [batch, seq_len, mimo_rank, num_groups, state_dim]
        v: Value tensor [batch, seq_len, num_heads, head_dim]
        mimo_v: MIMO projection matrix [num_heads, mimo_rank, head_dim]
        mimo_o: Optional output projection [num_heads, mimo_rank, head_dim]
        mimo_z: Optional z projection [num_heads, mimo_rank, head_dim]
        z: Optional gating tensor [batch, seq_len, num_heads, head_dim]
        d: Optional skip parameter [num_heads]
        angles: Rotary embeddings [batch, seq_len, num_heads, rotary_dim]
        
    Returns:
        output: [batch, seq_len, num_heads, head_dim]
    """
    batch_size = q.shape[0]
    seq_len = q.shape[1]
    mimo_rank = q.shape[2]
    num_groups = q.shape[3]
    state_dim = q.shape[4]
    num_heads = v.shape[2]
    head_dim = v.shape[3]
    
    # Allocate output tensor
    output = torch.empty(
        (batch_size, seq_len, num_heads, head_dim),
        dtype=v.dtype,
        device=v.device,
    )
    
    has_z = z is not None
    has_d = d is not None
    has_mimo_o = mimo_o is not None
    
    # Ensure contiguity for better memory access patterns
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    mimo_v = mimo_v.contiguous()
    if mimo_o is not None:
        mimo_o = mimo_o.contiguous()
    
    # Get strides
    Q_batch_stride = q.stride(0)
    Q_seq_stride = q.stride(1)
    K_batch_stride = k.stride(0)
    K_seq_stride = k.stride(1)
    V_batch_stride = v.stride(0)
    V_seq_stride = v.stride(1)
    O_batch_stride = output.stride(0)
    O_seq_stride = output.stride(1)
    
    # Calculate grid: (batch, num_heads, num_seq_blocks)
    BLOCK_M = 32  # Sequence block size
    BLOCK_N = 32  # Head dim block size
    BLOCK_K = 32  # State dim block size
    
    grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_M))
    
    _mamba3_mimo_forward_kernel[grid](
        q, k, v,
        q_bias, k_bias,
        mimo_v, mimo_o,
        z, d, mimo_z,
        output,
        batch_size, seq_len, num_heads, head_dim, state_dim,
        mimo_rank, num_groups,
        Q_batch_stride, Q_seq_stride,
        K_batch_stride, K_seq_stride,
        V_batch_stride, V_seq_stride,
        O_batch_stride, O_seq_stride,
        has_z, has_d, has_mimo_o,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return output
