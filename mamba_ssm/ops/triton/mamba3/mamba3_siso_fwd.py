"""
Mamba-3 SISO Forward Pass Triton Kernel.

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

import triton
import triton.language as tl
from mamba_ssm.ops.triton.mamba3.utils import cos_approx, sin_approx, tanh_approx, silu, sigmoid_approx

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [2, 4, 8]
    ],
    key=[
        "CHUNK_SIZE", "HEADDIM_QK", "HEADDIM_V", "STORE_SSM_STATES_ADT_OUTV", "HAS_D", 
        "HAS_Z", "HAS_INITIAL_STATES", "RETURN_FINAL_STATES", "IS_VARLEN"],
)
@triton.jit
def mamba3_siso_fwd_kernel(
    # Inputs
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, 
    Initial_SSM_State, Initial_K_State, Initial_V_State, Cu_Seqlens,
    # Outputs
    Out, Out_v, SSM_States, DA_CS_Store, DA_CS_SUM_Store, Q_store, K_store, QK_store,
    Scale_store, Gamma_store, Final_SSM_State, Final_K_State,
    # Input Strides
    stride_q_batch, stride_q_seqlen, stride_q_head, stride_q_qkdim,
    stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_qkdim,
    stride_v_batch, stride_v_seqlen, stride_v_head, stride_v_vdim,
    stride_adt_batch, stride_adt_head, stride_adt_seqlen,
    stride_dt_batch, stride_dt_head, stride_dt_seqlen,
    stride_trap_batch, stride_trap_head, stride_trap_seqlen,
    stride_q_bias_head, stride_q_bias_qkdim,
    stride_k_bias_head, stride_k_bias_qkdim,
    stride_angles_batch, stride_angles_seqlen, stride_angles_head, stride_angles_qkdim,
    stride_d_head,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_vdim,
    stride_init_ssm_state_seq, stride_init_ssm_state_head, stride_init_ssm_state_vdim, 
    stride_init_ssm_state_qkdim,
    stride_init_k_state_seq, stride_init_k_state_head, stride_init_k_state_qkdim,
    stride_init_v_state_seq, stride_init_v_state_head, stride_init_v_state_vdim,
    stride_cu_seqlen,
    # Output Strides
    stride_o_batch, stride_o_seqlen, stride_o_head, stride_o_vdim,
    stride_o_v_batch, stride_o_v_seqlen, stride_o_v_head, stride_o_v_vdim,
    stride_ssm_states_batch, stride_ssm_states_head, stride_ssm_states_vdim, stride_ssm_states_qkdim,
    stride_da_cs_store_batch, stride_da_cs_store_head, stride_da_cs_store_seqlen,
    stride_da_cs_sum_store_batch, stride_da_cs_sum_store_head, stride_da_cs_sum_store_seqlen,
    stride_q_store_batch, stride_q_store_seqlen, stride_q_store_head, stride_q_store_qkdim,
    stride_k_store_batch, stride_k_store_seqlen, stride_k_store_head, stride_k_store_qkdim,
    stride_qk_store_batch, stride_qk_store_head, stride_qk_store_seqlen,
    stride_scale_store_batch, stride_scale_store_head, stride_scale_store_seqlen,
    stride_gamma_store_batch, stride_gamma_store_head, stride_gamma_store_seqlen,
    stride_final_ssm_state_seq, stride_final_ssm_state_head, stride_final_ssm_state_vdim, 
    stride_final_ssm_state_qkdim,
    stride_final_k_state_seq, stride_final_k_state_head, stride_final_k_state_chunk, 
    stride_final_k_state_qkdim,
    # Dimensions
    seqlen, nheads_qk, headdim_qk, headdim_v, headdim_angles,
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    STORE_SSM_STATES_ADT_OUTV: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    RETURN_FINAL_STATES: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Mamba-3 forward kernel.

    Grid: (nheads, batch) for batched, (nheads, 1, num_sequences) for varlen

    Inputs:
        Q, K:                       (batch, seqlen, nheads_qk, headdim_qk)
        V:                          (batch, seqlen, nheads, headdim_v)  
        ADT, DT, Trap:              (batch, nheads, seqlen)
        Q_bias, K_bias:             (nheads, headdim_qk)
        Angles:                     (batch, seqlen, nheads, headdim_angles)
        D:                          (nheads,)
        Z:                          (batch, seqlen, nheads, headdim_v)
        Initial SSM State:          (num_sequences, nheads, headdim_v, headdim_qk)
        Initial K State:            (num_sequences, nheads, headdim_qk)
        Initial V State:            (num_sequences, nheads, headdim_v)
        Cu_Seqlens:                 (num_sequences + 1,)

    NOTE: num_sequences = batch for batched mode, or len(cu_seqlens)-1 for varlen mode.

    Compile-time constants:
        CHUNK_SIZE:                 Chunk size for processing sequences
        HEADDIM_QK:                 Head dimension for Q/K
        HEADDIM_V:                  Head dimension for V
        
        STORE_SSM_STATES_ADT_OUTV:  Whether to store SSM states, ADT, and Out_v for backward pass
                                    Set to FALSE for inference-only runs for efficiency
        HAS_INITIAL_STATES:         Whether input SSM states are provided for state passing
        RETURN_FINAL_STATES:        Whether to return final SSM states for state passing
        HAS_D:                      Whether D-skip connection is used
        HAS_Z:                      Whether Z-gating is used
        IS_VARLEN:                  Whether the input is a variable-length sequence

    NOTE:
        1. nheads % nheads_qk == 0
        2. Kernel is optimized for headdim_qk = 128 and headdim_v = 64

    Outputs:
        Out:                    (batch, seqlen, nheads, headdim_v)
        Out_v:                  (batch, seqlen, nheads, headdim_v) (if STORE_SSM_STATES_ADT_OUTV)
        SSM_States:             (batch, nheads, headdim_v, nchunks * headdim_qk) (if STORE_SSM_STATES_ADT_OUTV)
        DA_CS_Store:            (batch, nheads, seqlen) (if STORE_SSM_STATES_ADT_OUTV)
        DA_CS_SUM_Store:        (batch, nheads, nchunks) (if STORE_SSM_STATES_ADT_OUTV)
        Q_store:                (batch, seqlen, nheads, headdim_qk)
        K_store:                (batch, seqlen, nheads, headdim_qk)
        QK_store:               (batch, seqlen, nheads)
        Scale_store:            (batch, seqlen, nheads)
        Gamma_store:            (batch, seqlen, nheads)
        Final SSM State:        (num_sequences, nheads, headdim_v, headdim_qk) (if RETURN_FINAL_STATES)
        Final K State:          (num_sequences, nheads, chunk_size, headdim_qk) (if RETURN_FINAL_STATES)
    
    NOTE: 
    1. For batched inputs, nchunks = ceil(seqlen / CHUNK_SIZE) and for varlen inputs, nchunks = num_sequences + 
    total_seqlen//CHUNK_SIZE.
    2. Final K state has an additional chunk_size dimension since triton does not allow indexing within a chunk. We
    pick the correct index in the wrapper.
    """
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    if IS_VARLEN:
        pid_seq = tl.program_id(2)
        seq_idx = pid_seq

        cu_seqlen_start = tl.load(Cu_Seqlens + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_end = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        total_seqlen = seqlen
        seqlen = cu_seqlen_end - cu_seqlen_start
        seq_offset = cu_seqlen_start
        chunk_offset = pid_seq + cu_seqlen_start // CHUNK_SIZE
    else:
        seq_idx = pid_batch
        seq_offset = 0
        chunk_offset = 0
    
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    # Compute head index for Q/K (supports Grouped Query Attention)
    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)

    # Setup input pointers
    q_ptr = Q + pid_batch * stride_q_batch + head_idx_qk * stride_q_head + seq_offset * stride_q_seqlen
    k_ptr = K + pid_batch * stride_k_batch + head_idx_qk * stride_k_head + seq_offset * stride_k_seqlen
    v_ptr = V + pid_batch * stride_v_batch + pid_head * stride_v_head + seq_offset * stride_v_seqlen
    adt_ptr = ADT + pid_batch * stride_adt_batch + pid_head * stride_adt_head + seq_offset * stride_adt_seqlen
    dt_ptr = DT + pid_batch * stride_dt_batch + pid_head * stride_dt_head + seq_offset * stride_dt_seqlen
    trap_ptr = Trap + pid_batch * stride_trap_batch + pid_head * stride_trap_head + seq_offset * stride_trap_seqlen
    q_bias_ptr = Q_bias + pid_head * stride_q_bias_head
    k_bias_ptr = K_bias + pid_head * stride_k_bias_head
    angle_ptr = Angles + pid_batch * stride_angles_batch + pid_head * stride_angles_head + seq_offset * stride_angles_seqlen
    
    if HAS_D:
        D_ptr = D + pid_head * stride_d_head
        D_val = tl.load(D_ptr).to(tl.float32)
    if HAS_Z:
        z_ptr = Z + pid_batch * stride_z_batch + pid_head * stride_z_head + seq_offset * stride_z_seqlen
    
    # State pointers use seq_idx (unified for batched and varlen)
    if HAS_INITIAL_STATES:
        init_ssm_state_ptr = Initial_SSM_State + seq_idx * stride_init_ssm_state_seq + pid_head * stride_init_ssm_state_head
        init_k_state_ptr = Initial_K_State + seq_idx * stride_init_k_state_seq + pid_head * stride_init_k_state_head
        init_v_state_ptr = Initial_V_State + seq_idx * stride_init_v_state_seq + pid_head * stride_init_v_state_head

    # Setup output pointers
    o_ptr = Out + pid_batch * stride_o_batch + pid_head * stride_o_head + seq_offset * stride_o_seqlen
    if STORE_SSM_STATES_ADT_OUTV:
        out_v_ptr = Out_v + pid_batch * stride_o_v_batch + pid_head * stride_o_v_head + seq_offset * stride_o_v_seqlen
        ssm_states_ptr = SSM_States + pid_batch * stride_ssm_states_batch + pid_head * stride_ssm_states_head + chunk_offset * HEADDIM_QK * stride_ssm_states_qkdim
        da_cs_store_ptr = DA_CS_Store + pid_batch * stride_da_cs_store_batch + pid_head * stride_da_cs_store_head + seq_offset * stride_da_cs_store_seqlen
        da_cs_sum_store_ptr = DA_CS_SUM_Store + pid_batch * stride_da_cs_sum_store_batch + pid_head * stride_da_cs_sum_store_head + chunk_offset * stride_da_cs_sum_store_seqlen

    q_store_ptr = Q_store + pid_batch * stride_q_store_batch + pid_head * stride_q_store_head + seq_offset * stride_q_store_seqlen
    k_store_ptr = K_store + pid_batch * stride_k_store_batch + pid_head * stride_k_store_head + seq_offset * stride_k_store_seqlen
    qk_store_ptr = QK_store + pid_batch * stride_qk_store_batch + pid_head * stride_qk_store_head + seq_offset * stride_qk_store_seqlen
    scale_store_ptr = Scale_store + pid_batch * stride_scale_store_batch + pid_head * stride_scale_store_head + seq_offset * stride_scale_store_seqlen
    gamma_store_ptr = Gamma_store + pid_batch * stride_gamma_store_batch + pid_head * stride_gamma_store_head + seq_offset * stride_gamma_store_seqlen

    if RETURN_FINAL_STATES:
        final_ssm_state_ptr = Final_SSM_State + seq_idx * stride_final_ssm_state_seq + pid_head * stride_final_ssm_state_head
        final_k_state_ptr = Final_K_State + seq_idx * stride_final_k_state_seq + pid_head * stride_final_k_state_head

    # Create TMA tensor descriptors
    q_desc = tl.make_tensor_descriptor(
        q_ptr,
        shape=[seqlen, headdim_qk],
        strides=[stride_q_seqlen, stride_q_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr,
        shape=[seqlen, headdim_qk],
        strides=[stride_k_seqlen, stride_k_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr,
        shape=[seqlen, headdim_v],
        strides=[stride_v_seqlen, stride_v_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )
    if HAS_Z:
        z_desc = tl.make_tensor_descriptor(
            z_ptr,
            shape=[seqlen, headdim_v],
            strides=[stride_z_seqlen, stride_z_vdim],
            block_shape=[CHUNK_SIZE, HEADDIM_V],
        )
    
    q_store_desc = tl.make_tensor_descriptor(
        q_store_ptr,
        shape=[seqlen, headdim_qk],
        strides=[stride_q_store_seqlen, stride_q_store_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    k_store_desc = tl.make_tensor_descriptor(
        k_store_ptr,
        shape=[seqlen, headdim_qk],
        strides=[stride_k_store_seqlen, stride_k_store_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr,
        shape=[seqlen, headdim_v],
        strides=[stride_o_seqlen, stride_o_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )
    if STORE_SSM_STATES_ADT_OUTV:
        ssm_states_desc = tl.make_tensor_descriptor(
            ssm_states_ptr,
            shape=[headdim_v, num_chunks * headdim_qk],
            strides=[stride_ssm_states_vdim, stride_ssm_states_qkdim],
            block_shape=[HEADDIM_V, HEADDIM_QK],
        )

    # Phase 1: Preprocessing - Apply bias, rotary embeddings, compute QK dots.
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_seqlen = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_hd = tl.arange(0, HEADDIM_QK)
        offs_hdr = tl.arange(0, HEADDIM_QK // 2)

        # Load Q and K blocks via TMA
        q_pre_block = q_desc.load([chunk_start, 0])
        k_pre_block = k_desc.load([chunk_start, 0])
        
        # Load rotary angles
        angle_block = tl.load(
            angle_ptr + offs_seqlen[:, None] * stride_angles_seqlen + offs_hdr[None, :] * stride_angles_qkdim,
            mask=(offs_seqlen[:, None] < seqlen) & (offs_hdr[None, :] < headdim_angles), other=0.0
        )
        
        # Compute shifted gamma and scale
        dt = tl.load(dt_ptr + offs_seqlen * stride_dt_seqlen, mask=offs_seqlen < seqlen, other=0.0).to(tl.float32)
        dt_shifted = tl.load(
            dt_ptr + (offs_seqlen + 1) * stride_dt_seqlen, 
            mask=offs_seqlen + 1 < seqlen, other=0.0).to(tl.float32)
        trap = tl.load(trap_ptr + offs_seqlen * stride_trap_seqlen, mask=offs_seqlen < seqlen, other=0.0).to(tl.float32)
        trap = sigmoid_approx(trap)
        trap_shifted = tl.load(
            trap_ptr + (offs_seqlen + 1) * stride_trap_seqlen, 
            mask=offs_seqlen + 1 < seqlen, other=0.0).to(tl.float32)
        trap_shifted = sigmoid_approx(trap_shifted)

        shifted_gamma = dt_shifted * (1 - trap_shifted)
        gamma = dt * trap
        scale = shifted_gamma + gamma

        # Store scale and shifted gamma for backward pass
        tl.store(gamma_store_ptr + offs_seqlen * stride_gamma_store_seqlen, gamma, mask=offs_seqlen < seqlen)
        tl.store(scale_store_ptr + offs_seqlen * stride_scale_store_seqlen, scale, mask=offs_seqlen < seqlen)

        # Add biases to Q and K
        q_bias_block = tl.load(q_bias_ptr + offs_hd * stride_q_bias_qkdim, offs_hd < headdim_qk)
        q_pre_block += q_bias_block[None, :]
        k_bias_block = tl.load(k_bias_ptr + offs_hd * stride_k_bias_qkdim, offs_hd < headdim_qk)
        k_pre_block += k_bias_block[None, :]

        # Compute QK dot products for skip connection
        store_qk_dot = tl.dot(
            q_pre_block * k_pre_block,
            tl.full([HEADDIM_QK, 1], 1, dtype=q_pre_block.dtype)
        ).to(q_pre_block.dtype)
        store_qk_dot = store_qk_dot.reshape(CHUNK_SIZE)
        store_qk_dot *= gamma
        tl.store(qk_store_ptr + offs_seqlen * stride_qk_store_seqlen, store_qk_dot, mask=offs_seqlen < seqlen)
        
        # Compute rotary embedding cos/sin
        cos_block = cos_approx(angle_block.to(tl.float32))
        sin_block = sin_approx(angle_block.to(tl.float32))

        # Apply rotary embeddings to K and scale
        k0, k1 = tl.split(tl.reshape(k_pre_block, [CHUNK_SIZE, HEADDIM_QK // 2, 2]))
        ko0 = k0 * cos_block - k1 * sin_block
        ko1 = k0 * sin_block + k1 * cos_block
        k_pre_block = tl.reshape(tl.join(ko0, ko1), [CHUNK_SIZE, HEADDIM_QK]).to(k_pre_block.dtype)

        if chunk_idx == num_chunks - 1 and RETURN_FINAL_STATES:
            tl.store(final_k_state_ptr + tl.arange(0, CHUNK_SIZE)[:, None] * stride_final_k_state_chunk 
                + offs_hd[None, :] * stride_final_k_state_qkdim, 
                k_pre_block,
                mask=(offs_hd[None, :] < headdim_qk))
            
        k_pre_block *= scale[:, None]
        k_store_desc.store([chunk_start, 0], k_pre_block)

        # Apply rotary embeddings to Q
        q0, q1 = tl.split(tl.reshape(q_pre_block, [CHUNK_SIZE, HEADDIM_QK // 2, 2]))
        qo0 = q0 * cos_block - q1 * sin_block
        qo1 = q0 * sin_block + q1 * cos_block
        q_pre_block = tl.reshape(tl.join(qo0, qo1), [CHUNK_SIZE, HEADDIM_QK]).to(q_pre_block.dtype)
        q_store_desc.store([chunk_start, 0], q_pre_block)

    # Phase 2: Main computation and output generation.
    if HAS_INITIAL_STATES:
        acc_ssm_states = tl.load(
            init_ssm_state_ptr + tl.arange(0, HEADDIM_V)[:, None] * stride_init_ssm_state_vdim 
            + tl.arange(0, HEADDIM_QK)[None, :] * stride_init_ssm_state_qkdim,
            mask= (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (tl.arange(0, HEADDIM_QK)[None, :] < headdim_qk),
            other=0.0).to(tl.float32)
        input_k_state = tl.load(
            init_k_state_ptr + tl.arange(0, HEADDIM_QK) * stride_init_k_state_qkdim,
            mask=tl.arange(0, HEADDIM_QK) < headdim_qk, other=0.0).to(tl.float32)
        input_v_state = tl.load(
            init_v_state_ptr + tl.arange(0, HEADDIM_V) * stride_init_v_state_vdim,
            mask=tl.arange(0, HEADDIM_V) < headdim_v, other=0.0).to(tl.float32)

        dt_scalar = tl.load(dt_ptr).to(tl.float32)
        trap_scalar = tl.load(trap_ptr).to(tl.float32)
        trap_scalar = sigmoid_approx(trap_scalar)
        # Step on the SSM states with input K/V states to account for trapezoidal discretization
        acc_ssm_states += input_v_state[:, None] * input_k_state[None, :] * dt_scalar * (1 - trap_scalar)
    else:
        acc_ssm_states = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)

    if HAS_D:
        D_val = tl.load(D_ptr).to(tl.float32)
    else:
        D_val = 0.0

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_seqlen = chunk_start + tl.arange(0, CHUNK_SIZE)

        # Load decay factors (log2 scale for exp2 computation)
        adt_ptrs = adt_ptr + offs_seqlen * stride_adt_seqlen
        da = tl.load(adt_ptrs, mask=offs_seqlen < seqlen, other=0.0) * 1.44269504089  # log2(e)

        # Load preprocessed Q, K, V blocks
        q_block = q_store_desc.load([chunk_start, 0])
        k_block = k_store_desc.load([chunk_start, 0])
        v_block = v_desc.load([chunk_start, 0])
        if HAS_Z:
            z_block = z_desc.load([chunk_start, 0])

        # Compute cumulative decay for this chunk
        da_cs = tl.cumsum(da)
        da_cs_last = tl.sum(da)
        da_cs_rev = da_cs_last - da_cs

        # Store decay info for backward pass
        if STORE_SSM_STATES_ADT_OUTV:
            tl.store(da_cs_store_ptr + offs_seqlen * stride_da_cs_store_seqlen, da_cs, mask=offs_seqlen < seqlen)
            tl.store(da_cs_sum_store_ptr + chunk_idx * stride_da_cs_sum_store_seqlen, da_cs_last)

        # Output contribution from previous state: Q @ SSM_States^T * exp(da_cs)
        acc_o = tl.dot(q_block, tl.trans(acc_ssm_states).to(q_block.dtype))
        acc_o *= tl.math.exp2(da_cs)[:, None]

        # Output contribution from current chunk: causal(Q @ K^T * exp(decay)) @ V
        # NOTE: We compute the (i,i) component using QK dot to prevent non-causal numerical leakage
        s_block = tl.dot(q_block, tl.trans(k_block))
        s_block *= tl.math.exp2(tl.minimum((da_cs[:, None] - da_cs[None, :]), 0.0))
        s_block = tl.where(
            tl.arange(0, CHUNK_SIZE)[:, None] > tl.arange(0, CHUNK_SIZE)[None, :], 
            s_block, 
            0.0
        )
        acc_o += tl.dot(s_block.to(v_block.dtype), v_block)

        # Add D-skip connection and subtract QK dot contribution
        qk_dot = tl.load(qk_store_ptr + offs_seqlen * stride_qk_store_seqlen, mask=offs_seqlen < seqlen, other=0.0)
        acc_o += (D_val + qk_dot)[:, None] * v_block

        if STORE_SSM_STATES_ADT_OUTV:
            tl.store(out_v_ptr + offs_seqlen[:, None] * stride_o_v_seqlen 
                + tl.arange(0, HEADDIM_V)[None, :] * stride_o_v_vdim, acc_o, 
                mask=(offs_seqlen[:, None] < seqlen) & (tl.arange(0, HEADDIM_V)[None, :] < headdim_v))

        # Apply Z-gating if present
        if HAS_Z:
            acc_o = acc_o * silu(z_block.to(tl.float32))

        # Store output
        o_desc.store([chunk_start, 0], acc_o)

        if STORE_SSM_STATES_ADT_OUTV:
            ssm_states_desc.store([0, chunk_idx * headdim_qk], acc_ssm_states.to(ssm_states_desc.dtype))

        # Update recurrent states
        scale = tl.math.exp2(da_cs_rev)
        v_block *= scale[:, None]
        acc_ssm_states = acc_ssm_states * tl.math.exp2(da_cs_last) + tl.dot(
            tl.trans(v_block).to(k_block.dtype), k_block
        )

    # Store final states if requested
    if RETURN_FINAL_STATES:
        tl.store(final_ssm_state_ptr + tl.arange(0, HEADDIM_V)[:, None] * stride_final_ssm_state_vdim 
            + tl.arange(0, HEADDIM_QK)[None, :] * stride_final_ssm_state_qkdim, 
            acc_ssm_states,
            mask=(tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (tl.arange(0, HEADDIM_QK)[None, :] < headdim_qk))

# Memory Allocator for TMA Descriptors
def _alloc_fn(size: int, alignment: int, stream: Optional[int]):
    """Custom allocator for TMA descriptor global memory allocation."""
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(_alloc_fn)

def mamba3_siso_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Initial_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    chunk_size: int = 64,
    store_states_adt_outv: bool = False,
    return_final_states: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Mamba-3 forward pass wrapper.
    
    Args:
        Q: Query tensor                 (batch, seqlen, nheads_qk, headdim_qk)
        K: Key tensor                   (batch, seqlen, nheads_qk, headdim_qk)
        V: Value tensor                 (batch, seqlen, nheads, headdim_v)
        ADT: Decay tensor               (batch, nheads, seqlen)
        DT: DT tensor                   (batch, nheads, seqlen)
        Trap: Trap tensor               (batch, nheads, seqlen)
        Q_bias: Query bias              (nheads, headdim_qk)
        K_bias: Key bias                (nheads, headdim_qk)
        Angles: Rotary angles           (batch, seqlen, nheads, headdim_angles)
            - headdim_angles <= headdim_qk // 2 and headdim_angles % 2 == 0.
        D: Skip connection weight       (nheads,)
        Z: Gating tensor                (batch, seqlen, nheads, headdim_v)
            - Applies SiLU gating: out = out * silu(Z).
        Initial_States: Tuple of (SSM_State, K_State, V_State)
            SSM State shape:        (num_sequences, nheads, headdim_v, headdim_qk).
            K state shape:          (num_sequences, nheads, headdim_qk).
            V state shape:          (num_sequences, nheads, headdim_v).
                - K state is post bias and rotation and pre scaling
        cu_seqlens: Cumulative sequence lengths (num_sequences + 1,) for varlen
        chunk_size: Chunk size for processing
        store_states_adt_outv: Store intermediate states for backward pass
        return_final_states: Return final states
        
    Returns:
        Out: Output tensor                      (batch, seqlen, nheads, headdim_v)
        Out_v: Pre-gate output tensor           (batch, seqlen, nheads, headdim_v) (if store_states_adt_outv)
        SSM_States: Per-chunk SSM States        (batch, nheads, headdim_v, nchunks * headdim_qk) (if store_states_adt_outv)
        DA_CS_Store: Cumulative decay           (batch, nheads, seqlen) (if store_states_adt_outv)
        DA_CS_SUM_Store: Chunk decay sum        (batch, nheads, nchunks) (if store_states_adt_outv)
        Q_store: Rotated Q+bias                 (batch, seqlen, nheads, headdim_qk) (None if store_states_adt_outv=False)
        K_store: Rotated K+bias                 (batch, seqlen, nheads, headdim_qk) (None if store_states_adt_outv=False)
        QK_store: QK dot products               (batch, nheads, seqlen) (None if store_states_adt_outv=False)
        Scale_store: Scale factors              (batch, nheads, seqlen) (None if store_states_adt_outv=False)
        Gamma_store: Gamma factors              (batch, nheads, seqlen) (None if store_states_adt_outv=False)
        Final States: Final output state (None if return_output_state=False)
            Final SSM State                (num_sequences, nheads, headdim_v, headdim_qk)
            Final K state                  (num_sequences, nheads, headdim_qk)
            Final V state                  (num_sequences, nheads, headdim_v)
    
    Notes:
        1. For varlen mode: batch must be 1, cu_seqlens required
        2. num_sequences = batch for batched mode, len(cu_seqlens)-1 for varlen
        3. nheads % nheads_qk == 0
        4. nchunks = ceil(seqlen / chunk_size) for batched mode, num_sequences + total_seqlen//chunk_size for varlen mode.
    
    COMMENT:
        Design choice to store: Q_store, K_store, QK_store, is primarily an artifact of Triton's
        lack of programmatic access to shared memory---In the forward pass, we compute, store and then re-load
        these tensors in shared memory (using TMA) to prevent register spilling.
        
    """
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    device = Q.device
    is_varlen = cu_seqlens is not None
    assert seqlen > 0, "Sequence length must be greater than 0"

    # Determine number of sequences
    if is_varlen:
        assert batch == 1, "Varlen mode requires batch=1"
        num_sequences = cu_seqlens.shape[0] - 1
    else:
        num_sequences = batch
        cu_seqlens = None

    # Validate shapes
    assert Q.shape == K.shape, f"Q and K shape mismatch: {Q.shape} vs {K.shape}"
    assert nheads % nheads_qk == 0, f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})"
    assert ADT.shape == (batch, nheads, seqlen)
    assert DT.shape == (batch, nheads, seqlen)
    assert Trap.shape == (batch, nheads, seqlen)
    assert Q_bias.shape == (nheads, headdim_qk)
    assert K_bias.shape == (nheads, headdim_qk)
    
    headdim_angles = Angles.shape[-1]
    assert headdim_angles <= headdim_qk // 2 and headdim_angles % 2 == 0
    assert Angles.shape == (batch, seqlen, nheads, headdim_angles)
    
    if D is not None:
        assert D.shape == (nheads,)
    if Z is not None:
        assert Z.shape == (batch, seqlen, nheads, headdim_v)
    
    if Initial_States is not None:
        Init_SSM_State, Init_K_State, Init_V_State = Initial_States
        assert Init_SSM_State.shape == (num_sequences, nheads, headdim_v, headdim_qk), \
            f"Initial_States[0] shape mismatch: expected {(num_sequences, nheads, headdim_v, headdim_qk)}, got {Init_SSM_State.shape}"
        assert Init_K_State.shape == (num_sequences, nheads, headdim_qk), \
            f"Initial_States[1] shape mismatch: expected {(num_sequences, nheads, headdim_qk)}, got {Init_K_State.shape}"
        assert Init_V_State.shape == (num_sequences, nheads, headdim_v), \
            f"Initial_States[2] shape mismatch: expected {(num_sequences, nheads, headdim_v)}, got {Init_V_State.shape}"
    else:
        Init_SSM_State, Init_K_State, Init_V_State = None, None, None

    # Ensure contiguous
    Q = Q.contiguous() if not Q.is_contiguous() else Q
    K = K.contiguous() if not K.is_contiguous() else K
    V = V.contiguous() if not V.is_contiguous() else V
    ADT = ADT.contiguous() if not ADT.is_contiguous() else ADT
    DT = DT.contiguous() if not DT.is_contiguous() else DT
    Trap = Trap.contiguous() if not Trap.is_contiguous() else Trap
    Q_bias = Q_bias.contiguous() if not Q_bias.is_contiguous() else Q_bias
    K_bias = K_bias.contiguous() if not K_bias.is_contiguous() else K_bias
    Angles = Angles.contiguous() if not Angles.is_contiguous() else Angles
    
    if D is not None:
        D = D.contiguous() if not D.is_contiguous() else D
    if Z is not None:
        Z = Z.contiguous() if not Z.is_contiguous() else Z
    if Initial_States is not None:
        Init_SSM_State = Init_SSM_State.contiguous() if not Init_SSM_State.is_contiguous() else Init_SSM_State
        Init_K_State = Init_K_State.contiguous() if not Init_K_State.is_contiguous() else Init_K_State
        Init_V_State = Init_V_State.contiguous() if not Init_V_State.is_contiguous() else Init_V_State
    
    # Calculate nchunks
    if is_varlen:
        nchunks = num_sequences + seqlen // chunk_size
    else:
        nchunks = (seqlen + chunk_size - 1) // chunk_size

    # Allocate output tensors
    Out = torch.empty((batch, seqlen, nheads, headdim_v), device=device, dtype=V.dtype)
    if store_states_adt_outv:
        SSM_States = torch.zeros((batch, nheads, headdim_v, nchunks * headdim_qk), device=device, dtype=torch.bfloat16)
        DA_CS_Store = torch.empty((batch, nheads, seqlen), device=device, dtype=torch.float32)
        DA_CS_SUM_Store = torch.zeros((batch, nheads, nchunks), device=device, dtype=torch.float32)
        Out_v = torch.empty((batch, seqlen, nheads, headdim_v), device=device, dtype=V.dtype)
    else:
        SSM_States, DA_CS_Store, DA_CS_SUM_Store, Out_v = None, None, None, None
    
    Q_store = torch.empty((batch, seqlen, nheads, headdim_qk), device=device, dtype=Q.dtype)
    K_store = torch.empty((batch, seqlen, nheads, headdim_qk), device=device, dtype=K.dtype)
    QK_store = torch.empty((batch, nheads, seqlen), device=device, dtype=torch.float32)
    Scale_store = torch.empty((batch, nheads, seqlen), device=device, dtype=torch.float32)
    Gamma_store = torch.empty((batch, nheads, seqlen), device=device, dtype=torch.float32)
    
    if return_final_states:
        Final_SSM_State = torch.empty((num_sequences, nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32)
        Final_K_State = torch.empty((num_sequences, nheads, chunk_size, headdim_qk), device=device, dtype=torch.float32)
    else:
        Final_SSM_State, Final_K_State = None, None

    HEADDIM_V = triton.next_power_of_2(headdim_v)
    HEADDIM_QK = triton.next_power_of_2(headdim_qk)

    # Grid setup
    if is_varlen:
        grid = (nheads, batch, num_sequences) # batch = 1
    else:
        grid = (nheads, batch)

    mamba3_siso_fwd_kernel[grid](
        # Inputs
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, 
        Init_SSM_State, Init_K_State, Init_V_State, cu_seqlens,
        # Outputs
        Out, Out_v, SSM_States, DA_CS_Store, DA_CS_SUM_Store, 
        Q_store, K_store, QK_store, Scale_store, Gamma_store,
        Final_SSM_State, Final_K_State,
        # Input strides
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        ADT.stride(0), ADT.stride(1), ADT.stride(2),
        DT.stride(0), DT.stride(1), DT.stride(2),
        Trap.stride(0), Trap.stride(1), Trap.stride(2),
        Q_bias.stride(0), Q_bias.stride(1),
        K_bias.stride(0), K_bias.stride(1),
        Angles.stride(0), Angles.stride(1), Angles.stride(2), Angles.stride(3),
        D.stride(0) if D is not None else 0,
        Z.stride(0) if Z is not None else 0,
        Z.stride(1) if Z is not None else 0,
        Z.stride(2) if Z is not None else 0,
        Z.stride(3) if Z is not None else 0,
        Init_SSM_State.stride(0) if Init_SSM_State is not None else 0,
        Init_SSM_State.stride(1) if Init_SSM_State is not None else 0,
        Init_SSM_State.stride(2) if Init_SSM_State is not None else 0,
        Init_SSM_State.stride(3) if Init_SSM_State is not None else 0,
        Init_K_State.stride(0) if Init_K_State is not None else 0,
        Init_K_State.stride(1) if Init_K_State is not None else 0,
        Init_K_State.stride(2) if Init_K_State is not None else 0,
        Init_V_State.stride(0) if Init_V_State is not None else 0,
        Init_V_State.stride(1) if Init_V_State is not None else 0,
        Init_V_State.stride(2) if Init_V_State is not None else 0,
        cu_seqlens.stride(0) if cu_seqlens is not None else 0,
        # Output strides
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Out_v.stride(0) if Out_v is not None else 0,
        Out_v.stride(1) if Out_v is not None else 0,
        Out_v.stride(2) if Out_v is not None else 0,
        Out_v.stride(3) if Out_v is not None else 0,
        SSM_States.stride(0) if SSM_States is not None else 0,
        SSM_States.stride(1) if SSM_States is not None else 0,
        SSM_States.stride(2) if SSM_States is not None else 0,
        SSM_States.stride(3) if SSM_States is not None else 0,
        DA_CS_Store.stride(0) if DA_CS_Store is not None else 0,
        DA_CS_Store.stride(1) if DA_CS_Store is not None else 0,
        DA_CS_Store.stride(2) if DA_CS_Store is not None else 0,
        DA_CS_SUM_Store.stride(0) if DA_CS_SUM_Store is not None else 0,
        DA_CS_SUM_Store.stride(1) if DA_CS_SUM_Store is not None else 0,
        DA_CS_SUM_Store.stride(2) if DA_CS_SUM_Store is not None else 0,
        Q_store.stride(0), Q_store.stride(1), Q_store.stride(2), Q_store.stride(3),
        K_store.stride(0), K_store.stride(1), K_store.stride(2), K_store.stride(3),
        QK_store.stride(0), QK_store.stride(1), QK_store.stride(2),
        Scale_store.stride(0), Scale_store.stride(1), Scale_store.stride(2),
        Gamma_store.stride(0), Gamma_store.stride(1), Gamma_store.stride(2),
        Final_SSM_State.stride(0) if Final_SSM_State is not None else 0,
        Final_SSM_State.stride(1) if Final_SSM_State is not None else 0,
        Final_SSM_State.stride(2) if Final_SSM_State is not None else 0,
        Final_SSM_State.stride(3) if Final_SSM_State is not None else 0,
        Final_K_State.stride(0) if Final_K_State is not None else 0,
        Final_K_State.stride(1) if Final_K_State is not None else 0,
        Final_K_State.stride(2) if Final_K_State is not None else 0,
        Final_K_State.stride(3) if Final_K_State is not None else 0,
        # Dimensions
        seqlen, nheads_qk, headdim_qk, headdim_v, headdim_angles,
        # Compile-time constants
        chunk_size,
        HEADDIM_QK,
        HEADDIM_V,
        STORE_SSM_STATES_ADT_OUTV=store_states_adt_outv,
        HAS_INITIAL_STATES=Initial_States is not None,
        RETURN_FINAL_STATES=return_final_states,
        HAS_D=D is not None,
        HAS_Z=Z is not None,
        IS_VARLEN=is_varlen,
    )

    Final_States = None
    if return_final_states:
        if is_varlen:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            last_chunk_pos = (seq_lens - 1) % chunk_size 

            final_k = Final_K_State[
                torch.arange(num_sequences, device=device),
                :, 
                last_chunk_pos,
                :
            ]
            
            last_token_idx = cu_seqlens[1:] - 1
            final_v = V[0, last_token_idx]
        else:
            k_state_idx = (seqlen - 1) % chunk_size
            final_k = Final_K_State[:, :, k_state_idx, :]
            final_v = V[:, -1]
        
        Final_States = (Final_SSM_State, final_k, final_v)

    return (Out, Out_v, SSM_States, DA_CS_Store, DA_CS_SUM_Store, 
            Q_store, K_store, QK_store, Scale_store, Gamma_store, Final_States)