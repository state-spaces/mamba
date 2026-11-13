"""
Mamba-3 SISO Forward Pass Triton Kernel.

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

from typing import Optional, Tuple
import math

import torch

import triton
import triton.language as tl
from mamba_ssm.ops.triton.mamba3.utils import cos_approx, sin_approx, silu, sigmoid_approx
from mamba_ssm.ops.triton.mamba3.npu_compat import (
    alloc_device,
    autotune_regs,
    effective_chunk_size,
    make_config,
    precise_dqkv,
    set_allocator_safely,
    stage_choices,
)

# Ascend launches a 1D ``coreDim`` that must fit in UINT16_MAX.  The logical
# grid is ``nheads * batch`` (varlen: ``nheads * num_sequences``) and grows with
# the configuration, so it can exceed the limit for large batch / many sequences.
# Above it we switch to a grid-stride launch: at most ``_GRID_STRIDE_LIMIT``
# programs, each looping over several (head, batch) tiles.  Programs stay
# independent (the only recurrence is over chunks *inside* a program), so this
# is a pure scheduling change.  ``num_programs`` is passed explicitly instead of
# using ``tl.num_programs(0)`` because Triton-Ascend may fold the logical grid
# onto the physical AI cores, making that value unreliable.
_MAX_COREDIM = 65535
_GRID_STRIDE_LIMIT = _MAX_COREDIM


def _grid_programs(num_tiles: int) -> int:
    """Logical program count: all tiles when under the cap, else the cap."""
    return min(int(num_tiles), int(_GRID_STRIDE_LIMIT))


# Test-observable grid-stride signal.  Every ``mamba3_siso_fwd`` call records
# its launch geometry here so tests can assert that the capped path was really
# taken instead of merely inferring it from a successful launch:
#   * ``num_tiles``      logical tiles, ``nheads*batch`` / ``nheads*num_sequences``;
#   * ``num_programs``   launched grid extent (``<= _MAX_COREDIM``);
#   * ``grid_stride``    ``True`` iff ``num_programs < num_tiles``, i.e. the
#                        kernel's per-program tile loop runs more than once;
#   * ``calls`` / ``grid_stride_calls``  wrapper-call counters.
# PHASE=0 and PHASE=1 share the same grid, so one call updates this once.
_GRID_STRIDE_STATE = {
    "num_tiles": 0,
    "num_programs": 0,
    "grid_stride": False,
    "calls": 0,
    "grid_stride_calls": 0,
}


def set_grid_stride_limit(limit: Optional[int]) -> None:
    """Test hook: lower the effective ``coreDim`` cap (``None`` restores 65535).

    Production leaves this at ``_MAX_COREDIM``.  Lowering it forces the exact
    capped-grid + inner-tile-loop path on a problem small enough for one card,
    without changing the kernel code that the real overflow exercises.
    """
    global _GRID_STRIDE_LIMIT
    if limit is None:
        _GRID_STRIDE_LIMIT = _MAX_COREDIM
        return
    limit = int(limit)
    if not 1 <= limit <= _MAX_COREDIM:
        raise ValueError(f"grid-stride limit must be in [1, {_MAX_COREDIM}], got {limit}")
    _GRID_STRIDE_LIMIT = limit


def get_grid_stride_state() -> dict:
    """Snapshot of the last wrapper call's launch geometry (test hook)."""
    return dict(_GRID_STRIDE_STATE)


def reset_grid_stride_state() -> None:
    """Reset the test-observable counters (does not change the limit)."""
    _GRID_STRIDE_STATE.update(
        num_tiles=0, num_programs=0, grid_stride=False,
        calls=0, grid_stride_calls=0,
    )


def _prune_mamba3_siso_fwd_configs(configs, named_args, **kwargs):
    """Prune autotuning configs for mamba3_siso_fwd_kernel based on GPU architecture.

    On Blackwell GPUs (compute capability major 10 or 12), multi-stage software
    pipelining (num_stages > 1) silently corrupts the forward pass output of this
    recurrent kernel. This is a Triton codegen issue reproduced across Triton 3.5.1,
    3.6.0, and 3.7.0. Restrict to num_stages=1 on Blackwell to ensure correctness.
    Non-Blackwell architectures (e.g., Hopper/SM90) retain the full autotune space.

    The decision is driven purely by the *device type* of the input, so it is well
    defined on every backend: a CUDA device probes compute capability (Blackwell
    pruning), while an Ascend ``npu`` device keeps the full config space (the
    CUDA-codegen bug does not apply there).  Keeping the check device-driven also
    makes the behaviour identical for the CUDA-mocked unit test regardless of the
    backend that imported this module.

    Ascend note: on NPU this hook is a *constant no-op* -- ``device_type``
    is ``"npu"``, so it returns ``configs`` unchanged and prunes nothing.
    """
    device = named_args["Q"].device
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    if device_type != "cuda":
        return configs
    major, _ = torch.cuda.get_device_capability(device)
    if major in (10, 12):
        return [config for config in configs if config.num_stages == 1]
    return configs


@triton.autotune(
    configs=[
        make_config({}, s, w, r)
        for s in stage_choices([1, 2, 3])
        for w in [2, 4, 8]
        for r in autotune_regs([None, 128, 256])
    ],
    key=[
        "CHUNK_SIZE", "HEADDIM_QK", "HEADDIM_V", "STORE_SSM_STATES_ADT_OUTV", "HAS_D",
        "HAS_Z", "HAS_INITIAL_STATES", "RETURN_FINAL_STATES", "IS_VARLEN", "PHASE"],
    prune_configs_by={
        "early_config_prune": _prune_mamba3_siso_fwd_configs,
    },
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
    nheads, seqlen, nheads_qk, headdim_qk, headdim_v, headdim_angles, num_cblocks,
    num_tiles, num_programs,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    STORE_SSM_STATES_ADT_OUTV: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    RETURN_FINAL_STATES: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    PHASE: tl.constexpr,
):
    """
    Mamba-3 forward kernel.

    Grid: flat ``(num_programs,)``.  The logical tiles are ``(nheads, batch)``
    for batched and ``(nheads, num_sequences)`` for varlen; when that count
    exceeds the Ascend ``coreDim`` limit each program loops over several
    tiles (grid-stride).

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
    # Grid-stride over independent (head, batch/sequence) tiles.  With
    # num_programs == num_tiles (the common case) each program handles exactly
    # one tile and the mapping is identical to the former 2D/3D grid.  Above the
    # Ascend ``coreDim`` cap the wrapper launches at most ``_MAX_COREDIM``
    # programs and each one loops over the remaining tiles; tiles are fully
    # independent (the only state recurrence is over chunks *inside* a
    # program), so this is a pure scheduling change.
    #
    # The iteration count is per-program (``cdiv``) so the loop uses a constant
    # step of 1: Triton's ``range`` does not accept a runtime step.
    pid = tl.program_id(0)
    for i in range(0, tl.cdiv(num_tiles - pid, num_programs)):
        tile_idx = pid + i * num_programs
        pid_head = tile_idx % nheads

        if IS_VARLEN:
            pid_batch = 0
            pid_seq = tile_idx // nheads
            seq_idx = pid_seq

            cu_seqlen_start = tl.load(Cu_Seqlens + pid_seq * stride_cu_seqlen).to(tl.int32)
            cu_seqlen_end = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
            seqlen = cu_seqlen_end - cu_seqlen_start
            seq_offset = cu_seqlen_start
            chunk_offset = pid_seq + cu_seqlen_start // CHUNK_SIZE
        else:
            pid_batch = tile_idx // nheads
            seq_idx = pid_batch
            seq_offset = 0
            chunk_offset = 0

        num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

        # Compute head index for Q/K (supports Grouped Query Attention).
        # ``nheads`` is passed explicitly instead of using ``tl.num_programs(0)``:
        # Triton-Ascend folds a logical grid onto the physical AI cores, so the
        # program count observed inside the kernel is not guaranteed to be the
        # logical grid extent.
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

        # NOTE (Ascend migration): upstream used TMA tensor descriptors here
        # (``tl.make_tensor_descriptor``) for Q/K/V/Z, the store buffers and the
        # SSM states.  Triton-Ascend does not support TMA, so every ``desc.load``/
        # ``desc.store`` below is replaced with a plain strided pointer access and
        # an explicit mask that reproduces TMA's out-of-bounds zero padding.

        if PHASE == 0:
            # ------------------------------------------------------------------
            # Phase 1: preprocessing (bias + interleaved rotary + QK dot).
            #
            # Ascend UB mitigation:
            #   * the chunk is processed in BLOCK_C row sub-tiles;
            #   * Q/K are loaded/rotated/stored as even/odd headdim halves so no
            #     bf16 tl.split/tl.join is needed (Triton-Ascend cannot lower those:
            #     "cannot align ... axis");
            #   * Q/K bias vectors are chunk-independent and hoisted.
            # This phase is compiled as its own specialization (PHASE=0) so it does
            # not share UB with Phase 2.
            # ------------------------------------------------------------------
            offs_hd = tl.arange(0, HEADDIM_QK)
            offs_hdr = tl.arange(0, HEADDIM_QK // 2)
            hd_mask = offs_hd < headdim_qk
            hdr_mask = offs_hdr < headdim_angles
            offs_e = 2 * offs_hdr
            offs_o = 2 * offs_hdr + 1
            e_mask = offs_e < headdim_qk
            o_mask = offs_o < headdim_qk
            q_bias_e = tl.load(q_bias_ptr + offs_e * stride_q_bias_qkdim, mask=e_mask, other=0.0)
            q_bias_o = tl.load(q_bias_ptr + offs_o * stride_q_bias_qkdim, mask=o_mask, other=0.0)
            k_bias_e = tl.load(k_bias_ptr + offs_e * stride_k_bias_qkdim, mask=e_mask, other=0.0)
            k_bias_o = tl.load(k_bias_ptr + offs_o * stride_k_bias_qkdim, mask=o_mask, other=0.0)

            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * CHUNK_SIZE
                for cb in range(0, num_cblocks):
                    c0 = cb * BLOCK_C
                    rows = chunk_start + c0 + tl.arange(0, BLOCK_C)
                    rmask = rows < seqlen

                    # Q/K even/odd headdim halves (strided loads; no split/join)
                    q_e = tl.load(q_ptr + rows[:, None] * stride_q_seqlen + offs_e[None, :] * stride_q_qkdim,
                                  mask=rmask[:, None] & e_mask[None, :], other=0.0)
                    q_o = tl.load(q_ptr + rows[:, None] * stride_q_seqlen + offs_o[None, :] * stride_q_qkdim,
                                  mask=rmask[:, None] & o_mask[None, :], other=0.0)
                    k_e = tl.load(k_ptr + rows[:, None] * stride_k_seqlen + offs_e[None, :] * stride_k_qkdim,
                                  mask=rmask[:, None] & e_mask[None, :], other=0.0)
                    k_o = tl.load(k_ptr + rows[:, None] * stride_k_seqlen + offs_o[None, :] * stride_k_qkdim,
                                  mask=rmask[:, None] & o_mask[None, :], other=0.0)

                    # Load rotary angles
                    angle_block = tl.load(
                        angle_ptr + rows[:, None] * stride_angles_seqlen + offs_hdr[None, :] * stride_angles_qkdim,
                        mask=rmask[:, None] & hdr_mask[None, :], other=0.0,
                    )

                    # Compute shifted gamma and scale
                    dt = tl.load(dt_ptr + rows * stride_dt_seqlen, mask=rmask, other=0.0).to(tl.float32)
                    dt_shifted = tl.load(dt_ptr + (rows + 1) * stride_dt_seqlen,
                                         mask=(rows + 1) < seqlen, other=0.0).to(tl.float32)
                    trap = tl.load(trap_ptr + rows * stride_trap_seqlen, mask=rmask, other=0.0).to(tl.float32)
                    trap = sigmoid_approx(trap)
                    trap_shifted = tl.load(trap_ptr + (rows + 1) * stride_trap_seqlen,
                                           mask=(rows + 1) < seqlen, other=0.0).to(tl.float32)
                    trap_shifted = sigmoid_approx(trap_shifted)
                    shifted_gamma = dt_shifted * (1 - trap_shifted)
                    gamma = dt * trap
                    scale = shifted_gamma + gamma

                    # Store scale and shifted gamma for backward pass
                    tl.store(gamma_store_ptr + rows * stride_gamma_store_seqlen, gamma, mask=rmask)
                    tl.store(scale_store_ptr + rows * stride_scale_store_seqlen, scale, mask=rmask)

                    # Add biases to Q and K
                    q_e += q_bias_e[None, :]
                    q_o += q_bias_o[None, :]
                    k_e += k_bias_e[None, :]
                    k_o += k_bias_o[None, :]

                    # QK dot product for the skip connection (fp32 products + sum,
                    # matching an exactly-accumulating bf16 ``tl.dot``)
                    qk = (tl.sum(q_e.to(tl.float32) * k_e.to(tl.float32), axis=1)
                          + tl.sum(q_o.to(tl.float32) * k_o.to(tl.float32), axis=1))
                    store_qk_dot = qk.to(k_e.dtype) * gamma
                    tl.store(qk_store_ptr + rows * stride_qk_store_seqlen, store_qk_dot, mask=rmask)

                    # Interleaved rotary on the even/odd halves
                    cos_b = cos_approx(angle_block.to(tl.float32))
                    sin_b = sin_approx(angle_block.to(tl.float32))
                    qe_rot = q_e * cos_b - q_o * sin_b
                    qo_rot = q_e * sin_b + q_o * cos_b
                    ke_rot = k_e * cos_b - k_o * sin_b
                    ko_rot = k_e * sin_b + k_o * cos_b

                    if chunk_idx == num_chunks - 1 and RETURN_FINAL_STATES:
                        sel = c0 + tl.arange(0, BLOCK_C)
                        tl.store(final_k_state_ptr + sel[:, None] * stride_final_k_state_chunk
                                 + offs_e[None, :] * stride_final_k_state_qkdim,
                                 ke_rot, mask=e_mask[None, :])
                        tl.store(final_k_state_ptr + sel[:, None] * stride_final_k_state_chunk
                                 + offs_o[None, :] * stride_final_k_state_qkdim,
                                 ko_rot, mask=o_mask[None, :])

                    # Scale K and store
                    tl.store(k_store_ptr + rows[:, None] * stride_k_store_seqlen + offs_e[None, :] * stride_k_store_qkdim,
                             (ke_rot * scale[:, None]), mask=rmask[:, None] & e_mask[None, :])
                    tl.store(k_store_ptr + rows[:, None] * stride_k_store_seqlen + offs_o[None, :] * stride_k_store_qkdim,
                             (ko_rot * scale[:, None]), mask=rmask[:, None] & o_mask[None, :])

                    # Store rotated Q
                    tl.store(q_store_ptr + rows[:, None] * stride_q_store_seqlen + offs_e[None, :] * stride_q_store_qkdim,
                             qe_rot, mask=rmask[:, None] & e_mask[None, :])
                    tl.store(q_store_ptr + rows[:, None] * stride_q_store_seqlen + offs_o[None, :] * stride_q_store_qkdim,
                             qo_rot, mask=rmask[:, None] & o_mask[None, :])

        if PHASE == 1:
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

            offs_hd = tl.arange(0, HEADDIM_QK)
            hd_mask = offs_hd < headdim_qk
            offs_hdv = tl.arange(0, HEADDIM_V)
            hdv_mask = offs_hdv < headdim_v

            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * CHUNK_SIZE
                offs_seqlen = chunk_start + tl.arange(0, CHUNK_SIZE)
                seq2 = offs_seqlen < seqlen

                # Load decay factors (log2 scale for exp2 computation)
                da = tl.load(adt_ptr + offs_seqlen * stride_adt_seqlen, mask=seq2, other=0.0) * 1.44269504089  # log2(e)

                # Cumulative decay for this chunk (columns + recurrent state use it).
                da_cs = tl.cumsum(da)
                da_cs_last = tl.sum(da)
                da_cs_rev = da_cs_last - da_cs

                # Store decay info for backward pass
                if STORE_SSM_STATES_ADT_OUTV:
                    tl.store(da_cs_store_ptr + offs_seqlen * stride_da_cs_store_seqlen, da_cs, mask=seq2)
                    tl.store(da_cs_sum_store_ptr + chunk_idx * stride_da_cs_sum_store_seqlen, da_cs_last)

                # K and V of this chunk (all columns/rows are needed for attention and
                # for the recurrent-state update).
                k_block = tl.load(
                    k_store_ptr + offs_seqlen[:, None] * stride_k_store_seqlen + offs_hd[None, :] * stride_k_store_qkdim,
                    mask=seq2[:, None] & hd_mask[None, :], other=0.0,
                )
                k_t = tl.trans(k_block)  # (HEADDIM_QK, CHUNK_SIZE)
                v_block = tl.load(
                    v_ptr + offs_seqlen[:, None] * stride_v_seqlen + offs_hdv[None, :] * stride_v_vdim,
                    mask=seq2[:, None] & hdv_mask[None, :], other=0.0,
                )

                # Row sub-blocks of the output.  UB mitigation (Ascend): never
                # materialise the full CHUNK_SIZE x CHUNK_SIZE gate matrix nor a full
                # fp32 CHUNK_SIZE x V accumulator -- each block only holds BLOCK_C rows,
                # so every intermediate is O(BLOCK_C x ...).
                for cb in range(0, num_cblocks):
                    c0 = cb * BLOCK_C
                    rows = chunk_start + c0 + tl.arange(0, BLOCK_C)
                    rmask = rows < seqlen

                    # Rows-only cumulative decay = prefix before this sub-block + the
                    # in-block cumsum (avoids materialising / slicing the full da_cs).
                    base = tl.sum(tl.where(tl.arange(0, CHUNK_SIZE) < c0, da, 0.0))
                    da_row = tl.load(adt_ptr + rows * stride_adt_seqlen, mask=rmask, other=0.0) * 1.44269504089
                    da_row_cs = base + tl.cumsum(da_row)

                    q_sub = tl.load(
                        q_store_ptr + rows[:, None] * stride_q_store_seqlen + offs_hd[None, :] * stride_q_store_qkdim,
                        mask=rmask[:, None] & hd_mask[None, :], other=0.0,
                    )
                    v_sub = tl.load(
                        v_ptr + rows[:, None] * stride_v_seqlen + offs_hdv[None, :] * stride_v_vdim,
                        mask=rmask[:, None] & hdv_mask[None, :], other=0.0,
                    )

                    # Contribution from the running state: Q @ SSM_States^T * exp(da_cs)
                    acc_o = tl.dot(q_sub, tl.trans(acc_ssm_states).to(q_sub.dtype))
                    acc_o *= tl.math.exp2(da_row_cs)[:, None]

                    # Causal intra-chunk term: (Q @ K^T * exp(decay)) @ V, restricted to
                    # this block's rows -> (BLOCK_C x CHUNK_SIZE) gate instead of
                    # (CHUNK_SIZE x CHUNK_SIZE).
                    s_block = tl.dot(q_sub, k_t)
                    s_block *= tl.math.exp2(tl.minimum(da_row_cs[:, None] - da_cs[None, :], 0.0))
                    s_block = tl.where(rows[:, None] > offs_seqlen[None, :], s_block, 0.0)
                    acc_o += tl.dot(s_block.to(v_sub.dtype), v_block)

                    # Add D-skip connection and the QK dot contribution
                    qk_dot = tl.load(qk_store_ptr + rows * stride_qk_store_seqlen, mask=rmask, other=0.0)
                    acc_o += (D_val + qk_dot)[:, None] * v_sub

                    if STORE_SSM_STATES_ADT_OUTV:
                        tl.store(out_v_ptr + rows[:, None] * stride_o_v_seqlen
                            + offs_hdv[None, :] * stride_o_v_vdim, acc_o,
                            mask=rmask[:, None] & hdv_mask[None, :])

                    # Apply Z-gating if present
                    if HAS_Z:
                        z_sub = tl.load(
                            z_ptr + rows[:, None] * stride_z_seqlen + offs_hdv[None, :] * stride_z_vdim,
                            mask=rmask[:, None] & hdv_mask[None, :], other=0.0,
                        )
                        acc_o = acc_o * silu(z_sub.to(tl.float32))

                    # Store output
                    tl.store(
                        o_ptr + rows[:, None] * stride_o_seqlen + offs_hdv[None, :] * stride_o_vdim,
                        acc_o,
                        mask=rmask[:, None] & hdv_mask[None, :],
                    )

                if STORE_SSM_STATES_ADT_OUTV:
                    ssm_v = offs_hdv[:, None]
                    ssm_k = chunk_idx * headdim_qk + offs_hd[None, :]
                    tl.store(
                        ssm_states_ptr + ssm_v * stride_ssm_states_vdim + ssm_k * stride_ssm_states_qkdim,
                        acc_ssm_states.to(SSM_States.dtype.element_ty),
                        mask=(ssm_v < headdim_v) & (ssm_k < num_chunks * headdim_qk),
                    )

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

# Memory allocator for Triton runtime scratch (device agnostic: cuda/npu).
def _alloc_fn(size: int, alignment: int, stream: Optional[int]):
    """Custom allocator for Triton runtime scratch memory."""
    return torch.empty(size, device=alloc_device(), dtype=torch.int8)
set_allocator_safely(_alloc_fn)

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

    # Ascend UB policy: clamp the chunk size so the per-program O(CHUNK_SIZE^2)
    # gate matrices + O(CHUNK_SIZE*HEADDIM) tiles + persistent recurrent state
    # fit the 192 KB Unified Buffer.  Chunking only reorders the same sums, so
    # the result is unchanged.
    chunk_size = effective_chunk_size(chunk_size, seqlen)

    # Determine number of sequences
    if is_varlen:
        assert batch == 1, "Varlen mode requires batch=1"
        num_sequences = cu_seqlens.shape[0] - 1
    else:
        num_sequences = batch
        cu_seqlens = None

    # Normalise memory layout BEFORE validating shapes: the kernel consumes
    # packed contiguous tensors, and a non-contiguous caller view (e.g. a
    # transposed Q) must be materialised first so that validation and the kernel
    # operate on the same tensor.  ``.contiguous()`` is a no-op when the input is
    # already packed, so the CUDA path is unaffected.
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
        Init_SSM_State = Init_SSM_State.contiguous() if not Init_SSM_State.is_contiguous() else Init_SSM_State
        Init_K_State = Init_K_State.contiguous() if not Init_K_State.is_contiguous() else Init_K_State
        Init_V_State = Init_V_State.contiguous() if not Init_V_State.is_contiguous() else Init_V_State
    else:
        Init_SSM_State, Init_K_State, Init_V_State = None, None, None

    # Calculate nchunks
    if is_varlen:
        nchunks = num_sequences + seqlen // chunk_size
    else:
        nchunks = (seqlen + chunk_size - 1) // chunk_size

    # Allocate output tensors
    Out = torch.empty((batch, seqlen, nheads, headdim_v), device=device, dtype=V.dtype)
    if store_states_adt_outv:
        # PLUT-50: the backward's dQ_mid/dADT inter-chunk terms read these
        # per-chunk states; a bf16 round-trip here caps them below the CANN fp32
        # matched_ratio gate even after the backward's own re-quantisation is
        # removed.  Keep them fp32 on the precise path (see
        # ``npu_compat.precise_dqkv``); bf16 otherwise.
        ssm_states_dtype = torch.float32 if precise_dqkv() else torch.bfloat16
        SSM_States = torch.zeros((batch, nheads, headdim_v, nchunks * headdim_qk), device=device, dtype=ssm_states_dtype)
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

    # Ascend UB policy: process the chunk dimension in BLOCK_C row sub-tiles so
    # every fp32/bf16 temporary inside the kernel is O(BLOCK_C x HEADDIM) /
    # O(BLOCK_C x CHUNK_SIZE) instead of O(CHUNK_SIZE x ...).  This is the real
    # lever against `ub overflow` for this kernel -- capping chunk_size alone has
    # no effect when chunk_size already equals CHUNK_SIZE.
    BLOCK_C = min(16, chunk_size)
    # Runtime trip count for the in-chunk sub-block loops.  Passing this as a
    # runtime value (instead of using the constexpr CHUNK_SIZE as the loop bound)
    # keeps those loops rolled, so the UB allocator reserves the sub-block body's
    # buffers once instead of once per unrolled copy.
    num_cblocks = chunk_size // BLOCK_C

    # Grid setup: one flat program per (head, batch) / (head, sequence) tile,
    # capped at the Ascend coreDim limit (UINT16_MAX).  Above the cap each
    # program grid-strides over several tiles (see kernel); PHASE=0 and PHASE=1
    # share this same grid.  ``num_tiles`` is the logical count the kernel walks.
    num_tiles = nheads * (num_sequences if is_varlen else batch)
    num_programs = _grid_programs(num_tiles)
    grid = (num_programs,)
    _GRID_STRIDE_STATE["num_tiles"] = num_tiles
    _GRID_STRIDE_STATE["num_programs"] = num_programs
    _GRID_STRIDE_STATE["grid_stride"] = num_programs < num_tiles
    _GRID_STRIDE_STATE["calls"] += 1
    if num_programs < num_tiles:
        _GRID_STRIDE_STATE["grid_stride_calls"] += 1

    # Launch Phase 1 (PHASE=0) then Phase 2 (PHASE=1) as two separate compiled
    # specializations so the two phases never share the Unified Buffer budget.
    for _phase in (0, 1):
        # Empty tile set (batch == 0 / no sequences): nothing to launch and
        # ``cdiv(num_tiles - pid, num_programs)`` would divide by zero.
        if num_tiles == 0:
            break
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
            nheads, seqlen, nheads_qk, headdim_qk, headdim_v, headdim_angles, num_cblocks,
            num_tiles, num_programs,
            # Compile-time constants
            chunk_size,
            BLOCK_C,
            HEADDIM_QK,
            HEADDIM_V,
            STORE_SSM_STATES_ADT_OUTV=store_states_adt_outv,
            HAS_INITIAL_STATES=Initial_States is not None,
            RETURN_FINAL_STATES=return_final_states,
            HAS_D=D is not None,
            HAS_Z=Z is not None,
            IS_VARLEN=is_varlen,
            PHASE=_phase,
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