"""
Mamba-3 Backward Pass Triton Kernels.

Copyright (c) 2026, Dao AI Lab, Goombalab
"""

from typing import Optional, Tuple
import math
import os

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

import triton
import triton.language as tl
from mamba_ssm.ops.triton.mamba3.utils import cos_approx, sin_approx, sigmoid_approx
from mamba_ssm.ops.triton.mamba3.npu_compat import (
    IS_CANN,
    alloc_device,
    autotune_regs,
    chunk_choices,
    effective_chunk_size,
    make_config,
    precise_ddt,
    precise_dqkv,
    set_allocator_safely,
    stage_choices,
)

# Ascend launches a 1D ``coreDim`` that must fit in UINT16_MAX.  Four backward
# launches in this file can exceed it:
#   * ``mamba3_siso_bwd_kernel_dzdo``: ``nchunks * nheads * batch`` tiles, where
#     ``nchunks = cdiv(seqlen, CHUNK_SIZE)`` and ``CHUNK_SIZE`` is autotuned (so
#     the tile count is only known once autotune has picked the config);
#   * ``mamba3_siso_bwd_kernel_rotary_bias_angles``: ``batch * nchunks *
#     num_cblocks`` tiles (grows with ``seqlen`` / ``batch``);
#   * ``mamba3_siso_bwd_kernel_dk_state_post``: ``nheads * num_sequences``
#     tiles (grows with GQA head count / varlen sequence count);
#   * ``mamba3_siso_bwd_kernel_ddt_dtrap_dinput_states``: ``nheads *
#     num_sequences`` tiles (same growth; ``H=32`` needs only ``B>2047``).
# Above the cap they switch to the same grid-stride launch as
# ``mamba3_siso_fwd.py``: at most ``_GRID_STRIDE_LIMIT`` programs, each looping
# over several independent tiles.  ``num_programs`` is passed explicitly rather
# than read with ``tl.num_programs(0)`` because Triton-Ascend may fold the
# logical grid onto the physical AI cores.  ``compute_dzdo`` takes its cap from
# ``MAMBA3_DZDO_GRID_CAP`` (see ``_dzdo_grid_cap``); the others use
# ``set_grid_stride_limit``.
_MAX_COREDIM = 65535
_GRID_STRIDE_LIMIT = _MAX_COREDIM


def _grid_programs(num_tiles: int) -> int:
    """Logical program count: all tiles when under the cap, else the cap."""
    return min(int(num_tiles), int(_GRID_STRIDE_LIMIT))


def _dzdo_grid_cap() -> int:
    """Effective program cap for the ``compute_dzdo`` launch.

    Read per launch from ``MAMBA3_DZDO_GRID_CAP`` so a test can lower it (e.g.
    ``monkeypatch.setenv(..., "4")``) without touching module state.  Unset,
    non-integer, or out-of-range values (outside ``[1, _MAX_COREDIM]``) all
    fall back to ``_MAX_COREDIM``.  ``set_grid_stride_limit`` deliberately does
    not apply here: lowering it for the other two launches must not silently
    compress the dzdo grid.
    """
    raw = os.environ.get("MAMBA3_DZDO_GRID_CAP")
    if raw is None:
        return _MAX_COREDIM
    try:
        cap = int(raw.strip())
    except (AttributeError, TypeError, ValueError):
        return _MAX_COREDIM
    if not 1 <= cap <= _MAX_COREDIM:
        return _MAX_COREDIM
    return cap


# Test-observable launch geometry, one entry per launch (``main`` =
# ``compute_dqktheta``'s rotary/bias kernel, ``post`` = ``apply_dk_state_post``,
# ``dzdo`` = ``compute_dzdo``, ``ddt`` = ``compute_ddt_dtrap_dinput_states``).
# Tests can lower the cap (``set_grid_stride_limit`` for main/post/ddt,
# ``MAMBA3_DZDO_GRID_CAP`` for dzdo) and assert that the grid-stride path ran.
_GRID_STRIDE_STATE = {
    "main": {
        "num_tiles": 0,
        "num_programs": 0,
        "grid_stride": False,
        "calls": 0,
        "grid_stride_calls": 0,
    },
    "post": {
        "num_tiles": 0,
        "num_programs": 0,
        "grid_stride": False,
        "calls": 0,
        "grid_stride_calls": 0,
    },
    "ddt": {
        "num_tiles": 0,
        "num_programs": 0,
        "grid_stride": False,
        "calls": 0,
        "grid_stride_calls": 0,
    },
    "dzdo": {
        "num_tiles": 0,
        "num_programs": 0,
        "grid_stride": False,
        "calls": 0,
        "grid_stride_calls": 0,
        # dzdo-only extras: the autotune-selected CHUNK_SIZE, the cap that
        # governed this launch, and the largest number of tiles any single
        # program walked (evidence the grid-stride body was actually entered).
        "chunk_size": 0,
        "cap": _MAX_COREDIM,
        "max_tiles_per_program": 0,
    },
}


def set_grid_stride_limit(limit: Optional[int]) -> None:
    """Test hook: lower the effective ``coreDim`` cap (``None`` restores 65535)."""
    global _GRID_STRIDE_LIMIT
    if limit is None:
        _GRID_STRIDE_LIMIT = _MAX_COREDIM
        return
    limit = int(limit)
    if not 1 <= limit <= _MAX_COREDIM:
        raise ValueError(f"grid-stride limit must be in [1, {_MAX_COREDIM}], got {limit}")
    _GRID_STRIDE_LIMIT = limit


def get_grid_stride_state() -> dict:
    """Snapshot of the last launches' geometry, keyed by launch name."""
    return {name: dict(state) for name, state in _GRID_STRIDE_STATE.items()}


def reset_grid_stride_state() -> None:
    """Reset the test-observable counters (does not change the limit)."""
    for name, state in _GRID_STRIDE_STATE.items():
        state.update(
            num_tiles=0, num_programs=0, grid_stride=False,
            calls=0, grid_stride_calls=0,
        )
        if name == "dzdo":
            state.update(chunk_size=0, cap=_MAX_COREDIM, max_tiles_per_program=0)


def _record_launch(name: str, num_tiles: int, num_programs: int) -> None:
    state = _GRID_STRIDE_STATE[name]
    state["num_tiles"] = int(num_tiles)
    state["num_programs"] = int(num_programs)
    state["grid_stride"] = num_programs < num_tiles
    state["calls"] += 1
    if num_programs < num_tiles:
        state["grid_stride_calls"] += 1


# =============================================================================
# dZ Kernel
# =============================================================================

@triton.autotune(
    configs=[
        make_config({"CHUNK_SIZE": cs}, s, w, r)
        for cs in chunk_choices([32, 64])
        for s in stage_choices([1, 2, 3])
        for w in [2, 4, 8]
        for r in autotune_regs([None, 128, 256])
    ],
    key=["HEADDIM_V"]
)
@triton.jit
def mamba3_siso_bwd_kernel_dzdo(
    # Input tensors
    DO, Z, O,
    # Output tensors
    Dz, DO_scaled,
    # Strides for DO: (batch, seqlen, nheads, headdim_v)
    stride_do_batch, stride_do_seqlen, stride_do_head, stride_do_vdim,
    # Strides for Z: (batch, seqlen, nheads, headdim_v)
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_vdim,
    # Strides for O: (batch, seqlen, nheads, headdim_v)
    stride_o_batch, stride_o_seqlen, stride_o_head, stride_o_vdim,
    # Strides for Dz: (batch, seqlen, nheads, headdim_v)
    stride_dz_batch, stride_dz_seqlen, stride_dz_head, stride_dz_vdim,
    # Strides for DO_scaled: (batch, seqlen, nheads, headdim_v)
    stride_do_scaled_batch, stride_do_scaled_seqlen, stride_do_scaled_head, stride_do_scaled_vdim,
    # Dimensions
    seqlen, headdim_v,
    nheads, batch,
    grid_cap,
    # Compile-time constants
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_V: tl.constexpr,
):
    """
    Backward kernel for Z-gating: computes dZ and scales dO.

    In the forward pass, output is gated as: out = O * Z * sigmoid(Z) = O * silu(Z)

    This kernel computes:
        - dZ = dO * O * sigmoid(Z) * (1 + Z * (1 - sigmoid(Z)))
        - dO_scaled = dO * sigmoid(Z) * Z  (for downstream gradient computation)

    Grid: flat ``(num_programs,)`` with ``num_programs <= grid_cap``.  Each
    logical tile is one (batch, head, chunk) triplet and each program walks one
    or more of them (grid-stride); with ``num_programs == num_tiles`` every
    program handles exactly one tile.
    """
    pid = tl.program_id(0)

    # Grid-stride over independent (batch, head, chunk) tiles.  ``CHUNK_SIZE`` is
    # chosen by autotune, so the true logical tile count is derived from the
    # selected constexpr here (the wrapper's grid lambda derives the same value
    # from the selected META) rather than read back from ``tl.num_programs(0)``:
    # Triton-Ascend folds a logical grid onto the physical AI cores, so the
    # program count seen inside the kernel is not the logical grid extent.
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    heads_chunks = nheads * num_chunks
    num_tiles = heads_chunks * batch
    num_programs = tl.minimum(num_tiles, grid_cap)

    for tile_i in range(0, tl.cdiv(num_tiles - pid, num_programs)):
        tile_idx = pid + tile_i * num_programs
        pid_batch = tile_idx // heads_chunks
        tile_rest = tile_idx % heads_chunks
        pid_head = tile_rest // num_chunks
        pid_chunk = tile_rest % num_chunks

        # Compute offsets for this (batch, head) pair
        do_offset = pid_batch * stride_do_batch + pid_head * stride_do_head
        z_offset = pid_batch * stride_z_batch + pid_head * stride_z_head
        o_offset = pid_batch * stride_o_batch + pid_head * stride_o_head
        dz_offset = pid_batch * stride_dz_batch + pid_head * stride_dz_head
        do_scaled_offset = pid_batch * stride_do_scaled_batch + pid_head * stride_do_scaled_head

        chunk_start = pid_chunk * CHUNK_SIZE
        offs_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
        offs_dim = tl.arange(0, HEADDIM_V)
        mask = (offs_seq[:, None] < seqlen) & (offs_dim[None, :] < HEADDIM_V)

        # Load dO block: (CHUNK_SIZE, headdim_v)
        do_ptrs = DO + do_offset + offs_seq[:, None] * stride_do_seqlen + offs_dim[None, :] * stride_do_vdim
        do_block = tl.load(do_ptrs, mask=mask, other=0.0)
        # Load Z block: (CHUNK_SIZE, headdim_v)
        z_ptrs = Z + z_offset + offs_seq[:, None] * stride_z_seqlen + offs_dim[None, :] * stride_z_vdim
        z_block = tl.load(z_ptrs, mask=mask, other=0.0)
        # Load O block (pre-gating output): (CHUNK_SIZE, headdim_v)
        o_ptrs = O + o_offset + offs_seq[:, None] * stride_o_seqlen + offs_dim[None, :] * stride_o_vdim
        o_block = tl.load(o_ptrs, mask=mask, other=0.0)

        # Compute sigmoid(Z) for gating
        sigmoid_z = tl.sigmoid(z_block.to(tl.float32))

        # Scale dO by sigmoid(Z)
        do_block = do_block * sigmoid_z

        # Compute dZ gradient
        # d/dZ [O * Z * sigmoid(Z)] = O * sigmoid(Z) * (1 + Z * (1 - sigmoid(Z)))
        #                           = O * sigmoid(Z) + O * Z * sigmoid(Z) * (1 - sigmoid(Z))
        dz_block = do_block * o_block * (1 + z_block * (1 - sigmoid_z))

        # Store dZ
        dz_ptrs = Dz + dz_offset + offs_seq[:, None] * stride_dz_seqlen + offs_dim[None, :] * stride_dz_vdim
        tl.store(dz_ptrs, dz_block, mask=mask)

        # Complete scaling of dO: dO * sigmoid(Z) * Z
        do_block = do_block * z_block

        # Store scaled dO for downstream gradient computation
        do_scaled_ptrs = DO_scaled + do_scaled_offset + offs_seq[:, None] * stride_do_scaled_seqlen + offs_dim[None, :] * stride_do_scaled_vdim
        tl.store(do_scaled_ptrs, do_block, mask=mask)



def compute_dzdo(
    do: torch.Tensor,
    z: torch.Tensor,
    o: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Z-gating gradients for Mamba-3 backward pass.

    When Z-gating is used in the forward pass (out = O * silu(Z)), this function
    computes the gradient with respect to Z and scales dO for downstream
    gradient computation.

    Args:
        do: Output gradient tensor of shape (batch, seqlen, nheads, headdim_v)
        z: Gating tensor from forward pass of shape (batch, seqlen, nheads, headdim_v)
        o: Pre-gating output from forward pass of shape (batch, seqlen, nheads, headdim_v)
        chunk_size: **Unused** -- kept only for signature compatibility with the
            combined backward.  This operator does not chunk the sequence: the
            tile size is the autotuned ``CHUNK_SIZE`` (see below), which the
            z-gating pointwise maths does not depend on.  The value is *not*
            clamped or forwarded to the kernel (the previous
            ``effective_chunk_size`` assignment was dead: nothing read it).

    Returns:
        Tuple containing:
            - dz: Gradient for Z tensor of shape (batch, seqlen, nheads, headdim_v)
            - do_scaled: Scaled output gradient of shape (batch, seqlen, nheads, headdim_v)
                        This should be used as input to subsequent gradient kernels.

    Grid: flat ``(num_programs,)`` = ``min(nchunks * nheads * batch, cap)`` where
    ``nchunks = cdiv(seqlen, META["CHUNK_SIZE"])`` for the autotune-selected
    ``CHUNK_SIZE`` and ``cap`` comes from ``MAMBA3_DZDO_GRID_CAP`` (default /
    invalid => 65535, the Ascend UINT16 ``coreDim`` limit).  When the true tile
    count exceeds the cap the kernel grid-strides over the remaining tiles.
    """
    batch, seqlen, nheads, headdim_v = do.shape

    # Validate inputs
    assert z is not None and o is not None and do is not None, "Z, O, and DO tensors must be provided"
    assert z.device == do.device and o.device == do.device, (
        f"Z, O, and DO must be on the same device: Z={z.device}, O={o.device}, DO={do.device}"
    )
    # CPU tensors only make sense under Triton's interpreter (the CPU golden
    # path); otherwise a CPU input would fail later inside the launch.
    assert do.device.type in ("cuda", "npu") or os.environ.get("TRITON_INTERPRET", "0") == "1", (
        f"compute_dzdo requires CUDA/NPU tensors, got device={do.device}; "
        "set TRITON_INTERPRET=1 to run the Triton kernel on CPU tensors"
    )
    assert z.shape == do.shape and o.shape == do.shape, f"Shape mismatch: Z={z.shape}, O={o.shape}, DO={do.shape}"

    # Ensure contiguity for optimal memory access
    if do.stride(-1) != 1:
        do = do.contiguous()
    if z.stride(-1) != 1:
        z = z.contiguous()
    if o.stride(-1) != 1:
        o = o.contiguous()

    # Allocate output tensors
    dz = torch.empty_like(z, dtype=do.dtype)
    do_scaled = torch.empty_like(do, dtype=do.dtype)

    # Round up head dimension to power of 2 for efficient loading
    HEADDIM_V = triton.next_power_of_2(headdim_v)

    if seqlen == 0:
        # No tile to walk: a (0,) flat grid is not launchable and the kernel's
        # ``cdiv(num_tiles - pid, num_programs)`` bound would divide by zero.
        return dz, do_scaled

    cap = _dzdo_grid_cap()
    launch = {}

    # Launch kernel: flat grid over (batch, head, chunk) tiles, capped at
    # ``cap``.  CHUNK_SIZE is autotuned, so the tile count can only be resolved
    # inside this lambda (META carries the config autotune actually selected).
    # The kernel recomputes the same count from its constexpr ``CHUNK_SIZE``;
    # ``tl.num_programs(0)`` is never used because Triton-Ascend folds the
    # logical grid onto the physical AI cores.
    def grid(META):
        chunk = int(META["CHUNK_SIZE"])
        num_tiles = triton.cdiv(seqlen, chunk) * nheads * batch
        num_programs = min(num_tiles, cap)
        launch["CHUNK_SIZE"] = chunk
        launch["num_tiles"] = num_tiles
        launch["num_programs"] = num_programs
        return (num_programs,)

    mamba3_siso_bwd_kernel_dzdo[grid](
        do, z, o,
        dz, do_scaled,
        # DO strides
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        # Z strides
        z.stride(0), z.stride(1), z.stride(2), z.stride(3),
        # O strides
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Dz strides
        dz.stride(0), dz.stride(1), dz.stride(2), dz.stride(3),
        # DO_scaled strides
        do_scaled.stride(0), do_scaled.stride(1), do_scaled.stride(2), do_scaled.stride(3),
        # Dimensions
        seqlen, headdim_v,
        nheads, batch,
        cap,
        # Compile-time constants
        HEADDIM_V=HEADDIM_V,
    )

    # Record the geometry of the launch autotune actually performed (the grid
    # lambda runs last for the selected config).  Tests assert ``num_programs <=
    # cap`` and ``grid_stride`` / ``max_tiles_per_program`` to prove the
    # compressed path ran, not merely that the numbers matched.
    _record_launch("dzdo", launch["num_tiles"], launch["num_programs"])
    dzdo_state = _GRID_STRIDE_STATE["dzdo"]
    dzdo_state["chunk_size"] = launch["CHUNK_SIZE"]
    dzdo_state["cap"] = cap
    dzdo_state["max_tiles_per_program"] = triton.cdiv(launch["num_tiles"], launch["num_programs"])

    return dz, do_scaled


# =============================================================================
# dQKV Kernel
# =============================================================================

@triton.autotune(
    configs=[
        make_config({}, s, w, r)
        for s in stage_choices([1, 2, 3])
        for w in [2, 4, 8]
        for r in autotune_regs([None, 128, 256])
    ],
    key=["CHUNK_SIZE", "HEADDIM_QK", "HEADDIM_V", "IS_VARLEN", "PRECISE_FP32"]
)
@triton.jit
def mamba3_siso_bwd_kernel_dqkv(
    # Input tensors
    Q, K, V, DA_CS, DA_CS_SUM, QK_Dot, D, SSM_States, dO, d_OSSM_State, Cu_Seqlens, # dO is scaled with Z
    # Output tensors
    dQ, dK, dV, dADT, dQK_Dot, dD, d_ISSM_State, # dQK_Dot is scaled with scale
    # Strides for Inputs
    # Strides for Q: (batch, seqlen, nheads_qk, HEADDIM_QK)
    stride_q_batch, stride_q_seqlen, stride_q_head, stride_q_qkdim,
    # Strides for K: (batch, seqlen, nheads_qk, HEADDIM_QK)
    stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_qkdim,
    # Strides for V: (batch, seqlen, nheads, HEADDIM_V)
    stride_v_batch, stride_v_seqlen, stride_v_head, stride_v_vdim,
    # Strides for DA_CS: (batch, nheads, seqlen)
    stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seqlen,
    # Strides for DA_CS_SUM: (batch, nheads, nchunks)
    stride_da_cs_sum_batch, stride_da_cs_sum_head, stride_da_cs_sum_seqlen,
    # Strides for QK (QK dot products): (batch, nheads, nchunks*CHUNK_SIZE)
    stride_qk_dot_batch, stride_qk_dot_head, stride_qk_dot_seqlen,
    # Strides for D: (nheads,)
    stride_d_head,
    # Strides for SSM_States: (batch, nheads, HEADDIM_V, nchunks*HEADDIM_QK)
    stride_ssm_states_batch, stride_ssm_states_head, stride_ssm_states_vdim, stride_ssm_states_qkdim,
    # Strides for dO: (batch, seqlen, nheads, HEADDIM_V)
    stride_do_batch, stride_do_seqlen, stride_do_head, stride_do_vdim,
    # Strides for d_OSSM_State: (num_sequences, nheads, HEADDIM_V, HEADDIM_QK)
    stride_d_ossm_state_batch, stride_d_ossm_state_head, stride_d_ossm_state_vdim, stride_d_ossm_state_qkdim,
    # Strides for Cu_Seqlens: (num_sequences + 1,)
    stride_cu_seqlen,
    # Strides for Outputs
    # Strides for dQ: (batch, seqlen, nheads, HEADDIM_QK)
    stride_dq_batch, stride_dq_seqlen, stride_dq_head, stride_dq_qkdim,
    # Strides for dK: (batch, seqlen, nheads, HEADDIM_QK)
    stride_dk_batch, stride_dk_seqlen, stride_dk_head, stride_dk_qkdim,
    # Strides for dV: (batch, seqlen, nheads, HEADDIM_V)
    stride_dv_batch, stride_dv_seqlen, stride_dv_head, stride_dv_vdim,
    # Strides for dAdt: (batch, nheads, seqlen)
    stride_dadt_batch, stride_dadt_head, stride_dadt_seqlen,
    # Strides for dQK_dot: (batch, nheads, seqlen)
    stride_dQK_dot_batch, stride_dQK_dot_head, stride_dQK_dot_seqlen,
    # Strides for dD: (nheads,)
    stride_dd_batch, stride_dd_head,
    # Strides for d_ISSM_State: (num_sequences, nheads, HEADDIM_V, HEADDIM_QK)
    stride_d_issm_state_batch, stride_d_issm_state_head, stride_d_issm_state_vdim, stride_d_issm_state_qkdim,
    # Dimensions
    nheads, seqlen, nheads_qk, headdim_qk, headdim_v,
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    RECOMPUTE_MASK: tl.constexpr,
    HAS_D_OSSM_STATE: tl.constexpr,
    RETURN_D_ISSM_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    PRECISE_FP32: tl.constexpr,
):
    """
    Backward kernel for Mamba-3 attention mechanism.

    Each program instance handles one (head, batch/seq) pair and iterates through
    all chunks in reverse order. This reverse iteration is necessary because
    state gradients flow backward through the sequence.

    The kernel computes:
        - dQ, dK: Gradients for query/key from both intra-chunk attention and inter-chunk states
        - dV: Gradient for values
        - dADT: Gradient for the decay parameter (A * dt)
        - dQK_Dot: Gradient for the QK dot product term
        - dD: Gradient for the skip connection (if present)
        - dISSM_State: Gradient for the input SSM state (if present)

    Grid:
        - Normal mode: (nheads, batch)
        - Varlen mode: (nheads, num_sequences)
    """
    # ==================== Program Indexing ====================
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    if IS_VARLEN:
        pid_seq = pid_batch
        pid_batch = 0
        cu_seqlen = tl.load(Cu_Seqlens + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_next = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seqlen = cu_seqlen_next - cu_seqlen
        cu_chunks = pid_seq + cu_seqlen // CHUNK_SIZE
    else:
        cu_seqlen = 0
        cu_chunks = 0
        pid_seq = 0

    # Compute Q/K head index for GQA (grouped query attention).
    # ``nheads`` is passed explicitly instead of using ``tl.num_programs(0)``:
    # Triton-Ascend folds a logical grid onto the physical AI cores, so the
    # program count seen inside the kernel is not the logical grid extent.
    head_idx_qk = pid_head // (nheads // nheads_qk)

    # Input Pointer Offsets
    q_offset = pid_batch * stride_q_batch + head_idx_qk * stride_q_head + IS_VARLEN * cu_seqlen * stride_q_seqlen
    k_offset = pid_batch * stride_k_batch + head_idx_qk * stride_k_head + IS_VARLEN * cu_seqlen * stride_k_seqlen
    v_offset = pid_batch * stride_v_batch + pid_head * stride_v_head + IS_VARLEN * cu_seqlen * stride_v_seqlen
    da_cs_offset = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head + IS_VARLEN * cu_seqlen * stride_da_cs_seqlen
    da_cs_sum_offset = pid_batch * stride_da_cs_sum_batch + pid_head * stride_da_cs_sum_head + IS_VARLEN * cu_chunks * stride_da_cs_sum_seqlen
    qk_dot_offset = pid_batch * stride_qk_dot_batch + pid_head * stride_qk_dot_head + IS_VARLEN * cu_seqlen * stride_qk_dot_seqlen
    ssm_states_offset = pid_batch * stride_ssm_states_batch + pid_head * stride_ssm_states_head + IS_VARLEN * cu_chunks * HEADDIM_QK * stride_ssm_states_qkdim
    do_offset = pid_batch * stride_do_batch + pid_head * stride_do_head + IS_VARLEN * cu_seqlen * stride_do_seqlen
    if HAS_D_OSSM_STATE:
        d_ossm_state_offset = (pid_batch + IS_VARLEN * pid_seq) * stride_d_ossm_state_batch + pid_head * stride_d_ossm_state_head

    # Load skip connection value D if present
    if D is not None:
        D_offset = pid_head * stride_d_head
        D_val = tl.load(D + D_offset)

    # Output Pointer Offsets
    dq_offset = pid_batch * stride_dq_batch + pid_head * stride_dq_head + IS_VARLEN * cu_seqlen * stride_dq_seqlen
    dk_offset = pid_batch * stride_dk_batch + pid_head * stride_dk_head + IS_VARLEN * cu_seqlen * stride_dk_seqlen
    dv_offset = pid_batch * stride_dv_batch + pid_head * stride_dv_head + IS_VARLEN * cu_seqlen * stride_dv_seqlen
    dadt_offset = pid_batch * stride_dadt_batch + pid_head * stride_dadt_head + IS_VARLEN * cu_seqlen * stride_dadt_seqlen
    dQK_dot_offset = pid_batch * stride_dQK_dot_batch + pid_head * stride_dQK_dot_head + IS_VARLEN * cu_seqlen * stride_dQK_dot_seqlen

    if D is not None:
        dD_offset = pid_head * stride_dd_head + pid_batch * stride_dd_batch + IS_VARLEN * pid_seq * stride_dd_batch
        dD_acc = tl.zeros([1], dtype=tl.float32)

    if RETURN_D_ISSM_STATE:
        d_issm_state_offset = (pid_batch + IS_VARLEN * pid_seq) * stride_d_issm_state_batch + pid_head * stride_d_issm_state_head

    # Accumulates gradients flowing backward through states across chunks
    if HAS_D_OSSM_STATE:
        d_ssm_ptrs =  d_OSSM_State + d_ossm_state_offset + tl.arange(0, HEADDIM_V)[:, None] * stride_d_ossm_state_vdim + tl.arange(0, HEADDIM_QK)[None, :] * stride_d_ossm_state_qkdim
        d_ssm_states_mask = (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (tl.arange(0, HEADDIM_QK)[None, :] < headdim_qk)
        d_ssm_states_acc = tl.load(d_ssm_ptrs, mask=d_ssm_states_mask, other=0.0).to(tl.float32)
    else:
        d_ssm_states_acc = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    # NOTE (Ascend migration): upstream used TMA tensor descriptors here.  Since
    # Triton-Ascend has no TMA support, all descriptor loads/stores below are
    # rewritten as explicit strided pointer accesses with masks that reproduce
    # TMA's out-of-bounds zero padding.

    for chunk_idx_loop in range(num_chunks):
        chunk_idx = num_chunks - 1 - chunk_idx_loop  # Reverse order for backward pass
        chunk_start = chunk_idx * CHUNK_SIZE

        # Sequence-length mask for non-TMA loads/stores
        offs_cs = chunk_start + tl.arange(0, CHUNK_SIZE)
        seq_mask = offs_cs < seqlen
        # Ascend tail-block guard (PLUT-32): the Cube ``tl.dot`` that builds the
        # intra-chunk gate matrix ``dAinv`` may return non-zero values in the
        # masked (s >= seqlen) tail columns, which ``tl.sum(..., axis=1)`` then
        # folds into the *valid* rows.  ``seq_valid_2d`` is the 2-D zero-padding
        # mask used to re-assert that padding on ``dAinv`` below.
        seq_valid_2d = seq_mask[:, None] & seq_mask[None, :]

        # ============================================================
        # Load Decay Values
        # We load these first to overlap computation with TMA loads
        # ============================================================
        da_cs_ptrs = DA_CS + da_cs_offset + offs_cs * stride_da_cs_seqlen
        da_cs = tl.load(da_cs_ptrs, mask=seq_mask, other=0.0)  # Cumulative decay within chunk: (CHUNK_SIZE,)

        da_cs_sum_ptrs = DA_CS_SUM + da_cs_sum_offset + chunk_idx * stride_da_cs_sum_seqlen
        da_cs_chunk_sum = tl.load(da_cs_sum_ptrs)  # Total decay for this chunk: scalar

        # ============================================================
        # Load Q, K, V, dO, SSM_States (pointer loads replace TMA)
        # ============================================================
        offs_hd = tl.arange(0, HEADDIM_QK)
        offs_hv = tl.arange(0, HEADDIM_V)
        hd_mask = offs_hd[None, :] < headdim_qk
        hv_mask = offs_hv[None, :] < headdim_v
        do_block = tl.load(
            dO + do_offset + offs_cs[:, None] * stride_do_seqlen + offs_hv[None, :] * stride_do_vdim,
            mask=seq_mask[:, None] & hv_mask, other=0.0)  # (CHUNK_SIZE, HEADDIM_V)
        v_block = tl.load(
            V + v_offset + offs_cs[:, None] * stride_v_seqlen + offs_hv[None, :] * stride_v_vdim,
            mask=seq_mask[:, None] & hv_mask, other=0.0)  # (CHUNK_SIZE, HEADDIM_V)
        q_block = tl.load(
            Q + q_offset + offs_cs[:, None] * stride_q_seqlen + offs_hd[None, :] * stride_q_qkdim,
            mask=seq_mask[:, None] & hd_mask, other=0.0)  # (CHUNK_SIZE, HEADDIM_QK)
        k_block = tl.load(
            K + k_offset + offs_cs[:, None] * stride_k_seqlen + offs_hd[None, :] * stride_k_qkdim,
            mask=seq_mask[:, None] & hd_mask, other=0.0)  # (CHUNK_SIZE, HEADDIM_QK)
        ssm_v = offs_hv[:, None]
        ssm_k = chunk_idx * headdim_qk + offs_hd[None, :]
        ssm_states_block = tl.load(
            SSM_States + ssm_states_offset + ssm_v * stride_ssm_states_vdim + ssm_k * stride_ssm_states_qkdim,
            mask=(ssm_v < headdim_v) & (ssm_k < num_chunks * headdim_qk), other=0.0)  # (HEADDIM_V, HEADDIM_QK)

        # ============================================================
        # Compute Decay Scaling Factors
        # ============================================================
        # Reverse cumsum: how much decay from position i to end of chunk
        da_cs_rev = da_cs_chunk_sum - da_cs
        exp_da_cs_rev = tl.math.exp2(da_cs_rev)  # For scaling inter-chunk contributions
        exp_da_cs = tl.math.exp2(da_cs)          # For scaling intra-chunk contributions

        # Compute strictly causal mask with exponential decay (this is L^T)
        if not RECOMPUTE_MASK:
            causal_decay_mask = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0)),
                0.0
            )

        # ============================================================
        # Compute dADT Gradient (Part 1): From Intra-chunk Attention
        # This is register-heavy so we compute it early before spilling
        # ============================================================
        # Gradient contribution from (QK^T ⊙ L) V term
        dAinv = tl.dot(v_block, tl.trans(do_block))  # V @ dO^T
        if RECOMPUTE_MASK:
            dAinv *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            dAinv = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                dAinv,
                0.0
            )
        else:
            dAinv *= causal_decay_mask
        # Ascend tail-block guard (PLUT-32): the causal mask above keeps every
        # column with ``j > i`` regardless of validity, so the garbage the Cube
        # ``tl.dot`` leaves in the masked tail columns (``s >= seqlen``) would
        # survive and be folded into the *valid* rows by ``tl.sum(dAinv, axis=1)``
        # below, scaling ``dADT`` to O(1e2~1e3).  Re-assert the zero padding once
        # here (before the elementwise K@Q^T multiply, which carries the same
        # garbage in its tail).  On a backend whose dot already returns exact
        # zeros on masked columns this is a no-op.
        dAinv = tl.where(seq_valid_2d, dAinv, 0.0)
        dAinv *= tl.dot(k_block, tl.trans(q_block))  # Element-wise with K @ Q^T
        dM_rev_vector = tl.sum(dAinv, axis=0) - tl.sum(dAinv, axis=1)  # (CHUNK_SIZE,)

        # ============================================================
        # Compute dK: Key Gradient
        # dK = (V @ dO^T ⊙ mask)^T @ Q + V @ dStates * scale
        # ============================================================
        # Intra-chunk: dP^T @ Q where dP = dO @ V^T ⊙ mask
        dp_t_block = tl.dot(v_block, tl.trans(do_block))  # V @ dO^T: (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            dp_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            dp_t_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                dp_t_block,
                0.0
            )
        else:
            dp_t_block *= causal_decay_mask

        # PLUT-50: the second ``tl.dot`` operands (and the state-gradient
        # accumulator, below) are where the fp32 intermediates get re-quantised
        # to the input dtype; keep them fp32 when ``PRECISE_FP32`` is set (see
        # ``npu_compat.precise_dqkv``).
        if PRECISE_FP32:
            acc_dk = tl.dot(dp_t_block.to(tl.float32), q_block.to(tl.float32))  # (CHUNK_SIZE, HEADDIM_QK)
            # Inter-chunk: gradient flowing through accumulated states
            acc_dk += tl.dot(v_block.to(tl.float32), d_ssm_states_acc) * exp_da_cs_rev[:, None]
        else:
            acc_dk = tl.dot(dp_t_block.to(q_block.dtype), q_block)  # (CHUNK_SIZE, HEADDIM_QK)
            # Inter-chunk: gradient flowing through accumulated states
            acc_dk += tl.dot(v_block, d_ssm_states_acc.to(v_block.dtype)) * exp_da_cs_rev[:, None]

        tl.store(
            dK + dk_offset + offs_cs[:, None] * stride_dk_seqlen + offs_hd[None, :] * stride_dk_qkdim,
            acc_dk,
            mask=seq_mask[:, None] & hd_mask,
        )

        # ============================================================
        # Compute dQ: Query Gradient
        # dQ = (V @ dO^T ⊙ mask) @ K + dO @ States * scale
        # ============================================================
        # Intra-chunk: S^T @ K where S = V @ dO^T ⊙ mask
        s_block = tl.dot(v_block, tl.trans(do_block))  # (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            s_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            s_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                s_block,
                0.0
            )
        else:
            s_block *= causal_decay_mask

        if PRECISE_FP32:
            acc_dq = tl.dot(tl.trans(s_block).to(tl.float32), k_block.to(tl.float32))  # (CHUNK_SIZE, HEADDIM_QK)
            # Inter-chunk: gradient through states from previous chunks
            acc_dq += tl.dot(do_block.to(tl.float32), ssm_states_block.to(tl.float32)) * exp_da_cs[:, None]
        else:
            acc_dq = tl.dot(tl.trans(s_block).to(k_block.dtype), k_block)  # (CHUNK_SIZE, HEADDIM_QK)
            # Inter-chunk: gradient through states from previous chunks
            acc_dq += tl.dot(do_block, ssm_states_block) * exp_da_cs[:, None]

        tl.store(
            dQ + dq_offset + offs_cs[:, None] * stride_dq_seqlen + offs_hd[None, :] * stride_dq_qkdim,
            acc_dq,
            mask=seq_mask[:, None] & hd_mask,
        )

        # ============================================================
        # Compute dV: Value Gradient
        # dV = (K @ Q^T ⊙ mask) @ dO + K @ dStates^T * scale + dO * (D + qk_dot)
        # ============================================================
        # Intra-chunk: P^T @ dO where P = Q @ K^T ⊙ mask
        p_t_block = tl.dot(k_block, tl.trans(q_block))  # K @ Q^T: (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            p_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            p_t_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                p_t_block,
                0.0
            )
        else:
            p_t_block *= causal_decay_mask

        if PRECISE_FP32:
            acc_dv = tl.dot(p_t_block.to(tl.float32), do_block.to(tl.float32))  # (CHUNK_SIZE, HEADDIM_V)
            # Inter-chunk: gradient through states
            acc_dv += tl.dot(k_block.to(tl.float32), tl.trans(d_ssm_states_acc)) * exp_da_cs_rev[:, None]
        else:
            acc_dv = tl.dot(p_t_block.to(do_block.dtype), do_block)  # (CHUNK_SIZE, HEADDIM_V)
            # Inter-chunk: gradient through states
            acc_dv += tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype)) * exp_da_cs_rev[:, None]

        # Skip connection gradient contribution.
        # UB note (Ascend): reuse ``do_block`` / ``v_block`` / ``ssm_states_block``
        # that were already loaded above and are not mutated before their last
        # use, instead of reloading fresh copies.  This removes three redundant
        # CHUNK_SIZE*HEADDIM_V and HEADDIM_V*HEADDIM_QK temporaries (plus their
        # global loads), lowering the kernel's peak UB footprint.
        qk_dot = tl.load(QK_Dot + qk_dot_offset + offs_cs * stride_qk_dot_seqlen, mask=seq_mask, other=0.0)
        if D is not None:
            acc_dv += do_block * (D_val + qk_dot[:, None])
        else:
            acc_dv += do_block * qk_dot[:, None]

        tl.store(
            dV + dv_offset + offs_cs[:, None] * stride_dv_seqlen + offs_hv[None, :] * stride_dv_vdim,
            acc_dv,
            mask=seq_mask[:, None] & hv_mask,
        )

        # ============================================================
        # Compute dQK_Dot and dD: Skip Connection Gradients
        # ============================================================
        # dQK_dot = sum_v(dO * V) for each position.  An explicit fp32-product
        # accumulation replaces the upstream N=1 matmul (Ascend Cube needs N>=16;
        # computing the product in bf16 would lose precision versus tl.dot).
        dQK_dot_block = tl.sum(do_block.to(tl.float32) * v_block.to(tl.float32), axis=1)

        tl.store(
            dQK_Dot + dQK_dot_offset + offs_cs * stride_dQK_dot_seqlen,
            dQK_dot_block,
            mask=seq_mask
        )

        # Accumulate dD gradient
        if D is not None:
            dD_acc += tl.sum(dQK_dot_block)

        # ============================================================
        # Compute dADT Gradient (Part 2): From Inter-chunk States
        # ============================================================
        # Gradient from Q @ States^T term
        if PRECISE_FP32:
            QS = tl.dot(q_block.to(tl.float32), tl.trans(ssm_states_block).to(tl.float32))  # (CHUNK_SIZE, HEADDIM_V)
        else:
            QS = tl.dot(q_block, tl.trans(ssm_states_block))  # (CHUNK_SIZE, HEADDIM_V)
        dM_rev_vector += tl.sum(QS * do_block, axis=1) * exp_da_cs  # (CHUNK_SIZE,)

        # ============================================================
        # Compute dADT Gradient (Part 3): From State Accumulation
        # ============================================================
        # Gradient flowing through d_ssm_states_acc @ SSM_States.
        # ``ssm_states_block`` already holds exactly this slice of SSM_States.
        dM_scalar = tl.sum(ssm_states_block * d_ssm_states_acc) * tl.math.exp2(da_cs_chunk_sum)

        # ============================================================
        # Compute dADT Gradient (Part 4): From K @ dStates
        # ============================================================
        if PRECISE_FP32:
            dSK = tl.dot(k_block.to(tl.float32), tl.trans(d_ssm_states_acc))  # (CHUNK_SIZE, HEADDIM_V)
        else:
            dSK = tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype))  # (CHUNK_SIZE, HEADDIM_V)
        dM_vector = tl.sum(dSK * v_block, axis=1) * exp_da_cs_rev  # (CHUNK_SIZE,)

        # ============================================================
        # Combine dADT Gradient Components via Reverse Cumsum
        # ============================================================
        dM_rev_vector += (tl.sum(dM_rev_vector) + dM_scalar) + tl.cumsum(dM_vector - dM_rev_vector) - dM_vector

        # Store dADT
        dadt_ptrs = dADT + dadt_offset + offs_cs * stride_dadt_seqlen
        tl.store(dadt_ptrs, dM_rev_vector, mask=seq_mask)

        # ============================================================
        # Accumulate State Gradients for Previous Chunks
        # ============================================================
        # Scale dO by the per-position decay inline; ``do_block`` is dead after
        # this point, so no separate scaled copy is kept alive in UB.
        if PRECISE_FP32:
            d_ssm_states_acc = (tl.math.exp2(da_cs_chunk_sum) * d_ssm_states_acc +
                           tl.dot(tl.trans(do_block.to(tl.float32) * exp_da_cs[:, None]), q_block.to(tl.float32)))
        else:
            d_ssm_states_acc = (tl.math.exp2(da_cs_chunk_sum) * d_ssm_states_acc +
                           tl.dot(tl.trans((do_block * exp_da_cs[:, None]).to(q_block.dtype)), q_block))

    # Store Final dD Gradient
    if D is not None:
        tl.store(dD + dD_offset + tl.arange(0, 1), dD_acc)

    # Store d_ISSM_State
    if RETURN_D_ISSM_STATE:
        d_ISSM_State_ptrs = d_ISSM_State + d_issm_state_offset + tl.arange(0, HEADDIM_V)[:, None] * stride_d_issm_state_vdim + tl.arange(0, HEADDIM_QK)[None, :] * stride_d_issm_state_qkdim
        d_ISSM_State_mask = (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (tl.arange(0, HEADDIM_QK)[None, :] < headdim_qk)
        tl.store(d_ISSM_State_ptrs, d_ssm_states_acc, mask=d_ISSM_State_mask)


def compute_dqkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    da_cs: torch.Tensor,
    da_cs_sum: torch.Tensor,
    qk_dot: torch.Tensor,
    SSM_States: torch.Tensor,
    do: torch.Tensor,
    d_ossm_state: Optional[torch.Tensor] = None,
    d_ov_state: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    has_input_state: bool = False,
    Cu_Seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute gradients dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, d_issm_state for Mamba-3 backward pass.

    This kernel operates on the rotated/scaled Q and K tensors (Q_mid, K_mid from forward).

    Args:
        q: Rotated query tensor Q_mid (batch, seqlen, headdim_qk, headdim_qk)
        k: Rotated+scaled key tensor K_mid (batch, seqlen, headdim_qk, headdim_qk)
        v: Value tensor (batch, seqlen, nheads, headdim_v)
        da_cs: Cumulative decay per chunk (batch, nheads, seqlen)
        da_cs_sum: Sum of decay per chunk (batch, nheads, nchunks)
        qk_dot: QK dot products from forward (batch, nheads, seqlen)
        SSM_States: SSM states from forward pass (batch, nheads, headdim_v, nchunks * headdim_qk)
        do: Output gradient, possibly scaled by Z (batch, seqlen, nheads, headdim_v)
        d_ossm_state: Gradient of output SSM states (num_sequences, nheads, headdim_v, headdim_qk)
        d_ov_state: Gradient of output V state (num_sequences, nheads, headdim_v) - added to last token of dV
        D: Optional skip connection weight (nheads,)
        chunk_size: Chunk size (default: 64)
        has_input_state: Whether to compute gradient for input states

    Returns:
        Tuple of (dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, d_issm_state)
        where d_issm_state is None if has_input_state=False
    """
    batch, seqlen, nheads_qk, headdim_qk = q.shape
    _, _, nheads, headdim_v = v.shape
    is_varlen = Cu_Seqlens is not None

    if is_varlen:
        num_sequences = Cu_Seqlens.shape[0] - 1
        assert batch == 1
        nchunks = num_sequences + seqlen // chunk_size
    else:
        num_sequences = batch
        nchunks = (seqlen + chunk_size - 1) // chunk_size

    assert nheads % nheads_qk == 0, "nheads must be divisible by nheads_qk (for GQA support)"
    assert q.device == k.device == v.device == da_cs.device and da_cs_sum.device == do.device, "All tensors must be on the same device"

    assert k.shape == q.shape
    assert v.shape == (batch, seqlen, nheads, headdim_v)
    assert da_cs.shape == (batch, nheads, seqlen)
    assert da_cs_sum.shape == (batch, nheads, nchunks)
    assert qk_dot.shape == (batch, nheads, seqlen)
    assert SSM_States.shape == (batch, nheads, headdim_v, nchunks * headdim_qk)
    assert do.shape == (batch, seqlen, nheads, headdim_v)
    assert d_ossm_state is None or d_ossm_state.shape == (num_sequences, nheads, headdim_v, headdim_qk)
    assert d_ov_state is None or d_ov_state.shape == (num_sequences, nheads, headdim_v)
    if D is not None:
        assert D.shape == (nheads,)

    # Ensure all tensors satisfy the (post-TMA) strided-access constraints.
    #
    # The kernels now access global memory through explicit strided pointers,
    # so the only requirement is a unit innermost stride; non-contiguous
    # tensors are made contiguous for predictable access patterns.
    chunk_size = effective_chunk_size(chunk_size, seqlen)
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if da_cs.stride(-1) != 1:
        da_cs = da_cs.contiguous()
    if da_cs_sum.stride(-1) != 1:
        da_cs_sum = da_cs_sum.contiguous()
    if qk_dot.stride(-1) != 1:
        qk_dot = qk_dot.contiguous()
    if not SSM_States.is_contiguous():
        SSM_States = SSM_States.contiguous()
    if not do.is_contiguous():
        do = do.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if d_ossm_state is not None and not d_ossm_state.is_contiguous():
        d_ossm_state = d_ossm_state.contiguous()
    if d_ov_state is not None and not d_ov_state.is_contiguous():
        d_ov_state = d_ov_state.contiguous()

    # Allocate output tensors
    dq = torch.empty((batch, seqlen, nheads, headdim_qk), dtype=torch.float32, device=q.device)
    dk = torch.empty((batch, seqlen, nheads, headdim_qk), dtype=torch.float32, device=k.device)
    dv = torch.empty((batch, seqlen, nheads, headdim_v), dtype=v.dtype, device=v.device)
    dAdt = torch.empty_like(da_cs)
    dQK = torch.empty_like(da_cs)
    dD = torch.empty((num_sequences, nheads), dtype=torch.float32, device=q.device) if D is not None else None
    d_issm_state = torch.empty((num_sequences, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=q.device) if has_input_state else None

    # Round up head dimensions to power of 2 for efficient loading
    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    HEADDIM_V = triton.next_power_of_2(headdim_v)

    # Grid: each program handles one (head, batch/num_sequences) pair
    if is_varlen:
        grid = (nheads, num_sequences)
    else:
        grid = (nheads, batch)

    # Launch kernel
    mamba3_siso_bwd_kernel_dqkv[grid](
        q, k, v, da_cs, da_cs_sum, qk_dot, D, SSM_States, do, d_ossm_state, Cu_Seqlens,
        dq, dk, dv, dAdt, dQK, dD, d_issm_state,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # DA_CS strides
        da_cs.stride(0), da_cs.stride(1), da_cs.stride(2),
        # DA_CS_SUM strides
        da_cs_sum.stride(0), da_cs_sum.stride(1), da_cs_sum.stride(2),
        # QK_Dot strides
        qk_dot.stride(0), qk_dot.stride(1), qk_dot.stride(2),
        # D stride
        D.stride(0) if D is not None else 0,
        # SSM_States strides: (batch, nheads, headdim_v, nchunks*headdim_qk)
        SSM_States.stride(0), SSM_States.stride(1), SSM_States.stride(2),
        SSM_States.stride(3),
        # dO strides
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        # d_ossm_state strides
        d_ossm_state.stride(0) if d_ossm_state is not None else 0,
        d_ossm_state.stride(1) if d_ossm_state is not None else 0,
        d_ossm_state.stride(2) if d_ossm_state is not None else 0,
        d_ossm_state.stride(3) if d_ossm_state is not None else 0,
        # Cu_Seqlens strides
        Cu_Seqlens.stride(0) if Cu_Seqlens is not None else 0,
        # dQ strides
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        # dK strides
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        # dV strides
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        # dAdt strides
        dAdt.stride(0), dAdt.stride(1), dAdt.stride(2),
        # dQK strides
        dQK.stride(0), dQK.stride(1), dQK.stride(2),
        # dD strides
        dD.stride(0) if D is not None else 0,
        dD.stride(1) if D is not None else 0,
        # d_issm_state strides
        d_issm_state.stride(0) if d_issm_state is not None else 0,
        d_issm_state.stride(1) if d_issm_state is not None else 0,
        d_issm_state.stride(2) if d_issm_state is not None else 0,
        d_issm_state.stride(3) if d_issm_state is not None else 0,
        # Dimensions
        nheads, seqlen, nheads_qk, headdim_qk, headdim_v,
        # Compile-time constants
        CHUNK_SIZE=chunk_size,
        HEADDIM_QK=HEADDIM_QK,
        HEADDIM_V=HEADDIM_V,
        # Ascend UB policy: recompute the causal-decay mask inline instead of
        # materialising and keeping a CHUNK_SIZE x CHUNK_SIZE fp32 mask alive for
        # the whole loop body (removes C^2*4 bytes of persistent UB).
        RECOMPUTE_MASK=IS_CANN,
        HAS_D_OSSM_STATE=d_ossm_state is not None,
        RETURN_D_ISSM_STATE=has_input_state,
        IS_VARLEN=is_varlen,
        # PLUT-50 precision path: fp32 second-dot operands + fp32 state-gradient
        # accumulator (see ``npu_compat.precise_dqkv``).
        PRECISE_FP32=precise_dqkv(),
    )

    # Add output V state gradients to the last token
    if d_ov_state is not None:
        if is_varlen:
            last_token_idx = Cu_Seqlens[1:] - 1
            dv[0, last_token_idx] += d_ov_state
        else:
            dv[:, -1, :, :] += d_ov_state

    dD = dD.sum(dim=0) if dD is not None else None
    return dq, dk, dv, dAdt, dQK, dD, d_issm_state


# =============================================================================
#  d Rotary+Bias Kernel
# =============================================================================


@triton.autotune(
    configs=[
        make_config({}, s, w, r)
        for s in stage_choices([1, 2, 3])
        for w in [2, 4, 8]
        for r in autotune_regs([None, 128, 256])
    ],
    key=["CHUNK_SIZE", "BLOCK_HEADDIM_QK", "HEADDIM_QK", "BLOCK_C"]
)
@triton.jit
def mamba3_siso_bwd_kernel_rotary_bias_angles(
    # Input tensors
    Q, K, Scale, Gamma, Q_bias, K_bias, Angles, dQ_in, dK_in, dQK,
    # Output tensors
    dQ, dK, dAngles, dScale, dGamma, dQ_bias, dK_bias,
    # Strides for inputs -------------------------------------------------------
    # Q: (batch, seqlen, nheads_qk, BLOCK_HEADDIM_QK)
    stride_q_batch, stride_q_seqlen, stride_q_head, stride_q_qkdim,
    # K: (batch, seqlen, nheads_qk, BLOCK_HEADDIM_QK)
    stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_qkdim,
    # Scale: (batch, nheads, seqlen)
    stride_scale_batch, stride_scale_head, stride_scale_seqlen,
    # Gamma: (batch, nheads, seqlen)
    stride_gamma_batch, stride_gamma_head, stride_gamma_seqlen,
    # Q_bias: (nheads, BLOCK_HEADDIM_QK)
    stride_q_bias_head, stride_q_bias_qkdim,
    # K_bias: (nheads, BLOCK_HEADDIM_QK)
    stride_k_bias_head, stride_k_bias_qkdim,
    # Angles: (batch, seqlen, nheads, BLOCK_HEADDIM_QK/2)
    stride_angles_batch, stride_angles_seqlen, stride_angles_head, stride_angles_qkdim,
    # dQ_in: (batch, seqlen, nheads, BLOCK_HEADDIM_QK)
    stride_dq_in_batch, stride_dq_in_seqlen, stride_dq_in_head, stride_dq_in_qkdim,
    # dK_in: (batch, seqlen, nheads, BLOCK_HEADDIM_QK)
    stride_dk_in_batch, stride_dk_in_seqlen, stride_dk_in_head, stride_dk_in_qkdim,
    # dQK: (batch, nheads, seqlen)
    stride_dqk_batch, stride_dqk_head, stride_dqk_seqlen,
    # Strides for outputs ------------------------------------------------------
    # dQ: (batch, seqlen, nheads_qk, BLOCK_HEADDIM_QK)
    stride_dq_batch, stride_dq_seqlen, stride_dq_head, stride_dq_qkdim,
    # dK: (batch, seqlen, nheads_qk, BLOCK_HEADDIM_QK)
    stride_dk_batch, stride_dk_seqlen, stride_dk_head, stride_dk_qkdim,
    # dAngles: (batch, seqlen, nheads, BLOCK_HEADDIM_QK/2)
    stride_dangles_batch, stride_dangles_seqlen, stride_dangles_head, stride_dangles_qkdim,
    # dScale: (batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, seqlen)
    stride_dscale_batch, stride_dscale_head, stride_dscale_nqkchunks ,stride_dscale_seqlen,
    # dGamma: (batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, seqlen)
    stride_dgamma_batch, stride_dgamma_head, stride_dgamma_nqkchunks, stride_dgamma_seqlen,
    # dQ_bias: (batch, nchunks, nheads, BLOCK_HEADDIM_QK)
    stride_dq_bias_batch, stride_dq_bias_nchunks, stride_dq_bias_head, stride_dq_bias_qkdim,
    # dK_bias: (batch, nchunks, nheads, BLOCK_HEADDIM_QK)
    stride_dk_bias_batch, stride_dk_bias_nchunks, stride_dk_bias_head, stride_dk_bias_qkdim,
    # ---- sizes ----
    seqlen, nheads_qk, nheads, headdim_qk, headdim_angles,
    num_cblocks, gqa_ratio, num_qk_dim_blocks, num_ctiles, num_tiles, num_programs,
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    BLOCK_HEADDIM_QK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Grid: flat ``(num_programs,)``.
    The logical tiles are ``(batch, chunk, row-block)``; each program
    processes one or more of them, in order.

    Loop structure:
    - Outer loop: iterate over qk_heads (nheads_qk)
    - Inner loop: iterate over GQA group (GQA_RATIO heads per qk_head)
    """
    pid = tl.program_id(0)

    # Grid-stride over independent (batch, chunk, row-block) tiles.  With
    # ``num_programs == num_tiles`` (the common case) each program handles
    # exactly one tile and the mapping is identical to the former 2D grid; above
    # the Ascend coreDim cap only ``num_programs`` programs are launched and each
    # one walks the remaining tiles.  Tiles never depend on each other, so this
    # is a pure scheduling change.
    for tile_i in range(0, tl.cdiv(num_tiles - pid, num_programs)):
        tile_idx = pid + tile_i * num_programs
        pid_batch = tile_idx // num_ctiles
        pid_tile = tile_idx % num_ctiles

        # Decode the chunk and the in-chunk row sub-block so every live tile is
        # O(BLOCK_C x BLOCK_HEADDIM_QK) instead of O(CHUNK_SIZE x
        # BLOCK_HEADDIM_QK) (Ascend UB mitigation).  The bias partial is indexed
        # by the per-batch flattened tile id.
        pid_nchunk = pid_tile // num_cblocks
        pid_rb = pid_tile % num_cblocks
        row_start = pid_nchunk * CHUNK_SIZE + pid_rb * BLOCK_C

        # Base offsets for inputs
        q_offset_base = pid_batch * stride_q_batch
        k_offset_base = pid_batch * stride_k_batch
        scale_offset_base = pid_batch * stride_scale_batch
        gamma_offset_base = pid_batch * stride_gamma_batch
        angle_offset_base = pid_batch * stride_angles_batch
        dq_in_offset_base = pid_batch * stride_dq_in_batch
        dk_in_offset_base = pid_batch * stride_dk_in_batch
        dqk_offset_base = pid_batch * stride_dqk_batch

        # Base offsets for outputs
        dq_offset_base = pid_batch * stride_dq_batch
        dk_offset_base = pid_batch * stride_dk_batch
        dangle_offset_base = pid_batch * stride_dangles_batch
        dscale_offset_base = pid_batch * stride_dscale_batch
        dgamma_offset_base = pid_batch * stride_dgamma_batch
        dq_bias_offset_base = pid_batch * stride_dq_bias_batch + pid_tile * stride_dq_bias_nchunks
        dk_bias_offset_base = pid_batch * stride_dk_bias_batch + pid_tile * stride_dk_bias_nchunks

        offs_s = row_start + tl.arange(0, BLOCK_C)
        # Row mask: valid sequence position AND still inside this chunk.  The
        # second term is what makes the rounded-up ``num_cblocks`` safe.  For a
        # chunk size that is not a multiple of BLOCK_C (24/40/60 on the short
        # sequence path) the last row block extends past the chunk end; without
        # it those rows would be recomputed as part of the neighbouring chunk --
        # double-counted in the dQ_bias/dK_bias partials and stored twice into
        # dQ/dK/dAngles.
        rmask = (offs_s < seqlen) & (offs_s < (pid_nchunk + 1) * CHUNK_SIZE)
        for nhead_qk_id in range(num_qk_dim_blocks):
            offs_dr = tl.arange(0, BLOCK_HEADDIM_QK // 2) + nhead_qk_id * (BLOCK_HEADDIM_QK // 2)
            offs_e = 2 * offs_dr
            offs_o = offs_e + 1
            dr_mask = offs_dr < headdim_angles
            e_mask = offs_e < headdim_qk
            o_mask = offs_o < headdim_qk

            # Outer loop: iterate over qk_heads
            for qk_head_idx in range(nheads_qk):
                q_offset = q_offset_base + qk_head_idx * stride_q_head
                k_offset = k_offset_base + qk_head_idx * stride_k_head

                # Zero accumulators for this (qk_head, row sub-block).  Q/K are kept
                # as even/odd headdim halves so no bf16 tl.split/tl.join is needed.
                dq_acc_e = tl.zeros((BLOCK_C, BLOCK_HEADDIM_QK // 2), dtype=tl.float32)
                dq_acc_o = tl.zeros((BLOCK_C, BLOCK_HEADDIM_QK // 2), dtype=tl.float32)
                dk_acc_e = tl.zeros((BLOCK_C, BLOCK_HEADDIM_QK // 2), dtype=tl.float32)
                dk_acc_o = tl.zeros((BLOCK_C, BLOCK_HEADDIM_QK // 2), dtype=tl.float32)

                # Inner loop: iterate over GQA group
                for gqa_idx in range(gqa_ratio):
                    nhead_idx = qk_head_idx * gqa_ratio + gqa_idx

                    # Bias for this head (even/odd halves)
                    q_bias_e = tl.load(Q_bias + nhead_idx * stride_q_bias_head + offs_e * stride_q_bias_qkdim,
                                       mask=e_mask, other=0.0).to(tl.float32)
                    q_bias_o = tl.load(Q_bias + nhead_idx * stride_q_bias_head + offs_o * stride_q_bias_qkdim,
                                       mask=o_mask, other=0.0).to(tl.float32)
                    k_bias_e = tl.load(K_bias + nhead_idx * stride_k_bias_head + offs_e * stride_k_bias_qkdim,
                                       mask=e_mask, other=0.0).to(tl.float32)
                    k_bias_o = tl.load(K_bias + nhead_idx * stride_k_bias_head + offs_o * stride_k_bias_qkdim,
                                       mask=o_mask, other=0.0).to(tl.float32)

                    # Q + bias, K + bias (even/odd halves), row sub-block only
                    q0_e = tl.load(Q + q_offset + offs_s[:, None] * stride_q_seqlen + offs_e[None, :] * stride_q_qkdim,
                                   mask=rmask[:, None] & e_mask[None, :], other=0.0)
                    q0_o = tl.load(Q + q_offset + offs_s[:, None] * stride_q_seqlen + offs_o[None, :] * stride_q_qkdim,
                                   mask=rmask[:, None] & o_mask[None, :], other=0.0)
                    k0_e = tl.load(K + k_offset + offs_s[:, None] * stride_k_seqlen + offs_e[None, :] * stride_k_qkdim,
                                   mask=rmask[:, None] & e_mask[None, :], other=0.0)
                    k0_o = tl.load(K + k_offset + offs_s[:, None] * stride_k_seqlen + offs_o[None, :] * stride_k_qkdim,
                                   mask=rmask[:, None] & o_mask[None, :], other=0.0)
                    Q_e = q0_e + q_bias_e[None, :]
                    Q_o = q0_o + q_bias_o[None, :]
                    K_e = k0_e + k_bias_e[None, :]
                    K_o = k0_o + k_bias_o[None, :]

                    # dQK for this head
                    dqk_offset = dqk_offset_base + nhead_idx * stride_dqk_head
                    dqk = tl.load(dQK + dqk_offset + offs_s * stride_dqk_seqlen, mask=rmask, other=0.0)

                    # Scale, Gamma for this head
                    scale_offset = scale_offset_base + nhead_idx * stride_scale_head
                    gamma_offset = gamma_offset_base + nhead_idx * stride_gamma_head
                    scale = tl.load(Scale + scale_offset + offs_s * stride_scale_seqlen, mask=rmask, other=0.0).to(tl.float32)
                    gamma = tl.load(Gamma + gamma_offset + offs_s * stride_gamma_seqlen, mask=rmask, other=0.0).to(tl.float32)

                    # Angles for this head
                    angle_offset = angle_offset_base + nhead_idx * stride_angles_head
                    theta = tl.load(
                        Angles + angle_offset + offs_s[:, None] * stride_angles_seqlen + offs_dr[None, :] * stride_angles_qkdim,
                        mask=rmask[:, None] & dr_mask[None, :], other=0.0).to(tl.float32)

                    # dQ_in, dK_in for this head (even/odd halves)
                    dq_in_offset = dq_in_offset_base + nhead_idx * stride_dq_in_head
                    dk_in_offset = dk_in_offset_base + nhead_idx * stride_dk_in_head
                    dQ_in_e = tl.load(dQ_in + dq_in_offset + offs_s[:, None] * stride_dq_in_seqlen + offs_e[None, :] * stride_dq_in_qkdim,
                                      mask=rmask[:, None] & e_mask[None, :], other=0.0)
                    dQ_in_o = tl.load(dQ_in + dq_in_offset + offs_s[:, None] * stride_dq_in_seqlen + offs_o[None, :] * stride_dq_in_qkdim,
                                      mask=rmask[:, None] & o_mask[None, :], other=0.0)
                    dK_in_e = tl.load(dK_in + dk_in_offset + offs_s[:, None] * stride_dk_in_seqlen + offs_e[None, :] * stride_dk_in_qkdim,
                                      mask=rmask[:, None] & e_mask[None, :], other=0.0)
                    dK_in_o = tl.load(dK_in + dk_in_offset + offs_s[:, None] * stride_dk_in_seqlen + offs_o[None, :] * stride_dk_in_qkdim,
                                      mask=rmask[:, None] & o_mask[None, :], other=0.0)

                    # dGamma = dQK * (Q_wbias . K_wbias)  [fp32 products]
                    QK_dot = (tl.sum(Q_e.to(tl.float32) * K_e.to(tl.float32), axis=1)
                              + tl.sum(Q_o.to(tl.float32) * K_o.to(tl.float32), axis=1))
                    d_gamma = dqk * QK_dot
                    dgamma_store_offset = dgamma_offset_base + nhead_idx * stride_dgamma_head
                    tl.store(
                        dGamma + dgamma_store_offset + offs_s * stride_dgamma_seqlen + nhead_qk_id * stride_dgamma_nqkchunks,
                        d_gamma, mask=rmask)

                    # cos/sin for rotary
                    cos_angle = cos_approx(theta)
                    sin_angle = sin_approx(theta)

                    # dScale = sum(dK_in * rotary(K_wbias))
                    K_rot_e = K_e * cos_angle - K_o * sin_angle
                    K_rot_o = K_e * sin_angle + K_o * cos_angle
                    dscale_val = (tl.sum(dK_in_e.to(tl.float32) * K_rot_e, axis=1)
                                  + tl.sum(dK_in_o.to(tl.float32) * K_rot_o, axis=1))
                    dscale_store_offset = dscale_offset_base + nhead_idx * stride_dscale_head
                    tl.store(
                        dScale + dscale_store_offset + offs_s * stride_dscale_seqlen + nhead_qk_id * stride_dscale_nqkchunks,
                        dscale_val, mask=rmask)

                    # Inverse rotary for dQ_pre / dK_pre
                    dK_in_e_s = dK_in_e * scale[:, None]
                    dK_in_o_s = dK_in_o * scale[:, None]
                    dq0_e = dQ_in_e * cos_angle + dQ_in_o * sin_angle
                    dq0_o = -dQ_in_e * sin_angle + dQ_in_o * cos_angle
                    dk0_e = dK_in_e_s * cos_angle + dK_in_o_s * sin_angle
                    dk0_o = -dK_in_e_s * sin_angle + dK_in_o_s * cos_angle

                    dqk_scaled = (dqk * gamma)[:, None]
                    dQ_pre_e = dq0_e + dqk_scaled * K_e
                    dQ_pre_o = dq0_o + dqk_scaled * K_o
                    dK_pre_e = dk0_e + dqk_scaled * Q_e
                    dK_pre_o = dk0_o + dqk_scaled * Q_o

                    # Accumulate dQ/dK over the GQA group
                    dq_acc_e += dQ_pre_e
                    dq_acc_o += dQ_pre_o
                    dk_acc_e += dK_pre_e
                    dk_acc_o += dK_pre_o

                    # dQ_bias/dK_bias for this head: sum over this row sub-block only
                    # (the wrapper sums all flattened grid slots).
                    dq_bias_store_offset = dq_bias_offset_base + nhead_idx * stride_dq_bias_head
                    dk_bias_store_offset = dk_bias_offset_base + nhead_idx * stride_dk_bias_head
                    tl.store(dQ_bias + dq_bias_store_offset + offs_e * stride_dq_bias_qkdim,
                             tl.sum(dQ_pre_e, axis=0), mask=e_mask)
                    tl.store(dQ_bias + dq_bias_store_offset + offs_o * stride_dq_bias_qkdim,
                             tl.sum(dQ_pre_o, axis=0), mask=o_mask)
                    tl.store(dK_bias + dk_bias_store_offset + offs_e * stride_dk_bias_qkdim,
                             tl.sum(dK_pre_e, axis=0), mask=e_mask)
                    tl.store(dK_bias + dk_bias_store_offset + offs_o * stride_dk_bias_qkdim,
                             tl.sum(dK_pre_o, axis=0), mask=o_mask)

                    # dAngles
                    dtheta_q = dQ_in_e * (-Q_e * sin_angle - Q_o * cos_angle) + dQ_in_o * (Q_e * cos_angle - Q_o * sin_angle)
                    dtheta_k = dK_in_e_s * (-K_e * sin_angle - K_o * cos_angle) + dK_in_o_s * (K_e * cos_angle - K_o * sin_angle)
                    dtheta = dtheta_q + dtheta_k
                    dangle_store_offset = dangle_offset_base + nhead_idx * stride_dangles_head
                    tl.store(
                        dAngles + dangle_store_offset + offs_s[:, None] * stride_dangles_seqlen + offs_dr[None, :] * stride_dangles_qkdim,
                        dtheta, mask=rmask[:, None] & dr_mask[None, :])

                # Store accumulated dQ, dK for this qk_head (even/odd columns)
                dq_offset = dq_offset_base + qk_head_idx * stride_dq_head
                dk_offset = dk_offset_base + qk_head_idx * stride_dk_head
                tl.store(dQ + dq_offset + offs_s[:, None] * stride_dq_seqlen + offs_e[None, :] * stride_dq_qkdim,
                         dq_acc_e, mask=rmask[:, None] & e_mask[None, :])
                tl.store(dQ + dq_offset + offs_s[:, None] * stride_dq_seqlen + offs_o[None, :] * stride_dq_qkdim,
                         dq_acc_o, mask=rmask[:, None] & o_mask[None, :])
                tl.store(dK + dk_offset + offs_s[:, None] * stride_dk_seqlen + offs_e[None, :] * stride_dk_qkdim,
                         dk_acc_e, mask=rmask[:, None] & e_mask[None, :])
                tl.store(dK + dk_offset + offs_s[:, None] * stride_dk_seqlen + offs_o[None, :] * stride_dk_qkdim,
                         dk_acc_o, mask=rmask[:, None] & o_mask[None, :])


# NOTE: Do not autotune this kernel.  It used to ``tl.atomic_add`` into dK /
# dK_bias / dAngles, so an autotuner's repeated launches would accumulate
# several times.  It now writes deterministic fp32 partials (below), but the
# launch parameters stay pinned (num_warps=2 / num_stages=3) and un-autotuned so
# the measured cost is unchanged.
@triton.jit
def mamba3_siso_bwd_kernel_dk_state_post(
    # Inputs tensors
    dK_State, Angles, K, K_bias, Cu_Seqlens,
    # Outputs: deterministic fp32 partials, one row per (sequence, head)
    dK_State_Partial, dAngles_Partial,
    # Strides for dK_State: (num_sequences, nheads, headdim_qk)
    stride_dk_state_batch, stride_dk_state_head, stride_dk_state_qkdim,
    # Strides for Angles: (batch, seqlen, nheads, headdim_angles)
    stride_angles_batch, stride_angles_seqlen, stride_angles_head, stride_angles_qkdim,
    # Strides for K: (batch, seqlen, nheads_qk, headdim_qk)
    stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_qkdim,
    # Strides for K_bias: (nheads, headdim_qk)
    stride_k_bias_head, stride_k_bias_qkdim,
    # Strides for Cu_Seqlens: (num_sequences + 1,)
    stride_cu_seqlen,
    # Strides for dK_State_Partial: (num_sequences, nheads, headdim_qk)
    stride_dkp_seq, stride_dkp_head, stride_dkp_qkdim,
    # Strides for dAngles_Partial: (num_sequences, nheads, headdim_angles)
    stride_dap_seq, stride_dap_head, stride_dap_qkdim,
    # Dimensions
    seqlen, nheads, headdim_qk, headdim_angles,
    num_tiles, num_programs,
    HEADDIM_QK: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Post-kernel for d_ok_state contributions (grid-stride over tiles).

    Each logical tile is one (sequence, head) pair.  The kernel writes the
    inverse-rotated dK and the dAngles value into fp32 partial buffers -- one
    row per (sequence, head), never shared between tiles -- instead of
    accumulating with ``tl.atomic_add`` into the outputs.  ``apply_dk_state_post``
    then reduces them on the host with fixed-shape, atomic-free reductions:

    * dK      += sum over the GQA group of ``dK_State_Partial`` (for that
                 sequence's last token),
    * dK_bias += sum over sequences of ``dK_State_Partial``,
    * dAngles += ``dAngles_Partial`` (one writer per (batch, last_pos, head)).

    A zero-length segment (legal in a packed varlen batch) has no last token:
    ``cu_end - 1`` would be the previous segment's last token, so such a tile
    contributes nothing.  It still writes its partial rows (as zeros), because
    ``dK_bias`` sums every sequence row and the buffers are allocated with
    ``torch.empty``.

    Removing the atomics makes the result bitwise reproducible across runs (an
    fp32 atomic-add order is not).  This holds for both dK store dtypes: fp32 on
    this branch (``compute_dqkv`` returns fp32) and bf16 in the upstream
    ``077d78b`` contract (``dk = torch.empty(..., dtype=k.dtype)``).
    """
    pid = tl.program_id(0)

    # Grid-stride over independent (sequence, head) tiles; same mechanism as the
    # main kernel above and as ``mamba3_siso_fwd.py``.
    for tile_i in range(0, tl.cdiv(num_tiles - pid, num_programs)):
        tile_idx = pid + tile_i * num_programs
        pid_head = tile_idx % nheads
        pid_seq = tile_idx // nheads

        if IS_VARLEN:
            pid_batch = 0
            cu_start = tl.load(Cu_Seqlens + pid_seq * stride_cu_seqlen).to(tl.int32)
            cu_end = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
            # Emptiness is ``end == start``, not ``last_pos < 0``: a zero-length
            # segment in the middle of the pack has ``end - 1 >= 0`` and would
            # otherwise duplicate the previous segment's last token.
            has_last = cu_end > cu_start
            last_pos = tl.maximum(cu_end - 1, 0)
        else:
            pid_batch = pid_seq
            last_pos = seqlen - 1
            has_last = True

        qk_head_idx = pid_head // GQA_RATIO
        offs_d = tl.arange(0, HEADDIM_QK)
        offs_dr = tl.arange(0, HEADDIM_QK // 2)

        # Load dK_State as interleaved pairs
        dk_state_base = dK_State + pid_seq * stride_dk_state_batch + pid_head * stride_dk_state_head
        dk_state = tl.load(dk_state_base + offs_d * stride_dk_state_qkdim, mask=offs_d < headdim_qk, other=0.0).to(tl.float32)
        dk_state_r = tl.reshape(dk_state, [HEADDIM_QK // 2, 2])
        dk_state_r0, dk_state_r1 = tl.split(dk_state_r)  # shape: (HEADDIM_QK // 2,)

        # Load angles at last position
        angles_base = Angles + pid_batch * stride_angles_batch + last_pos * stride_angles_seqlen + pid_head * stride_angles_head
        angles_val = tl.load(angles_base + offs_dr * stride_angles_qkdim, mask=offs_dr < headdim_angles, other=0.0).to(tl.float32)  # shape: (HEADDIM_QK // 2,)

        cos_ang = cos_approx(angles_val)
        sin_ang = sin_approx(angles_val)

        # Inverse rotary: dk_rotated
        dk0 = dk_state_r0 * cos_ang + dk_state_r1 * sin_ang
        dk1 = -dk_state_r0 * sin_ang + dk_state_r1 * cos_ang
        dk_rotated = tl.reshape(tl.join(dk0, dk1), [HEADDIM_QK])

        # Compute dAngles.  Load K at last position (using qk_head_idx for GQA)
        k_base = K + pid_batch * stride_k_batch + last_pos * stride_k_seqlen + qk_head_idx * stride_k_head
        k_val = tl.load(k_base + offs_d * stride_k_qkdim, mask=offs_d < headdim_qk, other=0.0).to(tl.float32)
        kr = tl.reshape(k_val, [HEADDIM_QK // 2, 2])
        k_r0, k_r1 = tl.split(kr)  # shape: (HEADDIM_QK // 2,)

        # Load K_bias
        k_bias_base = K_bias + pid_head * stride_k_bias_head
        k_bias_val = tl.load(k_bias_base + offs_d * stride_k_bias_qkdim, mask=offs_d < headdim_qk, other=0.0).to(tl.float32)
        kbr = tl.reshape(k_bias_val, [HEADDIM_QK // 2, 2])
        kb_r0, kb_r1 = tl.split(kbr)  # shape: (HEADDIM_QK // 2,)

        # K_wbias = K + K_bias
        K_wbias_r0 = k_r0 + kb_r0
        K_wbias_r1 = k_r1 + kb_r1

        # dtheta = dk_r0 * (-K0*sin - K1*cos) + dk_r1 * (K0*cos - K1*sin)
        dtheta_k = (dk_state_r0 * (-K_wbias_r0 * sin_ang - K_wbias_r1 * cos_ang) +
                    dk_state_r1 * (K_wbias_r0 * cos_ang - K_wbias_r1 * sin_ang))

        # Store the partials (no atomics): each (sequence, head) row is written
        # by exactly one tile.  An empty segment stores zeros so the host-side
        # dK_bias reduction stays exact without an extra masked copy.
        dkp_base = dK_State_Partial + pid_seq * stride_dkp_seq + pid_head * stride_dkp_head
        tl.store(dkp_base + offs_d * stride_dkp_qkdim,
                 tl.where(has_last, dk_rotated, 0.0), mask=offs_d < headdim_qk)

        dap_base = dAngles_Partial + pid_seq * stride_dap_seq + pid_head * stride_dap_head
        tl.store(dap_base + offs_dr * stride_dap_qkdim,
                 tl.where(has_last, dtheta_k, 0.0), mask=offs_dr < headdim_angles)


def compute_dqktheta(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: torch.Tensor,
    gamma: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    angles: torch.Tensor,
    dq_in: torch.Tensor,
    dk_in: torch.Tensor,
    dqk: torch.Tensor,
    d_ok_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    Cu_Seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute gradients through rotary embeddings and biases for Mamba-3 backward pass.

    This kernel undoes the rotary embedding and computes gradients for the original Q, K,
    angles, scaling factors, and biases.

    Args:
        q: Original query tensor before bias/rotary (batch, seqlen, nheads_qk, headdim_qk)
        k: Original key tensor before bias/rotary (batch, seqlen, nheads_qk, headdim_qk)
        scale: Combined scale factor gamma + gamma (batch, nheads, seqlen)
        gamma: gamma factor (batch, nheads, seqlen)
        q_bias: Query bias (nheads, headdim_qk)
        k_bias: Key bias (nheads, headdim_qk)
        angles: Rotary angles (batch, seqlen, nheads, headdim_angles)
        dq_in: Gradient from downstream for Q_mid (batch, seqlen, nheads, headdim_qk)
        dk_in: Gradient from downstream for K_mid (batch, seqlen, nheads, headdim_qk)
        dqk: Gradient for QK dot products (batch, nheads, seqlen)
        d_ok_state: Gradient of output K state (batch, nheads, headdim_qk) - added to last token of dK (without scaling)
        chunk_size: Chunk size (default: 64)
        Cu_Seqlens: cumulative varlen offsets (num_sequences + 1,).  A
            zero-length segment is legal; its d_ok_state row contributes nothing
            (see apply_dk_state_post).

    Returns:
        Tuple of (dQ, dK, dQ_bias, dK_bias, dAngles, dScale, dSGamma)
        - dQ: (batch, seqlen, nheads_qk, headdim_qk)
        - dK: (batch, seqlen, nheads_qk, headdim_qk)
        - dQ_bias: (nheads, headdim_qk)
        - dK_bias: (nheads, headdim_qk)
        - dAngles: (batch, seqlen, nheads, headdim_angles)
        - dScale: (batch, nheads, seqlen)
        - dGamma: (batch, nheads, seqlen)
    """
    batch, seqlen, nheads_qk, headdim_qk = q.shape
    assert q.shape == k.shape

    nheads = scale.shape[1]
    chunk_size = effective_chunk_size(chunk_size, seqlen)
    nchunks = triton.cdiv(seqlen, chunk_size)
    GQA_RATIO = nheads // nheads_qk

    assert scale.shape == (batch, nheads, seqlen)
    assert gamma.shape == (batch, nheads, seqlen)
    assert q_bias.shape == (nheads, headdim_qk)
    assert k_bias.shape == (nheads, headdim_qk)
    headdim_angles = angles.shape[-1]
    assert angles.shape == (batch, seqlen, nheads, headdim_angles)
    assert dq_in.shape == (batch, seqlen, nheads, headdim_qk)
    assert dk_in.shape == (batch, seqlen, nheads, headdim_qk)
    assert dqk.shape == (batch, nheads, seqlen)
    if d_ok_state is not None:
        num_sequences = Cu_Seqlens.shape[0] - 1 if Cu_Seqlens is not None else batch
        assert d_ok_state.shape == (num_sequences, nheads, headdim_qk)
    assert nheads % nheads_qk == 0, "nheads must be multiple of nheads_qk for GQA support"

    # Ensure contiguity after reshaping
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()
    if not dqk.is_contiguous():
        dqk = dqk.contiguous()
    if not angles.is_contiguous():
        angles = angles.contiguous()
    if not dq_in.is_contiguous():
        dq_in = dq_in.contiguous()
    if not dk_in.is_contiguous():
        dk_in = dk_in.contiguous()
    if q_bias.stride(-1) != 1:
        q_bias = q_bias.contiguous()
    if k_bias.stride(-1) != 1:
        k_bias = k_bias.contiguous()
    if d_ok_state is not None and (not d_ok_state.is_contiguous()):
        d_ok_state = d_ok_state.contiguous()

    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    BLOCK_HEADDIM_QK = min(HEADDIM_QK, 64)

    # Ascend UB mitigation: split the chunk dimension into BLOCK_C row sub-blocks
    # and put them on the grid, so every tile is O(BLOCK_C x BLOCK_HEADDIM_QK).
    #
    # PLUT-70: ``num_cblocks`` is rounded *up*, not down.  With the old
    # ``chunk_size // BLOCK_C`` a chunk size that is not a multiple of 16 (24/40/
    # 60 on the short-sequence path) silently left the last
    # ``chunk_size % BLOCK_C`` rows of every chunk uncomputed: dQ/dK/dAngles
    # were never stored (torch.empty garbage) and the bias partials missed those
    # rows.  Rounding up alone would make the last row block spill into the next
    # chunk and double-count it, so the kernel also masks rows with
    # ``offs_s < (pid_nchunk + 1) * CHUNK_SIZE``.
    #
    # Kept BLOCK_C = min(16, chunk_size) (a power-of-two tile) instead of picking
    # a divisor of chunk_size: the divisor approach would make BLOCK_C 8/8/15 for
    # chunk 24/40/60 -- a different UB footprint and a different autotune-key
    # value per chunk size -- and would degrade to tiny tiles for prime chunk
    # sizes.  The mask keeps one BLOCK_C value for every chunk size.
    BLOCK_C = min(16, chunk_size)
    num_cblocks = triton.cdiv(chunk_size, BLOCK_C)
    num_qk_dim_blocks = HEADDIM_QK // BLOCK_HEADDIM_QK

    # Allocate output tensors layout
    dq = torch.empty((batch, seqlen, nheads_qk, headdim_qk),
                              dtype=dq_in.dtype, device=q.device)
    dk = torch.empty((batch, seqlen, nheads_qk, headdim_qk),
                              dtype=dk_in.dtype, device=k.device)
    dangles = torch.empty((batch, seqlen, nheads, headdim_angles),
                                   dtype=angles.dtype, device=angles.device)
    dscale = torch.empty((batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, seqlen),
                                  dtype=scale.dtype, device=scale.device)
    dgamma = torch.empty((batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, seqlen),
                                   dtype=gamma.dtype, device=gamma.device)
    # Bias partials are indexed by the flattened (chunk, row-block) grid id.
    dq_bias_partial = torch.empty((batch, nchunks * num_cblocks, nheads, headdim_qk),
                                   dtype=torch.float32, device=q.device)
    dk_bias_partial = torch.empty((batch, nchunks * num_cblocks, nheads, headdim_qk),
                                   dtype=torch.float32, device=k.device)

    # Flat grid-stride launch: one tile per (batch, chunk, row-block); capped at
    # the Ascend coreDim limit with the true logical tile count passed in.
    num_ctiles = nchunks * num_cblocks
    num_tiles = num_ctiles * batch
    num_programs = _grid_programs(num_tiles)
    _record_launch("main", num_tiles, num_programs)
    grid = (num_programs,)

    mamba3_siso_bwd_kernel_rotary_bias_angles[grid](
        # Input tensors
        q, k, scale, gamma, q_bias, k_bias, angles, dq_in, dk_in, dqk,
        # Output tensors
        dq, dk, dangles, dscale, dgamma, dq_bias_partial, dk_bias_partial,
        # Q strides: (batch, seqlen, nheads_qk, headdim_qk)
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # Scale strides: (batch, nheads, seqlen)
        scale.stride(0), scale.stride(1), scale.stride(2),
        # SGamma strides
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        # Q_bias strides: (nheads, headdim_qk)
        q_bias.stride(0), q_bias.stride(1),
        # K_bias strides
        k_bias.stride(0), k_bias.stride(1),
        # Angles strides: (batch, seqlen, nheads, headdim_qk//2)
        angles.stride(0), angles.stride(1), angles.stride(2), angles.stride(3),
        # dQ_in strides: (batch, seqlen, nheads, headdim_qk)
        dq_in.stride(0), dq_in.stride(1), dq_in.stride(2), dq_in.stride(3),
        # dK_in strides
        dk_in.stride(0), dk_in.stride(1), dk_in.stride(2), dk_in.stride(3),
        # dQK strides: (batch, nheads, seqlen)
        dqk.stride(0), dqk.stride(1), dqk.stride(2),
        # Output tensors
        # dQ strides: (batch, seqlen, nheads_qk, headdim_qk)
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        # dK strides
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        # dAngles strides: (batch, seqlen, nheads, headdim_qk//2)
        dangles.stride(0), dangles.stride(1), dangles.stride(2), dangles.stride(3),
        # dScale strides: (batch, nheads, seqlen)
        dscale.stride(0), dscale.stride(1), dscale.stride(2), dscale.stride(3),
        # dSGamma strides
        dgamma.stride(0), dgamma.stride(1), dgamma.stride(2), dgamma.stride(3),
        # dQ_bias_partial strides: (batch, nchunks, nheads, headdim_qk)
        dq_bias_partial.stride(0), dq_bias_partial.stride(1),
        dq_bias_partial.stride(2), dq_bias_partial.stride(3),
        # dK_bias_partial strides
        dk_bias_partial.stride(0), dk_bias_partial.stride(1),
        dk_bias_partial.stride(2), dk_bias_partial.stride(3),
        # Sizes
        seqlen, nheads_qk, nheads, headdim_qk, headdim_angles,
        num_cblocks, GQA_RATIO, num_qk_dim_blocks,
        num_ctiles, num_tiles, num_programs,
        CHUNK_SIZE=chunk_size,
        HEADDIM_QK=HEADDIM_QK,
        BLOCK_HEADDIM_QK=BLOCK_HEADDIM_QK,
        BLOCK_C=BLOCK_C,
    )

    # Reshape outputs back to original layout
    dscale = torch.sum(dscale, dim=2)  # Sum over headdim blocks
    dgamma = torch.sum(dgamma, dim=2)  # Sum over headdim blocks

    # Reduce bias gradients: (batch, nchunks, nheads, headdim_qk) -> (nheads, headdim_qk)
    dq_bias = dq_bias_partial.sum(dim=(0, 1))
    dk_bias = dk_bias_partial.sum(dim=(0, 1))

    # NOTE: We handle d_ok_state contributions in a different kernel because merging it in
    # causes a +800% increase in register spillage and a +200us increase in runtime. For now
    # this new kernel only introduces +5us.
    if d_ok_state is not None:
        apply_dk_state_post(
            d_ok_state, angles, k, k_bias, dk, dk_bias, dangles, Cu_Seqlens
        )
    return dq, dk, dq_bias, dk_bias, dangles, dscale, dgamma

def apply_dk_state_post(
    d_ok_state: torch.Tensor,
    angles: torch.Tensor,
    k: torch.Tensor,
    k_bias: torch.Tensor,
    dk: torch.Tensor,
    dk_bias: torch.Tensor,
    dangles: torch.Tensor,
    Cu_Seqlens: Optional[torch.Tensor] = None,
):
    """Add the final-K-state gradient into ``dK`` / ``dK_bias`` / ``dAngles``.

    ``dk`` is ``(batch, seqlen, nheads_qk, headdim_qk)``, ``dangles`` is
    ``(batch, seqlen, nheads, headdim_angles)`` and ``dk_bias`` is
    ``(nheads, headdim_qk)``; the three are updated in place.  The contribution
    lands on each sequence's last token (``dK`` at the GQA-reduced head,
    ``dK_bias`` summed over sequences, ``dAngles`` per head).

    Varlen contract: ``Cu_Seqlens`` is the usual non-decreasing
    ``(num_sequences + 1,)`` cumulative offset array, and a **zero-length
    segment is legal and skipped** -- it has no last token, so it contributes
    nothing and does not fail the call for the other sequences.  An empty
    segment is detected as ``Cu_Seqlens[s+1] == Cu_Seqlens[s]`` (not as
    ``last_pos < 0``: a zero-length segment in the middle of the pack would
    still have ``last_pos >= 0`` and would duplicate its predecessor's last
    token).
    """
    batch, seqlen, nheads, headdim_angles = angles.shape
    _, _, headdim_qk = d_ok_state.shape
    nheads_qk = k.shape[2]
    GQA_RATIO = nheads // nheads_qk

    is_varlen = Cu_Seqlens is not None
    if is_varlen:
        num_sequences = Cu_Seqlens.shape[0] - 1
        assert batch == 1
    else:
        num_sequences = batch

    # Ensure contiguity
    if not d_ok_state.is_contiguous():
        d_ok_state = d_ok_state.contiguous()
    if not angles.is_contiguous():
        angles = angles.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not k_bias.is_contiguous():
        k_bias = k_bias.contiguous()

    if seqlen == 0:
        # Fully empty pack (every segment zero-length): there is no token to
        # accumulate into.  Returning here also keeps the kernel from loading
        # position 0 of a zero-length sequence.
        return

    HEADDIM_QK = triton.next_power_of_2(headdim_qk)

    # PLUT-70: deterministic replacement for the fp32 atomic adds.  The kernel
    # writes one fp32 partial row per (sequence, head) -- every row has exactly
    # one writer -- and the reductions below are single fixed-shape kernels.
    # The old atomics made the fp32 summation order (and thus the low bits)
    # run-dependent; in the committed contract dK is bf16, where every atomic
    # add rounded again as well.
    dk_state_partial = torch.empty((num_sequences, nheads, headdim_qk),
                                   dtype=torch.float32, device=dk.device)
    dangles_partial = torch.empty((num_sequences, nheads, headdim_angles),
                                  dtype=torch.float32, device=dangles.device)

    # Flat grid-stride launch over (sequence, head) tiles, capped at coreDim.
    num_tiles = num_sequences * nheads
    num_programs = _grid_programs(num_tiles)
    _record_launch("post", num_tiles, num_programs)
    grid = (num_programs,)

    mamba3_siso_bwd_kernel_dk_state_post[grid](
        # Input tensors
        d_ok_state, angles, k, k_bias, Cu_Seqlens,
        # Output partials
        dk_state_partial, dangles_partial,
        # dK_State strides: (num_sequences, nheads, headdim_qk)
        d_ok_state.stride(0), d_ok_state.stride(1), d_ok_state.stride(2),
        # Angles strides: (batch, seqlen, nheads, headdim_angles)
        angles.stride(0), angles.stride(1), angles.stride(2), angles.stride(3),
        # K strides: (batch, seqlen, nheads_qk, headdim_qk)
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # K_bias strides: (nheads, headdim_qk)
        k_bias.stride(0), k_bias.stride(1),
        # Cu_Seqlens strides: (num_sequences + 1,)
        Cu_Seqlens.stride(0) if is_varlen else 0,
        # dK_State_Partial strides: (num_sequences, nheads, headdim_qk)
        dk_state_partial.stride(0), dk_state_partial.stride(1),
        dk_state_partial.stride(2),
        # dAngles_Partial strides: (num_sequences, nheads, headdim_angles)
        dangles_partial.stride(0), dangles_partial.stride(1),
        dangles_partial.stride(2),
        # Dimensions
        seqlen, nheads, headdim_qk, headdim_angles,
        num_tiles, num_programs,
        HEADDIM_QK=HEADDIM_QK,
        GQA_RATIO=GQA_RATIO,
        IS_VARLEN=is_varlen,
        num_warps=2,
        num_stages=3,
    )

    # --- deterministic host-side reductions (no atomics) ---------------------
    # Each reduction is a *single* fixed-shape kernel over the whole partial
    # buffer, so every run performs the same additions in the same order (no
    # atomic accumulation, no run-dependent partition).  A Python loop over
    # ``range(num_sequences)`` would launch one small kernel per sequence: at
    # ~15 us per launch that is ~60 ms of host-serialised work for a 4096-
    # sequence varlen batch, versus ~30 us for the single reduction below.
    #
    # dK_bias: sum over sequences, per head.
    dk_bias.add_(dk_state_partial.sum(dim=0))

    # dK / dAngles are written at each sequence's own last token.  The
    # (batch, last_pos) pairs are unique, so this scatter has no accumulation
    # order to be non-deterministic about.
    #
    # A zero-length segment is skipped: ``cu_end - 1`` would be the *previous*
    # segment's last token (a leading empty segment gives -1, which Python/Torch
    # wraps to the last row), so it must not reach the scatter.  The kernel
    # already stores zeros into that segment's partial rows, which keeps the
    # dK_bias sum above exact.
    if is_varlen:
        cu = Cu_Seqlens.to(torch.long)
        cu_start = cu[:num_sequences]
        cu_end = cu[1:num_sequences + 1]
        keep = cu_end > cu_start
        batch_idx = torch.zeros(num_sequences, dtype=torch.long, device=dk.device)
        last_idx = (cu_end - 1).clamp(min=0)
    else:
        keep = None
        batch_idx = torch.arange(num_sequences, dtype=torch.long, device=dk.device)
        last_idx = torch.full((num_sequences,), seqlen - 1, dtype=torch.long,
                              device=dk.device)

    # GQA reduction of the per-head dK partial over g = 0 .. R-1: one reduction
    # kernel over the fixed (num_sequences, nheads_qk, GQA_RATIO, headdim_qk)
    # view.  ``g`` is the fastest head index (``pid_head = qk_head*R + g``).
    dk_qk_acc = dk_state_partial.view(num_sequences, nheads_qk, GQA_RATIO,
                                      headdim_qk).sum(dim=2)

    # Drop the empty segments before the scatter (a no-op when every segment has
    # tokens, which is the only case on the non-varlen path).
    if keep is None:
        sel = torch.arange(num_sequences, dtype=torch.long, device=dk.device)
    else:
        sel = torch.nonzero(keep, as_tuple=False).squeeze(1)
    tgt_b = batch_idx[sel]
    tgt_t = last_idx[sel]
    # Ascend ``IndexPut`` rejects a mixed-dtype store (161002 "expected DT_FLOAT
    # but found DT_BFLOAT16"), so the fp32 sum is cast back to the destination
    # dtype before the scatter.  ``dk`` / ``dangles`` keep their own dtype
    # (``dk_in.dtype`` / ``angles.dtype``); the cast is a no-op when they are fp32.
    dk[tgt_b, tgt_t] = (dk[tgt_b, tgt_t].float() + dk_qk_acc[sel]).to(dk.dtype)
    dangles[tgt_b, tgt_t] = (dangles[tgt_b, tgt_t].float() + dangles_partial[sel]).to(dangles.dtype)


# =============================================================================
# dDT, dTrap, and dInput States Kernel
# =============================================================================
# QK-dimension tile for the input-state branch of this kernel.  The full
# (HEADDIM_V x HEADDIM_QK) fp32 state tile exceeds the 192 KB Ascend Unified
# Buffer for large head dims (measured 4206592 bits required vs 1572864
# available at Dv/Dqk = 256/128); blocking the QK axis by this much keeps every
# live tile O(HEADDIM_V x BLOCK_QK).  The bounded loop is runtime-counted so it
# is not unrolled (each unrolled copy would reserve its own UB buffers).
NPU_DDT_QK_BLOCK = 32

# Chunk-size space for this kernel.  The UB footprint here is driven by the
# state tile (bounded by NPU_DDT_QK_BLOCK above), NOT by CHUNK_SIZE: Part-1 is
# elementwise over a CHUNK_SIZE vector (~24*CHUNK_SIZE bytes) and Part-2 is
# chunk-independent.  The global `NPU_MAX_CHUNK_SIZE=64` cap therefore has no
# kernel-specific UB evidence, and the PLUT-43 NPU run of
# `test_ddt_dtrap_dinput_states_original_autotune_candidates` PASSED with the
# restored chunk {64,128,256} (no `ub overflow` / compile failure).  A collapse
# requires evidence (NPU_MIGRATION_mamba3_siso.md §14), so the full upstream
# chunk space is kept here; `chunk_choices` with this cap is a no-op filter.
NPU_DDT_MAX_CHUNK = 256

@triton.autotune(
    configs=[
        make_config({"CHUNK_SIZE": cs}, s, w, r)
        for cs in chunk_choices([64, 128, 256], cap=NPU_DDT_MAX_CHUNK)
        for s in stage_choices([1, 2, 3])
        for w in [2, 4, 8]
        for r in autotune_regs([None, 128, 256])
    ],
    key=["HEADDIM_V", "HEADDIM_QK", "HAS_INPUT_STATE", "IS_VARLEN", "PRECISE_FP32"]
)
@triton.jit
def mamba3_siso_bwd_kernel_ddt_dtrap_dinput_states(
    # Input tensors
    dScale, dGamma, DT, Trap,
    d_ISSM_State, Input_K_State, Input_V_State, Cu_Seqlens,
    # Output tensors
    dDT, dTrap,
    dInput_SSM_State, dInput_K_State, dInput_V_State,
    # Strides for dScale: (batch, nheads, seqlen)
    stride_dscale_batch, stride_dscale_head, stride_dscale_seqlen,
    # Strides for dGamma: (batch, nheads, seqlen)
    stride_dgamma_batch, stride_dgamma_head, stride_dgamma_seqlen,
    # Strides for DT: (batch, nheads, seqlen)
    stride_dt_batch, stride_dt_head, stride_dt_seqlen,
    # Strides for Trap: (batch, nheads, seqlen)
    stride_trap_batch, stride_trap_head, stride_trap_seqlen,
    # Strides for d_ISSM_State: (num_sequences, nheads, headdim_v, headdim_qk)
    stride_d_issm_state_batch, stride_d_issm_state_head, stride_d_issm_state_vdim, stride_d_issm_state_qkdim,
    # Strides for Input_K_State: (num_sequences, nheads, headdim_qk)
    stride_input_k_state_batch, stride_input_k_state_head, stride_input_k_state_qkdim,
    # Strides for Input_V_State: (num_sequences, nheads, headdim_v)
    stride_input_v_state_batch, stride_input_v_state_head, stride_input_v_state_vdim,
    # Stride for Cu_Seqlens
    stride_cu_seqlen,
    # Strides for dDT: (batch, nheads, seqlen)
    stride_ddt_batch, stride_ddt_head, stride_ddt_seqlen,
    # Strides for dTrap: (batch, nheads, seqlen)
    stride_dtrap_batch, stride_dtrap_head, stride_dtrap_seqlen,
    # Strides for dInput_SSM_State: (num_sequences, nheads, headdim_v, headdim_qk)
    stride_dinput_ssm_state_batch, stride_dinput_ssm_state_head, stride_dinput_ssm_state_vdim, stride_dinput_ssm_state_qkdim,
    # Strides for dInput_K_State: (num_sequences, nheads, headdim_qk)
    stride_dinput_k_state_batch, stride_dinput_k_state_head, stride_dinput_k_state_qkdim,
    # Strides for dInput_V_State: (num_sequences, nheads, headdim_v)
    stride_dinput_v_state_batch, stride_dinput_v_state_head, stride_dinput_v_state_vdim,
    # Dimensions
    seqlen, nheads, headdim_v, headdim_qk,
    # Launch geometry (logical tile count and capped program count)
    num_tiles, num_programs,
    # Compile-time constants
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    HAS_INPUT_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_QK: tl.constexpr,
    PRECISE_FP32: tl.constexpr,
):
    """
    Backward kernel for computing dDT, dTrap, and input state gradients.

    Part 1 - dDT and dTrap from dScale and dGamma:
        Forward: gamma_t = DT_t * Trap_t                    (used independently)
                 shifted_gamma_t = DT_{t+1} * (1 - Trap_{t+1})  (used as scale for position t)

        Backward: DT[t] appears in gamma[t] and shifted_gamma[t-1]:
                  dDT_t = dGamma_t * Trap_t + dScale_{t-1} * (1 - Trap_t)

                  Trap[t] appears in gamma[t] and shifted_gamma[t-1]:
                  dTrap_t = dGamma_t * DT_t - dScale_{t-1} * DT_t

    Part 2 - Input state gradients (first token only, if HAS_INPUT_STATE):
        Forward: scalar = DT_0 * (1 - Trap_0)
                 SSM_State = Input_SSM_State + outer(Input_V, Input_K) * scalar
        Backward: dInput_SSM_State = d_ISSM_State
                  dInput_V = einsum(d_ISSM_State, Input_K) * scalar
                  dInput_K = einsum(d_ISSM_State, Input_V) * scalar
                  dDT_0 += d_scalar * (1 - Trap_0)
                  dTrap_0 += d_scalar * (-DT_0)

        Part 2 only runs when the caller passes ``d_issm_state``: the SISO
        training chain forwards ``Input_States=None`` (``mamba3.py``), so
        ``HAS_INPUT_STATE=False`` there and Part 2 is not launched.  Every
        input-state operand is upcast to fp32 inside the kernel, so any floating
        dtype is accepted; the forward's ``Final_V_State`` is bf16, which is what
        a real Part-2 caller would feed.

    Grid:
        - Flat ``(num_programs,)``, ``num_programs <= coreDim``.  Tile
          ``tile_idx = pid + tile_i * num_programs`` maps to
          ``pid_head = tile_idx % nheads`` and ``pid_batch = tile_idx // nheads``
          (normal mode) or the varlen sequence index (``IS_VARLEN``).
        - ``num_programs == num_tiles`` (the common case) walks one tile per
          program; above the cap each program grid-strides over the rest.
    """
    pid = tl.program_id(0)

    # Grid-stride over independent (head, batch/num_sequences) tiles.  ``nheads``
    # and ``num_programs`` are passed explicitly instead of deriving them from
    # ``tl.num_programs(0)``: Triton-Ascend folds the logical grid onto the
    # physical AI cores, so the program count seen inside the kernel is not the
    # logical grid extent.
    for tile_i in range(0, tl.cdiv(num_tiles - pid, num_programs)):
        tile_idx = pid + tile_i * num_programs
        pid_head = tile_idx % nheads
        pid_batch = tile_idx // nheads

        if IS_VARLEN:
            pid_seq = pid_batch
            pid_batch = 0
            cu_seqlen = tl.load(Cu_Seqlens + pid_seq * stride_cu_seqlen).to(tl.int32)
            cu_seqlen_next = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
            seqlen = cu_seqlen_next - cu_seqlen
        else:
            pid_seq = 0
            cu_seqlen = 0

        # ==================== Pointer Offsets ====================
        dscale_offset = pid_batch * stride_dscale_batch + pid_head * stride_dscale_head + IS_VARLEN * cu_seqlen * stride_dscale_seqlen
        dgamma_offset = pid_batch * stride_dgamma_batch + pid_head * stride_dgamma_head + IS_VARLEN * cu_seqlen * stride_dgamma_seqlen
        dt_offset = pid_batch * stride_dt_batch + pid_head * stride_dt_head + IS_VARLEN * cu_seqlen * stride_dt_seqlen
        trap_offset = pid_batch * stride_trap_batch + pid_head * stride_trap_head + IS_VARLEN * cu_seqlen * stride_trap_seqlen
        ddt_offset = pid_batch * stride_ddt_batch + pid_head * stride_ddt_head + IS_VARLEN * cu_seqlen * stride_ddt_seqlen
        dtrap_offset = pid_batch * stride_dtrap_batch + pid_head * stride_dtrap_head + IS_VARLEN * cu_seqlen * stride_dtrap_seqlen

        # ==================== Part 1: dDT and dTrap ====================
        num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

        for chunk_idx in range(num_chunks):
            offs_s = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
            mask = offs_s < seqlen

            # Load dscale_t, dGamma_t, Trap_t, DT_t for current positions.
            # PLUT-51: upcast every Part-1 cotangent operand to fp32 BEFORE the
            # arithmetic when ``PRECISE_FP32`` is set.  With fp16 inputs
            # `dgamma_t + dscale_t` (and the `* dt_t` products in the dTrap path)
            # would otherwise be rounded in the input dtype, losing ~1e-3 before
            # the fp32 output is written (the golden upcasts the verbatim inputs
            # to fp64, so that intermediate rounding is not common-mode).  Casting
            # is a no-op on the fp32 real chain.  See ``npu_compat.precise_ddt``.
            dscale_t = tl.load(dScale + dscale_offset + offs_s * stride_dscale_seqlen, mask=mask, other=0.0)
            dgamma_t = tl.load(dGamma + dgamma_offset + offs_s * stride_dgamma_seqlen, mask=mask, other=0.0)
            trap_presig_t = tl.load(Trap + trap_offset + offs_s * stride_trap_seqlen, mask=mask, other=0.0).to(tl.float32)
            trap_t = sigmoid_approx(trap_presig_t)
            dt_t = tl.load(DT + dt_offset + offs_s * stride_dt_seqlen, mask=mask, other=0.0)

            # Load dScale_{t-1} (shifted by 1, with 0 at t=0)
            # shifted_gamma[t-1] = DT[t] * (1 - Trap[t]) feeds into scale[t-1]
            offs_s_prev = offs_s - 1
            mask_prev = (offs_s_prev >= 0) & (offs_s_prev < seqlen)
            dscale_prev = tl.load(
                dScale + dscale_offset + offs_s_prev * stride_dscale_seqlen,
                mask=mask_prev,
                other=0.0
            )

            if PRECISE_FP32:
                dscale_t = dscale_t.to(tl.float32)
                dgamma_t = dgamma_t.to(tl.float32)
                dt_t = dt_t.to(tl.float32)
                dscale_prev = dscale_prev.to(tl.float32)

            # Compute gradients:
            ddt_t = (dgamma_t + dscale_t) * trap_t + dscale_prev * (1.0 - trap_t)
            dtrap_t = (dgamma_t + dscale_t) * dt_t - dscale_prev * dt_t
            dtrap_presig_t = dtrap_t * trap_t * (1.0 - trap_t)

            # Store results
            tl.store(dDT + ddt_offset + offs_s * stride_ddt_seqlen, ddt_t, mask=mask)
            tl.store(dTrap + dtrap_offset + offs_s * stride_dtrap_seqlen, dtrap_presig_t, mask=mask)

        # ==================== Part 2: Input State Gradients ====================
        if HAS_INPUT_STATE:
            # Pointer offsets for input states
            d_issm_offset = (pid_batch + pid_seq) * stride_d_issm_state_batch + pid_head * stride_d_issm_state_head
            input_k_offset = (pid_batch + pid_seq) * stride_input_k_state_batch + pid_head * stride_input_k_state_head
            input_v_offset = (pid_batch + pid_seq) * stride_input_v_state_batch + pid_head * stride_input_v_state_head
            dinput_ssm_offset = (pid_batch + pid_seq) * stride_dinput_ssm_state_batch + pid_head * stride_dinput_ssm_state_head
            dinput_k_offset = (pid_batch + pid_seq) * stride_dinput_k_state_batch + pid_head * stride_dinput_k_state_head
            dinput_v_offset = (pid_batch + pid_seq) * stride_dinput_v_state_batch + pid_head * stride_dinput_v_state_head
            # Load DT_0 and Trap_0 (first token)
            dt_0 = tl.load(DT + dt_offset).to(tl.float32)
            trap_presig_0 = tl.load(Trap + trap_offset).to(tl.float32)
            trap_0 = sigmoid_approx(trap_presig_0)
            scalar = dt_0 * (1.0 - trap_0)

            # Dimension offsets.  The (HEADDIM_V x HEADDIM_QK) fp32 state tile is
            # processed in QK-blocks so the per-program Unified Buffer footprint
            # stays inside the 192 KB Ascend budget: at Dv/Dqk = 256/128 the fully
            # materialised tile needs 4206592 bits vs 1572864 available (bisheng
            # `ub overflow`).  ``BLOCK_QK`` is a compile-time tile, while the block
            # count is a RUNTIME value so the loop is not unrolled -- each unrolled
            # copy would reserve its own UB buffers (same remedy as the fwd kernel's
            # ``BLOCK_C`` sub-blocking, NPU_MIGRATION_mamba3_siso.md §9.2).  Blocking
            # only reorders the fp32 reductions (a pure floating-point reassociation;
            # the CPU probe measures the resulting deviation at <= ~5e-5, far below
            # the 1e-2 CANN aggregate limit).
            offs_v = tl.arange(0, HEADDIM_V)
            offs_qk_in = tl.arange(0, BLOCK_QK)
            num_qk_blocks = tl.cdiv(headdim_qk, BLOCK_QK)

            # Load Input_V_State (full V extent; only HEADDIM_V wide).
            input_v = tl.load(
                Input_V_State + input_v_offset + offs_v * stride_input_v_state_vdim,
                mask=offs_v < headdim_v,
                other=0.0
            ).to(tl.float32)

            # Accumulators carried across the QK-block loop.
            d_scalar_acc = tl.zeros((1,), tl.float32)
            dinput_v_acc = tl.zeros((HEADDIM_V,), tl.float32)

            for qb in range(num_qk_blocks):
                offs_qk = qb * BLOCK_QK + offs_qk_in
                qk_mask = offs_qk < headdim_qk
                vq_mask = (offs_v[:, None] < headdim_v) & qk_mask[None, :]

                # Load Input_K_State (this QK block).
                input_k = tl.load(
                    Input_K_State + input_k_offset + offs_qk * stride_input_k_state_qkdim,
                    mask=qk_mask,
                    other=0.0).to(tl.float32)

                # Load d_ISSM_State: (headdim_v, headdim_qk) block.
                d_issm = tl.load(
                    d_ISSM_State + d_issm_offset +
                    offs_v[:, None] * stride_d_issm_state_vdim +
                    offs_qk[None, :] * stride_d_issm_state_qkdim,
                    mask=vq_mask,
                    other=0.0
                ).to(tl.float32)

                # dInput_SSM_State = d_ISSM_State (direct copy), block by block.
                tl.store(
                    dInput_SSM_State + dinput_ssm_offset +
                    offs_v[:, None] * stride_dinput_ssm_state_vdim +
                    offs_qk[None, :] * stride_dinput_ssm_state_qkdim,
                    d_issm,
                    mask=vq_mask,
                )

                # d_scalar += sum(d_ISSM_State * outer(Input_V, Input_K))
                outer_product = input_v[:, None] * input_k[None, :]
                d_scalar_acc = d_scalar_acc + tl.sum(d_issm * outer_product)

                # dInput_V += sum_qk(d_ISSM_State * Input_K)  (scaled once at the end)
                dinput_v_acc = dinput_v_acc + tl.sum(d_issm * input_k[None, :], axis=1)

                # dInput_K = sum_v(d_ISSM_State * Input_V) * scalar (this QK block)
                dinput_k = tl.sum(d_issm * input_v[:, None], axis=0) * scalar
                tl.store(
                    dInput_K_State + dinput_k_offset + offs_qk * stride_dinput_k_state_qkdim,
                    dinput_k, mask=qk_mask)

            d_scalar = tl.sum(d_scalar_acc)
            dinput_v = dinput_v_acc * scalar

            # Store dInput_V_State.
            tl.store(dInput_V_State + dinput_v_offset + offs_v * stride_dinput_v_state_vdim, dinput_v, mask=offs_v < headdim_v)

            # Add contributions to dDT_0 and dTrap_0 from input state gradient
            ddt_0_contrib = d_scalar * (1.0 - trap_0)
            dtrap_0_contrib = d_scalar * (-dt_0)
            dtrap_0_presig_contrib = dtrap_0_contrib * trap_0 * (1.0 - trap_0)

            # Atomically add to the first position (already written in Part 1)
            tl.atomic_add(dDT + ddt_offset, ddt_0_contrib)
            tl.atomic_add(dTrap + dtrap_offset, dtrap_0_presig_contrib)


def compute_ddt_dtrap_dinput_states(
    dscale: torch.Tensor,
    dgamma: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    d_issm_state: Optional[torch.Tensor] = None,
    input_k_state: Optional[torch.Tensor] = None,
    input_v_state: Optional[torch.Tensor] = None,
    Cu_Seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute dDT, dTrap from dScale/dGamma, and optionally input state gradients.

    Args:
        dscale: Gradient of scale, shape (batch, nheads, seqlen)
        dgamma: Gradient of gamma, shape (batch, nheads, seqlen)
        dt: DT tensor from forward pass, shape (batch, nheads, seqlen)
        trap: Trap tensor from forward pass, shape (batch, nheads, seqlen)
        d_issm_state: Gradient of SSM_State_mid (optional), shape (batch, nheads, headdim_v, headdim_qk)
        input_k_state: Input K state from forward pass (optional), shape (batch, nheads, headdim_qk)
        input_v_state: Input V state from forward pass (optional), shape (batch, nheads, headdim_v)

    Returns:
        Tuple containing:
            - dDT: Gradient for DT, shape (batch, nheads, seqlen), fp32
            - dTrap: Gradient for Trap, shape (batch, nheads, seqlen), fp32
            - dInput_SSM_State: Gradient for Input_SSM_State (None if no input
              state); dtype follows ``d_issm_state.dtype`` -- it is written from
              an fp32 computation, so a low-precision dtype rounds on store
            - dInput_K_State: Gradient for Input_K_State, fp32 (None if no input
              state)
            - dInput_V_State: Gradient for Input_V_State, fp32 (None if no input
              state)

    Dtype contract:
        Part 1 accepts any floating dtype for ``dscale`` / ``dgamma`` / ``dt`` /
        ``trap`` and widens them to fp32 before the arithmetic when
        ``npu_compat.precise_ddt()`` is enabled (default on CANN); the outputs
        are always fp32.  Part 2 upcasts ``d_issm_state`` / ``input_k_state`` /
        ``input_v_state`` to fp32 as well, so those may be any floating dtype
        (the forward's ``Final_V_State`` is bf16).

    Part 2 scope:
        ``d_issm_state`` / ``input_k_state`` / ``input_v_state`` are an explicit
        caller-supplied path, not part of the SISO training chain: ``mamba3.py``
        forwards ``Input_States=None``, so ``mamba3_siso_combined`` runs with
        ``ctx.has_input_state=False`` and calls this with ``d_issm_state=None``;
        only Part 1 runs there.  Part 2 (including the ``BLOCK_QK`` QK-axis
        blocking for its state tile) runs only when input states are supplied
        explicitly (``mamba3_siso_combined(..., Input_States=...)`` or a direct
        call with all three input-state tensors).
    """
    batch, nheads, seqlen = dscale.shape
    has_input_state = d_issm_state is not None
    is_varlen = Cu_Seqlens is not None

    if is_varlen:
        num_sequences = Cu_Seqlens.shape[0] - 1
        assert batch == 1, "Batch size must be 1 when using variable-length sequences."
    else:
        num_sequences = batch

    # Validate inputs
    assert dgamma.shape == (batch, nheads, seqlen), f"dgamma shape mismatch: {dgamma.shape}"
    assert dt.shape == (batch, nheads, seqlen), f"dt shape mismatch: {dt.shape}"
    assert trap.shape == (batch, nheads, seqlen), f"trap shape mismatch: {trap.shape}"

    if has_input_state:
        assert input_k_state is not None and input_v_state is not None, \
            "input_k_state and input_v_state must be provided with d_issm_state"
        headdim_v, headdim_qk = d_issm_state.shape[2], d_issm_state.shape[3]
        assert d_issm_state.shape == (num_sequences, nheads, headdim_v, headdim_qk), \
            f"d_issm_state shape mismatch: {d_issm_state.shape}"
        assert input_k_state.shape == (num_sequences, nheads, headdim_qk), \
            f"input_k_state shape mismatch: {input_k_state.shape}"
        assert input_v_state.shape == (num_sequences, nheads, headdim_v), \
            f"input_v_state shape mismatch: {input_v_state.shape}"
    else:
        headdim_v, headdim_qk = 64, 128  # Dummy values for block size calculation

    # Ensure contiguity
    dscale = dscale.contiguous() if not dscale.is_contiguous() else dscale
    dgamma = dgamma.contiguous() if not dgamma.is_contiguous() else dgamma
    dt = dt.contiguous() if not dt.is_contiguous() else dt
    trap = trap.contiguous() if not trap.is_contiguous() else trap

    if has_input_state:
        d_issm_state = d_issm_state.contiguous() if not d_issm_state.is_contiguous() else d_issm_state
        input_k_state = input_k_state.contiguous() if not input_k_state.is_contiguous() else input_k_state
        input_v_state = input_v_state.contiguous() if not input_v_state.is_contiguous() else input_v_state

    # Allocate outputs
    dDT = torch.empty_like(dt, dtype=torch.float32)
    dTrap = torch.empty_like(trap, dtype=torch.float32)

    if has_input_state:
        d_Input_SSM_State = torch.empty_like(d_issm_state)
        d_Input_K_State = torch.empty((num_sequences, nheads, headdim_qk), dtype=torch.float32, device=dt.device)
        d_Input_V_State = torch.empty((num_sequences, nheads, headdim_v), dtype=torch.float32, device=dt.device)
    else:
        d_Input_SSM_State = None
        d_Input_K_State = None
        d_Input_V_State = None

    # Launch kernel
    HEADDIM_V = triton.next_power_of_2(headdim_v) if has_input_state else 64
    HEADDIM_QK = triton.next_power_of_2(headdim_qk) if has_input_state else 128

    # Grid: one tile per (head, batch/num_sequences) pair, flattened and capped
    # at the Ascend coreDim limit with the true logical tile count passed in.
    # Above the cap each program grid-strides over the remaining tiles.
    if seqlen == 0:
        # No tile to walk: a (0,) flat grid is not launchable and the kernel's
        # ``cdiv(num_tiles - pid, num_programs)`` bound would divide by zero.
        # Part 2 still has a defined value with zero tokens: no token
        # contribution exists, so ``dInput_SSM_State`` is the upstream cotangent
        # copied through and ``dInput_K/V_State`` are zero.  Leaving them as
        # ``torch.empty`` would hand the caller uninitialised memory, so fill
        # them here; ``dDT``/``dTrap`` are empty along seqlen and stay empty.
        if has_input_state:
            d_Input_SSM_State.copy_(d_issm_state)
            d_Input_K_State.zero_()
            d_Input_V_State.zero_()
        return dDT, dTrap, d_Input_SSM_State, d_Input_K_State, d_Input_V_State

    num_tiles = nheads * num_sequences
    num_programs = _grid_programs(num_tiles)
    _record_launch("ddt", num_tiles, num_programs)
    grid = (num_programs,)

    mamba3_siso_bwd_kernel_ddt_dtrap_dinput_states[grid](
        # Inputs
        dscale, dgamma, dt, trap,
        d_issm_state if has_input_state else dscale,  # Dummy pointer if not used
        input_k_state if has_input_state else dscale,
        input_v_state if has_input_state else dscale,
        Cu_Seqlens,
        # Outputs
        dDT, dTrap,
        d_Input_SSM_State if has_input_state else dDT,  # Dummy pointer if not used
        d_Input_K_State if has_input_state else dDT,
        d_Input_V_State if has_input_state else dDT,
        # Strides for dScale
        dscale.stride(0), dscale.stride(1), dscale.stride(2),
        # Strides for dSGamma
        dgamma.stride(0), dgamma.stride(1), dgamma.stride(2),
        # Strides for DT
        dt.stride(0), dt.stride(1), dt.stride(2),
        # Strides for Trap
        trap.stride(0), trap.stride(1), trap.stride(2),
        # Strides for d_ISSM_State
        d_issm_state.stride(0) if has_input_state else 0,
        d_issm_state.stride(1) if has_input_state else 0,
        d_issm_state.stride(2) if has_input_state else 0,
        d_issm_state.stride(3) if has_input_state else 0,
        # Strides for Input_K_State
        input_k_state.stride(0) if has_input_state else 0,
        input_k_state.stride(1) if has_input_state else 0,
        input_k_state.stride(2) if has_input_state else 0,
        # Strides for Input_V_State
        input_v_state.stride(0) if has_input_state else 0,
        input_v_state.stride(1) if has_input_state else 0,
        input_v_state.stride(2) if has_input_state else 0,
        # Stride for Cu_Seqlens
        Cu_Seqlens.stride(0) if Cu_Seqlens is not None else 0,
        # Strides for dDT
        dDT.stride(0), dDT.stride(1), dDT.stride(2),
        # Strides for dTrap
        dTrap.stride(0), dTrap.stride(1), dTrap.stride(2),
        # Strides for d_Input_SSM_State
        d_Input_SSM_State.stride(0) if has_input_state else 0,
        d_Input_SSM_State.stride(1) if has_input_state else 0,
        d_Input_SSM_State.stride(2) if has_input_state else 0,
        d_Input_SSM_State.stride(3) if has_input_state else 0,
        # Strides for d_Input_K_State
        d_Input_K_State.stride(0) if has_input_state else 0,
        d_Input_K_State.stride(1) if has_input_state else 0,
        d_Input_K_State.stride(2) if has_input_state else 0,
        # Strides for d_Input_V_State
        d_Input_V_State.stride(0) if has_input_state else 0,
        d_Input_V_State.stride(1) if has_input_state else 0,
        d_Input_V_State.stride(2) if has_input_state else 0,
        # Dimensions
        seqlen, nheads, headdim_v, headdim_qk,
        # Launch geometry
        num_tiles, num_programs,
        # Constants
        HEADDIM_V=HEADDIM_V,
        HEADDIM_QK=HEADDIM_QK,
        HAS_INPUT_STATE=has_input_state,
        IS_VARLEN=is_varlen,
        BLOCK_QK=NPU_DDT_QK_BLOCK,
        # PLUT-51 precision path: fp32 Part-1 operands (see npu_compat.precise_ddt).
        PRECISE_FP32=precise_ddt(),
    )

    return dDT, dTrap, d_Input_SSM_State, d_Input_K_State, d_Input_V_State


# Memory allocator for Triton runtime scratch (device agnostic: cuda/npu).
def _alloc_fn(size: int, alignment: int, stream: Optional[int]):
    """Custom allocator for Triton runtime scratch memory."""
    return torch.empty(size, device=alloc_device(), dtype=torch.int8)


set_allocator_safely(_alloc_fn)
