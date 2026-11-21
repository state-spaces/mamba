"""
Tilelang implementation of the Mamba3 forward kernel with MIMO support
and variable-length sequence (varlen) support.

This is the varlen counterpart of ``mamba3_mimo_fwd.py``.  The two files
share the same mathematical kernel; the differences are:

* **NS parameter** — number of packed sequences in the batch.  When NS > 1
  the kernel dispatches one thread-block per sequence (``i_ns``) in addition
  to the per-head (``i_h``) and per-batch (``i_b``) dimensions.
* **CU_SEQLENS tensor** — int32 prefix-sum array of shape ``[NS+1]``; entry
  ``i`` holds the token offset where sequence ``i`` begins in the packed
  ``[B, S, ...]`` tensors.
* **max_nchunks** — ``(S // chunk_size) + NS`` to accommodate the global
  chunk layout where every sequence boundary introduces an extra padding
  chunk slot.
* **SEGSUM shape** — ``[B, H, max_nchunks, C, C]`` (non-varlen uses
  ``[B, H, nchunks, C, C]`` with ``nchunks = S // chunk_size``).
* **FINAL_STATE / FINAL_K shapes** — ``(NS, H, N, P)`` / ``(NS, R, H, N)``
  when NS > 1, or ``(B, H, N, P)`` / ``(B, R, H, N)`` otherwise (matching
  the non-varlen shapes).

Public API:
    mamba_mimo_forward_varlen(..., cu_seqlens=None) — forward pass; falls back to
        the non-varlen path when cu_seqlens is None.

Copyright (c) 2026, Dao AI Lab, Goombalab
"""

import torch

from mamba_ssm.ops.cuda.mamba3.fwd import cuda_fwd
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from tilelang.autotuner import autotune

import itertools
import argparse
from typing import Optional, Tuple


# NOTE: Uncomment the following to autotune:
# def get_configs():
#     iter_params = dict(num_stages=[0, 1, 2, 3], threads=[128, 256, 512])
#     # iter_params = dict(num_stages=[2], threads=[128])
#     return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

# @autotune(
#     configs=get_configs(),
#     warmup=3,
#     rep=20,
# )
@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def mamba_mimo_fwd(
    B,
    H,
    G,
    N,
    P,
    R,
    hasZ,
    hasD,
    reduceO,
    fuse_pregate_headwise_rms_norm=False,
    isVarlen: bool = True,
    return_final_state=False,
    has_init_state: bool = False,
    state_only: bool = False,
    blocked: bool = False,
    has_initial_state=False,
    compute_outputs: bool = True,
    chunk_size: int = 16,
    rotary_dim_divisor = 4,
    dtype: str = 'bfloat16',
    outproj_norm_eps: float = 1e-5,
    threads: int = 128,
    num_stages: int = 0,
) -> torch.Tensor:
    """
    TileLang kernel factory for the varlen Mamba3 chunked forward pass.

    Varlen additions vs. ``mamba_mimo_fwd`` in
    ``mamba3_mimo_fwd.py``:

    * NS > 1 activates the per-sequence (``i_ns``) grid dimension and reads
      per-sequence token offsets from ``CU_SEQLENS``.
    * ``max_nchunks = (S // chunk_size) + NS`` accommodates the global chunk
      layout where each sequence boundary introduces one extra slot.
    * SEGSUM shape becomes ``[B, H, max_nchunks, C, C]``.
    * FINAL_STATE shape is ``(NS, H, N, P)`` (vs. ``(B, H, N, P)``).
    * FINAL_K shape is ``(NS, R, H, N)`` (vs. ``(B, R, H, N)``).

    Output modes match the dense kernel:

    * ``reduceO=False`` writes the full per-rank MIMO output
      ``raw_y`` with shape ``[B, S, R, H, P]``.
    * ``reduceO=True`` applies the MIMO output projection inside the kernel.
    * ``reduceO=True`` with ``fuse_pregate_headwise_rms_norm=True`` applies
      ``RMSNorm(raw_y) * silu(z_r)`` per valid token/rank/head before the
      MIMO output projection.  The RMS reduction is over headdim ``P``, not
      sequence lanes, so chunk-tail padding used for varlen execution does not
      enter the RMS mean.

    When NS == 1 (or cu_seqlens is None) the kernel degenerates to the
    non-varlen behaviour.
    """
    S = T.dynamic("S")
    NS = T.dynamic("NS")
    # Dynamic, NOT a Python int: an int is baked into the @tilelang.jit key,
    # and NBLK changes whenever a segment's last block rounds up differently
    # -- measured ~22 s of recompilation per distinct value, i.e. most steps.
    # Inferred from BLK_SEG's shape, the same way stock infers NS.
    NBLK = T.dynamic("NBLK")

    accum_dtype = 'float32'

    # Block sizes for K and V dimensions - use full dimensions (no tiling)
    # assert S % chunk_size == 0, "Sequence length must be divisible by chunk_size"

    if isVarlen:
        max_nchunks = (S//chunk_size) + NS
        Final_State_shape = (NBLK if blocked else NS, H, N, P)
        Final_K_shape = (NS, R, H, N)
        Init_SSM_State_shape = (NS, H, P, N)
        Init_K_State_shape = (NS, R, H, N)
        Init_V_State_shape = (NS, H, P)
    else:
        max_nchunks = tilelang.cdiv(S, chunk_size)
        Final_State_shape = (B, H, N, P)
        Final_K_shape = (B, R, H, N)
        Init_SSM_State_shape = (B, H, P, N)
        Init_K_State_shape = (B, R, H, N)
        Init_V_State_shape = (B, H, P)
    fused_chunk_size = chunk_size * R

    if reduceO:
        O_shape = (B, S, H, P)
    else:
        O_shape = (B, S, R, H, P)


    @T.prim_func
    def mamba_mimo_fwd_kernel(
            Q: T.Tensor([B, S, R, G, N], dtype),  # type: ignore
            K: T.Tensor([B, S, R, G, N], dtype),  # type: ignore
            V: T.Tensor([B, S, H, P], dtype),  # type: ignore
            O: T.Tensor(O_shape, dtype),  # type: ignore
            Q_BIAS: T.Tensor([H, R, N], T.float32),  # type: ignore
            K_BIAS: T.Tensor([H, R, N], T.float32),  # type: ignore
            MIMO_V: T.Tensor([H, R, P], T.float32), # type: ignore
            MIMO_O: T.Tensor([H, R, P], T.float32), # type: ignore
            OUT_NORM_WEIGHT: T.Tensor([H, P], T.float32), # type: ignore
            Z: T.Tensor([B, S, H, P], dtype),  # type: ignore
            D: T.Tensor([H], T.float32),  # type: ignore
            MIMO_Z: T.Tensor([H, R, P], T.float32), # type: ignore
            ANGLES: T.Tensor([B, S, H, N//rotary_dim_divisor], T.float32), # type: ignore
            DA_CS: T.Tensor([B, H, S], T.float32), # type: ignore
            DA_CS_REV: T.Tensor([B, H, S], T.float32), # type: ignore
            DT: T.Tensor([B, H, S], T.float32), # type: ignore
            TRAP: T.Tensor([B, H, S], dtype), # type: ignore
            SEGSUM: T.Tensor([B, H, max_nchunks, chunk_size, chunk_size], T.float32), # type: ignore
            CU_SEQLENS: T.Tensor([NS+1], dtype=T.int32), # type: ignore
            INIT_SSM_STATE: T.Tensor(Init_SSM_State_shape, T.float32),  # type: ignore
            INIT_K_STATE: T.Tensor(Init_K_State_shape, dtype),  # type: ignore
            INIT_V_STATE: T.Tensor(Init_V_State_shape, dtype),  # type: ignore

            FINAL_STATE: T.Tensor(Final_State_shape, T.float32),  # type: ignore
            FINAL_K: T.Tensor(Final_K_shape, dtype),  # type: ignore
            # Per-block initial recurrent state. One entry per block; with one
            # block per segment (steps 1-3) that is [NS, H, N, P]. Unused when
            # has_init_state=False -- TileLang accepts None for a tensor
            # argument that compile-time flags make unreachable.
            BLK_SEG: T.Tensor([NBLK if blocked else 1], dtype=T.int32),  # type: ignore
            BLK_C0: T.Tensor([NBLK if blocked else 1], dtype=T.int32),  # type: ignore
            BLK_NCH: T.Tensor([NBLK if blocked else 1], dtype=T.int32),  # type: ignore
            INIT_STATE: T.Tensor([NBLK if blocked else NS, H, N, P], T.float32)  # type: ignore
            ):
        """
        Overview:
            Varlen fused chunked forward pass.  Identical in math to the dense
            kernel in ``mamba3_mimo_fwd.py``; differs in the grid and
            sequence-bounds logic:

            * Grid: ``T.Kernel(H, NS, B)`` — extra ``i_ns`` dimension iterates
              over packed sequences.
            * Per-sequence bounds are read from ``CU_SEQLENS[i_ns]`` /
              ``CU_SEQLENS[i_ns+1]``; the loop only visits chunks that belong to
              sequence ``i_ns``.
            * SEGSUM is indexed by the *global* chunk index
              ``start_chunk_ind + i`` where
              ``start_chunk_ind = (CU_SEQLENS[i_ns] // chunk_size) + i_ns``.

            Output handling is the same as the dense MIMO kernel:
            - ``reduceO=False`` writes per-rank ``raw_y`` to
              ``[B, S, R, H, P]``.
            - ``reduceO=True`` applies ``MIMO_O`` in-kernel and writes
              ``[B, S, H, P]``.
            - ``reduceO=True`` and ``fuse_pregate_headwise_rms_norm=True``
              applies ``RMSNorm(raw_y) * silu(z_r)`` in-kernel before
              ``MIMO_O``.  The RMS mean is over ``P`` for each valid token,
              rank, and head; varlen tail padding is only used to execute
              full chunks and does not contribute sequence elements to the
              RMS reduction.

        Inputs:
            - Activations: Q, K, V.
            - Projection parameters/biases: MIMO_V (Psi), MIMO_O (Phi),
              optional MIMO_Z (Zeta), optional OUT_NORM_WEIGHT, ANGLES, and
              Q_BIAS/K_BIAS.
            - Optional modifiers: Z and D.
            - Discretization tensors: DA_CS, DA_CS_REV, DT, TRAP, and SEGSUM.
            - CU_SEQLENS: int32 prefix-sum of per-sequence lengths, shape
              ``[NS+1]``.

        Outputs:
            - O: forward output activations. Shape is ``[B, S, H, P]`` when
              ``reduceO`` is true, otherwise ``[B, S, R, H, P]``.
            - FINAL_STATE: final recurrent states per sequence (shape
              ``[NS, H, N, P]`` when NS > 1; written only when
              return_final_state is True).
            - FINAL_K: final K per sequence (shape ``[NS, R, H, N]`` when
              NS > 1; written only when return_final_state is True).

        Notation:
            - Psi: MIMO X projection (MIMO_V).
            - Phi: MIMO O projection (MIMO_O).
            - Zeta: MIMO Z projection (MIMO_Z).
            - Trap: convex-combination modulator used in
              exponential-trapezoidal discretization.
        """

        with T.Kernel(H, NBLK if blocked else NS, B, threads=threads) as (i_h, i_blk, i_b):
            # --- Kernel Setup ---
            # GQA support: map V head to Q/K head
            i_h_qk = i_h // (H // G)

            # --- Buffer Allocation ---
            # The state-only variant (compute_outputs=False, used as pass 1 of
            # the group-parallel prefill) needs no Q side and no output
            # staging, which also shrinks its shared-memory footprint.
            k_shared = T.alloc_shared([fused_chunk_size, N], dtype)
            k_bias_frag = T.alloc_fragment([R, N], dtype)

            angles_shared = T.alloc_shared([chunk_size, N], dtype)

            PsiV_shared = T.alloc_shared([fused_chunk_size, P], dtype)
            v_shared = T.alloc_shared([chunk_size, P], dtype)

            if compute_outputs:
                q_shared = T.alloc_shared([fused_chunk_size, N], dtype)
                q_bias_frag = T.alloc_fragment([R, N], dtype)
                qs_shared = T.alloc_shared([fused_chunk_size, P], dtype)
                o_shared = T.alloc_shared([chunk_size, P], dtype)
                states_accum_cast_shared = T.alloc_shared([N, P], dtype)
                qk_intrachunk_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], dtype)
                diag_scale_shared = T.alloc_shared([chunk_size], T.float32)
                da_cs_shared = T.alloc_shared([chunk_size], T.float32)
                decay_shared = T.alloc_shared([chunk_size, chunk_size], T.float32)

                # --- Swizzling Annotation ---
                T.annotate_layout({
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    v_shared: tilelang.layout.make_swizzled_layout(v_shared),

                    angles_shared: tilelang.layout.make_swizzled_layout(angles_shared),

                    PsiV_shared: tilelang.layout.make_swizzled_layout(PsiV_shared),
                    qs_shared: tilelang.layout.make_swizzled_layout(qs_shared),
                    o_shared: tilelang.layout.make_swizzled_layout(o_shared),
                    states_accum_cast_shared: tilelang.layout.make_swizzled_layout(states_accum_cast_shared),
                    qk_intrachunk_shared: tilelang.layout.make_swizzled_layout(qk_intrachunk_shared),
                })
            else:
                T.annotate_layout({
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    v_shared: tilelang.layout.make_swizzled_layout(v_shared),

                    angles_shared: tilelang.layout.make_swizzled_layout(angles_shared),

                    PsiV_shared: tilelang.layout.make_swizzled_layout(PsiV_shared),
                })
            T.use_swizzle(10, "row")

            T.no_set_max_nreg()

            # --- Per-Head Constants / Running State ---
            states_frag = T.alloc_fragment([N, P], accum_dtype)
            # i_blk aliases i_ns while there is one block per segment; step 4
            # gives it a grid of its own.
            # i_blk is the grid's block index; which SEGMENT it belongs to is a
            # table lookup, so every existing `i_ns` use (CU_SEQLENS bounds,
            # FINAL_K, the gi == full_nchunks-1 tests) keeps reading
            # SEGMENT-relative data unchanged -- which is the whole trap.
            # Inline tensor expressions, not T.alloc_var: same pattern the stock
            # kernel already uses for `full_nchunks`, itself a global read.
            i_ns = BLK_SEG[i_blk] if blocked else i_blk
            if has_init_state:
                T.copy(INIT_STATE[i_blk, i_h, :, :], states_frag)
            else:
                T.clear(states_frag)

            phi_frag_intrachunk = T.alloc_fragment([R, P], dtype=dtype)
            if reduceO:
                T.copy(MIMO_O[i_h, :, :], phi_frag_intrachunk)
            Psi_frag = T.alloc_fragment([R, P], dtype)
            T.copy(MIMO_V[i_h, :, :], Psi_frag)

            if compute_outputs:
                T.copy(Q_BIAS[i_h, :, :], q_bias_frag)
            T.copy(K_BIAS[i_h, :, :], k_bias_frag)

            # Determine the current sequence length for variable-length support.
            # Use alloc_var for mutable scalar values to avoid immutable rebind warnings.
            start_seq_ind = T.alloc_var(T.int32)
            start_chunk_ind = T.alloc_var(T.int32)
            seq_len = T.alloc_var(T.int32)
            seq_end = T.alloc_var(T.int32)
            full_nchunks = T.alloc_var(T.int32)
            tail_len = T.alloc_var(T.int32)
            if isVarlen:
                start_seq_ind = CU_SEQLENS[i_ns]
                start_chunk_ind = (start_seq_ind // chunk_size) + i_ns
                seq_len = CU_SEQLENS[i_ns + 1] - CU_SEQLENS[i_ns]
                seq_end = start_seq_ind + seq_len
                full_nchunks = seq_len // chunk_size
                tail_len = seq_len % chunk_size
            else:
                start_seq_ind = 0
                start_chunk_ind = 0
                seq_len = S
                seq_end = S
                full_nchunks = S // chunk_size
                tail_len = S % chunk_size
            if tail_len > 0:
                full_nchunks += 1

            # blk_c0 = this block's first chunk (segment-relative); blk_nch =
            # how many it owns. Whole segment until step 4 splits the grid.
            if blocked:
                blk_c0 = BLK_C0[i_blk]
                blk_nch = BLK_NCH[i_blk]
            else:
                blk_c0 = 0
                blk_nch = full_nchunks

            if has_initial_state and blk_c0 == 0:
                boundary_scale = T.alloc_var(T.float32)
                boundary_trap = T.alloc_var(dtype)
                T.copy(DT[i_b, i_h, start_seq_ind], boundary_scale)
                T.copy(TRAP[i_b, i_h, start_seq_ind], boundary_trap)
                boundary_scale *= T.sigmoid(-boundary_trap)
                if isVarlen:
                    for n, p in T.Parallel(N, P):
                        states_frag[n, p] = INIT_SSM_STATE[i_ns, i_h, p, n]
                        for r in T.serial(R):
                            states_frag[n, p] += (
                                INIT_K_STATE[i_ns, r, i_h, n]
                                * INIT_V_STATE[i_ns, i_h, p]
                                * MIMO_V[i_h, r, p]
                                * boundary_scale
                            )
                else:
                    for n, p in T.Parallel(N, P):
                        states_frag[n, p] = INIT_SSM_STATE[i_b, i_h, p, n]
                        for r in T.serial(R):
                            states_frag[n, p] += (
                                INIT_K_STATE[i_b, r, i_h, n]
                                * INIT_V_STATE[i_b, i_h, p]
                                * MIMO_V[i_h, r, p]
                                * boundary_scale
                            )

            # --- Chunk Loop ---
            for i in T.Pipelined(0, blk_nch, num_stages=num_stages):
                # gi = chunk index within the SEGMENT; the loop variable is
                # block-local. EVERY derived index must stay segment-relative --
                # including the SEGSUM lookup and the `gi == full_nchunks - 1`
                # tail tests, which read segment bounds.
                gi = blk_c0 + i
                chunk_start = start_seq_ind + gi * chunk_size

                # --- Discretization Factors (Shifted Gamma + Trap Scale) ---
                trap_shifted_frag = T.alloc_fragment([chunk_size], T.float32)
                T.copy(TRAP[i_b, i_h, chunk_start+1: chunk_start+chunk_size+1], trap_shifted_frag)
                dt_shifted_frag = T.alloc_fragment([chunk_size], dtype)
                if gi == full_nchunks - 1:
                    for cs in T.Parallel(chunk_size):
                        dt_shifted_frag[cs] = T.if_then_else(chunk_start + cs + 1 < seq_end, DT[i_b, i_h, chunk_start + 1 + cs], 0.0)
                else:
                    T.copy(DT[i_b, i_h, chunk_start+1: chunk_start+chunk_size+1], dt_shifted_frag)
                shifted_gamma_frag = T.alloc_fragment([chunk_size], dtype)
                for cs in T.Parallel(chunk_size):
                    shifted_gamma_frag[cs] = T.if_then_else(chunk_start + cs < (seq_end - 1),
                                                            dt_shifted_frag[cs] * T.sigmoid(-trap_shifted_frag[cs]),
                                                            0.0)
                shifted_gamma_shared = T.alloc_shared([chunk_size], dtype)
                T.copy(shifted_gamma_frag, shifted_gamma_shared)

                trap_frag = T.alloc_fragment([chunk_size], T.float32)
                T.copy(TRAP[i_b, i_h, chunk_start: chunk_start+chunk_size], trap_frag)
                dt_frag = T.alloc_fragment([chunk_size], dtype)
                T.copy(DT[i_b, i_h, chunk_start: chunk_start+chunk_size], dt_frag)
                gamma_frag = T.alloc_fragment([chunk_size], T.float32)
                for cs in T.Parallel(chunk_size):
                    gamma_frag[cs] = dt_frag[cs] * T.sigmoid(trap_frag[cs])
                trap_scale_frag = T.alloc_fragment([chunk_size], dtype)
                for cs in T.Parallel(chunk_size):
                    trap_scale_frag[cs] = gamma_frag[cs] + shifted_gamma_shared[cs]
                trap_scale_shared = T.alloc_shared([chunk_size], dtype)
                T.copy(trap_scale_frag, trap_scale_shared)

                # --- Up-Project V and Prepare Biased Q/K ---
                PsiV_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                for cs, p in T.Parallel(chunk_size, P):
                    v_shared[cs, p] = V[i_b, chunk_start+cs, i_h, p]
                for cs, r, p in T.Parallel(chunk_size, R, P):
                    PsiV_frag[cs, r, p] = v_shared[cs, p] * Psi_frag[r, p]
                PsiV_reshaped_frag = T.view(PsiV_frag, shape=[fused_chunk_size, P])
                T.copy(PsiV_reshaped_frag, PsiV_shared)

                if compute_outputs and not state_only:
                    q_frag = T.alloc_fragment([chunk_size, R, N], dtype)
                    T.copy(Q[i_b, chunk_start:chunk_start+chunk_size, :, i_h_qk, :], q_frag)
                    for cs, r, n in T.Parallel(chunk_size, R, N):
                        q_frag[cs, r, n] += q_bias_frag[r, n]
                    T.copy(T.view(q_frag, shape=[fused_chunk_size, N]), q_shared)

                k_frag = T.alloc_fragment([chunk_size, R, N], dtype)
                T.copy(K[i_b, chunk_start:chunk_start+chunk_size, :, i_h_qk, :], k_frag)
                for cs, r, n in T.Parallel(chunk_size, R, N):
                    k_frag[cs, r, n] += k_bias_frag[r, n]
                T.copy(T.view(k_frag, shape=[fused_chunk_size, N]), k_shared)

                if compute_outputs and not state_only:  # state-only passes skip: rotary-Q half split
                    # The same-token R x R (diagonal) block of the output is taken from the intrachunk
                    # GEMM below, q . (k * ts) scaled by gamma / ts, so no separate q . k^T GEMM is needed.
                    # --- Rotary Q + Interchunk Contribution ---
                    q_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                    q_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)

                    for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                        q_first_half_frag[cs, r, n] = q_shared[cs*R + r, n]
                        q_second_half_frag[cs, r, n] = q_shared[cs*R + r, N//2 + n]

                # NOTE: angles are casted to fp32 for numerical stability
                # (loaded unconditionally: rotary K feeds the state update)
                angles_frag = T.alloc_fragment([chunk_size, N//rotary_dim_divisor], T.float32)
                T.copy(ANGLES[i_b, chunk_start:chunk_start+chunk_size, i_h, :], angles_frag)
                if compute_outputs and not state_only:  # state-only passes skip: rotary-Q apply + interchunk GEMM

                    for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                        q_shared[cs*R + r, n] = T.cos(angles_frag[cs, n]) * q_first_half_frag[cs, r, n] - T.sin(angles_frag[cs, n]) * q_second_half_frag[cs, r, n]
                        q_shared[cs*R + r, N//2 + n] = T.sin(angles_frag[cs, n]) * q_first_half_frag[cs, r, n] + T.cos(angles_frag[cs, n]) * q_second_half_frag[cs, r, n]

                    o_mimo_accum_frag = T.alloc_fragment([fused_chunk_size, P], dtype=accum_dtype)
                    T.copy(states_frag, states_accum_cast_shared)
                    T.gemm(q_shared, states_accum_cast_shared, o_mimo_accum_frag, clear_accum=True)

                # --- Rotary K + Trap Scaling + Intrachunk Contribution ---
                k_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                k_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)

                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    k_first_half_frag[cs, r, n] = k_shared[cs*R + r, n]
                    k_second_half_frag[cs, r, n] = k_shared[cs*R + r, N//2 + n]

                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    k_shared[cs*R + r, n] = T.cos(angles_frag[cs, n]) * k_first_half_frag[cs, r, n] - T.sin(angles_frag[cs, n]) * k_second_half_frag[cs, r, n]
                    k_shared[cs*R + r, N//2 + n] = T.sin(angles_frag[cs, n]) * k_first_half_frag[cs, r, n] + T.cos(angles_frag[cs, n]) * k_second_half_frag[cs, r, n]

                if return_final_state and gi == full_nchunks - 1:
                    seq_boundary = seq_end - chunk_start
                    if isVarlen:
                        for csr, n in T.Parallel(fused_chunk_size, N):
                            if csr >= (seq_boundary - 1) * R and csr < seq_boundary * R:  # Only copy the last chunk's R rows to FINAL_K
                                FINAL_K[i_ns, csr % R, i_h, n] = k_shared[csr, n]
                    else:
                        for csr, n in T.Parallel(fused_chunk_size, N):
                            if csr >= (seq_boundary - 1) * R and csr < seq_boundary * R:  # Only copy the last chunk's R rows to FINAL_K
                                FINAL_K[i_b, csr % R, i_h, n] = k_shared[csr, n]

                k_trap_scaled_frag = T.alloc_fragment([fused_chunk_size, N], dtype)
                T.copy(k_shared, k_trap_scaled_frag)
                for csr, n in T.Parallel(fused_chunk_size, N):
                    k_trap_scaled_frag[csr, n] *= trap_scale_shared[csr//R]
                T.copy(k_trap_scaled_frag, k_shared)

                if compute_outputs and not state_only:
                    qk_intrachunk_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=accum_dtype)
                    T.gemm(q_shared, k_shared, qk_intrachunk_frag, transpose_B=True, clear_accum=True)

                    exp_da_cs_frag = T.alloc_fragment([chunk_size], dtype=T.float32)
                    T.copy(DA_CS[i_b, i_h, chunk_start:chunk_start+chunk_size], exp_da_cs_frag)
                    T.copy(exp_da_cs_frag, da_cs_shared)
                    # Segment decay exp(da_cs[a] - da_cs[b]) of the strictly causal part (SEGSUM is no longer read):
                    # the decay is first built as a [chunk_size, chunk_size] matrix and the mask reads it once per entry,
                    # like SEGSUM: indexing the 1-D da_cs at two different steps (csr_i//R, csr_j//R) inside this
                    # T.Parallel(fused_chunk_size, fused_chunk_size) loop compiles to wrong values (18 % rel. L2 vs fp64).
                    decay_frag = T.alloc_fragment([chunk_size, chunk_size], T.float32)
                    for a, b in T.Parallel(chunk_size, chunk_size):
                        decay_frag[a, b] = T.exp(T.min(da_cs_shared[a] - da_cs_shared[b], 0.0))
                    T.copy(decay_frag, decay_shared)
                    diag_frag = T.alloc_fragment([chunk_size], dtype=T.float32)
                    for cs in T.Parallel(chunk_size):
                        diag_frag[cs] = T.if_then_else(trap_scale_shared[cs] > 0.0,
                                                       gamma_frag[cs] / trap_scale_shared[cs], 0.0)
                    T.copy(diag_frag, diag_scale_shared)
                    # Causal mask: earlier steps get the segment decay (decay_shared, from DA_CS), the same
                    # step gets gamma / ts (the diagonal term), later steps 0.
                    qk_intrachunk_masked_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=dtype)
                    for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size):
                        qk_intrachunk_masked_frag[csr_i, csr_j] = T.if_then_else(
                            csr_i//R > csr_j//R,
                            qk_intrachunk_frag[csr_i, csr_j]
                            * decay_shared[csr_i//R, csr_j//R],
                            T.if_then_else(csr_i//R == csr_j//R,
                                           qk_intrachunk_frag[csr_i, csr_j] * diag_scale_shared[csr_i//R], 0.0))

                    for cs in T.Parallel(chunk_size):
                        exp_da_cs_frag[cs] = T.exp(exp_da_cs_frag[cs])
                    for csr, p in T.Parallel(fused_chunk_size, P):
                        o_mimo_accum_frag[csr, p] *= exp_da_cs_frag[csr//R]

                    # NOTE: if we gemm with qk_intrachunk_masked_frag the compiler will
                    # error with layout issue if threads != 128:
                    # Copy via shared memory to satisfy layout constraints before GEMM.
                    T.copy(qk_intrachunk_masked_frag, qk_intrachunk_shared)
                    # Adding the two intermediate outputs together (interchunk += intrachunk)
                    T.gemm(qk_intrachunk_shared, PsiV_shared, o_mimo_accum_frag, clear_accum=False)

                    # --- D skip (the diagonal qk term is already in the intrachunk GEMM) ---
                    if hasD:
                        D_var = T.alloc_var(T.float32)
                        T.copy(D[i_h], D_var)
                        for csr, p in T.Parallel(fused_chunk_size, P):
                            o_mimo_accum_frag[csr, p] += D_var * PsiV_shared[csr, p]

                    # --- Optional Z Gating + Down-Projection ---
                    if reduceO:
                        if fuse_pregate_headwise_rms_norm:
                            o_sq_frag = T.alloc_fragment([chunk_size, R, P], T.float32)
                            for cs, r, p in T.Parallel(chunk_size, R, P):
                                o_sq_frag[cs, r, p] = (
                                    o_mimo_accum_frag[cs * R + r, p]
                                    * o_mimo_accum_frag[cs * R + r, p]
                                )
                            o_rstd_frag = T.alloc_fragment([chunk_size, R], T.float32)
                            T.reduce_sum(o_sq_frag, o_rstd_frag, dim=-1, clear=True)
                            for cs, r in T.Parallel(chunk_size, R):
                                o_rstd_frag[cs, r] = 1.0 / T.sqrt(
                                    o_rstd_frag[cs, r] / P + outproj_norm_eps
                                )
                            z_frag = T.alloc_fragment([chunk_size, P], dtype)
                            T.copy(Z[i_b, chunk_start:chunk_start+chunk_size, i_h, :], z_frag)
                            z_expanded_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                            for cs, r, p in T.Parallel(chunk_size, R, P):
                                # Apply SiLU to z_expanded_frag[cs, r, p]:
                                o_gated = z_frag[cs, p] * MIMO_Z[i_h, r, p] * 0.5
                                z_expanded_frag[cs, r, p] = o_gated * T.tanh(o_gated) + o_gated
                        elif hasZ:
                            z_frag = T.alloc_fragment([chunk_size, P], dtype)
                            T.copy(Z[i_b, chunk_start:chunk_start+chunk_size, i_h, :], z_frag)
                            z_expanded_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                            for cs, r, p in T.Parallel(chunk_size, R, P):
                                # Apply SiLU to z_expanded_frag[cs, r, p]:
                                o_gated = z_frag[cs, p] * MIMO_Z[i_h, r, p] * 0.5
                                z_expanded_frag[cs, r, p] = o_gated * T.tanh(o_gated) + o_gated

                        lqk_PsiV_reshaped_frag = T.view(o_mimo_accum_frag, shape=[chunk_size, R, P])
                        if fuse_pregate_headwise_rms_norm:
                            for cs, r, p in T.Parallel(chunk_size, R, P):
                                lqk_PsiV_reshaped_frag[cs, r, p] *= (
                                    o_rstd_frag[cs, r]
                                    * OUT_NORM_WEIGHT[i_h, p]
                                    * phi_frag_intrachunk[r, p]
                                    * z_expanded_frag[cs, r, p]
                                )
                        elif hasZ:
                            for cs, r, p in T.Parallel(chunk_size, R, P):
                                lqk_PsiV_reshaped_frag[cs, r, p] *= phi_frag_intrachunk[r, p] * z_expanded_frag[cs, r, p]
                        else:
                            for cs, r, p in T.Parallel(chunk_size, R, P):
                                lqk_PsiV_reshaped_frag[cs, r, p] *= phi_frag_intrachunk[r, p]
                        lqk_PsiV_reshaped_shared = T.alloc_shared([chunk_size, R, P], dtype)
                        T.copy(lqk_PsiV_reshaped_frag, lqk_PsiV_reshaped_shared)
                        o_frag = T.alloc_fragment([chunk_size, P], dtype)
                        T.clear(o_frag)
                        for r in T.serial(R):
                            for cs, p in T.Parallel(chunk_size, P):
                                o_frag[cs, p] += lqk_PsiV_reshaped_shared[cs, r, p]
                        if gi == (full_nchunks - 1) and tail_len > 0:
                            for cs, p in T.Parallel(chunk_size, P):
                                if cs < tail_len:
                                    O[i_b, chunk_start+cs, i_h, p] = o_frag[cs, p]
                        else:
                            T.copy(o_frag, O[i_b, chunk_start:chunk_start+chunk_size, i_h, :])
                    else:
                        if hasZ:
                            z_frag = T.alloc_fragment([chunk_size, P], dtype)
                            T.copy(Z[i_b, chunk_start:chunk_start+chunk_size, i_h, :], z_frag)
                            z_expanded_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                            for cs, r, p in T.Parallel(chunk_size, R, P):
                                # Apply SiLU to z_expanded_frag[cs, r, p]:
                                o_gated = z_frag[cs, p] * MIMO_Z[i_h, r, p] * 0.5
                                z_expanded_frag[cs, r, p] = o_gated * T.tanh(o_gated) + o_gated
                            lqk_PsiV_reshaped_shared = T.alloc_shared([chunk_size, R, P], dtype)
                            for cs, r, p in T.Parallel(chunk_size, R, P):
                                lqk_PsiV_reshaped_shared[cs, r, p] = o_mimo_accum_frag[cs* R + r, p] * z_expanded_frag[cs, r, p]
                            # T.copy(lqk_PsiV_frag, lqk_PsiV_reshaped_shared)
                            # for cs, r, p in T.Parallel(chunk_size, R, P):
                            #     lqk_PsiV_reshaped_shared[cs, r, p] *= z_expanded_frag[cs, r, p]
                        else:
                            lqk_PsiV_reshaped_shared = T.alloc_shared([chunk_size, R, P], dtype)
                            # T.copy(lqk_PsiV_reshaped_frag, lqk_PsiV_reshaped_shared)
                            for cs, r, p in T.Parallel(chunk_size, R, P):
                                lqk_PsiV_reshaped_shared[cs, r, p] = o_mimo_accum_frag[cs* R + r, p]
                        if gi == (full_nchunks - 1) and tail_len > 0:
                            for cs, r, p in T.Parallel(chunk_size, R, P):
                                if cs < tail_len:
                                    O[i_b, chunk_start+cs, r, i_h, p] = lqk_PsiV_reshaped_shared[cs, r, p]
                        else:
                            T.copy(lqk_PsiV_reshaped_shared, O[i_b, chunk_start:chunk_start+chunk_size, :, i_h, :])

                # --- Recurrent State Update ---
                # DA_CS_REV scales per-step K contributions for state accumulation.
                dA_cs_rev_frag = T.alloc_fragment([chunk_size], T.float32)
                T.copy(DA_CS_REV[i_b, i_h, chunk_start:chunk_start+chunk_size], dA_cs_rev_frag)

                k_state_frag = T.alloc_fragment([fused_chunk_size, N], dtype)
                T.copy(k_shared, k_state_frag)
                for csr, n in T.Parallel(fused_chunk_size, N):
                    k_state_frag[csr, n] *= T.exp(dA_cs_rev_frag[csr//R])

                # DA_CS(last) applies the chunk-level decay to the carried state.
                da_cs_sum = T.alloc_var(T.float32)
                if return_final_state and gi == (full_nchunks - 1) and tail_len > 0:
                    T.copy(DA_CS[i_b, i_h, seq_end - 1], da_cs_sum)
                    for csr, n in T.Parallel(fused_chunk_size, N):
                        k_state_frag[csr, n] = T.if_then_else(csr < tail_len * R, k_state_frag[csr, n], 0.0)
                else:
                    T.copy(DA_CS[i_b, i_h, chunk_start+chunk_size-1], da_cs_sum)

                for n, p in T.Parallel(N, P):
                    states_frag[n, p] *= T.exp(da_cs_sum)
                T.gemm(k_state_frag, PsiV_shared, states_frag, transpose_A=True, clear_accum=False)

            # --- Save Last State (if applicable) ---
            if return_final_state:
                if isVarlen:
                    T.copy(states_frag, FINAL_STATE[i_blk, i_h, :, :])
                else:
                    T.copy(states_frag, FINAL_STATE[i_b, i_h, :, :])


    return mamba_mimo_fwd_kernel


def mamba_mimo_forward_varlen(q, k, v,
                       q_bias, k_bias,
                       mimo_v, mimo_o,
                       z, D,
                       mimo_z,
                       angles,
                       dA_cs,
                       dA_cs_rev,
                       dt,
                       trap,
                       segsum,
                       chunk_size, rotary_dim_divisor, dtype,
                       cu_seqlens=None,
                       return_state=False,
                       init_state=None,
                       state_only=False,
                       blk_seg=None, blk_c0=None, blk_nch=None,
                       initial_states=None,
                       fuse_pregate_headwise_rms_norm=False,
                       outproj_norm_weight=None,
                       outproj_norm_eps=1e-5,
                       threads=128,
                       num_stages=0,
                       compute_outputs=True):
    """
    Varlen wrapper around ``mamba_mimo_fwd``.

    Dispatches to the varlen kernel when ``cu_seqlens`` is provided;
    otherwise behaves identically to the non-varlen version in
    ``mamba3_mimo_fwd.py``.

    Args:
        q, k, v:       Packed activations, shapes ``[B, S, R, G, N]`` /
                       ``[B, S, H, P]``.
        q_bias, k_bias: Per-head per-R biases, shape ``[H, R, N]``.
        mimo_v:        Psi (MIMO X) projection, shape ``[H, R, P]``.
        mimo_o:        Phi (MIMO O) projection, shape ``[H, R, P]``, or
                       None to disable output reduction.
        z:             Optional gating tensor, shape ``[B, S, H, P]``.
        D:             Optional skip-connection vector, shape ``[H]``.
        mimo_z:        Optional Zeta (MIMO Z) projection, shape ``[H, R, P]``.
        angles:        Rotary angles, shape ``[B, S, H, N // rotary_dim_divisor]``.
        dA_cs:         Forward cumsum of discretised A, shape ``[B, H, S]``.
        dA_cs_rev:     Reverse cumsum of discretised A, shape ``[B, H, S]``.
        dt:            Discretised time step, shape ``[B, H, S]``.
        trap:          Pre-sigmoid trapezoidal modulator, shape ``[B, H, S]``.
        segsum:        Per-chunk lower-triangular log-decay matrix, shape
                       ``[B, H, max_nchunks, C, C]``.
        chunk_size:    Tokens per chunk (C).
        rotary_dim_divisor: Divisor for the rotary embedding head dimension.
        dtype:         Compute dtype (torch.dtype or string).
        cu_seqlens:    Optional int32 tensor of shape ``[NS+1]`` with
                       per-sequence token offsets.  None → dense mode.
        return_state:  If True, also return the final recurrent state and K.
        fuse_pregate_headwise_rms_norm: If True, fuse headwise
                       ``RMSNorm(raw_y) * silu(z_r)`` with MIMO down
                       projection in the forward kernel.  Requires ``mimo_o``,
                       ``z``, and ``mimo_z``.  Uses ``outproj_norm_weight`` when
                       supplied, otherwise a unit RMSNorm weight.  The RMS
                       reduction is over ``P`` and is unaffected by varlen
                       chunk-tail padding.
        outproj_norm_weight: Optional RMSNorm weight, shaped ``[H, P]`` or
                       flattened to ``[H * P]``.
        outproj_norm_eps: Epsilon for the fused RMSNorm path.
        threads:       Number of GPU threads per thread-block.
        num_stages:    Software-pipeline stages (0 = auto).

    Returns:
        (o, h, k_final) where:
            o        — output activations, shape ``[B, S, H, P]`` (reduceO)
                       or ``[B, S, R, H, P]``.
            h        — final hidden state per sequence, shape
                       ``[NS, H, N, P]`` when NS > 1, else ``[B, H, N, P]``,
                       or None if return_state is False.
            k_final  — final K per sequence, shape ``[NS, R, H, N]`` when
                       NS > 1, else ``[B, R, H, N]``, or None if
                       return_state is False.
    """
    B, S, R, G, N = q.shape
    H, P = v.shape[-2], v.shape[-1]
    if isinstance(dtype, torch.dtype):
        tl_dtype = str(dtype).replace("torch.", "")
    else:
        tl_dtype = dtype

    if cu_seqlens is not None:
        # assert B == 1, "cu_seqlens support not implemented for B > 1"
        NS = cu_seqlens.shape[0] - 1
    else:
        NS = 1

    reduceO = mimo_o is not None
    if fuse_pregate_headwise_rms_norm:
        if not reduceO:
            raise ValueError("fuse_pregate_headwise_rms_norm=True requires mimo_o.")
        if z is None or mimo_z is None:
            raise ValueError("fuse_pregate_headwise_rms_norm=True requires z and mimo_z.")
    # Hand-written CUDA forward (sm_90, Olala configuration, M3_CUDA_FWD=1); None -> the TileLang kernel below.
    cuda_run = cuda_fwd(B, H, G, N, P, R, z is not None, D is not None, reduceO, fuse_pregate_headwise_rms_norm,
                        isVarlen=cu_seqlens is not None, chunk_size=chunk_size,
                        rotary_dim_divisor=rotary_dim_divisor, dtype=tl_dtype,
                        has_initial_state=initial_states is not None, return_state=return_state,
                        state_only=state_only) if compute_outputs or state_only else None
    if cuda_run is not None:
        return cuda_run(q, k, v, q_bias, k_bias, mimo_v, mimo_o, z, D, mimo_z, angles, dA_cs, dA_cs_rev, dt, trap,
                        cu_seqlens, blk_seg, blk_c0, blk_nch, init_state)
    kernel = mamba_mimo_fwd(B, H, G,
                                       N, P, R,
                                       z is not None,
                                       D is not None,
                                       reduceO,
                                       fuse_pregate_headwise_rms_norm,
                                       isVarlen=cu_seqlens is not None,
                                       return_final_state=return_state,
                                       has_init_state=init_state is not None,
                                       state_only=state_only,
                                       blocked=blk_seg is not None,
                                       has_initial_state=initial_states is not None,
                                       compute_outputs=compute_outputs,
                                       chunk_size=chunk_size,
                                       rotary_dim_divisor=rotary_dim_divisor,
                                       dtype=tl_dtype,
                                       outproj_norm_eps=outproj_norm_eps,
                                       threads=threads,
                                       num_stages=num_stages)
    # print(kernel.get_kernel_source()) # NOTE: prints compiled CUDA code
    if not compute_outputs:
        o = None  # state-only pass: O is compile-time unused in the kernel
    elif reduceO:
        o = torch.empty((B, S, H, P), device='cuda', dtype=dtype)
    else:
        o = torch.empty((B, S, R, H, P), device='cuda', dtype=dtype)
    # TileLang accepts None for tensor arguments that are compile-time unused.
    mimo_o_arg = mimo_o if reduceO else None
    if not fuse_pregate_headwise_rms_norm:
        outproj_norm_weight_arg = None
    elif outproj_norm_weight is None:
        outproj_norm_weight_arg = torch.ones((H, P), device=q.device, dtype=torch.float32)
    else:
        if outproj_norm_weight.ndim == 1:
            if outproj_norm_weight.numel() != H * P:
                raise ValueError(
                    f"Expected flattened outproj_norm_weight to have {H * P} elements, "
                    f"got {outproj_norm_weight.numel()}."
                )
            outproj_norm_weight_arg = outproj_norm_weight.reshape(H, P).contiguous()
        elif outproj_norm_weight.shape == (H, P):
            outproj_norm_weight_arg = outproj_norm_weight.contiguous()
        else:
            raise ValueError(
                f"Expected outproj_norm_weight to have shape ({H}, {P}) or ({H * P},), "
                f"got {tuple(outproj_norm_weight.shape)}."
            )
        outproj_norm_weight_arg = outproj_norm_weight_arg.to(device=q.device, dtype=torch.float32)
    z_arg = z
    D_arg = D
    mimo_z_arg = mimo_z
    if initial_states is None:
        init_ssm_state_arg, init_k_state_arg, init_v_state_arg = None, None, None
    else:
        init_ssm_state_arg, init_k_state_arg, init_v_state_arg = initial_states
        expected_sequences = NS if cu_seqlens is not None else B
        if init_ssm_state_arg.shape != (expected_sequences, H, P, N):
            raise ValueError(
                f"Expected initial SSM state shape {(expected_sequences, H, P, N)}, "
                f"got {tuple(init_ssm_state_arg.shape)}"
            )
        if init_k_state_arg.shape != (expected_sequences, R, H, N):
            raise ValueError(
                f"Expected initial K state shape {(expected_sequences, R, H, N)}, "
                f"got {tuple(init_k_state_arg.shape)}"
            )
        if init_v_state_arg.shape != (expected_sequences, H, P):
            raise ValueError(
                f"Expected initial V state shape {(expected_sequences, H, P)}, "
                f"got {tuple(init_v_state_arg.shape)}"
            )
        init_ssm_state_arg = init_ssm_state_arg.contiguous()
        init_k_state_arg = init_k_state_arg.contiguous()
        init_v_state_arg = init_v_state_arg.contiguous()

    _nrow = NS if blk_seg is None else int(blk_seg.numel())
    if blk_seg is None:  # placeholders; `blocked=False` means they are never read
        blk_seg = blk_c0 = blk_nch = torch.zeros(1, device=q.device, dtype=torch.int32)
    if cu_seqlens is not None:
        h = torch.empty((_nrow, H, N, P), device='cuda', dtype=torch.float32) if return_state else None
        k_final = torch.empty((NS, R, H, N), device='cuda', dtype=dtype) if return_state else None
    else:
        h = torch.empty((B, H, N, P), device='cuda', dtype=torch.float32) if return_state else None
        k_final = torch.empty((B, R, H, N), device='cuda', dtype=dtype) if return_state else None

    kernel( q,
            k,
            v, o,
            q_bias, k_bias,
            mimo_v, mimo_o_arg,
            outproj_norm_weight_arg,
            z_arg, D_arg, mimo_z_arg,
            angles,
            dA_cs,
            dA_cs_rev,
            dt,
            trap,
            segsum,
            cu_seqlens,
            init_ssm_state_arg,
            init_k_state_arg,
            init_v_state_arg,
            h,
            k_final,
            blk_seg, blk_c0, blk_nch,
            init_state
            )
    return o, h, k_final
