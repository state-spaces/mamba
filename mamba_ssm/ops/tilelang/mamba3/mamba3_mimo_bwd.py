"""
Tilelang implementation of Mamba3 backward kernels,
with MIMO support.

Copyright (c) 2026, Dao AI Lab, Goombalab

"""

import math
import torch
import tilelang
import tilelang.language as T
from triton.testing import do_bench
from tilelang.autotuner import autotune


import itertools
import argparse
from einops import rearrange
from typing import Optional, Tuple

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import bwd_dadt_fused_triton, bwd_dtrap_ddt_triton


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
def mamba_mimo_bwd_fwd(
    B,
    S,
    H,
    G,
    N,
    P,
    R,
    hasZ,
    hasD,
    reduceO,
    chunk_size: int = 16,
    rotary_dim_divisor: int = 4,
    dtype: str = 'float16',
    threads: int = 128,
    num_stages: int = 0,
) -> torch.Tensor:

    accum_dtype = 'float32'

    nchunks = tilelang.cdiv(S, chunk_size)
    tail_len = S % chunk_size
    fused_chunk_size = chunk_size * R

    if reduceO:
        DOUT_shape = (B, S, H, P)
    else:
        DOUT_shape = (B, S, R, H, P)

    @T.prim_func
    def mamba_mimo_bwd_fwd_kernel(
            DOUT: T.Tensor(DOUT_shape, dtype),  # type: ignore
            Q: T.Tensor([B, S, R, G, N], dtype),  # type: ignore
            K: T.Tensor([B, S, R, G, N], dtype),  # type: ignore
            V: T.Tensor([B, S, H, P], dtype),  # type: ignore
            Q_BIAS: T.Tensor([H, R, N], T.float32),  # type: ignore
            K_BIAS: T.Tensor([H, R, N], T.float32),  # type: ignore
            MIMO_V: T.Tensor([H, R, P], T.float32), # type: ignore
            MIMO_O: T.Tensor([H, R, P], T.float32), # type: ignore
            
            DMIMO_O: T.Tensor([B, H, R, P], T.float32), # type: ignore
            STATES: T.Tensor([B, H, nchunks, N, P], dtype), # type: ignore 

            Z: T.Tensor([B, S, H, P], dtype),  # type: ignore
            MIMO_Z: T.Tensor([H, R, P], T.float32), # type: ignore
            DZ: T.Tensor([B, S, H, P], dtype),  # type: ignore
            DMIMO_Z: T.Tensor([B, H, R, P], T.float32), # type: ignore
            ANGLES: T.Tensor([B, S, H, N//rotary_dim_divisor], T.float32), # type: ignore
            DA_CS: T.Tensor([B, H, S], T.float32), # type: ignore
            DA_CS_REV: T.Tensor([B, H, S], T.float32), # type: ignore
            DT: T.Tensor([B, H, S], T.float32), # type: ignore
            TRAP: T.Tensor([B, H, S], dtype), # type: ignore
            D: T.Tensor([H], T.float32),  # type: ignore

            QK_DOT: T.Tensor([B, H, S, R, R], dtype), # type: ignore
            
            SEGSUM: T.Tensor([B, H, nchunks, chunk_size, chunk_size], T.float32), # type: ignore
            ):
        """
        Overview:
            Fused backward-forward pass over chunks. Recomputes local forward intermediates,
            accumulates projection gradients (DMIMO_O and optional DMIMO_Z), emits optional DZ,
            stores per-chunk recurrent STATES, and materializes QK_DOT for the second backward pass.

        Inputs:
            - Activations and upstream grad: DOUT, Q, K, V.
            - Projection weights/biases: Q_BIAS, K_BIAS, MIMO_V (Psi), MIMO_O (Phi), optional MIMO_Z (Zeta).
            - Optional forward modifiers: Z, D.
            - Discretization tensors: DA_CS, DA_CS_REV, DT, TRAP, and SEGSUM.

        Outputs:
            - MIMO projection grads: DMIMO_O and optional DMIMO_Z.
            - Optional activation grad: DZ.
            - Cached intermediates for pass 2: STATES and QK_DOT.

        Notation:
            - Psi: MIMO X projection.
            - Phi: MIMO O projection.
            - Zeta: MIMO Z projection.
            - Trap: convex-combination modulator used in exponential-trapezoidal discretization.
        """
        
        with T.Kernel(H, B, threads=threads) as (i_h, i_b):
            # --- Kernel Setup ---
            # GQA support: map V head to Q/K head
            i_h_qk = i_h // (H // G)

            # --- Buffer Allocation ---
            q_shared = T.alloc_shared([fused_chunk_size, N], dtype)
            k_shared = T.alloc_shared([fused_chunk_size, N], dtype)
            PsiV_shared = T.alloc_shared([fused_chunk_size, P], dtype)
            qs_shared = T.alloc_shared([fused_chunk_size, P], dtype)
            o_shared = T.alloc_shared([chunk_size, P], dtype)
            v_shared = T.alloc_shared([chunk_size, P], dtype)
            states_accum_cast_shared = T.alloc_shared([N, P], dtype)

            qk_dot_full_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], dtype)

            # --- Output Accumulators ---
            if reduceO:
                dPhi_shared = T.alloc_shared([R, P], accum_dtype)
                T.clear(dPhi_shared)

            dout_shared = T.alloc_shared([chunk_size, P], dtype)

            z_shared = T.alloc_shared([chunk_size, P], dtype)
            dZeta_shared = T.alloc_shared([R, P], accum_dtype)
            T.clear(dZeta_shared)

            # --- Swizzling Annotation ---
            T.annotate_layout({
                q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                k_shared: tilelang.layout.make_swizzled_layout(k_shared),

                PsiV_shared: tilelang.layout.make_swizzled_layout(PsiV_shared),
                qs_shared: tilelang.layout.make_swizzled_layout(qs_shared),
                o_shared: tilelang.layout.make_swizzled_layout(o_shared),
                states_accum_cast_shared: tilelang.layout.make_swizzled_layout(states_accum_cast_shared),
                qk_dot_full_shared: tilelang.layout.make_swizzled_layout(qk_dot_full_shared),
                dout_shared: tilelang.layout.make_swizzled_layout(dout_shared),
                z_shared: tilelang.layout.make_swizzled_layout(z_shared),

            })
            T.use_swizzle(10, "row")

            T.no_set_max_nreg()

            # --- Per-Head Constants / Running State ---
            states_frag = T.alloc_fragment([N, P], accum_dtype)
            T.clear(states_frag)

            if reduceO:
                phi_frag_intrachunk = T.alloc_fragment([R, P], dtype=dtype)
                T.copy(MIMO_O[i_h, :, :], phi_frag_intrachunk)
            Psi_frag = T.alloc_fragment([R, P], dtype)
            T.copy(MIMO_V[i_h, :, :], Psi_frag)

            q_bias_frag = T.alloc_fragment([R, N], dtype)
            k_bias_frag = T.alloc_fragment([R, N], dtype)
            T.copy(Q_BIAS[i_h, :, :], q_bias_frag)
            T.copy(K_BIAS[i_h, :, :], k_bias_frag)

            # --- Chunk Loop ---
            for i in T.Pipelined(0, nchunks, num_stages=num_stages):
                chunk_start = i * chunk_size
                fused_chunk_start = chunk_start * R

                # --- Discretization Factors (Shifted Gamma + Trap Scale) ---
                trap_shifted_frag = T.alloc_fragment([chunk_size], T.float32)
                dt_shifted_frag = T.alloc_fragment([chunk_size], dtype)
                for cs in T.Parallel(chunk_size):
                    trap_shifted_frag[cs] = T.if_then_else(
                        chunk_start + cs + 1 < S,
                        TRAP[i_b, i_h, chunk_start + cs + 1],
                        0.0,
                    )
                    dt_shifted_frag[cs] = T.if_then_else(
                        chunk_start + cs + 1 < S,
                        DT[i_b, i_h, chunk_start + cs + 1],
                        0.0,
                    )
                shifted_gamma_frag = T.alloc_fragment([chunk_size], dtype)
                for cs in T.Parallel(chunk_size):
                    shifted_gamma_frag[cs] = T.if_then_else(chunk_start + cs < (S - 1), 
                                                            dt_shifted_frag[cs] * (T.sigmoid(-trap_shifted_frag[cs])), 
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

                T.copy(V[i_b, chunk_start:chunk_start+chunk_size, i_h, :], v_shared)
                for cs, r, p in T.Parallel(chunk_size, R, P):
                    PsiV_frag[cs, r, p] = v_shared[cs, p] * Psi_frag[r, p]
                PsiV_reshaped_frag = T.view(PsiV_frag, shape=[fused_chunk_size, P])
                T.copy(PsiV_reshaped_frag, PsiV_shared)

                q_reshaped_shared = T.view(q_shared, shape=[chunk_size, R, N])
                T.copy(Q[i_b, chunk_start:chunk_start+chunk_size, :, i_h_qk, :], q_reshaped_shared)
                q_frag = T.alloc_fragment([chunk_size, R, N], dtype)
                T.copy(q_reshaped_shared, q_frag)
                for cs, r, n in T.Parallel(chunk_size, R, N):
                    q_frag[cs, r, n] += q_bias_frag[r, n]
                T.copy(q_frag, q_reshaped_shared)

                k_reshaped_shared = T.view(k_shared, shape=[chunk_size, R, N])
                T.copy(K[i_b, chunk_start:chunk_start+chunk_size, :, i_h_qk, :], k_reshaped_shared)
                k_frag = T.alloc_fragment([chunk_size, R, N], dtype)
                T.copy(k_reshaped_shared, k_frag)
                for cs, r, n in T.Parallel(chunk_size, R, N):
                    k_frag[cs, r, n] += k_bias_frag[r, n]
                T.copy(k_frag, k_reshaped_shared)

                # --- Cache Diagonal qk_dot Path ---
                # Keep full qk_dot in shared memory to reuse per-step R x R blocks.
                qk_dot_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=accum_dtype)
                T.gemm(q_shared, k_shared, qk_dot_frag, transpose_B=True, clear_accum=True)
                T.copy(qk_dot_frag, qk_dot_full_shared)
                # Output QK_DOT for the bwd_bwd kernel (per-time-step blocks only)
                for cs, r_out, r_in in T.Parallel(chunk_size, R, R):
                    QK_DOT[i_b, i_h, chunk_start + cs, r_out, r_in] = \
                        qk_dot_full_shared[cs * R + r_out, cs * R + r_in]

                # --- Rotary Q/K + Trap Scaling ---
                q_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                q_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)

                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    q_first_half_frag[cs, r, n] = q_shared[cs*R + r, n]
                    q_second_half_frag[cs, r, n] = q_shared[cs*R + r, N//2 + n]

                # NOTE: angles are casted to fp32 for numerical stability
                angles_frag = T.alloc_fragment([chunk_size, N//rotary_dim_divisor], T.float32)
                T.copy(ANGLES[i_b, chunk_start:chunk_start+chunk_size, i_h, :], angles_frag)

                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    q_shared[cs*R + r, n] = T.cos(angles_frag[cs, n]) * q_first_half_frag[cs, r, n] - T.sin(angles_frag[cs, n]) * q_second_half_frag[cs, r, n]
                    q_shared[cs*R + r, N//2 + n] = T.sin(angles_frag[cs, n]) * q_first_half_frag[cs, r, n] + T.cos(angles_frag[cs, n]) * q_second_half_frag[cs, r, n]

                k_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                k_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)

                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    k_first_half_frag[cs, r, n] = k_shared[cs*R + r, n]
                    k_second_half_frag[cs, r, n] = k_shared[cs*R + r, N//2 + n]
                
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    k_shared[cs*R + r, n] = T.cos(angles_frag[cs, n]) * k_first_half_frag[cs, r, n] - T.sin(angles_frag[cs, n]) * k_second_half_frag[cs, r, n]
                    k_shared[cs*R + r, N//2 + n] = T.sin(angles_frag[cs, n]) * k_first_half_frag[cs, r, n] + T.cos(angles_frag[cs, n]) * k_second_half_frag[cs, r, n]

                k_trap_scaled_frag = T.alloc_fragment([fused_chunk_size, N], dtype)
                T.copy(k_shared, k_trap_scaled_frag)
                for csr, n in T.Parallel(fused_chunk_size, N):
                    k_trap_scaled_frag[csr, n] *= trap_scale_shared[csr//R]
                T.copy(k_trap_scaled_frag, k_shared)

                # --- Interchunk + Intrachunk Output Accumulation ---
                q_state_out_frag = T.alloc_fragment([fused_chunk_size, P], dtype=accum_dtype)
                # NOTE: Tilelang unable to infer correct layout when trying to cast
                # states_frag to 16-bit within rmem
                T.copy(states_frag, states_accum_cast_shared)
                T.gemm(q_shared, states_accum_cast_shared, q_state_out_frag, clear_accum=True)

                qk_intrachunk_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=accum_dtype)
                T.gemm(q_shared, k_shared, qk_intrachunk_frag, transpose_B=True, clear_accum=True)

                # Strictly causal masking over chunk steps (exclude same-step diagonal).
                da_cs__or__exp_da_cs_shared = T.alloc_shared([chunk_size], T.float32)
                T.copy(DA_CS[i_b, i_h, chunk_start:chunk_start+chunk_size], da_cs__or__exp_da_cs_shared)       
                for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size):
                    qk_intrachunk_frag[csr_i, csr_j] = T.if_then_else(
                                                csr_i//R > csr_j//R,
                                                qk_intrachunk_frag[csr_i, csr_j] * T.exp(SEGSUM[i_b, i_h, i, csr_i//R, csr_j//R]),
                                                0.0
                                            )
                qk_intrachunk_masked_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], dtype=dtype)
                for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size):
                    qk_intrachunk_masked_shared[csr_i, csr_j] = qk_intrachunk_frag[csr_i, csr_j]
                
                # Exponentiate da_cs__or__exp_da_cs_shared so that later usage does not have to:
                for cs in T.Parallel(chunk_size):
                    da_cs__or__exp_da_cs_shared[cs] = T.exp(da_cs__or__exp_da_cs_shared[cs])

                exp_da_cs_frag = T.alloc_fragment([chunk_size], dtype=T.float32)
                T.copy(da_cs__or__exp_da_cs_shared, exp_da_cs_frag)
                for csr, p in T.Parallel(fused_chunk_size, P):
                    q_state_out_frag[csr, p] *= exp_da_cs_frag[csr//R]

                o_mimo_accum_frag = T.alloc_fragment([fused_chunk_size, P], dtype=accum_dtype)
                T.gemm(qk_intrachunk_masked_shared, PsiV_shared, o_mimo_accum_frag, clear_accum=True)

                # Merge interchunk and intrachunk contributions.
                for cs, p in T.Parallel(fused_chunk_size, P):
                    o_mimo_accum_frag[cs, p] += q_state_out_frag[cs, p]

                # --- Add Diagonal Terms (qk_dot and optional D) ---
                qkdot_psiv_frag = T.alloc_fragment([chunk_size, R, P], dtype=dtype)
                T.clear(qkdot_psiv_frag)
                for cs, r_out, p in T.Parallel(chunk_size, R, P):
                    for r_in in T.serial(R):
                        qkdot_psiv_frag[cs, r_out, p] += qk_dot_full_shared[cs * R + r_out, cs * R + r_in] * PsiV_shared[cs * R + r_in, p]
                    qkdot_psiv_frag[cs, r_out, p] *= gamma_frag[cs] # Apply gamma
                qkdot_psiv_reshaped_frag = T.view(qkdot_psiv_frag, shape=[fused_chunk_size, P])
                for csr, p in T.Parallel(fused_chunk_size, P):
                    o_mimo_accum_frag[csr, p] += qkdot_psiv_reshaped_frag[csr, p]

                if hasD:
                    D_var = T.alloc_var(T.float32)
                    T.copy(D[i_h], D_var)
                    PsiV_D_frag = T.alloc_fragment([fused_chunk_size, P], T.float32)
                    T.copy(PsiV_shared, PsiV_D_frag)
                    for csr, p in T.Parallel(fused_chunk_size, P):
                        o_mimo_accum_frag[csr, p] += D_var * PsiV_D_frag[csr, p]

                # --- Project to dMIMO_O and Optional Z Backward Path ---
                if reduceO:
                    out_prereduced_shared = T.alloc_shared([fused_chunk_size, P], dtype)
                    T.copy(o_mimo_accum_frag, out_prereduced_shared)
                    
                    o_gated_frag = T.alloc_fragment([chunk_size, R, P], T.float32)
                    if hasZ:
                        # Apply Z gating to out:
                        T.copy(Z[i_b, chunk_start:chunk_start+chunk_size, i_h, :], z_shared)
                        z_o_frag = T.alloc_fragment([chunk_size, P], T.float32)
                        T.copy(z_shared, z_o_frag)
                        Zeta_o_frag = T.alloc_fragment([R, P], T.float32)
                        T.copy(MIMO_Z[i_h, :, :], Zeta_o_frag)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            # Apply SiLU to o_gated_frag:
                            tmp = z_o_frag[cs, p] * Zeta_o_frag[r, p] * 0.5
                            o_gated_frag[cs, r, p] = tmp * T.tanh(tmp) + tmp
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            o_gated_frag[cs, r, p] *= out_prereduced_shared[cs*R + r, p]
                    else:
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            o_gated_frag[cs, r, p] = out_prereduced_shared[cs*R + r, p]
                    
                    # NOTE: keeping dPhi_frag in fp32 for numerical reason
                    dPhi_frag = T.alloc_fragment([R, P], T.float32)
                    T.copy(dPhi_shared, dPhi_frag)
                    dout_frag = T.alloc_fragment([chunk_size, P], dtype)
                    T.copy(DOUT[i_b, chunk_start:chunk_start+chunk_size, i_h, :], dout_shared)
                    T.copy(dout_shared, dout_frag)
                    for r, p in T.Parallel(R, P):
                        for cs in T.serial(chunk_size):
                            dPhi_frag[r, p] += o_gated_frag[cs, r, p] * dout_frag[cs, p]
                    T.copy(dPhi_frag, dPhi_shared)

                    if hasZ:
                        # Up-project DOUT from SISO to MIMO.
                        Phi_frag = T.alloc_fragment([R, P], dtype)
                        T.copy(MIMO_O[i_h, :, :], Phi_frag)
                        dPhiO_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                        dout_preexpand_frag = T.alloc_fragment([chunk_size, P], dtype)
                        T.copy(dout_shared, dout_preexpand_frag)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            dPhiO_frag[cs, r, p] = dout_frag[cs, p] * Phi_frag[r, p]

                        # NOTE: layout issue when trying to reuse o_mimo_accum_frag
                        # NOTE: note that it uses out_prereduced_shared, which is the pre-Z-gate version
                        # of out
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            dPhiO_frag[cs, r, p] *= out_prereduced_shared[cs*R + r, p]
                        # Backward of SILU(z) is sigmoid(z) * (1 + z * (1 - sigmoid(z)))
                        z_frag = T.alloc_fragment([chunk_size, P], T.float32)
                        T.copy(z_shared, z_frag)
                        Zeta_frag = T.alloc_fragment([R, P], T.float32)
                        T.copy(MIMO_Z[i_h, :, :], Zeta_frag)
                        dZetaZ_frag = T.alloc_fragment([chunk_size, R, P], T.float32)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            dZetaZ_frag[cs, r, p] = z_frag[cs, p] * Zeta_frag[r, p]
                            dZetaZ_frag[cs, r, p] = dPhiO_frag[cs, r, p]* T.sigmoid(dZetaZ_frag[cs, r, p]) * \
                                (1 + dZetaZ_frag[cs, r, p] * (T.sigmoid(-dZetaZ_frag[cs, r, p])))

                        dZ_frag = T.alloc_fragment([chunk_size, P], dtype)
                        T.clear(dZ_frag)
                        for cs, p in T.Parallel(chunk_size, P):
                            for r in T.serial(R):
                                dZ_frag[cs, p] += dZetaZ_frag[cs, r, p] * Zeta_frag[r, p]
                        T.copy(dZ_frag, DZ[i_b, chunk_start:chunk_start+chunk_size, i_h, :])

                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            dZetaZ_frag[cs, r, p] *= z_frag[cs, p]
                        dZeta_frag = T.alloc_fragment([R, P], T.float32)
                        T.copy(dZeta_shared, dZeta_frag)
                        T.reduce_sum(dZetaZ_frag, dZeta_frag, clear=False, dim=0)
                        T.copy(dZeta_frag, dZeta_shared)
                else:
                    if hasZ:
                        out_prereduced_shared = T.alloc_shared([fused_chunk_size, P], dtype)
                        T.copy(o_mimo_accum_frag, out_prereduced_shared)
                        T.copy(Z[i_b, chunk_start:chunk_start+chunk_size, i_h, :], z_shared)
                        dPhiO_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            dPhiO_frag[cs, r, p] = DOUT[i_b, chunk_start + cs, r, i_h, p]
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            dPhiO_frag[cs, r, p] *= out_prereduced_shared[cs*R + r, p]
                        # Backward of SILU(z) is sigmoid(z) * (1 + z * (1 - sigmoid(z)))
                        z_frag = T.alloc_fragment([chunk_size, P], T.float32)
                        T.copy(z_shared, z_frag)
                        Zeta_frag = T.alloc_fragment([R, P], T.float32)
                        T.copy(MIMO_Z[i_h, :, :], Zeta_frag)
                        dZetaZ_frag = T.alloc_fragment([chunk_size, R, P], T.float32)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            dZetaZ_frag[cs, r, p] = z_frag[cs, p] * Zeta_frag[r, p]
                            dZetaZ_frag[cs, r, p] = dPhiO_frag[cs, r, p]* T.sigmoid(dZetaZ_frag[cs, r, p]) * \
                                (1 + dZetaZ_frag[cs, r, p] * (T.sigmoid(-dZetaZ_frag[cs, r, p])))
                        ## Compute DZ
                        dZ_frag = T.alloc_fragment([chunk_size, P], dtype)
                        T.clear(dZ_frag)
                        for cs, p in T.Parallel(chunk_size, P):
                            for r in T.serial(R):
                                dZ_frag[cs, p] += dZetaZ_frag[cs, r, p] * Zeta_frag[r, p]
                        T.copy(dZ_frag, DZ[i_b, chunk_start:chunk_start+chunk_size, i_h, :])
                        ## Compute DMIMO_Z
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            dZetaZ_frag[cs, r, p] *= z_frag[cs, p]
                        dZeta_frag = T.alloc_fragment([R, P], T.float32)
                        T.copy(dZeta_shared, dZeta_frag)
                        T.reduce_sum(dZetaZ_frag, dZeta_frag, clear=False, dim=0)
                        T.copy(dZeta_frag, dZeta_shared)

                # --- Save and Update Recurrent State ---
                T.copy(states_frag, STATES[i_b, i_h, i, :, :])

                # DA_CS_REV scales stepwise K contribution into the new state.
                dA_cs_rev_frag = T.alloc_fragment([chunk_size], T.float32)
                T.copy(DA_CS_REV[i_b, i_h, chunk_start:chunk_start+chunk_size], dA_cs_rev_frag)
                # NOTE: we can recycle k_trap_scaled_frag from earlier, however,
                # that is slower, so choose to recopy from smem:
                k_state_frag = T.alloc_fragment([fused_chunk_size, N], dtype)
                T.copy(k_shared, k_state_frag)
                for csr, n in T.Parallel(fused_chunk_size, N):
                    k_state_frag[csr, n] *= T.exp(dA_cs_rev_frag[csr//R])

                # DA_CS(last) applies chunk-level decay to the carried state.
                da_cs_sum = T.alloc_var(T.float32)
                if tail_len > 0 and i == nchunks - 1:
                    T.copy(DA_CS[i_b, i_h, S - 1], da_cs_sum)
                else:
                    T.copy(DA_CS[i_b, i_h, chunk_start+chunk_size-1], da_cs_sum)
                for n, p in T.Parallel(N, P):
                    states_frag[n, p] *= T.exp(da_cs_sum)
                T.gemm(k_state_frag, PsiV_shared, states_frag, transpose_A=True, clear_accum=False)
            
            if reduceO:
                T.copy(dPhi_shared, DMIMO_O[i_b, i_h, :, :])
            if hasZ:
                T.copy(dZeta_shared, DMIMO_Z[i_b, i_h, :, :])

    return mamba_mimo_bwd_fwd_kernel

# def get_configs():
#     iter_params = dict(num_stages=[0], threads=[128, 256])
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
def mamba_mimo_bwd_bwd(
    B,
    S,
    H,
    G,
    N,
    P,
    R,
    hasZ,
    hasD,
    reduceO,
    chunk_size: int = 16,
    rotary_dim_divisor: int = 4,
    dtype: str = 'float16',
    threads: int = 256,
    num_stages: int = 0,
) -> torch.Tensor:

    accum_dtype = 'float32'

    nchunks = tilelang.cdiv(S, chunk_size)
    tail_len = S % chunk_size
    fused_chunk_size = chunk_size * R

    if reduceO:
        DOUT_shape = (B, S, H, P)
    else:
        DOUT_shape = (B, S, R, H, P)

    @T.prim_func
    def mamba_mimo_bwd_bwd_kernel(
            DOUT: T.Tensor(DOUT_shape, dtype),  # type: ignore
            Q: T.Tensor([B, S, R, G, N], dtype),  # type: ignore
            K: T.Tensor([B, S, R, G, N], dtype),  # type: ignore
            V: T.Tensor([B, S, H, P], dtype),  # type: ignore
            Q_BIAS: T.Tensor([H, R, N], T.float32),  # type: ignore
            K_BIAS: T.Tensor([H, R, N], T.float32),  # type: ignore
            MIMO_V: T.Tensor([H, R, P], T.float32), # type: ignore
            MIMO_O: T.Tensor([H, R, P], T.float32), # type: ignore
            DK: T.Tensor([B, S*R, H, N], dtype),  # type: ignore
            DV: T.Tensor([B, S, H, P], dtype),  # type: ignore
            DMIMO_V: T.Tensor([B, H, R, P], T.float32), # type: ignore
            STATES: T.Tensor([B, H, nchunks, N, P], dtype), # type: ignore 
            DQ: T.Tensor([B, S*R, H, N], dtype),  # type: ignore

            Z: T.Tensor([B, S, H, P], dtype),  # type: ignore
            MIMO_Z: T.Tensor([H, R, P], T.float32), # type: ignore
            ANGLES: T.Tensor([B, S, H, N//rotary_dim_divisor], T.float32), # type: ignore
            DA_CS: T.Tensor([B, H, S], T.float32), # type: ignore
            DA_CS_REV: T.Tensor([B, H, S], T.float32), # type: ignore
            DT: T.Tensor([B, H, S], T.float32), # type: ignore
            TRAP: T.Tensor([B, H, S], dtype), # type: ignore
            DFACTOR: T.Tensor([B, H, S], T.float32), # type: ignore
            DGAMMA_DIAG: T.Tensor([B, H, S], T.float32), # type: ignore
            DANGLES: T.Tensor([B, S, H, N//rotary_dim_divisor], T.float32), # type: ignore
            D: T.Tensor([H], T.float32), # type: ignore
            DD: T.Tensor([B, H], T.float32), # type: ignore

            QK_DOT: T.Tensor([B, H, S, R, R], dtype), # type: ignore
            # DQK_DOT: T.Tensor([B, H, S, R, R], dtype), # type: ignore
            DDA: T.Tensor([B, H, S], T.float32), # type: ignore
            DSSDA: T.Tensor([B, H, nchunks, chunk_size, chunk_size], T.float32), # type: ignore
            DDA_CS_REV: T.Tensor([B, H, S], T.float32), # type: ignore
            DDA_CS: T.Tensor([B, H, S], T.float32), # type: ignore

            SEGSUM: T.Tensor([B, H, nchunks, chunk_size, chunk_size], T.float32), # type: ignore
            ):
        """
        Overview:
            Reverse-chunk backward pass that consumes cached STATES and QK_DOT from the first pass
            to produce gradients for the fused Mamba3 attention block.

        Inputs:
            - Forward activations/tensors: DOUT, Q, K, V, optional Z, optional D.
            - Projection weights/biases: Q_BIAS, K_BIAS, MIMO_V (Psi), MIMO_O (Phi), optional MIMO_Z (Zeta).
            - Cached intermediates: STATES and QK_DOT.
            - Discretization grads and factors:
              DA_CS, DA_CS_REV, DT, TRAP, DDA, DSSDA, DDA_CS_REV, DDA_CS, and SEGSUM.

        Outputs:
            - QKV grads: DQ, DK, DV.
            - MIMO projection grads: DMIMO_V.
            - Discretization/rotation grads: DANGLES, DFACTOR, DGAMMA_DIAG, DDA_CS_REV, DDA_CS, DDA.
            - Additional grads: optional DD.

        Notation:
            - Psi: MIMO X projection.
            - Phi: MIMO O projection.
            - Zeta: MIMO Z projection.
            - Trap: convex-combination modulator used in exponential-trapezoidal discretization.
        """
        
        with T.Kernel(H, B, threads=threads) as (i_h, i_b):
            # --- Kernel Setup ---
            # GQA support: map V head to Q/K head
            i_h_qk = i_h // (H // G)

            # --- Buffer Allocation ---
            dstates_shared = T.alloc_shared([N, P], dtype)
            dstates_frag = T.alloc_fragment([N, P], accum_dtype)

            dout_shared = T.alloc_shared([chunk_size, P], dtype)
            dPhiO_shared = T.alloc_shared([fused_chunk_size, P], dtype)

            q_shared = T.alloc_shared([fused_chunk_size, N], dtype)

            k_shared = T.alloc_shared([fused_chunk_size, N], dtype)
            v_shared = T.alloc_shared([chunk_size, P], dtype)

            states_shared = T.alloc_shared([N, P], dtype)
            lkq_masked__or__dkq_masked_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], dtype)

            dPsiV_combined_shared = T.alloc_shared([fused_chunk_size, P], dtype)

            dqk_from_diag_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], accum_dtype)

            q_pre_rot_shared = T.alloc_shared([fused_chunk_size, N], dtype)
            k_pre_rot_shared = T.alloc_shared([fused_chunk_size, N], dtype)

            dk_shared = T.alloc_shared([fused_chunk_size, N], dtype)
            dq_shared = T.alloc_shared([fused_chunk_size, N], dtype)

            qk_dot_shared = T.alloc_shared([chunk_size, R, R], dtype)

            k_pre_trap_shared = T.alloc_shared([fused_chunk_size, N], dtype)

            dangle_dk__or__dq_shared = T.alloc_shared([fused_chunk_size, N//rotary_dim_divisor], T.float32)

            # --- Swizzling Annotation ---
            noswizzle_annot = threads == 256 and (N <= 32 or P >= 128) # NOTE: heuristics for when swizzling annotation causes kernel hang, needs more investigation
            if not noswizzle_annot:
                T.annotate_layout({
                    dstates_shared: tilelang.layout.make_swizzled_layout(dstates_shared),
                    dout_shared: tilelang.layout.make_swizzled_layout(dout_shared),
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),

                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    v_shared: tilelang.layout.make_swizzled_layout(v_shared),
                    states_shared: tilelang.layout.make_swizzled_layout(states_shared),
                    lkq_masked__or__dkq_masked_shared: tilelang.layout.make_swizzled_layout(lkq_masked__or__dkq_masked_shared),

                    dPsiV_combined_shared: tilelang.layout.make_swizzled_layout(dPsiV_combined_shared),
                    dqk_from_diag_shared: tilelang.layout.make_swizzled_layout(dqk_from_diag_shared),

                    k_pre_rot_shared: tilelang.layout.make_swizzled_layout(k_pre_rot_shared),
                    q_pre_rot_shared: tilelang.layout.make_swizzled_layout(q_pre_rot_shared),

                    dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
                    dq_shared: tilelang.layout.make_swizzled_layout(dq_shared),

                    k_pre_trap_shared: tilelang.layout.make_swizzled_layout(k_pre_trap_shared),
                    dangle_dk__or__dq_shared: tilelang.layout.make_swizzled_layout(dangle_dk__or__dq_shared),
                })
            T.use_swizzle(10, "row")
            T.no_set_max_nreg()

            # --- Per-Head Constants / Running State ---
            T.clear(dstates_frag)
            T.clear(dstates_shared)

            if reduceO:
                Phi_frag = T.alloc_fragment([R, P], dtype)
                T.copy(MIMO_O[i_h, :, :], Phi_frag)
            Psi_frag = T.alloc_fragment([R, P], dtype)
            T.copy(MIMO_V[i_h, :, :], Psi_frag)

            dPsi_acc = T.alloc_fragment([R, P], accum_dtype) # TODO
            T.clear(dPsi_acc)

            if hasD:
                dD_frag = T.alloc_fragment([1], accum_dtype)
                T.clear(dD_frag)

            q_bias_frag = T.alloc_fragment([R, N], dtype)
            k_bias_frag = T.alloc_fragment([R, N], dtype)
            T.copy(Q_BIAS[i_h, :, :], q_bias_frag)
            T.copy(K_BIAS[i_h, :, :], k_bias_frag)

            # --- Reverse Chunk Loop ---
            for chunk_idx_rev in T.Pipelined(0, nchunks, num_stages=num_stages):
                chunk_idx = nchunks - 1 - chunk_idx_rev
                chunk_start = chunk_idx * chunk_size
                fused_chunk_start = chunk_start * R

                # --- Discretization Factors (Shifted Gamma + Trap Scale) ---
                trap_shifted_frag = T.alloc_fragment([chunk_size], T.float32)
                dt_shifted_frag = T.alloc_fragment([chunk_size], dtype)
                for cs in T.Parallel(chunk_size):
                    trap_shifted_frag[cs] = T.if_then_else(
                        chunk_start + cs + 1 < S,
                        TRAP[i_b, i_h, chunk_start + cs + 1],
                        0.0,
                    )
                    dt_shifted_frag[cs] = T.if_then_else(
                        chunk_start + cs + 1 < S,
                        DT[i_b, i_h, chunk_start + cs + 1],
                        0.0,
                    )
                shifted_gamma_frag = T.alloc_fragment([chunk_size], dtype)
                for cs in T.Parallel(chunk_size):
                    shifted_gamma_frag[cs] = T.if_then_else(chunk_start + cs < (S - 1), 
                                                            dt_shifted_frag[cs] * T.sigmoid(-trap_shifted_frag[cs]), 
                                                            0.0)

                trap_frag = T.alloc_fragment([chunk_size], T.float32)
                T.copy(TRAP[i_b, i_h, chunk_start: chunk_start+chunk_size], trap_frag)
                dt_frag = T.alloc_fragment([chunk_size], dtype)
                T.copy(DT[i_b, i_h, chunk_start: chunk_start+chunk_size], dt_frag)
                gamma_frag = T.alloc_fragment([chunk_size], T.float32)
                for cs in T.Parallel(chunk_size):
                    gamma_frag[cs] = dt_frag[cs] * T.sigmoid(trap_frag[cs])
                gamma_cached_frag = T.alloc_fragment([chunk_size], T.float32)
                T.copy(gamma_frag, gamma_cached_frag)
                trap_scale_frag = T.alloc_fragment([chunk_size], dtype)
                for cs in T.Parallel(chunk_size):
                    trap_scale_frag[cs] = gamma_frag[cs] + shifted_gamma_frag[cs]
                trap_scale_shared = T.alloc_shared([chunk_size], dtype)
                T.copy(trap_scale_frag, trap_scale_shared)

                # --- DOUT Projection and Optional Z / D Paths ---
                dPhiO_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                if reduceO:
                    for cs, p in T.Parallel(chunk_size, P):
                        dout_shared[cs, p] = DOUT[i_b, chunk_start+cs, i_h, p]
                    for cs, r, p in T.Parallel(chunk_size, R, P):
                        dPhiO_frag[cs, r, p] = dout_shared[cs, p] * Phi_frag[r, p]
                else:
                    for cs, r, p in T.Parallel(chunk_size, R, P):
                        dPhiO_frag[cs, r, p] = DOUT[i_b, chunk_start + cs, r, i_h, p]

                if hasZ:
                    ## Backpropagate via *SILU(Z)
                    Zeta_frag = T.alloc_fragment([R, P], dtype)
                    T.copy(MIMO_Z[i_h, :, :], Zeta_frag)
                    z_frag = T.alloc_fragment([chunk_size, P], dtype)
                    T.copy(Z[i_b, chunk_start:chunk_start+chunk_size, i_h, :], z_frag)
                    for cs, r, p in T.Parallel(chunk_size, R, P):
                        tmp = z_frag[cs, p] * Zeta_frag[r, p] * 0.5
                        dPhiO_frag[cs, r, p] *= tmp * T.tanh(tmp) + tmp
                T.copy(T.view(dPhiO_frag, shape=[fused_chunk_size, P]), dPhiO_shared)

                T.copy(V[i_b, chunk_start:chunk_start+chunk_size, i_h, :], v_shared)
                if hasD:
                    # Compute dD via projected DOUT and V/Psi factors.
                    v_dD_frag =  T.alloc_fragment([chunk_size, P], accum_dtype)
                    Psi_dD_frag = T.alloc_fragment([R, P], accum_dtype)
                    T.copy(v_shared, v_dD_frag)
                    T.copy(MIMO_V[i_h, :, :], Psi_dD_frag)
                    for cs, r, p in T.Parallel(chunk_size, R, P):
                        dPhiO_frag[cs, r, p] *= v_dD_frag[cs, p] * Psi_dD_frag[r, p]
                    T.reduce_sum(T.view(dPhiO_frag, shape=[fused_chunk_size*P]), dD_frag, clear=False)

                # --- Prepare Rotated/Scaled QK and Compute dPsiV ---
                # Load q and apply q_bias to it:
                for cs, r, n in T.Parallel(chunk_size, R, N):
                    q_shared[cs*R + r, n] = Q[i_b, chunk_start+cs, r, i_h_qk, n]
                
                q_frag = T.alloc_fragment([chunk_size, R, N], dtype)
                for cs, r, n in T.Parallel(chunk_size, R, N):
                    q_frag[cs, r, n] = q_shared[cs*R + r, n]
                for cs, r, n in T.Parallel(chunk_size, R, N):
                    q_frag[cs, r, n] += q_bias_frag[r, n]
                for cs, r, n in T.Parallel(chunk_size, R, N):
                    q_shared[cs*R + r, n] = q_frag[cs, r, n]
                T.copy(q_shared, q_pre_rot_shared) # Save pre-rotated q for later:
                # Apply rotary to q:
                q_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                q_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    q_first_half_frag[cs, r, n] = q_shared[cs*R + r, n]
                    q_second_half_frag[cs, r, n] = q_shared[cs*R + r, N//2 + n]
                angles_frag = T.alloc_fragment([chunk_size, N//rotary_dim_divisor], T.float32)
                T.copy(ANGLES[i_b, chunk_start:chunk_start+chunk_size, i_h, :], angles_frag)
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    q_shared[cs*R + r, n] = T.cos(angles_frag[cs, n]) * q_first_half_frag[cs, r, n] - T.sin(angles_frag[cs, n]) * q_second_half_frag[cs, r, n]
                    q_shared[cs*R + r, N//2 + n] = T.sin(angles_frag[cs, n]) * q_first_half_frag[cs, r, n] + T.cos(angles_frag[cs, n]) * q_second_half_frag[cs, r, n]

                # Load k and apply k_bias to it:
                k_reshaped_shared = T.view(k_pre_trap_shared, shape=[chunk_size, R, N])
                T.copy(K[i_b, chunk_start:chunk_start+chunk_size, :, i_h_qk, :], k_reshaped_shared)
                k_frag = T.alloc_fragment([chunk_size, R, N], dtype)
                T.copy(k_reshaped_shared, k_frag)
                for cs, r, n in T.Parallel(chunk_size, R, N):
                    k_frag[cs, r, n] += k_bias_frag[r, n]
                T.copy(k_frag, k_reshaped_shared)
                # Save pre-rotated k for later:
                for csr, n in T.Parallel(fused_chunk_size, N):
                    k_pre_rot_shared[csr, n] = k_pre_trap_shared[csr, n]
                # Apply rotary to k:
                k_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                k_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    k_first_half_frag[cs, r, n] = k_reshaped_shared[cs, r, n]
                    k_second_half_frag[cs, r, n] = k_reshaped_shared[cs, r, N//2 + n]
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    k_reshaped_shared[cs, r, n] = T.cos(angles_frag[cs, n]) * k_first_half_frag[cs, r, n] - T.sin(angles_frag[cs, n]) * k_second_half_frag[cs, r, n]
                    k_reshaped_shared[cs, r, N//2 + n] = T.sin(angles_frag[cs, n]) * k_first_half_frag[cs, r, n] + T.cos(angles_frag[cs, n]) * k_second_half_frag[cs, r, n]
                # Apply Trap-specific scaling:
                k_trap_scaled_frag = T.alloc_fragment([fused_chunk_size, N], dtype)
                T.copy(k_pre_trap_shared, k_trap_scaled_frag)
                for csr, n in T.Parallel(fused_chunk_size, N):
                    k_trap_scaled_frag[csr, n] *= trap_scale_shared[csr//R]
                T.copy(k_trap_scaled_frag, k_shared)

                # Apply the effect of interchunk (state update):
                dPsiV_frag = T.alloc_fragment([fused_chunk_size, P], accum_dtype)
                T.gemm(k_shared, dstates_shared, dPsiV_frag, clear_accum=True)
                dA_cs_rev_frag = T.alloc_fragment([chunk_size], T.float32)
                dA_cs_rev_shared = T.alloc_shared([chunk_size], T.float32)
                T.copy(DA_CS_REV[i_b, i_h, chunk_start:chunk_start+chunk_size], dA_cs_rev_shared)
                T.copy(dA_cs_rev_shared, dA_cs_rev_frag)
                for csr, p in T.Parallel(fused_chunk_size, P):
                    # DA_CS_REV scales per-step state contribution into dPsiV.
                    dPsiV_frag[csr, p] *= T.exp(dA_cs_rev_frag[csr//R])

                # Apply the effect of intrachunk:
                lkq_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], accum_dtype)
                T.gemm(k_shared, q_shared, lkq_frag, transpose_B=True, clear_accum=True)
                T.copy(lkq_frag, lkq_masked__or__dkq_masked_shared) # NOTE: Save later for the computation of DSSDA, using lkq_masked__or__dkq_masked_shared which has the same shape
                if R == 1: # More smem efficient which is necessary for R=1, but slower due to the need for casting
                    lkq_masked_dtype_buf = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype)
                    T.copy(lkq_masked__or__dkq_masked_shared, lkq_masked_dtype_buf)
                    for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size):
                        # Reverse-causal mask for backward flow across chunk steps.
                        lkq_masked_dtype_buf[csr_i, csr_j] = T.if_then_else(
                            csr_i//R < csr_j//R,
                            lkq_masked_dtype_buf[csr_i, csr_j]
                            * T.exp(SEGSUM[i_b, i_h, chunk_idx, csr_j//R, csr_i//R]),
                            0.0
                        )
                else:
                    for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size):
                        # Reverse-causal mask for backward flow across chunk steps.
                        lkq_frag[csr_i, csr_j] = T.if_then_else(
                            csr_i//R < csr_j//R,
                            lkq_frag[csr_i, csr_j]
                            * T.exp(SEGSUM[i_b, i_h, chunk_idx, csr_j//R, csr_i//R]),
                            0.0
                        )
                    lkq_masked_dtype_buf = T.alloc_shared([fused_chunk_size, fused_chunk_size], dtype)
                    T.copy(lkq_frag, lkq_masked_dtype_buf) # Convert to dtype
                T.gemm(lkq_masked_dtype_buf, dPhiO_shared, dPsiV_frag, clear_accum=False)

                # --- Add Diagonal Contributions to dPsiV (D and qk_dot) ---
                dPsiV_D_fused_frag = T.alloc_fragment([fused_chunk_size, P], accum_dtype)
                if hasD:
                    D_frag = T.alloc_var(T.float32)
                    T.copy(D[i_h], D_frag)
                    for csr, p in T.Parallel(fused_chunk_size, P):
                        dPsiV_D_fused_frag[csr, p] = dPsiV_frag[csr, p] + dPhiO_shared[csr, p]*D_frag
                else:
                    T.copy(dPsiV_frag, dPsiV_D_fused_frag)
                # Compute the contribution from the qk_dot term:
                # NOTE: recomputing qk_dot here is much slower than just loading from
                # the result of the bwd_fwd kernel
                qk_dot_frag = T.alloc_fragment([chunk_size, R, R], dtype)
                T.copy(QK_DOT[i_b, i_h, chunk_start:chunk_start+chunk_size, :, :], qk_dot_shared)
                T.copy(qk_dot_shared, qk_dot_frag)
                gamma_dPsiV_frag = T.alloc_fragment([chunk_size], dtype)
                T.copy(gamma_frag, gamma_dPsiV_frag)
                for csr, p in T.Parallel(fused_chunk_size, P):
                    cs = csr // R
                    r_in = csr % R
                    for r_out in T.serial(R):
                        csr_out = cs * R + r_out
                        dPsiV_D_fused_frag[csr, p] += dPhiO_shared[csr_out, p] * qk_dot_frag[cs, r_out, r_in] * gamma_dPsiV_frag[cs]
                T.copy(dPsiV_D_fused_frag, dPsiV_combined_shared)

                # --- Compute dV and dPsi from dPsiV ---
                # Compute dV
                dv_frag = T.alloc_fragment([chunk_size, P], dtype)
                T.clear(dv_frag)
                for cs, p in T.Parallel(chunk_size, P):
                    for r in T.serial(R):
                        dv_frag[cs, p] += dPsiV_combined_shared[cs*R + r, p] * Psi_frag[r, p]
                T.copy(dv_frag, DV[i_b, chunk_start:chunk_start+chunk_size, i_h, :])

                dPsi_frag = T.alloc_fragment([R, P], accum_dtype)
                T.copy(dPsi_acc, dPsi_frag)
                v_frag = T.alloc_fragment([chunk_size, P], accum_dtype)
                T.copy(v_shared, v_frag)
                for r, p in T.Parallel(R, P):
                    for cs in T.serial(chunk_size):
                        dPsi_frag[r, p] += dPsiV_combined_shared[cs*R + r, p] * v_frag[cs, p]
                T.copy(dPsi_frag, dPsi_acc)

                # Compute Psi_V
                PsiV_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                T.clear(PsiV_frag)
                for cs, p in T.Parallel(chunk_size, P):
                    for r in T.serial(R):
                        PsiV_frag[cs, r, p] += v_frag[cs, p] * Psi_frag[r, p]
                # NOTE: Tilelang unable to perform gemm with reshaped PsiV_frag
                # so have to copy to smem
                PsiV_shared  = T.alloc_shared([fused_chunk_size, P], dtype)
                for cs, r, p in T.Parallel(chunk_size, R, P):
                    PsiV_shared[cs*R + r, p] = PsiV_frag[cs, r, p]

                # Compute dqk_from_diag, which is the contribution to dQ/dK from qk_dot:
                dqk_from_diag_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], accum_dtype)
                T.gemm(dPhiO_shared, PsiV_shared, dqk_from_diag_frag, transpose_B=True, clear_accum=True) # (cs*r_out, cs*r_in)
                # Compute dgamma_diag
                dgamma_diag_prereduce_frag = T.alloc_fragment([chunk_size, R, R], accum_dtype)
                T.copy(qk_dot_shared, dgamma_diag_prereduce_frag)
                T.copy(dqk_from_diag_frag, dqk_from_diag_shared)
                for cs, r_out, r_in in T.Parallel(chunk_size, R, R):
                    dgamma_diag_prereduce_frag[cs, r_out, r_in] *= dqk_from_diag_shared[cs*R + r_out, cs*R + r_in]

                dgamma_diag_reduced_frag = T.alloc_fragment([chunk_size], accum_dtype)
                T.reduce_sum(
                    T.view(dgamma_diag_prereduce_frag, shape=[chunk_size, R*R]),
                    dgamma_diag_reduced_frag,
                    dim=-1,
                    clear=True
                    )
                T.copy(dgamma_diag_reduced_frag, DGAMMA_DIAG[i_b, i_h, chunk_start:chunk_start+chunk_size])
                # Apply shifted gamma to dqk:
                gamma_qk_frag = T.alloc_fragment([chunk_size], accum_dtype)
                T.copy(gamma_cached_frag, gamma_qk_frag) # Apply shifted gamma
                for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size):
                    dqk_from_diag_frag[csr_i, csr_j] *= gamma_qk_frag[csr_i//R]
                T.copy(dqk_from_diag_frag, dqk_from_diag_shared)

                # --- dK Path + ddA Terms ---
                dk_frag = T.alloc_fragment([fused_chunk_size, N], accum_dtype)
                T.gemm(PsiV_shared, dstates_shared, dk_frag, transpose_B=True, clear_accum=True)

                # Compute contribution to ddA from KV part of state update (part 1 of 4)
                ddA_state_kv_prereduce_frag = T.alloc_fragment([fused_chunk_size, N], accum_dtype)
                T.copy(k_shared, ddA_state_kv_prereduce_frag)
                for csr, n in T.Parallel(fused_chunk_size, N):
                    ddA_state_kv_prereduce_frag[csr, n] *= dk_frag[csr, n]
                ddA_state_kv_prereduce_frag_reshaped = T.view(ddA_state_kv_prereduce_frag, shape=[chunk_size, R*N])
                ddA_state_kv_frag = T.alloc_fragment([chunk_size], accum_dtype)
                T.reduce_sum(ddA_state_kv_prereduce_frag_reshaped, ddA_state_kv_frag, dim=-1, clear=True)
                T.copy(ddA_state_kv_frag, DDA_CS_REV[i_b, i_h, chunk_start:chunk_start+chunk_size])

                # Interchunk path uses k_scaled * exp(dA_cs_rev) in forward,
                # so apply exp(dA_cs_rev) to the interchunk dk term only.
                dA_cs_rev_dk_frag = T.alloc_fragment([chunk_size], T.float32)
                T.copy(dA_cs_rev_shared, dA_cs_rev_dk_frag)
                for cs in T.Parallel(chunk_size):
                    dA_cs_rev_dk_frag[cs] = T.exp(dA_cs_rev_dk_frag[cs])
                for csr, n in T.Parallel(fused_chunk_size, N):
                    dk_frag[csr, n] *= dA_cs_rev_dk_frag[csr//R]

                dk_intrachunk_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], accum_dtype)
                T.gemm(PsiV_shared, dPhiO_shared, dk_intrachunk_frag, transpose_B=True, clear_accum=True)

                # Compute contribution to ddA from intrachunk (part 2 of 4)
                kq_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype)
                T.copy(lkq_masked__or__dkq_masked_shared, kq_frag)
                for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size):
                    kq_frag[csr_i, csr_j] *= dk_intrachunk_frag[csr_i, csr_j]
                kq_frag_reshaped = T.view(kq_frag, shape=[fused_chunk_size, chunk_size, R])
                interchunk_dda_prereduce_frag = T.alloc_fragment([fused_chunk_size, chunk_size], accum_dtype)
                T.reduce_sum(kq_frag_reshaped, interchunk_dda_prereduce_frag, dim=-1, clear=True)
                interchunk_dda_prereduce_frag_reshaped = T.view(interchunk_dda_prereduce_frag, shape=[chunk_size, R, chunk_size])
                interchunk_dda_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
                T.reduce_sum(interchunk_dda_prereduce_frag_reshaped, interchunk_dda_frag, dim=1, clear=True)
                T.copy(interchunk_dda_frag, DSSDA[i_b, i_h, chunk_idx, :, :])

                for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size):
                    # Reverse-causal mask for intrachunk gradient flow.
                    dk_intrachunk_frag[csr_i, csr_j] = T.if_then_else(
                        csr_i//R < csr_j//R,
                        dk_intrachunk_frag[csr_i, csr_j]
                        * T.exp(SEGSUM[i_b, i_h, chunk_idx, csr_j//R, csr_i//R]),
                        0.0
                    )

                T.copy(dk_intrachunk_frag, lkq_masked__or__dkq_masked_shared) # denote lkq_masked__or__dkq_masked_shared as dkq_intrachunk
                T.copy(dk_frag, dk_shared)
                dk_nodiag_frag = T.alloc_fragment([fused_chunk_size, N], accum_dtype)
                T.copy(dk_shared, dk_nodiag_frag)
                T.gemm(lkq_masked__or__dkq_masked_shared, q_shared, dk_nodiag_frag, clear_accum=False) # Adding dk_interchunk to dkq_intrachunk @ q
                # Compute dfactor, using dk_nodiag_frag:
                k_factor_frag = T.alloc_fragment([chunk_size, R, N], accum_dtype)
                T.copy(k_pre_trap_shared, T.view(k_factor_frag, shape=[fused_chunk_size, N]))
                dfactor_prereduce_frag = T.alloc_fragment([chunk_size, R, N], accum_dtype)
                for cs, r, n in T.Parallel(chunk_size, R, N):
                    dfactor_prereduce_frag[cs, r, n] = k_factor_frag[cs, r, n] * dk_nodiag_frag[cs*R + r, n]
                dfactor_frag = T.alloc_fragment([chunk_size], accum_dtype)
                T.reduce_sum(T.view(dfactor_prereduce_frag, shape=[chunk_size, R*N]), dfactor_frag, dim=-1, clear=True)
                T.copy(dfactor_frag, DFACTOR[i_b, i_h, chunk_start:chunk_start+chunk_size])
                # Account for the effect of trap_scale = gamma + shifted_gamma:
                trap_scale_dk_frag = T.alloc_fragment([chunk_size], dtype)
                T.copy(trap_scale_shared, trap_scale_dk_frag)
                for csr, n in T.Parallel(fused_chunk_size, N):
                    dk_nodiag_frag[csr, n] *= trap_scale_dk_frag[csr//R]
                T.copy(dk_nodiag_frag, dk_shared)

                # --- State-Passing ddA Terms + Interchunk dQ ---
                T.copy(STATES[i_b, i_h, chunk_idx, :, :], states_shared) # Load cached states from bwd_fwd
                # NOTE: Compute the contribution of state passing (part 3 of 4)
                states_frag = T.alloc_fragment([N, P], T.float32)
                T.copy(states_shared, states_frag)
                ddA_state_passing = T.alloc_fragment([1], T.float32)
                ddA_state_passing_prereduce_frag = T.alloc_fragment([N, P], T.float32)
                da_cs_sum = T.alloc_var(T.float32)
                if tail_len > 0 and chunk_idx == nchunks - 1:
                    T.copy(DA_CS[i_b, i_h, S - 1], da_cs_sum)
                else:
                    T.copy(DA_CS[i_b, i_h, chunk_start+chunk_size-1], da_cs_sum)
                for n, p in T.Parallel(N, P):
                    ddA_state_passing_prereduce_frag[n, p] = (
                        states_frag[n, p] 
                        * dstates_frag[n, p] 
                        * T.exp(da_cs_sum)
                    )
                T.reduce_sum(
                    T.view(ddA_state_passing_prereduce_frag, shape=[N*P]),
                    ddA_state_passing,
                    dim=-1, clear=True,
                )
                dda_frag = T.alloc_fragment([chunk_size,], T.float32)
                for cs in T.Parallel(chunk_size):
                    dda_frag[cs] = ddA_state_passing[0]
                T.copy(dda_frag, DDA[i_b, i_h, chunk_start:chunk_start+chunk_size])

                dq_frag = T.alloc_fragment([fused_chunk_size, N], accum_dtype)
                T.gemm(dPhiO_shared, states_shared, dq_frag, transpose_B=True, clear_accum=True)
                # NOTE: Compute the contribution to ddA from applying it to q*state (part 4 of 4)
                dda_cs_prereduce_frag = T.alloc_fragment([fused_chunk_size, N], accum_dtype)
                T.copy(q_shared, dda_cs_prereduce_frag)
                for csr, n in T.Parallel(fused_chunk_size, N):
                    dda_cs_prereduce_frag[csr, n] *= dq_frag[csr, n]
                dda_cs_frag = T.alloc_fragment([chunk_size], accum_dtype)
                T.reduce_sum(T.view(dda_cs_prereduce_frag, shape=[chunk_size, R*N]), 
                             dda_cs_frag, dim=-1, clear=True)
                T.copy(dda_cs_frag, DDA_CS[i_b, i_h, chunk_start:chunk_start+chunk_size])

                dA_cs_dq_frag = T.alloc_fragment([chunk_size], T.float32)
                dA_cs_shared = T.alloc_shared([chunk_size], T.float32)

                T.copy(DA_CS[i_b, i_h, chunk_start:chunk_start+chunk_size], dA_cs_shared)
                T.copy(dA_cs_shared, dA_cs_dq_frag)
                for csr, n in T.Parallel(fused_chunk_size, N):
                    # DA_CS scales interchunk q-state contribution in backward.
                    dq_frag[csr, n] *= T.exp(dA_cs_dq_frag[csr//R])
                # NOTE: Unable to reuse dk_intrachunk_frag_dtype due to layout issue
                # (we do gemm with the transpose of dk_intrachunk_frag_dtype)
                T.copy(dq_frag, dq_shared)
                dq_combined_frag = T.alloc_fragment([fused_chunk_size, N], accum_dtype)
                T.copy(dq_shared, dq_combined_frag)
                T.gemm(lkq_masked__or__dkq_masked_shared, k_shared, dq_combined_frag, transpose_A=True, clear_accum=False)
                T.copy(dq_combined_frag, dq_shared)

                # --- Inverse Rotary for dK and dQ + dAngles ---
                angles_dk_frag = T.alloc_fragment([chunk_size, N//rotary_dim_divisor], T.float32)
                T.copy(ANGLES[i_b, chunk_start:chunk_start+chunk_size, i_h, :], angles_dk_frag)
                dk_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                dk_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                k_prerot_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                k_prerot_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    dk_first_half_frag[cs, r, n] = dk_shared[cs*R + r, n]
                    dk_second_half_frag[cs, r, n] = dk_shared[cs*R + r, N//2 + n]
                    k_prerot_first_half_frag[cs, r, n] = k_pre_rot_shared[cs*R + r, n]
                    k_prerot_second_half_frag[cs, r, n] = k_pre_rot_shared[cs*R + r, N//2 + n]
                # Compute the contribution of dk to dangle:
                dangle_dk_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], T.float32)
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    dangle_dk_frag[cs, r, n] = dk_first_half_frag[cs, r, n] * (-k_prerot_first_half_frag[cs, r, n] * T.sin(angles_dk_frag[cs, n]) - k_prerot_second_half_frag[cs, r, n] * T.cos(angles_dk_frag[cs, n])) +\
                                            dk_second_half_frag[cs, r, n] * (k_prerot_first_half_frag[cs, r, n] * T.cos(angles_dk_frag[cs, n]) - k_prerot_second_half_frag[cs, r, n] * T.sin(angles_dk_frag[cs, n]))
                T.copy(T.view(dangle_dk_frag, shape=[fused_chunk_size, N//rotary_dim_divisor]), dangle_dk__or__dq_shared)
                
                # Rotate dk_shared:
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    dk_shared[cs*R + r, n] = T.cos(angles_dk_frag[cs, n]) * dk_first_half_frag[cs, r, n] + T.sin(angles_dk_frag[cs, n]) * dk_second_half_frag[cs, r, n]
                    dk_shared[cs*R + r, N//2 + n] = -T.sin(angles_dk_frag[cs, n]) * dk_first_half_frag[cs, r, n] + T.cos(angles_dk_frag[cs, n]) * dk_second_half_frag[cs, r, n]

                dk_combined_frag = T.alloc_fragment([fused_chunk_size, N], accum_dtype)
                T.copy(dk_shared, dk_combined_frag)

                # Compute the effect of dqk_from_diag
                q_dk_frag = T.alloc_fragment([fused_chunk_size, N], accum_dtype) # Keeping q_dk_frag in accum_dtype to avoid casting instructions
                T.copy(q_pre_rot_shared, q_dk_frag) # NOTE: we need to use the pre-rotated version of q
                q_dk_frag_reshaped = T.view(q_dk_frag, [chunk_size, R, N])
                for csr_in, n in T.Parallel(fused_chunk_size, N):
                    cs = csr_in // R
                    for r_out in T.serial(R):
                        csr_out = cs*R + r_out
                        dk_combined_frag[csr_in, n] += dqk_from_diag_shared[csr_out, csr_in] * q_dk_frag_reshaped[cs, r_out, n]  
                # Copy to gmem:
                T.copy(dk_combined_frag, DK[i_b, fused_chunk_start:fused_chunk_start+fused_chunk_size, i_h, :])

                angles_dq_frag = T.alloc_fragment([chunk_size, N//rotary_dim_divisor], T.float32)
                T.copy(ANGLES[i_b, chunk_start:chunk_start+chunk_size, i_h, :], angles_dq_frag)
                dq_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                dq_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    dq_first_half_frag[cs, r, n] = dq_shared[cs*R + r, n]
                    dq_second_half_frag[cs, r, n] = dq_shared[cs*R + r, N//2 + n]
                
                # Compute the contribution of dq to dangle:
                q_prerot_first_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                q_prerot_second_half_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], dtype)
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    q_prerot_first_half_frag[cs, r, n] = q_pre_rot_shared[cs*R + r, n]
                    q_prerot_second_half_frag[cs, r, n] = q_pre_rot_shared[cs*R + r, N//2 + n]
                dangle_dq_frag = T.alloc_fragment([chunk_size, R, N//rotary_dim_divisor], T.float32)
                T.copy(dangle_dk__or__dq_shared, T.view(dangle_dq_frag, shape=[fused_chunk_size, N//rotary_dim_divisor]))
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    dangle_dq_frag[cs, r, n] += dq_first_half_frag[cs, r, n] * (-q_prerot_first_half_frag[cs, r, n] * T.sin(angles_dq_frag[cs, n]) - q_prerot_second_half_frag[cs, r, n] * T.cos(angles_dq_frag[cs, n])) +\
                                            dq_second_half_frag[cs, r, n] * (q_prerot_first_half_frag[cs, r, n] * T.cos(angles_dq_frag[cs, n]) - q_prerot_second_half_frag[cs, r, n] * T.sin(angles_dq_frag[cs, n]))
                # Sum dangle across R, and copy to gmem
                dangle_frag_reduced = T.alloc_fragment([chunk_size,  N//rotary_dim_divisor], T.float32)
                T.clear(dangle_frag_reduced)
                for cs, n in T.Parallel(chunk_size, N//rotary_dim_divisor):
                    for r in T.serial(R):
                        dangle_frag_reduced[cs, n] += dangle_dq_frag[cs, r, n]
                T.copy(dangle_frag_reduced, DANGLES[i_b, chunk_start:chunk_start+chunk_size, i_h, :])
                # Rotate dq_shared:
                for cs, r, n in T.Parallel(chunk_size, R, N//rotary_dim_divisor):
                    dq_shared[cs*R + r, n] = T.cos(angles_dk_frag[cs, n]) * dq_first_half_frag[cs, r, n] + T.sin(angles_dk_frag[cs, n]) * dq_second_half_frag[cs, r, n]
                    dq_shared[cs*R + r, N//2 + n] = -T.sin(angles_dk_frag[cs, n]) * dq_first_half_frag[cs, r, n] + T.cos(angles_dk_frag[cs, n]) * dq_second_half_frag[cs, r, n]
                T.copy(dq_shared, dq_frag)

                # Compute the effect of dqk_from_diag
                for csr_out, n in T.Parallel(fused_chunk_size, N):
                    cs = csr_out // R
                    for r_in in T.serial(R):
                        csr_in = cs*R + r_in
                        dq_frag[csr_out, n] += dqk_from_diag_shared[csr_out, csr_in] * k_pre_rot_shared[csr_in, n]
                # Copy to gmem:
                T.copy(dq_frag, DQ[i_b, fused_chunk_start:fused_chunk_start+fused_chunk_size, i_h, :])

                # --- Update Reverse-Passed State Gradient ---
                da_cs_sum_dstates = T.alloc_var(T.float32)
                if tail_len > 0 and chunk_idx == nchunks - 1:
                    T.copy(DA_CS[i_b, i_h, S - 1], da_cs_sum_dstates)
                else:
                    T.copy(DA_CS[i_b, i_h, chunk_start+chunk_size-1], da_cs_sum_dstates)
                for n, p in T.Parallel(N, P):
                    dstates_frag[n, p] *= T.exp(da_cs_sum_dstates)
                dPhiO_scaled_frag = T.alloc_fragment([fused_chunk_size, P], dtype)
                T.copy(dPhiO_shared, dPhiO_scaled_frag)
                dA_cs_dPhiO_frag = T.alloc_fragment([chunk_size], T.float32)
                T.copy(dA_cs_shared, dA_cs_dPhiO_frag)
                for csr, p in T.Parallel(fused_chunk_size, P):
                    # DA_CS applies chunk-level decay to the passed gradient state.
                    dPhiO_scaled_frag[csr, p] *= T.exp(dA_cs_dPhiO_frag[csr//R])
                T.gemm(q_shared, dPhiO_scaled_frag, dstates_frag, transpose_A=True, clear_accum=False)
                T.copy(dstates_frag, dstates_shared)

            T.copy(dPsi_acc, DMIMO_V[i_b, i_h, :, :])
            if hasD:
                T.copy(dD_frag, DD[i_b, i_h])

    return mamba_mimo_bwd_bwd_kernel


def mamba_mimo_bwd_combined(
        dout,
        q, 
        k, 
        v, 
        q_bias,
        k_bias,
        mimo_v, 
        mimo_o,
        z,
        mimo_z,
        angles,
        dA_cs,
        dA_cs_rev,
        dt,
        trap,
        D,
        segsum,
        chunk_size,
        rotary_dim_divisor,
        dtype,
        bf_threads=128,
        bf_num_stages=0,
        bb_threads=256,
        bb_num_stages=0,
        ):
    # TileLang kernel expects contiguous last-dim strides for DOUT.
    B, S, R, G, N = q.shape
    H, P = v.shape[-2], v.shape[-1]
    reduceO = mimo_o is not None

    dmimo_o = torch.empty([B, H, R, P], dtype=mimo_v.dtype, device=mimo_v.device) if reduceO else None
    states = torch.empty([B, H, math.ceil(S/chunk_size), N, P], dtype=v.dtype, device=v.device) # NOTE: states dtype is set to v.dtype
    
    if z is not None:
        dz_tilelang = torch.empty_like(v)
        dmimo_z = torch.empty([B, H, R, P], dtype=mimo_v.dtype, device=mimo_v.device)
    else:
        dz_tilelang = None
        dmimo_z = None
    qk_dot = torch.zeros([B, H, S, R, R], dtype=q.dtype, device=q.device)


    if isinstance(dtype, torch.dtype):
        dtype_str = str(dtype).replace("torch.", "")
    else:
        dtype_str = dtype
    bwd_fwd_kernel = mamba_mimo_bwd_fwd(B, S, H, G, N, P, R, 
                                             z is not None,
                                             D is not None,
                                             reduceO,
                                             chunk_size, 
                                             rotary_dim_divisor,
                                             dtype_str,
                                             bf_threads,
                                             bf_num_stages)
    bwd_fwd_kernel(
                    dout,
                    q, 
                    k, 
                    v, 
                    q_bias,
                    k_bias,
                    mimo_v, 
                    mimo_o,
                    dmimo_o,
                    states,
                    z,
                    mimo_z,
                    dz_tilelang,
                    dmimo_z,
                    angles,
                    dA_cs,
                    dA_cs_rev,
                    dt,
                    trap,
                    D,
                    qk_dot,
                    segsum,
                    )
    if reduceO:
        dmimo_o = dmimo_o.sum(dim=0)

    
    dq_tilelang = torch.empty([B, S, R, H, N], dtype=q.dtype, device=q.device)
    dk_tilelang = torch.empty([B, S, R, H, N], dtype=k.dtype, device=k.device)
    dv_tilelang = torch.empty_like(v)
    dmimo_v = torch.empty([B, H, R, P], dtype=mimo_v.dtype, device=mimo_v.device)
    dD = torch.empty([B, H], dtype=D.dtype, device=D.device) if D is not None else None
    dangles = torch.zeros([B, S, H, N//rotary_dim_divisor], dtype=angles.dtype, device=angles.device)
    dfactor = torch.zeros([B, H, S], dtype=torch.float32, device=trap.device)
    dgamma_diag = torch.zeros([B, H, S], dtype=torch.float32, device=trap.device)
    ddA = torch.zeros([B, H, S], dtype=torch.float32, device=dt.device)
    dSSdA = torch.zeros([B, H, math.ceil(S/chunk_size), chunk_size, chunk_size], dtype=torch.float32, device=dt.device)
    ddA_cs_rev = torch.zeros([B, H, S], dtype=torch.float32, device=dt.device)
    ddA_cs = torch.zeros([B, H, S], dtype=torch.float32, device=dt.device)
    
    
    bwd_bwd_kernel = mamba_mimo_bwd_bwd(B, S, H, G, N, P, R, 
                                             z is not None,
                                             D is not None,
                                             reduceO,
                                             chunk_size, 
                                             rotary_dim_divisor,
                                             dtype_str,
                                             bb_threads,
                                             bb_num_stages)
    bwd_bwd_kernel(
            dout,
            q, 
            k,
            v,
            q_bias,
            k_bias,
            mimo_v, 
            mimo_o,
            dk_tilelang.view(B, S*R, H, N), 
            dv_tilelang, 
            dmimo_v,
            states,
            dq_tilelang.view(B, S*R, H, N),
            z,
            mimo_z,
            angles,
            dA_cs,
            dA_cs_rev,
            dt,
            trap,
            dfactor,
            dgamma_diag,
            dangles,
            D,
            dD,
            qk_dot,
            ddA,
            dSSdA,
            ddA_cs_rev,
            ddA_cs,
            segsum,
            )
    
    if G == 1:
        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dq_tilelang = dq_tilelang.sum(dim=3, keepdim=True)
        dk_tilelang = dk_tilelang.sum(dim=3, keepdim=True)
        dmimo_v = dmimo_v.sum(dim=0)
        dmimo_z = dmimo_z.sum(dim=0) if dmimo_z is not None else None
        dD = dD.sum(dim=0) if dD is not None else None
    elif G == H:
        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dmimo_v = dmimo_v.sum(dim=0)
        dmimo_z = dmimo_z.sum(dim=0) if dmimo_z is not None else None
        dD = dD.sum(dim=0) if dD is not None else None
    else:
        raise ValueError(f"G value of {G} is not currently supported!")

    ddt, dtrap = bwd_dtrap_ddt_triton(
        trap, dt, dfactor, dgamma_diag, chunk_size
    )

    ddA += bwd_dadt_fused_triton(
        dSSdA, segsum, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size
    )


    return (dq_tilelang, dk_tilelang, dv_tilelang, 
            ddA, ddt, dtrap, dq_bias_tilelang, dk_bias_tilelang,
            dmimo_v, dmimo_z, dmimo_o, dangles, 
            dD, dz_tilelang)
