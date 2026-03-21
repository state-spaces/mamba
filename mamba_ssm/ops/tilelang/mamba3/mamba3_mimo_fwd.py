"""
Tilelang implementation of Mamba3 forward kernel,
with MIMO support.

Copyright (c) 2026, Dao AI Lab, Goombalab

"""

import torch
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
    S,
    H,
    G,
    N,
    P,
    R,
    hasZ,
    hasD,
    reduceO,
    return_final_state=False,
    chunk_size: int = 16,
    rotary_dim_divisor = 4,
    dtype: str = 'bfloat16',
    threads: int = 128,
    num_stages: int = 0,
) -> torch.Tensor:

    accum_dtype = 'float32'

    # Block sizes for K and V dimensions - use full dimensions (no tiling)
    assert S % chunk_size == 0, "Sequence length must be divisible by chunk_size"

    nchunks = tilelang.cdiv(S, chunk_size)
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
            Z: T.Tensor([B, S, H, P], dtype),  # type: ignore
            D: T.Tensor([H], T.float32),  # type: ignore
            MIMO_Z: T.Tensor([H, R, P], T.float32), # type: ignore
            ANGLES: T.Tensor([B, S, H, N//rotary_dim_divisor], T.float32), # type: ignore
            DA_CS: T.Tensor([B, H, S], T.float32), # type: ignore
            DA_CS_REV: T.Tensor([B, H, S], T.float32), # type: ignore
            DT: T.Tensor([B, H, S], T.float32), # type: ignore
            TRAP: T.Tensor([B, H, S], dtype), # type: ignore
            SEGSUM: T.Tensor([B, H, nchunks, chunk_size, chunk_size], T.float32), # type: ignore

            FINAL_STATE: T.Tensor([B, H, N, P], T.float32),  # type: ignore
            FINAL_K: T.Tensor([B, R, H, N], dtype)  # type: ignore
            ):
        """
        Overview:
            Fused chunked forward pass that combines MIMO projections with recurrent state updates.
            Computes interchunk and intrachunk contributions with optional D and Z paths,
            then writes output activations.

        Inputs:
            - Activations: Q, K, V.
            - Projection parameters/biases: MIMO_V (Psi), MIMO_O (Phi), optional MIMO_Z (Zeta), ANGLES,
              and Q_BIAS/K_BIAS.
            - Optional modifiers: Z, and D.
            - Discretization tensors: DA_CS, DA_CS_REV, DT, TRAP, and SEGSUM.

        Outputs:
            - O: fused forward output activations.
            - FINAL_STATE: final recurrent states (if return_final_state is True).
            - FINAL_K: final K tensor (if return_state is True, for use in decode)

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
            q_bias_frag = T.alloc_fragment([R, N], dtype)
            k_bias_frag = T.alloc_fragment([R, N], dtype)

            angles_shared = T.alloc_shared([chunk_size, N], dtype)

            PsiV_shared = T.alloc_shared([fused_chunk_size, P], dtype)
            qs_shared = T.alloc_shared([fused_chunk_size, P], dtype)
            o_shared = T.alloc_shared([chunk_size, P], dtype)
            v_shared = T.alloc_shared([chunk_size, P], dtype)
            states_accum_cast_shared = T.alloc_shared([N, P], dtype)
            qk_intrachunk_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], dtype)
            qk_dot_full_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], dtype)

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
                qk_dot_full_shared: tilelang.layout.make_swizzled_layout(qk_dot_full_shared),
                qk_intrachunk_shared: tilelang.layout.make_swizzled_layout(qk_intrachunk_shared),
            })
            T.use_swizzle(10, "row")

            T.no_set_max_nreg()

            # --- Per-Head Constants / Running State ---
            states_frag = T.alloc_fragment([N, P], accum_dtype)
            T.clear(states_frag)

            phi_frag_intrachunk = T.alloc_fragment([R, P], dtype=dtype)
            if reduceO:
                T.copy(MIMO_O[i_h, :, :], phi_frag_intrachunk)
            Psi_frag = T.alloc_fragment([R, P], dtype)
            T.copy(MIMO_V[i_h, :, :], Psi_frag)

            T.copy(Q_BIAS[i_h, :, :], q_bias_frag)
            T.copy(K_BIAS[i_h, :, :], k_bias_frag)

            # --- Chunk Loop ---
            for i in T.Pipelined(0, nchunks, num_stages=num_stages):
                chunk_start = i * chunk_size

                # --- Discretization Factors (Shifted Gamma + Trap Scale) ---
                trap_shifted_frag = T.alloc_fragment([chunk_size], T.float32)
                T.copy(TRAP[i_b, i_h, chunk_start+1: chunk_start+chunk_size+1], trap_shifted_frag)
                dt_shifted_frag = T.alloc_fragment([chunk_size], dtype)
                T.copy(DT[i_b, i_h, chunk_start+1: chunk_start+chunk_size+1], dt_shifted_frag)
                shifted_gamma_frag = T.alloc_fragment([chunk_size], dtype)
                for cs in T.Parallel(chunk_size):
                    shifted_gamma_frag[cs] = T.if_then_else(chunk_start + cs < (S - 1), 
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

                # --- Cache Diagonal qk_dot Path ---
                # Keep full qk_dot in shared memory because we reuse same-step R x R blocks later.
                qk_dot_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=accum_dtype)
                T.gemm(q_shared, k_shared, qk_dot_frag, transpose_B=True, clear_accum=True)
                T.copy(qk_dot_frag, qk_dot_full_shared)
                # Option B: extremely slow
                # qk_dot_frag = T.alloc_fragment([chunk_size, R, R], dtype=accum_dtype)
                # T.clear(qk_dot_frag)
                # for cs, r_out, r_in in T.Parallel(chunk_size, R, R):
                #     for n in T.serial(N):
                #         qk_dot_frag[cs, r_out, r_in] += (
                #             q_frag[cs, r_out, n] * k_frag[cs, r_in, n]
                #         )
                # T.copy(T.view(qk_dot_frag, shape=[fused_chunk_size, R]), qk_dot_shared)
                # NOTE ("option C"): The following fails Tilelang compilation:
                # qk_predot_frag = T.alloc_fragment([chunk_size, R, R, N], dtype)
                # for cs, r_out, r_in, n in T.Parallel(chunk_size, R, R, N):
                #     qk_predot_frag[cs, r_out, r_in, n] = q_frag[cs, r_out, n] * k_frag[cs, r_in, n]
                # qk_dot_frag = T.alloc_fragment([chunk_size, R, R], dtype)
                # T.reduce_sum(qk_predot_frag, qk_dot_frag, dim=-1, clear=True)
                # T.copy(T.view(qk_dot_frag, shape=[fused_chunk_size, R]), qk_dot_shared)

                # --- Rotary Q + Interchunk Contribution ---
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

                if i == nchunks - 1 and return_final_state:
                    seq_boundary = T.min(chunk_start + chunk_size, S) - chunk_start
                    for csr, n in T.Parallel(fused_chunk_size, N):
                        if csr >= (seq_boundary - 1) * R and csr < seq_boundary * R:  # Only copy the last chunk's R rows to FINAL_K
                            FINAL_K[i_b, csr % R, i_h, n] = k_shared[csr, n]

                k_trap_scaled_frag = T.alloc_fragment([fused_chunk_size, N], dtype)
                T.copy(k_shared, k_trap_scaled_frag)
                for csr, n in T.Parallel(fused_chunk_size, N):
                    k_trap_scaled_frag[csr, n] *= trap_scale_shared[csr//R]
                T.copy(k_trap_scaled_frag, k_shared)

                qk_intrachunk_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=accum_dtype)
                T.gemm(q_shared, k_shared, qk_intrachunk_frag, transpose_B=True, clear_accum=True)

                # Strictly causal mask over chunk steps (exclude same-step diagonal).
                da_cs__or__exp_da_cs_shared = T.alloc_shared([chunk_size], T.float32)
                T.copy(DA_CS[i_b, i_h, chunk_start:chunk_start+chunk_size], da_cs__or__exp_da_cs_shared)
                qk_intrachunk_masked_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], dtype=dtype)
                for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size):
                    qk_intrachunk_masked_frag[csr_i, csr_j] = T.if_then_else(
                                                csr_i//R > csr_j//R, # NOTE: we do indeed want to exclude the diagonal
                                                qk_intrachunk_frag[csr_i, csr_j] 
                                                * T.exp(SEGSUM[i_b, i_h, i, csr_i//R, csr_j//R]),
                                                0.0
                                            )

                # Exponentiate da_cs__or__exp_da_cs_shared so that later usage does not have to:
                for cs in T.Parallel(chunk_size):
                    da_cs__or__exp_da_cs_shared[cs] = T.exp(da_cs__or__exp_da_cs_shared[cs])

                exp_da_cs_frag = T.alloc_fragment([chunk_size], dtype=T.float32)
                T.copy(da_cs__or__exp_da_cs_shared, exp_da_cs_frag)
                for csr, p in T.Parallel(fused_chunk_size, P):
                    o_mimo_accum_frag[csr, p] *= exp_da_cs_frag[csr//R]

                # NOTE: if we gemm with qk_intrachunk_masked_frag the compiler will
                # error with layout issue if threads != 128:
                # Copy via shared memory to satisfy layout constraints before GEMM.
                T.copy(qk_intrachunk_masked_frag, qk_intrachunk_shared)
                # Adding the two intermediate outputs together (interchunk += intrachunk)
                T.gemm(qk_intrachunk_shared, PsiV_shared, o_mimo_accum_frag, clear_accum=False)

                # --- Add Diagonal Terms (qk_dot and optional D) ---
                qkdot_psiv_frag = T.alloc_fragment([chunk_size, R, P], dtype=dtype)
                T.clear(qkdot_psiv_frag)
                for cs, r_out, p in T.Parallel(chunk_size, R, P):
                    for r_in in T.serial(R):
                        qkdot_psiv_frag[cs, r_out, p] += qk_dot_full_shared[cs * R + r_out, cs * R + r_in] * PsiV_shared[cs * R + r_in, p]                    
                    qkdot_psiv_frag[cs, r_out, p] *= gamma_frag[cs] # Apply shifted gamma

                if hasD:
                    PsiV_D_frag = T.alloc_fragment([chunk_size, R, P], T.float32)
                    for cs, r, p in T.Parallel(chunk_size, R, P):
                        PsiV_D_frag[cs, r, p] = PsiV_shared[cs * R + r, p]
                    D_var = T.alloc_var(T.float32)
                    T.copy(D[i_h], D_var)
                    for cs, r_out, p in T.Parallel(chunk_size, R, P):
                        qkdot_psiv_frag[cs, r_out, p] += D_var * PsiV_D_frag[cs, r_out, p]
                qkdot_psiv_reshaped_frag = T.view(qkdot_psiv_frag, shape=[fused_chunk_size, P])
                for csr, p in T.Parallel(fused_chunk_size, P):
                    o_mimo_accum_frag[csr, p] += qkdot_psiv_reshaped_frag[csr, p]

                # --- Optional Z Gating + Down-Projection ---
                if reduceO:
                    if hasZ:
                        z_frag = T.alloc_fragment([chunk_size, P], dtype)
                        T.copy(Z[i_b, chunk_start:chunk_start+chunk_size, i_h, :], z_frag)
                        z_expanded_frag = T.alloc_fragment([chunk_size, R, P], dtype)
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            # Apply SiLU to z_expanded_frag[cs, r, p]:
                            o_gated = z_frag[cs, p] * MIMO_Z[i_h, r, p] * 0.5
                            z_expanded_frag[cs, r, p] = o_gated * T.tanh(o_gated) + o_gated

                    lqk_PsiV_reshaped_frag = T.view(o_mimo_accum_frag, shape=[chunk_size, R, P])
                    if hasZ:
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
                T.copy(DA_CS[i_b, i_h, chunk_start+chunk_size-1], da_cs_sum)
                for n, p in T.Parallel(N, P):
                    states_frag[n, p] *= T.exp(da_cs_sum)
                T.gemm(k_state_frag, PsiV_shared, states_frag, transpose_A=True, clear_accum=False)
            
            # --- Save Last State (if applicable) ---
            if return_final_state:
                T.copy(states_frag, FINAL_STATE[i_b, i_h, :, :])
                

    return mamba_mimo_fwd_kernel


def mamba_mimo_forward(q, k, v, 
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
                       return_state=False,
                       threads=128, 
                       num_stages=0):
    B, S, R, G, N = q.shape
    H, P = v.shape[-2], v.shape[-1]
    if isinstance(dtype, torch.dtype):
        tl_dtype = str(dtype).replace("torch.", "")
    else:
        tl_dtype = dtype
    reduceO = mimo_o is not None
    kernel = mamba_mimo_fwd(B, S, H, G, N, P, R, 
                                       z is not None, 
                                       D is not None, 
                                       reduceO,
                                       return_final_state=return_state,
                                       chunk_size=chunk_size, 
                                       rotary_dim_divisor=rotary_dim_divisor, 
                                       dtype=tl_dtype, 
                                       threads=threads, 
                                       num_stages=num_stages)
    # print(kernel.get_kernel_source()) # NOTE: prints compiled CUDA code
    if reduceO:
        o = torch.empty((B, S, H, P), device='cuda', dtype=dtype)
    else:
        o = torch.empty((B, S, R, H, P), device='cuda', dtype=dtype)
    # Kernel always declares all tensor parameters; pass dummies for None args
    mimo_o_arg = mimo_o if reduceO else torch.empty((H, R, P), device=q.device, dtype=torch.float32)
    z_arg = z if z is not None else torch.empty((B, S, H, P), device=q.device, dtype=dtype)
    D_arg = D if D is not None else torch.empty((H,), device=q.device, dtype=torch.float32)
    mimo_z_arg = mimo_z if mimo_z is not None else torch.empty((H, R, P), device=q.device, dtype=torch.float32)

    h = torch.empty((B, H, N, P), device='cuda', dtype=torch.float32) if return_state else None
    k_final = torch.empty((B, R, H, N), device='cuda', dtype=dtype) if return_state else None

    kernel( q,
            k,
            v, o,
            q_bias, k_bias,
            mimo_v, mimo_o_arg,
            z_arg, D_arg, mimo_z_arg,
            angles,
            dA_cs,
            dA_cs_rev,
            dt,
            trap,
            segsum,
            h,
            k_final
            )
    return o, h, k_final
