"""Mamba-3 Triton Autograd Wrapper

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
import triton

# Import kernels
from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd
from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import compute_dzdo, compute_dqkv, compute_dqktheta, compute_ddt_dtrap_dinput_states
from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd, angle_dt_bwd


def _triton_alloc_fn(size: int, alignment: int, stream: Optional[int]):
    """Allocator for Triton runtime memory (TMA descriptors, scratch)."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


# Set allocator immediately at import time.
try:
    triton.set_allocator(_triton_alloc_fn)
except Exception:
    pass  # Allocator may already be set


@dataclass(frozen=True)
class Mamba3Output:
    """Container for Mamba-3 outputs and optional intermediates.
    
    Attributes:
        out: Main output tensor (batch, seqlen, nheads, headdim_v)
        final_angle_state: Final angle state (num_sequences, nheads, headdim_angles)
        final_ssm_state: Final SSM state (num_sequences, nheads, headdim_v, headdim_qk)
        final_k_state: Final K state (num_sequences, nheads, headdim_qk)
        final_v_state: Final V state (num_sequences, nheads, headdim_v)
    """
    out: Tensor
    final_angle_state: Optional[Tensor] = None
    final_ssm_state: Optional[Tensor] = None
    final_k_state: Optional[Tensor] = None
    final_v_state: Optional[Tensor] = None

class _Mamba3Function(torch.autograd.Function):
    """Custom autograd function for Mamba-3 with Triton kernels."""
    
    @staticmethod
    def forward(
        ctx,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        ADT: Tensor,
        DT: Tensor,
        Trap: Tensor,
        Q_bias: Tensor,
        K_bias: Tensor,
        Angles: Tensor,
        D: Optional[Tensor],
        Z: Optional[Tensor],
        Input_Angle_State: Optional[Tensor],
        Input_SSM_State: Optional[Tensor],
        Input_K_State: Optional[Tensor],
        Input_V_State: Optional[Tensor],
        cu_seqlens: Optional[Tensor],
        chunk_size: int,
        return_final_states: bool,
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass: call Triton kernel and save tensors for backward."""
        
        try:
            triton.set_allocator(_triton_alloc_fn)
        except Exception:
            pass
        
        needs_backward = any(ctx.needs_input_grad)
        has_varlen = cu_seqlens is not None

        all_states_present = (Input_SSM_State is not None) and (Input_K_State is not None) and (Input_V_State is not None) and (Input_Angle_State is not None)
        all_states_absent = (Input_SSM_State is None) and (Input_K_State is None) and (Input_V_State is None) and (Input_Angle_State is None)

        assert all_states_present or all_states_absent, "Input states must be provided together or all be None."
        
        Angles_Cumsum, Final_Angle_State = angle_dt_fwd(
            Angles, DT, 
            init_state=Input_Angle_State, 
            chunk_size=chunk_size, 
            return_output_state=True,
            cu_seqlens=cu_seqlens,
        )

        Input_States = (
            (Input_SSM_State, Input_K_State, Input_V_State)
            if Input_SSM_State is not None
            else None
        )

        Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, Gamma, Final_States = mamba3_siso_fwd(
            Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles_Cumsum, D, Z, Input_States,
            chunk_size=chunk_size,
            store_states_adt_outv=needs_backward,
            return_final_states=return_final_states,
            cu_seqlens=cu_seqlens,
        )

        Final_SSM_State = Final_States[0] if Final_States is not None else None
        Final_K_State = Final_States[1] if Final_States is not None else None
        Final_V_State = Final_States[2] if Final_States is not None else None
        
        if needs_backward:
            ctx.chunk_size = chunk_size
            ctx.has_D = D is not None
            ctx.has_Z = Z is not None
            ctx.has_input_state = Input_SSM_State is not None
            ctx.return_final_states = return_final_states
            ctx.has_varlen = has_varlen
            
            # Save tensors - use empty tensor placeholders for None values
            D_save = D if D is not None else torch.empty((), device=Q.device)
            Z_save = Z if Z is not None else torch.empty((), device=Q.device)
            Input_SSM_State_save = Input_SSM_State if Input_SSM_State is not None else torch.empty((), device=Q.device)
            Input_K_State_save = Input_K_State if Input_K_State is not None else torch.empty((), device=Q.device)
            Input_V_State_save = Input_V_State if Input_V_State is not None else torch.empty((), device=Q.device)
            Final_SSM_State_save = Final_SSM_State if Final_SSM_State is not None else torch.empty((), device=Q.device)
            cu_seqlens_save = cu_seqlens if cu_seqlens is not None else torch.empty((), device=Q.device, dtype=torch.int32)
            
            ctx.save_for_backward(
                Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, Angles_Cumsum,
                D_save, Z_save, Input_SSM_State_save, Input_K_State_save, Input_V_State_save,
                Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, Gamma,
                Final_SSM_State_save, cu_seqlens_save
            )
        else:
            ctx.chunk_size = chunk_size
            ctx.has_D = D is not None
            ctx.has_Z = Z is not None
            ctx.has_input_state = Input_SSM_State is not None
            ctx.return_final_states = return_final_states
            ctx.has_varlen = has_varlen
            ctx.save_for_backward()
        
        if return_final_states:
            return Out, Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State
        return Out
    
    @staticmethod
    def backward(
        ctx, 
        grad_out: Optional[Tensor] = None, 
        grad_final_angle_state: Optional[Tensor] = None,
        grad_final_ssm_state: Optional[Tensor] = None, 
        grad_final_k_state: Optional[Tensor] = None, 
        grad_final_v_state: Optional[Tensor] = None
    ) -> tuple:
        """Backward pass: compute gradients using Triton backward kernels."""
        
        try:
            triton.set_allocator(_triton_alloc_fn)
        except Exception:
            pass
        
        if len(ctx.saved_tensors) == 0:
            raise RuntimeError(
                "Backward called but forward ran without gradient tracking. "
                "Ensure inputs require grad or run under torch.enable_grad()."
            )
        if grad_out is None and grad_final_ssm_state is None and grad_final_k_state is None and grad_final_v_state is None and grad_final_angle_state is None:
            raise RuntimeError("No gradients provided for backward pass.")

        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, Angles_Cumsum,
        D_save, Z_save, Input_SSM_State_save, Input_K_State_save, Input_V_State_save,
        Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, Gamma,
        Final_SSM_State_save, cu_seqlens_save) = ctx.saved_tensors
        
        D = D_save if ctx.has_D else None
        Z = Z_save if ctx.has_Z else None
        Input_SSM_State = Input_SSM_State_save if ctx.has_input_state else None
        Input_K_State = Input_K_State_save if ctx.has_input_state else None
        Input_V_State = Input_V_State_save if ctx.has_input_state else None
        cu_seqlens = cu_seqlens_save if ctx.has_varlen else None
        
        if grad_out is None:
            grad_out = torch.zeros_like(Out)
        
        # Step 1: Compute dZ and scale grad_out if Z gating is present
        if Z is not None:
            dZ, grad_out_scaled = compute_dzdo(
                grad_out, Z, Out_v, chunk_size=ctx.chunk_size
            )
        else:
            dZ = None
            grad_out_scaled = grad_out

        # Step 2: Compute main gradients (dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, dInput_SSM_State)
        dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, dInput_SSM_State = compute_dqkv(
            q=Q_rot,
            k=K_scaled,
            v=V,
            da_cs=DA_CS,
            da_cs_sum=DA_CS_SUM,
            qk_dot=QK_dot,
            SSM_States=SSM_States,
            do=grad_out_scaled,
            d_ossm_state=grad_final_ssm_state,
            d_ov_state=grad_final_v_state,
            D=D,
            chunk_size=ctx.chunk_size,
            has_input_state=ctx.has_input_state,
            Cu_Seqlens=cu_seqlens,
        )
        
        # Step 3: Compute gradients through rotary embeddings and biases
        dQ, dK, dQ_bias, dK_bias, dAngles_Cumsum, dScale, dGamma = compute_dqktheta(
            q=Q,
            k=K,
            scale=Scale,
            gamma=Gamma,
            q_bias=Q_bias,
            k_bias=K_bias,
            angles=Angles_Cumsum,
            dq_in=dQ_mid,
            dk_in=dK_mid,
            dqk=dQK_dot,
            d_ok_state=grad_final_k_state,
            chunk_size=ctx.chunk_size,
            Cu_Seqlens=cu_seqlens,
        )
        
        # Step 4: Compute dDT, dTrap, and input state gradients
        dDT, dTrap, dInput_SSM_State_final, dInput_K_State, dInput_V_State = compute_ddt_dtrap_dinput_states(
            dscale=dScale,
            dgamma=dGamma,
            dt=DT,
            trap=Trap.float(),
            d_issm_state=dInput_SSM_State if ctx.has_input_state else None,
            input_k_state=Input_K_State,
            input_v_state=Input_V_State,
            Cu_Seqlens=cu_seqlens,
        )
        
        # Step 5: Compute gradients through angle_dt cumsum
        dAngles, dDT_angle, dInput_Angle_State = angle_dt_bwd(
            grad_out=dAngles_Cumsum,
            angle=Angles,
            dt=DT,
            has_init_state=ctx.has_input_state,
            chunk_size=ctx.chunk_size,
            grad_output_state=grad_final_angle_state if ctx.return_final_states else None,
            cu_seqlens=cu_seqlens,
        )
        
        # Accumulate DT gradients from angle_dt backward
        dDT = dDT + dDT_angle
        
        if ctx.has_input_state:
            dInput_SSM_State = dInput_SSM_State_final
        else:
            dInput_SSM_State = None
            dInput_K_State = None
            dInput_V_State = None
            dInput_Angle_State = None
        
        return (
            dQ,                     # Q
            dK,                     # K
            dV,                     # V
            dADT,                   # ADT
            dDT,                    # DT
            dTrap,                  # Trap
            dQ_bias,                # Q_bias
            dK_bias,                # K_bias
            dAngles,                # Angles
            dD,                     # D
            dZ,                     # Z
            dInput_Angle_State,     # Input_Angle_State
            dInput_SSM_State,       # Input_SSM_State
            dInput_K_State,         # Input_K_State
            dInput_V_State,         # Input_V_State
            None,                   # cu_seqlens (not differentiable)
            None,                   # chunk_size (not differentiable)
            None,                   # return_final_states (not differentiable)
        )


def mamba3_siso_combined(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Trap: Tensor,
    Q_bias: Tensor,
    K_bias: Tensor,
    Angles: Tensor,
    D: Optional[Tensor] = None,
    Z: Optional[Tensor] = None,
    Input_States: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
    chunk_size: int = 64,
    return_final_states: bool = False,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor | Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Mamba-3 attention with Triton kernels and automatic differentiation.

    This is the main entry point for Mamba-3 forward and backward passes using
    optimized Triton kernels. Supports GQA (grouped-query attention), rotary
    position embeddings, optional gating, skip connections, state passing
    for recurrent inference, and variable-length sequences.

    Internally computes cumulative angles: Angles_Cumsum = cumsum(Angles * DT) mod 2π

    Args:
        Q: Query tensor             (batch, seqlen, nheads_qk, headdim_qk)
        K: Key tensor               (batch, seqlen, nheads_qk, headdim_qk)
        V: Value tensor             (batch, seqlen, nheads, headdim_v)
        ADT: Decay factor A * dt    (batch, nheads, seqlen)
        DT: Time delta tensor dt    (batch, nheads, seqlen)
        Trap: Trapezoidal factor    (batch, nheads, seqlen)
            Mixing factor in [0, 1] for trapezoidal discretization.
        Q_bias: Query bias          (nheads, headdim_qk)
        K_bias: Key bias            (nheads, headdim_qk)
        Angles: Rotary angle rates  (batch, seqlen, nheads, headdim_angles)
            Raw angle values that get accumulated via cumsum(Angles * DT).
            If headdim_angles < headdim_qk // 2, remaining dims are unrotated.
        D: Skip connection          (nheads,)
            Optional per-head skip connection weight applied to V.
        Z: Gating tensor            (batch, seqlen, nheads, headdim_v)
            Optional gating applied as: out = out * silu(Z).
        Input_States: Optional initial state tuple for recurrent inference.
            Angle State:            (num_sequences, nheads, headdim_angles)
            SSM State:              (num_sequences, nheads, headdim_v, headdim_qk)
            K State:                (num_sequences, nheads, headdim_qk)
            V State:                (num_sequences, nheads, headdim_v)
        chunk_size: Chunk size for chunked state computation (default: 64).
        return_final_states: If True, return final states for recurrent inference.
        cu_seqlens: Cumulative sequence lengths for variable-length support.
            Shape: (num_sequences + 1,), dtype: torch.int32.
            Example: [0, 128, 256, 512] for 3 sequences of lengths 128, 128, 256.
            When using cu_seqlens, batch must be 1 and the seqlen dimension
            contains all sequences concatenated.

    Returns:
        If return_final_states=False:
            out: Output tensor      (batch, seqlen, nheads, headdim_v)
        If return_final_states=True:
            Tuple of:
                out: Output tensor              (batch, seqlen, nheads, headdim_v)
                final_angle_state: Angle state  (num_sequences, nheads, headdim_angles)
                final_ssm_state: SSM state      (num_sequences, nheads, headdim_v, headdim_qk)
                final_k_state: K state          (num_sequences, nheads, headdim_qk)
                final_v_state: V state          (num_sequences, nheads, headdim_v)

    Notes:
        - For GQA: nheads must be divisible by nheads_qk.
        - headdim_qk and headdim_v must be powers of two for TMA compatiblity,
        - Variable-length mode (cu_seqlens is not None) requires batch == 1.
        - num_sequences = batch for batched mode, len(cu_seqlens)-1 for varlen mode.


    Performance Notes:
        The kernel is optimized for:
            nheads_qk=1, nheads=32, headdim_qk=128, headdim_v=64, chunk_size=64.
    """
    
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    
    assert nheads % nheads_qk == 0, f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})"
    assert headdim_qk % 2 == 0, f"headdim_qk ({headdim_qk}) must be even for rotary embeddings"
    
    # Varlen mode checks
    has_varlen = cu_seqlens is not None
    if has_varlen:
        if batch != 1:
            raise ValueError(f"Batch size must be 1 with variable-length sequences (cu_seqlens), got {batch}.")
    
    Input_Angle_State, Input_SSM_State, Input_K_State, Input_V_State = (
        Input_States if Input_States is not None else (None, None, None, None)
    )

    all_states_present = (Input_SSM_State is not None) and (Input_K_State is not None) and (Input_V_State is not None) and (Input_Angle_State is not None)
    all_states_absent = (Input_SSM_State is None) and (Input_K_State is None) and (Input_V_State is None) and (Input_Angle_State is None)
    assert all_states_present or all_states_absent, "Input states must be provided together or all be None."

    # Typecast all derived tensors to bf16.
    # ADT, DT should be in fp32 for stability
    # Q_bias, K_bias, D should be in fp32 as they are model parameters
    Q = Q.to(torch.bfloat16)
    K = K.to(torch.bfloat16)
    V = V.to(torch.bfloat16)
    Trap = Trap.to(torch.bfloat16)
    Angles = Angles.to(torch.bfloat16)
    if Z is not None:
        Z = Z.to(torch.bfloat16)

    return _Mamba3Function.apply(
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z,
        Input_Angle_State, Input_SSM_State, Input_K_State, Input_V_State, cu_seqlens, chunk_size, return_final_states
    )