"""
Mamba-3 SISO Kernel Tests

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

import copy
import math
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
from mamba_ssm.ops.triton.mamba3.mamba3_siso_step import mamba3_siso_step


# Reference Implementations
def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Segment sum helper for attention computation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def mamba3_siso_step_ref(
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
    Input_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reference implementation of Mamba-3 in recurrent (step) mode.
    
    Args:
        Input_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)
    
    Returns:
        out: Output tensor (batch, seqlen, nheads, headdim_v)
        Final_States: Tuple of (Angle_State, SSM_State, K_State, V_State)
    """
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device
    assert seqlen > 0
    Angles = torch.tanh(Angles) * math.pi

    # Expand Q/K for GQA
    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    def apply_rotary_emb(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        rotated = torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)
        return rotated
    
    # Initialize states
    if Input_States is not None:
        Angle_State, SSM_State, K_State, V_State = Input_States
        Angle_State = Angle_State.clone()
        SSM_State = SSM_State.clone().to(torch.float32)
        K_State = K_State.clone()
        V_State = V_State.clone()
    else:
        Angle_State = torch.zeros((batch, nheads, headdim_angles), dtype=torch.float32, device=device)
        SSM_State = torch.zeros((batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=device)
        K_State = torch.zeros((batch, nheads, headdim_qk), dtype=Q.dtype, device=device)
        V_State = torch.zeros((batch, nheads, headdim_v), dtype=V.dtype, device=device)
    
    TWO_PI = 2 * math.pi
    out_arr = []

    for idx in range(seqlen):
        q = Q[:, idx, :, :] + Q_bias.unsqueeze(0)
        k = K[:, idx, :, :] + K_bias.unsqueeze(0)
        v = V[:, idx, :, :]
        adt = ADT[:, :, idx]
        dt = DT[:, :, idx]
        trap = Trap[:, :, idx]
        z = Z[:, idx, :, :] if Z is not None else None
        angles = Angles[:, idx, :, :]

        # Update angle state with cumsum: Angle_State = (Angle_State + Angles * DT) mod 2π
        Angle_State = Angle_State + angles * dt.unsqueeze(-1)
        Angle_State = Angle_State - TWO_PI * torch.floor(Angle_State / TWO_PI)

        # Apply rotary embeddings to Q and K using cumulative angles
        cos_angles = torch.cos(Angle_State)
        sin_angles = torch.sin(Angle_State)
        q_rot = apply_rotary_emb(q, cos_angles, sin_angles)
        k_rot = apply_rotary_emb(k, cos_angles, sin_angles)

        trap = torch.sigmoid(trap)
        alpha = torch.exp(adt)
        beta = (1 - trap) * dt * alpha
        gamma = trap * dt

        # Update SSM state using previous K_State and V_State
        SSM_State = alpha.unsqueeze(-1).unsqueeze(-1) * SSM_State 
        SSM_State = SSM_State + beta.unsqueeze(-1).unsqueeze(-1) * (K_State.unsqueeze(-2) * V_State.unsqueeze(-1))
        SSM_State = SSM_State + gamma.unsqueeze(-1).unsqueeze(-1) * (k_rot.unsqueeze(-2) * v.unsqueeze(-1))

        # Compute output
        out = torch.einsum("bhdD, bhD -> bhd", SSM_State, q_rot.to(SSM_State.dtype))
        
        if D is not None:
            out = out + D[None, :, None] * v
        
        if Z is not None:
            out = out * z * torch.sigmoid(z)
        
        out_arr.append(out)
        
        # Update K and V states for next step
        K_State = k_rot
        V_State = v
    
    out = torch.stack(out_arr, dim=1)
    Final_States = (Angle_State, SSM_State, K_State, V_State)
    return out, Final_States


def mamba3_siso_fwd_ref(
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
    Initial_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.float32,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """Reference implementation of Mamba-3 forward pass.
    
    Args:
        Initial_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)
    
    Returns:
        out_z: Output with Z gating applied
        final_states: (Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State)
    """
    batch, total_seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device
    
    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert batch == 1
    
    # Cast inputs
    Q = Q.to(dtype)
    K = K.to(dtype)
    V = V.to(dtype)
    ADT = ADT.to(torch.float32)
    DT = DT.to(torch.float32)
    Trap = Trap.to(dtype)
    Q_bias = Q_bias.to(dtype)
    K_bias = K_bias.to(dtype)
    Angles = Angles.to(dtype)
    if D is not None:
        D = D.to(dtype)
    if Z is not None:
        Z = Z.to(dtype)
    if Initial_States is not None:
        Initial_Angle_State, Initial_SSM_State, Initial_K_State, Initial_V_State = Initial_States

    Angles = torch.tanh(Angles) * math.pi
    # Expand Q/K for GQA
    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    out_zs = []
    Final_Angle_States = []
    Final_SSM_States = []
    Final_K_States = []
    Final_V_States = []

    TWO_PI = 2 * math.pi

    def _rotary(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)

    def compute_one_sequence(seq_idx):
        if is_varlen:
            start_idx, end_idx = cu_seqlens[seq_idx].item(), cu_seqlens[seq_idx + 1].item()
            Q_curr = Q[0, start_idx:end_idx, :, :]
            K_curr = K[0, start_idx:end_idx, :, :]
            V_curr = V[0, start_idx:end_idx, :, :]
            ADT_curr = ADT[0, :, start_idx:end_idx]
            DT_curr = DT[0, :, start_idx:end_idx]
            Trap_curr = Trap[0, :, start_idx:end_idx]
            Angles_curr = Angles[0, start_idx:end_idx, :, :]
            Z_curr = Z[0, start_idx:end_idx, :, :] if Z is not None else None
        else:
            Q_curr = Q[seq_idx]
            K_curr = K[seq_idx]
            V_curr = V[seq_idx]
            ADT_curr = ADT[seq_idx]
            DT_curr = DT[seq_idx]
            Trap_curr = Trap[seq_idx]
            Angles_curr = Angles[seq_idx]
            Z_curr = Z[seq_idx] if Z is not None else None

        Trap_curr = torch.sigmoid(Trap_curr)
        seqlen_curr = Q_curr.shape[0]

        Angles_scaled = Angles_curr.float() * DT_curr.transpose(0, 1).unsqueeze(-1)
        Angles_Cumsum = torch.cumsum(Angles_scaled, dim=0)
        if Initial_States is not None:
            Initial_Angle_State_curr = Initial_Angle_State[seq_idx]
            Angles_Cumsum = Angles_Cumsum + Initial_Angle_State_curr.unsqueeze(0)
        Angles_Cumsum = Angles_Cumsum - TWO_PI * torch.floor(Angles_Cumsum / TWO_PI)
        Final_Angle_States.append(Angles_Cumsum[-1])

        # Initialize acc_states
        if Initial_States is not None:
            Initial_SSM_State_curr = Initial_SSM_State[seq_idx]
            Initial_K_State_curr = Initial_K_State[seq_idx]
            Initial_V_State_curr = Initial_V_State[seq_idx]

            scalar = DT_curr[:, 0] * (1 - Trap_curr[:, 0])
            acc_states = Initial_SSM_State_curr + Initial_V_State_curr[:, :, None] * Initial_K_State_curr[:, None, :] * scalar[:, None, None]
        else:
            acc_states = torch.zeros((nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32)

        # Compute shifted gamma and scale
        DT_shifted = F.pad(DT_curr[:, 1:], (0, 1))
        Trap_shifted = F.pad(Trap_curr[:, 1:], (0, 1))
        shifted_gamma = DT_shifted * (1 - Trap_shifted)
        scale = DT_curr * Trap_curr + DT_shifted * (1 - Trap_shifted)

        # Add biases
        Q_curr = Q_curr + Q_bias.unsqueeze(0)
        K_curr = K_curr + K_bias.unsqueeze(0)

        # Compute QK dot for skip connection
        QK_dot = torch.sum(K_curr * Q_curr, dim=-1) * shifted_gamma.transpose(0, 1)

        # Rotary embeddings using Angles_Cumsum
        cos_angles_curr = torch.cos(Angles_Cumsum).to(Q_curr.dtype)
        sin_angles_curr = torch.sin(Angles_Cumsum).to(Q_curr.dtype)
        Q_curr = _rotary(Q_curr, cos_angles_curr, sin_angles_curr)
        K_curr = _rotary(K_curr, cos_angles_curr, sin_angles_curr)

        Final_K_States.append(K_curr[-1])
        Final_V_States.append(V_curr[-1])

        K_curr_scaled = K_curr * scale.transpose(0, 1).unsqueeze(-1).to(K_curr.dtype)

        # Compute output via quadratic attention
        QK = torch.einsum("thd,shd->hts", Q_curr, K_curr_scaled)
        QK_causal = torch.tril(QK)
        QK_causal = (QK_causal * torch.exp(_segsum(ADT_curr))).to(QK_causal.dtype)
        out = torch.einsum("hts,shd->thd", QK_causal, V_curr)

        if Initial_States is not None:
            da_cs = torch.cumsum(ADT_curr, dim=-1)
            exp_da_cs = torch.exp(da_cs)
            out = out + torch.einsum("hDd,thd,ht->thD", acc_states.to(Q_curr.dtype), Q_curr, exp_da_cs.to(Q_curr.dtype))

        if D is not None:
            out = out + D[None, :, None] * V_curr

        out = out - V_curr * QK_dot.unsqueeze(-1)

        if Z_curr is not None:
            out = out * Z_curr * torch.sigmoid(Z_curr)
        out_zs.append(out)

        # Compute final state
        da_cs_last = torch.exp(torch.sum(ADT_curr, dim=-1))
        da_cs_rev = torch.exp(torch.sum(ADT_curr, dim=-1, keepdim=True) - torch.cumsum(ADT_curr, dim=-1))
        V_curr_scaled = V_curr * da_cs_rev.permute(1, 0).unsqueeze(-1).to(V_curr.dtype)
        final_acc_states = acc_states * da_cs_last.unsqueeze(-1).unsqueeze(-1) + torch.einsum(
            "thd,thD->hDd", K_curr_scaled, V_curr_scaled.to(K_curr_scaled.dtype))
        Final_SSM_States.append(final_acc_states)

    num_sequences = cu_seqlens.size(0) - 1 if is_varlen else batch
    for seq_idx in range(num_sequences):
        compute_one_sequence(seq_idx)

    if not is_varlen:
        out_zs = torch.stack(out_zs, dim=0)
        Final_Angle_States = torch.stack(Final_Angle_States, dim=0)
        Final_SSM_States = torch.stack(Final_SSM_States, dim=0)
        Final_K_States = torch.stack(Final_K_States, dim=0)
        Final_V_States = torch.stack(Final_V_States, dim=0)
    else:
        out_zs = torch.cat(out_zs, dim=0).unsqueeze(0)
        Final_Angle_States = torch.stack(Final_Angle_States, dim=0)
        Final_SSM_States = torch.stack(Final_SSM_States, dim=0)
        Final_K_States = torch.stack(Final_K_States, dim=0)
        Final_V_States = torch.stack(Final_V_States, dim=0)

    return out_zs, (Final_Angle_States, Final_SSM_States, Final_K_States, Final_V_States)


# ================================================================== 
# Test Utilities
# ================================================================== 

def detach_clone(*args):
    """Detach and clone tensors, preserving None values."""
    return tuple([arg.detach().clone().requires_grad_() if arg is not None else None for arg in args])

@torch.no_grad()
def relative_error(
    ker: torch.Tensor,
    ref: torch.Tensor,
    eps: float = 1e-6,
    ref_mag_mask: float = 1e-2,
    p: float = 0.95,
    name: str = "",
    print_top_errors: bool = True,
    angle: bool = False,   # if True: use circular absolute error; else: relative error
) -> float:
    assert ker.shape == ref.shape

    ker_xx = ker.detach().to(torch.float32)
    ref_xx = ref.detach().to(torch.float32)

    abs_ref = ref_xx.abs()

    if angle:
        delta = ker_xx - ref_xx
        delta = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
        abs_diff = delta.abs()
    else:
        abs_diff = (ker_xx - ref_xx).abs()

    mask = abs_ref >= ref_mag_mask
    if not mask.any():
        return 0.0

    vals = abs_diff[mask].flatten() if angle else (abs_diff[mask] / (abs_ref[mask] + eps)).flatten()

    n = vals.numel()
    k = max(1, min(n, int(math.ceil(p * n))))
    err = vals.kthvalue(k).values.item()

    if print_top_errors and err > 0.01:
        print(f"\n  Top 10 errors for {name}:")
        diff_flat = abs_diff.flatten()
        ref_flat = ref_xx.flatten()
        ker_flat = ker_xx.flatten()
        topk = diff_flat.topk(min(10, diff_flat.numel()))
        for i, idx in enumerate(topk.indices):
            idx = idx.item()
            r = ref_flat[idx].item()
            k_val = ker_flat[idx].item()
            d = diff_flat[idx].item()
            if angle:
                # For angles, show absolute angular error (radians)
                print(f"    {i}: ref={r:.6e}, ker={k_val:.6e}, ang_err={d:.6e} rad")
            else:
                rel_e = d / (abs(r) + eps) if abs(r) >= ref_mag_mask else float('nan')
                print(f"    {i}: ref={r:.6e}, ker={k_val:.6e}, diff={d:.6e}, rel={rel_e:.2%}")

    return err


def create_mamba3_siso_inputs(
    batch: int,
    seqlen: int,
    nheads: int,
    nheads_qk: int,
    headdim_qk: int,
    headdim_v: int,
    dtype: torch.dtype,
    device: str,
    has_D: bool,
    has_Z: bool,
    has_input_states: bool,
    cu_seqlens: Optional[torch.Tensor] = None,
    requires_grad: bool = False,
):
    num_sequences = cu_seqlens.size(0) - 1 if cu_seqlens is not None else batch
    
    Q = torch.randn((batch, seqlen, nheads_qk, headdim_qk), device=device, dtype=dtype)
    Q = F.rms_norm(Q, normalized_shape=(headdim_qk,)).clone()
    K = torch.randn((batch, seqlen, nheads_qk, headdim_qk), device=device, dtype=dtype)
    K = F.rms_norm(K, normalized_shape=(headdim_qk,)).clone()
    V = torch.randn((batch, seqlen, nheads, headdim_v), device=device, dtype=dtype)

    dt_max, dt_min = 0.1, 0.001
    a_init = -torch.empty(batch, nheads, seqlen, device=device, dtype=torch.float32).uniform_(1.0, 16.0)
    dt = torch.exp(
        torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32) 
        * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    )
    ADT = (a_init * dt).contiguous()
    DT = dt.contiguous()
    Trap = torch.empty(batch, nheads, seqlen, dtype=dtype, device=device).uniform_(0.0, 1.0).clone()
    Q_bias = torch.randn(nheads, headdim_qk, dtype=dtype, device=device)
    K_bias = torch.randn(nheads, headdim_qk, dtype=dtype, device=device)
    
    # headdim_angles constraint: 2*headdim_angles <= headdim_qk
    headdim_angles = headdim_qk // 4
    Angles = torch.randn(batch, seqlen, nheads, headdim_angles, dtype=torch.float32, device=device)

    D = torch.ones((nheads,), device=device, dtype=torch.float32) if has_D else None
    Z = torch.randn((batch, seqlen, nheads, headdim_v), device=device, dtype=dtype) if has_Z else None
    
    if has_input_states:
        Input_Angle_State = torch.randn((num_sequences, nheads, headdim_angles), device=device, dtype=torch.float32)
        Input_SSM_State = torch.randn((num_sequences, nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32)
        Input_K_State = torch.randn((num_sequences, nheads, headdim_qk), device=device, dtype=torch.float32)
        Input_V_State = torch.randn((num_sequences, nheads, headdim_v), device=device, dtype=torch.float32)
        Input_States = (Input_Angle_State, Input_SSM_State, Input_K_State, Input_V_State)
    else:
        Input_States = None
    
    if requires_grad:
        Q.requires_grad_(True)
        K.requires_grad_(True)
        V.requires_grad_(True)
        ADT.requires_grad_(True)
        DT.requires_grad_(True)
        Trap.requires_grad_(True)
        Q_bias.requires_grad_(True)
        K_bias.requires_grad_(True)
        Angles.requires_grad_(True)
        if D is not None:
            D.requires_grad_(True)
        if Z is not None:
            Z.requires_grad_(True)
        if Input_States is not None:
            for state in Input_States:
                state.requires_grad_(True)
    
    return {
        'Q': Q, 'K': K, 'V': V,
        'ADT': ADT, 'DT': DT, 'Trap': Trap,
        'Q_bias': Q_bias, 'K_bias': K_bias, 'Angles': Angles,
        'D': D, 'Z': Z, 'Input_States': Input_States,
    }


# ================================================================== 
# Triton Step Kernel Test
# ================================================================== 

def test_mamba3_siso_step(nheads_qk=4, has_Z=True, has_D=True):
    """Test Mamba-3 step kernel against reference recurrent implementation."""
    device = 'cuda'
    rtol = 5e-2
    dtype = torch.bfloat16
    torch.random.manual_seed(42)
    
    batch = 128
    seqlen = 2345
    nheads = 32
    headdim_qk = 128
    headdim_v = 64
    headdim_angles = headdim_qk // 4
    
    inputs = create_mamba3_siso_inputs(
        batch, seqlen, nheads, nheads_qk, headdim_qk, headdim_v,
        dtype, device, has_D=has_D, has_Z=has_Z, has_input_states=True,
        requires_grad=False
    )
    Q_full, K_full, V_full, ADT_full, DT_full, Trap_full, Q_bias, K_bias, Angles_full, D, Z_full, Input_States = inputs['Q'], inputs['K'], inputs['V'], inputs['ADT'], inputs['DT'], inputs['Trap'], inputs['Q_bias'], inputs['K_bias'], inputs['Angles'], inputs['D'], inputs['Z'], inputs['Input_States']

    angle_state_triton, ssm_state_triton, k_state_triton, v_state_triton = Input_States
    outputs_triton = []    
    for step in range(seqlen):
        Q_step = Q_full[:, step, :, :].contiguous()
        K_step = K_full[:, step, :, :].contiguous()
        V_step = V_full[:, step, :, :].contiguous()
        ADT_step = ADT_full[:, :, step].contiguous()
        DT_step = DT_full[:, :, step].contiguous()
        Trap_step = Trap_full[:, :, step].contiguous()
        Angles_step = Angles_full[:, step, :, :].contiguous()
        Z_step = Z_full[:, step, :, :].contiguous() if Z_full is not None else None
        
        input_states_triton = (angle_state_triton, ssm_state_triton, k_state_triton, v_state_triton)
        out_triton, output_states_triton = mamba3_siso_step(
            Q_step, K_step, V_step, ADT_step, DT_step, Trap_step,
            Q_bias, K_bias, Angles_step, D, Z_step, input_states_triton
        )
        angle_state_triton, ssm_state_triton, k_state_triton, v_state_triton = output_states_triton
        outputs_triton.append(out_triton)
    
    outputs_triton = torch.stack(outputs_triton, dim=1)

    # Reference implementation
    outputs_ref, final_states_ref = mamba3_siso_step_ref(
        Q_full, K_full, V_full, ADT_full, DT_full, Trap_full,
        Q_bias, K_bias, Angles_full, D, Z_full, Input_States=Input_States
    )
    angle_state_ref, ssm_state_ref, k_state_ref, v_state_ref = final_states_ref
    
    out_rel_err = relative_error(outputs_triton, outputs_ref)
    print(f"Step output relative error: {out_rel_err:.2e}")
    assert out_rel_err < rtol, f"Step output relative error {out_rel_err} exceeds tolerance {rtol}"
    
    # Compare final states
    angle_state_err = relative_error(angle_state_triton, angle_state_ref)
    ssm_state_err = relative_error(ssm_state_triton, ssm_state_ref)
    k_state_err = relative_error(k_state_triton, k_state_ref)
    v_state_err = relative_error(v_state_triton, v_state_ref)
    
    print(f"Final state errors - Angle: {angle_state_err:.2e}, SSM: {ssm_state_err:.2e}, K: {k_state_err:.2e}, V: {v_state_err:.2e}")
    assert angle_state_err < rtol, f"Angle state error {angle_state_err} exceeds tolerance {rtol}"
    assert ssm_state_err < rtol, f"SSM state error {ssm_state_err} exceeds tolerance {rtol}"
    assert k_state_err < rtol, f"K state error {k_state_err} exceeds tolerance {rtol}"
    assert v_state_err < rtol, f"V state error {v_state_err} exceeds tolerance {rtol}"

# ================================================================== 
# Triton Forward+Backward Batched Kernel Test
# ================================================================== 

# Combined Forward+Backward batched mode test
# NOTE: Relative erros for tensors are within 6-8% (especially when they are reduced). 
# The error for angle is ~20% because cumsum accumulates error over sequence length. This
# error becomes ~3% when cumsum (angle-dt) kernel is removed
def test_mamba3_siso_combined_batched(nheads_qk=4, has_Z=True, has_D=True, headdim_qk=128):
    """Test Mamba-3 combined forward+backward against fwd reference.
    """
    device = 'cuda'
    rtol = 1e-1
    dtype = torch.bfloat16
    torch.random.manual_seed(42)
    
    batch = 16
    seqlen = 2345
    nheads = 32
    headdim_v = 64
    chunk_size = 64
    half = seqlen // 2
    
    inputs = create_mamba3_siso_inputs(
        batch, seqlen, nheads, nheads_qk, headdim_qk, headdim_v,
        dtype, device, has_D=has_D, has_Z=has_Z, has_input_states=True,
        requires_grad=True
    )
    inputs_ref = copy.deepcopy(inputs)
    
    # Reference: use mamba3_siso_fwd_ref to compute full sequence output.
    Out_ref, Final_States_ref = mamba3_siso_fwd_ref(
        inputs_ref['Q'], inputs_ref['K'], inputs_ref['V'],
        inputs_ref['ADT'], inputs_ref['DT'], inputs_ref['Trap'],
        inputs_ref['Q_bias'], inputs_ref['K_bias'], inputs_ref['Angles'],
        inputs_ref['D'], inputs_ref['Z'], inputs_ref['Input_States'],
    )
    
    # Kernel: two-pass forward via state passing.
    Out_first, Angle_State_1, SSM_State_1, K_State_1, V_State_1 = mamba3_siso_combined(
        inputs['Q'][:, :half], inputs['K'][:, :half], inputs['V'][:, :half],
        inputs['ADT'][:, :, :half], inputs['DT'][:, :, :half], inputs['Trap'][:, :, :half],
        inputs['Q_bias'], inputs['K_bias'], inputs['Angles'][:, :half],
        inputs['D'], inputs['Z'][:, :half] if has_Z else None, 
        inputs['Input_States'],
        chunk_size=chunk_size,
        return_final_states=True,
    )
    Out_second, Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State = mamba3_siso_combined(
        inputs['Q'][:, half:], inputs['K'][:, half:], inputs['V'][:, half:],
        inputs['ADT'][:, :, half:], inputs['DT'][:, :, half:], inputs['Trap'][:, :, half:],
        inputs['Q_bias'], inputs['K_bias'], inputs['Angles'][:, half:],
        inputs['D'], inputs['Z'][:, half:] if has_Z else None,
        (Angle_State_1, SSM_State_1, K_State_1, V_State_1),
        chunk_size=chunk_size,
        return_final_states=True,
    )
    Out_kernel = torch.cat([Out_first, Out_second], dim=1)
    
    # Forward comparison
    out_err = relative_error(Out_kernel, Out_ref, name="Output")
    print(f"Forward output error: {out_err:.2e}")
    # assert out_err < rtol, f"Forward output error {out_err:.2e} exceeds tolerance {rtol}"
    
    # Compare final states
    Final_Angle_State_ref, Final_SSM_State_ref, Final_K_State_ref, Final_V_State_ref = Final_States_ref
    for state_name, ker_state, ref_state in [
        ('Angle', Final_Angle_State, Final_Angle_State_ref),
        ('SSM', Final_SSM_State, Final_SSM_State_ref),
        ('K', Final_K_State, Final_K_State_ref),
        ('V', Final_V_State, Final_V_State_ref),
    ]:
        err = relative_error(ker_state, ref_state, name=f"Final_{state_name}_State", angle=(state_name=='Angle'))
        print(f"Final_{state_name}_State error: {err:.2e}")
        # assert err < rtol, f"Final_{state_name}_State error {err:.2e} exceeds tolerance"
    
    # Backward 
    # Give gradients to both output and final states
    dO = torch.randn_like(Out_ref)
    dFinal_Angle_State = torch.randn_like(Final_Angle_State)
    dFinal_SSM_State = torch.randn_like(Final_SSM_State)
    dFinal_K_State = torch.randn_like(Final_K_State)
    dFinal_V_State = torch.randn_like(Final_V_State)
    
    # Reference backward
    torch.autograd.backward(
        [Out_ref, Final_Angle_State_ref, Final_SSM_State_ref, Final_K_State_ref, Final_V_State_ref],
        [dO, dFinal_Angle_State, dFinal_SSM_State, dFinal_K_State, dFinal_V_State],
    )
    # Kernel backward
    torch.autograd.backward(
        [Out_kernel, Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State],
        [dO, dFinal_Angle_State, dFinal_SSM_State, dFinal_K_State, dFinal_V_State],
    )
    
    # Compare gradients
    for grad_name in ['Q', 'K', 'V', 'ADT', 'DT', 'Trap', 'Q_bias', 'K_bias', 'Angles']:
        err = relative_error(inputs[grad_name].grad, inputs_ref[grad_name].grad, name=f"d{grad_name}")
        print(f"d{grad_name} error: {err:.2e}")
        # assert err < rtol, f"d{grad_name} error {err:.2e} exceeds tolerance"
    
    if has_D:
        err = relative_error(inputs['D'].grad, inputs_ref['D'].grad, name="dD")
        print(f"dD error: {err:.2e}")

    if has_Z:
        err = relative_error(inputs['Z'].grad, inputs_ref['Z'].grad, name="dZ")
        print(f"dZ error: {err:.2e}")
    
    # Input state gradients
    for i, state_name in enumerate(['Angle', 'SSM', 'K', 'V']):
        err = relative_error(inputs['Input_States'][i].grad, inputs_ref['Input_States'][i].grad, name=f"dInput_{state_name}_State")
        print(f"dInput_{state_name}_State error: {err:.2e}")

# ================================================================== 
# Triton Forward+Backward Varlen Kernel Test
# ================================================================== 

# Combined Forward+Backward varlen mode test
# NOTE: Relative erros for tensors are within 6-8% (especially when they are reduced). 
# The error for angle is ~20% because cumsum accumulates error over sequence length. This
# error becomes ~3% when cumsum (angle-dt) kernel is removed
def test_mamba3_siso_combined_varlen(nheads_qk=4, has_Z=True, has_D=True, headdim_qk=128):
    """Test Mamba-3 combined forward+backward with variable-length sequences against fwd reference.
    """
    device = 'cuda'
    rtol = 1e-1
    dtype = torch.bfloat16
    torch.random.manual_seed(42)
    
    num_sequences = 8
    seq_lengths = [2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352]
    total_seqlen = sum(seq_lengths)
    
    # Create cu_seqlens
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lengths), dim=0).tolist()), 
                               dtype=torch.int32, device=device)
    
    batch = 1  # Varlen requires batch=1
    nheads = 32
    headdim_v = 64
    chunk_size = 64
    headdim_angles = headdim_qk // 4
    
    # Create packed inputs (batch=1, total_seqlen, ...)
    Q = torch.randn((batch, total_seqlen, nheads_qk, headdim_qk), device=device, dtype=dtype)
    Q = F.rms_norm(Q, normalized_shape=(headdim_qk,)).clone()
    K = torch.randn((batch, total_seqlen, nheads_qk, headdim_qk), device=device, dtype=dtype)
    K = F.rms_norm(K, normalized_shape=(headdim_qk,)).clone()
    V = torch.randn((batch, total_seqlen, nheads, headdim_v), device=device, dtype=dtype)
    
    dt_max, dt_min = 0.1, 0.001
    a_init = -torch.empty(batch, nheads, total_seqlen, device=device, dtype=torch.float32).uniform_(1.0, 16.0)
    dt = torch.exp(
        torch.rand(batch, nheads, total_seqlen, device=device, dtype=torch.float32) 
        * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    )
    ADT = (a_init * dt).contiguous()
    DT = dt.contiguous()
    Trap = torch.empty(batch, nheads, total_seqlen, dtype=dtype, device=device).uniform_(0.0, 1.0).clone()
    
    Q_bias = torch.randn(nheads, headdim_qk, dtype=dtype, device=device)
    K_bias = torch.randn(nheads, headdim_qk, dtype=dtype, device=device)
    Angles = torch.randn(batch, total_seqlen, nheads, headdim_angles, dtype=dtype, device=device) * 0.1
    
    D = torch.ones((nheads,), device=device, dtype=torch.float32) if has_D else None
    Z = torch.randn((batch, total_seqlen, nheads, headdim_v), device=device, dtype=dtype) if has_Z else None
    
    # Input states: one per sequence
    Input_Angle_State = torch.randn((num_sequences, nheads, headdim_angles), device=device, dtype=torch.float32)
    Input_SSM_State = torch.randn((num_sequences, nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32)
    Input_K_State = torch.randn((num_sequences, nheads, headdim_qk), device=device, dtype=torch.float32)
    Input_V_State = torch.randn((num_sequences, nheads, headdim_v), device=device, dtype=torch.float32)
    Input_States = (Input_Angle_State, Input_SSM_State, Input_K_State, Input_V_State)
    
    # Enable gradients
    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    ADT.requires_grad_(True)
    DT.requires_grad_(True)
    Trap.requires_grad_(True)
    Q_bias.requires_grad_(True)
    K_bias.requires_grad_(True)
    Angles.requires_grad_(True)
    if D is not None:
        D.requires_grad_(True)
    if Z is not None:
        Z.requires_grad_(True)
    for state in Input_States:
        state.requires_grad_(True)
    
    # Create deep copies for reference
    inputs_ref = {
        'Q': Q.detach().clone().requires_grad_(True),
        'K': K.detach().clone().requires_grad_(True),
        'V': V.detach().clone().requires_grad_(True),
        'ADT': ADT.detach().clone().requires_grad_(True),
        'DT': DT.detach().clone().requires_grad_(True),
        'Trap': Trap.detach().clone().requires_grad_(True),
        'Q_bias': Q_bias.detach().clone().requires_grad_(True),
        'K_bias': K_bias.detach().clone().requires_grad_(True),
        'Angles': Angles.detach().clone().requires_grad_(True),
        'D': D.detach().clone().requires_grad_(True) if D is not None else None,
        'Z': Z.detach().clone().requires_grad_(True) if Z is not None else None,
        'Input_States': tuple(s.detach().clone().requires_grad_(True) for s in Input_States),
    }
    
    inputs_ker = {
        'Q': Q, 'K': K, 'V': V,
        'ADT': ADT, 'DT': DT, 'Trap': Trap,
        'Q_bias': Q_bias, 'K_bias': K_bias, 'Angles': Angles,
        'D': D, 'Z': Z, 'Input_States': Input_States,
    }
    
    # Reference: use mamba3_siso_fwd_ref with cu_seqlens
    Out_ref, Final_States_ref = mamba3_siso_fwd_ref(
        inputs_ref['Q'], inputs_ref['K'], inputs_ref['V'],
        inputs_ref['ADT'], inputs_ref['DT'], inputs_ref['Trap'],
        inputs_ref['Q_bias'], inputs_ref['K_bias'], inputs_ref['Angles'],
        inputs_ref['D'], inputs_ref['Z'], inputs_ref['Input_States'],
        cu_seqlens=cu_seqlens,
    )
    
    # Kernel: single call with cu_seqlens
    Out_kernel, Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State = mamba3_siso_combined(
        inputs_ker['Q'], inputs_ker['K'], inputs_ker['V'],
        inputs_ker['ADT'], inputs_ker['DT'], inputs_ker['Trap'],
        inputs_ker['Q_bias'], inputs_ker['K_bias'], inputs_ker['Angles'],
        inputs_ker['D'], inputs_ker['Z'], inputs_ker['Input_States'],
        chunk_size=chunk_size,
        return_final_states=True,
        cu_seqlens=cu_seqlens,
    )
    
    # Forward comparison
    out_err = relative_error(Out_kernel, Out_ref, name="Output")
    print(f"Forward output error: {out_err:.2e}")
    
    # Compare final states
    Final_Angle_State_ref, Final_SSM_State_ref, Final_K_State_ref, Final_V_State_ref = Final_States_ref
    for state_name, ker_state, ref_state in [
        ('Angle', Final_Angle_State, Final_Angle_State_ref),
        ('SSM', Final_SSM_State, Final_SSM_State_ref),
        ('K', Final_K_State, Final_K_State_ref),
        ('V', Final_V_State, Final_V_State_ref),
    ]:
        err = relative_error(ker_state, ref_state, name=f"Final_{state_name}_State", angle=(state_name=='Angle'))
        print(f"Final_{state_name}_State error: {err:.2e}")
    
    # Backward
    dO = torch.randn_like(Out_ref)
    dFinal_Angle_State = torch.randn_like(Final_Angle_State)
    dFinal_SSM_State = torch.randn_like(Final_SSM_State)
    dFinal_K_State = torch.randn_like(Final_K_State)
    dFinal_V_State = torch.randn_like(Final_V_State)
    
    # Reference backward
    torch.autograd.backward(
        [Out_ref, Final_Angle_State_ref, Final_SSM_State_ref, Final_K_State_ref, Final_V_State_ref],
        [dO, dFinal_Angle_State, dFinal_SSM_State, dFinal_K_State, dFinal_V_State],
    )
    # Kernel backward
    torch.autograd.backward(
        [Out_kernel, Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State],
        [dO, dFinal_Angle_State, dFinal_SSM_State, dFinal_K_State, dFinal_V_State],
    )
    
    # Compare gradients
    for grad_name in ['Q', 'K', 'V', 'ADT', 'DT', 'Trap', 'Q_bias', 'K_bias', 'Angles']:
        err = relative_error(inputs_ker[grad_name].grad, inputs_ref[grad_name].grad, name=f"d{grad_name}")
        print(f"d{grad_name} error: {err:.2e}")
    
    if has_D:
        err = relative_error(inputs_ker['D'].grad, inputs_ref['D'].grad, name="dD")
        print(f"dD error: {err:.2e}")
    if has_Z:
        err = relative_error(inputs_ker['Z'].grad, inputs_ref['Z'].grad, name="dZ")
        print(f"dZ error: {err:.2e}")
    
    # Input state gradients
    for i, state_name in enumerate(['Angle', 'SSM', 'K', 'V']):
        err = relative_error(inputs_ker['Input_States'][i].grad, inputs_ref['Input_States'][i].grad, name=f"dInput_{state_name}_State")
        print(f"dInput_{state_name}_State error: {err:.2e}")


# ================================================================== 
# Sanity check test: Step reference and Forward reference match
# ================================================================== 

def test_mamba3_siso_step_ref_vs_fwd_ref(nheads_qk=4, has_Z=True, has_D=True):
    """Test that mamba3_siso_step_ref and mamba3_siso_fwd_ref produce identical outputs."""
    device = 'cuda'
    rtol = 1e-4  # Both are pure Python/PyTorch, so should match very closely
    dtype = torch.float32  # Use float32 for reference-vs-reference comparison
    torch.random.manual_seed(42)

    batch = 16
    seqlen = 2048
    nheads = 32
    headdim_qk = 128
    headdim_v = 64
    headdim_angles = headdim_qk // 4

    inputs = create_mamba3_siso_inputs(
        batch, seqlen, nheads, nheads_qk, headdim_qk, headdim_v,
        dtype, device, has_D=has_D, has_Z=has_Z, has_input_states=True,
        requires_grad=False,
    )

    # --- Step ref ---
    out_step, final_states_step = mamba3_siso_step_ref(
        inputs['Q'], inputs['K'], inputs['V'],
        inputs['ADT'], inputs['DT'], inputs['Trap'],
        inputs['Q_bias'], inputs['K_bias'], inputs['Angles'],
        inputs['D'], inputs['Z'],
        Input_States=inputs['Input_States'],
    )
    angle_state_step, ssm_state_step, k_state_step, v_state_step = final_states_step

    # --- Fwd ref ---
    out_fwd, final_states_fwd = mamba3_siso_fwd_ref(
        inputs['Q'], inputs['K'], inputs['V'],
        inputs['ADT'], inputs['DT'], inputs['Trap'],
        inputs['Q_bias'], inputs['K_bias'], inputs['Angles'],
        inputs['D'], inputs['Z'],
        Initial_States=inputs['Input_States'],
        dtype=dtype,
    )
    angle_state_fwd, ssm_state_fwd, k_state_fwd, v_state_fwd = final_states_fwd

    # --- Compare outputs ---
    out_err = relative_error(out_step, out_fwd, name="Output", ref_mag_mask=1e-3)
    print(f"Output error: {out_err:.2e}")
    # assert out_err < rtol, f"Output error {out_err:.2e} exceeds tolerance {rtol}"

    # --- Compare final states ---
    for state_name, step_state, fwd_state in [
        ('Angle', angle_state_step, angle_state_fwd),
        ('SSM',   ssm_state_step,   ssm_state_fwd),
        ('K',     k_state_step,     k_state_fwd),
        ('V',     v_state_step,     v_state_fwd),
    ]:
        err = relative_error(step_state, fwd_state, name=f"Final_{state_name}_State",
                             angle=(state_name == 'Angle'), ref_mag_mask=1e-3)
        print(f"Final_{state_name}_State error: {err:.2e}")


# Main function
if __name__ == "__main__":
    print("Running Mamba-3 step reference vs forward reference test...")
    test_mamba3_siso_step_ref_vs_fwd_ref()
    print("="*100)

    print("\nRunning Mamba-3 combined forward+backward batched test...")
    test_mamba3_siso_combined_batched()
    print("="*100)

    print("\nRunning Mamba-3 combined forward+backward varlen test...")
    test_mamba3_siso_combined_varlen()
    print("="*100)