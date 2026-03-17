"""
Mamba-3 MIMO Kernel Tests

Copyright (c) 2026, Dao AI Lab, Goombalab


Usage:
pytest -q -s -p no:warnings tests/ops/tilelang/test_mamba3_mimo.py -k bwd
pytest -q -s -p no:warnings tests/ops/tilelang/test_mamba3_mimo.py -k fwd
pytest -q -s -p no:warnings tests/ops/tilelang/test_mamba3_mimo.py -k smoke
pytest -q -s -p no:warnings tests/ops/tilelang/test_mamba3_mimo.py -k chunk_ref_matches_step_ref

Remove the -s flag for less verbose output.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


import pytest
import torch
from torch import Tensor
F = torch.nn.functional


FIXED_B = 4
FIXED_S = 2048
FIXED_H = 16
FIXED_G = 1
FIXED_ROTARY_DIM_DIVISOR = 4
FIXED_DTYPE = torch.bfloat16
REL_TOL = 0.10


CASE_GRID = [
    pytest.param(16, 64, 4, 8, 128, id="N16_P64_R4_C8_BB128"),
    pytest.param(32, 64, 4, 16, 256, id="N32_P64_R4_C16_BB256"),
    pytest.param(64, 64, 4, 16, 256, id="N64_P64_R4_C16_BB256"),
    pytest.param(128, 64, 4, 16, 256, id="N128_P64_R4_C16_BB256"),
    pytest.param(256, 64, 4, 8, 256, id="N256_P64_R4_C8_BB256"),
    pytest.param(64, 128, 4, 16, 256, id="N64_P128_R4_C16_BB256"),
    pytest.param(128, 32, 4, 16, 256, id="N128_P32_R4_C16_BB256"),
    pytest.param(128, 128, 4, 8, 256, id="N128_P128_R4_C8_BB256"),
    pytest.param(128, 64, 8, 8, 256, id="N128_P64_R8_C8_BB256"),
    pytest.param(128, 64, 2, 32, 256, id="N128_P64_R2_C32_BB256"),
    pytest.param(128, 64, 1, 64, 256, id="N128_P64_R1_C64_BB256"),
]


def _require_cuda_and_kernel_deps() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for mamba3 tilelang tests")
    pytest.importorskip("tilelang")
    pytest.importorskip("triton")


@pytest.fixture(scope="module")
def mods() -> SimpleNamespace:
    _require_cuda_and_kernel_deps()
    import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo as mamba3_top
    import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd as mamba3_bwd
    import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd as mamba3_fwd
    import mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils as mamba3_mimo_utils

    return SimpleNamespace(
        top=mamba3_top,
        bwd=mamba3_bwd,
        fwd=mamba3_fwd,
        utils=mamba3_mimo_utils,
    )


def max_rel_err(ours: Tensor, ref: Tensor, eps: float = 1e-5) -> float:
    ours_f = ours.float()
    ref_f = ref.float()
    num = (ours_f - ref_f).abs().max()
    den = ref_f.abs().max().clamp_min(eps)
    return float((num / den).item())


def assert_stable_rel(
    ours: Tensor,
    ref: Tensor,
    *,
    label: str,
    cfg: str,
    rel_tol: float = REL_TOL,
) -> None:
    ours_f = ours.float()
    ref_f = ref.float()
    rel = max_rel_err(ours_f, ref_f)
    close_mask = torch.isclose(ours_f, ref_f, rtol=REL_TOL, atol=0.1)
    bad_frac = float((~close_mask).float().mean().item())
    max_abs = float((ours_f - ref_f).abs().max().item())
    print(
        f"[debug] {label} ({cfg}) "
        f"stable_max_rel={rel:.6f} max_abs={max_abs:.6e} "
        f"bad_frac(rtol=0.1,atol=0.1)={bad_frac:.6f}"
    )
    if rel < rel_tol:
        return

    raise AssertionError(
        f"{label} stable_max_rel >= {rel_tol} for {cfg}: "
        f"stable_max_rel={rel:.6f}, max_abs={max_abs:.6e}, "
        f"diag_bad_frac_at_rtol0.1_atol0.1={bad_frac:.6f}"
    )


def build_inputs(
    *,
    mods: SimpleNamespace,
    n: int,
    p: int,
    r: int,
    chunk_size: int,
    seed: int,
    b: int = FIXED_B,
    s: int = FIXED_S,
    h: int = FIXED_H,
    g: int = FIXED_G,
    dtype: torch.dtype = FIXED_DTYPE,
    has_z: bool = True,
    has_d: bool = True,
    rotary_dim_divisor: int = FIXED_ROTARY_DIM_DIVISOR,
) -> dict:
    assert s % chunk_size == 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    q = torch.randn((b, s, r, g, n), device="cuda", dtype=dtype)
    k = torch.randn((b, s, r, g, n), device="cuda", dtype=dtype)
    v = torch.randn((b, s, h, p), device="cuda", dtype=dtype)

    q_bias = torch.randn((h, r, n), device="cuda", dtype=torch.float32)
    k_bias = torch.randn((h, r, n), device="cuda", dtype=torch.float32)
    mimo_v = torch.randn((h, r, p), device="cuda", dtype=torch.float32) / r
    mimo_o = torch.randn((h, r, p), device="cuda", dtype=torch.float32) / r

    z = torch.randn_like(v) if has_z else None
    mimo_z = torch.randn_like(mimo_v) if has_z else None
    d = torch.randn((h,), device="cuda", dtype=torch.float32) if has_d else None

    angles = torch.rand(
        (b, s, h, n // rotary_dim_divisor), device="cuda", dtype=torch.float32
    )
    dt = F.softplus(-3.0 + torch.randn((b, h, s), device="cuda", dtype=torch.float32))
    a = torch.rand((b, h, s), device="cuda", dtype=torch.float32)
    dA = (-dt * a).detach()
    dA_cs, dA_cs_rev, segsum = mods.utils.compute_dacs_segsum_triton(dA, chunk_size)
    trap = torch.rand((b, h, s), device="cuda", dtype=dtype)
    dout = torch.randn_like(v)

    return {
        "q": q,
        "k": k,
        "v": v,
        "q_bias": q_bias,
        "k_bias": k_bias,
        "mimo_v": mimo_v,
        "mimo_o": mimo_o,
        "z": z,
        "mimo_z": mimo_z,
        "D": d,
        "angles": angles,
        "dt": dt,
        "dA": dA,
        "dA_cs": dA_cs,
        "dA_cs_rev": dA_cs_rev,
        "segsum": segsum,
        "trap": trap,
        "dout": dout,
        "chunk_size": chunk_size,
        "rotary_dim_divisor": rotary_dim_divisor,
    }

def make_smoke_inputs(
    *,
    batch: int = 1,
    seqlen: int = 64,
    mimo_rank: int = 4,
    nheads_qk: int = 1,
    nheads: int = 8,
    headdim_qk: int = 64,
    headdim_v: int = 32,
    chunk_size: int = 16,
    rotary_dim_divisor: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    Q = torch.randn(
        (batch, seqlen, mimo_rank, nheads_qk, headdim_qk),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn(
        (batch, seqlen, nheads, headdim_v),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    import torch.nn.functional as F
    import math
    DT = F.softplus(
        -3.0
        + torch.randn(
            batch,
            nheads,
            seqlen,
            device=device,
            dtype=torch.float,
        )
    ).detach().requires_grad_(True)
    # Make ADT a leaf so .grad is populated without retain_grad().
    ADT = (-DT.detach() * math.log2(math.e)).clone().detach().requires_grad_(True)
    
    Trap = (
        torch.rand(
            (batch, nheads, seqlen),
            device=device,
            dtype=dtype,
        )
        * 0.5
    ).detach().requires_grad_(True)

    Q_bias = torch.randn(
        (nheads, mimo_rank, headdim_qk),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    K_bias = torch.randn_like(Q_bias, requires_grad=True)
    MIMO_V = torch.randn(
        (nheads, mimo_rank, headdim_v),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    MIMO_Z = (torch.randn_like(MIMO_V) / mimo_rank).detach().requires_grad_(True)
    MIMO_Out = (torch.randn_like(MIMO_V) / mimo_rank).detach().requires_grad_(True)
    Angles = torch.rand(
        (batch, seqlen, nheads, headdim_qk // rotary_dim_divisor),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    D = torch.randn(
        (nheads,),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    Z = torch.randn(
        (batch, seqlen, nheads, headdim_v),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    return dict(
        Q=Q,
        K=K,
        V=V,
        ADT=ADT,
        DT=DT,
        Trap=Trap,
        Q_bias=Q_bias,
        K_bias=K_bias,
        MIMO_V=MIMO_V,
        MIMO_Z=MIMO_Z,
        MIMO_Out=MIMO_Out,
        Angles=Angles,
        D=D,
        Z=Z,
        chunk_size=chunk_size,
        rotary_dim_divisor=rotary_dim_divisor,
        dtype=dtype,
    )

def grads_to_dA(grad_dA_cs: Tensor, grad_dA_cs_rev: Tensor, chunk_size: int) -> Tensor:
    b, h, s = grad_dA_cs.shape
    assert s % chunk_size == 0
    nchunks = s // chunk_size

    g_f = grad_dA_cs.view(b, h, nchunks, chunk_size)
    grad_from_f = torch.flip(torch.cumsum(torch.flip(g_f, dims=[-1]), dim=-1), dims=[-1])

    g_r = grad_dA_cs_rev.view(b, h, nchunks, chunk_size)
    prefix = torch.cumsum(g_r, dim=-1)
    grad_from_r = torch.cat([torch.zeros_like(prefix[..., :1]), prefix[..., :-1]], dim=-1)
    return (grad_from_f + grad_from_r).view(b, h, s)



def mamba3_MIMO_step_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    MIMO_V: torch.Tensor,
    MIMO_O: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    MIMO_Z: Optional[torch.Tensor] = None,
    Input_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reference implementation of Mamba-3 MIMO in recurrent (step) mode.

    Args:
        Input_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)

    Returns:
        out: Output tensor (batch, seqlen, nheads, headdim_v)
        Final_States: Tuple of (Angle_State, SSM_State, K_State, V_State)
    """
    batch, seqlen, mimo_rank, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device
    assert seqlen > 0

    # Expand Q/K for GQA
    if Q.shape[3] != V.shape[2]:
        Q = repeat(Q, "b s r h_bc d -> b s r (h_bc g) d", g=V.shape[2] // Q.shape[3])
    if K.shape[3] != V.shape[2]:
        K = repeat(K, "b s r h_bc d -> b s r (h_bc g) d", g=V.shape[2] // K.shape[3])

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

    q_bias = rearrange(Q_bias, "h r d -> r h d")
    k_bias = rearrange(K_bias, "h r d -> r h d")

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
        K_State = torch.zeros((batch, nheads, mimo_rank, headdim_qk), dtype=Q.dtype, device=device)
        V_State = torch.zeros((batch, nheads, mimo_rank, headdim_v), dtype=V.dtype, device=device)

    # MIMO up project x and z:
    v_proj = torch.einsum("bthd,hrd->btrhd", V, MIMO_V)
    if Z is not None:
        z_proj = torch.einsum("bthd,hrd->btrhd", Z, MIMO_Z)
    else:
        z_proj = None

    TWO_PI = 2 * math.pi
    out_arr = []

    # Main SSM recurrence
    for idx in range(seqlen):
        q = Q[:, idx, :, :, :] + q_bias.unsqueeze(0)
        k = K[:, idx, :, :, :] + k_bias.unsqueeze(0)
        v = v_proj[:, idx, :, :, :] # (B R H P)
        adt = ADT[:, :, idx]
        dt = DT[:, :, idx]
        trap = torch.nn.functional.sigmoid(Trap[:, :, idx])
        z = z_proj[:, idx, :, :, :] if z_proj is not None else None
        angles = Angles[:, idx, :, :] # (B H N)

        q = q.permute(0, 2, 1, 3) # (B H R N)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if z is not None:
            z = z.permute(0, 2, 1, 3)

        # Update angle state with cumsum: Angle_State = (Angle_State + Angles * DT) mod 2π
        # Angle_State = Angle_State + angles * dt.unsqueeze(-1)
        # Angle_State = Angle_State - TWO_PI * torch.floor(Angle_State / TWO_PI)
        Angle_State = Angle_State + torch.tanh(angles) * dt.unsqueeze(-1) * math.pi


        # Apply rotary embeddings to Q and K using cumulative angles
        cos_angles = torch.cos(Angle_State).unsqueeze(2) # (B H 1 N)
        sin_angles = torch.sin(Angle_State).unsqueeze(2)
        q_rot = apply_rotary_emb(q, cos_angles, sin_angles)
        k_rot = apply_rotary_emb(k, cos_angles, sin_angles)

        alpha = torch.exp(adt)
        beta = (1 - trap) * dt * alpha
        gamma = trap * dt

        # Update SSM state using previous K_State and V_State
        prev_kv = torch.einsum("bhrd,bhrp->bhpd", K_State, V_State)
        curr_kv = torch.einsum("bhrd,bhrp->bhpd", k_rot, v)
        SSM_State = alpha.unsqueeze(-1).unsqueeze(-1) * SSM_State
        SSM_State = SSM_State + beta.unsqueeze(-1).unsqueeze(-1) * prev_kv
        SSM_State = SSM_State + gamma.unsqueeze(-1).unsqueeze(-1) * curr_kv

        # Compute output
        out = torch.einsum("bhpd,bhrd->bhrp", SSM_State, q_rot.to(SSM_State.dtype))

        if D is not None:
            out = out + D[None, :, None, None] * v

        if z is not None:
            out = out * z * torch.sigmoid(z)

        out = torch.einsum("bhrp,hrp->bhp", out, MIMO_O)
        out_arr.append(out)

        # Update K and V states for next step
        K_State = k_rot
        V_State = v

    out = torch.stack(out_arr, dim=1)
    Final_States = (Angle_State, SSM_State, K_State, V_State)
    return out, Final_States


def apply_angle_dt_reference(
    angle: Tensor,  # (batch, seqlen, nheads, dim)
    dt: Tensor,     # (batch, seqlen, nheads)
) -> Tensor:
    # Match debug_mimo_step.py preprocessing for chunk reference path.
    base_vals = angle.to(torch.float32)
    base_vals = torch.tanh(base_vals) * dt[..., None].to(torch.float32) * torch.pi
    return torch.cumsum(base_vals, dim=1)


def mamba3_MIMO_chunk_ref(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_bias: Tensor,
    k_bias: Tensor,
    mimo_v: Tensor,
    mimo_o: Optional[Tensor],
    z: Optional[Tensor],
    mimo_z: Optional[Tensor],
    angles: Tensor,
    dA_cs: Tensor,
    dA_cs_rev: Tensor,
    dt: Tensor,
    trap: Tensor,
    D: Optional[Tensor],
    chunk_size: int = 64,
    rotary_dim_divisor: int = 4,
    return_final_state: bool = False,
    dtype: torch.dtype = torch.float32,
    rotate_pairwise: bool = False,
    contract_mimo_out: bool = True,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    # Local copy of the reference program so tests remain valid even if module-level
    # debug/reference helpers are removed from shipped kernels.
    from einops import rearrange, repeat

    nchunks = q.shape[1] // chunk_size
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if z is not None:
        z = z.to(dtype)
        mimo_z = mimo_z.to(dtype)
    if D is not None:
        D = D.to(dtype)
    q_bias, k_bias = q_bias.to(dtype), k_bias.to(dtype)
    mimo_v = mimo_v.to(dtype)
    if contract_mimo_out:
        assert mimo_o is not None
        mimo_o = mimo_o.to(dtype)
    if dA_cs is not None:
        dA_cs, dA_cs_rev = dA_cs.to(dtype), dA_cs_rev.to(dtype)
        dA_cs = rearrange(dA_cs, "b h (n c) -> b h n c", c=chunk_size)
        dA_cs_rev = rearrange(dA_cs_rev, "b h (n c) -> b h n c", c=chunk_size)

    batch, seqlen, mimo_rank, nheads_qk, dstate = q.shape
    nheads = v.shape[-2]
    if nheads_qk != nheads:
        q = repeat(q, "b s r h_qk d -> b s r (h_qk g) d", g=nheads // nheads_qk)
        k = repeat(k, "b s r h_qk d -> b s r (h_qk g) d", g=nheads // nheads_qk)

    angles = angles.to(dtype) if angles is not None else None
    trap = trap.to(dtype) if trap is not None else None
    dt = dt.to(dtype) if dt is not None else None

    q_bias = rearrange(q_bias, "h r d -> r h d")
    k_bias = rearrange(k_bias, "h r d -> r h d")
    q = q + q_bias[None, None, :, :, :]
    k = k + k_bias[None, None, :, :, :]

    qk_dot = torch.einsum("bsRhd,bsrhd->bsRrh", q, k)

    if angles is not None:
        angles = angles.unsqueeze(2)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        def apply_rotary_emb(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
            if rotate_pairwise:
                # Pairwise convention used by mamba3_MIMO_step_ref / debug_mimo_step.py.
                tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
                tensor_0 = tensor_reshaped[..., 0]
                tensor_1 = tensor_reshaped[..., 1]
                rotated_0 = tensor_0 * cos - tensor_1 * sin
                rotated_1 = tensor_0 * sin + tensor_1 * cos
                return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)
            # Kernel-aligned convention (kept as default for existing tests).
            tensor_reshaped = tensor.view(*tensor.shape[:-1], 2, -1)
            tensor_0 = tensor_reshaped[..., 0, :]
            tensor_1 = tensor_reshaped[..., 1, :]
            rotated_0 = tensor_0 * cos - tensor_1 * sin
            rotated_1 = tensor_0 * sin + tensor_1 * cos
            return torch.stack([rotated_0, rotated_1], dim=-2).view_as(tensor)

        def apply_rotary_emb_rotate_half(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
            tensor_reshaped = tensor.view(*tensor.shape[:-1], 4, -1)
            tensor_0 = tensor_reshaped[..., 0, :]
            tensor_1 = tensor_reshaped[..., 2, :]
            rotated_0 = tensor_0 * cos - tensor_1 * sin
            rotated_1 = tensor_0 * sin + tensor_1 * cos
            return torch.stack(
                [
                    rotated_0,
                    tensor_reshaped[..., 1, :],
                    rotated_1,
                    tensor_reshaped[..., 3, :],
                ],
                dim=-2,
            ).view_as(tensor)

        if rotary_dim_divisor == 4:
            q = apply_rotary_emb_rotate_half(q, cos_angles, sin_angles)
            k = apply_rotary_emb_rotate_half(k, cos_angles, sin_angles)
        elif rotary_dim_divisor == 2:
            q = apply_rotary_emb(q, cos_angles, sin_angles)
            k = apply_rotary_emb(k, cos_angles, sin_angles)
        else:
            raise ValueError(f"Invalid rotary_dim_divisor: {rotary_dim_divisor}")

    if return_final_state:
        final_k = k[:, -1].contiguous().clone()
    else:
        final_k = None

    trap = torch.nn.functional.sigmoid(trap)
    gamma = dt * trap
    dt_shifted = torch.nn.functional.pad(dt[:, :, 1:], (0, 1), value=0.0)
    trap_shifted = torch.nn.functional.pad(trap[:, :, 1:], (0, 1), value=0.0)
    shifted_gamma = dt_shifted * (1 - trap_shifted)
    factor = gamma + shifted_gamma
    k = torch.einsum("bsrhn,bhs->bsrhn", k, factor)
    qk_dot = torch.einsum("bsrRh,bhs->bsrRh", qk_dot, shifted_gamma)

    v = torch.einsum("bthd,hrd->btrhd", v, mimo_v)

    def segsum_unstable(x: Tensor) -> Tensor:
        x_segsum = x[..., :, None] - x[..., None, :]
        mask = torch.tril(torch.ones(x.size(-1), x.size(-1), device=x.device, dtype=torch.bool), diagonal=0)
        return x_segsum.masked_fill(~mask, -torch.inf)

    mimo_mask_outer = segsum_unstable(dA_cs)
    mimo_mask_inner = torch.ones(mimo_rank, mimo_rank, dtype=torch.bool, device=q.device)
    mimo_mask = torch.kron(mimo_mask_outer, mimo_mask_inner[None, None, None, :, :])

    q = rearrange(q, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    k_scaled = rearrange(k, "b (n c) r h d -> b h n c r d", c=chunk_size)
    k_scaled = torch.einsum("bhncrd,bhnc->bhncrd", k_scaled, torch.exp(dA_cs_rev))
    k_scaled = rearrange(k_scaled, "b h n c r d -> b h n (c r) d", c=chunk_size)
    k = rearrange(k, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    v = rearrange(v, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    kv = k_scaled.transpose(-1, -2) @ v

    curr_state = torch.zeros_like(kv[:, :, 0, :, :])
    for n in range(nchunks):
        curr_dA_sum = dA_cs[:, :, n, -1]
        next_state = (torch.exp(curr_dA_sum[:, :, None, None]) * curr_state) + kv[:, :, n, :, :]
        kv[:, :, n, :, :] = curr_state
        curr_state = next_state

    if return_final_state:
        final_state = next_state.float()
    else:
        final_state = None

    q_inter = q * torch.exp(repeat(dA_cs, "b h n c -> b h n (c r)", r=mimo_rank).unsqueeze(-1))
    inter = q_inter @ kv
    intra = ((q @ k.transpose(-1, -2)) * torch.exp(mimo_mask)) @ v
    o = inter + intra
    o = rearrange(o, "b h n (c r) d -> b h n c r d", r=mimo_rank)

    v = rearrange(v, "b h n (c r) d -> b h (n c) r d", r=mimo_rank)
    qk_dot = rearrange(qk_dot, "b t R r h -> b h t R r")
    qkv = torch.einsum("bhtRr,bhtrp->bhtRp", qk_dot, v)
    qkv = rearrange(qkv, "b h (n c) r d -> b h n c r d", c=chunk_size)
    o -= qkv

    if D is not None:
        vd = torch.einsum("bhtrp,h->bhtrp", v, D)
        vd = rearrange(vd, "b h (n c) r d -> b h n c r d", c=chunk_size)
        o += vd

    if z is not None:
        z = torch.einsum("bthd,hrd->btrhd", z, mimo_z)
        z = rearrange(z, "b (n c) r h d -> b h n c r d", c=chunk_size)
        o = o * torch.nn.functional.silu(z)

    if contract_mimo_out:
        assert mimo_o is not None
        o = torch.einsum("bhncrd,hrd->bhncd", o, mimo_o)
        return rearrange(o, "b h n c d -> b (n c) h d"), final_state, final_k

    return rearrange(o, "b h n c r d -> b (n c) r h d"), final_state, final_k


def run_ref_backward_fp32(
    mods: SimpleNamespace,
    inputs: dict,
    *,
    contract_mimo_out: bool = True,
    grad_output: Optional[Tensor] = None,
) -> dict:
    ref_dtype = torch.float32
    q = inputs["q"].detach().to(ref_dtype).requires_grad_(True)
    k = inputs["k"].detach().to(ref_dtype).requires_grad_(True)
    v = inputs["v"].detach().to(ref_dtype).requires_grad_(True)
    q_bias = inputs["q_bias"].detach().to(ref_dtype).requires_grad_(True)
    k_bias = inputs["k_bias"].detach().to(ref_dtype).requires_grad_(True)
    mimo_v = inputs["mimo_v"].detach().to(ref_dtype).requires_grad_(True)
    mimo_o = (
        inputs["mimo_o"].detach().to(ref_dtype).requires_grad_(True)
        if contract_mimo_out
        else None
    )
    z = (
        inputs["z"].detach().to(ref_dtype).requires_grad_(True)
        if inputs["z"] is not None
        else None
    )
    mimo_z = (
        inputs["mimo_z"].detach().to(ref_dtype).requires_grad_(True)
        if inputs["mimo_z"] is not None
        else None
    )
    angles = inputs["angles"].detach().to(ref_dtype).requires_grad_(True)
    dt = inputs["dt"].detach().to(ref_dtype).requires_grad_(True)
    trap = inputs["trap"].detach().to(ref_dtype).requires_grad_(True)
    d = inputs["D"].detach().to(ref_dtype).requires_grad_(True)

    dA_cs_base, dA_cs_rev_base, _ = mods.utils.compute_dacs_segsum_triton(
        inputs["dA"].detach().to(torch.float32), inputs["chunk_size"]
    )
    dA_cs = dA_cs_base.detach().to(ref_dtype).requires_grad_(True)
    dA_cs_rev = dA_cs_rev_base.detach().to(ref_dtype).requires_grad_(True)

    out, _, _ = mamba3_MIMO_chunk_ref(
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
        d,
        chunk_size=inputs["chunk_size"],
        rotary_dim_divisor=inputs["rotary_dim_divisor"],
        dtype=ref_dtype,
        contract_mimo_out=contract_mimo_out,
    )

    grad_input_items = [
        ("q", q),
        ("k", k),
        ("v", v),
        ("q_bias", q_bias),
        ("k_bias", k_bias),
        ("mimo_v", mimo_v),
        ("angles", angles),
        ("dA_cs", dA_cs),
        ("dA_cs_rev", dA_cs_rev),
        ("dt", dt),
        ("trap", trap),
        ("dD", d),
    ]
    if z is not None:
        grad_input_items.append(("z", z))
    if mimo_z is not None:
        grad_input_items.append(("mimo_z", mimo_z))
    if contract_mimo_out:
        grad_input_items.append(("mimo_o", mimo_o))

    if grad_output is None:
        grad_output = inputs["dout"]
    grads = torch.autograd.grad(
        outputs=out,
        inputs=tuple(t for _, t in grad_input_items),
        grad_outputs=grad_output.detach().to(ref_dtype),
        retain_graph=False,
        allow_unused=True,
    )
    grad_map = {name: grad for (name, _), grad in zip(grad_input_items, grads)}
    grad_map["dA"] = grads_to_dA(grad_map["dA_cs"], grad_map["dA_cs_rev"], inputs["chunk_size"])

    return {
        "dq": grad_map["q"],
        "dk": grad_map["k"],
        "dv": grad_map["v"],
        "dA": grad_map["dA"],
        "ddt": grad_map["dt"],
        "dtrap": grad_map["trap"],
        "dq_bias": grad_map["q_bias"],
        "dk_bias": grad_map["k_bias"],
        "dmimo_v": grad_map["mimo_v"],
        "dmimo_z": grad_map.get("mimo_z"),
        "dmimo_o": grad_map.get("mimo_o"),
        "dangles": grad_map["angles"],
        "dD": grad_map["dD"],
        "dz": grad_map.get("z"),
    }


def test_mamba3_MIMO_chunk_ref_matches_step_ref() -> None:
    # Lightweight deterministic ref-vs-ref consistency test.
    B, S, H, G, P, N, R, chunk_size = 1, 128, 8, 1, 32, 64, 4, 16
    dtype = torch.float32
    device = "cpu"
    torch.manual_seed(0)

    q = torch.randn((B, S, R, G, N), device=device, dtype=dtype)
    k = torch.randn((B, S, R, G, N), device=device, dtype=dtype)
    v = torch.randn((B, S, H, P), device=device, dtype=dtype)

    q_bias = torch.randn((H, R, N), device=device, dtype=dtype)
    k_bias = torch.randn((H, R, N), device=device, dtype=dtype)
    mimo_v = torch.rand((H, R, P), device=device, dtype=dtype)
    mimo_o = torch.rand((H, R, P), device=device, dtype=dtype)

    z = torch.randn_like(v)
    mimo_z = torch.rand_like(mimo_v)
    D = torch.randn((H,), device=device, dtype=dtype)

    angles = torch.rand((B, S, H, N // 2), device=device, dtype=dtype)
    dt = F.softplus(-3.0 + torch.randn(B, H, S, device=device, dtype=torch.float32))
    A_neg = -F.softplus(torch.randn((B, H, S), device=device, dtype=torch.float32))
    A_neg = torch.clamp(A_neg, max=-1e-4)
    ADT = A_neg * dt
    trap = torch.rand(B, H, S, device=device, dtype=dtype) * 0.5

    dA_cs = torch.cumsum(rearrange(ADT, "b h (n c) -> b h n c", c=chunk_size), dim=-1)
    dA_cs_rev = dA_cs[..., -1:] - dA_cs
    angles_prerotated = apply_angle_dt_reference(angles, dt.permute(0, 2, 1))

    chunk_out, _, _ = mamba3_MIMO_chunk_ref(
        q,
        k,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        z,
        mimo_z,
        angles_prerotated,
        dA_cs.view(B, H, S),
        dA_cs_rev.view(B, H, S),
        dt,
        trap,
        D,
        chunk_size=chunk_size,
        rotary_dim_divisor=2,
        return_final_state=True,
        dtype=dtype,
        rotate_pairwise=True,
    )

    step_out, _ = mamba3_MIMO_step_ref(
        q,
        k,
        v,
        ADT,
        dt,
        trap,
        q_bias,
        k_bias,
        angles,
        mimo_v,
        mimo_o,
        D=D,
        Z=z,
        MIMO_Z=mimo_z,
    )

    assert chunk_out.shape == step_out.shape
    assert_stable_rel(
        chunk_out,
        step_out,
        label="chunk_ref_vs_step_ref",
        cfg=f"B={B}, S={S}, H={H}, P={P}, N={N}, R={R}, C={chunk_size}",
        rel_tol=0.02,
    )


@pytest.mark.parametrize("n,p,r,chunk_size,bb_threads", CASE_GRID)
def test_fused_chunk_linear_attn_fwd_relative_error_lt_10pct(
    mods: SimpleNamespace, n: int, p: int, r: int, chunk_size: int, bb_threads: int
) -> None:
    del bb_threads
    inputs = build_inputs(
        mods=mods,
        n=n,
        p=p,
        r=r,
        chunk_size=chunk_size,
        seed=1234 + n + p + r + chunk_size,
    )

    out_tilelang, _, _ = mods.fwd.mamba_mimo_forward(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["mimo_v"],
        inputs["mimo_o"],
        inputs["z"],
        inputs["D"],
        inputs["mimo_z"],
        inputs["angles"],
        inputs["dA_cs"],
        inputs["dA_cs_rev"],
        inputs["dt"],
        inputs["trap"],
        inputs["segsum"],
        chunk_size=chunk_size,
        rotary_dim_divisor=inputs["rotary_dim_divisor"],
        dtype=FIXED_DTYPE,
    )

    out_ref_fp32, _, _ = mamba3_MIMO_chunk_ref(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["q_bias"].clone(),
        inputs["k_bias"].clone(),
        inputs["mimo_v"].clone(),
        inputs["mimo_o"].clone(),
        inputs["z"].clone(),
        inputs["mimo_z"].clone(),
        inputs["angles"].clone(),
        inputs["dA_cs"].clone(),
        inputs["dA_cs_rev"].clone(),
        inputs["dt"].clone(),
        inputs["trap"].clone(),
        inputs["D"].clone(),
        chunk_size=chunk_size,
        rotary_dim_divisor=inputs["rotary_dim_divisor"],
        dtype=torch.float32,
    )

    assert_stable_rel(
        out_tilelang,
        out_ref_fp32,
        label="forward",
        cfg=f"N={n}, P={p}, R={r}, chunk={chunk_size}",
    )


@pytest.mark.parametrize("n,p,r,chunk_size,bb_threads", CASE_GRID)
def test_fused_chunk_linear_attn_fwd_return_state_relative_error_lt_10pct(
    mods: SimpleNamespace, n: int, p: int, r: int, chunk_size: int, bb_threads: int
) -> None:
    del bb_threads
    inputs = build_inputs(
        mods=mods,
        n=n,
        p=p,
        r=r,
        chunk_size=chunk_size,
        seed=3456 + n + p + r + chunk_size,
    )

    out_tilelang, final_state_tilelang, final_k_tilelang = mods.fwd.mamba_mimo_forward(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["mimo_v"],
        inputs["mimo_o"],
        inputs["z"],
        inputs["D"],
        inputs["mimo_z"],
        inputs["angles"],
        inputs["dA_cs"],
        inputs["dA_cs_rev"],
        inputs["dt"],
        inputs["trap"],
        inputs["segsum"],
        return_state=True,
        chunk_size=chunk_size,
        rotary_dim_divisor=inputs["rotary_dim_divisor"],
        dtype=FIXED_DTYPE,
    )

    out_ref_fp32, final_state_ref, final_k_ref = mamba3_MIMO_chunk_ref(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["q_bias"].clone(),
        inputs["k_bias"].clone(),
        inputs["mimo_v"].clone(),
        inputs["mimo_o"].clone(),
        inputs["z"].clone(),
        inputs["mimo_z"].clone(),
        inputs["angles"].clone(),
        inputs["dA_cs"].clone(),
        inputs["dA_cs_rev"].clone(),
        inputs["dt"].clone(),
        inputs["trap"].clone(),
        inputs["D"].clone(),
        chunk_size=chunk_size,
        rotary_dim_divisor=inputs["rotary_dim_divisor"],
        return_final_state=True,
        dtype=torch.float32,
    )

    assert_stable_rel(
        out_tilelang,
        out_ref_fp32,
        label="forward_return_state_out",
        cfg=f"N={n}, P={p}, R={r}, chunk={chunk_size}",
    )
    assert_stable_rel(
        final_state_tilelang,
        final_state_ref,
        label="forward_return_state_final_state",
        cfg=f"N={n}, P={p}, R={r}, chunk={chunk_size}",
    )
    assert_stable_rel(
        final_k_tilelang,
        final_k_ref,
        label="forward_return_state_final_k",
        cfg=f"N={n}, P={p}, R={r}, chunk={chunk_size}",
    )


@pytest.mark.parametrize("n,p,r,chunk_size,bb_threads", CASE_GRID)
def test_fused_chunk_linear_attn_fwd_prereduce_relative_error_lt_10pct(
    mods: SimpleNamespace, n: int, p: int, r: int, chunk_size: int, bb_threads: int
) -> None:
    del bb_threads
    inputs = build_inputs(
        mods=mods,
        n=n,
        p=p,
        r=r,
        chunk_size=chunk_size,
        seed=2345 + n + p + r + chunk_size,
        has_z=False,
    )

    out_tilelang, _, _ = mods.fwd.mamba_mimo_forward(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["mimo_v"],
        None,
        inputs["z"],
        inputs["D"],
        inputs["mimo_z"],
        inputs["angles"],
        inputs["dA_cs"],
        inputs["dA_cs_rev"],
        inputs["dt"],
        inputs["trap"],
        inputs["segsum"],
        chunk_size=chunk_size,
        rotary_dim_divisor=inputs["rotary_dim_divisor"],
        dtype=FIXED_DTYPE,
    )

    out_ref_fp32, _, _ = mamba3_MIMO_chunk_ref(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["q_bias"].clone(),
        inputs["k_bias"].clone(),
        inputs["mimo_v"].clone(),
        None,
        None,
        None,
        inputs["angles"].clone(),
        inputs["dA_cs"].clone(),
        inputs["dA_cs_rev"].clone(),
        inputs["dt"].clone(),
        inputs["trap"].clone(),
        inputs["D"].clone(),
        chunk_size=chunk_size,
        rotary_dim_divisor=inputs["rotary_dim_divisor"],
        dtype=torch.float32,
        contract_mimo_out=False,
    )

    assert_stable_rel(
        out_tilelang,
        out_ref_fp32,
        label="forward_prereduce",
        cfg=f"N={n}, P={p}, R={r}, chunk={chunk_size}",
    )


@pytest.mark.parametrize("n,p,r,chunk_size,bb_threads", CASE_GRID)
def test_mamba_mimo_bwd_combined_relative_errors_lt_10pct(
    mods: SimpleNamespace, n: int, p: int, r: int, chunk_size: int, bb_threads: int
) -> None:
    inputs = build_inputs(
        mods=mods,
        n=n,
        p=p,
        r=r,
        chunk_size=chunk_size,
        seed=5678 + n + p + r + chunk_size,
    )

    ref_grads = run_ref_backward_fp32(mods, inputs)

    (
        dq,
        dk,
        dv,
        dA,
        ddt,
        dtrap,
        dq_bias,
        dk_bias,
        dmimo_v,
        dmimo_z,
        dmimo_o,
        dangles,
        dD,
        dz,
    ) = mods.bwd.mamba_mimo_bwd_combined(
        inputs["dout"],
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["mimo_v"],
        inputs["mimo_o"],
        inputs["z"],
        inputs["mimo_z"],
        inputs["angles"],
        inputs["dA_cs"],
        inputs["dA_cs_rev"],
        inputs["dt"],
        inputs["trap"],
        inputs["D"],
        inputs["segsum"],
        chunk_size,
        inputs["rotary_dim_divisor"],
        FIXED_DTYPE,
        bb_threads=bb_threads,
    )

    comparisons = {
        "dq": (dq, ref_grads["dq"]),
        "dk": (dk, ref_grads["dk"]),
        "dv": (dv, ref_grads["dv"]),
        "dA": (dA, ref_grads["dA"]),
        "ddt": (ddt, ref_grads["ddt"]),
        "dtrap": (dtrap, ref_grads["dtrap"]),
        "dq_bias": (dq_bias, ref_grads["dq_bias"]),
        "dk_bias": (dk_bias, ref_grads["dk_bias"]),
        "dmimo_v": (dmimo_v, ref_grads["dmimo_v"]),
        "dmimo_z": (dmimo_z, ref_grads["dmimo_z"]),
        "dmimo_o": (dmimo_o, ref_grads["dmimo_o"]),
        "dangles": (dangles, ref_grads["dangles"]),
        "dD": (dD, ref_grads["dD"]),
        "dz": (dz, ref_grads["dz"]),
    }

    for name, (ours, ref) in comparisons.items():
        assert_stable_rel(
            ours,
            ref,
            label=name,
            cfg=f"N={n}, P={p}, R={r}, chunk={chunk_size}, bb_threads={bb_threads}",
        )


@pytest.mark.parametrize("n,p,r,chunk_size,bb_threads", CASE_GRID)
def test_mamba_mimo_bwd_combined_prereduce_relative_errors_lt_10pct(
    mods: SimpleNamespace, n: int, p: int, r: int, chunk_size: int, bb_threads: int
) -> None:
    inputs = build_inputs(
        mods=mods,
        n=n,
        p=p,
        r=r,
        chunk_size=chunk_size,
        seed=6789 + n + p + r + chunk_size,
        has_z=False,
    )
    b, s, h, p_dim = inputs["v"].shape
    dout_prereduce = torch.randn((b, s, r, h, p_dim), device="cuda", dtype=FIXED_DTYPE)

    ref_grads = run_ref_backward_fp32(
        mods,
        inputs,
        contract_mimo_out=False,
        grad_output=dout_prereduce,
    )

    (
        dq,
        dk,
        dv,
        dA,
        ddt,
        dtrap,
        dq_bias,
        dk_bias,
        dmimo_v,
        dmimo_z,
        dmimo_o,
        dangles,
        dD,
        dz,
    ) = mods.bwd.mamba_mimo_bwd_combined(
        dout_prereduce,
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["mimo_v"],
        None,
        None,
        None,
        inputs["angles"],
        inputs["dA_cs"],
        inputs["dA_cs_rev"],
        inputs["dt"],
        inputs["trap"],
        inputs["D"],
        inputs["segsum"],
        chunk_size,
        inputs["rotary_dim_divisor"],
        FIXED_DTYPE,
        bb_threads=bb_threads,
    )
    assert dmimo_o is None
    assert dmimo_z is None
    assert dz is None

    comparisons = {
        "dq_prereduce": (dq, ref_grads["dq"]),
        "dk_prereduce": (dk, ref_grads["dk"]),
        "dv_prereduce": (dv, ref_grads["dv"]),
        "dA_prereduce": (dA, ref_grads["dA"]),
        "ddt_prereduce": (ddt, ref_grads["ddt"]),
        "dtrap_prereduce": (dtrap, ref_grads["dtrap"]),
        "dq_bias_prereduce": (dq_bias, ref_grads["dq_bias"]),
        "dk_bias_prereduce": (dk_bias, ref_grads["dk_bias"]),
        "dmimo_v_prereduce": (dmimo_v, ref_grads["dmimo_v"]),
        "dangles_prereduce": (dangles, ref_grads["dangles"]),
        "dD_prereduce": (dD, ref_grads["dD"]),
    }

    for name, (ours, ref) in comparisons.items():
        assert_stable_rel(
            ours,
            ref,
            label=name,
            cfg=f"N={n}, P={p}, R={r}, chunk={chunk_size}, bb_threads={bb_threads}",
        )


def test_mamba_mimo_smoke_forward_backward(mods: SimpleNamespace) -> None:
    inputs = make_smoke_inputs(
        batch=FIXED_B,
        seqlen=FIXED_S,
        mimo_rank=4,
        nheads_qk=FIXED_G,
        nheads=FIXED_H,
        headdim_qk=128,
        headdim_v=64,
        chunk_size=16,
        rotary_dim_divisor=FIXED_ROTARY_DIM_DIVISOR,
        device="cuda",
        dtype=FIXED_DTYPE,
        seed=999,
    )

    out = mods.top.mamba3_mimo(**inputs)
    assert out.shape == (FIXED_B, FIXED_S, FIXED_H, 64)

    loss = out.float().sum()
    loss.backward()

    grad_names = [
        "Q",
        "K",
        "V",
        "ADT",
        "DT",
        "Trap",
        "Q_bias",
        "K_bias",
        "MIMO_V",
        "MIMO_Z",
        "MIMO_Out",
        "Angles",
        "D",
        "Z",
    ]
    for name in grad_names:
        grad = inputs[name].grad
        assert grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(grad).all(), f"Non-finite gradient detected for {name}"
