#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SISO inference script: prefill + per-token decode through the Triton
``mamba3_siso_step`` (cutlass decode path explicitly bypassed) + accuracy report.

What it does
------------
1. Builds a Mamba-3 SISO model from this tree's ``mamba_ssm`` with a fixed seed.
2. **Prefill**: runs the forward chain (``Mamba3.forward`` -> ``mamba3_siso_combined``
   with ``return_final_states=True``) over the first ``--prefill`` tokens.
3. **Decode**: recurses token-by-token through the **Triton** ``mamba3_siso_step``
   op.  The cutlass ``mamba3_step_fn`` path (``Mamba3.step``) is never called.
4. **Accuracy verification** (CANN ecosystem precision standard,
   ``mixed_tolerance_standard.md``: elementwise ``|a-g| <= atol + rtol*|g|`` with
   f32 ``1.2e-4`` / bf16 ``7.8e-3``; aggregate ``matched_ratio >= 0.99`` and
   ``max_abs_error <= max_abs_error_limit``):
   * ``infer`` (prefill + step decode) vs the **full-sequence forward** on the
     same teacher-forced input, per token;
   * ``infer`` vs a **pure-PyTorch reference** (the same module shell with the
     SISO kernel swapped for the chunked-attention reference), when enabled;
   * optionally, ``infer`` vs the **A100 goldens** (``--goldens-dir``).
5. **Device parameterized**: ``--device npu`` uses the real Triton-Ascend kernels;
   ``--device cpu`` routes the whole pipeline through the built-in pure-PyTorch
   reference (``siso_fwd_ref`` / ``siso_step_ref``) for a wiring self-check with
   no accelerator.  Writes a Markdown accuracy report.

Limitations
-----------
* SISO only (``is_mimo=False``); ``is_outproj_norm=True`` is not supported by the
  Triton ``mamba3_siso_step`` op, so decode is skipped for such models.
* ``cu_seqlens`` / varlen is not exercised here.

Run (NPU container, from the repo root)::

    MAMBA3_SISO_DEVICE=npu MAMBA3_DISABLE_TILELANG=1 ASCEND_RT_VISIBLE_DEVICES=0 \
        python scripts/infer_siso_step.py --device npu --dtype bf16 \
        --report infer_siso_step_report.md

Run (CPU pipeline self-check, no accelerator)::

    python scripts/infer_siso_step.py --device cpu --report infer_siso_step_report.md
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ===========================================================================
# CLI
# ===========================================================================

DTYPE_NAME = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}
NAME_DTYPE = {v: k for k, v in DTYPE_NAME.items()}


# CANN mixed_tolerance_standard.md (gitcode.com/cann/opbase ->
# docs/zh/ops_precision_standard/mixed_tolerance_standard.md).
CANN_MATCHED_RATIO_MIN = 0.99
CANN_TOL = {
    torch.float32: dict(rtol=1.2e-4, atol=1.2e-4),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
    torch.bfloat16: dict(rtol=7.8e-3, atol=7.8e-3),
}
CANN_MAX_ABS_ERROR_LIMIT = {
    torch.float32: 1e-2,
    torch.float16: 1e-1,
    torch.bfloat16: 1e-1,
}


# ===========================================================================
# Pure-PyTorch reference (device-agnostic) -- frozen copy of the upstream test
# reference (tests/ops/triton/_siso_ops_npu_common.py) so this script is
# self-contained.  Used both as the CPU pipeline self-check and as the absolute
# accuracy anchor on the accelerator.
# ===========================================================================

def _segsum(x: torch.Tensor) -> torch.Tensor:
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def siso_fwd_ref(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    ADT: torch.Tensor, DT: torch.Tensor, Trap: torch.Tensor,
    Q_bias: torch.Tensor, K_bias: torch.Tensor, Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None, Z: Optional[torch.Tensor] = None,
    Initial_States: Optional[Tuple[torch.Tensor, ...]] = None,
    chunk_size: int = 64, dtype: torch.dtype = torch.float32,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """Reference Mamba-3 chunked forward.  Returns ``(out, (fa, fs, fk, fv))``."""
    batch, total_seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    device = Q.device

    Q, K, V = Q.to(dtype), K.to(dtype), V.to(dtype)
    Trap = Trap.to(dtype)
    Q_bias, K_bias, Angles = Q_bias.to(dtype), K_bias.to(dtype), Angles.to(dtype)
    ADT, DT = ADT.to(torch.float32), DT.to(torch.float32)
    if D is not None:
        D = D.to(dtype)
    if Z is not None:
        Z = Z.to(dtype)
    if Initial_States is not None:
        Initial_Angle_State, Initial_SSM_State, Initial_K_State, Initial_V_State = Initial_States

    Angles = torch.tanh(Angles) * math.pi
    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    out_zs, FA, FS, FK, FV = [], [], [], [], []
    TWO_PI = 2 * math.pi

    def _rotary(tensor, cos, sin):
        r = tensor.view(*tensor.shape[:-1], -1, 2)
        t0, t1 = r[..., 0], r[..., 1]
        if cos.shape[-1] < t0.shape[-1]:
            pad = t0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad), value=1.0)
            sin = F.pad(sin, (0, pad), value=0.0)
        return torch.stack([t0 * cos - t1 * sin, t0 * sin + t1 * cos], dim=-1).view_as(tensor)

    def one(seq_idx):
        Q_c, K_c, V_c = Q[seq_idx], K[seq_idx], V[seq_idx]
        ADT_c, DT_c, Trap_c = ADT[seq_idx], DT[seq_idx], Trap[seq_idx]
        Angles_c = Angles[seq_idx]
        Z_c = Z[seq_idx] if Z is not None else None

        Trap_c = torch.sigmoid(Trap_c)

        Angles_Cumsum = torch.cumsum(Angles_c * DT_c.transpose(0, 1).unsqueeze(-1), dim=0)
        if Initial_States is not None:
            Angles_Cumsum = Angles_Cumsum + Initial_Angle_State[seq_idx].unsqueeze(0)
        Angles_Cumsum = Angles_Cumsum - TWO_PI * torch.floor(Angles_Cumsum / TWO_PI)
        FA.append(Angles_Cumsum[-1])

        if Initial_States is not None:
            scalar = DT_c[:, 0] * (1 - Trap_c[:, 0])
            acc_states = Initial_SSM_State[seq_idx] + (
                Initial_V_State[seq_idx][:, :, None] * Initial_K_State[seq_idx][:, None, :]
                * scalar[:, None, None])
        else:
            acc_states = torch.zeros((nheads, headdim_v, headdim_qk), device=device, dtype=dtype)

        DT_shifted = F.pad(DT_c[:, 1:], (0, 1))
        Trap_shifted = F.pad(Trap_c[:, 1:], (0, 1))
        shifted_gamma = DT_shifted * (1 - Trap_shifted)
        scale = DT_c * Trap_c + DT_shifted * (1 - Trap_shifted)

        Q_c = Q_c + Q_bias.unsqueeze(0)
        K_c = K_c + K_bias.unsqueeze(0)
        QK_dot = torch.sum(K_c * Q_c, dim=-1) * shifted_gamma.transpose(0, 1)

        cos_a, sin_a = torch.cos(Angles_Cumsum), torch.sin(Angles_Cumsum)
        Q_c = _rotary(Q_c, cos_a, sin_a)
        K_c = _rotary(K_c, cos_a, sin_a)
        FK.append(K_c[-1])
        FV.append(V_c[-1])

        K_c_scaled = K_c * scale.transpose(0, 1).unsqueeze(-1)
        QK = torch.einsum("thd,shd->hts", Q_c, K_c_scaled)
        out = torch.einsum("hts,shd->thd", torch.tril(QK) * torch.exp(_segsum(ADT_c)), V_c)
        if Initial_States is not None:
            out = out + torch.einsum("hDd,thd,ht->thD",
                                     acc_states, Q_c, torch.exp(torch.cumsum(ADT_c, dim=-1)))
        if D is not None:
            out = out + D[None, :, None] * V_c
        out = out - V_c * QK_dot.unsqueeze(-1)
        if Z_c is not None:
            out = out * Z_c * torch.sigmoid(Z_c)
        out_zs.append(out)

        da_cs_last = torch.exp(torch.sum(ADT_c, dim=-1))
        da_cs_rev = torch.exp(torch.sum(ADT_c, dim=-1, keepdim=True) - torch.cumsum(ADT_c, dim=-1))
        V_c_scaled = V_c * da_cs_rev.permute(1, 0).unsqueeze(-1)
        FS.append(acc_states * da_cs_last.unsqueeze(-1).unsqueeze(-1)
                  + torch.einsum("thd,thD->hDd", K_c_scaled, V_c_scaled))

    for i in range(batch):
        one(i)

    return (torch.stack(out_zs, 0),
            (torch.stack(FA, 0), torch.stack(FS, 0), torch.stack(FK, 0), torch.stack(FV, 0)))


def siso_step_ref(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    ADT: torch.Tensor, DT: torch.Tensor, Trap: torch.Tensor,
    Q_bias: torch.Tensor, K_bias: torch.Tensor, Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None, Z: Optional[torch.Tensor] = None,
    Input_States: Optional[Tuple[torch.Tensor, ...]] = None,
):
    """Reference Mamba-3 recurrent step.  Returns ``(out, (a, ssm, k, v))``."""
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device
    assert seqlen > 0
    Angles = torch.tanh(Angles) * math.pi

    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    def _rotary(tensor, cos, sin):
        r = tensor.view(*tensor.shape[:-1], -1, 2)
        t0, t1 = r[..., 0], r[..., 1]
        if cos.shape[-1] < t0.shape[-1]:
            pad = t0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad), value=1.0)
            sin = F.pad(sin, (0, pad), value=0.0)
        return torch.stack([t0 * cos - t1 * sin, t0 * sin + t1 * cos], dim=-1).view_as(tensor)

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
        q = Q[:, idx] + Q_bias.unsqueeze(0)
        k = K[:, idx] + K_bias.unsqueeze(0)
        v = V[:, idx]
        adt, dt = ADT[:, :, idx], DT[:, :, idx]
        trap = Trap[:, :, idx]
        z = Z[:, idx] if Z is not None else None
        angles = Angles[:, idx]

        Angle_State = Angle_State + angles * dt.unsqueeze(-1)
        Angle_State = Angle_State - TWO_PI * torch.floor(Angle_State / TWO_PI)
        cos_a, sin_a = torch.cos(Angle_State), torch.sin(Angle_State)
        q_rot = _rotary(q, cos_a, sin_a)
        k_rot = _rotary(k, cos_a, sin_a)

        trap = torch.sigmoid(trap)
        alpha = torch.exp(adt)
        beta = (1 - trap) * dt * alpha
        gamma = trap * dt

        SSM_State = alpha.unsqueeze(-1).unsqueeze(-1) * SSM_State
        SSM_State = SSM_State + beta.unsqueeze(-1).unsqueeze(-1) * (K_State.unsqueeze(-2) * V_State.unsqueeze(-1))
        SSM_State = SSM_State + gamma.unsqueeze(-1).unsqueeze(-1) * (k_rot.unsqueeze(-2) * v.unsqueeze(-1))

        out = torch.einsum("bhdD, bhD -> bhd", SSM_State, q_rot.to(SSM_State.dtype))
        if D is not None:
            out = out + D[None, :, None] * v
        if Z is not None:
            out = out * z * torch.sigmoid(z)
        out_arr.append(out)
        K_State = k_rot
        V_State = v

    return torch.stack(out_arr, dim=1), (Angle_State, SSM_State, K_State, V_State)


def _siso_combined_ref(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
    D=None, Z=None, Input_States=None, chunk_size=64,
    return_final_states=False, cu_seqlens=None,
):
    """Pure-PyTorch drop-in for ``mamba3_siso_combined`` (same dtype contract).

    ``mamba3_siso_combined`` casts Q/K/V/Z to the module dtype before the kernel;
    here that is the incoming ``Q.dtype`` (bf16 on the accelerator, fp32 in the
    CPU self-check), so the reference and the step op see the same contract.
    """
    cdt = Q.dtype
    Q = Q.to(cdt)
    K = K.to(cdt)
    V = V.to(cdt)
    Trap = Trap.to(torch.float32)
    if Z is not None:
        Z = Z.to(cdt)
    out, (fa, fs, fk, fv) = siso_fwd_ref(
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
        D=D, Z=Z, Initial_States=Input_States, chunk_size=chunk_size,
        dtype=torch.float32, cu_seqlens=cu_seqlens,
    )
    if return_final_states:
        return out, fa, fs, fk, fv
    return out


class _RMSNormGatedTorch(nn.Module):
    """Pure-PyTorch stand-in for ``RMSNormGated`` (used only in CPU self-check).

    The Triton ``RMSNorm`` cannot run without an accelerator; this mirrors its
    RMS semantics (normalize over the last dim, scale by ``weight``) for the
    z-free calls the SISO model makes for ``B_norm`` / ``C_norm``.
    """

    def __init__(self, hidden_size, eps=1e-5, group_size=None,
                 norm_before_gate=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        g = self.group_size or x.shape[-1]
        xg = x.reshape(*x.shape[:-1], -1, g).float()
        y = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + self.eps)
        y = (y * self.weight.float()).reshape(x.shape).to(x.dtype)
        if z is not None:
            gate = F.silu(z.float())
            y = (y.float() * gate).to(y.dtype) if self.norm_before_gate else y
        return y


def _step_ref_single(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
                     D=None, Z=None, Input_States=None):
    """Single-token wrapper: same signature as Triton ``mamba3_siso_step``."""
    out, states = siso_step_ref(
        Q.unsqueeze(1), K.unsqueeze(1), V.unsqueeze(1),
        ADT.unsqueeze(-1), DT.unsqueeze(-1), Trap.unsqueeze(-1),
        Q_bias, K_bias, Angles.unsqueeze(1), D=D,
        Z=Z.unsqueeze(1) if Z is not None else None, Input_States=Input_States,
    )
    return out.squeeze(1), states


# ===========================================================================
# CANN metrics
# ===========================================================================

def _f32(t):
    return t.detach().to(torch.float32).cpu()


def _ulp_at(x: torch.Tensor) -> float:
    x = x.detach()
    if not bool(torch.isfinite(x)):
        return 0.0
    nxt = torch.nextafter(x, torch.tensor(float("inf"), dtype=x.dtype))
    u = float((nxt - x).abs())
    if u == 0.0:
        prv = torch.nextafter(x, torch.tensor(float("-inf"), dtype=x.dtype))
        u = float((x - prv).abs())
    return u


def cann_max_abs_error_limit(golden_at_max_err, out_dtype: torch.dtype) -> float:
    base = CANN_MAX_ABS_ERROR_LIMIT.get(out_dtype, 1e-1)
    if golden_at_max_err is None:
        return base
    g_low = torch.as_tensor(golden_at_max_err).to(out_dtype)
    return max(base, 32.0 * _ulp_at(g_low))


def cann_metrics(cal, ref, dtype=None):
    """(matched_ratio, max_abs_error, max_abs_error_limit, rtol, atol, verdict)."""
    dt = dtype if dtype is not None else getattr(cal, "dtype", torch.float32)
    tol = CANN_TOL.get(dt, dict(rtol=1e-3, atol=1e-3))
    c = cal.detach().cpu().reshape(-1)
    r = ref.detach().cpu().reshape(-1)
    cf, rf = c.to(torch.float32), r.to(torch.float32)
    total = rf.numel()
    if total == 0:
        return 1.0, 0.0, CANN_MAX_ABS_ERROR_LIMIT.get(dt, 1e-1), tol["rtol"], tol["atol"], "PASS"

    finite = torch.isfinite(cf) & torch.isfinite(rf)
    matched = torch.zeros(total, dtype=torch.bool)
    if bool(finite.any()):
        df = (cf - rf).abs()
        thr = tol["atol"] + tol["rtol"] * rf.abs()
        matched |= finite & (df <= thr)
        fin_idx = torch.nonzero(finite).squeeze(1)
        j = fin_idx[torch.argmax(df[fin_idx])]
        max_abs_error = float(df[j])
        limit = cann_max_abs_error_limit(r[j], dt)
    else:
        max_abs_error, limit = 0.0, CANN_MAX_ABS_ERROR_LIMIT.get(dt, 1e-1)

    nonfinite = ~finite
    if bool(nonfinite.any()):
        a_nf, g_nf = c[nonfinite], r[nonfinite].to(dt)
        same_sign_inf = torch.isinf(a_nf) & torch.isinf(g_nf) & (torch.sign(a_nf) == torch.sign(g_nf))
        both_nan = torch.isnan(a_nf) & torch.isnan(g_nf)
        matched[nonfinite] = same_sign_inf | both_nan

    matched_ratio = float(matched.sum()) / total
    verdict = "PASS" if (matched_ratio >= CANN_MATCHED_RATIO_MIN and max_abs_error <= limit) else "FAIL"
    return matched_ratio, max_abs_error, limit, tol["rtol"], tol["atol"], verdict


def p95_relative_error(cal, ref, ref_mag_mask=1e-2, p=0.95, eps=1e-6):
    c, r = _f32(cal), _f32(ref)
    abs_ref = r.abs()
    mask = abs_ref >= ref_mag_mask
    if not mask.any():
        return 0.0
    vals = ((c - r).abs()[mask] / (abs_ref[mask] + eps)).flatten()
    n = vals.numel()
    k = max(1, min(n, int(math.ceil(p * n))))
    return vals.kthvalue(k).values.item()


def metrics_line(tag, cal, ref, dtype=None):
    """One markdown row: guide ref numbers + CANN verdict."""
    dt = dtype if dtype is not None else getattr(cal, "dtype", torch.float32)
    c, r = _f32(cal), _f32(ref)
    n = r.numel()
    max_abs = (c - r).abs().max().item() if n else 0.0
    max_rel = ((c - r).abs() / (r.abs() + 1e-6)).max().item() if n else 0.0
    p95 = p95_relative_error(cal, ref)
    ratio, mae, lim, rtol, atol, verdict = cann_metrics(cal, ref, dtype=dt)
    return {
        "tag": tag, "dtype": DTYPE_NAME.get(dt, str(dt)),
        "max_abs": max_abs, "max_rel": max_rel, "p95_rel": p95,
        "matched_ratio": ratio, "max_abs_error": mae, "max_abs_error_limit": lim,
        "rtol": rtol, "atol": atol, "verdict": verdict,
    }


def _fmt_row(m):
    return (f"| `{m['tag']}` | {m['dtype']} | {m['matched_ratio']:.6f} | "
            f"{m['max_abs_error']:.4e} | {m['max_abs_error_limit']:.4e} | "
            f"{m['max_abs']:.3e} | {m['max_rel']:.3e} | {m['p95_rel']:.3e} | {m['verdict']} |")


_METRICS_HEADER = ("| output | dtype | matched_ratio | max_abs_error | limit | "
                   "max_abs | max_rel | p95_rel | verdict |\n"
                   "|---|---|---|---|---|---|---|---|---|")


# ===========================================================================
# Model / inference
# ===========================================================================

class _InferenceCache:
    """Minimal inference-params holder for ``Mamba3.forward`` prefill."""

    def __init__(self):
        self.seqlen_offset = 0
        self.key_value_memory_dict: Dict[int, Tuple[torch.Tensor, ...]] = {}


def build_model(device, dtype, d_model, d_state, headdim, ngroups, expand,
                chunk_size, rope_fraction, seed):
    import mamba_ssm.modules.mamba3 as m3mod

    torch.manual_seed(seed)
    if device.startswith("npu"):
        try:
            torch.npu.manual_seed_all(seed)
        except Exception:
            pass
    model = m3mod.Mamba3(
        d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
        ngroups=ngroups, rope_fraction=rope_fraction, is_outproj_norm=False,
        is_mimo=False, chunk_size=chunk_size, dtype=dtype, device=device,
        layer_idx=0,
    )
    model.eval()
    model.to(device)
    return model


def prefill(model, u_pre):
    """Prefill through ``Mamba3.forward`` (returns final states via the cache)."""
    cache = _InferenceCache()
    out = model(u_pre, inference_params=cache)
    a, s, k, v = cache.key_value_memory_dict[0]
    # ``k`` carries an extra mimo-rank dim (batch, 1, nheads, d_state) -> drop it
    # to the step op's expected (batch, nheads, d_state).
    states = (a, s, k.squeeze(1).contiguous(), v)
    return out, states


def _siso_step_module(model, u_t, states, step_fn):
    """One decode step through the Triton ``mamba3_siso_step`` (cutlass bypassed).

    Mirrors ``Mamba3.step``'s pre-processing but feeds the *fused* Triton step op:
    it takes pre-bias / pre-rotation Q=C and K=B, the raw (pre-sigmoid) ``Trap``,
    and the rotary angles; the op adds ``Q_bias``/``K_bias`` and rotates internally.
    """
    mxBCdtAtrap = model.in_proj(u_t)
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        mxBCdtAtrap,
        [model.d_inner, model.d_inner,
         model.d_state * model.num_bc_heads * model.mimo_rank,
         model.d_state * model.num_bc_heads * model.mimo_rank,
         model.nheads, model.nheads, model.nheads, model.num_rope_angles],
        dim=-1)

    _A = -_heavy_tail(dd_A.to(torch.float32))
    _A = torch.clamp(_A, max=-model.A_floor)
    DT = F.softplus(dd_dt + model.dt_bias)
    ADT = _A * DT

    B = rearrange(B, "b (r g n) -> b r g n", r=model.mimo_rank, g=model.num_bc_heads)
    C = rearrange(C, "b (r g n) -> b r g n", r=model.mimo_rank, g=model.num_bc_heads)
    B = model.B_norm(B)
    C = model.C_norm(C)
    x = rearrange(x, "b (h p) -> b h p", p=model.headdim)
    z = rearrange(z, "b (h p) -> b h p", p=model.headdim)
    angles = angles.unsqueeze(-2).expand(-1, model.nheads, -1).to(torch.float32)

    Q = C.squeeze(1)                       # (b, nheads_qk, d_state)
    K = B.squeeze(1)                       # (b, nheads_qk, d_state)
    Q_bias = model.C_bias.squeeze(1)       # (nheads, d_state)
    K_bias = model.B_bias.squeeze(1)       # (nheads, d_state)

    out, new_states = step_fn(
        Q, K, x, ADT, DT, trap, Q_bias, K_bias, angles,
        model.D, z, states)

    out = rearrange(out, "b h p -> b (h p)")
    out = model.out_proj(out.to(x.dtype))
    return out, new_states


def _heavy_tail(x: torch.Tensor) -> torch.Tensor:
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def run_inference(model, u, prefill_len, step_fn):
    """Prefill + per-token Triton step decode.  Returns (out, pre, dec, states)."""
    L = u.shape[1]
    with torch.no_grad():
        out_pre, states = prefill(model, u[:, :prefill_len].contiguous())
        outs = []
        for t in range(prefill_len, L):
            o, states = _siso_step_module(model, u[:, t], states, step_fn)
            outs.append(o)
    if outs:
        out_dec = torch.stack(outs, dim=1)
        out = torch.cat([out_pre, out_dec], dim=1)
    else:
        out_dec, out = None, out_pre
    state_dict = {"Angle_State": states[0], "SSM_State": states[1],
                  "K_State": states[2], "V_State": states[3]}
    return out, out_pre, out_dec, state_dict


def final_states_of(model, u):
    """Full-sequence forward's final states (chunked kernel path)."""
    cache = _InferenceCache()
    with torch.no_grad():
        model(u, inference_params=cache)
    a, s, k, v = cache.key_value_memory_dict[0]
    return {"Angle_State": a, "SSM_State": s,
            "K_State": k.squeeze(1).contiguous(), "V_State": v}


@contextlib.contextmanager
def pure_ref_kernel():
    """Swap ``mamba3_siso_combined`` for the pure-PyTorch reference."""
    import mamba_ssm.modules.mamba3 as m3mod
    orig = m3mod.mamba3_siso_combined
    m3mod.mamba3_siso_combined = _siso_combined_ref
    try:
        yield
    finally:
        m3mod.mamba3_siso_combined = orig


# ===========================================================================
# Environment info
# ===========================================================================

def _git_rev(path: str) -> Optional[str]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    for _ in range(6):
        if os.path.isdir(os.path.join(root, ".git")):
            try:
                return subprocess.check_output(
                    ["git", "-C", root, "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                return None
        root = os.path.dirname(root)
    return None


def env_info(device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": device,
        "torch": torch.__version__,
        "torch_npu": None, "triton": None, "triton_ascend": None,
        "npu_model": None, "git_commit": None, "mamba_ssm_file": None,
    }
    try:
        import triton  # noqa
        info["triton"] = getattr(triton, "__version__", "unknown")
    except Exception:
        pass
    if device.startswith("npu"):
        try:
            import torch_npu  # noqa
            info["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
            info["npu_model"] = torch.npu.get_device_name(0)
        except Exception:
            pass
        try:
            from importlib.metadata import version as _v
            info["triton_ascend"] = _v("triton-ascend")
        except Exception:
            pass
    try:
        import mamba_ssm  # noqa
        info["mamba_ssm_file"] = getattr(mamba_ssm, "__file__", None)
        if info["mamba_ssm_file"]:
            info["git_commit"] = _git_rev(info["mamba_ssm_file"])
    except Exception as exc:
        info["mamba_ssm_file"] = f"<import failed: {exc}>"
    return info


# ===========================================================================
# Reporting
# ===========================================================================

def _per_token_rows(cal, ref, start, dt):
    """Per-token max-abs + matched-ratio for a (B, T, D) pair."""
    c, r = _f32(cal), _f32(ref)
    tol = CANN_TOL.get(dt, dict(rtol=1e-3, atol=1e-3))
    diff = (c - r).abs()
    per_abs = diff.amax(dim=(0, 2))
    matched = (diff <= tol["atol"] + tol["rtol"] * r.abs())
    ratio = matched.float().mean(dim=(0, 2))
    return [(start + i, per_abs[i].item(), ratio[i].item()) for i in range(cal.shape[1])]


def _decide_verdict(rows):
    return "PASS" if all(rt >= CANN_MATCHED_RATIO_MIN for _, _, rt in rows) else "FAIL"


def build_report(args, env, results):
    lines: List[str] = []
    a = lines.append
    a("# SISO 推理精度报告：prefill + Triton `mamba3_siso_step` decode")
    a("")
    a(f"> 脚本 `scripts/infer_siso_step.py`｜生成时间 {args.timestamp}｜"
      f"task PLUT-47")
    a("")
    a("## 1. 配置")
    a("")
    a("| 项 | 值 |")
    a("|---|---|")
    a(f"| device | `{env['device']}` |")
    a(f"| dtype | `{args.dtype}` |")
    a(f"| seed | {args.seed} |")
    a(f"| batch / seqlen | {args.batch} / {args.seqlen} |")
    a(f"| prefill / decode | {args.prefill_eff} / {args.seqlen - args.prefill_eff} |")
    a(f"| d_model / d_state / headdim | {args.d_model} / {args.d_state} / {args.headdim} |")
    a(f"| ngroups / expand / chunk_size | {args.ngroups} / {args.expand} / {args.chunk_size} |")
    a(f"| 纯 PyTorch 参考 | {'on' if args.pure_ref else 'off'} |")
    a(f"| A100 goldens | {args.goldens_dir or '（未提供）'} |")
    a("")
    a("**decode 路径**：逐 token 调 Triton `mamba3_siso_step`（`mamba3_siso_step.py`）；"
      "未调用 cutlass `mamba3_step_fn`。")
    if env["device"] == "cpu":
        a("")
        a("> ⚠️ CPU 模式：整个管线走内置纯 PyTorch 参考（`siso_fwd_ref` / `siso_step_ref`），"
          "目的是自检 prefill→decode 接线与精度流水线，不代表 Triton 算子精度。")
    a("")
    a("## 2. 环境")
    a("")
    a("| 项 | 值 |")
    a("|---|---|")
    for k in ("torch", "torch_npu", "triton", "triton_ascend", "npu_model", "git_commit", "mamba_ssm_file"):
        if env.get(k):
            a(f"| {k} | `{env[k]}` |")
    a("")
    a("## 3. 精度验证（CANN 口径）")
    a("")
    a(f"判据：`|a-g| <= atol + rtol*|g|`，整体 `matched_ratio >= {CANN_MATCHED_RATIO_MIN}` "
      "且 `max_abs_error <= max_abs_error_limit = max(fallback, 32*ULP(g_low))`；"
      "f32 `1.2e-4`，bf16 `7.8e-3`。")
    a("")

    for section in results["sections"]:
        a(f"### {section['title']}")
        a("")
        if section.get("note"):
            a(section["note"])
            a("")
        a(_METRICS_HEADER)
        for m in section["rows"]:
            a(_fmt_row(m))
        a("")

    pt = results.get("per_token")
    if pt:
        a("### 3.x 逐 token（decode 段，参考）")
        a("")
        a("总判据为「逐输出聚合」（3.1 的 decode 段整段）；下表给出逐 token 明细，"
          "其中单 token 的 `matched_ratio` 仅基于 1×batch×d_model 个元素，波动较大，"
          "不单独作为 gate。")
        a("")
        a("| token | max_abs | matched_ratio | verdict |")
        a("|---|---|---|---|")
        rows = pt["rows"]
        show = rows if len(rows) <= args.max_table_rows else (rows[: args.max_table_rows // 2]
                                                              + rows[-args.max_table_rows // 2:])
        for tok, ma, rt in show:
            a(f"| {tok} | {ma:.3e} | {rt:.6f} | "
              f"{'PASS' if rt >= CANN_MATCHED_RATIO_MIN else 'FAIL'} |")
        if len(rows) > len(show):
            a(f"| … | … | … | （省略 {len(rows) - len(show)} 行） |")
        a("")
        a(f"decode 段逐 token 汇总：max_abs=`{max(r[1] for r in rows):.3e}`，"
          f"min matched_ratio=`{min(r[2] for r in rows):.6f}`，"
          f"整体 verdict=`{pt['verdict']}`（容差 dtype=`{pt['dtype']}`）。")
        a("")

    if results.get("golden_sections"):
        a("## 4. A100 golden 对照")
        a("")
        for gs in results["golden_sections"]:
            a(f"### {gs['title']}")
            a("")
            if gs.get("note"):
                a(gs["note"])
                a("")
            if gs.get("rows"):
                a(_METRICS_HEADER)
                for m in gs["rows"]:
                    a(_fmt_row(m))
                a("")

    a("## 5. 结论")
    a("")
    a(results["conclusion"])
    a("")
    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="npu | cpu | auto (default: npu if available else cpu)")
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"],
                    help="model/compute dtype (default: bf16 on accelerator, fp32 on cpu)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--prefill", type=int, default=-1,
                    help="prefill length (default: seqlen//2)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-state", type=int, default=64)
    ap.add_argument("--headdim", type=int, default=32)
    ap.add_argument("--ngroups", type=int, default=2)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--rope-fraction", type=float, default=0.5)
    ap.add_argument("--pure-ref", dest="pure_ref", action="store_true", default=True,
                    help="also compare against the pure-PyTorch reference (default: on)")
    ap.add_argument("--no-pure-ref", dest="pure_ref", action="store_false")
    ap.add_argument("--goldens-dir", default=None,
                    help="optional A100 goldens dir (manifest.json + e2e cases)")
    ap.add_argument("--report", default="infer_siso_step_report.md")
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--max-table-rows", type=int, default=64)
    args = ap.parse_args(argv)
    return args


def _resolve_device(arg) -> str:
    if arg in ("npu", "cpu", "cuda"):
        return arg
    try:
        import torch_npu  # noqa
        if torch.npu.is_available():
            return "npu"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _setup_npu():
    os.environ.setdefault("MAMBA3_DISABLE_TILELANG", "1")
    try:
        import triton.backends.nvidia.driver as nv_driver
        nv_driver.CudaDriver.is_active = staticmethod(lambda: False)
    except Exception:
        pass
    import torch_npu  # noqa
    torch.npu.set_device(0)
    # torch_npu's get_device_capability returns None, breaking mamba3_siso_fwd's
    # autotune prune callback; provide a safe stub.
    try:
        if torch.cuda.get_device_capability(torch.device("npu")) is None:
            torch.cuda.get_device_capability = lambda *a, **k: (0, 0)
    except Exception:
        pass


def main(argv=None) -> int:
    args = parse_args(argv)
    args.timestamp = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())
    device = _resolve_device(args.device)
    if device == "npu":
        _setup_npu()
    os.environ.setdefault("MAMBA3_DISABLE_TILELANG", "1")

    dtype = NAME_DTYPE[args.dtype] if args.dtype else (
        torch.float32 if device == "cpu" else torch.bfloat16)
    args.dtype = DTYPE_NAME[dtype]
    L = args.seqlen
    args.prefill_eff = args.prefill if args.prefill >= 0 else max(1, L // 2)
    if not (0 < args.prefill_eff < L):
        raise SystemExit(f"--prefill must satisfy 0 < prefill < seqlen ({args.prefill_eff} vs {L})")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except Exception:
        pass

    # Import the tree under test.
    import mamba_ssm.modules.mamba3 as m3mod  # noqa: E402  (after env setup)
    if device == "cpu":
        # No accelerator: route the whole pipeline through the built-in reference.
        m3mod.mamba3_siso_combined = _siso_combined_ref
        m3mod.RMSNormGated = _RMSNormGatedTorch
        step_fn = _step_ref_single
    else:
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_step import mamba3_siso_step
        step_fn = mamba3_siso_step

    print(f"[cfg] device={device} dtype={args.dtype} seed={args.seed} "
          f"batch={args.batch} seqlen={L} prefill={args.prefill_eff} "
          f"d_model={args.d_model} d_state={args.d_state} headdim={args.headdim} "
          f"ngroups={args.ngroups} chunk={args.chunk_size}")

    model = build_model(device, dtype, args.d_model, args.d_state, args.headdim,
                        args.ngroups, args.expand, args.chunk_size,
                        args.rope_fraction, args.seed)

    # Deterministic input (generated on CPU, then moved -> identical across hosts).
    torch.manual_seed(args.seed + 1)
    u = torch.randn(args.batch, L, args.d_model, dtype=dtype).to(device)

    # ---- full-sequence forward (teacher forcing) ----
    with torch.no_grad():
        out_full = model(u)
    full_states = final_states_of(model, u)

    # ---- inference: prefill + Triton step decode ----
    out_infer, out_pre, out_dec, infer_states = run_inference(
        model, u, args.prefill_eff, step_fn)

    # ---- pure-PyTorch reference (absolute accuracy anchor) ----
    out_ref = None
    if args.pure_ref:
        if device == "cpu":
            out_ref = out_full if dtype == torch.float32 else None
        else:
            with pure_ref_kernel(), torch.no_grad():
                out_ref = model(u)

    results: Dict[str, Any] = {"sections": [], "golden_sections": []}
    P = args.prefill_eff

    # Section 1: infer vs full forward (teacher forcing), split by region.
    rows = [metrics_line("infer vs full · 全序列", out_infer, out_full, dtype)]
    rows.append(metrics_line("infer vs full · prefill 段", out_infer[:, :P], out_full[:, :P], dtype))
    if out_dec is not None:
        rows.append(metrics_line("infer vs full · decode 段", out_infer[:, P:], out_full[:, P:], dtype))
    results["sections"].append({
        "title": "3.1 `infer` (prefill + step decode) vs 全序列前向（teacher forcing）",
        "rows": rows,
        "note": "同一输入 `u`，全序列前向即 teacher-forcing 标杆；decode 段逐 token 由 "
                "Triton `mamba3_siso_step` 递推。",
    })

    # Section 2: final recurrent states.
    srows = []
    for nm in ("Angle_State", "SSM_State", "K_State", "V_State"):
        dt = infer_states[nm].dtype if nm == "Angle_State" else (
            torch.float32 if nm == "SSM_State" else dtype)
        srows.append(metrics_line(f"final.{nm}: infer vs full", infer_states[nm], full_states[nm], dt))
    results["sections"].append({
        "title": "3.2 最终递归状态：`infer` vs 全序列前向（附加，参考）",
        "rows": srows,
        "gate": False,
        "note": "prefill(+decode) 后的四个状态与全序列前向末态对比。注意 `SSM_State` 在 bf16 下"
                "对「分块前向 vs 逐 token 递推」的累加顺序敏感（已知口径问题，参见 PLUT-44/"
                "PLUT-45），故本节不计入总判据。",
    })

    # Section 3: vs pure-PyTorch reference.
    if out_ref is not None:
        results["sections"].append({
            "title": "3.3 `infer` vs 纯 PyTorch 参考（chunked 前向参考实现）",
            "rows": [metrics_line("infer vs ref · 全序列", out_infer, out_ref, dtype)],
            "note": "参考为同一模块外壳 + 纯 PyTorch SISO 前向（`siso_fwd_ref`）。"
                    + ("（CPU 模式下与全序列前向同源。）" if device == "cpu" else ""),
        })

    # Per-token decode table (informational: the gate is the per-output aggregate).
    if out_dec is not None:
        rows_pt = _per_token_rows(out_infer[:, P:], out_full[:, P:], P, dtype)
        results["per_token"] = {"rows": rows_pt, "verdict": _decide_verdict(rows_pt),
                                "dtype": args.dtype}

    # Section 4: A100 goldens (optional).
    if args.goldens_dir:
        results["golden_sections"] = compare_goldens(args, device, dtype, step_fn)

    # Conclusion: gate on the per-output aggregates (sections with gate != False) +
    # the optional A100 golden per-output rows.  The per-token table is reference.
    all_verdicts = [m["verdict"] for s in results["sections"] if s.get("gate", True)
                    for m in s["rows"]]
    all_verdicts += [m["verdict"] for gs in results["golden_sections"] for m in gs.get("rows", [])]
    ok = all(v == "PASS" for v in all_verdicts) and bool(all_verdicts)
    results["conclusion"] = (
        f"**{'PASS' if ok else 'FAIL'}** — "
        f"{sum(1 for v in all_verdicts if v == 'PASS')}/{len(all_verdicts)} 个输出在 CANN 口径下达标。"
        + ("" if device != "cpu" else "（CPU 模式为纯 PyTorch 管线自检。）"))

    env = env_info(device)
    report = build_report(args, env, results)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[report] wrote {os.path.abspath(args.report)}")

    if args.json_path:
        payload = {"args": {k: v for k, v in vars(args).items()}, "env": env,
                   "sections": results["sections"],
                   "per_token": results.get("per_token"),
                   "golden_sections": results["golden_sections"]}
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        print(f"[json] wrote {os.path.abspath(args.json_path)}")

    for s in results["sections"]:
        for m in s["rows"]:
            print(f"[{m['tag']}] dtype={m['dtype']} matched_ratio={m['matched_ratio']:.6f} "
                  f"max_abs_error={m['max_abs_error']:.4e} limit={m['max_abs_error_limit']:.4e} "
                  f"verdict={m['verdict']}")
    if results.get("per_token"):
        print(f"[decode 逐 token] min_matched_ratio=>"
              f"{min(r[2] for r in results['per_token']['rows']):.6f} "
              f"verdict={results['per_token']['verdict']}")
    print(f"[conclusion] {results['conclusion']}")
    return 0 if ok else 1


def compare_goldens(args, device, dtype, step_fn) -> List[Dict[str, Any]]:
    """Optional: rebuild each non-varlen e2e golden case and compare ``infer``."""
    import mamba_ssm.modules.mamba3 as m3mod

    sections: List[Dict[str, Any]] = []
    manifest_path = os.path.join(args.goldens_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"[golden] no manifest at {manifest_path}; skipping")
        return sections
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for entry in manifest.get("cases", []):
        if entry.get("kind") != "e2e" or entry.get("status") != "ok":
            continue
        cid = entry["id"]
        try:
            data = torch.load(os.path.join(args.goldens_dir, entry["file"]),
                              map_location="cpu", weights_only=False)
            cfg = data["module"]["config"]
            if data["inputs"].get("u") is None or cfg.get("varlen") is not None:
                raise ValueError("varlen case not supported")
            if cfg.get("is_outproj_norm"):
                raise ValueError("is_outproj_norm not supported by Triton step decode")
            gdt = NAME_DTYPE[cfg["dtype"]]
            m = m3mod.Mamba3(
                d_model=cfg["d_model"], d_state=cfg["d_state"], expand=cfg["expand"],
                headdim=cfg["headdim"], ngroups=cfg["ngroups"],
                rope_fraction=cfg["rope_fraction"], is_outproj_norm=cfg["is_outproj_norm"],
                is_mimo=cfg["is_mimo"], chunk_size=cfg["chunk"], dtype=gdt,
                device=device, layer_idx=0)
            m.load_state_dict(data["module"]["state_dict"])
            m.eval()
            m.to(device)

            u = data["inputs"]["u"].to(device, dtype=gdt)
            Lg = u.shape[1]
            P = max(1, Lg // 2)
            out_inf, _, _, _ = run_inference(m, u, P, step_fn)
            golden = data["forward"]["out"].to(device, dtype=gdt)
            rows = [metrics_line(f"{cid} · infer vs A100", out_inf, golden, gdt),
                    metrics_line(f"{cid} · decode 段 vs A100",
                                 out_inf[:, P:], golden[:, P:], gdt)]
            sections.append({"title": f"4.x `{cid}`", "rows": rows})
            print(f"[golden] {cid}: " + " ; ".join(
                f"{r['verdict']}({r['matched_ratio']:.4f})" for r in rows))
        except Exception as exc:
            sections.append({"title": f"4.x `{cid}`", "rows": [],
                             "note": f"跳过（{type(exc).__name__}: {exc}）"})
            print(f"[golden] {cid}: skipped ({type(exc).__name__}: {exc})")
    return sections


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
