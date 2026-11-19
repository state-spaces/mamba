"""Shared helpers for the Mamba-3 SISO single-operator NPU test modules.

This module holds the device selection, the tolerance policy, the comparison
helpers and the pure-PyTorch reference implementations shared by the
per-operator test modules:

    test_angle_dt_fwd_ops_npu.py
    test_mamba3_siso_fwd_ops_npu.py
    test_mamba3_siso_step_ops_npu.py
    test_grouped_head_reduction_ops_npu.py
    test_angle_dt_ops_npu.py                 (legacy combined fwd + bwd)
    test_mamba3_siso_combined_ops_npu.py
    test_mamba3_siso_angle_dt_bwd_ops_npu.py         (reverse)
    test_mamba3_siso_dzdo_ops_npu.py                 (reverse)
    test_mamba3_siso_dqkv_ops_npu.py                 (reverse)
    test_mamba3_siso_dqktheta_ops_npu.py             (reverse)
    test_mamba3_siso_ddt_dtrap_dinput_states_ops_npu.py (reverse)

No test cases live here; the split is a pure reorganization (identical coverage
and assertions) for per-operator review.

ACCEPTANCE READING (read before interpreting a pytest run): each output prints
BOTH a guide-tolerance line (``max_abs=... max_rel=... p95_rel=... tol=...``)
and a CANN-standard line (``CANN matched_ratio=... max_abs_error=...
max_abs_error_limit=... verdict=PASS|FAIL``).  The **single** pass criterion is
the **CANN ecosystem precision standard** (gitcode.com/cann/opbase ->
docs/zh/ops_precision_standard/mixed_tolerance_standard.md): the pytest
assertion gates on the ``CANN ... verdict`` field (``matched_ratio >= 0.99``
AND ``max_abs_error <= max_abs_error_limit``).  The guide-tolerance line is
**reference-only** (the old migration-guide per-dtype tolerance is no longer a
gate), so a guide-line exceedance is informational and does not, by itself,
fail the case.

Golden strategy (see agent identity):
    * siso_step / siso_combined: the upstream pure-PyTorch reference recurrence /
      chunked forward (`mamba3_siso_step_ref`, `mamba3_siso_fwd_ref`), run on CPU,
      copied verbatim from `tests/ops/triton/test_mamba3_siso.py`.
    * combined backward: autograd through the CPU reference.
    * angle_dt / grouped-reduction goldens live in their own modules.
      Both sides are moved to `.cpu()` and compared in float32 before assertion.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from einops import repeat

try:  # register the "npu" device
    import torch_npu  # noqa: F401

    _HAS_TORCH_NPU = True
except Exception:  # pragma: no cover - CUDA/CPU-only environment
    _HAS_TORCH_NPU = False

TRITON_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"


def _default_device() -> str:
    """Prefer NPU, then CUDA.  A bare CPU host cannot run these Triton kernels."""
    if _HAS_TORCH_NPU and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("MAMBA3_SISO_DEVICE", _default_device())
if DEVICE == "npu":
    try:
        torch.npu.set_device(0)
    except Exception:  # pragma: no cover
        pass

# The SISO operators are Triton kernels and cannot execute on a bare CPU host.
# Skip the test modules unless an accelerator is present, or the caller
# explicitly opts into the Triton CPU interpreter.
_SKIP_REASON = None
if DEVICE == "npu" and not _HAS_TORCH_NPU:
    _SKIP_REASON = "MAMBA3_SISO_DEVICE=npu but torch_npu is not importable"
elif DEVICE == "cuda" and not torch.cuda.is_available():
    _SKIP_REASON = "MAMBA3_SISO_DEVICE=cuda but CUDA is not available"
elif DEVICE == "cpu" and not TRITON_INTERPRET:
    _SKIP_REASON = (
        "no NPU/CUDA accelerator for the Triton kernels; set "
        "MAMBA3_SISO_DEVICE, or run on CPU with TRITON_INTERPRET=1"
    )

# Each test module does ``pytestmark = MODULE_SKIP``.
MODULE_SKIP = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

# Per-dtype MIGRATION-GUIDE tolerances -- REFERENCE ONLY (no longer a gate).
# Kept so the historical ``[name] ... tol=...`` line still carries the
# guide's per-dtype values; the pass/fail gate is the CANN verdict
# (see ``_report_and_check``).  Do NOT assert on these values any more.
DTYPE_TOL = {
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
    torch.bfloat16: dict(rtol=5e-3, atol=5e-3),
}

# Upstream-observed tolerances for the bf16 siso kernels (fast tanh/sin/cos
# approximations + long-sequence accumulation). TODO: tighten after 910B
# re-measurement once the migration is numerically confirmed.
SISO_STEP_TOL = dict(rtol=5e-2, atol=5e-2)
SISO_COMBINED_TOL = dict(rtol=1e-1, atol=1e-1)

# angle_dt_bwd's dDT: the kernel's chunked reverse-cumsum reorders the fp32 sums
# versus torch's full-sequence cumsum, so grad_dt lands ~1e-4 relative away while
# the forward operator itself matches to ~1e-6.  Measured on 910B:
# abs=2.29e-5 / rel=1.38e-4.  This is an accumulation-ordering effect (fp32
# rounding), not a math error; dAngle/dInit stay at the strict fp32 tolerance.
# TODO: re-measure and tighten after the second-round kernel changes.
ANGLE_DT_BWD_DDT_TOL = dict(rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32).cpu()


def assert_close(cal, ref, rtol, atol, name="", equal_nan=True):
    """Assert two tensors are close after moving to CPU / float32."""
    assert cal.shape == ref.shape, f"{name}: shape {tuple(cal.shape)} != {tuple(ref.shape)}"
    c, r = _f32(cal), _f32(ref)
    torch.testing.assert_close(
        c, r, rtol=rtol, atol=atol, equal_nan=equal_nan, msg=f"[{name}] tensor mismatch",
    )


def assert_angle_close(cal, ref, atol, name=""):
    """Compare angles modulo 2*pi via the wrapped absolute delta."""
    d = _f32(cal) - _f32(ref)
    two_pi = 2 * math.pi
    d = torch.remainder(d + math.pi, two_pi) - math.pi
    torch.testing.assert_close(
        d, torch.zeros_like(d), rtol=0.0, atol=atol, equal_nan=True,
        msg=f"[{name}] wrapped angle delta exceeded atol={atol}",
    )


def p95_relative_error(cal, ref, ref_mag_mask=1e-2, p=0.95, eps=1e-6):
    """p-th percentile relative error over elements with |ref| >= ref_mag_mask.

    Mirrors the upstream ``relative_error`` (p95) semantics: it discounts
    small-|ref| cancellation points (where a tiny absolute deviation blows up
    the ratio) and reports the systematic relative error instead.
    """
    c, r = _f32(cal), _f32(ref)
    abs_ref = r.abs()
    abs_diff = (c - r).abs()
    mask = abs_ref >= ref_mag_mask
    if not mask.any():
        return 0.0
    vals = (abs_diff[mask] / (abs_ref[mask] + eps)).flatten()
    n = vals.numel()
    k = max(1, min(n, int(math.ceil(p * n))))
    return vals.kthvalue(k).values.item()


def assert_p95_relative(cal, ref, rtol, name="", ref_mag_mask=1e-2):
    """Magnitude-aware weak-golden check: p95 relative error <= rtol."""
    assert cal.shape == ref.shape, f"{name}: shape {tuple(cal.shape)} != {tuple(ref.shape)}"
    err = p95_relative_error(cal, ref, ref_mag_mask=ref_mag_mask)
    assert err <= rtol, f"[{name}] p95 relative error {err:.3e} exceeds rtol={rtol}"


def to_cpu_leaf(d, requires_grad=True):
    """Deep CPU/float32 copy of an input dict; graph leaves when `requires_grad`.

    Casting the bf16 inputs up to float32 is exact for the values the kernel
    sees, so gradients w.r.t. these fp32 leaves equal the gradients w.r.t. the
    bf16 device tensors.
    """
    out = {}
    for k, v in d.items():
        if k == "Input_States":
            out[k] = tuple(
                t.detach().clone().to(device="cpu", dtype=torch.float32).requires_grad_(requires_grad)
                for t in v
            ) if v is not None else None
        elif isinstance(v, torch.Tensor):
            out[k] = v.detach().clone().to(device="cpu", dtype=torch.float32).requires_grad_(requires_grad)
        else:
            out[k] = None
    return out


# ===========================================================================
# Reference implementations (pure PyTorch, copied from
# tests/ops/triton/test_mamba3_siso.py so the suite is self-contained).
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
):
    """Reference Mamba-3 recurrent (step) implementation (device-agnostic)."""
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
        return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)

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

        Angle_State = Angle_State + angles * dt.unsqueeze(-1)
        Angle_State = Angle_State - TWO_PI * torch.floor(Angle_State / TWO_PI)

        cos_angles = torch.cos(Angle_State)
        sin_angles = torch.sin(Angle_State)
        q_rot = apply_rotary_emb(q, cos_angles, sin_angles)
        k_rot = apply_rotary_emb(k, cos_angles, sin_angles)

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

    out = torch.stack(out_arr, dim=1)
    return out, (Angle_State, SSM_State, K_State, V_State)


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
    angles_cumsum: Optional[torch.Tensor] = None,
    qkvd_dtype: Optional[torch.dtype] = None,
):
    """Reference Mamba-3 chunked forward pass (device-agnostic).

    ``angles_cumsum`` (optional): precomputed cumulative rotary angles in the
    same ``(batch, total_seqlen, nheads, headdim_angles)`` layout as ``Angles``
    (already wrapped into ``[0, 2*pi)`` and including any init angle state).
    When given, the internal ``cumsum(tanh(Angles)*pi*DT)`` step is skipped and
    these angles are used directly, so the reference isolates the forward kernel
    from the ``angle_dt_fwd`` pre-processing (see
    ``test_mamba3_siso_fwd_ops_npu.py``).

    ``qkvd_dtype`` (optional): same-precision mode.  When set, the reference
    reproduces the kernel's *mixed-precision* contract instead of the plain
    fp32/``dtype`` math:
        * Q, K, V, Z      -> ``qkvd_dtype`` (the kernel's compute dtype);
        * ADT, DT, Trap, Angles, Q_bias, K_bias, D -> fp32;
        * the kernel's low-precision intermediate stores are mirrored:
          rotated Q / scaled K (``Q_store``/``K_store``), the gated causal
          block fed to the second matmul, and the stored output ``Out`` are all
          rounded to ``qkvd_dtype``; the recurrent state stays fp32.
    This makes both sides carry the same quantization, so the residual is only
    accumulation/reduction order plus the fast transcendentals.  Default
    (``None``) keeps the original behaviour exactly.
    """
    batch, total_seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert batch == 1

    if qkvd_dtype is None:
        Q = Q.to(dtype)
        K = K.to(dtype)
        V = V.to(dtype)
        Trap = Trap.to(dtype)
        Q_bias = Q_bias.to(dtype)
        K_bias = K_bias.to(dtype)
        Angles = Angles.to(dtype)
        if D is not None:
            D = D.to(dtype)
        if Z is not None:
            Z = Z.to(dtype)
    else:
        # Mixed-precision contract actually seen by `mamba3_siso_fwd`: Q/K/V/Z in
        # the compute dtype, everything else fp32 (Trap and model params must NOT
        # be down-cast, which would introduce a new mismatch).
        Q = Q.to(qkvd_dtype)
        K = K.to(qkvd_dtype)
        V = V.to(qkvd_dtype)
        Trap = Trap.to(torch.float32)
        Q_bias = Q_bias.to(torch.float32)
        K_bias = K_bias.to(torch.float32)
        Angles = Angles.to(torch.float32)
        if D is not None:
            D = D.to(torch.float32)
        if Z is not None:
            Z = Z.to(qkvd_dtype)
    ADT = ADT.to(torch.float32)
    DT = DT.to(torch.float32)
    if Initial_States is not None:
        Initial_Angle_State, Initial_SSM_State, Initial_K_State, Initial_V_State = Initial_States

    Angles = torch.tanh(Angles) * math.pi
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

        if angles_cumsum is not None:
            # Precomputed cumulative angles (e.g. produced by `angle_dt_fwd`).
            # Used as-is so the reference tests the forward kernel given the
            # same angle input, decoupled from the angle pre-processing.
            if is_varlen:
                Angles_Cumsum = angles_cumsum[0, start_idx:end_idx, :, :]
            else:
                Angles_Cumsum = angles_cumsum[seq_idx]
        else:
            Angles_scaled = Angles_curr.float() * DT_curr.transpose(0, 1).unsqueeze(-1)
            Angles_Cumsum = torch.cumsum(Angles_scaled, dim=0)
            if Initial_States is not None:
                Initial_Angle_State_curr = Initial_Angle_State[seq_idx]
                Angles_Cumsum = Angles_Cumsum + Initial_Angle_State_curr.unsqueeze(0)
            Angles_Cumsum = Angles_Cumsum - TWO_PI * torch.floor(Angles_Cumsum / TWO_PI)
        Final_Angle_States.append(Angles_Cumsum[-1])

        if Initial_States is not None:
            Initial_SSM_State_curr = Initial_SSM_State[seq_idx]
            Initial_K_State_curr = Initial_K_State[seq_idx]
            Initial_V_State_curr = Initial_V_State[seq_idx]
            scalar = DT_curr[:, 0] * (1 - Trap_curr[:, 0])
            acc_states = Initial_SSM_State_curr + Initial_V_State_curr[:, :, None] * Initial_K_State_curr[:, None, :] * scalar[:, None, None]
        else:
            acc_states = torch.zeros((nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32)

        DT_shifted = F.pad(DT_curr[:, 1:], (0, 1))
        Trap_shifted = F.pad(Trap_curr[:, 1:], (0, 1))
        shifted_gamma = DT_shifted * (1 - Trap_shifted)
        scale = DT_curr * Trap_curr + DT_shifted * (1 - Trap_shifted)

        Q_curr = Q_curr + Q_bias.unsqueeze(0)
        K_curr = K_curr + K_bias.unsqueeze(0)

        QK_dot = torch.sum(K_curr * Q_curr, dim=-1) * shifted_gamma.transpose(0, 1)

        cos_angles_curr = torch.cos(Angles_Cumsum).to(Q_curr.dtype)
        sin_angles_curr = torch.sin(Angles_Cumsum).to(Q_curr.dtype)
        Q_curr = _rotary(Q_curr, cos_angles_curr, sin_angles_curr)
        K_curr = _rotary(K_curr, cos_angles_curr, sin_angles_curr)

        Final_K_States.append(K_curr[-1])
        Final_V_States.append(V_curr[-1])

        K_curr_scaled = K_curr * scale.transpose(0, 1).unsqueeze(-1).to(K_curr.dtype)

        # Same-precision mode mirrors the kernel's low-precision stores: rotated Q
        # and scaled K go to the Q_store / K_store buffers (compute dtype) and are
        # reloaded for the causal math; the gated block and the output are stored
        # the same way.  State accumulation stays fp32.
        if qkvd_dtype is not None:
            Q_used = Q_curr.to(qkvd_dtype)
            K_scaled_used = K_curr_scaled.to(qkvd_dtype)
            V_used = V_curr.to(qkvd_dtype)
        else:
            Q_used = Q_curr
            K_scaled_used = K_curr_scaled
            V_used = V_curr

        QK = torch.einsum("thd,shd->hts", Q_used, K_scaled_used)
        QK_causal = torch.tril(QK)
        QK_causal = (QK_causal * torch.exp(_segsum(ADT_curr))).to(QK_causal.dtype)
        if qkvd_dtype is not None:
            QK_causal = QK_causal.to(qkvd_dtype)  # kernel: s_block.to(v_sub.dtype)
        out = torch.einsum("hts,shd->thd", QK_causal, V_used)

        if Initial_States is not None:
            da_cs = torch.cumsum(ADT_curr, dim=-1)
            exp_da_cs = torch.exp(da_cs)
            if qkvd_dtype is not None:
                out = out + torch.einsum("hDd,thd,ht->thD",
                                         acc_states.to(qkvd_dtype),
                                         Q_used, exp_da_cs)
            else:
                out = out + torch.einsum("hDd,thd,ht->thD",
                                         acc_states.to(Q_curr.dtype),
                                         Q_curr, exp_da_cs.to(Q_curr.dtype))

        if D is not None:
            out = out + D[None, :, None] * V_used

        # NOTE (same-precision probe): the kernel also down-casts this skip QK
        # dot (``store_qk_dot = qk.to(k_e.dtype) * gamma``, mamba3_siso_fwd.py:328)
        # and reloads it in PHASE 1; the ref keeps QK_dot fp32 here, so it is a
        # small known residual of the ``qkvd_dtype`` probe (not of the main run).
        out = out - V_used * QK_dot.unsqueeze(-1)

        if Z_curr is not None:
            if qkvd_dtype is not None:
                # kernel applies the SiLU gate in fp32 to the fp32 accumulator
                zg = Z_curr.float()
                out = out * (zg * torch.sigmoid(zg))
            else:
                out = out * Z_curr * torch.sigmoid(Z_curr)
        if qkvd_dtype is not None:
            out = out.to(qkvd_dtype)  # kernel stores Out in V.dtype
        out_zs.append(out)

        da_cs_last = torch.exp(torch.sum(ADT_curr, dim=-1))
        da_cs_rev = torch.exp(torch.sum(ADT_curr, dim=-1, keepdim=True) - torch.cumsum(ADT_curr, dim=-1))
        V_curr_scaled = V_used * da_cs_rev.permute(1, 0).unsqueeze(-1).to(V_used.dtype)
        final_acc_states = acc_states * da_cs_last.unsqueeze(-1).unsqueeze(-1) + torch.einsum(
            "thd,thD->hDd", K_scaled_used, V_curr_scaled.to(K_scaled_used.dtype))
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


def create_mamba3_siso_inputs(
    batch, seqlen, nheads, nheads_qk, headdim_qk, headdim_v, dtype, device,
    has_D, has_Z, has_input_states, cu_seqlens=None, requires_grad=False,
):
    """Build a Mamba-3 SISO input dict on `device` (upstream helper, verbatim)."""
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
        for t in (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles):
            t.requires_grad_(True)
        if D is not None:
            D.requires_grad_(True)
        if Z is not None:
            Z.requires_grad_(True)
        if Input_States is not None:
            for s in Input_States:
                s.requires_grad_(True)

    return {
        "Q": Q, "K": K, "V": V,
        "ADT": ADT, "DT": DT, "Trap": Trap,
        "Q_bias": Q_bias, "K_bias": K_bias, "Angles": Angles,
        "D": D, "Z": Z, "Input_States": Input_States,
    }


# ===========================================================================
# PLUT-31 forward-operator additions (merged into the shared helper)
# ---------------------------------------------------------------------------
# Everything below is additive: the reverse (PLUT-30) suites that import from
# this module keep the exact same names and behaviour.  Added here are the
# forward-operator goldens (angle_dt_fwd / decoupled mamba3_siso_fwd) and the
# shared per-output reporter `_report_and_check` / aggregator `_check_all`,
# extended with the CANN ecosystem precision standard alongside the guide line.
# ===========================================================================

DTYPE_NAME = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

# angle_dt wraps mod 2*pi, so it is compared via a wrapped absolute delta.
ANGLE_TOL = {
    torch.float32: 1e-4,
    torch.float16: 1e-2,
    torch.bfloat16: 3e-2,
}


# --- CANN ecosystem operator precision standard -----------------------------
# Source: gitcode.com/cann/opbase -> docs/zh/ops_precision_standard/
#         mixed_tolerance_standard.md
#   elementwise  |actual - golden| <= atol + rtol * |golden|
#   aggregate    matched_ratio >= 0.99  and  max_abs_error <= max_abs_error_limit
#   thresholds   f16 atol=rtol=1e-3, bf16 7.8e-3, f32 1.2e-4   (§2.2)
#   max_abs_error_limit = max(fallback, 32*ULP(g_low))           (§2.1.2)
#     * fallback: f16/bf16 1e-1, f32 1e-2 (per §2.2 max_abs_error_limit row)
#     * g_low = the golden value AT THE MAX-ABS-ERROR POINT, narrowed to the
#       operator's ACTUAL OUTPUT dtype (round-to-nearest-even), and the ULP is
#       taken in that same output dtype.
#   max_abs_error = max |actual-golden| over FINITE comparison points only
#       (both actual and golden finite); ±inf/NaN points are handled by §2.3.
#   §2.3: for non-finite points, narrow the golden to the actual output dtype;
#       same-sign ±inf on both sides counts as a pass.
# This standard is looser than the migration guide; it is reported alongside the
# (stricter) guide line, which remains the test's assertion.  The user's
# acceptance criterion is the CANN `verdict`, not the pytest pass/fail.
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


def _ulp_at(x: torch.Tensor) -> float:
    """ULP at a scalar tensor ``x``, in ``x``'s OWN dtype (nextafter-based).

    Handles ``x == 0`` (returns the smallest subnormal step) and the top of the
    dtype range (falls back to the previous representable value).  Non-finite
    ``x`` returns 0.0 (a non-finite anchor contributes no ULP allowance).
    """
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
    """max_abs_error_limit = max(fallback, 32*ULP(g_low))  (§2.1.2).

    ``golden_at_max_err`` is the golden value at the maximum-abs-error point; it
    is narrowed to the operator's actual output dtype (``out_dtype``) and the ULP
    is taken there (``g_low`` in the standard).
    """
    base = CANN_MAX_ABS_ERROR_LIMIT.get(out_dtype, 1e-1)
    if golden_at_max_err is None:
        return base
    g_low = torch.as_tensor(golden_at_max_err).to(out_dtype)
    return max(base, 32.0 * _ulp_at(g_low))


def cann_metrics(cal, ref, dtype=None):
    """(matched_ratio, max_abs_error, max_abs_error_limit, rtol, atol, verdict).

    Follows the CANN mixed-tolerance standard:
      * the elementwise criterion ``|a-g| <= atol + rtol*|g|`` is evaluated on
        the finite comparison points;
      * ``max_abs_error`` is the max |a-g| over FINITE points only (§2.1.2);
      * non-finite points follow §2.3 (narrow the golden to the actual output
        dtype, same-sign ±inf passes);
      * ``max_abs_error_limit`` uses ``g_low`` = the golden at the max-error
        point narrowed to the output dtype, ULP taken in that dtype (§2.1.2).
    ``dtype`` (default: the actual output tensor's dtype) selects the §2.2 rows.
    """
    dt = dtype if dtype is not None else getattr(cal, "dtype", torch.float32)
    tol = CANN_TOL.get(dt, dict(rtol=1e-3, atol=1e-3))
    # Flatten every tensor to 1-D first so the boolean masks tensor mask,
    # `matched`, the finite index set and the gathered values all share one
    # (numel,) shape -- otherwise a multi-dim output (e.g. Out (b,s,h,d)) makes
    # `finite & (df <= thr)` / `matched[nonfinite]` broadcast-fail.
    c = cal.detach().cpu().reshape(-1)
    r = ref.detach().cpu().reshape(-1)
    cf = c.to(torch.float32)
    rf = r.to(torch.float32)
    total = rf.numel()
    if total == 0:
        return 1.0, 0.0, CANN_MAX_ABS_ERROR_LIMIT.get(dt, 1e-1), tol["rtol"], tol["atol"], "PASS"

    finite = torch.isfinite(cf) & torch.isfinite(rf)
    matched = torch.zeros(total, dtype=torch.bool)

    # --- finite comparison points (elementwise + max_abs_error anchor) --------
    if bool(finite.any()):
        df = (cf - rf).abs()
        thr = tol["atol"] + tol["rtol"] * rf.abs()
        matched |= finite & (df <= thr)
        fin_idx = torch.nonzero(finite).squeeze(1)
        j = fin_idx[torch.argmax(df[fin_idx])]
        max_abs_error = float(df[j])
        g_low = r[j]  # golden at the max-error point
        limit = cann_max_abs_error_limit(g_low, dt)
    else:
        max_abs_error = 0.0
        limit = CANN_MAX_ABS_ERROR_LIMIT.get(dt, 1e-1)

    # --- §2.3 non-finite points: narrow golden to the output dtype ------------
    nonfinite = ~finite
    if bool(nonfinite.any()):
        a_nf = c[nonfinite]                      # actual output dtype
        g_nf = r[nonfinite].to(dt)               # golden narrowed to output dtype
        same_sign_inf = (torch.isinf(a_nf) & torch.isinf(g_nf)
                         & (torch.sign(a_nf) == torch.sign(g_nf)))
        both_nan = torch.isnan(a_nf) & torch.isnan(g_nf)
        matched[nonfinite] = same_sign_inf | both_nan

    matched_ratio = float(matched.sum()) / total
    verdict = "PASS" if (matched_ratio >= CANN_MATCHED_RATIO_MIN and max_abs_error <= limit) else "FAIL"
    return matched_ratio, max_abs_error, limit, tol["rtol"], tol["atol"], verdict


def _report_and_check(name, cal, ref, tol):
    """Print the guide (reference) line + the CANN line, then gate on CANN.

    Shared by every per-operator module (forward and reverse) so each output is
    reported against BOTH standards in one bound log:
        [name] dtype=... max_abs=... max_rel=... p95_rel=... tol=... (reference-only)
        [name] CANN matched_ratio=... max_abs_error=... limit=... verdict=...
    The single pass criterion is the CANN verdict (``matched_ratio >= 0.99`` AND
    ``max_abs_error <= max_abs_error_limit``).  The guide-tolerance line is
    reference-only: it keeps the historical ``max_abs/max_rel/p95_rel`` numbers
    but no longer gates the case, so there is exactly one PASS/FAIL judgement.
    ``tol`` is therefore only used to render the reference line, not asserted.
    """
    c, r = _f32(cal), _f32(ref)
    assert c.shape == r.shape, f"[{name}] shape {tuple(c.shape)} != {tuple(r.shape)}"
    n = r.numel()
    max_abs = (c - r).abs().max().item() if n else 0.0
    max_rel = ((c - r).abs() / (r.abs() + 1e-6)).max().item() if n else 0.0
    p95 = p95_relative_error(c, r)
    print(f"[{name}] dtype={cal.dtype} max_abs={max_abs:.3e} "
          f"max_rel={max_rel:.3e} p95_rel={p95:.3e} tol={tol} (reference-only)")
    m_ratio, m_abs, m_lim, crt, cat, verdict = cann_metrics(cal, ref)
    print(f"[{name}] CANN matched_ratio={m_ratio:.6f} (>= {CANN_MATCHED_RATIO_MIN}) "
          f"max_abs_error={m_abs:.6e} max_abs_error_limit={m_lim:.6e} "
          f"cann_tol=(rtol={crt:g},atol={cat:g}) verdict={verdict}")
    # Single pass criterion: the CANN verdict (mixed tolerance §2.1.2).
    assert verdict == "PASS", (
        f"[{name}] CANN verdict=FAIL matched_ratio={m_ratio:.6f} "
        f"(>= {CANN_MATCHED_RATIO_MIN}) max_abs_error={m_abs:.6e} "
        f"max_abs_error_limit={m_lim:.6e} dtype={cal.dtype} "
        f"cann_tol=(rtol={crt:g},atol={cat:g})"
    )


def _check_all(cal, ref, tag, tol=None, out_names=None):
    """Report every output (never stop at the first), then fail once.

    The pass/fail gate is the CANN verdict for each output (see
    ``_report_and_check``); ``tol`` (default: the guide value for each output's
    own dtype) only renders the reference guide-tolerance line. ``out_names``
    labels the outputs (defaults to ``out0..outN``); ``None`` entries are allowed
    (asserted ``None`` on both sides).
    """
    names = out_names or [f"out{i}" for i in range(len(cal))]
    failures = []
    for nm, c, r in zip(names, cal, ref):
        if r is None:
            assert c is None, f"{tag}.{nm}: expected None, got {type(c)}"
            continue
        eff = tol if tol is not None else DTYPE_TOL.get(c.dtype, dict(rtol=1e-5, atol=1e-5))
        try:
            _report_and_check(f"{tag}.{nm}", c, r, eff)
        except AssertionError as exc:  # noqa: PERF203 - collect, don't stop
            failures.append(f"{tag}.{nm}: {exc}")
    assert not failures, " | ".join(failures)


def report_check(tag, out_name, dtype, cal, ref, rtol, atol):
    """Single-output check + CANN-standard line; gates on the CANN verdict.

    ``rtol``/``atol`` only render the reference guide-tolerance line.
    """
    _report_and_check(f"{tag}.{out_name}", cal, ref, dict(rtol=rtol, atol=atol))


def report_angle_check(tag, out_name, dtype, cal, ref, atol):
    """Wrapped (mod 2*pi) absolute-delta check + CANN-standard line for angles.

    The single pass criterion is the CANN verdict on the ``CANN ... verdict=``
    line: ``matched_ratio >= 0.99`` AND ``wrapped max_abs_error <=
    max_abs_error_limit``.  The wrapped-atol line above it is reference-only.

    Angles are compared as a wrapped absolute delta, so the elementwise
    threshold is ``|wrapped_delta| <= atol`` (there is no meaningful
    ``rtol*|golden|`` term: angles wrap mod 2*pi and the correct golden is
    circular).  The aggregate hard cap stays the standard's
    ``max_abs_error_limit = max(fallback, 32*ULP(g_low))`` keyed on the output
    ``dtype`` (§2.2/§2.1.2), anchored at the wrapped-golden value at the
    max-error point.
    """
    d = _f32(cal) - _f32(ref)
    two_pi = 2 * math.pi
    d = torch.remainder(d + math.pi, two_pi) - math.pi
    n = d.numel()
    viol = int((d.abs() > atol).sum())
    max_abs = d.abs().max().item() if n else 0.0
    print(f"[{tag}.{out_name}] dtype={DTYPE_NAME.get(dtype, str(dtype))} "
          f"wrapped_max_abs={max_abs:.6e} tol=(atol={atol:g}) viol={viol}/{n} (reference-only)")
    matched_ratio = float((d.abs() <= atol).sum()) / n if n else 1.0
    # CANN hard cap: max(fallback, 32*ULP) anchored at the golden at the
    # max-|delta| point, narrowed to the operator's output dtype (§2.1.2).
    if n:
        j = int(d.abs().reshape(-1).argmax().item())
        limit = cann_max_abs_error_limit(_f32(ref).reshape(-1)[j], dtype)
    else:
        limit = CANN_MAX_ABS_ERROR_LIMIT.get(dtype, 1e-1)
    verdict = "PASS" if (matched_ratio >= CANN_MATCHED_RATIO_MIN and max_abs <= limit) else "FAIL"
    print(f"[{tag}.{out_name}] CANN matched_ratio={matched_ratio:.6f} (>= {CANN_MATCHED_RATIO_MIN}) "
          f"max_abs_error={max_abs:.6e} max_abs_error_limit={limit:.6e} verdict={verdict}")
    # Single pass criterion for wrapped angles: the CANN verdict above.
    assert verdict == "PASS", (
        f"[{tag}] {out_name} CANN verdict=FAIL matched_ratio={matched_ratio:.6f} "
        f"(>= {CANN_MATCHED_RATIO_MIN}) wrapped_max_abs={max_abs:.6e} "
        f"max_abs_error_limit={limit:.6e} atol={atol:g} viol={viol}/{n}"
    )


# --- forward-only goldens ---------------------------------------------------

def angle_dt_vals(angle, dt):
    """tanh(angle) * pi * dt in (batch, seqlen, nheads, dim)."""
    return torch.tanh(angle.float()) * math.pi * dt.float().permute(0, 2, 1).unsqueeze(-1)


def angle_dt_cumsum_ref(angle, dt, init_state=None):
    """Golden for angle_dt_fwd: cumsum(tanh(angle)*pi*dt) (+init), then mod 2*pi."""
    cum = torch.cumsum(angle_dt_vals(angle, dt), dim=1)
    if init_state is not None:
        cum = cum + init_state.float().unsqueeze(1)
    two_pi = 2 * math.pi
    return cum - two_pi * torch.floor(cum / two_pi)


def mamba3_siso_fwd_from_cumsum_ref(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles_Cumsum,
    D=None, Z=None, Initial_States=None, chunk_size=64,
    dtype=torch.float32, cu_seqlens=None,
):
    """Decoupled ``mamba3_siso_fwd`` golden: pre-cumsummed angles given directly.

    Thin wrapper over ``mamba3_siso_fwd_ref(..., angles_cumsum=Angles_Cumsum)``.
    ``Initial_States`` is the operator's 3-tuple ``(SSM, K, V)``; the angle state
    is already folded into ``Angles_Cumsum`` (so a ``None`` angle slot is used
    internally and never read on the ``angles_cumsum`` path).
    """
    init4 = (None, *Initial_States) if Initial_States is not None else None
    ref_out, (_r_angle, r_ssm, r_k, r_v) = mamba3_siso_fwd_ref(
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias,
        Angles_Cumsum,  # placeholder Angles (unused when angles_cumsum is given)
        D=D, Z=Z, Initial_States=init4,
        chunk_size=chunk_size, dtype=dtype, cu_seqlens=cu_seqlens,
        angles_cumsum=Angles_Cumsum,
    )
    return ref_out, (r_ssm, r_k, r_v)
