"""End-to-end NPU test for the Mamba-3 SISO module (PLUT-34).

Unlike the single-operator suites, this exercises the **real call chain**

    Mamba3.forward -> mamba3_siso_combined -> angle_dt_fwd + mamba3_siso_fwd
                                            (forward)
    loss.backward() -> compute_dzdo / compute_dqkv / compute_dqktheta /
                       compute_ddt_dtrap_dinput_states / angle_dt_bwd
                                            (backward)

on an Ascend NPU, on a real ``Mamba3`` module (in_proj / B,C RMSNorm / out_proj
shell included), and compares it against an equivalent pure-PyTorch golden.

Golden strategy
---------------
The golden reuses the very same ``Mamba3`` module shell and only swaps the SISO
Triton kernel for the pure-PyTorch chunked reference ``mamba3_siso_fwd_ref``.
Both runs therefore share every non-SISO operator (linear projections,
RMSNormGated, out_proj, the ADT/DT/angle pre-processing), so any residual is
attributable to the SISO kernel chain that this migration touches.  The
reference wrapper mirrors the kernel wrapper's dtype contract (Q/K/V/Z -> bf16,
ADT/DT/Trap/Q_bias/K_bias/D/Angles -> fp32) so the comparison isolates the
kernel's arithmetic from a dtype re-quantization.

The reference (``_segsum`` / ``mamba3_siso_fwd_ref``) is a frozen, verbatim copy
of ``tests/ops/triton/_siso_ops_npu_common.py`` so this end-to-end deliverable
does not depend on that shared helper's concurrent edits.

Reporting
---------
Every compared tensor is reported against three standards, one line each:
    [tag] dtype=... max_abs=... max_rel=... p95_rel=... guide_tol=...   (guide)
    [tag] CANN matched_ratio=... max_abs_error=... limit=... verdict=...
    [tag] upstream p95_rel=... global_max_rel=... n_masked=...         (gate)

The **gate** follows the upstream CUDA suite (``tests/ops/triton/test_mamba3_siso.py``,
``test_mamba3_siso_combined_batched``): ``relative_error()`` = p95 of the elementwise
relative error over the elements with ``|ref| >= 1e-2``, at ``rtol = 1e-1``.  That
mask is what makes the criterion usable on gradients: a bf16 chain produces an
absolute error floor set by the dominant terms of each sum (measured ~1.2e-2 on
``dout_proj.weight``, |g| up to 211, and **bit-identical** elements in the large-|g|
buckets), so the small-|g| tail of a cancellation-heavy tensor can never clear a
per-element ``atol`` -- which is why upstream leaves those asserts commented out.

Two deviations from upstream, both deliberate and both stricter than "print only":

* the gate is enabled (upstream has it commented out), with upstream's own threshold;
* a tensor with fewer than ``UPSTREAM_MIN_SAMPLES`` masked elements is gated on the
  globally normalised max relative error instead (``max|ours-ref| / max|ref|``,
  the criterion of the upstream MIMO suite, same 0.10 threshold), because a p95
  over a handful of per-channel scalars degenerates to the max anyway.

CANN (``mixed_tolerance_standard.md``: matched_ratio >= 0.99 and
max_abs_error <= max_abs_error_limit) and the stricter guide tolerances are printed
for every tensor.  CANN is *asserted* on forward outputs -- where the per-element
criterion is achievable and measured green -- and report-only on gradients, for the
reason above.

Run (inside the NPU container, from the repo root):
    MAMBA3_SISO_DEVICE=npu MAMBA3_DISABLE_TILELANG=1 ASCEND_RT_VISIBLE_DEVICES=0 \\
        python -m pytest -v -s -p no:cacheprovider \\
        tests/ops/triton/test_mamba3_siso_e2e_npu.py
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import repeat

try:  # register the "npu" device
    import torch_npu  # noqa: F401

    _HAS_TORCH_NPU = True
except Exception:  # pragma: no cover
    _HAS_TORCH_NPU = False

import pytest

import mamba_ssm.modules.mamba3 as m3mod  # noqa: E402
from mamba_ssm.modules.mamba3 import Mamba3  # noqa: E402


DEVICE = os.environ.get("MAMBA3_SISO_DEVICE", "npu")
if DEVICE == "npu" and _HAS_TORCH_NPU:
    torch.npu.set_device(0)

_SKIP_REASON = None
if DEVICE == "npu" and not _HAS_TORCH_NPU:
    _SKIP_REASON = "MAMBA3_SISO_DEVICE=npu but torch_npu is not importable"
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


# ---------------------------------------------------------------------------
# Tolerance policy (guide line + CANN ecosystem standard)
# ---------------------------------------------------------------------------
GUIDE_TOL = {
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
    torch.bfloat16: dict(rtol=5e-3, atol=5e-3),
}
# CANN mixed_tolerance_standard.md: elementwise atol/rtol; aggregate
# matched_ratio >= 0.99 and max_abs_error <= max_abs_error_limit.
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

# Upstream CUDA suite criterion (tests/ops/triton/test_mamba3_siso.py):
#   RTOL = 1e-1 on relative_error() = p95 elementwise relative error over |ref| >= 1e-2.
# Below this many masked elements a p95 is within a couple of order statistics of the
# max, so the gate switches to the globally normalised max relative error (the
# criterion of the upstream MIMO suite, same threshold).
UPSTREAM_RTOL = 1e-1
UPSTREAM_MIN_SAMPLES = 100


def _f32(t):
    return t.detach().to(torch.float32).cpu()


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


def _32ulp_of_max_abs(t):
    x = t.detach().abs().max()
    if float(x) == 0.0:
        return 0.0
    nxt = torch.nextafter(x, torch.tensor(float("inf"), dtype=x.dtype))
    return float(nxt - x) * 32.0


def cann_metrics(cal, ref, dtype=None):
    dt = dtype if dtype is not None else getattr(cal, "dtype", torch.float32)
    tol = CANN_TOL.get(dt, dict(rtol=1e-3, atol=1e-3))
    c, r = _f32(cal), _f32(ref)
    n = r.numel()
    if n == 0:
        return 1.0, 0.0, CANN_MAX_ABS_ERROR_LIMIT.get(dt, 1e-1), tol["rtol"], tol["atol"], "PASS"
    diff = (c - r).abs()
    thr = tol["atol"] + tol["rtol"] * r.abs()
    matched_ratio = float((diff <= thr).sum().item()) / n
    max_abs = float(diff.max().item())
    limit = max(CANN_MAX_ABS_ERROR_LIMIT.get(dt, 1e-1), _32ulp_of_max_abs(_f32(ref)))
    verdict = "PASS" if (matched_ratio >= CANN_MATCHED_RATIO_MIN and max_abs <= limit) else "FAIL"
    return matched_ratio, max_abs, limit, tol["rtol"], tol["atol"], verdict


# ---------------------------------------------------------------------------
# Frozen pure-PyTorch golden (verbatim copy of _siso_ops_npu_common.py)
# ---------------------------------------------------------------------------

def _segsum(x: torch.Tensor) -> torch.Tensor:
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def mamba3_siso_fwd_ref(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
    D=None, Z=None, Initial_States=None, chunk_size=64,
    dtype=torch.float32, cu_seqlens=None, angles_cumsum=None, qkvd_dtype=None,
):
    """Reference Mamba-3 chunked forward pass (device-agnostic, fp32 golden)."""
    batch, total_seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert batch == 1

    if qkvd_dtype is None:
        Q = Q.to(dtype); K = K.to(dtype); V = V.to(dtype); Trap = Trap.to(dtype)
        Q_bias = Q_bias.to(dtype); K_bias = K_bias.to(dtype); Angles = Angles.to(dtype)
        if D is not None:
            D = D.to(dtype)
        if Z is not None:
            Z = Z.to(dtype)
    else:
        Q = Q.to(qkvd_dtype); K = K.to(qkvd_dtype); V = V.to(qkvd_dtype)
        Trap = Trap.to(torch.float32); Q_bias = Q_bias.to(torch.float32)
        K_bias = K_bias.to(torch.float32); Angles = Angles.to(torch.float32)
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
            Q_curr = Q[seq_idx]; K_curr = K[seq_idx]; V_curr = V[seq_idx]
            ADT_curr = ADT[seq_idx]; DT_curr = DT[seq_idx]; Trap_curr = Trap[seq_idx]
            Angles_curr = Angles[seq_idx]
            Z_curr = Z[seq_idx] if Z is not None else None

        Trap_curr = torch.sigmoid(Trap_curr)

        if angles_cumsum is not None:
            if is_varlen:
                Angles_Cumsum = angles_cumsum[0, start_idx:end_idx, :, :]
            else:
                Angles_Cumsum = angles_cumsum[seq_idx]
        else:
            Angles_scaled = Angles_curr.float() * DT_curr.transpose(0, 1).unsqueeze(-1)
            Angles_Cumsum = torch.cumsum(Angles_scaled, dim=0)
            if Initial_States is not None:
                Angles_Cumsum = Angles_Cumsum + Initial_Angle_State[seq_idx].unsqueeze(0)
            Angles_Cumsum = Angles_Cumsum - TWO_PI * torch.floor(Angles_Cumsum / TWO_PI)
        Final_Angle_States.append(Angles_Cumsum[-1])

        if Initial_States is not None:
            Initial_SSM_State_curr = Initial_SSM_State[seq_idx]
            Initial_K_State_curr = Initial_K_State[seq_idx]
            Initial_V_State_curr = Initial_V_State[seq_idx]
            scalar = DT_curr[:, 0] * (1 - Trap_curr[:, 0])
            acc_states = (Initial_SSM_State_curr
                          + Initial_V_State_curr[:, :, None] * Initial_K_State_curr[:, None, :]
                          * scalar[:, None, None])
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
            QK_causal = QK_causal.to(qkvd_dtype)
        out = torch.einsum("hts,shd->thd", QK_causal, V_used)

        if Initial_States is not None:
            da_cs = torch.cumsum(ADT_curr, dim=-1)
            exp_da_cs = torch.exp(da_cs)
            if qkvd_dtype is not None:
                out = out + torch.einsum("hDd,thd,ht->thD", acc_states.to(qkvd_dtype), Q_used, exp_da_cs)
            else:
                out = out + torch.einsum("hDd,thd,ht->thD", acc_states.to(Q_curr.dtype),
                                         Q_curr, exp_da_cs.to(Q_curr.dtype))

        if D is not None:
            out = out + D[None, :, None] * V_used

        out = out - V_used * QK_dot.unsqueeze(-1)

        if Z_curr is not None:
            if qkvd_dtype is not None:
                zg = Z_curr.float()
                out = out * (zg * torch.sigmoid(zg))
            else:
                out = out * Z_curr * torch.sigmoid(Z_curr)
        if qkvd_dtype is not None:
            out = out.to(qkvd_dtype)
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


# ---------------------------------------------------------------------------
# Golden swap + reporting
# ---------------------------------------------------------------------------

def _siso_combined_ref(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
    D=None, Z=None, Input_States=None, chunk_size=64,
    return_final_states=False, cu_seqlens=None,
):
    """Pure-PyTorch drop-in for the Triton wrapper (same dtype contract)."""
    Q = Q.to(torch.bfloat16)
    K = K.to(torch.bfloat16)
    V = V.to(torch.bfloat16)
    Trap = Trap.to(torch.float32)
    if Z is not None:
        Z = Z.to(torch.bfloat16)
    out, (fa, fs, fk, fv) = mamba3_siso_fwd_ref(
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
        D=D, Z=Z, Initial_States=Input_States, chunk_size=chunk_size,
        dtype=torch.float32, cu_seqlens=cu_seqlens,
    )
    if return_final_states:
        return out, fa, fs, fk, fv
    return out


@contextlib.contextmanager
def _ref_golden():
    orig = m3mod.mamba3_siso_combined
    m3mod.mamba3_siso_combined = _siso_combined_ref
    try:
        yield
    finally:
        m3mod.mamba3_siso_combined = orig


def _report(name, cal, ref, guide_fails, cann_fails, guide_tol=None):
    """Print guide + CANN lines; collect (never stop) guide and CANN misses."""
    c, r = _f32(cal), _f32(ref)
    assert c.shape == r.shape, f"[{name}] shape {tuple(c.shape)} != {tuple(r.shape)}"
    n = r.numel()
    max_abs = (c - r).abs().max().item() if n else 0.0
    max_rel = ((c - r).abs() / (r.abs() + 1e-6)).max().item() if n else 0.0
    p95 = p95_relative_error(cal, ref)
    tol = guide_tol or GUIDE_TOL.get(cal.dtype, dict(rtol=1e-5, atol=1e-5))
    print(f"[{name}] dtype={cal.dtype} max_abs={max_abs:.3e} "
          f"max_rel={max_rel:.3e} p95_rel={p95:.3e} guide_tol={tol}")
    m_ratio, m_abs, m_lim, crt, cat, verdict = cann_metrics(cal, ref)
    print(f"[{name}] CANN matched_ratio={m_ratio:.6f} (>= 0.99) "
          f"max_abs_error={m_abs:.6e} max_abs_error_limit={m_lim:.6e} "
          f"cann_tol=(rtol={crt:g},atol={cat:g}) verdict={verdict}")
    guide_ok = bool((((c - r).abs()) <= tol["atol"] + tol["rtol"] * r.abs()).all().item())
    if not guide_ok:
        guide_fails.append(f"{name}: max_abs={max_abs:.3e} guide_tol={tol}")
    if verdict != "PASS":
        cann_fails.append(
            f"{name}: matched_ratio={m_ratio:.6f} max_abs_error={m_abs:.6e} limit={m_lim:.6e}"
        )


def _global_max_rel(cal, ref, eps=1e-5):
    """Upstream MIMO-suite criterion: max|ours-ref| / max|ref| (globally normalised)."""
    c, r = _f32(cal), _f32(ref)
    if r.numel() == 0:
        return 0.0
    return float(((c - r).abs().max() / r.abs().max().clamp_min(eps)).item())


def _report_upstream(name, cal, ref, fails):
    """Print + gate on the upstream criterion; collect misses (caller asserts).

    Large tensors are gated on ``p95_relative_error`` (upstream ``relative_error``);
    tensors with fewer than ``UPSTREAM_MIN_SAMPLES`` masked elements -- the per-channel
    scalars such as ``dD`` / ``dt_bias`` -- on the globally normalised max instead.
    """
    c, r = _f32(cal), _f32(ref)
    assert c.shape == r.shape, f"[{name}] shape {tuple(c.shape)} != {tuple(r.shape)}"
    n_masked = int((r.abs() >= 1e-2).sum().item())
    p95 = p95_relative_error(cal, ref)
    gmax = _global_max_rel(cal, ref)
    if n_masked >= UPSTREAM_MIN_SAMPLES:
        gated, value = "p95_rel", p95
    else:
        gated, value = "global_max_rel", gmax
    ok = value < UPSTREAM_RTOL
    print(f"[{name}] upstream {gated}={value:.3e} (< {UPSTREAM_RTOL}) "
          f"p95_rel={p95:.3e} global_max_rel={gmax:.3e} n_masked={n_masked} "
          f"{'ok' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"{name}: {gated}={value:.3e} >= {UPSTREAM_RTOL}")


def _build_module(d_model=128, d_state=64, headdim=32, ngroups=2, expand=2,
                  chunk_size=64, dtype=torch.bfloat16):
    torch.manual_seed(0)
    m = Mamba3(
        d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
        ngroups=ngroups, rope_fraction=0.5, is_mimo=False, is_outproj_norm=False,
        chunk_size=chunk_size, dtype=dtype, device=DEVICE,
    )
    m.train()
    return m


def _peak_mem_mb():
    try:
        return torch.npu.max_memory_allocated() / 1024.0 / 1024.0
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_e2e_fwd_bwd_batched():
    """Real Mamba3.forward + backward on NPU vs the PyTorch-reference golden."""
    batch, seqlen = 2, 128
    mk = _build_module()
    mr = copy.deepcopy(mk)

    torch.manual_seed(1234)
    u = torch.randn(batch, seqlen, mk.d_model, dtype=torch.bfloat16, device=DEVICE)
    uk = u.clone().detach().requires_grad_(True)
    ur = u.clone().detach().requires_grad_(True)
    dO = torch.randn(batch, seqlen, mk.d_model, dtype=torch.bfloat16, device=DEVICE)

    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    out_k = mk(uk)
    (out_k * dO).sum().backward()
    torch.npu.synchronize()
    peak = _peak_mem_mb()

    with _ref_golden():
        out_r = mr(ur)
        (out_r * dO).sum().backward()

    guide_fails, cann_fails, upstream_fails = [], [], []
    _report("e2e.batched.out", out_k, out_r, guide_fails, cann_fails)
    _report_upstream("e2e.batched.out", out_k, out_r, upstream_fails)
    for (nk, pk), (nr, pr) in zip(mk.named_parameters(), mr.named_parameters()):
        assert nk == nr
        if pk.grad is None and pr.grad is None:
            continue
        assert pk.grad is not None and pr.grad is not None, f"{nk}: grad None on one side"
        _report(f"e2e.batched.d{nk}", pk.grad, pr.grad, guide_fails, cann_fails)
        _report_upstream(f"e2e.batched.d{nk}", pk.grad, pr.grad, upstream_fails)
    _report("e2e.batched.du", uk.grad, ur.grad, guide_fails, cann_fails)
    _report_upstream("e2e.batched.du", uk.grad, ur.grad, upstream_fails)

    print(f"[e2e.batched] peak_npu_mem={peak:.1f} MB")
    if guide_fails:
        print("[e2e.batched] GUIDE-TOL VIOLATIONS (report-only):\n  " + "\n  ".join(guide_fails))
    if cann_fails:
        print("[e2e.batched] CANN FAILURES (report-only on gradients, see module docstring):\n  "
              + "\n  ".join(cann_fails))
    assert not upstream_fails, (
        f"upstream criterion (rtol={UPSTREAM_RTOL}): " + " ; ".join(upstream_fails)
    )


def test_e2e_fwd_large_shape():
    """Real forward on a larger sequence (UB / memory stress), golden-checked."""
    batch, seqlen = 1, 512
    mk = _build_module()
    mr = copy.deepcopy(mk)

    torch.manual_seed(7)
    u = torch.randn(batch, seqlen, mk.d_model, dtype=torch.bfloat16, device=DEVICE)

    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    with torch.no_grad():
        out_k = mk(u)
    torch.npu.synchronize()
    peak = _peak_mem_mb()

    with _ref_golden(), torch.no_grad():
        out_r = mr(u)

    guide_fails, cann_fails, upstream_fails = [], [], []
    _report("e2e.large.out", out_k, out_r, guide_fails, cann_fails)
    _report_upstream("e2e.large.out", out_k, out_r, upstream_fails)
    print(f"[e2e.large] peak_npu_mem={peak:.1f} MB")
    if guide_fails:
        print("[e2e.large] GUIDE-TOL VIOLATIONS (report-only):\n  " + "\n  ".join(guide_fails))
    if cann_fails:
        print("[e2e.large] CANN FAILURES:\n  " + "\n  ".join(cann_fails))
    assert not cann_fails, "CANN FAILURES: " + " ; ".join(cann_fails)
    assert not upstream_fails, (
        f"upstream criterion (rtol={UPSTREAM_RTOL}): " + " ; ".join(upstream_fails)
    )


def test_e2e_fwd_varlen():
    """Real forward with cu_seqlens (varlen), golden-checked."""
    lengths = [40, 64, 24]
    total = sum(lengths)
    mk = _build_module()
    mr = copy.deepcopy(mk)

    cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist()),
        dtype=torch.int32, device=DEVICE,
    )
    torch.manual_seed(9)
    u = torch.randn(1, total, mk.d_model, dtype=torch.bfloat16, device=DEVICE)

    with torch.no_grad():
        out_k = mk(u, cu_seqlens=cu)
    with _ref_golden(), torch.no_grad():
        out_r = mr(u, cu_seqlens=cu)

    guide_fails, cann_fails, upstream_fails = [], [], []
    _report("e2e.varlen.out", out_k, out_r, guide_fails, cann_fails)
    _report_upstream("e2e.varlen.out", out_k, out_r, upstream_fails)
    if guide_fails:
        print("[e2e.varlen] GUIDE-TOL VIOLATIONS (report-only):\n  " + "\n  ".join(guide_fails))
    if cann_fails:
        print("[e2e.varlen] CANN FAILURES:\n  " + "\n  ".join(cann_fails))
    assert not cann_fails, "CANN FAILURES: " + " ; ".join(cann_fails)
    assert not upstream_fails, (
        f"upstream criterion (rtol={UPSTREAM_RTOL}): " + " ; ".join(upstream_fails)
    )
