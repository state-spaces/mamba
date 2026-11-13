"""Module-level Mamba-3 SISO decode test -- cutlass step replaced by Triton ``mamba3_siso_step``.

Why this file exists
--------------------
``Mamba3.step`` (``mamba_ssm/modules/mamba3.py:343``) is the standard decode entry
(the only thing ``forward`` dispatches to when ``inference_params.seqlen_offset > 0``,
``mamba3.py:200-202``), but it calls ``mamba3_step_fn`` -- a cutlass/CuteDSL kernel
that has not been ported to Ascend.  The import guard leaves
``mamba3_step_fn = None`` on NPU and the ``assert`` at ``mamba3.py:365`` fires, so
the module-level decode path is unusable on this stack even though the Triton
``mamba3_siso_step`` op already implements the same recurrence (see
``scripts/infer_siso_step.py``, which bypasses cutlass for exactly this reason).

This test wires the *module* to that Triton op and checks the only thing that can
be checked without a golden: **decode must reproduce prefill**.  Same model, same
input, both starting from zero state, teacher forcing (the decode step is fed the
token that prefill already saw, so the two paths must agree).

    prefill : model(u)                        -> teacher-forcing baseline (b, L, d_model)
              model(u[:, :P]) + cache         -> the four states decode starts from
    decode  : per token, mamba3_siso_step     -> out_decode          (b, L-P, d_model)
              then out_proj

The comparison is therefore *decode segment vs the same segment of the full
sequence forward* on the same teacher-forced input -- exactly the "infer vs full
· decode 段" row of ``scripts/infer_siso_step.py``.

This is an **NPU-only supplement**: upstream has no counterpart (upstream decode is
``Mamba3.step`` -> cutlass).  It therefore cannot be compared against an upstream
test, and it is *not* a migration-accuracy statement -- both sides of the
comparison run on the same device.

How the module quantities map onto the Triton op (``mamba3.py:310-331`` vs the op
signature).  Getting any of these wrong does not raise -- it silently produces
wrong numbers, which is why two negative self-checks below pin them down:

    module (_preprocess / forward)              -> mamba3_siso_step argument
    ---------------------------------------------------------------------------
    C after C_norm, expand(-1,-1,nheads,-1)     -> Q      (b, nheads, d_state)
    B after B_norm, expand(-1,-1,nheads,-1)     -> K      (b, nheads, d_state)
    x (b, l, nheads*headdim) -> rearrange        -> V      (b, nheads, headdim)
    _A * DT  (_A = -heavy_tail_activation)       -> ADT    (b, nheads)
    softplus(dd_dt + dt_bias)                    -> DT     (b, nheads)
    trap_proj  **before** sigmoid                -> Trap   (b, nheads)
    angles (per-token, pre-cumsum, fp32)         -> Angles (b, nheads, num_rope_angles)
    self.C_bias.squeeze(1) / self.B_bias.squeeze(1) -> Q_bias / K_bias (nheads, d_state)
    self.D                                       -> D      (nheads,)
    z                                            -> Z      (b, nheads, headdim)
    (angle_state, ssm_state, k_state, v_state)   -> Input_States

Three traps this file guards against (all silent):
  1. ``Trap`` must be the pre-sigmoid projection.  ``Mamba3._preprocess`` applies
     ``torch.sigmoid`` for the *cutlass* kernel; the Triton step applies its own
     ``sigmoid_approx``, so feeding the post-sigmoid value sigmoids twice.
  2. ``B``/``C`` must go through ``B_norm``/``C_norm`` **and** be expanded to
     ``nheads`` (``_preprocess`` does both).
  3. ``Angles`` are the raw per-token angles -- the op maintains ``angle_state``
     and rotates internally, so the module's ``apply_rotary_qk_inference_fwd``
     step must be skipped (otherwise the rotation happens twice).

Scope: ``is_outproj_norm=False`` and ``is_mimo=False`` only.  With
``is_outproj_norm=True`` the module also needs ``_postprocess`` (RMSNorm + mimo_o)
after the step, which the Triton op does not do -- that variant is out of scope
here, as it is in ``scripts/infer_siso_step.py`` (see its "Limitations").

dtype contract (why this test only gates on bf16)
-------------------------------------------------
``mamba3_siso_combined`` is **not an fp32 op**: its wrapper unconditionally casts
the kernel inputs to bf16 before launching --

    mamba3_siso_combined.py:394-396   Q/K/V = Q/K/V.to(torch.bfloat16)
    mamba3_siso_combined.py:403       Z     = Z.to(torch.bfloat16)
    (Trap/Angles stay fp32 on the NPU side; upstream
     mamba-main/.../mamba3_siso_combined.py:393-397 casts those too)

-- while ``Out`` is written back as ``V.dtype``.  So feeding fp32 in yields
*bf16 precision in an fp32-looking tensor*, and the upstream file carries the same
unconditional cast (``:393-395``), i.e. this is the op's pre-existing dtype
contract, not something the migration introduced.  ``mamba3_siso_step`` does *not*
cast: fp32 in -> fp32 compute (an fp32 step-vs-reference probe measured 1.9e-7
against 1.2e-3 for the same input through ``combined``).

Consequence for this file: on the fp32 branch, prefill (``model(u)`` ->
``combined``) is bf16 while decode (``mamba3_siso_step``) is fp32, so the two
sides are **not the same precision** and "fp32 must be more accurate" has no
meaning -- measured on this stack, fp32 p95 was *worse* (2.774e-2 vs bf16
2.055e-2) while ``max_abs`` was *better* (5.098e-3 vs 7.812e-3), exactly the bf16
quantisation step appearing on one side only.  The fp32 cases that used to assert
this were therefore removed; their numbers are recorded as an observation in the
PLUT-83 thread, not as a criterion.  Only the **bf16** branch is a valid gate, and
even then it compares two bf16 paths (``combined`` bf16 in / bf16 out vs
``step`` bf16), which is the intended "decode reproduces prefill" question.

Metric and gates
----------------
``p95`` is the same statistic upstream's ``relative_error`` uses: the 0.95
quantile of ``|out - ref| / (|ref| + 1e-6)`` over the elements with
``|ref| >= 1e-2`` (``_p95_relative_error`` below; it pads with ``+inf`` instead of
boolean-indexing, which is equivalent and cheaper on Ascend).  ``max_abs`` is
``max|out - ref|`` and is reported, not gated.

  * bf16 gate ``5e-2``: same metric and same number as upstream
    ``test_mamba3_siso_step``'s ``rtol``.  Measured here on NPU: ``2.055e-2``
    (``max_abs=7.812e-3``), which also matches the CPU wiring self-check
    (``2.083e-2``) almost exactly -- so the chunked-prefill-vs-recurrent-decode
    ordering behaves the same on both.  The gate keeps ~2.4x of headroom over the
    measurement.

  * The two negative self-checks gate on "clearly worse than baseline": NPU
    measured 62.5x (post-sigmoid ``Trap``) and 71.2x (skipped ``B_norm``/
    ``C_norm``) over the same bf16 baseline, so a wiring mistake cannot hide in the
    5e-2 headroom.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from _siso_ops_npu_common import DEVICE, MODULE_SKIP

import mamba_ssm.modules.mamba3 as m3mod
from mamba_ssm.ops.triton.mamba3 import mamba3_siso_step as step_mod

pytestmark = MODULE_SKIP


# ---------------------------------------------------------------------------
# Configuration (small on purpose: one compile per dtype, seconds per case)
# ---------------------------------------------------------------------------
BATCH = 2
SEQ_LEN = 64
PREFILL = 16
D_MODEL = 64
D_STATE = 16
HEADDIM = 16
EXPAND = 2
NGROUPS = 1
CHUNK_SIZE = 64
ROPE_FRACTION = 0.5
SEED = 0

REF_MAG_MASK = 1e-2
QUANTILE = 0.95

# Only the bf16 branch is a valid criterion here -- see "dtype contract" above:
# ``mamba3_siso_combined`` (prefill) is bf16 unconditionally, so an fp32 decode
# would be compared against a bf16 prefill.
BF16_P95_GATE = 5e-2

# A wiring mistake must show up as a *large* error, not a marginal one.
NEGATIVE_FACTOR = 5.0


# ---------------------------------------------------------------------------
# Metric: upstream-style p95 relative error, without boolean indexing
# ---------------------------------------------------------------------------
def _p95_relative_error(cal, ref):
    """p95 of |cal-ref|/(|ref|+eps) over |ref| >= REF_MAG_MASK; returns (p95, max_abs).

    Same statistic as the upstream SISO tests' ``relative_error``; the filter is
    implemented as ``where(mask, ratio, +inf)`` + kthvalue on the flattened tensor
    (a ``+inf`` never wins a k-th-smallest), so no boolean index / ``NonZero`` is
    materialised -- on Ascend that op dies at 6e8 elements (PLUT-83).
    """
    cal = cal.detach().to(torch.float32)
    ref = ref.detach().to(torch.float32)
    abs_ref = ref.abs()
    abs_diff = (cal - ref).abs()

    mask = abs_ref >= REF_MAG_MASK
    count = int(mask.sum().item())
    max_abs = abs_diff.max().item()
    if count == 0:
        return 0.0, max_abs
    filled = torch.where(mask, abs_diff / (abs_ref + 1e-6), float("inf")).flatten()
    k = max(1, min(count, int(math.ceil(QUANTILE * count))))
    return filled.kthvalue(k).values.item(), max_abs


# ---------------------------------------------------------------------------
# Module-side harness (mirrors scripts/infer_siso_step.py:479-551)
# ---------------------------------------------------------------------------
class _InferenceCache:
    """Minimal ``inference_params`` holder for ``Mamba3.forward`` prefill."""

    def __init__(self):
        self.seqlen_offset = 0
        self.key_value_memory_dict = {}


def _build_model(dtype):
    torch.random.manual_seed(SEED)
    model = m3mod.Mamba3(
        d_model=D_MODEL,
        d_state=D_STATE,
        expand=EXPAND,
        headdim=HEADDIM,
        ngroups=NGROUPS,
        rope_fraction=ROPE_FRACTION,
        is_outproj_norm=False,
        is_mimo=False,
        chunk_size=CHUNK_SIZE,
        dtype=dtype,
        device=DEVICE,
        layer_idx=0,
    )
    model.eval()
    return model


def _prefill_states(model, u_pre):
    """Run ``model(u_pre)`` through the cache; return the four states it ended on.

    ``Mamba3.forward`` copies its final states into ``inference_params`` only when
    a cache is passed (``mamba3.py:295-300``), which is exactly the hand-off the
    real decode loop would need.
    """
    cache = _InferenceCache()
    with torch.no_grad():
        out = model(u_pre, inference_params=cache)
    a, s, k, v = cache.key_value_memory_dict[0]
    # ``k`` carries an extra mimo-rank dim -> the step op's (b, nheads, d_state)
    # contract (scripts/infer_siso_step.py:505-507).
    return out, (a, s, k.squeeze(1).contiguous(), v)


def _decode_step(model, u_t, states, step_fn, trap_post_sigmoid=False, skip_bc_norm=False):
    """One token through ``mamba3_siso_step`` (cutlass ``Mamba3.step`` bypassed).

    Mirrors ``mamba3.py:367-469`` up to the kernel call, then ``out_proj``.
    The two keyword flags are the deliberate wiring mistakes used by the negative
    self-checks; the default path is the one the assertions gate on.
    """
    zxBCdtddA = model.in_proj(u_t)
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        zxBCdtddA,
        [
            model.d_inner,
            model.d_inner,
            model.d_state * model.num_bc_heads * model.mimo_rank,
            model.d_state * model.num_bc_heads * model.mimo_rank,
            model.nheads,
            model.nheads,
            model.nheads,
            model.num_rope_angles,
        ],
        dim=-1,
    )

    _A = -m3mod.heavy_tail_activation(dd_A.to(torch.float32))
    _A = torch.clamp(_A, max=-model.A_floor)
    DT = F.softplus(dd_dt + model.dt_bias)
    ADT = _A * DT

    B = rearrange(B, "b (r g n) -> b r g n", r=model.mimo_rank, g=model.num_bc_heads)
    C = rearrange(C, "b (r g n) -> b r g n", r=model.mimo_rank, g=model.num_bc_heads)
    if not skip_bc_norm:  # negative check: the module always normalises B/C
        B = model.B_norm(B)
        C = model.C_norm(C)
    B = B.expand(-1, -1, model.nheads, -1)  # (b, r, nheads, d_state)
    C = C.expand(-1, -1, model.nheads, -1)

    x = rearrange(x, "b (h p) -> b h p", p=model.headdim)
    z = rearrange(z, "b (h p) -> b h p", p=model.headdim)
    angles = angles.unsqueeze(-2).expand(-1, model.nheads, -1).to(torch.float32)

    if trap_post_sigmoid:  # negative check: the op sigmoids Trap itself
        trap = torch.sigmoid(trap)

    out, new_states = step_fn(
        C.squeeze(1),
        B.squeeze(1),
        x,
        ADT,
        DT,
        trap,
        model.C_bias.squeeze(1),
        model.B_bias.squeeze(1),
        angles,
        model.D,
        z,
        states,
    )

    out = rearrange(out, "b h p -> b (h p)")
    out = model.out_proj(out.to(x.dtype))
    return out, new_states


def _run_decode(model, u, states, step_fn, first_token=PREFILL, **broken):
    """Decode tokens ``first_token..L-1``; returns (out_decode, final states)."""
    outs = []
    for t in range(first_token, u.shape[1]):
        out, states = _decode_step(model, u[:, t], states, step_fn, **broken)
        outs.append(out)
    return torch.stack(outs, dim=1), states


def _measure(model, u, step_fn, **broken):
    """Decode segment vs the same segment of the full-sequence forward.

    Returns ``(p95, max_abs)``.  ``model(u)`` is the teacher-forcing baseline (the
    "prefill" the decode has to reproduce); the cached prefill on the first
    ``PREFILL`` tokens only supplies the states decode starts from
    (``scripts/infer_siso_step.py:560-576``).
    """
    with torch.no_grad():
        out_prefill = model(u)
        _, states = _prefill_states(model, u[:, :PREFILL])
        out_dec, _ = _run_decode(model, u, states, step_fn, **broken)
    return _p95_relative_error(out_dec, out_prefill[:, PREFILL:])


# ---------------------------------------------------------------------------
# Main assertion: decode (bf16) == prefill (bf16)
# ---------------------------------------------------------------------------
def test_decode_matches_prefill_bf16():
    """The one criterion in this file: bf16 decode reproduces the bf16 prefill."""
    torch.random.manual_seed(SEED)
    model = _build_model(torch.bfloat16)
    u = torch.randn(BATCH, SEQ_LEN, D_MODEL, dtype=torch.bfloat16, device=DEVICE)
    p95, max_abs = _measure(model, u, step_mod.mamba3_siso_step)
    print(f"[decode-vs-prefill bf16] p95_rel={p95:.3e} max_abs={max_abs:.3e} gate={BF16_P95_GATE:.1e}")
    assert p95 <= BF16_P95_GATE, f"bf16 decode-vs-prefill p95 {p95:.3e} > {BF16_P95_GATE:.1e}"


# ---------------------------------------------------------------------------
# Negative self-checks: the two silent wiring traps must be detectable
# ---------------------------------------------------------------------------
def test_negative_post_sigmoid_trap_is_detected():
    """Feeding the post-sigmoid Trap (the ``_preprocess`` value) must break the match."""
    torch.random.manual_seed(SEED)
    model = _build_model(torch.bfloat16)
    u = torch.randn(BATCH, SEQ_LEN, D_MODEL, dtype=torch.bfloat16, device=DEVICE)

    p95_ok, _ = _measure(model, u, step_mod.mamba3_siso_step)
    p95_bad, max_abs_bad = _measure(model, u, step_mod.mamba3_siso_step, trap_post_sigmoid=True)

    print(
        f"[negative: post-sigmoid Trap] p95_ok={p95_ok:.3e} p95_broken={p95_bad:.3e} "
        f"max_abs_broken={max_abs_bad:.3e} factor={p95_bad / max(p95_ok, 1e-12):.1f}x"
    )
    assert p95_bad > NEGATIVE_FACTOR * p95_ok, (
        f"post-sigmoid Trap gave p95 {p95_bad:.3e} vs baseline {p95_ok:.3e} -- "
        f"the double-sigmoid trap is not being detected"
    )
    assert p95_bad > BF16_P95_GATE


def test_negative_skipping_bc_norm_is_detected():
    """Skipping B_norm/C_norm (the ``_preprocess`` step) must break the match."""
    torch.random.manual_seed(SEED)
    model = _build_model(torch.bfloat16)
    u = torch.randn(BATCH, SEQ_LEN, D_MODEL, dtype=torch.bfloat16, device=DEVICE)

    p95_ok, _ = _measure(model, u, step_mod.mamba3_siso_step)
    p95_bad, max_abs_bad = _measure(model, u, step_mod.mamba3_siso_step, skip_bc_norm=True)

    print(
        f"[negative: no B_norm/C_norm] p95_ok={p95_ok:.3e} p95_broken={p95_bad:.3e} "
        f"max_abs_broken={max_abs_bad:.3e} factor={p95_bad / max(p95_ok, 1e-12):.1f}x"
    )
    assert p95_bad > NEGATIVE_FACTOR * p95_ok, (
        f"missing B_norm/C_norm gave p95 {p95_bad:.3e} vs baseline {p95_ok:.3e} -- "
        f"the norm trap is not being detected"
    )
    assert p95_bad > BF16_P95_GATE
