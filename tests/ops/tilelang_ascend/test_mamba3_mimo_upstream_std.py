"""NPU migration of the upstream CUDA suite ``tests/ops/tilelang/test_mamba3_mimo.py``.

The upstream file **is** the repo's own ``tests/ops/tilelang/test_mamba3_mimo.py``
and is executed as-is through :mod:`._upstream`, which applies only the four
mechanical NPU adaptations (device, CUDA RNG seeding, CUDA admission check, torch
decay chain).  Nothing here re-derives the standard: the case grid, the input
seeds, the dtypes, the fp32 references and -- the point of this file -- the
accuracy criterion all come from upstream:

    max|ours - ref| / max|ref| < REL_TOL,   REL_TOL = 0.10

i.e. ``upstream.assert_stable_rel``.  If upstream changes a tolerance, this suite
follows.

The test bodies are the upstream test bodies with ``mods.fwd.mamba_mimo_forward`` /
``mods.bwd.mamba_mimo_bwd_combined`` replaced by the A2 port's
``mamba_mimo_forward`` / ``mamba_mimo_backward``, which take the same arguments in
the same order.  ``bb_threads`` is a CUDA block-count knob and is dropped.
"""

from __future__ import annotations

import os

import pytest
import torch

from . import _upstream

try:
    P = _upstream.load()
except FileNotFoundError as exc:                       # wheel install, no tests/
    pytest.skip(str(exc), allow_module_level=True)

#: Upstream ``CASE_GRID``, with the CUDA-only ``bb_threads`` column dropped.  This
#: is the grid the upstream fwd / return_state / bwd tests are parametrized over.
CASES = [tuple(p.values[:4]) for p in P.CASE_GRID]
CASE_IDS = [p.id for p in P.CASE_GRID]

#: The port's argument order -- identical to the upstream call sites.
FWD_ORDER = ("q", "k", "v", "q_bias", "k_bias", "mimo_v", "mimo_o", "z", "D",
             "mimo_z", "angles", "dA_cs", "dA_cs_rev", "dt", "trap", "segsum")
BWD_ORDER = ("q", "k", "v", "q_bias", "k_bias", "mimo_v", "mimo_o", "z", "mimo_z",
             "angles", "dA_cs", "dA_cs_rev", "dt", "trap", "D", "segsum")

#: ``(port key, upstream key)`` in the upstream comparison order.  Upstream ``dA`` is
#: the decay-chain gradient, which the port calls ``ddA``.
BWD_MAP = (("dq", "dq"), ("dk", "dk"), ("dv", "dv"), ("ddA", "dA"),
           ("ddt", "ddt"), ("dtrap", "dtrap"), ("dq_bias", "dq_bias"),
           ("dk_bias", "dk_bias"), ("dmimo_v", "dmimo_v"),
           ("dmimo_z", "dmimo_z"), ("dmimo_o", "dmimo_o"),
           ("dangles", "dangles"), ("dD", "dD"), ("dz", "dz"))


def _build(n, p, r, chunk_size, seed):
    """Upstream ``build_inputs`` with upstream defaults (B4 S2048 H16, bf16, rot=4).

    ``A2_STD_S`` / ``A2_STD_B`` / ``A2_STD_H`` shrink the sequence geometry only --
    the case grid, the seeds, the dtypes and the criterion stay upstream.  They
    exist because the upstream sizes are a multi-hour soak on this port (one
    full-size bf16 forward plus its fp32 reference, and every shape recompiles its
    kernels), not because the verdict depends on them.
    """

    kw = {}
    for env, key in (("A2_STD_S", "s"), ("A2_STD_B", "b"), ("A2_STD_H", "h")):
        raw = os.environ.get(env)
        if raw:
            kw[key] = int(raw)
    return P.build_inputs(mods=None, n=n, p=p, r=r, chunk_size=chunk_size,
                          seed=seed, **kw)


def _cfg(n, p, r, chunk_size):
    return f"N={n}, P={p}, R={r}, chunk={chunk_size}"


def _assert(ours, ref, *, label, cfg):
    """Upstream ``assert_stable_rel`` with both operands on the host.

    Upstream calls it with both operands on the device (its reference runs there),
    so the helper assumes one device; ours is host fp32, so both sides are moved to
    the host.  ``max|err| / max|ref|`` is unchanged by the move, and ``.float()``
    keeps the widened bf16 output exact.
    """

    P.assert_stable_rel(ours.detach().to("cpu", torch.float32),
                        ref.detach().to("cpu", torch.float32),
                        label=label, cfg=cfg)


@pytest.mark.parametrize("n,p,r,chunk_size", CASES, ids=CASE_IDS)
def test_fused_chunk_linear_attn_fwd_relative_error_lt_10pct(
        n: int, p: int, r: int, chunk_size: int) -> None:
    """Upstream ``test_fused_chunk_linear_attn_fwd_relative_error_lt_10pct``."""

    from mamba_ssm.ops.tilelang_ascend.mamba3.mamba3_mimo_fwd import (
        mamba_mimo_forward,
    )

    inputs = _build(n, p, r, chunk_size, seed=1234 + n + p + r + chunk_size)
    rot = inputs["rotary_dim_divisor"]

    out_port, _, _ = mamba_mimo_forward(
        *[inputs[k] for k in FWD_ORDER], chunk_size=chunk_size,
        rotary_dim_divisor=rot, dtype=P.FIXED_DTYPE)
    torch.npu.synchronize()
    out_ref, _, _ = _upstream.ref_forward(inputs, chunk_size=chunk_size,
                                          rotary_dim_divisor=rot)

    _assert(out_port, out_ref, label="forward",
            cfg=_cfg(n, p, r, chunk_size))


@pytest.mark.parametrize("n,p,r,chunk_size", CASES, ids=CASE_IDS)
def test_fused_chunk_linear_attn_fwd_return_state_relative_error_lt_10pct(
        n: int, p: int, r: int, chunk_size: int) -> None:
    """Upstream ``..._fwd_return_state_relative_error_lt_10pct``."""

    from mamba_ssm.ops.tilelang_ascend.mamba3.mamba3_mimo_fwd import (
        mamba_mimo_forward,
    )

    inputs = _build(n, p, r, chunk_size, seed=3456 + n + p + r + chunk_size)
    rot = inputs["rotary_dim_divisor"]

    out_port, state_port, k_port = mamba_mimo_forward(
        *[inputs[k] for k in FWD_ORDER], return_state=True,
        chunk_size=chunk_size, rotary_dim_divisor=rot, dtype=P.FIXED_DTYPE)
    torch.npu.synchronize()
    out_ref, state_ref, k_ref = _upstream.ref_forward(
        inputs, chunk_size=chunk_size, rotary_dim_divisor=rot, return_state=True)

    cfg = _cfg(n, p, r, chunk_size)
    _assert(out_port, out_ref, label="forward_return_state_out", cfg=cfg)
    _assert(state_port, state_ref,
            label="forward_return_state_final_state", cfg=cfg)
    _assert(k_port, k_ref, label="forward_return_state_final_k", cfg=cfg)


@pytest.mark.parametrize("n,p,r,chunk_size", CASES, ids=CASE_IDS)
def test_mamba_mimo_bwd_relative_errors_lt_10pct(
        n: int, p: int, r: int, chunk_size: int) -> None:
    """Upstream ``test_mamba_mimo_bwd_combined_relative_errors_lt_10pct``.

    Upstream drives ``mamba_mimo_bwd_combined``, a fused convenience wrapper that is
    out of scope for this port; the comparison -- same 14 gradients, same criterion
    -- is therefore run against the port's separate backward.
    """

    from mamba_ssm.ops.tilelang_ascend.mamba3.mamba3_mimo_bwd import (
        mamba_mimo_backward,
    )

    inputs = _build(n, p, r, chunk_size, seed=5678 + n + p + r + chunk_size)
    rot = inputs["rotary_dim_divisor"]

    got = mamba_mimo_backward(
        inputs["dout"], *[inputs[k] for k in BWD_ORDER], chunk_size=chunk_size,
        rotary_dim_divisor=rot, dtype=P.FIXED_DTYPE)
    torch.npu.synchronize()
    ref = _upstream.ref_backward(inputs, inputs["dout"], chunk_size=chunk_size,
                                 rotary_dim_divisor=rot)

    # Upstream asserts the fused-norm weight gradient is absent when the epilogue
    # is off.
    assert got["dout_norm_weight"] is None

    cfg = _cfg(n, p, r, chunk_size)
    for port_key, ref_key in BWD_MAP:
        _assert(got[port_key], ref[ref_key], label=ref_key, cfg=cfg)
