"""Compare every Helion kernel with its Triton reference inside a real pass.

One ``MambaMixerMin`` forward+backward runs with the pipeline on Triton, and each
dispatched kernel is wrapped so that both implementations are invoked on the
*production* arguments and their outputs compared. The wrapper returns the Triton
result, so every later stage -- and therefore every other kernel's inputs -- stays
exact production data and a difference is attributable to the kernel that
produced it.

The arguments are not rebuilt by hand on purpose. They are views into shared
buffers (``x``/``B``/``C``/``dx`` slice one packed ``xBC``, ``z`` slices
``zxbcdt``), several are outputs of preceding stages recomputed in a specific
order (``state_passing_bwd`` consumes the backward's *recomputed* forward states,
not ``chunk_state_fwd``'s output), and ``dout`` has already been through the
z-gating backward. Hand-building all of that is how you end up testing a
specialization, or a magnitude regime, that production never sees.

``MAMBA_HELION_CONFIG_DIR`` selects the config set. With it unset the tests fall back
to the checked-in example set so they run with no setup; autotuning stays disabled
either way, so a config that is missing -- or that does not compile on this GPU --
fails the run with a message pointing at ``autotune_mamba_mixer``.

One process tests one variant -- unpacked by default, packed (a ragged
``seq_idx``, exercising every kernel's ``HAS_SEQ_IDX`` branch) with
``MAMBA_HELION_TEST_PACKED=1`` -- because each needs its own config set::

    pytest tests/ops/helion/test_kernels.py                       # checked-in configs
    MAMBA_HELION_CONFIG_DIR=<unpacked configs> pytest tests/ops/helion/test_kernels.py
    MAMBA_HELION_TEST_PACKED=1 MAMBA_HELION_CONFIG_DIR=<packed configs> \
        pytest tests/ops/helion/test_kernels.py

A config carries one ``indexing`` entry per load/store, and the packed path adds
the ``seq_idx`` loads, so an unpacked config does not merely run slower there --
it fails to compile. Generate the packed set with
``autotune_mamba_mixer --sequence-packing`` into a separate directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

helion = pytest.importorskip("helion")
pytest.importorskip("causal_conv1d")

from mamba_ssm.ops.helion.dispatch import (
    get_helion_layer_norm_bwd,
    get_helion_ssd_kernels,
)
from mamba_ssm.ops.helion.mamba_mixer_min import MambaMixerMin
from mamba_ssm.ops.triton import layernorm_gated, ssd_combined

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Helion kernel tests require CUDA"
)


# --- what this process tests -------------------------------------------------

# The variant cannot be a parametrization: a config autotuned unpacked has one
# `indexing` entry per load/store and the packed path adds the seq_idx loads, so it does
# not even compile there -- and Helion pins a kernel's config on its first call, so one
# process cannot use both config sets.
_PACKED = os.environ.get("MAMBA_HELION_TEST_PACKED") == "1"

_SSD_KERNELS = (
    "chunk_state_fwd",
    "state_passing_fwd",
    "chunk_scan_fwd",
    "chunk_scan_bwd_dstates",
    "state_passing_bwd",
    "chunk_scan_chunk_state_bwd_dx",
    "chunk_state_bwd_db",
    "chunk_scan_bwd_dC",
    "chunk_scan_bwd_ddAcs_stable",
)
_LAYER_NORM_BWD = "layer_norm_bwd"


# --- the model the kernels run inside ----------------------------------------

# benchmarks/helion/autotune_mamba_mixer.py defaults (Nemotron3 Nano, seq 8k, bf16).
_D_MODEL = 2688
_BATCH = 1
_SEQLEN = 8192
_NHEADS = 64
_HEADDIM = 64
_DSTATE = 128
_NGROUPS = 8
_CHUNK = 128
_DTYPE = torch.bfloat16

# Ragged sub-sequence lengths summing to _SEQLEN, none of them a multiple of _CHUNK, so
# every boundary lands mid-chunk -- that is what exercises the kernels' HAS_SEQ_IDX index
# arithmetic rather than just switching it on.
_PACKED_SEQLENS = (1531, 2048, 129, 4484)


def _build_seq_idx() -> torch.Tensor:
    """Return a (1, _SEQLEN) int32 seq_idx, as Megatron builds for the THD path."""
    assert sum(_PACKED_SEQLENS) == _SEQLEN
    idx = torch.arange(len(_PACKED_SEQLENS), device="cuda")
    return (
        torch.repeat_interleave(idx, torch.tensor(_PACKED_SEQLENS, device="cuda"))
        .to(torch.int32)
        .unsqueeze(0)
    )


# --- tolerances --------------------------------------------------------------

# Both tolerances are one bf16 ulp. Every kernel here multiplies bf16 operands, so that is
# the granularity at which two correct implementations may legitimately disagree: when
# their fp32 accumulators straddle a rounding boundary, the bf16 results differ by exactly
# 1 ulp with neither being wrong.
_ULP = 2.0**-8
_RTOL = _ULP


def _atol_for(expected: torch.Tensor) -> float:
    """One ulp of the tensor's *own* largest magnitude, rather than a constant.

    These outputs span five orders of magnitude in typical element size -- 3e-3 for the
    chunk states up to 4e2 for dD -- so any fixed atol is either impossibly tight for the
    large tensors or vacuous for the small ones: at the atol=1e-1 this started with, a
    deliberate 2% error injected into chunk_scan_fwd went undetected. Scaling by the max
    rather than the mean is deliberate too, since the differences land on the large
    elements. The measured worst case is 1.3e-3 of the max (layer_norm_bwd's dx), leaving
    3x margin, and together the two tolerances detect a systematic error above
    _RTOL + _ULP = 0.78% -- close to the floor of what a bf16-vs-bf16 comparison can do.
    """
    return _ULP * expected.abs().max().item()


def _assert_close(name: str, actual: Any, expected: Any) -> None:
    if actual is None or expected is None:
        assert actual is expected, f"{name}: one implementation returned None"
        return
    if isinstance(actual, tuple) or isinstance(expected, tuple):
        assert isinstance(actual, tuple) and isinstance(expected, tuple)
        assert len(actual) == len(expected), f"{name}: output count differs"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_close(f"{name}[{index}]", actual_item, expected_item)
        return
    try:
        torch.testing.assert_close(
            actual, expected, rtol=_RTOL, atol=_atol_for(expected)
        )
    except AssertionError as error:
        raise AssertionError(
            f"{name}: Helion output differs from Triton\n{error}"
        ) from error
    else:
        diff = (actual - expected).abs().float()
        print(
            f"{name}: diff abs mean={diff.mean().item():.3e} max={diff.max().item():.3e} | "
            f"actual abs mean={actual.float().abs().mean().item():.3e} "
            f"max={actual.float().abs().max().item():.3e} | "
            f"expected abs mean={expected.float().abs().mean().item():.3e} "
            f"max={expected.float().abs().max().item():.3e}"
        )


# --- shadowing the dispatched kernels ----------------------------------------

# Arguments a kernel writes through in place, and the index of the corresponding return
# value. Triton is given the caller's buffer, as production does, so later stages see the
# reference result; Helion gets a private one, and the returned tensor must BE the buffer
# it was handed -- dx is a view into the dxBC that causal_conv1d_bwd reads afterwards, so
# writing through it is the contract, not returning the right values.
_IN_PLACE_OUTPUTS = {
    "chunk_scan_chunk_state_bwd_dx": {"dx": 0},
}


class _ShadowKernel:
    """Run both implementations on one kernel's production arguments.

    Inputs are passed through untouched rather than copied: they are views into
    larger buffers and Helion compiles per stride, so a copy would test a
    different specialization, and none of these kernels writes to an input except
    through the declared in-place arguments. Returns the Triton result so the
    surrounding pipeline stays on the reference path.

    Comparison failures are recorded instead of raised -- every kernel is
    shadowed in a single pass, and one kernel's mismatch must not stop the others
    from being checked.
    """

    def __init__(self, name, triton_fn, helion_fn, in_place=None):
        self.name = name
        self.triton_fn = triton_fn
        self.helion_fn = helion_fn
        self.in_place = in_place or {}
        self.calls = 0
        self.errors: list[AssertionError] = []

    def __call__(self, *args, **kwargs):
        expected = self.triton_fn(*args, **kwargs)

        # NaN-filled so a Helion kernel that leaves part of the buffer unwritten
        # shows up as a mismatch rather than reading whatever was there.
        scratch = {
            param: torch.full_like(kwargs[param], float("nan"))
            for param in self.in_place
        }
        actual = self.helion_fn(*args, **{**kwargs, **scratch})

        label = f"{self.name}#{self.calls}"
        try:
            _assert_close(label, actual, expected)
            for param, index in self.in_place.items():
                assert expected[index] is kwargs[param], (
                    f"{label}: Triton did not write the caller's {param}"
                )
                assert actual[index] is scratch[param], (
                    f"{label}: Helion allocated its own {param} instead of "
                    "writing the caller's buffer"
                )
        except AssertionError as error:
            self.errors.append(error)

        self.calls += 1
        return expected


def _run_shadowed(packed: bool, monkeypatch) -> dict[str, _ShadowKernel]:
    """Drive one MambaMixerMin fwd+bwd with every dispatched kernel shadowed."""
    triton_kernels = ssd_combined._TRITON_SSD_KERNELS
    shadows = {
        name: _ShadowKernel(
            name,
            getattr(triton_kernels, name),
            getattr(get_helion_ssd_kernels(), name),
            in_place=_IN_PLACE_OUTPUTS.get(name, {}),
        )
        for name in vars(triton_kernels)
    }
    monkeypatch.setattr(
        ssd_combined, "_ssd_kernel_impls", lambda: SimpleNamespace(**shadows)
    )

    shadows[_LAYER_NORM_BWD] = _ShadowKernel(
        _LAYER_NORM_BWD,
        layernorm_gated._layer_norm_bwd,
        get_helion_layer_norm_bwd(),
    )
    monkeypatch.setattr(layernorm_gated, "use_helion", lambda: True)
    monkeypatch.setattr(
        layernorm_gated, "get_helion_layer_norm_bwd", lambda: shadows[_LAYER_NORM_BWD]
    )

    torch.manual_seed(0)
    mixer = MambaMixerMin(
        d_model=_D_MODEL,
        nheads=_NHEADS,
        headdim=_HEADDIM,
        d_state=_DSTATE,
        ngroups=_NGROUPS,
        chunk_size=_CHUNK,
        device="cuda",
        dtype=_DTYPE,
    )
    mixer.train()
    hidden_states = torch.randn(
        _SEQLEN, _BATCH, _D_MODEL, device="cuda", dtype=_DTYPE, requires_grad=True
    )
    out = mixer(hidden_states, seq_idx=_build_seq_idx() if packed else None)
    (out * torch.randn_like(out)).sum().backward()
    torch.cuda.synchronize()
    return shadows


# --- the config set the kernels compile with ---------------------------------

# Used when MAMBA_HELION_CONFIG_DIR is unset, so the tests run with no setup. A config is
# a set of block sizes and indexing choices rather than anything GPU-specific, so this set
# usually compiles and passes off its own environment too. Update if it is renamed.
_EXAMPLE_CONFIG_DIR = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "helion"
    / "configs"
    / "b200-helion1.4-triton3.6-nemotron3-nano-bf16-seq8k"
)

_AUTOTUNE_HINT = (
    "Autotune a set of your own and point MAMBA_HELION_CONFIG_DIR at it:\n"
    "    MAMBA_HELION_CONFIG_DIR=<dir> python "
    "benchmarks/helion/autotune_mamba_mixer.py{packed}"
)


def _resolve_config_dir(monkeypatch):
    """Point MAMBA_HELION_CONFIG_DIR at a config set, and forbid autotuning.

    Must run before the kernels are first called, so it lives with the module-scoped
    fixture that drives the pass rather than in an autouse one: pytest instantiates
    higher-scoped fixtures first, so a function-scoped fixture would patch the
    environment only after the configs had already been resolved.
    """
    if not os.environ.get("MAMBA_HELION_CONFIG_DIR"):
        # The packed path needs its own set, and none is checked in.
        assert not _PACKED, (
            "MAMBA_HELION_TEST_PACKED=1 needs a packed config set of its own; there is "
            "no checked-in example for it.\n"
            + _AUTOTUNE_HINT.format(packed=" --sequence-packing --batch-size 1")
        )
        assert _EXAMPLE_CONFIG_DIR.is_dir(), (
            f"no config dir given and the checked-in example set is missing at "
            f"{_EXAMPLE_CONFIG_DIR} (it lives in the repo, not in the installed "
            "package, so run from a source checkout).\n"
            + _AUTOTUNE_HINT.format(packed="")
        )
        monkeypatch.setenv("MAMBA_HELION_CONFIG_DIR", str(_EXAMPLE_CONFIG_DIR))
        print(f"\nMAMBA_HELION_CONFIG_DIR unset; using {_EXAMPLE_CONFIG_DIR}")
    monkeypatch.delenv("MAMBA_HELION_AUTOTUNE", raising=False)


# --- tests -------------------------------------------------------------------


@pytest.fixture(
    scope="module", params=[_PACKED], ids=["packed" if _PACKED else "unpacked"]
)
def shadows(request):
    # monkeypatch is function-scoped, so undo the patches by hand here.
    patched = pytest.MonkeyPatch()
    try:
        _resolve_config_dir(patched)
        try:
            yield _run_shadowed(request.param, patched)
        except Exception as error:
            # Anything raised out of the pass -- a missing config, or a Helion compile
            # error from a config that does not fit -- means the config set is wrong for
            # this environment, which is not obvious from the exception itself.
            raise RuntimeError(
                "the Helion pass failed with the config set in "
                f"{os.environ['MAMBA_HELION_CONFIG_DIR']} -- it may not fit this GPU, "
                "toolchain version or shape.\n"
                + _AUTOTUNE_HINT.format(
                    packed=" --sequence-packing --batch-size 1" if request.param else ""
                )
            ) from error
    finally:
        patched.undo()
        torch.cuda.empty_cache()


def _check(shadows, name):
    shadow = shadows[name]
    assert shadow.calls > 0, f"{name} was not reached by the pipeline"
    if shadow.errors:
        raise shadow.errors[0]


@pytest.mark.parametrize("kernel_name", _SSD_KERNELS)
def test_ssd_kernel_against_triton(shadows, kernel_name):
    _check(shadows, kernel_name)


def test_layer_norm_bwd_against_triton(shadows):
    _check(shadows, _LAYER_NORM_BWD)


def test_kernel_namespaces_match():
    """The two dispatch namespaces and _SSD_KERNELS are maintained by hand in
    three places; a kernel added to one and not the others would otherwise just
    go untested, with nothing failing."""
    triton_names = vars(ssd_combined._TRITON_SSD_KERNELS).keys()
    assert vars(get_helion_ssd_kernels()).keys() == triton_names
    assert set(_SSD_KERNELS) == triton_names
