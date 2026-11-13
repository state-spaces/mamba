"""Load the repo's Mamba-3 MIMO test assets and re-point them at the A2 port.

This module lives in ``tests/ops/tilelang_ascend/``, next to :mod:`__init__`, so
``from . import _upstream`` resolves the way it did inside the operator package on
the migration branch.  ``_REPO`` is therefore ``parents[3]``
(file -> ``tilelang_ascend`` -> ``tests/ops`` -> repository root).

The file this loads is ``<repo>/tests/ops/tilelang/test_mamba3_mimo.py`` -- the
**upstream CUDA suite, unmodified**.  Everything it provides is device agnostic and
is used unchanged: input construction, both fp32 references, the case grids, and
the accuracy standard itself (``assert_stable_rel`` / ``REL_TOL``).

Four adaptations are needed, all applied here and none of them touching the
standard:

1. the device the suite allocates on -- upstream hardcodes ``device="cuda"``, so
   ``device="cuda"`` / ``device: str = "cuda"`` become ``A2_DEVICE``;
2. ``torch.cuda.manual_seed_all`` (CUDA only) becomes a no-op;
3. ``_require_cuda_and_kernel_deps`` (which skips without CUDA) is neutered;
4. the decay chain -- upstream calls ``mods.utils.compute_dacs_segsum_triton``,
   which faults on the NPU (``SUSPECT REMOTE ERROR``), so a torch implementation of
   the same formula is supplied in its place.

They are applied as **source edits**, not as parameters, because the upstream file
has no ``device``/``dacs`` knobs (a fork that added them is also understood, so this
loader keeps working if such a copy is put back).

The upstream file lives at ``<repo>/tests/ops/tilelang/test_mamba3_mimo.py``, i.e.
*outside* the installed package.  When it is missing (a wheel install rather than a
source checkout) :func:`load` raises and the test modules skip.

Scope note: the migration branch keeps two further groups of helpers in this module
-- the packed-varlen builders / references, and the CANN mixed-tolerance judge
(``tools/mixed_tolerance.py``) -- because its internal suites use them.  Neither is
part of this repository, so only what ``test_mamba3_mimo_upstream_std.py`` consumes
is kept here.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

DEVICE = "npu"

_REPO = Path(__file__).resolve().parents[3]
UPSTREAM_TEST = _REPO / "tests/ops/tilelang/test_mamba3_mimo.py"
_MODULE_NAME = "a2_upstream_mimo"

__all__ = [
    "DEVICE",
    "UPSTREAM_TEST",
    "build_inputs",
    "load",
    "ref_backward",
    "ref_forward",
]


def _upstream_path() -> Path:
    """``A2_UPSTREAM_TEST`` if set (e.g. a separate CUDA checkout), else the repo's."""

    env = os.environ.get("A2_UPSTREAM_TEST")
    return Path(env) if env else UPSTREAM_TEST


def dacs_torch(dA: torch.Tensor, chunk_size: int):
    """The triton decay-chain kernel, recomputed in torch/fp32 on the host.

    Same formula as ``compute_dacs_segsum_triton``: per-chunk cumulative sums, the
    reversed sums, and the causal segment sums, all clamped at 0.  Computed on the
    CPU because the NPU triton path faults, then moved back to ``dA.device``.
    """

    b, h, s = dA.shape
    nchunks = s // chunk_size
    da = dA.detach().to("cpu", torch.float32).reshape(b, h, nchunks, chunk_size)
    cs = torch.cumsum(da, dim=-1)
    cs_rev = da.sum(dim=-1)[..., None] - cs
    grid = torch.ones(chunk_size, chunk_size, dtype=torch.bool).tril(diagonal=-1)
    segsum = torch.cumsum(
        da.unsqueeze(-1).expand(-1, -1, -1, -1, chunk_size).masked_fill(~grid, 0),
        dim=-2)
    return (cs.clamp(max=0.0).reshape(b, h, s).to(dA.device),
            cs_rev.clamp(max=0.0).reshape(b, h, s).to(dA.device),
            segsum.clamp(max=0.0).to(dA.device))


#: Upstream ``build_inputs`` only reaches into ``mods`` for the decay chain.
DACS_MODS = SimpleNamespace(utils=SimpleNamespace(
    compute_dacs_segsum_triton=dacs_torch,
    compute_dacs_segsum_triton_varlen=dacs_torch))


def _patch_source(src: str) -> str:
    """Apply adaptations 1-3 to the upstream source."""

    assert src.count("F = torch.nn.functional\n") == 1
    src = src.replace("F = torch.nn.functional\n",
                      'F = torch.nn.functional\nA2_DEVICE = "npu"\n')
    src = src.replace('device="cuda"', "device=A2_DEVICE")
    src = src.replace('device: str = "cuda"', "device: str = A2_DEVICE")
    src = src.replace("torch.cuda.manual_seed_all(seed)",
                      "pass  # CUDA-only upstream")
    guard = "def _require_cuda_and_kernel_deps() -> None:\n"
    assert src.count(guard) == 1
    src = src.replace(guard, guard + "    return  # NPU run: admission check off\n")
    return src


def load() -> SimpleNamespace:
    """Import (once) the upstream test module with the NPU adaptations applied."""

    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    path = _upstream_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"upstream test file not found: {path} -- the NPU suite needs the "
            "CUDA repo's tests/ops/tilelang/test_mamba3_mimo.py in the source tree")
    src = path.read_text(encoding="utf-8")
    forked = "dacs: str" in src
    # The test file's directory goes on the path because a fork may import
    # helpers that live next to it.
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))

    if forked:
        # A fork that added device/dacs parameters: import it, then inject.
        import importlib.util

        spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = mod
        spec.loader.exec_module(mod)
    else:
        # The pristine upstream file: adapt the source before executing it.
        mod = types.ModuleType(_MODULE_NAME)
        mod.__file__ = str(path)
        sys.modules[_MODULE_NAME] = mod
        exec(compile(_patch_source(src), str(path), "exec"), mod.__dict__)
    _patch(mod, forked=forked)
    return mod


def _patch(mod: SimpleNamespace, *, forked: bool) -> None:
    raw_build = mod.build_inputs

    if forked:
        def build_inputs(*args, **kwargs):
            kwargs.setdefault("device", DEVICE)
            kwargs.setdefault("dacs", "torch")
            return raw_build(*args, **kwargs)
    else:
        def build_inputs(*args, **kwargs):
            # No such knobs upstream: the device is compiled in, and the decay
            # chain is always the torch implementation (the triton one faults).
            kwargs.pop("device", None)
            kwargs.pop("dacs", None)
            if kwargs.get("mods") is None:
                kwargs["mods"] = DACS_MODS
            return raw_build(*args, **kwargs)

    mod.build_inputs = build_inputs
    mod._require_cuda_and_kernel_deps = lambda *a, **k: None


# --------------------------------------------------------------------------- #
# references (CPU fp32 -- the NPU Mul kernel rejects the >8-D kron the reference
# builds, and fp32 on the host is the standard anyway)
# --------------------------------------------------------------------------- #
REF_ORDER = ("q", "k", "v", "q_bias", "k_bias", "mimo_v", "mimo_o", "z",
             "mimo_z", "angles", "dA_cs", "dA_cs_rev", "dt", "trap", "D")


def to_cpu(inputs: dict) -> dict:
    return {k: (v.detach().to("cpu", torch.float32)
                if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}


def ref_forward(inputs: dict, *, chunk_size: int, rotary_dim_divisor: int,
                return_state: bool = False, fused_norm: bool = False):
    """The upstream fp32 reference forward on the CPU.

    ``fused_norm`` selects the ``fuse_pregate_headwise_rms_norm`` epilogue, which
    also needs ``OUT_NORM_WEIGHT``.
    """

    up = load()
    cpu = to_cpu(inputs)
    extra = ({"fused_norm": True,
              "outproj_norm_weight": cpu["outproj_norm_weight"].clone()}
             if fused_norm else {})
    return up.mamba3_MIMO_chunk_ref(
        *[None if cpu[k] is None else cpu[k].clone() for k in REF_ORDER],
        chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor,
        return_final_state=return_state, dtype=torch.float32, **extra)


def ref_backward(inputs: dict, grad_output: torch.Tensor, *,
                 chunk_size: int, rotary_dim_divisor: int,
                 fused_norm: bool = False) -> dict:
    """The upstream fp32 reference backward (autograd).

    ``run_ref_backward_fp32`` rebuilds the decay chain with
    ``compute_dacs_segsum_triton``, which faults on the NPU; the stub hands it the
    exact tensors the kernel consumed so both sides share one chain.
    """

    up = load()
    cpu = to_cpu(inputs)
    stub = SimpleNamespace(
        utils=SimpleNamespace(compute_dacs_segsum_triton=lambda dA, cs: (
            cpu["dA_cs"], cpu["dA_cs_rev"], cpu["segsum"])))
    return up.run_ref_backward_fp32(
        stub, cpu, grad_output=grad_output.detach().to("cpu", torch.float32),
        **({"fused_norm": True} if fused_norm else {}))
