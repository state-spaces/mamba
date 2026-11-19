"""Backend dispatch for the Mamba-3 operator families.

Three operator families, each with a CUDA implementation and an Ascend NPU port.
Neither side imports the other; only this module picks one, so the model entry
(``mamba_ssm.modules.mamba3``) stays backend-agnostic and the CUDA packages stay
byte-identical to upstream:

=======================  =============================  ================================
family                   CUDA / ROCm                    Ascend NPU
=======================  =============================  ================================
``triton``   (SISO)      ``mamba_ssm.ops.triton``       ``mamba_ssm.ops.triton_ascend``
``tilelang`` (MIMO)      ``mamba_ssm.ops.tilelang``     ``mamba_ssm.ops.tilelang_ascend``
``step``     (decode)    ``mamba_ssm.ops.cute``         ``mamba_ssm.ops.ascendc``
=======================  =============================  ================================

``triton`` and ``tilelang`` are *mirrored* families: the same module path exists
in both packages, so an export is ``(family, module, attribute)`` and switching
backends only rewrites the package prefix.  The decode step is not mirrored --
CUDA calls a cutlass/CuteDSL function, the NPU calls the Ascend C
``mamba3_decode_step`` operator, and the two live in differently named modules
as well -- so it gets an explicit per-backend table.

Resolution order
----------------
1. ``MAMBA3_TRITON_BACKEND`` environment variable, when set:
   ``cuda`` (also ``gpu``/``nvidia``) or ``ascend`` (also ``npu``/``cann``).
   An unrecognized value raises, because silently running the wrong numeric
   path is worse than a loud failure.
2. Auto-detection: Triton-Ascend exposes CANN libdevice
   (``triton.language.extra.cann.libdevice``) on a CANN build, so its presence
   selects ``ascend``.  This is the same probe ``triton_ascend``'s
   ``npu_compat.IS_CANN`` uses, so dispatch and in-kernel branching always
   agree.
3. ``torch.npu`` reported as available selects ``ascend`` as well (covers builds
   whose CANN libdevice lives under a different import path).
4. Otherwise ``cuda``.

The choice is memoized on first use: setting the environment variable after an
operator has been imported has no effect.

Required versus optional families
---------------------------------
``triton`` is required: a failure to import it propagates.  ``tilelang`` and
``step`` resolve to ``None`` instead, because each needs an extra toolchain that
a given machine may simply not have -- a TileLang built with
``USE_ASCEND910=ON`` for MIMO, and the compiled ``libmamba3_decode_step_ops.so``
for the Ascend C step (CUDA needs ``nvidia-cutlass-dsl`` + ``quack-kernels`` for
its own).  An ``ImportError`` is silent (the operator simply is not installed);
any other exception during import is logged, because that is a broken install
rather than an absent one.  The model code keeps its ``is not None`` assertions,
so a degraded family still fails loudly at the first use, never silently.

``MAMBA3_DISABLE_TILELANG=1`` (also ``true``/``yes``/``on``) forbids the
``tilelang`` family outright, on either backend.  The import can fail with a
non-``ImportError`` on a CANN build (see ``USAGE_mamba3_mimo_npu.md``), and it is
eager: ``modules/mamba3.py`` binds ``mamba3_mimo_combined`` at import time.  A
SISO-only run therefore sets this switch to skip the TileLang import (and its
build requirements) altogether; nothing on the SISO path needs it.

Call sites
----------
Model code imports the public names from here.  Code that is written for one
backend on purpose imports its package directly:

* CUDA-side code (upstream tests, ``scripts/golden_a100``, ``ops/tilelang``)
  imports ``mamba_ssm.ops.triton...`` / ``mamba_ssm.ops.tilelang...``;
* Ascend-side code (``ops/ascendc``, ``ops/tilelang_ascend``, NPU tests and
  scripts) imports ``mamba_ssm.ops.triton_ascend...`` and
  ``mamba_ssm.ops.tilelang_ascend...``.
"""

from __future__ import annotations

import functools
import importlib
import logging
import os
from typing import Optional

#: ``cuda`` keeps the upstream implementation, ``ascend``/``npu`` selects the port.
BACKEND_ENV = "MAMBA3_TRITON_BACKEND"

#: Forbids the TileLang (MIMO) family on either backend.
DISABLE_TILELANG_ENV = "MAMBA3_DISABLE_TILELANG"

_ASCEND_ALIASES = ("ascend", "npu", "cann")
_CUDA_ALIASES = ("cuda", "gpu", "nvidia")
_TRUTHY = ("1", "true", "yes", "on")

#: Mirrored families: the same module layout exists inside both packages.
_FAMILY_PACKAGES = {
    "triton": {
        "cuda": "mamba_ssm.ops.triton",
        "ascend": "mamba_ssm.ops.triton_ascend",
    },
    "tilelang": {
        "cuda": "mamba_ssm.ops.tilelang",
        "ascend": "mamba_ssm.ops.tilelang_ascend",
    },
}

#: The decode step is not mirrored: a different package *and* module per backend.
_STEP_MODULES = {
    "cuda": ("mamba_ssm.ops.cute.mamba3.mamba3_step_fn", "mamba3_step_fn"),
    "ascend": ("mamba_ssm.ops.ascendc.mamba3_step", "mamba3_step_fn"),
}

#: Families that may legitimately be absent on a machine (see module docstring).
_OPTIONAL_FAMILIES = frozenset({"tilelang", "step"})

#: Public name -> (family, module relative to the family package, attribute).
#:
#: ``RMSNormGated`` is the gated normalisation class that upstream imports as
#: ``from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated``;
#: the non-gated ``RMSNorm`` lives in ``layer_norm`` and is not on the Mamba-3
#: SISO path, so it is not re-exported here.
_EXPORTS = {
    # --- Mamba-3 SISO forward / backward / decode -------------------------
    "mamba3_siso_combined": ("triton", "mamba3.mamba3_siso_combined", "mamba3_siso_combined"),
    "mamba3_siso_fwd": ("triton", "mamba3.mamba3_siso_fwd", "mamba3_siso_fwd"),
    "mamba3_siso_step": ("triton", "mamba3.mamba3_siso_step", "mamba3_siso_step"),
    "angle_dt_fwd": ("triton", "mamba3.angle_dt", "angle_dt_fwd"),
    "angle_dt_bwd": ("triton", "mamba3.angle_dt", "angle_dt_bwd"),
    "compute_dzdo": ("triton", "mamba3.mamba3_siso_bwd", "compute_dzdo"),
    "compute_dqkv": ("triton", "mamba3.mamba3_siso_bwd", "compute_dqkv"),
    "compute_dqktheta": ("triton", "mamba3.mamba3_siso_bwd", "compute_dqktheta"),
    "compute_ddt_dtrap_dinput_states": (
        "triton",
        "mamba3.mamba3_siso_bwd",
        "compute_ddt_dtrap_dinput_states",
    ),
    "apply_rotary_qk_inference_fwd": (
        "triton",
        "mamba3.mamba3_mimo_rotary_step",
        "apply_rotary_qk_inference_fwd",
    ),
    # --- Mamba-3 MIMO (TileLang on both sides) ----------------------------
    "mamba3_mimo": ("tilelang", "mamba3.mamba3_mimo", "mamba3_mimo"),
    # --- gated normalisation used by the Mamba-3 block -------------------
    # (the non-gated ``layer_norm`` pair is not on the Mamba-3 SISO path, so it
    # is deliberately not dispatched: ``mamba2``/``block`` keep importing the
    # upstream module)
    "RMSNormGated": ("triton", "layernorm_gated", "RMSNorm"),
    "layernorm_fn": ("triton", "layernorm_gated", "layernorm_fn"),
    "rmsnorm_fn": ("triton", "layernorm_gated", "rmsnorm_fn"),
}

#: Names that resolve through :data:`_STEP_MODULES` instead of a mirrored family.
_STEP_EXPORTS = frozenset({"mamba3_step_fn"})

__all__ = sorted(set(_EXPORTS) | _STEP_EXPORTS)


def _cann_libdevice_available() -> bool:
    """True when Triton ships the CANN libdevice (i.e. this is Triton-Ascend)."""
    try:
        from triton.language.extra.cann import libdevice  # noqa: F401
    except Exception:  # CUDA / ROCm / plain Triton builds do not have it
        return False
    return True


def _torch_npu_available() -> bool:
    """True when a torch build exposing an available ``torch.npu`` is imported."""
    try:
        import torch
    except Exception:
        return False
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    try:
        return bool(npu.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def backend() -> str:
    """Return the active backend, ``"cuda"`` or ``"ascend"``."""
    forced = os.environ.get(BACKEND_ENV, "").strip().lower()
    if forced:
        if forced in _ASCEND_ALIASES:
            chosen, source = "ascend", f"{BACKEND_ENV}={forced}"
        elif forced in _CUDA_ALIASES:
            chosen, source = "cuda", f"{BACKEND_ENV}={forced}"
        else:
            raise ValueError(
                f"{BACKEND_ENV}={forced!r} is not a backend; "
                f"use one of {_CUDA_ALIASES + _ASCEND_ALIASES}"
            )
    elif _cann_libdevice_available():
        chosen, source = "ascend", "triton CANN libdevice"
    elif _torch_npu_available():
        chosen, source = "ascend", "torch.npu is available"
    else:
        chosen, source = "cuda", "no CANN libdevice / torch.npu"

    logging.getLogger(__name__).info("Mamba-3 Triton backend: %s (%s)", chosen, source)
    return chosen


def is_ascend() -> bool:
    """Convenience predicate for call sites that only need a yes/no."""
    return backend() == "ascend"


def triton_package() -> str:
    """Package holding the Triton operator families (SISO + the norms)."""
    return _FAMILY_PACKAGES["triton"][backend()]


def ops_package() -> str:
    """Alias of :func:`triton_package`, kept for older call sites and probes."""
    return triton_package()


def _tilelang_disabled() -> bool:
    return os.environ.get(DISABLE_TILELANG_ENV, "").strip().lower() in _TRUTHY


def tilelang_package() -> Optional[str]:
    """Package holding the MIMO operators, or ``None`` when that route is off."""
    if _tilelang_disabled():
        return None
    return _FAMILY_PACKAGES["tilelang"][backend()]


def step_module() -> str:
    """Dotted path of the module that holds ``mamba3_step_fn`` on this backend."""
    return _STEP_MODULES[backend()][0]


def import_ops(name: str):
    """Import ``<triton package>.<name>`` for the active backend.

    ``name`` is a dotted path relative to the package, e.g.
    ``"mamba3.mamba3_siso_bwd"``.  Used by tests and probes that need the module
    object itself (to inspect autotune configs or monkeypatch a kernel).
    """
    return importlib.import_module(f"{triton_package()}.{name}")


def _module_for(name: str) -> tuple:
    """Return ``(family, module path, attribute)`` for an exported name."""
    if name in _STEP_EXPORTS:
        module_path, attribute = _STEP_MODULES[backend()]
        return "step", module_path, attribute
    family, module, attribute = _EXPORTS[name]
    package = tilelang_package() if family == "tilelang" else _FAMILY_PACKAGES[family][backend()]
    if package is None:  # only tilelang can be switched off explicitly
        return family, "", attribute
    return family, f"{package}.{module}", attribute


def _resolve(name: str):
    """Resolve an exported name, or ``None`` when its optional family is absent."""
    family, module_path, attribute = _module_for(name)
    if not module_path:
        return None
    log = logging.getLogger(__name__)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        if family not in _OPTIONAL_FAMILIES:
            raise
        log.debug("%s is not installed on the %s backend (%s)", module_path, backend(), exc)
        return None
    except Exception as exc:
        if family not in _OPTIONAL_FAMILIES:
            raise
        log.warning(
            "%s failed to import on the %s backend (%s: %s); "
            "the operator is unavailable",
            module_path,
            backend(),
            type(exc).__name__,
            exc,
        )
        return None
    return getattr(module, attribute)


def __getattr__(name: str):
    """Resolve a public operator lazily from the selected backend."""
    if name not in _EXPORTS and name not in _STEP_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = _resolve(name)
    globals()[name] = value  # memoize: the second lookup skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
