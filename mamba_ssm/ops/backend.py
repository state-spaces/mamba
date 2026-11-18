"""Backend dispatch for the Mamba-3 Triton operator families.

Two implementations of the same operator set live side by side and neither one
imports the other:

``mamba_ssm.ops.triton``
    The upstream implementation (CUDA/ROCm), byte-identical to
    ``state-spaces/mamba``.  It contains no Ascend code at all.

``mamba_ssm.ops.triton_ascend``
    The Ascend NPU port (Triton-Ascend / CANN libdevice): pointer-based loads
    instead of TMA descriptors, ``tl.sum`` instead of N=1 ``tl.dot``, UB-bounded
    chunk sizes, grid-stride launches under the ``UINT16_MAX`` coreDim limit.

Only the model entry (``mamba_ssm.modules.mamba3``) has to care which one is
active; it imports the public names from here.  Code that is written for one
backend on purpose imports its package directly:

* CUDA-side code (upstream tests, ``scripts/golden_a100``, ``ops/tilelang``)
  imports ``mamba_ssm.ops.triton...``;
* Ascend-side code (``ops/ascendc``, ``ops/tilelang_ascend``, NPU tests and
  scripts) imports ``mamba_ssm.ops.triton_ascend...``.

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
"""

from __future__ import annotations

import functools
import importlib
import logging
import os

#: ``cuda`` keeps the upstream implementation, ``ascend``/``npu`` selects the port.
BACKEND_ENV = "MAMBA3_TRITON_BACKEND"

CUDA_PACKAGE = "mamba_ssm.ops.triton"
ASCEND_PACKAGE = "mamba_ssm.ops.triton_ascend"

_ASCEND_ALIASES = ("ascend", "npu", "cann")
_CUDA_ALIASES = ("cuda", "gpu", "nvidia")

#: Public name -> (submodule relative to the selected package, attribute name).
#:
#: ``RMSNormGated`` is the gated normalisation class that upstream imports as
#: ``from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated``;
#: the non-gated ``RMSNorm`` lives in ``layer_norm`` and is not on the Mamba-3
#: SISO path, so it is not re-exported here.
_EXPORTS = {
    # --- Mamba-3 SISO forward / backward / decode -------------------------
    "mamba3_siso_combined": ("mamba3.mamba3_siso_combined", "mamba3_siso_combined"),
    "mamba3_siso_fwd": ("mamba3.mamba3_siso_fwd", "mamba3_siso_fwd"),
    "mamba3_siso_step": ("mamba3.mamba3_siso_step", "mamba3_siso_step"),
    "angle_dt_fwd": ("mamba3.angle_dt", "angle_dt_fwd"),
    "angle_dt_bwd": ("mamba3.angle_dt", "angle_dt_bwd"),
    "compute_dzdo": ("mamba3.mamba3_siso_bwd", "compute_dzdo"),
    "compute_dqkv": ("mamba3.mamba3_siso_bwd", "compute_dqkv"),
    "compute_dqktheta": ("mamba3.mamba3_siso_bwd", "compute_dqktheta"),
    "compute_ddt_dtrap_dinput_states": (
        "mamba3.mamba3_siso_bwd",
        "compute_ddt_dtrap_dinput_states",
    ),
    "apply_rotary_qk_inference_fwd": (
        "mamba3.mamba3_mimo_rotary_step",
        "apply_rotary_qk_inference_fwd",
    ),
    # --- gated normalisation used by the Mamba-3 block -------------------
    # (the non-gated ``layer_norm`` pair is not on the Mamba-3 SISO path, so it
    # is deliberately not dispatched: ``mamba2``/``block`` keep importing the
    # upstream module)
    "RMSNormGated": ("layernorm_gated", "RMSNorm"),
    "layernorm_fn": ("layernorm_gated", "layernorm_fn"),
    "rmsnorm_fn": ("layernorm_gated", "rmsnorm_fn"),
}

__all__ = sorted(_EXPORTS)


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


def ops_package() -> str:
    """Fully qualified package holding the operators for the active backend."""
    return ASCEND_PACKAGE if is_ascend() else CUDA_PACKAGE


def import_ops(name: str):
    """Import ``<package>.<name>`` for the active backend.

    ``name`` is a dotted path relative to the package, e.g.
    ``"mamba3.mamba3_siso_bwd"``.  Used by tests and probes that need the module
    object itself (to inspect autotune configs or monkeypatch a kernel).
    """
    return importlib.import_module(f"{ops_package()}.{name}")


def __getattr__(name: str):
    """Resolve a public operator lazily from the selected package."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_ops(module_name), attribute)
    globals()[name] = value  # memoize: the second lookup skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
