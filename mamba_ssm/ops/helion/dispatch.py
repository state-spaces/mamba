import os
import warnings
from functools import cache
from types import SimpleNamespace

from ...utils.determinism import use_deterministic_mode


MAMBA_USE_HELION = "MAMBA_USE_HELION"

_deterministic_warning_issued = False


@cache
def _helion_opt_in() -> bool:
    """Return the process-level Helion opt-in captured at first dispatch."""
    return os.environ.get(MAMBA_USE_HELION, "0") == "1"


def use_helion() -> bool:
    """Return whether Helion kernels are selected, warning about determinism.

    The env read stays cached, but the determinism check does not: deterministic
    mode can be turned on after the first dispatch, and every call site asks on
    every dispatch, so this catches that too. Warns at most once per process.
    """
    enabled = _helion_opt_in()
    if enabled:
        _warn_if_deterministic()
    return enabled


def _warn_if_deterministic() -> None:
    """Warn that the Helion kernels ignore deterministic mode.

    Unlike the Triton kernels, which route their reductions through
    ``alloc_tile_workspace``/``finalize_tile_workspace`` when
    ``use_deterministic_mode()`` is on, the Helion backward kernels decide at
    compile time and never ask.
    """
    global _deterministic_warning_issued
    if _deterministic_warning_issued or not use_deterministic_mode():
        return
    _deterministic_warning_issued = True
    warnings.warn(
        "MAMBA_USE_HELION=1 while deterministic mode is enabled, but the Helion "
        "kernels do not honour it: chunk_scan_chunk_state_bwd_dx (ddt and 1-D dD) "
        "and chunk_scan_bwd_dC (ddA_cumsum_prev) always reduce with atomic adds, "
        "so the backward is not bitwise reproducible. Unset MAMBA_USE_HELION for "
        "the deterministic Triton path.",
        stacklevel=2,
    )


@cache
def get_helion_ssd_kernels():
    """Lazily import Helion SSD implementations only after explicit opt-in."""
    from ._chunk_scan_bwd_dstates import _chunk_scan_bwd_dstates
    from ._chunk_scan_chunk_state_bwd_dx import _chunk_scan_chunk_state_bwd_dx
    from ._chunk_scan_fwd import _chunk_scan_fwd
    from ._chunk_state_fwd import _chunk_state_fwd
    from ._state_passing_bwd import _state_passing_bwd
    from ._state_passing_fwd import _state_passing_fwd_v2

    return SimpleNamespace(
        chunk_scan_bwd_dstates=_chunk_scan_bwd_dstates,
        chunk_scan_chunk_state_bwd_dx=_chunk_scan_chunk_state_bwd_dx,
        chunk_scan_fwd=_chunk_scan_fwd,
        chunk_state_fwd=_chunk_state_fwd,
        state_passing_bwd=_state_passing_bwd,
        state_passing_fwd=_state_passing_fwd_v2,
    )
