# Copyright (c) 2024, Tri Dao, Albert Gu.

import os
import warnings
from packaging import version

import torch

try:
    import triton
    TRITON_VERSION = version.parse(triton.__version__)
except ImportError:
    TRITON_VERSION = version.parse("0.0.0")

TRITON_HAS_CACHE_RESULTS = TRITON_VERSION >= version.parse("3.4.0")
_autotune_warning_issued = False

_deterministic_override = None


def use_deterministic_mode():
    if _deterministic_override is not None:
        return _deterministic_override
    env = os.environ.get('MAMBA_DETERMINISTIC')
    if env:
        return env[0] == '1'
    return torch.are_deterministic_algorithms_enabled()


def set_deterministic_mode(value):
    global _deterministic_override
    _deterministic_override = value


def _estimate_config_cost(cfg):
    """Estimate shared memory cost of a config. Lower is cheaper."""
    block_product = 1
    for key, val in cfg.kwargs.items():
        if key.startswith('BLOCK_SIZE_'):
            block_product *= val
    return block_product * (getattr(cfg, 'num_stages', 1) or 1)


def _filter_configs_by_block_sizes(configs):
    """Pin tile sizes from ``TRITON_AUTOTUNE_BLOCK*`` environment variables.

    ``TRITON_AUTOTUNE_<BLOCK_KEY>=<int>`` constrains the configs whose kwargs contain
    ``<BLOCK_KEY>`` (for example ``TRITON_AUTOTUNE_BLOCK_SIZE_H=2`` or
    ``TRITON_AUTOTUNE_BLOCK_SIZE_M=64``). Variables naming a key that none of
    this kernel's configs use are ignored for this kernel, so one environment
    can pin different kernels at once; the constraints that do apply are ANDed.

    Returns ``[first matching config]``, or ``None`` when no constraint applies
    to this kernel or none of its configs satisfies them (the caller then falls
    back to the cheapest config).
    """
    prefix = "TRITON_AUTOTUNE_"
    keys_present = {key for cfg in configs for key in cfg.kwargs}
    applicable = {}
    for env_key, env_val in os.environ.items():
        key = env_key[len(prefix):]
        if env_key.startswith(prefix + "BLOCK") and key in keys_present:
            applicable[key] = int(env_val)  # only values for this kernel's keys are parsed; "" is an error
    if not applicable:
        return None
    matching = [c for c in configs if all(c.kwargs.get(k) == v for k, v in applicable.items())]
    return matching[:1] if matching else None


def autotune_configs(configs):
    """Select autotune configs for deterministic mode.
    
    Uses cached autotuning (TRITON_CACHE_AUTOTUNING=1) if Triton >= 3.4.0,
    otherwise auto-selects the cheapest config by block size * stages.
    """
    if not configs or not use_deterministic_mode():
        return configs
    if TRITON_HAS_CACHE_RESULTS and os.environ.get("TRITON_CACHE_AUTOTUNING") == "1":
        return configs
    global _autotune_warning_issued
    if not _autotune_warning_issued:
        _autotune_warning_issued = True
        msg = "Deterministic mode: set TRITON_CACHE_AUTOTUNING=1 for cached autotuning." if TRITON_HAS_CACHE_RESULTS else "Deterministic mode: upgrade to Triton >= 3.4.0 for cached autotuning."
        warnings.warn(msg)
    filtered = _filter_configs_by_block_sizes(configs)
    if filtered:
        return filtered
    return [min(configs, key=_estimate_config_cost)]


def alloc_tile_workspace(base_shape, tile_dim, dtype, device, deterministic, *, zero_init=True):
    """Allocate buffer for deterministic per-program reductions."""
    if base_shape is None:
        return None, 0
    if deterministic:
        factory = torch.zeros if zero_init else torch.empty
        tensor = factory(*base_shape, tile_dim, device=device, dtype=dtype)
        return tensor, tensor.stride(-1)
    return torch.empty(*base_shape, device=device, dtype=dtype), 0


def finalize_tile_workspace(tensor, deterministic):
    if tensor is None:
        return None
    if deterministic:
        tensor = tensor.sum(dim=-1)
    return tensor
