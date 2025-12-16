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


def autotune_configs(configs):
    """Wrap autotune configs for determinism. Uses cached autotuning if available,
    otherwise selects single config via TRITON_AUTOTUNE_BLOCK_SIZE_N or TRITON_AUTOTUNE_CONFIG_INDEX."""
    if not configs or not use_deterministic_mode():
        return configs
    
    if TRITON_HAS_CACHE_RESULTS and os.environ.get("TRITON_CACHE_AUTOTUNING") == "1":
        return configs
    
    global _autotune_warning_issued
    if not _autotune_warning_issued:
        _autotune_warning_issued = True
        if TRITON_HAS_CACHE_RESULTS:
            msg = "Deterministic mode: set TRITON_CACHE_AUTOTUNING=1 for cached autotuning."
        else:
            msg = "Deterministic mode: upgrade to Triton >= 3.4.0 for cached autotuning."
        warnings.warn(msg)

    block_size_n = os.environ.get("TRITON_AUTOTUNE_BLOCK_SIZE_N")
    if block_size_n is not None:
        target_n = int(block_size_n)
        matching = [c for c in configs if c.kwargs.get('BLOCK_SIZE_N') == target_n]
        if matching:
            return matching[:1]

    idx = int(os.environ.get("TRITON_AUTOTUNE_CONFIG_INDEX", "-1"))
    if idx < 0:
        idx += len(configs)
    idx = max(0, min(idx, len(configs) - 1))
    return configs[idx:idx + 1]


def alloc_tile_workspace(base_shape, tile_dim, dtype, device, deterministic, *, zero_init=True):
    """Allocate buffer for deterministic per-program reductions."""
    if base_shape is None:
        return None, 0
    if deterministic:
        factory = torch.zeros if zero_init else torch.empty
        tensor = factory(*base_shape, tile_dim, device=device, dtype=dtype)
        return tensor, tensor.stride(-1)
    tensor = torch.empty(*base_shape, device=device, dtype=dtype)
    return tensor, 0


def finalize_tile_workspace(tensor, deterministic, *, target_dtype=torch.float32):
    """Collapse extra tile dimension (if needed) and optionally cast."""
    if tensor is None:
        return None
    if deterministic:
        tensor = tensor.sum(dim=-1)
    if target_dtype is not None and tensor.dtype != target_dtype:
        tensor = tensor.to(target_dtype)
    return tensor
