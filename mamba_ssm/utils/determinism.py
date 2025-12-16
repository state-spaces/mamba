# Copyright (c) 2024, Tri Dao, Albert Gu.

import os
import torch

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
