"""Shared TileLang tensor layout helpers."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def ensure_tilelang_contiguous(t: Optional[Tensor]) -> Optional[Tensor]:
    """Make tensor truly contiguous with last-stride=1.

    PyTorch's `.contiguous()` treats a size-1 trailing dimension as contiguous even
    when `stride(-1) != 1` (e.g. after transpose). TileLang kernels require physical
    stride 1 — without this, Mamba3 MIMO `forward` fails at `seqlen=1` (#985).
    """
    if t is None:
        return None
    t = t.contiguous()
    if t.stride(-1) != 1:
        t = torch.empty(t.shape, device=t.device, dtype=t.dtype).copy_(t)
    return t
