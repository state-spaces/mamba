"""Regression tests for varlen packing helpers."""

import torch
import torch.nn.functional as F


def _cu_seqlens(lengths, device):
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    return F.pad(torch.cumsum(lens, dim=0).to(torch.int32), (1, 0))


def test_cu_seqlens_stays_int32_on_cuda():
    if not torch.cuda.is_available():
        return
    cu = _cu_seqlens([17, 25], device="cuda")
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 17, 42]
