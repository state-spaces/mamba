"""CPU tests for TileLang contiguity helper used by Mamba3 MIMO (#985)."""

import importlib.util
from pathlib import Path

import torch

_LAYOUT = Path(__file__).resolve().parents[3] / "mamba_ssm/ops/tilelang/mamba3/tilelang_layout.py"
_spec = importlib.util.spec_from_file_location("tilelang_layout", _LAYOUT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
ensure_tilelang_contiguous = _mod.ensure_tilelang_contiguous


def test_ensure_tilelang_contiguous_none():
    assert ensure_tilelang_contiguous(None) is None


def test_ensure_tilelang_contiguous_already_ok():
    t = torch.randn(2, 3, 4)
    out = ensure_tilelang_contiguous(t)
    assert out is not None
    assert out.stride(-1) == 1
    assert out.shape == t.shape


def test_ensure_tilelang_contiguous_fixes_size1_trailing_stride():
    src = torch.randn(2, 8)
    t = torch.as_strided(src, size=(2, 3, 1), stride=(8, 1, 2))
    assert t.shape[-1] == 1
    out = ensure_tilelang_contiguous(t)
    assert out is not None
    assert out.shape == t.shape
    assert out.stride(-1) == 1
    assert torch.allclose(out, t)
