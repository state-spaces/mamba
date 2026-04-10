"""
Mamba-3 module-level varlen tests.

Tests that MIMO (and SISO) produce identical outputs for packed varlen sequences
vs. running each sequence independently in non-varlen mode.

Copyright (c) 2026, Dao AI Lab, Goombalab.
"""
import pytest
import torch
import torch.nn.functional as F

from mamba_ssm.modules.mamba3 import Mamba3


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(is_mimo: bool, **kwargs) -> Mamba3:
    defaults = dict(
        d_model=128,
        d_state=64,
        expand=2,
        headdim=64,
        ngroups=1,
        is_mimo=is_mimo,
        mimo_rank=2,
        chunk_size=16,
        layer_idx=0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    defaults.update(kwargs)
    model = Mamba3(**defaults).eval()
    return model


def _pack_sequences(seqs):
    """Pack a list of (1, Li, d) tensors into (1, sum(Li), d)."""
    return torch.cat(seqs, dim=1)


def _cu_seqlens(lengths, device):
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    # cumsum on int32 promotes to int64 on CUDA; cast back before padding
    return F.pad(torch.cumsum(lens, dim=0).to(torch.int32), (1, 0))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_mimo", [False, True])
def test_mamba3_varlen_matches_nonvarlen(is_mimo):
    """Packed varlen output should match per-sequence non-varlen output."""
    _require_cuda()
    torch.manual_seed(0)

    model = _make_model(is_mimo=is_mimo)
    d_model = model.d_model
    lengths = [17, 25]  # two sequences, neither a multiple of chunk_size=16
    device = "cuda"

    # Build individual inputs
    seqs = [torch.randn(1, L, d_model, device=device, dtype=torch.bfloat16) for L in lengths]

    # Run model on each sequence independently (non-varlen reference)
    with torch.no_grad():
        outs_ref = [model(s) for s in seqs]

    # Run model on packed sequence (varlen)
    packed = _pack_sequences(seqs)
    cu_sl = _cu_seqlens(lengths, device)
    with torch.no_grad():
        out_varlen = model(packed, cu_seqlens=cu_sl)

    # Compare each slice against its reference
    atol = 5e-2  # bfloat16 + chunked cumsum accumulates some error
    offset = 0
    for i, (L, ref) in enumerate(zip(lengths, outs_ref)):
        slc = out_varlen[:, offset : offset + L]
        assert torch.allclose(slc, ref, atol=atol), (
            f"Sequence {i} (length {L}): max diff = {(slc - ref).abs().max():.4f}"
        )
        offset += L


@pytest.mark.parametrize("is_mimo", [False, True])
def test_mamba3_varlen_backward(is_mimo):
    """Backward pass through varlen mode should produce finite gradients."""
    _require_cuda()
    torch.manual_seed(1)

    model = _make_model(is_mimo=is_mimo)
    d_model = model.d_model
    lengths = [19, 29, 37]
    device = "cuda"

    packed = torch.randn(
        1, sum(lengths), d_model, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    cu_sl = _cu_seqlens(lengths, device)

    out = model(packed, cu_seqlens=cu_sl)
    out.float().sum().backward()

    assert packed.grad is not None, "No gradient for input"
    assert torch.isfinite(packed.grad).all(), "Non-finite gradient for input"
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for param {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for param {name}"
