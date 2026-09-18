"""Regression tests for issue #1024: OOB writes in Mamba3 step() when
headdim % 64 != 0 (CuTe step kernel hardcodes tile_D=64, no bounds checking).

Fixed by routing such SISO configs to the Triton step kernel.
"""
import pytest
import torch

from mamba_ssm import Mamba3


def _build(headdim, d_model=64, d_state=64, dtype=torch.float32):
    torch.manual_seed(0)
    m = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim,
               chunk_size=16, is_mimo=False, dtype=dtype).cuda().eval()
    cache = m.allocate_inference_cache(1, None, dtype=dtype, device="cuda")
    return m, cache


@pytest.mark.parametrize("headdim", [16, 32, 64])  # 16/32: Triton fallback, 64: CuTe
def test_step_deterministic_and_no_param_corruption(headdim):
    """Contract: step() is bitwise deterministic and never mutates parameters.

    Before the fix, headdim=16 corrupted dt_bias via OOB writes (#1024)."""
    m, cache = _build(headdim)
    u = torch.randn(1, 64, dtype=torch.float32).cuda()
    params_before = {n: p.detach().clone() for n, p in m.named_parameters()}
    with torch.no_grad():
        o1, *_ = m.step(u, *[c.clone() for c in cache])
        o2, *_ = m.step(u, *[c.clone() for c in cache])
        for _ in range(20):
            m.step(torch.randn_like(u), *[c.clone() for c in cache])
    torch.cuda.synchronize()
    assert torch.equal(o1, o2), "step() is non-deterministic"
    for n, p in m.named_parameters():
        assert torch.equal(p, params_before[n]), f"step() corrupted parameter {n}"


@pytest.mark.parametrize("headdim", [16, 64])
def test_step_matches_forward(headdim):
    """Streaming step() loop must match forward() within kernel tolerance."""
    m, cache = _build(headdim)
    seq = torch.randn(1, 10, 64, dtype=torch.float32).cuda()
    with torch.no_grad():
        ref = m(seq)
        states = list(cache)
        outs = []
        for t in range(10):
            out, *states = m.step(seq[:, t, :], *states)
            outs.append(out)
    streamed = torch.stack(outs, dim=1)
    max_diff = (ref - streamed).abs().max().item()
    assert max_diff < 2e-2, f"step-vs-forward diff {max_diff:.2e} exceeds tolerance"


def test_step_unsupported_headdim_raises():
    """headdim=48: unsafe for CuTe (64 does not divide 48), not power-of-two
    for Triton -> must fail loudly instead of corrupting memory."""
    m, cache = _build(headdim=48, d_model=96)
    u = torch.randn(1, 96, dtype=torch.float32).cuda()
    with pytest.raises(NotImplementedError, match="headdim"):
        with torch.no_grad():
            m.step(u, *cache)
