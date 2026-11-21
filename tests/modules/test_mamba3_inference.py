"""
Mamba-3 module-level inference tests.

Copyright (c) 2026, Dao AI Lab, Goombalab.
"""

import pytest
import torch

from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.utils.generation import InferenceParams


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


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
    return Mamba3(**defaults).eval()


@pytest.mark.parametrize("is_mimo", [False, True])
def test_mamba3_inference_params_prefill_uses_cached_states(is_mimo):
    """Prefill should consume cached states when seqlen_offset stays at 0."""
    _require_cuda()
    torch.manual_seed(2)

    model = _make_model(is_mimo=is_mimo)
    d_model = model.d_model
    split = 32
    seqlen = 48
    u = torch.randn(1, seqlen, d_model, device="cuda", dtype=torch.bfloat16)
    inference_params = InferenceParams(max_seqlen=seqlen, max_batch_size=1)

    with torch.no_grad():
        full = model(u)
        first = model(u[:, :split], inference_params=inference_params)
        second = model(u[:, split:], inference_params=inference_params)

    combined = torch.cat([first, second], dim=1)
    assert torch.allclose(combined, full, atol=8e-2, rtol=8e-2), (
        f"max diff = {(combined - full).abs().max():.4f}"
    )
