"""Shape contract for Mamba3.step — matches Mamba2 decode (B, 1, D)."""
import types

import pytest
import torch

from mamba_ssm.modules import mamba3 as mamba3_mod


class _DummyNorm:
    def __init__(self):
        self.weight = torch.ones(1)
        self.eps = 1e-5

    def __call__(self, y, z=None):
        return y


def _stub_step_fn(*args, **kwargs):
    # Fill `out` if provided; otherwise no-op for the fused path.
    out = kwargs.get("out")
    if out is not None:
        out.zero_()
    return None


@pytest.mark.parametrize("keep_seq", [False, True])
def test_mamba3_step_accepts_batched_token(monkeypatch, keep_seq):
    monkeypatch.setattr(mamba3_mod, "mamba3_step_fn", _stub_step_fn)
    monkeypatch.setattr(
        mamba3_mod,
        "apply_rotary_qk_inference_fwd",
        lambda q, k, angle_state, angle_proj, dt, bias_q, bias_k, **kw: (
            q,
            k,
            angle_state.clone(),
        ),
    )

    batch, d_model = 2, 64
    m = mamba3_mod.Mamba3(
        d_model=d_model,
        d_state=16,
        headdim=16,
        expand=2,
        is_mimo=False,
        is_outproj_norm=False,
        device="cpu",
        dtype=torch.float32,
    )
    # Avoid needing the real CuteDSL path beyond the stub.
    m.norm = _DummyNorm()

    angle, ssm, k_state, v_state = m.allocate_inference_cache(
        batch_size=batch, max_seqlen=8, device="cpu", dtype=torch.float32
    )
    if keep_seq:
        u = torch.randn(batch, 1, d_model)
    else:
        u = torch.randn(batch, d_model)

    out, *_ = m.step(u, angle, ssm, k_state, v_state)
    assert out.shape == u.shape
