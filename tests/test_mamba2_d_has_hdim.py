"""Test that Mamba2 forward() and step() produce consistent outputs when D_has_hdim=True.

Regression test for https://github.com/state-spaces/mamba/issues/887
"""
import torch
import pytest
from einops import rearrange

from mamba_ssm.modules.mamba2 import Mamba2


@pytest.mark.parametrize("D_has_hdim", [True, False])
def test_mamba2_step_forward_consistency(D_has_hdim):
    """step() must match forward() for D_has_hdim=True and D_has_hdim=False."""
    torch.manual_seed(42)
    batch, seqlen = 2, 16
    d_model, headdim, d_state = 256, 64, 64
    device = "cuda"
    dtype = torch.float32

    model = Mamba2(
        d_model=d_model,
        headdim=headdim,
        d_state=d_state,
        d_conv=4,
        D_has_hdim=D_has_hdim,
        ngroups=1,
        rmsnorm=False,
        use_mem_eff_path=False,
        device=device,
        dtype=dtype,
    )
    model.eval()

    # Randomize D so non-uniform values expose the bug
    with torch.no_grad():
        model.D.copy_(torch.randn_like(model.D))

    x = torch.randn(batch, seqlen, d_model, device=device, dtype=dtype)

    # Forward pass — reference output
    with torch.no_grad():
        out_forward = model(x)

    # Step pass — one token at a time
    conv_state, ssm_state = model.allocate_inference_cache(batch, seqlen, dtype=dtype)
    step_outputs = []
    with torch.no_grad():
        for t in range(seqlen):
            out_t, conv_state, ssm_state = model.step(
                x[:, t : t + 1, :], conv_state, ssm_state
            )
            step_outputs.append(out_t)
    out_step = torch.cat(step_outputs, dim=1)

    # After conv warmup (d_conv - 1 = 3 tokens), outputs should match
    d_conv = model.d_conv
    out_fwd_tail = out_forward[:, d_conv - 1 :, :]
    out_step_tail = out_step[:, d_conv - 1 :, :]

    max_diff = (out_fwd_tail - out_step_tail).abs().max().item()
    print(f"D_has_hdim={D_has_hdim}, max diff after conv warmup: {max_diff:.2e}")
    assert torch.allclose(out_fwd_tail, out_step_tail, rtol=1e-3, atol=1e-3), (
        f"forward/step mismatch with D_has_hdim={D_has_hdim}: max diff {max_diff:.2e}"
    )
