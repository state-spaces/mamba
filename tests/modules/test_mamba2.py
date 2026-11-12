import pytest
import torch

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.utils.generation import InferenceParams


@pytest.mark.parametrize("dt_limit", [(0.0, float("inf")), (0.0, 0.01)])
def test_mamba2_step_matches_forward(dt_limit):
    """Decoding token by token with Mamba2.step must match one forward pass over the same tokens."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size, seqlen, prefill, d_model = 2, 64, 16, 256
    model = Mamba2(d_model=d_model, d_state=64, headdim=64, chunk_size=64, dt_limit=dt_limit,
                   use_mem_eff_path=False, layer_idx=0, device=device, dtype=torch.float32)
    u = torch.randn(batch_size, seqlen, d_model, device=device)
    with torch.no_grad():
        out_ref = model(u)
        inference_params = InferenceParams(max_seqlen=seqlen, max_batch_size=batch_size)
        outs = [model(u[:, :prefill], inference_params=inference_params)]
        inference_params.seqlen_offset += prefill
        for i in range(prefill, seqlen):
            outs.append(model(u[:, i:i + 1], inference_params=inference_params))
            inference_params.seqlen_offset += 1
    out = torch.cat(outs, dim=1)
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    assert torch.allclose(out, out_ref, rtol=1e-3, atol=1e-3)
