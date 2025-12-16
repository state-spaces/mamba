import torch
import pytest

from mamba_ssm.models.mixer_seq_simple import MixerModel
from mamba_ssm.utils.generation import InferenceParams


def _make_mamba(d_model=32, n_layers=2, d_state=16, vocab_size=100, device="cuda", dtype=torch.float32):
    """Create a simple Mamba model for testing."""
    model = MixerModel(
        d_model=d_model,
        n_layer=n_layers,
        d_intermediate=0,  # No MLP for simplicity
        vocab_size=vocab_size,
        ssm_cfg=dict(layer="Mamba1"),
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model


def _empty_caches_for_model(model, batch_size, device, dtype):
    """Create empty inference caches for a model."""
    max_seqlen = 1024  # Large enough for tests
    caches = model.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
    # Initialize inference params
    inference_params = InferenceParams(
        max_seqlen=max_seqlen,
        max_batch_size=batch_size,
        seqlen_offset=0,
    )
    inference_params.key_value_memory_dict = caches
    return inference_params


@pytest.mark.parametrize("device", ["cuda"])
def test_one_forward_matches_two_state_continuity_forward(device):
    """Test that processing two chunks with state continuity matches full forward pass."""
    torch.manual_seed(0)
    B, L, D = 2, 30, 32
    model = _make_mamba(d_model=D, n_layers=2, d_state=16, device=device, dtype=torch.float32)
    
    vocab_size = 100
    x = torch.randint(0, vocab_size, (B, L), device=device, dtype=torch.long)
    L1 = L // 2
    
    with torch.no_grad():
        # Full forward pass
        gold = model(x)  # (B, L, D)
        
        # Process in two chunks with state continuity
        # Use seqlen_offset=0 to use parallel scan with initial_state support
        inference_params = _empty_caches_for_model(model, B, device, torch.float32)
        y1 = model(x[:, :L1], inference_params=inference_params)  # (B, L1, D)
        # State is updated in inference_params.key_value_memory_dict
        # Process second chunk with same inference_params (state continuity)
        y2 = model(x[:, L1:], inference_params=inference_params)  # (B, L-L1, D)
        got = torch.cat([y1, y2], dim=1)
    
    assert torch.allclose(got, gold, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda"])
def forward_matches_steps(device):
    """Test that processing a chunk matches sequential step-by-step processing."""
    torch.manual_seed(0)
    B, L, D = 2, 25, 32
    model = _make_mamba(d_model=D, n_layers=2, d_state=16, device=device, dtype=torch.float32)
    
    vocab_size = 100
    x = torch.randint(0, vocab_size, (B, L), device=device, dtype=torch.long)
    
    with torch.no_grad():
        # Sequential step-by-step processing using forward with seqlen_offset > 0
        # This triggers step() method internally
        inference_params_seq = _empty_caches_for_model(model, B, device, torch.float32)
        y_list = []
        for t in range(L):
            x_t = x[:, t:t+1]  # (B, 1)
            # seqlen_offset > 0 triggers step() method internally
            inference_params_seq.seqlen_offset = t
            y_t = model(x_t, inference_params=inference_params_seq)  # (B, 1, D)
            y_list.append(y_t)
        logits_step = torch.cat(y_list, dim=1)  # (B, L, D)
        
        # Process as single chunk using forward with seqlen_offset=0
        # This uses parallel scan with initial_state support
        inference_params_chunk = _empty_caches_for_model(model, B, device, torch.float32)
        logits_chunk = model(x, inference_params=inference_params_chunk)  # (B, L, D)
    
    assert torch.allclose(logits_chunk, logits_step, atol=1e-5, rtol=1e-5)
