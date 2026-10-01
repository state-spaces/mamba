import pytest
import torch

from mamba_ssm.utils.generation import modify_logits_for_top_p_filtering, sample


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("top_p", [1e-8, 1e-4, 0.2])
def test_top_p_keeps_most_likely_token(dtype, top_p):
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=dtype)
    filtered = logits.clone()
    modify_logits_for_top_p_filtering(filtered, top_p)
    assert torch.isfinite(filtered).sum(dim=-1).tolist() == [1, 1]
    torch.testing.assert_close(filtered.argmax(dim=-1), logits.argmax(dim=-1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("top_k", [0, 2])
def test_sample_with_tiny_top_p(dtype, top_k):
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=dtype)
    original = logits.clone()
    actual = sample(logits, top_k=top_k, top_p=1e-8)
    torch.testing.assert_close(actual, logits.argmax(dim=-1))
    torch.testing.assert_close(logits, original)


@pytest.mark.parametrize("top_p", [0.0, 1.0])
def test_top_p_filtering_disabled(top_p):
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    original = logits.clone()
    modify_logits_for_top_p_filtering(logits, top_p)
    torch.testing.assert_close(logits, original)


def test_top_p_retains_the_nucleus():
    logits = torch.tensor([[0.1, 0.3, 0.6]]).log()
    modify_logits_for_top_p_filtering(logits, 0.8)
    assert torch.isfinite(logits).tolist() == [[False, True, True]]
