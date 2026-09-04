import torch

from mamba_ssm.utils.generation import sample


def test_nonpositive_temperature_is_greedy():
    logits = torch.tensor([[0.0, 3.0, 1.0], [2.0, 0.0, 0.5]])
    assert sample(logits, top_k=0, temperature=0.0).tolist() == [1, 0]
    assert sample(logits, top_k=5, temperature=-1.0).tolist() == [1, 0]
