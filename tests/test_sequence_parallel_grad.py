import importlib.util
from pathlib import Path

import pytest

import torch
from torch import nn

spec = importlib.util.spec_from_file_location(
    "distributed_grad",
    Path(__file__).parents[1] / "mamba_ssm/distributed/distributed_utils.py",
)
distributed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(distributed)


def test_sequence_parallel_reduction_skips_parameters_without_gradients(monkeypatch):
    model = nn.Linear(2, 2)
    model.weight._sequence_parallel = True
    model.bias._sequence_parallel = True
    model.weight.grad = torch.ones_like(model.weight)
    calls = []

    def reduce(tensor, group):
        calls.append(tensor.clone())
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    distributed.allreduce_sequence_parallel_grad(model, object())
    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 2))
    assert model.bias.grad is None
    assert len(calls) == 1


def test_sequence_parallel_reduction_is_noop_without_gradients(monkeypatch):
    model = nn.Linear(2, 2)
    model.weight._sequence_parallel = True
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *a, **kw: pytest.fail("unexpected collective"),
    )
    distributed.allreduce_sequence_parallel_grad(model, object())


def test_sequence_parallel_reduction_preserves_sorted_gradient_layout(monkeypatch):
    model = nn.Linear(2, 2)
    for p in model.parameters():
        p._sequence_parallel = True
        p.grad = torch.ones_like(p)
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, group: tensor.mul_(3)
    )
    distributed.allreduce_sequence_parallel_grad(model, object())
    for p in model.parameters():
        torch.testing.assert_close(p.grad, torch.full_like(p, 3))
