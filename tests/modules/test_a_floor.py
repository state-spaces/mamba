import pytest
import torch

from mamba_ssm.modules.a_floor import apply_a_floor

DEFAULT_FLOOR = 1e-4


def test_a_floor_clamps_near_zero():
    A = torch.tensor([-1e-8, -1e-5, -1e-4, -1.0])
    expected = torch.tensor([-1e-4, -1e-4, -1e-4, -1.0])
    assert torch.equal(apply_a_floor(A, DEFAULT_FLOOR), expected)


def test_a_floor_is_noop_when_already_below_floor():
    A = torch.tensor([-1e-3, -1.0])
    assert torch.equal(apply_a_floor(A, DEFAULT_FLOOR), A)


def test_a_floor_is_idempotent():
    A = torch.tensor([-1e-8, -1e-4, -1.0])
    once = apply_a_floor(A, DEFAULT_FLOOR)
    assert torch.equal(apply_a_floor(once, DEFAULT_FLOOR), once)


def test_a_floor_leaves_a_released_checkpoint_untouched():
    # mamba2-130m spans roughly -3.63e4 to -3.98e-4 across its 24 layers, so the
    # default floor only activates once a layer has drifted past it.
    A = torch.tensor([-3.63e4, -1.0, -1e-3, -3.98e-4])
    assert torch.equal(apply_a_floor(A, DEFAULT_FLOOR), A)


def test_a_floor_can_be_disabled():
    A = torch.tensor([-1e-8, -1.0])
    assert torch.equal(apply_a_floor(A, None), A)


def test_a_floor_disabled_returns_the_same_tensor():
    A = torch.tensor([-1e-8])
    assert apply_a_floor(A, None) is A


@pytest.mark.parametrize("A_floor", [0.0, -1e-4, float("nan"), float("inf")])
def test_a_floor_rejects_values_that_invert_the_clamp(A_floor):
    with pytest.raises(ValueError, match="positive finite"):
        apply_a_floor(torch.tensor([-1.0]), A_floor)
