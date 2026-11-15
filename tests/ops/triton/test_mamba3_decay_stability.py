"""Regression checks for reverse-decay roundoff at large negative magnitudes."""
import pytest
import torch

from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import compute_dqkv
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined


@pytest.mark.parametrize("roundoff", [0.0, 128.0])
@pytest.mark.parametrize("headdim", [64, 128])
def test_dqkv_zero_upstream_gradient_with_rounded_decay(roundoff, headdim):
    # Both stored decay tensors are finite. An independent FP32 total reduction
    # can differ from the last prefix by an ulp-scale positive residual.
    shape = (1, 64, 1, headdim)
    value = torch.ones(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.ones((1, 64, 1, 64), device="cuda", dtype=torch.bfloat16)
    prefix = -562400768.0
    result = compute_dqkv(
        q=value, k=value, v=v,
        da_cs=torch.full((1, 1, 64), prefix, device="cuda"),
        da_cs_sum=torch.full((1, 1, 1), prefix + roundoff, device="cuda"),
        qk_dot=torch.zeros((1, 1, 64), device="cuda"),
        SSM_States=torch.zeros((1, 1, 64, headdim), device="cuda", dtype=torch.bfloat16),
        do=torch.zeros_like(v), D=torch.ones(1, device="cuda"),
        chunk_size=64, has_input_state=False,
    )
    # A zero output adjoint must produce zero gradients, independently of the
    # decay magnitude. In particular, zero state gradients must not become 0*Inf.
    for gradient in result:
        if gradient is not None:
            assert torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("seqlen,return_states", [(64, False), (64, True), (128, False)])
def test_siso_large_negative_decay_zero_adjoint(seqlen, return_states):
    # Generate valid nonpositive ADT rather than supplying cached intermediates.
    # The large dynamic range exposes differing FP32 reduction/scan roundoff.
    torch.manual_seed(42)
    batch, heads, dim = 16, 8, 64
    shape = (batch, seqlen, heads, dim)
    q = torch.full(shape, 0.5, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.full(shape, 0.25, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.full(shape, 0.125, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    adt = -(torch.rand(batch, heads, seqlen, device="cuda") + 0.5) * 1.0e8
    adt.requires_grad_()
    dt = torch.ones_like(adt, requires_grad=True)
    trap = torch.zeros_like(adt, dtype=torch.bfloat16, requires_grad=True)
    q_bias = torch.zeros(heads, dim, device="cuda", requires_grad=True)
    k_bias = torch.zeros_like(q_bias, requires_grad=True)
    angles = torch.zeros(batch, seqlen, heads, dim // 4, device="cuda", requires_grad=True)
    skip = torch.ones(heads, device="cuda", requires_grad=True)
    inputs = (q, k, v, adt, dt, trap, q_bias, k_bias, angles, skip)
    assert all(torch.isfinite(t).all() for t in inputs)
    assert (adt <= 0).all()
    result = mamba3_siso_combined(
        q, k, v, adt, dt, trap, q_bias, k_bias, angles,
        D=skip, chunk_size=64, return_final_states=return_states,
    )
    outputs = result if return_states else (result,)
    assert all(torch.isfinite(t).all() for t in outputs)
    # Off-diagonal decays underflow to zero, so each output is exactly
    # V * (DT * sigmoid(Trap) * dot(Q, K) + D) = 0.125 * (4 + 1).
    torch.testing.assert_close(outputs[0], torch.full_like(outputs[0], 0.625), rtol=0, atol=0)
    torch.autograd.backward(outputs, [torch.zeros_like(t) for t in outputs])
    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) == 0
