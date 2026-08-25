import torch
import torch.nn.functional as F

import pytest

from mamba_ssm.ops.triton.layer_norm import (
    layer_norm_fn,
    layer_norm_ref,
    rms_norm_ref,
    _layer_norm_fwd,
    _layer_norm_bwd,
)


@pytest.mark.parametrize("has_x1", [False, True])
# @pytest.mark.parametrize("has_x1", [False])
@pytest.mark.parametrize("has_residual", [False, True])
# @pytest.mark.parametrize("has_residual", [False])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize("has_bias", [False])
@pytest.mark.parametrize("is_rms_norm", [False, True])
# @pytest.mark.parametrize("is_rms_norm", [False])
@pytest.mark.parametrize("dropout_p", [0.0, 0.3])
# @pytest.mark.parametrize("dropout_p", [0.3])
@pytest.mark.parametrize("has_rowscale", [False, True])
# @pytest.mark.parametrize("has_rowscale", [True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
# @pytest.mark.parametrize('dtype', [torch.float32])
def test_layer_norm(dtype, has_rowscale, dropout_p, is_rms_norm, has_bias, has_residual, has_x1):
    device = "cuda"
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
    # The kernel accumulates the weight gradients in fp32 while the reference
    # computes them in the input dtype, so allow some slack for fp16.
    rtol_grad, atol_grad = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 2e-2)
    # set seed
    torch.random.manual_seed(0)
    batch = 4
    seqlen = 32
    d = 256
    x = torch.randn(batch, seqlen, d, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(d, dtype=dtype, device=device, requires_grad=True)
    if has_bias:
        bias = torch.randn(d, dtype=dtype, device=device, requires_grad=True)
    else:
        bias = None
    if has_residual:
        residual = torch.randn(batch, seqlen, d, dtype=dtype, device=device, requires_grad=True)
    else:
        residual = None
    if has_x1:
        x1 = torch.randn(batch, seqlen, d, dtype=dtype, device=device, requires_grad=True)
    else:
        x1 = None
    if has_rowscale:
        # With parallel LayerNorm (x1 is not None) rowscale has 2 * rows
        # entries: the first rows for x and the last rows for x1.
        rowscale = torch.rand(
            2 * batch * seqlen if has_x1 else batch * seqlen, dtype=dtype, device=device
        )
    else:
        rowscale = None
    out, dropout_mask, dropout_mask1 = layer_norm_fn(
        x,
        weight,
        bias,
        residual=residual,
        x1=x1,
        eps=1e-5,
        dropout_p=dropout_p,
        rowscale=rowscale,
        is_rms_norm=is_rms_norm,
        return_dropout_mask=True,
    )
    # Reference computation that reuses the exact dropout mask from the
    # forward pass, so that both the output and the gradients are deterministic
    # and directly comparable.
    ref = rms_norm_ref if is_rms_norm else layer_norm_ref
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if has_bias else None
    residual_ref = residual.detach().clone().requires_grad_() if has_residual else None
    x1_ref = x1.detach().clone().requires_grad_() if has_x1 else None
    out_ref = ref(
        x_ref,
        weight_ref,
        bias_ref,
        residual_ref,
        x1_ref,
        None,
        None,
        1e-5,
        dropout_p=dropout_p,
        rowscale=rowscale.detach().clone() if has_rowscale else None,
        dropout_mask=dropout_mask,
        dropout_mask1=dropout_mask1,
        upcast=dtype == torch.float16,
    )
    torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)

    dy = torch.randn_like(out)
    inputs = [t for t in (x, weight, bias, residual, x1) if t is not None]
    inputs_ref = [t for t in (x_ref, weight_ref, bias_ref, residual_ref, x1_ref) if t is not None]
    grads = torch.autograd.grad(out, inputs, grad_outputs=dy)
    grads_ref = torch.autograd.grad(out_ref, inputs_ref, grad_outputs=dy)
    for grad, grad_ref, input_ in zip(grads, grads_ref, inputs):
        # The gradient of the dropout'd path is exactly zero where the dropout
        # mask is False, so compare only the entries that are actually trained.
        assert grad.dtype == input_.dtype
        torch.testing.assert_close(grad, grad_ref, rtol=rtol_grad, atol=atol_grad)


@pytest.mark.parametrize("dropout_p", [0.0, 0.3])
# @pytest.mark.parametrize("dropout_p", [0.3])
def test_layer_norm_prenorm(dropout_p):
    # prenorm=True exercises the dresidual path: the gradient from the
    # upstream of the residual connection is added inside the backward kernel.
    device = "cuda"
    torch.random.manual_seed(0)
    batch, seqlen, d = 4, 32, 256
    x = torch.randn(batch, seqlen, d, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(d, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(d, device=device, dtype=torch.float32, requires_grad=True)
    residual = torch.randn(batch, seqlen, d, device=device, dtype=torch.float32, requires_grad=True)
    out, residual_out, dropout_mask, _ = layer_norm_fn(
        x, weight, bias, residual=residual, eps=1e-5, dropout_p=dropout_p, prenorm=True,
        return_dropout_mask=True,
    )
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_()
    residual_ref = residual.detach().clone().requires_grad_()
    out_ref, residual_out_ref = layer_norm_ref(
        x_ref, weight_ref, bias_ref, residual_ref, eps=1e-5, dropout_p=dropout_p,
        prenorm=True, dropout_mask=dropout_mask,
    )
    dy = torch.randn_like(out)
    dy_res = torch.randn_like(residual_out)
    torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(residual_out, residual_out_ref, rtol=1e-4, atol=1e-4)
    dx, dw, db, dresidual = torch.autograd.grad(
        (out, residual_out), (x, weight, bias, residual), grad_outputs=(dy, dy_res)
    )
    dx_ref, dw_ref, db_ref, dresidual_ref = torch.autograd.grad(
        (out_ref, residual_out_ref), (x_ref, weight_ref, bias_ref, residual_ref),
        grad_outputs=(dy, dy_res),
    )
    torch.testing.assert_close(dx, dx_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dw, dw_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(db, db_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dresidual, dresidual_ref, rtol=1e-4, atol=1e-4)


def test_layer_norm_parallel():
    # Parallel LayerNorm: x1 gets its own norm output (y1) and its own norm
    # weights, which exercises the DY1/DW1/DB1 branches of the backward kernel.
    device = "cuda"
    torch.random.manual_seed(0)
    batch, seqlen, d = 4, 32, 256
    dropout_p = 0.3
    x = torch.randn(batch, seqlen, d, device=device, dtype=torch.float32, requires_grad=True)
    x1 = torch.randn(batch, seqlen, d, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(d, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(d, device=device, dtype=torch.float32, requires_grad=True)
    weight1 = torch.randn(d, device=device, dtype=torch.float32, requires_grad=True)
    bias1 = torch.randn(d, device=device, dtype=torch.float32, requires_grad=True)
    out, out1, dropout_mask, dropout_mask1 = layer_norm_fn(
        x, weight, bias, x1=x1, weight1=weight1, bias1=bias1, eps=1e-5,
        dropout_p=dropout_p, return_dropout_mask=True,
    )
    x_ref = x.detach().clone().requires_grad_()
    x1_ref = x1.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_()
    weight1_ref = weight1.detach().clone().requires_grad_()
    bias1_ref = bias1.detach().clone().requires_grad_()
    out_ref, out1_ref = layer_norm_ref(
        x_ref, weight_ref, bias_ref, None, x1_ref, weight1_ref, bias1_ref, 1e-5,
        dropout_p=dropout_p, dropout_mask=dropout_mask, dropout_mask1=dropout_mask1,
    )
    dy = torch.randn_like(out)
    dy1 = torch.randn_like(out1)
    torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out1, out1_ref, rtol=1e-4, atol=1e-4)
    grads = torch.autograd.grad(
        (out, out1), (x, x1, weight, bias, weight1, bias1), grad_outputs=(dy, dy1)
    )
    grads_ref = torch.autograd.grad(
        (out_ref, out1_ref), (x_ref, x1_ref, weight_ref, bias_ref, weight1_ref, bias1_ref),
        grad_outputs=(dy, dy1),
    )
    for grad, grad_ref in zip(grads, grads_ref):
        torch.testing.assert_close(grad, grad_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("has_x1", [False, True])
# @pytest.mark.parametrize("has_x1", [True])
def test_bwd_reuses_fwd_dropout_mask(has_x1):
    # Regression test: the backward kernel must reuse the dropout mask stored
    # in the forward pass, not regenerate its own from the seeds. We hand the
    # backward pass a mask that differs from the one the forward pass applied
    # (its complement): the backward gradients must follow the given mask. If
    # the kernel ignored the stored mask and regenerated one from the seeds,
    # the gradient pattern would follow the original mask and this test fails.
    device = "cuda"
    torch.random.manual_seed(0)
    M, N = 32, 256
    dropout_p = 0.3
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    x1 = torch.randn(M, N, device=device, dtype=torch.float32) if has_x1 else None
    weight = torch.randn(N, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)
    y, y1, mean, rstd, _, _, dropout_mask = _layer_norm_fwd(
        x, weight, bias, 1e-5, x1=x1, dropout_p=dropout_p
    )
    # The forward always stores the mask when dropout is enabled.
    assert dropout_mask is not None
    assert dropout_mask.shape == (M if not has_x1 else 2 * M, N)
    fake_mask = ~dropout_mask
    dy = torch.randn_like(y)
    dx, _, _, _, dx1, _, _ = _layer_norm_bwd(
        dy, x, weight, bias, 1e-5, mean, rstd, dropout_mask=fake_mask,
        dropout_p=dropout_p, has_x1=has_x1,
    )
    assert torch.equal(dx != 0, fake_mask[:M])
    if has_x1:
        assert torch.equal(dx1 != 0, fake_mask[M:])


def test_rowscale_x1_backward():
    # Regression test: with parallel LayerNorm, the second half of rowscale
    # applies to x1, so dx1 must be scaled by it (it used to be written out
    # unscaled, and the combination was rejected by the forward asserts).
    device = "cuda"
    torch.random.manual_seed(0)
    M, N = 32, 256
    x = torch.randn(M, N, device=device, dtype=torch.float32, requires_grad=True)
    x1 = torch.randn(M, N, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
    rowscale = torch.rand(2 * M, device=device, dtype=torch.float32)
    out = layer_norm_fn(x, weight, None, x1=x1, eps=1e-5, dropout_p=0.0, rowscale=rowscale)
    dy = torch.randn_like(out)
    dx, dx1 = torch.autograd.grad(out, (x, x1), grad_outputs=dy)
    # Closed-form reference: sum = rowscale[:M] * x + rowscale[M:] * x1, then
    # layer norm; d(x) = rowscale[:M] * d(sum), d(x1) = rowscale[M:] * d(sum).
    sum_in = x.detach() * rowscale[:M][:, None] + x1.detach() * rowscale[M:][:, None]
    sum_in.requires_grad_()
    out_ref = F.layer_norm(sum_in, (N,), weight.detach(), None, 1e-5)
    dsum = torch.autograd.grad(out_ref, sum_in, grad_outputs=dy)[0]
    torch.testing.assert_close(dx, dsum * rowscale[:M][:, None], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dx1, dsum * rowscale[M:][:, None], rtol=1e-4, atol=1e-4)


def test_dropout_default_no_mask_return():
    # The default call (return_dropout_mask=False) must keep working: the
    # backward pass still needs the mask internally, so it is stored and
    # reused even though it is not returned to the caller.
    device = "cuda"
    M, N = 32, 256
    torch.random.manual_seed(0)
    x = torch.randn(M, N, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
    out = layer_norm_fn(x, weight, None, eps=1e-5, dropout_p=0.3)
    dy = torch.randn_like(out)
    dx, dw = torch.autograd.grad(out, (x, weight), grad_outputs=dy)
    # Re-run with the same seed (same input, same dropout mask) and compare
    # the gradients: the backward pass must reproduce the exact same result
    # even though the mask was never returned to the caller.
    torch.random.manual_seed(0)
    x2 = torch.randn(M, N, device=device, dtype=torch.float32, requires_grad=True)
    weight2 = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
    out2, mask, _ = layer_norm_fn(x2, weight2, None, eps=1e-5, dropout_p=0.3,
                                  return_dropout_mask=True)
    dx2, dw2 = torch.autograd.grad(out2, (x2, weight2), grad_outputs=dy)
    torch.testing.assert_close(dx, dx2, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dw, dw2, rtol=1e-4, atol=1e-4)
    # The gradient is exactly zero where the dropout mask dropped the input.
    assert torch.equal(dx != 0, mask)
