import math

import torch
import torch.nn.functional as F
import triton

import pytest

from einops import rearrange, repeat

from mamba_ssm.ops.triton.layernorm_gated import layernorm_fn, rms_norm_ref

from overflow_test_utils import assert_isfinite, bwd_row_start_max, gpu_memory_skipif


@pytest.mark.parametrize("norm_before_gate", [True, False])
# @pytest.mark.parametrize("norm_before_gate", [False])
@pytest.mark.parametrize("has_group", [False, True])
# @pytest.mark.parametrize("has_group", [False])
@pytest.mark.parametrize("is_rms_norm", [False, True])
# @pytest.mark.parametrize("is_rms_norm", [True])
@pytest.mark.parametrize("has_z", [False, True])
# @pytest.mark.parametrize("has_z", [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize("has_bias", [False])
# @pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('dtype', [torch.float16])
# @pytest.mark.parametrize("wtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wtype", [torch.float32])
@pytest.mark.parametrize('d', [2048, 4096])
# @pytest.mark.parametrize('d', [4096])
def test_layer_norm_gated(d, dtype, wtype, has_bias, has_z, is_rms_norm, has_group, norm_before_gate):
    if not has_z and not norm_before_gate:
        pytest.skip()
    if not norm_before_gate and not is_rms_norm:  # Reference LN isn't implemented for this case yet
        pytest.skip()
    device = 'cuda'
    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 8e-3)
    group_size = None if not has_group else 64
    # set seed
    torch.random.manual_seed(0)
    batch = 16
    seqlen = 1024
    x = torch.randn(batch, seqlen, d, dtype=dtype, device=device, requires_grad=True)
    if has_z:
        z = torch.randn(batch, seqlen, d, dtype=dtype, device=device, requires_grad=True)
    else:
        z = None
    weight = torch.randn(d, dtype=wtype, device=device, requires_grad=True)
    if has_bias:
        bias = torch.randn(d, dtype=wtype, device=device, requires_grad=True)
    else:
        bias = None
    x_ref = x.detach().clone().requires_grad_()
    x_pt = x.detach().clone().requires_grad_()
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    z_pt = z.detach().clone().requires_grad_() if z is not None else None
    weight_ref = weight.detach().clone().requires_grad_()
    weight_pt = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    bias_pt = bias.detach().clone().requires_grad_() if bias is not None else None
    out = layernorm_fn(x, weight, bias, z=z, eps=1e-5, group_size=group_size, norm_before_gate=norm_before_gate,
                       is_rms_norm=is_rms_norm)
    if not is_rms_norm:
        if not has_group:
            out_ref = F.layer_norm(x_ref.float(), (d,), weight=weight_ref.float(), bias=bias_ref.float() if bias_ref is not None else None, eps=1e-5)
            out_pt = F.layer_norm(x_pt.to(wtype), (d,), weight=weight_pt, bias=bias_pt, eps=1e-5)
        else:
            out_ref = rearrange(F.layer_norm(rearrange(x_ref, "... (g d) -> ... g d", d=group_size).float(), (group_size,), eps=1e-5), "... g d -> ... (g d)") * weight_ref.float()
            if has_bias:
                out_ref = out_ref + bias_ref.float()
            out_pt = rearrange(F.layer_norm(rearrange(x_pt, "... (g d) -> ... g d", d=group_size), (group_size,), eps=1e-5), "... g d -> ... (g d)") * weight_pt
            if has_bias:
                out_pt = out_pt + bias_pt
        if has_z and norm_before_gate:
            out_ref = out_ref * F.silu(z_ref.float())
            out_pt = out_pt * F.silu(z_pt)
    else:
        out_ref = rms_norm_ref(x_ref, weight_ref, bias_ref, z=z_ref, eps=1e-5, group_size=group_size,
                               norm_before_gate=norm_before_gate)
        out_pt = rms_norm_ref(x_pt, weight_pt, bias_pt, z=z_pt, eps=1e-5, group_size=group_size,
                              norm_before_gate=norm_before_gate, upcast=False)
    print(f"Max diff = {(out - out_ref).abs().max().item()}")
    print(f"Max diff Pytorch = {(out_pt - out_ref).abs().max().item()}")
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + atol

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    out_pt.backward(g)
    print(f"Max dx diff = {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"Max dx diff Pytorch = {(x_pt.grad - x_ref.grad).abs().max().item()}")
    if has_z:
        print(f"Max dz diff = {(z.grad - z_ref.grad).abs().max().item()}")
        print(f"Max dz diff Pytorch = {(z_pt.grad - z_ref.grad).abs().max().item()}")
    print(f"Max dw diff = {(weight.grad - weight_ref.grad).abs().max().item()}")
    print(f"Max dw diff Pytorch = {(weight_pt.grad - weight_ref.grad).abs().max().item()}")
    if has_bias:
        print(f"Max db diff = {(bias.grad - bias_ref.grad).abs().max().item()}")
        print(f"Max db diff Pytorch = {(bias_pt.grad - bias_ref.grad).abs().max().item()}")
    assert (x.grad - x_ref.grad).abs().max().item() <= 2 * (x_pt.grad - x_ref.grad).abs().max().item() + atol
    if has_z:
        assert (z.grad - z_ref.grad).abs().max().item() <= 2 * (z_pt.grad - z_ref.grad).abs().max().item() + atol
    assert (weight.grad - weight_ref.grad).abs().max().item() <= 2 * (weight_pt.grad - weight_ref.grad).abs().max().item() + atol
    if has_bias:
        assert (bias.grad - bias_ref.grad).abs().max().item() <= 2 * (bias_pt.grad - bias_ref.grad).abs().max().item() + atol


@gpu_memory_skipif(21)
def test_layer_norm_gated_large_row_count_no_overflow() -> None:
    # int64 to avoid int32 overflow, see https://github.com/state-spaces/mamba/issues/1015.
    device = 'cuda'
    torch.manual_seed(0)
    N = 32_768  # hard cap: layernorm_gated.py raises if group_size/N exceeds 64KB / dtype_size
    M = 80_000  # (nrow_groups - 1) * ceil(M / nrow_groups) * N still clears 2**31 - 1 with margin (asserted below)

    element_size = 2  # bfloat16
    max_fused_size = 65536 // element_size
    block_n = min(max_fused_size, triton.next_power_of_2(N))
    num_warps = min(max(block_n // 256, 1), 8)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    nrow_groups = math.ceil(sm_count * math.ceil(4 / num_warps))
    assert bwd_row_start_max(M, nrow_groups) * N > 2**31 - 1, (
        f"test parameters too small to overflow the backward kernel on this GPU "
        f"(sm_count={sm_count}, nrow_groups={nrow_groups}): increase N"
    )

    x = torch.randn(M, N, dtype=torch.bfloat16, device=device, requires_grad=True)
    weight = torch.randn(N, dtype=torch.float32, device=device, requires_grad=True)

    out = layernorm_fn(x, weight, bias=None, is_rms_norm=True)
    torch.cuda.synchronize(device)
    assert out.shape == (M, N)
    assert_isfinite(out)

    out.sum().backward()
    torch.cuda.synchronize(device)
    assert x.grad is not None
    assert_isfinite(x.grad)
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
