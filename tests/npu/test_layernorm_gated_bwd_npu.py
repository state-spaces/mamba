import torch
import torch_npu
from mamba_ssm.ops.triton.layernorm_gated import layernorm_fn, rmsnorm_fn

# 门控 LayerNorm 反向单算子校验：触发 _layer_norm_bwd（附 z 门控 z*sigmoid(z)）
# 前向输出与 torch 参考比对，反向 x/z/w/b 的梯度与 torch.autograd 参考比对
# 两种配置：LayerNorm+gate（is_rms_norm=False）、RMSNorm+gate（mamba2 实际用法，is_rms_norm=True）
torch.manual_seed(0)
device = "npu"
dtype = torch.float32
batch, seqlen, hidden = 2, 32, 64
eps = 1e-5


def gate(z):
    return z * torch.sigmoid(z)


def check(is_rms_norm):
    tag = "RMSNorm+gate" if is_rms_norm else "LayerNorm+gate"
    x = torch.randn(batch, seqlen, hidden, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    z = torch.randn(batch, seqlen, hidden, device=device, dtype=dtype, requires_grad=True)
    dy = torch.randn(batch, seqlen, hidden, device=device, dtype=dtype)

    fn = rmsnorm_fn if is_rms_norm else layernorm_fn
    y = fn(x, w, b, z=z, eps=eps, norm_before_gate=True)
    torch.npu.synchronize()

    # 前向参考
    if is_rms_norm:
        xhat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    else:
        xhat = (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
    y_ref = (xhat * w + b) * gate(z)
    diff_y = (y - y_ref).abs().max().item()

    # 反向参考（torch.autograd）
    xr = x.detach().requires_grad_(True)
    wr = w.detach().requires_grad_(True)
    br = b.detach().requires_grad_(True)
    zr = z.detach().requires_grad_(True)
    if is_rms_norm:
        xhat_r = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + eps)
    else:
        xhat_r = (xr - xr.mean(-1, keepdim=True)) * torch.rsqrt(xr.var(-1, unbiased=False, keepdim=True) + eps)
    y_r = (xhat_r * wr + br) * gate(zr)
    y_r.backward(dy)
    y.backward(dy)
    torch.npu.synchronize()

    diffs = {
        "fwd_y": diff_y,
        "dx": (x.grad - xr.grad).abs().max().item(),
        "dz": (z.grad - zr.grad).abs().max().item(),
        "dw": (w.grad - wr.grad).abs().max().item(),
        "db": (b.grad - br.grad).abs().max().item(),
    }
    worst = max(diffs.values())
    for k, v in diffs.items():
        print(f"  [{tag}] {k} max abs diff: {v}")
    return worst


d1 = check(False)
print("LayerNorm+gate bwd:", "PASS" if d1 < 1e-3 else "MISMATCH")
d2 = check(True)
print("RMSNorm+gate bwd:", "PASS" if d2 < 1e-3 else "MISMATCH")
