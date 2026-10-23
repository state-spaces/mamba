import torch
import torch_npu
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn

# 非门控 LayerNorm 单算子校验（layer_norm.py，mamba-1/通用 RMSNorm，适配时顺带修改的层）
# 触发 _layer_norm_fwd + _layer_norm_bwd；两种配置：LayerNorm、RMSNorm
torch.manual_seed(0)
device = "npu"
dtype = torch.float32
batch, seqlen, hidden = 2, 32, 64
eps = 1e-5


def check(is_rms_norm):
    tag = "RMSNorm" if is_rms_norm else "LayerNorm"
    x = torch.randn(batch, seqlen, hidden, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    dy = torch.randn(batch, seqlen, hidden, device=device, dtype=dtype)

    y = layer_norm_fn(x, w, b, eps=eps, is_rms_norm=is_rms_norm)
    torch.npu.synchronize()

    # 前向参考
    if is_rms_norm:
        xhat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    else:
        xhat = (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
    y_ref = xhat * w + b
    diff_y = (y - y_ref).abs().max().item()

    # 反向参考（torch.autograd）
    xr = x.detach().requires_grad_(True)
    wr = w.detach().requires_grad_(True)
    br = b.detach().requires_grad_(True)
    if is_rms_norm:
        xhat_r = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + eps)
    else:
        xhat_r = (xr - xr.mean(-1, keepdim=True)) * torch.rsqrt(xr.var(-1, unbiased=False, keepdim=True) + eps)
    y_r = xhat_r * wr + br
    y_r.backward(dy)
    y.backward(dy)
    torch.npu.synchronize()

    diffs = {
        "fwd_y": diff_y,
        "dx": (x.grad - xr.grad).abs().max().item(),
        "dw": (w.grad - wr.grad).abs().max().item(),
        "db": (b.grad - br.grad).abs().max().item(),
    }
    worst = max(diffs.values())
    for k, v in diffs.items():
        print(f"  [{tag}] {k} max abs diff: {v}")
    return worst


d1 = check(False)
print("LayerNorm bwd:", "PASS" if d1 < 1e-3 else "MISMATCH")
d2 = check(True)
print("RMSNorm bwd:", "PASS" if d2 < 1e-3 else "MISMATCH")
