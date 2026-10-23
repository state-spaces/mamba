import torch
import torch_npu
from mamba_ssm.ops.triton.k_activations import _swiglu_fwd, _swiglu_bwd

# SwiGLU 单算子校验：out = x * sigmoid(x) * y，xy 最后一维为 2N，切成两半
# 前向对比 PyTorch 公式；反向对比 torch.autograd 参考梯度
torch.manual_seed(0)
device = "npu"
dtype = torch.float32
batch, seqlen, d = 2, 32, 64  # xy 最后一维 2d

xy = torch.randn(batch, seqlen, 2 * d, device=device, dtype=dtype)
dout = torch.randn(batch, seqlen, d, device=device, dtype=dtype)

# ---- 前向 ----
out = _swiglu_fwd(xy)
torch.npu.synchronize()
x, y = xy.chunk(2, dim=-1)
ref = x * torch.sigmoid(x) * y
diff_out = (out - ref).abs().max().item()
print("fwd max abs diff:", diff_out)

# ---- 反向（对比 torch.autograd 参考梯度）----
xy_ref = xy.detach().clone().requires_grad_(True)
xr, yr = xy_ref.chunk(2, dim=-1)
ref_b = xr * torch.sigmoid(xr) * yr
ref_b.backward(dout)
dxy = _swiglu_bwd(xy, dout)
torch.npu.synchronize()
dx, dy_got = dxy.chunk(2, dim=-1)
dx_ref, dy_ref = xy_ref.grad.chunk(2, dim=-1)
diff_dx = (dx - dx_ref).abs().max().item()
diff_dy = (dy_got - dy_ref).abs().max().item()
print("bwd dx max abs diff:", diff_dx)
print("bwd dy max abs diff:", diff_dy)

ok = diff_out < 1e-4 and diff_dx < 1e-3 and diff_dy < 1e-3
print("PASS" if ok else "MISMATCH")
