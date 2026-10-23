import torch
import torch_npu
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, ssd_chunk_scan_combined_ref

torch.manual_seed(1)
device = "npu"
dtype = torch.float32
b, s, h, p, n = 2, 64, 2, 32, 16

x = (torch.randn(b, s, h, p, device=device, dtype=dtype) * 0.5).requires_grad_()
dt0 = torch.rand(b, s, h, device=device, dtype=dtype) * 0.1 + 0.01
A0 = -torch.rand(h, device=device, dtype=dtype) - 0.5
B0 = (torch.randn(b, s, 1, n, device=device, dtype=dtype) * 0.5).requires_grad_()
C0 = (torch.randn(b, s, 1, n, device=device, dtype=dtype) * 0.5).requires_grad_()
D0 = torch.randn(h, device=device, dtype=dtype)
dt_bias0 = torch.rand(h, device=device, dtype=dtype) * 0.1 - 0.05

def run(xx, BB, CC):
    return mamba_chunk_scan_combined(xx, dt0, A0, BB, CC, 16, D=D0, dt_bias=dt_bias0, dt_softplus=True)[0]

out = run(x, B0, C0)
loss = (out.float() ** 2).sum()
loss.backward()
torch.npu.synchronize()
print("backward OK, x.grad norm:", x.grad.norm().item())
print("B.grad norm:", B0.grad.norm().item(), "C.grad norm:", C0.grad.norm().item())
