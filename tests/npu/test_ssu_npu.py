import torch
import torch_npu
from mamba_ssm.ops.triton.selective_state_update import selective_state_update, selective_state_update_ref

torch.manual_seed(2)
device = "npu"
dtype = torch.float32
b, h, p, n = 2, 2, 32, 16

state0 = torch.randn(b, h, p, n, device=device, dtype=dtype) * 0.5
x = torch.randn(b, h, p, device=device, dtype=dtype) * 0.5
dt = torch.rand(b, h, p, device=device, dtype=dtype) * 0.1 + 0.01
A = -torch.rand(h, p, n, device=device, dtype=dtype) - 0.5
B = torch.randn(b, 1, n, device=device, dtype=dtype) * 0.5
C = torch.randn(b, 1, n, device=device, dtype=dtype) * 0.5
D = torch.randn(h, p, device=device, dtype=dtype)
dt_bias = torch.rand(h, p, device=device, dtype=dtype) * 0.1 - 0.05

out = selective_state_update(state0.clone(), x, dt, A, B, C, D=D, dt_bias=dt_bias, dt_softplus=True)
ref = selective_state_update_ref(state0, x, dt, A, B, C, D=D, dt_bias=dt_bias, dt_softplus=True)
torch.npu.synchronize()
diff = (out - ref).abs().max().item()
print("max abs diff:", diff, "| ref absmax:", ref.abs().max().item())
print("PASS" if diff / ref.abs().max().item() < 1e-3 else "MISMATCH")
