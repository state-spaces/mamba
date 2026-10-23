import torch
import torch_npu
from mamba_ssm.ops.triton.layernorm_gated import layernorm_fn

torch.manual_seed(0)
device = "npu"
dtype = torch.float32
batch, seqlen, hidden = 2, 32, 64

x = torch.randn(batch, seqlen, hidden, device=device, dtype=dtype)
weight = torch.randn(hidden, device=device, dtype=dtype)
bias = torch.randn(hidden, device=device, dtype=dtype)
eps = 1e-5

out = layernorm_fn(x, weight, bias, eps=eps, group_size=None, norm_before_gate=False, is_rms_norm=False)
torch.npu.synchronize()

mean = x.mean(-1, keepdim=True)
var = x.var(-1, unbiased=False, keepdim=True)
ref = (x - mean) / torch.sqrt(var + eps) * weight + bias

diff = (out - ref).abs().max().item()
print("max abs diff:", diff)
print("PASS" if diff < 1e-4 else "MISMATCH")
