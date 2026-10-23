# Mamba2 module-level acceptance on Ascend NPU (forward + full backward).
# Mirrors the official README Mamba-2 usage example with the NPU constraints:
#   - use_mem_eff_path=False (the default fused path requires CUDA-only causal_conv1d)
#   - d_model=16 / headdim=32 so d_ssm is divisible by headdim
# Verified on 910B3: forward loss = 0.3435635566711426, backward OK.
import os
import torch
import torch_npu
from mamba_ssm.modules.mamba2 import Mamba2

batch, length, dim = 2, 64, 16
torch.manual_seed(0)
x = torch.randn(batch, length, dim).to("npu:0").requires_grad_()
model = Mamba2(d_model=dim, d_state=64, d_conv=4, expand=2,
               headdim=32, use_mem_eff_path=False).to("npu:0")
y = model(x)
loss = y.square().mean()
print("forward OK, loss =", loss.item(), flush=True)

loss.backward()
print("backward OK", flush=True)
print("x.grad norm:", x.grad.norm().item(), flush=True)

if os.environ.get("EXPORT_TRACE"):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
        model(x).square().mean().backward()
    prof.export_chrome_trace(os.path.join(os.getcwd(), "trace_mamba2_bwd.json"))
    print("trace exported", flush=True)
