import torch
import torch_npu
import torch.nn.functional as F
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_cumsum_bwd

# _chunk_cumsum 前向+反向单算子校验
# 前向：dt_out = softplus(dt + dt_bias)（fp32），dA_cumsum = 块内 cumsum(dt_out * A)
# 反向语义（读 kernel 源码确认）：ddA 是"对块内逐位置乘积 r = dt_out * A"的余向量（非对 cumsum 结果），
#   ddt = (ddA * A + ddt_out) * dsoftplus(dt + bias)；dA = sum_l ddA[l] * dt_out[l]；ddt_bias = sum_l ddt[l]
# 参考：r / dt_out 为中间量，autograd 对叶子 dt / A / dt_bias 求梯度
# 另：dA / ddt_bias 依赖 autotune pre_hook init_to_zero 清零缓冲区（非确定性路径 atomic_add），
#   连跑两次同输入对比，检查是否存在未清零/不稳定（E9 排查协议）
torch.manual_seed(0)
device = "npu"
dtype = torch.float32
batch, seqlen, nheads, chunk = 2, 64, 4, 32
nc = seqlen // chunk

dt = torch.randn(batch, seqlen, nheads, device=device, dtype=dtype) * 0.5
A = -torch.exp(torch.randn(nheads, device=device, dtype=dtype))
dt_bias = torch.randn(nheads, device=device, dtype=dtype)

# ---- 前向 ----
dA_cs, dt_out = _chunk_cumsum_fwd(dt, A, chunk, dt_bias=dt_bias, dt_softplus=True)
torch.npu.synchronize()

dt_full = F.softplus(dt + dt_bias)
dt_full_c = dt_full.reshape(batch, nc, chunk, nheads).permute(0, 3, 1, 2)
dA_cs_ref = torch.cumsum(dt_full_c * A.view(1, nheads, 1, 1), dim=-1)

d1 = (dt_out - dt_full_c).abs().max().item()
d2 = (dA_cs - dA_cs_ref).abs().max().item()
print("fwd dt_out max abs diff:", d1)
print("fwd dA_cs  max abs diff:", d2, "（量级 ~50，rel ~%.1e）" % (d2 / dA_cs_ref.abs().max().item()))

# ---- 反向参考（r = dt_full * A 为中间量）----
dt_l = dt.clone().requires_grad_(True)
A_l = A.clone().requires_grad_(True)
bias_l = dt_bias.clone().requires_grad_(True)
dt_full_l = F.softplus(dt_l + bias_l)
dt_full_lc = dt_full_l.reshape(batch, nc, chunk, nheads).permute(0, 3, 1, 2)
r_l = dt_full_lc * A_l.view(1, nheads, 1, 1)
g_r = torch.randn_like(r_l)       # dL/dr（逐位置乘积的余向量）
g_out = torch.randn_like(dt_full_lc)  # dL/dt_out
g_dt, g_A, g_bias = torch.autograd.grad([r_l, dt_full_lc], [dt_l, A_l, bias_l],
                                        grad_outputs=[g_r, g_out])

# ---- 反向跑两遍：同输入稳定性（init_to_zero / atomic 清零检查）----
outs = []
for trial in range(2):
    ddt, dA, ddt_bias = _chunk_cumsum_bwd(g_r, g_out, dt, A, dt_bias=dt_bias, dt_softplus=True)
    torch.npu.synchronize()
    outs.append((ddt.clone(), dA.clone(), ddt_bias.clone()))

for i, (ddt, dA, ddt_bias) in enumerate(outs):
    print(f"[run{i}] bwd ddt      max abs diff:", (ddt - g_dt).abs().max().item())
    print(f"[run{i}] bwd dA       max abs diff:", (dA - g_A).abs().max().item(),
          "（量级 ~%.1f，rel ~%.1e）" % (g_A.abs().max().item(), ((dA - g_A).abs().max() / g_A.abs().max()).item()))
    print(f"[run{i}] bwd ddt_bias max abs diff:", (ddt_bias - g_bias).abs().max().item())

ddA_stab = max((outs[0][j] - outs[1][j]).abs().max().item() for j in range(3))
print("同输入两次运行最大互差:", ddA_stab)

ok = (d1 < 1e-4 and d2 < 1e-3
      and (outs[0][0] - g_dt).abs().max() < 1e-2
      and ((outs[0][1] - g_A).abs().max() / g_A.abs().max()) < 1e-3
      and (outs[0][2] - g_bias).abs().max() < 1e-2
      and ddA_stab < 1e-4)
print("PASS" if ok else "MISMATCH")
