import torch
import torch_npu
from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_bwd

# _bmm_chunk 前向+反向单算子校验
# 前向：按块分段的矩阵乘 out = a @ b^T（按块内位置 l,s，合约 k 维），带 group 维
# 反向两种调用形态（复刻组合层 488/489 行）：
#   形态1 _bmm_chunk_bwd(C, dCB)                -> grad_B
#   形态2 _bmm_chunk_bwd(B, dCB^T, residual=r)  -> grad_C + residual
torch.manual_seed(0)
device = "npu"
dtype = torch.float32
batch, seqlen, ngroups, k, chunk = 2, 64, 2, 32, 32
nc = seqlen // chunk

a = torch.randn(batch, seqlen, ngroups, k, device=device, dtype=dtype)   # 相当于 C
b = torch.randn(batch, seqlen, ngroups, k, device=device, dtype=dtype)   # 相当于 B

# ---- 前向 ----
CB = _bmm_chunk_fwd(a, b, chunk)   # (b, c, g, cl, cl)
torch.npu.synchronize()
a_c = a.reshape(batch, nc, chunk, ngroups, k)
b_c = b.reshape(batch, nc, chunk, ngroups, k)
CB_ref = torch.einsum("bclgk,bcsgk->bcgls", a_c, b_c)
print("fwd CB max abs diff:", (CB - CB_ref).abs().max().item())

# ---- 反向 ----
dCB = torch.randn_like(CB)
a_l = a.clone().requires_grad_(True)
b_l = b.clone().requires_grad_(True)
CB_l = torch.einsum("bclgk,bcsgk->bcgls", a_l.reshape(batch, nc, chunk, ngroups, k),
                    b_l.reshape(batch, nc, chunk, ngroups, k))
g_a, g_b = torch.autograd.grad(CB_l, [a_l, b_l], grad_outputs=dCB)

# 形态1：grad_B = sum_l dCB[l,s] * C[l,n]
gB = _bmm_chunk_bwd(a, dCB)
torch.npu.synchronize()
print("bwd grad_B max abs diff:", (gB - g_b).abs().max().item())

# 形态2：grad_C + residual，dCB 转置传入（复刻组合层 489 行）
residual = torch.randn(batch, seqlen, ngroups, k, device=device, dtype=dtype)
gC = _bmm_chunk_bwd(b, dCB.transpose(-1, -2), residual=residual)
torch.npu.synchronize()
print("bwd grad_C+res max abs diff:", (gC - (g_a + residual)).abs().max().item())

ok = ((CB - CB_ref).abs().max() < 1e-3 and (gB - g_b).abs().max() < 1e-2
      and (gC - (g_a + residual)).abs().max() < 1e-2)
print("PASS" if ok else "MISMATCH")
