import torch
import torch_npu
import torch.nn.functional as F
from mamba_ssm.ops.triton.ssd_chunk_state import (
    _chunk_cumsum_fwd, _chunk_state_bwd_dx, _chunk_state_bwd_db, _chunk_state_bwd_ddAcs_stable,
)

# chunk_state 反向三 kernel 单算子校验
# 参考（chunk_state_ref 同款数学）：
#   states = sum_l decay[l] * dt[l] * x[l] ⊗ B[l]，decay = exp(cs[-1] - cs)
# 叶子：x / B / dt / dA_cumsum，随机余向量 dstates，autograd 得参考梯度
# 预期：_chunk_state_bwd_db 在 wrapper 层 TypeError（sm_count=None，E17 同款，ssd_chunk_state.py:937）
torch.manual_seed(0)
device = "npu"
dtype = torch.float32
batch, seqlen, nheads, headdim, ngroups, dstate, chunk = 2, 64, 4, 32, 2, 32, 32
nc = seqlen // chunk

x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
Bm = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype)
dt_raw = torch.randn(batch, seqlen, nheads, device=device, dtype=dtype) * 0.5
A = -torch.exp(torch.randn(nheads, device=device, dtype=dtype))
dt_bias = torch.randn(nheads, device=device, dtype=dtype)
cs, dt_f = _chunk_cumsum_fwd(dt_raw, A, chunk, dt_bias=dt_bias, dt_softplus=True)
torch.npu.synchronize()

ratio = nheads // ngroups


def states_ref(x_, B_, cs_, dtf_):
    Bg = B_.reshape(batch, nc, chunk, ngroups, dstate).repeat_interleave(ratio, dim=3)  # b c l h n
    decay = torch.exp(cs_[..., -1:] - cs_)                                              # b h c l
    return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", Bg, decay, dtf_, x_.reshape(batch, nc, chunk, nheads, headdim))


# autograd 参考梯度
x_l = x.clone().requires_grad_(True)
B_l = Bm.clone().requires_grad_(True)
cs_l = cs.clone().requires_grad_(True)
dtf_l = dt_f.clone().requires_grad_(True)
states_l = states_ref(x_l, B_l, cs_l, dtf_l)
dstates = torch.randn_like(states_l)
dx_ref, dB_ref, dcs_ref, ddt_ref = torch.autograd.grad(states_l, [x_l, B_l, cs_l, dtf_l], grad_outputs=dstates)

# ---- 1) _chunk_state_bwd_dx：dx / ddt / ddA_cumsum ----
# 注：ddA 输出语义按 _chunk_cumsum_bwd 的消费方式应为 dL/dr（r=逐位置乘积），
#     即 dL/dcs 的逆序累积 flip.cumsum.flip；三种假设并列打印
flip_ref = dcs_ref.flip(-1).cumsum(-1).flip(-1)
try:
    dx_t, ddt_t, ddA_t = _chunk_state_bwd_dx(Bm, x, dt_f, cs, dstates)
    torch.npu.synchronize()
    print("[bwd_dx] dx  max abs diff:", (dx_t - dx_ref).abs().max().item())
    print("[bwd_dx] ddt max abs diff:", (ddt_t - ddt_ref).abs().max().item())
    print("[bwd_dx] ddA direct  max abs diff:", (ddA_t - dcs_ref).abs().max().item())
    print("[bwd_dx] ddA anchored max abs diff:", (ddA_t - (dcs_ref - dcs_ref[..., :1])).abs().max().item())
    print("[bwd_dx] ddA flip(dL/dr) max abs diff:", (ddA_t - flip_ref).abs().max().item())
except Exception as e:
    print("[bwd_dx] RUNTIME FAIL:", type(e).__name__, str(e)[:120])

# ---- 2) _chunk_state_bwd_db：预期 wrapper 层 TypeError（E17 同款）----
try:
    dB_t, ddA2_t = _chunk_state_bwd_db(x, dt_f, cs, dstates, B=Bm, ngroups=ngroups)
    torch.npu.synchronize()
    print("[bwd_db] dB max abs diff:", (dB_t - dB_ref).abs().max().item())
except Exception as e:
    print("[bwd_db] RUNTIME FAIL:", type(e).__name__, str(e)[:120])

# ---- 3) _chunk_state_bwd_ddAcs_stable：wrapper 对 kernel 输出做 cumsum 后处理 ----
# 语义假设并列：direct(dL/dcs) / anchored(pos0 锚定) / flip(dL/dr，与 _chunk_cumsum_bwd 消费方式一致)
try:
    ddA_t = _chunk_state_bwd_ddAcs_stable(Bm, x, dt_f, cs, dstates)
    torch.npu.synchronize()
    diff_direct = (ddA_t - dcs_ref).abs().max().item()
    anchored = dcs_ref - dcs_ref[..., :1]
    diff_anchor = (ddA_t - anchored).abs().max().item()
    diff_flip = (ddA_t - flip_ref).abs().max().item()
    print(f"[bwd_ddAcs] direct(dL/dcs): {diff_direct:.3e} | anchored: {diff_anchor:.3e} | flip(dL/dr): {diff_flip:.3e}")
except Exception as e:
    print("[bwd_ddAcs] RUNTIME FAIL:", type(e).__name__, str(e)[:120])
