import torch
import torch_npu
import torch.nn.functional as F
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd
from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd
from mamba_ssm.ops.triton.ssd_chunk_scan import (
    _chunk_scan_bwd_dz, _chunk_scan_bwd_dstates, _chunk_scan_bwd_dC, _chunk_scan_bwd_dcb,
    _chunk_scan_bwd_ddAcs_stable, _chunk_scan_bwd_ddAcs_prev,
)
from mamba_ssm.ops.triton.ssd_combined import _chunk_scan_chunk_state_bwd_dx

# chunk_scan 反向 kernel 单算子校验（chunk_scan_ref 同款数学，叶子 + autograd 参考）
# 预期：_chunk_scan_bwd_dC（ssd_chunk_scan.py:1485）与 _chunk_scan_bwd_dcb（:1534）
#       在 wrapper 层 TypeError（sm_count=None，E17 同款）
torch.manual_seed(0)
device = "npu"
dtype = torch.float32
batch, seqlen, nheads, headdim, ngroups, dstate, chunk = 2, 64, 4, 32, 2, 32, 32
nc = seqlen // chunk
ratio = nheads // ngroups

x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
Bm = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype)
Cm = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype)
D = torch.randn(nheads, headdim, device=device, dtype=dtype)
z = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
prev = torch.randn(batch, nc, nheads, headdim, dstate, device=device, dtype=dtype)
dt_raw = torch.randn(batch, seqlen, nheads, device=device, dtype=dtype) * 0.5
A = -torch.exp(torch.randn(nheads, device=device, dtype=dtype))
dt_bias = torch.randn(nheads, device=device, dtype=dtype)
cs, dt_f = _chunk_cumsum_fwd(dt_raw, A, chunk, dt_bias=dt_bias, dt_softplus=True)
CB = _bmm_chunk_fwd(Cm, Bm, chunk)   # (b, c, g, cl, cl)
torch.npu.synchronize()
x_c = x.reshape(batch, nc, chunk, nheads, headdim)
Cm_c = Cm.reshape(batch, nc, chunk, ngroups, dstate).repeat_interleave(ratio, dim=3)  # b c l h n
Bm_c = Bm.reshape(batch, nc, chunk, ngroups, dstate).repeat_interleave(ratio, dim=3)


def out_x_ref(x_, C_, B_, prev_, cs_, dtf_, D_):
    """pre-gate 输出（chunk_scan_ref 同款）：局部 CB 路径 + prev_states 路径 + x*D"""
    CB_r = torch.einsum("bclhn,bcshn->bchls", C_, B_)                    # b c h l s
    seg = cs_[..., :, None] - cs_[..., None, :]                          # b h c l s
    mask = torch.tril(torch.ones(chunk, chunk, device=device, dtype=torch.bool))
    seg = seg.masked_fill(~mask, float("-inf"))                          # 先 mask 再 exp：杜绝上三角 exp(+大数)=inf 及 0*inf=NaN 反传
    decay = torch.exp(seg)
    scores = CB_r * decay.permute(0, 2, 1, 3, 4)                         # b c h l s
    scores = scores.masked_fill(~mask, 0)
    out_local = torch.einsum("bchls,bhcs,bcshp->bclhp", scores, dtf_, x_)  # b c l h p
    state_decay = torch.exp(cs_).permute(0, 2, 3, 1).unsqueeze(-1)       # b c l h 1
    out_prev = torch.einsum("bclhn,bchpn->bclhp", C_, prev_) * state_decay
    out_flat = (out_local + out_prev).reshape(batch, seqlen, nheads, headdim)
    return out_flat + x_.reshape(batch, seqlen, nheads, headdim) * D_


# ---- 1) _chunk_scan_bwd_dz：dz / dout_x / dD ----
try:
    g = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)  # 门控后余向量
    z_l = z.clone().requires_grad_(True)
    D_l = D.clone().requires_grad_(True)
    out_gated = out_x_ref(x_c, Cm_c, Bm_c, prev, cs, dt_f, D_l) * F.silu(z_l)
    out_x_data = out_x_ref(x_c, Cm_c, Bm_c, prev, cs, dt_f, D)
    dz_ref, dD_ref = torch.autograd.grad((out_gated * g).sum(), [z_l, D_l])
    dz_t, dout_x_t, dD_t = _chunk_scan_bwd_dz(x, z, out_x_data, g,
                                              chunk, has_ddAcs=False, D=D)
    torch.npu.synchronize()
    sig = torch.sigmoid(z)
    dz_ref = g * out_x_data * (sig * (1 + z * (1 - sig)))   # kernel 414 行：dz = dout*out*silu'(z)
    dout_x_ref = g * F.silu(z)                              # kernel 416 行：dout_x = dout*silu(z)（门控前余向量）
    print("[bwd_dz] dz     max abs diff:", (dz_t - dz_ref).abs().max().item())
    print("[bwd_dz] dout_x max abs diff:", (dout_x_t - dout_x_ref).abs().max().item())
    print("[bwd_dz] dD     max abs diff:", (dD_t - dD_ref).abs().max().item())
except Exception as e:
    print("[bwd_dz] RUNTIME FAIL:", type(e).__name__, str(e)[:120])

# ---- 2) _chunk_scan_bwd_dstates：dprev_states ----
try:
    g2 = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)  # 门控前余向量
    prev_l = prev.clone().requires_grad_(True)
    L = (out_x_ref(x_c, Cm_c, Bm_c, prev_l, cs, dt_f, D) * g2).sum()
    dstates_ref = torch.autograd.grad(L, prev_l)[0]
    dstates_t = _chunk_scan_bwd_dstates(Cm, cs, g2, dtype=torch.float32)
    torch.npu.synchronize()
    print("[bwd_dstates] max abs diff:", (dstates_t - dstates_ref).abs().max().item())
except Exception as e:
    print("[bwd_dstates] RUNTIME FAIL:", type(e).__name__, str(e)[:120])

# ---- 3) _chunk_scan_bwd_dC：E17 修复后可达；参考值 = 仅 prev_states 路径对 C 的导数（局部 CB 路径由 bwd_dcb 负责）----
try:
    g2 = g2 if 'g2' in dir() else torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
    Cm_l = Cm.clone().requires_grad_(True)
    Cm_c_l = Cm_l.reshape(batch, nc, chunk, ngroups, dstate).repeat_interleave(ratio, dim=3)  # b c l h n
    state_decay = torch.exp(cs).permute(0, 2, 3, 1).unsqueeze(-1)          # b c l h 1
    out_prev = torch.einsum("bclhn,bchpn->bclhp", Cm_c_l, prev) * state_decay
    L = (out_prev.reshape(batch, seqlen, nheads, headdim) * g2).sum()
    dC_ref = torch.autograd.grad(L, Cm_l)[0]            # (batch, seqlen, ngroups, dstate)
    dC_t, ddA_prev_t = _chunk_scan_bwd_dC(prev, cs, g2, C=Cm, ngroups=ngroups)
    torch.npu.synchronize()
    dC_t = dC_t.sum(dim=2) if dC_t.dim() == 5 else dC_t  # nsplits>1 时合并该维；nsplits==1 时 wrapper 已压缩
    print("[bwd_dC] dC_ref NaN:", torch.isnan(dC_ref).sum().item(), "/", dC_ref.numel(),
          " dC_t NaN:", torch.isnan(dC_t).sum().item(), "/", dC_t.numel(),
          " dC_ref absmax:", torch.nan_to_num(dC_ref, nan=0.0).abs().max().item())
    print("[bwd_dC] dC max abs diff:", (dC_t - dC_ref).abs().max().item())
except Exception as e:
    print("[bwd_dC] RUNTIME FAIL:", type(e).__name__, str(e)[:120])

# ---- 4) _chunk_scan_bwd_dcb：预期 wrapper TypeError（E17 同款）----
try:
    dCB_t, ddA_tmp_t = _chunk_scan_bwd_dcb(x, dt_f, cs, g2, ngroups=ngroups)
    torch.npu.synchronize()
    print("[bwd_dcb] dCB shape:", tuple(dCB_t.shape))
except Exception as e:
    print("[bwd_dcb] RUNTIME FAIL:", type(e).__name__, str(e)[:120])

# ---- 5) _chunk_scan_bwd_ddAcs_stable：语义假设 direct(dL/dcs) vs flip(dL/dr，下游 _chunk_cumsum_bwd 的消费方式）----
try:
    cs_l = cs.clone().requires_grad_(True)
    L = (out_x_ref(x_c, Cm_c, Bm_c, prev, cs_l, dt_f, D) * g2).sum()
    ddA_ref = torch.autograd.grad(L, cs_l)[0]
    flip_ref = ddA_ref.flip(-1).cumsum(-1).flip(-1)
    ddA_t = _chunk_scan_bwd_ddAcs_stable(x, dt_f, cs, g2, CB)
    torch.npu.synchronize()
    print("[bwd_ddAcs_stable] direct(dL/dcs) max abs diff:", (ddA_t - ddA_ref).abs().max().item())
    print("[bwd_ddAcs_stable] flip(dL/dr)  max abs diff:", (ddA_t - flip_ref).abs().max().item())
except Exception as e:
    print("[bwd_ddAcs_stable] RUNTIME FAIL:", type(e).__name__, str(e)[:120])

# ---- 6) _chunk_scan_bwd_ddAcs_prev：grad w.r.t. dA_cumsum（仅 prev_states 路径）----
# 语义假设同 5：direct vs flip（下游同样并入 cumsum_bwd 的 ddA）
try:
    cs_l = cs.clone().requires_grad_(True)
    state_decay = torch.exp(cs_l).permute(0, 2, 3, 1).unsqueeze(-1)
    out_prev_only = (torch.einsum("bclhn,bchpn->bclhp", Cm_c, prev) * state_decay
                     ).reshape(batch, seqlen, nheads, headdim)
    ddA_prev_ref = torch.autograd.grad((out_prev_only * g2).sum(), cs_l)[0]
    flip_prev_ref = ddA_prev_ref.flip(-1).cumsum(-1).flip(-1)
    ddA_prev_t = _chunk_scan_bwd_ddAcs_prev(prev, Cm, g2, cs)
    torch.npu.synchronize()
    print("[bwd_ddAcs_prev] direct(dL/dcs) max abs diff:", (ddA_prev_t - ddA_prev_ref).abs().max().item())
    print("[bwd_ddAcs_prev] flip(dL/dr)  max abs diff:", (ddA_prev_t - flip_prev_ref).abs().max().item())
except Exception as e:
    print("[bwd_ddAcs_prev] RUNTIME FAIL:", type(e).__name__, str(e)[:120])

# ---- 7) _chunk_scan_chunk_state_bwd_dx（ssd_combined.py 内）：两段拆解验证 ----
# 该 kernel 的 dx = 局部 CB 路径 + D 路径 + states 路径（dstates 收缩），三项可加，分别验证：
# 7a) dstates=0：dx = 局部+D 路径，参考 = out_x_ref（prev 独立叶子）
# 7b) dout=0：dx = states 路径，参考 = sum(chunk_state(x) * dstates_in) 对 x 的梯度
try:
    dout_pre = g2 * F.silu(z)   # 门控前余向量（上游 _chunk_scan_bwd_dz 的 dout 输出语义）
    # 7a
    dx_t, _, dD2_t = _chunk_scan_chunk_state_bwd_dx(x, dt_f, cs, Bm, CB, dout_pre,
                                                    torch.zeros_like(prev), D=D)
    torch.npu.synchronize()
    x_l = x.clone().requires_grad_(True)
    D_l = D.clone().requires_grad_(True)
    Lx = (out_x_ref(x_l.reshape(batch, nc, chunk, nheads, headdim), Cm_c, Bm_c, prev, cs, dt_f, D_l) * dout_pre).sum()
    dx_ref, dD_ref2 = torch.autograd.grad(Lx, [x_l, D_l])
    print("[combined_bwd_dx|局部+D] dx max abs diff:", (dx_t - dx_ref).abs().max().item())
    print("[combined_bwd_dx|局部+D] dD max abs diff:", (dD2_t - dD_ref2).abs().max().item())
    # 7b
    dstates_in = torch.randn_like(prev)
    dx_t2, _, _ = _chunk_scan_chunk_state_bwd_dx(x, dt_f, cs, Bm, CB,
                                                 torch.zeros_like(dout_pre), dstates_in, D=None)
    torch.npu.synchronize()
    Bg = Bm.reshape(batch, nc, chunk, ngroups, dstate).repeat_interleave(ratio, dim=3)
    decay = torch.exp(cs[..., -1:] - cs)
    x_l2 = x.clone().requires_grad_(True)
    states_l = torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", Bg, decay, dt_f,
                            x_l2.reshape(batch, nc, chunk, nheads, headdim))
    Ls = (states_l * dstates_in).sum()
    dx_ref2 = torch.autograd.grad(Ls, x_l2)[0]
    print("[combined_bwd_dx|states] dx max abs diff:", (dx_t2 - dx_ref2).abs().max().item())
except Exception as e:
    print("[combined_bwd_dx] RUNTIME FAIL:", type(e).__name__, str(e)[:120])
