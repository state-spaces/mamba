# Modified by Mzero #20240123
# Copyright (C) 2023, Tri Dao, Albert Gu.
import math
import torch
import torch.nn.functional as F
import pytest
from einops import rearrange, repeat


def selective_scan_easy(us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, chunksize=64):
    """
    # B: batch_size, G: groups, D: dim, N: state dim, L: seqlen
    us: B, G * D, L 
    dts: B, G * D, L
    As: G * D, N
    Bs: B, G, N, L
    Cs: B, G, N, L
    Ds: G * D
    delta_bias: G * D
    # chunksize can be any as you like. But as the chunksize raises, hs may get None, as exp(sum(delta) A) is really small
    """
    def selective_scan_chunk(us, dts, As, Bs, Cs, Ds, hprefix):
        """
        partial(h) / partial(t) = Ah + Bu; y = Ch + Du;
        => partial(h*exp(-At)) / partial(t) = Bu*exp(-At);
        => h_t = h_0 + sum_{0}_{t}_{Bu*exp(A(t-v)) dv};
        => h_b = exp(A(dt_a + ... + dt_{b-1})) * (h_a + sum_{a}_{b-1}_{Bu*exp(-A(dt_a + ... + dt_i)) dt_i});
           y_i = C_i*h_i + D*u_i
        """
        """
        us: (L, B, G, D) # L is chunk_size
        dts: (L, B, G, D)
        As: (G, D, N)
        Bs: (L, B, G, N)
        Cs: (L, B, G, N)
        Ds: (G, D)
        hprefix: (1, B, G, D, N)
        """
        
        Ats = torch.einsum("gdn,lbgd->lbgdn", As, dts.cumsum(dim=0)).exp()
        dtBus = torch.einsum("lbgd,lbgn,lbgd->lbgdn", dts, Bs, us)
        hs = Ats * (dtBus / Ats + 1e-11).cumsum(dim=0) + Ats * hprefix.unsqueeze(0)
        ys = torch.einsum("lbgn,lbgdn->lbgd", Cs, hs) 
        if Ds is not None:
            ys = ys + Ds * us
        return ys, hs[-1]
    
    dts = dts.float()
    if delta_bias is not None:
        dts = dts + delta_bias.view(1, -1, 1).float()
    if delta_softplus:
        dts = torch.nn.functional.softplus(dts)
    
    if len(Bs.shape) == 3:
        Bs = Bs.unsqueeze(1)
    if len(Cs.shape) == 3:
        Cs = Cs.unsqueeze(1)
    B, G, N, L = Bs.shape
    us = us.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    As = As.view(G, -1, N).float()
    Bs = Bs.permute(3, 0, 1, 2).float()
    Cs = Cs.permute(3, 0, 1, 2).float()
    Ds = Ds.view(G, -1).float() if Ds is not None else None
    D = As.shape[1]
    
    oys = []
    hprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
    for i in range(0, L, chunksize):
        ys, hprefix = selective_scan_chunk(
            us[i:i + chunksize], dts[i:i + chunksize], 
            As, Bs[i:i + chunksize], Cs[i:i + chunksize], Ds, hprefix, 
        )
        oys.append(ys)

    oys = torch.cat(oys, dim=0).permute(1, 2, 3, 0).view(B, -1, L)
    return oys


def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if B.dim() == 3:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    else:
        B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
        deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C.dim() == 3:
            y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        else:
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    out = out.to(dtype=dtype_in)
    return out


@pytest.mark.parametrize('wtype', [torch.float32])
# @pytest.mark.parametrize('itype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('itype', [torch.float32])
@pytest.mark.parametrize('seqlen', [256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize('has_delta_bias', [False, True])
@pytest.mark.parametrize('delta_softplus', [True])
@pytest.mark.parametrize('has_D', [False, True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
@pytest.mark.parametrize("chunksize", [64])
def test_selective_scan(varBC_groups, has_D, has_delta_bias,
                        delta_softplus, seqlen, itype, wtype, chunksize):
    device = 'cuda'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    dim = 8
    dstate = 8
    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_()
    if varBC_groups == 1:
        B_shape = (batch_size, dstate, seqlen)
    else:
        B_shape = (batch_size, varBC_groups, dstate, seqlen)
    B = torch.randn(*B_shape, device=device, dtype=itype,
                    requires_grad=True)
    if varBC_groups == 1:
        C_shape = (batch_size, dstate, seqlen)
    else:
        C_shape = (batch_size, varBC_groups, dstate, seqlen)
    C = torch.randn(*C_shape, device=device, dtype=itype,
                    requires_grad=True)
    if has_D:
        D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        D = None
    if has_delta_bias:
        delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_()
    else:
        delta_bias = None
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    delta = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_()
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    u_ref = u.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
    out = selective_scan_easy(
        u, delta, A, B, C, D,
        delta_bias=delta_bias, delta_softplus=delta_softplus, chunksize=chunksize,
    )
    out_ref = selective_scan_ref(
        u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref,
        delta_bias=delta_bias_ref, delta_softplus=delta_softplus,
    )
    # dA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # dt_u = delta * u

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f'du max diff: {(u.grad - u_ref.grad).abs().max().item()}')
    print(f'ddelta max diff: {(delta.grad - delta_ref.grad).abs().max().item()}')
    print(f'dA max diff: {(A.grad - A_ref.grad).abs().max().item()}')
    print(f'dB max diff: {(B.grad - B_ref.grad).abs().max().item()}')
    print(f'dC max diff: {(C.grad - C_ref.grad).abs().max().item()}')
    if has_D:
        print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')
    if has_delta_bias:
        print(f'ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}')

    assert torch.allclose(u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2)
    assert torch.allclose(delta.grad, delta_ref.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10)
    assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
    assert torch.allclose(B.grad, B_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(C.grad, C_ref.grad, rtol=rtol, atol=atol)
    if has_D:
        assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
    if has_delta_bias:
        assert torch.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)

