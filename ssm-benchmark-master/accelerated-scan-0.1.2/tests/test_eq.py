import pytest

import torch
from accelerated_scan.ref import scan as scan_ref
from accelerated_scan.warp import scan as scan_warp
from accelerated_scan.triton import scan as scan_triton

# https://arxiv.org/abs/2109.08203
seeds = [3407,4,42,57]
seqlens = [2**i for i in range(5, 17)]
scans = [scan_warp, scan_triton]
dtypes = [torch.float32, torch.bfloat16, torch.float16]
atol = {torch.float32: 1e-7, torch.bfloat16: 1e-1, torch.float16: 12e-3}

def init(seed, batch_size=3, dim=1536, seqlen=32, requires_grad=False, dtype=torch.float32):
    torch.manual_seed(seed)
    gates = torch.rand(batch_size, dim, seqlen, requires_grad=requires_grad, dtype=dtype, device="cuda")
    tokens = torch.rand(batch_size, dim, seqlen, requires_grad=requires_grad, dtype=dtype, device="cuda")
    if requires_grad:
        gates.retain_grad()
        tokens.retain_grad()
    return gates, tokens


@pytest.mark.parametrize("scan", scans)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("dtype", dtypes)
@torch.inference_mode()
def test_eq_forward(scan, seed, seqlen, dtype):
    gates, tokens = init(seed, seqlen=seqlen, dtype=dtype)
    out = scan(gates, tokens)
    out_ref = scan_ref(gates, tokens)

    print('max abs error', (out - out_ref).abs().max().item(), 'seqlen', seqlen, 'dtype', dtype)
    #print(out, 'out')
    #print(out_ref, 'ref')

    assert torch.allclose(out, out_ref, atol=atol[dtype])


@pytest.mark.parametrize("scan", scans)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("dtype", dtypes)
def test_eq_backward(scan, seed, seqlen, dtype):
    gates, tokens = init(seed, seqlen=seqlen, requires_grad=True, dtype=dtype)
    scan(gates, tokens).mean(dim=-1).sum().backward()
    gates_grad = gates.grad
    tokens_grad = tokens.grad
    del gates
    del tokens

    gates_ref, tokens_ref = init(seed, seqlen=seqlen, requires_grad=True, dtype=dtype)
    scan_ref(gates_ref, tokens_ref).mean(dim=-1).sum().backward()

    print('gate max abs error', (gates_grad - gates_ref.grad).abs().max().item(), 'seqlen', seqlen, 'dtype', dtype)
    print('token max abs error', (tokens_grad - tokens_ref.grad).abs().max().item(), 'seqlen', seqlen, 'dtype', dtype)

    assert torch.allclose(gates_grad, gates_ref.grad, atol=atol[dtype])
    assert torch.allclose(tokens_grad, tokens_ref.grad, atol=atol[dtype])


@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("seqlen", seqlens)
def test_eq_ref_reverse(seed, seqlen):
    generator = torch.Generator().manual_seed(seed)
    B,C,T = 1, 1, seqlen
    f = torch.randn(B, C, T, generator=generator, requires_grad=True)
    x = torch.randn(B, C, T, generator=generator, requires_grad=True)

    c = scan_ref(f, x)

    dldc = torch.ones_like(c)

    fpx = torch.cat([f, torch.ones_like(f[:, :, :1])], dim=-1)[:, :, 1:].contiguous()
    dcdx = scan_ref(fpx, dldc, reverse=True)
    cp = torch.cat([torch.zeros_like(c[:, :, :1]), c], dim=-1)[:, :, :-1].contiguous()
    dcdf = dcdx * cp

    c.sum().backward()
    print(dcdx, 'dcdx')
    print(x.grad, 'x.grad')
    print((x.grad - dcdx).abs().max(), 'x error')
    assert torch.allclose(x.grad, dcdx, atol=1e-5)
    print(dcdf, 'dcdf')
    print(f.grad, 'f.grad')
    print((f.grad - dcdf).abs().max(), 'f error')
    assert torch.allclose(f.grad, dcdf, atol=2e-5)
