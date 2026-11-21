"""The opt-in CUDA forward kernels (M3_CUDA_FWD=1: full pass + state-only two-level Pass A) against an fp64
reference, next to the TileLang kernel: finite on ragged packings (chunk tails, 1-token segments, the last chunk
past the buffer end), bitwise deterministic, and at least as accurate as TileLang -- for the stock single-launch
call and for the two-level decomposition (several blocks per segment)."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="sm_90 only")

H, N, P, R, C, ROT = 12, 128, 64, 4, 16, 4
LENS = [1, 17, 2000, 2078]


def _inputs():
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton_varlen
    torch.manual_seed(0)
    S = sum(LENS)
    cu = torch.tensor([0] + torch.tensor(LENS).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
    q = torch.randn(1, S, R, 1, N, device="cuda").bfloat16()
    k = torch.randn_like(q)
    v = torch.randn(1, S, H, P, device="cuda").bfloat16()
    dt = torch.nn.functional.softplus(-3.0 + torch.randn(1, H, S, device="cuda"))
    dA = (-dt * torch.rand(1, H, S, device="cuda")).contiguous()
    dA_cs, dA_cs_rev, segsum = compute_dacs_segsum_triton_varlen(dA, C, cu_seqlens=cu)
    return dict(q=q, k=k, v=v, q_bias=torch.randn(H, R, N, device="cuda"), k_bias=torch.randn(H, R, N, device="cuda"),
                mimo_v=torch.randn(H, R, P, device="cuda") / R, mimo_o=torch.randn(H, R, P, device="cuda") / R,
                z=torch.randn_like(v), D=torch.randn(H, device="cuda"), mimo_z=torch.randn(H, R, P, device="cuda") / R,
                angles=angle_dt_fwd(torch.rand(1, S, H, N // ROT, device="cuda"), dt, chunk_size=C, cu_seqlens=cu),
                dA_cs=dA_cs, dA_cs_rev=dA_cs_rev, dt=dt, trap=torch.rand(1, H, S, device="cuda").bfloat16(),
                segsum=segsum, cu_seqlens=cu)


def _kwargs(x):
    return dict(x, chunk_size=C, rotary_dim_divisor=ROT, dtype=torch.bfloat16)


def _with_cuda(cuda, fn):
    os.environ["M3_CUDA_FWD"] = "1" if cuda else "0"
    try:
        return fn()
    finally:
        os.environ.pop("M3_CUDA_FWD", None)


def _stock(x, cuda):
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd_varlen import mamba_mimo_forward_varlen
    return _with_cuda(cuda, lambda: mamba_mimo_forward_varlen(**_kwargs(x))[0])


def _two_level(x, cuda, scan_block=4):
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd_varlen import mamba_mimo_forward_varlen
    from mamba_ssm.ops.tilelang.mamba3.mamba3_twolevel import two_level_forward
    return _with_cuda(cuda, lambda: two_level_forward(mamba_mimo_forward_varlen, _kwargs(x), scan_block)[0])


def _pass_a(x, cuda, scan_block=4):
    """Two-level Pass A: the state each block produces from zero, [NBLK, H, N, P]."""
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd_varlen import mamba_mimo_forward_varlen
    from mamba_ssm.ops.tilelang.mamba3.mamba3_twolevel import _zeros, get_plan
    pl = get_plan(x["cu_seqlens"], C, scan_block)
    kw = dict(_kwargs(x), blk_seg=pl["seg"], blk_c0=pl["c0"], blk_nch=pl["nch"], state_only=True, return_state=True,
              init_state=_zeros((pl["nblk"], H, N, P), x["q"].device))
    return _with_cuda(cuda, lambda: mamba_mimo_forward_varlen(**kw)[1])


def _reference(x):
    spec = importlib.util.spec_from_file_location("t3", Path(__file__).parents[1] / "tilelang" / "test_mamba3_mimo.py")
    t3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t3)
    keys = ["q", "k", "v", "q_bias", "k_bias", "mimo_v", "mimo_o", "z", "mimo_z", "angles", "dA_cs", "dA_cs_rev", "dt",
            "trap", "D"]
    with torch.no_grad():
        out, _, _ = t3.mamba3_MIMO_chunk_ref(*[x[kk].double() for kk in keys], chunk_size=C, rotary_dim_divisor=ROT,
                                             dtype=torch.float64, cu_seqlens=x["cu_seqlens"])
    return out


def _rel(a, r):
    a, r = a.double().reshape(r.shape), r.double()
    return ((a - r).norm() / r.norm()).item()


@pytest.mark.parametrize("mode", ["stock", "two_level"])
def test_cuda_fwd_finite_deterministic_and_as_accurate_as_tilelang(mode):
    x = _inputs()
    ref = _reference(x)
    run = _stock if mode == "stock" else _two_level
    tl, cu1, cu2 = run(x, False), run(x, True), run(x, True)
    assert torch.isfinite(cu1).all()
    assert torch.equal(cu1, cu2), "not deterministic"
    e_tl, e_cu = _rel(tl, ref), _rel(cu1, ref)
    assert e_cu <= 1.1 * e_tl + 1e-4, f"CUDA {e_cu:.2e} vs TileLang {e_tl:.2e} (rel L2 vs fp64)"


def test_cuda_state_only_pass_matches_tilelang():
    x = _inputs()
    tl, cu = _pass_a(x, False), _pass_a(x, True)
    assert torch.isfinite(cu).all()
    assert _rel(cu, tl) < 2e-2, f"Pass A states differ: {_rel(cu, tl):.2e}"
