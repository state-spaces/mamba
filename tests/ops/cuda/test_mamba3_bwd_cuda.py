"""The opt-in CUDA backward kernels (M3_CUDA_BWD=1: bwd_fwd + bwd_bwd) against an fp64 autograd reference, next to the TileLang kernel:
it must be finite on ragged packings (chunk tails at the end of the buffer), bitwise deterministic, and at
least as accurate as TileLang."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="sm_90 only")

H, N, P, R, C, ROT = 12, 128, 64, 4, 16, 4
LENS = [1, 17, 2000, 2078]          # tails, a 1-token segment, the last chunk past the buffer end
NAMES = ["dq", "dk", "dv", "dA", "ddt", "dtrap", "dq_bias", "dk_bias", "dmimo_v", "dmimo_z", "dmimo_o", "dangles", "dD", "dz"]


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
    return dict(dout=torch.randn_like(v), q=q, k=k, v=v, q_bias=torch.randn(H, R, N, device="cuda"),
                k_bias=torch.randn(H, R, N, device="cuda"), mimo_v=torch.randn(H, R, P, device="cuda") / R,
                mimo_o=torch.randn(H, R, P, device="cuda") / R, z=torch.randn_like(v), mimo_z=torch.randn(H, R, P, device="cuda") / R,
                angles=angle_dt_fwd(torch.rand(1, S, H, N // ROT, device="cuda"), dt, chunk_size=C, cu_seqlens=cu),
                dA_cs=dA_cs, dA_cs_rev=dA_cs_rev, dt=dt, trap=torch.rand(1, H, S, device="cuda").bfloat16(),
                D=torch.randn(H, device="cuda"), segsum=segsum, cu=cu)


def _bwd(x, cuda):
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd_varlen import mamba_mimo_bwd_combined_varlen
    os.environ["M3_CUDA_BWD"] = "1" if cuda else "0"
    try:
        out = mamba_mimo_bwd_combined_varlen(
            x["dout"], x["q"], x["k"], x["v"], x["q_bias"], x["k_bias"], x["mimo_v"], x["mimo_o"], x["z"], x["mimo_z"],
            x["angles"], x["dA_cs"], x["dA_cs_rev"], x["dt"], x["trap"], x["D"], x["segsum"], C, ROT, torch.bfloat16,
            cu_seqlens=x["cu"])
    finally:
        os.environ.pop("M3_CUDA_BWD", None)
    return dict(zip(NAMES, out[:14]))


def _reference(x):
    spec = importlib.util.spec_from_file_location("t3", Path(__file__).parents[1] / "tilelang" / "test_mamba3_mimo.py")
    t3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t3)
    keys = ["q", "k", "v", "q_bias", "k_bias", "mimo_v", "mimo_o", "z", "mimo_z", "angles", "dA_cs", "dA_cs_rev", "dt", "trap", "D"]
    leaves = {kk: x[kk].detach().double().requires_grad_(True) for kk in keys}
    out, _, _ = t3.mamba3_MIMO_chunk_ref(*[leaves[kk] for kk in keys], chunk_size=C, rotary_dim_divisor=ROT,
                                         dtype=torch.float64, cu_seqlens=x["cu"])
    g = torch.autograd.grad(out, [leaves[kk] for kk in keys], x["dout"].double())
    m = dict(zip(keys, g))
    return dict(dq=m["q"], dk=m["k"], dv=m["v"], dangles=m["angles"], ddt=m["dt"], dtrap=m["trap"], dq_bias=m["q_bias"],
                dk_bias=m["k_bias"], dmimo_v=m["mimo_v"], dmimo_o=m["mimo_o"], dmimo_z=m["mimo_z"], dz=m["z"])


def test_cuda_bwd_finite_deterministic_and_as_accurate_as_tilelang():
    x = _inputs()
    ref = _reference(x)
    tl, cu1, cu2 = _bwd(x, False), _bwd(x, True), _bwd(x, True)
    for name, r in ref.items():
        a, b, c = tl[name].double().reshape(r.shape), cu1[name].double().reshape(r.shape), cu2[name]
        assert torch.isfinite(b).all(), name
        assert torch.equal(cu1[name], c), f"{name} not deterministic"
        e_tl, e_cu = ((a - r).norm() / r.norm()).item(), ((b - r).norm() / r.norm()).item()
        assert e_cu <= 1.1 * e_tl + 1e-4, f"{name}: CUDA {e_cu:.2e} vs TileLang {e_tl:.2e} (rel L2 vs fp64)"
