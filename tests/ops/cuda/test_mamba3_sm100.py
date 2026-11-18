"""The Blackwell (sm_100 / sm_103) Mamba-3 MIMO kernels (mamba_ssm/ops/cuda/mamba3/sm100), end to end through
mamba3_mimo's autograd function: forward output and every gradient against an fp64 autograd reference (which also
differentiates the angle cumsum and the ADT chunk cumsums), next to the TileLang kernels. The sm100 path must be
finite on ragged packings, bitwise deterministic, and at least as accurate as TileLang."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="sm_100 / sm_103 only")

H, N, P, R, C, ROT = 12, 128, 64, 4, 16, 4
LENS = [1, 17, 200, 333, 70, 1000, 31, 33]   # 1-token and 32 +- 1 segments, chunk tails, several sub-blocks
NAMES = ["Q", "K", "V", "Q_bias", "K_bias", "MIMO_V", "MIMO_Out", "Z", "MIMO_Z", "Angles", "DT", "ADT", "Trap", "D"]


def _inputs(lens, seed=0):
    torch.manual_seed(seed)
    S = sum(lens)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
    q = torch.randn(1, S, R, 1, N, device="cuda").bfloat16()
    v = torch.randn(1, S, H, P, device="cuda").bfloat16()
    dt = torch.nn.functional.softplus(-3.0 + torch.randn(1, H, S, device="cuda"))
    return dict(Q=q, K=torch.randn_like(q), V=v, ADT=(-dt * torch.rand(1, H, S, device="cuda")).contiguous(), DT=dt,
                Trap=torch.rand(1, H, S, device="cuda").bfloat16(), Q_bias=torch.randn(H, R, N, device="cuda"),
                K_bias=torch.randn(H, R, N, device="cuda"), MIMO_V=torch.randn(H, R, P, device="cuda") / R,
                MIMO_Z=torch.randn(H, R, P, device="cuda") / R, MIMO_Out=torch.randn(H, R, P, device="cuda") / R,
                Angles=torch.rand(1, S, H, N // ROT, device="cuda"), D=torch.randn(H, device="cuda"),
                Z=torch.randn_like(v)), cu


def _run(x, cu, dout, sm100):
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo
    os.environ["M3_SM100"] = "1" if sm100 else "0"
    try:
        leaves = {k: t.detach().requires_grad_(True) for k, t in x.items()}
        out = mamba3_mimo(chunk_size=C, rotary_dim_divisor=ROT, dtype=torch.bfloat16, cu_seqlens=cu, **leaves)
        grads = torch.autograd.grad(out, [leaves[k] for k in NAMES], dout)
    finally:
        os.environ.pop("M3_SM100", None)
    return out, dict(zip(NAMES, grads))


def _reference(x, cu, lens, dout):
    spec = importlib.util.spec_from_file_location("t3", Path(__file__).parents[1] / "tilelang" / "test_mamba3_mimo.py")
    t3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t3)
    d = {k: x[k].detach().double().requires_grad_(True) for k in NAMES}

    def segcs(a, ch):  # cumsum within each sequence (and within chunks of ch tokens), and the reverse-exclusive one
        cs, rv = [], []
        for i in range(len(lens)):
            seg = a[..., int(cu[i]):int(cu[i + 1])]
            if ch is None:
                cs.append(seg.cumsum(-1))
                continue
            L = seg.shape[-1]
            sp = torch.nn.functional.pad(seg, (0, (-L) % ch)).reshape(*seg.shape[:-1], -1, ch)
            cs.append(sp.cumsum(-1).reshape(*seg.shape[:-1], -1)[..., :L])
            rv.append((sp.flip(-1).cumsum(-1).flip(-1) - sp).reshape(*seg.shape[:-1], -1)[..., :L])
        return torch.cat(cs, -1), (torch.cat(rv, -1) if rv else None)

    dacs, dacsr = segcs(d["ADT"], C)
    inc = torch.tanh(d["Angles"]) * torch.pi * d["DT"][0].transpose(0, 1)[None, :, :, None]
    ang = segcs(inc.permute(0, 2, 3, 1), None)[0].permute(0, 3, 1, 2)
    out, _, _ = t3.mamba3_MIMO_chunk_ref(
        d["Q"], d["K"], d["V"], d["Q_bias"], d["K_bias"], d["MIMO_V"], d["MIMO_Out"], d["Z"], d["MIMO_Z"], ang, dacs,
        dacsr, d["DT"], d["Trap"], d["D"], chunk_size=C, rotary_dim_divisor=ROT, dtype=torch.float64, cu_seqlens=cu)
    return out, dict(zip(NAMES, torch.autograd.grad(out, [d[k] for k in NAMES], dout.double())))


def test_sm100_fwd_bwd_finite_deterministic_and_as_accurate_as_tilelang():
    x, cu = _inputs(LENS)
    dout = torch.randn_like(x["V"])
    ref_out, ref = _reference(x, cu, LENS, dout)
    tl_out, tl = _run(x, cu, dout, sm100=False)
    o1, g1 = _run(x, cu, dout, sm100=True)
    o2, g2 = _run(x, cu, dout, sm100=True)
    rel = lambda a, r: ((a.double().reshape(r.shape) - r).norm() / r.norm()).item()  # noqa: E731
    assert torch.equal(o1, o2), "forward not deterministic"
    assert rel(o1, ref_out) <= 1.1 * rel(tl_out, ref_out) + 1e-4
    for name in NAMES:
        assert torch.isfinite(g1[name]).all(), name
        assert torch.equal(g1[name], g2[name]), f"{name} not deterministic"
        e_sm, e_tl = rel(g1[name], ref[name]), rel(tl[name], ref[name])
        assert e_sm <= 1.1 * e_tl + 1e-4, f"{name}: sm100 {e_sm:.2e} vs TileLang {e_tl:.2e} (rel L2 vs fp64)"


def test_sm100_falls_back_outside_the_olala_configuration():
    from mamba_ssm.ops.cuda.mamba3.sm100.common import config_ok
    x, cu = _inputs([64])
    q, k, v = x["Q"], x["K"], x["V"]
    args = (q, k, v, H, x["MIMO_Out"], x["Z"], x["D"], x["Angles"], C, ROT, torch.bfloat16, False, cu, False)
    assert config_ok(*args)
    assert not config_ok(*args[:-1], True)                          # initial states
    assert not config_ok(*args[:11], True, *args[12:])             # fused pre-gate RMSNorm
    assert not config_ok(*args[:12], None, False)                   # dense (non-varlen) layout
    assert not config_ok(q, k, v, H, None, *args[5:])               # no MIMO_Out reduction
