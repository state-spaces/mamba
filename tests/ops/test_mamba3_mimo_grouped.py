# Copyright (c) 2026, Tri Dao.
"""Parity: group-parallel varlen prefill vs the single-pass kernel.

``mamba3_mimo_varlen_grouped`` restructures long-prompt prefill as two kernel
passes over a virtual-sequence layout plus a closed-form state chain. Outputs
and final states must match ``mamba3_mimo(..., return_state=True)`` to bf16
rounding on: a single long sequence, a mixed varlen batch, and a continuation
carrying input states.
"""
import pytest
import torch

from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import (
    mamba3_mimo,
    mamba3_mimo_varlen_grouped,
)

H, P, N, R, GQK, C, ROT = 48, 64, 128, 4, 1, 16, 4

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _inputs(S, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=g, device="cuda",
                           dtype=torch.float32).to(dtype)

    dt = torch.rand(1, H, S, generator=g, device="cuda") * 0.25 + 0.05
    A = -torch.rand(1, H, S, generator=g, device="cuda")
    return dict(
        Q=rn(1, S, R, GQK, N) * 0.5, K=rn(1, S, R, GQK, N) * 0.5,
        V=rn(1, S, H, P) * 0.5, Z=rn(1, S, H, P) * 0.5,
        ADT=(A * dt).contiguous(), DT=dt.contiguous(),
        Trap=rn(1, H, S) * 0.7,
        Q_bias=rn(H, R, N, dtype=torch.float32) * 0.1,
        K_bias=rn(H, R, N, dtype=torch.float32) * 0.1,
        MIMO_V=rn(H, R, P, dtype=torch.float32) * 0.3,
        MIMO_Z=rn(H, R, P, dtype=torch.float32) * 0.3,
        MIMO_Out=rn(H, R, P, dtype=torch.float32) * 0.3,
        Angles=rn(1, S, H, N // ROT, dtype=torch.float32) * 0.02,
        D=torch.rand(H, generator=g, device="cuda"),
    )


def _rel(x, y):
    x, y = x.float(), y.float()
    return ((x - y).abs().max() / y.abs().max().clamp_min(1e-6)).item()


def _check(inp, cu, group_tokens, init=None):
    base = mamba3_mimo(**inp, chunk_size=C, rotary_dim_divisor=ROT,
                       dtype=torch.bfloat16, return_state=True,
                       cu_seqlens=cu, Input_States=init)
    grp = mamba3_mimo_varlen_grouped(
        **inp, chunk_size=C, rotary_dim_divisor=ROT, dtype=torch.bfloat16,
        cu_seqlens=cu, Input_States=init, group_tokens=group_tokens)
    out_d, ang_d, ssm_d, k_d = (_rel(g, b) for g, b in zip(grp[:4], base[:4]))
    assert out_d < 3e-2, f"Out rel diff {out_d}"
    assert ang_d < 1e-5, f"Angle rel diff {ang_d}"
    assert ssm_d < 3e-2, f"SSM rel diff {ssm_d}"
    assert k_d < 3e-2, f"K rel diff {k_d}"
    assert torch.equal(grp[4], base[4])  # Final_V is a pure slice


@requires_cuda
@pytest.mark.parametrize("group_tokens", [2048, 4096])
def test_grouped_single_long(group_tokens):
    torch.manual_seed(0)
    S = 16384
    _check(_inputs(S), torch.tensor([0, S], dtype=torch.int32, device="cuda"),
           group_tokens)


@requires_cuda
def test_grouped_mixed_varlen():
    """One short (unsplit) + one long (split) sequence in the same batch."""
    torch.manual_seed(0)
    S = 2048 + 16384
    cu = torch.tensor([0, 2048, S], dtype=torch.int32, device="cuda")
    _check(_inputs(S, seed=1), cu, 2048)


@requires_cuda
def test_grouped_with_input_states():
    """Continuation prefill: input states must chain through the groups."""
    torch.manual_seed(0)
    S = 16384
    init = (
        torch.rand(1, H, N // ROT, device="cuda") * 3.0,
        torch.randn(1, H, P, N, device="cuda") * 0.3,
        (torch.randn(1, R, H, N, device="cuda") * 0.3).bfloat16(),
        (torch.randn(1, H, P, device="cuda") * 0.3).bfloat16(),
    )
    _check(_inputs(S), torch.tensor([0, S], dtype=torch.int32, device="cuda"),
           2048, init=init)


@requires_cuda
def test_grouped_short_fallback():
    """Below min_split_tokens the wrapper must take the single-pass path
    and still return the exact 5-tuple contract."""
    torch.manual_seed(0)
    S = 1024
    _check(_inputs(S, seed=2),
           torch.tensor([0, S], dtype=torch.int32, device="cuda"), 2048)
