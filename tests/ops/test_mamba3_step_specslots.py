# Copyright (c) 2026.
"""Parity: dual-slot step (state_batch_indices_out) vs the in-place step.

The spec-decode verify primitive: launch t reads the state at slot[t-1] and
writes slot[t]. Reference = the plain in-place step advanced sequentially on
one slot, checkpointing the pool after every token. Every per-position state
(ssm/k/v/angle) and every per-token output must be BIT-IDENTICAL: the dual
slot changes addressing only, not math.
"""
import pytest
import torch

from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn

H, D, N, R = 48, 64, 128, 4
ROT, A_HALF = 64, 32

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _mk_inputs(B, dev):
    return dict(
        A=-torch.rand(B, H, device=dev, dtype=torch.float32),
        x=torch.randn(B, H, D, device=dev, dtype=torch.bfloat16),
        dt=torch.rand(B, H, device=dev, dtype=torch.float32),
        trap=torch.rand(B, H, device=dev, dtype=torch.float32),
        B_pre=torch.randn(B, R, 1, N, device=dev, dtype=torch.bfloat16),
        C_pre=torch.randn(B, R, 1, N, device=dev, dtype=torch.bfloat16),
        angle_proj=torch.randn(B, 1, A_HALF, device=dev, dtype=torch.float32),
        z=torch.randn(B, H, D, device=dev, dtype=torch.bfloat16),
    )


def _step(pools, slots_in, slots_out, inp, weights, y):
    ssm, k, v, ang = pools
    mamba3_step_fn(
        ssm, k, v, inp["A"],
        inp["B_pre"].expand(-1, -1, H, -1),
        inp["C_pre"].expand(-1, -1, H, -1),
        weights["Dp"], inp["x"], inp["dt"], inp["trap"],
        weights["xpj"], weights["opj"], None, y,
        z=inp["z"], zproj=weights["zpj"],
        state_batch_indices=slots_in,
        state_batch_indices_out=slots_out,
        update_kv_state=True,
        rotary_dim=ROT,
        rotary_bias_q=weights["bias_q"], rotary_bias_k=weights["bias_k"],
        rotary_angle_proj=inp["angle_proj"].expand(-1, H, -1),
        rotary_angle_state=ang,
        tile_D=64, num_warps=4,
    )


@requires_cuda
@pytest.mark.parametrize("n_req,T", [(1, 4), (3, 5), (8, 2)])
def test_dual_slot_chain_matches_inplace(n_req: int, T: int):
    """T dual-slot launches == T in-place launches, per-position states."""
    torch.manual_seed(0)
    dev = "cuda"
    # Slot layout: request r owns main slot r and scratch slots
    # n_req + r*T + t for position t.
    P = n_req * (T + 1) + 3
    weights = dict(
        Dp=torch.randn(H, device=dev, dtype=torch.float32),
        xpj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        opj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        zpj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        bias_q=torch.randn(R, H, N, device=dev, dtype=torch.float32),
        bias_k=torch.randn(R, H, N, device=dev, dtype=torch.float32),
    )
    ssm0 = torch.randn(P, H, D, N, device=dev, dtype=torch.float32)
    k0 = torch.randn(P, R, H, N, device=dev, dtype=torch.bfloat16)
    v0 = torch.randn(P, H, D, device=dev, dtype=torch.bfloat16)
    ang0 = torch.randn(P, H, A_HALF, device=dev, dtype=torch.float32)
    steps = [_mk_inputs(n_req, dev) for _ in range(T)]

    # --- reference: in-place chain on the main slots, checkpoint each t ----
    pools_r = (ssm0.clone(), k0.clone(), v0.clone(), ang0.clone())
    main = torch.arange(n_req, device=dev, dtype=torch.int32)
    y_ref, ckpt = [], []
    for t in range(T):
        y = torch.empty(n_req, H, D, device=dev, dtype=torch.bfloat16)
        _step(pools_r, main, None, steps[t], weights, y)
        y_ref.append(y.clone())
        ckpt.append(tuple(p[main.long()].clone() for p in pools_r))

    # --- dual-slot chain: read slot[t-1], write scratch slot[t] ------------
    pools_s = (ssm0.clone(), k0.clone(), v0.clone(), ang0.clone())
    scratch = (
        n_req
        + torch.arange(n_req, device=dev, dtype=torch.int32)[:, None] * T
        + torch.arange(T, device=dev, dtype=torch.int32)[None, :]
    )  # (n_req, T)
    y_spec = []
    for t in range(T):
        slots_in = main if t == 0 else scratch[:, t - 1].contiguous()
        slots_out = scratch[:, t].contiguous()
        y = torch.empty(n_req, H, D, device=dev, dtype=torch.bfloat16)
        _step(pools_s, slots_in, slots_out, steps[t], weights, y)
        y_spec.append(y.clone())

    for t in range(T):
        assert torch.equal(y_spec[t], y_ref[t]), f"output mismatch at t={t}"
        rows = scratch[:, t].long()
        for name, pool, ref in zip(
            ("ssm", "k", "v", "angle"),
            pools_s,
            ckpt[t],
        ):
            assert torch.equal(pool[rows], ref), f"{name} state mismatch t={t}"

    # Main slots (and every untouched row) must be unmodified by the
    # dual-slot chain except main's first read.
    assert torch.equal(pools_s[0][main.long()], ssm0[main.long()])
    assert torch.equal(pools_s[3][main.long()], ang0[main.long()])


@requires_cuda
def test_out_equals_in_matches_inplace():
    """slots_out == slots_in must be bit-identical to plain in-place."""
    torch.manual_seed(1)
    dev = "cuda"
    B, P = 6, 11
    weights = dict(
        Dp=torch.randn(H, device=dev, dtype=torch.float32),
        xpj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        opj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        zpj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        bias_q=torch.randn(R, H, N, device=dev, dtype=torch.float32),
        bias_k=torch.randn(R, H, N, device=dev, dtype=torch.float32),
    )
    slots = torch.randperm(P, device=dev)[:B].to(torch.int32)
    ssm0 = torch.randn(P, H, D, N, device=dev, dtype=torch.float32)
    k0 = torch.randn(P, R, H, N, device=dev, dtype=torch.bfloat16)
    v0 = torch.randn(P, H, D, device=dev, dtype=torch.bfloat16)
    ang0 = torch.randn(P, H, A_HALF, device=dev, dtype=torch.float32)
    inp = _mk_inputs(B, dev)

    pools_a = (ssm0.clone(), k0.clone(), v0.clone(), ang0.clone())
    y_a = torch.empty(B, H, D, device=dev, dtype=torch.bfloat16)
    _step(pools_a, slots, None, inp, weights, y_a)

    pools_b = (ssm0.clone(), k0.clone(), v0.clone(), ang0.clone())
    y_b = torch.empty(B, H, D, device=dev, dtype=torch.bfloat16)
    _step(pools_b, slots, slots.clone(), inp, weights, y_b)

    assert torch.equal(y_a, y_b)
    for pa, pb in zip(pools_a, pools_b):
        assert torch.equal(pa, pb)


@requires_cuda
def test_pad_lane_out_slot_suppressed():
    """A padded out slot (-1) must leave every pool row untouched."""
    torch.manual_seed(2)
    dev = "cuda"
    B, P = 4, 9
    weights = dict(
        Dp=torch.randn(H, device=dev, dtype=torch.float32),
        xpj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        opj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        zpj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        bias_q=torch.randn(R, H, N, device=dev, dtype=torch.float32),
        bias_k=torch.randn(R, H, N, device=dev, dtype=torch.float32),
    )
    slots_in = torch.tensor([0, 1, 2, -1], device=dev, dtype=torch.int32)
    slots_out = torch.tensor([4, 5, -1, -1], device=dev, dtype=torch.int32)
    ssm0 = torch.randn(P, H, D, N, device=dev, dtype=torch.float32)
    k0 = torch.randn(P, R, H, N, device=dev, dtype=torch.bfloat16)
    v0 = torch.randn(P, H, D, device=dev, dtype=torch.bfloat16)
    ang0 = torch.randn(P, H, A_HALF, device=dev, dtype=torch.float32)
    pools = (ssm0.clone(), k0.clone(), v0.clone(), ang0.clone())
    y = torch.empty(B, H, D, device=dev, dtype=torch.bfloat16)
    _step(pools, slots_in, slots_out, _mk_inputs(B, dev), weights, y)

    written = {4, 5}
    untouched = [i for i in range(P) if i not in written]
    for pool, ref in zip(pools, (ssm0, k0, v0, ang0)):
        assert torch.equal(pool[untouched], ref[untouched])
    # Lane 2 (valid in, pad out): output still computed, nothing written.
    assert torch.isfinite(y[:3].float()).all()
