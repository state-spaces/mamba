# Copyright (c) 2026, Tri Dao.
"""Parity: paged state pools (``state_batch_indices``) vs the torch reference.

The indexed path reads/updates rows of a larger state pool through an
arbitrary (non-contiguous, unsorted) index vector. This pins it against
``selective_state_update_fused_ref_v2`` — an independent pure-PyTorch oracle
— rather than against the dense kernel: for every pooled row the kernel must
produce the same output and post-step SSM state as the reference evaluated
on the gathered per-request states, and rows the batch does not reference
must remain untouched.
"""
import pytest
import torch

from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import (
    mamba3_step_fn,
    selective_state_update_fused_ref_v2,
)

H, D, N, R = 48, 64, 128, 4
P = 53  # pool rows, deliberately larger than any batch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _scattered_slots(B: int, seed: int) -> torch.Tensor:
    """Non-contiguous, unsorted, gap-ridden slot assignment into the pool."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(P, generator=g)[:B].to(device="cuda", dtype=torch.int32)


@requires_cuda
@pytest.mark.parametrize("B", [1, 5, 24])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("update_kv", [False, True])
def test_pool_indexed_matches_reference(B, state_dtype, update_kv):
    torch.manual_seed(0)
    dev = "cuda"
    slots = _scattered_slots(B, seed=B)
    if B > 1:  # truly non-contiguous slot layout
        assert not torch.all(torch.diff(slots.long()) == 1)

    ssm_pool = torch.randn(P, H, D, N, device=dev, dtype=state_dtype)
    k_pool = torch.randn(P, R, H, N, device=dev, dtype=torch.bfloat16)
    v_pool = torch.randn(P, H, D, device=dev, dtype=torch.bfloat16)
    ssm0, k0, v0 = ssm_pool.clone(), k_pool.clone(), v_pool.clone()

    A = -torch.rand(B, H, device=dev, dtype=torch.float32)
    Bt = torch.randn(B, R, H, N, device=dev, dtype=torch.bfloat16)
    Ct = torch.randn(B, R, H, N, device=dev, dtype=torch.bfloat16)
    Dp = torch.randn(H, device=dev, dtype=torch.float32)
    x = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    dt = torch.rand(B, H, device=dev, dtype=torch.float32)
    trap = torch.rand(B, H, device=dev, dtype=torch.float32)
    xpj = torch.randn(R, H, D, device=dev, dtype=torch.bfloat16)
    opj = torch.randn(R, H, D, device=dev, dtype=torch.bfloat16)
    z = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    zpj = torch.randn(R, H, D, device=dev, dtype=torch.bfloat16)
    y = torch.empty(B, H, D, device=dev, dtype=torch.bfloat16)

    # Torch reference on the gathered (dense) per-request states.
    rows = slots.long()
    ref_out, ref_state = selective_state_update_fused_ref_v2(
        ssm0[rows], A, Bt, Ct, xpj, x, zpj, z, dt,
        k0[rows], v0[rows], trap, Dp, opj,
    )

    mamba3_step_fn(
        ssm_pool, k_pool, v_pool, A, Bt, Ct, Dp, x, dt, trap, xpj, opj,
        None, y, z=z, zproj=zpj,
        state_batch_indices=slots,
        update_kv_state=update_kv,
        tile_D=64, num_warps=4,
    )
    if not update_kv:
        k_pool[rows] = Bt
        v_pool[rows] = x
    torch.cuda.synchronize()

    # Output: bf16 storage of an fp32 accumulation; the reference accumulates
    # in fp32 with a different reduction order.
    out_scale = ref_out.float().abs().max().clamp_min(1.0)
    assert (y.float() - ref_out.float()).abs().max() / out_scale < 2e-2

    # Post-step SSM state at the scattered rows.
    st = ssm_pool[rows].float()
    st_scale = ref_state.float().abs().max().clamp_min(1.0)
    tol = 2e-2 if state_dtype == torch.bfloat16 else 2e-3
    assert (st - ref_state.float()).abs().max() / st_scale < tol

    # New key/value states are this step's B/x, at the scattered rows.
    assert torch.equal(k_pool[rows], Bt)
    assert torch.equal(v_pool[rows], x)

    # Every row the batch does not reference is bit-untouched.
    untouched = torch.ones(P, dtype=torch.bool)
    untouched[rows.cpu()] = False
    for pool, ref in ((ssm_pool, ssm0), (k_pool, k0), (v_pool, v0)):
        assert torch.equal(pool[untouched], ref[untouched])


@requires_cuda
def test_pool_indexed_descending_slots():
    """Order-independence: reversed (descending) slots give per-request
    results identical to the same requests with ascending slots."""
    torch.manual_seed(1)
    dev = "cuda"
    B = 8
    asc = torch.arange(3, 3 + 2 * B, 2, device=dev, dtype=torch.int32)
    desc = asc.flip(0)

    # One pool + one batch; run twice with the batch order permuted so
    # request i targets the same slot both times.
    ssm0 = torch.randn(P, H, D, N, device=dev, dtype=torch.float32)
    k0 = torch.randn(P, R, H, N, device=dev, dtype=torch.bfloat16)
    v0 = torch.randn(P, H, D, device=dev, dtype=torch.bfloat16)
    inp = dict(
        A=-torch.rand(B, H, device=dev, dtype=torch.float32),
        Bt=torch.randn(B, R, H, N, device=dev, dtype=torch.bfloat16),
        Ct=torch.randn(B, R, H, N, device=dev, dtype=torch.bfloat16),
        Dp=torch.randn(H, device=dev, dtype=torch.float32),
        x=torch.randn(B, H, D, device=dev, dtype=torch.bfloat16),
        dt=torch.rand(B, H, device=dev, dtype=torch.float32),
        trap=torch.rand(B, H, device=dev, dtype=torch.float32),
        xpj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        opj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        z=torch.randn(B, H, D, device=dev, dtype=torch.bfloat16),
        zpj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
    )

    def step(slots, order):
        ssm, k, v = ssm0.clone(), k0.clone(), v0.clone()
        y = torch.empty(B, H, D, device=dev, dtype=torch.bfloat16)
        mamba3_step_fn(
            ssm, k, v,
            inp["A"][order], inp["Bt"][order], inp["Ct"][order], inp["Dp"],
            inp["x"][order], inp["dt"][order], inp["trap"][order],
            inp["xpj"], inp["opj"], None, y,
            z=inp["z"][order], zproj=inp["zpj"],
            state_batch_indices=slots,
            update_kv_state=True,
            tile_D=64, num_warps=4,
        )
        return ssm, k, v, y

    fwd = torch.arange(B, device=dev)
    rev = fwd.flip(0)
    ssm_a, k_a, v_a, y_a = step(asc, fwd)
    ssm_d, k_d, v_d, y_d = step(desc, rev)

    assert torch.equal(ssm_a, ssm_d)
    assert torch.equal(k_a, k_d)
    assert torch.equal(v_a, v_d)
    assert torch.equal(y_a, y_d[rev])
