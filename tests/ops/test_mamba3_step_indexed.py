# Copyright (c) 2025, Tri Dao.
"""Parity tests for mamba3_step_fn's indexed state-pool mode.

The indexed path (``state_indices``) must be byte-identical to the
gather -> dense kernel -> scatter reference, including:
- negative (PAD_SLOT_ID) indices from CUDA-graph batch padding,
- untouched pool rows staying untouched,
- ``update_kv_state=True`` storing the new B/x states exactly as the
  caller-side scatter would.
"""
import pytest
import torch

from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn

H, D, N, R = 48, 64, 128, 4
P = 37  # pool rows

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _rand_inputs(B, state_dtype=torch.float32):
    dev = "cuda"
    ssm_pool = torch.randn(P, H, D, N, device=dev, dtype=state_dtype)
    k_pool = torch.randn(P, R, H, N, device=dev, dtype=torch.bfloat16)
    v_pool = torch.randn(P, H, D, device=dev, dtype=torch.bfloat16)
    return dict(
        ssm_pool=ssm_pool, k_pool=k_pool, v_pool=v_pool,
        A=-torch.rand(B, H, device=dev, dtype=torch.float32),
        B=torch.randn(B, R, H, N, device=dev, dtype=torch.bfloat16),
        C=torch.randn(B, R, H, N, device=dev, dtype=torch.bfloat16),
        Dp=torch.randn(H, device=dev, dtype=torch.float32),
        x=torch.randn(B, H, D, device=dev, dtype=torch.bfloat16),
        dt=torch.rand(B, H, device=dev, dtype=torch.float32),
        trap=torch.rand(B, H, device=dev, dtype=torch.float32),
        xpj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        opj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
        z=torch.randn(B, H, D, device=dev, dtype=torch.bfloat16),
        zpj=torch.randn(R, H, D, device=dev, dtype=torch.bfloat16),
    )


@requires_cuda
@pytest.mark.parametrize("B,npad", [(1, 0), (5, 0), (32, 0), (8, 3), (24, 8)])
@pytest.mark.parametrize("update_kv", [False, True])
def test_indexed_matches_gather_scatter(B: int, npad: int, update_kv: bool):
    torch.manual_seed(0)
    t = _rand_inputs(B)
    real = B - npad
    slots = torch.randperm(P, device="cuda")[:B].to(torch.int32)
    slots[real:] = -1  # PAD_SLOT_ID lanes
    rs = slots[:real].long()

    # Reference: gather -> dense kernel -> scatter (real lanes only).
    pool_ref = t["ssm_pool"].clone()
    k_ref, v_ref = t["k_pool"].clone(), t["v_pool"].clone()
    st = pool_ref[rs]
    st_out = torch.empty_like(st)
    y_ref = torch.empty_like(t["x"][:real])
    mamba3_step_fn(st, k_ref[rs], v_ref[rs], t["A"][:real], t["B"][:real],
                   t["C"][:real], t["Dp"], t["x"][:real], t["dt"][:real],
                   t["trap"][:real], t["xpj"], t["opj"], st_out, y_ref,
                   z=t["z"][:real], zproj=t["zpj"], tile_D=64, num_warps=4)
    pool_ref[rs] = st_out
    k_ref[rs] = t["B"][:real]  # caller-side kv update (old semantics)
    v_ref[rs] = t["x"][:real]

    # Indexed: in-place pool update via slot indices.
    pool_new = t["ssm_pool"].clone()
    k_new, v_new = t["k_pool"].clone(), t["v_pool"].clone()
    y_new = torch.empty_like(t["x"])
    mamba3_step_fn(pool_new, k_new, v_new, t["A"], t["B"], t["C"], t["Dp"],
                   t["x"], t["dt"], t["trap"], t["xpj"], t["opj"], None,
                   y_new, z=t["z"], zproj=t["zpj"], state_batch_indices=slots,
                   update_kv_state=update_kv, tile_D=64, num_warps=4)
    if not update_kv:
        k_new[torch.where(slots >= 0, slots, 0).long()[:real]] = t["B"][:real]
        v_new[torch.where(slots >= 0, slots, 0).long()[:real]] = t["x"][:real]
    torch.cuda.synchronize()

    assert torch.equal(y_new[:real], y_ref)
    # Padding lanes produce zeroed outputs (selective_state_update semantics).
    if npad:
        assert torch.equal(y_new[real:], torch.zeros_like(y_new[real:]))
    assert torch.equal(pool_new, pool_ref)
    assert torch.equal(k_new, k_ref)
    assert torch.equal(v_new, v_ref)
    # untouched pool rows must be untouched
    mask = torch.ones(P, dtype=torch.bool, device="cuda")
    mask[rs] = False
    assert torch.equal(pool_new[mask], t["ssm_pool"][mask])


@requires_cuda
def test_update_kv_state_requires_single_d_tile():
    torch.manual_seed(0)
    t = _rand_inputs(4)
    slots = torch.arange(4, device="cuda", dtype=torch.int32)
    y = torch.empty_like(t["x"])
    with pytest.raises(AssertionError, match="single D-tile"):
        mamba3_step_fn(t["ssm_pool"], t["k_pool"], t["v_pool"], t["A"],
                       t["B"], t["C"], t["Dp"], t["x"], t["dt"], t["trap"],
                       t["xpj"], t["opj"], None, y, z=t["z"], zproj=t["zpj"],
                       state_batch_indices=slots, update_kv_state=True,
                       tile_D=32, num_warps=4)
