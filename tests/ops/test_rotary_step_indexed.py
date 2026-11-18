# Copyright (c) 2025, Tri Dao.
"""Parity for apply_rotary_qk_inference_fwd's state_batch_indices mode:
indexed in-place pool update must match gather -> kernel -> scatter, with
negative (padding) indices skipping the token (q/k zeroed, pool untouched)."""
import pytest
import torch

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import (
    apply_rotary_qk_inference_fwd,
)

H, D, R = 48, 128, 4
ROT_HALF = 32  # rotary_dim // 2
P = 37

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@requires_cuda
@pytest.mark.parametrize("B,npad", [(1, 0), (5, 0), (24, 0), (8, 3), (24, 8)])
def test_indexed_rotary_matches_gather_scatter(B: int, npad: int):
    torch.manual_seed(0)
    dev = "cuda"
    real = B - npad
    slots = torch.randperm(P, device=dev)[:B].to(torch.int32)
    slots[real:] = -1

    angle_pool = torch.randn(P, H, ROT_HALF, device=dev, dtype=torch.float32)
    q = torch.randn(B, R, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, R, H, D, device=dev, dtype=torch.bfloat16)
    angle_proj = torch.randn(B, H, ROT_HALF, device=dev, dtype=torch.float32)
    dt = torch.rand(B, H, device=dev, dtype=torch.float32)
    bias_q = torch.randn(R, H, D, device=dev, dtype=torch.float32)
    bias_k = torch.randn(R, H, D, device=dev, dtype=torch.float32)

    # Reference: gather -> kernel -> scatter (real lanes only).
    rs = slots[:real].long()
    pool_ref = angle_pool.clone()
    q_ref, k_ref, nxt = apply_rotary_qk_inference_fwd(
        q=q[:real], k=k[:real], angle_state=pool_ref[rs],
        angle_proj=angle_proj[:real], dt=dt[:real],
        bias_q=bias_q, bias_k=bias_k,
        conjugate=False, inplace=False, rotate_pairwise=False)
    pool_ref[rs] = nxt

    # Indexed: pool + slot indices, in-place.
    pool_new = angle_pool.clone()
    q_new, k_new, ret = apply_rotary_qk_inference_fwd(
        q=q, k=k, angle_state=pool_new,
        angle_proj=angle_proj, dt=dt,
        bias_q=bias_q, bias_k=bias_k,
        conjugate=False, inplace=False, rotate_pairwise=False,
        state_batch_indices=slots)
    torch.cuda.synchronize()

    assert ret is pool_new  # in-place pool return
    assert torch.equal(q_new[:real], q_ref)
    assert torch.equal(k_new[:real], k_ref)
    if npad:
        assert torch.equal(q_new[real:], torch.zeros_like(q_new[real:]))
        assert torch.equal(k_new[real:], torch.zeros_like(k_new[real:]))
    assert torch.equal(pool_new, pool_ref)
    mask = torch.ones(P, dtype=torch.bool, device=dev)
    mask[rs] = False
    assert torch.equal(pool_new[mask], angle_pool[mask])
