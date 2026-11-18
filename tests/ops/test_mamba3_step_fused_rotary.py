# Copyright (c) 2025, Tri Dao.
"""Parity: mamba3_step_fn fused bias+rotary vs the two-kernel reference
(apply_rotary_qk_inference_fwd -> mamba3_step_fn).

Tolerances are bf16-level: the two paths use different cos/sin backends
(Triton libdevice vs CuteDSL), so bit-identity is not expected; everything
else (single bf16 rounding of the rotated values, fp32 state math) matches.
"""
import pytest
import torch

from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import (
    apply_rotary_qk_inference_fwd,
)

H, D, N, R = 48, 64, 128, 4
ROT, A_HALF = 64, 32
P = 37

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@requires_cuda
@pytest.mark.parametrize("B,npad", [(1, 0), (5, 0), (24, 0), (8, 3), (24, 8)])
def test_fused_rotary_matches_two_kernel_reference(B: int, npad: int):
    torch.manual_seed(0)
    dev = "cuda"
    real = B - npad
    slots = torch.randperm(P, device=dev)[:B].to(torch.int32)
    slots[real:] = -1

    ssm_pool = torch.randn(P, H, D, N, device=dev, dtype=torch.float32)
    k_pool = torch.randn(P, R, H, N, device=dev, dtype=torch.bfloat16)
    v_pool = torch.randn(P, H, D, device=dev, dtype=torch.bfloat16)
    angle_pool = torch.randn(P, H, A_HALF, device=dev, dtype=torch.float32)

    # Pre-rotation B/C: shared across heads (Dragon: ngroups == 1).
    B_pre = torch.randn(B, R, 1, N, device=dev, dtype=torch.bfloat16)
    C_pre = torch.randn(B, R, 1, N, device=dev, dtype=torch.bfloat16)
    bias_q = torch.randn(R, H, N, device=dev, dtype=torch.float32)
    bias_k = torch.randn(R, H, N, device=dev, dtype=torch.float32)
    angle_proj = torch.randn(B, 1, A_HALF, device=dev, dtype=torch.float32)

    A = -torch.rand(B, H, device=dev, dtype=torch.float32)
    Dp = torch.randn(H, device=dev, dtype=torch.float32)
    x = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    dt = torch.rand(B, H, device=dev, dtype=torch.float32)
    trap = torch.rand(B, H, device=dev, dtype=torch.float32)
    xpj = torch.randn(R, H, D, device=dev, dtype=torch.bfloat16)
    opj = torch.randn(R, H, D, device=dev, dtype=torch.bfloat16)
    zz = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    zpj = torch.randn(R, H, D, device=dev, dtype=torch.bfloat16)

    B_exp = B_pre.expand(-1, -1, H, -1)
    C_exp = C_pre.expand(-1, -1, H, -1)
    ap_exp = angle_proj.expand(-1, H, -1)

    # --- reference: rotary kernel then step kernel --------------------------
    pool_r, k_r, v_r = ssm_pool.clone(), k_pool.clone(), v_pool.clone()
    ang_r = angle_pool.clone()
    C_rot, B_rot, _ = apply_rotary_qk_inference_fwd(
        q=C_exp, k=B_exp, angle_state=ang_r, angle_proj=ap_exp, dt=dt,
        bias_q=bias_q, bias_k=bias_k, conjugate=False, inplace=False,
        rotate_pairwise=False, state_batch_indices=slots)
    y_r = torch.empty_like(x)
    mamba3_step_fn(pool_r, k_r, v_r, A, B_rot.to(torch.bfloat16),
                   C_rot.to(torch.bfloat16), Dp, x, dt, trap, xpj, opj,
                   None, y_r, z=zz, zproj=zpj, state_batch_indices=slots,
                   update_kv_state=True, tile_D=64, num_warps=4)

    # --- fused: single step kernel ------------------------------------------
    pool_f, k_f, v_f = ssm_pool.clone(), k_pool.clone(), v_pool.clone()
    ang_f = angle_pool.clone()
    y_f = torch.empty_like(x)
    mamba3_step_fn(pool_f, k_f, v_f, A, B_exp, C_exp, Dp, x, dt, trap,
                   xpj, opj, None, y_f, z=zz, zproj=zpj,
                   state_batch_indices=slots, update_kv_state=True,
                   rotary_dim=ROT, rotary_bias_q=bias_q,
                   rotary_bias_k=bias_k, rotary_angle_proj=ap_exp,
                   rotary_angle_state=ang_f, tile_D=64, num_warps=4)
    torch.cuda.synchronize()

    assert torch.allclose(ang_f, ang_r, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_f.float(), k_r.float(), atol=2e-2, rtol=2e-2)
    assert torch.equal(v_f, v_r)
    assert torch.allclose(pool_f, pool_r, atol=2e-2, rtol=2e-2)
    assert torch.allclose(y_f.float(), y_r.float(), atol=4e-2, rtol=4e-2)
    if npad:
        assert torch.equal(y_f[real:], torch.zeros_like(y_f[real:]))
    # untouched pool rows untouched
    mask = torch.ones(P, dtype=torch.bool, device=dev)
    mask[slots[:real].long()] = False
    assert torch.equal(pool_f[mask], ssm_pool[mask])
    assert torch.equal(ang_f[mask], angle_pool[mask])
