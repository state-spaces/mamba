# Copyright (C) 2023, Tri Dao.

import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange, repeat

from mamba_ssm.ops.triton.selective_state_update import selective_state_update, selective_state_update_ref


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("has_z", [False, True])
# @pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize("dstate", [16, 32, 64])
# @pytest.mark.parametrize("dstate", [16])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_selective_state_update(dim, dstate, has_z, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
        if torch.version.hip:
            atol *= 2
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    state = torch.randn(batch_size, dim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)
    if has_z:
        z = torch.randn_like(x)
    else:
        z = None
    state_ref = state.detach().clone()
    out = selective_state_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("has_z", [False, True])
# @pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize("tie_hdim", [False, True])
# @pytest.mark.parametrize('tie_hdim', [True])
@pytest.mark.parametrize("ngroups", [1, 2, 4])
# @pytest.mark.parametrize("ngroups", [2])
@pytest.mark.parametrize("dstate", [16, 32, 64])
# @pytest.mark.parametrize("dstate", [16])
@pytest.mark.parametrize("dim", [2048, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_selective_state_update_with_heads(dim, dstate, ngroups, has_z, tie_hdim, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 3e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-1
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    headdim = 64
    nheads = dim // headdim
    state = torch.randn(batch_size, nheads, headdim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
    if not tie_hdim:
        dt = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
        dt_bias = torch.rand(nheads, headdim, device=device) - 4.0
        A = -torch.rand(nheads, headdim, dstate, device=device) - 1.0
        D = torch.randn(nheads, headdim, device=device)
    else:
        dt = repeat(torch.randn(batch_size, nheads, device=device, dtype=itype), "b h -> b h p", p=headdim)
        dt_bias = repeat(torch.rand(nheads, device=device) - 4.0, "h -> h p", p=headdim)
        A = repeat(-torch.rand(nheads, device=device) - 1.0, "h -> h p n", p=headdim, n=dstate)
        D = repeat(torch.randn(nheads, device=device), "h -> h p", p=headdim)
    B = torch.randn(batch_size, ngroups, dstate, device=device)
    C = torch.randn(batch_size, ngroups, dstate, device=device)
    if has_z:
        z = torch.randn_like(x)
    else:
        z = None
    state_ref = state.detach().clone()
    state_og = state.detach().clone()
    out = selective_state_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("has_z", [False, True])
# @pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize("dstate", [16, 32, 64])
# @pytest.mark.parametrize("dstate", [16])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_selective_state_update_with_batch_indices(dim, dstate, has_z, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 6e-2, 6e-2
        if torch.version.hip:
            atol *= 2
    # set seed
    torch.random.manual_seed(0)
    batch_size = 16

    total_entries = 10 * batch_size
    state = torch.randn(total_entries, dim, dstate, dtype=itype, device=device)
    state_indices = torch.randperm(total_entries)[:batch_size].to(dtype=torch.int32, device=device)

    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)
    if has_z:
        z = torch.randn_like(x)
    else:
        z = None
    state_ref = state[state_indices,:].detach().clone()
    out = selective_state_update(state, x, dt, A, B, C, D=D, z=z,
                                 dt_bias=dt_bias, dt_softplus=True, state_batch_indices=state_indices)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(state[state_indices,:], state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
#@pytest.mark.parametrize('itype', [torch.float32])
@pytest.mark.parametrize("has_z", [False, True])
# @pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize("tie_hdim", [False, True])
# @pytest.mark.parametrize('tie_hdim', [True])
@pytest.mark.parametrize("ngroups", [1, 2, 4])
# @pytest.mark.parametrize("ngroups", [2])
@pytest.mark.parametrize("dstate", [16, 32, 64])
# @pytest.mark.parametrize("dstate", [16])
@pytest.mark.parametrize("dim", [2048, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_selective_state_update_with_heads_with_batch_indices(dim, dstate, ngroups, has_z, tie_hdim, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 3e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1
    # set seed
    torch.random.manual_seed(0)
    batch_size = 16
    headdim = 64
    nheads = dim // headdim

    total_entries = 10 * batch_size
    state = torch.randn(total_entries, nheads, headdim, dstate, dtype=itype, device=device)
    state_indices = torch.randperm(total_entries)[:batch_size].to(dtype=torch.int32, device=device)

    x = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
    if not tie_hdim:
        dt = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
        dt_bias = torch.rand(nheads, headdim, device=device) - 4.0
        A = -torch.rand(nheads, headdim, dstate, device=device) - 1.0
        D = torch.randn(nheads, headdim, device=device)
    else:
        dt = repeat(torch.randn(batch_size, nheads, device=device, dtype=itype), "b h -> b h p", p=headdim)
        dt_bias = repeat(torch.rand(nheads, device=device) - 4.0, "h -> h p", p=headdim)
        A = repeat(-torch.rand(nheads, device=device) - 1.0, "h -> h p n", p=headdim, n=dstate)
        D = repeat(torch.randn(nheads, device=device), "h -> h p", p=headdim)
    B = torch.randn(batch_size, ngroups, dstate, device=device)
    C = torch.randn(batch_size, ngroups, dstate, device=device)
    if has_z:
        z = torch.randn_like(x)
    else:
        z = None
    state_ref = state[state_indices,:].detach().clone()
    state_og = state[state_indices,:].detach().clone()
    out = selective_state_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True, state_batch_indices=state_indices)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(state[state_indices,:], state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
