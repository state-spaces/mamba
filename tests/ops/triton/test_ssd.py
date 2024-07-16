import math

import torch
import torch.nn.functional as F

import pytest

from einops import rearrange, repeat

from mamba_ssm.ops.triton.ssd_chunk_state import chunk_state, chunk_state_ref
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from mamba_ssm.ops.triton.ssd_chunk_state import chunk_state_varlen
from mamba_ssm.ops.triton.ssd_state_passing import state_passing, state_passing_ref
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd
from mamba_ssm.ops.triton.ssd_chunk_scan import chunk_scan, chunk_scan_ref
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_chunk_scan, ssd_chunk_scan_combined_ref, ssd_selective_scan
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined, mamba_split_conv1d_scan_ref


def detach_clone(*args):
    return tuple([arg.detach().clone().requires_grad_() if arg is not None else None for arg in args])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('ngroups', [1, 2, 8, "max"])
# @pytest.mark.parametrize('ngroups', [1])
@pytest.mark.parametrize('chunk_size', [64, 128])
# @pytest.mark.parametrize('chunk_size', [128])
def test_chunk_state_varlen(chunk_size, ngroups, dtype):
    device = 'cuda'
    rtol, atol = (1e-2, 3e-3)
    # set seed
    torch.random.manual_seed(chunk_size + (ngroups if ngroups != "max" else 64))
    batch = 300
    seqlens = torch.randint(1, 200, (batch,), device=device)
    # batch = 3
    # seqlens = torch.tensor([201, 56, 5], device=device)
    cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0))
    total_seqlen = seqlens.sum().item()
    seq_idx = torch.cat([torch.full((s,), i, dtype=torch.int32, device=device) for i, s in enumerate(seqlens)], dim=0).unsqueeze(0)
    dim = 4096
    # dim = 64
    headdim = 64
    # dim = 32
    dstate = 32
    assert dim % headdim == 0
    nheads = dim // headdim
    if ngroups == "max":
        ngroups = nheads
    assert nheads % ngroups == 0
    B = torch.randn(total_seqlen, ngroups, dstate, dtype=dtype, device=device) / 5
    x = torch.randn(total_seqlen, nheads, headdim, dtype=dtype, device=device)
    A = -0.1 * (torch.rand(nheads, device=device))
    dt = F.softplus(torch.randn(total_seqlen, nheads, device=device, dtype=torch.float32) - 4)
    dA_cumsum, dt_rounded = _chunk_cumsum_fwd(dt.unsqueeze(0), A, chunk_size)
    chunk_states = _chunk_state_fwd(B.unsqueeze(0), x.unsqueeze(0), dt_rounded, dA_cumsum, seq_idx=seq_idx)
    chunk_states, _ = _state_passing_fwd(rearrange(chunk_states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                         seq_idx=seq_idx, chunk_size=chunk_size)
    chunk_states = rearrange(chunk_states, "... (p n) -> ... p n", n=dstate)
    chunk_states = chunk_states.squeeze(0)
    dA_cumsum = dA_cumsum.squeeze(0)
    dt_rounded = dt_rounded.squeeze(0)
    out = chunk_state_varlen(B, x, dt_rounded, dA_cumsum, cu_seqlens, chunk_states)
    out_ref = []
    for b in range(batch):
        x_s = x[cu_seqlens[b]:cu_seqlens[b + 1]].unsqueeze(0)
        B_s = B[cu_seqlens[b]:cu_seqlens[b + 1]].unsqueeze(0)
        dt_s = dt[cu_seqlens[b]:cu_seqlens[b + 1]].unsqueeze(0)
        dA_cumsum_s, dt_rounded_s = _chunk_cumsum_fwd(dt_s, A, chunk_size)
        states = chunk_state(B_s, x_s, dt_rounded_s, dA_cumsum_s)
        _, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum_s[:, :, :, -1],
                                             chunk_size=chunk_size)
        final_states = rearrange(final_states, "... (p n) -> ... p n", n=dstate)
        out_ref.append(final_states)
    out_ref = torch.cat(out_ref, dim=0)
    print(f"Max diff = {(out - out_ref).abs().max().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
