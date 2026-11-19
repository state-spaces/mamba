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


@pytest.mark.parametrize("target", ["x", "B"])
@pytest.mark.parametrize("nonfinite_value", [float("inf"), float("nan")], ids=["inf", "nan"])
@pytest.mark.parametrize("len_a", [5, 16], ids=["same_chunk", "chunk_boundary"])
def test_mamba_chunk_scan_isolates_nonfinite_request(len_a, nonfinite_value, target):
    """One request's non-finite input (x or B, inf or nan) must not corrupt another
    request's output or final state in the same packed, varlen prefill.

    `len_a=5` ("same_chunk") puts both requests in chunk 0, exercising the
    chunk-local masking in `_chunk_state_fwd_kernel` / `_chunk_scan_fwd_kernel`.
    `len_a=16` ("chunk_boundary") makes request A exactly fill chunk 0
    (chunk_size is 16), so request B starts in chunk 1 and the only remaining
    contamination path is the cross-chunk state carry in
    `_state_passing_fwd_kernel` and the `prev_states` handling in
    `_chunk_scan_fwd_kernel`, which mask the same way
    (`scale = where(mask, decay, 0.0); result = scale * possibly_nonfinite + ...`)
    and are just as vulnerable to `0 * inf = nan` under IEEE 754.
    """
    device = "cuda"
    torch.manual_seed(0)
    n_heads, d_head, d_state = 2, 4, 8
    len_b = 30
    seqlen = len_a + len_b

    x = torch.ones(1, seqlen, n_heads, d_head, device=device)
    B = torch.ones(1, seqlen, 1, d_state, device=device)
    if target == "x":
        x[0, 2, 0, 1] = nonfinite_value  # only request A (the first len_a tokens)
    else:
        B[0, 2, 0, 1] = nonfinite_value
    dt = torch.ones(1, seqlen, n_heads, device=device)
    A = torch.zeros(n_heads, device=device)
    C = torch.ones(1, seqlen, 1, d_state, device=device)
    seq_idx = torch.repeat_interleave(
        torch.arange(2, dtype=torch.int32, device=device),
        torch.tensor([len_a, len_b], device=device),
    ).unsqueeze(0)
    cu_seqlens = torch.tensor([0, len_a, seqlen], dtype=torch.int32, device=device)

    out, varlen_states = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        16,  # chunk_size
        seq_idx=seq_idx,
        cu_seqlens=cu_seqlens,
        return_varlen_states=True,
    )

    victim_out = out[:, len_a:]
    states_b = varlen_states[1]
    expected = (
        d_state * torch.arange(1, len_b + 1, device=device, dtype=torch.float32)
    )[None, :, None, None].expand(1, len_b, n_heads, d_head)

    assert states_b.isfinite().all(), "request B's final state must stay finite"
    assert victim_out.isfinite().all(), "request B's output must stay finite"
    torch.testing.assert_close(victim_out, expected)
    torch.testing.assert_close(states_b, torch.full_like(states_b, float(len_b)))


def test_mamba_chunk_scan_lone_request_nonfinite_input_stays_nonfinite():
    """A single, unbatched request's own non-finite input must still surface as
    non-finite output, not be silently sanitized into a finite-but-wrong value.

    The sanitization that isolates sequences from each other only guards the
    masked-select paths that exist when `seq_idx` is given; without it there is
    one sequence per batch row and no `0 * inf` path to guard against, so the
    fix must not apply there.
    """
    device = "cuda"
    torch.manual_seed(0)
    n_heads, d_head, d_state = 2, 4, 8
    seqlen = 30

    x = torch.ones(1, seqlen, n_heads, d_head, device=device)
    x[0, 2, 0, 1] = float("inf")
    dt = torch.ones(1, seqlen, n_heads, device=device)
    A = torch.zeros(n_heads, device=device)
    B = torch.ones(1, seqlen, 1, d_state, device=device)
    C = torch.ones(1, seqlen, 1, d_state, device=device)

    out = mamba_chunk_scan_combined(x, dt, A, B, C, 16)

    assert not out.isfinite().all(), (
        "a lone request's own non-finite input must not be silently sanitized "
        "into finite-but-wrong output"
    )


@pytest.mark.parametrize("target", ["x", "B"])
@pytest.mark.parametrize("nonfinite_value", [float("inf"), float("nan")], ids=["inf", "nan"])
@pytest.mark.parametrize(
    "lens,poisoned",
    [
        pytest.param([5, 30], 0, id="two_same_chunk-first"),
        pytest.param([5, 30], 1, id="two_same_chunk-second"),
        pytest.param([16, 30], 0, id="chunk_boundary-first"),
        pytest.param([16, 30], 1, id="chunk_boundary-second"),
        pytest.param([3, 4, 9, 30], 0, id="three_in_chunk-first"),
        pytest.param([3, 4, 9, 30], 1, id="three_in_chunk-middle"),
        pytest.param([3, 4, 9, 30], 2, id="three_in_chunk-last"),
        pytest.param([3, 4, 9, 30], 3, id="three_in_chunk-after"),
    ],
)
def test_mamba_chunk_scan_nonfinite_stays_with_its_own_request(lens, poisoned, nonfinite_value, target):
    """A request's own non-finite input must still reach its own output and final
    state, while every other request in the packed batch matches its standalone run.

    Isolating requests by dropping non-finite values from a shared chunk would hide a
    request's own `nan` from itself (finite, wrong output instead of `nan`).
    """
    device = "cuda"
    n_heads, d_head, d_state, chunk_size = 2, 4, 8, 16
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)
    seqlen = starts[-1]
    pos = starts[poisoned] + 2  # the third token of the poisoned request

    x = torch.ones(1, seqlen, n_heads, d_head, device=device)
    B = torch.ones(1, seqlen, 1, d_state, device=device)
    if target == "x":
        x[0, pos, 0, 1] = nonfinite_value
    else:
        B[0, pos, 0, 1] = nonfinite_value
    dt = torch.ones(1, seqlen, n_heads, device=device)
    A = torch.zeros(n_heads, device=device)
    C = torch.ones(1, seqlen, 1, d_state, device=device)
    seq_idx = torch.repeat_interleave(
        torch.arange(len(lens), dtype=torch.int32, device=device),
        torch.tensor(lens, device=device),
    ).unsqueeze(0)
    cu_seqlens = torch.tensor(starts, dtype=torch.int32, device=device)

    out, varlen_states = mamba_chunk_scan_combined(
        x, dt, A, B, C, chunk_size,
        seq_idx=seq_idx, cu_seqlens=cu_seqlens, return_varlen_states=True,
    )

    for i, n in enumerate(lens):
        seq_out = out[0, starts[i]:starts[i + 1]]
        state = varlen_states[i]
        if i != poisoned:
            expected = (d_state * torch.arange(1, n + 1, device=device, dtype=torch.float32))
            expected = expected[:, None, None].expand(n, n_heads, d_head)
            assert seq_out.isfinite().all(), f"request {i}'s output must stay finite"
            assert state.isfinite().all(), f"request {i}'s final state must stay finite"
            torch.testing.assert_close(seq_out, expected)
            torch.testing.assert_close(state, torch.full_like(state, float(n)))
        else:
            rel = pos - starts[i]
            own = seq_out[rel:, 0, 1] if target == "x" else seq_out[rel:]
            assert (~own.isfinite()).all(), (
                "the poisoned request's own output must stay non-finite from its non-finite token on"
            )
            assert not state.isfinite().all(), "the poisoned request's own final state must stay non-finite"
