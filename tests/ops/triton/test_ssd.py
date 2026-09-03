import math

import torch
import torch.nn.functional as F

import pytest

from einops import rearrange

from mamba_ssm.ops.triton.ssd_chunk_state import (
    _chunk_cumsum_bwd,
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
    chunk_state,
    chunk_state_varlen,
)
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd, _state_passing_bwd
from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_bwd
from mamba_ssm.ops.triton import ssd_combined
from mamba_ssm.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined,
    mamba_split_conv1d_scan_combined,
    _mamba_chunk_scan_combined_fwd,
    _mamba_chunk_scan_combined_bwd,
    ensure_stride,
    causal_conv1d_bwd_function,
)

from overflow_test_utils import gpu_memory_skipif, wide_noncontiguous_slices


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
    rtol, atol = (1e-2, 3e-3) if dtype != torch.bfloat16 else (1e-2, 6e-3)
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


@gpu_memory_skipif(9)
def test_chunk_cumsum_fwd_noncontiguous_wide_view_no_overflow_known_answer() -> None:
    # Same overflow as test_chunk_cumsum_fwd_bwd_noncontiguous_wide_view_no_overflow
    # below, but instead of comparing against a second (contiguous) kernel
    # run, use inputs where the correct output is known in closed form.
    # With dt_softplus=False, no bias, and the default dt_limit=(0, inf),
    # _chunk_cumsum_fwd reduces to: dt_out = dt, dA_cumsum[..., k] =
    # A[h] * cumsum_k(dt). A constant dt (e.g. all 1s) is a weak check: a
    # wrapped-but-in-bounds read landing on the wrong (h, k) would still
    # dereference the same constant value by chance and go undetected.
    # Instead, set dt to the within-chunk position j = 0..chunk_size-1
    # (repeating every chunk), so dt_out[..., k] = k and
    # dA_cumsum[b, h, c, k] = A[h] * sum_{j=0}^{k} j = A[h] * k * (k+1) / 2 --
    # every (h, k) has a distinct expected value, so a misaddressed read
    # is caught even if it lands in-bounds.
    device = 'cuda'
    seqlen = 115_866
    nheads = 128
    chunk_size = 128
    parent_width = 18_560  # same overflow margin as the sibling test below

    A = torch.arange(1, nheads + 1, dtype=torch.float32, device=device)

    parent = torch.zeros((1, seqlen, parent_width), dtype=torch.float32, device=device)
    dt = parent[:, :, -nheads:]
    position = torch.arange(seqlen, device=device, dtype=torch.float32) % chunk_size
    dt.copy_(position[None, :, None])
    assert not dt.is_contiguous()
    nchunks = math.ceil(seqlen / chunk_size)
    assert (nchunks - 1) * chunk_size * dt.stride(1) > 2**31 - 1, \
        "test parameters too small to overflow the pid_c term"

    with torch.no_grad():
        dA_cumsum, dt_out = _chunk_cumsum_fwd(dt, A, chunk_size)
    torch.cuda.synchronize(device)

    last_chunk_len = seqlen - (nchunks - 1) * chunk_size
    j = torch.arange(chunk_size, dtype=torch.float32, device=device)
    expected_dt_out = j[None, None, None, :].expand(1, nheads, nchunks, chunk_size).clone()
    expected_dt_out[:, :, -1, last_chunk_len:] = 0.0  # padding past seqlen in the last (partial) chunk
    torch.testing.assert_close(dt_out, expected_dt_out, rtol=0, atol=0)

    triangular = j * (j + 1) / 2  # sum_{i=0}^{j} i
    expected_dA_cumsum = (A[None, :, None, None] * triangular[None, None, None, :]).expand(1, nheads, nchunks, chunk_size).clone()
    # Padding past seqlen in the last (partial) chunk: dt there is masked to 0
    # before the cumsum, so the running sum plateaus at its last real value
    # instead of resetting to 0.
    last_triangular = (last_chunk_len - 1) * last_chunk_len / 2
    expected_dA_cumsum[:, :, -1, last_chunk_len:] = (A * last_triangular)[None, :, None]
    torch.testing.assert_close(dA_cumsum, expected_dA_cumsum, rtol=1e-4, atol=1e-4)


def test_chunk_cumsum_fwd_bwd_noncontiguous_wide_view_no_overflow() -> None:
    # int32 overflow in _chunk_cumsum_fwd_kernel/_chunk_cumsum_bwd_kernel,
    # see https://github.com/state-spaces/mamba/issues/1015. Needs dt as a non-contiguous view
    # into a much wider parent tensor (large stride(1)) to trigger.
    #
    # isfinite() can't reliably catch this: a wrapped offset can land on
    # another in-bounds address and read finite-but-wrong values. Compare
    # against the same kernel run on a contiguous copy of the same values
    # instead (same pattern as the batch-axis variant below).
    device = 'cuda'
    torch.manual_seed(0)
    seqlen = 115_866
    nheads = 128
    chunk_size = 128
    parent_width = 18_560  # wide enough that (nchunks - 1) * chunk_size * stride(1) overflows int32

    A = -torch.exp(torch.randn(nheads, dtype=torch.float32, device=device))
    dt_bias = torch.randn(nheads, dtype=torch.float32, device=device)

    parent = torch.zeros((1, seqlen, parent_width), dtype=torch.bfloat16, device=device)
    dt = parent[:, :, -nheads:]
    assert not dt.is_contiguous()
    nchunks = math.ceil(seqlen / chunk_size)
    assert (nchunks - 1) * chunk_size * dt.stride(1) > 2**31 - 1, \
        "test parameters too small to overflow the pid_c term"
    dt_c = dt.contiguous()

    with torch.no_grad():
        dA_cumsum, dt_out = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=True)
        dA_cumsum_c, dt_out_c = _chunk_cumsum_fwd(dt_c, A, chunk_size, dt_bias=dt_bias, dt_softplus=True)
    torch.cuda.synchronize(device)

    assert dA_cumsum.shape == (1, nheads, nchunks, chunk_size)
    assert dt_out.shape == (1, nheads, nchunks, chunk_size)
    assert torch.isfinite(dA_cumsum).all()
    torch.testing.assert_close(dA_cumsum, dA_cumsum_c, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dt_out, dt_out_c, rtol=1e-4, atol=1e-4)

    with torch.no_grad():
        ddA = torch.randn_like(dA_cumsum)
        ddt_out = torch.randn_like(dt_out)
        ddt, dA, ddt_bias = _chunk_cumsum_bwd(ddA, ddt_out, dt, A, dt_bias=dt_bias, dt_softplus=True)
        ddt_c, dA_c, ddt_bias_c = _chunk_cumsum_bwd(ddA, ddt_out, dt_c, A, dt_bias=dt_bias, dt_softplus=True)
    torch.cuda.synchronize(device)

    assert ddt.shape == dt.shape
    assert dA.shape == (nheads,)
    assert torch.isfinite(ddt).all()
    torch.testing.assert_close(ddt, ddt_c, rtol=1e-4, atol=1e-4)
    # Slightly looser: dA is summed over all chunks, and summing in a
    # different order (contiguous vs. wide-sliced memory layout) causes tiny
    # float roundoff differences at the ~1e-3 level even for a correct kernel.
    torch.testing.assert_close(dA, dA_c, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ddt_bias, ddt_bias_c, rtol=1e-3, atol=1e-3)


@gpu_memory_skipif(11)
def test_chunk_cumsum_fwd_noncontiguous_wide_view_batch_axis_no_overflow_known_answer() -> None:
    # Batch-axis counterpart to
    # test_chunk_cumsum_fwd_noncontiguous_wide_view_no_overflow_known_answer:
    # targets `pid_b * stride_dt_batch` instead of the pid_c term, only
    # exercised at batch > 1. Give each batch its own additive offset in dt
    # (BATCH_OFFSET * b) on top of the same within-chunk position used
    # there, so a wrapped read that lands on the wrong batch produces a
    # value offset by a distinguishable multiple of BATCH_OFFSET rather
    # than one that could coincidentally match.
    device = 'cuda'

    batch = 8
    seqlen = 8_192
    nheads = 128
    chunk_size = 128
    parent_width = 40_960  # (batch - 1) * seqlen * parent_width > 2**31 - 1, with ~9% margin
    BATCH_OFFSET = 1000.0

    A = torch.arange(1, nheads + 1, dtype=torch.float32, device=device)

    parent = torch.zeros((batch, seqlen, parent_width), dtype=torch.float32, device=device)
    dt = parent[:, :, -nheads:]
    position = torch.arange(seqlen, device=device, dtype=torch.float32) % chunk_size
    batch_term = torch.arange(batch, device=device, dtype=torch.float32) * BATCH_OFFSET
    dt.copy_((position[None, :] + batch_term[:, None])[:, :, None])
    assert not dt.is_contiguous()
    assert dt.stride(0) * (batch - 1) > 2**31 - 1, "test parameters too small to overflow the batch-axis term"

    with torch.no_grad():
        dA_cumsum, dt_out = _chunk_cumsum_fwd(dt, A, chunk_size)
    torch.cuda.synchronize(device)

    nchunks = math.ceil(seqlen / chunk_size)
    assert nchunks * chunk_size == seqlen, "test assumes seqlen is a whole number of chunks (no padding term below)"

    j = torch.arange(chunk_size, dtype=torch.float32, device=device)
    expected_dt_out = (j[None, :] + batch_term[:, None])[:, None, None, :].expand(batch, nheads, nchunks, chunk_size)
    torch.testing.assert_close(dt_out, expected_dt_out, rtol=0, atol=0)

    # sum_{i=0}^{k} (i + b * BATCH_OFFSET) = k(k+1)/2 + (k+1) * b * BATCH_OFFSET
    triangular = j * (j + 1) / 2
    per_k = triangular[None, :] + (j[None, :] + 1) * batch_term[:, None]  # (batch, chunk_size)
    expected_dA_cumsum = (A[None, :, None, None] * per_k[:, None, None, :]).expand(batch, nheads, nchunks, chunk_size)
    torch.testing.assert_close(dA_cumsum, expected_dA_cumsum, rtol=1e-4, atol=1e-4)


@gpu_memory_skipif(6)
def test_chunk_cumsum_fwd_bwd_noncontiguous_wide_view_batch_axis_no_overflow() -> None:
    # Distinct overflow term in the same two kernels: `pid_b * stride_dt_batch`,
    # separate from the pid_c term above and only exercised at batch > 1 (the
    # test above uses batch=1, so pid_b is always 0). See https://github.com/state-spaces/mamba/issues/1015.
    # batch=8 here (vs. the batch=4 real-world crash) trades a larger batch
    # for a narrower/shorter parent tensor at the same overflow margin, to
    # keep memory down -- see (batch - 1) * seqlen * parent_width below.
    #
    # isfinite() can't reliably catch this: a wrapped offset can land on
    # another in-bounds address and read finite-but-wrong values. Compare
    # against the same kernel run on a contiguous copy of the same values
    # instead.
    device = 'cuda'

    torch.manual_seed(0)
    batch = 8
    seqlen = 8_192
    nheads = 128
    chunk_size = 128
    parent_width = 40_960  # (batch - 1) * seqlen * parent_width > 2**31 - 1, with ~9% margin
    assert parent_width >= nheads

    dt, = wide_noncontiguous_slices(device, seqlen, parent_width, [(nheads,)], batch=batch)
    assert not dt.is_contiguous()
    assert dt.stride(0) * (batch - 1) > 2**31 - 1, "test parameters too small to overflow the batch-axis term"
    dt_c = dt.contiguous()

    A = -torch.exp(torch.randn(nheads, dtype=torch.float32, device=device))
    dt_bias = torch.randn(nheads, dtype=torch.float32, device=device)

    with torch.no_grad():
        dA_cumsum, dt_out = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=True)
        dA_cumsum_c, dt_out_c = _chunk_cumsum_fwd(dt_c, A, chunk_size, dt_bias=dt_bias, dt_softplus=True)
    torch.cuda.synchronize(device)

    nchunks = math.ceil(seqlen / chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert torch.isfinite(dA_cumsum).all()
    torch.testing.assert_close(dA_cumsum, dA_cumsum_c, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dt_out, dt_out_c, rtol=1e-4, atol=1e-4)

    with torch.no_grad():
        ddA = torch.randn_like(dA_cumsum)
        ddt_out = torch.randn_like(dt_out)
        ddt, dA, ddt_bias = _chunk_cumsum_bwd(ddA, ddt_out, dt, A, dt_bias=dt_bias, dt_softplus=True)
        ddt_c, dA_c, ddt_bias_c = _chunk_cumsum_bwd(ddA, ddt_out, dt_c, A, dt_bias=dt_bias, dt_softplus=True)
    torch.cuda.synchronize(device)

    assert ddt.shape == dt.shape
    assert torch.isfinite(ddt).all()
    torch.testing.assert_close(ddt, ddt_c, rtol=1e-4, atol=1e-4)
    # Slightly looser: dA is summed over all chunks, and summing in a
    # different order (contiguous vs. wide-sliced memory layout) causes tiny
    # float roundoff differences at the ~1e-3 level even for a correct kernel.
    torch.testing.assert_close(dA, dA_c, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ddt_bias, ddt_bias_c, rtol=1e-3, atol=1e-3)


@gpu_memory_skipif(9)
def test_chunk_state_varlen_noncontiguous_wide_view_no_overflow_known_answer() -> None:
    # Known-answer counterpart to the sibling test below. chunk_state_varlen
    # recombines the inter-chunk chunk_states with an intra-chunk running
    # sum up to a per-sequence end_idx loaded from cu_seqlens -- the same
    # underlying recurrence as mamba_chunk_scan_combined's state (see its
    # known-answer test above), just returning the raw final SSM state
    # instead of a C-projected output. With A = 0 (no decay), dt = 1,
    # x = B = 1: the state at the end of a sequence of length L is exactly
    # L (every head/headdim/dstate position accumulates the same running
    # count). Here cu_seqlens = [0, total_seqlen] (one sequence spanning
    # the whole call), so out[0, h, p, n] = total_seqlen exactly --
    # a wrong end_idx (from a misaddressed cu_seqlens load) would give some
    # other, wrong count instead, still exact and still distinguishable.
    device = 'cuda'

    nheads = 8
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 906  # (nchunks - 1) * chunk_size * 18_560 > 2**31 - 1
    total_seqlen = nchunks * chunk_size
    cu_seqlens = torch.tensor([0, total_seqlen], device=device, dtype=torch.int32)
    parent_width = 18_560

    parent = torch.zeros(total_seqlen, parent_width, dtype=torch.float32, device=device)
    x = parent[:, :nheads * headdim].view(total_seqlen, nheads, headdim)
    B = parent[:, nheads * headdim:nheads * headdim + ngroups * dstate].view(total_seqlen, ngroups, dstate)
    x.fill_(1.0)
    B.fill_(1.0)
    assert not x.is_contiguous()
    assert not B.is_contiguous()
    assert (nchunks - 1) * chunk_size * x.stride(0) > 2**31 - 1

    dt = torch.ones(total_seqlen, nheads, dtype=torch.float32, device=device)
    A = torch.zeros(nheads, device=device)
    dA_cumsum, dt_rounded = _chunk_cumsum_fwd(dt.unsqueeze(0), A, chunk_size)
    chunk_states = _chunk_state_fwd(B.unsqueeze(0), x.unsqueeze(0), dt_rounded, dA_cumsum)
    # chunk_state_varlen expects chunk_states to be the cross-chunk running
    # state ENTERING each chunk, not _chunk_state_fwd's raw intra-chunk
    # contribution -- _state_passing_fwd does that propagation (same as the
    # parametrized test_chunk_state_varlen above).
    chunk_states, _ = _state_passing_fwd(rearrange(chunk_states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                         chunk_size=chunk_size)
    chunk_states = rearrange(chunk_states, "... (p n) -> ... p n", n=dstate)
    dA_cumsum, dt_rounded = dA_cumsum.squeeze(0), dt_rounded.squeeze(0)
    chunk_states = chunk_states.squeeze(0)

    out = chunk_state_varlen(B, x, dt_rounded, dA_cumsum, cu_seqlens, chunk_states)
    torch.cuda.synchronize(device)

    assert out.shape == (1, nheads, headdim, dstate)
    expected_out = torch.full_like(out, float(total_seqlen))
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)


@gpu_memory_skipif(6)
def test_chunk_state_varlen_noncontiguous_wide_view_no_overflow() -> None:
    # Same overflow class in _chunk_state_varlen_kernel, see
    # https://github.com/state-spaces/mamba/issues/1015. pid_c here comes from a loaded
    # cu_seqlens value rather than tl.program_id() directly.
    device = 'cuda'

    torch.manual_seed(0)
    nheads = 8
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 906  # (nchunks - 1) * chunk_size * 18_560 > 2**31 - 1
    total_seqlen = nchunks * chunk_size
    # dtype=torch.int32 matters: a plain torch.tensor([...]) defaults to
    # int64, which would make end_idx already 64-bit regardless of the
    # kernel's own cast, silently defeating this test.
    cu_seqlens = torch.tensor([0, total_seqlen], device=device, dtype=torch.int32)
    parent_width = 18_560

    # x and B share ONE wide parent tensor (disjoint column ranges) instead
    # of one each -- see wide_noncontiguous_slices' docstring.
    x, B = wide_noncontiguous_slices(
        device, total_seqlen, parent_width, [(nheads, headdim), (ngroups, dstate)],
    )
    assert not x.is_contiguous()
    assert not B.is_contiguous()
    assert (nchunks - 1) * chunk_size * x.stride(0) > 2**31 - 1

    x_c, B_c = x.contiguous(), B.contiguous()

    dt = F.softplus(torch.randn(total_seqlen, nheads, device=device, dtype=torch.float32) - 4)
    A = -0.1 * torch.rand(nheads, device=device)
    dA_cumsum, dt_rounded = _chunk_cumsum_fwd(dt.unsqueeze(0), A, chunk_size)
    chunk_states = _chunk_state_fwd(B.unsqueeze(0), x.unsqueeze(0), dt_rounded, dA_cumsum)
    chunk_states_c = _chunk_state_fwd(B_c.unsqueeze(0), x_c.unsqueeze(0), dt_rounded, dA_cumsum)
    dA_cumsum, dt_rounded = dA_cumsum.squeeze(0), dt_rounded.squeeze(0)
    chunk_states, chunk_states_c = chunk_states.squeeze(0), chunk_states_c.squeeze(0)

    out = chunk_state_varlen(B, x, dt_rounded, dA_cumsum, cu_seqlens, chunk_states)
    out_c = chunk_state_varlen(B_c, x_c, dt_rounded, dA_cumsum, cu_seqlens, chunk_states_c)
    torch.cuda.synchronize(device)

    assert out.shape == (1, nheads, headdim, dstate)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), out_c.float(), rtol=1e-4, atol=1e-4)


@gpu_memory_skipif(11)
def test_chunk_state_varlen_batch_axis_no_overflow_known_answer() -> None:
    # Known-answer counterpart to the sibling test below, targeting the
    # same `pid_b * stride_states_batch` term. Every sequence here is
    # exactly one token, so it's always fully contained within a single
    # global chunk -- the kernel's "if start_idx < pid_c * chunk_size:
    # add chunk_states" branch (the cross-CHUNK carry) never fires, and its
    # start_idx_cur masking excludes every other token packed into the same
    # chunk. So states[b] reduces to exactly that one token's own
    # contribution: dt[b] * x[b] * B[b], with no decay term (a single
    # token's own last-step scale is exp(dA_cs_last - dA_cs_last) = 1
    # regardless of A) and no cross-sequence leakage from chunk_states.
    #
    # Setting x = B = 1 and dt[i] = i + 1 (i = the token's position in the
    # packed stream, which for one-token sequences is exactly its batch
    # index b) makes states[b] = b + 1 exactly -- distinct per batch, so a
    # wrapped store landing in the wrong batch slot is caught.
    #
    device = 'cuda'

    batch = 65_000
    nheads, headdim, dstate, ngroups = 8, 64, 72, 8
    chunk_size = 128
    total_seqlen = batch  # one token per sequence

    assert (batch - 1) * nheads * headdim * dstate > 2**31 - 1, \
        "test parameters too small to overflow the batch-axis term"
    assert batch < 65_535, "batch must stay under CUDA's grid-dim-Y limit"

    cu_seqlens = torch.arange(0, batch + 1, device=device, dtype=torch.int32)
    nchunks = math.ceil((total_seqlen - 1) / chunk_size) + 1
    padded_len = nchunks * chunk_size

    x = torch.ones(total_seqlen, nheads, headdim, dtype=torch.float32, device=device)
    B = torch.ones(total_seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    i = torch.arange(padded_len, dtype=torch.float32, device=device)
    dt = (i + 1)[None, :, None].expand(1, padded_len, nheads).contiguous()
    A = torch.zeros(nheads, device=device)
    dA_cumsum, dt_rounded = _chunk_cumsum_fwd(dt, A, chunk_size)
    dA_cumsum, dt_rounded = dA_cumsum.squeeze(0), dt_rounded.squeeze(0)
    x_pad = F.pad(x, (0, 0, 0, 0, 0, padded_len - total_seqlen))
    B_pad = F.pad(B, (0, 0, 0, 0, 0, padded_len - total_seqlen))
    chunk_states = _chunk_state_fwd(B_pad.unsqueeze(0), x_pad.unsqueeze(0), dt_rounded.unsqueeze(0),
                                    dA_cumsum.unsqueeze(0)).squeeze(0)

    states = chunk_state_varlen(B, x, dt_rounded, dA_cumsum, cu_seqlens, chunk_states)
    torch.cuda.synchronize(device)

    b = torch.arange(batch, dtype=torch.float32, device=device)
    expected_col = b + 1  # (batch,) -- same for every head/headdim/dstate position
    # Compare a few individual slices, not the full (batch, nheads, headdim,
    # dstate) tensor: torch.testing.assert_close materializes the
    # broadcasted comparison, which alone uses ~35 GiB on top of the ~10
    # GiB the kernel itself needs -- every slice is identical by
    # construction anyway, so this still fully exercises the batch-axis
    # addressing under test at a fraction of the memory.
    for h, p, n in ((0, 0, 0), (nheads - 1, headdim - 1, dstate - 1), (nheads // 2, headdim // 2, dstate // 2)):
        torch.testing.assert_close(states[:, h, p, n].float(), expected_col, rtol=1e-3, atol=0)


@gpu_memory_skipif(8)
def test_chunk_state_varlen_batch_axis_no_overflow() -> None:
    # Distinct overflow term in the same kernel: `pid_b * stride_states_batch`
    # (states_ptr, the varlen output buffer), separate from the pid_c term
    # above. See https://github.com/state-spaces/mamba/issues/1015. Unlike every other
    # batch-axis case in this file, `states` here is freshly allocated
    # (ordinary contiguous) inside chunk_state_varlen() itself -- its batch
    # stride is just nheads * headdim * dstate, its own natural size, not
    # an injected wide stride. So this needs a genuinely large *batch*
    # (many packed sequences), not a synthetic parent tensor -- a real
    # continuous-batching / packed-sequence scenario with tens of thousands
    # of sequences in one call, which is realistic at serving scale.
    #
    # batch is capped by CUDA's ~65535 grid-dim-Y limit (this kernel grids
    # over (·, batch, nheads)), so batch alone can't reach 2**31 - 1; nheads
    # * headdim * dstate has to make up the rest.
    #
    # Confirmed via direct reproduction before this fix: reverting pid_b's
    # cast crashes with an illegal memory access at this exact scale.
    device = 'cuda'

    torch.manual_seed(0)
    batch = 65_000
    nheads, headdim, dstate, ngroups = 8, 64, 72, 8
    chunk_size = 128
    total_seqlen = batch  # one token per sequence -- keeps x/B/dt/dA_cumsum small

    assert (batch - 1) * nheads * headdim * dstate > 2**31 - 1, \
        "test parameters too small to overflow the batch-axis term"
    assert batch < 65_535, "batch must stay under CUDA's grid-dim-Y limit"

    cu_seqlens = torch.arange(0, batch + 1, device=device, dtype=torch.int32)
    nchunks = math.ceil((total_seqlen - 1) / chunk_size) + 1
    padded_len = nchunks * chunk_size

    x = torch.randn(total_seqlen, nheads, headdim, dtype=torch.bfloat16, device=device)
    B = torch.randn(total_seqlen, ngroups, dstate, dtype=torch.bfloat16, device=device)
    dt = F.softplus(torch.randn(1, padded_len, nheads, device=device, dtype=torch.float32) - 4)
    A = -0.1 * torch.rand(nheads, device=device)
    dA_cumsum, dt_rounded = _chunk_cumsum_fwd(dt, A, chunk_size)
    dA_cumsum, dt_rounded = dA_cumsum.squeeze(0), dt_rounded.squeeze(0)
    x_pad = F.pad(x, (0, 0, 0, 0, 0, padded_len - total_seqlen))
    B_pad = F.pad(B, (0, 0, 0, 0, 0, padded_len - total_seqlen))
    chunk_states = _chunk_state_fwd(B_pad.unsqueeze(0), x_pad.unsqueeze(0), dt_rounded.unsqueeze(0),
                                    dA_cumsum.unsqueeze(0)).squeeze(0)

    states = chunk_state_varlen(B, x, dt_rounded, dA_cumsum, cu_seqlens, chunk_states)
    torch.cuda.synchronize(device)
    assert torch.isfinite(states).all()

    # Reference: recompute a handful of sequences (first, last, and a
    # couple in between) independently via the same chunk_state/
    # _state_passing_fwd path the batched varlen kernel is meant to match --
    # a full independent recompute for all 65,000 sequences would be slow
    # and isn't necessary to catch a wrapped-but-in-bounds batch offset.
    for b in [0, 1, batch // 2, batch - 2, batch - 1]:
        start, end = cu_seqlens[b].item(), cu_seqlens[b + 1].item()
        x_s = x[start:end].unsqueeze(0)
        B_s = B[start:end].unsqueeze(0)
        dt_s = dt[:, start:end]
        dA_cumsum_s, dt_rounded_s = _chunk_cumsum_fwd(dt_s, A, chunk_size)
        st = chunk_state(B_s, x_s, dt_rounded_s, dA_cumsum_s)
        _, final_states = _state_passing_fwd(rearrange(st, "... p n -> ... (p n)"), dA_cumsum_s[:, :, :, -1],
                                             chunk_size=chunk_size)
        final_states = rearrange(final_states, "... (p n) -> ... p n", n=dstate).squeeze(0)
        torch.testing.assert_close(states[b].float(), final_states.float(), rtol=1e-4, atol=1e-4,
                                   msg=f"batch index {b} mismatch")
