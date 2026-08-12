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


@gpu_memory_skipif(11)
def test_bmm_chunk_fwd_noncontiguous_wide_view_batch_axis_no_overflow_known_answer() -> None:
    # Batch-axis counterpart to the chunk_cumsum known-answer tests above,
    # targeting _bmm_chunk_fwd_kernel/_bmm_chunk_bwd_kernel's `pid_b`
    # instead. Give a and b the same constant-per-batch vector value
    # (batch_val = b + 1, repeated across seqlen/group/dstate) so
    # out[b, c, h, m, n] = dstate * batch_val[b]**2 exactly, independent of
    # m, n, c, h -- a wrapped read landing on the wrong batch reads a
    # different, distinguishable batch_val.
    #
    # For the backward pass, _bmm_chunk_bwd(a, dout) computes the gradient
    # for the OTHER matrix from forward (out = a @ b^T), i.e.
    # da[b, n, k] = sum_m(dout[b, ..., m, n]) * a[b, m, k] -- and since a is
    # fixed to the same constant-per-batch vector for every m,
    # da[b, s, h, k] = batch_val[b] * sum_m(dout[b, ..., m]) for every k --
    # linear in dout, so this holds for ANY dout (no need to also make
    # dout a known constant).
    device = 'cuda'

    batch = 8
    seqlen = 8_192
    chunk_size = 128
    ngroups = 1
    dstate = 32
    parent_width = 40_960  # (batch - 1) * seqlen * parent_width > 2**31 - 1, with ~9% margin
    assert seqlen % chunk_size == 0, "test assumes a whole number of chunks (no padding term below)"
    nchunks = seqlen // chunk_size

    parent = torch.zeros((batch, seqlen, parent_width), dtype=torch.float32, device=device)
    a = parent[..., :dstate].view(batch, seqlen, ngroups, dstate)
    b = parent[..., dstate:2 * dstate].view(batch, seqlen, ngroups, dstate)
    batch_val = torch.arange(1, batch + 1, dtype=torch.float32, device=device)
    a.copy_(batch_val[:, None, None, None])
    b.copy_(batch_val[:, None, None, None])
    assert not a.is_contiguous()
    assert a.stride(0) * (batch - 1) > 2**31 - 1, "test parameters too small to overflow the batch-axis term"

    out = _bmm_chunk_fwd(a, b, chunk_size)
    torch.cuda.synchronize(device)
    expected_out = (dstate * batch_val**2)[:, None, None, None, None].expand(batch, nchunks, ngroups, chunk_size, chunk_size)
    torch.testing.assert_close(out.float(), expected_out, rtol=1e-4, atol=1e-4)

    dout = torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size, device=device)
    da = _bmm_chunk_bwd(a, dout)
    torch.cuda.synchronize(device)
    col_sum = dout.sum(dim=-2)  # (batch, nchunks, ngroups, chunk_size), one value per sequence position n
    expected_da = (batch_val[:, None, None, None] * col_sum).reshape(batch, seqlen, ngroups)[..., None].expand(batch, seqlen, ngroups, dstate)
    # Looser, mostly-absolute tolerance: summing chunk_size=128 random terms
    # in blocks (kernel) vs. one shot (torch) causes float roundoff up to
    # ~0.1 in absolute terms even for a correct kernel -- and relative error
    # blows up whenever the true sum happens to land near 0 by cancellation,
    # so atol has to carry most of the budget here, same idea as the dA
    # comparison above (which doesn't hit this since dA is never near 0).
    torch.testing.assert_close(da.float(), expected_da, rtol=1e-2, atol=0.2)


@gpu_memory_skipif(6)
def test_bmm_chunk_fwd_bwd_noncontiguous_wide_view_batch_axis_no_overflow() -> None:
    # _bmm_chunk_fwd_kernel/_bmm_chunk_bwd_kernel left `pid_b` uncast (only
    # `pid_ch` was fixed), so `pid_b * stride_a_batch` overflows int32 at
    # batch > 1 with a wide fused-projection stride -- unlike every other
    # kernel in this codebase with an analogous batch term. See
    # https://github.com/state-spaces/mamba/issues/1015. Confirmed via direct reproduction: this
    # crashes with an illegal memory access before the pid_b cast is added,
    # not just wrong-but-finite values -- so isfinite() alone would actually
    # catch this one, but compare against a contiguous copy anyway to also
    # catch a wrapped-but-in-bounds offset if the margin were different.
    device = 'cuda'

    torch.manual_seed(0)
    batch = 8
    seqlen = 8_192
    chunk_size = 128
    ngroups = 1
    dstate = 32
    parent_width = 40_960  # (batch - 1) * seqlen * parent_width > 2**31 - 1, with ~9% margin

    a, b = wide_noncontiguous_slices(device, seqlen, parent_width, [(ngroups, dstate), (ngroups, dstate)], batch=batch)
    assert not a.is_contiguous()
    assert a.stride(0) * (batch - 1) > 2**31 - 1, "test parameters too small to overflow the batch-axis term"
    a_c, b_c = a.contiguous(), b.contiguous()

    out = _bmm_chunk_fwd(a, b, chunk_size)
    out_c = _bmm_chunk_fwd(a_c, b_c, chunk_size)
    torch.cuda.synchronize(device)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), out_c.float(), rtol=1e-4, atol=1e-4)

    dout = torch.randn_like(out)
    da = _bmm_chunk_bwd(a, dout)
    da_c = _bmm_chunk_bwd(a_c, dout)
    torch.cuda.synchronize(device)
    assert torch.isfinite(da).all()
    torch.testing.assert_close(da.float(), da_c.float(), rtol=1e-4, atol=1e-4)


@gpu_memory_skipif(16)
def test_mamba_chunk_scan_combined_noncontiguous_wide_view_no_overflow_known_answer() -> None:
    # Same overflow (see https://github.com/state-spaces/mamba/issues/1015) and shared-wide-parent
    # setup as the sibling test below, but with constant inputs chosen so the
    # SSM recurrence has an exact closed-form answer, instead of relying on
    # ssd_chunk_scan_combined_ref (unusable here per its own docstring) or a
    # self-consistency check against a contiguous copy.
    #
    # With A = 0 (no decay), dt = 1, x = B = C = 1, D = None, z = None: the
    # recurrence state[t] = state[t-1] + dt[t] * B[t] * x[t] simplifies to
    # state[t][p, n] = t + 1 (0-indexed t) for every head/p/n, and
    # out[t, h, p] = C[t]^T state[t] = dstate * (t + 1) -- exact regardless
    # of how the chunked algorithm internally splits the recurrence, since
    # this is just the true SSM math, not an approximation of it. z is
    # dropped here (silu(z) has no simple closed form) -- the sibling test
    # below still covers z's own overflow addressing.
    #
    # float32 (not the sibling's bfloat16) throughout: state/out values here
    # reach dstate * seqlen ~= 3.7M, comfortably inside float32's exact
    # integer range (2**24) but well past bfloat16's (2**8).
    device = 'cuda'

    batch = 1
    nheads = 4  # overflow depends only on seqlen/chunk_size/stride, not head count
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 906  # (nchunks - 1) * chunk_size * 18_560 > 2**31 - 1
    seqlen = nchunks * chunk_size
    parent_width = 18_560  # same width that overflowed in the ssd_chunk_state.py bug

    parent = torch.zeros((batch, seqlen, parent_width), dtype=torch.float32, device=device)
    offset = 0
    x = parent[..., offset:offset + nheads * headdim].view(batch, seqlen, nheads, headdim)
    offset += nheads * headdim
    B = parent[..., offset:offset + ngroups * dstate].view(batch, seqlen, ngroups, dstate)
    offset += ngroups * dstate
    C = parent[..., offset:offset + ngroups * dstate].view(batch, seqlen, ngroups, dstate)
    offset += ngroups * dstate
    x.fill_(1.0)
    B.fill_(1.0)
    C.fill_(1.0)
    assert not x.is_contiguous()
    assert not B.is_contiguous()
    assert not C.is_contiguous()
    assert (nchunks - 1) * chunk_size * B.stride(1) > 2**31 - 1, \
        "test parameters too small to overflow the pid_c term"

    dt = torch.ones(batch, seqlen, nheads, dtype=torch.float32, device=device)
    A = torch.zeros(nheads, dtype=torch.float32, device=device)

    out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size)
    torch.cuda.synchronize(device)

    assert out.shape == (batch, seqlen, nheads, headdim)
    t = torch.arange(seqlen, dtype=torch.float32, device=device)
    expected_out = (dstate * (t + 1))[None, :, None, None].expand(batch, seqlen, nheads, headdim)
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)


@gpu_memory_skipif(16)
def test_mamba_chunk_scan_combined_bwd_noncontiguous_wide_view_no_overflow_known_answer() -> None:
    # Backward counterpart to
    # test_mamba_chunk_scan_combined_noncontiguous_wide_view_no_overflow_known_answer.
    # Unrolling the recurrence with decay exp(A*(t-t')) between steps t' and
    # t gives state[t,p,n] = sum_{t'<=t} exp(A*(t-t')) * dt[t'] * x[t'][p] * B[t'][n].
    # Differentiating at the same operating point (A=0, dt=x=B=C=1) and
    # summing over the L = out.sum() loss gives, with T = seqlen and
    # 0-indexed t (ngroups=1 below, so all nheads heads share the same B/C --
    # their gradients pick up a factor of nheads from that sharing; x/dt/A
    # are per-head already and don't):
    #   dC[t]  = nheads * headdim * (t + 1)
    #   dB[t]  = nheads * headdim * (T - t)
    #   dx[t]  = dstate * (T - t)
    #   ddt[t] = headdim * dstate * (T - t)
    #   dA     = headdim * dstate * (T + 1) * T * (T - 1) / 6
    # dt and A only pick up their direct multiplicative contribution here:
    # d(exp(A*dt))/dt = A*exp(A*dt) = 0 at A=0, so unlike dA, ddt's formula
    # doesn't need to account for the decay term at all. Verified by hand
    # against a T=2, headdim=dstate=1 toy case, and numerically against this
    # test's actual nheads/headdim/dstate before adding tolerances below.
    device = 'cuda'

    batch = 1
    nheads = 4  # overflow depends only on seqlen/chunk_size/stride, not head count
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 906  # (nchunks - 1) * chunk_size * 18_560 > 2**31 - 1
    seqlen = nchunks * chunk_size
    parent_width = 18_560  # same width that overflowed in the ssd_chunk_state.py bug

    parent = torch.zeros((batch, seqlen, parent_width), dtype=torch.float32, device=device)
    offset = 0
    x = parent[..., offset:offset + nheads * headdim].view(batch, seqlen, nheads, headdim)
    offset += nheads * headdim
    B = parent[..., offset:offset + ngroups * dstate].view(batch, seqlen, ngroups, dstate)
    offset += ngroups * dstate
    C = parent[..., offset:offset + ngroups * dstate].view(batch, seqlen, ngroups, dstate)
    offset += ngroups * dstate
    x.fill_(1.0)
    B.fill_(1.0)
    C.fill_(1.0)
    assert not x.is_contiguous()
    assert not B.is_contiguous()
    assert not C.is_contiguous()
    assert (nchunks - 1) * chunk_size * B.stride(1) > 2**31 - 1, \
        "test parameters too small to overflow the pid_c term"

    dt = torch.ones(batch, seqlen, nheads, dtype=torch.float32, device=device)
    A = torch.zeros(nheads, dtype=torch.float32, device=device)
    # requires_grad_() must be called in place, not via .clone() first --
    # .clone() silently makes x/B/C contiguous, defeating this test entirely.
    for tensor in (x, B, C, dt, A):
        tensor.requires_grad_()

    out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size)
    out.sum().backward()
    torch.cuda.synchronize(device)

    T = seqlen
    t = torch.arange(seqlen, dtype=torch.float32, device=device)
    for name, grad, expected in [
        ("C", C.grad, (nheads * headdim * (t + 1))[None, :, None, None].expand(batch, seqlen, ngroups, dstate)),
        ("B", B.grad, (nheads * headdim * (T - t))[None, :, None, None].expand(batch, seqlen, ngroups, dstate)),
        ("x", x.grad, (dstate * (T - t))[None, :, None, None].expand(batch, seqlen, nheads, headdim)),
        ("dt", dt.grad, (headdim * dstate * (T - t))[None, :, None].expand(batch, seqlen, nheads)),
        ("A", A.grad, torch.full((nheads,), headdim * dstate * (T + 1) * T * (T - 1) / 6, device=device)),
    ]:
        assert grad is not None, f"{name}.grad is None"
        # Loose relative tolerance: these sums grow with T (up to ~1e17 for
        # dA), well past float32's exact-integer range, and the kernels
        # accumulate in float32 across ~900 chunks -- roundoff at the
        # ~1e-3 relative level is expected here even for a correct kernel.
        torch.testing.assert_close(grad.float(), expected, rtol=1e-3, atol=0, msg=f"{name}.grad mismatch")


@gpu_memory_skipif(6)
def test_mamba_chunk_scan_combined_noncontiguous_wide_view_no_overflow() -> None:
    # Same overflow class hit through the full mamba_chunk_scan_combined
    # path (see https://github.com/state-spaces/mamba/issues/1015), covering ssd_chunk_scan.py,
    # ssd_combined.py, and ssd_bmm.py's kernels too. x, B, C, and z are all
    # sliced non-contiguously out of a shared wide parent tensor (not just
    # x). Backward is exercised too, since several of the fixed kernels
    # only run there, not under a forward-only no_grad call.
    #
    # Compares the same kernel run on wide-sliced vs. an ordinary
    # .contiguous() copy of identical values, rather than
    # ssd_chunk_scan_combined_ref (unusable here: NaNs at this chunk count
    # even for a correct kernel, per its own docstring, and OOMs at
    # realistic head counts). isfinite() alone isn't sufficient either --
    # a wrapped offset can land in-bounds and produce finite, wrong values.
    device = 'cuda'

    torch.manual_seed(0)
    batch = 1
    # Small nheads: holding x/z/B/C, their contiguous copies, and both
    # autograd graphs at once OOMs at realistic head counts; the overflow
    # depends only on seqlen/chunk_size/stride, not head count.
    nheads = 16
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 906  # (nchunks - 1) * chunk_size * 18_560 > 2**31 - 1
    seqlen = nchunks * chunk_size
    parent_width = 18_560  # same width that overflowed in the ssd_chunk_state.py bug

    # x, z, B, and C are ALL sliced non-contiguously out of the SAME wide
    # parent tensor here (disjoint column ranges, via wide_noncontiguous_slices),
    # rather than one wide parent per tensor -- one ~4 GiB allocation instead
    # of four.
    x, z, B, C = wide_noncontiguous_slices(
        device, seqlen, parent_width,
        [(nheads, headdim), (nheads, headdim), (ngroups, dstate), (ngroups, dstate)],
        batch=batch,
    )
    assert not x.is_contiguous()
    assert not z.is_contiguous()
    assert not B.is_contiguous()
    assert not C.is_contiguous()
    # (nchunks - 1) * chunk_size * stride(1) overflow check for B/C
    assert (nchunks - 1) * chunk_size * B.stride(1) > 2**31 - 1

    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4)
    A = -torch.rand(nheads, dtype=torch.float32, device=device) - 0.01
    D = torch.randn(nheads, headdim, dtype=torch.float32, device=device)

    # x/z/B/C: requires_grad_() must be called in place, not via .clone()
    # first -- .clone() silently makes a non-contiguous tensor contiguous,
    # which would defeat this test entirely.
    x_c, z_c, B_c, C_c = [t.contiguous() for t in (x, z, B, C)]
    dt_c, A_c, D_c = [t.clone() for t in (dt, A, D)]
    for t in (x, z, B, C, dt, A, D, x_c, z_c, B_c, C_c, dt_c, A_c, D_c):
        t.requires_grad_()
    assert not x.is_contiguous() and x_c.is_contiguous()
    assert not B.is_contiguous() and B_c.is_contiguous()

    out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z)
    out_c = mamba_chunk_scan_combined(x_c, dt_c, A_c, B_c, C_c, chunk_size, D=D_c, z=z_c)
    torch.cuda.synchronize(device)

    assert out.shape == (batch, seqlen, nheads, headdim)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), out_c.float(), rtol=1e-4, atol=1e-4)

    out.sum().backward()
    out_c.sum().backward()
    torch.cuda.synchronize(device)
    grad_pairs = [("x", x, x_c), ("z", z, z_c), ("B", B, B_c), ("C", C, C_c),
                  ("dt", dt, dt_c), ("A", A, A_c), ("D", D, D_c)]
    for name, t, t_c in grad_pairs:
        assert t.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite values"
        # Slightly looser than the forward check: A.grad and D.grad are
        # summed over all ~900 chunks, and summing in a different order
        # (contiguous vs. wide-sliced memory layout) causes tiny float
        # roundoff differences at the ~1e-3 level even for a correct kernel.
        torch.testing.assert_close(t.grad.float(), t_c.grad.float(), rtol=1e-3, atol=1e-3,
                                   msg=f"{name}.grad mismatch vs contiguous-equivalent")


@gpu_memory_skipif(12)
def test_mamba_chunk_scan_combined_noncontiguous_wide_view_batch_axis_no_overflow_known_answer() -> None:
    # Batch-axis counterpart to the chunk-axis known-answer fwd+bwd tests
    # above (combined into one test here, like the sibling test below,
    # since batch=8/nchunks=64 is cheap enough not to need splitting).
    #
    # Same A=0, dt=1, D=None, z=None setup, but x = B = C = v[b] = b + 1
    # (batch-dependent, not just 1) so a wrapped read landing on the wrong
    # batch reads a different, distinguishable v. With decay=1:
    #   state[t,p,n] = v[b]^2 * (t + 1)
    #   out[t,p]     = dstate * v[b]^3 * (t + 1)
    # Differentiating (ngroups=1, so B/C are shared across all nheads
    # heads and pick up a factor of nheads; A is shared across all
    # batches, so its gradient sums v[b]^3 over every batch):
    #   dC[t,b]  = nheads * headdim * v[b]^2 * (t + 1)
    #   dB[t,b]  = nheads * headdim * v[b]^2 * (T - t)
    #   dx[t,b]  = dstate * v[b]^2 * (T - t)
    #   ddt[t,b] = headdim * dstate * v[b]^3 * (T - t)
    #   dA[h]    = headdim * dstate * (T + 1) * T * (T - 1) / 6 * sum_b(v[b]^3)
    # Derived the same way as the chunk-axis version: general partial
    # derivatives of the trilinear-in-(x,B,C) recurrence, then evaluated
    # at x=B=C=v[b] instead of x=B=C=1 -- every term above is exactly the
    # chunk-axis formula times the extra v[b]^2 or v[b]^3 its derivation
    # picks up from the extra non-unit constant.
    device = 'cuda'

    batch = 8
    nheads = 16
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 64
    seqlen = nchunks * chunk_size
    parent_width = 40_960  # (batch - 1) * seqlen * parent_width > 2**31 - 1, with ~9% margin

    parent = torch.zeros((batch, seqlen, parent_width), dtype=torch.float32, device=device)
    offset = 0
    x = parent[..., offset:offset + nheads * headdim].view(batch, seqlen, nheads, headdim)
    offset += nheads * headdim
    B = parent[..., offset:offset + ngroups * dstate].view(batch, seqlen, ngroups, dstate)
    offset += ngroups * dstate
    C = parent[..., offset:offset + ngroups * dstate].view(batch, seqlen, ngroups, dstate)
    offset += ngroups * dstate
    v = torch.arange(1, batch + 1, dtype=torch.float32, device=device)
    x.copy_(v[:, None, None, None])
    B.copy_(v[:, None, None, None])
    C.copy_(v[:, None, None, None])
    assert not x.is_contiguous()
    assert not B.is_contiguous()
    assert not C.is_contiguous()
    assert (batch - 1) * x.stride(0) > 2**31 - 1, "test parameters too small to overflow the batch-axis term"

    dt = torch.ones(batch, seqlen, nheads, dtype=torch.float32, device=device)
    A = torch.zeros(nheads, dtype=torch.float32, device=device)
    for tensor in (x, B, C, dt, A):
        tensor.requires_grad_()

    out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size)
    torch.cuda.synchronize(device)

    assert out.shape == (batch, seqlen, nheads, headdim)
    T = seqlen
    t = torch.arange(seqlen, dtype=torch.float32, device=device)
    expected_out = (dstate * v[:, None] ** 3 * (t + 1)[None, :])[:, :, None, None].expand(batch, seqlen, nheads, headdim)
    torch.testing.assert_close(out, expected_out, rtol=1e-3, atol=0)

    out.sum().backward()
    torch.cuda.synchronize(device)

    v2, v3 = v**2, v**3
    for name, grad, expected in [
        ("C", C.grad, (nheads * headdim * v2[:, None] * (t + 1)[None, :])[:, :, None, None].expand(batch, seqlen, ngroups, dstate)),
        ("B", B.grad, (nheads * headdim * v2[:, None] * (T - t)[None, :])[:, :, None, None].expand(batch, seqlen, ngroups, dstate)),
        ("x", x.grad, (dstate * v2[:, None] * (T - t)[None, :])[:, :, None, None].expand(batch, seqlen, nheads, headdim)),
        ("dt", dt.grad, (headdim * dstate * v3[:, None] * (T - t)[None, :])[:, :, None].expand(batch, seqlen, nheads)),
        ("A", A.grad, torch.full((nheads,), headdim * dstate * (T + 1) * T * (T - 1) / 6 * v3.sum().item(), device=device)),
    ]:
        assert grad is not None, f"{name}.grad is None"
        # Same loose relative tolerance as the chunk-axis version: float32
        # accumulation roundoff over many chunks/timesteps, not a sign of a
        # real bug.
        torch.testing.assert_close(grad.float(), expected, rtol=1e-3, atol=0, msg=f"{name}.grad mismatch")


@gpu_memory_skipif(8)
def test_mamba_chunk_scan_combined_noncontiguous_wide_view_batch_axis_no_overflow() -> None:
    # Batch-axis counterpart to test_mamba_chunk_scan_combined_noncontiguous_wide_view_no_overflow
    # above: `pid_b * stride_x_batch` (and the analogous B/C/z terms) in
    # _chunk_state_fwd/bwd_dx/bwd_db_kernel, the seven live ssd_chunk_scan.py
    # kernels, and _chunk_scan_chunk_state_bwd_dx_kernel -- all reached
    # through this same mamba_chunk_scan_combined call, all sharing the
    # pid_bc = tl.program_id(1).to(tl.int64) pattern -- only ever had their
    # chunk-axis half (pid_c) exercised by a real test; batch=1 there meant
    # pid_b was always 0. See https://github.com/state-spaces/mamba/issues/1015.
    #
    # Same setup as the chunk-axis version, but batch=8 with a much smaller
    # nchunks (cheap: seqlen no longer needs to be large on its own, since
    # (batch - 1) * seqlen * parent_width is what has to clear the
    # threshold here, not (nchunks - 1) * chunk_size * parent_width).
    device = 'cuda'

    torch.manual_seed(0)
    batch = 8
    nheads = 16
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 64
    seqlen = nchunks * chunk_size
    parent_width = 40_960  # (batch - 1) * seqlen * parent_width > 2**31 - 1, with ~9% margin

    x, z, B, C = wide_noncontiguous_slices(
        device, seqlen, parent_width,
        [(nheads, headdim), (nheads, headdim), (ngroups, dstate), (ngroups, dstate)],
        batch=batch,
    )
    assert not x.is_contiguous()
    assert not z.is_contiguous()
    assert not B.is_contiguous()
    assert not C.is_contiguous()
    assert (batch - 1) * x.stride(0) > 2**31 - 1, "test parameters too small to overflow the batch-axis term"

    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4)
    A = -torch.rand(nheads, dtype=torch.float32, device=device) - 0.01
    D = torch.randn(nheads, headdim, dtype=torch.float32, device=device)

    x_c, z_c, B_c, C_c = [t.contiguous() for t in (x, z, B, C)]
    dt_c, A_c, D_c = [t.clone() for t in (dt, A, D)]
    for t in (x, z, B, C, dt, A, D, x_c, z_c, B_c, C_c, dt_c, A_c, D_c):
        t.requires_grad_()
    assert not x.is_contiguous() and x_c.is_contiguous()
    assert not B.is_contiguous() and B_c.is_contiguous()

    out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z)
    out_c = mamba_chunk_scan_combined(x_c, dt_c, A_c, B_c, C_c, chunk_size, D=D_c, z=z_c)
    torch.cuda.synchronize(device)

    assert out.shape == (batch, seqlen, nheads, headdim)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), out_c.float(), rtol=1e-4, atol=1e-4)

    out.sum().backward()
    out_c.sum().backward()
    torch.cuda.synchronize(device)
    grad_pairs = [("x", x, x_c), ("z", z, z_c), ("B", B, B_c), ("C", C, C_c),
                  ("dt", dt, dt_c), ("A", A, A_c), ("D", D, D_c)]
    for name, t, t_c in grad_pairs:
        assert t.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite values"
        torch.testing.assert_close(t.grad.float(), t_c.grad.float(), rtol=1e-3, atol=1e-3,
                                   msg=f"{name}.grad mismatch vs contiguous-equivalent")


@gpu_memory_skipif(12)
def test_mamba_chunk_scan_combined_bwd_noncontiguous_dout_no_overflow_known_answer() -> None:
    # Known-answer counterpart to the sibling test below, targeting the
    # same wide, non-contiguous dout (the incoming gradient from autograd
    # can arrive with an arbitrary, caller-controlled large stride, unlike
    # x/dt/B/C which we control ourselves). A constant dout = 1 is exactly
    # the adjoint of out.sum().backward() -- the operating point used by
    # test_mamba_chunk_scan_combined_bwd_noncontiguous_wide_view_no_overflow_known_answer
    # above -- so the same closed-form gradients apply here unchanged:
    #   dC[t]  = nheads * headdim * (t + 1)
    #   dB[t]  = nheads * headdim * (T - t)
    #   dx[t]  = dstate * (T - t)
    #   ddt[t] = headdim * dstate * (T - t)
    #   dA     = headdim * dstate * (T + 1) * T * (T - 1) / 6
    # D is set to 0 (not None) so dD gets returned rather than skipped --
    # 0*x adds nothing to the forward output, so it doesn't disturb the
    # formulas above, and dD = sum_t(dout * x) = T at this operating point.
    device = 'cuda'

    batch = 1
    nheads = 16
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 906
    seqlen = nchunks * chunk_size
    parent_width = 18_560

    x = torch.ones(batch, seqlen, nheads, headdim, dtype=torch.float32, device=device)
    B = torch.ones(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    C = torch.ones(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    dt = torch.ones(batch, seqlen, nheads, dtype=torch.float32, device=device)
    A = torch.zeros(nheads, dtype=torch.float32, device=device)
    D = torch.zeros(nheads, headdim, dtype=torch.float32, device=device)

    out, *_ = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D)

    dout_parent = torch.ones(batch, seqlen, parent_width, dtype=torch.float32, device=device)
    dout = dout_parent[:, :, -nheads * headdim:].view(batch, seqlen, nheads, headdim)
    assert not dout.is_contiguous()
    assert (nchunks - 1) * chunk_size * dout.stride(1) > 2**31 - 1

    dx, ddt, dA, dB, dC, dD, *_ = _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, chunk_size, D=D)
    torch.cuda.synchronize(device)

    T = seqlen
    t = torch.arange(seqlen, dtype=torch.float32, device=device)
    checks = [
        ("dC", dC, (nheads * headdim * (t + 1))[None, :, None, None].expand(batch, seqlen, ngroups, dstate)),
        ("dB", dB, (nheads * headdim * (T - t))[None, :, None, None].expand(batch, seqlen, ngroups, dstate)),
        ("dx", dx, (dstate * (T - t))[None, :, None, None].expand(batch, seqlen, nheads, headdim)),
        ("ddt", ddt, (headdim * dstate * (T - t))[None, :, None].expand(batch, seqlen, nheads)),
        ("dA", dA, torch.full((nheads,), headdim * dstate * (T + 1) * T * (T - 1) / 6, device=device)),
        ("dD", dD, torch.full((nheads, headdim), float(T), device=device)),
    ]
    for name, g, expected in checks:
        assert torch.isfinite(g).all(), f"{name} has non-finite values"
        # Same loose relative tolerance as the sibling fwd+bwd known-answer
        # test above: float32 accumulation roundoff over ~900 chunks, not a
        # sign of a real bug.
        torch.testing.assert_close(g.float(), expected, rtol=1e-3, atol=0, msg=f"{name} mismatch")


@gpu_memory_skipif(6)
def test_mamba_chunk_scan_combined_bwd_noncontiguous_dout_no_overflow() -> None:
    # int64 to avoid int32 overflow, see https://github.com/state-spaces/mamba/issues/1015.
    # dout is the incoming gradient from autograd, so unlike x/dt/B/C (all
    # produced internally by our own ops), it can arrive non-contiguous
    # with an arbitrary, caller-controlled large stride -- e.g. sliced out
    # of a wider fused gradient tensor upstream.
    device = 'cuda'

    torch.manual_seed(0)
    batch = 1
    nheads = 16
    headdim = 64
    ngroups = 1
    dstate = 32
    chunk_size = 128
    nchunks = 906
    seqlen = nchunks * chunk_size
    parent_width = 18_560

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32, device=device)
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float32, device=device)
    D = torch.randn(nheads, headdim, dtype=torch.float32, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4)
    A = -torch.rand(nheads, dtype=torch.float32, device=device) - 0.01

    out, *_ = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D)

    dout_parent = torch.randn(batch, seqlen, parent_width, dtype=torch.float32, device=device)
    dout = dout_parent[:, :, -nheads * headdim:].view(batch, seqlen, nheads, headdim)
    assert not dout.is_contiguous()
    assert (nchunks - 1) * chunk_size * dout.stride(1) > 2**31 - 1
    dout_c = dout.contiguous()

    def run(dout_):
        return _mamba_chunk_scan_combined_bwd(dout_, x, dt, A, B, C, out, chunk_size, D=D)

    dx, ddt, dA, dB, dC, dD, *_ = run(dout)
    dx_ref, ddt_ref, dA_ref, dB_ref, dC_ref, dD_ref, *_ = run(dout_c)
    torch.cuda.synchronize(device)
    # dA/dD are summed over all ~900 chunks; summing in a different order
    # (contiguous vs. wide-sliced memory layout) causes tiny float roundoff
    # differences at the ~1e-3 level even for a correct kernel -- same as
    # the combined test above.
    tols = {"dA": (1e-3, 1e-3), "dD": (1e-3, 1e-3)}
    for name, g, g_ref in [("dx", dx, dx_ref), ("ddt", ddt, ddt_ref), ("dA", dA, dA_ref),
                           ("dB", dB, dB_ref), ("dC", dC, dC_ref), ("dD", dD, dD_ref)]:
        assert torch.isfinite(g).all(), f"{name} has non-finite values"
        rtol, atol = tols.get(name, (None, None))
        torch.testing.assert_close(g, g_ref, rtol=rtol, atol=atol, msg=f"{name} mismatch vs contiguous dout")


def test_causal_conv1d_bwd_dx_given_ensure_stride_copy_path() -> None:
    # MambaSplitConv1dScanCombinedFn.backward passes
    # dx=rearrange(ensure_stride(dxBC_given), "b s d -> b d s") directly into
    # causal_conv1d_bwd_function, then detects+repairs any aliasing break via
    # a stride-comparison-and-copy afterward. This exercises the repair path:
    # force ensure_stride(dxBC_given) to return a copy (rather than a view)
    # via a monkeypatched impossible _UINT32_MAX (not real overflow scale --
    # see test_causal_conv1d_bwd_dx_given_wide_batch_stride_no_overflow for
    # that), and confirm the if/else correctly detects and repairs it against
    # a ground-truth reference (an ordinary, unsliced dx).
    device = 'cuda'
    torch.manual_seed(0)

    batch, dim, seqlen, width = 2, 16, 64, 4
    # x/dout must be genuinely channels-last (via ensure_stride + rearrange),
    # matching how the real code constructs them -- a plain contiguous x/dout
    # is NOT representative and can make the kernel reject a channels-last dx
    # for an unrelated reason (layout mismatch between x and dx).
    xBC = torch.randn(batch, seqlen, dim, dtype=torch.float32, device=device)
    x = rearrange(ensure_stride(xBC), "b s d -> b d s")
    weight = torch.randn(dim, width, dtype=torch.float32, device=device)
    bias = torch.randn(dim, dtype=torch.float32, device=device)
    doutBC = torch.randn(batch, seqlen, dim, dtype=torch.float32, device=device)
    dout = rearrange(ensure_stride(doutBC), "b s d -> b d s")

    dx_ref, *_ = causal_conv1d_bwd_function(x, weight, bias, dout, None, None, None, None, False, False)
    dx_ref_bsd = rearrange(dx_ref, "b d s -> b s d")

    def make_wide_dx_given():
        # non-contiguous slice of a wider contiguous parent, matching
        # dxBC_given's real construction as a slice of dzxbcdt.
        parent = torch.zeros(batch, seqlen, dim * 3, dtype=torch.float32, device=device)
        given = parent[:, :, :dim]
        assert not given.is_contiguous()
        return given

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ssd_combined, "_UINT32_MAX", -1)  # force ensure_stride to always copy

        dxBC_given = make_wide_dx_given()
        dx_in = rearrange(ensure_stride(dxBC_given), "b s d -> b d s")
        assert dx_in.data_ptr() != dxBC_given.data_ptr(), "ensure_stride did not copy as expected"
        dxBC_given_update, *_ = causal_conv1d_bwd_function(x, weight, bias, dout, None, None, None, dx_in, False, False)
        dxBC_given_update = rearrange(dxBC_given_update, "b d s -> b s d")
        if dxBC_given.stride() != dxBC_given_update.stride():
            dxBC_given.copy_(dxBC_given_update)
        else:
            dxBC_given = dxBC_given_update

    torch.testing.assert_close(dxBC_given, dx_ref_bsd, msg="copy-path repair diverged from ground truth")


@gpu_memory_skipif(25)
def test_causal_conv1d_bwd_dx_given_wide_batch_stride_no_overflow() -> None:
    # Real bug, not a hypothetical: dxBC_given is a slice of the wide dzxbcdt tensor,
    # so it inherits dzxbcdt's batch stride. Passing it directly as the `dx` output
    # buffer into causal_conv1d_bwd_function, with no protection at all, corrupts the
    # result at genuine Nemotron scale (batch=4, seqlen=40960, in_proj width=35072)
    # because (batch - 1) * seqlen * width exceeds 2**32 -- the batch-3 offset wraps
    # into batch 0 inside the CUDA kernel. See https://github.com/state-spaces/mamba/issues/1015.
    #
    # The actual code in MambaSplitConv1dScanCombinedFn.backward passes
    # ensure_stride(dxBC_given) as `dx`, relying on ensure_stride's own overflow
    # check to force a protective copy at this scale, then the stride-check-and-repair
    # to copy that back into dxBC_given. This test proves that reliance is
    # necessary -- not just a style choice -- by reproducing the corruption directly
    # against two broken alternatives: passing dxBC_given as `dx` with no
    # ensure_stride at all, and with ensure_stride but no copy-back repair.
    device = 'cuda'
    torch.manual_seed(0)

    batch, seqlen, parent_width, channels, width = 4, 40960, 35072, 8, 4
    assert (batch - 1) * seqlen * parent_width > 2**32 - 1

    x_small = torch.randn(batch, seqlen, channels, device=device)
    x = rearrange(ensure_stride(x_small), "b s d -> b d s")
    weight = torch.randn(channels, width, device=device)
    bias = torch.randn(channels, device=device)
    dout_small = torch.randn(batch, seqlen, channels, device=device)
    dout = rearrange(ensure_stride(dout_small), "b s d -> b d s")

    dx_ref, *_ = causal_conv1d_bwd_function(x, weight, bias, dout, None, None, None, None, False, False)
    dx_ref_bsd = rearrange(dx_ref, "b d s -> b s d").clone()

    def make_wide_dx_given():
        # non-contiguous slice of a much wider contiguous parent, matching
        # dxBC_given's real construction as a slice of dzxbcdt.
        parent = torch.zeros(batch, seqlen, parent_width, device=device)
        given = parent[:, :, :channels]
        assert not given.is_contiguous()
        max_offset = sum((size - 1) * stride for size, stride in zip(given.shape, given.stride()))
        assert max_offset > 2**32 - 1
        return given

    # Current code's approach: ensure_stride(dxBC_given) as dx, with the stride-check-and-repair.
    dxBC_given_current = make_wide_dx_given()
    dx_in_current = rearrange(ensure_stride(dxBC_given_current), "b s d -> b d s")
    dxBC_given_update, *_ = causal_conv1d_bwd_function(x, weight, bias, dout, None, None, None, dx_in_current, False, False)
    dxBC_given_update = rearrange(dxBC_given_update, "b d s -> b s d")
    if dxBC_given_current.stride() != dxBC_given_update.stride():
        dxBC_given_current.copy_(dxBC_given_update)
    else:
        dxBC_given_current = dxBC_given_update
    torch.testing.assert_close(dxBC_given_current, dx_ref_bsd,
                               msg="current code (ensure_stride(dxBC_given) as dx + repair) corrupted at genuine overflow scale")
    del dxBC_given_current, dx_in_current, dxBC_given_update

    # Naive alternative: dx=dxBC_given directly, no ensure_stride at all.
    dxBC_given_naive = make_wide_dx_given()
    dx_in_naive = rearrange(dxBC_given_naive, "b s d -> b d s")
    causal_conv1d_bwd_function(x, weight, bias, dout, None, None, None, dx_in_naive, False, False)
    assert (dxBC_given_naive - dx_ref_bsd).abs().max().item() > 1.0, \
        "expected the naive dx=dxBC_given approach (no ensure_stride) to actually corrupt at this scale"
    del dxBC_given_naive, dx_in_naive

    # ensure_stride(dxBC_given) as dx, but no copy-back repair at all.
    dxBC_given_no_repair = make_wide_dx_given()
    dx_in_no_repair = rearrange(ensure_stride(dxBC_given_no_repair), "b s d -> b d s")
    causal_conv1d_bwd_function(x, weight, bias, dout, None, None, None, dx_in_no_repair, False, False)
    assert (dxBC_given_no_repair - dx_ref_bsd).abs().max().item() > 1.0, \
        "expected ensure_stride(dxBC_given) as dx with no copy-back repair to actually corrupt at this scale"


def test_mamba_split_conv1d_scan_combined_bwd_ensure_stride_copy_path() -> None:
    # The test above exercises ensure_stride's copy path against the bare
    # causal_conv1d_bwd_function primitive directly -- it never runs through
    # MambaSplitConv1dScanCombinedFn.backward() itself (the dxBC_given
    # stride-check-and-repair around causal_conv1d_bwd_function, see
    # https://github.com/state-spaces/mamba/issues/1015), so it doesn't prove that *that* code, as
    # actually invoked by autograd, is unaffected by ensure_stride returning
    # a copy instead of a view. Force the copy path (as above, via the same
    # _UINT32_MAX monkeypatch -- not real overflow scale) and compare
    # gradients against an unpatched run (the common case: ensure_stride
    # passes dxBC_given straight through) of the exact same real backward path.
    device = 'cuda'
    torch.manual_seed(0)
    batch, nheads, headdim, ngroups, dstate, chunk_size, seqlen = 2, 4, 32, 2, 16, 64, 256
    dim = nheads * headdim
    width_conv1d = dim + 2 * ngroups * dstate
    conv_width = 4

    def run(force_copy):
        torch.manual_seed(0)
        zxbcdt = torch.randn(batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads,
                             device=device, requires_grad=True)
        conv1d_weight = torch.randn(width_conv1d, conv_width, device=device)
        conv1d_bias = torch.randn(width_conv1d, device=device)
        dt_bias = torch.randn(nheads, device=device)
        A = -torch.rand(nheads, device=device) - 0.01
        D = torch.randn(nheads, headdim, device=device)
        with pytest.MonkeyPatch.context() as mp:
            if force_copy:
                mp.setattr(ssd_combined, "_UINT32_MAX", -1)
            out = mamba_split_conv1d_scan_combined(
                zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, ngroups=ngroups)
            out.sum().backward()
        torch.cuda.synchronize(device)
        return out.detach().clone(), zxbcdt.grad.clone()

    out_copy, grad_copy = run(force_copy=True)
    out_view, grad_view = run(force_copy=False)

    assert torch.isfinite(grad_copy).all()
    torch.testing.assert_close(out_copy, out_view, msg="forward output changed by forcing ensure_stride's copy path")
    torch.testing.assert_close(grad_copy, grad_view, msg="backward gradient changed by forcing ensure_stride's copy path")


@gpu_memory_skipif(26)
def test_mamba_split_conv1d_scan_combined_fwd_noncontiguous_no_overflow_known_answer() -> None:
    # Known-answer counterpart to the sibling test below, same real
    # (not artificially injected) non-contiguous x/B/C split. Two knobs
    # make the whole pipeline closed-form instead of just isfinite:
    #
    # 1. conv1d_weight's last tap = 1 (rest 0), bias = 0, activation=None
    #    (passed to causal_conv1d_fn) makes the causal conv an exact
    #    identity -- verified directly against causal_conv1d_fn beforehand.
    #    So xBC_conv == the raw xBC we write into zxbcdt.
    # 2. dt_softplus is hardcoded True inside this function, so instead of
    #    fighting that, dt (pre-softplus) = 1 with dt_bias = 0 gives a
    #    known dt_after = softplus(1) (computed via the same
    #    F.softplus primitive, not the kernel under test).
    #
    # With x = B = C = 1 (post-"conv"), A = 0, D = 0: this is exactly the
    # A=0/dt=x=B=C=1 recurrence from the mamba_chunk_scan_combined
    # known-answer test above, scaled by dt_after instead of 1:
    #   out_x[t] = dstate * dt_after * (t + 1)
    # z is not conv'd at all (split raw from zxbcdt), so setting z to a
    # constant z_const gates the whole thing by a known F.silu(z_const):
    #   out[t] = out_x[t] * F.silu(z_const)
    device = 'cuda'

    batch = 1
    nheads = 289
    headdim = 64
    ngroups = 1
    dstate = 32
    dim = nheads * headdim
    chunk_size = 128
    nchunks = 906
    seqlen = nchunks * chunk_size
    conv_width = 4

    width_conv1d = dim + 2 * ngroups * dstate
    assert width_conv1d == 18_560

    z_const = 2.0
    dt_raw_val = 1.0

    zxbcdt = torch.zeros(batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads,
                         dtype=torch.bfloat16, device=device)
    z_part, xbc_part, dt_part = torch.split(zxbcdt, [dim, width_conv1d, nheads], dim=-1)
    z_part.fill_(z_const)
    xbc_part.fill_(1.0)
    dt_part.fill_(dt_raw_val)
    assert not xbc_part.is_contiguous()  # real torch.split() view, not an injected wide slice

    conv1d_weight = torch.zeros(width_conv1d, conv_width, dtype=torch.bfloat16, device=device)
    conv1d_weight[:, -1] = 1.0  # identity tap, verified directly against causal_conv1d_fn
    conv1d_bias = torch.zeros(width_conv1d, dtype=torch.bfloat16, device=device)
    dt_bias = torch.zeros(nheads, dtype=torch.float32, device=device)
    A = torch.zeros(nheads, dtype=torch.float32, device=device)
    D = torch.zeros(nheads, headdim, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = mamba_split_conv1d_scan_combined(
            zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, activation=None)
    torch.cuda.synchronize(device)

    assert out.shape == (batch, seqlen, dim)
    dt_after = F.softplus(torch.tensor(dt_raw_val, device=device))
    silu_z = F.silu(torch.tensor(z_const, device=device))
    t = torch.arange(seqlen, dtype=torch.float32, device=device)
    expected_col = dstate * dt_after * (t + 1) * silu_z  # (seqlen,) -- same for every dim column
    # Compare a single column, not the full (seqlen, dim) tensor: the
    # comparison itself (not the forward pass) OOMs at the full width on a
    # 48 GiB GPU once both the actual and expected tensors materialize, and
    # every column is identical by construction anyway -- this still
    # exercises addressing across the full overflow-triggering seqlen axis.
    torch.testing.assert_close(out[0, :, 0].float(), expected_col, rtol=1e-2, atol=0)
    # Also spot-check a few other columns/heads to confirm the uniformity
    # assumption itself (not just column 0) at negligible extra memory cost.
    for col in (1, dim // 2, dim - 1):
        torch.testing.assert_close(out[0, :, col].float(), expected_col, rtol=1e-2, atol=0)


@gpu_memory_skipif(25)
def test_mamba_split_conv1d_scan_combined_fwd_noncontiguous_no_overflow() -> None:
    # int64 to avoid int32 overflow, see https://github.com/state-spaces/mamba/issues/1015.
    # x/B/C are non-contiguous here not via an artificial wide-slice trick
    # but for real: causal_conv1d's output (xBC_conv) is contiguous, and
    # x/B/C are plain torch.split() views of it along the last dim -- the
    # same "large stride(1)" pattern as elsewhere in this file, except this
    # one arises unavoidably on every real call. Sized at realistic
    # Nemotron-scale in_proj width (dim + 2*ngroups*dstate = 18,560) so the
    # split's stride(1) alone exceeds the int32 threshold over ~900 chunks.
    device = 'cuda'

    torch.manual_seed(0)
    batch = 1
    # small dstate: the intermediate `states` tensor scales with
    # nheads * headdim * dstate * nchunks, so most of width_conv1d needs to
    # come from dim (nheads * headdim), not dstate, to keep memory down.
    nheads = 289
    headdim = 64
    ngroups = 1
    dstate = 32
    dim = nheads * headdim
    chunk_size = 128
    nchunks = 906
    seqlen = nchunks * chunk_size
    conv_width = 4

    width_conv1d = dim + 2 * ngroups * dstate
    assert width_conv1d == 18_560

    zxbcdt = torch.randn(batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads,
                         dtype=torch.bfloat16, device=device)
    conv1d_weight = torch.randn(width_conv1d, conv_width, dtype=torch.bfloat16, device=device)
    conv1d_bias = torch.randn(width_conv1d, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(nheads, dtype=torch.float32, device=device)
    A = -torch.rand(nheads, dtype=torch.float32, device=device) - 0.01
    D = torch.randn(nheads, headdim, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = mamba_split_conv1d_scan_combined(
            zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size)
        torch.cuda.synchronize(device)
        assert torch.isfinite(out).all()


@gpu_memory_skipif(12)
def test_state_passing_fwd_bwd_noncontiguous_dA_chunk_cumsum_no_overflow_known_answer() -> None:
    # Known-answer counterpart to the sibling test below, same wide,
    # non-contiguous dA_chunk_cumsum setup. The fwd kernel implements
    # y[0] = 0 (no initial_states); y[j+1] = exp(dA_chunk_cumsum[c]) * y[j] + states[c],
    # out[j] = y[j] for j = 0..nchunks-1, final_states = y[nchunks].
    #
    # Setting states = 1 (constant) and dA_chunk_cumsum[h, c] = a[h] (a
    # per-head constant, same for every c -- so a wrong-head read is
    # distinguishable, unlike a shared constant) makes this a plain
    # geometric series with ratio r[h] = exp(a[h]):
    #   out[j, h] = S_h(j) := (1 - r[h]**j) / (1 - r[h])   (sum_{k=0}^{j-1} r[h]**k)
    #   final_states[h] = S_h(nchunks)
    # a[h] is kept negative (r[h] < 1) so the series stays bounded over
    # nchunks steps instead of exploding.
    #
    # For backward, _state_passing_bwd's "states" argument is documented as
    # the forward's own running-sum output (see StatePassingFn.backward,
    # which passes `out`, not the raw per-chunk `states` input) -- so this
    # test does the same, unlike the sibling test below (which just needs
    # *some* same-shape tensor for a self-consistency check and reuses the
    # raw `states` input for convenience). With dout = 1 (adjoint of
    # out.sum()), the standard backward-recursion adjoint G_j = dout[j] +
    # r[h]*G_{j+1} (G_nchunks = 0, no dfinal_states) solves to
    # G_j = S_h(nchunks - j), giving:
    #   dstates[c, h] = G_{c+1} = S_h(nchunks - c - 1)
    #   ddA[c, h]     = dim * G_{c+1} * r[h] * y_c = dim * S_h(nchunks - c - 1) * r[h] * S_h(c)
    #                   (the extra `dim` factor is _state_passing_bwd summing
    #                   out[p]*dstates[p]*scale over the dim axis, which is
    #                   constant across p here since out/dstates are)
    # Verified by hand against a small (nchunks=4) case, cross-checked
    # against torch.autograd through the real StatePassingFn, before
    # committing to this closed form.
    device = 'cuda'

    batch = 1
    nheads = 128
    nchunks = 64
    dim = 64
    width = 300_000  # padding only, not a real chunk_size

    parent = torch.zeros(batch, nheads, nchunks, width, dtype=torch.float32, device=device)
    dA_chunk_cumsum = parent[:, :, :, -1]
    a = -(torch.arange(1, nheads + 1, dtype=torch.float32, device=device)) * 0.01
    dA_chunk_cumsum.copy_(a[None, :, None])
    assert not dA_chunk_cumsum.is_contiguous()
    assert (nheads - 1) * dA_chunk_cumsum.stride(1) > 2**31 - 1

    states = torch.ones(batch, nchunks, nheads, dim, dtype=torch.float32, device=device)

    out, final_states = _state_passing_fwd(states, dA_chunk_cumsum)
    torch.cuda.synchronize(device)

    r = torch.exp(a)  # (nheads,)

    def S(n):
        return (1 - r ** n) / (1 - r)

    j = torch.arange(nchunks, dtype=torch.float32, device=device)
    S_j = S(j[:, None])  # (nchunks, nheads), matches out's (chunk, head) axis order
    expected_out = S_j[None, :, :, None].expand(batch, nchunks, nheads, dim)
    torch.testing.assert_close(out, expected_out, rtol=1e-3, atol=0)
    expected_final_states = S(torch.tensor(float(nchunks), device=device))[None, :, None].expand(batch, nheads, dim)
    torch.testing.assert_close(final_states, expected_final_states, rtol=1e-3, atol=0)

    dout = torch.ones(batch, nchunks, nheads, dim, dtype=torch.float32, device=device)
    dstates, ddA, _ = _state_passing_bwd(out, dA_chunk_cumsum, dout, has_initial_states=False)
    torch.cuda.synchronize(device)

    c = torch.arange(nchunks, dtype=torch.float32, device=device)
    G_next = S((nchunks - c - 1)[:, None])  # (nchunks, nheads): G_{c+1} for each chunk c
    expected_dstates = G_next[None, :, :, None].expand(batch, nchunks, nheads, dim)
    torch.testing.assert_close(dstates, expected_dstates, rtol=1e-3, atol=0)
    # ddA sums out[p]*dstates[p]*scale over the full dim axis inside the
    # kernel; out and dstates are both constant across dim here, so that
    # sum is just dim * (the scalar formula).
    expected_ddA_jh = dim * G_next * r[None, :] * S_j  # (nchunks, nheads): G_{c+1} * r[h] * S_h(c)
    expected_ddA = expected_ddA_jh.transpose(0, 1)[None, :, :].expand(batch, nheads, nchunks)
    torch.testing.assert_close(ddA, expected_ddA, rtol=1e-3, atol=0)


@gpu_memory_skipif(12)
def test_state_passing_fwd_bwd_noncontiguous_dA_chunk_cumsum_no_overflow() -> None:
    # int64 to avoid int32 overflow, see https://github.com/state-spaces/mamba/issues/1015.
    # dA_chunk_cumsum is *always* non-contiguous in production
    # (ssd_combined.py slices dA_cumsum[:, :, :, -1]), but only reaches
    # overflow-relevant scale once nheads * nchunks is large. Reproduced
    # directly here via a padded parent tensor -- states/dout stay small
    # and cheap; only the padding dim (not nheads/nchunks/dim) needs to be
    # huge to hit the threshold.
    device = 'cuda'

    torch.manual_seed(0)
    batch = 1
    nheads = 128
    nchunks = 64
    dim = 64
    width = 300_000  # padding only, not a real chunk_size

    parent = torch.randn(batch, nheads, nchunks, width, dtype=torch.float32, device=device)
    dA_chunk_cumsum = parent[:, :, :, -1]
    assert not dA_chunk_cumsum.is_contiguous()
    assert (nheads - 1) * dA_chunk_cumsum.stride(1) > 2**31 - 1
    dA_chunk_cumsum_c = dA_chunk_cumsum.contiguous()

    states = torch.randn(batch, nchunks, nheads, dim, dtype=torch.float32, device=device)
    dout = torch.randn(batch, nchunks, nheads, dim, dtype=torch.float32, device=device)

    out, final_states = _state_passing_fwd(states, dA_chunk_cumsum)
    out_ref, final_states_ref = _state_passing_fwd(states, dA_chunk_cumsum_c)
    torch.cuda.synchronize(device)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close(final_states, final_states_ref)

    dstates, ddA, _ = _state_passing_bwd(states, dA_chunk_cumsum, dout, has_initial_states=False)
    dstates_ref, ddA_ref, _ = _state_passing_bwd(states, dA_chunk_cumsum_c, dout, has_initial_states=False)
    torch.cuda.synchronize(device)
    assert torch.isfinite(dstates).all()
    torch.testing.assert_close(dstates, dstates_ref)
    torch.testing.assert_close(ddA, ddA_ref)


@gpu_memory_skipif(12)
def test_state_passing_fwd_bwd_noncontiguous_states_batch_axis_no_overflow_known_answer() -> None:
    # Batch-axis counterpart to the dA_chunk_cumsum known-answer test above,
    # targeting `pid_b * stride_states_batch` instead. `states` (the
    # per-chunk increment s_c) is what needs to be wide/batch-addressed
    # here, so it -- not dA_chunk_cumsum -- carries the batch-distinguishing
    # value: states[b, c] = v[b] = b + 1 (constant across c), decay
    # dA_chunk_cumsum[b] = a[b] = -(b + 1) * 0.05 (also batch-distinct, for
    # extra rigor, though not itself the tensor under test here).
    #
    # This just rescales the same geometric-series formulas from above by
    # v[b] (since the recurrence is linear in s_c, and dstates doesn't
    # depend on the states' own values, only on the decay and dout):
    #   out[j, b]        = v[b] * S_b(j)
    #   final_states[b]  = v[b] * S_b(nchunks)
    #   dstates[c, b]    = S_b(nchunks - c - 1)                (unscaled)
    #   ddA[c, b]        = S_b(nchunks - c - 1) * r[b] * v[b] * S_b(c)
    # nheads = dim = 1 here (matching the sibling test's minimal shape), so
    # there's no extra dim-summation factor this time.
    device = 'cuda'

    batch = 8
    nchunks = 2
    nheads = 1
    dim = 1
    width = 335_000_000  # (batch - 1) * width > 2**31 - 1, with ~9% margin

    parent = torch.zeros(batch, width, dtype=torch.float32, device=device)
    states = parent[:, :nchunks * nheads * dim].view(batch, nchunks, nheads, dim)
    v = torch.arange(1, batch + 1, dtype=torch.float32, device=device)
    states.copy_(v[:, None, None, None])
    assert not states.is_contiguous()
    assert (batch - 1) * states.stride(0) > 2**31 - 1, "test parameters too small to overflow the batch-axis term"

    a = -(torch.arange(1, batch + 1, dtype=torch.float32, device=device)) * 0.05
    dA_chunk_cumsum = a[:, None, None].expand(batch, nheads, nchunks).contiguous()

    out, final_states = _state_passing_fwd(states, dA_chunk_cumsum)
    torch.cuda.synchronize(device)

    r = torch.exp(a)  # (batch,)

    def S(n):
        return (1 - r ** n) / (1 - r)

    j = torch.arange(nchunks, dtype=torch.float32, device=device)
    S_j = S(j[:, None])  # (nchunks, batch)
    vs_j = (v[None, :] * S_j).transpose(0, 1)  # (batch, nchunks)
    expected_out = vs_j[:, :, None, None].expand(batch, nchunks, nheads, dim)
    torch.testing.assert_close(out, expected_out, rtol=1e-4, atol=0)
    expected_final_states = (v * S(torch.tensor(float(nchunks), device=device)))[:, None, None].expand(batch, nheads, dim)
    torch.testing.assert_close(final_states, expected_final_states, rtol=1e-4, atol=0)

    dout = torch.ones(batch, nchunks, nheads, dim, dtype=torch.float32, device=device)
    dstates, ddA, _ = _state_passing_bwd(out, dA_chunk_cumsum, dout, has_initial_states=False)
    torch.cuda.synchronize(device)

    c = torch.arange(nchunks, dtype=torch.float32, device=device)
    G_next = S((nchunks - c - 1)[:, None])  # (nchunks, batch)
    expected_dstates = G_next.transpose(0, 1)[:, :, None, None].expand(batch, nchunks, nheads, dim)
    torch.testing.assert_close(dstates, expected_dstates, rtol=1e-4, atol=0)
    expected_ddA_jb = G_next * r[None, :] * v[None, :] * S_j  # (nchunks, batch)
    expected_ddA = expected_ddA_jb.transpose(0, 1)[:, None, :].expand(batch, nheads, nchunks)
    torch.testing.assert_close(ddA, expected_ddA, rtol=1e-4, atol=0)


@gpu_memory_skipif(12)
def test_state_passing_fwd_bwd_noncontiguous_states_batch_axis_no_overflow() -> None:
    # Batch-axis counterpart to test_state_passing_fwd_bwd_noncontiguous_dA_chunk_cumsum_no_overflow
    # above: same pid_bc = tl.program_id(...).to(tl.int64) pattern, but for
    # `pid_b * stride_states_batch` instead of `pid_h * stride_dA_cs_head`.
    # `states` here isn't naturally non-contiguous in production the way
    # dA_chunk_cumsum is, so this reproduces the pattern directly via a
    # padded parent tensor on the *batch* dimension -- unlike padding an
    # inner dim, the padding width here is inherently tied to the
    # (batch - 1) * width > 2**31 - 1 threshold regardless of how it's
    # split, so there's no way to shrink this one below roughly
    # threshold * dtype_size total memory.
    device = 'cuda'

    torch.manual_seed(0)
    batch = 8
    nchunks = 2
    nheads = 1
    dim = 1
    width = 335_000_000  # (batch - 1) * width > 2**31 - 1, with ~9% margin

    parent = torch.randn(batch, width, dtype=torch.float32, device=device)
    states = parent[:, :nchunks * nheads * dim].view(batch, nchunks, nheads, dim)
    assert not states.is_contiguous()
    assert (batch - 1) * states.stride(0) > 2**31 - 1, "test parameters too small to overflow the batch-axis term"
    states_c = states.contiguous()

    dA_chunk_cumsum = torch.randn(batch, nheads, nchunks, dtype=torch.float32, device=device)
    dout = torch.randn(batch, nchunks, nheads, dim, dtype=torch.float32, device=device)

    out, final_states = _state_passing_fwd(states, dA_chunk_cumsum)
    out_ref, final_states_ref = _state_passing_fwd(states_c, dA_chunk_cumsum)
    torch.cuda.synchronize(device)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close(final_states, final_states_ref)

    dstates, ddA, _ = _state_passing_bwd(states, dA_chunk_cumsum, dout, has_initial_states=False)
    dstates_ref, ddA_ref, _ = _state_passing_bwd(states_c, dA_chunk_cumsum, dout, has_initial_states=False)
    torch.cuda.synchronize(device)
    assert torch.isfinite(dstates).all()
    torch.testing.assert_close(dstates, dstates_ref)
    torch.testing.assert_close(ddA, ddA_ref)


@gpu_memory_skipif(12)
def test_state_passing_fwd_bwd_noncontiguous_states_dim_axis_no_overflow_known_answer() -> None:
    # Dim-axis counterpart to the batch-axis known-answer test above,
    # targeting `offs_m * stride_states_dim` instead. With nchunks = 1
    # (matching the sibling test below): the fwd kernel's loop runs exactly
    # once, and since c(=0) < nchunks-1(=0) is false, that iteration writes
    # straight to final_states instead of out -- so out[0] keeps its
    # initial-store value of 0 (out never reflects states or dA_chunk_cumsum
    # at all here), and final_states = exp(dA_cs)*0 + states[0] = states[0]
    # exactly, regardless of dA_chunk_cumsum's value. Setting
    # states[0, p] = p (distinct per dim position, not a shared constant)
    # makes final_states[p] = p a trivial but exact, dim-position-sensitive
    # closed form -- a misaddressed dim read is caught by reading a
    # different position's distinct value.
    #
    # The backward loop runs range(nchunks - 1) = range(0) times -- zero
    # iterations -- so with no dfinal_states, dstates is just zeros and
    # ddA_chunk_cumsum is never written by the kernel at all (its output
    # buffer is uninitialized `torch.empty`, not meaningfully defined at
    # this nchunks=1 shape) -- so only dstates is checked below, not ddA.
    device = 'cuda'

    batch = 1
    nchunks = 1
    nheads = 1
    dim = 100_000
    pad = 23_400  # (dim - 1) * pad > 2**31 - 1, with ~9% margin

    parent = torch.zeros(batch, nchunks, nheads, dim, pad, dtype=torch.float32, device=device)
    states = parent[:, :, :, :, 0]
    p = torch.arange(dim, dtype=torch.float32, device=device)
    states.copy_(p[None, None, None, :])
    assert not states.is_contiguous()
    assert (dim - 1) * states.stride(3) > 2**31 - 1, "test parameters too small to overflow the dim-axis term"

    dA_chunk_cumsum = torch.randn(batch, nheads, nchunks, dtype=torch.float32, device=device)

    out, final_states = _state_passing_fwd(states, dA_chunk_cumsum)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
    expected_final_states = p[None, None, :]
    torch.testing.assert_close(final_states, expected_final_states, rtol=0, atol=0)

    dout = torch.zeros(batch, nchunks, nheads, dim, dtype=torch.float32, device=device)
    dstates, _, _ = _state_passing_bwd(out, dA_chunk_cumsum, dout, has_initial_states=False)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(dstates, torch.zeros_like(dstates), rtol=0, atol=0)


@gpu_memory_skipif(12)
def test_state_passing_fwd_bwd_noncontiguous_states_dim_axis_no_overflow() -> None:
    # Third pid cast in the same kernels: `offs_m * stride_states_dim`,
    # where offs_m is derived from pid_m = tl.program_id(axis=0) (the
    # per-head state-dim block index), separate from the pid_b/pid_h terms
    # above. See https://github.com/state-spaces/mamba/issues/1015. `dim`'s own stride is 1 in
    # production (states is freshly allocated, contiguous), so unlike
    # batch/head this doesn't arise naturally -- reproduced directly here
    # via a parent tensor padded on a *new* trailing axis (same technique
    # as the dA_chunk_cumsum test), so dim's own size can stay small while
    # its stride is large.
    #
    # Confirmed via direct reproduction before this fix: reverting pid_m's
    # cast crashes with an illegal memory access at this exact scale, same
    # as the pid_b/ssd_bmm cases -- this was not just a theoretical gap.
    device = 'cuda'

    torch.manual_seed(0)
    batch = 1
    nchunks = 1
    nheads = 1
    dim = 100_000
    pad = 23_400  # (dim - 1) * pad > 2**31 - 1, with ~9% margin

    parent = torch.randn(batch, nchunks, nheads, dim, pad, dtype=torch.bfloat16, device=device)
    states = parent[:, :, :, :, 0]
    assert not states.is_contiguous()
    assert (dim - 1) * states.stride(3) > 2**31 - 1, "test parameters too small to overflow the dim-axis term"
    states_c = states.contiguous()

    dA_chunk_cumsum = torch.randn(batch, nheads, nchunks, dtype=torch.float32, device=device)
    dout = torch.randn(batch, nchunks, nheads, dim, dtype=torch.float32, device=device)

    out, final_states = _state_passing_fwd(states, dA_chunk_cumsum)
    out_ref, final_states_ref = _state_passing_fwd(states_c, dA_chunk_cumsum)
    torch.cuda.synchronize(device)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), out_ref.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(final_states, final_states_ref)

    dstates, ddA, _ = _state_passing_bwd(states, dA_chunk_cumsum, dout, has_initial_states=False)
    dstates_ref, ddA_ref, _ = _state_passing_bwd(states_c, dA_chunk_cumsum, dout, has_initial_states=False)
    torch.cuda.synchronize(device)
    assert torch.isfinite(dstates).all()
    torch.testing.assert_close(dstates.float(), dstates_ref.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ddA, ddA_ref)


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
