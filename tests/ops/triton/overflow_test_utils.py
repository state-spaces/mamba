import math

import pytest
import torch


def gpu_memory_skipif(required_gib, device='cuda'):
    """Shared by the int32-overflow regression tests, which all need a wide
    (large stride(1)) tensor to trigger the bug -- skip rather than OOM on
    GPUs that are otherwise perfectly capable of running the suite:

        @gpu_memory_skipif(9)
        def test_something_large() -> None:
            ...

    Evaluates the GPU's total memory at collection time (not skipped lazily
    inside the test), so it shows up as SKIPPED (not run) in pytest's summary.
    """
    total_memory = torch.cuda.get_device_properties(device).total_memory
    required_memory = required_gib * 1024**3
    return pytest.mark.skipif(
        total_memory < required_memory,
        reason=f"GPU has {total_memory / 1024**3:.1f} GiB, need >= {required_gib} GiB",
    )


def assert_isfinite(t, chunk_rows=4096):
    # torch.isfinite(t).all() on a large low-precision tensor (e.g. bf16)
    # internally promotes to a full-size fp32 intermediate before reducing,
    # roughly tripling peak memory over t's own footprint at row counts like
    # the ones these overflow tests use -- checking row-chunks at a time
    # keeps that intermediate bounded, at the same full-coverage guarantee.
    for i in range(0, t.shape[0], chunk_rows):
        assert torch.isfinite(t[i:i + chunk_rows]).all(), f"non-finite value in rows [{i}, {i + chunk_rows})"


def wide_noncontiguous_slices(device, seqlen, parent_width, shapes, batch=None):
    """Allocate ONE (batch, seqlen, parent_width) parent tensor -- or
    (seqlen, parent_width) if batch is None -- and return disjoint,
    non-contiguous slices of it, one per entry in `shapes` (each a tuple of
    trailing dims whose product is that slice's width). Every slice shares
    the same large stride(1) == parent_width regardless of which columns it
    takes, since slicing the last dim of a contiguous tensor doesn't change
    the stride of an earlier dim -- so this is used by every int32-overflow
    regression test needing multiple wide, non-contiguous tensors, at
    the memory cost of ONE parent instead of one per tensor.
    """
    widths = [math.prod(shape) for shape in shapes]
    assert parent_width >= sum(widths)
    parent_shape = (seqlen, parent_width) if batch is None else (batch, seqlen, parent_width)
    parent = torch.randn(parent_shape, dtype=torch.bfloat16, device=device)
    slices = []
    offset = 0
    for width, shape in zip(widths, shapes):
        leading = () if batch is None else (batch,)
        slices.append(parent[..., offset:offset + width].view(*leading, seqlen, *shape))
        offset += width
    return slices


def bwd_row_start_max(M, num_programs):
    # Shared by layer_norm.py/layernorm_gated.py overflow tests: both
    # backward kernels launch only `num_programs` programs (not one per
    # row), each covering `rows_per_program = ceil(M / num_programs)` rows,
    # so the largest row_start is (num_programs - 1) * rows_per_program --
    # well under M - 1. N must be sized so this, not just M * N, overflows.
    rows_per_program = math.ceil(M / num_programs)
    return (num_programs - 1) * rows_per_program
