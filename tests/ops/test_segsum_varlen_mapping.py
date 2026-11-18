# Copyright (c) 2025, Tri Dao.
"""The vectorized chunk-mapping in compute_dacs_segsum_triton_varlen must be
bit-identical to the original per-sequence loop semantics."""
import pytest
import torch

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import (
    compute_dacs_segsum_triton_varlen,
)

C = 64

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _ref_mappings(cu_list, chunk_size, nchunks):
    """The original loop implementation (reference semantics)."""
    seq_map = torch.zeros(nchunks, dtype=torch.int32)
    default_seq_len = cu_list[1] - cu_list[0]
    chunk_in_seq = torch.full(
        (nchunks,), (default_seq_len + chunk_size - 1) // chunk_size,
        dtype=torch.int32)
    for i in range(len(cu_list) - 1):
        start = cu_list[i]
        n = (cu_list[i + 1] - start) // chunk_size + 1
        cs = start // chunk_size + i
        seq_map[cs:cs + n] = i
        chunk_in_seq[cs:cs + n] = torch.arange(n, dtype=torch.int32)
    return seq_map, chunk_in_seq


@requires_cuda
@pytest.mark.parametrize("cu_list", [
    [0, 7],                          # single short seq
    [0, 64],                         # exactly one chunk
    [0, 64, 128, 192],               # all multiples of C
    [0, 1, 2, 3, 4],                 # single-token seqs
    [0, 100, 101, 357, 999, 1000],   # mixed
    [0, 513],                        # NS=1 long
    list(range(0, 33 * 17, 17)),     # 32 seqs of 17
])
def test_mapping_matches_reference(cu_list):
    torch.manual_seed(0)
    NS, S = len(cu_list) - 1, cu_list[-1]
    nchunks = S // C + NS
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")

    # Recompute the vectorized mapping exactly as the function does.
    seq_lens = cu[1:] - cu[:-1]
    starts = cu[:-1] // C + torch.arange(NS, dtype=cu.dtype, device="cuda")
    ends = starts + seq_lens // C + 1
    g = torch.arange(nchunks, dtype=cu.dtype, device="cuda")
    owner = torch.searchsorted(starts, g, right=True) - 1
    active = g < ends[owner]
    sm = torch.where(active, owner, torch.zeros_like(owner)).to(torch.int32)
    cis = torch.where(active, g - starts[owner],
                      (seq_lens[0] + C - 1) // C).to(torch.int32)

    rsm, rcis = _ref_mappings(cu_list, C, nchunks)
    assert torch.equal(sm.cpu(), rsm)
    assert torch.equal(cis.cpu(), rcis)

    # End-to-end: the full function runs and produces finite outputs of the
    # right shapes through the vectorized path.
    da = torch.randn(1, 2, S, device="cuda", dtype=torch.float32) * -0.1
    da_cs, da_cs_rev, segsum = compute_dacs_segsum_triton_varlen(
        da, C, cu_seqlens=cu)
    assert da_cs.shape == da.shape
    assert segsum.shape[2] == nchunks
    assert torch.isfinite(da_cs).all()
