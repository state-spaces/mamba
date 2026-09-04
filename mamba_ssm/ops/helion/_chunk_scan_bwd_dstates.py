from __future__ import annotations

import math
from typing import Optional

from einops import rearrange
import torch
import triton

import helion
import helion.language as hl

from ..triton.ssd_chunk_scan import (
    _chunk_scan_bwd_dstates as _chunk_scan_bwd_dstates_ref,
)

from .utils import json_cached_autotune


@json_cached_autotune
@helion.kernel(
    autotune_accuracy_check=False,
    autotune_baseline_fn=_chunk_scan_bwd_dstates_ref,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _chunk_scan_bwd_dstates_kernel(
    C: torch.Tensor,
    dA_cumsum: torch.Tensor,
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    batch, seqlen, nheads, headdim = dout.shape
    _, _, nchunks, chunk_size = dA_cumsum.shape
    _, _, ngroups, dstate = C.shape
    assert seqlen % chunk_size == 0
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    dtype = C.dtype if dtype is None else dtype
    dprev_states = torch.empty(
        batch,
        nchunks,
        nheads,
        headdim,
        dstate,
        device=C.device,
        dtype=dtype,
    )

    HAS_SEQ_IDX = seq_idx is not None

    block_m = hl.register_block_size(headdim)
    block_n = hl.register_block_size(dstate)
    block_k = hl.register_block_size(chunk_size)

    dout = torch.transpose(dout, 1, 3)

    for tile_c, tile_b, tile_h, tile_m, tile_n in hl.tile(
        [nchunks, batch, nheads, headdim, dstate],
        block_size=[1, 1, 1, block_m, block_n],
    ):
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        if HAS_SEQ_IDX:
            if tile_c.id == 0:
                seq_idx_prev_local = hl.zeros([], dtype=seq_idx.dtype)
            else:
                seq_idx_prev_local = seq_idx[
                    tile_b.begin,
                    tile_c.begin * chunk_size - 1,
                ]
        for tile_k in hl.tile(chunk_size, block_size=block_k):
            dout_local = dout[
                tile_b.begin,
                tile_m,
                tile_h.begin,
                tile_c.begin * chunk_size + tile_k.index,
            ].to(torch.float32)
            dA_cs_k_local = dA_cumsum[
                tile_b.begin,
                tile_h.begin,
                tile_c.begin,
                tile_k,
            ].to(torch.float32)
            scale_k = torch.exp(dA_cs_k_local)
            if HAS_SEQ_IDX:
                seq_idx_k_local = seq_idx[
                    tile_b.begin,
                    tile_c.begin * chunk_size + tile_k.index,
                ]
                scale_k = torch.where(
                    seq_idx_k_local == seq_idx_prev_local,
                    scale_k,
                    torch.zeros_like(scale_k),
                )
            dout_local = (dout_local * scale_k).to(dout.dtype)
            c_local = C[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_k.index,
                tile_h.begin // (nheads // ngroups),
                tile_n,
            ].to(dout.dtype)
            acc = hl.dot(dout_local, c_local, acc=acc)
        dprev_states[
            tile_b.begin,
            tile_c.begin,
            tile_h.begin,
            tile_m,
            tile_n,
        ] = acc

    return dprev_states


def _chunk_scan_bwd_dstates(
    C: torch.Tensor,
    dA_cumsum: torch.Tensor,
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return _chunk_scan_bwd_dstates_kernel(C, dA_cumsum, dout, seq_idx, dtype)
