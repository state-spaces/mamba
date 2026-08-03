# https://github.com/pytorch/helion/blob/v1.0.0/examples/mamba2_chunk_state.py

from __future__ import annotations

from typing import Optional

import torch

import helion
import helion.language as hl

from .utils import json_cached_autotune

from ..triton.ssd_chunk_state import (
    _chunk_state_fwd as _chunk_state_fwd_ref,
)


@json_cached_autotune
@helion.kernel(
    autotune_accuracy_check=False,
    autotune_baseline_fn=_chunk_state_fwd_ref,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _chunk_state_fwd_kernel(
    B: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    seq_idx: Optional[torch.Tensor] = None,
    states: Optional[torch.Tensor] = None,
    states_in_fp32: hl.constexpr = True,
) -> torch.Tensor:
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert seqlen % chunk_size == 0
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    assert states is not None
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)

    acc_dtype = torch.float32
    # 1 / ln(2)
    p = 1.44269504089
    block_m = hl.register_block_size(headdim)
    block_n = hl.register_block_size(dstate)
    block_k = hl.register_block_size(chunk_size)

    HAS_SEQ_IDX = seq_idx is not None

    for tile_h, tile_m, tile_n, tile_b, tile_c in hl.tile(
        [nheads, headdim, dstate, batch, nchunks],
        block_size=[1, block_m, block_n, 1, 1],
    ):
        dA_cs_last_local = (
            dA_cumsum[
                tile_b.begin,
                tile_h.begin,
                tile_c.begin,
                chunk_size - 1,
            ].to(torch.float32)
            * p
        )
        if HAS_SEQ_IDX:
            seq_idx_last_local = seq_idx[
                tile_b.begin,
                tile_c.begin * chunk_size + chunk_size - 1,
            ]

        acc = hl.zeros([tile_m, tile_n], dtype=acc_dtype)
        for tile_k in hl.tile(chunk_size, block_size=block_k):
            x_local = x[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_k.index,
                tile_h.begin,
                tile_m,
            ]
            dA_cs_k_local = (
                dA_cumsum[
                    tile_b.begin,
                    tile_h.begin,
                    tile_c.begin,
                    tile_k,
                ].to(torch.float32)
                * p
            )
            dt_k_local = dt[
                tile_b.begin,
                tile_h.begin,
                tile_c.begin,
                tile_k,
            ].to(torch.float32)
            scale = (
                torch.exp2((dA_cs_last_local - dA_cs_k_local).clamp(max=0.0))
                * dt_k_local
            )
            if HAS_SEQ_IDX:
                seq_idx_k_local = seq_idx[
                    tile_b.begin,
                    tile_c.begin * chunk_size + tile_k.index,
                ]
                scale_mask = seq_idx_k_local == seq_idx_last_local
                scale = torch.where(scale_mask, scale, torch.zeros_like(scale))
            b_local = B[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_k.index,
                tile_h.begin // (nheads // ngroups),
                tile_n,
            ].to(torch.float32)
            b_local *= scale[:, None]
            b_local = b_local.to(x.dtype)
            acc = hl.dot(x_local.T, b_local, acc=acc)

        states[
            tile_b.begin,
            tile_c.begin,
            tile_h.begin,
            tile_m,
            tile_n,
        ] = acc

    return states


def _chunk_state_fwd(
    B: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    seq_idx: Optional[torch.Tensor] = None,
    states: Optional[torch.Tensor] = None,
    states_in_fp32: bool = True,
) -> torch.Tensor:
    batch, _, nheads, headdim = x.shape
    _, _, nchunks, _ = dt.shape
    _, _, _, dstate = B.shape
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype)
    return _chunk_state_fwd_kernel(B, x, dt, dA_cumsum, seq_idx, states, states_in_fp32)
