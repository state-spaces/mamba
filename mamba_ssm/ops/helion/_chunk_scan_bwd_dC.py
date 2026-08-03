from __future__ import annotations

import math
from typing import Optional

import torch

import helion
import helion.language as hl
from helion.autotuner import IntegerFragment

from ..triton.ssd_chunk_scan import (
    _chunk_scan_bwd_dC as _chunk_scan_bwd_dC_ref,
)

from .utils import json_cached_autotune


@json_cached_autotune
@helion.kernel(
    autotune_accuracy_check=False,
    autotune_baseline_fn=_chunk_scan_bwd_dC_ref,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _chunk_scan_bwd_dC_kernel(
    prev_states: torch.Tensor,
    dA_cumsum: torch.Tensor,
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor] = None,
    C: Optional[torch.Tensor] = None,
    ngroups: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    _, _, _, chunk_size = dA_cumsum.shape
    assert seqlen % chunk_size == 0
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == (batch, seqlen, nheads, headdim)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if C is not None:
        assert C.shape == (batch, seqlen, ngroups, dstate)
        ddA_cumsum_prev = torch.zeros(
            batch,
            nheads,
            nchunks,
            chunk_size,
            device=dout.device,
            dtype=torch.float32,
        )
    else:
        ddA_cumsum_prev = None

    ngroups = hl.specialize(ngroups)
    nheads_ngroups_ratio = nheads // ngroups
    nheads_per_program = hl.register_tunable(
        "nheads_per_program",
        IntegerFragment(1, nheads_ngroups_ratio),
    )
    nsplits = helion.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dC = torch.empty(
        batch,
        seqlen,
        nsplits,
        ngroups,
        dstate,
        device=dout.device,
        dtype=torch.float32,
    )

    HAS_DDA_CS = ddA_cumsum_prev is not None
    HAS_SEQ_IDX = seq_idx is not None
    block_m = hl.register_block_size(chunk_size)
    block_n = hl.register_block_size(dstate)

    for tile_c, tile_b, tile_s, tile_g, tile_m, tile_n in hl.tile(
        [nchunks, batch, nsplits, ngroups, chunk_size, dstate],
        block_size=[1, 1, 1, 1, block_m, block_n],
    ):
        acc = hl.zeros((block_m, block_n), dtype=torch.float32)
        if HAS_DDA_CS:
            c_local = C[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_m.index,
                tile_g.begin,
                tile_n,
            ].to(torch.float32)
        if HAS_SEQ_IDX:
            if tile_c.id == 0:
                seq_idx_prev_local = hl.zeros([], dtype=seq_idx.dtype)
            else:
                seq_idx_prev_local = seq_idx[
                    tile_b.begin,
                    tile_c.begin * chunk_size - 1,
                ]
            seq_idx_m_local = seq_idx[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_m.index,
            ]

        nheads_iter = min(
            nheads_per_program,
            nheads // ngroups - tile_s.id * nheads_per_program,
        )
        for tile_h in hl.tile(nheads_iter, block_size=1):
            head_idx = (
                tile_g.begin * (nheads // ngroups)
                + tile_s.begin * nheads_per_program
                + tile_h.begin
            )
            dout_local = dout[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_m.index,
                head_idx,
                :,
            ]
            prev_states_local = prev_states[
                tile_b.begin,
                tile_c.begin,
                head_idx,
                :,
                tile_n,
            ].to(dout.dtype)
            dc = hl.dot(dout_local, prev_states_local)
            dA_cs_m_local = dA_cumsum[
                tile_b.begin,
                head_idx,
                tile_c.begin,
                tile_m,
            ].to(torch.float32)
            scale = torch.exp(dA_cs_m_local)
            if HAS_SEQ_IDX:
                scale = torch.where(
                    seq_idx_m_local == seq_idx_prev_local,
                    scale,
                    torch.zeros_like(scale),
                )
            dc *= scale[:, None]
            if HAS_DDA_CS:
                ddA_cs_local = torch.sum(dc * c_local, dim=1)
                hl.atomic_add(
                    ddA_cumsum_prev,
                    [
                        tile_b.begin,
                        head_idx,
                        tile_c.begin,
                        tile_m,
                    ],
                    ddA_cs_local,
                )
            acc += dc

        dC[
            tile_b.begin,
            tile_c.begin * chunk_size + tile_m.index,
            tile_s.begin,
            tile_g.begin,
            tile_n,
        ] = acc

    dC = dC.sum(2)
    return dC if C is None else (dC, ddA_cumsum_prev)


def _chunk_scan_bwd_dC(
    prev_states: torch.Tensor,
    dA_cumsum: torch.Tensor,
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor] = None,
    C: Optional[torch.Tensor] = None,
    ngroups: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _chunk_scan_bwd_dC_kernel(
        prev_states,
        dA_cumsum,
        dout,
        seq_idx,
        C,
        ngroups,
    )
