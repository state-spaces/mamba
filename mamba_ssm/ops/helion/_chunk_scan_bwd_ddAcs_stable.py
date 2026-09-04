from __future__ import annotations

from typing import Optional

import torch

import helion
import helion.language as hl

from ..triton.ssd_chunk_scan import (
    _chunk_scan_bwd_ddAcs_stable as _chunk_scan_bwd_ddAcs_stable_ref,
)

from .utils import json_cached_autotune


@json_cached_autotune
@helion.kernel(
    autotune_accuracy_check=False,
    autotune_baseline_fn=_chunk_scan_bwd_ddAcs_stable_ref,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _chunk_scan_bwd_ddAcs_stable_kernel(
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    dout: torch.Tensor,
    cb: torch.Tensor,
) -> torch.Tensor:
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen % chunk_size == 0
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == x.shape
    assert dA_cumsum.shape == dt.shape
    ngroups = cb.shape[2]
    assert nheads % ngroups == 0
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)

    BLOCK_SIZE_M_min = 32
    ddA_cumsum = torch.zeros(
        batch,
        nheads,
        nchunks,
        helion.cdiv(chunk_size, BLOCK_SIZE_M_min),
        chunk_size,
        device=x.device,
        dtype=torch.float32,
    )

    p = 1.44269504089  # 1 / ln(2)
    block_m = hl.register_block_size(BLOCK_SIZE_M_min, chunk_size)
    block_n = hl.register_block_size(chunk_size)

    x = torch.transpose(x, 1, 3)

    for tile_c, tile_b, tile_h, tile_m in hl.tile(
        [nchunks, batch, nheads, chunk_size],
        block_size=[1, 1, 1, block_m],
    ):
        rowsum_local = hl.zeros([block_m], dtype=torch.float32)
        dout_local = dout[
            tile_b.begin,
            tile_c.begin * chunk_size + tile_m.index,
            tile_h.begin,
            :,
        ]
        dA_cs_m_local = (
            dA_cumsum[
                tile_b.begin,
                tile_h.begin,
                tile_c.begin,
                tile_m,
            ].to(torch.float32)
            * p
        )

        lo, hi = 0, (tile_m.id + 1) * block_m
        for tile_n in hl.tile(lo, hi, block_size=block_n):
            x_local = x[
                tile_b.begin,
                :,
                tile_h.begin,
                tile_c.begin * chunk_size + tile_n.index,
            ]
            acc = hl.dot(dout_local, x_local)
            dt_n_local = dt[
                tile_b.begin,
                tile_h.begin,
                tile_c.begin,
                tile_n,
            ].to(torch.float32)
            acc *= dt_n_local
            cb_local = cb[
                tile_b.begin,
                tile_c.begin,
                tile_h.begin // (nheads // ngroups),
                tile_m,
                tile_n,
            ].to(torch.float32)
            acc *= cb_local
            dA_cs_n_local = (
                dA_cumsum[
                    tile_b.begin,
                    tile_h.begin,
                    tile_c.begin,
                    tile_n,
                ].to(torch.float32)
                * p
            )
            acc *= torch.exp2(
                (dA_cs_m_local[:, None] - dA_cs_n_local[None, :]).clamp(max=0.0)
            )
            mask = (tile_m.index + 0)[:, None] >= (tile_n.index + 0)[None, :] + 1
            acc = torch.where(mask, acc, torch.zeros_like(acc))
            rowsum_new_local = rowsum_local + torch.sum(acc, dim=1)
            acc = rowsum_local[:, None] + torch.cumsum(acc, dim=1)
            rowsum_local = rowsum_new_local
            acc = torch.where(mask, acc, torch.zeros_like(acc))
            ddA_cs_local = torch.sum(acc, dim=0)
            hl.store(
                ddA_cumsum,
                (
                    tile_b.begin,
                    tile_h.begin,
                    tile_c.begin,
                    tile_m.id,
                    tile_n.index + 1,
                ),
                ddA_cs_local,
                extra_mask=tile_n.index + 1 < chunk_size,
            )

    n_valid_blocks = (chunk_size + block_m - 1) // block_m
    ddA_cumsum = ddA_cumsum[:, :, :, :n_valid_blocks].sum(dim=3)
    return ddA_cumsum


def _chunk_scan_bwd_ddAcs_stable(
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    dout: torch.Tensor,
    cb: torch.Tensor,
) -> torch.Tensor:
    return _chunk_scan_bwd_ddAcs_stable_kernel(x, dt, dA_cumsum, dout, cb)
