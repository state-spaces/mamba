# https://github.com/pytorch/helion/blob/v1.0.0/examples/mamba2_chunk_scan.py

from __future__ import annotations

from typing import Optional

import torch

import helion
import helion.language as hl

from .utils import json_cached_autotune

from ..triton.ssd_chunk_scan import _chunk_scan_fwd as _chunk_scan_fwd_ref


@json_cached_autotune
@helion.kernel(
    autotune_accuracy_check=False,
    autotune_baseline_fn=_chunk_scan_fwd_ref,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _chunk_scan_fwd_kernel(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    states: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert seqlen % chunk_size == 0
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

    # Allocates output.
    out = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
    if z is not None:
        out_x = torch.empty(
            batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype
        )
        assert out_x.stride() == out.stride()
    else:
        out_x = None

    acc_dtype = torch.float32
    # 1 / ln(2)
    p = 1.44269504089
    block_m = hl.register_block_size(chunk_size)
    block_n = hl.register_block_size(headdim)
    block_k = hl.register_block_size(chunk_size)

    HAS_D = D is not None
    D_HAS_HDIM = D.dim() == 2 if D is not None else False
    HAS_Z = z is not None
    HAS_SEQ_IDX = seq_idx is not None

    for tile_h, tile_m, tile_n, tile_b, tile_c in hl.tile(
        [nheads, chunk_size, headdim, batch, nchunks],
        block_size=[1, block_m, block_n, 1, 1],
    ):
        acc = hl.zeros([tile_m, tile_n], dtype=acc_dtype)

        dA_cs_m_local = (
            dA_cumsum[
                tile_b.begin,
                tile_h.begin,
                tile_c.begin,
                tile_m,
            ].to(torch.float32)
            * p
        )
        scale_m_local = torch.exp2(dA_cs_m_local)
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
            scale_m_local_mask = seq_idx_m_local == seq_idx_prev_local
            scale_m_local = torch.where(
                scale_m_local_mask, scale_m_local, torch.zeros_like(scale_m_local)
            )

        c_local = C[
            tile_b.begin,
            tile_m.index + tile_c.begin * chunk_size,
            tile_h.begin // (nheads // ngroups),
            :,
        ]
        prev_states_local = states[
            tile_b.begin,
            tile_c.begin,
            tile_h.begin,
            tile_n,
            :,
        ].to(C.dtype)
        acc = hl.dot(c_local, prev_states_local.T, acc=acc)
        acc *= scale_m_local[:, None]

        for tile_k in hl.tile((tile_m.id + 1) * block_m, block_size=block_k):
            cb_local = cb[
                tile_b.begin,
                tile_c.begin,
                tile_h.begin // (nheads // ngroups),
                tile_m,
                tile_k,
            ].to(torch.float32)
            dA_cs_k_local = (
                dA_cumsum[
                    tile_b.begin,
                    tile_h.begin,
                    tile_c.begin,
                    tile_k,
                ].to(torch.float32)
                * p
            )
            cb_local *= torch.exp2(
                (dA_cs_m_local[:, None] - dA_cs_k_local[None, :]).clamp(max=0.0)
            )
            dt_local = dt[
                tile_b.begin,
                tile_h.begin,
                tile_c.begin,
                tile_k,
            ].to(torch.float32)
            cb_local = cb_local * dt_local[None, :]
            cb_local_mask = (tile_m.index + 0)[:, None] >= (tile_k.index + 0)[None, :]
            cb_local = torch.where(cb_local_mask, cb_local, torch.zeros_like(cb_local))
            cb_local = cb_local.to(x.dtype)
            x_local = x[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_k.index,
                tile_h.begin,
                tile_n,
            ]
            acc = hl.dot(cb_local, x_local, acc=acc)

        if HAS_D:
            if D_HAS_HDIM:
                d_local = D[tile_h.begin, tile_n].to(torch.float32)
            else:
                d_local = D[tile_h.begin].to(torch.float32)
            x_res_local = x[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_m.index,
                tile_h.begin,
                tile_n,
            ].to(torch.float32)
            acc += x_res_local * d_local

        if HAS_Z:
            out_x[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_m.index,
                tile_h.begin,
                tile_n,
            ] = acc
            z_local = z[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_m.index,
                tile_h.begin,
                tile_n,
            ].to(torch.float32)
            acc *= z_local * torch.sigmoid(z_local)

        out[
            tile_b.begin,
            tile_c.begin * chunk_size + tile_m.index,
            tile_h.begin,
            tile_n,
        ] = acc

    return out, out_x


def _chunk_scan_fwd(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    states: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    return _chunk_scan_fwd_kernel(
        cb, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx
    )
