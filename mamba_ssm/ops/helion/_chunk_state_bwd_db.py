from __future__ import annotations

import math
from typing import Optional

import torch

import triton
import triton.language as tl

import helion
import helion.language as hl
from helion.autotuner import IntegerFragment

from ..triton.ssd_chunk_state import (
    _chunk_state_bwd_db as _chunk_state_bwd_db_ref,
)

from .utils import json_cached_autotune

from mamba_ssm.utils.determinism import (
    alloc_tile_workspace,
    finalize_tile_workspace,
)


@triton.jit
def _masked_atomic_add(out_ptr, offs, val, mask):
    tl.atomic_add(out_ptr + offs, val, mask=mask)


@json_cached_autotune
@helion.kernel(
    autotune_accuracy_check=False,
    autotune_baseline_fn=_chunk_state_bwd_db_ref,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _chunk_state_bwd_db_kernel(
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    dstates: torch.Tensor,
    seq_idx: Optional[torch.Tensor] = None,
    B: Optional[torch.Tensor] = None,
    ngroups: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = dstates.shape[-1]
    assert seqlen % chunk_size == 0
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

    block_n = hl.register_block_size(dstate)
    # Fixed, not autotuned: it selects between a per-tile ddA_cumsum workspace and the
    # masked-atomic path below, so it also decides ddA_cumsum's shape.
    # TODO: make it a tunable and let autotuning pick.
    DETERMINISTIC_REDUCTION = True
    DETERMINISTIC_REDUCTION = hl.specialize(DETERMINISTIC_REDUCTION)
    if B is not None:
        assert B.shape == (batch, seqlen, ngroups, dstate)
        tile_count = math.ceil(dstate / block_n)
        ddA_cumsum, _ = alloc_tile_workspace(
            (batch, nheads, nchunks, chunk_size),
            tile_count,
            torch.float32,
            x.device,
            DETERMINISTIC_REDUCTION,
            zero_init=True,
        )
    else:
        ddA_cumsum = None

    ngroups = hl.specialize(ngroups)
    nheads_ngroups_ratio = nheads // ngroups
    nheads_per_program = hl.register_tunable(
        "nheads_per_program",
        IntegerFragment(1, nheads_ngroups_ratio),
    )
    nsplits = helion.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dB = torch.empty(
        batch,
        seqlen,
        nsplits,
        ngroups,
        dstate,
        device=x.device,
        dtype=torch.float32,
    )

    HAS_DDA_CS = ddA_cumsum is not None
    HAS_SEQ_IDX = seq_idx is not None
    p = 1.44269504089  # 1 / ln(2)
    block_m = hl.register_block_size(chunk_size)

    for tile_c, tile_b, tile_s, tile_g, tile_m, tile_n in hl.tile(
        [nchunks, batch, nsplits, ngroups, chunk_size, dstate],
        block_size=[1, 1, 1, 1, block_m, block_n],
    ):
        acc = hl.zeros((block_m, block_n), dtype=torch.float32)
        if HAS_DDA_CS:
            b_local = B[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_m.index,
                tile_g.begin,
                tile_n,
            ].to(torch.float32)
        if HAS_SEQ_IDX:
            seq_idx_m_local = seq_idx[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_m.index,
            ]
            seq_idx_last_local = seq_idx[
                tile_b.begin,
                tile_c.begin * chunk_size + chunk_size - 1,
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
            x_local = x[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_m.index,
                head_idx,
                :,
            ]
            dstates_local = dstates[
                tile_b.begin,
                tile_c.begin,
                head_idx,
                :,
                tile_n,
            ].to(x.dtype)
            db = hl.dot(x_local, dstates_local)
            dA_cs_last_local = (
                dA_cumsum[
                    tile_b.begin,
                    head_idx,
                    tile_c.begin,
                    chunk_size - 1,
                ].to(torch.float32)
                * p
            )
            dA_cs_m_local = (
                dA_cumsum[
                    tile_b.begin,
                    head_idx,
                    tile_c.begin,
                    tile_m,
                ].to(torch.float32)
                * p
            )
            dt_m_local = dt[
                tile_b.begin,
                head_idx,
                tile_c.begin,
                tile_m,
            ].to(torch.float32)
            scale = torch.exp2((dA_cs_last_local - dA_cs_m_local).clamp(max=0.0))
            if HAS_SEQ_IDX:
                scale = torch.where(
                    seq_idx_m_local == seq_idx_last_local,
                    scale,
                    torch.zeros_like(scale),
                )
            db *= (scale * dt_m_local)[:, None]
            if HAS_DDA_CS:
                ddA_cs_local = torch.sum(db * b_local, dim=1)
                if DETERMINISTIC_REDUCTION:
                    hl.store(
                        ddA_cumsum,
                        (
                            tile_b.begin,
                            head_idx,
                            tile_c.begin,
                            tile_m.index + 1,
                            tile_n.id,
                        ),
                        ddA_cs_local,
                        extra_mask=tile_m.index < (chunk_size - 1),
                    )
                else:
                    # hl.atomic_add takes no extra_mask, so the masked atomic add goes
                    # through a hand-written Triton kernel via hl.triton_kernel.
                    ddA_off = (
                        tile_b.begin * ddA_cumsum.stride(0)
                        + head_idx * ddA_cumsum.stride(1)
                        + tile_c.begin * ddA_cumsum.stride(2)
                        + (tile_m.index + 1) * ddA_cumsum.stride(3)
                    )
                    mask = tile_m.index < (chunk_size - 1)
                    hl.triton_kernel(
                        _masked_atomic_add,
                        args={
                            "out_ptr": ddA_cumsum,
                            "offs": ddA_off,
                            "val": ddA_cs_local,
                            "mask": mask,
                        },
                        output_like=None,
                    )
            acc += db

        dB[
            tile_b.begin,
            tile_c.begin * chunk_size + tile_m.index,
            tile_s.begin,
            tile_g.begin,
            tile_n,
        ] = acc

    dB = dB.sum(2)
    if ddA_cumsum is not None:
        ddA_cumsum = finalize_tile_workspace(ddA_cumsum, DETERMINISTIC_REDUCTION)
        torch.cumsum(ddA_cumsum, dim=-1, out=ddA_cumsum)
    return dB if B is None else (dB, ddA_cumsum)


def _chunk_state_bwd_db(
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    dstates: torch.Tensor,
    seq_idx: Optional[torch.Tensor] = None,
    B: Optional[torch.Tensor] = None,
    ngroups: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _chunk_state_bwd_db_kernel(
        x,
        dt,
        dA_cumsum,
        dstates,
        seq_idx,
        B,
        ngroups,
    )
