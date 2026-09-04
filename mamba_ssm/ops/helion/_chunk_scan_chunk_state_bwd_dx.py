from __future__ import annotations

import math
from typing import Optional

import torch
import triton

import helion
import helion.language as hl

from mamba_ssm.utils.determinism import (
    alloc_tile_workspace,
    finalize_tile_workspace,
)

from .utils import json_cached_autotune


def _chunk_scan_chunk_state_bwd_dx_ref(
    x, dt, dA_cumsum, B, CB, dout, dstates, D=None, seq_idx=None, dx=None
):
    # Autotune baseline for ``_chunk_scan_chunk_state_bwd_dx_kernel``. It drives the
    # low-level Triton kernel directly (rather than ssd_combined's wrapper) so that
    # its input/output exactly matches the Helion kernel: it returns ``dD`` at rank 2
    # ``(nheads, dD_hdim)`` -- (nheads, headdim) for 2-D D and (nheads, 1) for 1-D D.
    # Unlike ssd_combined's wrapper it does NOT squeeze the trailing dim for 1-D D;
    # the Helion kernel keeps dD at rank 2 and defers that squeeze to its host
    # wrapper, so the baseline must match that shape. Deterministic reduction is
    # forced off so that 1-D D keeps dD_hdim == 1 and the reduction below yields
    # (nheads, 1) directly, with no separate squeeze step.
    #
    # The ssd_combined import stays lazy to avoid a circular import: ssd_combined
    # imports this module at top level, so importing it at module load time would
    # fail with a partially-initialized-module error. It is only needed when
    # autotuning runs, by which point ssd_combined is fully initialized.
    from mamba_ssm.ops.triton.ssd_combined import (
        _CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N,
        TRITON_22,
        _chunk_scan_chunk_state_bwd_dx_kernel,
    )

    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    deterministic = False
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        pid_m_tiles = triton.cdiv(chunk_size, BLOCK_SIZE_min)
        pid_n_tiles = math.ceil(headdim / _CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N)
        if D.dim() == 2:
            dD_hdim = headdim
        elif deterministic:
            dD_hdim = pid_n_tiles
        else:
            dD_hdim = 1
        dD = torch.zeros(
            pid_m_tiles,
            batch,
            nchunks,
            nheads,
            dD_hdim,
            device=D.device,
            dtype=torch.float32,
        )
        dD_strides = (
            dD.stride(0),
            dD.stride(1),
            dD.stride(2),
            dD.stride(3),
            dD.stride(4),
        )
    else:
        dD = None
        dD_strides = (0, 0, 0, 0, 0)
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape
    tile_count = math.ceil(headdim / _CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N)
    ddt, stride_ddt_tile = alloc_tile_workspace(
        (batch, nheads, nchunks, chunk_size),
        tile_count,
        torch.float32,
        dout.device,
        deterministic,
        zero_init=True,
    )
    grid_dx = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x,
            CB,
            dout,
            dt,
            dA_cumsum,
            seq_idx,
            D,
            B,
            dstates,
            dx,
            ddt,
            dD,
            chunk_size,
            headdim,
            dstate,
            batch,
            seqlen,
            nheads // ngroups,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            CB.stride(0),
            CB.stride(1),
            CB.stride(2),
            CB.stride(-1),
            CB.stride(-2),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            *(
                (seq_idx.stride(0), seq_idx.stride(1))
                if seq_idx is not None
                else (0, 0)
            ),
            D.stride(0) if D is not None else 0,
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            dstates.stride(0),
            dstates.stride(1),
            dstates.stride(2),
            dstates.stride(3),
            dstates.stride(4),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            dx.stride(3),
            ddt.stride(0),
            ddt.stride(2),
            ddt.stride(1),
            ddt.stride(3),
            stride_ddt_tile,
            dD_strides[1],
            dD_strides[2],
            dD_strides[3],
            dD_strides[0],
            dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            IS_TRITON_22=TRITON_22,
            DETERMINISTIC_REDUCTION=deterministic,
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs[
            "BLOCK_SIZE_M"
        ]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2))
    ddt = finalize_tile_workspace(ddt, deterministic)
    return dx, ddt, dD


@json_cached_autotune
@helion.kernel(
    autotune_accuracy_check=False,
    autotune_baseline_fn=_chunk_scan_chunk_state_bwd_dx_ref,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _chunk_scan_chunk_state_bwd_dx_kernel(
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    B: torch.Tensor,
    CB: torch.Tensor,
    dout: torch.Tensor,
    dstates: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,  # required in practice; see host wrapper
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert seqlen % chunk_size == 0
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    BLOCK_M_MIN = 32
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        pid_m_tiles = helion.cdiv(chunk_size, BLOCK_M_MIN)
        if D.dim() == 2:
            dD_hdim = headdim
        else:
            dD_hdim = 1
        dD = torch.zeros(
            pid_m_tiles,
            batch,
            nchunks,
            nheads,
            dD_hdim,
            device=D.device,
            dtype=torch.float32,
        )
    else:
        dD = None
    # dx must already be a tensor: allocating it here would make Helion's type
    # propagation merge LiteralType(None) with TensorType and fail at bind() time.
    assert dx is not None, "dx must be preallocated by the caller"
    assert dx.shape == x.shape
    ddt = torch.zeros(
        (batch, nheads, nchunks, chunk_size),
        device=dt.device,
        dtype=torch.float32,
    )

    accum_dtype = torch.float32
    block_m = hl.register_block_size(BLOCK_M_MIN, chunk_size)
    block_n = hl.register_block_size(headdim)
    block_k = hl.register_block_size(chunk_size)

    CB = torch.transpose(CB, 3, 4)
    dstates = torch.transpose(dstates, 3, 4)

    HAS_D = D is not None
    if D is not None:
        D_HAS_HDIM = D.dim() == 2
    else:
        D_HAS_HDIM = True
    HAS_SEQ_IDX = seq_idx is not None

    for tile_c, tile_b, tile_h, tile_m, tile_n in hl.tile(
        [nchunks, batch, nheads, chunk_size, headdim],
        block_size=[1, 1, 1, block_m, block_n],
    ):
        chunk_size_limit = min(chunk_size, seqlen - tile_c.id * chunk_size)

        # [block_m, block_n]
        acc = hl.zeros([block_m, block_n], dtype=accum_dtype)

        # [block_m]
        dA_cs_m_local = dA_cumsum[
            tile_b.begin,
            tile_h.begin,
            tile_c.begin,
            tile_m,
        ].to(torch.float32)
        # [1]
        dA_cs_last_local = dA_cumsum[
            tile_b.begin,
            tile_h.begin,
            tile_c.begin,
            chunk_size - 1,
        ].to(torch.float32)
        scale = torch.exp((dA_cs_last_local - dA_cs_m_local).clamp(max=0.0))
        if HAS_SEQ_IDX:
            seq_idx_m_local = seq_idx[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_m.index,
            ]
            seq_idx_last_local = seq_idx[
                tile_b.begin,
                tile_c.begin * chunk_size + chunk_size - 1,
            ]
            scale_mask = seq_idx_m_local == seq_idx_last_local
            scale = torch.where(scale_mask, scale, torch.zeros_like(scale))

        # [block_m, dstate]
        b_local = B[
            tile_b.begin,
            tile_c.begin * chunk_size + tile_m.index,
            tile_h.begin // (nheads // ngroups),
            :,
        ]
        # [dstate, block_n]
        dstates_local = dstates[
            tile_b.begin,
            tile_c.begin,
            tile_h.begin,
            :,
            tile_n,
        ].to(B.dtype)
        acc = hl.dot(b_local, dstates_local, acc=acc)
        acc *= scale[:, None]

        K_MIN = tile_m.id * block_m
        K_MAX = chunk_size_limit
        for tile_k in hl.tile(K_MIN, K_MAX, block_size=block_k):
            # [block_m, block_k]
            cb_local = CB[
                tile_b.begin,
                tile_c.begin,
                tile_h.begin // (nheads // ngroups),
                tile_m,
                tile_k,
            ]
            # [block_k, block_n]
            dout_local = dout[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_k.index,
                tile_h.begin,
                tile_n,
            ]
            # [block_k]
            dA_cs_k_local = dA_cumsum[
                tile_b.begin,
                tile_h.begin,
                tile_c.begin,
                tile_k,
            ].to(torch.float32)
            cb_local *= torch.exp(
                (dA_cs_k_local[None, :] - dA_cs_m_local[:, None]).clamp(max=0.0)
            )
            cb_local_mask = (tile_k.index + 0)[None, :] >= (tile_m.index + 0)[:, None]
            cb_local = torch.where(cb_local_mask, cb_local, torch.zeros_like(cb_local))
            cb_local = cb_local.to(dout.dtype)
            acc = hl.dot(cb_local, dout_local, acc=acc)

        # [block_m]
        dt_m_local = dt[
            tile_b.begin,
            tile_h.begin,
            tile_c.begin,
            tile_m,
        ].to(torch.float32)
        dx_local = acc * dt_m_local[:, None]

        if HAS_D:
            # [block_m, block_n]
            dout_res_local = dout[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_m.index,
                tile_h.begin,
                tile_n,
            ].to(torch.float32)
            if D_HAS_HDIM:
                d_local = D[tile_h.begin, tile_n].to(torch.float32)
            else:
                d_local = D[tile_h.begin].to(torch.float32)
            dx_local += dout_res_local * d_local
        dx[
            tile_b.begin,
            tile_c.begin * chunk_size + tile_m.index,
            tile_h.begin,
            tile_n,
        ] = dx_local

        # [block_m, block_n]
        x_local = x[
            tile_b.begin,
            tile_c.begin * chunk_size + tile_m.index,
            tile_h.begin,
            tile_n,
        ].to(torch.float32)
        if HAS_D:
            if D_HAS_HDIM:
                dd_local = torch.sum(dout_res_local * x_local, dim=0)
                dD[
                    tile_m.id,
                    tile_b.begin,
                    tile_c.begin,
                    tile_h.begin,
                    tile_n,
                ] = dd_local
            else:
                dd_local = torch.sum(dout_res_local * x_local, dim=1)
                dd_local = torch.sum(dd_local, dim=0)
                hl.atomic_add(
                    dD,
                    [
                        tile_m.id,
                        tile_b.begin,
                        tile_c.begin,
                        tile_h.begin,
                        0,
                    ],
                    dd_local,
                )

        # [block_m]
        ddt_local = torch.sum(acc * x_local, dim=1)
        hl.atomic_add(
            ddt,
            [
                tile_b.begin,
                tile_h.begin,
                tile_c.begin,
                tile_m,
            ],
            ddt_local,
        )

    if D is not None:
        n_valid_blocks = (chunk_size + block_m - 1) // block_m
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2))

    return dx, ddt, dD


def _chunk_scan_chunk_state_bwd_dx(
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    B: torch.Tensor,
    CB: torch.Tensor,
    dout: torch.Tensor,
    dstates: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape
    dx, ddt, dD = _chunk_scan_chunk_state_bwd_dx_kernel(
        x, dt, dA_cumsum, B, CB, dout, dstates, D, seq_idx, dx
    )
    if D is not None:
        if D.dim() == 1:
            dD = dD.sum(dim=-1)
        dD = dD.to(dtype=D.dtype)
    return dx, ddt, dD
