from __future__ import annotations

import math
from typing import Optional, Union

import torch

import helion
import helion.language as hl

from ..triton.ssd_state_passing import (
    _state_passing_bwd as _state_passing_bwd_ref,
)

from .utils import json_cached_autotune


def _affine_combine(left, right):
    # The recurrence s_i = a_i * s_{i-1} + b_i as composition of the affine maps
    # T_i(x) = a_i * x + b_i, which is associative and so scannable:
    # T_right o T_left: (a_l, b_l), (a_r, b_r) -> (a_r*a_l, a_r*b_l + b_r)
    a1, b1 = left
    a2, b2 = right
    return a1 * a2, a2 * b1 + b2


@json_cached_autotune
@helion.kernel(
    autotune_accuracy_check=False,
    autotune_baseline_fn=_state_passing_bwd_ref,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _state_passing_bwd_kernel(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    dout: torch.Tensor,
    dfinal_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    has_initial_states: hl.constexpr = False,
    dstates_dtype: Optional[torch.dtype] = None,
    states_dtype: Optional[torch.dtype] = None,
    chunk_size: int = 0,
) -> Union[
    tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
]:
    """
    states contains the initial_states at index 0. The final states are not included in states.
    """
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    assert dout.shape == (batch, nchunks, nheads, dim)
    if seq_idx is not None:
        assert chunk_size > 0
        chunk_size = hl.specialize(chunk_size)
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)

    dstates = torch.empty_like(
        dout,
        dtype=dstates_dtype if dstates_dtype is not None else dout.dtype,
    )
    if states_dtype is not None and states_dtype != states.dtype:
        states_converted = torch.empty_like(
            states,
            dtype=dstates_dtype if dstates_dtype is not None else dout.dtype,
        )
        assert states_converted.stride() == states.stride()
    else:
        states_converted = None

    if dfinal_states is not None:
        assert dfinal_states.shape == (batch, nheads, dim)

    if has_initial_states:
        dinitstates = torch.empty(
            (batch, nheads, dim),
            dtype=dstates_dtype if dstates_dtype is not None else dout.dtype,
            device=dout.device,
        )
    else:
        dinitstates = None

    block_m = hl.register_block_size(dim)
    n_blocks = (dim + block_m - 1) // block_m
    ddA_chunk_cumsum = torch.empty(
        batch,
        nheads,
        nchunks,
        n_blocks,
        dtype=torch.float32,
        device=dA_chunk_cumsum.device,
    )

    CONVERT_STATES = states_converted is not None
    HAS_DFINAL_STATES = dfinal_states is not None
    HAS_DINITSTATES = has_initial_states
    HAS_SEQ_IDX = seq_idx is not None

    block_c = hl.register_block_size(nchunks)

    for tile_b, tile_h, tile_m in hl.tile(
        [batch, nheads, dim],
        block_size=[1, 1, block_m],
    ):
        if HAS_DFINAL_STATES:
            dstates_local = dfinal_states[tile_b.begin, tile_h.begin, tile_m].to(
                torch.float32
            )
        else:
            dstates_local = hl.zeros((block_m,), dtype=torch.float32)
        # Chunk nchunks - 1 seeds the reverse scan: its dstates is dfinal_states (or
        # zero), and the loop below walks the remaining chunks backwards.
        dstates[
            tile_b.begin,
            nchunks - 1,
            tile_h.begin,
            tile_m,
        ] = dstates_local
        dA_cs_last_local = dA_chunk_cumsum[tile_b.begin, tile_h.begin, nchunks - 1].to(
            torch.float32
        )
        scale_last = torch.exp(dA_cs_last_local)
        if HAS_SEQ_IDX:
            seq_idx_last_local = seq_idx[tile_b.begin, seqlen - 1]
            seq_idx_new_last_local = seq_idx[tile_b.begin, seqlen - 1 - chunk_size]
            scale_last = torch.where(
                seq_idx_last_local == seq_idx_new_last_local,
                scale_last,
                torch.zeros_like(scale_last),
            )
        out_last_local = states[
            tile_b.begin,
            nchunks - 1,
            tile_h.begin,
            tile_m,
        ].to(torch.float32)
        if CONVERT_STATES:
            states_converted[
                tile_b.begin,
                nchunks - 1,
                tile_h.begin,
                tile_m,
            ] = out_last_local
        ddA_last_local = torch.sum(out_last_local * dstates_local) * scale_last
        ddA_chunk_cumsum[
            tile_b.begin,
            tile_h.begin,
            nchunks - 1,
            tile_m.id,
        ] = ddA_last_local

        for tile_c in hl.tile(1, nchunks, block_size=block_c):
            # tile_c.index runs 1..nchunks-1, so this reads chunk
            # (nchunks - 1) - (tile_c.index - 1): chunks nchunks - 1 down to 1.
            dout_prev_local = dout[
                tile_b.begin,
                (nchunks - 1) - (tile_c.index - 1),
                tile_h.begin,
                tile_m,
            ].to(torch.float32)
            dA_cs_prev_local = dA_chunk_cumsum[
                tile_b.begin,
                tile_h.begin,
                (nchunks - 1) - (tile_c.index - 1),
            ].to(torch.float32)
            scale_prev = torch.exp(dA_cs_prev_local)
            if HAS_SEQ_IDX:
                seq_idx_prev_local = seq_idx[
                    tile_b.begin,
                    (nchunks - (tile_c.index - 1)) * chunk_size - 1,
                ]
                seq_idx_new_prev_local = seq_idx[
                    tile_b.begin,
                    (nchunks - tile_c.index) * chunk_size - 1,
                ]
                scale_prev = torch.where(
                    seq_idx_prev_local == seq_idx_new_prev_local,
                    scale_prev,
                    torch.zeros_like(scale_prev),
                )
            A_cum, B_cum = hl.associative_scan(
                _affine_combine,
                (scale_prev[:, None].expand(-1, block_m), dout_prev_local),
                dim=0,
            )
            tmp_dstates_local = A_cum * dstates_local[None, :] + B_cum
            # On a partial last tile (nchunks - 1 not a multiple of block_c) lane
            # block_c - 1 is out of range and would carry garbage into the next tile;
            # tile_c.end - 1 is this tile's last valid index.
            last_mask = tile_c.index == (tile_c.end - 1)
            dstates_local = torch.where(
                last_mask[:, None],
                tmp_dstates_local,
                torch.zeros_like(tmp_dstates_local),
            ).sum(dim=0)
            # Chunk (nchunks - 1) - tile_c.index: chunks nchunks - 2 down to 0.
            dstates[
                tile_b.begin,
                (nchunks - 1) - tile_c.index,
                tile_h.begin,
                tile_m,
            ] = tmp_dstates_local
            # Chunk (nchunks - 1) - tile_c.index: chunks nchunks - 2 down to 0.
            dA_cs_local = dA_chunk_cumsum[
                tile_b.begin,
                tile_h.begin,
                (nchunks - 1) - tile_c.index,
            ].to(torch.float32)
            scale = torch.exp(dA_cs_local)
            if HAS_SEQ_IDX:
                seq_idx_local = seq_idx_new_prev_local
                seq_idx_new_local = hl.load(
                    seq_idx,
                    [
                        tile_b.begin,
                        (nchunks - (tile_c.index + 1)) * chunk_size - 1,
                    ],
                    extra_mask=tile_c.index + 1 < nchunks,
                )
                seq_idx_new_local = torch.where(
                    tile_c.index + 1 < nchunks,
                    seq_idx_new_local,
                    torch.zeros_like(seq_idx_new_local),
                )
                scale = torch.where(
                    seq_idx_new_local == seq_idx_local,
                    scale,
                    torch.zeros_like(scale),
                )
            out_local = states[
                tile_b.begin,
                (nchunks - 1) - tile_c.index,
                tile_h.begin,
                tile_m,
            ].to(torch.float32)
            if CONVERT_STATES:
                states_converted[
                    tile_b.begin,
                    (nchunks - 1) - tile_c.index,
                    tile_h.begin,
                    tile_m,
                ] = out_local
            ddA_local = torch.sum(out_local * tmp_dstates_local, dim=1) * scale
            ddA_chunk_cumsum[
                tile_b.begin,
                tile_h.begin,
                (nchunks - 1) - tile_c.index,
                tile_m.id,
            ] = ddA_local

        if not HAS_DINITSTATES:
            ddA_chunk_cumsum[
                tile_b.begin,
                tile_h.begin,
                0,
                tile_m.id,
            ] = hl.zeros((), dtype=torch.float32)
        else:
            # Triton applies the recurrence once more for chunk 0 after leaving the
            # chunk loop (ssd_state_passing.py:182-193):
            #     dinitial_states = scale_0 * dstates[0] + dout[0]
            # Without this step dstates[0] would leave as initial_states' gradient.
            # (chunk 0's ddA_chunk_cumsum is already produced by the loop's last
            # iteration.)
            dA_cs_first_local = dA_chunk_cumsum[tile_b.begin, tile_h.begin, 0].to(
                torch.float32
            )
            scale_first = torch.exp(dA_cs_first_local)
            if HAS_SEQ_IDX:
                # The seq_idx Triton tracks is seq_idx[chunk_size - 1] at this point,
                # compared against the value 0 it started the loop with.
                seq_idx_first_local = seq_idx[tile_b.begin, chunk_size - 1]
                scale_first = torch.where(
                    seq_idx_first_local == 0,
                    scale_first,
                    torch.zeros_like(scale_first),
                )
            dout_first_local = dout[tile_b.begin, 0, tile_h.begin, tile_m].to(
                torch.float32
            )
            dinitstates[tile_b.begin, tile_h.begin, tile_m] = (
                scale_first * dstates_local + dout_first_local
            )

    ddA_chunk_cumsum = ddA_chunk_cumsum.sum(dim=-1).to(dtype=dA_chunk_cumsum.dtype)
    if states_dtype is not None and states_dtype == states.dtype:
        states_converted = states
    return (
        (dstates, ddA_chunk_cumsum, dinitstates)
        if states_dtype is None
        else (dstates, ddA_chunk_cumsum, dinitstates, states_converted)
    )


def _state_passing_bwd(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    dout: torch.Tensor,
    dfinal_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    has_initial_states: bool = False,
    dstates_dtype: Optional[torch.dtype] = None,
    states_dtype: Optional[torch.dtype] = None,
    chunk_size: int = 0,
) -> Union[
    tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
]:
    return _state_passing_bwd_kernel(
        states,
        dA_chunk_cumsum,
        dout,
        dfinal_states=dfinal_states,
        seq_idx=seq_idx,
        has_initial_states=has_initial_states,
        dstates_dtype=dstates_dtype,
        states_dtype=states_dtype,
        chunk_size=chunk_size,
    )
