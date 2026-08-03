from __future__ import annotations

from typing import Optional

import torch

import helion
import helion.language as hl

from ..triton.ssd_state_passing import (
    _state_passing_fwd as _state_passing_fwd_ref,
)

from .utils import json_cached_autotune


@json_cached_autotune
@helion.kernel(
    autotune_accuracy_check=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _state_passing_fwd_kernel_v1(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)

    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)

    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)

    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty(
        (batch, nchunks, nheads, dim),
        device=states.device,
        dtype=out_dtype,
    )
    final_states = torch.empty(
        (batch, nheads, dim),
        device=states.device,
        dtype=torch.float32,
    )

    HAS_INITSTATES = initial_states is not None
    HAS_SEQ_IDX = seq_idx is not None
    if chunk_size is not None:
        chunk_size = hl.specialize(chunk_size)

    block = hl.register_block_size(dim)

    for tile_b, tile_h, tile_m in hl.tile(
        [batch, nheads, dim],
        block_size=[1, 1, block],
    ):
        if not HAS_INITSTATES:
            states_local = hl.zeros((block,), dtype=torch.float32)
        else:
            states_local = initial_states[tile_b.begin, tile_h.begin, tile_m].to(
                torch.float32
            )

        out[tile_b.begin, 0, tile_h.begin, tile_m] = states_local

        if HAS_SEQ_IDX:
            seq_idx_local = hl.zeros([], dtype=seq_idx.dtype)

        # One chunk per step: the recurrence is sequential in tile_c. The v2 kernel
        # below does the same work with block_size > 1 via an associative scan.
        for tile_c in hl.tile(nchunks, block_size=1):
            new_states_local = states[
                tile_b.begin, tile_c.begin, tile_h.begin, tile_m
            ].to(torch.float32)
            dA_cs_local = dA_chunk_cumsum[tile_b.begin, tile_h.begin, tile_c.begin].to(
                torch.float32
            )
            scale = torch.exp(dA_cs_local)

            if HAS_SEQ_IDX:
                seq_idx_new_local = seq_idx[
                    tile_b.begin, (tile_c.begin + 1) * chunk_size - 1
                ]
                scale_mask = seq_idx_new_local == seq_idx_local
                scale = torch.where(scale_mask, scale, torch.zeros_like(scale))
                seq_idx_local = seq_idx_new_local

            states_local = scale * states_local + new_states_local

            if tile_c.begin < nchunks - 1:
                out[tile_b.begin, tile_c.begin + 1, tile_h.begin, tile_m] = states_local
            else:
                final_states[tile_b.begin, tile_h.begin, tile_m] = states_local

    return out, final_states


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
    autotune_baseline_fn=_state_passing_fwd_ref,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _state_passing_fwd_kernel_v2(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)

    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)

    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)

    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty(
        (batch, nchunks, nheads, dim),
        device=states.device,
        dtype=out_dtype,
    )
    final_states = torch.empty(
        (batch, nheads, dim),
        device=states.device,
        dtype=torch.float32,
    )

    HAS_INITSTATES = initial_states is not None
    HAS_SEQ_IDX = seq_idx is not None
    if chunk_size is not None:
        chunk_size = hl.specialize(chunk_size)

    block_m = hl.register_block_size(dim)
    block_c = hl.register_block_size(nchunks)

    for tile_b, tile_h, tile_m in hl.tile(
        [batch, nheads, dim],
        block_size=[1, 1, block_m],
    ):
        if not HAS_INITSTATES:
            states_local = hl.zeros((block_m,), dtype=torch.float32)
        else:
            states_local = initial_states[tile_b.begin, tile_h.begin, tile_m].to(
                torch.float32
            )

        out[tile_b.begin, 0, tile_h.begin, tile_m] = states_local

        for tile_c in hl.tile(nchunks, block_size=block_c):
            new_states_local = states[tile_b.begin, tile_c, tile_h.begin, tile_m].to(
                torch.float32
            )
            dA_cs_local = dA_chunk_cumsum[tile_b.begin, tile_h.begin, tile_c].to(
                torch.float32
            )
            scale = torch.exp(dA_cs_local)

            if HAS_SEQ_IDX:
                seq_idx_new_local = seq_idx[
                    tile_b.begin, (tile_c.index + 1) * chunk_size - 1
                ]
                seq_idx_local = seq_idx[tile_b.begin, tile_c.index * chunk_size - 1]
                scale_mask = seq_idx_new_local == seq_idx_local
                scale = torch.where(scale_mask, scale, torch.zeros_like(scale))

            # scale is broadcast to (block_c, block_m) explicitly: both elements of
            # the scan tuple must have the same shape.
            A_cum, B_cum = hl.associative_scan(
                _affine_combine,
                (scale[:, None].expand(-1, block_m), new_states_local),
                dim=0,
            )

            tmp_states_local = A_cum * states_local[None, :] + B_cum
            # Carry the scan's last row into the next tile. A kernel tensor cannot be
            # indexed by an integer, so the row is extracted with a one-hot mask + sum.
            last_mask = hl.arange(0, block_c) == (block_c - 1)
            states_local = torch.where(
                last_mask[:, None], tmp_states_local, torch.zeros_like(tmp_states_local)
            ).sum(dim=0)

            # Shifted by one: out[c + 1] is the state entering chunk c + 1. The mask
            # drops the last chunk's, which becomes final_states instead.
            hl.store(
                out,
                (tile_b.begin, tile_c.index + 1, tile_h.begin, tile_m),
                tmp_states_local,
                extra_mask=(tile_c.index + 1 < nchunks)[:, None],
            )

        final_states[tile_b.begin, tile_h.begin, tile_m] = states_local

    return out, final_states


def _state_passing_fwd_v1(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    return _state_passing_fwd_kernel_v1(
        states, dA_chunk_cumsum, initial_states, seq_idx, chunk_size, out_dtype
    )


def _state_passing_fwd_v2(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    return _state_passing_fwd_kernel_v2(
        states, dA_chunk_cumsum, initial_states, seq_idx, chunk_size, out_dtype
    )
