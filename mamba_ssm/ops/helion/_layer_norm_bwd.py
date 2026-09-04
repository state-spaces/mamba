from __future__ import annotations

import math
from typing import Optional

import torch

import helion
import helion.language as hl
from helion.autotuner import IntegerFragment

from .utils import json_cached_autotune


@json_cached_autotune
@helion.kernel(
    autotune_accuracy_check=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _layer_norm_bwd_kernel(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_N: hl.constexpr,
    group_size: int,
    z: Optional[torch.Tensor] = None,
    norm_before_gate: hl.constexpr = True,
    is_rms_norm: hl.constexpr = False,
    recompute_output: hl.constexpr = False,
    dz: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], ...]:
    M, N = x.shape
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    dx = torch.empty_like(x)

    nrow_groups = hl.register_tunable("nrow_groups", IntegerFragment(1, M))
    _dw = torch.empty(
        (nrow_groups, N),
        dtype=torch.float32,
        device=weight.device,
    )
    if bias is not None:
        _db = torch.empty(
            (nrow_groups, N),
            dtype=torch.float32,
            device=bias.device,
        )
    else:
        _db = None
    rows_per_program = math.ceil(M / nrow_groups)

    NORM_BEFORE_GATE = norm_before_gate
    IS_RMS_NORM = is_rms_norm
    HAS_BIAS = bias is not None
    HAS_Z = z is not None
    RECOMPUTE_OUTPUT = out is not None

    for tile_rb, tile_g in hl.tile(
        [nrow_groups, ngroups],
        block_size=[1, 1],
    ):
        cols = hl.arange(0, BLOCK_N)
        mask = cols < group_size
        w_local = hl.load(
            weight,
            [tile_g.begin * group_size + cols],
            extra_mask=mask,
        ).to(torch.float32)
        if (RECOMPUTE_OUTPUT or HAS_Z) and HAS_BIAS:
            b_local = hl.load(
                bias,
                [tile_g.begin * group_size + cols],
                extra_mask=mask,
            ).to(torch.float32)
        dw_local = hl.zeros([BLOCK_N], dtype=torch.float32)
        if HAS_BIAS:
            db_local = hl.zeros([BLOCK_N], dtype=torch.float32)
        row_start = tile_rb.begin * rows_per_program
        row_end = min((tile_rb.begin + 1) * rows_per_program, M)
        for tile_r in hl.tile(row_start, row_end, block_size=1):
            x_local = hl.load(
                x,
                [
                    tile_r.begin,
                    tile_g.begin * group_size + cols,
                ],
                extra_mask=mask,
            )
            x_local = torch.where(mask, x_local, torch.zeros_like(x_local)).to(
                torch.float32
            )
            dy_local = hl.load(
                dy,
                [
                    tile_r.begin,
                    tile_g.begin * group_size + cols,
                ],
                extra_mask=mask,
            )
            dy_local = torch.where(mask, dy_local, torch.zeros_like(dy_local)).to(
                torch.float32
            )
            if not IS_RMS_NORM:
                mean_local = mean[tile_g.begin * M + tile_r.begin]
            if HAS_Z and not NORM_BEFORE_GATE:
                z_local = hl.load(
                    z,
                    [
                        tile_r.begin,
                        tile_g.begin * group_size + cols,
                    ],
                    extra_mask=mask,
                )
                z_local = torch.where(mask, z_local, torch.zeros_like(z_local)).to(
                    torch.float32
                )
                x_og = x_local
                x_local = x_og * z_local * torch.sigmoid(z_local)
            rstd_local = rstd[tile_g.begin * M + tile_r.begin]
            if not IS_RMS_NORM:
                xhat = (x_local - mean_local) * rstd_local
            else:
                xhat = x_local * rstd_local
            xhat = torch.where(mask, xhat, torch.zeros_like(xhat))
            if HAS_Z and NORM_BEFORE_GATE:
                z_local = hl.load(
                    z,
                    [
                        tile_r.begin,
                        tile_g.begin * group_size + cols,
                    ],
                    extra_mask=mask,
                )
                z_local = torch.where(mask, z_local, torch.zeros_like(z_local)).to(
                    torch.float32
                )
                z_sigmoid = torch.sigmoid(z_local)
                if HAS_BIAS:
                    y = xhat * w_local + b_local
                else:
                    y = xhat * w_local
                if RECOMPUTE_OUTPUT:
                    hl.store(
                        out,
                        [
                            tile_r.begin,
                            tile_g.begin * group_size + cols,
                        ],
                        y * z_local * z_sigmoid,
                        extra_mask=mask,
                    )
                dz_local = dy_local * y * z_sigmoid * (1 + z_local * (1 - z_sigmoid))
                hl.store(
                    dz,
                    [
                        tile_r.begin,
                        tile_g.begin * group_size + cols,
                    ],
                    dz_local,
                    extra_mask=mask,
                )
                dy_local *= z_local * z_sigmoid
            else:
                if RECOMPUTE_OUTPUT:
                    if HAS_BIAS:
                        y = xhat * w_local + b_local
                    else:
                        y = xhat * w_local
                    hl.store(
                        out,
                        [
                            tile_r.begin,
                            tile_g.begin * group_size + cols,
                        ],
                        y,
                        extra_mask=mask,
                    )
            wdy = w_local * dy_local
            c1 = torch.sum(xhat * wdy, dim=0) / group_size
            if not IS_RMS_NORM:
                c2 = torch.sum(wdy, dim=0) / group_size
                dx_local = (wdy - (xhat * c1 + c2)) * rstd_local
            else:
                dx_local = (wdy - xhat * c1) * rstd_local
            dw_local += dy_local * xhat
            if HAS_BIAS:
                db_local += dy_local
            if HAS_Z and not NORM_BEFORE_GATE:
                z_sigmoid = torch.sigmoid(z_local)
                dz_local = dx_local * x_og * z_sigmoid * (1 + z_local * (1 - z_sigmoid))
                hl.store(
                    dz,
                    [
                        tile_r.begin,
                        tile_g.begin * group_size + cols,
                    ],
                    dz_local,
                    extra_mask=mask,
                )
                dx_local *= z_local * z_sigmoid
            hl.store(
                dx,
                [
                    tile_r.begin,
                    tile_g.begin * group_size + cols,
                ],
                dx_local,
                extra_mask=mask,
            )
        hl.store(
            _dw,
            [tile_rb.begin, tile_g.begin * group_size + cols],
            dw_local,
            extra_mask=mask,
        )
        if HAS_BIAS:
            hl.store(
                _db,
                [tile_rb.begin, tile_g.begin * group_size + cols],
                db_local,
                extra_mask=mask,
            )

    dw = _dw.sum(0).to(weight.dtype)
    if bias is not None:
        db = _db.sum(0).to(bias.dtype)
    else:
        db = None
    return (dx, dw, db, dz) if not recompute_output else (dx, dw, db, dz, out)


def _layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    recompute_output: bool = False,
    dz: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], ...]:
    _, N = x.shape
    if group_size is None:
        group_size = N
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
        assert dz.stride(-1) == 1
    else:
        dz = torch.empty_like(z) if z is not None else None
    if recompute_output:
        if out is None:
            out = torch.empty_like(x)
        assert out.shape == x.shape
    # One group's features must fit in 64KB, as in the Triton kernel: BLOCK_N covers a
    # whole group, so the kernel keeps a row of it live at a time.
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, helion.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    return _layer_norm_bwd_kernel(
        dy,
        x,
        weight,
        bias,
        eps,
        mean,
        rstd,
        BLOCK_N,
        group_size,
        z,
        norm_before_gate,
        is_rms_norm,
        recompute_output,
        dz,
        out,
    )
