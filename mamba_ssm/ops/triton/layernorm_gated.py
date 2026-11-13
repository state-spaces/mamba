# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

import math

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange

# Single implementation of the device guard and the core-count heuristic; the
# non-gated LayerNorm/RMSNorm path (``layer_norm.py``) uses the same helpers.
from mamba_ssm.ops.triton.mamba3.npu_compat import (
    _device_ctx,
    _npu_aiv_block_limit,
    _sm_count,
)


# Ascend folds a 2D Triton grid onto a 1D ``coreDim`` that must fit
# ``UINT16_MAX = 65535``.  The forward grid is ``(M, ngroups)`` where ``M`` is
# the product of the leading dims and ``ngroups = N // group_size``, so the
# constrained quantity is ``M * ngroups`` (the number of normalisation groups).
# CUDA's ``grid.x`` limit is 2^31-1, so the CUDA launch is left exactly as
# before (one row per program); on every other device the row extent is capped
# and the kernel grid-strides over rows.
_MAX_COREDIM = 65535
_FWD_GRID_LIMIT = _MAX_COREDIM


def set_fwd_grid_limit(limit):
    """Test hook: lower the effective forward coreDim cap (``None`` = 65535).

    Production leaves this at ``_MAX_COREDIM``.  Lowering it exercises the
    capped-grid + inner-row-loop path on shapes small enough to run quickly,
    without touching the kernel code the real overflow uses.
    """
    global _FWD_GRID_LIMIT
    if limit is None:
        _FWD_GRID_LIMIT = _MAX_COREDIM
        return
    limit = int(limit)
    if not 1 <= limit <= _MAX_COREDIM:
        raise ValueError(f"fwd grid limit must be in [1, {_MAX_COREDIM}], got {limit}")
    _FWD_GRID_LIMIT = limit


def _fwd_row_programs(M, ngroups, limit=None):
    """Row extent for the forward launch: ``(row_programs, rows_per_program)``.

    The flattened coreDim is ``row_programs * ngroups``; it is kept at or below
    ``limit`` by capping the row extent at ``limit // ngroups`` and letting each
    program loop over rows with a stride of ``row_programs``:

        max_rows  = limit // ngroups
        rows_per  = 1                              if M <= max_rows
                    ceil(M / max_rows)              otherwise
        programs  = ceil(M / rows_per)  (== M when not compressed)

    A "tile" is one row, so there is no intra-tile tail mask: the loop bound
    ``M`` short-circuits programs whose next row is out of range.
    """
    limit = _FWD_GRID_LIMIT if limit is None else int(limit)
    M, ngroups = int(M), int(ngroups)
    if ngroups > limit:
        raise ValueError(
            f"ngroups={ngroups} alone exceeds the coreDim limit {limit}: the group "
            f"extent cannot be compressed by a row grid-stride")
    max_row_programs = limit // ngroups
    if M <= max_row_programs:
        return M, 1
    rows_per_program = -(-M // max_row_programs)  # ceil
    return -(-M // rows_per_program), rows_per_program


# Test-observable launch geometry.  Every forward launch records what it
# actually emitted so tests can prove the capped path ran instead of inferring
# it from a successful call:
#   * ``num_tiles``       logical group count ``M * ngroups``;
#   * ``num_row_programs`` row extent actually launched (``<= _MAX_COREDIM //
#                         ngroups`` whenever the device is not CUDA);
#   * ``rows_per_program`` rows each program visits (grid-stride length);
#   * ``grid``            the emitted ``(row_programs, ngroups)`` tuple;
#   * ``grid_stride``     True iff a program visits more than one row;
#   * ``launched``        False for the ``M == 0`` short-circuit.
_FWD_GRID_STATE = {
    "M": 0,
    "ngroups": 0,
    "num_tiles": 0,
    "num_row_programs": 0,
    "rows_per_program": 0,
    "grid": (0, 0),
    "grid_stride": False,
    "launched": False,
    "calls": 0,
    "grid_stride_calls": 0,
}


def reset_fwd_grid_state():
    """Reset the launch-geometry record (does not change the cap)."""
    _FWD_GRID_STATE.update(
        M=0, ngroups=0, num_tiles=0, num_row_programs=0, rows_per_program=0,
        grid=(0, 0), grid_stride=False, launched=False, calls=0,
        grid_stride_calls=0,
    )


def get_fwd_grid_state():
    """Snapshot of the last forward launch's geometry (test hook)."""
    return dict(_FWD_GRID_STATE)


def _record_fwd_grid(M, ngroups, row_programs, rows_per_program, launched):
    grid_stride = bool(launched and int(row_programs) < int(M))
    _FWD_GRID_STATE.update(
        M=int(M),
        ngroups=int(ngroups),
        num_tiles=int(M) * int(ngroups),
        num_row_programs=int(row_programs),
        rows_per_program=int(rows_per_program),
        grid=(int(row_programs), int(ngroups)),
        grid_stride=grid_stride,
        launched=bool(launched),
        calls=_FWD_GRID_STATE["calls"] + 1,
        grid_stride_calls=_FWD_GRID_STATE["grid_stride_calls"] + int(grid_stride),
    )


def rms_norm_ref(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    num_row_programs,  # rows launched (row grid-stride step)
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Row grid-stride: this program handles rows ``pid``, ``pid +
    # num_row_programs``, ... so the flattened coreDim
    # (``num_row_programs * ngroups``) can stay under UINT16_MAX while every row
    # is still visited exactly once.  When the grid is not compressed,
    # ``num_row_programs == M`` and the loop runs a single iteration, which is
    # the original one-row-per-program schedule.
    # ``num_row_programs`` is passed explicitly rather than read from
    # ``tl.num_programs(0)`` because Triton-Ascend may fold the logical grid
    # onto the physical AI cores, making that value unreliable.
    pid = tl.program_id(0)
    group = tl.program_id(1)
    X += group * N
    Y += group * N
    if HAS_Z:
        Z += group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    # weight/bias are shared by every row of this program's group
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    # Rows owned by this program: pid, pid + num_row_programs, ...  A unit-step
    # loop is used because some Triton backends do not accept a dynamic step.
    num_rows = (M - pid + num_row_programs - 1) // num_row_programs
    for i in range(0, num_rows):
        row = pid + i * num_row_programs
        # Compute mean and variance
        x = tl.load(X + row * stride_x_row + cols, mask=cols < N, other=0.).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z + row * stride_z_row + cols, mask=cols < N).to(tl.float32)
            x *= z * tl.sigmoid(z)
        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=0) / N
            tl.store(Mean + row, mean)
            xbar = tl.where(cols < N, x - mean, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        else:
            xbar = tl.where(cols < N, x, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd + row, rstd)
        # Normalize and apply linear transformation
        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z + row * stride_z_row + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        # Write output
        tl.store(Y + row * stride_y_row + cols, y, mask=mask)


def _layer_norm_fwd(x, weight, bias, eps, z=None, out=None, group_size=None, norm_before_gate=True, is_rms_norm=False):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    if M == 0:
        # Zero rows: the outputs (and mean/rstd) are already empty and there is
        # no valid grid, so short-circuit instead of launching (zero-tile rule).
        _record_fwd_grid(M, ngroups, 0, 0, launched=False)
        return out, mean, rstd
    if getattr(x.device, "type", None) == "cuda":
        # CUDA grid.x is 2^31-1: keep the historical one-row-per-program launch.
        row_programs, rows_per_program = M, 1
    else:
        row_programs, rows_per_program = _fwd_row_programs(M, ngroups)
    grid = (row_programs, ngroups)
    _record_fwd_grid(M, ngroups, row_programs, rows_per_program, launched=True)
    with _device_ctx(x.device):
        _layer_norm_fwd_1pass_kernel[grid](x, out, weight, bias, z, mean, rstd,
                                           x.stride(0), out.stride(0), z.stride(0) if z is not None else 0,
                                           M, group_size, eps, row_programs,
                                           BLOCK_N=BLOCK_N,
                                           NORM_BEFORE_GATE=norm_before_gate,
                                           IS_RMS_NORM=is_rms_norm,
                                           num_warps=num_warps)
    return out, mean, rstd



@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
    X,   # pointer to the input
    W,   # pointer to the weights
    B,   # pointer to the biases
    Z,   # pointer to the other branch
    Y,   # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DZ,  # pointer to the other branch
    Mean,   # pointer to the mean
    Rstd,   # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_z_row,
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dz_row,
    stride_dw_row,
    stride_db_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    group = tl.program_id(1)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row + group * N
    if HAS_Z:
        Z += row_start * stride_z_row + group * N
        DZ += row_start * stride_dz_row + group * N
    DY += row_start * stride_dy_row + group * N
    DX += row_start * stride_dx_row + group * N
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if (RECOMPUTE_OUTPUT or HAS_Z) and HAS_BIAS:
        B += group * N
        b = tl.load(B + cols, mask=mask, other=0.).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.).to(tl.float32)
            x_og = x
            x = x_og * z * tl.sigmoid(z)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.)
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.).to(tl.float32)
            z_sigmoid = tl.sigmoid(z)
            y = xhat * w + b if HAS_BIAS else xhat * w
            if RECOMPUTE_OUTPUT:
                tl.store(Y + cols, y * z * z_sigmoid, mask=mask)
            dz = dy * y * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dy *= z * z_sigmoid
        else:
            if RECOMPUTE_OUTPUT:
                y = xhat * w + b if HAS_BIAS else xhat * w
                tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        c1 = tl.sum(xhat * wdy, axis=0) / N
        if not IS_RMS_NORM:
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            dx = (wdy - xhat * c1) * rstd
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_Z and not NORM_BEFORE_GATE:
            z_sigmoid = tl.sigmoid(z)
            dz = dx * x_og * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dx *= z * z_sigmoid
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_Z:
            Z += stride_z_row
            DZ += stride_dz_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * stride_dw_row + group * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * stride_db_row + group * N + cols, db, mask=mask)


def _layer_norm_bwd(dy, x, weight, bias, eps, mean, rstd, z=None, group_size=None,
                    norm_before_gate=True, is_rms_norm=False, recompute_output=False, dz=None, out=None):
    M, N = x.shape
    if group_size is None:
        group_size = N
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

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    sm_count = _sm_count(x.device)
    # If group size is small (e.g., 64), we're only using 1 warp. So having just 108 programs
    # would limit the occupancy.
    nrow_groups = math.ceil(sm_count * math.ceil(4 / num_warps) / ngroups)
    _dw = torch.empty((nrow_groups, N), dtype=torch.float32, device=weight.device)
    _db = torch.empty((nrow_groups, N), dtype=torch.float32, device=bias.device) if bias is not None else None
    rows_per_program = math.ceil(M / nrow_groups)
    grid = (nrow_groups, ngroups)
    with _device_ctx(x.device):
        _layer_norm_bwd_kernel[grid](x, weight, bias, z, out if recompute_output else None,
                                     dy, dx, _dw, _db, dz, mean, rstd,
                                     x.stride(0),
                                     z.stride(0) if z is not None else 0,
                                     0 if not recompute_output else out.stride(0),
                                     dy.stride(0), dx.stride(0),
                                     dz.stride(0) if dz is not None else 0,
                                     _dw.stride(0),
                                     _db.stride(0) if _db is not None else 0,
                                     M, group_size, eps,
                                     rows_per_program,
                                     BLOCK_N=BLOCK_N,
                                     NORM_BEFORE_GATE=norm_before_gate,
                                     IS_RMS_NORM=is_rms_norm,
                                     num_warps=num_warps)
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    return (dx, dw, db, dz) if not recompute_output else (dx, dw, db, dz, out)


class LayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True,
                is_rms_norm=False):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """

        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = _layer_norm_fwd(x, weight, bias, eps, z=z, group_size=group_size, norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm)
        ctx.save_for_backward(x, weight, bias, mean, rstd, z)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm
        return y.reshape(x_shape_og)

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, mean, rstd, z = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        dx, dw, db, dz = _layer_norm_bwd(dy, x, weight, bias, ctx.eps, mean, rstd, z, ctx.group_size,
                                         ctx.norm_before_gate, ctx.is_rms_norm)
        return dx.reshape(ctx.x_shape_og), dw, db, dz.reshape(ctx.x_shape_og) if dz is not None else None, None, None, None, None


def layernorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, is_rms_norm=False):
    return LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)


def rmsnorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    return LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, True)


class LayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, device=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        return layernorm_fn(x, self.weight, self.bias, z=z, group_size=self.group_size, eps=self.eps,
                            norm_before_gate=self.norm_before_gate)


class RMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, device=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        return rmsnorm_fn(x, self.weight, self.bias, z=z, eps=self.eps, group_size=self.group_size,
                          norm_before_gate=self.norm_before_gate)
