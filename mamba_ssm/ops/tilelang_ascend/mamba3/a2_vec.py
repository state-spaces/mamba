"""Vector (AIV/UB) primitives for the A2 Mamba-3 port.

What the A2 backend supports, measured on 910B3 (``MIGRATION.md`` §4.4-4.5):

* same-shape, same-dtype arithmetic inside ``T.Parallel`` -- ``a*b+c``, ``a*b``,
  ``a/(b+1)`` all reproduce torch bit-closely;
* the whole-buffer intrinsics ``T.npuir_exp`` and ``T.npuir_sigmoid``;
* **not** broadcasts (a 1-D UB buffer indexed by the outer loop variable) and
  **not** ``T.if_then_else`` whose result is cast -- both silently produce wrong
  values or kill the frontend;
* a kernel whose UB buffers exceed the 192 KB AIV local memory fails to compile,
  so every kernel here processes a **row block** and the grid walks the blocks.

The port therefore keeps all *input-only* coefficient maths on the host (cos/sin
of the rotary angles, the segsum causal mask, the per-token decay factors, the
gating product) and pre-expands every broadcast to the operand shape.  Device
kernels then only ever see equal-shaped operands, which is the style that works.
"""

import math
import os

import torch

import tilelang
import tilelang.ascend910.language as T

from mamba_ssm.ops.tilelang_ascend.mamba3.a2_prims import compile_kernel

#: Rows per vector kernel block.  64 rows x 256 cols of fp16 is 32 KB per buffer,
#: four buffers stay well inside the 192 KB AIV local memory.
BR = 64


def _nblocks(rows: int, br: int) -> int:
    if rows % br:
        raise ValueError(f"rows={rows} is not a multiple of the block size {br}")
    return rows // br


def _pad_rows_to(rows: int, br: int = BR) -> int:
    """Round ``rows`` up to a whole number of ``br`` blocks.

    ``BR`` is a *contract* of the kernels, not just a performance knob: every tile
    is allocated and copied as a full ``(br, cols)`` block, and a partial tile
    reads garbage.  ``br = 32`` in particular is miscompiled -- the ``_rotary``
    kernel returns stale shared memory (measured: ``max`` jumping between 5.6 and
    6e4 on identical input, then NaN).  Callers therefore zero pad the row axis up
    to a block multiple and slice the pad away; the kernels are row-independent,
    so the pad cannot change the real rows.
    """

    if rows % br == 0:
        return rows
    return (rows + br - 1) // br * br


def _pad_rows_tensor(x: "torch.Tensor", rows: int, br: int = BR):
    """Zero pad ``x`` to ``rows`` rows (already rounded up by :func:`_pad_rows_to`)."""

    if rows == x.shape[0]:
        return x
    pad = torch.zeros((rows - x.shape[0], x.shape[1]), dtype=x.dtype,
                      device=x.device)
    return torch.cat((x, pad), dim=0)


#: Narrowest legal tile row, in elements.  The tile's row width doubles as the DMA
#: row stride, so a row below one 32 B block makes the row-strided copy degenerate
#: (measured on A2: cols 4/6/8/10/12/14 return garbage, 16 and above are exact).
#: Element counts, not bytes, because both fp16 and bf16 are 2 B wide.
_MIN_TILE_COLS = 16

#: AIV local memory (UB) available to a kernel, in bytes.
_UB_BYTES = 192 * 1024

#: Peak UB per tile element of the wide (bf16) elementwise path, in bytes.
#: Measured from the compiler's own accounting rather than assumed: a ``64x256``
#: bf16 tile was rejected with "requires 1835008 bits while 1572864 bits
#: available", i.e. 229376 B for 16384 elements = 14 B/element (fp16 and bf16 are
#: both 2 B, so this is dtype independent in this port).
_WIDE_BYTES_PER_ELEM = 14

#: Widest elementwise tile, in columns.  ``64 x 128`` needs 64 * 128 * 14 = 114 KB of
#: the 192 KB UB (41 % headroom); ``64 x 256`` needs 229 KB and fails to compile with
#: ``ub overflow``.  N=256 therefore becomes two column blocks -- see
#: :func:`run_ewise`.  Kept a power of two so blocking a 256-wide operand does not
#: produce two differently shaped kernels.
_EWISE_MAX_COLS = 128

assert BR * _EWISE_MAX_COLS * _WIDE_BYTES_PER_ELEM <= _UB_BYTES, (
    f"elementwise tile {BR}x{_EWISE_MAX_COLS} does not fit the {_UB_BYTES} B UB")


def _pad_cols(x: "torch.Tensor", cols: int):
    """Zero pad ``x`` to ``cols`` columns, or return it unchanged if already wide."""

    if cols == x.shape[1]:
        return x
    pad = torch.zeros((x.shape[0], cols - x.shape[1]), dtype=x.dtype,
                      device=x.device)
    return torch.cat((x, pad), dim=1)


#: Accumulation dtype of the wide (``bfloat16``) path.
_ACC = "float32"


def _wide(dtype: str) -> bool:
    """Whether ``dtype`` needs the explicit fp32 staging path.

    ``bfloat16`` has no arithmetic in the ``hivm`` dialect the vectorizer emits
    (its verifier accepts only i16/i32/f16/f32/i64), and the compiler's widening
    pass runs *before* the vectorizer that creates those ops, so a bf16
    ``T.Parallel`` body reaches codegen un-widened.  The kernels therefore spell
    the widening out themselves.  fp16/fp32 keep the plain path so their numerics
    stay bit-identical.
    """

    return dtype in ("bfloat16", "bf16")


def _ewise_builder(op):
    """``Y = f(A, B, C)`` over a row block of equal-shaped fp16 operands."""

    def build_plain(rows, cols, dtype="float16", br=BR):
        nb = _nblocks(rows, br)

        @T.prim_func
        def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((rows, cols), dtype),
                 C: T.Tensor((rows, cols), dtype), Y: T.Tensor((rows, cols), dtype)):
            with T.Kernel(nb, is_npu=True) as (cid, _sid):
                a = T.alloc_shared((br, cols), dtype, scope="shared")
                b = T.alloc_shared((br, cols), dtype, scope="shared")
                c = T.alloc_shared((br, cols), dtype, scope="shared")
                y = T.alloc_shared((br, cols), dtype, scope="shared")
                T.copy(A[cid * br, 0], a)
                T.copy(B[cid * br, 0], b)
                T.copy(C[cid * br, 0], c)
                T.pipe_barrier("PIPE_ALL")
                if op == "add":
                    for i, j in T.Parallel(br, cols):
                        y[i, j] = a[i, j] + b[i, j]
                elif op == "sub":
                    for i, j in T.Parallel(br, cols):
                        y[i, j] = a[i, j] - b[i, j]
                elif op == "mul":
                    for i, j in T.Parallel(br, cols):
                        y[i, j] = a[i, j] * b[i, j]
                elif op == "axpy":
                    for i, j in T.Parallel(br, cols):
                        y[i, j] = a[i, j] * b[i, j] + c[i, j]
                elif op == "axmy":
                    for i, j in T.Parallel(br, cols):
                        y[i, j] = a[i, j] * b[i, j] - c[i, j]
                elif op == "div":
                    for i, j in T.Parallel(br, cols):
                        y[i, j] = a[i, j] / b[i, j]
                else:
                    raise KeyError(op)
                T.pipe_barrier("PIPE_ALL")
                T.copy(y, Y[cid * br, 0])

        return main

    def build_wide(rows, cols, dtype="bfloat16", br=BR):
        """Same kernel with the arithmetic staged through fp32.

        Inputs are cast into fp32 shared buffers with ``T.npuir_cast``, the
        arithmetic runs on those, and the result is cast back before the store.
        An expression-level cast (``a[i, j].astype("float32")``) also compiles on
        this CANN build but mis-lowers: measured ``rel_l2 = 1.3e+01`` against the
        fp32 reference, i.e. garbage rather than a rounding difference.

        The staging doubles the footprint of this kernel's tiles (fp16 keeps the
        single-dtype path for that reason); the shapes that reach it are
        ``(rows, 64)``, which is 96 KB at ``br=64`` and fits the 192 KB AIV UB.
        """

        nb = _nblocks(rows, br)

        @T.prim_func
        def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((rows, cols), dtype),
                 C: T.Tensor((rows, cols), dtype), Y: T.Tensor((rows, cols), dtype)):
            with T.Kernel(nb, is_npu=True) as (cid, _sid):
                a = T.alloc_shared((br, cols), dtype, scope="shared")
                b = T.alloc_shared((br, cols), dtype, scope="shared")
                c = T.alloc_shared((br, cols), dtype, scope="shared")
                y = T.alloc_shared((br, cols), dtype, scope="shared")
                af = T.alloc_shared((br, cols), _ACC, scope="shared")
                bf = T.alloc_shared((br, cols), _ACC, scope="shared")
                cf = T.alloc_shared((br, cols), _ACC, scope="shared")
                yf = T.alloc_shared((br, cols), _ACC, scope="shared")
                T.copy(A[cid * br, 0], a)
                T.copy(B[cid * br, 0], b)
                T.copy(C[cid * br, 0], c)
                T.pipe_barrier("PIPE_ALL")
                T.npuir_cast(a, af)
                T.npuir_cast(b, bf)
                T.npuir_cast(c, cf)
                T.pipe_barrier("PIPE_ALL")
                if op == "add":
                    for i, j in T.Parallel(br, cols):
                        yf[i, j] = af[i, j] + bf[i, j]
                elif op == "sub":
                    for i, j in T.Parallel(br, cols):
                        yf[i, j] = af[i, j] - bf[i, j]
                elif op == "mul":
                    for i, j in T.Parallel(br, cols):
                        yf[i, j] = af[i, j] * bf[i, j]
                elif op == "axpy":
                    for i, j in T.Parallel(br, cols):
                        yf[i, j] = af[i, j] * bf[i, j] + cf[i, j]
                elif op == "axmy":
                    for i, j in T.Parallel(br, cols):
                        yf[i, j] = af[i, j] * bf[i, j] - cf[i, j]
                elif op == "div":
                    for i, j in T.Parallel(br, cols):
                        yf[i, j] = af[i, j] / bf[i, j]
                else:
                    raise KeyError(op)
                T.pipe_barrier("PIPE_ALL")
                T.npuir_cast(yf, y)
                T.pipe_barrier("PIPE_ALL")
                T.copy(y, Y[cid * br, 0])

        return main

    def build(rows, cols, dtype="float16", br=BR):
        maker = build_wide if _wide(dtype) else build_plain
        return maker(rows, cols, dtype, br)

    return build


_EWISE = {op: _ewise_builder(op) for op in ("add", "sub", "mul", "axpy", "axmy", "div")}


def _zeros_like_shape(x):
    return torch.zeros_like(x)


_KCACHE = {}


def _cached(build, key, *args, out_idx=-1):
    kern = _KCACHE.get(key)
    if kern is None:
        kern = compile_kernel(build(*args), out_idx=out_idx)
        _KCACHE[key] = kern
    return kern


def run_ewise(op, a, b=None, c=None, br=BR):
    """Elementwise kernel over ``[rows, cols]`` fp16/bf16 operands.

    ``b``/``c`` default to zeros/ones so a two-operand op can be expressed
    without materialising a dummy tensor at the call site.

    Operands wider than :data:`_EWISE_MAX_COLS` are evaluated in column blocks and
    concatenated.  The column axis is the one to cut: ``br`` is a contract of the
    kernels (``br = 32`` is miscompiled, see :func:`_pad_rows_to`), while the tile
    width is what drives the UB footprint.  Blocking also keeps the block width a
    power of two, so a 256-wide operand compiles one 128-wide kernel instead of two
    differently shaped ones.
    """

    cols = a.shape[1]
    if cols > _EWISE_MAX_COLS:
        return torch.cat(
            [run_ewise(op,
                       a[:, c0:c0 + _EWISE_MAX_COLS].contiguous(),
                       None if b is None else b[:, c0:c0 + _EWISE_MAX_COLS].contiguous(),
                       None if c is None else c[:, c0:c0 + _EWISE_MAX_COLS].contiguous(),
                       br)
             for c0 in range(0, cols, _EWISE_MAX_COLS)],
            dim=1)

    rows = a.shape[0]
    b = torch.ones_like(a) if b is None else b
    c = torch.zeros_like(a) if c is None else c
    dt = str(a.dtype).replace("torch.", "")
    rows_p = _pad_rows_to(rows, br)
    kern = _cached(_EWISE[op], (op, rows_p, cols, dt, br), rows_p, cols, dt, br)
    y = kern(_pad_rows_tensor(a, rows_p, br), _pad_rows_tensor(b, rows_p, br),
             _pad_rows_tensor(c, rows_p, br))
    return y[:rows]


def _unary_builder(op):
    def build(rows, cols, dtype="float16", br=BR):
        nb = _nblocks(rows, br)

        @T.prim_func
        def main(A: T.Tensor((rows, cols), dtype), Y: T.Tensor((rows, cols), dtype)):
            with T.Kernel(nb, is_npu=True) as (cid, _sid):
                a = T.alloc_shared((br, cols), dtype, scope="shared")
                y = T.alloc_shared((br, cols), dtype, scope="shared")
                T.copy(A[cid * br, 0], a)
                T.pipe_barrier("PIPE_ALL")
                if op == "exp":
                    T.npuir_exp(a, y)
                elif op == "sigmoid":
                    T.npuir_sigmoid(a, y)
                else:
                    raise KeyError(op)
                T.pipe_barrier("PIPE_ALL")
                T.copy(y, Y[cid * br, 0])

        return main

    return build


_UNARY = {op: _unary_builder(op) for op in ("exp", "sigmoid")}


def run_unary(op, a, br=BR):
    rows, cols = a.shape
    dt = str(a.dtype).replace("torch.", "")
    rows_p = _pad_rows_to(rows, br)
    kern = _cached(_UNARY[op], (op, rows_p, cols, dt, br), rows_p, cols, dt, br,
                   out_idx=None)
    y = torch.empty((rows_p, cols), dtype=a.dtype, device=a.device)
    kern(_pad_rows_tensor(a, rows_p, br), y)
    return y[:rows]


def _rotary_builder():
    """Rotary pair: ``y1 = a*c - b*s``, ``y2 = a*s + b*c``.

    Written as two same-shape outputs so no column-half gather is needed; the
    caller passes the two halves of the state vector as separate tensors.
    """

    def build_plain(rows, cols, dtype="float16", br=BR):
        nb = _nblocks(rows, br)

        @T.prim_func
        def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((rows, cols), dtype),
                 CO: T.Tensor((rows, cols), dtype), SI: T.Tensor((rows, cols), dtype),
                 Y1: T.Tensor((rows, cols), dtype), Y2: T.Tensor((rows, cols), dtype)):
            with T.Kernel(nb, is_npu=True) as (cid, _sid):
                a = T.alloc_shared((br, cols), dtype, scope="shared")
                b = T.alloc_shared((br, cols), dtype, scope="shared")
                co = T.alloc_shared((br, cols), dtype, scope="shared")
                si = T.alloc_shared((br, cols), dtype, scope="shared")
                y1 = T.alloc_shared((br, cols), dtype, scope="shared")
                y2 = T.alloc_shared((br, cols), dtype, scope="shared")
                T.copy(A[cid * br, 0], a)
                T.copy(B[cid * br, 0], b)
                T.copy(CO[cid * br, 0], co)
                T.copy(SI[cid * br, 0], si)
                T.pipe_barrier("PIPE_ALL")
                for i, j in T.Parallel(br, cols):
                    y1[i, j] = a[i, j] * co[i, j] - b[i, j] * si[i, j]
                for i, j in T.Parallel(br, cols):
                    y2[i, j] = a[i, j] * si[i, j] + b[i, j] * co[i, j]
                T.pipe_barrier("PIPE_ALL")
                T.copy(y1, Y1[cid * br, 0])
                T.copy(y2, Y2[cid * br, 0])

        return main

    def build_wide(rows, cols, dtype="bfloat16", br=BR):
        """Rotary pair with the arithmetic staged through fp32 (see ``_wide``).

        ``cos``/``sin`` arrive as precomputed tensors -- the kernel has no
        transcendental function -- so this is the same expression-level widening
        the elementwise kernels use, applied to two outputs.
        """

        nb = _nblocks(rows, br)

        @T.prim_func
        def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((rows, cols), dtype),
                 CO: T.Tensor((rows, cols), dtype), SI: T.Tensor((rows, cols), dtype),
                 Y1: T.Tensor((rows, cols), dtype), Y2: T.Tensor((rows, cols), dtype)):
            with T.Kernel(nb, is_npu=True) as (cid, _sid):
                a = T.alloc_shared((br, cols), dtype, scope="shared")
                b = T.alloc_shared((br, cols), dtype, scope="shared")
                co = T.alloc_shared((br, cols), dtype, scope="shared")
                si = T.alloc_shared((br, cols), dtype, scope="shared")
                y1 = T.alloc_shared((br, cols), dtype, scope="shared")
                y2 = T.alloc_shared((br, cols), dtype, scope="shared")
                af = T.alloc_shared((br, cols), _ACC, scope="shared")
                bf = T.alloc_shared((br, cols), _ACC, scope="shared")
                cof = T.alloc_shared((br, cols), _ACC, scope="shared")
                sif = T.alloc_shared((br, cols), _ACC, scope="shared")
                T.copy(A[cid * br, 0], a)
                T.copy(B[cid * br, 0], b)
                T.copy(CO[cid * br, 0], co)
                T.copy(SI[cid * br, 0], si)
                T.pipe_barrier("PIPE_ALL")
                T.npuir_cast(a, af)
                T.npuir_cast(b, bf)
                T.npuir_cast(co, cof)
                T.npuir_cast(si, sif)
                T.pipe_barrier("PIPE_ALL")
                for i, j in T.Parallel(br, cols):
                    y1[i, j] = af[i, j] * cof[i, j] - bf[i, j] * sif[i, j]
                for i, j in T.Parallel(br, cols):
                    y2[i, j] = af[i, j] * sif[i, j] + bf[i, j] * cof[i, j]
                T.pipe_barrier("PIPE_ALL")
                T.copy(y1, Y1[cid * br, 0])
                T.copy(y2, Y2[cid * br, 0])

        return main

    def build(rows, cols, dtype="float16", br=BR):
        maker = build_wide if _wide(dtype) else build_plain
        return maker(rows, cols, dtype, br)

    return build


_rotary = _rotary_builder()


def run_rotary(a, b, cos, sin, br=BR):
    """Apply the rotary pair to the two halves of ``[rows, cols]`` operands.

    ``cols`` is both the tile's row width and the DMA row stride, so a row
    narrower than one 32 B block makes the row-strided copy degenerate: measured
    on A2, ``cols`` 4/6/8/10/12/14 all come back wrong (``y1`` partly NaN, ``y2``
    off by 1.1-1.6) while 16 and above are correct.  Narrow tiles are therefore
    zero padded to 16 columns here and sliced back afterwards, which keeps
    ``N // rotary_dim_divisor < 16`` (``N=32`` at ``rotary_dim_divisor=4``)
    correct instead of rejecting it.
    """

    rows, cols = a.shape
    cols_p = max(cols, _MIN_TILE_COLS)
    dtype = str(a.dtype).replace("torch.", "")
    rows_p = _pad_rows_to(rows, br)
    kern = _cached(_rotary, ("rotary", rows_p, cols_p, dtype, br), rows_p,
                   cols_p, dtype, br, out_idx=None)
    y1 = torch.empty((rows_p, cols_p), dtype=a.dtype, device=a.device)
    y2 = torch.empty((rows_p, cols_p), dtype=b.dtype, device=b.device)
    kern(_pad_cols(_pad_rows_tensor(a, rows_p, br), cols_p),
         _pad_cols(_pad_rows_tensor(b, rows_p, br), cols_p),
         _pad_cols(_pad_rows_tensor(cos.contiguous(), rows_p, br), cols_p),
         _pad_cols(_pad_rows_tensor(sin.contiguous(), rows_p, br), cols_p),
         y1, y2)
    return y1[:rows, :cols], y2[:rows, :cols]
# NOTE (known gap): the rank reduction ``sum_r out[cs*R + r, p]`` has no device
# primitive yet.  The flat ``[C*R, P]`` view makes the rank axis strided, and the
# candidate kernels (strided row copies, a 3-D UB tile, a flattened 3-D tile)
# either lose the loop variable in layout inference ("cannot find variable i") or
# fail the vendor compile.  The forward path therefore reduces on the host for
# now; deviceising it is a follow-up once a working stride-aware copy exists.