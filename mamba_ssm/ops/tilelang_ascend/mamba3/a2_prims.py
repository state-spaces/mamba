"""Ascend910 (A2) primitives shared by the ported Mamba-3 MIMO kernels.

Design notes
------------
The GPU kernels in ``mamba_ssm/ops/tilelang/mamba3`` are single fused per-(head,
batch) kernels written for a unified SM: ``T.Kernel(H, B, threads=...)``, register
fragments, ``T.Pipelined`` software pipelining, L2 swizzling and warp-level
reductions.  None of that exists on the A2 backend:

* ``T.Kernel`` takes a **1-D** grid and has no ``threads=``; thread domains only
  exist inside ``T.SimdVF``/``T.SimtVF``.
* there is no register "fragment" level -- on-chip storage is UB
  (``T.alloc_shared(..., scope="shared")``), L1 (``T.alloc_L1``) and the cube
  buffers ``T.alloc_L0A/L0B/L0C``.
* cube (AIC) and vector (AIV) are **different cores**; a kernel's core type is
  inferred from the buffer scopes it touches, so one kernel body cannot freely
  mix matrix and elementwise work.
* the vendor compiler segfaults in its auto-multibuffer pass on ordinary
  multi-dot kernels, so ``PASS_CONFIGS`` below must disable it and pair it with
  ``disable_hivm_auto_inject_sync`` -- both flags are load-bearing, dropping
  either brings the crash back (``MIGRATION.md`` §4.2).

Therefore the port keeps the *host-side* maths and the workspace layout of the
GPU version and expresses each step as an explicit A2 kernel operating on global
memory.  ``mamba3_mimo_fwd.py`` composes them; this module holds the reusable
pieces.

Conventions
-----------
* All tensor arguments are GM tensors declared with ``T.Tensor``.
* fp16/bf16 activations, fp32 accumulation, matching the GPU version.
* Matrix kernels always accumulate in L0C and write the finished tile once, so
  no GM read-modify-write is required.
"""

# NOTE: no ``from __future__ import annotations`` here.  It turns the
# ``T.Tensor(...)`` parameter annotations of a ``@T.prim_func`` into strings that
# tilelang's eager builder re-evaluates in module globals, so closure variables
# (tile sizes, dtypes) would be unresolvable ("name 'M' is not defined").

import os

import tilelang
import tilelang.ascend910.language as T
from tilelang.ascend910.language.gemm import gemm as a2_gemm

#: Pass configuration required on A2.  ``enable_auto_multi_buffer`` makes
#: ``bishengir-compile`` die with SIGSEGV as soon as a kernel contains more than
#: one ``npuir_dot`` (or a small/odd-shaped tile), and the auto sync injection
#: only pairs with that pass -- both flags are load-bearing.
#:
#: The underlying crash is a vendor bug, not a tile problem:
#: ``HFusionToHIVM.cpp:1351`` calls ``getResult(0)`` on a buffer-semantics
#: ``hivm.hir.mmadL1`` (which has zero SSA results), i.e. an out-of-bounds read,
#: so a Release build only dies *sometimes* (an assertion build aborts 100%).
#: A patched compiler is shipped in ``3rdparty/bishengir/``; its README says how
#: to point TileLang at it.
PASS_CONFIGS = {
    "npuir.enable_auto_multi_buffer": False,
    "npuir.disable_hivm_auto_inject_sync": True,
}

#: Cube tile: validated on 910B3.  ``bishengir-compile`` SIGSEGVs as soon as one
#: kernel contains more than a single ``npuir_dot``, so ``BK`` is the *whole* K
#: depth of a dot and K is walked by launching the kernel once per K tile
#: (accumulating in GM with ``T.npuir_atomic_add``), never by looping dots inside
#: one kernel.  ``BK = 128`` is the deepest single dot observed to work.
#:
#: The compiled kernel declares **tile-sized** buffers (``bm x bk``, ``bn x bk``,
#: ``bm x bn``): declaring the full padded operand (``Mp x K``) and copying the
#: tile with an offset instead makes ``bishengir-compile`` die with SIGSEGV as
#: soon as the declared extent -- or a tile offset -- reaches 128, which is what
#: used to cap ``F = chunk_size * rank`` at 128.  ``run_gemm`` therefore slices
#: the operands on the host; see ``tools/A2_mamba3_mimo_accuracy_report.md`` §11.1.
BM = 128
BN = 64
BK = 128


def _k_chunk(dtype: str, transpose_b: bool = True) -> int:
    """Deepest K one ``npuir_dot`` computes correctly at.

    Only ``fp16`` in the ``a @ b^T`` form survives the full ``BK = 128``:

    * bf16 and fp32 die in the Ascend NPU IR compile stage at depth 128 and work
      again at 64 (with the split, K=128 is exact for both -- ``rel ~ 3e-7``);
    * the ``a @ b`` form is *wrong* at depth 128 for every dtype tried -- fp16
      gives ``rel_l2`` between 1.0 and 40 or NaN depending on the run, with the
      kernel cache disabled, so it is a miscompile rather than a stale artifact.
      Nothing in this package uses that form; the port only multiplies ``a @ b^T``.

    Splitting costs one extra launch and nothing else: the partials are fp32 and
    summed by :func:`run_add`.  Caller and kernel must agree, hence the single
    helper: the host walks K in chunks of ``_k_chunk(dtype, transpose_b)`` while the
    prim_func is *built* at the same depth, and a disagreement fails the
    ``kk < n_kt`` check instead of computing the wrong tile.
    """

    if not transpose_b:
        return BK // 2
    return BK if dtype in ("float16", "fp16") else BK // 2


def compile_kernel(fn, out_idx=-1, retries=None):
    """Compile an A2 prim_func with the mandatory pass configuration.

    ``bishengir-compile`` segfaults intermittently on *identical* input: the same
    tile kernel compiles cleanly on one attempt and dies with ``Return code:
    -11`` on the next.  The cause is deterministic (see ``PASS_CONFIGS``) but a
    Release build only crashes *sometimes*, so retrying is a weak mitigation --
    the reliable path is a warm ``~/.tilelang/cache`` or the patched compiler in
    ``3rdparty/bishengir/``.  ``A2_COMPILE_RETRIES`` (default 4) bounds the
    attempts; 1 makes it fail fast.
    """

    if retries is None:
        retries = int(os.environ.get("A2_COMPILE_RETRIES", "4"))
    last = None
    for attempt in range(retries):
        try:
            return tilelang.compile(
                fn,
                target="ascend910",
                out_idx=out_idx,
                pass_configs=dict(PASS_CONFIGS),
            )
        except RuntimeError as exc:
            if "Return code: -11" not in str(exc):
                raise
            last = exc
            if os.environ.get("A2_COMPILE_VERBOSE"):
                print(f"  retry {attempt + 1}/{retries}: compiler segfault")
    raise last


def _pad_to(n: int, m: int) -> int:
    return (n + m - 1) // m * m


# ---------------------------------------------------------------- cube kernels


def _dot_tile(transpose_b):
    """Build a one-dot kernel factory: one BM x BN tile from one K chunk.

    ``bishengir-compile`` crashes on any kernel with two ``npuir_dot`` ops, so a
    single compiled kernel does exactly one dot and the caller walks the M/N/K
    tile grid by launching it (see :func:`run_gemm`).  ``kk > 0`` accumulates
    into ``C`` with a device atomic add, which keeps split-K in the kernel
    instead of in host code.
    """

    def build(M, N, K, im=0, jn=0, kk=0, bm=None, bn=None,
              dtype="float16", accum="float32"):
        bm = BM if bm is None else int(bm)
        bn = BN if bn is None else int(bn)
        # Must match ``run_gemm``'s chunking exactly (see ``_k_chunk``).
        kdep = min(K, _k_chunk(dtype, transpose_b))
        n_kt = _pad_to(K, kdep) // kdep
        assert 0 <= kk < n_kt, f"kk={kk} out of range for K={K}, kdep={kdep}"

        @T.prim_func
        def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype),
                 C: T.Tensor((M, N), accum)):
            with T.Kernel(1, is_npu=True) as (_cid, _sid):
                a_l1 = T.alloc_L1((bm, kdep), dtype)
                if transpose_b:
                    b_l1 = T.alloc_L1((bn, kdep), dtype)
                else:
                    b_l1 = T.alloc_L1((kdep, bn), dtype)
                c_l0 = T.alloc_L0C((bm, bn), accum)
                T.copy(A[im * bm, kk * kdep], a_l1)
                if transpose_b:
                    T.copy(B[jn * bn, kk * kdep], b_l1)
                else:
                    T.copy(B[kk * kdep, jn * bn], b_l1)
                T.pipe_barrier("PIPE_ALL")
                if transpose_b:
                    a2_gemm(a_l1, b_l1, c_l0, size=[bm, kdep, bn],
                            b_transpose=True, initC=True)
                else:
                    a2_gemm(a_l1, b_l1, c_l0, size=[bm, kdep, bn], initC=True)
                T.pipe_barrier("PIPE_ALL")
                # Each launch handles exactly one K chunk and writes a partial;
                # the chunks are summed on the device afterwards.  An in-kernel
                # ``npuir_atomic_add`` accumulation is not usable: layout
                # inference fails with "cannot find variable c_ub".
                T.npuir_store_fixpipe(c_l0, C[im * bm, jn * bn],
                                      size=[bm, bn], enable_nz2nd=True)

        return main

    return build


_GEMM_CACHE = {}

gemm_nt_tile = _dot_tile(True)
gemm_nn_tile = _dot_tile(False)


def _dtype_name(x) -> str:
    """``"bfloat16"``/``"float16"``/... for a torch tensor or a dtype object."""

    return str(x.dtype).replace("torch.", "") if hasattr(x, "dtype") else str(x)


def run_gemm(a, b, M, N, K, transpose_b=True, dtype=None, bm=None, bn=None):
    """Launch the single-dot tile kernels to compute ``a @ b^T`` (or ``a @ b``).

    Returns a fresh ``[M, N]`` fp32 tensor.  The K axis is walked by launching the
    kernel once per K chunk into a *separate* partial buffer and summing the
    partials with :func:`run_add`; an in-kernel atomic accumulation does not
    compile (layout inference fails), and a kernel may contain only one dot.

    Every tile is launched with **tile-sized** operands, so exactly one kernel is
    compiled per ``(bm, bn, kdep, dtype)`` no matter how large ``M``/``N`` are: the
    tile is sliced out on the host.  Declaring the full padded operand instead and
    copying the tile with an offset makes ``ConvertHFusionToHIVM`` in
    ``bishengir-compile`` segfault at a high rate once an extent reaches 128
    (measured 5/6 .. 9/10, and the retry loop below only sometimes rescues it);
    see ``tools/A2_mamba3_mimo_accuracy_report.md`` §11.1.

    ``dtype`` is only an *assertion* about the operands' type, never a request to
    compile at some other type: the kernel reads the raw buffers, so a mismatch is
    caught nowhere downstream and silently produces garbage (measured
    ``rel_l2 = 2.4`` with bf16 buffers at ``float16``, ``86.3`` the other way
    round, and NaN for fp32 buffers at ``float16``).  Omitting it uses the
    operands' own dtype, which is what every caller in this package does.  It used
    to default to a hard-coded ``"float16"`` -- invisible for fp16 calls, but it
    mis-read bf16 buffers (``rel_l2 = 1.27`` on a 64x64x64 gemm against ``4e-8``
    when the type matches, and a forward ~1.75x too large).
    """

    actual = _dtype_name(a)
    if dtype is None:
        dtype = actual
    else:
        dtype = _dtype_name(dtype)
        # ``raise`` rather than ``assert``: ``python -O`` strips asserts, which
        # would quietly re-open exactly the hole this check closes.
        if not (dtype == actual == _dtype_name(b)):
            raise ValueError(
                f"run_gemm: operands are {actual} and {_dtype_name(b)}, but "
                f"dtype={dtype!r} was requested; the kernel does not convert, it "
                "reads the buffers at whatever type it was compiled for")

    import torch

    bm = BM if bm is None else int(bm)
    bn = BN if bn is None else int(bn)
    Mp, Np = _pad_to(M, bm), _pad_to(N, bn)
    kdep = min(K, _k_chunk(dtype, transpose_b))
    if dtype in ("bfloat16", "bf16"):
        # A 16-deep bf16 dot still segfaults in ConvertHFusionToHIVM (fp16 at the
        # same depth is fine).  Zero-padding K up to 32 is numerically free.
        kdep = max(kdep, 32)
    n_kt = _pad_to(K, kdep) // kdep
    # The single-dot kernels always read and write a whole tile, so the operands
    # are zero padded to a tile (and K chunk) multiple; the pad is sliced away at
    # the end.  ``Kp`` also covers a K that is not a whole number of chunks.
    Kp = n_kt * kdep
    a_p = torch.zeros((Mp, Kp), device=a.device, dtype=a.dtype)
    a_p[:M, :K] = a
    if transpose_b:
        b_p = torch.zeros((Np, Kp), device=b.device, dtype=b.dtype)
        b_p[:N, :K] = b
    else:
        b_p = torch.zeros((Kp, Np), device=b.device, dtype=b.dtype)
        b_p[:K, :N] = b

    build = gemm_nt_tile if transpose_b else gemm_nn_tile
    # One kernel serves every tile: the compiled buffers are exactly bm x bn x kdep.
    key = (transpose_b, bm, bn, kdep, dtype)
    kern = _GEMM_CACHE.get(key)
    if kern is None:
        kern = compile_kernel(build(bm, bn, kdep, 0, 0, 0, bm=bm, bn=bn,
                                    dtype=dtype), out_idx=None)
        _GEMM_CACHE[key] = kern

    tile = torch.zeros((bm, bn), device=a.device, dtype=torch.float32)
    out = None
    for kk in range(n_kt):
        part = torch.zeros((Mp, Np), device=a.device, dtype=torch.float32)
        for im in range(Mp // bm):
            for jn in range(Np // bn):
                rows = a_p[im * bm:(im + 1) * bm,
                           kk * kdep:(kk + 1) * kdep].contiguous()
                if transpose_b:
                    cols = b_p[jn * bn:(jn + 1) * bn,
                               kk * kdep:(kk + 1) * kdep].contiguous()
                else:
                    cols = b_p[kk * kdep:(kk + 1) * kdep,
                               jn * bn:(jn + 1) * bn].contiguous()
                kern(rows, cols, tile)
                part[im * bm:(im + 1) * bm, jn * bn:(jn + 1) * bn] = tile
        out = part if out is None else run_add(out, part)
    return out[:M, :N]


def _add_builder():
    """``Y = A + B`` for any float dtype (used for the K-chunk partials)."""

    def build(rows, cols, dtype, br=64):
        nb = rows // br

        @T.prim_func
        def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((rows, cols), dtype),
                 Y: T.Tensor((rows, cols), dtype)):
            with T.Kernel(nb, is_npu=True) as (cid, _sid):
                a = T.alloc_shared((br, cols), dtype, scope="shared")
                b = T.alloc_shared((br, cols), dtype, scope="shared")
                y = T.alloc_shared((br, cols), dtype, scope="shared")
                T.copy(A[cid * br, 0], a)
                T.copy(B[cid * br, 0], b)
                T.pipe_barrier("PIPE_ALL")
                for i, j in T.Parallel(br, cols):
                    y[i, j] = a[i, j] + b[i, j]
                T.pipe_barrier("PIPE_ALL")
                T.copy(y, Y[cid * br, 0])

        return main

    return build


_add = _add_builder()
_ADD_CACHE = {}


def run_add(x, y):
    """Elementwise fp32/fp16 add of two equal-shaped GM tensors."""

    import torch as _torch

    rows, cols = x.shape
    dt = str(x.dtype).replace("torch.", "")
    key = (rows, cols, dt)
    kern = _ADD_CACHE.get(key)
    if kern is None:
        kern = compile_kernel(_add(rows, cols, dt), out_idx=None)
        _ADD_CACHE[key] = kern
    out = _torch.empty_like(x)
    kern(x, y, out)
    return out
