"""Ascend (A2) test suite for the Mamba-3 MIMO TileLang port.

Scope: this package holds the **upstream-standard** suite only --
``test_mamba3_mimo_upstream_std.py``, i.e. the repo's own
``tests/ops/tilelang/test_mamba3_mimo.py`` re-pointed at the NPU.  It lives here,
not inside ``mamba_ssm/ops/tilelang_ascend/``, because a test suite is not part of
the operator package.

The upstream suite cannot be imported as-is: its ``mods`` fixture requires CUDA and
imports ``mamba_ssm.ops.tilelang.mamba3`` (the CUDA kernels).  Its *assets*,
however, are device agnostic and are reused verbatim through :mod:`._upstream`:

* ``build_inputs``            -- input construction (re-pointed at the NPU)
* ``mamba3_MIMO_chunk_ref``   -- fp32 reference forward
* ``run_ref_backward_fp32``   -- fp32 reference backward (autograd)

Everything else -- the case grid, the input seeds, the dtypes and the accuracy
criterion (``max|ours - ref| / max|ref| < REL_TOL``, ``REL_TOL = 0.10``, i.e.
upstream's ``assert_stable_rel``) -- is taken from the upstream file at run time, so
a tolerance change upstream is followed here instead of being re-derived.

Run it with::

    pytest -q -p no:warnings tests/ops/tilelang_ascend/test_mamba3_mimo_upstream_std.py

``A2_STD_S`` / ``A2_STD_B`` / ``A2_STD_H`` shrink the sequence geometry (default:
upstream's ``B4 S2048 H16``); the verdict does not depend on them.  See
``USAGE_mamba3_mimo_npu.md`` at the repository root.
"""
