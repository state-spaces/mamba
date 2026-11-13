"""A2 kernels for the Mamba-3 MIMO operator family.

``a2_prims`` holds the reusable pieces (matrix + elementwise primitives); the
``mamba3_mimo_*`` modules compose them into the operator entry points.
"""

from __future__ import annotations

from mamba_ssm.ops.tilelang_ascend.mamba3 import a2_prims

__all__ = ["a2_prims"]
