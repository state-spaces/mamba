"""TileLang Ascend910 (A2) port of the Mamba-3 MIMO kernels.

This package replaces the earlier AscendC/PTO-dialect port, which targeted a
different TileLang release (``tilelang-ascend`` v0.1.4, ``ascendc`` dialect).
The current port targets the installed ``tilelang 0.1.15`` wheel and the
``tilelang.ascend910.language`` dialect, and runs on 910B3 with the
``bishengir-compile`` that ships inside CANN 9.1.0.

See ``MIGRATION.md`` §3 (toolchain pairing), §4.2 (the mandatory pass
configuration) and §5 (the host-side maths that the port preserves).  The
``bishengir-compile`` in CANN 9.1.0 has a crash bug that the port works around;
a patched build is shipped in ``3rdparty/bishengir/`` (see its README).
"""

from __future__ import annotations

from mamba_ssm.ops.tilelang_ascend.mamba3 import a2_prims

__all__ = ["a2_prims"]
