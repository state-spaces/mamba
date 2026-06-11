"""
Mamba-3 Util Functions.

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

import triton
import triton.language as tl


def _is_hip():
    """Detect whether the active GPU backend is AMD HIP/ROCm.

    Checks torch.version.hip first (set at PyTorch build time, always
    available) then falls back to querying the Triton runtime.  This
    avoids depending on the Triton driver being fully initialized at
    import time.
    """
    try:
        import torch
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return True
    except ImportError:
        pass
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


_IS_HIP: bool = _is_hip()   # evaluated once at import time


def _maxnreg(value):
    """Return maxnreg kwarg dict, empty on HIP where it is unsupported."""
    if value is None or _IS_HIP:
        return {}
    return {"maxnreg": value}


# maxnreg values to sweep in autotune config generators.
# On HIP all values collapse to {} so only [None] is needed to avoid
# generating duplicate configs that waste autotuning time.
MAXNREG_VALUES = [None] if _IS_HIP else [None, 128, 256]
MAXNREG_VALUES_SMALL = [None] if _IS_HIP else [None, 64, 128]


# ---------------------------------------------------------------------------
# Backend-conditional trig/activation helpers.
#
# On NVIDIA: use PTX SFU inline assembly (single-cycle approximate
#   instructions: cos.approx.f32, sin.approx.f32, tanh.approx.f32).
# On AMD HIP/ROCm: use portable Triton builtins (tl.cos, tl.sin,
#   tl.sigmoid) which compile to the appropriate AMDGCN instructions.
#
# The NVIDIA PTX inline asm uses the "=f,f" register constraint which is
# not recognized by the AMDGCN backend, causing:
#   "error: couldn't allocate output register for constraint 'f'"
#
# sigmoid_approx and silu already used tl.sigmoid on both backends
# (the original authors found no speed benefit from PTX for those).
# ---------------------------------------------------------------------------

if _IS_HIP:
    @triton.jit
    def cos_approx(x):
        """Cosine via portable Triton builtin (AMD/ROCm path)."""
        return tl.cos(x)

    @triton.jit
    def sin_approx(x):
        """Sine via portable Triton builtin (AMD/ROCm path)."""
        return tl.sin(x)

    @triton.jit
    def tanh_approx(x):
        """tanh(x) = 2*sigmoid(2x) - 1 (AMD/ROCm path)."""
        return 2.0 * tl.sigmoid(2.0 * x) - 1.0

    @triton.jit
    def sech2_approx(x):
        """sech^2(x) = 1 - tanh^2(x) (AMD/ROCm path)."""
        tanh_x = 2.0 * tl.sigmoid(2.0 * x) - 1.0
        return 1.0 - tanh_x * tanh_x

else:
    @triton.jit
    def cos_approx(x):
        """Fast cosine via PTX SFU instruction (NVIDIA path)."""
        return tl.inline_asm_elementwise(
            "cos.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def sin_approx(x):
        """Fast sine via PTX SFU instruction (NVIDIA path)."""
        return tl.inline_asm_elementwise(
            "sin.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def tanh_approx(x):
        """Fast tanh via PTX SFU instruction (NVIDIA path)."""
        return tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def sech2_approx(x):
        """Fast sech^2 via PTX SFU tanh + arithmetic (NVIDIA path)."""
        tanh_x = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return 1.0 - tanh_x * tanh_x


@triton.jit
def sigmoid_approx(x):
    """Sigmoid via Triton builtin (portable, both backends)."""
    return tl.sigmoid(x)

@triton.jit
def silu(x):
    """SiLU (Swish) activation: x * sigmoid(x) (portable, both backends)."""
    return x * tl.sigmoid(x)
