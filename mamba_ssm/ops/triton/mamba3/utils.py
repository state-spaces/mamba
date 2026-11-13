"""
Mamba-3 Util Functions.

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

import triton
import triton.language as tl

from mamba_ssm.ops.triton.mamba3.npu_compat import IS_CANN, libdevice as _cann_ld

# Upstream used PTX approximations (tanh.approx.f32 / sin.approx.f32 /
# cos.approx.f32) to trade a bit of accuracy for much faster speed.  Those
# instructions are NVIDIA-only and fail to compile on Triton-Ascend with
# "Unsupported other types than int64 for inline".
#
# We keep the fast PTX path on CUDA and use the mathematically equivalent CANN
# libdevice implementations on Ascend NPU.  The selection happens once, at
# import time, based on the active Triton backend.


if IS_CANN:

    @triton.jit
    def cos_approx(x):
        """Cosine of ``x`` (radians).  CANN libdevice implementation."""
        return _cann_ld.cos(x)

    @triton.jit
    def sin_approx(x):
        """Sine of ``x`` (radians).  CANN libdevice implementation."""
        return _cann_ld.sin(x)

    @triton.jit
    def tanh_approx(x):
        """Hyperbolic tangent of ``x``.  CANN libdevice implementation."""
        return _cann_ld.tanh(x)

    @triton.jit
    def sech2_approx(x):
        """Square of the hyperbolic secant, ``1 - tanh(x)^2``."""
        tanh_x = _cann_ld.tanh(x)
        return 1.0 - tanh_x * tanh_x

else:

    @triton.jit
    def cos_approx(x):
        """
        (Fast) Cosine approximation using PTX inline assembly.

        Args:
            x: Input triton tensor (any shape) in float32
        Returns:
            Approximate cosine values in float32
        """
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
        """
        (Fast) Sine approximation using PTX inline assembly.

        Args:
            x: Input triton tensor (any shape) in float32
        Returns:
            Approximate sine values in float32
        """
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
        """
        (Fast) hyperbolic tangent approximation using PTX inline assembly.

        Args:
            x: Input triton tensor (any shape) in float32
        Returns:
            Approximate tanh values in float32
        """
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
        """
        (Fast) square of the hyperbolic secant approximation using PTX inline assembly.

        Args:
            x: Input triton tensor (any shape) in float32
        Returns:
            Approximate sech^2 values in float32
        """
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
    """
    (Fast) Sigmoid approximation.

    Formula: sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))
    Leverages fast tanh approximation for speed.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate sigmoid values in float32
    """
    # NOTE: We ended up using the built-in sigmoid for better performance, as the
    # tanh-based approximation was not faster in this case.  ``tl.sigmoid`` is
    # backend-agnostic and works on both CUDA and Ascend NPU.
    return tl.sigmoid(x)


@triton.jit
def silu(x):
    """
    SiLU (Swish) activation function: x * sigmoid(x).

    Args:
        x: Input triton tensor (any shape) in float32

    Returns:
        SiLU activation output in float32
    """
    return x * tl.sigmoid(x)
