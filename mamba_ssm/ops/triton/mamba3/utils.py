"""
Mamba-3 Util Functions.

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

import triton
import triton.language as tl

# Portable trig/activation helpers using Triton builtins.
# These compile to platform-appropriate instructions on both NVIDIA and AMD.

@triton.jit
def cos_approx(x):
    """
    Cosine via Triton builtin (portable across NVIDIA and AMD backends).

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Cosine values in float32
    """
    return tl.cos(x)


@triton.jit
def sin_approx(x):
    """
    Sine via Triton builtin (portable across NVIDIA and AMD backends).

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Sine values in float32
    """
    return tl.sin(x)

@triton.jit
def tanh_approx(x):
    """
    Hyperbolic tangent computed as 2*sigmoid(2x) - 1.

    Mathematically equivalent to tanh(x). Uses tl.sigmoid which is
    portable across all Triton backends.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Tanh values in float32
    """
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0

@triton.jit
def sech2_approx(x):
    """
    Square of hyperbolic secant: sech^2(x) = 1 - tanh^2(x).

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        sech^2 values in float32
    """
    tanh_x = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    return 1.0 - tanh_x * tanh_x

@triton.jit
def sigmoid_approx(x):
    """
    Sigmoid via Triton builtin.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Sigmoid values in float32
    """
    return tl.sigmoid(x)

@triton.jit
def silu(x):
    """
    SiLU (Swish) activation: x * sigmoid(x).

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        SiLU activation output in float32
    """
    return x * tl.sigmoid(x)
