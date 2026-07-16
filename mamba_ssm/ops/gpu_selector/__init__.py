"""
GPU-aware MIMO kernel selection module.

Provides automatic detection and selection of appropriate MIMO kernels
based on GPU architecture (Hopper, Ada, Ampere, Volta).
"""

from .mimo_kernel_selector import (
    GPUArchitecture,
    MIMOKernelSelector,
    get_mimo_kernel,
    enable_mimo_on_device,
    mamba3_mimo_combined_auto,
)

__all__ = [
    "GPUArchitecture",
    "MIMOKernelSelector",
    "get_mimo_kernel",
    "enable_mimo_on_device",
    "mamba3_mimo_combined_auto",
]
