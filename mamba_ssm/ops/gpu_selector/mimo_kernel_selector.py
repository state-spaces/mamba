"""
GPU device detection and kernel selection for Mamba3 MIMO.

This module provides automatic selection between different kernel implementations
based on the target GPU architecture, enabling seamless compatibility across
Blackwell, Hopper, Ada, Ampere, Volta, and other NVIDIA GPU architectures.

Supported GPUs:
- Blackwell (RTX 50xx, GB100, GB200): Uses optimized TileLang kernel with TMA
- Hopper (H100, H200): Uses optimized TileLang kernel with TMA
- Ada (RTX 40xx, L40, L40S): Uses GPU-compatible TileLang kernel
- Ampere (RTX 30xx, A100): Uses Triton fallback kernel
- Volta (V100): Uses Triton fallback kernel (limited support)

Copyright (c) 2026, Dao AI Lab, Goombalab
"""

import torch
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


# GPU Architecture Detection Mapping
GPU_ARCHITECTURE_MAP = {
    # Blackwell (10.0)
    "NVIDIA GeForce RTX 5050": (10, 0),
    "NVIDIA GeForce RTX 5050 Laptop GPU": (10, 0),
    "NVIDIA GeForce RTX 5060": (10, 0),
    "NVIDIA GeForce RTX 5060 Laptop GPU": (10, 0),
    "NVIDIA GeForce RTX 5070": (10, 0),
    "NVIDIA GeForce RTX 5070 Laptop GPU": (10, 0),
    "NVIDIA GeForce RTX 5080": (10, 0),
    "NVIDIA GeForce RTX 5090": (10, 0),
    "GB100": (10, 0),
    "GB200": (10, 0),
    
    # Hopper (9.0)
    "H100": (9, 0),
    "H200": (9, 0),
    
    # Ada (8.9)
    "RTX 6000 Ada": (8, 9),
    "RTX 5880 Ada": (8, 9),
    "RTX 5000 Ada": (8, 9),
    "RTX 6000 ADA": (8, 9),
    "L40S": (8, 9),
    "L40": (8, 9),
    
    # Ampere (8.0)
    "RTX A6000": (8, 0),
    "RTX A5900": (8, 0),
    "RTX A5000": (8, 0),
    "RTX A4500": (8, 0),
    "RTX A4000": (8, 0),
    "RTX A2000": (8, 0),
    "RTX 3090": (8, 6),
    "RTX 3090 Ti": (8, 6),
    "RTX 3080 Ti": (8, 6),
    "RTX 3080": (8, 6),
    "RTX 3070 Ti": (8, 6),
    "RTX 3070": (8, 6),
    "RTX 3060 Ti": (8, 6),
    "RTX 3060": (8, 6),
    "A100-PCIE-40GB": (8, 0),
    "A100-PCIE-80GB": (8, 0),
    "A100-SXM-40GB": (8, 0),
    "A100-SXM-80GB": (8, 0),
    "A30": (8, 0),
    "A10": (8, 6),
    "A10G": (8, 6),
    
    # Volta (7.0)
    "V100-PCIE-16GB": (7, 0),
    "V100-PCIE-32GB": (7, 0),
    "V100-SXM2-16GB": (7, 0),
    "V100-SXM2-32GB": (7, 0),
}

# Compute capability ranges for kernel selection
BLACKWELL_MIN_CC = (10, 0)
HOPPER_MIN_CC = (9, 0)
ADA_MIN_CC = (8, 9)
AMPERE_MIN_CC = (8, 0)
VOLTA_MIN_CC = (7, 0)


class GPUArchitecture:
    """Represents GPU architecture and capabilities."""
    
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.device = torch.device(f"cuda:{device_index}")
        self.device_name = torch.cuda.get_device_name(device_index)
        self.compute_capability = torch.cuda.get_device_capability(device_index)
        self.major, self.minor = self.compute_capability
        
        logger.info(f"GPU Device {device_index}: {self.device_name}")
        logger.info(f"Compute Capability: {self.major}.{self.minor}")
    
    @property
    def is_blackwell(self) -> bool:
        """Check if GPU is Blackwell architecture."""
        return self.compute_capability >= BLACKWELL_MIN_CC
    
    @property
    def is_hopper(self) -> bool:
        """Check if GPU is Hopper architecture."""
        return self.compute_capability >= HOPPER_MIN_CC and not self.is_blackwell
    
    @property
    def is_ada(self) -> bool:
        """Check if GPU is Ada architecture."""
        return self.compute_capability >= ADA_MIN_CC and not self.is_blackwell and not self.is_hopper
    
    @property
    def is_ampere(self) -> bool:
        """Check if GPU is Ampere architecture."""
        return self.compute_capability >= AMPERE_MIN_CC and not self.is_blackwell and not self.is_hopper and not self.is_ada
    
    @property
    def is_volta(self) -> bool:
        """Check if GPU is Volta architecture."""
        return self.compute_capability >= VOLTA_MIN_CC and not self.is_blackwell and not self.is_hopper and not self.is_ada and not self.is_ampere
    
    @property
    def architecture_name(self) -> str:
        """Get human-readable architecture name."""
        if self.is_blackwell:
            return "Blackwell"
        elif self.is_hopper:
            return "Hopper"
        elif self.is_ada:
            return "Ada"
        elif self.is_ampere:
            return "Ampere"
        elif self.is_volta:
            return "Volta"
        else:
            return "Unknown"
    
    def __repr__(self) -> str:
        return f"GPUArchitecture({self.device_name}, CC {self.major}.{self.minor})"


class MIMOKernelSelector:
    """
    Selects appropriate MIMO kernel based on GPU architecture.
    
    Provides a unified interface for kernel selection across different GPU types,
    with automatic fallback chain:
    1. Try optimized Blackwell/Hopper TileLang kernel (with TMA)
    2. Fall back to compatible TileLang kernel (Ada/Ampere)
    3. Fall back to Triton kernel (older GPUs, fallback)
    """
    
    def __init__(self, device_index: int = 0):
        self.gpu = GPUArchitecture(device_index)
        self._load_kernels()
    
    def _load_kernels(self):
        """Load appropriate kernel implementations."""
        # Try to import TileLang-based kernels
        self._tilelang_hopper_kernel = None
        self._tilelang_compat_kernel = None
        self._triton_kernel = None
        
        try:
            from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as mamba3_mimo_combined
            self._tilelang_hopper_kernel = mamba3_mimo_combined
            logger.debug("Loaded Hopper-optimized TileLang kernel (works on Blackwell/Hopper)")
        except (ImportError, Exception) as e:
            logger.debug(f"Could not load Hopper TileLang kernel: {e}")
        
        try:
            from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd_compat import mamba_mimo_forward_compat
            self._tilelang_compat_kernel = mamba_mimo_forward_compat
            logger.debug("Loaded GPU-compatible TileLang kernel")
        except (ImportError, Exception) as e:
            logger.debug(f"Could not load compatible TileLang kernel: {e}")
        
        try:
            from mamba_ssm.ops.triton.mamba3.mamba3_mimo_triton_fallback import mamba3_mimo_forward_triton
            self._triton_kernel = mamba3_mimo_forward_triton
            logger.debug("Loaded Triton fallback kernel")
        except (ImportError, Exception) as e:
            logger.debug(f"Could not load Triton kernel: {e}")
    
    def select_kernel(self, verbose: bool = True) -> Tuple[Any, str]:
        """
        Select the best kernel for the current GPU.
        
        Args:
            verbose: Whether to log kernel selection details
            
        Returns:
            Tuple of (kernel_function, kernel_name_str)
        """
        if self.gpu.is_blackwell or self.gpu.is_hopper:
            if self._tilelang_hopper_kernel is not None:
                if verbose:
                    logger.info(f"Using {self.gpu.architecture_name}-optimized TileLang kernel on {self.gpu.device_name}")
                return self._tilelang_hopper_kernel, f"{self.gpu.architecture_name.lower()}_tilelang"
        
        if self.gpu.is_blackwell or self.gpu.is_hopper or self.gpu.is_ada or self.gpu.is_ampere:
            if self._tilelang_compat_kernel is not None:
                if verbose:
                    logger.info(f"Using GPU-compatible TileLang kernel on {self.gpu.architecture_name} ({self.gpu.device_name})")
                return self._tilelang_compat_kernel, "compat_tilelang"
        
        if self._triton_kernel is not None:
            if verbose:
                logger.info(f"Using Triton fallback kernel on {self.gpu.architecture_name} ({self.gpu.device_name})")
            return self._triton_kernel, "triton_fallback"
        
        raise RuntimeError(
            f"No suitable MIMO kernel found for GPU {self.gpu.device_name} (CC {self.gpu.major}.{self.gpu.minor}). "
            "Please ensure TileLang or Triton is installed."
        )
    
    def get_kernel_features(self) -> Dict[str, bool]:
        """Get supported features for the selected kernel."""
        has_hopper = self._tilelang_hopper_kernel is not None
        has_compat = self._tilelang_compat_kernel is not None
        has_triton = self._triton_kernel is not None
        
        return {
            "blackwell_optimized": has_hopper and self.gpu.is_blackwell,
            "hopper_optimized": has_hopper and self.gpu.is_hopper,
            "tma_support": has_hopper and (self.gpu.is_blackwell or self.gpu.is_hopper),
            "warp_specialized": has_hopper and (self.gpu.is_blackwell or self.gpu.is_hopper),
            "tilelang_compatible": has_compat,
            "triton_fallback": has_triton,
        }


def get_mimo_kernel(device_index: int = 0, verbose: bool = True):
    """
    Get the appropriate MIMO kernel for the specified device.
    
    Args:
        device_index: CUDA device index
        verbose: Whether to log kernel selection
        
    Returns:
        Tuple of (kernel_function, kernel_name, gpu_architecture)
    """
    selector = MIMOKernelSelector(device_index)
    kernel, kernel_name = selector.select_kernel(verbose=verbose)
    
    return kernel, kernel_name, selector.gpu


def enable_mimo_on_device(device_index: int = 0) -> Dict[str, Any]:
    """
    Configure device for MIMO support and return kernel selection details.
    
    Args:
        device_index: CUDA device index
        
    Returns:
        Dictionary with kernel selection and device info
    """
    torch.cuda.set_device(device_index)
    
    kernel, kernel_name, gpu = get_mimo_kernel(device_index, verbose=True)
    
    features = {
        "device_index": device_index,
        "device_name": gpu.device_name,
        "compute_capability": f"{gpu.major}.{gpu.minor}",
        "architecture": gpu.architecture_name,
        "kernel_type": kernel_name,
        "features": MIMOKernelSelector(device_index).get_kernel_features(),
    }
    
    logger.info(f"MIMO enabled on device {device_index}: {features}")
    
    return features


# Legacy compatibility function
def mamba3_mimo_combined_auto(
    Q,
    K,
    V,
    ADT,
    DT,
    Trap,
    Q_bias,
    K_bias,
    MIMO_V,
    MIMO_Z,
    MIMO_Out,
    Angles,
    D,
    Z,
    chunk_size,
    rotary_dim_divisor,
    dtype,
    return_state=False,
    cu_seqlens=None,
    fuse_pregate_headwise_rms_norm=False,
    outproj_norm_weight=None,
    outproj_norm_eps=1e-5,
    device_index: int = 0,
):
    """
    Automatic MIMO kernel selector wrapper.
    
    This function automatically selects and calls the appropriate kernel
    based on the GPU architecture, providing a drop-in replacement for
    the original mamba3_mimo_combined function.
    
    All parameters are passed through to the selected kernel.
    """
    kernel, kernel_name, gpu = get_mimo_kernel(device_index, verbose=False)
    
    logger.debug(f"Using {kernel_name} kernel for MIMO forward pass")
    
    # Call the selected kernel with all provided arguments
    return kernel(
        Q=Q,
        K=K,
        V=V,
        ADT=ADT,
        DT=DT,
        Trap=Trap,
        Q_bias=Q_bias,
        K_bias=K_bias,
        MIMO_V=MIMO_V,
        MIMO_Z=MIMO_Z,
        MIMO_Out=MIMO_Out,
        Angles=Angles,
        D=D,
        Z=Z,
        chunk_size=chunk_size,
        rotary_dim_divisor=rotary_dim_divisor,
        dtype=dtype,
        return_state=return_state,
        cu_seqlens=cu_seqlens,
        fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
        outproj_norm_weight=outproj_norm_weight,
        outproj_norm_eps=outproj_norm_eps,
    )
