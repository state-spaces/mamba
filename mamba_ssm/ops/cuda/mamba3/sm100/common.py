"""Shared pieces of the Blackwell (sm_100 / sm_103) Mamba-3 MIMO kernels: JIT build, chunk layout, applicability.

The Olala configuration only: R=4, N=128, P=64, 32 rotary angles (rotary_dim_divisor 4), one Q/K group, reduceO with
the Z gate and D skip (no fused pre-gate RMSNorm), bf16 activations, varlen (B=1), no initial states.
"""

import os
import warnings
from functools import lru_cache

import torch

R, N, P, NQ = 4, 128, 64, 32


def sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@lru_cache(maxsize=None)
def _load(name: str, source: str):
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"  # tcgen05 needs the arch-specific target
    build = os.path.join(os.environ.get("M3_CUDA_BUILD_DIR", os.path.join(os.path.expanduser("~"), ".cache", "mamba3_cuda")),
                         f"{name}_sm{arch}_torch{torch.__version__.replace('+', '_')}")
    os.makedirs(build, exist_ok=True)
    return load(
        name=f"{name}_sm{arch}",
        sources=[os.path.join(here, source)],
        extra_include_paths=[here],
        extra_cuda_cflags=["-O3", "-std=c++20", "--use_fast_math", *(["-lineinfo"] if os.environ.get("M3_LINEINFO") else []), f"-gencode=arch=compute_{arch},code=sm_{arch}"],
        extra_ldflags=["-lcuda"],
        build_directory=build,
        verbose=False,
    )


def build(name: str, source: str):
    """The JIT-built extension, or None (with a warning) if the build fails."""
    try:
        return _load(name, source)
    except Exception as e:  # noqa: BLE001 - build failure falls back to TileLang
        warnings.warn(f"mamba3 sm100 kernel {name} unavailable ({e}); using the TileLang kernels")
        return None


@lru_cache(maxsize=1)
def _kernels_built() -> bool:
    return build("mamba3_scan_tc", "scan_tc.cu") is not None and build("mamba3_bwd_tc", "bwd_tc.cu") is not None


def enabled() -> bool:
    """On sm_100 / sm_103 unless M3_SM100=0, and only if both JIT builds succeed (else: TileLang, with a warning)."""
    return os.environ.get("M3_SM100", "1") != "0" and sm100() and _kernels_built()


def config_ok(q, k, v, H, mimo_o, z, D, angles, chunk_size, rotary_dim_divisor, dtype,
              fuse_pregate_headwise_rms_norm, cu_seqlens, has_initial_state) -> bool:
    return (
        enabled()
        and cu_seqlens is not None and q.shape[0] == 1 and q.dim() == 5 and q.shape[2] == R and q.shape[3] == 1
        and q.shape[4] == N and v.shape[3] == P and angles.shape[-1] == NQ and rotary_dim_divisor == 4
        and mimo_o is not None and z is not None and D is not None and not fuse_pregate_headwise_rms_norm
        and not has_initial_state
        and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
        and z.dtype == torch.bfloat16 and str(dtype).replace("torch.", "") == "bfloat16"
    )


_LAYOUT: dict = {}


def chunk_layout(cu_list: list, ls: int, device):
    """Chunks of at most ``ls`` tokens that never straddle a sequence.

    Returns (NC, meta) with int32 device tensors: start, len, send (end of its sequence), seq_start (start of its
    sequence), last (-1: the forward kernel writes no final K), and per sequence the first chunk / chunk count."""
    key = (tuple(cu_list), ls, str(device))
    hit = _LAYOUT.get(key)
    if hit is not None:
        return hit
    start, ln, send, sst, c0, nc = [], [], [], [], [], []
    for s in range(len(cu_list) - 1):
        a, b = cu_list[s], cu_list[s + 1]
        c0.append(len(start))
        for x in range(a, b, ls):
            start.append(x)
            ln.append(min(ls, b - x))
            send.append(b)
            sst.append(a)
        nc.append(len(start) - c0[-1])
    NC = len(start)
    t = lambda x: torch.tensor(x, dtype=torch.int32).to(device, non_blocking=True)  # noqa: E731
    meta = dict(start=t(start), len=t(ln), send=t(send), sst=t(sst), last=t([-1] * max(NC, 1)),
                seq_c0=t(c0), seq_nc=t(nc), NS=len(c0), max_nc=max(nc, default=0))
    if len(_LAYOUT) > 64:
        _LAYOUT.clear()
    _LAYOUT[key] = (NC, meta)
    return NC, meta
