"""Ascend (Triton-Ascend) backend compatibility helpers for Mamba-3 kernels.

The upstream Mamba-3 Triton kernels are written for NVIDIA GPUs.  This module
keeps every backend-specific decision in one place so the same source can run on
CUDA and on Ascend NPUs (via Triton-Ascend) without changing the CUDA code path.

Two layers of adaptation live here:

1. **Syntax layer** -- CANN libdevice instead of NVIDIA PTX inline asm, device
   string for the scratch allocator, CUDA-only ``maxnreg`` removal.
2. **Architecture layer** -- Unified-Buffer (UB) budget policy: the Ascend
   per-core UB is small (192 KB on 910B / Atlas A2), the chunked SISO kernels
   materialise O(CHUNK_SIZE^2) fp32 gate matrices plus O(CHUNK_SIZE*HEADDIM)
   tiles and a persistent HEADDIM_V*HEADDIM_QK fp32 recurrent state, and
   multi-buffering duplicates the tile buffers per pipeline stage.  The helpers
   below bound the tile/state footprint so a single program stays inside UB:

   * ``effective_chunk_size()`` -- clamp the chunk size on CANN;
   * ``stage_choices()``        -- keep the verbatim upstream ``num_stages`` space;
   * ``chunk_choices()``        -- drop UB-unsafe chunk sizes on CANN.

   Only collapses backed by a measured ``ub overflow`` (or a hard compile
   failure) are kept; an axis with no such evidence keeps its upstream extent.
   See ``NPU_MIGRATION_mamba3_siso.md`` §14 for the per-axis ledger.

Public operator signatures and return structures are never changed by this
module; only blocking / config selection differs between backends.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from typing import List, Optional

import triton
import torch

try:  # Triton-Ascend exposes CANN libdevice under this path.
    from triton.language.extra.cann import libdevice as libdevice  # noqa: F401

    IS_CANN = True
except Exception:  # CUDA / ROCm / plain-Triton builds do not ship CANN libdevice
    libdevice = None
    IS_CANN = False


# ---------------------------------------------------------------------------
# Device / launch helpers shared by the gated (RMSNormGated) and non-gated
# (LayerNorm/RMSNorm) kernels.  Keeping one implementation avoids the two
# paths drifting apart.  They live here because this module is the existing
# backend-compat home shared by both chains; the dependency stays one-way
# (``layer_norm.py`` / ``layernorm_gated.py`` import this module, and this
# module imports no ``mamba_ssm`` code), so the edge is safe on CUDA and NPU.
# ---------------------------------------------------------------------------


def _device_ctx(device):
    """Device-scoping context for a Triton launch.

    CUDA needs the ``torch.cuda.device(index)`` context; Ascend NPU (and plain
    CPU) have no such notion, so use a no-op context there.  This keeps the CUDA
    behaviour byte-for-byte identical while letting the same kernels be
    dispatched from the Mamba-3 SISO chain as well as from the generic
    LayerNorm/RMSNorm path on NPU.
    """
    if getattr(device, "type", None) == "cuda":
        return torch.cuda.device(device.index)
    return contextlib.nullcontext()


# This AIV-specific value controls the _dw shard count and therefore the
# summation order of its fp32 partial sums (the backward kernel uses stores,
# not atomic additions).  ``MAMBA3_SM_COUNT`` is the single override knob for
# both the gated (RMSNormGated) and the non-gated (LayerNorm/RMSNorm) backward
# grids; it is named after the Mamba-3 work but applies to both.
@functools.lru_cache(maxsize=None)
def _npu_aiv_block_limit(device, env_override):
    """Return the cached AIV core count used to size the backward grid.

    Shared by the gated and the non-gated (``layer_norm.py``) backward grids, so
    ``MAMBA3_SM_COUNT`` affects both.
    """
    if env_override is not None:
        try:
            count = int(env_override)
            if count > 0:
                logging.getLogger(__name__).info(
                    "AIV block limit: device=%s value=%d source=MAMBA3_SM_COUNT",
                    device,
                    count,
                )
                return count
        except ValueError:
            pass
        logging.getLogger(__name__).warning(
            "Ignoring invalid MAMBA3_SM_COUNT=%r", env_override
        )

    try:
        properties = torch.npu.get_device_properties(torch.device(device))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        properties = None

    if properties is not None:
        for name in ("vector_core_num", "multi_processor_count"):
            try:
                count = int(getattr(properties, name, 0))
            except (TypeError, ValueError):
                continue
            if count > 0:
                logging.getLogger(__name__).info(
                    "AIV block limit: device=%s value=%d source=%s",
                    device,
                    count,
                    name,
                )
                return count

    logging.getLogger(__name__).info(
        "AIV block limit: device=%s value=40 source=fallback", device
    )
    return 40


def _sm_count(device):
    """AIV core count for the backward grid heuristic (CUDA/NPU/CPU)."""
    if getattr(device, "type", None) == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    if getattr(device, "type", None) == "npu":
        return _npu_aiv_block_limit(str(device), os.getenv("MAMBA3_SM_COUNT"))
    # Keep CPU-only callers valid; NPU properties are unavailable on CPU.
    return 40


# Per-core Unified Buffer on Ascend 910B / Atlas A2: 192 KB.
UB_BYTES = 192 * 1024

# Largest chunk size that keeps every SISO kernel's peak UB inside ``UB_BYTES``.
# At large sequences the chunk is halved (NPU_LARGE_SEQ_CHUNK_SIZE): the backward
# dqkv kernel at CHUNK_SIZE=64 / HEADDIM_QK=128 was 1.256x over UB.  Short
# sequences keep the original chunk -- a tiny chunk with seqlen=1 trips a
# Triton-Ascend GraphOptimize failure, and there is no UB pressure at short
# lengths.  The chunked maths is invariant to the chunk size (chunking only
# reorders the same sums), so this only changes blocking, not results.
NPU_MAX_CHUNK_SIZE = 64
NPU_LARGE_SEQ_THRESHOLD = 64
NPU_LARGE_SEQ_CHUNK_SIZE = 32


def effective_chunk_size(chunk_size: int, seqlen: Optional[int] = None) -> int:
    """Clamp the chunk size to a UB-safe value on Ascend NPU.

    Applied consistently in every wrapper that participates in the chunked
    forward/backward (fwd / angle_dt / bwd), so the per-chunk state layout
    (``SSM_States`` columns, ``DA_CS_SUM`` length) computed on the forward side
    matches the backward side.

    For large sequences the chunk is additionally halved to bound the per-tile UB
    footprint; short sequences keep the original chunk.
    """
    if IS_CANN:
        cs = min(int(chunk_size), NPU_MAX_CHUNK_SIZE)
        if seqlen is not None and seqlen >= NPU_LARGE_SEQ_THRESHOLD:
            cs = min(cs, NPU_LARGE_SEQ_CHUNK_SIZE)
        return cs
    return chunk_size


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean env override; fall back to ``default`` when unset/garbage."""
    val = os.environ.get(name)
    if val is None:
        return default
    val = val.strip().lower()
    if val in ("0", "false", "no", "off", ""):
        return False
    if val in ("1", "true", "yes", "on"):
        return True
    return default


def precise_dqkv() -> bool:
    """PLUT-50: full-fp32 precision path for the ``compute_dqkv`` backward.

    When enabled:

    * the **second** ``tl.dot`` operands and the state-gradient accumulator in
      ``mamba3_siso_bwd_kernel_dqkv`` are computed in fp32 (the kernel otherwise
      rounds each second-dot operand and the fp32 accumulator to the input dtype
      -- ``mamba3_siso_bwd.py:463/466/490/517/520/575/593``), and
    * the forward stores per-chunk ``SSM_States`` in **fp32** instead of bf16.

    Both are required together: the CPU root-cause probe
    (``tests/ops/triton/repro_pl50_dqkv_fp32.py``) shows the in-kernel fix alone
    lifts ``dK_mid``/``dV`` to ``matched_ratio=1.0`` but leaves
    ``dQ_mid``/``dADT`` < 0.99 at the CANN fp32 criterion, because those outputs
    read the bf16-stored per-chunk states as an input.

    Defaults to ``IS_CANN`` so the CUDA/ROCm numeric path is unchanged.  Override
    with ``MAMBA3_SISO_PRECISE_DQKV=0/1`` to A/B the precision and cost on NPU.
    """
    return _env_flag("MAMBA3_SISO_PRECISE_DQKV", IS_CANN)


def precise_ddt() -> bool:
    """PLUT-51: fp32 Part-1 path for ``compute_ddt_dtrap_dinput_states``.

    When enabled, the Part-1 cotangent operands (``dScale``, ``dGamma``, ``DT``
    and the shifted ``dScale_{t-1}``) are widened to fp32 *before* the
    arithmetic.  Otherwise ``dgamma_t + dscale_t`` (and the ``* dt_t`` products
    in the dTrap path) are rounded in the **input** dtype and only later promoted
    by the fp32 ``sigmoid(trap)`` term, which for fp16 loses ~1e-3 before the
    fp32 output is written.

    The fp64 CPU probe ``tests/ops/triton/repro_pl51_ddt_fp16.py`` reproduces the
    fp16 failure (dDT ``matched_ratio`` 0.858 pre-fix) and shows the up-cast lifts
    every output to ``matched_ratio=1.0``.  The cast is a no-op on the fp32 real
    chain (nothing in the combined backward feeds fp16/bf16 here), so the default
    is ``IS_CANN``; override with ``MAMBA3_SISO_PRECISE_DDT=0/1`` to A/B the
    precision on NPU.
    """
    return _env_flag("MAMBA3_SISO_PRECISE_DDT", IS_CANN)


def alloc_device() -> str:
    """Device string for Triton's runtime scratch/descriptor allocator."""
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return "npu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def stage_choices(choices: List[int]) -> List[int]:
    """``num_stages`` candidates: the verbatim upstream space, on every backend.

    The first NPU migration collapsed this axis to ``[1]`` (single stage, no
    multi-buffering) on the *hypothesis* that pipeline stages multiply the
    per-tile UB footprint.  That hypothesis has no measurement behind it: every
    ``ub overflow`` actually observed on this backend (fwd 1581312 / angle_dt
    bwd 2129920 / bwd dqkv 1974528 bits, ``NPU_MIGRATION_mamba3_siso.md`` §6.1,
    §9.1, §13) was driven by the tile / ``CHUNK_SIZE`` / ``BLOCK_D`` size, never
    by ``num_stages``.

    Under the "default == upstream candidate space; collapse only with evidence"
    rule the axis is therefore restored to the full upstream extent.  ``choices``
    is the upstream list (``[1, 2, 3]``); callers must keep passing it verbatim
    so the restored space stays auditable.  If a future NPU run shows a concrete
    ``num_stages>1`` config overflowing or failing to compile, re-add the
    collapse *here*, quoting that error text and the minimal affected range.
    """
    return list(choices)


def chunk_choices(choices: List[int], cap: int = NPU_MAX_CHUNK_SIZE) -> List[int]:
    """``CHUNK_SIZE`` autotune candidates, capped to the UB-safe range on CANN."""
    if IS_CANN:
        kept = [c for c in choices if c <= cap]
        return kept or [cap]
    return list(choices)


def make_config(
    extra: dict,
    num_stages: int,
    num_warps: int,
    maxnreg: Optional[int] = None,
) -> "triton.Config":
    """Build a ``triton.Config``.

    ``maxnreg`` is a CUDA-only knob (register caps do not exist on NPU) and is
    therefore dropped on the CANN backend.
    """
    if IS_CANN or maxnreg is None:
        return triton.Config(extra, num_stages=num_stages, num_warps=num_warps)
    return triton.Config(
        extra, num_stages=num_stages, num_warps=num_warps, maxnreg=maxnreg
    )


def autotune_regs(candidates: List[Optional[int]]) -> List[Optional[int]]:
    """``maxnreg`` candidates; on CANN all collapse to a single ``None`` entry."""
    if IS_CANN:
        return [None]
    return candidates


def set_allocator_safely(fn) -> None:
    """Install a Triton scratch allocator, ignoring backends without support."""
    if not hasattr(triton, "set_allocator"):
        return
    try:
        triton.set_allocator(fn)
    except Exception:
        pass
