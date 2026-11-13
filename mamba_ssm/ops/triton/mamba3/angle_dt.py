from typing import Tuple, Optional

import torch
from torch import Tensor

import triton
import triton.language as tl
from mamba_ssm.ops.triton.mamba3.utils import tanh_approx, sech2_approx
from mamba_ssm.ops.triton.mamba3.npu_compat import IS_CANN


# -----------------------------------------------------------------------------
# NPU (Ascend) autotune / UB policy (operator-local; npu_compat.py is shared)
# -----------------------------------------------------------------------------
# The per-program Unified-Buffer footprint of these kernels scales with the
# compile-time tile CHUNK_SIZE * BLOCK_D and does NOT depend on seqlen.  Measured
# on Ascend 910B / Atlas A2 (192 KB = 1572864 bits of per-core UB):
#   * DIM=32, chunk=32  (tile 1024) -> fwd/bwd fit
#   * DIM=32, chunk=64  (tile 2048) -> fwd 1581312 / bwd 2129920 bits: overflow
#   * DIM=64, chunk=32  (tile 2048) -> fwd 1589504 / bwd 2138112 bits: overflow
# so a 32*32 = 1024-element tile is the largest that reliably fits.  Cap the
# chunk on CANN so CHUNK_SIZE*BLOCK_D stays inside that budget: this covers both
# the large-DIM case (DIM=64 -> chunk<=16) and the short-seqlen case (which the
# old seqlen>=64 gate let keep chunk=64 and overflow).  CUDA has no such UB
# pressure and is left untouched.
_UB_TILE_BUDGET = 32 * 32

# Ascend launches a 1D `coreDim` that must fit in UINT16_MAX.  The logical grid
# is `nheads * batch` (varlen: `nheads * num_sequences`) and grows with the
# configuration, so it can exceed the limit for large batch / many sequences.
# Above it we switch to a grid-stride launch: at most `_MAX_COREDIM` programs,
# each looping over several (batch, head) tiles.  Programs stay independent
# (the only recurrence is over chunks *inside* a program), so this is a pure
# scheduling change.  `NUM_PROGRAMS` is passed explicitly instead of using
# ``tl.num_programs(0)`` because Triton-Ascend may fold the logical grid onto
# the physical AI cores, making that value unreliable.
_MAX_COREDIM = 65535


def _grid_programs(num_tiles: int) -> int:
    """Logical program count: all tiles when under UINT16_MAX, else the cap."""
    return min(int(num_tiles), _MAX_COREDIM)


def _npu_chunk_size(chunk_size: int, block_d: int) -> int:
    """NPU-only: largest requested chunk with ``CHUNK_SIZE * BLOCK_D`` <= budget."""
    if not IS_CANN:
        return chunk_size
    cap = max(1, _UB_TILE_BUDGET // int(block_d))
    return min(int(chunk_size), cap)


def _num_warps_choices() -> list:
    """``num_warps`` autotune candidates.

    The audited Triton-Ascend (3.6.0+dev20260921) emits byte-identical
    npubin/ttir/source for num_warps=2/4/8 (same bishengir command line), so on
    CANN the extra warps only inflate autotune rounds, compile time and cache
    entries.  Keep one choice on CANN; CUDA keeps the full tuning space.
    """
    return [4] if IS_CANN else [2, 4, 8]


# -----------------------------------------------------------------------------
# Forward kernel
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in _num_warps_choices()
    ],
    key=["CHUNK_SIZE", "BLOCK_D", "HAS_INIT_STATE", "RETURN_OUTPUT_STATE", "IS_VARLEN"],
)
@triton.jit
def angle_dt_fwd_kernel(
    # Outputs
    OUT, OUTPUT_STATE,
    # Inputs
    ANGLE, DT, INIT_STATE, CU_SEQLENS,
    # Strides for OUT (batch, seqlen, nheads, dim)
    stride_out_batch, stride_out_seq, stride_out_head, stride_out_dim,
    # Strides for OUTPUT_STATE (num_sequences, nheads, dim)
    stride_output_state_seq, stride_output_state_head, stride_output_state_dim,
    # Strides for ANGLE (batch, seqlen, nheads, dim)
    stride_angle_batch, stride_angle_seq, stride_angle_head, stride_angle_dim,
    # Strides for DT (batch, nheads, seqlen)
    stride_dt_batch, stride_dt_head, stride_dt_seq,
    # Strides for INIT_STATE (num_sequences, nheads, dim)
    stride_init_seq, stride_init_head, stride_init_dim,
    # Stride for CU_SEQLENS
    stride_cu_seqlen,
    # Dimensions
    seqlen, dim,
    nheads, num_tiles, num_programs,
    # Meta-parameters
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    RETURN_OUTPUT_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)

    dim_range = tl.arange(0, BLOCK_D)
    dim_mask = dim_range < dim

    PI = 3.141592653589793
    TWO_PI = 2 * PI

    # Grid-stride over independent (batch, head) tiles.  With num_programs ==
    # num_tiles (the common case) each program handles exactly one tile and the
    # mapping is identical to the former 2D/3D grid.
    #
    # The iteration count is per-program (``cdiv``) so the loop uses a constant
    # step of 1: Triton's ``range`` does not accept a runtime step.
    for i in range(0, tl.cdiv(num_tiles - pid, num_programs)):
        tile_idx = pid + i * num_programs
        if IS_VARLEN:
            pid_h = tile_idx % nheads
            pid_b = 0
            seq_idx = tile_idx // nheads
            cu_seqlen_start = tl.load(CU_SEQLENS + seq_idx * stride_cu_seqlen).to(tl.int32)
            cu_seqlen_end = tl.load(CU_SEQLENS + (seq_idx + 1) * stride_cu_seqlen).to(tl.int32)
            seq_len = cu_seqlen_end - cu_seqlen_start
            seq_offset = cu_seqlen_start
        else:
            pid_h = tile_idx % nheads
            pid_b = tile_idx // nheads
            seq_idx = pid_b
            seq_len = seqlen
            seq_offset = 0

        nchunks = tl.cdiv(seq_len, CHUNK_SIZE)

        # Offset base pointers by batch and head
        angle_ptr = ANGLE + pid_b * stride_angle_batch + pid_h * stride_angle_head + seq_offset * stride_angle_seq
        dt_ptr = DT + pid_b * stride_dt_batch + pid_h * stride_dt_head + seq_offset * stride_dt_seq
        out_ptr = OUT + pid_b * stride_out_batch + pid_h * stride_out_head + seq_offset * stride_out_seq

        # Initialize state from init_state or zeros
        if HAS_INIT_STATE:
            init_ptrs = INIT_STATE + seq_idx * stride_init_seq + pid_h * stride_init_head + dim_range * stride_init_dim
            state = tl.load(init_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
        else:
            state = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for chunk_idx in range(nchunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            seq_range = tl.arange(0, CHUNK_SIZE)
            seq_mask = (chunk_start + seq_range) < seq_len

            # Load angle (CHUNK_SIZE, BLOCK_D)
            angle_ptrs = angle_ptr + (chunk_start + seq_range[:, None]) * stride_angle_seq + dim_range[None, :] * stride_angle_dim
            angle_vals = tl.load(angle_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
            angle_vals = tanh_approx(angle_vals) * PI

            # Load dt (CHUNK_SIZE,)
            dt_ptrs = dt_ptr + (chunk_start + seq_range) * stride_dt_seq
            dt_vals = tl.load(dt_ptrs, mask=seq_mask, other=0.0).to(tl.float32)

            # Compute vals = angle * dt
            vals = angle_vals * dt_vals[:, None]

            # Cumsum within chunk + add state from previous chunks
            chunk_cumsum = tl.cumsum(vals, axis=0)
            out_vals = chunk_cumsum + state[None, :]

            # Apply mod 2*pi for rotary angle normalization
            out_vals = out_vals - TWO_PI * tl.floor(out_vals / TWO_PI)

            # Store output
            out_ptrs = out_ptr + (chunk_start + seq_range[:, None]) * stride_out_seq + dim_range[None, :] * stride_out_dim
            tl.store(out_ptrs, out_vals, mask=seq_mask[:, None] & dim_mask[None, :])

            # Update state: add chunk sum and apply mod 2*pi
            chunk_sum = tl.sum(vals, axis=0)
            state = state + chunk_sum
            state = state - TWO_PI * tl.floor(state / TWO_PI)

        # Store final state if requested
        if RETURN_OUTPUT_STATE:
            output_state_ptrs = OUTPUT_STATE + seq_idx * stride_output_state_seq + pid_h * stride_output_state_head + dim_range * stride_output_state_dim
            tl.store(output_state_ptrs, state, mask=dim_mask)


def angle_dt_fwd(
    angle: Tensor,
    dt: Tensor,
    init_state: Optional[Tensor] = None,
    chunk_size: int = 64,
    return_output_state: bool = False,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Forward pass for angle * dt cumsum.

    Args:
        angle: Angle tensor             (batch, seqlen, nheads, dim)
        dt: Time delta tensor           (batch, nheads, seqlen)
        init_state: Initial state       (num_sequences, nheads, dim) or None
        chunk_size: Chunk size for chunked computation
        return_output_state: Whether to return final state
        cu_seqlens: Cumulative sequence lengths (num_sequences + 1,) for varlen mode

    Returns:
        If return_output_state=False:
            out: Cumulative output      (batch, seqlen, nheads, dim)
        If return_output_state=True:
            Tuple of:
                out: Cumulative output      (batch, seqlen, nheads, dim)
                output_state: Final state   (num_sequences, nheads, dim)
    """
    batch, seqlen, nheads, dim = angle.shape
    is_varlen = cu_seqlens is not None
    BLOCK_D = triton.next_power_of_2(dim)
    chunk_size = _npu_chunk_size(chunk_size, BLOCK_D)

    # Determine number of sequences
    if is_varlen:
        assert batch == 1, "Varlen mode requires batch=1"
        num_sequences = cu_seqlens.shape[0] - 1
    else:
        num_sequences = batch

    assert dt.shape == (batch, nheads, seqlen), f"dt shape mismatch: {dt.shape}"
    if init_state is not None:
        assert init_state.shape == (num_sequences, nheads, dim), f"init_state shape mismatch: {init_state.shape}"

    out = torch.empty_like(angle)

    # Handle None init_state for kernel
    HAS_INIT_STATE = init_state is not None
    if not HAS_INIT_STATE:
        init_state = angle  # dummy, won't be accessed
        stride_init = (0, 0, 0)
    else:
        stride_init = init_state.stride()

    # Handle output_state
    if return_output_state:
        output_state = torch.empty(num_sequences, nheads, dim, device=angle.device, dtype=angle.dtype)
        stride_output_state = output_state.stride()
    else:
        output_state = out  # dummy, won't be accessed
        stride_output_state = (0, 0, 0)

    # Handle cu_seqlens
    if cu_seqlens is not None:
        stride_cu_seqlen = cu_seqlens.stride(0)
    else:
        cu_seqlens = angle  # dummy, won't be accessed
        stride_cu_seqlen = 0

    # Grid setup: one flat program per (batch, head) tile, capped at UINT16_MAX.
    # Above the cap each program grid-strides over several tiles (see kernel).
    num_tiles = nheads * (num_sequences if is_varlen else batch)

    # Zero tiles: dense ``batch == 0``, varlen ``num_sequences == 0``, or
    # ``nheads == 0``.  Short-circuit instead of launching ``grid = (0,)``
    # (the backend's coreDim must be >= 1, and the kernel's
    # ``cdiv(num_tiles - pid, num_programs)`` would divide by zero) and instead
    # of asserting: an empty batch / empty sequence list is a legitimate tensor
    # state, whose contract is "no work, input-shaped empty outputs", not an
    # error.  With a valid zero-tile input the preallocated ``out`` and
    # ``output_state`` are already empty (numel == 0); for varlen this means
    # ``cu_seqlens == [0]`` and therefore zero packed tokens.
    if num_tiles == 0:
        if return_output_state:
            return out, output_state
        return out

    num_programs = _grid_programs(num_tiles)
    grid = (num_programs,)

    angle_dt_fwd_kernel[grid](
        out, output_state,
        angle, dt, init_state, cu_seqlens,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        stride_output_state[0], stride_output_state[1], stride_output_state[2],
        angle.stride(0), angle.stride(1), angle.stride(2), angle.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2),
        stride_init[0], stride_init[1], stride_init[2],
        stride_cu_seqlen,
        seqlen, dim,
        nheads, num_tiles, num_programs,
        CHUNK_SIZE=chunk_size,
        BLOCK_D=BLOCK_D,
        HAS_INIT_STATE=HAS_INIT_STATE,
        RETURN_OUTPUT_STATE=return_output_state,
        IS_VARLEN=is_varlen,
    )

    if return_output_state:
        return out, output_state
    return out


# -----------------------------------------------------------------------------
# Backward kernel
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in _num_warps_choices()
    ],
    key=["CHUNK_SIZE", "BLOCK_D", "HAS_INIT_STATE", "HAS_GRAD_OUTPUT_STATE", "IS_VARLEN"],
)
@triton.jit
def angle_dt_bwd_kernel(
    # Outputs
    GRAD_ANGLE, GRAD_DT, GRAD_INIT_STATE,
    # Inputs
    GRAD_OUT, GRAD_OUTPUT_STATE, ANGLE, DT, CU_SEQLENS,
    # Strides for GRAD_ANGLE (batch, seqlen, nheads, dim)
    stride_grad_angle_batch, stride_grad_angle_seq, stride_grad_angle_head, stride_grad_angle_dim,
    # Strides for GRAD_DT (batch, nheads, seqlen)
    stride_grad_dt_batch, stride_grad_dt_head, stride_grad_dt_seq,
    # Strides for GRAD_INIT_STATE (num_sequences, nheads, dim)
    stride_grad_init_seq, stride_grad_init_head, stride_grad_init_dim,
    # Strides for GRAD_OUT (batch, seqlen, nheads, dim)
    stride_grad_out_batch, stride_grad_out_seq, stride_grad_out_head, stride_grad_out_dim,
    # Strides for GRAD_OUTPUT_STATE (num_sequences, nheads, dim)
    stride_grad_output_state_seq, stride_grad_output_state_head, stride_grad_output_state_dim,
    # Strides for ANGLE (batch, seqlen, nheads, dim)
    stride_angle_batch, stride_angle_seq, stride_angle_head, stride_angle_dim,
    # Strides for DT (batch, nheads, seqlen)
    stride_dt_batch, stride_dt_head, stride_dt_seq,
    # Stride for CU_SEQLENS
    stride_cu_seqlen,
    # Dimensions
    seqlen, dim,
    nheads, num_tiles, num_programs,
    # Meta-parameters
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    HAS_GRAD_OUTPUT_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)

    dim_range = tl.arange(0, BLOCK_D)
    dim_mask = dim_range < dim
    PI = 3.141592653589793

    # Grid-stride over independent (batch, head) tiles (see the forward kernel).
    for i in range(0, tl.cdiv(num_tiles - pid, num_programs)):
        tile_idx = pid + i * num_programs
        if IS_VARLEN:
            pid_h = tile_idx % nheads
            pid_b = 0
            seq_idx = tile_idx // nheads
            cu_seqlen_start = tl.load(CU_SEQLENS + seq_idx * stride_cu_seqlen).to(tl.int32)
            cu_seqlen_end = tl.load(CU_SEQLENS + (seq_idx + 1) * stride_cu_seqlen).to(tl.int32)
            seq_len = cu_seqlen_end - cu_seqlen_start
            seq_offset = cu_seqlen_start
        else:
            pid_h = tile_idx % nheads
            pid_b = tile_idx // nheads
            seq_idx = pid_b
            seq_len = seqlen
            seq_offset = 0

        nchunks = tl.cdiv(seq_len, CHUNK_SIZE)

        # Offset base pointers by batch and head
        grad_angle_ptr = GRAD_ANGLE + pid_b * stride_grad_angle_batch + pid_h * stride_grad_angle_head + seq_offset * stride_grad_angle_seq
        grad_dt_ptr = GRAD_DT + pid_b * stride_grad_dt_batch + pid_h * stride_grad_dt_head + seq_offset * stride_grad_dt_seq
        grad_out_ptr = GRAD_OUT + pid_b * stride_grad_out_batch + pid_h * stride_grad_out_head + seq_offset * stride_grad_out_seq
        angle_ptr = ANGLE + pid_b * stride_angle_batch + pid_h * stride_angle_head + seq_offset * stride_angle_seq
        dt_ptr = DT + pid_b * stride_dt_batch + pid_h * stride_dt_head + seq_offset * stride_dt_seq

        # Initialize gradient state from grad_output_state or zeros
        if HAS_GRAD_OUTPUT_STATE:
            grad_output_state_ptrs = GRAD_OUTPUT_STATE + seq_idx * stride_grad_output_state_seq + pid_h * stride_grad_output_state_head + dim_range * stride_grad_output_state_dim
            grad_state = tl.load(grad_output_state_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
        else:
            grad_state = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # Loop in reverse: derivative of cumsum is reverse cumsum
        for chunk_idx in range(nchunks - 1, -1, -1):
            chunk_start = chunk_idx * CHUNK_SIZE
            seq_range = tl.arange(0, CHUNK_SIZE)
            seq_mask = (chunk_start + seq_range) < seq_len

            # Load grad_out (CHUNK_SIZE, BLOCK_D)
            grad_out_ptrs = grad_out_ptr + (chunk_start + seq_range[:, None]) * stride_grad_out_seq + dim_range[None, :] * stride_grad_out_dim
            grad_out_vals = tl.load(grad_out_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)

            # Reverse cumsum within chunk: rev_cumsum = total - cumsum + x
            # But we need to handle the mask properly for partial chunks
            chunk_sum = tl.sum(grad_out_vals, axis=0)
            fwd_cumsum = tl.cumsum(grad_out_vals, axis=0)
            rev_cumsum = chunk_sum[None, :] - fwd_cumsum + grad_out_vals

            # Add gradient from future chunks
            grad_vals = rev_cumsum + grad_state[None, :]

            # Load angle and dt
            angle_ptrs = angle_ptr + (chunk_start + seq_range[:, None]) * stride_angle_seq + dim_range[None, :] * stride_angle_dim
            pretanh_angle_vals = tl.load(angle_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
            angle_vals = tanh_approx(pretanh_angle_vals) * PI

            dt_ptrs = dt_ptr + (chunk_start + seq_range) * stride_dt_seq
            dt_vals = tl.load(dt_ptrs, mask=seq_mask, other=0.0).to(tl.float32)

            # Compute gradients: out = angle * dt
            grad_angle_vals = grad_vals * dt_vals[:, None] * PI * sech2_approx(pretanh_angle_vals)
            grad_dt_vals = tl.sum(grad_vals * angle_vals, axis=1)

            # Store gradients
            grad_angle_ptrs = grad_angle_ptr + (chunk_start + seq_range[:, None]) * stride_grad_angle_seq + dim_range[None, :] * stride_grad_angle_dim
            tl.store(grad_angle_ptrs, grad_angle_vals, mask=seq_mask[:, None] & dim_mask[None, :])

            grad_dt_ptrs = grad_dt_ptr + (chunk_start + seq_range) * stride_grad_dt_seq
            tl.store(grad_dt_ptrs, grad_dt_vals, mask=seq_mask)

            # Update state for previous chunk
            grad_state = grad_state + chunk_sum

        # Store gradient for init_state if provided
        if HAS_INIT_STATE:
            grad_init_ptrs = GRAD_INIT_STATE + seq_idx * stride_grad_init_seq + pid_h * stride_grad_init_head + dim_range * stride_grad_init_dim
            tl.store(grad_init_ptrs, grad_state, mask=dim_mask)


def angle_dt_bwd(
    grad_out: Tensor,
    angle: Tensor,
    dt: Tensor,
    has_init_state: bool = False,
    chunk_size: int = 64,
    grad_output_state: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Backward pass for angle * dt cumsum.

    Args:
        grad_out: Gradient of output         (batch, seqlen, nheads, dim)
        angle: Angle tensor                  (batch, seqlen, nheads, dim)
        dt: Time delta tensor                (batch, nheads, seqlen)
        has_init_state: Whether init_state was provided in forward
        chunk_size: Chunk size for chunked computation
        grad_output_state: Gradient of output state (num_sequences, nheads, dim) or None
        cu_seqlens: Cumulative sequence lengths (num_sequences + 1,) for varlen mode

    Returns:
        grad_angle: Gradient for angle       (batch, seqlen, nheads, dim)
        grad_dt: Gradient for dt             (batch, nheads, seqlen)
        grad_init_state: Gradient for init_state (num_sequences, nheads, dim) or None
    """
    batch, seqlen, nheads, dim = angle.shape
    is_varlen = cu_seqlens is not None
    BLOCK_D = triton.next_power_of_2(dim)
    chunk_size = _npu_chunk_size(chunk_size, BLOCK_D)

    # Determine number of sequences
    if is_varlen:
        assert batch == 1, "Varlen mode requires batch=1"
        num_sequences = cu_seqlens.shape[0] - 1
    else:
        num_sequences = batch
    
    grad_angle = torch.empty_like(angle)
    grad_dt = torch.empty_like(dt)

    # Handle init_state gradient
    if has_init_state:
        grad_init_state = torch.empty(num_sequences, nheads, dim, device=angle.device, dtype=torch.float32)
        stride_grad_init = grad_init_state.stride()
    else:
        grad_init_state = None
        stride_grad_init = (0, 0, 0)
        grad_init_dummy = grad_angle  # dummy pointer

    # Handle grad_output_state
    HAS_GRAD_OUTPUT_STATE = grad_output_state is not None
    if not HAS_GRAD_OUTPUT_STATE:
        grad_output_state = grad_angle  # dummy, won't be accessed
        stride_grad_output_state = (0, 0, 0)
    else:
        stride_grad_output_state = grad_output_state.stride()

    # Handle cu_seqlens
    if cu_seqlens is not None:
        stride_cu_seqlen = cu_seqlens.stride(0)
    else:
        cu_seqlens = angle  # dummy, won't be accessed
        stride_cu_seqlen = 0

    # Grid setup: same flat, capped tile grid as the forward pass.
    num_tiles = nheads * (num_sequences if is_varlen else batch)

    # Same zero-tile contract as the forward pass: dense ``batch == 0``,
    # varlen ``num_sequences == 0``, or ``nheads == 0`` returns the already
    # allocated, input-shaped (empty) gradients without launching.
    if num_tiles == 0:
        return grad_angle, grad_dt, grad_init_state

    num_programs = _grid_programs(num_tiles)
    grid = (num_programs,)

    angle_dt_bwd_kernel[grid](
        grad_angle, grad_dt, grad_init_state if has_init_state else grad_init_dummy,
        grad_out, grad_output_state, angle, dt, cu_seqlens,
        grad_angle.stride(0), grad_angle.stride(1), grad_angle.stride(2), grad_angle.stride(3),
        grad_dt.stride(0), grad_dt.stride(1), grad_dt.stride(2),
        stride_grad_init[0], stride_grad_init[1], stride_grad_init[2],
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
        stride_grad_output_state[0], stride_grad_output_state[1], stride_grad_output_state[2],
        angle.stride(0), angle.stride(1), angle.stride(2), angle.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2),
        stride_cu_seqlen,
        seqlen, dim,
        nheads, num_tiles, num_programs,
        CHUNK_SIZE=chunk_size,
        BLOCK_D=BLOCK_D,
        HAS_INIT_STATE=has_init_state,
        HAS_GRAD_OUTPUT_STATE=HAS_GRAD_OUTPUT_STATE,
        IS_VARLEN=is_varlen,
    )
    return grad_angle, grad_dt, grad_init_state