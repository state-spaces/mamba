# Copyright (c) 2025, Tri Dao.

from typing import Optional
import math

import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice

class AngleDtFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                angle: torch.Tensor,   # (B, S, H, D)
                dt: torch.Tensor,      # (B, S, H)
                chunk_size: int = 128  # power of 2
                ) -> torch.Tensor:
        # run Triton fwd
        out = apply_angle_dt_fwd(angle, dt, chunk_size=chunk_size)
        # save for bwd
        ctx.save_for_backward(angle, dt)
        ctx.chunk_size = int(chunk_size)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        angle, dt = ctx.saved_tensors
        # run Triton bwd
        grad_dt, grad_angle = apply_angle_dt_bwd(
            grad_out=grad_out, angle=angle, dt=dt, chunk_size=ctx.chunk_size
        )
        # grads align with (angle, dt, chunk_size)
        return grad_angle, grad_dt, None


def angle_dt(angle: torch.Tensor,
             dt: torch.Tensor,
             *,
             chunk_size: int = 128) -> torch.Tensor:
    return AngleDtFn.apply(angle, dt, chunk_size)


@triton.jit
def cumsum_kernel(
    OUT,        # Output tensor (batch, seqlen, nheads, dim)
    X,          # Input tensor (batch, seqlen, nheads, dim)
    seqlen,
    dim,
    stride_out, # (batch, seqlen, nheads, dim)
    stride_x,   # (batch, seqlen, nheads, dim)
    # Meta-parameters
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Program IDs
    pid_h = tl.program_id(axis=0)  # Head index (one per head)
    pid_d = tl.program_id(axis=1)  # Dim block
    pid_b = tl.program_id(axis=2)  # Batch index (one per batch element)

    # Offset pointers by batch and head
    X = X + pid_b * stride_x[0] + pid_h * stride_x[2]
    OUT = OUT + pid_b * stride_out[0] + pid_h * stride_out[2]

    # Compute ranges
    dim_range = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_range < dim

    # Load entire sequence for this batch, head, and dim block
    seq_range = tl.arange(0, BLOCK_S)[:, None]  # (BLOCK_S, 1)

    # Load input: (seqlen, dim) for this batch and head
    x_ptrs = X + seq_range * stride_x[1] + dim_range[None, :] * stride_x[3]
    x_mask = (seq_range < seqlen) & dim_mask[None, :]
    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

    # Compute cumulative sum along sequence dimension (axis 0)
    cumsum_vals = tl.cumsum(x_vals, axis=0)

    # Store output: (seqlen, dim) for this batch and head
    out_ptrs = OUT + seq_range * stride_out[1] + dim_range[None, :] * stride_out[3]
    out_mask = (seq_range < seqlen) & dim_mask[None, :]
    tl.store(out_ptrs, cumsum_vals, mask=out_mask)


@triton.jit
def angle_dt_fwd_kernel(
    OUT,        # Output tensor (batch, seqlen, nheads, dim)
    OUT_SUM,    # Output sum tensor (batch, seqlen // chunk_size, nheads, dim)
    ANGLE,      # Angle tensor (batch, seqlen, nheads, dim)
    DT,         # Delta time tensor (batch, seqlen, nheads)
    PREFIX,     # Prefix tensor (batch, numchunks, nheads, dim) - optional
    seqlen,
    dim,
    chunk_size,
    stride_out,     # (batch, seqlen, nheads, dim)
    stride_out_sum, # (batch, seqlen // chunk_size, nheads, dim)
    stride_angle,   # (batch, seqlen, nheads, dim)
    stride_dt,      # (batch, seqlen, nheads)
    stride_prefix,  # (batch, numchunks, nheads, dim)
    # Meta-parameters
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WRITE_OUTPUT: tl.constexpr,  # Whether to write the full output
    WRITE_CHUNK_SUM: tl.constexpr,  # Whether to write the chunk sum
    HAS_PREFIX: tl.constexpr,  # Whether prefix is provided
):
    # Program IDs
    pid_b = tl.program_id(axis=2)  # Batch index (one per batch element)
    pid_s = tl.program_id(axis=1)  # Sequence block (chunk index)
    pid_h = tl.program_id(axis=0)  # Head index (one per head)

    # Offset pointers by batch and head
    ANGLE = ANGLE + pid_b * stride_angle[0] + pid_h * stride_angle[2]
    DT = DT + pid_b * stride_dt[0] + pid_h * stride_dt[2]
    if WRITE_OUTPUT:
        OUT = OUT + pid_b * stride_out[0] + pid_h * stride_out[2]
    if WRITE_CHUNK_SUM:
        OUT_SUM = OUT_SUM + pid_b * stride_out_sum[0] + pid_h * stride_out_sum[2]
    if HAS_PREFIX:
        PREFIX = PREFIX + pid_b * stride_prefix[0] + pid_h * stride_prefix[2]

    # Compute ranges - each block processes exactly chunk_size elements
    seq_start = pid_s * chunk_size
    seq_range = seq_start + tl.arange(0, BLOCK_S)
    dim_range = tl.arange(0, BLOCK_D)

    # Masks
    seq_mask = seq_range < seqlen
    dim_mask = dim_range < dim

    # Load angle: (seqlen, dim) for this batch and head
    angle_ptrs = ANGLE + seq_range[:, None] * stride_angle[1] + dim_range[None, :] * stride_angle[3]
    angle_mask = (seq_mask[:, None] & dim_mask[None, :])
    angle_vals = tl.load(angle_ptrs, mask=angle_mask, other=0.0).to(tl.float32)

    # Load dt: (seqlen,) for this batch and head
    dt_ptrs = DT + seq_range * stride_dt[1]
    dt_mask = seq_mask
    dt_vals = tl.load(dt_ptrs, mask=dt_mask, other=0.0).to(tl.float32)

    # Multiply: angle (S, D) * dt (S, 1) -> output (S, D)
    # angle_vals: (BLOCK_S, BLOCK_D)
    # dt_vals: (BLOCK_S,)
    #output_vals = angle_vals * dt_vals[:, None]  # (BLOCK_S, BLOCK_D)
    output_vals = tl.sigmoid(2.0 * angle_vals) * 2.0 - 1.0
    # output_vals = libdevice.tanh(output_vals)  # This is pretty slow
    # This is still not super fast, idk how to enable fastmath
    #output_vals = tl.sigmoid(2.0 * output_vals) * 2.0 - 1.0
    output_vals = output_vals * dt_vals[:, None]
    # This is the fastest, but with reduced accuracy. We probably don't need it
    # output_vals = tl.inline_asm_elementwise(
    #     "tanh.approx.f32 $0, $1;",
    #     "=f,f",
    #     [output_vals],
    #     dtype=tl.float32,
    #     is_pure=True,
    #     pack=1,
    # )
    output_vals *= 3.141592653589793  # pi

    # Conditionally compute and store chunk sum
    if WRITE_CHUNK_SUM:
        # Compute sum along sequence dimension (within this chunk)
        # Sum over the sequence dimension (axis 0)
        chunk_sum = tl.sum(output_vals, axis=0)  # (BLOCK_D,)
        # Store chunk sum: (seqlen // chunk_size, dim) for this batch and head
        sum_ptrs = OUT_SUM + pid_s * stride_out_sum[1] + dim_range * stride_out_sum[3]
        sum_mask = dim_mask
        tl.store(sum_ptrs, chunk_sum, mask=sum_mask)

    # Conditionally store output: (seqlen, dim) for this batch and head
    if WRITE_OUTPUT:
        output_vals = tl.cumsum(output_vals, axis=0)  # Cumulative sum along sequence dimension (axis 0)
        # Add prefix if provided
        if HAS_PREFIX:
            # If chunk idx is 0, prefix is 0. If chunk idx is i, read from prefix at location i-1
            if pid_s > 0:
                # Load prefix for this chunk from location pid_s - 1
                prefix_ptrs = PREFIX + (pid_s - 1) * stride_prefix[1] + dim_range * stride_prefix[3]
                prefix_mask = dim_mask
                prefix_vals = tl.load(prefix_ptrs, mask=prefix_mask, other=0.0).to(tl.float32)
                # Add prefix to all elements in this chunk
                output_vals = output_vals + prefix_vals[None, :]  # Broadcast prefix across sequence dimension
            # For pid_s == 0, prefix is implicitly 0, so no addition needed
        out_ptrs = OUT + seq_range[:, None] * stride_out[1] + dim_range[None, :] * stride_out[3]
        out_mask = (seq_mask[:, None] & dim_mask[None, :])
        tl.store(out_ptrs, output_vals, mask=out_mask)


# The kernel expects inputs to be flipped in the sequence dimension.
# This is because it processes chunks in reverse order.
@triton.jit
def angle_dt_bwd_kernel(
    GRAD_DT,      # Grad dt tensor (batch, seqlen, nheads)
    GRAD_ANGLE,   # Grad angle tensor (batch, seqlen, nheads, dim)
    GRAD_SUM,     # Grad sum tensor (batch, seqlen // chunk_size, nheads, dim)
    GRAD_OUT,     # Grad input tensor (batch, seqlen, nheads, dim)
    ANGLE,        # Angle tensor (batch, seqlen, nheads, dim)
    DT,           # Delta time tensor (batch, seqlen, nheads)
    PREFIX,       # Prefix tensor (batch, numchunks, nheads, dim) - optional
    seqlen,
    dim,
    chunk_size,
    stride_grad_dt,     # (batch, seqlen, nheads)
    stride_grad_angle,  # (batch, seqlen, nheads, dim)
    stride_grad_sum,    # (batch, seqlen // chunk_size, nheads, dim)
    stride_grad_out,    # (batch, seqlen, nheads, dim)
    stride_angle,       # (batch, seqlen, nheads, dim)
    stride_dt,          # (batch, seqlen, nheads)
    stride_prefix,      # (batch, numchunks, nheads, dim)
    # Meta-parameters
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WRITE_GRAD: tl.constexpr,       # Whether to write the full output
    WRITE_CHUNK_SUM: tl.constexpr,  # Whether to write the chunk sum
    HAS_PREFIX: tl.constexpr,       # Whether prefix is provided
):
    # Program IDs
    pid_b = tl.program_id(axis=2)  # Batch index (one per batch element)
    pid_s = tl.program_id(axis=1)  # Sequence block (chunk index)
    pid_h = tl.program_id(axis=0)  # Head index (one per head)

    # Offset pointers by batch and head
    GRAD_OUT = GRAD_OUT + pid_b * stride_grad_out[0] + pid_h * stride_grad_out[2]
    if WRITE_GRAD:
        GRAD_DT = GRAD_DT + pid_b * stride_grad_dt[0] + pid_h * stride_grad_dt[2]
        GRAD_ANGLE = GRAD_ANGLE + pid_b * stride_grad_angle[0] + pid_h * stride_grad_angle[2]
        DT = DT + pid_b * stride_dt[0] + pid_h * stride_dt[2]
        ANGLE = ANGLE + pid_b * stride_angle[0] + pid_h * stride_angle[2]
    if WRITE_CHUNK_SUM:
        GRAD_SUM = GRAD_SUM + pid_b * stride_grad_sum[0] + pid_h * stride_grad_sum[2]
    if HAS_PREFIX:
        PREFIX = PREFIX + pid_b * stride_prefix[0] + pid_h * stride_prefix[2]

    # Compute ranges - each block processes exactly chunk_size elements
    seq_start = pid_s * chunk_size
    seq_range = seq_start + tl.arange(0, BLOCK_S)
    dim_range = tl.arange(0, BLOCK_D)

    # Masks
    seq_mask = seq_range < seqlen
    dim_mask = dim_range < dim

    # Load angle: (seqlen, dim) for this batch and head
    grad_out_ptrs = GRAD_OUT + seq_range[:, None] * stride_grad_out[1] + dim_range[None, :] * stride_grad_out[3]
    grad_out_mask = (seq_mask[:, None] & dim_mask[None, :])
    grad_out_vals = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0).to(tl.float32)

    # Conditionally compute and store chunk sum
    if WRITE_CHUNK_SUM:
        # Compute sum along sequence dimension (within this chunk)
        # Sum over the sequence dimension (axis 0)
        chunk_sum = tl.sum(grad_out_vals, axis=0)  # (BLOCK_D,)
        # Store chunk sum: (seqlen // chunk_size, dim) for this batch and head
        sum_ptrs = GRAD_SUM + pid_s * stride_grad_sum[1] + dim_range * stride_grad_sum[3]
        sum_mask = dim_mask
        tl.store(sum_ptrs, chunk_sum, mask=sum_mask)

    # Conditionally store output: (seqlen, dim) for this batch and head
    if WRITE_GRAD:
        grad_out_vals = tl.cumsum(grad_out_vals, axis=0)  # Cumulative sum along sequence dimension (axis 0)

        # Add prefix if provided
        if HAS_PREFIX:
            # If chunk idx is 0, prefix is 0. If chunk idx is i, read from prefix at location i-1
            if pid_s > 0:
                # Load prefix for this chunk from location pid_s - 1
                prefix_ptrs = PREFIX + (pid_s - 1) * stride_prefix[1] + dim_range * stride_prefix[3]
                prefix_mask = dim_mask
                prefix_vals = tl.load(prefix_ptrs, mask=prefix_mask, other=0.0).to(tl.float32)
                # Add prefix to all elements in this chunk
                grad_out_vals = grad_out_vals + prefix_vals[None, :]  # Broadcast prefix across sequence dimension
            # For pid_s == 0, prefix is implicitly 0, so no addition needed

        # Load angle: (seqlen, dim) for this batch and head
        angle_ptrs = ANGLE + seq_range[:, None] * stride_angle[1] + dim_range[None, :] * stride_angle[3]
        angle_mask = (seq_mask[:, None] & dim_mask[None, :])
        angle_vals = tl.load(angle_ptrs, mask=angle_mask, other=0.0).to(tl.float32)

        # Load dt: (seqlen,) for this batch and head
        dt_ptrs = DT + seq_range * stride_dt[1]
        dt_mask = seq_mask
        dt_vals = tl.load(dt_ptrs, mask=dt_mask, other=0.0).to(tl.float32)  # (BLOCK_S,)

        # Compute dt gradients
        tanh_angle_vals = tl.sigmoid(2.0 * angle_vals) * 2.0 - 1.0  # (BLOCK_S, BLOCK_D)
        pi_tanh_angle_vals = tanh_angle_vals*3.141592653589793
        dt_grad_vals = grad_out_vals * pi_tanh_angle_vals # (BLOCK_S, BLOCK_D)
        dt_grad_vals = tl.sum(dt_grad_vals, axis=1)  # Sum over dim to get (BLOCK_S,)

        # Store dt gradients
        grad_dt_ptrs = GRAD_DT + seq_range * stride_grad_dt[1]
        grad_dt_mask = seq_mask
        tl.store(grad_dt_ptrs, dt_grad_vals, mask=grad_dt_mask)

        # Compute angle gradients
        d_tanh = 1.0 - tanh_angle_vals * tanh_angle_vals
        grad_angle_vals = (3.141592653589793 * dt_vals[:, None]) * d_tanh * grad_out_vals

        # Store angle gradients
        grad_angle_ptrs = GRAD_ANGLE + seq_range[:, None] * stride_grad_angle[1] + dim_range[None, :] * stride_grad_angle[3]
        grad_angle_mask = (seq_mask[:, None] & dim_mask[None, :])
        tl.store(grad_angle_ptrs, grad_angle_vals, mask=grad_angle_mask)


def apply_angle_dt_fwd(
    angle: torch.Tensor,  # (batch, seqlen, nheads, dim)
    dt: torch.Tensor,     # (batch, seqlen, nheads)
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multiply angle and dt tensors element-wise and compute chunk sums.

    Arguments:
        angle: (batch, seqlen, nheads, dim)
        dt: (batch, seqlen, nheads)
        chunk_size: Size of chunks for summing (must be power of 2)
        write_output: Whether to write the full output tensor
        write_chunk_sum: Whether to write the chunk sum tensor
        prefix: Optional prefix to add before cumsum (batch, numchunks, nheads, dim)

    Returns:
        output: (batch, seqlen, nheads, dim) - may contain uninitialized data if write_output=False
        output_sum: (batch, seqlen // chunk_size, nheads, dim) - may contain uninitialized data if write_chunk_sum=False
    """
    batch, seqlen, nheads, dim = angle.shape
    assert angle.shape == (batch, seqlen, nheads, dim)
    assert dt.shape == (batch, seqlen, nheads)
    assert chunk_size > 0 and (chunk_size & (chunk_size - 1)) == 0, "chunk_size must be power of 2"

    # Calculate output dimensions
    num_chunks = math.ceil(seqlen / chunk_size)

    # Create output tensors (always fp32)
    output = torch.empty(batch, seqlen, nheads, dim, device=angle.device, dtype=torch.float32)
    output_sum = torch.empty(batch, num_chunks, nheads, dim, device=angle.device, dtype=torch.float32)

    # Launch kernel
    BLOCK_S = chunk_size  # Use chunk_size as BLOCK_S
    BLOCK_D = triton.next_power_of_2(dim)

    # Step 1: compute the sum of each chunk. Don't write the output
    grid = lambda META: (nheads, num_chunks, batch)
    with torch.cuda.device(angle.device.index):
        torch.library.wrap_triton(angle_dt_fwd_kernel)[grid](
            None,  # output
            output_sum,
            angle,
            dt,
            None,  # prefix
            seqlen,
            dim,
            chunk_size,
            (0, 0, 0, 0),  # output_stride
            output_sum.stride(),
            angle.stride(),
            dt.stride(),
            (0, 0, 0, 0),   # prefix_stride
            BLOCK_S=BLOCK_S,
            BLOCK_D=BLOCK_D,
            WRITE_OUTPUT=False,
            WRITE_CHUNK_SUM=True,
            HAS_PREFIX=False,
        )

    # Step 2: compute cumsum on output_sum to get prefix
    prefix = apply_cumsum(output_sum)  # Shape: (batch, num_chunks, nheads, dim)

    # Step 3: call angle_dt_kernel again with output and prefix, don't need to write output_sum
    with torch.cuda.device(angle.device.index):
        torch.library.wrap_triton(angle_dt_fwd_kernel)[grid](
            output,  # output
            None,    # output_sum (don't need to write)
            angle,
            dt,
            prefix,  # prefix
            seqlen,
            dim,
            chunk_size,
            output.stride(),  # output_stride
            (0, 0, 0, 0),     # output_sum_stride
            angle.stride(),
            dt.stride(),
            prefix.stride(),  # prefix_stride
            BLOCK_S=BLOCK_S,
            BLOCK_D=BLOCK_D,
            WRITE_OUTPUT=True,
            WRITE_CHUNK_SUM=False,
            HAS_PREFIX=True,
        )

    return output

def apply_angle_dt_bwd(
    grad_out: torch.Tensor,  # (batch, seqlen, nheads, dim)
    angle: torch.Tensor,     # (batch, seqlen, nheads, dim)
    dt: torch.Tensor,        # (batch, seqlen, nheads)
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multiply angle and dt tensors element-wise and compute chunk sums.

    Arguments:
        grad_out: (batch, seqlen, nheads, dim) - gradient of the output
        angle: (batch, seqlen, nheads, dim) - stored angle tensor
        dt: (batch, seqlen, nheads) - stored delta time tensor
        chunk_size: Size of chunks for summing (must be power of 2)
        write_output: Whether to write the full output tensor
        write_chunk_sum: Whether to write the chunk sum tensor
        prefix: Optional prefix to add before cumsum (batch, numchunks, nheads, dim)

    Returns:
        output: (batch, seqlen, nheads, dim) - may contain uninitialized data if write_output=False
        output_sum: (batch, seqlen // chunk_size, nheads, dim) - may contain uninitialized data if write_chunk_sum=False
    """
    batch, seqlen, nheads, dim = grad_out.shape
    assert grad_out.shape == (batch, seqlen, nheads, dim)
    assert angle.shape == (batch, seqlen, nheads, dim)
    assert dt.shape == (batch, seqlen, nheads)
    assert chunk_size > 0 and (chunk_size & (chunk_size - 1)) == 0, "chunk_size must be power of 2"

    # Calculate output dimensions
    num_chunks = math.ceil(seqlen / chunk_size)

    # Reverse the sequence dimension of grad_out, angle, dt
    grad_out = grad_out.flip(dims=(1,))  # Reverse along sequence dimension
    angle = angle.flip(dims=(1,))
    dt = dt.flip(dims=(1,))

    # Create output tensors (always fp32)
    grad_dt = torch.empty_like(dt) # (batch, seqlen, nheads)
    grad_angle = torch.empty_like(angle) # (batch, seqlen, nheads, dim)
    grad_sum = torch.empty(batch, num_chunks, nheads, dim, device=angle.device, dtype=torch.float32)

    # Launch kernel
    BLOCK_S = chunk_size  # Use chunk_size as BLOCK_S
    BLOCK_D = triton.next_power_of_2(dim)

    # Step 1: compute the sum of each chunk. Don't write the output
    grid = lambda META: (nheads, num_chunks, batch)
    with torch.cuda.device(angle.device.index):
        torch.library.wrap_triton(angle_dt_bwd_kernel)[grid](
            None,  # GRAD_DT
            None,  # GRAD_ANGLE
            grad_sum,  # GRAD_SUM
            grad_out,  # GRAD_OUT
            angle,
            dt,
            None,  # PREFIX
            seqlen,
            dim,
            chunk_size,
            (0, 0, 0),             # stride_grad_dt
            (0, 0, 0, 0),          # stride_grad_angle
            grad_sum.stride(),     # stride_grad_sum
            grad_out.stride(),     # stride_grad_out
            angle.stride(),
            dt.stride(),
            (0, 0, 0, 0),          # stride_prefix
            BLOCK_S=BLOCK_S,
            BLOCK_D=BLOCK_D,
            WRITE_GRAD=False,      # Don't write grad_dt and grad_angle yet
            WRITE_CHUNK_SUM=True,  # Write chunk sums to grad_sum
            HAS_PREFIX=False,      # No prefix provided
        )

    # Step 2: compute cumsum on output_sum to get prefix
    prefix = apply_cumsum(grad_sum)  # Shape: (batch, num_chunks, nheads, dim)

    # Step 3: call angle_dt_fwd_chunksum_kernel again with output and prefix, don't need to write output_sum
    with torch.cuda.device(angle.device.index):
        torch.library.wrap_triton(angle_dt_bwd_kernel)[grid](
            grad_dt,
            grad_angle,
            None,               # GRAD_SUM (don't need to write)
            grad_out,
            angle,
            dt,
            prefix,             # prefix
            seqlen,
            dim,
            chunk_size,
            grad_dt.stride(),       # stride_grad_dt
            grad_angle.stride(),    # stride_grad_angle
            (0, 0, 0),              # stride_grad_sum
            grad_out.stride(),      # stride_grad_out
            angle.stride(),
            dt.stride(),
            prefix.stride(),        # stride_prefix
            BLOCK_S=BLOCK_S,
            BLOCK_D=BLOCK_D,
            WRITE_GRAD=True,        # Write grad_dt and grad_angle
            WRITE_CHUNK_SUM=False,  # Don't write chunk sums again
            HAS_PREFIX=True,        # Use the computed prefix
        )

    grad_dt = grad_dt.flip(dims=(1,))
    grad_angle = grad_angle.flip(dims=(1,)) 

    return grad_dt, grad_angle


def apply_cumsum(
    x: torch.Tensor,  # (batch, seqlen, nheads, dim)
) -> torch.Tensor:
    """
    Compute cumulative sum along sequence dimension using Triton.

    Arguments:
        x: (batch, seqlen, nheads, dim)

    Returns:
        output: (batch, seqlen, nheads, dim) - cumulative sum along seqlen dimension
    """
    batch, seqlen, nheads, dim = x.shape
    assert seqlen <= 512, f"seqlen must be <= 512, got {seqlen}"
    # Create output tensor (always fp32)
    output = torch.empty_like(x, dtype=torch.float32)

    # Launch kernel
    BLOCK_S = triton.next_power_of_2(seqlen)
    BLOCK_D = triton.next_power_of_2(min(dim, 16))

    grid = lambda META: (nheads, triton.cdiv(dim, META["BLOCK_D"]), batch)
    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(cumsum_kernel)[grid](
            output,
            x,
            seqlen,
            dim,
            output.stride(),
            x.stride(),
            BLOCK_S=BLOCK_S,
            BLOCK_D=BLOCK_D,
        )

    return output


def apply_angle_dt_reference(
    angle: torch.Tensor,  # (batch, seqlen, nheads, dim)
    dt: torch.Tensor,     # (batch, seqlen, nheads)
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference PyTorch implementation."""
    batch, seqlen, nheads, dim = angle.shape

    # Element-wise multiply: angle (B, S, H, D) * dt (B, S, H, 1) -> (B, S, H, D)
    #base_vals = (angle * dt[..., None]).to(torch.float32)  # Always return fp32
    base_vals = (angle).to(torch.float32)

    # Apply tanh then multiply by pi
    base_vals = torch.tanh(base_vals) * dt[..., None].to(torch.float32) * torch.pi

    # Simple cumulative sum along seqlen dimension
    output = torch.cumsum(base_vals, dim=1)
    return output


def test_correctness():
    """Test correctness against reference implementation."""
    print("Testing angle_dt kernel correctness...")

    # Test parameters
    batch, seqlen, nheads, dim = 2, 512, 4, 32
    chunk_size = 64
    device = "cuda"
    dtype = torch.float32

    # Create test tensors
    #torch.manual_seed(42)
    angle = torch.randn(batch, seqlen, nheads, dim, device=device, dtype=dtype)
    dt = torch.randn(batch, seqlen, nheads, device=device, dtype=dtype)

    # Test kernel vs reference
    out_triton = apply_angle_dt_fwd(angle, dt, chunk_size)
    out_ref = apply_angle_dt_reference(angle, dt, chunk_size)

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"Output max difference: {max_diff:.6f}")
    assert max_diff < 1e-3, f"Too large difference in output: {max_diff}"
    print("Test passed! ✓")
    print("All basic tests passed! ✓")


def test_cumsum_correctness():
    """Test cumsum kernel correctness against PyTorch."""
    print("Testing cumsum kernel correctness...")

    # Test parameters
    batch, seqlen, nheads, dim = 4, 128, 8, 64
    device = "cuda"
    dtype = torch.float32

    # Create test tensors
    #torch.manual_seed(42)
    x = torch.randn(batch, seqlen, nheads, dim, device=device, dtype=dtype)

    # Test kernel vs PyTorch
    out_triton = apply_cumsum(x)
    out_ref = torch.cumsum(x, dim=1).to(torch.float32)

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"Cumsum max difference: {max_diff:.6f}")

    assert max_diff < 1e-4, f"Too large difference in cumsum: {max_diff}"

    print("Cumsum test passed! ✓")

def test_backward_correctness():
    """Backward correctness vs PyTorch autograd on small cases."""
    print("Testing backward correctness...")

    device = "cuda"
    tol = 5e-3  # fp32

    cases = [
        (2, 257, 3, 17, 64),  # odd S/D, non-power-of-two
        (1, 129, 4, 33, 32),
    ]

    for (batch, seqlen, nheads, dim, chunk_size) in cases:
        angle = torch.randn(batch, seqlen, nheads, dim, device=device, dtype=torch.float32)
        dt    = torch.randn(batch, seqlen, nheads,      device=device, dtype=torch.float32)
        grad_out = torch.randn(batch, seqlen, nheads, dim, device=device, dtype=torch.float32)

        # Triton bwd
        grad_dt_tri, grad_angle_tri = apply_angle_dt_bwd(grad_out, angle, dt, chunk_size)
        # Reference bwd via autograd
        angle_ref = angle.detach().clone().requires_grad_(True)
        dt_ref    = dt.detach().clone().requires_grad_(True)
        out_ref = apply_angle_dt_reference(angle_ref, dt_ref, chunk_size)
        out_ref.backward(grad_out)
        grad_angle_ref = angle_ref.grad.detach()
        grad_dt_ref    = dt_ref.grad.detach()

        max_da = (grad_angle_tri - grad_angle_ref).abs().max().item()
        max_dd = (grad_dt_tri    - grad_dt_ref   ).abs().max().item()
        print(f"  Case B={batch} S={seqlen} H={nheads} D={dim} chunk={chunk_size} | "
              f"max|Δ angle|={max_da:.3e}  max|Δ dt|={max_dd:.3e}")
        assert max_da < tol, f"angle grad mismatch {max_da}"
        assert max_dd < tol, f"dt grad mismatch {max_dd}"

    print("Backward correctness test passed! ✓")

def benchmark_angle_dt():
    """Benchmark angle_dt kernel and measure memory bandwidth."""
    print("\nBenchmarking angle_dt kernel...")

    # Benchmark parameters
    batch, seqlen, nheads, dim = 8, 4096, 32, 32
    # batch, seqlen, nheads, dim = 1, 128, 1, 1
    chunk_size = 128
    device = "cuda"
    dtype = torch.bfloat16

    # Create input tensors
    #torch.manual_seed(42)
    # Generate angle by expanding from (batch, seqlen, 1, dim) to (batch, seqlen, nheads, dim)
    angle_base = torch.randn(batch, seqlen, 1, dim, device=device, dtype=dtype)
    angle = angle_base.expand(batch, seqlen, nheads, dim)
    dt = torch.randn(batch, seqlen, nheads, device=device, dtype=dtype)

    fn = lambda: apply_angle_dt_fwd(angle, dt, chunk_size)
    out = fn()
    # Warmup
    for _ in range(10):
        fn()

    # Benchmark
    torch.cuda.synchronize()
    import time
    time.sleep(0.5)

    # Run benchmark
    time_ms = triton.testing.do_bench(fn, warmup=10, rep=100)

    # Calculate memory bandwidth
    # Read: angle_base (actual underlying data) + dt
    # Write: output + output_sum (always fp32, so 4 bytes per element)
    # Note: angle is expanded so actual memory read is only angle_base.numel()
    bytes_read = angle_base.untyped_storage().nbytes() + dt.untyped_storage().nbytes()
    bytes_write = out.untyped_storage().nbytes()  # Both output and output_sum (fp32 = 4 bytes)
    total_bytes = bytes_read + bytes_write

    # Convert to GB/s
    time_s = time_ms / 1000.0
    bandwidth_gb_s = (total_bytes / 1e9) / time_s

    print(f"Angle base shape: {angle_base.shape}")
    print(f"Angle expanded shape: {angle.shape}")
    print(f"Angle stride: {angle.stride()}")
    print(f"DT shape: {dt.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Chunk size: {chunk_size}")
    print(f"Time: {time_ms:.3f} ms")
    print(f"Memory transferred: {total_bytes / 1e9:.3f} GB")
    print(f"Memory bandwidth: {bandwidth_gb_s:.1f} GB/s")

    # from flash_attn.utils.benchmark import pytorch_profiler
    # pytorch_profiler(fn)

    return time_ms, bandwidth_gb_s

def benchmark_angle_dt_backward():
    """Benchmark backward pass and report rough memory bandwidth."""
    print("\nBenchmarking angle_dt backward...")

    batch, seqlen, nheads, dim = 8, 4096, 32, 32
    chunk_size = 128
    device = "cuda"

    # Use fp32 for bwd accumulations
    angle = torch.randn(batch, seqlen, nheads, dim, device=device, dtype=torch.float32)
    dt    = torch.randn(batch, seqlen, nheads,      device=device, dtype=torch.float32)
    grad_out = torch.randn(batch, seqlen, nheads, dim, device=device, dtype=torch.float32)

    fn = lambda: apply_angle_dt_bwd(grad_out, angle, dt, chunk_size)
    _ = fn()
    # Warmup
    for _ in range(10):
        fn()

    torch.cuda.synchronize()
    import time
    time.sleep(0.5)
    time_ms = triton.testing.do_bench(fn, warmup=10, rep=100)

    # Rough traffic estimate (two-stage + prefixes), conservative:
    num_chunks = (seqlen + chunk_size - 1) // chunk_size
    bytes_read = (
        grad_out.numel() * 4 +  # read grad_out
        angle.numel()   * 4 +   # read angle
        dt.numel()      * 4 +   # read dt
        (batch * num_chunks * nheads * dim) * 4 +  # read grad_sum for prefix
        (batch * num_chunks * nheads * dim) * 4    # read prefix in stage 2
    )
    bytes_write = (
        (batch * num_chunks * nheads * dim) * 4 +  # write grad_sum (stage 1)
        (batch * seqlen * nheads) * 4 +            # write grad_dt
        (batch * seqlen * nheads * dim) * 4        # write grad_angle
    )
    total_bytes = bytes_read + bytes_write
    bandwidth_gb_s = (total_bytes / 1e9) / (time_ms / 1000.0)

    print(f"B={batch} S={seqlen} H={nheads} D={dim} chunk={chunk_size}")
    print(f"Time: {time_ms:.3f} ms")
    print(f"Memory transferred (est): {total_bytes / 1e9:.3f} GB")
    print(f"Memory bandwidth (est): {bandwidth_gb_s:.1f} GB/s")

    return time_ms, bandwidth_gb_s

if __name__ == "__main__":
    test_correctness()
    test_cumsum_correctness()
    benchmark_angle_dt()