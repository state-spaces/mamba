"""Utilities for reducing per-head backward outputs to grouped Q/K heads."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def reduce_grouped_qk_grads(qk_grads: torch.Tensor, num_qk_groups: int) -> torch.Tensor:
    """Collapse per-head Q/K grads from ``H`` heads back to ``G`` grouped heads.

    The TileLang backward kernels materialize Q/K gradients per value head with shape
    ``[B, S, R, H, N]``. The forward kernels map each value head ``i_h`` to a Q/K
    group via ``i_h // (H // G)``, so the corresponding grouped reduction must sum
    contiguous blocks of ``H // G`` heads.
    """
    if qk_grads.ndim != 5:
        raise ValueError(
            f"Expected qk_grads to have shape [B, S, R, H, N], got {tuple(qk_grads.shape)}."
        )

    num_heads = qk_grads.shape[3]
    if num_heads % num_qk_groups != 0:
        raise ValueError(
            f"Expected H ({num_heads}) to be divisible by G ({num_qk_groups})."
        )

    group_size = num_heads // num_qk_groups
    return qk_grads.reshape(
        *qk_grads.shape[:3],
        num_qk_groups,
        group_size,
        qk_grads.shape[-1],
    ).sum(dim=4)


@triton.jit
def _reduce_grouped_qk_grads_and_bias_partial_kernel(
    dq_raw,
    dk_raw,
    dq_out,
    dk_out,
    dq_bias_partial,
    dk_bias_partial,
    TOTAL_ROWS: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STORE_GROUPED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_gn = tl.program_id(2)
    pid_g = pid_gn // N_BLOCKS
    pid_n = pid_gn - pid_g * N_BLOCKS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < TOTAL_ROWS) & (offs_n[None, :] < N)

    grouped_dq = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    grouped_dk = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for h_in_group in range(0, GROUP_SIZE):
        h = pid_g * GROUP_SIZE + h_in_group
        raw_offsets = ((offs_m[:, None] * R + pid_r) * H + h) * N + offs_n[None, :]
        dq_vals = tl.load(dq_raw + raw_offsets, mask=mask, other=0.0).to(tl.float32)
        dk_vals = tl.load(dk_raw + raw_offsets, mask=mask, other=0.0).to(tl.float32)

        grouped_dq += dq_vals
        grouped_dk += dk_vals

        bias_offsets = ((pid_m * H + h) * R + pid_r) * N + offs_n
        tl.store(
            dq_bias_partial + bias_offsets,
            tl.sum(dq_vals, axis=0),
            mask=offs_n < N,
        )
        tl.store(
            dk_bias_partial + bias_offsets,
            tl.sum(dk_vals, axis=0),
            mask=offs_n < N,
        )

    if STORE_GROUPED:
        out_offsets = ((offs_m[:, None] * R + pid_r) * G + pid_g) * N + offs_n[None, :]
        tl.store(dq_out + out_offsets, grouped_dq, mask=mask)
        tl.store(dk_out + out_offsets, grouped_dk, mask=mask)


@triton.jit
def _finalize_qk_bias_kernel(
    dq_bias_partial,
    dk_bias_partial,
    dq_bias,
    dk_bias,
    NUM_ROW_BLOCKS: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_b = tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_dq = tl.zeros((BLOCK_N,), tl.float32)
    acc_dk = tl.zeros((BLOCK_N,), tl.float32)

    for block_start in range(0, NUM_ROW_BLOCKS, BLOCK_B):
        block_ids = block_start + offs_b
        offsets = ((block_ids[:, None] * H + pid_h) * R + pid_r) * N + offs_n[None, :]
        mask = (block_ids[:, None] < NUM_ROW_BLOCKS) & (offs_n[None, :] < N)
        dq_vals = tl.load(dq_bias_partial + offsets, mask=mask, other=0.0)
        dk_vals = tl.load(dk_bias_partial + offsets, mask=mask, other=0.0)
        acc_dq += tl.sum(dq_vals, axis=0)
        acc_dk += tl.sum(dk_vals, axis=0)

    out_offsets = (pid_h * R + pid_r) * N + offs_n
    tl.store(dq_bias + out_offsets, acc_dq, mask=offs_n < N)
    tl.store(dk_bias + out_offsets, acc_dk, mask=offs_n < N)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def reduce_grouped_qk_grads_and_bias_triton(
    dq_raw: torch.Tensor,
    dk_raw: torch.Tensor,
    num_qk_groups: int,
    block_m: int = 64,
    block_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce raw per-value-head Q/K grads and compute Q/K bias grads in one pass.

    ``dq_raw`` and ``dk_raw`` are TileLang outputs shaped ``[B, S, R, H, N]``.
    This returns grouped Q/K grads shaped ``[B, S, R, G, N]`` and bias grads
    shaped ``[H, R, N]`` while avoiding separate full-tensor reads for the bias
    and grouped-head reductions.
    """
    if dq_raw.shape != dk_raw.shape:
        raise ValueError(
            f"Expected dq_raw and dk_raw to have the same shape, got "
            f"{tuple(dq_raw.shape)} and {tuple(dk_raw.shape)}."
        )
    if dq_raw.ndim != 5:
        raise ValueError(
            f"Expected raw Q/K grads to have shape [B, S, R, H, N], got {tuple(dq_raw.shape)}."
        )
    if not dq_raw.is_contiguous() or not dk_raw.is_contiguous():
        raise ValueError("Expected raw Q/K grads to be contiguous.")

    B, S, R, H, N = dq_raw.shape
    if H % num_qk_groups != 0:
        raise ValueError(f"Expected H ({H}) to be divisible by G ({num_qk_groups}).")

    total_rows = B * S
    group_size = H // num_qk_groups
    num_row_blocks = triton.cdiv(total_rows, block_m)
    if block_n is None:
        block_n = min(16, _next_power_of_2(N))
    block_n = max(1, block_n)

    dq_bias_partial = torch.empty(
        (num_row_blocks, H, R, N), dtype=torch.float32, device=dq_raw.device
    )
    dk_bias_partial = torch.empty_like(dq_bias_partial)

    if num_qk_groups == H:
        dq_out = dq_raw
        dk_out = dk_raw
        store_grouped = False
    else:
        dq_out = torch.empty(
            (B, S, R, num_qk_groups, N), dtype=dq_raw.dtype, device=dq_raw.device
        )
        dk_out = torch.empty_like(dq_out)
        store_grouped = True

    n_blocks = triton.cdiv(N, block_n)
    grid = (num_row_blocks, R, num_qk_groups * n_blocks)
    _reduce_grouped_qk_grads_and_bias_partial_kernel[grid](
        dq_raw,
        dk_raw,
        dq_out,
        dk_out,
        dq_bias_partial,
        dk_bias_partial,
        total_rows,
        R,
        H,
        num_qk_groups,
        N,
        group_size,
        n_blocks,
        block_m,
        block_n,
        store_grouped,
        num_warps=8,
    )

    dq_bias = torch.empty((H, R, N), dtype=torch.float32, device=dq_raw.device)
    dk_bias = torch.empty_like(dq_bias)
    block_b = min(1024, _next_power_of_2(num_row_blocks))
    _finalize_qk_bias_kernel[(H, R, n_blocks)](
        dq_bias_partial,
        dk_bias_partial,
        dq_bias,
        dk_bias,
        num_row_blocks,
        H,
        R,
        N,
        block_b,
        block_n,
        num_warps=8,
    )

    return dq_out, dk_out, dq_bias, dk_bias
