# Copyright (c) 2024, Tri Dao, Albert Gu.
# Mamba-3 fused RoPE (Rotary Position Embedding) Triton kernels for B and C.
#
# Replaces `apply_rotary_emb_to_bc` in mamba3_ssd.py with fused Triton kernels:
# 1. Forward: expand B/C from groups to heads + apply sincos rotation
# 2. Backward: reverse rotation + reduce heads to groups + dtheta gradient
#
# Strategy (approach a): the cumulative sum of theta is computed in PyTorch
# (one kernel launch), then the Triton kernel handles the group->head expansion
# and sincos rotation. This avoids in-kernel sequential scan complexity.

import torch
import triton
import triton.language as tl


# =============================================================================
# Forward kernel: group->head expansion + RoPE rotation
# =============================================================================

@triton.jit
def _mamba3_rope_fwd_kernel(
    # Input pointers
    b_ptr, c_ptr, cos_ptr, sin_ptr,
    # Output pointers
    b_out_ptr, c_out_ptr,
    # Dimensions
    batch, seqlen, nheads, ngroups, dstate, half_d,
    nheads_per_group,
    # B strides (batch, seqlen, ngroups, dstate)
    stride_b_batch, stride_b_seqlen, stride_b_group, stride_b_dstate,
    # C strides (same layout as B)
    stride_c_batch, stride_c_seqlen, stride_c_group, stride_c_dstate,
    # cos/sin strides (batch, seqlen, nheads, half_d)
    stride_cs_batch, stride_cs_seqlen, stride_cs_head, stride_cs_halfd,
    # B_out strides (batch, seqlen, nheads, dstate)
    stride_bo_batch, stride_bo_seqlen, stride_bo_head, stride_bo_dstate,
    # C_out strides (same layout as B_out)
    stride_co_batch, stride_co_seqlen, stride_co_head, stride_co_dstate,
    # Meta-parameters
    BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Fused group->head expansion and RoPE rotation for B and C.

    Grid: (batch, cdiv(seqlen, BLOCK_L), nheads)

    For each (batch, seqlen_tile, head):
      1. Load B from the corresponding group (head // nheads_per_group)
      2. Split into even (first half_d) and odd (second half_d) halves
      3. Apply rotation: B_out_even = B_even*cos - B_odd*sin
                          B_out_odd  = B_even*sin + B_odd*cos
      4. Same for C
    """
    pid_b = tl.program_id(axis=0)
    pid_l = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Which group does this head belong to?
    pid_g = pid_h // nheads_per_group

    # Sequence offsets for this tile
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    # Half-dstate offsets
    offs_d = tl.arange(0, BLOCK_D)
    l_mask = offs_l < seqlen
    d_mask = offs_d < half_d

    # --- Load cos and sin for this head ---
    cs_base = pid_b * stride_cs_batch + pid_h * stride_cs_head
    cos_ptrs = cos_ptr + cs_base + offs_l[:, None] * stride_cs_seqlen + offs_d[None, :] * stride_cs_halfd
    sin_ptrs = sin_ptr + cs_base + offs_l[:, None] * stride_cs_seqlen + offs_d[None, :] * stride_cs_halfd
    mask_ld = l_mask[:, None] & d_mask[None, :]
    cos_val = tl.load(cos_ptrs, mask=mask_ld, other=1.0).to(tl.float32)
    sin_val = tl.load(sin_ptrs, mask=mask_ld, other=0.0).to(tl.float32)

    # --- Process B ---
    b_base = pid_b * stride_b_batch + pid_g * stride_b_group
    # Even half (first half_d elements of dstate)
    b_even_ptrs = b_ptr + b_base + offs_l[:, None] * stride_b_seqlen + offs_d[None, :] * stride_b_dstate
    # Odd half (second half_d elements of dstate)
    b_odd_ptrs = b_ptr + b_base + offs_l[:, None] * stride_b_seqlen + (offs_d[None, :] + half_d) * stride_b_dstate
    b_even = tl.load(b_even_ptrs, mask=mask_ld, other=0.0).to(tl.float32)
    b_odd = tl.load(b_odd_ptrs, mask=mask_ld, other=0.0).to(tl.float32)

    # Rotation
    b_out_even = b_even * cos_val - b_odd * sin_val
    b_out_odd = b_even * sin_val + b_odd * cos_val

    # Store to B_out at head level
    bo_base = pid_b * stride_bo_batch + pid_h * stride_bo_head
    bo_even_ptrs = b_out_ptr + bo_base + offs_l[:, None] * stride_bo_seqlen + offs_d[None, :] * stride_bo_dstate
    bo_odd_ptrs = b_out_ptr + bo_base + offs_l[:, None] * stride_bo_seqlen + (offs_d[None, :] + half_d) * stride_bo_dstate
    tl.store(bo_even_ptrs, b_out_even.to(b_out_ptr.dtype.element_ty), mask=mask_ld)
    tl.store(bo_odd_ptrs, b_out_odd.to(b_out_ptr.dtype.element_ty), mask=mask_ld)

    # --- Process C (identical logic) ---
    c_base = pid_b * stride_c_batch + pid_g * stride_c_group
    c_even_ptrs = c_ptr + c_base + offs_l[:, None] * stride_c_seqlen + offs_d[None, :] * stride_c_dstate
    c_odd_ptrs = c_ptr + c_base + offs_l[:, None] * stride_c_seqlen + (offs_d[None, :] + half_d) * stride_c_dstate
    c_even = tl.load(c_even_ptrs, mask=mask_ld, other=0.0).to(tl.float32)
    c_odd = tl.load(c_odd_ptrs, mask=mask_ld, other=0.0).to(tl.float32)

    c_out_even = c_even * cos_val - c_odd * sin_val
    c_out_odd = c_even * sin_val + c_odd * cos_val

    co_base = pid_b * stride_co_batch + pid_h * stride_co_head
    co_even_ptrs = c_out_ptr + co_base + offs_l[:, None] * stride_co_seqlen + offs_d[None, :] * stride_co_dstate
    co_odd_ptrs = c_out_ptr + co_base + offs_l[:, None] * stride_co_seqlen + (offs_d[None, :] + half_d) * stride_co_dstate
    tl.store(co_even_ptrs, c_out_even.to(c_out_ptr.dtype.element_ty), mask=mask_ld)
    tl.store(co_odd_ptrs, c_out_odd.to(c_out_ptr.dtype.element_ty), mask=mask_ld)


def _mamba3_rope_fwd(B, C, theta, nheads, ngroups):
    """Apply RoPE to B and C, expanding from groups to heads.

    Args:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        theta: (batch, seqlen, nheads, dstate//2) -- per-step rotation angles

    Returns:
        B_heads: (batch, seqlen, nheads, dstate)
        C_heads: (batch, seqlen, nheads, dstate)
        theta_cumsum: (batch, seqlen, nheads, dstate//2) -- saved for backward
    """
    batch, seqlen, ngroups_B, dstate = B.shape
    assert C.shape == B.shape
    assert theta.shape == (batch, seqlen, nheads, dstate // 2)
    assert nheads % ngroups == 0
    half_d = dstate // 2
    nheads_per_group = nheads // ngroups

    # Step 1: Compute cumulative sum of theta in PyTorch
    theta_cumsum = torch.cumsum(theta.float(), dim=1)  # (batch, seqlen, nheads, half_d)
    cos_theta = torch.cos(theta_cumsum)
    sin_theta = torch.sin(theta_cumsum)

    # Step 2: Allocate output at head level
    B_heads = torch.empty(batch, seqlen, nheads, dstate, device=B.device, dtype=B.dtype)
    C_heads = torch.empty(batch, seqlen, nheads, dstate, device=C.device, dtype=C.dtype)

    # Choose block sizes
    BLOCK_L = min(triton.next_power_of_2(seqlen), 128)
    BLOCK_D = triton.next_power_of_2(half_d)

    grid = (batch, triton.cdiv(seqlen, BLOCK_L), nheads)

    with torch.cuda.device(B.device.index):
        _mamba3_rope_fwd_kernel[grid](
            B, C, cos_theta, sin_theta,
            B_heads, C_heads,
            batch, seqlen, nheads, ngroups, dstate, half_d,
            nheads_per_group,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            cos_theta.stride(0), cos_theta.stride(1), cos_theta.stride(2), cos_theta.stride(3),
            B_heads.stride(0), B_heads.stride(1), B_heads.stride(2), B_heads.stride(3),
            C_heads.stride(0), C_heads.stride(1), C_heads.stride(2), C_heads.stride(3),
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D,
        )

    return B_heads, C_heads, theta_cumsum


# =============================================================================
# Backward kernel: reverse rotation + head->group reduction + dtheta
# =============================================================================

@triton.jit
def _mamba3_rope_bwd_kernel(
    # Gradient inputs (at head level)
    db_heads_ptr, dc_heads_ptr,
    # Forward outputs (for dtheta computation)
    b_heads_ptr, c_heads_ptr,
    # cos/sin from forward
    cos_ptr, sin_ptr,
    # Outputs
    db_ptr, dc_ptr, dtheta_ptr,
    # Dimensions
    batch, seqlen, nheads, ngroups, dstate, half_d,
    nheads_per_group,
    # dB_heads strides (batch, seqlen, nheads, dstate)
    stride_dbh_batch, stride_dbh_seqlen, stride_dbh_head, stride_dbh_dstate,
    # dC_heads strides
    stride_dch_batch, stride_dch_seqlen, stride_dch_head, stride_dch_dstate,
    # B_heads strides (forward outputs)
    stride_bh_batch, stride_bh_seqlen, stride_bh_head, stride_bh_dstate,
    # C_heads strides
    stride_ch_batch, stride_ch_seqlen, stride_ch_head, stride_ch_dstate,
    # cos/sin strides
    stride_cs_batch, stride_cs_seqlen, stride_cs_head, stride_cs_halfd,
    # dB output strides (batch, seqlen, ngroups, dstate)
    stride_db_batch, stride_db_seqlen, stride_db_group, stride_db_dstate,
    # dC output strides
    stride_dc_batch, stride_dc_seqlen, stride_dc_group, stride_dc_dstate,
    # dtheta strides (batch, seqlen, nheads, half_d)
    stride_dth_batch, stride_dth_seqlen, stride_dth_head, stride_dth_halfd,
    # Meta-parameters
    BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Backward through RoPE for B and C.

    Grid: (batch, cdiv(seqlen, BLOCK_L), nheads)

    Computes:
      dB[group] += reverse_rotate(dB_heads[head]) for all heads in group
      dC[group] += reverse_rotate(dC_heads[head]) for all heads in group
      dtheta[head] = (-B_out_odd)*dB_out_even + B_out_even*dB_out_odd
                   + (-C_out_odd)*dC_out_even + C_out_even*dC_out_odd

    Note: dtheta here is the gradient w.r.t. theta_cumsum. The reverse cumsum
    to get gradient w.r.t. theta is done in PyTorch outside this kernel.

    For dB reduction across heads in a group, we use atomic_add since multiple
    heads map to the same group.
    """
    pid_b = tl.program_id(axis=0)
    pid_l = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_g = pid_h // nheads_per_group

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, BLOCK_D)
    l_mask = offs_l < seqlen
    d_mask = offs_d < half_d
    mask_ld = l_mask[:, None] & d_mask[None, :]

    # Load cos/sin
    cs_base = pid_b * stride_cs_batch + pid_h * stride_cs_head
    cos_val = tl.load(cos_ptr + cs_base + offs_l[:, None] * stride_cs_seqlen + offs_d[None, :] * stride_cs_halfd,
                      mask=mask_ld, other=1.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + cs_base + offs_l[:, None] * stride_cs_seqlen + offs_d[None, :] * stride_cs_halfd,
                      mask=mask_ld, other=0.0).to(tl.float32)

    # --- dB backward ---
    dbh_base = pid_b * stride_dbh_batch + pid_h * stride_dbh_head
    db_out_even = tl.load(db_heads_ptr + dbh_base + offs_l[:, None] * stride_dbh_seqlen + offs_d[None, :] * stride_dbh_dstate,
                          mask=mask_ld, other=0.0).to(tl.float32)
    db_out_odd = tl.load(db_heads_ptr + dbh_base + offs_l[:, None] * stride_dbh_seqlen + (offs_d[None, :] + half_d) * stride_dbh_dstate,
                         mask=mask_ld, other=0.0).to(tl.float32)

    # Reverse rotation: dB_even = dB_out_even*cos + dB_out_odd*sin
    #                    dB_odd  = -dB_out_even*sin + dB_out_odd*cos
    db_even = db_out_even * cos_val + db_out_odd * sin_val
    db_odd = -db_out_even * sin_val + db_out_odd * cos_val

    # Atomic add to dB at group level (multiple heads contribute to same group)
    db_base = pid_b * stride_db_batch + pid_g * stride_db_group
    db_even_ptrs = db_ptr + db_base + offs_l[:, None] * stride_db_seqlen + offs_d[None, :] * stride_db_dstate
    db_odd_ptrs = db_ptr + db_base + offs_l[:, None] * stride_db_seqlen + (offs_d[None, :] + half_d) * stride_db_dstate
    tl.atomic_add(db_even_ptrs, db_even.to(db_ptr.dtype.element_ty), mask=mask_ld)
    tl.atomic_add(db_odd_ptrs, db_odd.to(db_ptr.dtype.element_ty), mask=mask_ld)

    # --- dC backward (identical structure) ---
    dch_base = pid_b * stride_dch_batch + pid_h * stride_dch_head
    dc_out_even = tl.load(dc_heads_ptr + dch_base + offs_l[:, None] * stride_dch_seqlen + offs_d[None, :] * stride_dch_dstate,
                          mask=mask_ld, other=0.0).to(tl.float32)
    dc_out_odd = tl.load(dc_heads_ptr + dch_base + offs_l[:, None] * stride_dch_seqlen + (offs_d[None, :] + half_d) * stride_dch_dstate,
                         mask=mask_ld, other=0.0).to(tl.float32)

    dc_even = dc_out_even * cos_val + dc_out_odd * sin_val
    dc_odd = -dc_out_even * sin_val + dc_out_odd * cos_val

    dc_base = pid_b * stride_dc_batch + pid_g * stride_dc_group
    dc_even_ptrs = dc_ptr + dc_base + offs_l[:, None] * stride_dc_seqlen + offs_d[None, :] * stride_dc_dstate
    dc_odd_ptrs = dc_ptr + dc_base + offs_l[:, None] * stride_dc_seqlen + (offs_d[None, :] + half_d) * stride_dc_dstate
    tl.atomic_add(dc_even_ptrs, dc_even.to(dc_ptr.dtype.element_ty), mask=mask_ld)
    tl.atomic_add(dc_odd_ptrs, dc_odd.to(dc_ptr.dtype.element_ty), mask=mask_ld)

    # --- dtheta computation ---
    # dtheta_cumsum = (-B_out_odd)*dB_out_even + B_out_even*dB_out_odd
    #               + (-C_out_odd)*dC_out_even + C_out_even*dC_out_odd
    bh_base = pid_b * stride_bh_batch + pid_h * stride_bh_head
    b_out_even = tl.load(b_heads_ptr + bh_base + offs_l[:, None] * stride_bh_seqlen + offs_d[None, :] * stride_bh_dstate,
                         mask=mask_ld, other=0.0).to(tl.float32)
    b_out_odd = tl.load(b_heads_ptr + bh_base + offs_l[:, None] * stride_bh_seqlen + (offs_d[None, :] + half_d) * stride_bh_dstate,
                        mask=mask_ld, other=0.0).to(tl.float32)

    ch_base = pid_b * stride_ch_batch + pid_h * stride_ch_head
    c_out_even = tl.load(c_heads_ptr + ch_base + offs_l[:, None] * stride_ch_seqlen + offs_d[None, :] * stride_ch_dstate,
                         mask=mask_ld, other=0.0).to(tl.float32)
    c_out_odd = tl.load(c_heads_ptr + ch_base + offs_l[:, None] * stride_ch_seqlen + (offs_d[None, :] + half_d) * stride_ch_dstate,
                        mask=mask_ld, other=0.0).to(tl.float32)

    dtheta_cs = ((-b_out_odd) * db_out_even + b_out_even * db_out_odd
                 + (-c_out_odd) * dc_out_even + c_out_even * dc_out_odd)

    dth_base = pid_b * stride_dth_batch + pid_h * stride_dth_head
    dth_ptrs = dtheta_ptr + dth_base + offs_l[:, None] * stride_dth_seqlen + offs_d[None, :] * stride_dth_halfd
    tl.store(dth_ptrs, dtheta_cs.to(dtheta_ptr.dtype.element_ty), mask=mask_ld)


def _mamba3_rope_bwd(dB_heads, dC_heads, B_heads, C_heads, theta_cumsum, ngroups):
    """Backward through RoPE.

    Args:
        dB_heads: (batch, seqlen, nheads, dstate) -- gradient of rotated B
        dC_heads: (batch, seqlen, nheads, dstate) -- gradient of rotated C
        B_heads: (batch, seqlen, nheads, dstate) -- forward output (rotated B)
        C_heads: (batch, seqlen, nheads, dstate) -- forward output (rotated C)
        theta_cumsum: (batch, seqlen, nheads, dstate//2) -- cumulative theta from forward
        ngroups: int

    Returns:
        dB: (batch, seqlen, ngroups, dstate)
        dC: (batch, seqlen, ngroups, dstate)
        dtheta: (batch, seqlen, nheads, dstate//2)
    """
    batch, seqlen, nheads, dstate = dB_heads.shape
    assert dC_heads.shape == dB_heads.shape
    assert B_heads.shape == dB_heads.shape
    assert C_heads.shape == dB_heads.shape
    half_d = dstate // 2
    assert theta_cumsum.shape == (batch, seqlen, nheads, half_d)
    assert nheads % ngroups == 0
    nheads_per_group = nheads // ngroups

    cos_theta = torch.cos(theta_cumsum)
    sin_theta = torch.sin(theta_cumsum)

    # Allocate outputs -- dB and dC are zero-initialized for atomic_add
    dB = torch.zeros(batch, seqlen, ngroups, dstate, device=dB_heads.device, dtype=torch.float32)
    dC = torch.zeros(batch, seqlen, ngroups, dstate, device=dC_heads.device, dtype=torch.float32)
    dtheta_cumsum = torch.empty(batch, seqlen, nheads, half_d, device=dB_heads.device, dtype=torch.float32)

    BLOCK_L = min(triton.next_power_of_2(seqlen), 128)
    BLOCK_D = triton.next_power_of_2(half_d)

    grid = (batch, triton.cdiv(seqlen, BLOCK_L), nheads)

    with torch.cuda.device(dB_heads.device.index):
        _mamba3_rope_bwd_kernel[grid](
            dB_heads, dC_heads,
            B_heads, C_heads,
            cos_theta, sin_theta,
            dB, dC, dtheta_cumsum,
            batch, seqlen, nheads, ngroups, dstate, half_d,
            nheads_per_group,
            # dB_heads strides
            dB_heads.stride(0), dB_heads.stride(1), dB_heads.stride(2), dB_heads.stride(3),
            # dC_heads strides
            dC_heads.stride(0), dC_heads.stride(1), dC_heads.stride(2), dC_heads.stride(3),
            # B_heads strides
            B_heads.stride(0), B_heads.stride(1), B_heads.stride(2), B_heads.stride(3),
            # C_heads strides
            C_heads.stride(0), C_heads.stride(1), C_heads.stride(2), C_heads.stride(3),
            # cos/sin strides
            cos_theta.stride(0), cos_theta.stride(1), cos_theta.stride(2), cos_theta.stride(3),
            # dB strides
            dB.stride(0), dB.stride(1), dB.stride(2), dB.stride(3),
            # dC strides
            dC.stride(0), dC.stride(1), dC.stride(2), dC.stride(3),
            # dtheta strides
            dtheta_cumsum.stride(0), dtheta_cumsum.stride(1), dtheta_cumsum.stride(2), dtheta_cumsum.stride(3),
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D,
        )

    # Reverse cumsum to get dtheta from dtheta_cumsum:
    # dtheta[t] = sum_{s>=t} dtheta_cumsum[s]
    # = flip(cumsum(flip(dtheta_cumsum, dim=1), dim=1), dim=1)
    dtheta = dtheta_cumsum.flip(1).cumsum(1).flip(1)

    return dB, dC, dtheta
