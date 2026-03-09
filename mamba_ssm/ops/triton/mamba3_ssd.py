# Copyright (c) 2024, Tri Dao, Albert Gu.
# Mamba-3 SSD operations: chunked parallel forward + Triton decode kernel.
#
# Chunked parallel SSD with exponential-trapezoidal discretization:
#   h_t = α_t * h_{t-1} + β_t * B_{t-1} * x_{t-1} + γ_t * B_t * x_t
#
# Strategy:
# 1. Pre-convolve the state-input: v_t = γ_t * B_t ⊗ x_t + β_t * B_{t-1} ⊗ x_{t-1}
#    Then the recurrence is h_t = α_t * h_{t-1} + v_t (standard linear recurrence).
# 2. Apply RoPE to B, C before chunked computation (doesn't affect kernel structure).
# 3. Use the SSD chunked algorithm (matmuls within chunks, sequential across chunks).

import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    import triton
    import triton.language as tl
    from mamba_ssm.ops.triton.softplus import softplus
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ============================================================================
# Chunked Parallel SSD for Mamba-3 (PyTorch reference, differentiable)
# ============================================================================

def segsum(x):
    """Stable segment sum for causal decay matrix."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def mamba3_ssd_chunked(
    X, dt, A, B, C,
    block_len,
    gamma=None,
    beta=None,
    D=None,
    z=None,
    initial_states=None,
    return_final_states=False,
    initial_prev_Bx=None,
    seq_idx=None,
):
    """
    Chunked parallel SSD for Mamba-3 with exponential-trapezoidal discretization.

    Supports both SISO and MIMO. For MIMO, B/C/X have an extra trailing rank dimension.

    Arguments:
        X: (batch, length, n_heads, d_head[, mimo_rank])
        dt: (batch, length, n_heads)
        A: (n_heads,) -- negative SSM eigenvalues
        B: (batch, length, n_heads, d_state[, mimo_rank])
        C: (batch, length, n_heads, d_state[, mimo_rank])
        block_len: int -- chunk size
        gamma: (batch, length, n_heads) or None -- trapezoidal current weight (λ * dt)
        beta: (batch, length, n_heads) or None -- trapezoidal lookback weight ((1-λ) * dt * α)
        D: (n_heads,) or (n_heads, d_head) or None -- skip connection
        initial_states: (batch, n_heads, d_head, d_state) or None
        return_final_states: bool
        initial_prev_Bx: (batch, n_heads, d_head, d_state) or None -- prev B*x for trapezoidal t=0
        seq_idx: (batch, length) int or None -- document indices for packed training
    Return:
        Y: (batch, length, n_heads, d_head)
        final_state: (batch, n_heads, d_head, d_state) if return_final_states
    """
    is_mimo = B.dim() == 5
    batch, seqlen, nheads, headdim = X.shape[:4]
    mimo_rank = X.shape[4] if is_mimo else 0
    dstate = B.shape[-2] if is_mimo else B.shape[-1]

    assert seqlen % block_len == 0
    nchunks = seqlen // block_len
    out_dtype = X.dtype  # preserve original dtype for output

    # Cast to float32 for numerical stability in matmuls
    X = X.float()
    B = B.float()
    C = C.float()
    dt = dt.float()
    if gamma is not None:
        gamma = gamma.float()
    if beta is not None:
        beta = beta.float()

    # Compute dA = dt * A per timestep
    dA = dt * A.float().view(1, 1, nheads)  # (batch, seqlen, nheads)

    # Prepare seq_idx masks for cross-document boundary handling
    has_seq_idx = seq_idx is not None
    if has_seq_idx:
        seq_idx_c = rearrange(seq_idx, "b (c l) -> b c l", l=block_len)

    # Reshape into chunks
    if is_mimo:
        X_c = rearrange(X, "b (c l) h p r -> b c l h p r", l=block_len)
        B_c = rearrange(B, "b (c l) h n r -> b c l h n r", l=block_len)
        C_c = rearrange(C, "b (c l) h n r -> b c l h n r", l=block_len)
    else:
        X_c = rearrange(X, "b (c l) h p -> b c l h p", l=block_len)
        B_c = rearrange(B, "b (c l) h n -> b c l h n", l=block_len)
        C_c = rearrange(C, "b (c l) h n -> b c l h n", l=block_len)
    dA_c = rearrange(dA, "b (c l) h -> b h c l", l=block_len)
    dt_c = rearrange(dt, "b (c l) h -> b c l h", l=block_len)

    # Cumsum of dA within each chunk
    dA_cumsum = torch.cumsum(dA_c, dim=-1)  # (batch, nheads, nchunks, block_len)

    # For SISO, C_c is used directly. For MIMO, we compute per output rank.

    # === Handle trapezoidal convolution on state-input ===
    use_trapezoidal = gamma is not None and beta is not None

    if use_trapezoidal:
        gamma_c = rearrange(gamma, "b (c l) h -> b c l h", l=block_len)
        beta_c = rearrange(beta, "b (c l) h -> b c l h", l=block_len)

        # Shifted B and X for the lookback term (shift by 1 within each chunk)
        B_shifted = torch.zeros_like(B_c)
        X_shifted = torch.zeros_like(X_c)
        B_shifted[:, :, 1:] = B_c[:, :, :-1]
        X_shifted[:, :, 1:] = X_c[:, :, :-1]
        # Cross-chunk boundary: position 0 of chunk c gets last position of chunk c-1
        B_shifted[:, 1:, 0] = B_c[:, :-1, -1]
        X_shifted[:, 1:, 0] = X_c[:, :-1, -1]

        # Mask shifted values at document boundaries (no lookback across documents)
        if has_seq_idx:
            # shift_valid[b,c,t] = True if position t can look back to t-1
            shift_valid = torch.ones(batch, nchunks, block_len, dtype=torch.bool, device=X.device)
            shift_valid[:, :, 1:] = seq_idx_c[:, :, 1:] == seq_idx_c[:, :, :-1]
            shift_valid[:, 1:, 0] = seq_idx_c[:, 1:, 0] == seq_idx_c[:, :-1, -1]
            shift_valid[:, 0, 0] = False  # no lookback for very first position
            if is_mimo:
                sv = shift_valid[:, :, :, None, None, None]
            else:
                sv = shift_valid[:, :, :, None, None]
            B_shifted = B_shifted * sv
            X_shifted = X_shifted * sv

        # Streaming: position 0 of chunk 0 from initial_prev_Bx
        # We can't split prev_Bx into separate B and x, so handle as state correction below
    else:
        gamma_c = dt_c  # Fall back to Euler: γ = dt
        beta_c = None

    # === 1. Intra-chunk computation (diagonal blocks) ===
    L = torch.exp(segsum(dA_c))  # (batch, nheads, nchunks, block_len, block_len)

    # Mask L for cross-document boundaries: L[i,j]=0 when seq_idx[i] != seq_idx[j]
    if has_seq_idx:
        seq_mask = seq_idx_c[:, :, :, None] == seq_idx_c[:, :, None, :]  # (b, c, l, l)
        L = L * rearrange(seq_mask.float(), "b c l s -> b 1 c l s")

    gamma_scale = rearrange(gamma_c, "b c l h -> b h c 1 l")

    if is_mimo:
        # Per output rank: Y[r_out] = Σ_{r_in} L * γ * C[r_out]^T B[r_in] * X[r_in]
        Y_diag = torch.zeros(batch, nchunks, block_len, nheads, headdim, mimo_rank,
                             device=X.device, dtype=X.dtype)
        for r_out in range(mimo_rank):
            for r_in in range(mimo_rank):
                CB_r = torch.einsum("bclhn,bcshn->bhcls", C_c[..., r_out], B_c[..., r_in])
                Y_diag[..., r_out] = Y_diag[..., r_out] + torch.einsum(
                    "bhcls,bhcls,bcshp->bclhp",
                    L * gamma_scale, CB_r, X_c[..., r_in],
                )
        if use_trapezoidal:
            beta_scale = rearrange(beta_c, "b c l h -> b h c 1 l")
            for r_out in range(mimo_rank):
                for r_in in range(mimo_rank):
                    CB_shifted_r = torch.einsum("bclhn,bcshn->bhcls", C_c[..., r_out], B_shifted[..., r_in])
                    Y_diag[..., r_out] = Y_diag[..., r_out] + torch.einsum(
                        "bhcls,bhcls,bcshp->bclhp",
                        L * beta_scale, CB_shifted_r, X_shifted[..., r_in],
                    )
    else:
        CB = torch.einsum("bclhn,bcshn->bhcls", C_c, B_c)
        Y_diag = torch.einsum(
            "bhcls,bhcls,bcshp->bclhp",
            L * gamma_scale, CB, X_c,
        )
        if use_trapezoidal:
            CB_shifted = torch.einsum("bclhn,bcshn->bhcls", C_c, B_shifted)
            beta_scale = rearrange(beta_c, "b c l h -> b h c 1 l")
            Y_diag = Y_diag + torch.einsum(
                "bhcls,bhcls,bcshp->bclhp",
                L * beta_scale, CB_shifted, X_shifted,
            )

    # === 2. Per-chunk state computation ===
    decay_states = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)  # (batch, nheads, nchunks, block_len)

    # Mask: only accumulate state from tokens in same document as chunk's last token
    if has_seq_idx:
        state_doc_mask = (seq_idx_c == seq_idx_c[:, :, -1:]).float()  # (b, c, l)
        decay_states = decay_states * rearrange(state_doc_mask, "b c l -> b 1 c l")

    gamma_decay = decay_states * rearrange(gamma_c, "b c l h -> b h c l")

    if is_mimo:
        states = torch.zeros(batch, nchunks, nheads, headdim, dstate,
                             device=X.device, dtype=torch.float32)
        for r in range(mimo_rank):
            states = states + torch.einsum(
                "bclhn,bhcl,bclhp->bchpn",
                B_c[..., r].float(), gamma_decay, X_c[..., r].float(),
            )
        if use_trapezoidal:
            beta_decay = decay_states * rearrange(beta_c, "b c l h -> b h c l")
            for r in range(mimo_rank):
                states = states + torch.einsum(
                    "bclhn,bhcl,bclhp->bchpn",
                    B_shifted[..., r].float(), beta_decay, X_shifted[..., r].float(),
                )
    else:
        states = torch.einsum(
            "bclhn,bhcl,bclhp->bchpn",
            B_c, gamma_decay, X_c,
        )
        if use_trapezoidal:
            beta_decay = decay_states * rearrange(beta_c, "b c l h -> b h c l")
            states_trap = torch.einsum(
                "bclhn,bhcl,bclhp->bchpn",
                B_shifted, beta_decay, X_shifted,
            )
            states = states + states_trap

    # === Handle initial_prev_Bx correction ===
    # At t=0, trapezoidal adds β_0 * initial_prev_Bx to state. This was missed by
    # B_shifted/X_shifted (which are zero at position 0 of chunk 0).
    # We add the correction both to the state and the output.
    if initial_prev_Bx is not None and use_trapezoidal:
        beta_0 = beta[:, 0, :]  # (batch, nheads)
        correction = rearrange(beta_0, "b h -> b h 1 1") * initial_prev_Bx.float()
        # Decay correction from position 0 to end of chunk 0
        decay_from_0 = decay_states[:, :, 0, 0]
        states[:, 0] = states[:, 0] + rearrange(decay_from_0, "b h -> b h 1 1") * correction
        # Output correction within chunk 0
        decay_0_to_m = torch.exp(dA_cumsum[:, :, 0, :] - dA_cumsum[:, :, 0, 0:1])  # (b, h, block_len)
        if is_mimo:
            for r_out in range(mimo_rank):
                Y_corr = torch.einsum(
                    "bclhn,bhpn,bhcl->bclhp",
                    C_c[:, 0:1, :, :, :, r_out], correction.to(C_c.dtype), decay_0_to_m.unsqueeze(2),
                )
                Y_diag[:, 0:1, :, :, :, r_out] = Y_diag[:, 0:1, :, :, :, r_out] + Y_corr
        else:
            Y_correction = torch.einsum(
                "bclhn,bhpn,bhcl->bclhp",
                C_c[:, 0:1], correction.to(C_c.dtype), decay_0_to_m.unsqueeze(2),
            )
            Y_diag[:, 0:1] = Y_diag[:, 0:1] + Y_correction

    # === 3. Inter-chunk recurrence ===
    if initial_states is None:
        initial_states_flat = torch.zeros(
            batch, nheads, headdim * dstate, device=X.device, dtype=torch.float32,
        )
    else:
        initial_states_flat = rearrange(initial_states.float(), "b h p n -> b h (p n)")

    states_flat = rearrange(states.float(), "b c h p n -> b c h (p n)")

    # Total decay per chunk
    dA_chunk_cumsum = dA_cumsum[:, :, :, -1]  # (batch, nheads, nchunks)

    # Mask inter-chunk propagation at document boundaries
    if has_seq_idx:
        # Compare last token of each chunk with last token of previous chunk
        chunk_end_idx = seq_idx_c[:, :, -1]  # (b, c)
        # chunk_same[b, c] = True if chunk c's last token is same doc as chunk c-1's last token
        chunk_same = torch.ones(batch, nchunks, dtype=torch.bool, device=X.device)
        chunk_same[:, 1:] = chunk_end_idx[:, 1:] == chunk_end_idx[:, :-1]
        chunk_same[:, 0] = initial_states is not None  # propagate initial state only if provided
        chunk_propagate = rearrange(chunk_same.float(), "b c -> b 1 c")  # (b, 1, c) for heads

    # Sequential scan across chunks
    all_states = []
    prev_state = initial_states_flat
    for c in range(nchunks):
        scale = torch.exp(dA_chunk_cumsum[:, :, c]).unsqueeze(-1)  # (batch, nheads, 1)
        if has_seq_idx:
            scale = scale * chunk_propagate[:, :, c].unsqueeze(-1)
        prev_state = scale * prev_state + states_flat[:, c]
        all_states.append(prev_state)

    # Propagated states at chunk boundaries
    boundary_states = [initial_states_flat] + all_states[:-1]
    boundary_states = torch.stack(boundary_states, dim=1)  # (batch, nchunks, nheads, headdim*dstate)
    boundary_states = rearrange(boundary_states, "b c h (p n) -> b c h p n", p=headdim, n=dstate)

    final_state = rearrange(all_states[-1], "b h (p n) -> b h p n", p=headdim, n=dstate)

    # === 4. Inter-chunk output (off-diagonal blocks) ===
    state_decay_out = torch.exp(dA_cumsum)  # (batch, nheads, nchunks, block_len)

    # Mask: only apply boundary state to positions in same document as chunk start
    if has_seq_idx:
        # For each position in a chunk, check if it's in the same document as
        # the boundary state (which was propagated from the previous chunk's last token)
        # The boundary state represents the document at the END of the previous chunk.
        # We need: seq_idx at each position == seq_idx at start of current doc segment in this chunk.
        # Simpler correct approach: mask positions where seq_idx differs from chunk's first token
        # of the same document run that includes the boundary.
        # Actually, the correct mask is: state_decay_out[pos] = 0 if pos belongs to a different
        # document than the boundary state. The boundary state is from the previous chunk's last token.
        boundary_doc = torch.zeros(batch, nchunks, dtype=seq_idx.dtype, device=seq_idx.device)
        boundary_doc[:, 0] = seq_idx_c[:, 0, 0]  # initial state's document (or first token)
        boundary_doc[:, 1:] = seq_idx_c[:, :-1, -1]  # previous chunk's last token
        # mask[b,c,t] = (seq_idx_c[b,c,t] == boundary_doc[b,c])
        off_diag_mask = (seq_idx_c == boundary_doc[:, :, None]).float()
        state_decay_out = state_decay_out * rearrange(off_diag_mask, "b c l -> b 1 c l")
    if is_mimo:
        Y_off = torch.zeros(batch, nchunks, block_len, nheads, headdim, mimo_rank,
                            device=X.device, dtype=X.dtype)
        for r_out in range(mimo_rank):
            Y_off[..., r_out] = torch.einsum(
                "bclhn,bchpn,bhcl->bclhp",
                C_c[..., r_out], boundary_states.to(C_c.dtype), state_decay_out,
            )
    else:
        Y_off = torch.einsum(
            "bclhn,bchpn,bhcl->bclhp",
            C_c, boundary_states.to(C_c.dtype), state_decay_out,
        )

    # === Combine ===
    if is_mimo:
        Y = rearrange(Y_diag + Y_off, "b c l h p r -> b (c l) h p r")
    else:
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    # D skip connection
    if D is not None:
        if is_mimo:
            if D.dim() == 1:
                Y = Y + X * rearrange(D, "h -> 1 1 h 1 1")
            else:
                Y = Y + X * rearrange(D, "h p -> 1 1 h p 1")
        else:
            if D.dim() == 1:
                Y = Y + X * rearrange(D, "h -> 1 1 h 1")
            else:
                Y = Y + X * rearrange(D, "h p -> 1 1 h p")

    # z gating (SiLU)
    if z is not None:
        z = z.float()
        Y = Y * F.silu(z)

    # Cast back to original dtype
    Y = Y.to(out_dtype)

    if return_final_states:
        return Y, final_state
    return Y


def apply_rotary_emb_to_bc(B, C, theta, nheads, ngroups):
    """Apply cumulative data-dependent RoPE to B and C.

    RoPE is applied AFTER expanding B, C from groups to heads so that each head
    gets its own rotation (matching the reference recurrence). This is called
    before group→head expansion in mamba3_chunk_scan_combined, so we expand
    internally, apply per-head RoPE, then return at head level.

    Args:
        B: (batch, seqlen, ngroups, d_state) or (..., d_state, mimo_rank)
        C: same as B
        theta: (batch, seqlen, nheads, d_state//2) -- per-step rotation angles
        nheads: int
        ngroups: int
    Returns:
        B_rot, C_rot at head level: (batch, seqlen, nheads, d_state[, mimo_rank])
    """
    if theta is None:
        return B, C

    batch, seqlen = theta.shape[:2]
    is_mimo = B.dim() == 5
    dstate = B.shape[-2] if is_mimo else B.shape[-1]
    nheads_per_group = nheads // ngroups
    half_d = dstate // 2

    # Expand B, C from groups to heads BEFORE applying RoPE
    if is_mimo:
        B = repeat(B, "b l g n r -> b l (g h) n r", h=nheads_per_group)
        C = repeat(C, "b l g n r -> b l (g h) n r", h=nheads_per_group)
    else:
        B = repeat(B, "b l g n -> b l (g h) n", h=nheads_per_group)
        C = repeat(C, "b l g n -> b l (g h) n", h=nheads_per_group)

    # Cumulative sum of per-head angles
    theta_cumsum = torch.cumsum(theta, dim=1)  # (batch, seqlen, nheads, dstate//2)
    cos_h = torch.cos(theta_cumsum)  # (batch, seqlen, nheads, dstate//2)
    sin_h = torch.sin(theta_cumsum)

    if is_mimo:
        # B: (batch, seqlen, nheads, d_state, mimo_rank)
        B1, B2 = B[..., :half_d, :], B[..., half_d:, :]
        B_rot = torch.cat([
            B1 * cos_h.unsqueeze(-1) - B2 * sin_h.unsqueeze(-1),
            B1 * sin_h.unsqueeze(-1) + B2 * cos_h.unsqueeze(-1),
        ], dim=-2)
        C1, C2 = C[..., :half_d, :], C[..., half_d:, :]
        C_rot = torch.cat([
            C1 * cos_h.unsqueeze(-1) - C2 * sin_h.unsqueeze(-1),
            C1 * sin_h.unsqueeze(-1) + C2 * cos_h.unsqueeze(-1),
        ], dim=-2)
    else:
        # B: (batch, seqlen, nheads, d_state)
        B1, B2 = B[..., :half_d], B[..., half_d:]
        B_rot = torch.cat([B1 * cos_h - B2 * sin_h, B1 * sin_h + B2 * cos_h], dim=-1)
        C1, C2 = C[..., :half_d], C[..., half_d:]
        C_rot = torch.cat([C1 * cos_h - C2 * sin_h, C1 * sin_h + C2 * cos_h], dim=-1)

    return B_rot, C_rot


def mamba3_chunk_scan_combined(
    x, dt, A, B, C,
    chunk_size,
    gamma=None,
    beta=None,
    theta=None,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    initial_states=None,
    initial_prev_Bx=None,
    return_final_states=False,
    ngroups=1,
    seq_idx=None,
):
    """
    Combined chunked SSD for Mamba-3.

    Args:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads,) -- negative
        B: (batch, seqlen, ngroups, d_state)
        C: (batch, seqlen, ngroups, d_state)
        chunk_size: int
        gamma: (batch, seqlen, nheads) -- λ * dt (trapezoidal current weight)
        beta: (batch, seqlen, nheads) -- (1-λ) * dt * exp(dt*A) (trapezoidal lookback weight)
        theta: (batch, seqlen, nheads, d_state//2) -- RoPE angles
        D, z, dt_bias, dt_softplus, dt_limit: same as mamba_chunk_scan_combined
        initial_states: (batch, nheads, headdim, d_state)
        return_final_states: bool
        ngroups: int
    """
    batch, seqlen, nheads, headdim = x.shape[:4]
    is_mimo = B.dim() == 5
    dstate = B.shape[-2] if is_mimo else B.shape[-1]
    nheads_per_group = nheads // ngroups

    # Process dt
    if dt_bias is not None:
        dt = dt + dt_bias.view(1, 1, nheads)
    if dt_softplus:
        dt = F.softplus(dt)
    if dt_limit != (0.0, float("inf")):
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])

    # Apply RoPE to B, C before chunked computation
    # apply_rotary_emb_to_bc expands from groups→heads internally (per-head RoPE)
    if theta is not None:
        B, C = apply_rotary_emb_to_bc(B, C, theta, nheads, ngroups)
        is_mimo = B.dim() == 5  # refresh after expansion
    else:
        # No RoPE: expand B, C from groups to heads
        is_mimo = B.dim() == 5
        if is_mimo:
            B = repeat(B, "b l g n r -> b l (g h) n r", h=nheads_per_group)
            C = repeat(C, "b l g n r -> b l (g h) n r", h=nheads_per_group)
        else:
            B = repeat(B, "b l g n -> b l (g h) n", h=nheads_per_group)
            C = repeat(C, "b l g n -> b l (g h) n", h=nheads_per_group)

    # Pad sequence to multiple of chunk_size
    pad_len = (chunk_size - seqlen % chunk_size) % chunk_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len)) if not is_mimo else \
            F.pad(x, (0, 0, 0, 0, 0, 0, 0, pad_len))
        dt = F.pad(dt, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len)) if not is_mimo else \
            F.pad(B, (0, 0, 0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len)) if not is_mimo else \
            F.pad(C, (0, 0, 0, 0, 0, 0, 0, pad_len))
        if gamma is not None:
            gamma = F.pad(gamma, (0, 0, 0, pad_len))
        if beta is not None:
            beta = F.pad(beta, (0, 0, 0, pad_len))
        if z is not None:
            z = F.pad(z, (0, 0, 0, 0, 0, pad_len)) if not is_mimo else \
                F.pad(z, (0, 0, 0, 0, 0, 0, 0, pad_len))
        if seq_idx is not None:
            # Pad with -1 so padded positions are never equal to real doc indices
            seq_idx = F.pad(seq_idx, (0, pad_len), value=-1)

    result = mamba3_ssd_chunked(
        x, dt, A, B, C,
        block_len=chunk_size,
        gamma=gamma,
        beta=beta,
        D=D,
        z=z,
        initial_states=initial_states,
        return_final_states=return_final_states,
        initial_prev_Bx=initial_prev_Bx,
        seq_idx=seq_idx,
    )

    # Un-pad
    if pad_len > 0:
        if return_final_states:
            Y, final_state = result
            Y = Y[:, :seqlen]
            result = (Y, final_state)
        else:
            result = result[:, :seqlen]

    return result


# ============================================================================
# Triton kernel for Mamba-3 single-step decode
# ============================================================================
# LIMITATION: This kernel operates on B, C at group level without RoPE, BCNorm,
# or BC bias. It is a low-level primitive — the caller must pre-process B, C
# (apply norm, expand to heads, apply bias, apply RoPE) before calling.
# The step() method in mamba3.py handles this correctly in PyTorch.
# Supports both SISO and MIMO decode (per-rank B, C, X for MIMO).
# ============================================================================

if HAS_TRITON:

    @triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
    @triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
    @triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
    @triton.heuristics({"HAS_PREV_BX": lambda args: args["prev_Bx_ptr"] is not None})
    @triton.heuristics({"HAS_BETA": lambda args: args["beta_ptr"] is not None})
    @triton.heuristics({"HAS_GAMMA": lambda args: args["gamma_ptr"] is not None})
    @triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
    @triton.heuristics({"IS_MIMO": lambda args: args["mimo_rank"] > 0})
    @triton.heuristics({"MIMO_RANK": lambda args: args["mimo_rank"]})
    @triton.jit
    def _mamba3_state_update_kernel(
        # Pointers
        state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr,
        B_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
        prev_Bx_ptr, beta_ptr, gamma_ptr,
        # Dims
        batch, nheads, dim, dstate, nheads_ngroups_ratio, mimo_rank,
        # Strides
        stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
        stride_x_batch, stride_x_head, stride_x_dim,
        stride_dt_batch, stride_dt_head,
        stride_A_head,
        stride_B_batch, stride_B_group, stride_B_dstate,
        stride_C_batch, stride_C_group, stride_C_dstate,
        stride_D_head,
        stride_z_batch, stride_z_head, stride_z_dim,
        stride_out_batch, stride_out_head, stride_out_dim,
        stride_prev_Bx_batch, stride_prev_Bx_head, stride_prev_Bx_dim, stride_prev_Bx_dstate,
        stride_beta_batch, stride_beta_head,
        stride_gamma_batch, stride_gamma_head,
        # MIMO strides (only used when IS_MIMO)
        stride_x_rank, stride_B_rank, stride_C_rank, stride_out_rank,
        # Meta
        DT_SOFTPLUS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        HAS_DT_BIAS: tl.constexpr,
        HAS_D: tl.constexpr,
        HAS_Z: tl.constexpr,
        HAS_PREV_BX: tl.constexpr,
        HAS_BETA: tl.constexpr,
        HAS_GAMMA: tl.constexpr,
        BLOCK_SIZE_DSTATE: tl.constexpr,
        IS_MIMO: tl.constexpr,
        MIMO_RANK: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_b = tl.program_id(axis=1)
        pid_h = tl.program_id(axis=2)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)

        # Load dt and A (scalar per head)
        dt = tl.load(dt_ptr + pid_b * stride_dt_batch + pid_h * stride_dt_head).to(tl.float32)
        if HAS_DT_BIAS:
            dt_bias_stride = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
            dt += dt_bias_stride
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, softplus(dt), dt)

        A = tl.load(A_ptr + pid_h * stride_A_head).to(tl.float32)
        dA = tl.exp(A * dt)  # decay uses original dt

        # Load gamma (input scaling) — separate from dt for trapezoidal
        if HAS_GAMMA:
            input_scale = tl.load(gamma_ptr + pid_b * stride_gamma_batch + pid_h * stride_gamma_head).to(tl.float32)
        else:
            input_scale = dt  # Euler mode: gamma = dt

        # Load state
        state_ptr_base = state_ptr + pid_b * stride_state_batch + pid_h * stride_state_head
        state_ptrs = state_ptr_base + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)

        # Load x, B and compute Bx (unscaled) and dBx (scaled by input_scale)
        x_ptr_base = x_ptr + pid_b * stride_x_batch + pid_h * stride_x_head
        B_ptr_base = B_ptr + pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
        C_ptr_base = C_ptr + pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group

        if IS_MIMO:
            # MIMO: Bx = Σ_r x[m,r] * B[n,r], summed over rank
            Bx_unscaled = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_DSTATE], dtype=tl.float32)
            for r in range(MIMO_RANK):
                x_r = tl.load(x_ptr_base + offs_m * stride_x_dim + r * stride_x_rank,
                              mask=offs_m < dim, other=0.0).to(tl.float32)
                B_r = tl.load(B_ptr_base + offs_n * stride_B_dstate + r * stride_B_rank,
                              mask=offs_n < dstate, other=0.0).to(tl.float32)
                Bx_unscaled += x_r[:, None] * B_r[None, :]
            dBx = Bx_unscaled * input_scale
        else:
            # SISO: dBx = input_scale * B * x
            x = tl.load(x_ptr_base + offs_m * stride_x_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
            B = tl.load(B_ptr_base + offs_n * stride_B_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)
            dBx = B[None, :] * input_scale * x[:, None]

        # State update: h = dA * h + dBx
        state = state * dA + dBx

        # Add trapezoidal lookback: + beta * prev_Bx
        if HAS_PREV_BX and HAS_BETA:
            beta = tl.load(beta_ptr + pid_b * stride_beta_batch + pid_h * stride_beta_head).to(tl.float32)
            prev_Bx_ptrs = (prev_Bx_ptr + pid_b * stride_prev_Bx_batch + pid_h * stride_prev_Bx_head
                            + offs_m[:, None] * stride_prev_Bx_dim + offs_n[None, :] * stride_prev_Bx_dstate)
            prev_Bx = tl.load(prev_Bx_ptrs,
                               mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
            state = state + beta * prev_Bx

        # Store updated state
        tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))

        # Store current Bx (unscaled) for next step's trapezoidal lookback
        if HAS_PREV_BX:
            prev_Bx_ptrs_s = (prev_Bx_ptr + pid_b * stride_prev_Bx_batch + pid_h * stride_prev_Bx_head
                              + offs_m[:, None] * stride_prev_Bx_dim + offs_n[None, :] * stride_prev_Bx_dstate)
            if IS_MIMO:
                tl.store(prev_Bx_ptrs_s, Bx_unscaled,
                         mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
            else:
                raw_Bx = B[None, :] * x[:, None]
                tl.store(prev_Bx_ptrs_s, raw_Bx,
                         mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))

        # Output
        out_ptr_base = out_ptr + pid_b * stride_out_batch + pid_h * stride_out_head
        if IS_MIMO:
            # Per-rank output: y[p,r] = Σ_n state[p,n] * C[n,r]
            for r in range(MIMO_RANK):
                C_r = tl.load(C_ptr_base + offs_n * stride_C_dstate + r * stride_C_rank,
                              mask=offs_n < dstate, other=0.0).to(tl.float32)
                out_r = tl.sum(state * C_r[None, :], axis=1)
                tl.store(out_ptr_base + offs_m * stride_out_dim + r * stride_out_rank,
                         out_r, mask=offs_m < dim)
        else:
            C = tl.load(C_ptr_base + offs_n * stride_C_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)
            out = tl.sum(state * C[None, :], axis=1)
            if HAS_D:
                D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
                out += x * D
            if HAS_Z:
                z = tl.load(z_ptr + pid_b * stride_z_batch + pid_h * stride_z_head + offs_m * stride_z_dim,
                            mask=offs_m < dim, other=0.0).to(tl.float32)
                out *= z * tl.sigmoid(z)
            tl.store(out_ptr_base + offs_m * stride_out_dim, out, mask=offs_m < dim)


def mamba3_state_update(
    state, x, dt, A, B, C,
    D=None, z=None, dt_bias=None, dt_softplus=False,
    prev_Bx=None, beta=None, gamma=None,
):
    """
    Mamba-3 single-step decode with fused Triton kernel. Supports SISO and MIMO.

    Args:
        state: (batch, nheads, dim, dstate)
        x: (batch, nheads, dim) for SISO, (batch, nheads, dim, mimo_rank) for MIMO
        dt: (batch, nheads)
        A: (nheads,)
        B: (batch, ngroups, dstate) for SISO, (batch, ngroups, dstate, mimo_rank) for MIMO
        C: (batch, ngroups, dstate) for SISO, (batch, ngroups, dstate, mimo_rank) for MIMO
        D: (nheads,) or None -- NOT applied for MIMO (handled outside)
        z: (batch, nheads, dim) or None -- NOT applied for MIMO (handled outside)
        dt_bias: (nheads,) or None
        dt_softplus: bool
        prev_Bx: (batch, nheads, dim, dstate) or None
        beta: (batch, nheads) or None
        gamma: (batch, nheads) or None
    Returns:
        out: (batch, nheads, dim) for SISO, (batch, nheads, dim, mimo_rank) for MIMO
    """
    is_mimo = x.dim() == 4
    mr = x.shape[3] if is_mimo else 0

    if not HAS_TRITON:
        return _mamba3_state_update_ref(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus, prev_Bx, beta, gamma)

    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    assert nheads % ngroups == 0

    if is_mimo:
        out = torch.empty((batch, nheads, dim, mr), device=x.device, dtype=x.dtype)
    else:
        out = torch.empty((batch, nheads, dim), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)
    BLOCK_SIZE_M = 8 if dstate <= 32 else (4 if dstate <= 128 else 2)

    with torch.cuda.device(x.device.index):
        _mamba3_state_update_kernel[grid](
            state, x, dt, dt_bias, A, B, C, D, z, out,
            prev_Bx, beta, gamma,
            batch, nheads, dim, dstate, nheads // ngroups, mr,
            # state strides
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            # x strides
            x.stride(0), x.stride(1), x.stride(2),
            # dt strides
            dt.stride(0), dt.stride(1),
            # A stride
            A.stride(0),
            # B strides
            B.stride(0), B.stride(1), B.stride(2),
            # C strides
            C.stride(0), C.stride(1), C.stride(2),
            # D stride
            D.stride(0) if D is not None else 0,
            # z strides
            *((z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)),
            # out strides
            out.stride(0), out.stride(1), out.stride(2),
            # prev_Bx strides
            *((prev_Bx.stride(0), prev_Bx.stride(1), prev_Bx.stride(2), prev_Bx.stride(3))
              if prev_Bx is not None else (0, 0, 0, 0)),
            # beta strides
            *((beta.stride(0), beta.stride(1)) if beta is not None else (0, 0)),
            # gamma strides
            *((gamma.stride(0), gamma.stride(1)) if gamma is not None else (0, 0)),
            # MIMO strides
            x.stride(3) if is_mimo else 0,
            B.stride(3) if is_mimo else 0,
            C.stride(3) if is_mimo else 0,
            out.stride(3) if is_mimo else 0,
            # Meta
            dt_softplus,
            BLOCK_SIZE_M,
            num_warps=4,
        )
    return out


def _mamba3_state_update_ref(state, x, dt, A, B, C, D=None, z=None,
                              dt_bias=None, dt_softplus=False,
                              prev_Bx=None, beta=None, gamma=None):
    """Reference PyTorch implementation for Mamba-3 decode step. Supports SISO and MIMO."""
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    is_mimo = x.dim() == 4

    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)

    dA = torch.exp(dt * A)  # (batch, nheads)

    # Input scaling: gamma for trapezoidal, dt for Euler
    input_scale = gamma if gamma is not None else dt

    # Expand B, C from groups to heads
    nheads_per_group = nheads // ngroups
    if is_mimo:
        B_exp = repeat(B, "b g n r -> b (g h) n r", h=nheads_per_group)
        C_exp = repeat(C, "b g n r -> b (g h) n r", h=nheads_per_group)
    else:
        B_exp = repeat(B, "b g n -> b (g h) n", h=nheads_per_group)
        C_exp = repeat(C, "b g n -> b (g h) n", h=nheads_per_group)

    # Compute Bx (unscaled): sum over rank for MIMO
    if is_mimo:
        Bx = torch.einsum("bhpr,bhnr->bhpn", x.float(), B_exp.float())
    else:
        Bx = torch.einsum("bhp,bhn->bhpn", x.float(), B_exp.float())

    # Scaled dBx
    dBx = rearrange(input_scale, "b h -> b h 1 1") * Bx

    # State update
    state.copy_(state * rearrange(dA, "b h -> b h 1 1") + dBx)

    # Trapezoidal lookback
    if prev_Bx is not None and beta is not None:
        state.add_(rearrange(beta, "b h -> b h 1 1") * prev_Bx)

    # Store current raw Bx for next step (unscaled)
    if prev_Bx is not None:
        prev_Bx.copy_(Bx)

    # Output
    if is_mimo:
        out = torch.einsum("bhpn,bhnr->bhpr", state.to(C_exp.dtype), C_exp)
        # D and z not applied for MIMO (handled outside)
    else:
        out = torch.einsum("bhpn,bhn->bhp", state.to(C_exp.dtype), C_exp)
        if D is not None:
            out = out + x * rearrange(D, "h -> 1 h 1")
        if z is not None:
            out = out * F.silu(z)

    return out
