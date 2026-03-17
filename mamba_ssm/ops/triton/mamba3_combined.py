# Copyright (c) 2024, Tri Dao, Albert Gu.

"""Mamba-3 fused chunked SSD with Triton forward and backward.

Uses Triton kernels for the forward pass (speed improvement over pure PyTorch).
Backward uses a Triton backward pipeline when available (SISO, CUDA), falling
back to PyTorch autograd recomputation otherwise (MIMO, CPU, missing kernels).

Architecture:
  - Forward: Triton kernels for SISO on CUDA; PyTorch fallback for MIMO or CPU.
  - Backward: Triton backward pipeline for SISO on CUDA; PyTorch recompute fallback.

This provides a drop-in replacement for mamba3_chunk_scan_combined.
"""

import math

import torch
import torch.nn.functional as F

from mamba_ssm.utils.torch import custom_bwd, custom_fwd

from einops import rearrange, repeat

from mamba_ssm.ops.triton.mamba3_ssd import (
    mamba3_ssd_chunked,
    mamba3_chunk_scan_combined as _mamba3_chunk_scan_combined_ref,
    apply_rotary_emb_to_bc,
)


def _triton_forward_available():
    """Check if the Mamba-3 Triton forward kernels are importable.

    These kernels are provided by separate modules (mamba3_chunk_state, mamba3_chunk_scan)
    and may not be available if they have not been compiled or if the GPU does not support them.
    """
    try:
        from mamba_ssm.ops.triton.mamba3_chunk_state import _mamba3_chunk_state_fwd  # noqa: F401
        from mamba_ssm.ops.triton.mamba3_chunk_scan import _mamba3_chunk_scan_fwd  # noqa: F401
        return True
    except ImportError:
        return False


def _triton_backward_available():
    """Check if the Mamba-3 Triton backward kernels are importable.

    These kernels are provided by separate modules and may not be available
    if they have not been compiled.
    """
    try:
        from mamba_ssm.ops.triton.mamba3_chunk_scan_bwd import (  # noqa: F401
            _mamba3_chunk_scan_chunk_state_bwd_dx,
            _mamba3_chunk_scan_bwd_dcb,
            _mamba3_chunk_scan_bwd_ddAcs_stable,
        )
        from mamba_ssm.ops.triton.mamba3_chunk_state_bwd import (  # noqa: F401
            _mamba3_chunk_state_bwd_db,
            _mamba3_chunk_state_bwd_ddAcs_stable,
        )
        return True
    except ImportError:
        return False


# Cache the availability checks so we only do them once.
_TRITON_FWD_AVAILABLE = None
_TRITON_BWD_AVAILABLE = None


def _check_triton_fwd():
    global _TRITON_FWD_AVAILABLE
    if _TRITON_FWD_AVAILABLE is None:
        _TRITON_FWD_AVAILABLE = _triton_forward_available()
    return _TRITON_FWD_AVAILABLE


def _check_triton_bwd():
    global _TRITON_BWD_AVAILABLE
    if _TRITON_BWD_AVAILABLE is None:
        _TRITON_BWD_AVAILABLE = _triton_backward_available()
    return _TRITON_BWD_AVAILABLE


def _mamba3_chunk_scan_combined_bwd(
    dout, x, dt, A, B, C, out, chunk_size,
    D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None,
    dt_softplus=False, dt_limit=(0.0, float("inf")),
    gamma=None, beta=None, theta=None, initial_prev_Bx=None,
    ngroups=1,
    dfinal_states=None,
):
    """Triton backward for Mamba-3 chunked SSD.

    Follows the Mamba-2 backward pattern from _mamba_chunk_scan_combined_bwd
    with extensions for Mamba-3's trapezoidal discretization, RoPE, and shift.

    Steps:
    1. Pad seqlen to multiple of chunk_size (same as forward)
    2. Recompute forward intermediates (dA_cumsum, dt_out, states, CB)
    3. If z: compute dz via _chunk_scan_bwd_dz (reuse Mamba-2)
    4. Compute dstates via _chunk_scan_bwd_dstates (reuse Mamba-2)
    5. Backward state passing via _state_passing_bwd (reuse Mamba-2)
    6. Compute dx, dgamma, dbeta, ddt, dD, dx_shifted via _mamba3_chunk_scan_chunk_state_bwd_dx
    7. Accumulate dx_shifted into dx (shift backward)
    8. Compute dB, dB_shifted, ddA_next via _mamba3_chunk_state_bwd_db
    9. Compute dC, ddA_cumsum_prev via _chunk_scan_bwd_dC (reuse Mamba-2)
    10. Compute dCB, dCB_shifted via _mamba3_chunk_scan_bwd_dcb
    11. Convert dCB -> dB_scan, dC_scan via _bmm_chunk_bwd (reuse Mamba-2)
    12. Handle dCB_shifted -> additional dB, dC via _bmm_chunk_bwd
    13. Compute ddA_cumsum from scan path via _mamba3_chunk_scan_bwd_ddAcs_stable
    14. If initial_prev_Bx: compute backward through state+output corrections
    15. Accumulate all ddA contributions (ddA_next, ddA_prev, ddA_ipBx)
    16. Convert ddA -> ddt, dA, ddt_bias via _chunk_cumsum_bwd (reuse Mamba-2)
    17. If theta: backward through RoPE; else reduce heads to groups
    18. Convert dgamma_c, dbeta_c from chunked layout to flat layout
    19. Unpad all outputs to original seqlen
    """
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_cumsum_bwd
    from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd, _state_passing_bwd
    from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_bwd
    from mamba_ssm.ops.triton.ssd_chunk_scan import (
        _chunk_scan_bwd_dz,
        _chunk_scan_bwd_dstates,
        _chunk_scan_bwd_dC,
    )
    from mamba_ssm.ops.triton.mamba3_chunk_state import _mamba3_chunk_state_fwd
    from mamba_ssm.ops.triton.mamba3_chunk_scan_bwd import (
        _mamba3_chunk_scan_chunk_state_bwd_dx,
        _mamba3_chunk_scan_bwd_dcb as _mamba3_chunk_scan_bwd_dcb_fn,
        _mamba3_chunk_scan_bwd_ddAcs_stable as _mamba3_scan_bwd_ddAcs,
    )
    from mamba_ssm.ops.triton.mamba3_chunk_state_bwd import (
        _mamba3_chunk_state_bwd_db,
        _mamba3_chunk_state_bwd_ddAcs_stable,
    )

    if dout.stride(-1) != 1:
        dout = dout.contiguous()

    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups_bc, dstate = B.shape
    nheads_per_group = nheads // ngroups_bc
    use_trapezoidal = gamma is not None

    assert dout.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert nheads % ngroups_bc == 0
    assert B.shape == (batch, seqlen, ngroups_bc, dstate)
    assert C.shape == B.shape
    assert out.shape == x.shape

    # ---- Step 1: Pad seqlen to multiple of chunk_size ----
    pad_len = (chunk_size - seqlen % chunk_size) % chunk_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        dt = F.pad(dt, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
        dout = F.pad(dout, (0, 0, 0, 0, 0, pad_len))
        out = F.pad(out, (0, 0, 0, 0, 0, pad_len))
        if gamma is not None:
            gamma = F.pad(gamma, (0, 0, 0, pad_len))
        if beta is not None:
            beta = F.pad(beta, (0, 0, 0, pad_len))
        if z is not None:
            z = F.pad(z, (0, 0, 0, 0, 0, pad_len))
        if theta is not None:
            theta = F.pad(theta, (0, 0, 0, 0, 0, pad_len))
        if seq_idx is not None:
            seq_idx = F.pad(seq_idx, (0, pad_len), value=-1)

    padded_seqlen = seqlen + pad_len
    nchunks = padded_seqlen // chunk_size

    # ---- Step 2: Recompute forward intermediates ----
    # Clone dt to avoid Triton context issues (same as Mamba-2)
    dt_in = dt.clone()
    dA_cumsum, dt_out = _chunk_cumsum_fwd(
        dt_in, A, chunk_size, dt_bias=dt_bias,
        dt_softplus=dt_softplus, dt_limit=dt_limit,
    )

    # RoPE
    if theta is not None:
        B_heads, C_heads = apply_rotary_emb_to_bc(B, C, theta, nheads, ngroups_bc)
    else:
        B_heads = repeat(B, "b l g n -> b l (g h) n", h=nheads_per_group)
        C_heads = repeat(C, "b l g n -> b l (g h) n", h=nheads_per_group)

    if B_heads.stride(-1) != 1:
        B_heads = B_heads.contiguous()
    if C_heads.stride(-1) != 1:
        C_heads = C_heads.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:
        x = x.contiguous()

    # Shift computation for trapezoidal
    if use_trapezoidal:
        gamma_c = rearrange(gamma, "b (c l) h -> b h c l", l=chunk_size)
        beta_c = rearrange(beta, "b (c l) h -> b h c l", l=chunk_size) if beta is not None else None

        B_shifted = torch.zeros_like(B_heads)
        x_shifted = torch.zeros_like(x)
        B_shifted[:, 1:] = B_heads[:, :-1]
        x_shifted[:, 1:] = x[:, :-1]

        if seq_idx is not None:
            shift_valid = torch.ones(batch, padded_seqlen, dtype=torch.bool, device=x.device)
            shift_valid[:, 1:] = seq_idx[:, 1:] == seq_idx[:, :-1]
            shift_valid[:, 0] = False
            sv = shift_valid[:, :, None, None]
            B_shifted = B_shifted * sv
            x_shifted = x_shifted * sv

        if B_shifted.stride(-1) != 1:
            B_shifted = B_shifted.contiguous()
        if x_shifted.stride(-1) != 1 and x_shifted.stride(1) != 1:
            x_shifted = x_shifted.contiguous()
    else:
        gamma_c = dt_out
        beta_c = None
        B_shifted = None
        x_shifted = None

    # Chunk states
    states = _mamba3_chunk_state_fwd(
        B_heads, x, dt_out, dA_cumsum,
        gamma=gamma_c,
        beta=beta_c, B_shifted=B_shifted, x_shifted=x_shifted,
        seq_idx=seq_idx,
        states_in_fp32=True,
    )

    # initial_prev_Bx correction on chunk 0 state
    if initial_prev_Bx is not None and use_trapezoidal and beta is not None:
        beta_flat = rearrange(beta, "b (c l) h -> b c l h", l=chunk_size)
        beta_0 = beta_flat[:, 0, 0, :]
        correction = rearrange(beta_0, "b h -> b h 1 1") * initial_prev_Bx.float()
        decay_states = torch.exp(dA_cumsum[:, :, 0, -1:] - dA_cumsum[:, :, 0, :])
        decay_from_0 = decay_states[:, :, 0]
        states[:, 0] = states[:, 0] + rearrange(decay_from_0, "b h -> b h 1 1") * correction

    # State passing
    states, _ = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
        seq_idx=seq_idx, chunk_size=chunk_size,
    )
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)

    # CB
    CB = _bmm_chunk_fwd(C_heads, B_heads, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    CB_shifted = None
    if use_trapezoidal and B_shifted is not None:
        CB_shifted = _bmm_chunk_fwd(C_heads, B_shifted, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)

    # ---- Step 3: dz computation (if z gating present) ----
    if z is not None:
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(
            x, z, out, dout, chunk_size=chunk_size, has_ddAcs=False, D=D,
        )
    else:
        dz = None

    # ---- Step 4: dstates ----
    dstates = _chunk_scan_bwd_dstates(C_heads, dA_cumsum, dout, seq_idx=seq_idx, dtype=states.dtype)

    # ---- Step 5: Backward state passing ----
    dstates, ddA_chunk_cumsum, dinitial_states, states = _state_passing_bwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        rearrange(dstates, "... p n -> ... (p n)"),
        dfinal_states=rearrange(dfinal_states, "... p n -> ... (p n)") if dfinal_states is not None else None,
        seq_idx=seq_idx,
        has_initial_states=initial_states is not None,
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
        chunk_size=chunk_size,
    )
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    dstates = rearrange(dstates, "... (p n) -> ... p n", n=dstate)
    dinitial_states = rearrange(dinitial_states, "... (p n) -> ... p n", n=dstate) if dinitial_states is not None else None

    # ---- Step 6: dx, dgamma, dbeta, ddt, dD, dx_shifted ----
    # The Mamba-3 dx kernel handles both current and lookback terms internally,
    # returning all 6 outputs: dx, ddt, dD, dgamma_c, dbeta_c, dx_shifted.
    # B_heads is per-head, so we pass ngroups=nheads (nheads_ngroups_ratio=1).
    dx, ddt, dD_from_x, dgamma_c, dbeta_c, dx_shifted = _mamba3_chunk_scan_chunk_state_bwd_dx(
        x, dt_out, dA_cumsum, B_heads, CB, dout, dstates,
        D=D, seq_idx=seq_idx,
        gamma=gamma_c, beta=beta_c,
        CB_shifted=CB_shifted, x_shifted=x_shifted,
        B_shifted=B_shifted,
    )

    # ---- Step 7: Accumulate dx_shifted into dx (shift backward) ----
    # Contribution at position t in shifted maps back to position t-1 in original.
    if dx_shifted is not None:
        dx[:, :-1] += dx_shifted[:, 1:]

    # ---- Step 8: dB, dB_shifted, ddA_next from chunk state backward ----
    # B_heads is per-head (nheads groups of 1), so ngroups=nheads.
    if use_trapezoidal:
        # In trapezoidal mode, the db kernel only computes current term's ddA
        # (lookback ddA is skipped). Use B=None to skip ddA in the db kernel,
        # then call _mamba3_chunk_state_bwd_ddAcs_stable for the full ddA
        # (both current and lookback terms).
        dB_heads, dB_shifted_state = _mamba3_chunk_state_bwd_db(
            x, dA_cumsum, dstates,
            seq_idx=seq_idx, B=None, ngroups=nheads,
            gamma=gamma_c, beta=beta_c,
            x_shifted=x_shifted,
        )
        ddA_next = _mamba3_chunk_state_bwd_ddAcs_stable(
            x, dA_cumsum, dstates, B_heads,
            seq_idx=seq_idx, ngroups=nheads,
            gamma=gamma_c, beta=beta_c,
            x_shifted=x_shifted, B_shifted=B_shifted,
        )
    else:
        # Mamba-2 mode: db kernel folds ddA into its return (no lookback).
        dB_heads, dB_shifted_state, ddA_next = _mamba3_chunk_state_bwd_db(
            x, dA_cumsum, dstates,
            seq_idx=seq_idx, B=B_heads, ngroups=nheads,
            gamma=gamma_c, beta=beta_c,
            x_shifted=x_shifted,
        )

    # Accumulate dB_shifted_state into dB_heads (shift backward)
    if dB_shifted_state is not None:
        dB_heads[:, :-1] += dB_shifted_state[:, 1:]

    # ---- Step 9: dC via chunk scan backward (reuse Mamba-2) ----
    # C_heads is per-head, so pass ngroups=nheads for correct nheads_ngroups_ratio=1.
    dC_heads, ddA_cumsum_prev = _chunk_scan_bwd_dC(
        states.to(x.dtype), dA_cumsum, dout,
        seq_idx=seq_idx, C=C_heads, ngroups=nheads,
    )

    # ---- Step 10: dCB, dCB_shifted via chunk scan backward ----
    # The Mamba-3 dcb kernel handles both current and lookback terms internally.
    dCB, dCB_shifted = _mamba3_chunk_scan_bwd_dcb_fn(
        x, dA_cumsum, dout,
        seq_idx=seq_idx, ngroups=nheads,
        gamma=gamma_c, beta=beta_c,
        x_shifted=x_shifted,
    )

    # ---- Step 11: Convert dCB -> additional dB, dC via BMM backward ----
    dCB = dCB.to(CB.dtype)
    # dCB[b,c,g,m,n]: C^T @ B product gradient
    # dB_from_cb = C^T @ dCB (adds to dB_heads)
    # dC_from_cb = dCB^T @ B (adds to dC_heads)
    dB_scan = torch.empty_like(B_heads)
    _bmm_chunk_bwd(C_heads, dCB, residual=dB_heads, out=dB_scan)
    dC_scan = torch.empty_like(C_heads)
    _bmm_chunk_bwd(B_heads, rearrange(dCB, "... l s -> ... s l"), residual=dC_heads, out=dC_scan)

    # ---- Step 12: Handle dCB_shifted -> additional dB, dC ----
    if dCB_shifted is not None:
        dCB_shifted = dCB_shifted.to(CB_shifted.dtype)
        # dB_shifted_from_cb = C^T @ dCB_shifted
        dB_shifted_bmm = _bmm_chunk_bwd(C_heads, dCB_shifted)
        # dC_from_lb_bmm = dCB_shifted^T @ B_shifted
        dC_from_lb_bmm = _bmm_chunk_bwd(B_shifted, rearrange(dCB_shifted, "... l s -> ... s l"))
        # Shift backward for dB_shifted: position t in shifted -> position t-1 in original
        dB_scan[:, :-1] += dB_shifted_bmm[:, 1:]
        # dC is NOT shifted (C appears unshifted in CB_shifted = C^T @ B_shifted)
        dC_scan += dC_from_lb_bmm

    # If z is not None, dD was already computed in step 3
    if z is None:
        dD = dD_from_x

    # ---- Step 13: ddA_cumsum from scan path ----
    ddA_scan = _mamba3_scan_bwd_ddAcs(
        x, dA_cumsum, dout, CB,
        seq_idx=seq_idx,
        gamma=gamma_c, beta=beta_c,
        x_shifted=x_shifted, CB_shifted=CB_shifted,
        ngroups=nheads,
    )

    # Note: ddA from the state path (ddA_next) is computed in step 8.
    # In trapezoidal mode, it uses _mamba3_chunk_state_bwd_ddAcs_stable
    # (covers both current and lookback terms). In Mamba-2 mode, it's
    # folded into _mamba3_chunk_state_bwd_db's return (current term only).
    # ddA_cumsum_prev is already computed by _chunk_scan_bwd_dC (step 9).

    # ---- Step 14: initial_prev_Bx backward ----
    # Both corrections (state + output) are additive, so the existing Triton
    # backward kernels produce correct gradients for all OTHER parameters.
    # We compute gradients through the ipBx corrections and accumulate ddA/dC/dbeta.
    # Must run before Step 15 (ddA accumulation) and Step 17 (RoPE/group reduction).
    d_initial_prev_Bx = None
    ddA_ipBx = None
    dbeta_0_ipBx = None

    if initial_prev_Bx is not None and use_trapezoidal and beta is not None:
        ipBx = initial_prev_Bx.float()
        beta_flat = rearrange(beta, "b (c l) h -> b c l h", l=chunk_size)
        beta_0 = beta_flat[:, 0, 0, :]  # (batch, nheads)
        beta_0_r = rearrange(beta_0, "b h -> b h 1 1")
        correction = beta_0_r * ipBx  # (b, h, P, N)

        # -- State correction backward --
        # Forward: states[:, 0] += decay_from_0 * correction
        # where decay_from_0 = exp(dA_cumsum[:,:,0,-1] - dA_cumsum[:,:,0,0])
        decay_states_ipBx = torch.exp(dA_cumsum[:, :, 0, -1:] - dA_cumsum[:, :, 0, :])
        decay_from_0 = decay_states_ipBx[:, :, 0]  # (b, h)
        decay_from_0_r = rearrange(decay_from_0, "b h -> b h 1 1")

        dstates_0 = dstates[:, 0].float()  # (b, h, P, N)
        d_decay_from_0 = (dstates_0 * correction).sum(dim=(-2, -1))  # (b, h)
        d_correction_state = decay_from_0_r * dstates_0  # (b, h, P, N)
        d_ipBx_state = d_correction_state * beta_0_r  # (b, h, P, N)
        d_beta_0_state = (d_correction_state * ipBx).sum(dim=(-2, -1))  # (b, h)
        ddA_from_state = d_decay_from_0 * decay_from_0  # (b, h)

        # -- Output correction backward --
        # Forward: Y_corr = einsum("blhn,bhpn,bhl->blhp", C_chunk0, correction, decay_0_to_m)
        # out[:, :chunk_size] += Y_corr
        decay_0_to_m = torch.exp(dA_cumsum[:, :, 0, :] - dA_cumsum[:, :, 0, 0:1])  # (b, h, L)
        C_chunk0 = C_heads[:, :chunk_size].float()  # (b, L, h, N)
        dout_chunk0 = dout[:, :chunk_size].float()  # (b, L, h, P)

        d_correction_out = torch.einsum(
            "blhn,blhp,bhl->bhpn", C_chunk0, dout_chunk0, decay_0_to_m,
        )
        dC_ipBx_chunk0 = torch.einsum(
            "blhp,bhpn,bhl->blhn", dout_chunk0, correction, decay_0_to_m,
        )
        d_decay_0_to_m = torch.einsum(
            "blhn,bhpn,blhp->bhl", C_chunk0, correction, dout_chunk0,
        )

        d_ipBx_out = d_correction_out * beta_0_r  # (b, h, P, N)
        d_beta_0_out = (d_correction_out * ipBx).sum(dim=(-2, -1))  # (b, h)
        ddA_from_output = d_decay_0_to_m * decay_0_to_m  # (b, h, L)

        # -- Accumulate d_initial_prev_Bx --
        d_initial_prev_Bx = (d_ipBx_state + d_ipBx_out).to(initial_prev_Bx.dtype)
        dbeta_0_ipBx = d_beta_0_state + d_beta_0_out  # (b, h)

        # -- Build ddA_ipBx at chunk 0 --
        ddA_ipBx = torch.zeros_like(dA_cumsum)  # (b, h, nchunks, L)
        # State correction: d/d(dA_cumsum[:,:,0,-1]) += ddA_from_state, [:,:,0,0] -= ddA_from_state
        ddA_ipBx[:, :, 0, -1] += ddA_from_state
        ddA_ipBx[:, :, 0, 0] -= ddA_from_state
        # Output correction: d/d(dA_cumsum[:,:,0,:]) += ddA_from_output, [:,:,0,0] -= sum
        ddA_ipBx[:, :, 0, :] += ddA_from_output
        ddA_ipBx[:, :, 0, 0] -= ddA_from_output.sum(dim=-1)

        # -- Accumulate dC from output correction (per-head, at chunk 0 positions) --
        dC_scan[:, :chunk_size] += dC_ipBx_chunk0.to(dC_scan.dtype)

    # ---- Step 15: Accumulate all ddA contributions ----
    # ddA_cumsum_prev is in cumsum space → convert to per-position dA space via reverse cumsum
    ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
    if ddA_ipBx is not None:
        # ddA_ipBx is also in cumsum space, merge before reverse cumsum
        ddA_cumsum_prev = ddA_cumsum_prev + ddA_ipBx
    ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])

    ddA = ddA_scan + ddA_next + ddA_prev

    # ---- Step 16: ddA -> ddt, dA, ddt_bias ----
    if use_trapezoidal:
        ddt_for_cumsum = torch.zeros_like(ddt)
    else:
        ddt_for_cumsum = dgamma_c if dgamma_c is not None else ddt
    ddt_out, dA, ddt_bias_out = _chunk_cumsum_bwd(
        ddA, ddt_for_cumsum, dt_in, A, dt_bias=dt_bias,
        dt_softplus=dt_softplus, dt_limit=dt_limit,
    )

    # ---- Step 17: RoPE backward ----
    # dB_scan and dC_scan are at head level. We need to convert back to group level
    # and also backprop through RoPE if theta is present.
    if theta is not None:
        dtheta, dB_group, dC_group = _rope_bwd_pytorch(
            dB_scan, dC_scan, B, C, theta, nheads, ngroups_bc,
        )
    else:
        dtheta = None
        # Reduce from heads back to groups
        dB_group = rearrange(dB_scan, "b l (g h) n -> b l g h n", g=ngroups_bc).sum(dim=3)
        dC_group = rearrange(dC_scan, "b l (g h) n -> b l g h n", g=ngroups_bc).sum(dim=3)

    # ---- Step 18: Convert dgamma_c, dbeta_c from chunked layout ----
    dgamma = None
    dbeta = None
    if use_trapezoidal:
        if dgamma_c is not None:
            dgamma = rearrange(dgamma_c, "b h c l -> b (c l) h")
        if dbeta_c is not None:
            dbeta = rearrange(dbeta_c, "b h c l -> b (c l) h")
        # Accumulate ipBx contribution to dbeta at position (chunk=0, pos=0)
        if dbeta_0_ipBx is not None:
            if dbeta is None:
                dbeta = torch.zeros_like(dgamma) if dgamma is not None else torch.zeros(batch, padded_seqlen, nheads, device=x.device, dtype=x.dtype)
            dbeta[:, 0] += dbeta_0_ipBx.to(dbeta.dtype)

    # ---- Step 19: Unpad all outputs to original seqlen ----
    if pad_len > 0:
        dx = dx[:, :seqlen]
        dB_group = dB_group[:, :seqlen]
        dC_group = dC_group[:, :seqlen]
        if dz is not None:
            dz = dz[:, :seqlen]
        if dgamma is not None:
            dgamma = dgamma[:, :seqlen]
        if dbeta is not None:
            dbeta = dbeta[:, :seqlen]
        if dtheta is not None:
            dtheta = dtheta[:, :seqlen]

    # ddt_out is (batch, seqlen, nheads) from _chunk_cumsum_bwd -- already correct shape
    # since dt_in was padded and _chunk_cumsum_bwd produces matching shape.
    # We need to unpad it too.
    if pad_len > 0:
        ddt_out = ddt_out[:, :seqlen]

    return (
        dx, ddt_out, dA, dB_group, dC_group, dD, dz, ddt_bias_out,
        dinitial_states, dgamma, dbeta, dtheta, d_initial_prev_Bx,
    )


def _rope_bwd_pytorch(dB_heads, dC_heads, B, C, theta, nheads, ngroups):
    """Backward through RoPE using PyTorch autograd.

    This is a fallback for when Triton RoPE backward kernels are unavailable.
    We recompute the RoPE forward with autograd enabled, then use
    torch.autograd.grad to get gradients.
    """
    B_detach = B.detach().requires_grad_(True)
    C_detach = C.detach().requires_grad_(True)
    theta_detach = theta.detach().requires_grad_(True)

    with torch.enable_grad():
        B_rot, C_rot = apply_rotary_emb_to_bc(
            B_detach, C_detach, theta_detach, nheads, ngroups,
        )

        # Compute gradients
        grads = torch.autograd.grad(
            [B_rot, C_rot],
            [B_detach, C_detach, theta_detach],
            [dB_heads, dC_heads],
            allow_unused=True,
        )

    dB_group = grads[0]
    dC_group = grads[1]
    dtheta = grads[2]

    return dtheta, dB_group, dC_group


class Mamba3ChunkScanCombinedFn(torch.autograd.Function):
    """Autograd function for Mamba-3 chunked SSD with Triton-accelerated forward and backward.

    Forward: Uses Triton kernels when available (SISO, CUDA tensors). Falls back to
    the PyTorch reference implementation for MIMO or when Triton is unavailable.

    Backward: Uses Triton backward pipeline when available (SISO, CUDA, kernels present).
    Falls back to PyTorch autograd recomputation otherwise.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, x, dt, A, B, C, chunk_size,
                D=None, z=None, dt_bias=None,
                initial_states=None, seq_idx=None,
                dt_softplus=False, dt_limit=(0.0, float("inf")),
                return_final_states=False,
                gamma=None, beta=None, theta=None,
                initial_prev_Bx=None, mimo_rank=0, ngroups=1):
        """Forward pass using Triton kernels when possible, PyTorch otherwise.

        Falls back to PyTorch for:
          - MIMO (mimo_rank > 0): Triton kernels are SISO-only.
          - CPU tensors: Triton requires CUDA.
          - Missing Triton kernel modules.

        All non-tensor arguments are stored on ctx as attributes (not saved_tensors).
        """
        # Determine whether to use Triton forward
        use_triton = (
            mimo_rank == 0
            and _check_triton_fwd()
            and x.is_cuda
        )

        # Track which output to save for backward (out_x for dz kernel when z is present)
        out_for_bwd = None

        if use_triton:
            try:
                out, out_x, final_states = _mamba3_triton_fwd(
                    x, dt, A, B, C, chunk_size,
                    D=D, z=z, dt_bias=dt_bias,
                    initial_states=initial_states, seq_idx=seq_idx,
                    dt_softplus=dt_softplus, dt_limit=dt_limit,
                    return_final_states=return_final_states,
                    gamma=gamma, beta=beta, theta=theta,
                    initial_prev_Bx=initial_prev_Bx,
                    ngroups=ngroups,
                )
                # Save pre-z output when z is present (needed by _chunk_scan_bwd_dz)
                out_for_bwd = out if z is None else out_x
            except Exception as e:
                # Graceful fallback if Triton kernels fail at runtime
                # (e.g., unsupported GPU, shape issues)
                import warnings
                warnings.warn(
                    f"Triton forward failed ({type(e).__name__}: {e}), "
                    "falling back to PyTorch forward.",
                    stacklevel=2,
                )
                out, final_states = _mamba3_pytorch_fwd(
                    x, dt, A, B, C, chunk_size,
                    D=D, z=z, dt_bias=dt_bias,
                    initial_states=initial_states, seq_idx=seq_idx,
                    dt_softplus=dt_softplus, dt_limit=dt_limit,
                    return_final_states=return_final_states,
                    gamma=gamma, beta=beta, theta=theta,
                    initial_prev_Bx=initial_prev_Bx,
                    mimo_rank=mimo_rank, ngroups=ngroups,
                )
                out_for_bwd = out
        else:
            out, final_states = _mamba3_pytorch_fwd(
                x, dt, A, B, C, chunk_size,
                D=D, z=z, dt_bias=dt_bias,
                initial_states=initial_states, seq_idx=seq_idx,
                dt_softplus=dt_softplus, dt_limit=dt_limit,
                return_final_states=return_final_states,
                gamma=gamma, beta=beta, theta=theta,
                initial_prev_Bx=initial_prev_Bx,
                mimo_rank=mimo_rank, ngroups=ngroups,
            )
            out_for_bwd = out

        # Save inputs for backward (recompute strategy -- we only save tensors,
        # non-tensor config goes on ctx as attributes).
        # When z is present and we used the Triton path, out_for_bwd is out_x
        # (the pre-z output), which is needed by _chunk_scan_bwd_dz.
        # When z is None, out_for_bwd == out.
        ctx.save_for_backward(x, dt, A, B, C, D, z, dt_bias,
                              initial_states, seq_idx, gamma, beta, theta,
                              initial_prev_Bx, out_for_bwd)
        ctx.chunk_size = chunk_size
        ctx.dt_softplus = dt_softplus
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.mimo_rank = mimo_rank
        ctx.ngroups = ngroups

        if return_final_states:
            return out, final_states
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        """Backward pass using Triton backward pipeline when available.

        Falls back to PyTorch autograd recomputation for:
          - MIMO (mimo_rank > 0)
          - CPU tensors
          - Missing Triton backward kernels
        """
        (x, dt, A, B, C, D, z, dt_bias,
         initial_states, seq_idx, gamma, beta, theta,
         initial_prev_Bx, out) = ctx.saved_tensors

        dfinal_states = args[0] if ctx.return_final_states and len(args) > 0 else None

        # Determine whether to use Triton backward
        # Note: z + initial_prev_Bx forces PyTorch backward because the Triton
        # forward asserts z is None when ipBx is present (they interact in the
        # output correction). If z is present with ipBx, the forward fell back
        # to PyTorch, so the saved out is the final output (not pre-z), making
        # _chunk_scan_bwd_dz incorrect.
        use_triton_bwd = (
            ctx.mimo_rank == 0
            and x.is_cuda
            and _check_triton_bwd()
            and not (z is not None and initial_prev_Bx is not None)
        )

        if use_triton_bwd:
            try:
                grads = _mamba3_chunk_scan_combined_bwd(
                    dout, x, dt, A, B, C, out, ctx.chunk_size,
                    D=D, z=z, dt_bias=dt_bias,
                    initial_states=initial_states, seq_idx=seq_idx,
                    dt_softplus=ctx.dt_softplus, dt_limit=ctx.dt_limit,
                    gamma=gamma, beta=beta, theta=theta,
                    initial_prev_Bx=initial_prev_Bx,
                    ngroups=ctx.ngroups,
                    dfinal_states=dfinal_states,
                )
                (dx, ddt, dA, dB, dC, dD_val, dz_val, ddt_bias,
                 dinitial_states, dgamma, dbeta, dtheta,
                 d_initial_prev_Bx) = grads

                return (
                    dx,                        # x
                    ddt,                       # dt
                    dA,                        # A
                    dB,                        # B
                    dC,                        # C
                    None,                      # chunk_size (int)
                    dD_val,                    # D
                    dz_val,                    # z
                    ddt_bias,                  # dt_bias
                    dinitial_states,           # initial_states
                    None,                      # seq_idx (int tensor)
                    None,                      # dt_softplus (bool)
                    None,                      # dt_limit (tuple)
                    None,                      # return_final_states (bool)
                    dgamma,                    # gamma
                    dbeta,                     # beta
                    dtheta,                    # theta
                    d_initial_prev_Bx,         # initial_prev_Bx
                    None,                      # mimo_rank (int)
                    None,                      # ngroups (int)
                )
            except Exception as e:
                # Fall through to PyTorch backward.
                # Log the exception so silent fallbacks are visible during development.
                import warnings
                warnings.warn(
                    f"Triton backward failed ({type(e).__name__}: {e}), "
                    "falling back to PyTorch recompute backward.",
                    stacklevel=2,
                )

        # ---- PyTorch recompute fallback ----
        return _mamba3_pytorch_backward(
            ctx, dout, x, dt, A, B, C, D, z, dt_bias,
            initial_states, seq_idx, gamma, beta, theta,
            initial_prev_Bx, dfinal_states,
        )


def _mamba3_pytorch_backward(ctx, dout, x, dt, A, B, C, D, z, dt_bias,
                              initial_states, seq_idx, gamma, beta, theta,
                              initial_prev_Bx, dfinal_states):
    """PyTorch autograd recomputation backward (fallback path).

    Strategy:
    1. Detach all saved tensors.
    2. Re-enable requires_grad on differentiable inputs.
    3. Run PyTorch reference forward with grad tracking.
    4. Use torch.autograd.grad to compute gradients w.r.t. differentiable inputs.
    5. Return gradients in the exact order of forward's arguments.
    """
    tensor_inputs = [
        ("x", x, True),
        ("dt", dt, True),
        ("A", A, True),
        ("B", B, True),
        ("C", C, True),
        ("D", D, D is not None),
        ("z", z, z is not None),
        ("dt_bias", dt_bias, dt_bias is not None),
        ("initial_states", initial_states, initial_states is not None),
        ("seq_idx", seq_idx, False),
        ("gamma", gamma, gamma is not None),
        ("beta", beta, beta is not None),
        ("theta", theta, theta is not None),
        ("initial_prev_Bx", initial_prev_Bx, initial_prev_Bx is not None),
    ]

    recomp_tensors = {}
    grad_tensors = []
    for name, tensor, needs_grad in tensor_inputs:
        if tensor is None:
            recomp_tensors[name] = None
        else:
            t = tensor.detach()
            if needs_grad:
                t = t.requires_grad_(True)
                grad_tensors.append((name, t))
            recomp_tensors[name] = t

    with torch.enable_grad():
        result = _mamba3_chunk_scan_combined_ref(
            recomp_tensors["x"],
            recomp_tensors["dt"],
            recomp_tensors["A"],
            recomp_tensors["B"],
            recomp_tensors["C"],
            ctx.chunk_size,
            gamma=recomp_tensors["gamma"],
            beta=recomp_tensors["beta"],
            theta=recomp_tensors["theta"],
            D=recomp_tensors["D"],
            z=recomp_tensors["z"],
            dt_bias=recomp_tensors["dt_bias"],
            dt_softplus=ctx.dt_softplus,
            dt_limit=ctx.dt_limit,
            initial_states=recomp_tensors["initial_states"],
            initial_prev_Bx=recomp_tensors["initial_prev_Bx"],
            return_final_states=ctx.return_final_states,
            ngroups=ctx.ngroups,
            seq_idx=recomp_tensors["seq_idx"],
        )

        if ctx.return_final_states:
            recomp_out, recomp_final_states = result
        else:
            recomp_out = result
            recomp_final_states = None

        outputs = [recomp_out]
        grad_outputs = [dout]

        if ctx.return_final_states and recomp_final_states is not None and dfinal_states is not None:
            outputs.append(recomp_final_states)
            grad_outputs.append(dfinal_states)

        diff_inputs = [t for _, t in grad_tensors]
        grads = torch.autograd.grad(
            outputs,
            diff_inputs,
            grad_outputs,
            allow_unused=True,
        )

    grad_map = {}
    for (name, _), g in zip(grad_tensors, grads):
        grad_map[name] = g

    return (
        grad_map.get("x"),
        grad_map.get("dt"),
        grad_map.get("A"),
        grad_map.get("B"),
        grad_map.get("C"),
        None,                      # chunk_size (int)
        grad_map.get("D"),
        grad_map.get("z"),
        grad_map.get("dt_bias"),
        grad_map.get("initial_states"),
        None,                      # seq_idx (int tensor)
        None,                      # dt_softplus (bool)
        None,                      # dt_limit (tuple)
        None,                      # return_final_states (bool)
        grad_map.get("gamma"),
        grad_map.get("beta"),
        grad_map.get("theta"),
        grad_map.get("initial_prev_Bx"),
        None,                      # mimo_rank (int)
        None,                      # ngroups (int)
    )


def _mamba3_triton_fwd(x, dt, A, B, C, chunk_size,
                       D=None, z=None, dt_bias=None,
                       initial_states=None, seq_idx=None,
                       dt_softplus=False, dt_limit=(0.0, float("inf")),
                       return_final_states=False,
                       gamma=None, beta=None, theta=None,
                       initial_prev_Bx=None, ngroups=1):
    """Forward pass using Triton kernels for SISO Mamba-3 chunked SSD.

    Pipeline:
    1. dt preprocessing (reuse _chunk_cumsum_fwd from Mamba-2)
    2. RoPE on B, C (PyTorch -- will be Triton-ized later)
    3. Compute shifted B, x for trapezoidal (PyTorch preprocessing)
    4. Chunk state computation (_mamba3_chunk_state_fwd -- Triton)
    5. State passing (_state_passing_fwd -- reuse from Mamba-2)
    6. BMM: C^T @ B and C^T @ B_shifted (_bmm_chunk_fwd -- reuse from Mamba-2)
    7. Chunk scan output (_mamba3_chunk_scan_fwd -- Triton)
    8. initial_prev_Bx correction (PyTorch)

    This function is SISO-only (mimo_rank=0). MIMO falls back to PyTorch.
    """
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd
    from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd
    from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd
    from mamba_ssm.ops.triton.mamba3_chunk_state import _mamba3_chunk_state_fwd
    from mamba_ssm.ops.triton.mamba3_chunk_scan import _mamba3_chunk_scan_fwd

    batch, seqlen, nheads, headdim = x.shape
    ngroups_bc = B.shape[2]  # B is (batch, seqlen, ngroups, dstate)
    dstate = B.shape[-1]
    nheads_per_group = nheads // ngroups_bc
    out_dtype = x.dtype

    # Pad sequence length to multiple of chunk_size
    pad_len = (chunk_size - seqlen % chunk_size) % chunk_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        dt = F.pad(dt, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
        if gamma is not None:
            gamma = F.pad(gamma, (0, 0, 0, pad_len))
        if beta is not None:
            beta = F.pad(beta, (0, 0, 0, pad_len))
        if z is not None:
            z = F.pad(z, (0, 0, 0, 0, 0, pad_len))
        if theta is not None:
            theta = F.pad(theta, (0, 0, 0, 0, 0, pad_len))
        if seq_idx is not None:
            seq_idx = F.pad(seq_idx, (0, pad_len), value=-1)

    padded_seqlen = seqlen + pad_len
    nchunks = padded_seqlen // chunk_size

    # ---- Stage 1: dt preprocessing ----
    # Reuse Mamba-2's _chunk_cumsum_fwd for dt bias, softplus, limit, and cumsum
    dA_cumsum, dt_out = _chunk_cumsum_fwd(
        dt.contiguous(), A.contiguous(), chunk_size,
        dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit,
    )
    # dA_cumsum: (batch, nheads, nchunks, chunk_size)
    # dt_out: (batch, nheads, nchunks, chunk_size) -- processed dt values

    # ---- Stage 2: RoPE on B, C (PyTorch) ----
    if theta is not None:
        B_heads, C_heads = apply_rotary_emb_to_bc(B, C, theta, nheads, ngroups_bc)
    else:
        # Expand B, C from groups to heads without RoPE
        B_heads = repeat(B, "b l g n -> b l (g h) n", h=nheads_per_group)
        C_heads = repeat(C, "b l g n -> b l (g h) n", h=nheads_per_group)

    # Make contiguous for kernel consumption
    if B_heads.stride(-1) != 1:
        B_heads = B_heads.contiguous()
    if C_heads.stride(-1) != 1:
        C_heads = C_heads.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:
        x = x.contiguous()

    # ---- Stage 3: Compute gamma/beta and shifted tensors (PyTorch) ----
    use_trapezoidal = gamma is not None

    if use_trapezoidal:
        # Reshape gamma, beta to match dt_out layout: (batch, nheads, nchunks, chunk_size)
        gamma_c = rearrange(gamma, "b (c l) h -> b h c l", l=chunk_size)
        beta_c = rearrange(beta, "b (c l) h -> b h c l", l=chunk_size) if beta is not None else None

        # Compute shifted B and x for the lookback term
        B_shifted = torch.zeros_like(B_heads)
        x_shifted = torch.zeros_like(x)
        # Within-sequence shift by 1
        B_shifted[:, 1:] = B_heads[:, :-1]
        x_shifted[:, 1:] = x[:, :-1]

        # Handle seq_idx masking on shifted tensors
        if seq_idx is not None:
            shift_valid = torch.ones(batch, padded_seqlen, dtype=torch.bool, device=x.device)
            shift_valid[:, 1:] = seq_idx[:, 1:] == seq_idx[:, :-1]
            shift_valid[:, 0] = False  # no lookback for first position
            # Mask invalid shifts
            sv = shift_valid[:, :, None, None]  # (b, l, 1, 1)
            B_shifted = B_shifted * sv
            x_shifted = x_shifted * sv

        if B_shifted.stride(-1) != 1:
            B_shifted = B_shifted.contiguous()
        if x_shifted.stride(-1) != 1 and x_shifted.stride(1) != 1:
            x_shifted = x_shifted.contiguous()
    else:
        # Euler mode: gamma = dt (already in dt_out)
        gamma_c = dt_out
        beta_c = None
        B_shifted = None
        x_shifted = None

    # ---- Stage 4: Chunk state computation (Triton) ----
    # Compute per-chunk states using the Mamba-3 variant that handles
    # both the current (gamma) and lookback (beta) terms.
    states = _mamba3_chunk_state_fwd(
        B_heads, x, dt_out, dA_cumsum,
        gamma=gamma_c,
        beta=beta_c, B_shifted=B_shifted, x_shifted=x_shifted,
        seq_idx=seq_idx,
        states_in_fp32=True,
    )

    # Handle initial_prev_Bx correction on chunk 0 state
    if initial_prev_Bx is not None and use_trapezoidal and beta is not None:
        beta_flat = rearrange(beta, "b (c l) h -> b c l h", l=chunk_size)
        beta_0 = beta_flat[:, 0, 0, :]  # (batch, nheads)
        correction = rearrange(beta_0, "b h -> b h 1 1") * initial_prev_Bx.float()
        # Decay correction from position 0 to end of chunk 0
        decay_states = torch.exp(dA_cumsum[:, :, 0, -1:] - dA_cumsum[:, :, 0, :])
        decay_from_0 = decay_states[:, :, 0]  # decay from pos 0 to end of chunk
        states[:, 0] = states[:, 0] + rearrange(decay_from_0, "b h -> b h 1 1") * correction

    # ---- Stage 5: State passing (reuse Mamba-2 Triton kernel) ----
    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
        seq_idx=seq_idx, chunk_size=chunk_size,
        out_dtype=C_heads.dtype,
    )
    states, final_states = [
        rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]
    ]

    # ---- Stage 6: BMM for C^T @ B products (reuse Mamba-2 Triton kernel) ----
    # Main CB product for current-time term
    CB = _bmm_chunk_fwd(C_heads, B_heads, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)

    # Shifted CB product for trapezoidal lookback term
    CB_shifted = None
    if use_trapezoidal and B_shifted is not None:
        CB_shifted = _bmm_chunk_fwd(C_heads, B_shifted, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)

    # ---- Stage 7: Chunk scan output (Triton) ----
    out, out_x = _mamba3_chunk_scan_fwd(
        CB, x, dt_out, dA_cumsum, gamma_c, C_heads, states,
        D=D, z=z,
        beta=beta_c, CB_shifted=CB_shifted, x_shifted=x_shifted,
        seq_idx=seq_idx,
    )

    # ---- Stage 8: initial_prev_Bx output correction (PyTorch) ----
    # NOTE: This correction is applied after the scan kernel, which may have applied z gating.
    # If z is not None, the correction would be added after z gating, which is incorrect.
    # In practice, z is always None here (modules apply z gating externally).
    if initial_prev_Bx is not None and use_trapezoidal and beta is not None:
        assert z is None, (
            "initial_prev_Bx correction is not compatible with z gating in the Triton path. "
            "Pass z=None and apply z gating externally."
        )
        beta_flat = rearrange(beta, "b (c l) h -> b c l h", l=chunk_size)
        beta_0 = beta_flat[:, 0, 0, :]  # (batch, nheads)
        correction = rearrange(beta_0, "b h -> b h 1 1") * initial_prev_Bx.float()
        # Decay from position 0 to each position m within chunk 0
        decay_0_to_m = torch.exp(dA_cumsum[:, :, 0, :] - dA_cumsum[:, :, 0, 0:1])  # (b, h, chunk_size)

        # C_heads for chunk 0: (batch, chunk_size, nheads, dstate)
        C_chunk0 = C_heads[:, :chunk_size]
        # Y_correction = C_chunk0^T @ correction, weighted by decay
        Y_correction = torch.einsum(
            "blhn,bhpn,bhl->blhp",
            C_chunk0.float(), correction, decay_0_to_m,
        )
        out[:, :chunk_size] = out[:, :chunk_size] + Y_correction.to(out.dtype)

    # Un-pad if necessary
    if pad_len > 0:
        out = out[:, :seqlen]
        if out_x is not None:
            out_x = out_x[:, :seqlen]

    out = out.to(out_dtype)

    if return_final_states:
        return out, out_x, final_states
    return out, out_x, None


def _mamba3_pytorch_fwd(x, dt, A, B, C, chunk_size,
                        D=None, z=None, dt_bias=None,
                        initial_states=None, seq_idx=None,
                        dt_softplus=False, dt_limit=(0.0, float("inf")),
                        return_final_states=False,
                        gamma=None, beta=None, theta=None,
                        initial_prev_Bx=None, mimo_rank=0, ngroups=1):
    """Forward pass using PyTorch reference (for backward recompute and MIMO fallback).

    Delegates to the existing mamba3_chunk_scan_combined reference implementation.
    Returns (out, final_states) tuple where final_states is None if not requested.
    """
    # Note: mamba3_chunk_scan_combined does not accept mimo_rank as a kwarg;
    # it detects MIMO from B.dim() == 5. We do not pass mimo_rank here.
    result = _mamba3_chunk_scan_combined_ref(
        x, dt, A, B, C, chunk_size,
        gamma=gamma,
        beta=beta,
        theta=theta,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        initial_states=initial_states,
        initial_prev_Bx=initial_prev_Bx,
        return_final_states=return_final_states,
        ngroups=ngroups,
        seq_idx=seq_idx,
    )

    if return_final_states:
        return result  # already (out, final_states)
    else:
        return result, None  # normalize to (out, None)


def mamba3_chunk_scan_combined_triton(x, dt, A, B, C, chunk_size, **kwargs):
    """Drop-in replacement for mamba3_chunk_scan_combined with Triton acceleration.

    Uses Triton kernels for the forward pass (SISO on CUDA) and Triton backward
    pipeline when available. Falls back to pure PyTorch for MIMO or when Triton
    kernels are unavailable.

    Usage:
        Replace calls to mamba3_chunk_scan_combined with this function.
        The signature and return values are identical.

    Args:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads,) -- negative
        B: (batch, seqlen, ngroups, d_state)
        C: (batch, seqlen, ngroups, d_state)
        chunk_size: int
        **kwargs: All keyword arguments from mamba3_chunk_scan_combined:
            gamma, beta, theta, D, z, dt_bias, dt_softplus, dt_limit,
            initial_states, initial_prev_Bx, return_final_states,
            ngroups, seq_idx, mimo_rank.
    Returns:
        Same as mamba3_chunk_scan_combined:
          - Y: (batch, seqlen, nheads, headdim[, mimo_rank]) if not return_final_states
          - (Y, final_state) if return_final_states
    """
    # Extract mimo_rank to decide if we need to handle MIMO return shapes
    mimo_rank = kwargs.get("mimo_rank", 0)

    # For the "ngroups" kwarg, mamba3_chunk_scan_combined uses it but it's also
    # derivable from B.shape. We pass it through.
    result = Mamba3ChunkScanCombinedFn.apply(
        x, dt, A, B, C, chunk_size,
        kwargs.get("D"),
        kwargs.get("z"),
        kwargs.get("dt_bias"),
        kwargs.get("initial_states"),
        kwargs.get("seq_idx"),
        kwargs.get("dt_softplus", False),
        kwargs.get("dt_limit", (0.0, float("inf"))),
        kwargs.get("return_final_states", False),
        kwargs.get("gamma"),
        kwargs.get("beta"),
        kwargs.get("theta"),
        kwargs.get("initial_prev_Bx"),
        mimo_rank,
        kwargs.get("ngroups", 1),
    )

    return result
