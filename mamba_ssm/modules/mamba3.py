# Copyright (c) 2024, Tri Dao, Albert Gu.
# Mamba-3 implementation based on "Mamba-3: Improved Sequence Modeling Using State Space Principles"
# Key changes from Mamba-2:
#   1. Exponential-trapezoidal discretization (lookback recurrence)
#   2. Complex-valued SSM via data-dependent RoPE on B, C
#   3. MIMO (multi-input, multi-output) SSM option
#   4. BCNorm (RMSNorm on B, C projections)
#   5. Learnable B, C biases (head-specific, channel-wise)
#   6. No short causal convolution

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None

try:
    from mamba_ssm.ops.triton.mamba3_ssd import (
        mamba3_chunk_scan_combined,
        mamba3_state_update,
    )
except ImportError:
    mamba3_chunk_scan_combined = None
    mamba3_state_update = None

from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from huggingface_hub import PyTorchModelHubMixin


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embedding to x. x: (..., N), cos/sin: (..., N/2)."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


def compute_cumulative_rotary(theta_cumsum, dstate):
    """Compute cumulative rotation angles for data-dependent RoPE.

    Args:
        theta_cumsum: (batch, seqlen, nheads, dstate//2) cumulative sum of theta angles
    Returns:
        cos, sin: (batch, seqlen, nheads, dstate//2) cumulative cos/sin
    """
    return torch.cos(theta_cumsum), torch.sin(theta_cumsum)


class Mamba3(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=64,
        expand=2,
        headdim=64,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        # Mamba-3 specific
        use_rope=True,
        use_trapezoidal=True,
        use_bc_norm=True,
        use_bc_bias=True,
        mimo_rank=0,  # 0 = SISO, >0 = MIMO with this rank
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,  # gradient checkpointing for memory-efficient training
        layer_idx=None,
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert self.d_ssm == self.d_inner, (
            f"Mamba3 requires d_ssm == d_inner (got d_ssm={self.d_ssm}, d_inner={self.d_inner}). "
            f"Unlike Mamba-2, Mamba-3 does not support partial SSM dimension."
        )
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Mamba-3 specific
        self.use_rope = use_rope
        self.use_trapezoidal = use_trapezoidal
        self.use_bc_norm = use_bc_norm
        self.use_bc_bias = use_bc_bias
        self.mimo_rank = mimo_rank
        self.is_mimo = mimo_rank > 0

        # Projection sizes
        # For MIMO: B and C project to (ngroups * d_state * mimo_rank) instead of (ngroups * d_state)
        bc_dim = self.ngroups * self.d_state
        if self.is_mimo:
            bc_proj_dim = bc_dim * self.mimo_rank
        else:
            bc_proj_dim = bc_dim

        # dt: nheads
        # theta (for RoPE): nheads * (d_state // 2) if use_rope else 0
        theta_dim = self.nheads * (self.d_state // 2) if self.use_rope else 0
        # lambda (for trapezoidal): nheads if use_trapezoidal else 0
        lambda_dim = self.nheads if self.use_trapezoidal else 0

        # Order: [z, x, B, C, dt, theta, lambda]
        d_in_proj = (
            self.d_inner  # z (gate)
            + self.d_ssm  # x (SSM input)
            + bc_proj_dim  # B
            + bc_proj_dim  # C
            + self.nheads  # dt
            + theta_dim  # theta for RoPE
            + lambda_dim  # lambda for trapezoidal
        )

        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(
                self.d_model, d_in_proj * self.world_size, bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        # For MIMO, x also needs rank R projection
        if self.is_mimo:
            # x goes from (batch, seqlen, d_ssm) to (batch, seqlen, nheads, headdim, mimo_rank)
            # We project headdim -> headdim * mimo_rank per head
            self.x_mimo_proj = nn.Linear(self.headdim, self.headdim * self.mimo_rank, bias=False, **factory_kwargs)
            # Learned MIMO output projection: PR -> P per head (paper Section D, W_{O'})
            self.mimo_out_proj = nn.Linear(self.headdim * self.mimo_rank, self.headdim, bias=False, **factory_kwargs)

        # dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter (data-dependent in Mamba-3, but we keep log-space init)
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        # BC Norm (Mamba-3: RMSNorm on B and C after projection)
        if self.use_bc_norm:
            self.B_norm = nn.RMSNorm(self.d_state, eps=1e-5, **factory_kwargs)
            self.C_norm = nn.RMSNorm(self.d_state, eps=1e-5, **factory_kwargs)

        # BC Bias (Mamba-3: learnable head-specific channel-wise biases, init=1.0 per paper Table 9a)
        if self.use_bc_bias:
            self.B_bias = nn.Parameter(torch.ones(self.nheads, self.d_state, **factory_kwargs))
            self.C_bias = nn.Parameter(torch.ones(self.nheads, self.d_state, **factory_kwargs))

        # Output norm (gated RMSNorm)
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // self.ngroups, **factory_kwargs,
            )

        # Output projection
        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(
                self.d_inner * self.world_size, self.d_model, bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        # Store split sizes for forward
        self._split_sizes = [
            self.d_inner,  # z
            self.d_ssm,  # x
            bc_proj_dim,  # B
            bc_proj_dim,  # C
            self.nheads,  # dt
        ]
        if self.use_rope:
            self._split_sizes.append(theta_dim)
        if self.use_trapezoidal:
            self._split_sizes.append(lambda_dim)

    def _process_bc(self, B_raw, C_raw):
        """Apply BCNorm to B and C projections (at group level, before expansion).

        BC bias is applied separately after group→head expansion to preserve
        head-specificity (paper Section 3.4: "head-specific, channel-wise biases").

        Args:
            B_raw: (batch, seqlen, ngroups, d_state [, mimo_rank])
            C_raw: same shape
        Returns:
            B, C with norm applied (bias applied later after group→head expansion)
        """
        if self.use_bc_norm:
            orig_shape = B_raw.shape
            if B_raw.dim() == 5:
                # MIMO: (b, l, g, d_state, mimo_rank) — move rank before d_state to normalize correctly
                B_raw = self.B_norm(B_raw.movedim(-1, -2).reshape(-1, self.d_state)).reshape(
                    *orig_shape[:-2], orig_shape[-1], orig_shape[-2]
                ).movedim(-1, -2)
                C_raw = self.C_norm(C_raw.movedim(-1, -2).reshape(-1, self.d_state)).reshape(
                    *orig_shape[:-2], orig_shape[-1], orig_shape[-2]
                ).movedim(-1, -2)
            else:
                B_raw = self.B_norm(B_raw.reshape(-1, self.d_state)).reshape(orig_shape)
                C_raw = self.C_norm(C_raw.reshape(-1, self.d_state)).reshape(orig_shape)

        return B_raw, C_raw

    def _apply_bc_bias(self, B, C):
        """Apply head-specific BC bias after group→head expansion.

        Args:
            B: (batch, seqlen, nheads, d_state [, mimo_rank])
            C: same shape
        Returns:
            B, C with per-head bias applied
        """
        if not self.use_bc_bias:
            return B, C
        if self.is_mimo:
            # B_bias: (nheads, d_state) -> broadcast over (batch, seqlen, nheads, d_state, mimo_rank)
            B = B + self.B_bias.view(1, 1, self.nheads, self.d_state, 1)
            C = C + self.C_bias.view(1, 1, self.nheads, self.d_state, 1)
        else:
            # B_bias: (nheads, d_state) -> broadcast over (batch, seqlen, nheads, d_state)
            B = B + self.B_bias
            C = C + self.C_bias
        return B, C

    def _ssd_trapezoidal(self, x, dt, A, B, C, theta=None, lam=None,
                         initial_states=None, return_final_states=False,
                         initial_prev_Bx=None, seq_idx=None):
        """Reference implementation of Mamba-3 SSD with trapezoidal discretization.

        This is a step-by-step recurrence (not chunked). For production, this should
        be replaced with optimized Triton kernels.

        Args:
            x: (batch, seqlen, nheads, headdim) or (batch, seqlen, nheads, headdim, mimo_rank) for MIMO
            dt: (batch, seqlen, nheads) - already processed (softplus applied)
            A: (nheads,) - negative values
            B: (batch, seqlen, nheads, d_state) or (..., d_state, mimo_rank) — already expanded
            C: same as B
            theta: (batch, seqlen, nheads, d_state//2) or None
            lam: (batch, seqlen, nheads) or None - trapezoidal lambda in [0, 1]
            initial_states: (batch, nheads, headdim, d_state) or None
            return_final_states: bool
        Returns:
            y: (batch, seqlen, nheads, headdim)
            final_states: (batch, nheads, headdim, d_state) if return_final_states
        """
        batch, seqlen, nheads, headdim = x.shape[:4]
        dstate = B.shape[-2] if self.is_mimo else B.shape[-1]
        # B, C are already at head level (expanded + biased in forward())

        # Apply RoPE if enabled
        if self.use_rope and theta is not None:
            # theta: (batch, seqlen, nheads, d_state//2)
            # Cumulative sum of theta for data-dependent RoPE
            theta_cumsum = torch.cumsum(theta, dim=1)  # (batch, seqlen, nheads, d_state//2)
            cos_t, sin_t = compute_cumulative_rotary(theta_cumsum, dstate)

            if self.is_mimo:
                # Apply RoPE to each rank slice (out-of-place for autograd safety)
                B_parts = [apply_rotary_emb(B[:, :, :, :, r], cos_t, sin_t) for r in range(self.mimo_rank)]
                C_parts = [apply_rotary_emb(C[:, :, :, :, r], cos_t, sin_t) for r in range(self.mimo_rank)]
                B = torch.stack(B_parts, dim=-1)
                C = torch.stack(C_parts, dim=-1)
            else:
                B = apply_rotary_emb(B, cos_t, sin_t)
                C = apply_rotary_emb(C, cos_t, sin_t)

        # Discretize
        alpha = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, nheads, 1))  # (batch, seqlen, nheads, 1)

        if self.use_trapezoidal and lam is not None:
            gamma = lam * dt  # (batch, seqlen, nheads) — λ * Δt
            beta = (1 - lam) * dt * torch.exp(dt * A.view(1, 1, nheads))  # (1-λ) * Δt * α
        else:
            gamma = dt  # Euler fallback: γ = Δt
            beta = None

        # Initialize state
        if self.is_mimo:
            h = torch.zeros(batch, nheads, headdim, dstate, device=x.device, dtype=torch.float32)
        else:
            h = torch.zeros(batch, nheads, headdim, dstate, device=x.device, dtype=torch.float32)

        if initial_states is not None:
            h = initial_states.float()

        ys = []
        prev_Bx = initial_prev_Bx  # For trapezoidal: B_{t-1} * x_{t-1}

        for t in range(seqlen):
            # Reset state at document boundaries
            if seq_idx is not None and t > 0:
                boundary = (seq_idx[:, t] != seq_idx[:, t - 1])  # (batch,)
                if boundary.any():
                    mask = boundary.view(-1, 1, 1, 1).float()
                    h = h * (1 - mask)  # zero state at boundary
                    if initial_states is not None:
                        h = h + mask * initial_states.float()
                    prev_Bx = None if boundary.all() else (
                        prev_Bx * (1 - mask) if prev_Bx is not None else None
                    )

            # State transition: h_t = alpha_t * h_{t-1} + beta_t * B_{t-1} * x_{t-1} + gamma_t * B_t * x_t
            alpha_t = alpha[:, t]  # (batch, nheads, 1)

            if self.is_mimo:
                x_t = x[:, t]
                B_t = B[:, t]
                Bx_t = torch.einsum("bhpr,bhnr->bhpn", x_t.float(), B_t.float())
            else:
                x_t = x[:, t]
                B_t = B[:, t]
                Bx_t = torch.einsum("bhp,bhn->bhpn", x_t.float(), B_t.float())

            gamma_t = gamma[:, t].unsqueeze(-1).unsqueeze(-1)  # (batch, nheads, 1, 1)

            h = alpha_t.unsqueeze(-1) * h + gamma_t * Bx_t

            if beta is not None and prev_Bx is not None:
                beta_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)  # (batch, nheads, 1, 1)
                h = h + beta_t * prev_Bx

            prev_Bx = Bx_t

            # Output: y_t = C_t^T @ h_t
            if self.is_mimo:
                C_t = C[:, t]  # (batch, nheads, d_state, mimo_rank)
                y_t = torch.einsum("bhpn,bhnr->bhpr", h.to(C_t.dtype), C_t)
                # Per-rank output: (batch, nheads, headdim, mimo_rank)
            else:
                C_t = C[:, t]  # (batch, nheads, d_state)
                y_t = torch.einsum("bhpn,bhn->bhp", h.to(C_t.dtype), C_t)

            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (batch, seqlen, nheads, headdim)

        if return_final_states:
            return y, h
        return y

    def _ssd_chunked(self, x, dt, A, B, C, theta=None, lam=None,
                     initial_states=None, seq_idx=None, return_final_states=False,
                     initial_prev_Bx=None):
        """Chunked parallel implementation.

        Uses mamba3_chunk_scan_combined (matmul-based parallel within chunks) when available.
        Falls back to step-by-step recurrence otherwise.
        """
        nheads = self.nheads

        # Compute trapezoidal weights
        gamma = None
        beta = None
        if self.use_trapezoidal and lam is not None:
            gamma = lam * dt  # (batch, seqlen, nheads)
            beta = (1 - lam) * dt * torch.exp(dt * A.view(1, 1, nheads))
        else:
            gamma = dt  # Euler fallback

        if mamba3_chunk_scan_combined is not None:
            # B, C are already expanded to head level, so ngroups=nheads
            return mamba3_chunk_scan_combined(
                x, dt, A, B, C,
                chunk_size=self.chunk_size,
                gamma=gamma,
                beta=beta if self.use_trapezoidal else None,
                theta=theta,
                D=None,  # D is applied outside
                initial_states=initial_states,
                initial_prev_Bx=initial_prev_Bx,
                return_final_states=return_final_states,
                ngroups=nheads,
                seq_idx=seq_idx,
            )
        else:
            # Fallback to step-by-step (if chunked kernels unavailable)
            return self._ssd_trapezoidal(
                x, dt, A, B, C, theta=theta, lam=lam,
                initial_states=initial_states,
                return_final_states=return_final_states,
                initial_prev_Bx=initial_prev_Bx,
                seq_idx=seq_idx,
            )

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
        Returns: same shape as u
        """
        if seqlen is None:
            batch = u.shape[0]
        else:
            batch = u.shape[0] // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        # Memory-efficient path: gradient checkpointing to avoid storing intermediates
        if self.use_mem_eff_path and inference_params is None and torch.is_grad_enabled():
            return gradient_checkpoint(
                self._forward_inner, u, seqlen, seq_idx, conv_state, ssm_state,
                use_reentrant=False,
            )
        return self._forward_inner(u, seqlen, seq_idx, conv_state, ssm_state)

    def _forward_inner(self, u, seqlen, seq_idx, conv_state, ssm_state):
        """Core forward computation, factored out for gradient checkpointing."""
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        proj = self.in_proj(u)  # (B, L, d_in_proj)
        if seqlen_og is not None:
            proj = rearrange(proj, "(b l) d -> b l d", l=seqlen)

        # Split projection
        splits = torch.split(proj, self._split_sizes, dim=-1)
        idx = 0
        z = splits[idx]; idx += 1
        x = splits[idx]; idx += 1
        B_raw = splits[idx]; idx += 1
        C_raw = splits[idx]; idx += 1
        dt_raw = splits[idx]; idx += 1
        theta_raw = splits[idx] if self.use_rope else None; idx += (1 if self.use_rope else 0)
        lam_raw = splits[idx] if self.use_trapezoidal else None

        A = -torch.exp(self.A_log.float())

        # Process dt
        dt = F.softplus(dt_raw + self.dt_bias)  # (batch, seqlen, nheads)
        if self.dt_limit != (0.0, float("inf")):
            dt = dt.clamp(min=self.dt_limit[0], max=self.dt_limit[1])

        # Process B, C
        if self.is_mimo:
            B = rearrange(B_raw, "b l (g n r) -> b l g n r", g=self.ngroups, r=self.mimo_rank)
            C = rearrange(C_raw, "b l (g n r) -> b l g n r", g=self.ngroups, r=self.mimo_rank)
        else:
            B = rearrange(B_raw, "b l (g n) -> b l g n", g=self.ngroups)
            C = rearrange(C_raw, "b l (g n) -> b l g n", g=self.ngroups)

        B, C = self._process_bc(B, C)

        # Expand B, C from groups to heads and apply per-head bias
        nheads_per_group = self.nheads // self.ngroups
        if self.is_mimo:
            B = repeat(B, "b l g n r -> b l (g h) n r", h=nheads_per_group)
            C = repeat(C, "b l g n r -> b l (g h) n r", h=nheads_per_group)
        else:
            B = repeat(B, "b l g n -> b l (g h) n", h=nheads_per_group)
            C = repeat(C, "b l g n -> b l (g h) n", h=nheads_per_group)
        B, C = self._apply_bc_bias(B, C)

        # Process theta (RoPE angles)
        theta = None
        if self.use_rope and theta_raw is not None:
            theta = rearrange(theta_raw, "b l (h d) -> b l h d", h=self.nheads)

        # Process lambda (trapezoidal parameter)
        lam = None
        if self.use_trapezoidal and lam_raw is not None:
            lam = torch.sigmoid(lam_raw)  # (batch, seqlen, nheads) in [0, 1]

        # Process x
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        if self.is_mimo:
            # Project x to (batch, seqlen, nheads, headdim, mimo_rank)
            x = self.x_mimo_proj(x)  # (batch, seqlen, nheads, headdim * mimo_rank)
            x = rearrange(x, "b l h (p r) -> b l h p r", r=self.mimo_rank)

        # Extract initial_prev_Bx from conv_state for segmented prefill continuity
        initial_prev_Bx = None
        if conv_state is not None and self.use_trapezoidal:
            prev_Bx_flat_size = self.nheads * self.headdim * self.d_state
            prev_Bx_data = conv_state[:, :prev_Bx_flat_size]
            if prev_Bx_data.abs().sum() > 0:  # non-zero means state was populated by a previous segment
                initial_prev_Bx = prev_Bx_data.view(-1, self.nheads, self.headdim, self.d_state)

        # Run SSM (B, C already at head level with bias applied)
        result = self._ssd_chunked(
            x, dt, A, B, C,
            theta=theta, lam=lam,
            seq_idx=seq_idx,
            return_final_states=ssm_state is not None,
            initial_prev_Bx=initial_prev_Bx,
        )

        if ssm_state is not None:
            y, last_state = result
            ssm_state.copy_(last_state)
        else:
            y = result if not isinstance(result, tuple) else result[0]

        # Store inference state for decode continuity after prefill
        if conv_state is not None:
            prev_Bx_flat_size = self.nheads * self.headdim * self.d_state
            half_d = self.d_state // 2

            # Compute cumulative theta first (needed for both RoPE on B_last and storage)
            theta_cumsum = None
            if self.use_rope and theta is not None:
                theta_cumsum = torch.cumsum(theta, dim=1)  # (batch, seqlen, nheads, d_state//2)

            # Store last step's B*x for trapezoidal lookback (with RoPE applied)
            # B is already at head level with bias applied; just need RoPE
            B_last = B[:, -1].clone()  # (batch, nheads, d_state[, R])
            if self.is_mimo:
                # Apply RoPE to B_last so prev_Bx matches decode step's convention
                if theta_cumsum is not None:
                    theta_last = theta_cumsum[:, -1]  # (batch, nheads, d_state//2)
                    cos_last = torch.cos(theta_last)
                    sin_last = torch.sin(theta_last)
                    for r in range(self.mimo_rank):
                        B_last[:, :, :, r] = apply_rotary_emb(B_last[:, :, :, r], cos_last, sin_last)
                x_last = x[:, -1]  # (batch, nheads, headdim, mimo_rank)
                Bx_last = torch.einsum("bhpr,bhnr->bhpn", x_last.float(), B_last.float())
            else:
                # Apply RoPE to B_last so prev_Bx matches decode step's convention
                if theta_cumsum is not None:
                    theta_last = theta_cumsum[:, -1]  # (batch, nheads, d_state//2)
                    cos_last = torch.cos(theta_last)
                    sin_last = torch.sin(theta_last)
                    B_last = apply_rotary_emb(B_last, cos_last, sin_last)
                x_last = x[:, -1]  # (batch, nheads, headdim)
                Bx_last = torch.einsum("bhp,bhn->bhpn", x_last.float(), B_last.float())
            conv_state[:, :prev_Bx_flat_size] = Bx_last.reshape(batch, -1)

            # Store cumulative theta for RoPE continuity
            if theta_cumsum is not None:
                theta_total = theta_cumsum[:, -1]  # (batch, nheads, d_state//2)
                cum_theta_offset = prev_Bx_flat_size
                conv_state[:, cum_theta_offset:cum_theta_offset + self.nheads * half_d] = \
                    theta_total.reshape(batch, -1).float()

        # D skip connection + flatten (cast D to input dtype to avoid float32 promotion)
        D = self.D.to(dtype=y.dtype)
        if self.is_mimo:
            # y: (B, L, H, P, R), x: (B, L, H, P, R)
            if self.D_has_hdim:
                y = y + x * rearrange(D, "(h p) -> 1 1 h p 1", p=self.headdim)
            else:
                y = y + x * repeat(D, "h -> 1 1 h 1 1")
            # Learned MIMO output projection: (P*R) -> P per head
            y = self.mimo_out_proj(rearrange(y, "b l h p r -> b l h (p r)"))
            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            y = rearrange(y, "b l h p -> b l (h p)")
            x_flat = rearrange(x, "b l h p -> b l (h p)")
            if self.D_has_hdim:
                y = y + x_flat * rearrange(D, "(h p) -> h p", p=self.headdim).reshape(1, 1, -1)
            else:
                y = y + x_flat * repeat(D, "h -> (h p)", p=self.headdim).reshape(1, 1, -1)

        # Gated output norm
        if self.rmsnorm:
            y = self.norm(y, z)
        else:
            y = y * F.silu(z)

        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")

        out = self.out_proj(y)

        if self.process_group is not None:
            reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
            out = reduce_fn(out, self.process_group)

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        """Single-token decoding step.

        conv_state is a dict-like object (or tuple) with:
          - prev_Bx: (batch, nheads, headdim, d_state) for trapezoidal lookback
          - cum_theta: (batch, nheads, d_state//2) for cumulative RoPE angles
        For simplicity we pack them into a single tensor:
          conv_state: (batch, nheads, headdim * d_state + d_state // 2)
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time"
        proj = self.in_proj(hidden_states.squeeze(1))  # (B, d_in_proj)

        # Split
        splits = torch.split(proj, self._split_sizes, dim=-1)
        idx = 0
        z = splits[idx]; idx += 1
        x = splits[idx]; idx += 1
        B_raw = splits[idx]; idx += 1
        C_raw = splits[idx]; idx += 1
        dt_raw = splits[idx]; idx += 1
        theta_raw = splits[idx] if self.use_rope else None; idx += (1 if self.use_rope else 0)
        lam_raw = splits[idx] if self.use_trapezoidal else None

        A = -torch.exp(self.A_log.float())

        # Process dt
        dt = F.softplus(dt_raw + self.dt_bias.to(dtype=dt_raw.dtype))  # (batch, nheads)
        if self.dt_limit != (0.0, float("inf")):
            dt = dt.clamp(min=self.dt_limit[0], max=self.dt_limit[1])

        dA = torch.exp(dt * A)  # (batch, nheads)

        # Process B, C
        if self.is_mimo:
            B = rearrange(B_raw, "b (g n r) -> b g n r", g=self.ngroups, r=self.mimo_rank)
            C = rearrange(C_raw, "b (g n r) -> b g n r", g=self.ngroups, r=self.mimo_rank)
        else:
            B = rearrange(B_raw, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C_raw, "b (g n) -> b g n", g=self.ngroups)

        # BC Norm
        if self.use_bc_norm:
            orig = B.shape
            if self.is_mimo:
                # MIMO: (b, g, d_state, mimo_rank) — move rank before d_state
                B = self.B_norm(B.movedim(-1, -2).reshape(-1, self.d_state)).reshape(
                    *orig[:-2], orig[-1], orig[-2]
                ).movedim(-1, -2)
                C = self.C_norm(C.movedim(-1, -2).reshape(-1, self.d_state)).reshape(
                    *orig[:-2], orig[-1], orig[-2]
                ).movedim(-1, -2)
            else:
                B = self.B_norm(B.reshape(-1, self.d_state)).reshape(orig)
                C = self.C_norm(C.reshape(-1, self.d_state)).reshape(orig)

        # Expand B, C from groups to heads
        nheads_per_group = self.nheads // self.ngroups
        if self.is_mimo:
            B = repeat(B, "b g n r -> b (g h) n r", h=nheads_per_group)
            C = repeat(C, "b g n r -> b (g h) n r", h=nheads_per_group)
        else:
            B = repeat(B, "b g n -> b (g h) n", h=nheads_per_group)
            C = repeat(C, "b g n -> b (g h) n", h=nheads_per_group)

        # Apply head-specific BC bias (after expansion for true per-head bias)
        if self.use_bc_bias:
            if self.is_mimo:
                B = B + self.B_bias.view(1, self.nheads, self.d_state, 1)
                C = C + self.C_bias.view(1, self.nheads, self.d_state, 1)
            else:
                B = B + self.B_bias
                C = C + self.C_bias

        # Unpack conv_state -> prev_Bx and cum_theta
        prev_Bx_flat_size = self.nheads * self.headdim * self.d_state
        half_d = self.d_state // 2
        prev_Bx = conv_state[:, :prev_Bx_flat_size].view(
            -1, self.nheads, self.headdim, self.d_state
        )

        # Apply RoPE to B, C
        if self.use_rope and theta_raw is not None:
            theta = rearrange(theta_raw, "b (h d) -> b h d", h=self.nheads)
            cum_theta_offset = prev_Bx_flat_size
            cum_theta = conv_state[:, cum_theta_offset:cum_theta_offset + self.nheads * half_d].view(
                -1, self.nheads, half_d
            )
            cum_theta = cum_theta + theta
            conv_state[:, cum_theta_offset:cum_theta_offset + self.nheads * half_d] = cum_theta.view(
                -1, self.nheads * half_d
            )

            cos_t = torch.cos(cum_theta)
            sin_t = torch.sin(cum_theta)

            if self.is_mimo:
                for r in range(self.mimo_rank):
                    B[:, :, :, r] = apply_rotary_emb(B[:, :, :, r], cos_t, sin_t)
                    C[:, :, :, r] = apply_rotary_emb(C[:, :, :, r], cos_t, sin_t)
            else:
                B = apply_rotary_emb(B, cos_t, sin_t)
                C = apply_rotary_emb(C, cos_t, sin_t)

        # Process x
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        if self.is_mimo:
            x = self.x_mimo_proj(x)
            x = rearrange(x, "b h (p r) -> b h p r", r=self.mimo_rank)

        # Compute trapezoidal weights
        lam_val = None
        gamma_scalar = None
        beta_scalar = None
        if self.use_trapezoidal and lam_raw is not None:
            lam_val = torch.sigmoid(lam_raw)  # (batch, nheads)
            gamma_scalar = lam_val * dt  # (batch, nheads)
            beta_scalar = (1 - lam_val) * dt * dA  # (batch, nheads)
        else:
            gamma_scalar = dt

        # Use Triton kernel for decode when available (supports both SISO and MIMO)
        use_triton_decode = (
            mamba3_state_update is not None
            and ssm_state.is_cuda
        )

        if use_triton_decode:
            # Triton kernel handles: state update + output in one fused op
            # B, C are already at head level after preprocessing — pass as ngroups=nheads
            if self.is_mimo:
                # MIMO: kernel returns (B, H, P, R), D and z not applied by kernel
                y = mamba3_state_update(
                    ssm_state, x, dt, A,
                    B, C,
                    D=None, z=None,
                    prev_Bx=prev_Bx,
                    beta=beta_scalar,
                    gamma=gamma_scalar,
                )
                D_val = self.D.to(dtype)
                if self.D_has_hdim:
                    y = y + x * rearrange(D_val, "(h p) -> 1 h p 1", p=self.headdim)
                else:
                    y = y + x * rearrange(D_val, "h -> 1 h 1 1")
                y = self.mimo_out_proj(rearrange(y, "b h p r -> b h (p r)"))
                y = rearrange(y, "b h p -> b (h p)")
            else:
                # SISO: kernel also applies D (scalar per head)
                y = mamba3_state_update(
                    ssm_state, x, dt, A,
                    B, C,
                    D=self.D if not self.D_has_hdim else None,
                    z=None,  # we handle norm+gate separately
                    prev_Bx=prev_Bx,
                    beta=beta_scalar,
                    gamma=gamma_scalar,
                )
                if self.D_has_hdim:
                    y = y + rearrange(self.D.to(dtype), "(h p) -> h p", p=self.headdim) * x
                y = rearrange(y, "b h p -> b (h p)")
            # prev_Bx was updated in-place by the kernel
        else:
            # PyTorch fallback (MIMO or no Triton)
            if self.is_mimo:
                Bx = torch.einsum("bhpr,bhnr->bhpn", x.float(), B.float())
            else:
                Bx = torch.einsum("bhp,bhn->bhpn", x.float(), B.float())

            gamma_4d = gamma_scalar.unsqueeze(-1).unsqueeze(-1)
            if beta_scalar is not None:
                beta_4d = beta_scalar.unsqueeze(-1).unsqueeze(-1)
                ssm_state.copy_(
                    ssm_state * rearrange(dA, "b h -> b h 1 1")
                    + gamma_4d * Bx
                    + beta_4d * prev_Bx
                )
            else:
                ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + gamma_4d * Bx)

            # Store current Bx as prev_Bx for next step
            conv_state[:, :prev_Bx_flat_size] = Bx.view(-1, prev_Bx_flat_size)

            # Output: y = C^T @ h
            if self.is_mimo:
                y = torch.einsum("bhpn,bhnr->bhpr", ssm_state.to(dtype), C)
                if self.D_has_hdim:
                    y = y + x * rearrange(self.D.to(dtype), "(h p) -> 1 h p 1", p=self.headdim)
                else:
                    y = y + x * rearrange(self.D.to(dtype), "h -> 1 h 1 1")
                y = self.mimo_out_proj(rearrange(y, "b h p r -> b h (p r)"))
                y = rearrange(y, "b h p -> b (h p)")
            else:
                y = torch.einsum("bhpn,bhn->bhp", ssm_state.to(dtype), C)
                if self.D_has_hdim:
                    y = y + rearrange(self.D.to(dtype), "(h p) -> h p", p=self.headdim) * x
                else:
                    y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
                y = rearrange(y, "b h p -> b (h p)")

        # Gated output
        if self.rmsnorm:
            y = self.norm(y, z)
        else:
            y = y * F.silu(z)

        out = self.out_proj(y)
        if self.process_group is not None:
            reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
            out = reduce_fn(out, self.process_group)
        return out.unsqueeze(1), conv_state, ssm_state

    def _conv_state_size(self):
        """Size of the flattened conv_state for inference."""
        prev_Bx_size = self.nheads * self.headdim * self.d_state
        theta_size = self.nheads * (self.d_state // 2) if self.use_rope else 0
        return prev_Bx_size + theta_size

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        # Always float32 to avoid precision loss when storing Bx and cumulative theta
        conv_state = torch.zeros(
            batch_size, self._conv_state_size(), device=device, dtype=torch.float32,
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size, self._conv_state_size(),
                device=self.in_proj.weight.device,
                dtype=torch.float32,
            )
            ssm_state = torch.zeros(
                batch_size, self.nheads, self.headdim, self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
