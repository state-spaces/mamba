# Copyright (c) 2024, Tri Dao, Albert Gu.
# Mamba-3 simplified implementation (no TP, no inference/step support).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

from mamba_ssm.modules.mamba3 import apply_rotary_emb, compute_cumulative_rotary


class Mamba3Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        bias=False,
        # Mamba-3 specific
        use_rope=True,
        use_trapezoidal=True,
        use_bc_norm=True,
        use_bc_bias=True,
        mimo_rank=0,
        # Kernel options
        chunk_size=256,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # Mamba-3 specific
        self.use_rope = use_rope
        self.use_trapezoidal = use_trapezoidal
        self.use_bc_norm = use_bc_norm
        self.use_bc_bias = use_bc_bias
        self.mimo_rank = mimo_rank
        self.is_mimo = mimo_rank > 0

        bc_dim = self.ngroups * self.d_state
        bc_proj_dim = bc_dim * self.mimo_rank if self.is_mimo else bc_dim
        theta_dim = self.nheads * (self.d_state // 2) if self.use_rope else 0
        lambda_dim = self.nheads if self.use_trapezoidal else 0

        d_in_proj = (
            self.d_inner + self.d_inner + bc_proj_dim + bc_proj_dim
            + self.nheads + theta_dim + lambda_dim
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        if self.is_mimo:
            self.x_mimo_proj = nn.Linear(self.headdim, self.headdim * self.mimo_rank, bias=False, **factory_kwargs)
            self.mimo_out_proj = nn.Linear(self.headdim * self.mimo_rank, self.headdim, bias=False, **factory_kwargs)

        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )
            self.init_states._no_weight_decay = True

        # dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True

        # D
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # BC Norm
        if self.use_bc_norm:
            self.B_norm = nn.RMSNorm(self.d_state, eps=1e-5, **factory_kwargs)
            self.C_norm = nn.RMSNorm(self.d_state, eps=1e-5, **factory_kwargs)

        # BC Bias (init=1.0 per paper Table 9a)
        if self.use_bc_bias:
            self.B_bias = nn.Parameter(torch.ones(self.nheads, self.d_state, **factory_kwargs))
            self.C_bias = nn.Parameter(torch.ones(self.nheads, self.d_state, **factory_kwargs))

        # Output norm
        assert RMSNormGated is not None
        self.norm = RMSNormGated(
            self.d_inner, eps=1e-5, norm_before_gate=False,
            group_size=self.d_inner // self.ngroups, **factory_kwargs,
        )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Split sizes
        self._split_sizes = [self.d_inner, self.d_inner, bc_proj_dim, bc_proj_dim, self.nheads]
        if self.use_rope:
            self._split_sizes.append(theta_dim)
        if self.use_trapezoidal:
            self._split_sizes.append(lambda_dim)

    def forward(self, u, seq_idx=None):
        """u: (B, L, D). Returns same shape."""
        batch, seqlen, dim = u.shape

        proj = self.in_proj(u)
        A = -torch.exp(self.A_log.float())

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

        # Process dt
        dt = F.softplus(dt_raw + self.dt_bias)
        if self.dt_limit != (0.0, float("inf")):
            dt = dt.clamp(min=self.dt_limit[0], max=self.dt_limit[1])

        # Reshape B, C
        if self.is_mimo:
            B = rearrange(B_raw, "b l (g n r) -> b l g n r", g=self.ngroups, r=self.mimo_rank)
            C = rearrange(C_raw, "b l (g n r) -> b l g n r", g=self.ngroups, r=self.mimo_rank)
        else:
            B = rearrange(B_raw, "b l (g n) -> b l g n", g=self.ngroups)
            C = rearrange(C_raw, "b l (g n) -> b l g n", g=self.ngroups)

        # BC Norm
        if self.use_bc_norm:
            orig = B.shape
            if self.is_mimo:
                # MIMO: (b, l, g, d_state, mimo_rank) — move rank before d_state to normalize correctly
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
            B = repeat(B, "b l g n r -> b l (g h) n r", h=nheads_per_group)
            C = repeat(C, "b l g n r -> b l (g h) n r", h=nheads_per_group)
        else:
            B = repeat(B, "b l g n -> b l (g h) n", h=nheads_per_group)
            C = repeat(C, "b l g n -> b l (g h) n", h=nheads_per_group)

        # BC Bias (applied per-head after expansion for true head-specificity)
        if self.use_bc_bias:
            if self.is_mimo:
                B = B + self.B_bias.view(1, 1, self.nheads, self.d_state, 1)
                C = C + self.C_bias.view(1, 1, self.nheads, self.d_state, 1)
            else:
                B = B + self.B_bias
                C = C + self.C_bias

        # Apply RoPE
        if self.use_rope and theta_raw is not None:
            theta = rearrange(theta_raw, "b l (h d) -> b l h d", h=self.nheads)
            theta_cumsum = torch.cumsum(theta, dim=1)
            cos_t, sin_t = compute_cumulative_rotary(theta_cumsum, self.d_state)
            if self.is_mimo:
                B_parts = [apply_rotary_emb(B[:, :, :, :, r], cos_t, sin_t) for r in range(self.mimo_rank)]
                C_parts = [apply_rotary_emb(C[:, :, :, :, r], cos_t, sin_t) for r in range(self.mimo_rank)]
                B = torch.stack(B_parts, dim=-1)
                C = torch.stack(C_parts, dim=-1)
            else:
                B = apply_rotary_emb(B, cos_t, sin_t)
                C = apply_rotary_emb(C, cos_t, sin_t)

        # Process lambda
        lam = torch.sigmoid(lam_raw) if self.use_trapezoidal and lam_raw is not None else None

        # Process x
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        if self.is_mimo:
            x = self.x_mimo_proj(x)
            x = rearrange(x, "b l h (p r) -> b l h p r", r=self.mimo_rank)

        # Initial states
        initial_states = repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None

        # Run step-by-step recurrence (reference impl)
        y = self._recurrence(x, dt, A, B, C, lam=lam, initial_states=initial_states, seq_idx=seq_idx)

        # D skip + flatten (cast D to y's dtype to avoid float32 promotion)
        D = self.D.to(dtype=y.dtype)
        if self.is_mimo:
            # y: (B, L, H, P, R), x: (B, L, H, P, R)
            y = y + x * repeat(D, "h -> 1 1 h 1 1")
            y = self.mimo_out_proj(rearrange(y, "b l h p r -> b l h (p r)"))
            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            y = rearrange(y, "b l h p -> b l (h p)")
            x_flat = rearrange(x, "b l h p -> b l (h p)")
            y = y + x_flat * repeat(D, "h -> (h p)", p=self.headdim).reshape(1, 1, -1)

        # Norm + gate
        y = self.norm(y, z)
        out = self.out_proj(y)
        return out

    def _recurrence(self, x, dt, A, B, C, lam=None, initial_states=None, seq_idx=None):
        """Step-by-step reference recurrence with trapezoidal discretization."""
        batch, seqlen = x.shape[0], x.shape[1]
        nheads = self.nheads
        headdim = self.headdim
        dstate = self.d_state

        alpha = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, nheads, 1))  # (B, L, H, 1)

        if self.use_trapezoidal and lam is not None:
            gamma = lam * dt  # λ * Δt
            beta = (1 - lam) * dt * torch.exp(dt * A.view(1, 1, nheads))
        else:
            gamma = dt  # Euler fallback
            beta = None

        h = torch.zeros(batch, nheads, headdim, dstate, device=x.device, dtype=torch.float32)
        if initial_states is not None:
            h = initial_states.float()

        ys = []
        prev_Bx = None

        for t in range(seqlen):
            # Reset state at document boundaries
            if seq_idx is not None and t > 0:
                boundary = (seq_idx[:, t] != seq_idx[:, t - 1])  # (batch,)
                if boundary.any():
                    mask = boundary.view(-1, 1, 1, 1).float()
                    h = h * (1 - mask)
                    if initial_states is not None:
                        h = h + mask * initial_states.float()
                    prev_Bx = None if boundary.all() else (
                        prev_Bx * (1 - mask) if prev_Bx is not None else None
                    )

            x_t = x[:, t]
            B_t = B[:, t]
            C_t = C[:, t]

            if self.is_mimo:
                Bx_t = torch.einsum("bhpr,bhnr->bhpn", x_t.float(), B_t.float())
            else:
                Bx_t = torch.einsum("bhp,bhn->bhpn", x_t.float(), B_t.float())

            alpha_t = alpha[:, t].unsqueeze(-1)  # (B, H, 1, 1)
            gamma_t = gamma[:, t].unsqueeze(-1).unsqueeze(-1)

            h = alpha_t * h + gamma_t * Bx_t

            if beta is not None and prev_Bx is not None:
                beta_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)
                h = h + beta_t * prev_Bx

            prev_Bx = Bx_t

            if self.is_mimo:
                y_t = torch.einsum("bhpn,bhnr->bhpr", h.to(C_t.dtype), C_t)
            else:
                y_t = torch.einsum("bhpn,bhn->bhp", h.to(C_t.dtype), C_t)

            ys.append(y_t)

        return torch.stack(ys, dim=1)
