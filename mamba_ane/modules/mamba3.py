"""
mamba_ane/modules/mamba3.py — ANE-native Mamba3 building blocks.

RMSNormANE: ANE-friendly RMSNorm (no .float() casts; x*x instead of pow(2)).
MambaBlock:  Full Mamba3 SISO with RoPE, trapezoid discretization, SSM
             recurrence, and stateful buffers ready for ct.StateType export.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormANE(nn.Module):
    """ANE-friendly RMSNorm. No .float() casts; x*x replaces pow(2)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = (x * x).mean(-1, keepdim=True)
        rms_inv = torch.rsqrt(variance + self.eps)
        return x * rms_inv * self.weight


class MambaBlock(nn.Module):
    """
    Mamba3 SISO — ANE-native implementation.

    Stages:
      1. in_proj + explicit slicing
      2. Activations (softplus, sigmoid, silu) + dt_bias
      3. K/Q RMSNorm + head expand + bias
      4. RoPE angle accumulation
      5. Trapezoid discretization (alpha/beta/gamma)
      6. SSM recurrence via outer products; in-place state updates
      7. Output contraction + D skip connection
      8. out_proj

    State buffers (angle_state, ssm_state, k_state, v_state) are registered
    with register_buffer() so CoreML ct.StateType export picks them up.

    Args:
        d_model:   input/output dimension
        d_state:   SSM state dimension (default 64)
        headdim:   head dimension (default 64)
        num_heads: number of attention heads (default 8)
        ema_alpha: kept for API compatibility; not used in forward
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 64,
        headdim: int = 64,
        num_heads: int = 8,
        ema_alpha: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.num_heads = num_heads
        self.ema_alpha = ema_alpha

        self.d_inner = num_heads * headdim
        self.num_rope_angles = 16
        self.rotary_dim = self.num_rope_angles * 2  # 32

        # Projection slice boundaries
        self._s0 = self.d_inner
        self._s1 = self._s0 + self.d_inner
        self._s2 = self._s1 + d_state
        self._s3 = self._s2 + d_state
        self._s4 = self._s3 + num_heads
        self._s5 = self._s4 + num_heads
        self._s6 = self._s5 + num_heads
        self._s7 = self._s6 + self.num_rope_angles

        self.in_proj  = nn.Linear(d_model, self._s7, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.B_norm = RMSNormANE(d_state)
        self.C_norm = RMSNormANE(d_state)
        self.B_bias = nn.Parameter(torch.zeros(1, num_heads, d_state))
        self.C_bias = nn.Parameter(torch.zeros(1, num_heads, d_state))

        self.dt_bias = nn.Parameter(torch.zeros(num_heads))
        self.D       = nn.Parameter(torch.ones(num_heads))

        self.register_buffer("angle_state", torch.zeros(1, num_heads, self.num_rope_angles))
        self.register_buffer("ssm_state",   torch.zeros(1, num_heads, headdim, d_state))
        self.register_buffer("k_state",     torch.zeros(1, 1, num_heads, d_state))
        self.register_buffer("v_state",     torch.zeros(1, num_heads, headdim))

    def apply_rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Stateless RoPE rotation; x: (1,H,D), cos/sin: (1,H,num_rope_angles)."""
        x_rot  = x[..., :self.rotary_dim]
        x_pass = x[..., self.rotary_dim:]
        x0 = x_rot[..., 0::2]
        x1 = x_rot[..., 1::2]
        out_rot = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)
        return torch.cat([out_rot, x_pass], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: project + slice
        zxBCdt     = self.in_proj(x)
        z_raw      = zxBCdt[..., :self._s0]
        x_raw      = zxBCdt[..., self._s0:self._s1]
        B_raw      = zxBCdt[..., self._s1:self._s2]
        C_raw      = zxBCdt[..., self._s2:self._s3]
        dd_dt      = zxBCdt[..., self._s3:self._s4]
        dd_A       = zxBCdt[..., self._s4:self._s5]
        trap_raw   = zxBCdt[..., self._s5:self._s6]
        angles_raw = zxBCdt[..., self._s6:self._s7]

        # Stage 2: activations
        A   = -(F.softplus(dd_A) + 1e-4)
        DT  = F.softplus(dd_dt + self.dt_bias)
        lam = torch.sigmoid(trap_raw)
        z   = F.silu(z_raw)

        # Stage 3: K/Q norm + expand + bias
        K_norm     = self.B_norm(B_raw.view(1, 1, 1, self.d_state))
        Q_norm     = self.C_norm(C_raw.view(1, 1, 1, self.d_state))
        K_expanded = K_norm.repeat(1, 1, self.num_heads, 1).squeeze(1)
        Q_expanded = Q_norm.repeat(1, 1, self.num_heads, 1).squeeze(1)
        K_pre      = K_expanded + self.B_bias
        Q_pre      = Q_expanded + self.C_bias

        # Stage 4: RoPE
        angles      = angles_raw.view(1, 1, self.num_rope_angles).repeat(1, self.num_heads, 1)
        delta_theta = torch.tanh(angles) * math.pi * DT.unsqueeze(-1)
        theta       = self.angle_state + delta_theta
        K_rot       = self.apply_rope(K_pre, torch.cos(theta), torch.sin(theta))
        Q_rot       = self.apply_rope(Q_pre, torch.cos(theta), torch.sin(theta))

        # Stage 5: trapezoid coefficients
        alpha = torch.exp(A * DT)
        beta  = (1.0 - lam) * DT * alpha
        gamma = lam * DT
        g4    = gamma[:, :, None, None]
        b4    = beta[:, :, None, None]

        # Stage 6: SSM recurrence
        V           = x_raw.reshape(1, self.num_heads, self.headdim)
        outer_curr  = torch.matmul(V.unsqueeze(-1),              K_rot.unsqueeze(-2))
        outer_prev  = torch.matmul(self.v_state.unsqueeze(-1),   self.k_state.squeeze(1).unsqueeze(-2))
        new_h       = alpha[:, :, None, None] * self.ssm_state + b4 * outer_prev + g4 * outer_curr

        self.ssm_state[:]   = new_h
        self.k_state[:]     = K_rot.unsqueeze(1)
        self.v_state[:]     = V
        self.angle_state[:] = theta

        # Stage 7-8: output
        y_ssm = torch.matmul(new_h, Q_rot.unsqueeze(-1)).squeeze(-1)
        y_ssm = y_ssm + (V * self.D.view(1, -1, 1))
        return self.out_proj(y_ssm.reshape(1, self.d_inner) * z)
