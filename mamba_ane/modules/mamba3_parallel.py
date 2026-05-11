# mamba_ane/modules/mamba3_parallel.py
"""
Mamba3ParallelPortable — ANE-safe parallel (sequence-in/sequence-out) Mamba-3 SISO.

Algorithm: unchunked SSD via materialised (B, H, L, L) L-matrix.
No register_buffer, no ct.StateType — stateless, static shapes baked at trace time.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ane.modules.mamba3 import RMSNormANE


class Mamba3ParallelPortable(nn.Module):
    """
    Pure-PyTorch parallel Mamba-3 SISO forward pass.

    forward(u: (B, L, d_model)) -> (B, L, d_model)

    No stateful buffers.  Load weights from an original Mamba3 instance via
    load_from_original().  Works with any L at runtime; shapes are baked in
    statically when torch.jit.trace is called for CoreML export.

    ANE constraints observed throughout:
      - No .float() casts — input dtype is preserved end-to-end.
      - x * x instead of pow(x, 2) in RMSNormANE.
      - No torch.split — explicit index slices precomputed in __init__.
      - Outer products via matmul(unsqueeze(-1), unsqueeze(-2)).
      - additive A floor: -(softplus(x) + A_floor) instead of clamp.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        rope_fraction: float = 0.5,
        A_floor: float = 1e-4,
        rms_eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim
        self.headdim = headdim
        self.A_floor = A_floor

        # RoPE sizing (matches mamba_ssm Mamba3 with rope_fraction=0.5)
        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2   # 16
        self.rotary_dim      = split_tensor_size         # 32

        # Projection slice boundaries (no torch.split in forward)
        self._s0 = self.d_inner                              # z   end
        self._s1 = self._s0 + self.d_inner                   # x   end
        self._s2 = self._s1 + d_state                        # B   end
        self._s3 = self._s2 + d_state                        # C   end
        self._s4 = self._s3 + self.nheads                    # dd_dt end
        self._s5 = self._s4 + self.nheads                    # dd_A  end
        self._s6 = self._s5 + self.nheads                    # trap  end
        self._s7 = self._s6 + self.num_rope_angles           # angles end

        self.in_proj  = nn.Linear(d_model, self._s7, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.B_norm = RMSNormANE(d_state, eps=rms_eps)
        self.C_norm = RMSNormANE(d_state, eps=rms_eps)

        # (nheads, d_state) — squeezed from Mamba3's (nheads, 1, d_state)
        self.B_bias  = nn.Parameter(torch.zeros(self.nheads, d_state))
        self.C_bias  = nn.Parameter(torch.zeros(self.nheads, d_state))
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.D       = nn.Parameter(torch.ones(self.nheads))

    # ------------------------------------------------------------------
    def apply_rope(
        self,
        x:   torch.Tensor,   # (B, L, H, D)
        cos: torch.Tensor,   # (B, L, H, num_rope_angles)
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise RoPE — works for any leading (B, L) dims."""
        x_rot  = x[..., :self.rotary_dim]   # (B, L, H, 32)
        x_pass = x[..., self.rotary_dim:]
        x0 = x_rot[..., 0::2]               # (B, L, H, 16)
        x1 = x_rot[..., 1::2]
        out_rot = torch.stack(
            [x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1
        ).flatten(-2)                        # (B, L, H, 32)
        return torch.cat([out_rot, x_pass], dim=-1)

    # ------------------------------------------------------------------
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, d_model) → (B, L, d_model)"""
        B, L, _ = u.shape

        # Stage 1: in_proj + explicit slices
        proj       = self.in_proj(u)
        z_raw      = proj[..., :self._s0]                   # (B, L, d_inner)
        x_raw      = proj[..., self._s0:self._s1]
        B_raw      = proj[..., self._s1:self._s2]           # (B, L, d_state)
        C_raw      = proj[..., self._s2:self._s3]
        dd_dt      = proj[..., self._s3:self._s4]           # (B, L, nheads)
        dd_A       = proj[..., self._s4:self._s5]
        trap_raw   = proj[..., self._s5:self._s6]
        angles_raw = proj[..., self._s6:self._s7]           # (B, L, num_rope_angles)

        # Stage 2: activations — no .float() casts
        A   = -(F.softplus(dd_A)   + self.A_floor)         # (B, L, H)
        DT  = F.softplus(dd_dt + self.dt_bias)             # (B, L, H)
        lam = torch.sigmoid(trap_raw)                       # (B, L, H)
        z   = F.silu(z_raw)                                 # (B, L, d_inner)
        ADT = A * DT                                        # log(alpha) per position

        # Stage 3: K/Q norm + expand to heads + bias
        K_norm = self.B_norm(B_raw)                         # (B, L, d_state)
        Q_norm = self.C_norm(C_raw)
        # expand: (B, L, 1, d_state) -> (B, L, H, d_state)
        K_pre = K_norm.unsqueeze(2).expand(-1, -1, self.nheads, -1) + self.B_bias
        Q_pre = Q_norm.unsqueeze(2).expand(-1, -1, self.nheads, -1) + self.C_bias

        # Stage 4: RoPE — cumulative angle along sequence
        angles      = angles_raw.unsqueeze(2).expand(-1, -1, self.nheads, -1)
        delta_theta = torch.tanh(angles) * math.pi * DT.unsqueeze(-1)
        theta       = torch.cumsum(delta_theta, dim=1)      # (B, L, H, num_rope_angles)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        K_rot = self.apply_rope(K_pre, cos_t, sin_t)        # (B, L, H, d_state)
        Q_rot = self.apply_rope(Q_pre, cos_t, sin_t)

        # Stage 5: discretization coefficients
        gamma     = lam * DT                                # (B, L, H)
        # beta_next[b,r,h] = (1-lam[b,r+1,h]) * DT[b,r+1,h]; zero at last position
        beta_next = torch.zeros_like(gamma)
        beta_next[:, :-1, :] = (1.0 - lam[:, 1:, :]) * DT[:, 1:, :]

        # Stage 6: L-matrix (vectorised across all heads)
        # cum_log_alpha: (B, L, H)
        cum = torch.cumsum(ADT, dim=1)

        # decay_mat[b,t,r,h] = exp(cum[b,t,h] - cum[b,r,h])
        # shape broadcast: (B, L, 1, H) - (B, 1, L, H) = (B, L, L, H)
        log_decay = cum.unsqueeze(2) - cum.unsqueeze(1)

        # causal mask (lower-triangular) — constant at trace time
        causal = torch.tril(torch.ones(L, L, device=u.device, dtype=u.dtype))
        # apply mask in log-space before exp: prevents inf overflow (exp(+large)*0 = NaN)
        log_decay = log_decay + (1.0 - causal).unsqueeze(-1) * (-1e9)
        decay_mat = torch.exp(log_decay) * causal.unsqueeze(-1)        # (B, L, L, H)

        # g_col = gamma + beta_next; diagonal needs gamma only
        g_col  = gamma + beta_next                          # (B, L, H)
        L_mat  = decay_mat * g_col.unsqueeze(1)             # (B, L, L, H) — col broadcast
        # diagonal correction: subtract beta_next from L_mat[t,t,h]
        eye    = torch.eye(L, device=u.device, dtype=u.dtype)
        L_mat  = L_mat - eye.unsqueeze(-1) * beta_next.unsqueeze(1)  # (B, L, L, H)

        # Stage 7: attention-like contraction
        # Permute all to (B, H, L, *) for batch matmul
        K_bhl    = K_rot.permute(0, 2, 1, 3)               # (B, H, L, d_state)
        Q_bhl    = Q_rot.permute(0, 2, 1, 3)
        V        = x_raw.reshape(B, L, self.nheads, self.headdim)
        V_bhl    = V.permute(0, 2, 1, 3)                   # (B, H, L, headdim)
        L_bhll   = L_mat.permute(0, 3, 1, 2)               # (B, H, L, L)

        score    = torch.matmul(Q_bhl, K_bhl.transpose(-1, -2))     # (B, H, L, L)
        Y_bhl    = torch.matmul(L_bhll * score, V_bhl)              # (B, H, L, headdim)
        Y        = Y_bhl.permute(0, 2, 1, 3)               # (B, L, H, headdim)

        # Stage 8: D skip + gate + out_proj
        Y     = Y + V * self.D.view(1, 1, -1, 1)
        y_flat = Y.reshape(B, L, self.d_inner)              # (B, L, d_inner)
        return self.out_proj(y_flat * z)                    # (B, L, d_model)

    # ------------------------------------------------------------------
    def load_from_original(self, src: nn.Module) -> None:
        """
        Copy weights from a mamba_ssm.modules.mamba3.Mamba3 instance (SISO, ngroups=1).

        Handles the B_bias/C_bias reshape: Mamba3 stores (nheads, 1, d_state),
        portable stores (nheads, d_state).
        """
        with torch.no_grad():
            self.in_proj.weight.copy_(src.in_proj.weight)
            self.out_proj.weight.copy_(src.out_proj.weight)
            self.dt_bias.copy_(src.dt_bias)
            self.D.copy_(src.D)
            self.B_norm.weight.copy_(src.B_norm.weight)
            self.C_norm.weight.copy_(src.C_norm.weight)
            # Mamba3 B_bias shape: (nheads, 1, d_state) → squeeze dim 1
            self.B_bias.copy_(src.B_bias.squeeze(1))
            self.C_bias.copy_(src.C_bias.squeeze(1))
