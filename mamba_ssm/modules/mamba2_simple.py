# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None

if RMSNormGated is None:
    class RMSNormGated(nn.Module):
        def __init__(self, dim, eps=1e-5, norm_before_gate=False, **kwargs):
            super().__init__()
            self.eps = eps
            self.norm_before_gate = norm_before_gate
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x, z=None):
            if z is not None:
                gated = F.silu(z)
                if self.norm_before_gate:
                    variance = x.pow(2).mean(-1, keepdim=True)
                    normed = x * torch.rsqrt(variance + self.eps) * self.weight
                    return normed * gated
                else:
                    out = x * gated
                    variance = out.pow(2).mean(-1, keepdim=True)
                    return out * torch.rsqrt(variance + self.eps) * self.weight
            else:
                variance = x.pow(2).mean(-1, keepdim=True)
                return x * torch.rsqrt(variance + self.eps) * self.weight

if mamba_chunk_scan_combined is None:
    def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None, seq_idx=None, initial_states=None, **kwargs):
        batch, seqlen, nheads, headdim = x.shape
        ngroups = B.shape[2]
        
        # Pad sequence length to a multiple of chunk_size
        pad_len = (chunk_size - (seqlen % chunk_size)) % chunk_size
        if pad_len > 0:
            x = rearrange(F.pad(rearrange(x, 'b l h d -> b h d l'), (0, pad_len), value=0.0), 'b h d l -> b l h d')
            dt = rearrange(F.pad(rearrange(dt, 'b l h -> b h l'), (0, pad_len), value=0.0), 'b h l -> b l h')
            B = rearrange(F.pad(rearrange(B, 'b l g d -> b g d l'), (0, pad_len), value=0.0), 'b g d l -> b l g d')
            C = rearrange(F.pad(rearrange(C, 'b l g d -> b g d l'), (0, pad_len), value=0.0), 'b g d l -> b l g d')
            if z is not None:
                z = rearrange(F.pad(rearrange(z, 'b l h d -> b h d l'), (0, pad_len), value=0.0), 'b h d l -> b l h d')

        if ngroups < nheads:
            B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
            C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
        A_discrete = A.view(1, 1, -1) * dt
        X_discrete = x * dt.unsqueeze(-1)
        from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete
        y, final_state = ssd_minimal_discrete(X_discrete, A_discrete, B, C, chunk_size, initial_states=initial_states)
        
        if pad_len > 0:
            y = y[:, :-pad_len]
            x = x[:, :-pad_len]
            if z is not None:
                z = z[:, :-pad_len]
                
        if D is not None:
            if D.dim() == 2:
                y = y + x * D.unsqueeze(0).unsqueeze(0)
            else:
                y = y + x * D.view(1, 1, -1, 1)
        if z is not None:
            y = y * F.silu(z)
        return y

if mamba_split_conv1d_scan_combined is None:
    def mamba_split_conv1d_scan_combined(
        zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D=None, chunk_size=256,
        seq_idx=None, activation="swish", rmsnorm_weight=None, rmsnorm_eps=1e-5,
        outproj_weight=None, outproj_bias=None, headdim=128, ngroups=1,
        norm_before_gate=False, initial_states=None, **kwargs
    ):
        batch, seqlen, _ = zxbcdt.shape
        d_inner = rmsnorm_weight.shape[0]
        nheads = A.shape[0]
        d_state = (zxbcdt.shape[-1] - 2 * d_inner - nheads) // (2 * ngroups)
        z, xBC, dt = torch.split(zxbcdt, [d_inner, d_inner + 2 * ngroups * d_state, nheads], dim=-1)
        dt = F.softplus(dt + dt_bias)
        if causal_conv1d_fn is not None:
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=conv1d_weight,
                bias=conv1d_bias,
                activation=activation,
            ).transpose(1, 2)
        else:
            d_conv = conv1d_weight.shape[-1]
            xBC_padded = F.pad(xBC.transpose(1, 2), (d_conv - 1, 0))
            xBC_conv = F.conv1d(xBC_padded, conv1d_weight.unsqueeze(1), bias=conv1d_bias, groups=xBC.shape[-1])
            xBC = F.silu(xBC_conv).transpose(1, 2)
        x, B, C = torch.split(xBC, [d_inner, ngroups * d_state, ngroups * d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=ngroups),
            chunk_size=chunk_size,
            D=D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        gated = F.silu(z)
        if norm_before_gate:
            variance = y.pow(2).mean(-1, keepdim=True)
            normed = y * torch.rsqrt(variance + rmsnorm_eps) * rmsnorm_weight
            y = normed * gated
        else:
            out = y * gated
            variance = out.pow(2).mean(-1, keepdim=True)
            y = out * torch.rsqrt(variance + rmsnorm_eps) * rmsnorm_weight
        return F.linear(y, outproj_weight, outproj_bias)


class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out
