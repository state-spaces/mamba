"""Minimal, Megatron-free replica of Megatron-LM's ``MambaMixer`` training path,
so the Helion kernels can be autotuned / profiled inside the mamba repo against
the exact kernel set production uses.

Unlike ``mamba_ssm.modules.mamba2.Mamba2`` -- which *fuses* the gated RMSNorm and
out_proj into ``mamba_split_conv1d_scan_combined`` and so runs the plain Triton
``_layer_norm_bwd`` -- this mirrors Megatron's ``MambaMixer._ssm_training``: the
kernel is called WITHOUT ``rmsnorm_weight`` / ``outproj_weight`` and a separate
``RMSNormGated`` runs afterwards, routing through Helion's ``_layer_norm_bwd``.
The un-fused layout must be mirrored exactly to autotune the right kernels.

Reference: ``MambaMixer._ssm_training`` in Megatron-LM at commit 571370c --
https://github.com/NVIDIA/Megatron-LM/blob/571370c829ca768fe37244f4e2e7f28d8accc4ab/megatron/core/ssm/mamba_mixer.py#L683
Verified bit-for-bit identical (fwd+bwd) to a ``Float16Module``-wrapped Megatron
``MambaMixer`` at tp=cp=1, except for rounding-level in_proj/out_proj differences
(``nn.Linear`` here vs Megatron's TE/fused bf16 GEMM) that don't affect the kernels.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from einops import rearrange

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class MambaMixerMin(nn.Module):
    """Single-GPU MambaMixer mirroring Megatron's training-path layout.

    Forward expects ``hidden_states`` of shape ``(seqlen, batch, d_model)`` (the
    ``l b d`` layout Megatron uses) and returns the same shape.
    """

    def __init__(
        self,
        d_model: int,
        nheads: int,
        headdim: int = 64,
        d_state: int = 128,
        ngroups: int = 8,
        d_conv: int = 4,
        chunk_size: int = 128,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        D_has_hdim: bool = False,
        bias: bool = False,
        conv_bias: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        assert not bias
        assert not norm_before_gate
        assert nheads % ngroups == 0, "nheads must be evenly divisible by ngroups"

        self.d_model = d_model
        self.nheads = nheads
        self.headdim = headdim
        self.d_state = d_state
        self.ngroups = ngroups
        self.d_conv = d_conv
        self.chunk_size = chunk_size
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.D_has_hdim = D_has_hdim
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.d_inner = nheads * headdim

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state  # x B C
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        # dt bias so that softplus(dt_bias) lands in (dt_min, dt_max).
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))  # keep A_log in fp32

        self.D = nn.Parameter(
            torch.ones(self.d_inner if self.D_has_hdim else self.nheads, device=device)
        )  # keep in fp32

        if self.rmsnorm:
            self.norm = RMSNormGated(
                self.d_inner,
                eps=1e-5,
                group_size=self.d_inner // self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **factory_kwargs,
            )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Megatron -- and so this constructor above -- initializes D and the gated-norm
        # weight to all ones. Randomize them: with every element equal to 1, a kernel
        # that indexes D by the wrong head, or normalizes with the wrong group, still
        # produces the right answer, so those bugs would be invisible to the tests or to
        # autotune_baseline_fn's accuracy check against Triton. A trained checkpoint has
        # non-trivial values anyway; the norm weight is kept around the ones-init
        # magnitude so the norm still behaves like a trained one.
        with torch.no_grad():
            self.D.normal_()
            if self.rmsnorm:
                self.norm.weight.uniform_(0.5, 1.5)

        # Production wraps the whole model in Float16Module, whose .bfloat16()/.half()
        # casts EVERY float param to params_dtype -- including A_log and D, which the
        # constructor above deliberately created in fp32. The Megatron profiler
        # replicates this with Float16Module(...).bfloat16(); without it, D reaches
        # the Helion chunk-scan kernels as fp32 instead of bf16/fp16 (it is passed in
        # without a .float() in the forward), diverging from real training. Mirror
        # that whole-module cast here so the kernels see production dtypes.
        if dtype in (torch.float16, torch.bfloat16):
            self.to(dtype)

    def forward(
        self, hidden_states: torch.Tensor, seq_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """hidden_states: (seqlen, batch, d_model) -> (seqlen, batch, d_model)."""
        _, _, dim = hidden_states.shape
        assert dim == self.d_model

        zxBCdt = self.in_proj(hidden_states)  # (l, b, d_in_proj)

        # The Helion / Triton split-conv-scan kernel works in (b, l, d) layout.
        zxBCdt = rearrange(zxBCdt, "l b d -> b l d").contiguous()

        A = -torch.exp(self.A_log.float())  # (nheads,)

        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )

        # NOTE: rmsnorm_weight / outproj_weight are intentionally NOT passed, so the
        # gated RMSNorm and out_proj stay un-fused (the Megatron training layout).
        y = mamba_split_conv1d_scan_combined(
            zxBCdt,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias.float(),
            A,
            D=(
                rearrange(self.D.float(), "(h p) -> h p", p=self.headdim)
                if self.D_has_hdim
                else self.D
            ),
            chunk_size=self.chunk_size,
            activation=self.activation,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            seq_idx=seq_idx,
            **dt_limit_kwargs,
        )

        y = rearrange(y, "b l d -> l b d").contiguous()

        if self.rmsnorm:
            # Separate gated RMSNorm -> routes through Helion _layer_norm_bwd.
            y = self.norm(y)

        out = self.out_proj(y)
        return out
