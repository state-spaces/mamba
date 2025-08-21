# Copyright (c) 2024, Tri Dao, Albert Gu.
from typing import Optional, Tuple, Type, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class DiffBlockPaper(nn.Module):
    """
    Diff-Mamba block: Add->Norm -> (mixer1 || mixer2) -> Norm each -> subtract with λ -> Linear -> Norm
                      -> (optional MLP sublayer like vanilla Block)

    Returns (hidden_states, residual) with the SAME contract as mamba_ssm.modules.block.Block:
      - If no MLP: residual is the pre-norm Add sum, hidden_states is the sublayer output (no add here).
      - If MLP: we do residual += hidden_states before norm2+MLP, as in vanilla.
    """

    def __init__(
        self,
        dim: int,
        mixer_cls1: Callable[[int], nn.Module],
        mixer_cls2: Callable[[int], nn.Module],
        mlp_cls: Callable[[int], nn.Module],
        *,
        norm_cls: Callable[[int], nn.Module] = nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        layer_idx: int = 0,
        use_postscale: bool = False,     # optional extra scaling by (1 - lambda_init)
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = dim
        self.layer_idx = layer_idx
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.use_postscale = bool(use_postscale)

        # Prenorm for Add->Norm input
        self.norm = norm_cls(dim)

        # Two parallel mixers
        self.mixer1 = mixer_cls1(dim)
        self.mixer2 = mixer_cls2(dim)

        # Post-mixer norms (separate for each branch) and post-sub norm
        self.subln      = norm_cls(dim)

        # Per-layer scalar λ (σ(λ̄)+λ_init), λ̄ initialized very negative -> small λ
        self.lambda_init = 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * self.layer_idx))
        self.lambda_q1 = nn.Parameter(torch.randn(self.d_model))

        # Optional second sublayer (MLP) mirrors vanilla Block
        if mlp_cls is nn.Identity:
            self.mlp = None
        else:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)

        if self.fused_add_norm:
            assert layer_norm_fn is not None, "fused_add_norm=True requires Triton layer_norm_fn"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)) if RMSNorm is not None else isinstance(self.norm, nn.LayerNorm)

    # -------- cache helper (lets each mixer see its own cache view) --------
    class _SwapCache:
        def __init__(self, ip, idx: int, view):
            self.ip, self.idx, self.view, self.orig = ip, idx, view, None
        def __enter__(self):
            if self.ip is not None:
                self.orig = self.ip.key_value_memory_dict.get(self.idx, None)
                self.ip.key_value_memory_dict[self.idx] = self.view
        def __exit__(self, exc_type, exc, tb):
            if self.ip is not None:
                if self.orig is None:
                    self.ip.key_value_memory_dict.pop(self.idx, None)
                else:
                    self.ip.key_value_memory_dict[self.idx] = self.orig

    def _run_mixers(self, x: Tensor, inference_params=None, **mixer_kwargs) -> Tuple[Tensor, Tensor]:
        if inference_params is None:
            y1 = self.mixer1(x, inference_params=None, **mixer_kwargs)
            y2 = self.mixer2(x, inference_params=None, **mixer_kwargs)
            return y1, y2

        slot = inference_params.key_value_memory_dict.get(self.layer_idx, None)
        if isinstance(slot, tuple) and len(slot) == 2:
            c1, c2 = slot
            with DiffBlock._SwapCache(inference_params, self.layer_idx, c1):
                y1 = self.mixer1(x, inference_params=inference_params, **mixer_kwargs)
            with DiffBlock._SwapCache(inference_params, self.layer_idx, c2):
                y2 = self.mixer2(x, inference_params=inference_params, **mixer_kwargs)
        else:
            y1 = self.mixer1(x, inference_params=inference_params, **mixer_kwargs)
            y2 = self.mixer2(x, inference_params=inference_params, **mixer_kwargs)
        return y1, y2

    @staticmethod
    def _to_norm_dtype(norm: nn.Module, x: Tensor) -> Tensor:
        w = getattr(norm, "weight", None)
        return x.to(w.dtype) if isinstance(w, torch.Tensor) else x

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        **mixer_kwargs,
    ) -> Tuple[Tensor, Tensor]:
        # ---- Add -> Norm (prenorm) ----
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        
        # ---- Scalar λ per layer ----
        lambda_q1 = torch.sum(self.lambda_q1, dim=-1).float()
        lambda_full = torch.sigmoid(lambda_q1) + self.lambda_init

        # ---- Parallel mixers ----
        y1, y2 = self._run_mixers(hidden_states, inference_params, **mixer_kwargs)

        # ---- Differential combine -> out proj -> post-sub norm ----
        attn = y1 - lambda_full * y2
        attn = self.subln(attn)

        # First sublayer output
        hidden_states = attn * (1.0 - self.lambda_init)


        # ---- Optional MLP sublayer (mirrors vanilla Block) ----
        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)
        else:
            residual = hidden_states + residual

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        cache1 = getattr(self.mixer1, "allocate_inference_cache", None)
        cache2 = getattr(self.mixer2, "allocate_inference_cache", None)
        c1 = cache1(batch_size, max_seqlen, dtype=dtype, **kwargs) if callable(cache1) else None
        c2 = cache2(batch_size, max_seqlen, dtype=dtype, **kwargs) if callable(cache2) else None
        return (c1, c2)

    @classmethod
    def from_pretrained_block(
        cls,
        block: nn.Module,
        mixer_cls: Optional[Callable[[int], nn.Module]] = None,
        mlp_cls: Optional[Callable[[int], nn.Module]] = None,
        norm_cls: Optional[Callable[[int], nn.Module]] = None,
        fused_add_norm: Optional[bool] = None,
        residual_in_fp32: Optional[bool] = None,
        lambda_init: float = 0.1,
        use_postscale: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "DiffBlock":
        """Build a DiffBlock from a vanilla Block and copy weights into both mixers."""
        src_mixer = getattr(block, "mixer", None)
        src_mlp   = getattr(block, "mlp", None)
        src_norm  = getattr(block, "norm", None)

        mixer_cls = mixer_cls or (src_mixer.__class__)
        mlp_cls   = mlp_cls   or (src_mlp.__class__ if src_mlp is not None else nn.Identity)
        norm_cls  = norm_cls  or (src_norm.__class__ if src_norm is not None else nn.LayerNorm)

        fused_add_norm    = fused_add_norm    if fused_add_norm    is not None else getattr(block, "fused_add_norm", False)
        residual_in_fp32  = residual_in_fp32  if residual_in_fp32  is not None else getattr(block, "residual_in_fp32", False)

        newb = cls(
            dim=block.d_model,
            mixer_cls1=mixer_cls,
            mixer_cls2=mixer_cls,
            mlp_cls=mlp_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            layer_idx=getattr(block, "layer_idx", 0),
            lambda_init=lambda_init,
            use_postscale=use_postscale,
            device=device,
            dtype=dtype,
        )

        # copy prenorm
        if src_norm is not None:
            newb.norm.load_state_dict(src_norm.state_dict(), strict=False)
            # seed post norms with same stats
            newb.post_mamba_norm1.load_state_dict(newb.norm.state_dict(), strict=False)
            newb.post_mamba_norm2.load_state_dict(newb.norm.state_dict(), strict=False)
            newb.post_sub_norm.load_state_dict(newb.norm.state_dict(), strict=False)

        # copy mixer weights into both mixers
        if src_mixer is not None:
            st = src_mixer.state_dict()
            newb.mixer1.load_state_dict(st, strict=False)
            newb.mixer2.load_state_dict(st, strict=False)

        # copy mlp & norm2 if present
        if src_mlp is not None and newb.mlp is not None:
            newb.mlp.load_state_dict(src_mlp.state_dict(), strict=False)
            if hasattr(block, "norm2") and hasattr(newb, "norm2"):
                newb.norm2.load_state_dict(block.norm2.state_dict(), strict=False)

        return newb
