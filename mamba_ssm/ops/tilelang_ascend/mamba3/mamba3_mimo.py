"""Mamba-3 MIMO autograd entry point for the A2 kernels.

This module is a **drop-in mirror of** ``mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py``:
the public signature, the returned tuple, the backward gradient order and the
triton preprocessing/postprocessing are all identical, so model code can switch
between the CUDA tilelang kernels and the A2 kernels by changing one import:

    # CUDA tilelang
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo
    # Ascend A2
    from mamba_ssm.ops.tilelang_ascend.mamba3.mamba3_mimo import mamba3_mimo

The triton helpers (``angle_dt_fwd`` / ``angle_dt_bwd`` /
``compute_dacs_segsum_triton*``) are **not** reimplemented here: they are imported
from ``mamba_ssm.ops.triton.mamba3`` exactly as the upstream wrapper does, and
run on the NPU through ``triton-ascend``.  That keeps the discretisation maths
bit-identical between the two back ends and avoids a second source of truth.

What differs from upstream (and only this):

* the forward/backward kernel calls target
  ``mamba_ssm.ops.tilelang_ascend.mamba3.*`` and therefore pass ``chunk_size`` /
  ``rotary_dim_divisor`` positionally;
* the A2 backward returns a dict, which is unpacked into upstream's gradient
  order here;
* ``fuse_pregate_headwise_rms_norm`` is ported: the fused epilogue runs inside
  the A2 forward/backward composition, and ``dOut_Norm_Weight`` is summed over
  the batch and rank axes and reshaped like ``Out_Norm_Weight``, exactly as the
  CUDA wrapper does.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

# --- triton preprocessing, shared verbatim with the CUDA tilelang operator ----
from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import (
    compute_dacs_segsum_triton,
    compute_dacs_segsum_triton_varlen,
)
from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_bwd, angle_dt_fwd

# --- A2 tilelang kernels ----------------------------------------------------
from mamba_ssm.ops.tilelang_ascend.mamba3.mamba3_mimo_bwd import mamba_mimo_backward
from mamba_ssm.ops.tilelang_ascend.mamba3.mamba3_mimo_fwd import (
    mamba_mimo_forward,
    mamba_mimo_forward_varlen,
)


class _Mamba3Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        ADT: Tensor,
        DT: Tensor,
        Trap: Tensor,
        Q_bias: Tensor,
        K_bias: Tensor,
        MIMO_V: Tensor,
        MIMO_Z: Tensor,
        MIMO_Out: Union[Tensor, None],
        Out_Norm_Weight: Union[Tensor, None],
        Angles: Tensor,
        D: Tensor,
        Z: Tensor,
        chunk_size: int,
        rotary_dim_divisor: int,
        dtype: torch.dtype,
        return_state: bool,
        cu_seqlens: Optional[Tensor],
        fuse_pregate_headwise_rms_norm: bool,
        outproj_norm_eps: float,
    ) -> Union[Tensor, Tuple[Tensor, Tuple]]:
        if fuse_pregate_headwise_rms_norm:
            if MIMO_Out is None:
                raise ValueError(
                    "fuse_pregate_headwise_rms_norm=True requires MIMO_Out.")
            if Z is None or MIMO_Z is None:
                raise ValueError(
                    "fuse_pregate_headwise_rms_norm=True requires Z and MIMO_Z.")

        ctx.chunk_size = chunk_size
        ctx.rotary_dim_divisor = rotary_dim_divisor
        ctx.dtype = dtype
        ctx.fuse_pregate_headwise_rms_norm = fuse_pregate_headwise_rms_norm
        ctx.outproj_norm_eps = outproj_norm_eps

        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out, Out_Norm_Weight, Angles, D, Z = tuple(
            t.contiguous() if t is not None else None
            for t in (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z,
                      MIMO_Out, Out_Norm_Weight, Angles, D, Z)
        )
        if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)

        # cumulative rotary angles (varlen aware) -- same helper as upstream, so
        # the kernel sees exactly the same Angles_Cumsum on both back ends
        Angles_Cumsum, angle_output_state = angle_dt_fwd(
            Angles, DT,
            chunk_size=chunk_size,
            return_output_state=True,
            cu_seqlens=cu_seqlens,
        )

        if cu_seqlens is not None:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(
                ADT, chunk_size, cu_seqlens=cu_seqlens)
            Out, Final_SSM_State, Final_K = mamba_mimo_forward_varlen(
                Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out,
                Z, D, MIMO_Z, Angles_Cumsum,
                DA_CS, DA_CS_REV, DT, Trap, Segsum,
                chunk_size, rotary_dim_divisor,
                dtype=dtype, cu_seqlens=cu_seqlens, return_state=return_state,
                fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
                outproj_norm_weight=Out_Norm_Weight,
                outproj_norm_eps=outproj_norm_eps,
            )
        else:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton(ADT, chunk_size)
            Out, Final_SSM_State, Final_K = mamba_mimo_forward(
                Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out,
                Z, D, MIMO_Z, Angles_Cumsum,
                DA_CS, DA_CS_REV, DT, Trap, Segsum,
                chunk_size, rotary_dim_divisor,
                dtype=dtype, return_state=return_state,
                fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
                outproj_norm_weight=Out_Norm_Weight,
                outproj_norm_eps=outproj_norm_eps,
            )

        ctx.save_for_backward(
            Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, Angles_Cumsum,
            D, Z,
            MIMO_V, MIMO_Out, MIMO_Z, Out_Norm_Weight,
            cu_seqlens,
        )

        if not return_state:
            return Out

        # upstream layout: state comes back as (ns, H, N, P) and is exposed as
        # (ns, H, P, N); the A2 kernel uses the same (ns, H, N, P) layout
        Final_Angle = torch.remainder(angle_output_state, 2 * torch.pi).contiguous().detach()
        Final_SSM_State = Final_SSM_State.permute(0, 1, 3, 2).contiguous().detach()
        Final_K = Final_K.contiguous().detach()
        Final_V = V[:, -1, :, :].contiguous().detach()
        ctx.mark_non_differentiable(Final_Angle, Final_SSM_State, Final_K, Final_V)
        return Out, Final_Angle, Final_SSM_State, Final_K, Final_V

    @staticmethod
    def backward(ctx, dout, *args) -> tuple:
        if len(ctx.saved_tensors) == 0:
            raise RuntimeError(
                "Backward called but forward ran without gradient tracking.")
        dout = dout.contiguous()

        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, Angles_Cumsum,
         D, Z, MIMO_V, MIMO_Out, MIMO_Z, Out_Norm_Weight, cu_seqlens) = ctx.saved_tensors

        if cu_seqlens is not None:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(
                ADT, ctx.chunk_size, cu_seqlens=cu_seqlens)
        else:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton(ADT, ctx.chunk_size)

        g = mamba_mimo_backward(
            dout,
            Q, K, V, Q_bias, K_bias,
            MIMO_V, MIMO_Out, Z, MIMO_Z,
            Angles_Cumsum,
            DA_CS, DA_CS_REV, DT, Trap, D, Segsum,
            ctx.chunk_size, ctx.rotary_dim_divisor, ctx.dtype,
            cu_seqlens=cu_seqlens,
            fuse_pregate_headwise_rms_norm=ctx.fuse_pregate_headwise_rms_norm,
            outproj_norm_weight=Out_Norm_Weight,
            outproj_norm_eps=ctx.outproj_norm_eps,
        )

        dQ, dK, dV = g["dq"], g["dk"], g["dv"]
        dQ_bias, dK_bias = g["dq_bias"], g["dk_bias"]
        # The A2 backward always fills its accumulators, while the forward may
        # have been called with these inputs as None (no gate / no skip term /
        # no rank reduction).  Autograd rejects a gradient for a None input, so
        # report None exactly where upstream's CUDA backward does.
        dMIMO_V = g["dmimo_v"]
        dMIMO_Z = g["dmimo_z"] if MIMO_Z is not None else None
        dMIMO_Out = g["dmimo_o"] if MIMO_Out is not None else None
        dD = g["dD"] if D is not None else None
        dZ = g["dz"] if Z is not None else None
        # Same reduction as the CUDA wrapper (batch and rank axes summed, then
        # reshaped like the weight that was passed in).
        dOut_Norm_Weight = None
        if ctx.fuse_pregate_headwise_rms_norm:
            dOut_Norm_Weight = g["dout_norm_weight"].sum(dim=(0, 2))
            if Out_Norm_Weight is not None:
                dOut_Norm_Weight = dOut_Norm_Weight.reshape_as(Out_Norm_Weight)

        # A2 returns the fully composed discretisation gradients
        # (ddA / ddt / dtrap == upstream's dADT / dDT / dTrap), so they are used
        # as-is.  "dangles" is the gradient w.r.t. the *cumulative* angles, i.e.
        # upstream's dAngles_Cumsum, so the angle chain is applied here exactly
        # like the CUDA wrapper does.
        dADT, dDT, dTrap = g["ddA"], g["ddt"], g["dtrap"]
        dAngles_Cumsum = g["dangles"]
        dAngles, dDT_angle, _ = angle_dt_bwd(
            grad_out=dAngles_Cumsum,
            angle=Angles,
            dt=DT,
            has_init_state=False,
            chunk_size=ctx.chunk_size,
            cu_seqlens=cu_seqlens,
        )
        dDT = dDT + dDT_angle

        return (
            dQ, dK, dV,
            dADT, dDT, dTrap,
            dQ_bias, dK_bias,
            dMIMO_V, dMIMO_Z, dMIMO_Out,
            dOut_Norm_Weight,
            dAngles, dD, dZ,
            None, None, None, None, None, None, None,
        )


def mamba3_mimo(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Trap: Tensor,
    Q_bias: Tensor,
    K_bias: Tensor,
    MIMO_V: Tensor,
    MIMO_Z: Tensor,
    MIMO_Out: Tensor,
    Angles: Tensor,
    D: Tensor,
    Z: Tensor,
    chunk_size: int,
    rotary_dim_divisor: int,
    dtype: torch.dtype,
    return_state: bool = False,
    cu_seqlens: Optional[Tensor] = None,
    fuse_pregate_headwise_rms_norm: bool = False,
    outproj_norm_weight: Optional[Tensor] = None,
    outproj_norm_eps: float = 1e-5,
) -> Union[Tensor, Tuple[Tensor, Tuple]]:
    """Mamba-3 MIMO with the A2 kernels; signature identical to the CUDA version.

    Args:
        Q: Query tensor (batch, seqlen, mimo_rank, nheads_qk, headdim_qk)
        K: Key tensor (batch, seqlen, mimo_rank, nheads_qk, headdim_qk)
        V: Value tensor (batch, seqlen, nheads, headdim_v)
        ADT: Decay factor A * dt (batch, nheads, seqlen)
        DT: Time delta tensor dt (batch, nheads, seqlen)
        Trap: Trapezoidal mixing factor, pre-sigmoid (batch, nheads, seqlen)
        Q_bias: Query bias (nheads, mimo_rank, headdim_qk)
        K_bias: Key bias (nheads, mimo_rank, headdim_qk)
        MIMO_V: Mimo up projection for V (nheads, mimo_rank, headdim_v)
        MIMO_Z: Mimo up projection for Z (nheads, mimo_rank, headdim_v)
        MIMO_Out: Mimo down projection for output (nheads, mimo_rank, headdim_v)
        Angles: Rotary position embeddings (batch, seqlen, nheads, headangles)
        D: Skip connection weight (nheads,)
        Z: Gating tensor (batch, seqlen, nheads, headdim_v)
        chunk_size: Chunk size for state computation.  Typical values are
            [16, 32, 64, 128].  Any value is supported: the kernels work on a row
            axis of ``chunk_size * mimo_rank`` and zero pad it up to
            ``a2_vec.BR``, so small chunks (e.g. 8) are correct but pay for the
            padding.
        rotary_dim_divisor: Divisor for rotary embedding dimensions
        dtype: Data type for lower-precision computation (e.g. torch.bfloat16)
        return_state: Whether to return the final state for decoding
        cu_seqlens: Cumulative sequence lengths (num_seq + 1,) for packed input
        fuse_pregate_headwise_rms_norm: fuse headwise RMSNorm(y) * silu(z) with
            the MIMO down projection, exactly like the CUDA operator.  Requires
            ``MIMO_Out``, ``Z`` and ``MIMO_Z``.
        outproj_norm_weight: RMSNorm weight, shaped ``(nheads, headdim_v)`` or
            flattened; only used when the fused switch is on
        outproj_norm_eps: epsilon of the fused headwise RMSNorm

    Returns:
        ``Out`` or ``(Out, Final_Angle, Final_SSM_State, Final_K, Final_V)`` --
        the same tuple, in the same order and layout, as the CUDA operator.
    """

    return _Mamba3Function.apply(
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias,
        MIMO_V, MIMO_Z, MIMO_Out, outproj_norm_weight,
        Angles, D, Z,
        chunk_size, rotary_dim_divisor, dtype,
        return_state, cu_seqlens,
        fuse_pregate_headwise_rms_norm, outproj_norm_eps,
    )
