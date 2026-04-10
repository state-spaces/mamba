"""Mamba-3 Tilelang Autograd Wrapper

Interface for Mamba-3 Tilelang kernels with automatic differentiation

Copyright (c) 2026, Dao AI Lab, Goombalab
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

# Import kernels
from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton, compute_dacs_segsum_triton_varlen
from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd, angle_dt_bwd

from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd import mamba_mimo_forward
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd_varlen import mamba_mimo_forward_varlen

from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import mamba_mimo_bwd_combined
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd_varlen import mamba_mimo_bwd_combined_varlen


# =============================================================================
# Autograd Function
# =============================================================================

class _Mamba3Function(torch.autograd.Function):
    """Custom autograd function for Mamba-3 with Triton/Tilelang kernels."""
    
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
        Angles: Tensor,
        D: Tensor,
        Z: Tensor,
        chunk_size: int,
        rotary_dim_divisor: int,
        dtype: torch.dtype,
        return_state: bool,
        cu_seqlens: Optional[Tensor],
    ) -> Tensor | Tuple[Tensor, Tuple]:
        """Forward pass: call Triton/Tilelang kernel and save tensors for backward."""
        ctx.chunk_size = chunk_size
        ctx.rotary_dim_divisor = rotary_dim_divisor
        ctx.dtype = dtype
        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out, Angles, D, Z) = tuple(
            t.contiguous() if t is not None else None
            for t in (
                Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out, Angles, D, Z,
            )
        )
        # Kernels require cu_seqlens as int32; torch.cumsum promotes int32→int64 on CUDA
        if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)

        # Compute cumulative angles (varlen-aware)
        Angles_Cumsum, angle_output_state = angle_dt_fwd(
            Angles, DT,
            chunk_size=chunk_size,
            return_output_state=True,
            cu_seqlens=cu_seqlens,
        )
        if return_state:
            Final_Angle = torch.remainder(angle_output_state, 2 * torch.pi).contiguous().detach()

        if cu_seqlens is not None:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(ADT, chunk_size, cu_seqlens=cu_seqlens)
            Out, Final_SSM_State, Final_K = mamba_mimo_forward_varlen(
                Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out,
                Z, D, MIMO_Z, Angles_Cumsum,
                DA_CS, DA_CS_REV, DT, Trap, Segsum,
                cu_seqlens=cu_seqlens,
                return_state=return_state,
                chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor,
                dtype=dtype,
            )

        else:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton(ADT, chunk_size)
            Out, Final_SSM_State, Final_K = mamba_mimo_forward(
                Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out,
                Z, D, MIMO_Z, Angles_Cumsum,
                DA_CS, DA_CS_REV, DT, Trap, Segsum,
                return_state=return_state,
                chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor,
                dtype=dtype,
            )

        ctx.chunk_size = chunk_size
        ctx.save_for_backward(
            Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, Angles_Cumsum,
            D, Z,
            MIMO_V, MIMO_Out, MIMO_Z,
            cu_seqlens,
        )

        if not return_state:
            return Out
        else:
            Final_SSM_State = Final_SSM_State.permute(0, 1, 3, 2).contiguous().detach()
            Final_K = Final_K.contiguous().detach()
            Final_V = V[:, -1, :, :].contiguous().detach()
            ctx.mark_non_differentiable(Final_Angle, Final_SSM_State, Final_K, Final_V)
            return Out, Final_Angle, Final_SSM_State, Final_K, Final_V
    
    @staticmethod
    def backward(ctx, dout, *args) -> tuple:
        """Backward pass: compute gradients using Tilelang backward kernels."""
        
        if len(ctx.saved_tensors) == 0:
            raise RuntimeError(
                "Backward called but forward ran without gradient tracking. "
                "Ensure inputs require grad or run under torch.enable_grad()."
            )
        dout = dout.contiguous()

        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, Angles_Cumsum,
            D, Z,
            MIMO_V, MIMO_Out, MIMO_Z,
            cu_seqlens
            ) = ctx.saved_tensors

        if cu_seqlens is not None:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(ADT, ctx.chunk_size, cu_seqlens=cu_seqlens)
            (dQ, dK, dV,
                dADT, dDT, dTrap, dQ_bias, dK_bias,
                dMIMO_V, dMIMO_Z, dMIMO_Out, dAngles_Cumsum,
                dD, dZ) = mamba_mimo_bwd_combined_varlen(
                    dout,
                    Q,
                    K,
                    V,
                    Q_bias,
                    K_bias,
                    MIMO_V,
                    MIMO_Out,
                    Z,
                    MIMO_Z,
                    Angles_Cumsum,
                    DA_CS,
                    DA_CS_REV,
                    DT,
                    Trap,
                    D,
                    Segsum,
                    ctx.chunk_size,
                    ctx.rotary_dim_divisor,
                    ctx.dtype,
                    cu_seqlens=cu_seqlens,
                )
        else:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton(ADT, ctx.chunk_size)
            (dQ, dK, dV,
                dADT, dDT, dTrap, dQ_bias, dK_bias,
                dMIMO_V, dMIMO_Z, dMIMO_Out, dAngles_Cumsum,
                dD, dZ) = mamba_mimo_bwd_combined(
                    dout,
                    Q,
                    K,
                    V,
                    Q_bias,
                    K_bias,
                    MIMO_V,
                    MIMO_Out,
                    Z,
                    MIMO_Z,
                    Angles_Cumsum,
                    DA_CS,
                    DA_CS_REV,
                    DT,
                    Trap,
                    D,
                    Segsum,
                    ctx.chunk_size,
                    ctx.rotary_dim_divisor,
                    ctx.dtype,
                )

        # Backprop through angle_dt cumsum (varlen-aware)
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
            dQ,
            dK,
            dV,
            dADT,
            dDT,
            dTrap,
            dQ_bias,
            dK_bias,
            dMIMO_V,
            dMIMO_Z,
            dMIMO_Out,
            dAngles,
            dD,
            dZ,
            None, None, None, None, None,
        )


# =============================================================================
# Public API
# =============================================================================

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
) -> Tensor | Tuple[Tensor, Tuple]:
    """Mamba-3 attention with Tilelang kernels and automatic differentiation.
    
    Args:
        Q: Query tensor (batch, seqlen, mimo_rank, nheads_qk, headdim_qk)
        K: Key tensor (batch, seqlen, mimo_rank, nheads_qk, headdim_qk)
        V: Value tensor (batch, seqlen, nheads, headdim_v)
        ADT: Decay factor A * dt (batch, nheads, seqlen)
        DT: Time delta tensor dt (batch, nheads, seqlen)
        Trap: Trapezoidal mixing factor, pre-sigmoid (batch, nheads, seqlen)
        Q_bias: Query bias (nheads, mimo_rank, headdim_qk)
        K_bias: Key bias (nheads, mimo_rank, headdim_qk)
        MIMO_V: Mimo up projection for V (nheads, mimo_rank, headdim_v),
        MIMO_Z: Mimo up projection for Z (nheads, mimo_rank, headdim_v),
        MIMO_Out: Mimo down projection for output (nheads, mimo_rank, headdim_v). If None, does not reduce output with MIMO_Out,
        Angles: Rotary position embeddings (batch, seqlen, nheads, headangles)
        D: Optional skip connection weight (nheads,)
        Z: Optional gating tensor (batch, seqlen, nheads, headdim_v)
        chunk_size: Chunk size for state computation (default: 64//R)  
        rotary_dim_divisor: Divisor for rotary embedding dimensions (default: 4, meaning angles have 1/4 of headdim_qk)
        dtype: Data type for lower-precision computation (e.g., torch.bfloat16)
        return_state: Whether to return final state for autoregressive decoding (default: False)
        cu_seqlens: Optional tensor of cumulative sequence lengths for variable-length sequences. 
         If provided, should be a tensor of shape (num_seq + 1,) where cu_seqlens[i] is 
         the cumulative sequence length up to sequence i. This is used for efficient processing of 
         variable-length sequences in the kernel.
              
    Returns:
        output: (batch, seqlen, nheads, headdim_v) if MIMO_Out is not None
                (batch, seqlen, mimo_rank, nheads, headdim_v) if MIMO_Out is None
        final_state: Tuple of tensors representing the running Angle sum, final SSM state, final K, and final V for autoregressive decoding. Only returned if return_state=True.

    NOTE: The kernel is most optimized for seqlen: 2048, nheads_qk: 1, nheads: 32
     headdim_qk: 128, headdim_v: 64, mimo_rank: 4, and chunk_size: 16. On H100.
    NOTE: The code is still prone to smem over-allocation and Tilelang compilation error
     once headdim_qk, headdim_v, mimo_rank, chunk_size, or hardware type deviate from the combinations tested.
    NOTE: Chunk size of 64/R is recommended, where R is the MIMO rank. However, it may be necessary to reduce chunk size
     in case of smem over-allocation, which can occur with larger headdim_qk, headdim_v, or mimo_rank values.
    NOTE: Currently final_state is intended to be a non-differentiable side output. In particular,
     loss = f(output) is fine, but loss = f(output, final_state) will not work properly since the backward does not compute gradients for final_state components.
    NOTE: Currently we have a separate set of kernels for variable-length sequences with cu_seqlens input, 
     which can be more efficient than padding to max length. However, the variable-length kernels 
     incur noticeable overhead, so we have separate scripts for the non-varlen case.

    """
    
    batch, seqlen, mimo_rank, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    
    assert chunk_size >= 8, f"chunk_size must be at least 8"
    assert nheads % nheads_qk == 0, f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})"
    assert headdim_qk % 2 == 0, f"headdim_qk ({headdim_qk}) must be even for rotary embeddings"
    assert rotary_dim_divisor in [2, 4], f"currently only supports rotary embedding on entire or half of headdim_qk"
    # NOTE: the following (headdim_qk, headdim_v) values currently can result in compilation errors: (16, 32), (256, 128) 
    if headdim_qk not in [16, 32, 64, 128, 256]:
        print(f"WARNING: The value headdim_qk={headdim_qk} has not been tested. " +\
              "Proceed with caution and consider one of the tested headdim_qk: 16, 32, 64, 128, 256.")
    if headdim_v not in [32, 64, 128]:
        print(f"WARNING: The value headdim_v={headdim_v} has not been tested. " +\
              "Proceed with caution and consider one of the tested headdim_v: 32, 64, 128.")
    if mimo_rank not in [1, 2, 4, 8]:
        print(f"WARNING: The value mimo_rank={mimo_rank} has not been tested. " +\
              "Proceed with caution and consider one of the tested mimo_rank: 1, 2, 4, 8.")

    if chunk_size*mimo_rank > 64:
        print(f"WARNING: chunk_size * mimo_rank = {chunk_size*mimo_rank} exceeds 64, which may result in smem over-allocation. Consider decreasing chunk_size.")

    return _Mamba3Function.apply(
        Q,
        K,
        V,
        ADT,
        DT,
        Trap,
        Q_bias,
        K_bias,
        MIMO_V,
        MIMO_Z,
        MIMO_Out,
        Angles,
        D,
        Z,
        chunk_size,
        rotary_dim_divisor,
        dtype,
        return_state,
        cu_seqlens,
    )