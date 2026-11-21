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

# Two-level (block-decomposed) scan, ON by default with scan_block 32 (the
# measured optimum at chunk_size 16), used per packing only where it pays
# (two_level_pays: the longest segment, not the total work, bounds the stock
# scan). M3_SCAN_BLOCK=0 forces the stock scan everywhere.
from mamba_ssm.ops.cuda.mamba3.sm100 import common as _sm100
from mamba_ssm.ops.tilelang.mamba3.mamba3_twolevel import (
    cu_host_list,
    scan_block_from_env,
    two_level_forward,
    two_level_pays,
)
_M3_SCAN_BLOCK = scan_block_from_env(32)


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
        Out_Norm_Weight: Union[Tensor, None],
        Angles: Tensor,
        D: Tensor,
        Z: Tensor,
        Input_Angle_State: Optional[Tensor],
        Input_SSM_State: Optional[Tensor],
        Input_K_State: Optional[Tensor],
        Input_V_State: Optional[Tensor],
        chunk_size: int,
        rotary_dim_divisor: int,
        dtype: torch.dtype,
        return_state: bool,
        cu_seqlens: Optional[Tensor],
        fuse_pregate_headwise_rms_norm: bool,
        outproj_norm_eps: float,
    ) -> Tensor | Tuple[Tensor, Tuple]:
        """Forward pass: call Triton/Tilelang kernel and save tensors for backward."""
        ctx.chunk_size = chunk_size
        ctx.rotary_dim_divisor = rotary_dim_divisor
        ctx.dtype = dtype
        ctx.fuse_pregate_headwise_rms_norm = fuse_pregate_headwise_rms_norm
        ctx.outproj_norm_eps = outproj_norm_eps
        ctx.has_input_state = Input_SSM_State is not None
        all_states_present = (
            Input_Angle_State is not None
            and Input_SSM_State is not None
            and Input_K_State is not None
            and Input_V_State is not None
        )
        all_states_absent = (
            Input_Angle_State is None
            and Input_SSM_State is None
            and Input_K_State is None
            and Input_V_State is None
        )
        assert all_states_present or all_states_absent, "Input states must be provided together or all be None."
        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out, Out_Norm_Weight, Angles, D, Z,
            Input_Angle_State, Input_SSM_State, Input_K_State, Input_V_State) = tuple(
            t.contiguous() if t is not None else None
            for t in (
                Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out, Out_Norm_Weight, Angles, D, Z,
                Input_Angle_State, Input_SSM_State, Input_K_State, Input_V_State,
            )
        )
        # Kernels require cu_seqlens as int32; torch.cumsum promotes int32→int64 on CUDA
        if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)

        # Blackwell (sm_100 / sm_103), Olala configuration: tcgen05 forward (outputs only) and chunk-parallel
        # tcgen05 backward (mamba_ssm/ops/cuda/mamba3/sm100); M3_SM100=0 keeps the TileLang kernels.
        ctx.sm100 = (
            not return_state
            and _sm100.config_ok(Q, K, V, V.shape[2], MIMO_Out, Z, D, Angles, chunk_size, rotary_dim_divisor, dtype,
                                 fuse_pregate_headwise_rms_norm, cu_seqlens, all_states_present)
            and Out_Norm_Weight is None
        )
        Angle_Base = None
        if ctx.sm100:
            # the angle cumsum is folded into the forward kernels: Angles_Cumsum holds the phases relative to each
            # 256-token chunk, Angle_Base the chunk's entering phase (both for the backward)
            from mamba_ssm.ops.cuda.mamba3.sm100 import fwd_tc
            ctx.scan_block = 0
            Out, Angles_Cumsum, Angle_Base = fwd_tc.fwd(Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out, Z, D, MIMO_Z, Angles,
                                                        ADT, DT, Trap, cu_host_list(cu_seqlens))
        else:
            # Compute cumulative angles (varlen-aware)
            Angles_Cumsum, angle_output_state = angle_dt_fwd(
                Angles, DT,
                init_state=Input_Angle_State,
                chunk_size=chunk_size,
                return_output_state=True,
                cu_seqlens=cu_seqlens,
            )
            if return_state:
                Final_Angle = torch.remainder(angle_output_state, 2 * torch.pi).contiguous().detach()
        if ctx.sm100:
            pass  # done above
        elif cu_seqlens is not None:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(ADT, chunk_size, cu_seqlens=cu_seqlens)
            _fwd_kwargs = dict(
                q=Q, k=K, v=V, q_bias=Q_bias, k_bias=K_bias,
                mimo_v=MIMO_V, mimo_o=MIMO_Out,
                z=Z, D=D, mimo_z=MIMO_Z, angles=Angles_Cumsum,
                dA_cs=DA_CS, dA_cs_rev=DA_CS_REV, dt=DT, trap=Trap, segsum=Segsum,
                cu_seqlens=cu_seqlens,
                initial_states=(Input_SSM_State, Input_K_State, Input_V_State) if all_states_present else None,
                return_state=return_state,
                chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor,
                dtype=dtype,
                fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
                outproj_norm_weight=Out_Norm_Weight,
                outproj_norm_eps=outproj_norm_eps,
            )
            ctx.scan_block = (
                _M3_SCAN_BLOCK
                if _M3_SCAN_BLOCK > 0 and two_level_pays(cu_seqlens, V.shape[2], V.device)
                else 0
            )
            if ctx.scan_block > 0:
                # `initial_states` flows unchanged to BOTH passes: the kernel
                # applies it only when blk_c0 == 0, so the segment's first block
                # folds it into C_local (Pass A) and seeds from it (Pass C),
                # while later blocks use the state Pass B propagated.
                Out, Final_SSM_State, Final_K = two_level_forward(
                    mamba_mimo_forward_varlen, _fwd_kwargs, ctx.scan_block)
            else:
                Out, Final_SSM_State, Final_K = mamba_mimo_forward_varlen(**_fwd_kwargs)

        else:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton(ADT, chunk_size)
            Out, Final_SSM_State, Final_K = mamba_mimo_forward(
                Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out,
                Z, D, MIMO_Z, Angles_Cumsum,
                DA_CS, DA_CS_REV, DT, Trap, Segsum,
                initial_states=(Input_SSM_State, Input_K_State, Input_V_State) if all_states_present else None,
                return_state=return_state,
                chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor,
                dtype=dtype,
                fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
                outproj_norm_weight=Out_Norm_Weight,
                outproj_norm_eps=outproj_norm_eps,
            )

        ctx.chunk_size = chunk_size
        ctx.save_for_backward(
            Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, Angles_Cumsum,
            D, Z,
            MIMO_V, MIMO_Out, MIMO_Z, Out_Norm_Weight,
            cu_seqlens, Angle_Base,
        )

        if not return_state:
            return Out
        else:
            Final_SSM_State = Final_SSM_State.permute(0, 1, 3, 2).contiguous().detach()
            Final_K = Final_K.contiguous().detach()
            if cu_seqlens is None:
                Final_V = torch.empty((V.shape[0], V.shape[2], V.shape[3]), device=V.device, dtype=V.dtype)
                Final_V.copy_(V[:, -1, :, :])
            else:
                final_v_indices = cu_seqlens[1:].to(torch.long) - 1
                Final_V = torch.empty((final_v_indices.shape[0], V.shape[2], V.shape[3]), device=V.device, dtype=V.dtype)
                Final_V.copy_(V[0, final_v_indices, :, :])
            Final_V = Final_V.detach()
            ctx.mark_non_differentiable(Final_Angle, Final_SSM_State, Final_K, Final_V)
            return Out, Final_Angle, Final_SSM_State, Final_K, Final_V
    
    @staticmethod
    def backward(ctx, dout, *args) -> tuple:
        """Backward pass: compute gradients using Tilelang backward kernels."""
        
        # Access ctx.saved_tensors exactly once: each access re-runs the
        # saved-tensor unpack hooks, and non-reentrant gradient checkpointing
        # (torch.utils.checkpoint use_reentrant=False) only permits a single
        # unpack per tensor — a second access raises CheckpointError.
        saved = ctx.saved_tensors
        if len(saved) == 0:
            raise RuntimeError(
                "Backward called but forward ran without gradient tracking. "
                "Ensure inputs require grad or run under torch.enable_grad()."
            )
        if ctx.has_input_state:
            raise NotImplementedError(
                "Mamba-3 MIMO backward with Input_States is not implemented; "
                "use input states only for inference/prefill."
            )
        dout = dout.contiguous()

        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, Angles_Cumsum,
            D, Z,
            MIMO_V, MIMO_Out, MIMO_Z, Out_Norm_Weight,
            cu_seqlens, Angle_Base,
            ) = saved

        if ctx.sm100:
            from mamba_ssm.ops.cuda.mamba3.sm100 import bwd_cuda
            (dQ, dK, dV,
                dADT, dDT, dTrap, dQ_bias, dK_bias,
                dMIMO_V, dMIMO_Z, dMIMO_Out, dAngles_Cumsum,
                dD, dZ, dOut_Norm_Weight) = bwd_cuda.bwd(
                    dout, Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out, Z, MIMO_Z, Angles_Cumsum, ADT, DT, Trap, D,
                    cu_host_list(cu_seqlens), cu_seqlens, ain=Angle_Base)
        elif cu_seqlens is not None:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(ADT, ctx.chunk_size, cu_seqlens=cu_seqlens)
            (dQ, dK, dV,
                dADT, dDT, dTrap, dQ_bias, dK_bias,
                dMIMO_V, dMIMO_Z, dMIMO_Out, dAngles_Cumsum,
                dD, dZ, dOut_Norm_Weight) = mamba_mimo_bwd_combined_varlen(
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
                    fuse_pregate_headwise_rms_norm=ctx.fuse_pregate_headwise_rms_norm,
                    outproj_norm_weight=Out_Norm_Weight,
                    outproj_norm_eps=ctx.outproj_norm_eps,
                    scan_block=ctx.scan_block,
                )
        else:
            DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton(ADT, ctx.chunk_size)
            (dQ, dK, dV,
                dADT, dDT, dTrap, dQ_bias, dK_bias,
                dMIMO_V, dMIMO_Z, dMIMO_Out, dAngles_Cumsum,
                dD, dZ, dOut_Norm_Weight) = mamba_mimo_bwd_combined(
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
                    fuse_pregate_headwise_rms_norm=ctx.fuse_pregate_headwise_rms_norm,
                    outproj_norm_weight=Out_Norm_Weight,
                    outproj_norm_eps=ctx.outproj_norm_eps,
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
        if dOut_Norm_Weight is not None and Out_Norm_Weight is not None:
            dOut_Norm_Weight = dOut_Norm_Weight.reshape_as(Out_Norm_Weight)

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
            dOut_Norm_Weight,
            dAngles,
            dD,
            dZ,
            None, None, None, None,
            None, None, None, None, None, None, None,
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
    Input_States: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
    fuse_pregate_headwise_rms_norm: bool = False,
    outproj_norm_weight: Optional[Tensor] = None,
    outproj_norm_eps: float = 1e-5,
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
        Input_States: Optional tuple of initial states (angle, ssm, k, v). Dense shapes are
         (batch, nheads, angle_dim), (batch, nheads, headdim_v, headdim_qk),
         (batch, mimo_rank, nheads, headdim_qk), and (batch, nheads, headdim_v).
         Varlen mode uses num_sequences as the leading dimension.
        cu_seqlens: Optional tensor of cumulative sequence lengths for variable-length sequences. 
         If provided, should be a tensor of shape (num_seq + 1,) where cu_seqlens[i] is 
         the cumulative sequence length up to sequence i. This is used for efficient processing of 
         variable-length sequences in the kernel.
        fuse_pregate_headwise_rms_norm: Whether to fuse headwise RMSNorm(y) * silu(z)
         with MIMO down projection in the dense or varlen forward/backward kernels.
        outproj_norm_weight: Optional RMSNorm weight, shaped (nheads, headdim_v) or flattened.
        outproj_norm_eps: Epsilon used by the fused headwise RMSNorm path.
              
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
    num_sequences = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
    if Input_States is None:
        Input_Angle_State, Input_SSM_State, Input_K_State, Input_V_State = None, None, None, None
    else:
        assert len(Input_States) == 4, "Input_States must be a tuple of (angle_state, ssm_state, k_state, v_state)"
        Input_Angle_State, Input_SSM_State, Input_K_State, Input_V_State = Input_States
        angle_dim = Angles.shape[-1]
        assert Input_Angle_State.shape == (num_sequences, nheads, angle_dim), (
            f"Input angle state shape mismatch: expected {(num_sequences, nheads, angle_dim)}, got {Input_Angle_State.shape}"
        )
        assert Input_SSM_State.shape == (num_sequences, nheads, headdim_v, headdim_qk), (
            f"Input SSM state shape mismatch: expected {(num_sequences, nheads, headdim_v, headdim_qk)}, got {Input_SSM_State.shape}"
        )
        assert Input_K_State.shape == (num_sequences, mimo_rank, nheads, headdim_qk), (
            f"Input K state shape mismatch: expected {(num_sequences, mimo_rank, nheads, headdim_qk)}, got {Input_K_State.shape}"
        )
        assert Input_V_State.shape == (num_sequences, nheads, headdim_v), (
            f"Input V state shape mismatch: expected {(num_sequences, nheads, headdim_v)}, got {Input_V_State.shape}"
        )
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
        outproj_norm_weight,
        Angles,
        D,
        Z,
        Input_Angle_State,
        Input_SSM_State,
        Input_K_State,
        Input_V_State,
        chunk_size,
        rotary_dim_divisor,
        dtype,
        return_state,
        cu_seqlens,
        fuse_pregate_headwise_rms_norm,
        outproj_norm_eps,
    )


# =============================================================================
# Group-parallel varlen prefill (inference only)
# =============================================================================

@torch.no_grad()
def mamba3_mimo_varlen_grouped(
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
    cu_seqlens: Tensor,
    Input_States: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
    group_tokens: int = 1536,
    min_split_tokens: int = 4096,
    cu_seqlens_cpu: Optional[Tensor] = None,
    fuse_pregate_headwise_rms_norm: bool = False,
    outproj_norm_weight: Optional[Tensor] = None,
    outproj_norm_eps: float = 1e-5,
    threads: int = 128,
    num_stages: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Group-parallel exact prefill: same math as ``mamba3_mimo`` (varlen,
    ``return_state=True``), restructured for GPU occupancy on long prompts.

    The chunked-scan kernel launches one thread-block per (head, sequence)
    and walks the sequence's chunks serially, so a single long prefill uses
    only ``H`` blocks.  This wrapper splits every sequence longer than
    ``min_split_tokens`` into virtual sequences of ``group_tokens`` tokens
    and runs the kernel twice over the virtual layout:

    1. groups run with zero initial state, returning each group's local
       final state and last rotated K;
    2. group input states are chained with the closed form
       ``final = local_final + exp(sum ADT) * folded_input`` (the state
       recurrence is linear, with scalar-per-head decay), reusing the same
       boundary K/V fold the kernel applies for ``Input_States``;
    3. a second kernel pass consumes the chained states and produces exact
       outputs and final states.

    Cost is 2x kernel FLOPs for ~(S / group_tokens)x more parallelism —
    a large net win whenever ``H * num_groups`` was below GPU saturation.

    Group boundaries reuse the battle-tested chunked-prefill ``Input_States``
    machinery, so numerics match the single-pass kernel to bf16 rounding.

    Returns ``(Out, Final_Angle, Final_SSM, Final_K, Final_V)`` — the exact
    contract of ``mamba3_mimo(..., return_state=True)``.
    """
    assert cu_seqlens is not None, "grouped prefill is varlen-only"
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    tensors = (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z,
               MIMO_Out, Angles, D, Z)
    (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z,
     MIMO_Out, Angles, D, Z) = tuple(
        t.contiguous() if t is not None else None for t in tensors)

    if Input_States is not None:
        In_Angle, In_SSM, In_K, In_V = (
            t.contiguous() if t is not None else None for t in Input_States)
    else:
        In_Angle = In_SSM = In_K = In_V = None

    # ---- virtual group layout ------------------------------------------------
    # cu_seqlens_cpu (when given) avoids a device sync for the host-side split.
    cu_list = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens).tolist()
    ns_true = len(cu_list) - 1
    v_bounds = [0]           # virtual cu_seqlens
    seq_of_group: list[int] = []
    last_group_rows = [0] * ns_true
    for s in range(ns_true):
        start, end = cu_list[s], cu_list[s + 1]
        length = end - start
        if length >= min_split_tokens and length > group_tokens:
            starts = list(range(start, end, group_tokens))
        else:
            starts = [start]
        for g in starts:
            v_bounds.append(min(g + group_tokens, end) if len(starts) > 1 else end)
            seq_of_group.append(s)
            last_group_rows[s] = len(seq_of_group) - 1

    if len(seq_of_group) == ns_true:  # nothing split: single-pass fast path
        Angles_Cumsum, angle_out_state = angle_dt_fwd(
            Angles, DT, init_state=In_Angle, chunk_size=chunk_size,
            return_output_state=True, cu_seqlens=cu_seqlens,
        )
        Final_Angle = torch.remainder(angle_out_state, 2 * torch.pi).contiguous()
        DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(
            ADT, chunk_size, cu_seqlens=cu_seqlens)
        o, h, k_fin = mamba_mimo_forward_varlen(
            Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out, Z, D, MIMO_Z,
            Angles_Cumsum, DA_CS, DA_CS_REV, DT, Trap, Segsum,
            chunk_size, rotary_dim_divisor, dtype, cu_seqlens=cu_seqlens,
            return_state=True,
            initial_states=(In_SSM, In_K, In_V) if In_SSM is not None else None,
            fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
            outproj_norm_weight=outproj_norm_weight,
            outproj_norm_eps=outproj_norm_eps,
            threads=threads, num_stages=num_stages,
        )
        ends = cu_seqlens[1:].to(torch.long) - 1
        return (o, Final_Angle, h.permute(0, 1, 3, 2).contiguous(),
                k_fin.contiguous(), V[0, ends].contiguous())

    v_cu = torch.tensor(v_bounds, dtype=torch.int32, device=cu_seqlens.device)
    nsv = len(seq_of_group)

    # ---- rotary phases: same two-pass trick over the virtual layout ---------
    # angle_dt walks chunks serially per (head, sequence), so it has the same
    # occupancy cliff as the scan kernel. Pass 1 gets per-group phase totals
    # (mod 2*pi is kept bounded inside the kernel), a tiny chain turns them
    # into per-group phase offsets, and pass 2 emits the continuous cumsum.
    TWO_PI = 2 * torch.pi
    _, ang_local = angle_dt_fwd(
        Angles, DT, init_state=None, chunk_size=chunk_size,
        return_output_state=True, cu_seqlens=v_cu,
    )
    # Phase offsets have no decay, so the chain is an exclusive cumsum per
    # true sequence (group phases are bounded by the kernel's mod 2*pi, and
    # there are few groups, so fp32 cumsum is exact enough).
    first_rows = [0] + [last_group_rows[s - 1] + 1 for s in range(1, ns_true)]
    ang_in = torch.empty_like(ang_local)
    for s in range(ns_true):
        sl = slice(first_rows[s], last_group_rows[s] + 1)
        base = In_Angle[s].float() if In_Angle is not None \
            else torch.zeros_like(ang_local[0])
        excl = torch.cat([torch.zeros_like(ang_local[:1]),
                          torch.cumsum(ang_local[sl][:-1], dim=0)])
        ang_in[sl] = torch.remainder(base + excl, TWO_PI)
    Angles_Cumsum, ang_state2 = angle_dt_fwd(
        Angles, DT, init_state=ang_in, chunk_size=chunk_size,
        return_output_state=True, cu_seqlens=v_cu,
    )

    DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(
        ADT, chunk_size, cu_seqlens=v_cu)

    # ---- pass 1: local group states only (zero input, no output compute) ----
    _, h1, k1 = mamba_mimo_forward_varlen(
        Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out, Z, D, MIMO_Z,
        Angles_Cumsum, DA_CS, DA_CS_REV, DT, Trap, Segsum,
        chunk_size, rotary_dim_divisor, dtype, cu_seqlens=v_cu,
        return_state=True, initial_states=None, compute_outputs=False,
        fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
        outproj_norm_weight=outproj_norm_weight,
        outproj_norm_eps=outproj_norm_eps,
        threads=threads, num_stages=num_stages,
    )

    # ---- chain group states (linear recurrence, closed form) ----------------
    H_, P_ = V.shape[-2], V.shape[-1]
    N_ = Q.shape[-1]
    R_ = Q.shape[2]
    dev = V.device

    # Per-group total decay exp(sum ADT) [nsv, H] and boundary trap scale
    # DT[first] * sigmoid(-Trap[first]) [nsv, H] — the same quantities the
    # kernel derives per chunk / at Input_States fold time.
    adt_cs = ADT[0].float().cumsum(-1)                      # [H, S]
    starts_t = torch.tensor(v_bounds[:-1], device=dev, dtype=torch.long)
    ends_t = torch.tensor(v_bounds[1:], device=dev, dtype=torch.long)
    tot = adt_cs[:, ends_t - 1] - torch.where(
        starts_t > 0, adt_cs[:, (starts_t - 1).clamp(min=0)],
        torch.zeros_like(adt_cs[:, :1]))
    decay = tot.exp().t().contiguous()                      # [nsv, H]
    bscale = (DT[0, :, starts_t].float()
              * torch.sigmoid(-Trap[0, :, starts_t].float())).t()  # [nsv, H]

    mimo_v_f = MIMO_V.float()                               # [H, R, P]
    ssm_in = torch.zeros(nsv, H_, N_, P_, device=dev, dtype=torch.float32)
    k_in = torch.zeros(nsv, R_, H_, N_, device=dev, dtype=k1.dtype)
    v_in = torch.zeros(nsv, H_, P_, device=dev, dtype=V.dtype)

    # k/v inputs are state-independent, so they are known for every group up
    # front: pass-1 finals for continuation groups, Input_States for firsts.
    first_set = set(first_rows)
    cont = [g for g in range(nsv) if g not in first_set]
    if cont:
        cont_t = torch.tensor(cont, device=dev, dtype=torch.long)
        k_in[cont_t] = k1[cont_t - 1]
        v_in[cont_t] = V[0, torch.tensor([v_bounds[g] - 1 for g in cont],
                                         device=dev, dtype=torch.long)]
    if In_SSM is not None:
        fr = torch.tensor(first_rows, device=dev, dtype=torch.long)
        ssm_in[fr] = In_SSM.float().transpose(-1, -2)  # [NS,H,P,N]->[NS,H,N,P]
        k_in[fr] = In_K
        v_in[fr] = In_V

    # pre[g] = local_final + decay * folded_kv; only the scalar-decay affine
    # accumulation stays sequential (one addcmul per group).
    fold = torch.einsum("grhn,ghp,hrp,gh->ghnp",
                        k_in.float(), v_in.float(), mimo_v_f, bscale)
    pre = h1 + decay.view(nsv, H_, 1, 1) * fold
    for g in cont:
        ssm_in[g] = torch.addcmul(pre[g - 1],
                                  decay[g - 1].view(H_, 1, 1), ssm_in[g - 1])

    # ---- pass 2: exact outputs with chained input states --------------------
    o2, h2, k2 = mamba_mimo_forward_varlen(
        Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out, Z, D, MIMO_Z,
        Angles_Cumsum, DA_CS, DA_CS_REV, DT, Trap, Segsum,
        chunk_size, rotary_dim_divisor, dtype, cu_seqlens=v_cu,
        return_state=True,
        initial_states=(ssm_in.transpose(-1, -2).contiguous(), k_in, v_in),
        fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
        outproj_norm_weight=outproj_norm_weight,
        outproj_norm_eps=outproj_norm_eps,
        threads=threads, num_stages=num_stages,
    )

    rows = torch.tensor(last_group_rows, device=dev, dtype=torch.long)
    ends = cu_seqlens[1:].to(torch.long) - 1
    Final_Angle = torch.remainder(ang_state2[rows], TWO_PI).contiguous()
    return (o2, Final_Angle,
            h2[rows].permute(0, 1, 3, 2).contiguous(),
            k2[rows].contiguous(),
            V[0, ends].contiguous())
