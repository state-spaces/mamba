# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from mamba_ssm.utils.torch import custom_bwd, custom_fwd

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.cpp_functions import causal_conv1d_fwd_function, causal_conv1d_bwd_function, causal_conv1d_update_function
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_fwd_function = None
    causal_conv1d_bwd_function = None
    causal_conv1d_update_function = None

try:
    from mamba_ssm.ops.triton.layer_norm import _layer_norm_fwd
except ImportError:
    _layer_norm_fwd = None

try:
    import selective_scan_cuda
except ImportError:
    selective_scan_cuda = None


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def rms_norm_forward(
    x,
    weight,
    bias,
    eps=1e-6,
    is_rms_norm=True,
):
    # x (b l) d
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if _layer_norm_fwd is None:
        if is_rms_norm:
            variance = x.pow(2).mean(-1, keepdim=True)
            output = x * torch.rsqrt(variance + eps) * weight
            if bias is not None:
                output = output + bias
            return output
        else:
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False)
            output = (x - mean) * torch.rsqrt(var + eps) * weight
            if bias is not None:
                output = output + bias
            return output
    y = _layer_norm_fwd(
        x, weight, bias, eps, None, residual_dtype=None, is_rms_norm=is_rms_norm
    )[0]
    # y (b l) d
    return y


import math

def hybrid_chunk_scan_pytorch(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                             return_last_state=False, chunk_size=64):
    """
    Pure PyTorch implementation of selective scan using a Hybrid Chunked Associative Scan.
    Fully parallelized and vectorized.
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
        
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
        
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
            
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    # Rearrange deltaA and deltaB_u to (B, D, N, L)
    a = rearrange(deltaA, 'b d l n -> b d n l')
    b = rearrange(deltaB_u, 'b d l n -> b d n l')
    
    L = u.shape[2]
    
    # Pad L to a multiple of chunk_size if necessary
    pad_len = (chunk_size - (L % chunk_size)) % chunk_size
    if pad_len > 0:
        a = F.pad(a, (0, pad_len), value=1.0)
        b = F.pad(b, (0, pad_len), value=0.0)
        
    L_padded = a.shape[-1]
    num_chunks = L_padded // chunk_size
    
    # Reshape to (B, D, N, C, W)
    a = rearrange(a, 'b d n (c w) -> b d n c w', w=chunk_size)
    b = rearrange(b, 'b d n (c w) -> b d n c w', w=chunk_size)
    
    # 1. Intra-chunk sequential scan (vectorized over chunks, batch, and dims)
    A_loc = []
    X_loc = []
    
    curr_A = torch.ones((batch, dim, dstate, num_chunks), device=a.device, dtype=a.dtype)
    curr_X = torch.zeros((batch, dim, dstate, num_chunks), device=b.device, dtype=b.dtype)
    
    for w in range(chunk_size):
        curr_A = a[..., w] * curr_A
        curr_X = a[..., w] * curr_X + b[..., w]
        A_loc.append(curr_A)
        X_loc.append(curr_X)
        
    # Shape of A_loc and X_loc is (chunk_size, B, D, N, C)
    # Stack them to shape (B, D, N, C, W)
    A_loc = torch.stack(A_loc, dim=-1)
    X_loc = torch.stack(X_loc, dim=-1)
    
    # Chunk summary:
    # chunk_a: decay factor for each chunk (B, D, N, C)
    # chunk_b: output term for each chunk (B, D, N, C)
    chunk_a = A_loc[..., -1]
    chunk_b = X_loc[..., -1]
    
    # 2. Inter-chunk parallel scan (Kogge-Stone style scan over C chunks)
    # s_c: boundary state at start of chunk c (B, D, N, C)
    # s_0 = 0.
    s = torch.zeros((batch, dim, dstate, num_chunks), device=a.device, dtype=a.dtype)
    
    # We want to solve s_{c+1} = chunk_a_c * s_c + chunk_b_c
    # This means the state after chunk c is chunk_x_c = prefix_scan(chunk_a, chunk_b)
    # Let's perform parallel prefix scan on chunk_a and chunk_b
    scan_a = chunk_a.clone()
    scan_b = chunk_b.clone()
    
    num_steps = int(math.ceil(math.log2(num_chunks)))
    for i in range(num_steps):
        stride = 2 ** i
        if stride >= num_chunks:
            break
        # Shift and combine
        a_left = scan_a[..., :num_chunks - stride]
        b_left = scan_b[..., :num_chunks - stride]
        
        a_right = scan_a[..., stride:]
        b_right = scan_b[..., stride:]
        
        scan_a[..., stride:] = a_right * a_left
        scan_b[..., stride:] = a_right * b_left + b_right
        
    # The state at the end of chunk c is scan_b[..., c]
    # So the state at the start of chunk c+1 is scan_b[..., c]
    # The state at the start of chunk 0 is s_0 = 0
    if num_chunks > 1:
        s[..., 1:] = scan_b[..., :-1]
        
    # 3. Final state reconstruction for all steps in all chunks
    # x_{c, w} = A_loc_{c, w} * s_c + X_loc_{c, w}
    # A_loc: (B, D, N, C, W)
    # s: (B, D, N, C) -> unsqueeze to (B, D, N, C, 1)
    # X_loc: (B, D, N, C, W)
    x = A_loc * s.unsqueeze(-1) + X_loc
    
    # Reshape back to (B, D, N, L_padded) and truncate to (B, D, N, L)
    x = rearrange(x, 'b d n c w -> b d n (c w)')
    if pad_len > 0:
        x = x[..., :-pad_len]
        
    # Get last state if needed
    last_state = x[..., -1]
    
    # Compute output y:
    # x: (B, D, N, L)
    if not is_variable_C:
        y = torch.einsum('bdnl,dn->bdl', x, C)
    else:
        if C.dim() == 3:
            y = torch.einsum('bdnl,bnl->bdl', x, C)
        else:
            y = torch.einsum('bdnl,bdnl->bdl', x, C)
            
    if y.is_complex():
        y = y.real * 2
        
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
        
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    if selective_scan_cuda is None:
        return hybrid_chunk_scan_pytorch(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight=None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_fwd_function(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
            
        if b_rms_weight is not None:
            B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if c_rms_weight is not None:
            C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if dt_rms_weight is not None:
            delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
            delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
        
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.b_rms_weight = b_rms_weight
        ctx.c_rms_weight = c_rms_weight
        ctx.dt_rms_weight = dt_rms_weight
        ctx.b_c_dt_rms_eps = b_c_dt_rms_eps
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_fwd_function(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
            if dt_rms_weight is not None:
                delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
                delta = rms_norm_forward(delta, ctx.dt_rms_weight, None, ctx.b_c_dt_rms_eps)
                delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
            if b_rms_weight is not None:
                # Recompute & RMSNorm B
                B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                B = rms_norm_forward(
                    B, ctx.b_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            if c_rms_weight is not None:
                # Recompute & RMSNorm C
                C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                C = rms_norm_forward(
                    C, ctx.c_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_bwd_function(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                # 6-None are delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps
                dB_proj_bias, dC_proj_bias, None, None, None, None, None, None)


def mamba_inner_ref_pure(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1,
    b_rms_weight=None, c_rms_weight=None, dt_rms_weight=None, b_c_dt_rms_eps=1e-6
):
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    
    if causal_conv1d_fn is not None:
        x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    else:
        d_conv = conv1d_weight.shape[-1]
        x_padded = F.pad(x, (d_conv - 1, 0))
        x = F.conv1d(x_padded, conv1d_weight, bias=conv1d_bias, groups=x.shape[1])
        x = F.silu(x)
        
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
    
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable C
        C = x_dbl[:, -d_state:]  # (bl dstate)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
            
    if b_rms_weight is not None:
        B = rearrange(B, "b dstate l -> (b l) dstate").contiguous()
        B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
    if c_rms_weight is not None:
        C = rearrange(C, "b dstate l -> (b l) dstate").contiguous()
        C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
    if dt_rms_weight is not None:
        delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
        delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
        delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
        
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)


def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight= None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6
):
    if selective_scan_cuda is None or causal_conv1d_fwd_function is None:
        return mamba_inner_ref_pure(
            xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
            out_proj_weight, out_proj_bias,
            A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, checkpoint_lvl,
            b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps
        )
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps)


def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)
