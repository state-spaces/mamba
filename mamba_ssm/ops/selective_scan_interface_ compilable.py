import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple

import selective_scan_cuda


@torch.library.custom_op(
    "custom_ops::selective_scan_fwd",
    device_types=["cuda"],
    mutates_args=(),
)
def custom_selective_scan_fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    delta_bias: Optional[torch.Tensor],
    delta_softplus: bool,
    return_last_state: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool, bool]:
    pass

@torch.library.register_fake("custom_ops::selective_scan_fwd")
def custom_selective_scan_fwd_fake(
    u,
    delta,
    A,
    B,
    C,
    D,
    z,
    delta_bias,
    delta_softplus,
    return_last_state,
):
    final_out = torch.empty_like(u)
    dstate = A.size(1) * (2 if A.is_complex() else 1)
    last_state_fake = u.new_empty((u.size(0), u.size(1), dstate)) if return_last_state else u.new_empty(0)
    out_fake = torch.empty_like(u)
    x_fake = u.new_empty((u.size(0), u.size(1), u.size(2), 2 * dstate))
    return final_out, last_state_fake, out_fake, x_fake, False, False, z is not None

@torch.library.register_kernel("custom_ops::selective_scan_fwd", "cuda")
def custom_selective_scan_fwd_cuda(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    delta_bias: Optional[torch.Tensor],
    delta_softplus: bool,
    return_last_state: bool,
):
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

    squeeze_B = False
    if B.dim() == 3:
        B = rearrange(B, "b dstate l -> b 1 dstate l").contiguous()
        squeeze_B = True

    squeeze_C = False
    if C.dim() == 3:
        C = rearrange(C, "b dstate l -> b 1 dstate l").contiguous()
        squeeze_C = True

    out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
    has_z = z is not None
    final_out = rest[0].clone() if has_z else out.clone()
    last_state = x[:, :, -1, 1::2].clone() if return_last_state else u.new_empty(0)
    return final_out, last_state, out, x, squeeze_B, squeeze_C, has_z

@torch.library.custom_op(
    "custom_ops::selective_scan_bwd",
    device_types=["cuda"],
    mutates_args=(),
)
def custom_selective_scan_bwd(
    dout: torch.Tensor,
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    delta_bias: Optional[torch.Tensor],
    delta_softplus: bool,
    out: torch.Tensor,
    x: torch.Tensor,
    squeeze_B: bool,
    squeeze_C: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass

@torch.library.register_fake("custom_ops::selective_scan_bwd")
def custom_selective_scan_bwd_fake(
    dout,
    u,
    delta,
    A,
    B,
    C,
    D,
    z,
    delta_bias,
    delta_softplus,
    out,
    x,
    squeeze_B,
    squeeze_C,
):
    du = torch.empty_like(u)
    ddelta = torch.empty_like(delta)
    dA = torch.empty_like(A)
    dB = torch.empty_like(B)
    dC = torch.empty_like(C)
    dD = torch.empty_like(D) if (D is not None and D.numel() > 0) else u.new_empty(0)
    dz = torch.empty_like(z) if (z is not None and z.numel() > 0) else u.new_empty(0)
    ddelta_bias = torch.empty_like(delta_bias) if (delta_bias is not None and delta_bias.numel() > 0) else u.new_empty(0)
    return du, ddelta, dA, dB, dC, dD, dz, ddelta_bias

@torch.library.register_kernel("custom_ops::selective_scan_bwd", "cuda")
def custom_selective_scan_bwd_cuda(
    dout: torch.Tensor,
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    delta_bias: Optional[torch.Tensor],
    delta_softplus: bool,
    out: torch.Tensor,
    x: torch.Tensor,
    squeeze_B: bool,
    squeeze_C: bool,
):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    results = selective_scan_cuda.bwd(
        u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, delta_softplus, False
    )
    has_z = z is not None
    if has_z:
        du, ddelta, dA, dB, dC, dD, ddelta_bias, dz = results
    else:
        du, ddelta, dA, dB, dC, dD, ddelta_bias = results
        dz = u.new_empty(0)

    if squeeze_B and dB.numel() > 0:
        dB = dB.squeeze(1)
    if squeeze_C and dC.numel() > 0:
        dC = dC.squeeze(1)

    return du, ddelta, dA, dB, dC, dD, dz, ddelta_bias

def custom_bridge(ctx, *grads):
    dout = grads[0] if grads else ctx.saved_tensors[0].new_empty(0)
    saved = ctx.saved_tensors
    if not ctx.has_z:
        u, delta, A, B, C, D, delta_bias, x, out = saved
        z = None
    else:
        u, delta, A, B, C, D, z, delta_bias, x, out = saved

    du, ddelta, dA, dB, dC, dD, dz, ddelta_bias = torch.ops.custom_ops.selective_scan_bwd(
        dout,
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        ctx.delta_softplus,
        out,
        x,
        ctx.squeeze_B,
        ctx.squeeze_C
    )

    return (
        du,
        ddelta,
        dA,
        dB,
        dC,
        dD if D is not None else None,
        dz if z is not None else None,
        ddelta_bias if delta_bias is not None else None,
        None,
        None,
    )

def custom_setup_context(ctx, inputs, output):
    (u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state) = inputs
    (final_out, last_state, out, x, squeeze_B, squeeze_C, has_z) = output

    ctx.delta_softplus = delta_softplus
    ctx.squeeze_B = squeeze_B
    ctx.squeeze_C = squeeze_C
    ctx.has_z = has_z

    B = B.contiguous()
    C = C.contiguous()
    if squeeze_B and B.dim() == 3:
        B = rearrange(B, "b dstate l -> b 1 dstate l").contiguous()
    if squeeze_C and C.dim() == 3:
        C = rearrange(C, "b dstate l -> b 1 dstate l").contiguous()

    if not has_z:
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x, out)
    else:
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)

torch.library.register_autograd(
    "custom_ops::selective_scan_fwd", custom_bridge, setup_context=custom_setup_context
)

def selective_scan_fn_custom_op(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    delta_bias: Optional[torch.Tensor],
    delta_softplus: bool,
    return_last_state: bool,
) -> torch.Tensor:
    # Pass all arguments positionally, exactly in schema order:
    final_out, last_state, _, _, _, _, _ = torch.ops.custom_ops.selective_scan_fwd(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        delta_softplus,
        return_last_state
    )

    if return_last_state:
        return final_out, last_state
    else:
        return final_out
