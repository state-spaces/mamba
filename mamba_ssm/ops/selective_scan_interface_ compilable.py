import torch
from einops import rearrange
from typing import Optional, Tuple

import selective_scan_cuda

@torch.library.custom_op(
    "custom_ops::selective_scan_fwd",
    device_types=["cuda"],
    mutates_args=(),
    schema="(Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C, Tensor? D, Tensor? z, Tensor? delta_bias, bool delta_softplus, bool return_last_state) -> (Tensor, Tensor, Tensor, Tensor, bool, bool, bool)",
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
):
    pass


@torch.library.register_fake("custom_ops::selective_scan_fwd")
def custom_selective_scan_fwd_fake(
    u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
):
    dstate = A.size(1) * (2 if A.is_complex() else 1)
    seqlen = u.size(2)
    n_chunks = (seqlen + 2048 - 1) // 2048

    squeeze_B = B.dim() == 3
    squeeze_C = C.dim() == 3
    has_z = z is not None

    final_out = torch.empty_like(delta)
    out_fake = torch.empty_like(delta)
    last_state_fake = (
        u.new_empty((u.size(0), u.size(1), dstate))
        if return_last_state
        else u.new_empty(0)
    )
    x_fake = u.new_empty((u.size(0), u.size(1), n_chunks, 2 * A.size(1)), dtype=A.dtype)

    return final_out, last_state_fake, out_fake, x_fake, squeeze_B, squeeze_C, has_z


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

    out, x, *rest = selective_scan_cuda.fwd(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus
    )
    has_z = z is not None
    if has_z:
        final_out = rest[0].clone()
    else:
        final_out = out.clone()
    last_state = x[:, :, -1, 1::2].clone() if return_last_state else u.new_empty(0)
    return final_out, last_state, out, x, squeeze_B, squeeze_C, has_z


@torch.library.custom_op(
    "custom_ops::selective_scan_bwd",
    device_types=["cuda"],
    mutates_args=(),
    schema="(Tensor dout, Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C, Tensor? D, Tensor? z, Tensor? delta_bias, bool delta_softplus, Tensor out, Tensor x, bool squeeze_B, bool squeeze_C, bool recompute_out_z) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor?, Tensor?, Tensor?)",
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
    recompute_out_z: bool,
):
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
    recompute_out_z,
):
    # Here we just return shape-compatible fake tensors
    du = torch.empty_like(u)
    ddelta = torch.empty_like(delta)
    dA = torch.empty_like(A)

    # Decide if variable B/C
    is_variable_B = B.dim() > 3
    is_variable_C = C.dim() > 3

    dB = torch.empty_like(
        B, dtype=B.dtype
    )  # If variable_B, still float32 is okay for fake
    dC = torch.empty_like(C, dtype=C.dtype)

    dD = torch.empty_like(D) if (D is not None) else None
    ddelta_bias_out = torch.empty_like(delta_bias) if (delta_bias is not None) else None
    dz = torch.empty_like(z) if (z is not None) else None

    if squeeze_B and dB.numel() > 0:
        dB = dB.squeeze(1)
    if squeeze_C and dC.numel() > 0:
        dC = dC.squeeze(1)

    return du, ddelta, dA, dB, dC, dD, ddelta_bias_out, dz


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
    recompute_out_z: bool,
):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()

    results = selective_scan_cuda.bwd(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        dout,
        x,
        out,
        None,
        delta_softplus,
        recompute_out_z,
    )

    has_z = z is not None
    if has_z:
        du, ddelta, dA, dB, dC, dD, ddelta_bias_out, dz = results
    else:
        du, ddelta, dA, dB, dC, dD, ddelta_bias_out = results
        dz = None

    if squeeze_B and dB.numel() > 0:
        dB = dB.squeeze(1)
    if squeeze_C and dC.numel() > 0:
        dC = dC.squeeze(1)

    return du, ddelta, dA, dB, dC, dD, ddelta_bias_out, dz


def custom_bridge(ctx, *grads):
    dout = grads[0] if grads else ctx.saved_tensors[0].new_empty(0)
    saved = ctx.saved_tensors

    if not ctx.has_z:
        u, delta, A, B, C, D, delta_bias, x, out = saved
        z = None
    else:
        u, delta, A, B, C, D, z, delta_bias, x, out = saved

    du, ddelta, dA, dB, dC, dD, ddelta_bias_out, dz = (
        torch.ops.custom_ops.selective_scan_bwd(
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
            ctx.squeeze_C,
            False,
        )
    )

    # For optional inputs, return None if not provided in forward
    if D is None:
        dD = None
    if z is None:
        dz = None
    if delta_bias is None:
        ddelta_bias_out = None

    # Return gradients in the order of forward inputs:
    # (u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
    # `delta_softplus` and `return_last_state` are bools -> gradient = None
    d_delta_softplus = None
    d_return_last_state = None

    return (
        du,
        ddelta,
        dA,
        dB,
        dC,
        dD,
        dz,
        ddelta_bias_out,
        d_delta_softplus,
        d_return_last_state,
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
    final_out, last_state, _, _, _, _, _ = torch.ops.custom_ops.selective_scan_fwd(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
    )
    if return_last_state:
        return final_out, last_state
    else:
        return final_out
