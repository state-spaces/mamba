# Copyright (c) 2025, Tri Dao.
# We need a pretty recent version of triton to support tuples. 3.3 definitely will work,
# idk which is the minimum version.

import math
from typing import Optional, Tuple

import torch

import triton
import triton.language as tl
import triton.testing
#from flash_attn.cute.benchmark import pytorch_profiler

@triton.jit
def rotary_qk_inference_kernel(
    OUT_Q,  # Pointers to matrices
    OUT_K,
    OUT_ANGLE_STATE,
    Q,
    K,
    ANGLE_STATE,
    ANGLE_PROJ,
    DT,
    BIAS_Q,
    BIAS_K,
    nheads,
    headdim,
    stride_out_q,           # (batch, mimo_dim, nheads, headdim)
    stride_out_k,           # (batch, mimo_dim, nheads, headdim)
    stride_out_angle_state, # (batch, nheads, rotary_dim // 2)
    stride_q,               # (batch, mimo_dim, nheads, headdim)
    stride_k,               # (batch, mimo_dim, nheads, headdim)
    stride_angle_state,     # (batch, nheads, rotary_dim // 2)
    stride_angle_proj,      # (batch, nheads, rotary_dim // 2)
    stride_dt,              # (batch, nheads)
    stride_bias_q,          # (mimo_dim, nheads, headdim)
    stride_bias_k,          # (mimo_dim, nheads, headdim)
    # Meta-parameters
    ROTARY_DIM: tl.constexpr,
    CONJUGATE: tl.constexpr,
    HAS_BIAS_Q: tl.constexpr,
    HAS_BIAS_K: tl.constexpr,
    MIMO_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr, # headdim, no chunking
    ROTATE_PAIRWISE: tl.constexpr, # If true, rotate every pair of dimensions together. Otherwise, rotate the first half and second half separately (like in the original RoPE paper)
):
    pid_nheads = tl.program_id(axis=0) # heads
    pid_batch = tl.program_id(axis=1)

    Q = Q + pid_batch * stride_q[0] + pid_nheads * stride_q[2]
    K = K + pid_batch * stride_k[0] + pid_nheads * stride_k[2]
    ANGLE_STATE = ANGLE_STATE + pid_batch * stride_angle_state[0] + pid_nheads * stride_angle_state[1]  # FIX: [1]
    ANGLE_PROJ = ANGLE_PROJ + pid_batch * stride_angle_proj[0] + pid_nheads * stride_angle_proj[1]      # FIX: [1]
    DT = DT + pid_batch * stride_dt[0] + pid_nheads * stride_dt[1]

    OUT_Q = OUT_Q + pid_batch * stride_out_q[0] + pid_nheads * stride_out_q[2]
    OUT_K = OUT_K + pid_batch * stride_out_k[0] + pid_nheads * stride_out_k[2]
    OUT_ANGLE_STATE = OUT_ANGLE_STATE + pid_batch * stride_out_angle_state[0] + pid_nheads * stride_out_angle_state[1]  # FIX: [1]

    rm = tl.arange(0, MIMO_DIM)
    rd = tl.arange(0, BLOCK_D)
    rd_half = tl.arange(0, BLOCK_D // 2)

    # Load angle and compute cos/sin (same for both q and k)
    ANGLE_STATE = ANGLE_STATE + rd_half * stride_angle_state[2]  # (rotary_dim // 2)
    mask_angle = rd_half < ROTARY_DIM // 2
    angle_state = tl.load(ANGLE_STATE, mask=mask_angle, other=0.0).to(tl.float32)

    ANGLE_PROJ = ANGLE_PROJ + rd_half * stride_angle_proj[2]     # (rotary_dim // 2)
    angle_proj = tl.load(ANGLE_PROJ, mask=mask_angle, other=0.0).to(tl.float32)

    dt = tl.load(DT, mask=True, other=0.0).to(tl.float32)

    # Match angle_dt: tanh(angle_proj) * dt * pi
    angle_proj = tl.sigmoid(2.0 * angle_proj) * 2.0 - 1.0  # tanh
    angle = angle_state + angle_proj * dt * 3.141592653589793  # (rotary_dim // 2)

    OUT_ANGLE_STATE = OUT_ANGLE_STATE + rd_half * stride_out_angle_state[2]
    tl.store(OUT_ANGLE_STATE, angle, mask=mask_angle)

    angle = angle[None, :]  # (1, rotary_dim // 2) for mimo_dim broadcasting
    cos = tl.cos(angle)
    sin = tl.sin(angle)
    if CONJUGATE:
        sin = -sin

    # Process Q tensor
    Q = Q + (rm[:, None] * stride_q[1] + rd[None, :] * stride_q[3])
    OUT_Q = OUT_Q + (rm[:, None] * stride_out_q[1] + rd[None, :] * stride_out_q[3])
    mask = rd[None, :] < headdim
    q = tl.load(Q, mask=mask, other=0.0).to(tl.float32)  # (mimo_dim, headdim)

    # Add bias to Q if present
    if HAS_BIAS_Q:
        BIAS_Q = BIAS_Q + pid_nheads * stride_bias_q[1]                                                   
        BIAS_Q = BIAS_Q + (rm[:, None] * stride_bias_q[0] + rd[None, :] * stride_bias_q[2])
        bias_q = tl.load(BIAS_Q, mask=mask, other=0.0).to(tl.float32)
        q = q + bias_q

    if ROTATE_PAIRWISE:
        # Apply rotary to Q
        q0, q1 = tl.split(tl.reshape(q, [MIMO_DIM, BLOCK_D // 2, 2]))
        qo0 = q0 * cos - q1 * sin
        qo1 = q0 * sin + q1 * cos
        qo = tl.reshape(tl.join(qo0, qo1), [MIMO_DIM, BLOCK_D])
        tl.store(OUT_Q, qo, mask=mask)
    else:
        # Apply rotary to Q
        q_reshaped = tl.reshape(q, [MIMO_DIM, 2, BLOCK_D // 2])
        q_permuted = tl.permute(q_reshaped, (0, 2, 1))  # (mimo_dim, block_d // 2, 2)
        q0, q1 = tl.split(q_permuted)
        qo0 = q0 * cos - q1 * sin
        qo1 = q0 * sin + q1 * cos
        q_joined = tl.join(qo0, qo1)
        q_final = tl.permute(q_joined, (0, 2, 1))  # (mimo_dim, 2, block_d // 2)
        qo = tl.reshape(q_final, [MIMO_DIM, BLOCK_D])
        tl.store(OUT_Q, qo, mask=mask)

    # Process K tensor
    K = K + (rm[:, None] * stride_k[1] + rd[None, :] * stride_k[3])
    OUT_K = OUT_K + (rm[:, None] * stride_out_k[1] + rd[None, :] * stride_out_k[3])
    k = tl.load(K, mask=mask, other=0.0).to(tl.float32)

    # Add bias to K if present
    if HAS_BIAS_K:
        BIAS_K = BIAS_K + pid_nheads * stride_bias_k[1]                                                 
        BIAS_K = BIAS_K + (rm[:, None] * stride_bias_k[0] + rd[None, :] * stride_bias_k[2])
        bias_k = tl.load(BIAS_K, mask=mask, other=0.0).to(tl.float32)
        k = k + bias_k

    if ROTATE_PAIRWISE:
        # Apply rotary to K
        k0, k1 = tl.split(tl.reshape(k, [MIMO_DIM, BLOCK_D // 2, 2]))
        ko0 = k0 * cos - k1 * sin
        ko1 = k0 * sin + k1 * cos
        ko = tl.reshape(tl.join(ko0, ko1), [MIMO_DIM, BLOCK_D])
        tl.store(OUT_K, ko, mask=mask)
    else:
        # Apply rotary to K
        k_reshaped = tl.reshape(k, [MIMO_DIM, 2, BLOCK_D // 2])
        k_permuted = tl.permute(k_reshaped, (0, 2, 1))  # (mimo_dim, block_d // 2, 2)
        k0, k1 = tl.split(k_permuted)
        ko0 = k0 * cos - k1 * sin
        ko1 = k0 * sin + k1 * cos
        k_joined = tl.join(ko0, ko1)
        k_final = tl.permute(k_joined, (0, 2, 1))  # (mimo_dim, 2, block_d // 2)
        ko = tl.reshape(k_final, [MIMO_DIM, BLOCK_D])
        tl.store(OUT_K, ko, mask=mask)

def apply_rotary_qk_inference_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    angle_state: torch.Tensor,
    angle_proj: torch.Tensor,
    dt: torch.Tensor,
    bias_q: Optional[torch.Tensor] = None,
    bias_k: Optional[torch.Tensor] = None,
    inplace=False,
    conjugate=False,
    rotate_pairwise=True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply rotary embedding to both q and k tensors using the same angle.
    Also computes output angle state for next step.

    Arguments:
        q: (batch, mimo_dim, nheads, headdim)
        k: (batch, mimo_dim, nheads, headdim)
        angle_state: (batch, nheads, rotary_dim / 2)
        angle_proj: (batch, nheads, rotary_dim / 2)
        dt: (batch, nheads)
        bias_q: Optional (mimo_dim, nheads, headdim) - bias to add to q before rotary
        bias_k: Optional (mimo_dim, nheads, headdim) - bias to add to k before rotary
    Returns:
        (q_out, k_out, angle_state_out): q_out and k_out are (batch, mimo_dim, nheads, headdim),
                               angle_state_out is (batch, nheads, rotary_dim / 2)
    """
    batch, mimo_dim, nheads, headdim = q.shape
    assert headdim % 2 == 0
    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"

    rotary_dim = angle_state.shape[-1] * 2
    assert angle_state.shape == (batch, nheads, rotary_dim // 2)
    assert angle_state.shape == angle_proj.shape
    assert dt.shape == (batch, nheads)
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"

    if bias_q is not None:
        assert bias_q.shape == (mimo_dim, nheads, headdim), f"bias_q shape {bias_q.shape} != (mimo_dim, nheads, headdim) {(mimo_dim, nheads, headdim)}"
        bias_q = bias_q.contiguous()

    if bias_k is not None:
        assert bias_k.shape == (mimo_dim, nheads, headdim), f"bias_k shape {bias_k.shape} != (mimo_dim, nheads, headdim) {(mimo_dim, nheads, headdim)}"
        bias_k = bias_k.contiguous()

    output_q = torch.empty_like(q) if not inplace else q
    output_k = torch.empty_like(k) if not inplace else k
    output_angle_state = torch.empty_like(angle_state) if not inplace else angle_state

    grid = lambda META: (nheads, batch)  # noqa
    with torch.cuda.device(q.device.index):
        torch.library.wrap_triton(rotary_qk_inference_kernel)[grid](
            output_q,  # data ptrs
            output_k,
            output_angle_state,
            q,
            k,
            angle_state,
            angle_proj,
            dt,
            bias_q,
            bias_k,
            nheads,
            headdim,
            output_q.stride(),  # output strides tuples
            output_k.stride(),
            output_angle_state.stride(),
            q.stride(),  # input strides tuples
            k.stride(),
            angle_state.stride(),
            angle_proj.stride(),
            dt.stride(),
            bias_q.stride() if bias_q is not None else (0, 0, 0),
            bias_k.stride() if bias_k is not None else (0, 0, 0),
            rotary_dim,
            conjugate,
            bias_q is not None,
            bias_k is not None,
            MIMO_DIM=mimo_dim,
            BLOCK_D=triton.next_power_of_2(headdim),
            num_warps=8,  # important, 4 warps is slower if we compute qk_sum
            ROTATE_PAIRWISE=rotate_pairwise,
        )
    return output_q, output_k, output_angle_state


def apply_rotary_qk_inference_reference(
    q: torch.Tensor, # (B, R, N, D)
    k: torch.Tensor, # (B, R, N, D)
    angle_state: torch.Tensor, # (B, N, S) S: num_rope_angles
    angle_proj: torch.Tensor, # (B, N, S)
    dt: torch.Tensor, # (B, N)
    bias_q: Optional[torch.Tensor] = None, # (R, N, D)
    bias_k: Optional[torch.Tensor] = None, # (R, N, D)
    conjugate=False,
    rotate_pairwise=True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference PyTorch implementation for QK rotary embedding with qk_sum."""
    batch, mimo_dim, nheads, headdim = q.shape
    rotary_dim = angle_state.shape[-1] * 2

    # Match angle_dt: tanh(angle_proj) * dt * pi
    angle_proj = torch.tanh(angle_proj)
    angle = angle_state + angle_proj * dt[:, :, None] * math.pi  # (B, N, S)
    angle_state_new = angle
    angle = angle.unsqueeze(1).expand(-1, mimo_dim, -1, -1)  # (B, R, N, S)

    # Add biases if present
    if bias_q is not None:
        q = q + bias_q[None, :, :, :]  # Broadcast bias_q
    if bias_k is not None:
        k = k + bias_k[None, :, :, :]  # Broadcast bias_k

    # Only apply rotary to the rotary dimensions
    q_rot = q[..., :rotary_dim] # (B, R, N, rotary_dim)
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    # Compute cos and sin from angle (same for both q and k)
    cos = torch.cos(angle) # (B, N, S)
    sin = torch.sin(angle)
    if conjugate:
        sin = -sin

    if rotate_pairwise:
        # Interleaved rotary: pairs are (x0,x1), (x2,x3), ...
        q_rot = q_rot.reshape(batch, mimo_dim, nheads, rotary_dim // 2, 2)
        q0, q1 = q_rot[..., 0], q_rot[..., 1]
        k_rot = k_rot.reshape(batch, mimo_dim, nheads, rotary_dim // 2, 2)
        k0, k1 = k_rot[..., 0], k_rot[..., 1]

        qo0 = q0 * cos - q1 * sin
        qo1 = q0 * sin + q1 * cos
        ko0 = k0 * cos - k1 * sin
        ko1 = k0 * sin + k1 * cos

        qout_rot = torch.stack([qo0, qo1], dim=-1).reshape(batch, mimo_dim, nheads, rotary_dim)
        kout_rot = torch.stack([ko0, ko1], dim=-1).reshape(batch, mimo_dim, nheads, rotary_dim)

        # Concatenate rotated and pass-through dimensions
        if rotary_dim < headdim:
            q_out = torch.cat([qout_rot, q_pass], dim=-1)
            k_out = torch.cat([kout_rot, k_pass], dim=-1)
        else:
            q_out = qout_rot
            k_out = kout_rot
    else:
        # Halved rotary: split full headdim in half, pairs are (dim_i, dim_{i+D/2})
        # Matches kernel which splits BLOCK_D in half; cos(0)=1/sin(0)=0 gives identity
        # for pairs beyond rotary_dim//2
        half = headdim // 2
        q0, q1 = q[..., :half], q[..., half:]
        k0, k1 = k[..., :half], k[..., half:]

        # Pad cos/sin from rotary_dim//2 to headdim//2 with cos=1, sin=0
        rdim_half = rotary_dim // 2
        if half > rdim_half:
            pad_shape = list(cos.shape)
            pad_shape[-1] = half - rdim_half
            cos = torch.cat([cos, torch.ones(pad_shape, device=cos.device, dtype=cos.dtype)], dim=-1)
            sin = torch.cat([sin, torch.zeros(pad_shape, device=sin.device, dtype=sin.dtype)], dim=-1)

        qo0 = q0 * cos - q1 * sin
        qo1 = q0 * sin + q1 * cos
        ko0 = k0 * cos - k1 * sin
        ko1 = k0 * sin + k1 * cos

        q_out = torch.cat([qo0, qo1], dim=-1)
        k_out = torch.cat([ko0, ko1], dim=-1)

    return q_out, k_out, angle_state_new


def test_correctness_qk_inference():
    print("Testing QK Inference correctness...")

    device = "cuda"
    torch.manual_seed(2025)
    dtype_qk = torch.bfloat16  # common inference dtype
    dtype_ang = torch.float32

    def run_case(B, R, N, D, RD, with_bias, conjugate, expanded_heads, rotate_pairwise):
        assert D % 2 == 0
        # Build q,k with optional head broadcasting
        q0 = torch.randn(B, R, 1 if expanded_heads else N, D, device=device, dtype=dtype_qk)
        k0 = torch.randn(B, R, 1 if expanded_heads else N, D, device=device, dtype=dtype_qk)
        q  = q0.expand(B, R, N, D) if expanded_heads else q0
        k  = k0.expand(B, R, N, D) if expanded_heads else k0

        angle_state = torch.randn(B, N, RD // 2, device=device, dtype=dtype_ang)
        angle_proj  = torch.randn(B, N, RD // 2, device=device, dtype=dtype_ang)
        dt          = torch.randn(B, N, device=device, dtype=dtype_ang)

        bias_q = torch.randn(R, N, D, device=device, dtype=dtype_qk) if with_bias else None
        bias_k = torch.randn(R, N, D, device=device, dtype=dtype_qk) if with_bias else None

        # Reference
        q_ref, k_ref, updated_angle_ref = apply_rotary_qk_inference_reference(
            q, k, angle_state, angle_proj, dt,
            bias_q=bias_q, bias_k=bias_k, conjugate=conjugate,
            rotate_pairwise=rotate_pairwise,
        )

        # Kernel
        q_out, k_out, updated_angle = apply_rotary_qk_inference_fwd(
            q, k, angle_state, angle_proj, dt,
            bias_q=bias_q, bias_k=bias_k, conjugate=conjugate, inplace=False,
            rotate_pairwise=rotate_pairwise,
        )

        def _chk(name, a, b, atol=1e-1, rtol=1e-1):
            diff = (a - b).abs().max().item()
            if not torch.allclose(a, b, atol=atol, rtol=rtol):
                raise AssertionError(f"{name} mismatch: max|Δ|={diff:.3e}  got={tuple(a.shape)}  ref={tuple(b.shape)}")
            print(f"  {name:18s} ok   max|Δ|={diff:.2e}")

        print(f"\nInference [{B=}, {R=}, {N=}, {D=}, {RD=} | bias={with_bias}, conj={conjugate}, expanded={expanded_heads}, pairwise={rotate_pairwise}]")
        _chk("q_out", q_out.float(), q_ref.float(), atol=1e-1, rtol=1e-1)
        _chk("k_out", k_out.float(), k_ref.float(), atol=1e-1, rtol=1e-1)
        _chk("updated_angle", updated_angle, updated_angle_ref, atol=1e-1, rtol=1e-1)

    # standard config
    B, R, N, D, RD = 2, 4, 64, 128, 64
    for with_bias in [False, True]:
        for conjugate in [False, True]:
            for expanded in [True, False]:
                for pairwise in [True, False]:
                    run_case(B, R, N, D, RD, with_bias, conjugate, expanded, pairwise)

    # light shape sweep
    for (BB, RR, NN, DD, RRd) in [
        (1, 2, 64, 64,  32),
        (3, 1, 32, 128, 64),
        (2, 8, 32, 128, 64),
    ]:
        for pairwise in [True, False]:
            run_case(BB, RR, NN, DD, RRd, with_bias=True,  conjugate=False, expanded_heads=True, rotate_pairwise=pairwise)
            run_case(BB, RR, NN, DD, RRd, with_bias=False, conjugate=True,  expanded_heads=False, rotate_pairwise=pairwise)

    print("\nAll QK Inference tests passed! ✓")


if __name__ == "__main__":
    test_correctness_qk_inference()
