"""GPU tests for Mamba-3 Triton backward pass correctness.

Compares gradients from the Triton backward pipeline (mamba3_chunk_scan_combined_triton
from mamba3_combined.py) against the PyTorch reference backward (mamba3_chunk_scan_combined
from mamba3_ssd.py) using manual gradient comparison.

Test strategy:
1. Create identical inputs for both paths (with requires_grad=True).
2. Run forward + backward on both.
3. Compare all gradients (dx, ddt, dA, dB, dC, dgamma, dbeta, dD, dz, dtheta, etc.)
   with appropriate tolerances per dtype.

All tests require an NVIDIA CUDA GPU.
"""

import pytest
import torch
import torch.nn.functional as F

DEVICE = "cuda"

# Skip entire module if CUDA is not available.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# Tolerances per dtype.
FP32_ATOL = 5e-3
FP32_RTOL = 5e-2
BF16_ATOL = 5e-3
BF16_RTOL = 5e-2


def _tols(dtype):
    if dtype == torch.bfloat16:
        return BF16_ATOL, BF16_RTOL
    return FP32_ATOL, FP32_RTOL


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def _make_inputs(
    batch=2,
    seqlen=128,
    nheads=4,
    headdim=32,
    dstate=16,
    ngroups=None,
    dtype=torch.float32,
    with_gamma_beta=True,
    with_theta=False,
    with_z=False,
    with_D=False,
    D_2d=False,
    with_dt_bias=False,
    dt_softplus=False,
    with_seq_idx=False,
    with_initial_states=False,
    return_final_states=False,
    with_initial_prev_Bx=False,
    seed=42,
):
    """Create test inputs.  Returns a dict of tensors (no requires_grad yet)."""
    if ngroups is None:
        ngroups = nheads

    torch.manual_seed(seed)
    device = DEVICE

    x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
    dt = torch.rand(batch, seqlen, nheads, device=device, dtype=dtype) * 0.1 + 0.01
    A = -torch.rand(nheads, device=device, dtype=dtype) * 3 - 0.5
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) * 0.1
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) * 0.1

    gamma = beta = None
    if with_gamma_beta:
        gamma = torch.rand(batch, seqlen, nheads, device=device, dtype=dtype) * 0.1 + 0.01
        beta = torch.rand(batch, seqlen, nheads, device=device, dtype=dtype) * 0.1 + 0.01

    theta = None
    if with_theta:
        theta = torch.randn(batch, seqlen, nheads, dstate // 2, device=device, dtype=dtype) * 0.05

    z = None
    if with_z:
        z = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)

    D = None
    if with_D:
        if D_2d:
            D = torch.randn(nheads, headdim, device=device, dtype=dtype) * 0.1
        else:
            D = torch.randn(nheads, device=device, dtype=dtype) * 0.1

    dt_bias = None
    if with_dt_bias:
        dt_bias = torch.rand(nheads, device=device, dtype=dtype) * 0.005

    seq_idx = None
    if with_seq_idx:
        seq_idx = torch.zeros(batch, seqlen, dtype=torch.long, device=device)
        seq_idx[:, seqlen // 2:] = 1

    initial_states = None
    if with_initial_states:
        initial_states = torch.randn(batch, nheads, headdim, dstate, device=device, dtype=dtype) * 0.1

    initial_prev_Bx = None
    if with_initial_prev_Bx and with_gamma_beta:
        initial_prev_Bx = torch.randn(batch, nheads, headdim, dstate, device=device, dtype=dtype) * 0.1

    return dict(
        x=x, dt=dt, A=A, B=B, C=C,
        gamma=gamma, beta=beta, theta=theta,
        z=z, D=D, dt_bias=dt_bias,
        seq_idx=seq_idx,
        initial_states=initial_states,
        initial_prev_Bx=initial_prev_Bx,
        ngroups=ngroups,
        dt_softplus=dt_softplus,
        return_final_states=return_final_states,
    )


# Names of all tensor inputs that are potentially differentiable.
_DIFF_NAMES = [
    "x", "dt", "A", "B", "C",
    "gamma", "beta", "theta",
    "z", "D", "dt_bias",
    "initial_states", "initial_prev_Bx",
]


def _clone_with_grad(inputs):
    """Clone all tensors; set requires_grad on differentiable ones."""
    cloned = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and k in _DIFF_NAMES:
            c = v.detach().clone().requires_grad_(True)
            cloned[k] = c
        elif isinstance(v, torch.Tensor):
            cloned[k] = v.detach().clone()  # seq_idx etc.
        else:
            cloned[k] = v
    return cloned


# ---------------------------------------------------------------------------
# Core comparison helper
# ---------------------------------------------------------------------------

def _compare_gradients(chunk_size=64, atol_override=None, rtol_override=None, **input_kwargs):
    """Run Triton and PyTorch reference backward, compare all gradients.

    ``input_kwargs`` are forwarded to ``_make_inputs``.
    """
    from mamba_ssm.ops.triton.mamba3_combined import mamba3_chunk_scan_combined_triton
    from mamba_ssm.ops.triton.mamba3_ssd import (
        mamba3_chunk_scan_combined as mamba3_ref,
    )

    raw = _make_inputs(**input_kwargs)
    dtype = input_kwargs.get("dtype", torch.float32)
    atol, rtol = _tols(dtype)
    if atol_override is not None:
        atol = atol_override
    if rtol_override is not None:
        rtol = rtol_override
    return_final_states = raw.get("return_final_states", False)

    # --- Triton path ---
    tri = _clone_with_grad(raw)
    tri_result = mamba3_chunk_scan_combined_triton(
        tri["x"], tri["dt"], tri["A"], tri["B"], tri["C"], chunk_size,
        gamma=tri["gamma"], beta=tri["beta"], theta=tri["theta"],
        D=tri["D"], z=tri["z"], dt_bias=tri["dt_bias"],
        dt_softplus=tri.get("dt_softplus", False),
        initial_states=tri["initial_states"],
        initial_prev_Bx=tri["initial_prev_Bx"],
        return_final_states=return_final_states,
        ngroups=tri["ngroups"],
        seq_idx=tri.get("seq_idx"),
    )
    if return_final_states:
        tri_out, tri_final = tri_result
        tri_loss = tri_out.float().sum() + tri_final.float().sum()
    else:
        tri_out = tri_result
        tri_loss = tri_out.float().sum()
    tri_loss.backward()

    # --- PyTorch reference path ---
    ref = _clone_with_grad(raw)
    ref_result = mamba3_ref(
        ref["x"], ref["dt"], ref["A"], ref["B"], ref["C"], chunk_size,
        gamma=ref["gamma"], beta=ref["beta"], theta=ref["theta"],
        D=ref["D"], z=ref["z"], dt_bias=ref["dt_bias"],
        dt_softplus=ref.get("dt_softplus", False),
        initial_states=ref["initial_states"],
        initial_prev_Bx=ref["initial_prev_Bx"],
        return_final_states=return_final_states,
        ngroups=ref["ngroups"],
        seq_idx=ref.get("seq_idx"),
    )
    if return_final_states:
        ref_out, ref_final = ref_result
        ref_loss = ref_out.float().sum() + ref_final.float().sum()
    else:
        ref_out = ref_result
        ref_loss = ref_out.float().sum()
    ref_loss.backward()

    # --- Forward comparison ---
    torch.testing.assert_close(
        tri_out.float(), ref_out.float(), atol=atol, rtol=rtol,
        msg="Forward output mismatch",
    )
    if return_final_states:
        torch.testing.assert_close(
            tri_final.float(), ref_final.float(), atol=atol, rtol=rtol,
            msg="Final state mismatch",
        )

    # --- Gradient comparison ---
    for name in _DIFF_NAMES:
        t_tensor = tri.get(name)
        r_tensor = ref.get(name)
        if t_tensor is None or r_tensor is None:
            continue
        t_grad = t_tensor.grad
        r_grad = r_tensor.grad
        if t_grad is None and r_grad is None:
            continue
        assert t_grad is not None, f"Triton grad for '{name}' is None but ref grad exists"
        assert r_grad is not None, f"Ref grad for '{name}' is None but Triton grad exists"
        assert torch.isfinite(t_grad).all(), f"Non-finite Triton grad for '{name}'"
        assert torch.isfinite(r_grad).all(), f"Non-finite ref grad for '{name}'"
        torch.testing.assert_close(
            t_grad.float(), r_grad.float(), atol=atol, rtol=rtol,
            msg=f"Gradient mismatch for '{name}'",
        )


# ============================================================================
# 1. Basic SISO (ngroups=nheads, rank=1)
# ============================================================================

class TestBasicSISO:
    """Basic SISO backward: trapezoidal (Mamba-3) and Euler (Mamba-2 fallback)."""

    def test_trapezoidal_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
        )

    def test_trapezoidal_bf16(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.bfloat16, with_gamma_beta=True,
        )

    def test_euler_fp32(self):
        """Mamba-2 fallback: no gamma/beta."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=False,
        )

    def test_euler_bf16(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.bfloat16, with_gamma_beta=False,
        )


# ============================================================================
# 2. With RoPE (theta parameter)
# ============================================================================

class TestWithRoPE:
    """Backward with RoPE rotary embeddings on B and C."""

    def test_rope_trapezoidal_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_theta=True,
        )

    def test_rope_trapezoidal_bf16(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.bfloat16, with_gamma_beta=True, with_theta=True,
        )

    def test_rope_euler_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=False, with_theta=True,
        )

    def test_rope_ngroups_lt_nheads_fp32(self):
        """RoPE with ngroups < nheads (group-to-head expansion)."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=8, headdim=32, dstate=16,
            ngroups=2, dtype=torch.float32,
            with_gamma_beta=True, with_theta=True,
        )


# ============================================================================
# 3. With z gating (SiLU)
# ============================================================================

class TestWithZGating:

    def test_z_trapezoidal_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_z=True,
        )

    def test_z_trapezoidal_bf16(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.bfloat16, with_gamma_beta=True, with_z=True,
        )

    def test_z_euler_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=False, with_z=True,
        )


# ============================================================================
# 4. With D skip connection
# ============================================================================

class TestWithDSkip:

    def test_D_1d_trapezoidal_fp32(self):
        """D as (nheads,) vector."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_D=True, D_2d=False,
        )

    def test_D_2d_trapezoidal_fp32(self):
        """D as (nheads, headdim) matrix."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_D=True, D_2d=True,
        )

    def test_D_1d_bf16(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.bfloat16, with_gamma_beta=True, with_D=True, D_2d=False,
        )

    def test_D_with_z_fp32(self):
        """D + z gating combined."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
            with_D=True, with_z=True,
        )


# ============================================================================
# 5. With seq_idx (packed multi-document sequences)
# ============================================================================

class TestWithSeqIdx:

    def test_seq_idx_trapezoidal_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_seq_idx=True,
        )

    def test_seq_idx_trapezoidal_bf16(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.bfloat16, with_gamma_beta=True, with_seq_idx=True,
        )

    def test_seq_idx_euler_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=False, with_seq_idx=True,
        )

    def test_seq_idx_with_rope_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32,
            with_gamma_beta=True, with_theta=True, with_seq_idx=True,
        )


# ============================================================================
# 6. With initial_states / return_final_states
# ============================================================================

class TestWithInitialStates:

    def test_initial_states_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
            with_initial_states=True, return_final_states=True,
        )

    def test_initial_states_bf16(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.bfloat16, with_gamma_beta=True,
            with_initial_states=True, return_final_states=True,
        )

    def test_return_final_states_only_fp32(self):
        """return_final_states without providing initial_states."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
            with_initial_states=False, return_final_states=True,
        )

    def test_initial_prev_Bx_fp32(self):
        """initial_prev_Bx for trapezoidal lookback at t=0."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
            with_initial_prev_Bx=True,
        )

    def test_initial_states_and_prev_Bx_fp32(self):
        """Both initial_states and initial_prev_Bx."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
            with_initial_states=True, with_initial_prev_Bx=True,
            return_final_states=True,
        )


# ============================================================================
# 7. Trapezoidal mode (gamma+beta) -- core Mamba-3
# ============================================================================

class TestTrapezoidal:
    """Dedicated trapezoidal-mode tests at various configurations."""

    def test_trapezoidal_small_chunk_fp32(self):
        _compare_gradients(
            chunk_size=32,
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
        )

    def test_trapezoidal_large_dstate_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=64,
            dtype=torch.float32, with_gamma_beta=True,
        )

    def test_trapezoidal_with_dt_bias_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_dt_bias=True,
        )

    def test_trapezoidal_with_dt_softplus_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
            with_dt_bias=True, dt_softplus=True,
        )


# ============================================================================
# 8. Mamba-2 fallback (no gamma/beta)
# ============================================================================

class TestMamba2Fallback:
    """Euler discretization (Mamba-2 compatible) backward tests."""

    def test_euler_with_D_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=False, with_D=True,
        )

    def test_euler_with_z_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=False, with_z=True,
        )

    def test_euler_with_initial_states_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=False,
            with_initial_states=True, return_final_states=True,
        )

    def test_euler_with_rope_bf16(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.bfloat16, with_gamma_beta=False, with_theta=True,
        )


# ============================================================================
# 9. Different shapes
# ============================================================================

class TestDifferentShapes:

    def test_seqlen_256_nheads_4_fp32(self):
        _compare_gradients(
            batch=2, seqlen=256, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
        )

    def test_nheads_8_headdim_64_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=8, headdim=64, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
        )

    def test_dstate_64_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=64,
            dtype=torch.float32, with_gamma_beta=True,
        )

    def test_ngroups_1_nheads_8_fp32(self):
        """Large head-to-group ratio."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=8, headdim=32, dstate=16,
            ngroups=1, dtype=torch.float32, with_gamma_beta=True,
        )

    def test_ngroups_2_nheads_8_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=8, headdim=32, dstate=16,
            ngroups=2, dtype=torch.float32, with_gamma_beta=True,
        )

    def test_small_shapes_fp32(self):
        """Minimal configuration."""
        _compare_gradients(
            chunk_size=32,
            batch=1, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
        )

    def test_large_shapes_bf16(self):
        """Larger configuration in bf16 — wider tolerance for bf16 accumulation noise."""
        _compare_gradients(
            batch=2, seqlen=256, nheads=8, headdim=64, dstate=64,
            dtype=torch.bfloat16, with_gamma_beta=True,
            atol_override=0.05, rtol_override=0.1,
        )

    def test_headdim_64_dstate_64_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=64, dstate=64,
            dtype=torch.float32, with_gamma_beta=True,
        )


# ============================================================================
# 10. dt processing (bias + softplus)
# ============================================================================

class TestDtProcessing:

    def test_dt_bias_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_dt_bias=True,
        )

    def test_dt_softplus_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
            dt_softplus=True,
        )

    def test_dt_bias_and_softplus_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
            with_dt_bias=True, dt_softplus=True,
        )

    def test_dt_bias_bf16(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.bfloat16, with_gamma_beta=True,
            with_dt_bias=True,
        )


# ============================================================================
# 11. Combined features
# ============================================================================

class TestCombinedFeatures:
    """Tests combining multiple optional features at once."""

    def test_all_features_fp32(self):
        """gamma/beta + RoPE + z + D + dt_bias + softplus + initial_states + final_states."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32,
            with_gamma_beta=True, with_theta=True,
            with_z=True, with_D=True, with_dt_bias=True,
            dt_softplus=True,
            with_initial_states=True, return_final_states=True,
        )

    def test_all_features_bf16(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.bfloat16,
            with_gamma_beta=True, with_theta=True,
            with_z=True, with_D=True, with_dt_bias=True,
            dt_softplus=True,
            with_initial_states=True, return_final_states=True,
        )

    def test_trapezoidal_rope_seq_idx_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32,
            with_gamma_beta=True, with_theta=True, with_seq_idx=True,
        )

    def test_trapezoidal_D_z_initial_prev_Bx_fp32(self):
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32,
            with_gamma_beta=True, with_D=True, with_z=True,
            with_initial_states=True, with_initial_prev_Bx=True,
            return_final_states=True,
        )

    def test_euler_all_extras_fp32(self):
        """Euler mode with all other optional features."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32,
            with_gamma_beta=False, with_theta=True,
            with_z=True, with_D=True, with_dt_bias=True,
            dt_softplus=True,
            with_initial_states=True, return_final_states=True,
        )

    def test_ngroups_with_all_features_fp32(self):
        """ngroups < nheads with all features."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=8, headdim=32, dstate=16,
            ngroups=2, dtype=torch.float32,
            with_gamma_beta=True, with_theta=True,
            with_z=True, with_D=True,
            with_initial_states=True, return_final_states=True,
        )

    def test_seq_idx_initial_states_rope_trapezoidal_bf16(self):
        """Everything together in bf16 — wider tolerance for bf16 accumulation noise."""
        _compare_gradients(
            batch=2, seqlen=128, nheads=4, headdim=32, dstate=16,
            ngroups=2, dtype=torch.bfloat16,
            with_gamma_beta=True, with_theta=True,
            with_seq_idx=True, with_D=True,
            with_initial_states=True, return_final_states=True,
            atol_override=0.02, rtol_override=0.1,
        )


# ============================================================================
# 12. Individual gradient sanity checks
# ============================================================================

class TestIndividualGradients:
    """Verify specific gradient components in isolation (useful for debugging)."""

    def _get_grad(self, name, **kwargs):
        """Run both paths and return (triton_grad, ref_grad) for a named parameter."""
        from mamba_ssm.ops.triton.mamba3_combined import mamba3_chunk_scan_combined_triton
        from mamba_ssm.ops.triton.mamba3_ssd import (
            mamba3_chunk_scan_combined as mamba3_ref,
        )

        raw = _make_inputs(**kwargs)
        chunk_size = 64

        results = {}
        for label, fn in [("tri", mamba3_chunk_scan_combined_triton), ("ref", mamba3_ref)]:
            inp = _clone_with_grad(raw)
            out = fn(
                inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], chunk_size,
                gamma=inp["gamma"], beta=inp["beta"], theta=inp["theta"],
                D=inp["D"], z=inp["z"], dt_bias=inp["dt_bias"],
                dt_softplus=inp.get("dt_softplus", False),
                initial_states=inp["initial_states"],
                initial_prev_Bx=inp["initial_prev_Bx"],
                return_final_states=inp.get("return_final_states", False),
                ngroups=inp["ngroups"],
                seq_idx=inp.get("seq_idx"),
            )
            if isinstance(out, tuple):
                loss = out[0].float().sum() + out[1].float().sum()
            else:
                loss = out.float().sum()
            loss.backward()
            results[label] = inp[name].grad if inp[name] is not None else None

        return results["tri"], results["ref"]

    def test_dx_trapezoidal(self):
        tri_g, ref_g = self._get_grad(
            "x", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_ddt_euler(self):
        tri_g, ref_g = self._get_grad(
            "dt", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=False,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_dA(self):
        tri_g, ref_g = self._get_grad(
            "A", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_dB_trapezoidal(self):
        tri_g, ref_g = self._get_grad(
            "B", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_dC_trapezoidal(self):
        tri_g, ref_g = self._get_grad(
            "C", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_dgamma(self):
        tri_g, ref_g = self._get_grad(
            "gamma", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_dbeta(self):
        tri_g, ref_g = self._get_grad(
            "beta", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_dD(self):
        tri_g, ref_g = self._get_grad(
            "D", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_D=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_dtheta(self):
        tri_g, ref_g = self._get_grad(
            "theta", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_theta=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_dz(self):
        tri_g, ref_g = self._get_grad(
            "z", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_z=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_d_initial_states(self):
        tri_g, ref_g = self._get_grad(
            "initial_states", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True,
            with_initial_states=True, return_final_states=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_d_initial_prev_Bx(self):
        tri_g, ref_g = self._get_grad(
            "initial_prev_Bx", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_initial_prev_Bx=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)

    def test_d_dt_bias(self):
        tri_g, ref_g = self._get_grad(
            "dt_bias", batch=2, seqlen=64, nheads=4, headdim=32, dstate=16,
            dtype=torch.float32, with_gamma_beta=True, with_dt_bias=True,
        )
        assert tri_g is not None and ref_g is not None
        torch.testing.assert_close(tri_g.float(), ref_g.float(), atol=FP32_ATOL, rtol=FP32_RTOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
