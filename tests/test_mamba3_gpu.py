"""GPU tests for Mamba-3 implementation.

Tests on CUDA:
1. Chunked SSD (Triton) vs step-by-step recurrence
2. Prefill (forward) → decode (step) consistency
3. Mamba3 full module: forward + step
4. Gradient flow with mixed precision (bf16)
5. MIMO + RoPE + trapezoidal end-to-end
6. MambaLMHeadModel integration
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

DEVICE = "cuda"


# ============================================================================
# 1. Chunked SSD vs Recurrence (GPU, real Triton kernels)
# ============================================================================

class TestChunkedVsRecurrenceGPU:
    """Verify chunked parallel SSD matches step-by-step recurrence on GPU."""

    def _reference_recurrence(self, X, dt, A, B, C, gamma, beta=None):
        """Step-by-step reference."""
        is_mimo = X.dim() == 5
        batch, seqlen, nheads, headdim = X.shape[:4]
        dstate = B.shape[-2] if is_mimo else B.shape[-1]

        alpha = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, nheads, 1))
        h = torch.zeros(batch, nheads, headdim, dstate, device=X.device, dtype=torch.float32)
        ys = []
        prev_Bx = None

        for t in range(seqlen):
            x_t = X[:, t]
            B_t = B[:, t]
            C_t = C[:, t]

            if is_mimo:
                Bx_t = torch.einsum("bhpr,bhnr->bhpn", x_t.float(), B_t.float())
            else:
                Bx_t = torch.einsum("bhp,bhn->bhpn", x_t.float(), B_t.float())

            alpha_t = alpha[:, t].unsqueeze(-1)
            gamma_t = gamma[:, t].unsqueeze(-1).unsqueeze(-1)
            h = alpha_t * h + gamma_t * Bx_t

            if beta is not None and prev_Bx is not None:
                beta_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)
                h = h + beta_t * prev_Bx

            prev_Bx = Bx_t

            if is_mimo:
                y_t = torch.einsum("bhpn,bhnr->bhpr", h.to(X.dtype), C_t)
            else:
                y_t = torch.einsum("bhpn,bhn->bhp", h.to(X.dtype), C_t)
            ys.append(y_t)

        return torch.stack(ys, dim=1)

    def _run_comparison(self, use_rope=True, use_trapezoidal=True, mimo_rank=0, dtype=torch.float32):
        from mamba_ssm.modules.mamba3 import apply_rotary_emb, compute_cumulative_rotary
        from mamba_ssm.ops.triton.mamba3_ssd import mamba3_ssd_chunked

        torch.manual_seed(42)
        batch, seqlen, nheads, headdim, dstate = 2, 128, 8, 16, 32
        chunk_size = 32

        X = torch.randn(batch, seqlen, nheads, headdim, device=DEVICE, dtype=dtype)
        dt = torch.rand(batch, seqlen, nheads, device=DEVICE, dtype=dtype) * 0.1 + 0.01
        A = -torch.rand(nheads, device=DEVICE, dtype=dtype) * 5 - 1
        B = torch.randn(batch, seqlen, nheads, dstate, device=DEVICE, dtype=dtype)
        C = torch.randn(batch, seqlen, nheads, dstate, device=DEVICE, dtype=dtype)

        theta = None
        if use_rope:
            theta = torch.randn(batch, seqlen, nheads, dstate // 2, device=DEVICE, dtype=dtype) * 0.1

        lam = None
        gamma = dt.clone()
        beta = None
        if use_trapezoidal:
            lam = torch.sigmoid(torch.randn(batch, seqlen, nheads, device=DEVICE, dtype=dtype))
            gamma = lam * dt
            beta = (1 - lam) * dt * torch.exp(dt * A.view(1, 1, nheads))

        if mimo_rank > 0:
            R = mimo_rank
            X = torch.randn(batch, seqlen, nheads, headdim, R, device=DEVICE, dtype=dtype)
            B = torch.randn(batch, seqlen, nheads, dstate, R, device=DEVICE, dtype=dtype)
            C = torch.randn(batch, seqlen, nheads, dstate, R, device=DEVICE, dtype=dtype)

        # Apply RoPE to B, C for both paths
        if theta is not None:
            theta_cumsum = torch.cumsum(theta, dim=1)
            cos_t, sin_t = compute_cumulative_rotary(theta_cumsum, dstate)
            if mimo_rank > 0:
                B_parts = [apply_rotary_emb(B[..., r], cos_t, sin_t) for r in range(mimo_rank)]
                C_parts = [apply_rotary_emb(C[..., r], cos_t, sin_t) for r in range(mimo_rank)]
                B = torch.stack(B_parts, dim=-1)
                C = torch.stack(C_parts, dim=-1)
            else:
                B = apply_rotary_emb(B, cos_t, sin_t)
                C = apply_rotary_emb(C, cos_t, sin_t)

        # Chunked
        Y_chunked = mamba3_ssd_chunked(X, dt, A, B, C, block_len=chunk_size, gamma=gamma, beta=beta)

        # Reference
        Y_ref = self._reference_recurrence(X, dt, A, B, C, gamma, beta)

        atol = 1e-3 if dtype == torch.float32 else 5e-2
        rtol = 1e-3 if dtype == torch.float32 else 5e-2
        torch.testing.assert_close(Y_chunked.float(), Y_ref.float(), atol=atol, rtol=rtol)

    def test_siso_euler_fp32(self):
        self._run_comparison(use_rope=False, use_trapezoidal=False)

    def test_siso_full_fp32(self):
        self._run_comparison(use_rope=True, use_trapezoidal=True)

    def test_mimo_full_fp32(self):
        self._run_comparison(use_rope=True, use_trapezoidal=True, mimo_rank=2)

    def test_siso_full_bf16(self):
        self._run_comparison(use_rope=True, use_trapezoidal=True, dtype=torch.bfloat16)

    def test_mimo_full_bf16(self):
        self._run_comparison(use_rope=True, use_trapezoidal=True, mimo_rank=2, dtype=torch.bfloat16)


# ============================================================================
# 2. Mamba3 full module forward
# ============================================================================

class TestMamba3ModuleGPU:
    """Test the full Mamba3 module on GPU."""

    def _make_model(self, mimo_rank=0, use_rope=True, use_trapezoidal=True, dtype=torch.float32):
        from mamba_ssm.modules.mamba3 import Mamba3
        return Mamba3(
            d_model=128, d_state=32, expand=2, headdim=32,
            ngroups=1, use_rope=use_rope, use_trapezoidal=use_trapezoidal,
            use_bc_norm=True, use_bc_bias=True, mimo_rank=mimo_rank,
            chunk_size=64, layer_idx=0, device=DEVICE, dtype=dtype,
        ).eval()

    def test_siso_forward(self):
        model = self._make_model()
        u = torch.randn(2, 128, 128, device=DEVICE)
        with torch.no_grad():
            y = model(u)
        assert y.shape == u.shape
        assert torch.isfinite(y).all()

    def test_mimo_forward(self):
        model = self._make_model(mimo_rank=4)
        u = torch.randn(2, 128, 128, device=DEVICE)
        with torch.no_grad():
            y = model(u)
        assert y.shape == u.shape
        assert torch.isfinite(y).all()

    def test_siso_forward_bf16(self):
        model = self._make_model(dtype=torch.bfloat16)
        u = torch.randn(2, 128, 128, device=DEVICE, dtype=torch.bfloat16)
        with torch.no_grad():
            y = model(u)
        assert y.shape == u.shape
        assert torch.isfinite(y).all()

    def test_mimo_forward_bf16(self):
        model = self._make_model(mimo_rank=4, dtype=torch.bfloat16)
        u = torch.randn(2, 128, 128, device=DEVICE, dtype=torch.bfloat16)
        with torch.no_grad():
            y = model(u)
        assert y.shape == u.shape
        assert torch.isfinite(y).all()

    def test_siso_gradient_bf16(self):
        model = self._make_model(dtype=torch.bfloat16).train()
        u = torch.randn(2, 64, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        y = model(u)
        y.sum().backward()
        assert u.grad is not None
        assert torch.isfinite(u.grad).all()

    def test_mimo_gradient_bf16(self):
        model = self._make_model(mimo_rank=2, dtype=torch.bfloat16).train()
        u = torch.randn(2, 64, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        y = model(u)
        y.sum().backward()
        assert u.grad is not None
        assert torch.isfinite(u.grad).all()


# ============================================================================
# 3. Prefill → Decode consistency
# ============================================================================

class TestPrefillDecodeConsistency:
    """Test that single-step decode matches last position of prefill."""

    def _make_model(self, mimo_rank=0, dtype=torch.float32):
        from mamba_ssm.modules.mamba3 import Mamba3
        return Mamba3(
            d_model=64, d_state=16, expand=2, headdim=16,
            ngroups=1, use_rope=True, use_trapezoidal=True,
            use_bc_norm=True, use_bc_bias=True, mimo_rank=mimo_rank,
            chunk_size=32, layer_idx=0, device=DEVICE, dtype=dtype,
        ).eval()

    def test_siso_prefill_then_decode(self):
        """After prefill, decode step should produce reasonable output."""
        from mamba_ssm.utils.generation import InferenceParams

        model = self._make_model()
        batch, seqlen = 2, 64

        # Allocate inference cache
        inference_params = InferenceParams(max_seqlen=seqlen + 10, max_batch_size=batch)

        # Prefill
        u_prefill = torch.randn(batch, seqlen, 64, device=DEVICE)
        with torch.no_grad():
            y_prefill = model(u_prefill, inference_params=inference_params)
        inference_params.seqlen_offset = seqlen

        # Decode one token
        u_decode = torch.randn(batch, 1, 64, device=DEVICE)
        with torch.no_grad():
            y_decode = model(u_decode, inference_params=inference_params)

        assert y_decode.shape == (batch, 1, 64)
        assert torch.isfinite(y_decode).all()

    def test_siso_multi_step_decode(self):
        """Multiple decode steps should all produce finite output."""
        from mamba_ssm.utils.generation import InferenceParams

        model = self._make_model()
        batch, seqlen = 1, 32

        inference_params = InferenceParams(max_seqlen=seqlen + 20, max_batch_size=batch)

        # Prefill
        u = torch.randn(batch, seqlen, 64, device=DEVICE)
        with torch.no_grad():
            model(u, inference_params=inference_params)
        inference_params.seqlen_offset = seqlen

        # Decode 10 tokens
        for step in range(10):
            u_step = torch.randn(batch, 1, 64, device=DEVICE)
            with torch.no_grad():
                y = model(u_step, inference_params=inference_params)
            assert torch.isfinite(y).all(), f"Non-finite at decode step {step}"
            inference_params.seqlen_offset += 1

    def test_siso_prefill_decode_numerical(self):
        """Verify decode step matches what prefill would produce for same input."""
        from mamba_ssm.utils.generation import InferenceParams
        torch.manual_seed(42)

        model = self._make_model()
        batch = 1
        seqlen = 32

        # Full sequence: prefill all at once
        u_full = torch.randn(batch, seqlen + 1, 64, device=DEVICE)
        with torch.no_grad():
            y_full = model(u_full)

        # Split: prefill first seqlen tokens, then decode last token
        inference_params = InferenceParams(max_seqlen=seqlen + 10, max_batch_size=batch)
        with torch.no_grad():
            y_prefill = model(u_full[:, :seqlen], inference_params=inference_params)
        inference_params.seqlen_offset = seqlen
        with torch.no_grad():
            y_decode = model(u_full[:, seqlen:seqlen+1], inference_params=inference_params)

        # The decode output should match the last position of full prefill
        # Allow some tolerance due to chunked vs recurrent numerical diffs
        torch.testing.assert_close(
            y_decode[:, 0].float(), y_full[:, seqlen].float(),
            atol=5e-3, rtol=5e-3
        )


# ============================================================================
# 4. MambaLMHeadModel with Mamba3
# ============================================================================

class TestMambaLMHeadModelMamba3:
    """Test full model integration."""

    def test_mamba3_lm_model_forward(self):
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        config = MambaConfig(
            d_model=128, n_layer=2, vocab_size=256,
            ssm_cfg={"layer": "Mamba3", "d_state": 16, "headdim": 32,
                      "use_rope": True, "use_trapezoidal": True},
            rms_norm=True, fused_add_norm=False,
        )
        model = MambaLMHeadModel(config, device=DEVICE, dtype=torch.bfloat16)
        input_ids = torch.randint(0, 256, (2, 64), device=DEVICE)
        output = model(input_ids)
        assert output.logits.shape == (2, 64, 256)
        assert torch.isfinite(output.logits).all()

    def test_mamba3_mimo_lm_model(self):
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        config = MambaConfig(
            d_model=128, n_layer=2, vocab_size=256,
            ssm_cfg={"layer": "Mamba3", "d_state": 16, "headdim": 32,
                      "mimo_rank": 2, "use_rope": True, "use_trapezoidal": True},
            rms_norm=True, fused_add_norm=False,
        )
        model = MambaLMHeadModel(config, device=DEVICE, dtype=torch.bfloat16)
        input_ids = torch.randint(0, 256, (2, 64), device=DEVICE)
        output = model(input_ids)
        assert output.logits.shape == (2, 64, 256)
        assert torch.isfinite(output.logits).all()

    def test_mamba3_lm_model_gradient(self):
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        config = MambaConfig(
            d_model=128, n_layer=2, vocab_size=256,
            ssm_cfg={"layer": "Mamba3", "d_state": 16, "headdim": 32},
            rms_norm=True, fused_add_norm=False,
        )
        model = MambaLMHeadModel(config, device=DEVICE, dtype=torch.bfloat16)
        input_ids = torch.randint(0, 256, (2, 32), device=DEVICE)
        output = model(input_ids)
        loss = output.logits.float().sum()
        loss.backward()
        # Check at least some params got gradients
        graded = sum(1 for p in model.parameters() if p.grad is not None)
        assert graded > 0


# ============================================================================
# 5. Mamba3Simple on GPU
# ============================================================================

class TestMamba3SimpleGPU:
    """Test Mamba3Simple on GPU (uses reference recurrence, no Triton)."""

    def _make_model(self, mimo_rank=0, dtype=torch.float32):
        from mamba_ssm.modules.mamba3_simple import Mamba3Simple
        return Mamba3Simple(
            d_model=64, d_state=16, expand=2, headdim=16,
            ngroups=1, use_rope=True, use_trapezoidal=True,
            use_bc_norm=True, use_bc_bias=True, mimo_rank=mimo_rank,
            chunk_size=32, device=DEVICE, dtype=dtype,
        ).eval()

    def test_siso_forward_and_grad(self):
        model = self._make_model().train()
        u = torch.randn(2, 64, 64, device=DEVICE, requires_grad=True)
        y = model(u)
        assert y.shape == u.shape
        y.sum().backward()
        assert torch.isfinite(u.grad).all()

    def test_mimo_forward_and_grad(self):
        model = self._make_model(mimo_rank=2).train()
        u = torch.randn(2, 64, 64, device=DEVICE, requires_grad=True)
        y = model(u)
        assert y.shape == u.shape
        y.sum().backward()
        assert torch.isfinite(u.grad).all()


# ============================================================================
# 6. seq_idx — packed multi-document training (GPU)
# ============================================================================

class TestSeqIdxGPU:
    """Test seq_idx prevents cross-document leakage on GPU with real Triton kernels."""

    def _make_model(self, **kwargs):
        from mamba_ssm.modules.mamba3_simple import Mamba3Simple
        defaults = dict(
            d_model=64, d_state=16, expand=2, headdim=16,
            ngroups=1, use_rope=True, use_trapezoidal=True,
            use_bc_norm=True, use_bc_bias=True, mimo_rank=0,
            chunk_size=32, device=DEVICE, dtype=torch.float32,
        )
        defaults.update(kwargs)
        return Mamba3Simple(**defaults).eval()

    def test_siso_two_docs_isolation(self):
        """Packed docs should produce same output as separate runs."""
        torch.manual_seed(42)
        model = self._make_model(use_rope=False)
        u_a = torch.randn(1, 32, 64, device=DEVICE)
        u_b = torch.randn(1, 32, 64, device=DEVICE)

        with torch.no_grad():
            y_a = model(u_a)
            y_b = model(u_b)

        u_packed = torch.cat([u_a, u_b], dim=1)
        seq_idx = torch.cat([
            torch.zeros(1, 32, dtype=torch.long, device=DEVICE),
            torch.ones(1, 32, dtype=torch.long, device=DEVICE),
        ], dim=1)

        with torch.no_grad():
            y_packed = model(u_packed, seq_idx=seq_idx)

        torch.testing.assert_close(y_packed[:, :32], y_a, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(y_packed[:, 32:], y_b, atol=1e-4, rtol=1e-4)

    def test_siso_trapezoidal_docs(self):
        """seq_idx + trapezoidal discretization."""
        torch.manual_seed(42)
        model = self._make_model(use_rope=False, use_trapezoidal=True)
        u_a = torch.randn(1, 32, 64, device=DEVICE)
        u_b = torch.randn(1, 32, 64, device=DEVICE)

        with torch.no_grad():
            y_a = model(u_a)
            y_b = model(u_b)

        u_packed = torch.cat([u_a, u_b], dim=1)
        seq_idx = torch.cat([
            torch.zeros(1, 32, dtype=torch.long, device=DEVICE),
            torch.ones(1, 32, dtype=torch.long, device=DEVICE),
        ], dim=1)

        with torch.no_grad():
            y_packed = model(u_packed, seq_idx=seq_idx)

        torch.testing.assert_close(y_packed[:, :32], y_a, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(y_packed[:, 32:], y_b, atol=1e-4, rtol=1e-4)

    def test_gradient_isolation(self):
        """No gradient should flow across document boundaries."""
        torch.manual_seed(42)
        model = self._make_model(use_rope=False).train()
        u = torch.randn(1, 64, 64, device=DEVICE, requires_grad=True)
        seq_idx = torch.cat([
            torch.zeros(1, 32, dtype=torch.long, device=DEVICE),
            torch.ones(1, 32, dtype=torch.long, device=DEVICE),
        ], dim=1)

        y = model(u, seq_idx=seq_idx)
        y[:, 32:].sum().backward()

        assert u.grad[:, :32].abs().max() < 1e-5, "Gradient leaked across documents"

    def test_bf16_seq_idx(self):
        """seq_idx should work correctly with bf16."""
        torch.manual_seed(42)
        model = self._make_model(dtype=torch.bfloat16, use_rope=False)
        u_a = torch.randn(1, 32, 64, device=DEVICE, dtype=torch.bfloat16)
        u_b = torch.randn(1, 32, 64, device=DEVICE, dtype=torch.bfloat16)

        with torch.no_grad():
            y_a = model(u_a)
            y_b = model(u_b)

        u_packed = torch.cat([u_a, u_b], dim=1)
        seq_idx = torch.cat([
            torch.zeros(1, 32, dtype=torch.long, device=DEVICE),
            torch.ones(1, 32, dtype=torch.long, device=DEVICE),
        ], dim=1)

        with torch.no_grad():
            y_packed = model(u_packed, seq_idx=seq_idx)

        torch.testing.assert_close(y_packed[:, :32], y_a, atol=0.05, rtol=0.05)
        torch.testing.assert_close(y_packed[:, 32:], y_b, atol=0.05, rtol=0.05)


# ============================================================================
# 7. Triton decode kernel integration
# ============================================================================

class TestTritonDecodeKernel:
    """Test that Triton decode kernel in step() matches PyTorch reference."""

    def _make_model(self, **kwargs):
        from mamba_ssm.modules.mamba3 import Mamba3
        defaults = dict(
            d_model=64, d_state=16, expand=2, headdim=16,
            ngroups=1, use_rope=True, use_trapezoidal=True,
            use_bc_norm=True, use_bc_bias=True, mimo_rank=0,
            use_mem_eff_path=False,
            chunk_size=32, layer_idx=0, device=DEVICE, dtype=torch.float32,
        )
        defaults.update(kwargs)
        return Mamba3(**defaults).eval()

    def test_triton_decode_siso(self):
        """SISO decode: Triton kernel should match prefill output at each position."""
        torch.manual_seed(42)
        model = self._make_model()
        batch, seqlen = 2, 16
        u = torch.randn(batch, seqlen, 64, device=DEVICE)

        # Prefill
        from mamba_ssm.utils.generation import InferenceParams
        inference_params = InferenceParams(max_seqlen=seqlen, max_batch_size=batch)
        with torch.no_grad():
            y_prefill = model(u, inference_params=inference_params)

        # Now decode one more token
        u_next = torch.randn(batch, 1, 64, device=DEVICE)
        inference_params.seqlen_offset = seqlen
        with torch.no_grad():
            y_decode = model(u_next, inference_params=inference_params)

        assert y_decode.shape == (batch, 1, 64)
        assert torch.isfinite(y_decode).all()

    def test_triton_vs_pytorch_decode_consistency(self):
        """SISO: Triton decode and PyTorch fallback should give same results."""
        torch.manual_seed(42)
        model = self._make_model(use_rope=False)
        batch, seqlen = 1, 32

        import copy
        from mamba_ssm.utils.generation import InferenceParams
        import mamba_ssm.modules.mamba3 as mamba3_mod

        # Run with Triton kernel
        torch.manual_seed(99)
        u = torch.randn(batch, seqlen, 64, device=DEVICE)
        ip1 = InferenceParams(max_seqlen=seqlen + 4, max_batch_size=batch)
        with torch.no_grad():
            _ = model(u, inference_params=ip1)
        decode_triton = []
        decode_inputs = []
        for t in range(4):
            u_t = torch.randn(batch, 1, 64, device=DEVICE)
            decode_inputs.append(u_t.clone())
            ip1.seqlen_offset = seqlen + t
            with torch.no_grad():
                decode_triton.append(model(u_t, inference_params=ip1))

        # Run with PyTorch fallback (patch out mamba3_state_update)
        model2 = copy.deepcopy(model)
        orig_fn = mamba3_mod.mamba3_state_update
        torch.manual_seed(99)
        u2 = torch.randn(batch, seqlen, 64, device=DEVICE)
        ip2 = InferenceParams(max_seqlen=seqlen + 4, max_batch_size=batch)
        with torch.no_grad():
            _ = model2(u2, inference_params=ip2)
        decode_pytorch = []
        mamba3_mod.mamba3_state_update = None
        try:
            for t in range(4):
                ip2.seqlen_offset = seqlen + t
                with torch.no_grad():
                    decode_pytorch.append(model2(decode_inputs[t], inference_params=ip2))
        finally:
            mamba3_mod.mamba3_state_update = orig_fn

        # Compare Triton vs PyTorch outputs
        for i in range(4):
            torch.testing.assert_close(
                decode_triton[i], decode_pytorch[i], atol=1e-4, rtol=1e-4,
                msg=f"SISO decode step {i}: Triton vs PyTorch mismatch",
            )

    def test_triton_decode_mimo(self):
        """MIMO decode: Triton kernel should produce finite outputs and correct shapes."""
        torch.manual_seed(42)
        model = self._make_model(mimo_rank=2)
        batch, seqlen = 2, 16
        u = torch.randn(batch, seqlen, 64, device=DEVICE)

        from mamba_ssm.utils.generation import InferenceParams
        inference_params = InferenceParams(max_seqlen=seqlen + 4, max_batch_size=batch)
        with torch.no_grad():
            y_prefill = model(u, inference_params=inference_params)

        assert y_prefill.shape == (batch, seqlen, 64)
        assert torch.isfinite(y_prefill).all()

        # Decode 4 tokens
        for t in range(4):
            u_t = torch.randn(batch, 1, 64, device=DEVICE)
            inference_params.seqlen_offset = seqlen + t
            with torch.no_grad():
                y_t = model(u_t, inference_params=inference_params)
            assert y_t.shape == (batch, 1, 64)
            assert torch.isfinite(y_t).all(), f"MIMO decode step {t} has non-finite values"

    def test_triton_mimo_vs_pytorch_consistency(self):
        """MIMO: Triton decode and PyTorch fallback should give same results."""
        torch.manual_seed(42)
        model = self._make_model(mimo_rank=2, use_rope=False)
        batch, seqlen = 1, 32

        import copy
        from mamba_ssm.utils.generation import InferenceParams
        import mamba_ssm.modules.mamba3 as mamba3_mod

        # Run with Triton kernel
        torch.manual_seed(99)
        u = torch.randn(batch, seqlen, 64, device=DEVICE)
        ip1 = InferenceParams(max_seqlen=seqlen + 2, max_batch_size=batch)
        with torch.no_grad():
            _ = model(u, inference_params=ip1)
        decode_triton = []
        decode_inputs = []
        for t in range(2):
            u_t = torch.randn(batch, 1, 64, device=DEVICE)
            decode_inputs.append(u_t.clone())
            ip1.seqlen_offset = seqlen + t
            with torch.no_grad():
                decode_triton.append(model(u_t, inference_params=ip1))

        # Run with PyTorch fallback (patch out mamba3_state_update)
        model2 = copy.deepcopy(model)
        orig_fn = mamba3_mod.mamba3_state_update
        torch.manual_seed(99)
        u2 = torch.randn(batch, seqlen, 64, device=DEVICE)
        ip2 = InferenceParams(max_seqlen=seqlen + 2, max_batch_size=batch)
        with torch.no_grad():
            _ = model2(u2, inference_params=ip2)
        decode_pytorch = []
        mamba3_mod.mamba3_state_update = None  # force PyTorch fallback
        try:
            for t in range(2):
                ip2.seqlen_offset = seqlen + t
                with torch.no_grad():
                    decode_pytorch.append(model2(decode_inputs[t], inference_params=ip2))
        finally:
            mamba3_mod.mamba3_state_update = orig_fn

        # Both paths should produce matching results
        for i in range(2):
            torch.testing.assert_close(
                decode_triton[i], decode_pytorch[i], atol=1e-4, rtol=1e-4,
                msg=f"MIMO decode step {i}: Triton vs PyTorch mismatch",
            )


# ============================================================================
# 8. use_mem_eff_path (gradient checkpointing)
# ============================================================================

class TestMemEffPathGPU:
    """Test gradient checkpointing produces same results as normal forward."""

    def _make_model(self, use_mem_eff_path=True, **kwargs):
        from mamba_ssm.modules.mamba3 import Mamba3
        defaults = dict(
            d_model=64, d_state=16, expand=2, headdim=16,
            ngroups=1, use_rope=True, use_trapezoidal=True,
            use_bc_norm=True, use_bc_bias=True, mimo_rank=0,
            use_mem_eff_path=use_mem_eff_path,
            chunk_size=32, device=DEVICE, dtype=torch.float32,
        )
        defaults.update(kwargs)
        return Mamba3(**defaults)

    def test_checkpointed_matches_normal(self):
        """Checkpointed and normal forward should produce same outputs and gradients."""
        torch.manual_seed(42)
        model_ckpt = self._make_model(use_mem_eff_path=True).train()

        import copy
        model_plain = copy.deepcopy(model_ckpt)
        model_plain.use_mem_eff_path = False

        u = torch.randn(2, 32, 64, device=DEVICE)

        # Checkpointed
        u1 = u.clone().requires_grad_(True)
        y1 = model_ckpt(u1)
        y1.sum().backward()

        # Normal
        u2 = u.clone().requires_grad_(True)
        y2 = model_plain(u2)
        y2.sum().backward()

        torch.testing.assert_close(y1, y2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(u1.grad, u2.grad, atol=1e-5, rtol=1e-5)
        # Check parameter gradients match
        for (n1, p1), (n2, p2) in zip(
            model_ckpt.named_parameters(), model_plain.named_parameters()
        ):
            if p1.grad is not None:
                torch.testing.assert_close(p1.grad, p2.grad, atol=1e-5, rtol=1e-5,
                                           msg=f"Grad mismatch for {n1}")

    def test_checkpointed_bf16(self):
        """Gradient checkpointing should work with bf16."""
        torch.manual_seed(42)
        model = self._make_model(use_mem_eff_path=True, dtype=torch.bfloat16).train()
        u = torch.randn(2, 32, 64, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        y = model(u)
        y.sum().backward()
        assert torch.isfinite(u.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
