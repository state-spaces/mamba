# Copyright (c) 2024, Tri Dao, Albert Gu.

import os

import pytest
import torch


def _set_deterministic(enabled: bool) -> None:
    torch.use_deterministic_algorithms(enabled)
    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _make_inputs(*, seed: int, headdim: int, dstate: int) -> dict[str, torch.Tensor]:
    """Inputs for determinism-enabled backward kernels."""
    import math

    _set_seeds(seed)
    device = "cuda"

    batch = 2
    seqlen = 2048
    nheads = 8
    ngroups = 1
    chunk_size = 256
    nchunks = math.ceil(seqlen / chunk_size)

    x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    dout = torch.randn_like(x)
    dt = torch.randn(batch, nheads, nchunks, chunk_size, device=device, dtype=torch.float32)
    dA_cumsum = torch.randn_like(dt)
    cb = torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size, device=device, dtype=torch.bfloat16)

    B = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=torch.bfloat16).contiguous()
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=torch.bfloat16).contiguous()
    dstates = torch.randn(batch, nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32)
    prev_states = torch.randn_like(dstates)

    ddA = torch.randn(batch, nheads, nchunks, chunk_size, device=device, dtype=torch.float32)
    ddt_out = torch.randn_like(ddA)
    dt_raw = torch.randn(batch, seqlen, nheads, device=device, dtype=torch.bfloat16)
    A = (torch.randn(nheads, device=device, dtype=torch.float32) * -1.0).contiguous()
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32).contiguous()
    D = torch.randn(nheads, device=device, dtype=torch.float32)

    return {
        "x": x,
        "dout": dout,
        "dt": dt,
        "dA_cumsum": dA_cumsum,
        "cb": cb,
        "B": B,
        "C": C,
        "dstates": dstates,
        "prev_states": prev_states,
        "ddA": ddA,
        "ddt_out": ddt_out,
        "dt_raw": dt_raw,
        "A": A,
        "dt_bias": dt_bias,
        "D": D,
    }


def _run_case_outputs(
    *, case: str, deterministic: bool, seed: int, headdim: int = 64,
) -> dict[str, torch.Tensor]:
    """Run one kernel wrapper and return named outputs (as fp32)."""
    _set_deterministic(deterministic)
    t = _make_inputs(seed=seed, headdim=headdim, dstate=64)

    if case == "chunk_scan_bwd_dx":
        from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dx
        dx, ddt = _chunk_scan_bwd_dx(t["cb"], t["x"], t["dt"], t["dA_cumsum"], t["dout"])
        out = {"dx": dx, "ddt": ddt}
    elif case == "chunk_scan_bwd_dC":
        from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dC
        dC, ddA_prev = _chunk_scan_bwd_dC(t["prev_states"], t["dA_cumsum"], t["dout"], C=t["C"], ngroups=1)
        out = {"dC": dC, "ddA_cumsum_prev": ddA_prev}
    elif case == "chunk_state_bwd_dx":
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_bwd_dx
        dx, ddt, ddA = _chunk_state_bwd_dx(t["B"], t["x"], t["dt"], t["dA_cumsum"], t["dstates"])
        out = {"dx": dx, "ddt": ddt, "ddA_cumsum": ddA}
    elif case == "chunk_state_bwd_db":
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_bwd_db
        dB, ddA = _chunk_state_bwd_db(t["x"], t["dt"], t["dA_cumsum"], t["dstates"], B=t["B"], ngroups=1)
        out = {"dB": dB, "ddA_cumsum": ddA}
    elif case == "chunk_state_bwd_ddAcs_stable":
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_bwd_ddAcs_stable
        ddA = _chunk_state_bwd_ddAcs_stable(t["B"], t["x"], t["dt"], t["dA_cumsum"], t["dstates"])
        out = {"ddA_cumsum": ddA}
    elif case == "chunk_cumsum_bwd":
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_bwd
        ddt, dA, ddt_bias = _chunk_cumsum_bwd(t["ddA"], t["ddt_out"], t["dt_raw"], t["A"], dt_bias=t["dt_bias"], dt_softplus=True)
        out = {"ddt": ddt, "dA": dA, "ddt_bias": ddt_bias}
    elif case == "combined_bwd_dx":
        from mamba_ssm.ops.triton.ssd_combined import _chunk_scan_chunk_state_bwd_dx
        dx, ddt, dD = _chunk_scan_chunk_state_bwd_dx(t["x"], t["dt"], t["dA_cumsum"], t["B"], t["cb"], t["dout"], t["dstates"], D=t["D"])
        out = {"dx": dx, "ddt": ddt, "dD": dD}
    else:
        raise AssertionError(f"Unknown case: {case}")

    torch.cuda.synchronize()
    return {k: v.detach().clone().float() for k, v in out.items() if v is not None}


_CASES = [
    "chunk_scan_bwd_dx",
    "chunk_scan_bwd_dC",
    "chunk_state_bwd_dx",
    "chunk_state_bwd_db",
    "chunk_state_bwd_ddAcs_stable",
    "chunk_cumsum_bwd",
    "combined_bwd_dx",
]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("headdim", [32, 64])
@pytest.mark.parametrize("case", _CASES)
def test_all_determinism_enabled_kernels_reproducible(case: str, headdim: int):
    runs = 5
    outs = [_run_case_outputs(case=case, deterministic=True, seed=123, headdim=headdim) for _ in range(runs)]
    ref = outs[0]
    for i in range(1, runs):
        for k in ref:
            assert _max_abs_diff(ref[k], outs[i][k]) == 0.0, f"{case} output {k} differs (headdim={headdim})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_default_mode_is_not_reproducible():
    from mamba_ssm.modules.mamba2 import Mamba2

    device = "cuda"
    dtype = torch.bfloat16
    seed = 123
    runs = 20
    batch = 4
    seqlen = 4096

    _set_seeds(seed)
    model = Mamba2(
        d_model=256,
        d_state=64,
        headdim=64,
        expand=2,
        d_conv=4,
        chunk_size=256,
        use_mem_eff_path=True,
        device=device,
        dtype=dtype,
    ).train()
    x_data = torch.randn(batch, seqlen, model.d_model, device=device, dtype=dtype)

    def _run() -> dict[str, torch.Tensor]:
        _set_deterministic(False)
        model.zero_grad(set_to_none=True)
        x = x_data.clone().requires_grad_(True)
        y = model(x)
        (y.float().square().mean()).backward()
        torch.cuda.synchronize()
        grads = {"input": x.grad.detach().float().clone()}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.detach().float().clone()
        return grads

    _run()  # warmup
    ref = _run()
    observed_diff = False
    for _ in range(runs - 1):
        g = _run()
        for k in ref:
            if _max_abs_diff(ref[k], g[k]) != 0.0:
                observed_diff = True
                break
        if observed_diff:
            break

    if not observed_diff:
        pytest.xfail(
            f"Did not observe nondeterminism in default mode after {runs} runs. "
            "This GPU may have deterministic atomic behavior at these shapes."
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("headdim", [32, 64])
@pytest.mark.parametrize("case", _CASES)
def test_all_determinism_enabled_kernels_close_to_default(case: str, headdim: int):
    atol = 1e-2
    rtol = atol
    det = _run_case_outputs(case=case, deterministic=True, seed=123, headdim=headdim)
    for _ in range(3):
        default = _run_case_outputs(case=case, deterministic=False, seed=123, headdim=headdim)
        for k in det:
            assert torch.allclose(default[k], det[k], atol=atol, rtol=rtol), f"{case} output {k} not close (headdim={headdim})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("headdim", [32, 128])
def test_mamba2_fwd_bwd_deterministic_mode_is_reproducible(headdim: int):
    from mamba_ssm.modules.mamba2 import Mamba2

    device = "cuda"
    dtype = torch.bfloat16
    seed = 123
    runs = 5
    batch = 2
    seqlen = 2048

    _set_seeds(seed)
    _set_deterministic(True)

    model = Mamba2(
        d_model=headdim,
        d_state=16,
        headdim=headdim,
        expand=2,
        d_conv=4,
        chunk_size=16,
        use_mem_eff_path=True,
        device=device,
        dtype=dtype,
    ).train()
    x_data = torch.randn(batch, seqlen, model.d_model, device=device, dtype=dtype)

    def _run() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad(set_to_none=True)
        x = x_data.clone().requires_grad_(True)
        y = model(x)
        (y.float().square().mean()).backward()
        torch.cuda.synchronize()
        grads: dict[str, torch.Tensor] = {"input": x.grad.detach().float().clone()}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.detach().float().clone()
        return y.detach().float().clone(), grads

    _run()  # warmup
    y0, g0 = _run()
    for _ in range(runs - 1):
        y, g = _run()
        assert _max_abs_diff(y0, y) == 0.0
        assert g.keys() == g0.keys()
        for k in g0:
            assert _max_abs_diff(g0[k], g[k]) == 0.0, f"Mamba2 grad {k} differs (headdim={headdim})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("headdim", [32, 64])
def test_mamba2_fwd_bwd_deterministic_close_to_default(headdim: int):
    from mamba_ssm.modules.mamba2 import Mamba2

    device = "cuda"
    dtype = torch.bfloat16
    seed = 123
    batch = 2
    seqlen = 2048
    atol = 1e-2
    rtol = 1e-2

    def _run(deterministic: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        _set_seeds(seed)
        model = Mamba2(
            d_model=headdim * 4,
            d_state=32,
            headdim=headdim,
            expand=2,
            d_conv=4,
            chunk_size=64,
            use_mem_eff_path=True,
            device=device,
            dtype=dtype,
        ).train()
        x = torch.randn(batch, seqlen, model.d_model, device=device, dtype=dtype).requires_grad_(True)
        y = model(x)
        (y.float().square().mean()).backward()
        torch.cuda.synchronize()
        grads: dict[str, torch.Tensor] = {"input": x.grad.detach().float().clone()}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.detach().float().clone()
        return y.detach().float().clone(), grads

    _run(False)  # warmup
    y_default, g_default = _run(False)
    y_det, g_det = _run(True)

    assert torch.allclose(y_default, y_det, atol=atol, rtol=rtol), "Mamba2 output differs"
    for k in g_default:
        assert torch.allclose(g_default[k], g_det[k], atol=atol, rtol=rtol), f"Mamba2 grad {k} not close (headdim={headdim})"
