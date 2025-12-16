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


def _make_inputs(*, seed: int, headdim: int, dstate: int, scale: float = 1.0) -> dict[str, torch.Tensor]:
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

    x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16) * scale
    dout = torch.randn_like(x) * scale
    dt = torch.randn(batch, nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * scale
    dA_cumsum = torch.randn_like(dt) * scale
    cb = torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size, device=device, dtype=torch.bfloat16) * scale

    B = (torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=torch.bfloat16) * scale).contiguous()
    C = (torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=torch.bfloat16) * scale).contiguous()
    dstates = torch.randn(batch, nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32) * scale
    prev_states = torch.randn_like(dstates) * scale

    ddA = torch.randn(batch, nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * scale
    ddt_out = torch.randn_like(ddA) * scale
    dt_raw = torch.randn(batch, seqlen, nheads, device=device, dtype=torch.bfloat16) * scale
    A = (torch.randn(nheads, device=device, dtype=torch.float32) * -1.0).contiguous()
    dt_bias = (torch.randn(nheads, device=device, dtype=torch.float32) * scale).contiguous()

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
    }


def _run_case_outputs(*, case: str, deterministic: bool, seed: int, scale: float = 1.0) -> dict[str, torch.Tensor]:
    """Run one kernel wrapper and return named outputs (as fp32)."""
    _set_deterministic(deterministic)
    if case in ("chunk_scan_bwd_dC", "chunk_state_bwd_db"):
        headdim = 256
    else:
        headdim = 384
    dstate = 384
    t = _make_inputs(seed=seed, headdim=headdim, dstate=dstate, scale=scale)

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
        dx, ddt, _ = _chunk_scan_chunk_state_bwd_dx(t["x"], t["dt"], t["dA_cumsum"], t["B"], t["cb"], t["dout"], t["dstates"])
        out = {"dx": dx, "ddt": ddt}
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
@pytest.mark.parametrize("case", _CASES)
def test_all_determinism_enabled_kernels_reproducible(case: str):
    runs = 5
    outs = [_run_case_outputs(case=case, deterministic=True, seed=123) for _ in range(runs)]
    ref = outs[0]
    for i in range(1, runs):
        for k in ref:
            assert _max_abs_diff(ref[k], outs[i][k]) == 0.0, f"{case} output {k} differs"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_default_mode_is_not_reproducible_for_atomics_path():
    runs = 50
    outs = [_run_case_outputs(case="chunk_scan_bwd_dx", deterministic=False, seed=123) for _ in range(runs)]
    ref = outs[0]["ddt"]
    observed = any(_max_abs_diff(ref, outs[i]["ddt"]) != 0.0 for i in range(1, runs))
    if not observed:
        pytest.xfail(
            "Did not observe nondeterminism in default mode after "
            f"{runs} runs. If you expect nondeterminism on this GPU, increase "
            "the run count and/or adjust shapes to increase atomic contention."
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", _CASES)
def test_all_determinism_enabled_kernels_close_to_default(case: str):
    scale = 1.0
    atol = 1e-2
    rtol = atol
    det = _run_case_outputs(case=case, deterministic=True, seed=123, scale=scale)
    for _ in range(3):
        default = _run_case_outputs(case=case, deterministic=False, seed=123, scale=scale)
        for k in det:
            assert torch.allclose(default[k], det[k], atol=atol, rtol=rtol), f"{case} output {k} not close"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mamba2_fwd_bwd_deterministic_mode_is_reproducible():
    from mamba_ssm.modules.mamba2 import Mamba2

    device = "cuda"
    dtype = torch.bfloat16
    seed = 123
    runs = 5
    scale = 1.0
    batch = 2
    seqlen = 2048

    _set_seeds(seed)
    _set_deterministic(True)

    model = Mamba2(
        d_model=256,
        d_state=384,
        headdim=128,
        expand=2,
        d_conv=4,
        chunk_size=256,
        use_mem_eff_path=True,
        device=device,
        dtype=dtype,
    ).train()
    x_data = torch.randn(batch, seqlen, model.d_model, device=device, dtype=dtype) * scale

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
            assert _max_abs_diff(g0[k], g[k]) == 0.0, f"Mamba2 grad {k} differs"
