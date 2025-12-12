# Copyright (c) 2024, Tri Dao, Albert Gu.

import os

import pytest
import torch

from mamba_ssm.utils.determinism import set_deterministic_mode

MODEL_PRESETS = {
    "small": {"d_model": 256, "headdim": 64, "d_state": 64, "ngroups": 1},
    "nemotronh-56b": {"d_model": 8192, "headdim": 64, "d_state": 256, "ngroups": 8},
}


def _configure(deterministic: bool) -> None:
    os.environ["MAMBA_DETERMINISTIC"] = "1" if deterministic else "0"
    os.environ["CAUSAL_CONV1D_DETERMINISTIC"] = "1" if deterministic else "0"
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_deterministic_mode(deterministic)


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _run_mamba2_backward(*, cfg: dict, seed: int) -> dict[str, torch.Tensor]:
    from mamba_ssm.modules.mamba2 import Mamba2

    _set_seeds(seed)
    model = Mamba2(
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        d_conv=4,
        expand=2,
        headdim=cfg["headdim"],
        ngroups=cfg["ngroups"],
        use_mem_eff_path=True,
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = torch.randn(cfg["batch"], cfg["seqlen"], cfg["d_model"], device="cuda", dtype=torch.bfloat16)
    out = model(x)
    out.sum().backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deterministic_backend_reproducible_small():
    cfg = {**MODEL_PRESETS["small"], "batch": 2, "seqlen": 2048}
    _configure(True)
    grads0 = _run_mamba2_backward(cfg=cfg, seed=123)
    grads1 = _run_mamba2_backward(cfg=cfg, seed=123)
    for k in grads0:
        assert _max_abs_diff(grads0[k], grads1[k]) == 0.0


def main() -> int:
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(description="Mamba2 determinism check (manual)")
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS.keys()), default="small")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--mode", choices=["det", "default", "both"], default="det")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    if args.mode == "both":
        # Run each mode in a fresh process to avoid environment / library init leakage.
        base = [
            sys.executable, __file__,
            "--preset", args.preset,
            "--batch", str(args.batch),
            "--seqlen", str(args.seqlen),
            "--runs", str(args.runs),
        ]
        subprocess.check_call(base + ["--mode", "default"])
        subprocess.check_call(base + ["--mode", "det"])
        return 0

    deterministic = args.mode == "det"
    _configure(deterministic)

    cfg = {**MODEL_PRESETS[args.preset], "batch": args.batch, "seqlen": args.seqlen}
    grads = [_run_mamba2_backward(cfg=cfg, seed=123) for _ in range(args.runs)]

    max_diff = 0.0
    max_name = None
    for name in grads[0]:
        for i in range(1, args.runs):
            diff = _max_abs_diff(grads[0][name], grads[i][name])
            if diff > max_diff:
                max_diff = diff
                max_name = name

    print(f"mode={args.mode} max_grad_diff={max_diff:.6e} max_param={max_name}")
    if args.mode == "default" and max_diff == 0.0:
        print(
            "note: default path can appear deterministic for some small configs; "
            "try --preset nemotronh-56b --seqlen 16384 (or larger batch/seqlen) to reproduce nondeterminism."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


