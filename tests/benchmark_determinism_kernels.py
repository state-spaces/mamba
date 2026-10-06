#!/usr/bin/env python
# Copyright (c) 2024, Tri Dao, Albert Gu.

import gc
import math

import torch
from triton.testing import do_bench

from mamba_ssm.utils.determinism import set_deterministic_mode

MODEL_PRESETS = {
    "small": {"nheads": 32, "headdim": 64, "dstate": 64, "ngroups": 1},
    "nemotronh-56b": {"nheads": 256, "headdim": 64, "dstate": 256, "ngroups": 8},
}


def _reset_peak_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _peak_memory_mb(fn, *, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    _reset_peak_memory()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def make_tensors(*, batch: int, seqlen: int, nheads: int, headdim: int, dstate: int, ngroups: int, chunk_size: int,
                 dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    device = "cuda"
    nchunks = math.ceil(seqlen / chunk_size)
    return {
        "x": torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype),
        "B": torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype),
        "C": torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype),
        "dt": torch.randn(batch, nheads, nchunks, chunk_size, device=device, dtype=torch.float32),
        "dA_cumsum": torch.randn(batch, nheads, nchunks, chunk_size, device=device, dtype=torch.float32),
        "dstates": torch.randn(batch, nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32),
        "dout": torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype),
        "ddA": torch.randn(batch, nheads, nchunks, chunk_size, device=device, dtype=torch.float32),
        "ddt_out": torch.randn(batch, nheads, nchunks, chunk_size, device=device, dtype=torch.float32),
        "dt_raw": torch.randn(batch, seqlen, nheads, device=device, dtype=dtype),
        "A": torch.randn(nheads, device=device, dtype=torch.float32) * -1,
        "dt_bias": torch.randn(nheads, device=device, dtype=torch.float32),
        "prev_states": torch.randn(batch, nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32),
        "cb": torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size, device=device, dtype=dtype),
    }


def get_benchmarks(t: dict[str, torch.Tensor], *, ngroups: int):
    from mamba_ssm.ops.triton.ssd_chunk_state import (
        _chunk_cumsum_bwd,
        _chunk_state_bwd_db,
        _chunk_state_bwd_ddAcs_stable,
        _chunk_state_bwd_dx,
    )
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dC, _chunk_scan_bwd_dx
    from mamba_ssm.ops.triton.ssd_combined import _chunk_scan_chunk_state_bwd_dx

    x = t["x"].contiguous()
    B = t["B"].contiguous()
    C = t["C"].contiguous()
    dout = t["dout"].contiguous()
    dstates = t["dstates"].contiguous()

    return [
        ("chunk_cumsum_bwd", lambda: _chunk_cumsum_bwd(t["ddA"], t["ddt_out"], t["dt_raw"], t["A"], dt_bias=t["dt_bias"], dt_softplus=True)),
        ("chunk_state_bwd_dx", lambda: _chunk_state_bwd_dx(B, x, t["dt"], t["dA_cumsum"], dstates)),
        ("chunk_state_bwd_db", lambda: _chunk_state_bwd_db(x, t["dt"], t["dA_cumsum"], dstates, B=B, ngroups=ngroups)),
        ("chunk_state_bwd_ddAcs", lambda: _chunk_state_bwd_ddAcs_stable(B, x, t["dt"], t["dA_cumsum"], dstates)),
        ("chunk_scan_bwd_dC", lambda: _chunk_scan_bwd_dC(t["prev_states"], t["dA_cumsum"], dout, C=C, ngroups=ngroups)),
        ("chunk_scan_bwd_dx", lambda: _chunk_scan_bwd_dx(t["cb"], x, t["dt"], t["dA_cumsum"], dout)),
        ("combined_bwd_dx", lambda: _chunk_scan_chunk_state_bwd_dx(x, t["dt"], t["dA_cumsum"], B, t["cb"], dout, dstates)),
    ]


def _run_one(fn, *, deterministic: bool, warmup: int, rep: int):
    set_deterministic_mode(deterministic)
    ms = do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    peak_mb = _peak_memory_mb(fn, warmup=1)
    return ms, peak_mb


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark determinism overhead for key Triton backward kernels")
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS.keys()), default="small")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    p = MODEL_PRESETS[args.preset]
    tensors = make_tensors(
        batch=args.batch,
        seqlen=args.seqlen,
        nheads=p["nheads"],
        headdim=p["headdim"],
        dstate=p["dstate"],
        ngroups=p["ngroups"],
        chunk_size=args.chunk_size,
    )
    benches = get_benchmarks(tensors, ngroups=p["ngroups"])

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"preset={args.preset} batch={args.batch} seqlen={args.seqlen} chunk_size={args.chunk_size}")
    print(f"{'kernel':<20} {'ms':>9} {'det_ms':>9} {'ms_%':>6} {'MB':>9} {'det_MB':>9} {'MB_%':>6}")

    rows = []
    try:
        for name, fn in benches:
            ms, mb = _run_one(fn, deterministic=False, warmup=args.warmup, rep=args.rep)
            det_ms, det_mb = _run_one(fn, deterministic=True, warmup=args.warmup, rep=args.rep)
            ms_pct = (det_ms / ms - 1.0) * 100.0
            mb_pct = (det_mb / mb - 1.0) * 100.0 if mb else 0.0
            rows.append((name, ms, det_ms, ms_pct, mb, det_mb, mb_pct))
            print(f"{name:<20} {ms:>9.3f} {det_ms:>9.3f} {ms_pct:>+6.0f}% {mb:>9.1f} {det_mb:>9.1f} {mb_pct:>+6.0f}%")
    finally:
        set_deterministic_mode(None)

    total_ms = sum(r[1] for r in rows)
    total_det_ms = sum(r[2] for r in rows)
    max_mb = max(r[4] for r in rows) if rows else 0.0
    max_det_mb = max(r[5] for r in rows) if rows else 0.0
    total_pct = (total_det_ms / total_ms - 1.0) * 100.0 if total_ms else 0.0
    max_mb_pct = (max_det_mb / max_mb - 1.0) * 100.0 if max_mb else 0.0
    print(f"{'TOTAL/MAX':<20} {total_ms:>9.3f} {total_det_ms:>9.3f} {total_pct:>+6.0f}% {max_mb:>9.1f} {max_det_mb:>9.1f} {max_mb_pct:>+6.0f}%")


if __name__ == "__main__":
    main()


