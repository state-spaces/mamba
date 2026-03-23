#!/usr/bin/env python3
"""Benchmark forward/backward speed: Mamba1 vs Mamba2 vs Mamba3.

Compares all three SSM generations on the same workload (same d_model,
sequence length, batch size) measuring forward/backward speed and VRAM.

Usage:
    python benchmarks/benchmark_speed_mamba123.py
"""

import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import torch

matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})

GB = 1024**3
OUTPUT_PATH = Path(__file__).parent / "speed_mamba123.png"

# Common benchmark parameters
D_MODEL = 768
BATCH = 2
SEQ_LEN = 2048
DTYPE = torch.bfloat16
N_WARMUP = 2
N_RUNS = 5


def bench_module(name, create_fn):
    """Benchmark a single Mamba module, returning timing and VRAM stats."""
    print(f"  Benchmarking {name}...")
    model = create_fn().to("cuda")
    params = sum(p.numel() for p in model.parameters()) / 1e6

    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, dtype=DTYPE, device="cuda")

    # Warm-up
    for _ in range(N_WARMUP):
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()

    # Forward
    torch.cuda.reset_peak_memory_stats()
    fwd_times = []
    for _ in range(N_RUNS):
        x_fwd = torch.randn(BATCH, SEQ_LEN, D_MODEL, dtype=DTYPE, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model(x_fwd)
        torch.cuda.synchronize()
        fwd_times.append((time.time() - t0) * 1000)
    fwd_ms = sum(fwd_times) / len(fwd_times)
    fwd_vram = torch.cuda.max_memory_allocated() / GB

    # Backward
    torch.cuda.reset_peak_memory_stats()
    bwd_times = []
    for _ in range(N_RUNS):
        x_bwd = torch.randn(BATCH, SEQ_LEN, D_MODEL, dtype=DTYPE, device="cuda", requires_grad=True)
        y = model(x_bwd)
        loss = y.sum()
        torch.cuda.synchronize()
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bwd_times.append((time.time() - t0) * 1000)
    bwd_ms = sum(bwd_times) / len(bwd_times)
    bwd_vram = torch.cuda.max_memory_allocated() / GB

    del model, x
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "params_m": params,
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "fwd_vram_gb": fwd_vram,
        "bwd_vram_gb": bwd_vram,
    }


def collect_results():
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.modules.mamba3 import Mamba3

    models = {
        "Mamba1": lambda: Mamba(d_model=D_MODEL, d_state=16, dtype=DTYPE),
        "Mamba2": lambda: Mamba2(d_model=D_MODEL, d_state=128, headdim=64, dtype=DTYPE),
        "Mamba3\n(SISO)": lambda: Mamba3(
            d_model=D_MODEL, d_state=128, headdim=64,
            is_mimo=False, chunk_size=64, is_outproj_norm=False, dtype=DTYPE,
        ),
    }

    results = {}
    for name, create_fn in models.items():
        results[name] = bench_module(name, create_fn)
    return results


def plot_results(results):
    names = list(results.keys())
    colors = ["#4C78A8", "#F58518", "#E45756"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Mamba1 vs Mamba2 vs Mamba3 — {torch.cuda.get_device_name(0)}\n"
        f"d_model={D_MODEL}, seq_len={SEQ_LEN}, batch={BATCH}, dtype=bf16 | "
        f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}",
        fontsize=12, fontweight="bold",
    )

    def add_bar_labels(ax, bars, values, fmt="{:.1f}"):
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.03,
                    fmt.format(v), ha="center", va="bottom", fontsize=10)

    # 1) Forward time
    ax = axes[0, 0]
    vals = [results[n]["fwd_ms"] for n in names]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("ms")
    ax.set_title("Forward Pass (lower is better)")
    add_bar_labels(ax, bars, vals, "{:.1f}ms")

    # 2) Backward time
    ax = axes[0, 1]
    vals = [results[n]["bwd_ms"] for n in names]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("ms")
    ax.set_title("Backward Pass (lower is better)")
    add_bar_labels(ax, bars, vals, "{:.1f}ms")

    # 3) VRAM (forward vs fwd+bwd side by side)
    ax = axes[1, 0]
    x_pos = range(len(names))
    w = 0.35
    fwd_vram = [results[n]["fwd_vram_gb"] for n in names]
    bwd_vram = [results[n]["bwd_vram_gb"] for n in names]
    ax.bar([p - w / 2 for p in x_pos], fwd_vram, w, label="Forward", color="#4C78A8")
    ax.bar([p + w / 2 for p in x_pos], bwd_vram, w, label="Fwd+Backward", color="#E45756")
    ax.axhline(y=11.6, color="red", linestyle="--", alpha=0.4, label="VRAM limit (11.6 GB)")
    ax.set_ylabel("GB")
    ax.set_title("Peak VRAM Usage")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(names)
    ax.legend(fontsize=9)

    # 4) Parameters
    ax = axes[1, 1]
    vals = [results[n]["params_m"] for n in names]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Module Parameters")
    add_bar_labels(ax, bars, vals, "{:.1f}M")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_PATH}")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Benchmark: d_model={D_MODEL}, seq_len={SEQ_LEN}, batch={BATCH}, bf16")
    print(f"Averaging over {N_RUNS} runs after {N_WARMUP} warmup\n")

    results = collect_results()

    print("\n--- Results ---")
    for name, r in results.items():
        print(f"  {name:12s}  params={r['params_m']:.1f}M  "
              f"fwd={r['fwd_ms']:.1f}ms  bwd={r['bwd_ms']:.1f}ms  "
              f"vram_fwd={r['fwd_vram_gb']:.2f}GB  vram_bwd={r['bwd_vram_gb']:.2f}GB")

    plot_results(results)


if __name__ == "__main__":
    main()
