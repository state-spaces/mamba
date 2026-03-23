#!/usr/bin/env python3
"""Benchmark text generation latency across Mamba model sizes.

Loads pretrained Mamba models, generates tokens from a prompt, and
measures end-to-end latency (prompt processing + token decoding).

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
        .venv/bin/python benchmarks/benchmark_text_generation_latency.py
"""

import argparse
import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import torch

from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})

GB = 1024**3
OUTPUT_PATH = Path(__file__).parent / "text_generation_latency.png"

MODELS = [
    ("mamba-130m", "state-spaces/mamba-130m"),
    ("mamba-370m", "state-spaces/mamba-370m"),
]

REPEATS = 3
GENLEN = 100
PROMPT = "The capital of France is"


def bench_model(name, pretrained, tokenizer, dtype, genlen):
    """Benchmark a single pretrained model, returning latency and throughput."""
    print(f"  Loading {name}...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    model = MambaLMHeadModel.from_pretrained(pretrained, device="cuda", dtype=dtype)
    model.eval()
    load_time = time.time() - t0

    params = sum(p.numel() for p in model.parameters()) / 1e6
    vram_model = torch.cuda.memory_allocated() / GB

    input_ids = torch.tensor([tokenizer.encode(PROMPT)], device="cuda")
    prompt_len = input_ids.shape[1]
    max_length = prompt_len + genlen

    # Warmup
    with torch.no_grad():
        model.generate(input_ids=input_ids, max_length=prompt_len + 10, temperature=0.7)

    # Timed runs
    torch.cuda.synchronize()
    latencies = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.time()
        out = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
        torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000)

    avg_latency_ms = sum(latencies) / len(latencies)
    tokens_gen = out.shape[1] - prompt_len
    tok_per_sec = tokens_gen / (avg_latency_ms / 1000)
    peak_vram = torch.cuda.max_memory_allocated() / GB
    generated_text = tokenizer.decode(out[0].cpu().tolist())

    del model, out, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "params_m": params,
        "load_time_s": load_time,
        "vram_model_gb": vram_model,
        "peak_vram_gb": peak_vram,
        "latency_ms": avg_latency_ms,
        "tok_per_sec": tok_per_sec,
        "tokens_gen": tokens_gen,
        "text": generated_text,
    }


def collect_results(dtype, genlen):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    results = {}
    for name, pretrained in MODELS:
        results[name] = bench_model(name, pretrained, tokenizer, dtype, genlen)
    return results


def plot_results(results):
    names = list(results.keys())
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Text Generation Latency — {torch.cuda.get_device_name(0)}\n"
        f"prompt={PROMPT!r}, genlen={GENLEN}, {REPEATS} runs | "
        f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}",
        fontsize=12, fontweight="bold",
    )

    def add_bar_labels(ax, bars, values, fmt="{:.1f}"):
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.03,
                    fmt.format(v), ha="center", va="bottom", fontsize=10)

    # 1) Generation throughput (tok/s)
    ax = axes[0, 0]
    vals = [results[n]["tok_per_sec"] for n in names]
    bars = ax.bar(names, vals, color=colors[:len(names)])
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Generation Throughput (higher is better)")
    add_bar_labels(ax, bars, vals, "{:.1f}")

    # 2) End-to-end latency (ms)
    ax = axes[0, 1]
    vals = [results[n]["latency_ms"] for n in names]
    bars = ax.bar(names, vals, color=colors[:len(names)])
    ax.set_ylabel("ms")
    ax.set_title(f"Total Latency for {GENLEN} tokens (lower is better)")
    add_bar_labels(ax, bars, vals, "{:.0f}ms")

    # 3) VRAM usage
    ax = axes[1, 0]
    x_pos = range(len(names))
    w = 0.35
    vram_model = [results[n]["vram_model_gb"] for n in names]
    vram_peak = [results[n]["peak_vram_gb"] for n in names]
    ax.bar([p - w / 2 for p in x_pos], vram_model, w, label="Model weights", color="#4C78A8")
    ax.bar([p + w / 2 for p in x_pos], vram_peak, w, label="Peak (generation)", color="#E45756")
    ax.axhline(y=11.6, color="red", linestyle="--", alpha=0.4, label="VRAM limit (11.6 GB)")
    ax.set_ylabel("GB")
    ax.set_title("VRAM Usage")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(names)
    ax.legend(fontsize=9)

    # 4) Model size
    ax = axes[1, 1]
    vals = [results[n]["params_m"] for n in names]
    bars = ax.bar(names, vals, color=colors[:len(names)])
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Model Size")
    add_bar_labels(ax, bars, vals, "{:.0f}M")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark text generation latency")
    parser.add_argument("--genlen", type=int, default=GENLEN, help="Number of tokens to generate")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Generation length: {args.genlen} tokens, dtype: {args.dtype}")
    print(f"Averaging over {REPEATS} runs\n")

    results = collect_results(dtype, args.genlen)

    print("\n--- Results ---")
    for name, r in results.items():
        print(f"  {name:12s}  params={r['params_m']:.0f}M  "
              f"latency={r['latency_ms']:.0f}ms  "
              f"throughput={r['tok_per_sec']:.1f} tok/s  "
              f"vram_peak={r['peak_vram_gb']:.2f}GB")

    plot_results(results)

    # Print sample output from first model
    first = next(iter(results.values()))
    print(f"\nSample output ({MODELS[0][0]}):")
    print(first["text"][:300])


if __name__ == "__main__":
    main()
