#!/usr/bin/env python3
"""Run Mamba tests on RTX 4080 and generate a visual report."""

import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import torch

matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 120})

GB = 1024**3
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
OUTPUT_PATH = Path(__file__).parent / "test_results_rtx4080.png"


def load_gpu_config():
    with open(CONFIGS_DIR / "rtx4080.json") as f:
        return json.load(f)


def collect_mamba1_results(config):
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    prompt = "The capital of France is"
    results = {}

    # Test models one at a time, aggressively freeing VRAM between runs
    test_models = ["mamba1_130m", "mamba1_370m"]
    for name in test_models:
        cfg = config["models"].get(name)
        if not cfg or "pretrained" not in cfg:
            continue
        dtype = getattr(torch, cfg.get("dtype", "bfloat16"))

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        print(f"Testing {name}...")
        t0 = time.time()
        model = MambaLMHeadModel.from_pretrained(cfg["pretrained"], device="cuda", dtype=dtype)
        model.eval()
        load_time = time.time() - t0

        params = sum(p.numel() for p in model.parameters()) / 1e6
        vram_model = torch.cuda.memory_allocated() / GB

        input_ids = torch.tensor([tokenizer.encode(prompt)], device="cuda")
        with torch.no_grad():
            t0 = time.time()
            out = model.generate(input_ids=input_ids, max_length=input_ids.shape[1] + 50, temperature=0.7)
            gen_time = time.time() - t0

        tokens_gen = out.shape[1] - input_ids.shape[1]
        tok_per_sec = tokens_gen / gen_time
        peak_vram = torch.cuda.max_memory_allocated() / GB
        text = tokenizer.decode(out[0].cpu().tolist())

        results[name] = {
            "params_m": params, "load_time": load_time, "vram_model_gb": vram_model,
            "peak_vram_gb": peak_vram, "tok_per_sec": tok_per_sec, "gen_time": gen_time,
            "tokens_gen": tokens_gen, "text": text,
        }
        del model, out, input_ids
        gc.collect()
        torch.cuda.empty_cache()

    return results


def collect_mamba3_results(config):
    from mamba_ssm.modules.mamba3 import Mamba3

    cfg = config["models"]["mamba3_test"]
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    is_mimo = cfg.get("is_mimo", False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Testing mamba3_test...")
    kwargs = dict(
        d_model=cfg["d_model"], d_state=cfg["d_state"], headdim=cfg["headdim"],
        is_mimo=is_mimo, chunk_size=cfg["chunk_size"], is_outproj_norm=False, dtype=dtype,
    )
    if is_mimo:
        kwargs["mimo_rank"] = cfg.get("mimo_rank", 4)

    model = Mamba3(**kwargs).to("cuda")
    params = sum(p.numel() for p in model.parameters()) / 1e6

    batch, length, dim = 2, 2048, cfg["d_model"]

    # Warm-up
    x = torch.randn(batch, length, dim, dtype=dtype, device="cuda")
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()

    # Forward
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, length, dim, dtype=dtype, device="cuda")
    with torch.no_grad():
        t0 = time.time()
        y = model(x)
        torch.cuda.synchronize()
        fwd_ms = (time.time() - t0) * 1000
    fwd_vram = torch.cuda.max_memory_allocated() / GB

    # Backward
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, length, dim, dtype=dtype, device="cuda", requires_grad=True)
    y = model(x)
    loss = y.sum()
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    bwd_ms = (time.time() - t0) * 1000
    bwd_vram = torch.cuda.max_memory_allocated() / GB

    return {
        "params_m": params, "fwd_ms": fwd_ms, "bwd_ms": bwd_ms,
        "fwd_vram_gb": fwd_vram, "bwd_vram_gb": bwd_vram,
        "mode": "MIMO" if is_mimo else "SISO",
    }


def plot_results(mamba1_results, mamba3_result):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Mamba Test Results — {torch.cuda.get_device_name(0)}\n"
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory / GB:.1f} GB | "
        f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    names = list(mamba1_results.keys())
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]

    # 1) Parameters
    ax1 = fig.add_subplot(2, 3, 1)
    params = [mamba1_results[n]["params_m"] for n in names]
    bars = ax1.bar(names, params, color=colors[:len(names)])
    ax1.set_ylabel("Parameters (M)")
    ax1.set_title("Model Size")
    for bar, v in zip(bars, params):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(params) * 0.02,
                 f"{v:.0f}M", ha="center", va="bottom", fontsize=9)
    ax1.tick_params(axis="x", rotation=20)

    # 2) Generation speed
    ax2 = fig.add_subplot(2, 3, 2)
    speeds = [mamba1_results[n]["tok_per_sec"] for n in names]
    bars = ax2.bar(names, speeds, color=colors[:len(names)])
    ax2.set_ylabel("Tokens / sec")
    ax2.set_title("Generation Speed (50 tokens)")
    for bar, v in zip(bars, speeds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(speeds) * 0.02,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax2.tick_params(axis="x", rotation=20)

    # 3) VRAM usage
    ax3 = fig.add_subplot(2, 3, 3)
    vram_model = [mamba1_results[n]["vram_model_gb"] for n in names]
    vram_peak = [mamba1_results[n]["peak_vram_gb"] for n in names]
    x_pos = range(len(names))
    w = 0.35
    ax3.bar([p - w / 2 for p in x_pos], vram_model, w, label="Model", color="#4C78A8")
    ax3.bar([p + w / 2 for p in x_pos], vram_peak, w, label="Peak", color="#E45756")
    ax3.axhline(y=11.6, color="red", linestyle="--", alpha=0.5, label="VRAM limit (11.6 GB)")
    ax3.set_ylabel("GB")
    ax3.set_title("VRAM Usage")
    ax3.set_xticks(list(x_pos))
    ax3.set_xticklabels(names, rotation=20)
    ax3.legend(fontsize=8)

    # 4) Mamba3 forward/backward
    ax4 = fig.add_subplot(2, 3, 4)
    m3_labels = ["Forward", "Backward"]
    m3_times = [mamba3_result["fwd_ms"], mamba3_result["bwd_ms"]]
    bars = ax4.bar(m3_labels, m3_times, color=["#4C78A8", "#E45756"])
    ax4.set_ylabel("ms")
    ax4.set_title(f"Mamba3 {mamba3_result['mode']} (d=768, L=2048)")
    for bar, v in zip(bars, m3_times):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(m3_times) * 0.02,
                 f"{v:.0f}ms", ha="center", va="bottom", fontsize=9)

    # 5) Mamba3 VRAM
    ax5 = fig.add_subplot(2, 3, 5)
    m3_vram = [mamba3_result["fwd_vram_gb"], mamba3_result["bwd_vram_gb"]]
    bars = ax5.bar(["Forward", "Fwd+Backward"], m3_vram, color=["#4C78A8", "#E45756"])
    ax5.axhline(y=11.6, color="red", linestyle="--", alpha=0.5, label="VRAM limit")
    ax5.set_ylabel("GB")
    ax5.set_title(f"Mamba3 {mamba3_result['mode']} VRAM")
    for bar, v in zip(bars, m3_vram):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(m3_vram) * 0.02,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax5.legend(fontsize=8)

    # 6) Generated text sample (from smallest model)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    first_model = names[0]
    text = mamba1_results[first_model]["text"][:300]
    ax6.set_title(f"Sample Output ({first_model})", fontsize=11)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=8,
             verticalalignment="top", family="monospace", wrap=True,
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"\nSaved visual report to {OUTPUT_PATH}")


def main():
    config = load_gpu_config()
    mamba1_results = collect_mamba1_results(config)
    mamba3_result = collect_mamba3_results(config)
    plot_results(mamba1_results, mamba3_result)


if __name__ == "__main__":
    main()
