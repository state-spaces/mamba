#!/usr/bin/env python3
"""Test Mamba models on RTX 4080 (sm_89, 12GB VRAM).

Usage:
    python tests/test_rtx4080.py [--model MODEL] [--config CONFIG]

Models: mamba1_130m, mamba1_370m, mamba1_1.4b, mamba1_2.8b, mamba3_test
"""

import argparse
import json
import time
from pathlib import Path

import torch

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
GB = 1024**3


def load_gpu_config(config_path: str = None):
    config_path = config_path or str(CONFIGS_DIR / "rtx4080.json")
    with open(config_path) as f:
        return json.load(f)


def print_gpu_info():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / GB:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()


def test_mamba1_pretrained(model_name: str, config: dict):
    """Test Mamba1/2 pretrained model via MambaLMHeadModel."""
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    model_cfg = config["models"][model_name]
    pretrained = model_cfg["pretrained"]
    dtype = getattr(torch, model_cfg.get("dtype", config["defaults"]["dtype"]))

    print(f"--- Loading {model_name} from {pretrained} ---")
    t0 = time.time()
    model = MambaLMHeadModel.from_pretrained(pretrained, device="cuda", dtype=dtype)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"VRAM used: {torch.cuda.memory_allocated() / GB:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    prompt = "The capital of France is"
    input_ids = torch.tensor([tokenizer.encode(prompt)], device="cuda")

    print(f"\nPrompt: {prompt!r}")
    with torch.no_grad():
        t0 = time.time()
        out = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + 50,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        elapsed = time.time() - t0

    generated = tokenizer.decode(out[0].cpu().tolist())
    tokens_generated = out.shape[1] - input_ids.shape[1]
    print(f"Generated ({tokens_generated} tokens in {elapsed:.2f}s, {tokens_generated/elapsed:.1f} tok/s):")
    print(generated)
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / GB:.2f} GB")
    print()


def test_mamba3(config: dict):
    """Test Mamba3 module with random input (no pretrained model available yet)."""
    from mamba_ssm.modules.mamba3 import Mamba3

    model_cfg = config["models"]["mamba3_test"]
    dtype = getattr(torch, model_cfg.get("dtype", config["defaults"]["dtype"]))

    is_mimo = model_cfg.get("is_mimo", False)
    mode_str = "MIMO" if is_mimo else "SISO"
    print(f"--- Testing Mamba3 module ({mode_str}, random weights) ---")

    mamba3_kwargs = dict(
        d_model=model_cfg["d_model"],
        d_state=model_cfg["d_state"],
        headdim=model_cfg["headdim"],
        is_mimo=is_mimo,
        chunk_size=model_cfg["chunk_size"],
        is_outproj_norm=False,
        dtype=dtype,
    )
    if is_mimo:
        mamba3_kwargs["mimo_rank"] = model_cfg.get("mimo_rank", 4)

    model = Mamba3(**mamba3_kwargs).to("cuda")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    batch, length, dim = 2, 2048, model_cfg["d_model"]
    x = torch.randn(batch, length, dim, dtype=dtype, device="cuda")

    print(f"Input shape: {x.shape}")
    with torch.no_grad():
        t0 = time.time()
        y = model(x)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

    print(f"Output shape: {y.shape}")
    print(f"Forward pass: {elapsed*1000:.1f}ms")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / GB:.2f} GB")

    # Test backward pass
    torch.cuda.reset_peak_memory_stats()
    x.requires_grad_(True)
    y = model(x)
    loss = y.sum()
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Backward pass: {elapsed*1000:.1f}ms")
    print(f"Peak VRAM (fwd+bwd): {torch.cuda.max_memory_allocated() / GB:.2f} GB")
    print()


def main():
    parser = argparse.ArgumentParser(description="Test Mamba models on RTX 4080")
    parser.add_argument(
        "--model", default="mamba1_130m",
        choices=["mamba1_130m", "mamba1_370m", "mamba1_1.4b", "mamba1_2.8b", "mamba3_test", "all"],
        help="Model to test (default: mamba1_130m)",
    )
    parser.add_argument("--config", default=None, help="Path to GPU config JSON")
    args = parser.parse_args()

    config = load_gpu_config(args.config)
    print_gpu_info()

    models = list(config["models"]) if args.model == "all" else [args.model]
    for name in models:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if name == "mamba3_test":
            test_mamba3(config)
        else:
            test_mamba1_pretrained(name, config)


if __name__ == "__main__":
    main()
