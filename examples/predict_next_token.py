#!/usr/bin/env python3
"""Interactive next-token prediction with Mamba.

Mamba is a BASE language model (not a chatbot). It predicts the most likely
continuation of your text, as if completing a document from its training data
(The Pile: web text, code, Wikipedia, academic papers, etc.).

Usage:
    python examples/predict_next_token.py [--model MODEL] [--genlen N] [--skip-examples]

Models: state-spaces/mamba-130m (default), state-spaces/mamba-370m,
        state-spaces/mamba-1.4b, state-spaces/mamba-2.8b,
        state-spaces/mamba2-130m, state-spaces/mamba2-370m,
        state-spaces/mamba2-1.3b, state-spaces/mamba2-2.7b
"""

import argparse

import torch
from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

EXAMPLES = [
    ("The theory of relativity states that",
     "Science/factual continuation"),
    ("Once upon a time, in a land far away,",
     "Story continuation"),
    ("def fibonacci(n):",
     "Code completion"),
    ("The capital of Japan is",
     "Factual knowledge"),
]


def run_examples(model, tokenizer, genlen, temperature, top_k, top_p):
    """Run example prompts to show what the model does."""
    print("=" * 60)
    print("EXAMPLE OUTPUTS (showing what next-token prediction does)")
    print("=" * 60)

    for prompt, description in EXAMPLES:
        input_ids = torch.tensor([tokenizer.encode(prompt)], device="cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + min(genlen, 50),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        generated = tokenizer.decode(out[0].cpu().tolist())
        print(f"\n[{description}]")
        print(f"  Prompt:    {prompt}")
        print(f"  Mamba ->   {generated}")

    print("\n" + "=" * 60)
    print("NOTE: This is NOT a chatbot. It predicts the next tokens")
    print("based on training data. Longer, specific prompts work best.")
    print("Short inputs like 'hello' may produce random document fragments.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Interactive Mamba next-token prediction")
    parser.add_argument("--model", default="state-spaces/mamba-130m", help="Pretrained model name")
    parser.add_argument("--genlen", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--skip-examples", action="store_true", help="Skip example outputs on launch")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model, device="cuda", dtype=dtype)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Ready! ({params:.0f}M params, {args.dtype})\n")

    if not args.skip_examples:
        run_examples(model, tokenizer, args.genlen, args.temperature, args.top_k, args.top_p)

    print(f"\nSettings: genlen={args.genlen}, temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("Enter a prompt and Mamba will predict the continuation.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in ("quit", "exit", "q"):
            break
        if not prompt.strip():
            continue

        input_ids = torch.tensor([tokenizer.encode(prompt)], device="cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + args.genlen,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        print(tokenizer.decode(out[0].cpu().tolist()))
        print()


if __name__ == "__main__":
    main()
