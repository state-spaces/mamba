"""Autotune the Helion kernels via a repo-internal MambaMixer.

Drives :class:`MambaMixerMin` (a Megatron-free replica of MambaMixer's training
path) through one forward+backward, hitting every Helion kernel in the production
path on a single GPU with no Megatron-LM dependency. Dimensions default to the
Nemotron3 Nano production config (seq_len 8k).

``MAMBA_HELION_AUTOTUNE=1`` autotunes only kernels whose ``<kernel>.json`` is
missing from ``MAMBA_HELION_CONFIG_DIR``; ``=2`` forces a re-autotune of all of
them (e.g. for a new GPU / Helion version). Both default on: AUTOTUNE=1 and
CONFIG_DIR=``<this dir>/configs``. The harness also enables
``MAMBA_USE_HELION=1`` so production dispatch selects Helion.

    python benchmarks/helion/autotune_mamba_mixer.py

After the run the harness checks that every dispatched kernel (the nine SSD
kernels plus the gated LayerNorm backward) left a ``<kernel>.json`` behind, and
fails if one did not -- otherwise a kernel this configuration never reaches would
only surface later as a runtime ``ensure_helion_config`` assertion.

A config is specialized to the run's shapes/path (dims, dtype, ``--sequence-packing``,
...), so autotune each distinct variant into its own ``MAMBA_HELION_CONFIG_DIR`` and
point the runtime at the matching set.
"""

import argparse
import os
from pathlib import Path

import torch

# NOTE: mamba_ssm.ops.helion.mamba_mixer_min (and through it ssd_combined /
# layernorm_gated) is imported inside main(), after configure_env() -- dispatch's
# use_helion() is @cache'd, so any import-time read of MAMBA_USE_HELION would pin
# the whole harness to Triton.

_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def parse_args():
    p = argparse.ArgumentParser(description="Autotune Helion kernels via MambaMixer")
    # Dimensions — defaults match the Nemotron3 Nano production config (seq_len 8k).
    p.add_argument("--hidden-size", type=int, default=2688)
    p.add_argument("--mamba-num-heads", type=int, default=64)
    p.add_argument("--mamba-head-dim", type=int, default=64)
    p.add_argument("--mamba-state-dim", type=int, default=128)
    p.add_argument("--mamba-num-groups", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    # Sets HELION_AUTOTUNE_PRECOMPILE. "spawn" precompiles each candidate config in a
    # subprocess, so a config that hard-crashes the CUDA context (e.g. Triton
    # "misaligned address") is skipped instead of aborting the whole autotune run;
    # it costs more memory and cold-start time. "fork" is Helion's default, "off"
    # precompiles in-process.
    p.add_argument(
        "--precompile",
        choices=["spawn", "fork", "off"],
        default=None,
        help="Helion autotune precompile mode. Explicit flag wins; otherwise an "
        "exported HELION_AUTOTUNE_PRECOMPILE is kept, else 'fork'. Use 'spawn' to "
        "isolate configs that crash the CUDA context.",
    )
    # Sequence packing (THD) — feeds a seq_idx tensor so the packed-sequence kernel
    # path is autotuned. Requires --batch-size 1.
    p.add_argument(
        "--sequence-packing",
        action="store_true",
        help="Build a seq_idx and run the packed (THD) path.",
    )
    p.add_argument(
        "--packed-num-seqs",
        type=int,
        default=4,
        help="Number of sub-sequences to split --seq-len into (uniform).",
    )
    p.add_argument(
        "--packed-seqlens",
        type=str,
        default=None,
        help="Comma-separated sub-seq lengths summing to --seq-len. "
        "Overrides --packed-num-seqs.",
    )
    return p.parse_args()


def build_seq_idx(args):
    """Return a (1, seq_len) int32 seq_idx mapping each token to its sub-seq index.

    Mirrors Megatron MambaMixer._create_packed_seq_idx: for sub-seq lengths
    [5, 2, 4] the result is [0,0,0,0,0, 1,1, 2,2,2,2]. Returns None when packing
    is disabled.
    """
    if not args.sequence_packing:
        return None
    if args.batch_size != 1:
        raise ValueError(
            f"Sequence packing requires --batch-size 1 (got {args.batch_size})."
        )

    if args.packed_seqlens:
        lengths = [int(s) for s in args.packed_seqlens.split(",") if s.strip()]
        if sum(lengths) != args.seq_len:
            raise ValueError(
                f"--packed-seqlens sum ({sum(lengths)}) != --seq-len ({args.seq_len})."
            )
    else:
        n = args.packed_num_seqs
        if not 1 <= n <= args.seq_len:
            raise ValueError(
                f"--packed-num-seqs must be in [1, seq_len={args.seq_len}] (got {n})."
            )
        # Random sub-seq lengths (not a uniform split) so the packed path sees
        # ragged boundaries like real data. Pick n-1 distinct interior cut points;
        # the gaps give n lengths, each >= 1, summing to seq_len. Reproducible via
        # --seed (main seeds torch).
        cuts, _ = torch.sort(torch.randperm(args.seq_len - 1)[: n - 1] + 1)
        bounds = torch.cat([cuts.new_zeros(1), cuts, cuts.new_full((1,), args.seq_len)])
        lengths = torch.diff(bounds).tolist()

    idx = torch.arange(len(lengths), device="cuda")
    seq_idx = (
        torch.repeat_interleave(idx, torch.tensor(lengths, device="cuda"))
        .to(torch.int32)
        .unsqueeze(0)
    )  # (1, seq_len)
    print(
        f"sequence_packing : {len(lengths)} sub-seqs, lengths={lengths[:10]}"
        f"{'...' if len(lengths) > 10 else ''}"
    )
    return seq_idx


def expected_kernel_names():
    """Return ``{dispatch wrapper name: [helion kernel name, ...]}``.

    Walks the dispatch namespaces -- the nine SSD kernels plus the gated
    LayerNorm backward -- and, for each wrapper, picks out the
    ``json_cached_autotune``-wrapped kernels it references, so the list stays in
    sync with dispatch.py instead of being restated here. The kernel names are
    the ``<name>.json`` basenames ``ensure_helion_config`` looks for at runtime.
    """
    import helion

    from mamba_ssm.ops.helion.dispatch import (
        get_helion_layer_norm_bwd,
        get_helion_ssd_kernels,
    )

    wrappers = list(vars(get_helion_ssd_kernels()).values())
    wrappers.append(get_helion_layer_norm_bwd())

    names = {}
    for wrapper in wrappers:
        referenced = [
            obj.kernel.name
            for obj in (wrapper.__globals__.get(n) for n in wrapper.__code__.co_names)
            if isinstance(getattr(obj, "kernel", None), helion.Kernel)
        ]
        assert referenced, f"no Helion kernel found in {wrapper.__name__}"
        names[wrapper.__name__] = referenced
    return names


def verify_configs(config_dir):
    """Fail unless every dispatched kernel has a config file in ``config_dir``.

    A kernel that the harness never reaches would otherwise leave the run
    "successful" with its JSON missing, and only blow up later in
    ``ensure_helion_config`` at runtime -- exactly the invariant this harness
    exists to guarantee.
    """
    missing = []
    for wrapper_name, kernel_names in expected_kernel_names().items():
        for kernel_name in kernel_names:
            found = (config_dir / f"{kernel_name}.json").is_file()
            if not found:
                missing.append(f"{kernel_name}.json (dispatched as {wrapper_name})")
            print(f"  [{'ok' if found else 'MISSING'}] {kernel_name}.json")
    assert not missing, (
        f"{len(missing)} Helion config(s) missing from {config_dir} after the "
        "autotune run -- the kernel(s) were not reached by this configuration:\n  "
        + "\n  ".join(missing)
    )


def configure_env(args):
    """Set the Helion env vars. Must run before anything imports the dispatch path.

    ``dispatch.use_helion()`` is ``@cache``d, so the first read of
    ``MAMBA_USE_HELION`` wins for the whole process: setting it after an import
    that already asked would silently run this harness on Triton and autotune
    nothing.
    """
    # Defaults so the script autotunes into the repo's configs dir out of the box;
    # explicit env vars still win (setdefault). For the packed (THD) path you must set
    # MAMBA_HELION_CONFIG_DIR yourself to a separate dir -- see the sequence-packing
    # note above on why packed configs must stay separate from the non-packed ones.
    os.environ.setdefault("MAMBA_USE_HELION", "1")
    assert os.environ["MAMBA_USE_HELION"] == "1", (
        "The Helion autotune harness requires MAMBA_USE_HELION=1"
    )
    os.environ.setdefault("MAMBA_HELION_AUTOTUNE", "1")
    default_config_dir = Path(__file__).resolve().parent / "configs"
    os.environ.setdefault("MAMBA_HELION_CONFIG_DIR", str(default_config_dir))
    Path(os.environ["MAMBA_HELION_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    # An explicit --precompile wins; otherwise keep an exported value, like the
    # three above, and fall back to the harness default.
    if args.precompile is not None:
        os.environ["HELION_AUTOTUNE_PRECOMPILE"] = (
            "0" if args.precompile == "off" else args.precompile
        )
    else:
        os.environ.setdefault("HELION_AUTOTUNE_PRECOMPILE", "fork")

    if os.environ.get("MAMBA_HELION_AUTOTUNE") not in ("1", "2"):
        print(
            "WARNING: MAMBA_HELION_AUTOTUNE not in {'1','2'}; existing configs are "
            "loaded and nothing is autotuned."
        )
    print(
        f"MAMBA_USE_HELION={os.environ['MAMBA_USE_HELION']} "
        f"MAMBA_HELION_AUTOTUNE={os.environ['MAMBA_HELION_AUTOTUNE']} "
        f"MAMBA_HELION_CONFIG_DIR={os.environ['MAMBA_HELION_CONFIG_DIR']} "
        f"HELION_AUTOTUNE_PRECOMPILE={os.environ['HELION_AUTOTUNE_PRECOMPILE']}"
    )

    # Importing this module already pulled in the mamba_ssm package, so deferring
    # the mixer import is not by itself proof that nothing read MAMBA_USE_HELION
    # too early. Calling the @cache'd helper here either caches the right answer
    # or reveals a stale cached one, instead of quietly running on Triton.
    from mamba_ssm.ops.helion.dispatch import use_helion

    assert use_helion(), (
        "dispatch.use_helion() is already cached False -- something read "
        "MAMBA_USE_HELION before configure_env(); the harness would run on Triton "
        "and autotune nothing"
    )


def main():
    assert torch.cuda.is_available(), "CUDA is required."
    args = parse_args()
    configure_env(args)

    # Imported only now: this pulls in ssd_combined / layernorm_gated, which reach
    # the @cache'd dispatch helpers.
    from mamba_ssm.ops.helion.mamba_mixer_min import MambaMixerMin

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    # Do not initialize torch.distributed here. These kernels do not declare
    # distributed intent and normally use ordinary CUDA tensors, so Helion 1.4 treats
    # them as ordinary kernels even in distributed jobs. The config helper enables
    # Helion's timeout-protected subprocess benchmark path for autotuning.

    dtype = _DTYPE_MAP[args.dtype]
    mixer = MambaMixerMin(
        d_model=args.hidden_size,
        nheads=args.mamba_num_heads,
        headdim=args.mamba_head_dim,
        d_state=args.mamba_state_dim,
        ngroups=args.mamba_num_groups,
        chunk_size=args.chunk_size,
        device="cuda",
        dtype=dtype,
    )
    mixer.train()

    print(
        f"d_model={args.hidden_size} nheads={args.mamba_num_heads} headdim={args.mamba_head_dim} "
        f"d_state={args.mamba_state_dim} ngroups={args.mamba_num_groups} "
        f"chunk_size={args.chunk_size} seq_len={args.seq_len} batch={args.batch_size} dtype={args.dtype}"
    )
    seq_idx = build_seq_idx(args)
    print("running one fwd+bwd to trigger Helion autotune for any missing configs...")

    x = torch.randn(
        args.seq_len,
        args.batch_size,
        args.hidden_size,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    out = mixer(x, seq_idx=seq_idx)
    (out * torch.randn_like(out)).sum().backward()
    torch.cuda.synchronize()

    print("verifying every dispatched kernel has a config...")
    verify_configs(Path(os.environ["MAMBA_HELION_CONFIG_DIR"]))
    print("done.")


if __name__ == "__main__":
    main()
