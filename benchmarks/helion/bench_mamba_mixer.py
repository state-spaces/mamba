"""Compare Triton and Helion SSD kernel speed via torch profiler traces.

Runs :class:`MambaMixerMin` (the same harness :mod:`autotune_mamba_mixer` uses,
so the shapes and the kernel set match production) through fwd+bwd under
``torch.profiler``, exports a chrome trace, and attributes every device kernel to
the dispatch stage that launched it. Doing it per stage rather than per kernel
name is the point: the two backends emit differently named kernels, so
``chunk_scan_fwd`` on Triton and on Helion are only comparable through the
dispatch wrapper they share.

    python benchmarks/helion/bench_mamba_mixer.py                  # both backends, then compare
    python benchmarks/helion/bench_mamba_mixer.py --backend helion # one trace only
    python benchmarks/helion/bench_mamba_mixer.py --compare a.json b.json

The default mode runs each backend in its own subprocess because
``dispatch.use_helion()`` is ``@cache``d -- the first read of ``MAMBA_USE_HELION``
pins the whole process to one backend. Autotuning is disabled (``MAMBA_HELION_AUTOTUNE``
is removed from the environment), so a missing config in ``MAMBA_HELION_CONFIG_DIR``
fails the run exactly as it would at runtime; generate configs with
``autotune_mamba_mixer`` first.
"""

import argparse
import bisect
import json
import os
import subprocess
import sys
from collections import defaultdict
from functools import wraps
from pathlib import Path

import torch

# Sibling script, importable because it sits next to this one; both are run by path.
from autotune_mamba_mixer import _DTYPE_MAP, build_seq_idx

_REPO_ROOT = Path(__file__).resolve().parents[2]

_STAGE_PREFIX = "ssd::"
_ITER_MARK = "bench::iter"
# Chrome-trace categories: device work, the launch APIs that correlate to it, and
# record_function markers.
_DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
_LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Dimensions -- keep in sync with autotune_mamba_mixer so a trace matches the
    # configs that harness produced.
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
    p.add_argument("--sequence-packing", action="store_true")
    p.add_argument("--packed-num-seqs", type=int, default=4)
    p.add_argument("--packed-seqlens", type=str, default=None)

    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--backend",
        choices=["triton", "helion"],
        default=None,
        help="Profile this backend only. Omit to run both in subprocesses and compare.",
    )
    p.add_argument(
        "--trace-dir",
        type=Path,
        default=_REPO_ROOT / "logs",
        help="Where the chrome traces are written.",
    )
    p.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("TRITON_JSON", "HELION_JSON"),
        default=None,
        help="Skip profiling; report on two existing traces.",
    )
    return p.parse_args()


def annotate_dispatch():
    """Wrap the dispatch entry points in ``record_function`` markers.

    Patches both the Triton namespace and the Helion getters in each consumer
    module, so the same instrumentation works whichever backend
    ``_ssd_kernel_impls()`` selects. Both call sites resolve these names at call
    time (``ssd_combined._ssd_kernel_impls`` and the ternary in
    ``LayerNormFn.backward``), which is what makes patching the module globals
    enough.
    """
    from types import SimpleNamespace

    from mamba_ssm.ops.triton import layernorm_gated, ssd_combined

    def marked(name, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with torch.profiler.record_function(f"{_STAGE_PREFIX}{name}"):
                return fn(*args, **kwargs)

        return wrapper

    kernels = ssd_combined._ssd_kernel_impls()
    wrapped = SimpleNamespace(
        **{name: marked(name, fn) for name, fn in vars(kernels).items()}
    )
    ssd_combined._TRITON_SSD_KERNELS = wrapped
    ssd_combined.get_helion_ssd_kernels = lambda: wrapped

    layer_norm_bwd = (
        layernorm_gated.get_helion_layer_norm_bwd()
        if layernorm_gated.use_helion()
        else layernorm_gated._layer_norm_bwd
    )
    wrapped_ln = marked("layer_norm_bwd", layer_norm_bwd)
    layernorm_gated._layer_norm_bwd = wrapped_ln
    layernorm_gated.get_helion_layer_norm_bwd = lambda: wrapped_ln

    return sorted(vars(wrapped)) + ["layer_norm_bwd"]


def profile_backend(args, backend, trace_path):
    """Profile ``args.iters`` fwd+bwd iterations and write a chrome trace."""
    os.environ["MAMBA_USE_HELION"] = "1" if backend == "helion" else "0"
    # Never autotune while benchmarking: it would both dwarf the measurement and
    # install a config tuned for whatever shapes happen to be here.
    os.environ.pop("MAMBA_HELION_AUTOTUNE", None)
    os.environ.setdefault(
        "MAMBA_HELION_CONFIG_DIR", str(Path(__file__).resolve().parent / "configs")
    )

    from mamba_ssm.ops.helion.dispatch import use_helion
    from mamba_ssm.ops.helion.mamba_mixer_min import MambaMixerMin

    assert use_helion() == (backend == "helion"), (
        f"dispatch.use_helion()={use_helion()} does not match --backend {backend}; "
        "something read MAMBA_USE_HELION before this ran"
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

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
    seq_idx = build_seq_idx(args)
    stages = annotate_dispatch()
    print(f"backend={backend} instrumented stages: {', '.join(stages)}")

    x = torch.randn(
        args.seq_len,
        args.batch_size,
        args.hidden_size,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    # Fixed grad-output so no RNG kernels land inside the timed region.
    dout = torch.randn_like(x)

    def step():
        mixer.zero_grad(set_to_none=True)
        x.grad = None
        mixer(x, seq_idx=seq_idx).backward(dout)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(args.iters):
            with torch.profiler.record_function(_ITER_MARK):
                step()
        torch.cuda.synchronize()

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    print(f"backend={backend} trace -> {trace_path}")


def read_trace(path):
    """Return ``(per-iteration stage totals in ms, diagnostics)`` for a trace.

    Device work is tied to a stage through the launch API call that produced it:
    kernels carry a ``correlation`` id shared with their ``cudaLaunchKernel`` /
    ``cuLaunchKernel`` event, and that launch sits inside the ``record_function``
    interval on the launching thread. Matching on the marker interval (rather
    than on ``External id``, which points at the innermost aten op) keeps the
    attribution correct for the backward pass too, where the launches happen on
    the autograd thread.
    """
    with open(path) as f:
        events = json.load(f)["traceEvents"]

    markers = defaultdict(list)  # tid -> [(ts, end, name)]
    iters = 0
    launch_stage = {}  # correlation -> stage
    device = []  # (correlation, dur, name)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat, name = ev.get("cat"), ev.get("name", "")
        if cat == "user_annotation":
            if name == _ITER_MARK:
                iters += 1
            elif name.startswith(_STAGE_PREFIX):
                markers[ev["tid"]].append(
                    (ev["ts"], ev["ts"] + ev["dur"], name[len(_STAGE_PREFIX) :])
                )
        elif cat in _LAUNCH_CATS:
            corr = ev.get("args", {}).get("correlation")
            if corr is not None:
                launch_stage[corr] = (ev["tid"], ev["ts"])
        elif cat in _DEVICE_CATS:
            device.append(
                (ev.get("args", {}).get("correlation"), ev["dur"], name)
            )

    assert iters, f"no {_ITER_MARK} markers in {path} -- was it written by this script?"
    for tid in markers:
        markers[tid].sort()

    def stage_of(corr):
        launch = launch_stage.get(corr)
        if launch is None:
            return None
        tid, ts = launch
        intervals = markers.get(tid)
        if not intervals:
            return None
        # Rightmost marker starting at or before the launch; markers of interest
        # never nest, so containment in that one decides it.
        i = bisect.bisect_right(intervals, (ts, float("inf"), "")) - 1
        if i >= 0 and intervals[i][1] >= ts:
            return intervals[i][2]
        return None

    stage_ms = defaultdict(float)
    kernel_ms = defaultdict(lambda: defaultdict(float))
    unattributed = 0.0
    for corr, dur, name in device:
        stage = stage_of(corr)
        if stage is None:
            unattributed += dur
        else:
            stage_ms[stage] += dur
            kernel_ms[stage][name] += dur

    scale = 1e-3 / iters  # us total -> ms per iteration
    return (
        {k: v * scale for k, v in stage_ms.items()},
        {s: {n: v * scale for n, v in ks.items()} for s, ks in kernel_ms.items()},
        {
            "iters": iters,
            "other_ms": unattributed * scale,
            "total_ms": (sum(stage_ms.values()) + unattributed) * scale,
        },
    )


def generated_only(kernels):
    """Sum only the backend's own generated kernel(s) from a stage's kernels.

    A Triton or Helion kernel is launched under its Python function name
    (``_chunk_scan_fwd_kernel``, ``_helion__chunk_scan_fwd_kernel``), so its trace
    name is a valid identifier; the helpers a stage also launches are not --
    aten/cutlass kernels arrive as demangled C++ signatures (``void
    at::native::...``) and memset/memcpy as ``Memset (Device)``. Dropping them
    isolates the kernel comparison from the different amount of work the two
    backends leave to aten (e.g. Triton reduces dD with a separate
    ``reduce_kernel`` where Helion uses in-kernel atomics).
    """
    return sum(ms for name, ms in kernels.items() if name.isidentifier())


def report(labelled_paths):
    """Print the per-stage comparison table across the given traces."""
    traces = [(label, *read_trace(path)) for label, path in labelled_paths]
    for label, _, _, diag in traces:
        print(
            f"{label:>8}: {diag['iters']} iters, device time/iter "
            f"{diag['total_ms']:.3f} ms (SSD+LN stages "
            f"{diag['total_ms'] - diag['other_ms']:.3f} ms, "
            f"other {diag['other_ms']:.3f} ms)"
        )

    stages = sorted(
        {s for _, stage_ms, _, _ in traces for s in stage_ms},
        key=lambda s: -max(stage_ms.get(s, 0.0) for _, stage_ms, _, _ in traces),
    )
    # In --compare mode the labels are file stems; keep the distinguishing tail so
    # they still fit one numeric column.
    labels = [label[-11:] for label, *_ in traces]
    width = max(len(s) for s in stages) + 2
    paired = len(traces) == 2
    group = 12 * len(labels) + (9 if paired else 0)

    def row(name, whole, kernel):
        cells = "".join(f"{v:>12.3f}" for v in whole)
        if paired:
            cells += f"{(whole[0] / whole[1] if whole[1] else float('nan')):>8.2f}x"
        cells += "   " + "".join(f"{v:>12.3f}" for v in kernel)
        if paired:
            cells += f"{(kernel[0] / kernel[1] if kernel[1] else float('nan')):>8.2f}x"
        return f"{name:<{width}}{cells}"

    groups = f"\n{'':<{width}}{'stage device time':^{group}}   {'generated kernel only':^{group}}"
    header = f"{'stage':<{width}}" + 2 * (
        "".join(f"{lb:>12}" for lb in labels) + (f"{'speedup':>9}" if paired else "") + "   "
    )
    print(groups)
    print(header)
    print("-" * len(header))
    for stage in stages:
        print(
            row(
                stage,
                [stage_ms.get(stage, 0.0) for _, stage_ms, _, _ in traces],
                [generated_only(km.get(stage, {})) for _, _, km, _ in traces],
            )
        )
    print(
        row(
            "TOTAL (stages)",
            [sum(stage_ms.values()) for _, stage_ms, _, _ in traces],
            [sum(generated_only(ks) for ks in km.values()) for _, _, km, _ in traces],
        )
    )
    print("\nall times are ms of device time per iteration; "
          "speedup = first / second (>1 means the second is faster). "
          "'stage device time' includes the aten helper kernels each stage launches; "
          "'generated kernel only' is just the Triton/Helion kernel.")

    print("\nper-kernel breakdown:")
    for label, _, kernel_ms, _ in traces:
        print(f"  [{label}]")
        for stage in stages:
            for name, ms in sorted(
                kernel_ms.get(stage, {}).items(), key=lambda kv: -kv[1]
            ):
                print(f"    {stage:<32} {ms:8.3f}  {name}")


def run_both(args):
    """Profile each backend in its own subprocess, then compare the traces."""
    paths = {}
    for backend in ("triton", "helion"):
        path = args.trace_dir / f"bench_mamba_mixer_{backend}.json"
        cmd = [sys.executable, __file__, *sys.argv[1:], "--backend", backend]
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        paths[backend] = path
    report([("triton", paths["triton"]), ("helion", paths["helion"])])


def main():
    assert torch.cuda.is_available(), "CUDA is required."
    args = parse_args()
    if args.compare:
        report([(p.stem, p) for p in args.compare])
    elif args.backend:
        profile_backend(
            args,
            args.backend,
            args.trace_dir / f"bench_mamba_mixer_{args.backend}.json",
        )
    else:
        run_both(args)


if __name__ == "__main__":
    main()
