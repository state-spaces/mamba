# Helion kernel backend for the Mamba-2 SSD path

An opt-in [Helion](https://github.com/pytorch/helion) implementation of the nine SSD
kernels used by `mamba_split_conv1d_scan_combined` plus the gated-LayerNorm backward.
The Triton kernels remain the default and the numerical reference; nothing changes
unless `MAMBA_USE_HELION=1` is set.

## What this branch adds

Runtime code lives in this directory; the offline tooling and the example configs live
outside the installed package, under `benchmarks/helion/`, so nothing autotunes or writes
into site-packages.

| File | Role |
| --- | --- |
| `dispatch.py` | The only import of this package from production code. Resolves the backend once per process and lazily imports the kernels **after** `MAMBA_USE_HELION=1` is confirmed, so an install without `helion` still works. |
| `utils.py` | `json_cached_autotune` / `ensure_helion_config`: resolve each kernel's config from `MAMBA_HELION_CONFIG_DIR/<kernel.name>.json` on first call, autotune only when explicitly asked. |
| `_chunk_state_fwd.py`, `_state_passing_fwd.py`, `_chunk_scan_fwd.py` | Forward stages. |
| `_chunk_scan_bwd_dstates.py`, `_state_passing_bwd.py`, `_chunk_scan_chunk_state_bwd_dx.py`, `_chunk_state_bwd_db.py`, `_chunk_scan_bwd_dC.py`, `_chunk_scan_bwd_ddAcs_stable.py` | Backward stages. |
| `_layer_norm_bwd.py` | Gated LayerNorm/RMSNorm backward (`layernorm_gated`). |
| `mamba_mixer_min.py` | `MambaMixerMin`, a Megatron-free replica of Megatron-LM's `MambaMixer._ssm_training` layout. Imported by the harness, the benchmark and the tests, which is why it stays in the package. |

Outside the package:

| Path | Role |
| --- | --- |
| `benchmarks/helion/autotune_mamba_mixer.py` | Offline autotune harness — one fwd+bwd that reaches every kernel, then verifies each left a config behind. |
| `benchmarks/helion/run_autotune.sh` | Runs the harness detached under `nohup`, logging to `logs/`. |
| `benchmarks/helion/bench_mamba_mixer.py` | Triton-vs-Helion per-stage device-time comparison from `torch.profiler` traces. |
| `benchmarks/helion/configs/` | Where the harness autotunes into, and where the checked-in example set lives (see below). Not shipped in the wheel. |
| `tests/ops/helion/test_kernels.py` | Every kernel against its Triton reference, on production arguments. |

Consumers in the Triton package:

- [`ssd_combined.py`](../triton/ssd_combined.py) — `_ssd_kernel_impls()` returns either
  `_TRITON_SSD_KERNELS` or `get_helion_ssd_kernels()`, both `SimpleNamespace`s with the
  same nine attribute names; the fwd/bwd functions call through it.
- [`layernorm_gated.py`](../triton/layernorm_gated.py) — `LayerNormFn.backward` picks
  `get_helion_layer_norm_bwd()` over `_layer_norm_bwd`.

Adding or renaming a kernel means keeping those two namespaces, `_SSD_KERNELS` in the
test, and the per-kernel file in sync — `test_kernel_namespaces_match` guards exactly that.

Requirements beyond the base install come from the `helion` extra (`helion==1.4.0` and
`triton>=3.6`), plus `causal_conv1d` for the harness, benchmark and tests:

```sh
pip install -e '.[helion,causal-conv1d]' --no-build-isolation
```

`helion` is pinned exactly, not lower-bounded: a config is autotuned for one toolchain and
Helion's config schema is still moving, so another version can need re-autotuning — or fail
to load a config — even with the kernels unchanged. Raise the pin deliberately, together
with a re-autotuned config set.

## Validated environment

Everything below — the checked-in config set, the test results and the benchmark numbers —
was produced on a single node of:

| | |
| --- | --- |
| GPU | NVIDIA B200 183GB (compute capability 10.0), driver 580.95.05; all runs single-GPU |
| OS / Python | Linux 6.8.0 x86_64, glibc 2.39, CPython 3.12.3 |
| torch | 2.11.0a0+a6c236b9fd.nv26.03 (NGC 26.03 container), CUDA 13.2 |
| triton | 3.6.0 |
| helion | 1.4.0 |
| causal_conv1d | 1.6.2.post1 |
| einops / pytest | 0.8.2 / 8.1.1 |

Model shape for all of it, the autotune harness defaults (Nemotron3 Nano): `d_model` 2688,
64 heads x 64 head dim, `d_state` 128, 8 groups, `chunk_size` 128, seq_len 8192, batch 1,
bf16, `MambaMixerMin` un-fused layout.

The kernels themselves are not Blackwell-specific, but a config set is (see
[Autotuning configs](#autotuning-configs)): on any other GPU, toolchain version or model
shape, re-autotune and re-run the tests before trusting the backend.

## Enabling it

```sh
export MAMBA_USE_HELION=1
export MAMBA_HELION_CONFIG_DIR=$PWD/benchmarks/helion/configs/b200-helion1.4-triton3.6-nemotron3-nano-bf16-seq8k
```

| Env var | Meaning |
| --- | --- |
| `MAMBA_USE_HELION` | `1` selects the Helion kernels. Read once per process (`@cache`), so set it before importing anything that dispatches. |
| `MAMBA_HELION_CONFIG_DIR` | Directory holding `<kernel>.json`. A missing config is an assertion failure at first call. |
| `MAMBA_HELION_AUTOTUNE` | Offline only: `1` fills in missing configs (resumable), `2` forces a re-tune and overwrites. Unset at runtime. |

Autotuning must never happen at runtime — not under CUDA graph capture, not in a
distributed job — which is why a missing config fails loudly instead of tuning itself.
The config loaded on first call becomes the kernel's single config, so every later
specialization reuses it.

Note: with `MAMBA_USE_HELION=1` and deterministic mode enabled, `dispatch.use_helion()`
warns once — `chunk_scan_chunk_state_bwd_dx` (ddt, 1-D dD) and `chunk_scan_bwd_dC`
(ddA_cumsum_prev) always reduce with atomic adds, so the backward is not bitwise
reproducible. Use the Triton path when you need determinism.

## Autotuning configs

```sh
python benchmarks/helion/autotune_mamba_mixer.py                    # unpacked, Nemotron3 Nano dims, seq 8k, bf16
python benchmarks/helion/autotune_mamba_mixer.py --sequence-packing # packed (THD) path, needs --batch-size 1
./benchmarks/helion/run_autotune.sh --precompile fork               # same, detached, logs to logs/
```

The harness sets `MAMBA_USE_HELION=1`, `MAMBA_HELION_AUTOTUNE=1` and
`MAMBA_HELION_CONFIG_DIR=benchmarks/helion/configs` by default (explicit env vars win), runs one
`MambaMixerMin` fwd+bwd, then asserts that every dispatched kernel left a
`<kernel>.json` behind — a kernel this configuration never reached would otherwise
surface only later, as a runtime `ensure_helion_config` assertion.

Useful flags: the dimension flags (`--hidden-size`, `--mamba-num-heads`,
`--mamba-head-dim`, `--mamba-state-dim`, `--mamba-num-groups`, `--chunk-size`,
`--seq-len`, `--batch-size`, `--dtype`), the packing flags (`--sequence-packing`,
`--packed-num-seqs`, `--packed-seqlens`), and `--precompile {spawn,fork,off}` — `spawn`
precompiles each candidate config in a subprocess so a config that hard-crashes the CUDA
context is skipped rather than aborting the whole run, at the cost of memory and
cold-start time.

**A config set is specialized to GPU, toolchain, dims, dtype and packing.** Give each
variant its own directory; the naming scheme used here is
`<gpu>-helion<minor>-triton<minor>-<model>-<dtype>-seq<len>[-packed]`, e.g. the
checked-in [`benchmarks/helion/configs/b200-helion1.4-triton3.6-nemotron3-nano-bf16-seq8k/`](../../../benchmarks/helion/configs/b200-helion1.4-triton3.6-nemotron3-nano-bf16-seq8k)
(its `README.md` records the exact environment). Packed configs in particular *cannot*
be reused: a config carries one `indexing` entry per load/store and the packed path adds
the `seq_idx` loads, so an unpacked config fails to compile there.

Configs are repo artifacts, deliberately not packaged: `benchmarks/` is excluded from the
wheel, so a `pip install mamba-ssm` ships no config set and nothing can autotune into the
installed package. Enabling the backend from an installed wheel therefore means bringing
your own directory — autotuned from a source checkout, or copied from one — and pointing
`MAMBA_HELION_CONFIG_DIR` at it.

## Tests

```sh
pytest tests/ops/helion/test_kernels.py                       # uses the checked-in configs
MAMBA_HELION_CONFIG_DIR=<unpacked configs> pytest tests/ops/helion/test_kernels.py
MAMBA_HELION_TEST_PACKED=1 MAMBA_HELION_CONFIG_DIR=<packed configs> \
    pytest tests/ops/helion/test_kernels.py
pytest 'tests/ops/helion/test_kernels.py::test_ssd_kernel_against_triton[unpacked-chunk_scan_fwd]'
```

With `MAMBA_HELION_CONFIG_DIR` unset the tests fall back to the checked-in
`benchmarks/helion/configs/b200-…-seq8k/` set, so a reviewer can run
them with no setup. A config is a set of block sizes and indexing choices rather than
anything GPU-specific, so that set usually compiles and passes on other hardware too — but
if it does not, the failure carries the `autotune_mamba_mixer` command to generate your
own. Autotuning never happens implicitly, in the tests as at runtime. The packed variant
has no checked-in set and asks for one explicitly.

One `MambaMixerMin` fwd+bwd runs with the pipeline on **Triton**, every dispatched kernel
wrapped so both implementations see the same production arguments and their outputs are
compared. The wrapper returns the Triton result, so every later stage still consumes
exact production data and a difference is attributable to the kernel that produced it.
Arguments are deliberately not rebuilt by hand — they are views into shared buffers
(`x`/`B`/`C`/`dx` slice one packed `xBC`, `z` slices `zxbcdt`) and outputs of preceding
stages; Helion compiles per stride, so a contiguous stand-in would test a specialization
that never runs.

Tolerances are one bf16 ulp for `rtol`, and an absolute floor of one ulp of the tensor's
*own* max magnitude — these outputs span orders of magnitude, so any fixed atol is either
impossibly tight for the large tensors or vacuous for the small ones.

One process tests one variant (unpacked by default, packed under
`MAMBA_HELION_TEST_PACKED=1`), because Helion pins a kernel's config on first call and the
two variants need different config sets. `MAMBA_HELION_CONFIG_DIR` must already contain
them: the fixture deletes `MAMBA_HELION_AUTOTUNE`, making a missing config a hard failure
exactly as at runtime.

On the environment above, with the checked-in config set, the unpacked variant is 11 passed
(nine SSD kernels, the gated-norm backward, and the namespace guard) in ~34s — one module
fixture runs the single fwd+bwd that all of them read.

## Benchmark

```sh
python benchmarks/helion/bench_mamba_mixer.py                    # both backends in subprocesses, then compare
python benchmarks/helion/bench_mamba_mixer.py --backend helion   # one trace only
python benchmarks/helion/bench_mamba_mixer.py --compare logs/bench_mamba_mixer_triton.json logs/bench_mamba_mixer_helion.json
```

Runs `MambaMixerMin` fwd+bwd under `torch.profiler` (`--warmup`, `--iters`), writes a
chrome trace per backend to `--trace-dir` (default `logs/`), and attributes every device
kernel to the **dispatch stage** that launched it — per stage rather than per kernel name
because the two backends emit differently named kernels, so `chunk_scan_fwd` on Triton and
on Helion are only comparable through the dispatch wrapper they share. Attribution goes
through each kernel's `correlation` id to its launch API call, and from there to the
enclosing `record_function` interval, which keeps the backward pass (launched on the
autograd thread) correct too.

The report prints two columns per backend: *stage device time* (including the aten helper
kernels a stage launches) and *generated kernel only* (just the Triton/Helion kernel) —
the split matters because the backends leave different amounts of work to aten, e.g.
Triton reduces dD with a separate `reduce_kernel` where Helion uses in-kernel atomics.

Each backend runs in its own subprocess since `use_helion()` is cached per process, and
autotuning is disabled while benchmarking (a missing config fails the run, as at runtime).
Dimension flags mirror the autotune harness — keep them in sync with the config set you
point at.

### Measured example

```sh
CUDA_VISIBLE_DEVICES=1 python benchmarks/helion/bench_mamba_mixer.py
```

Harness defaults (warmup 3, 10 timed iterations) with the checked-in
`benchmarks/helion/configs/b200-helion1.4-triton3.6-nemotron3-nano-bf16-seq8k/` set, on
the [environment above](#validated-environment).

Device time per fwd+bwd iteration, in ms:

| | Triton | Helion | Speedup |
| --- | ---: | ---: | ---: |
| whole iteration | 3.084 | 2.725 | 1.13x |
| SSD + gated-norm stages | 1.295 | 0.935 | **1.39x** |
| everything else (in/out projections, conv, aten) | 1.789 | 1.791 | 1.00x |

Per stage — the left group is the stage's whole device time, the right group only the
Triton/Helion kernel it generates, excluding the aten helpers the stage launches:

| Stage | Triton | Helion | Speedup | Triton<br>(kernel) | Helion<br>(kernel) | Speedup<br>(kernel) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_norm_bwd` | 0.190 | 0.049 | **3.90x** | 0.182 | 0.035 | **5.24x** |
| `chunk_scan_fwd` | 0.174 | 0.138 | 1.26x | 0.174 | 0.138 | 1.26x |
| `chunk_scan_chunk_state_bwd_dx` | 0.169 | 0.146 | 1.16x | 0.155 | 0.132 | 1.17x |
| `chunk_scan_bwd_ddAcs_stable` | 0.142 | 0.112 | 1.27x | 0.137 | 0.104 | 1.32x |
| `chunk_state_bwd_db` | 0.122 | 0.099 | 1.22x | 0.088 | 0.064 | 1.39x |
| `chunk_state_fwd` | 0.121 | 0.115 | 1.05x | 0.121 | 0.115 | 1.05x |
| `chunk_scan_bwd_dC` | 0.114 | 0.083 | 1.38x | 0.089 | 0.058 | 1.53x |
| `state_passing_bwd` | 0.102 | 0.063 | 1.63x | 0.099 | 0.060 | 1.65x |
| `state_passing_fwd` | 0.101 | 0.078 | 1.30x | 0.101 | 0.078 | 1.30x |
| `chunk_scan_bwd_dstates` | 0.059 | 0.052 | 1.13x | 0.059 | 0.052 | 1.13x |
| **TOTAL (stages)** | **1.295** | **0.935** | **1.39x** | **1.206** | **0.836** | **1.44x** |

Speedup is Triton / Helion, so >1 means Helion is faster. Every stage is faster here, the
gated LayerNorm backward by the largest margin; the untouched work around the stages is
unchanged, as expected. The numbers are specific to this GPU, toolchain and config set —
re-run the benchmark for any other combination rather than reading these as general
figures. To re-print a table from traces already in `logs/`, without profiling again:

```sh
python benchmarks/helion/bench_mamba_mixer.py --compare \
    logs/bench_mamba_mixer_triton.json logs/bench_mamba_mixer_helion.json
```
