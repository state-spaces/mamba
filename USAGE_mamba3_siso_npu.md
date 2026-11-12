# Mamba-3 SISO on Ascend NPU: usage notes

Scope: `is_mimo=False` only. The training path (forward + backward) and the inference (decode) path
are both shown as runnable snippets. `is_outproj_norm=True` decode is out of scope: it also needs
`_postprocess` (RMSNorm + `mimo_o`), which the Triton step does not implement.

## Validation status

- The merged operator code is **byte-identical** to the PLUT-82 validated version: `e30337d..d7c6f08`
  touched only 2 test files, and that version was validated with 148 cases (147 passed + 1 registered
  xfail). The CANN-side evidence therefore applies to this code as-is.
- **CUDA was not re-tested**; the user accepted this as a known residual risk on 2026-10-22. Be aware
  that two dtype normalisations in `mamba3_siso_combined.py` (`Trap` and `Angles`, both unconditionally
  `fp32`) also take effect on CUDA and change its numbers relative to upstream. The per-change analysis
  is kept on the `mamba-main-npu` branch (`NPU_MIGRATION_mamba3_siso.md`), not in this repository.

## Environment (measured, read-only)

| Package / component | Version (verbatim) |
|---|---|
| python | `3.12.0` |
| torch | `2.12.0+cu130` (dist metadata `2.12.0`; `torch.version.cuda=13.0`) |
| torch_npu | `2.12.0.post2` |
| triton | `3.6.0` (this string is overwritten by triton-ascend; do not use it alone to identify the backend) |
| triton-ascend | `3.6.0+dev20260921` |
| CANN toolkit / opp | `9.1.0` / `9.1.0` (toolkit `timestamp=20260730_231653901`) |

Notes:

1. `torch 2.12.0+cu130` is a CUDA 13.0 wheel; on NPU it is used after `torch_npu 2.12.0.post2` patches
   it -- this is not a wrong install.
2. `triton-ascend` and `triton` are **two packages installed into the same `triton/` directory**:
   `triton-3.6.0.dist-info/RECORD` has 0 `extra/cann` entries, while
   `triton_ascend-3.6.0+dev20260921.dist-info/RECORD` has 34, and the latter also overwrites
   `triton/__init__.py` -- so `IS_CANN=True` depends on the **triton-ascend** version, not on
   `triton.__version__`.

## A. Training: forward + backward

### A0. NPU prerequisites (run once, before any snippet below)
```python
import os
os.environ.setdefault("MAMBA3_DISABLE_TILELANG", "1")   # skip the tilelang MIMO import (cannot build on Ascend)

# The three shims applied by scripts/infer_siso_step.py:_setup_npu() for every NPU run.
# Library code has no equivalent guard yet, so apply them yourself:
try:                                                    # Triton-Ascend is not the CUDA driver
    import triton.backends.nvidia.driver as nv_driver
    nv_driver.CudaDriver.is_active = staticmethod(lambda: False)
except Exception:
    pass
import torch_npu  # noqa: F401                          # registers torch.npu
torch.npu.set_device(0)
try:                                                    # replace the probe only if it returns None
    if torch.cuda.get_device_capability(torch.device("npu")) is None:
        torch.cuda.get_device_capability = lambda *a, **k: (0, 0)
    # if it returns a real capability, leave the original function untouched
except Exception:
    pass
```

Why each one: without `MAMBA3_DISABLE_TILELANG=1` the tilelang MIMO import raises a
non-`ImportError` on Ascend (e.g. `PassConfigKey.TL_ENABLE_FAST_MATH`) and the module import fails;
without the `CudaDriver.is_active` patch Triton probes an absent CUDA driver. The third shim is a
**measured requirement whose exact failure path is not yet localized** -- it is deliberately *not*
attributed to `mamba3_siso_fwd`'s `early_config_prune`, because that callback returns early on any
non-CUDA device (`mamba3_siso_fwd.py:111-113`) and never reaches `torch.cuda.get_device_capability`;
the likely consumer is Triton's own device probing on Ascend. It is applied because it is required
in practice -- confirm the reason by A/B-disabling it on NPU before folding any of these shims into
`npu_compat` (deferred until then).

**Path prerequisite** (also measured): `mamba_ssm` is **not installed as a package** -- a bare
`import mamba_ssm` from outside the tree raises `ModuleNotFoundError`. Run the snippets below from the
repository root, or put the repo root on `PYTHONPATH`.

```python
import torch
from mamba_ssm.modules.mamba3 import Mamba3

model = Mamba3(
    d_model=128, d_state=16, expand=2, headdim=16, ngroups=1,
    is_mimo=False, is_outproj_norm=False, chunk_size=64,
    dtype=torch.bfloat16, device="npu",
)
u = torch.randn(2, 128, 128, dtype=torch.bfloat16, device="npu", requires_grad=True)
y = model(u)                        # (b, l, d_model)
loss = y.float().square().mean()
loss.backward()
print(y.shape, y.dtype, u.grad is not None)
```

Two constraints:

1. **The chunk size is clamped.** On NPU `chunk_size` is capped at <=64 by `effective_chunk_size`, and for `seqlen >= 64` it is reduced further to 32. Chunking is **mathematically equivalent but not bitwise equivalent**: the number of `mod 2*pi` folds changes with the chunk size, and so does the cross-chunk accumulation order.
2. **The measured safe upper bound for `RMSNormGated`'s `group_size` is 8192** (with `is_outproj_norm=True` this is `headdim`). Exceeding it is a **compile-time** `ub overflow`, never a silent wrong result; `16384` measured **1.83x** over the limit.

## B. Inference (decode): `mamba3_siso_step` on NPU
Upstream `Mamba3.step()` depends on the cutlass/CuteDSL `mamba3_step_fn`, which has **not been migrated to
Ascend**. The import guard in `mamba3.py` leaves it as `None`, so `forward(inference_params=...)` with
`seqlen_offset > 0` dispatches into `step()` and **fails an assertion**. On NPU, call the SISO decode step
directly: keep using the module for prefill (internally `mamba3_siso_combined(..., return_final_states=True)`)
to obtain the four states, then call step once per token. Runnable example: `scripts/infer_siso_step.py`;
test version: `tests/ops/triton/test_mamba3_siso_npu.py`.

Two interchangeable implementations of the single-token step (same signature, same
`(out, [angle_state, ssm_state, k_state, v_state])` return, so they are drop-in replacements):

| Implementation | Import path | Notes |
|---|---|---|
| **Ascend C (default on NPU)** | `mamba_ssm.ops.ascendc.mamba3_siso_step` | The step itself runs the NPU kernel-direct operator `mamba3_decode_step` (i.e. `mamba3_step_fn`); rotary / angle-state reuse the Triton `apply_rotary_qk_inference_fwd`. Build `build/libmamba3_decode_step_ops.so` first (see section C) |
| Triton (previous implementation, kept) | `mamba_ssm.ops.triton.mamba3.mamba3_siso_step` | Pure Triton, unchanged; useful as an A/B reference |

`tests/ops/triton/test_mamba3_siso_npu.py` uses Ascend C by default; set `MAMBA3_SISO_STEP_IMPL=triton` to switch back.

### B1. Prefill: obtain the four states
```python
from types import SimpleNamespace
import torch
from mamba_ssm.modules.mamba3 import Mamba3

model = Mamba3(d_model=128, d_state=16, expand=2, headdim=16, ngroups=1,
               is_mimo=False, is_outproj_norm=False, chunk_size=64,
               dtype=torch.bfloat16, device="npu", layer_idx=0).eval()
u = torch.randn(2, 64, 128, dtype=torch.bfloat16, device="npu")

cache = SimpleNamespace(seqlen_offset=0, key_value_memory_dict={})
with torch.no_grad():
    out_prefill = model(u, inference_params=cache)  # internally: combined(..., return_final_states=True)
a, s, k, v = cache.key_value_memory_dict[0]
states = (a, s, k.squeeze(1).contiguous(), v)       # k: (b, 1, H, N) -> (b, H, N)
```

`angle`, `ssm` and `k` are fp32; `v` uses `V.dtype` (bf16). Do **not** feed the bf16 `k_state` from `allocate_inference_cache` into this path: it belongs to the cutlass route -- it type-checks but silently loses precision.

### B2. Per-token decode
```python
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.modules.mamba3 import heavy_tail_activation
from mamba_ssm.ops.ascendc.mamba3_siso_step import mamba3_siso_step   # default on NPU (Ascend C)
# from mamba_ssm.ops.triton.mamba3.mamba3_siso_step import mamba3_siso_step   # previous Triton path, interchangeable

NSPLIT = [
    model.d_inner, model.d_inner,
    model.d_state * model.num_bc_heads * model.mimo_rank,
    model.d_state * model.num_bc_heads * model.mimo_rank,
    model.nheads, model.nheads, model.nheads, model.num_rope_angles,
]


def decode_step(model, u_t, states):
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(model.in_proj(u_t), NSPLIT, dim=-1)
    _A = torch.clamp(-heavy_tail_activation(dd_A.to(torch.float32)), max=-model.A_floor)
    DT = F.softplus(dd_dt + model.dt_bias)
    ADT = _A * DT
    B = rearrange(B, "b (r g n) -> b r g n", r=model.mimo_rank, g=model.num_bc_heads)
    C = rearrange(C, "b (r g n) -> b r g n", r=model.mimo_rank, g=model.num_bc_heads)
    B = model.B_norm(B).expand(-1, -1, model.nheads, -1)
    C = model.C_norm(C).expand(-1, -1, model.nheads, -1)
    x = rearrange(x, "b (h p) -> b h p", p=model.headdim)
    z = rearrange(z, "b (h p) -> b h p", p=model.headdim)
    angles = angles.unsqueeze(-2).expand(-1, model.nheads, -1).to(torch.float32)
    out, states = mamba3_siso_step(
        C.squeeze(1), B.squeeze(1), x, ADT, DT, trap,     # trap is the pre-sigmoid projection
        model.C_bias.squeeze(1), model.B_bias.squeeze(1), angles,
        model.D, z, states,
    )
    out = model.out_proj(rearrange(out, "b h p -> b (h p)").to(x.dtype))
    return out, states


outs = []
with torch.no_grad():
    for t in range(16, u.shape[1]):     # first 16 tokens were used for prefill
        o, states = decode_step(model, u[:, t], states)
        outs.append(o)
out_decode = torch.stack(outs, dim=1)   # (b, l - 16, d_model)
print(out_decode.shape, out_decode.dtype, out_prefill.shape)
```

Extra shape constraints of the Ascend C path (violations are reported by `mamba3_step_fn` itself, never silently):
`d_state % 8 == 0`; `headdim_v` must be divisible by one of `32/64/128` (`tile_D` is picked automatically);
`headdim_qk % 2 == 0`; NPU only. `D=None` is treated as zeros, whereas the Triton path simply omits the `D*v` term.

### B3. Four traps (all fail **silently**; none of them raises)

1. `Trap` must be the projection **before** sigmoid (the kernel applies sigmoid itself). Passing the post-sigmoid value sigmoids twice: measured p95 degrades from `2.1e-2` to `1.28` (x62).
2. `C/B` must be **normalised before expanding** (`C_norm/B_norm` -> `expand(-1,-1,nheads,-1)`); skipping the norm measured x71.
3. `Angles` must be the **raw per-token angles (fp32)**; do not cumsum them yourself, and do **not** call `apply_rotary_qk_inference_fwd` (rotation happens inside the kernel).
4. The state order is `(angle, ssm, k, v)`; the prefill `Final_K_State` needs `.squeeze(1)` to drop the mimo-rank axis.

### B4. Precision contract (summary of the PLUT-62 final report v3)
**Equivalence (main conclusion).** Against the fp32/fp64 anchors migration equivalence passes: `real_defect = 0` over 831 primary verdicts, and there is no output that passes its anchor on A100 but fails on NPU.

**Absolute CANN gate.** 418/831 do not clear the absolute gate, but all of them are A100-identical and are delivered with an explanation and an exemption list signed off item by item; this is not an NPU-only regression. Never read "NPU vs A100 FAIL" as an NPU defect: such a comparison must always quote A100 vs anchor as well.

**Per-op status (representative).** `angle_dt_fwd` PASS (24/24) | `mamba3_siso_step` PASS | `grouped_head_reduction` PASS | `compute_dzdo` PASS (15/15) | `compute_dqktheta` PASS (19/19) | `angle_dt_bwd` PASS (36/36) |
`compute_ddt_dtrap_dinput_states` PASS (real chain fp32, 26/26; the OFF fp16 direct-call failures are not the real chain) | `mamba3_siso_fwd` conditional PASS: `Out` 35/43, `final_v` 43/43, `final_k` 42/43 (the single failure is an fp16-input case) | `compute_dqkv` 90/90 against the fp32 anchor.

**`Final_SSM_State` (dtype contract, already ruled on).** It is still declared `torch.float32`, but the recurrence requantises it in place in the input dtype (`mamba3_siso_fwd.py:576-581`, declared at `:769`), so a bf16 chain is effectively bf16; it passes 43/43 under the bf16 criterion and the code stays fp32 (unchanged). Caveat: that 43/43 is an **A100-side** measurement plus NPU CPU-probe inference, not an NPU re-measurement.

**Inference.** Prefill + decode step PASS in bf16 (decode step through both the Ascend C `mamba3_decode_step`
path and the previous Triton path; `tests/ops/triton/test_mamba3_siso_npu.py` is green); fp16/fp32 inference is
not supported.

## C. Building the `mamba3_decode_step` operator (Ascend C kernel-direct .so)

The Ascend C path in section B needs the Python extension `libmamba3_decode_step_ops.so`, which wraps
`mamba3_decode_step` (the operator behind `mamba3_step_fn`). **That .so is not in the repository and is not
shipped with the package**: build it once per machine and once per conda env.

### C1. Prerequisites

| Item | Requirement |
|---|---|
| CANN | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` has been run and `ASCEND_HOME_PATH` is set (built and tested against CANN 9.1.0) |
| Compiler | the `bisheng` / `ccec` binaries bundled with CANN (inside the toolkit; `set_env.sh` puts them on `PATH`, nothing else to install) |
| cmake | >= 3.16 |
| python | **the same env you run in**, with `torch` and `torch_npu` installed. The .so links against that env's libtorch, so a different env means a rebuild |

### C2. Build

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd mamba_ssm/ops/ascendc
PYTHON=/path/to/conda/envs/<env>/bin/python bash build.sh           # normal
PYTHON=/path/to/conda/envs/<env>/bin/python bash build.sh --clean   # after switching env, or if the build cache looks stale
```

- Output: `mamba_ssm/ops/ascendc/build/libmamba3_decode_step_ops.so`
- `PYTHON` falls back to whatever `python` is on `PATH`: **always pass it explicitly** on a machine with several
  envs, otherwise cmake may bind the wrong torch
- A full `--clean` rebuild takes roughly 40 s on an 8-core host and is dominated by the `op_kernel/*.asc` pass
  through bisheng; changes that only touch Python (`mamba3_step.py` / `mamba3_siso_step.py`) need no rebuild

### C3. Loading

`mamba_ssm/ops/ascendc/mamba3_step.py` loads the library lazily on first call and looks for
`<repo>/mamba_ssm/ops/ascendc/build/libmamba3_decode_step_ops.so` by default. To use a different location:

```bash
export MAMBA3_DECODE_STEP_LIB=/abs/path/to/libmamba3_decode_step_ops.so
```

The same .so exposes two entry points, and the Python side uses both:

| Entry | Used for |
|---|---|
| `torch.ops.npu.mamba3_decode_step` | the dispatch path, registered through `torch.ops.load_library` |
| module `mamba3_decode_step_ext` with `validate` / `step` / `launch_mode` | host-side argument validation; `step` validates and launches inside one pybind11 call and falls back to the two-step path when the symbol is absent; `launch_mode` reports which launch form the current thread took |

`op_extension/mamba3_decode_step_validate.cpp` is what defines the pybind11 module, so it has to be part of the
build. A .so compiled without it still registers the torch op but has no `mamba3_decode_step_ext` (see C5).

### C4. Verifying a build

```bash
# Smoke: confirm the .so loads and computes, with the smallest legal shape
# (d_state % 8 == 0, headdim divisible by tile_D; shapes are exactly the operator contract)
PYTHONPATH=$PWD python - <<'PY'
import torch
import torch_npu  # noqa: F401
from mamba_ssm.ops.ascendc import mamba3_step_fn

B, R, H, D, N = 1, 1, 2, 32, 16
f = lambda *s: torch.randn(*s, device="npu", dtype=torch.float32)
state = f(B, H, D, N)
state_out = torch.empty_like(state)
out = torch.empty(B, H, D, device="npu", dtype=torch.float32)
mamba3_step_fn(
    state,                # state    (B, H, D, N)
    f(B, R, H, N),        # Bstate   (B, R, H, N)
    f(B, H, D),           # Xstate   (B, H, D)
    f(B, H),              # A        (B, H)
    f(B, R, H, N),        # B        (B, R, H, N)
    f(B, R, H, N),        # C        (B, R, H, N)
    f(H),                 # D        (H,)
    f(B, H, D),           # x        (B, H, D)
    f(B, H),              # dt       (B, H)
    f(B, H),              # trap     (B, H)
    f(R, H, D),           # xproj    (R, H, D)
    outproj=f(R, H, D), state_out=state_out, out=out, tile_D=32,
)
torch.npu.synchronize()
print("ok", tuple(out.shape), float(out.abs().sum()))   # -> ok (1, 2, 32) <sum>

# Both entry points, and the launch form this call took
#   launch_mode() -> 1 = aclrtLaunchKernelWithHostArgs (fast path), 0 = generated stub, -1 = no launch yet
import importlib.util, sys
m = sys.modules["mamba_ssm.ops.ascendc.mamba3_step"]
spec = importlib.util.spec_from_file_location("mamba3_decode_step_ext", str(m._lib_path()))
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
print("entries:", [a for a in ("validate", "step", "launch_mode") if hasattr(ext, a)])
print("launch_mode:", ext.launch_mode())   # -> -1 on the very first call (kernel handle resolved), 1 from the 2nd on
PY

# The whole SISO chain (angle_dt + decode step + combined fwd/bwd), i.e. the section B acceptance suite
PYTHONPATH=$PWD python tests/ops/triton/test_mamba3_siso_npu.py
```

### C5. Common failures

The error strings in the left column are quoted verbatim from the code, so they match what you will see.

| Symptom | Cause / fix |
|---|---|
| `ERROR: ASCEND_HOME_PATH 未设置，请先 source CANN 的 set_env.sh` | `set_env.sh` was not sourced; `build.sh` refuses to run |
| cmake cannot find `torch` / `Python3` | `PYTHON` points at the wrong env, or that env has no torch installed |
| `undefined symbol` / `ImportError` at `torch.ops.load_library` | the .so was built against a different torch ABI: rebuild with the target env (`--clean`) |
| `mamba3_decode_step: 未找到算子动态库 <path>；请先执行 bash .../build.sh 编译，或通过环境变量 MAMBA3_DECODE_STEP_LIB 指定 .so 路径` | no artifact at the default path, or `MAMBA3_DECODE_STEP_LIB` points at a missing file |
| results change after switching env | a stale .so is still there: `bash build.sh --clean` |
| `ImportError: dynamic module does not define module export function (PyInit_mamba3_decode_step_ext)` | the .so was built without `op_extension/mamba3_decode_step_validate.cpp` (older source set or CMakeLists): rebuild from this tree with `--clean` |

## D. Test suites shipped here

Two NPU suites ship with this repository. Upstream has no counterpart for either: they are our own,
and both are runnable scripts (`python <file>`) that print a measured report and exit non-zero on
failure. Both need `torch_npu`, `triton-ascend` and, for the decode step, the built
`libmamba3_decode_step_ops.so` (section C).

| File | What it checks |
|---|---|
| `tests/ops/triton/test_mamba3_siso_npu.py` | Operator level, bf16 on `npu`, against the upstream PyTorch reference: `angle_dt`, a 64-step decode recurrence (the four states are carried across steps), and combined chunked forward + backward with every gradient. `MAMBA3_SISO_STEP_IMPL=ascendc` (default) or `triton` selects which decode step is measured, so one run per value gives the A/B comparison. |
| `tests/ops/triton/test_mamba3_siso_module_decode_npu.py` | Module level: `Mamba3` prefill and token-by-token decode must agree, same device, teacher-forced input. It drives the Triton step (not the Ascend C operator) and skips itself when no accelerator is present. |

The upstream CUDA suite `tests/ops/triton/test_mamba3_siso.py` is untouched and stays CUDA-only; the
reference implementations the first suite compares against are imported from it.