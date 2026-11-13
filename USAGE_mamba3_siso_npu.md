# Mamba-3 SISO on Ascend NPU: usage notes

Scope: `is_mimo=False` only. The training path (forward + backward) and the inference (decode) path
are both shown as runnable snippets. `is_outproj_norm=True` decode is out of scope: it also needs
`_postprocess` (RMSNorm + `mimo_o`), which the Triton step does not implement.

## 0. Toolchain install: `triton-ascend` and `tilelang-mlir-ascend`

Neither ships with `torch_npu`, and neither works from a plain `pip install` on Ascend.

| Package | What it compiles | Needed for | Version validated here |
|---|---|---|---|
| `triton-ascend` | every Triton kernel of this path (`angle_dt*`, `mamba3_siso_fwd`, the `compute_*` / `angle_dt_bwd` chain, the rotary step) | always | `3.6.0+dev20260921` -- it must match the Triton your `torch` pairs with (`torch 2.12` <-> `triton 3.6`) |
| `tilelang-mlir-ascend` (imports as `tilelang`) | the MIMO forward `mamba3_mimo` | **not** this path -- with `MAMBA3_DISABLE_TILELANG=1` (set in every command below) nothing imports it; only `is_mimo=True` needs it, see `USAGE_mamba3_mimo_npu.md` | `0.1.15+cuda.git0b31c2f4`, i.e. branch `v0.1.15` @ `0b31c2f4`, built with `USE_ASCEND910=ON` |

Validated combination: 910B3, CANN 9.1.0, Python 3.12, `torch 2.12.0+cpu`, `torch_npu 2.12.0.post2`,
`triton-ascend 3.6.0+dev20260921`, `tilelang 0.1.15+cuda.git0b31c2f4`.

### 0.1 `triton-ascend`

The wheels come from Huawei's nightly repo -- **browse it and take the file that matches your
interpreter and CPU arch**:

    https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/triton-ascend/

The file name encodes everything needed:
`triton_ascend-<triton version>+dev<YYYYMMDD>-cp<XY>-cp<XY>-manylinux_2_27_<arch>.manylinux_2_34_<arch>.whl`.

* `3.6.0` is the Triton version -- it must be the one your `torch` pairs with (`torch 2.12` <-> `3.6`).
* `cp312` is the interpreter (`cp310`..`cp314` are published), `aarch64` / `x86_64` the arch
  (`aarch64` on 910B3 hosts).
* `+dev<YYYYMMDD>` is just the nightly build date; take a recent one. This document was measured with
  `3.6.0+dev20260921` (cp312, aarch64).
* The newest builds carry `manylinux_2_34`: they need glibc >= 2.34. Older ones are `manylinux_2_28`.

```bash
# install the URL you picked from the page above, e.g.
pip install https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/triton-ascend/triton_ascend-3.6.0+dev20260921-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_34_aarch64.whl

# or let pip resolve from that repo
pip install triton-ascend --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/
```

From source, following the vendor's own steps (repo `https://gitcode.com/Ascend/triton-ascend`):

```bash
# Ubuntu 22.04 package names; other distros: clang >= 15 + lld, then point CC/CXX at them
apt install zlib1g-dev clang-15 lld-15
update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-15   100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
pip install ninja cmake wheel pybind11
git clone https://gitcode.com/Ascend/triton-ascend.git && cd triton-ascend
pip install -e .
# custom LLVM: run scripts/build-llvm-project.sh first, then export LLVM_INSTALL_PREFIX=<prefix>
```

`python -c "import triton; print(triton.__version__)"` must print the version `torch` expects. A
mismatched pair does not fail at import -- it fails at the first kernel launch, so section D is the
real check.

### 0.2 `tilelang-mlir-ascend`

Only `is_mimo=True` needs it, but it is documented here because one container usually serves both
paths (`USAGE_mamba3_mimo_npu.md` is the document that runs it).

A wheel built without `USE_ASCEND910=ON` -- e.g. the default PyPI `tilelang` -- has no `ascend910`
backend at all. That backend is a build-time switch
(`-DUSE_ASCEND910=ON`) that hard-requires a BishengIR = AscendNPU-IR **install tree**:
`bin/bishengir-{compile,opt}`, `include/mlir/IR/BuiltinOps.h`, `lib/*.a` (see
`src/ascend910/CMakeLists.txt`). CANN 9.1.0 ships bishengir binaries and bitcode only, so that tree
has to be built first.

**Source.** `https://github.com/tile-ai/tilelang-mlir-ascend/tree/v0.1.15` -- branch `v0.1.15`, HEAD
`0b31c2f4` (the exact commit the wheel we run was built from). Clone it, or download that branch as an
archive and unpack it; the tree is then compiled in place (stage 2). There is no PyPI / nightly wheel
that carries the `ascend910` backend, which is why this one has to be built.

**Stage 1 -- AscendNPU-IR.** Use the commit that tilelang `main` pins,
`31f690369d1247fbd5529a3f88b758f7d470ae4f`: `v0.1.15` does not pin it at all, and the newer tags
(`v1.1.0`, `v1.2.0`, master 2.0.0) dropped builder parameters that `src/ascend910` still passes
(`FixpipeOp`'s `enable_nz2nd`), so they do not compile against this tilelang.

```bash
git clone https://github.com/Ascend/AscendNPU-IR.git      # or the gitcode mirror
cd AscendNPU-IR && git checkout 31f690369d1247fbd5529a3f88b758f7d470ae4f
git submodule update --init --depth 1
export CC=/usr/bin/clang CXX=/usr/bin/clang++
./build-tools/build.sh -o build \
    --c-compiler=clang --cxx-compiler=clang++ \
    --add-cmake-options="-DCMAKE_LINKER=lld -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_RTTI=ON" \
    --apply-patches --bishengir-publish=off -j $(nproc)
cd build && cmake --install . --prefix ./install
```

Budget ~2.3 GB of source plus ~1.7 GB of build directory. Watch the install step: an interrupted
`cmake --install` can silently leave whole header trees out, so check that `bin/bishengir-compile`,
`include/mlir/IR/BuiltinOps.h`, `include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.h.inc` (it must contain
`enable_nz2nd`) and a non-empty `lib/*.a` are all present before moving on.

**Stage 2 -- the wheel.**

```bash
# source: https://github.com/tile-ai/tilelang-mlir-ascend/tree/v0.1.15
#        (branch v0.1.15, HEAD 0b31c2f4; "Code -> Download ZIP" there works too)
git clone -b v0.1.15 https://github.com/tile-ai/tilelang-mlir-ascend.git tl && cd tl
python -m pip wheel . --no-deps --no-build-isolation -w dist \
    --config-settings=cmake.define.USE_ASCEND=ON \
    --config-settings=cmake.define.USE_ASCEND910=ON \
    --config-settings=cmake.define.USE_CUDA=OFF \
    --config-settings=cmake.define.BISHENGIR_ROOT_PATH=<AscendNPU-IR>/build/install \
    --config-settings=cmake.define.CMAKE_CXX_FLAGS=-DBSPUB_DAVINCI_BISHENGIR_A5
pip install dist/tilelang-0.1.15+cuda.git0b31c2f4-cp39-abi3-linux_aarch64.whl
```

`-DBSPUB_DAVINCI_BISHENGIR_A5` is required, not cosmetic: AscendNPU-IR configures itself with
`LLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON`, and its installed `mlir/Interfaces/CallInterfaces.h` declares
`mlir::call_interface_impl` only under that macro -- a consumer without it fails with
`no member named 'call_interface_impl' in namespace 'mlir'`.

**Stage 3 -- which `bishengir-compile` runs.** This is *not* the tree built in stage 1: stage 1 is what
links the wheel, this is the compiler binary tilelang spawns for every kernel at run time. Resolution
order in `tilelang/ascend910/compiler.py`: `$TILELANG_NPU_COMPILER_PATH` (must be a **file**), then
`$ASCEND_HOME_PATH/tools/bishengir/bin/bishengir-compile`, then `PATH`. The compiler that ships with
CANN 9.1.0 crashes intermittently on A2 kernels with `Return code: -11` (SIGSEGV);
`3rdparty/bishengir/` carries a patched build of the same 1.2.0 version (byte-identical objects for the
kernels that do compile). Decide how to get it **before** installing anything -- the preflight check
runs on the target machine, takes 30 s, and the tarball does not even have to be there yet:

```bash
bash 3rdparty/bishengir/check_target_abi.sh     # aarch64 / glibc >= 2.34 / GLIBCXX_3.4.30 / CANN
```

| Preflight result | Which way |
|---|---|
| `exit 0` (no `[FAIL]`) | **A** -- use the prebuilt package, ~5 min |
| `exit 1` (any `[FAIL]`) | **B** -- build it on this machine, ~35 min; the prebuilt binary is linked against glibc 2.35, so an older glibc cannot be shimmed |

**A. the prebuilt package.**

```bash
sha256sum 3rdparty/bishengir/bishengir-patched-1.2.0.tar.gz
# e169b46c1687d0458fd4c4943f17c9fcc612033638d2ce9c16ec79f53ecf3a7e  <- the contract
tar -xzf 3rdparty/bishengir/bishengir-patched-1.2.0.tar.gz -C /opt/
export TILELANG_NPU_COMPILER_PATH=/opt/bishengir-patched-1.2.0/bin/tl-bishengir-compile
bash 3rdparty/bishengir/verify_patched_bishengir.sh /opt/bishengir-patched-1.2.0
```

The package is self-locating (the wrapper derives its root from `readlink -f $0`), so it can be
unpacked anywhere; `libstdc++` is bundled in its `lib/`.

**B. build it on this machine.**

```bash
git clone --depth 1 --branch v1.2.0 --recurse-submodules --shallow-submodules \
    https://gitcode.com/Ascend/AscendNPU-IR.git          # branch must be v1.2.0; ~2.3 GB with submodules
bash 3rdparty/bishengir/build_patched_bishengir.sh \
    --src AscendNPU-IR --out /opt/bishengir-patched --jobs $(nproc)
export TILELANG_NPU_COMPILER_PATH=/opt/bishengir-patched/bin/tl-bishengir-compile
bash 3rdparty/bishengir/verify_patched_bishengir.sh /opt/bishengir-patched
```

The script does preflight -> patch -> `build.sh` -> assemble `pkg/` -> acceptance check, and copies the
14 device-side `.bc` files out of this machine's CANN (without them the compiler dies with
`External hivmc run fails`); `--dry-run` stops after patching.

Both branches end in the same acceptance check: the compiler reports its version, a real cube kernel
compiles to an `.o` with `rc=0`, a minimal IR does not crash, and the effective
`TILELANG_NPU_COMPILER_PATH` is printed. If the *stock* CANN compiler also passes the cube-kernel case,
you are still on the stock compiler -- check where the variable points (`rc=139` is the stock failure).
If neither branch fits the machine, leave the variable unset and accept that a kernel compile may fail
and need a retry.

Point it at the `tl-bishengir-compile` **wrapper** (not the bare binary), leave
`TILELANG_NPU_OPT_PATH` unset, and put the `export` in your environment script rather than in the
current shell. Rollback is `unset TILELANG_NPU_COMPILER_PATH`. Details and troubleshooting:
`3rdparty/bishengir/README.md`.

Finally check that the A2 backend is really inside the wheel -- this is the same probe the MIMO step
suite uses to decide skip vs run:

```bash
python -c "import tilelang, tilelang.ascend910.language; print(tilelang.__version__)"
```

## A. Training: forward + backward
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
test version: `tests/ops/triton/test_mamba3_siso_module_decode_npu.py`.

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

### B4. Precision contract
**Equivalence (main conclusion).** Against the fp32/fp64 anchors there is no regression over the 831 comparisons, and no output passes its anchor on A100 while failing on NPU.

**Absolute CANN gate.** 418/831 do not clear the absolute gate, but all of them are A100-identical, i.e. they are a property of the bf16 chain rather than of the port. Never read "NPU vs A100 FAIL" as an NPU defect: such a comparison must always quote A100 vs anchor as well.

**Per-op status (representative).** `angle_dt_fwd` PASS (24/24) | `mamba3_siso_step` PASS | `grouped_head_reduction` PASS | `compute_dzdo` PASS (15/15) | `compute_dqktheta` PASS (19/19) | `angle_dt_bwd` PASS (36/36) |
`compute_ddt_dtrap_dinput_states` PASS (real chain fp32, 26/26; the OFF fp16 direct-call failures are not the real chain) | `mamba3_siso_fwd` conditional PASS: `Out` 35/43, `final_v` 43/43, `final_k` 42/43 (the single failure is an fp16-input case) | `compute_dqkv` 90/90 against the fp32 anchor.

**`Final_SSM_State` (dtype contract, already ruled on).** It is still declared `torch.float32`, but the recurrence requantises it in place in the input dtype (`mamba3_siso_fwd.py:576-581`, declared at `:769`), so a bf16 chain is effectively bf16; it passes 43/43 under the bf16 criterion and the code stays fp32 (unchanged). Caveat: that 43/43 is an **A100-side** measurement plus NPU CPU-probe inference, not an NPU re-measurement.

**Inference.** Prefill + decode step PASS in bf16 (decode step through both the Ascend C `mamba3_decode_step` path and the previous Triton path; `tests/ops/triton/test_mamba3_siso_npu.py` is green); fp16/fp32 inference is not supported.

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
| module `mamba3_decode_step_ext` with `validate` / `step` / `launch_mode` | host-side argument validation; `step` validates and launches inside one pybind11 call and falls back to the two-step path when the symbol is absent; `launch_mode` reports which launch form the current thread took (1 = self-managed `aclrtLaunchKernelWithHostArgs`, 0 = generated stub, -1 = not resolved yet, i.e. before the second call) |

`op_extension/mamba3_decode_step_validate.cpp` is what defines the pybind11 module, so it has to be part of the
build. A .so compiled without it still registers the torch op but has no `mamba3_decode_step_ext` (see C5).

### C4. Verifying a build (smoke first, then the suites)

```bash
# 1) The operator on its own: every case, NPU vs CPU float64 golden
#    (1000 precision cases + negative cases + determinism repeats)
python mamba_ssm/ops/ascendc/scripts/selftest.py --out /tmp/selftest_results.json

# 2) Just to confirm the .so loads and computes: filter by case id (about 40 s; every run also
#    executes the 19 negative cases and the 3 determinism cases).
#    Always pass --out explicitly, otherwise the default file is written to the current directory.
python mamba_ssm/ops/ascendc/scripts/selftest.py --list
python mamba_ssm/ops/ascendc/scripts/selftest.py --filter p0001 --out /tmp/selftest_p0001.json

# 3) The whole SISO chain (angle_dt + decode step + combined fwd/bwd), i.e. the section B acceptance suite
PYTHONPATH=$PWD python tests/ops/triton/test_mamba3_siso_npu.py

# 4) Both entry points present, and the launch is self-managed
#    (launch_mode: 1 = aclrtLaunchKernelWithHostArgs, 0 = generated stub, -1 = not resolved yet)
PYTHONPATH=$PWD python - <<'PY'
import importlib.util
import sys

import torch
import torch_npu  # noqa: F401
import mamba_ssm.ops.ascendc.mamba3_step as _st

m = sys.modules["mamba_ssm.ops.ascendc.mamba3_step"]
spec = importlib.util.spec_from_file_location("mamba3_decode_step_ext", str(m._lib_path()))
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
print("entries:", [a for a in ("validate", "step", "launch_mode") if hasattr(ext, a)])
print("launch_mode:", ext.launch_mode())   # -> -1 until one launch happened, then 1
PY
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

Three suites cover the paths this document describes. Upstream has no counterpart for any of them:
they are our own. The first two are runnable scripts (`python <file>`) that print a measured report
and exit non-zero on failure; the third is a pytest module. All three need `torch_npu`,
`triton-ascend` and, for the decode step, the built `libmamba3_decode_step_ops.so` (section C).

| File | What it checks |
|---|---|
| `tests/ops/triton/test_mamba3_siso_npu.py` | Operator level, bf16 on `npu`, against the upstream PyTorch reference: `angle_dt`, a 64-step decode recurrence (the four states are carried across steps), and combined chunked forward + backward with every gradient. `MAMBA3_SISO_STEP_IMPL=ascendc` (default) or `triton` selects which decode step is measured, so one run per value gives the A/B comparison. |
| `tests/ops/triton/test_mamba3_siso_module_decode_npu.py` | Module level: `Mamba3` prefill and token-by-token decode must agree, same device, teacher-forced input. It drives the Triton step (not the Ascend C operator) and skips itself when no accelerator is present. |
| `tests/ops/triton/test_mamba3_siso_e2e_npu.py` | Module level, **the only end-to-end suite for SISO**: a real `Mamba3` (`d_model=128`, `d_state=64`, `headdim=32`, `ngroups=2`, bf16) is deep-copied and one copy runs with `mamba3_siso_combined` swapped for the frozen fp32 `mamba3_siso_fwd_ref`, so `in_proj`, the B/C norms, `out_proj` and the ADT/DT/angle preprocessing cancel out and only the SISO kernel chain is measured -- the dtype contract between module and kernel (Q/K/V/Z cast to bf16 at the wrapper entry, `trap` kept fp32) is exactly what this catches. 3 cases: batched forward + backward at `B=2, S=128` (`out`, `du` and 8 parameter gradients, plus peak NPU memory), a larger forward at `B=1, S=512` (UB / memory pressure), and varlen (`cu_seqlens=[40, 64, 24]`). Gates on the upstream criterion -- p95 relative error over `|ref| >= 1e-2` at `rtol=1e-1`, with the globally normalised max for the per-channel scalars -- while the guide and CANN lines are printed for every tensor. |

```bash
# the third suite (pytest; ~30 s once the Triton kernels are cached, ~10 min on a cold cache)
MAMBA3_DISABLE_TILELANG=1 ASCEND_RT_VISIBLE_DEVICES=0 \
    python -m pytest -q -s -p no:cacheprovider tests/ops/triton/test_mamba3_siso_e2e_npu.py
```

The upstream CUDA suite `tests/ops/triton/test_mamba3_siso.py` is untouched and stays CUDA-only; the
reference implementations the first suite compares against are imported from it, while the third
carries its own frozen copy of the reference (taken from `_siso_ops_npu_common.py`) so that it runs
standalone.

The MIMO suites (`is_mimo=True`, including the decode step's MIMO path) live in their own document:
`USAGE_mamba3_mimo_npu.md`.