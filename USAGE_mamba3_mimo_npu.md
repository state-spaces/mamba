# Mamba-3 MIMO on Ascend NPU: usage notes

Scope: `is_mimo=True`. Companion to `USAGE_mamba3_siso_npu.md`: only what differs from SISO is spelled
out here, and that document's section C (building `libmamba3_decode_step_ops.so`) applies unchanged.

Measured on Ascend 910B3 / CANN 9.1.0, `torch` 2.12.0+cpu, `torch_npu` 2.12.0.post2, `tilelang`
0.1.15 (ascend910 backend), `triton` 3.6.0.

## 0. Toolchain install: `triton-ascend` and `tilelang-mlir-ascend`

Neither ships with `torch_npu`, and neither works from a plain `pip install` on Ascend.

| Package | What it compiles | Needed for | Version validated here |
|---|---|---|---|
| `triton-ascend` | every Triton kernel of these paths (`angle_dt*`, `mamba3_siso_fwd`, the `compute_*` / `angle_dt_bwd` chain, the rotary step) | always | `3.6.0+dev20260921` -- it must match the Triton your `torch` pairs with (`torch 2.12` <-> `triton 3.6`) |
| `tilelang-mlir-ascend` (imports as `tilelang`) | the MIMO forward `mamba3_mimo` | `is_mimo=True`, i.e. this document | `0.1.15+cuda.git0b31c2f4`, i.e. branch `v0.1.15` @ `0b31c2f4`, built with `USE_ASCEND910=ON` |

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
mismatched pair does not fail at import -- it fails at the first kernel launch, so section E is the
real check.

### 0.2 `tilelang-mlir-ascend`

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

Finally check that the A2 backend is really inside the wheel -- this is the same probe the step suite
uses to decide skip vs run:

```bash
python -c "import tilelang, tilelang.ascend910.language; print(tilelang.__version__)"
```

## A. What runs where

| Piece | Implementation | Notes |
|---|---|---|
| `Mamba3.forward`, MIMO branch | `mamba_ssm.ops.tilelang_ascend.mamba3.mamba3_mimo` (TileLang, A2) | Drop-in mirror of the CUDA `mamba3_mimo`: same arguments including `return_state`, `fuse_pregate_headwise_rms_norm`, `outproj_norm_weight`. Also exports `mamba3_mimo_fwd` / `mamba3_mimo_bwd` used by the op-level tests. |
| `Mamba3.step()` | `mamba_ssm.ops.ascendc.mamba3_step.mamba3_step_fn` (Ascend C, kernel-direct `mamba3_decode_step`) | The **same** operator as SISO, with `R = mimo_rank`. `Mamba3.step` calls it for both variants, so one rebinding covers both. |
| step's rotary / angle state | `mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step.apply_rotary_qk_inference_fwd` | `rotate_pairwise = not is_mimo` (`mamba3.py:392`): SISO rotates adjacent pairs, MIMO uses the blockwise/halved rotation. `angle = angle_state + tanh(angle_proj) * dt * pi`, `tanh(2sigmoid(2x) - 1)` inside the kernel. |

## B. Module wiring: the default route is still the CUDA kernel

`mamba_ssm/modules/mamba3.py:27-44` binds `mamba3_mimo_combined` to the **CUDA** tilelang module
(`mamba_ssm.ops.tilelang.mamba3.mamba3_mimo`). On NPU that import raises a non-`ImportError`
(`PassConfigKey.TL_ENABLE_FAST_MATH`), so `MAMBA3_DISABLE_TILELANG=1` is recommended to skip it
entirely; `mamba3_mimo_combined` then stays `None` and `Mamba3(is_mimo=True)` trips
`mamba3.py:119`:

```
assert mamba3_mimo_combined is not None, "Fails to import Mamba-3 MIMO kernels. ..."
```

So a MIMO run needs the A2 kernels installed explicitly. Two lines, no operator source touched:

```python
import os
os.environ.setdefault("MAMBA3_DISABLE_TILELANG", "1")   # before importing mamba3

import torch
import mamba_ssm.modules.mamba3 as m3mod
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.ops.ascendc.mamba3_step import mamba3_step_fn
from mamba_ssm.ops.tilelang_ascend.mamba3.mamba3_mimo import mamba3_mimo

m3mod.mamba3_mimo_combined = mamba3_mimo
m3mod.mamba3_step_fn = mamba3_step_fn    # normalises A/dt/trap/D itself (section D)

model = Mamba3(d_model=2048, d_state=128, headdim=64, is_mimo=True, mimo_rank=4,
               chunk_size=64 // 4, dtype=torch.bfloat16, device="npu",
               layer_idx=0, is_outproj_norm=False).eval()
u = torch.randn(2, 32, 2048, dtype=torch.bfloat16, device="npu")
with torch.no_grad():
    y = model(u)                                   # forward (MIMO, tilelang)
    st = model.allocate_inference_cache(2, 1, device="npu", dtype=torch.bfloat16)
    outs = []
    for t in range(u.shape[1]):                     # decode step-by-token
        o, angle, ssm, k, v = model.step(u[:, t], *st)
        st = (angle, ssm, k, v)
        outs.append(o)
```

SISO is unaffected by this: `mamba3_siso_combined` already resolves to the ported Triton kernel, so
`is_mimo=False` works with `MAMBA3_DISABLE_TILELANG=1` alone.

## C. Shapes, dtypes, and what the module passes

**Validated op-level grid** (upstream `CASE_GRID`, 11 shapes): `d_state` 16/32/64/128/256, `headdim`
32/64/128, `mimo_rank` 1/2/4/8, `chunk_size` 8..64 (coupled to `r` as `chunk = 64 // r` in the
module: `r=4 -> 16`, `r=1 -> 64`). The CUDA-only `bb_threads` knob is dropped on NPU.

```python
# step kernel constraints (mamba3_step_fn)
d_state % 8 == 0            # vectorised read/write
headdim % tile_D == 0       # tile_D in {32, 64, 128}; Mamba3.step passes tile_D=64
```

**dtypes.** The step operator takes five independent dtype groups (`mamba3_step.py`): state/state_out;
`B`/`Bstate`/`C`; `x`/`Xstate`/`z`/`out`; `xproj`/`outproj`/`zproj`; and `A`/`dt`/`trap`/`D` -- dtype is
only checked *within* a group. The module feeds the first three in the activation dtype, while `xproj` /
`zproj` / `outproj` come from the fp32 `mamba_x` / `mamba_z` / `mamba_o` parameters and stay fp32. The
fifth group has to be **fp32**: the kernel takes a single dtype code for those four tensors, so the entry
normalises them to fp32 rather than letting a caller mix them (see D).

**Outproj-norm.** With `is_outproj_norm=True` the step writes `(B, R, H, P)` (`outproj=None`, no `z`)
and `Mamba3._postprocess` applies the gate, `RMSNormGated` and `mimo_o`; with `False` the operator
returns `(B, H, P)` with `outproj` and the `z` gate fused. Both routes are covered by the step suite.

## D. dtype contract: `A` / `dt` / `trap` / `D` must share one dtype

The tiling carries a **single** dtype code for these four (`tiling.dtypeScalar`, taken from `A`) and the
kernel reads all four at that width, so a mixed group is read at the wrong stride and yields wrong numbers
with **no exception**: measured state rel-L2 `6.0e-1`, 12.9 % of the output off by >0.1, against `3.0e-8`
and 0 % once all four are fp32. `Mamba3.step` mixes them by construction -- `A` is fp32, while `dt` and
`trap` fall out of bf16 projections -- so the operator guards this in two places, and both must stay:

* `mamba3_step_fn` normalises half-precision members of the group to fp32 before dispatch. **Do not drop
  that cast when refactoring the entry**: it is what keeps `Mamba3.step` correct, and section B needs no
  adapter because of it.
* the C++ validator treats the four as a fifth dtype group and rejects a mixed group with
  `kRcDtypeScalar` (`rc=34`), so a caller that bypasses the Python entry fails loudly instead of reading
  garbage.

`selftest.py` pins both: one case asserts `rc=34`, one asserts the normalised call is bit-identical to an
explicit fp32 call.

## E. Test suites

| File | What it checks | Run |
|---|---|---|
| `tests/ops/tilelang_ascend/test_mamba3_mimo_upstream_std.py` | **Acceptance suite**: the upstream CUDA file `tests/ops/tilelang/test_mamba3_mimo.py` executed as-is (only device/RNG/admission adapted), 33 items = 11 shapes x (fwd, fwd+`return_state`, bwd with 14 gradients). Criterion is upstream's own: `max|ours-ref| / max|ref| < REL_TOL`, `REL_TOL = 0.10` | `pytest ... -q` |
| `tests/ops/ascendc/test_mamba3_mimo_step.py` | **The only module-level suite that covers the MIMO step** (ported from `tests/ops/cute/test_mamba3_mimo_step.py`). 4 cases = SISO/MIMO x `is_outproj_norm` False/True, `B=128, S=32, bf16`: 32 per-token `Mamba3.step()` calls against an fp32 `Mamba3.forward`, plus a mixed "16-token forward prefix + step suffix" decode through `InferenceParams`, at `rtol=atol=0.1` | see below |

```bash
# module-level step suite: 4 passed in ~13.5 min on 910B3 (kernel compilation dominates)
MAMBA3_STEP_DEVICE=npu MAMBA3_DISABLE_TILELANG=1 ASCEND_RT_VISIBLE_DEVICES=0 \
    python -m pytest -q -s -p no:cacheprovider tests/ops/ascendc/test_mamba3_mimo_step.py

# same file, benchmark instead of assertions (per-step latency + achieved bandwidth, 4 variants)
MAMBA3_DISABLE_TILELANG=1 python tests/ops/ascendc/test_mamba3_mimo_step.py

# op-level acceptance (33 items)
python -m pytest -q tests/ops/tilelang_ascend/test_mamba3_mimo_upstream_std.py
```

`MAMBA3_STEP_BATCH` / `MAMBA3_STEP_SEQLEN` shrink the step suite's workload (defaults are the upstream
ones, 128 / 32). It needs `torch_npu`, `tilelang` with the `ascend910` backend and the built
`libmamba3_decode_step_ops.so`, and skips itself with a message naming whatever is missing.

## F. Measured status

* **Op level:** 33/33 acceptance items pass, worst `max|ours-ref|/max|ref| = 1.31e-2` against the 0.10
  gate -- and the worst item is the forward output `O` itself, not a corner-case gradient.
* **Module level (step):** 4/4 cases pass at the upstream workload; step vs forward max|err| 3.9e-2
  over 32 tokens (tolerance 0.1), including the mixed prefix/suffix decode.
* **Speed** (`B=8`, single step, NPU event timing, mean of 30 steps after 5 warmup; reproduce with
  `MAMBA3_STEP_BATCH=8 MAMBA3_DISABLE_TILELANG=1 python tests/ops/ascendc/test_mamba3_mimo_step.py`):

  | Variant | rotation | ms/step | GB/s |
  |---|---|---|---|
  | MIMO, `is_outproj_norm=False` | halved | 2.33 | 15.0 |
  | MIMO, `is_outproj_norm=True` | halved | 3.16 | 11.1 |

  The same run also prints the SISO pair for reference (2.03 ms / 16.8 GB/s and 2.77 ms / 12.4 GB/s).
  Run-to-run spread on this 910B3 is a few percent, so read the table as a magnitude, not a target.
  The step reads and writes the fp32 `(B, H, P, N)` state, which is only 16.8 MB at `B=8`; 15 GB/s
  is ~1 % of the 910B3 HBM peak, so at this size the step is dominated by launch/serialisation rather
  than by memory. At `B=128` the same state is 268 MB and the kernel does become memory-bound -- that
  point is exercised for correctness by the suite but not benchmarked here, so treat the table as a
  small-batch baseline, not a tuned number.
* **CANN mixed tolerance on the MIMO operator is only partly met, and that is a bf16 property rather
  than a defect of the port**: cast the fp32 reference itself to bf16 and it stops clearing the gate
  too, and on every tensor that has an A100 reference the NPU result is at least as close as the A100
  one. Judge this path with the criterion above (`max|ours-ref|/max|ref| < 0.10`, 33/33), not with the
  per-element CANN gate.
