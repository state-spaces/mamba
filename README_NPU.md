# Running Mamba-2 on Ascend NPU (910B series)

This document describes how to run the Mamba-2 Triton kernels in this repository
on Huawei Ascend NPUs (validated on **8× Ascend 910B3**, 64 GB HBM per card,
aarch64/Kunpeng host, CANN 9.1.0 container), and how to verify the
installation with the bundled test suite (`tests/npu/`).

> Every command below was executed and verified on a real 910B3 machine
> (2026-10-11 ~ 2026-10-16; the full `tests/npu/` suite of this branch was
> re-run end-to-end on 2026-10-20 against the patched tree — all PASS).
> Expected outputs are the actual recorded outputs.

## 0. What works

| Capability | triton-ascend 3.2.2 | triton-ascend ≥ 3.6.0 (nightly) |
|---|---|---|
| Selective-state-update decoding (`selective_state_update`) | ✅ | ✅ |
| LayerNorm / RMSNorm (gated & plain), fwd + bwd | ✅ | ✅ |
| SwiGLU fwd + bwd | ✅ | ✅ |
| SSD prefill **forward** (`mamba_chunk_scan_combined`) | ✅ | ✅ |
| SSD prefill **backward** (full autograd chain) | ❌ kernel compile error (fixed upstream in 3.6.0) | ✅ |
| `Mamba2` module forward + backward training | forward only | ✅ |
| Mamba-1 / Mamba-3 | ❌ out of scope (CUDA-only extensions / tilelang) | ❌ |

The process-level intermittent wrong-result bug in the SSD forward
(`_state_passing_fwd_kernel` pipelining race, ~25–46 % of processes) is **fixed
in this branch** by pinning `num_stages=1` when running on NPU (GPU configs are
untouched); see "Known issues" below.

## 1. Environment setup

### 1.1 Host check

```bash
uname -m                     # expect: aarch64 (x86_64 hosts work too with the x86 image)
npu-smi info | head -12      # expect: 8x 910B3 (or your card list)
docker images | grep 910b    # expect: cann:9.1.0-910b-ubuntu22.04-py3.11
```

### 1.2 Start the container

The official CANN image is required (host Python 3.9 cannot run this stack).
`--privileged` and the driver bind mounts are required — without them
`torch.npu.is_available()` stays `False`.

```bash
docker run -itd --name mamba2-npu --privileged \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v $HOME/mamba2-work:/work \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.11 \
  /bin/bash -c "sleep infinity"

docker exec -it mamba2-npu bash   # all subsequent steps run INSIDE the container
```

### 1.3 Python dependencies (inside the container)

```bash
pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
pip config set global.extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

pip install torch==2.8.0 torch_npu==2.8.0.post1 triton-ascend==3.2.2
# For the SSD backward you need the nightly instead of 3.2.2 (upstream fixed the
# backward-kernel compile errors in 3.6.0; see the table in section 0):
#   pip install https://repo.huaweicloud.com/ascend/repos/pypi/nightly/triton-ascend/\
#     triton_ascend-3.6.0+dev20260921-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
# Note: pick the wheel matching your architecture (aarch64 vs x86_64) — the
# filename must keep all five segments or pip rejects it with "Invalid wheel filename".
```

Known side effects (harmless, recorded for completeness): pip also installs
`triton 3.5.0` from the PyPI mirror while `triton.__version__` reports 3.2.0 at
runtime (triton-ascend overrides it); `pip check` may report conflicts that the
image ships with. The CANN image already ships `einops`/`packaging`/`pytest`
(used by mamba_ssm and the test suite); on a leaner image add
`pip install einops packaging pytest`.

### 1.4 One-time torch_npu header patch (CANN 9.1.0)

torch_npu 2.8.0.post1 bundles an ACL header that predates CANN 9.1.0. Without
this patch the first Triton kernel compile fails in `npu_utils.cpp` with
`aclOpExecutor` undefined.

```bash
H=$(python3 -c "import torch_npu, os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'include/third_party/acl/inc/acl/acl_op_compiler.h'))")
cp "$H" "$H.bak"
printf '\n/* mamba2 NPU: typedef moved to aclnn/acl_meta.h in CANN 9.1.0 */\ntypedef struct aclOpExecutor aclOpExecutor;\n' >> "$H"
```

### 1.5 Environment fingerprint check

```bash
python3 -c "import torch, torch_npu, triton; print(torch.__version__, torch_npu.__version__, triton.__version__)"
# expect: 2.8.0+cpu 2.8.0.post1 3.2.0   (or 3.6.0 with the nightly wheel)
python3 -c "import torch, torch_npu; print('npu available:', torch.npu.is_available())"
# expect: True
python3 -c "import torch, torch_npu; p=torch.npu.get_device_properties(0); print(type(p).__name__, p.multi_processor_count)"
# expect on torch_npu 2.8.0.post1: _NPUDeviceProperties None   <-- this branch's code handles it (see PR)
# expect on torch_npu 2.7.1.post11: _NPUDeviceProperties 40    (populated there; no special handling needed)
```

> **torch_npu version note (verified 2026-10-20 on 910B3):** torch_npu
> 2.8.0.post1 reports `multi_processor_count=None` (use
> `vector_core_num` / `cube_core_num` instead — this branch does, with the
> original `torch.cuda` line kept for non-NPU devices). torch_npu 2.7.1.post11
> reports a real count (40 on 910B3), and the unmodified upstream sm_count
> lines were re-verified to pass all backward tests there with bit-identical
> gradients (rel-diff table row below). This branch's device-conditional code
> works on both.

## 2. Get the code

### Option A — use this branch as-is (recommended for evaluation)

```bash
cd /work
git clone <this-repo-url> mamba2-npu
cd mamba2-npu
git checkout npu/ascend-910b-adapt
```

### Option B — apply the patch to upstream `main`

```bash
cd /work && git clone https://github.com/state-spaces/mamba.git mamba2-npu
cd mamba2-npu                        # main @ e9594ce or newer
git apply npu-adapt-ascend-910b.patch
```

### Install (pick one)

```bash
# A) Editable source install (recommended; pure Python wheel, no CUDA build needed):
pip install -e . --no-build-isolation

# B) Or run straight from the source tree without installing:
cd /work/mamba2-npu
export PYTHONPATH=/work/mamba2-npu:$PYTHONPATH   # APPEND, never overwrite — CANN's
                                                 # tbe paths live in PYTHONPATH (see Known issues)
```

If you use the official `pip install . --no-build-isolation` path, note two
dependency mines found during validation:

- `tilelang==0.1.8` + `apache-tvm-ffi==0.1.12` + Python 3.11 crashes at import
  (`tvm_ffi/registry.py:403`). Mitigation: `pip install apache-tvm-ffi==0.1.9`.
- `tilelang` pulls `numpy>=1.23.5`, which may silently upgrade numpy past
  triton-ascend's `numpy==1.26.4` pin. After installing, run `pip check` and
  keep triton-ascend's numpy pin (installing the triton-ascend 3.6.0 nightly
  resolves numpy back to 1.26.4).

## 3. Verify the installation

Run the whole suite (SSD tests are looped over 8 fresh processes — a single
PASS is not evidence, see section 5):

```bash
cd /work/mamba2-npu/tests/npu
bash run_npu_tests.sh 8
```

Or run the individual steps (all from `tests/npu/`):

```bash
# Triton smoke tests
python3 example1_add.py
# expect: versions: 2.8.0+cpu 2.8.0.post1 3.2.0 / max diff: 0.0
python3 -m pytest example2_broadcast.py -q
# expect: 3 passed

# Decoding kernel (stable, 6/6 PASS during validation)
python3 test_ssu_npu.py
# expect: max abs diff: 2.384185791015625e-07 / PASS

# LayerNorm family
python3 test_layernorm_npu.py           # expect: max abs diff: 9.5367431640625e-07 / PASS
python3 test_swiglu_npu.py              # expect: fwd/bwd max diff ~4.77e-07 / PASS
python3 test_layernorm_gated_bwd_npu.py # expect: dx/dz/dw/db all <= 1e-3 / PASS
python3 test_layer_norm_plain_npu.py    # expect: dx/dw/db all <= 1e-3 / PASS

# SSD building blocks
python3 test_chunk_cumsum_npu.py        # expect: ddt 0.0, dA rel ~2.1e-07
python3 test_bmm_chunk_npu.py           # expect: all diffs 0.0
python3 test_chunk_state_bwd_npu.py     # expect: dx/ddt/ddA <= 9.6e-06, dB <= 1e-3
python3 test_chunk_scan_bwd_npu.py      # expect: dz/dstates/dx <= 1.1e-05, dC/dcb <= 1e-3

# SSD combined operator — MUST be run in multiple fresh processes
for i in $(seq 1 8); do python3 test_ssd_npu.py | tail -1; done
# expect: 8/8 PASS, and on this branch the good-process diffs are bit-identical
#   across processes (recorded 2026-10-20 on 910B3, triton-ascend 3.6.0):
#   rel diff: 3.7164144395433736e-07 / 4.657614670182079e-07 — every run PASS.
#   (Pre-num_stages-fix baseline was ~2-4/8 FAIL with process-level intermittent
#   wrong values; the earlier journey recorded half-batch diffs out 1.37e-06 /
#   final 1.01e-06 because the script then unpacked the single default return
#   value along batch. The shipped script compares the full batch.)

# SSD backward (requires triton-ascend >= 3.6.0)
python3 test_ssd_bwd_npu.py
# expect: backward OK, x.grad norm ~161.74 / B.grad norm ~326.08 / C.grad norm ~338.71

# Mamba2 module forward + backward training step
python3 test_mamba2_module.py
# expect (recorded 2026-10-20, seeded): forward OK, loss = 0.3475983738899231 /
#   backward OK / x.grad norm: 0.020339107140898705

# Diagnostics
python3 test_coredim.py ub       # expect: error "ub overflow, requires 2097152 bits while 1572864 bits available"
python3 test_coredim.py coredim  # expect: kernel done, sum = 0.0
```

## 4. Using the `Mamba2` module on NPU

Two constraints differ from the CUDA README:

1. **`use_mem_eff_path=False`** — the default fused path calls into
   `causal_conv1d`, which is a CUDA-only extension. The built-in non-fused
   fallback (`F.conv1d` + `mamba_chunk_scan_combined`) works on NPU.
2. **`d_model % headdim == 0`** — the module asserts `d_ssm` divisibility.

```python
import torch, torch_npu
from mamba_ssm.modules.mamba2 import Mamba2

batch, length, dim = 2, 64, 16            # dim = headdim * nheads
x = torch.randn(batch, length, dim).to("npu:0")
model = Mamba2(d_model=dim, d_state=64, d_conv=4, expand=2,
               headdim=32,                 # 32 divides d_ssm = dim * expand = 32
               use_mem_eff_path=False,     # required on NPU
               ).to("npu:0")
y = model(x)
loss = y.square().mean()
loss.backward()                            # requires triton-ascend >= 3.6.0
```

## 5. Acceptance protocol — always multi-process

The nastiest bug found during validation (a pipelining race in
`_state_passing_fwd_kernel`) is **process-level intermittent**: a bad process
reproduces bit-identical wrong values, a good process is bit-stable, and a
single process that starts "good" stays correct for every in-process repeat.
Therefore:

- **Never accept a single run** of any SSD-path test (forward or backward).
  Loop over ≥ 8 fresh processes (`run_npu_tests.sh` does this).
- Ops not on the SSD path (LayerNorm, SwiGLU, selective-state-update) were
  stable in-process; 3 consecutive runs still recommended.
- Same-process determinism checks are useful but **not sufficient**.

## 6. Known issues and limitations

| Issue | Status in this branch |
|---|---|
| SSD forward process-level intermittent wrong results (triton-ascend pipelining race in `_state_passing_fwd/_bwd`) | **Fixed**: `num_stages=1` pinned for the state-passing autotune configs **on NPU only** (detected via `torch_npu`; GPU/ROCm configs unchanged, kernel bodies untouched). 16/16 multi-process PASS; backward gradients bit-identical; re-verified after the NPU-only refinement (8/8 bit-identical). Compiler-side fix tracked at [triton-lang/triton-ascend#2662](https://github.com/triton-lang/triton-ascend/issues/2662) (filed 2026-10-20), root-cause issue with compiler dumps at [Ascend/AscendNPU-IR#801](https://gitcode.com/Ascend/AscendNPU-IR/issues/801) (filed 2026-10-23). |
| SSD backward kernel compile errors (`_state_passing_bwd`, `_chunk_scan_bwd_ddAcs_stable`) on triton-ascend 3.2.2 | **Not fixable in this repo** — fixed upstream in triton-ascend 3.6.0. Use the nightly wheel for training. |
| `multi_processor_count` is `None` on torch_npu **2.8.0.post1** (2.7.1.post11 reports a real count, 40 on 910B3 — version-dependent) | **Fixed, and proven necessary on 2.8.0.post1**: reverting the 5 sm_count sites to the upstream `torch.cuda` line crashes every affected backward wrapper (`AssertionError: Torch not compiled with CUDA enabled`, verified 2026-10-20). The same reverted code was run on torch_npu 2.7.1.post11 + triton-ascend 3.6.0 and **passed all backward tests with bit-identical gradients** (x.grad 161.7418 / B.grad 326.0809 / C.grad 338.7113). The shipped code dispatches on `device.type`: NPU falls back to `vector_core_num` / `cube_core_num` (present in both torch_npu versions); the non-NPU branch is the original upstream line unchanged. |
| Default `Mamba2` path needs CUDA-only `causal_conv1d` | Use `use_mem_eff_path=False` (documented above). |
| `PYTHONPATH=<path>` (overwrite) breaks CANN GE graph compilation (`No module named 'tbe'`) | Always **append**: `PYTHONPATH=/work/mamba2-npu:$PYTHONPATH`. |
| `import tilelang` rewrites `PYTHONPATH` and breaks CANN graph-compile subprocesses | Avoid importing the mamba3 stack on NPU; this branch's lazy `mamba_ssm/__init__` keeps `import mamba_ssm.modules.mamba2` free of tilelang. |
| Mamba-3 (tilelang/triton) kernels | Not adapted; not expected to work on triton-ascend. |
| Performance vs GPU | Not benchmarked yet (shared multi-tenant machine). |
| Kernels with the same "in-loop store of carried value + loop-exit store" pattern as state-passing may share the pipelining risk | Audit item; the num_stages fix currently covers state-passing only. |

## 7. Environment summary (validated fingerprint)

| Component | Version |
|---|---|
| Hardware | 8× Ascend 910B3 (64 GB HBM), aarch64 host |
| OS / container | openEuler 22.03 SP4 host; `cann:9.1.0-910b-ubuntu22.04-py3.11` |
| Python | 3.11.15 (container image) |
| torch / torch_npu | 2.8.0+cpu / 2.8.0.post1 |
| triton / triton-ascend | 3.2.0 / 3.2.2 (inference); 3.6.0+dev (training) |
| torch_npu device props (2.8.0.post1; on 2.7.1.post11 `multi_processor_count=40`) | `cube_core_num=20`, `vector_core_num=40`, `multi_processor_count=None` |
