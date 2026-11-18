# `mamba_ssm.ops.triton_ascend` — Ascend NPU port of the Mamba-3 Triton operators

This package holds the Triton-Ascend (CANN) implementation of the Mamba-3 SISO
operator family.  It is a **separate package on purpose**: `mamba_ssm.ops.triton`
stays byte-identical to upstream, and only the model entry picks a backend.

The user-facing contract - which entry point to call, what the step returns,
how to switch the step implementation - is documented in
[`USAGE_mamba3_siso_npu.md`](../../../USAGE_mamba3_siso_npu.md) and
[`USAGE_mamba3_mimo_npu.md`](../../../USAGE_mamba3_mimo_npu.md).

## Who imports what

| Caller | Import from |
|---|---|
| `mamba_ssm/modules/mamba3.py` (model entry, backend-agnostic) | `mamba_ssm.ops.backend` |
| `mamba_ssm/ops/ascendc`, `mamba_ssm/ops/tilelang_ascend`, NPU tests/scripts | `mamba_ssm.ops.triton_ascend...` (explicit) |
| Upstream tests, `scripts/golden_a100`, `ops/tilelang` | `mamba_ssm.ops.triton...` (explicit) |

```python
# backend-agnostic (model code)
from mamba_ssm.ops.backend import mamba3_siso_combined, RMSNormGated

# Ascend-side code that must not fall back to CUDA
from mamba_ssm.ops.triton_ascend.mamba3.mamba3_siso_bwd import compute_dqkv
```

## Backend selection

`mamba_ssm.ops.backend.backend()` resolves, in order:

1. `MAMBA3_TRITON_BACKEND=cuda|gpu|nvidia|ascend|npu|cann` (anything else raises
   `ValueError` — a typo must not silently pick the wrong numeric path);
2. auto: `triton.language.extra.cann.libdevice` importable → `ascend`
   (same probe as `npu_compat.IS_CANN`, so dispatch and in-kernel branching
   always agree);
3. auto: `torch.npu.is_available()` → `ascend`;
4. otherwise `cuda`.

The result is memoized on first use.  To A/B against the upstream kernels on an
Ascend box, set `MAMBA3_TRITON_BACKEND=cuda`; expect it to fail on TMA / PTX /
`maxnreg`, which is exactly the point of keeping that path pristine.

## Module map

| Module | Responsibility | Notable NPU adaptations |
|---|---|---|
| `mamba3/npu_compat.py` | backend probe, UB/chunk policy, precision switches | `IS_CANN`, `effective_chunk_size`, `chunk_choices`, `make_config` (drops `maxnreg`), `precise_dqkv`, `precise_ddt` |
| `mamba3/utils.py` | elementwise approximations | `tanh/sin/cos/sech2` → CANN libdevice instead of PTX `*.approx.f32` |
| `mamba3/angle_dt.py` | angle cumsum fwd/bwd | 1-D `coreDim` grid-stride (`UINT16_MAX`), UB-bounded `CHUNK_SIZE = 1024 // BLOCK_D` |
| `mamba3/grouped_head_reduction.py` | per-head bias reduction | `num_warps=1/num_stages=1` on CANN (`GraphOptimize` failure otherwise) |
| `mamba3/mamba3_siso_fwd.py` | chunked forward | TMA → strided pointers, `PHASE` specialization, `BLOCK_C=16` sub-blocks, even/odd rotary (no `tl.split/join`) |
| `mamba3/mamba3_siso_bwd.py` | `dzdo`/`dqkv`/`dqktheta`/`ddt_dtrap_dinput_states` | `tl.sum` instead of N=1 `tl.dot`, buffer reuse, optional fp32 `PRECISE_FP32` path |
| `mamba3/mamba3_siso_combined.py` | autograd wrapper | device-agnostic scratch allocator, chunk clamp, fp32 `Trap`/`Angles` |
| `mamba3/mamba3_siso_step.py` | single-token decode | `tl.sum` instead of N=1 `tl.dot`, explicit `nheads`, fp32 rotated Q/K |
| `mamba3/mamba3_mimo_rotary_step.py` | rotary in the decode step | device-agnostic launches |
| `layernorm_gated.py` | gated RMSNorm/LayerNorm | `coreDim` row grid-stride, `_device_ctx`/`_sm_count` (AIV count) helpers |
| `layer_norm.py` | non-gated LayerNorm/RMSNorm | same device helpers (kept for parity with the gated path) |

## Status of the `IS_CANN` branches

The kernels keep their `IS_CANN` conditionals even though this is the Ascend
side.  On a build without CANN libdevice they take the CUDA formulation instead
of raising, which keeps CPU-side static checks, `TRITON_INTERPRET` runs and the
forced A/B mode working.  Do **not** add new backend conditionals to
`mamba_ssm/ops/triton` — new NPU work belongs here, registered in
`mamba_ssm/ops/backend.py::_EXPORTS` (same name and signature on both sides).

## Verification

```bash
# upstream tree untouched (empty diff) + no Ascend markers
git diff 08a1420 -- mamba_ssm/ops/triton

# syntax, no triton / NPU required
python -m py_compile mamba_ssm/ops/triton_ascend/mamba3/*.py

# dispatch
python -c "from mamba_ssm.ops import backend as b; print(b.backend(), b.ops_package())"

# functional / precision regression, inside the CANN container
python tests/ops/triton/test_mamba3_siso_npu.py
python tests/ops/triton/test_mamba3_siso_e2e_npu.py
```
