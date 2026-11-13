"""
Mamba-3 MIMO Step Function Tests  --  Ascend NPU port

Copyright (c) 2026, Dao AI Lab, Goombalab

Pytest coverage for Mamba3.step() and mixed forward/step decoding.

Upstream: ``tests/ops/cute/test_mamba3_mimo_step.py`` (CUDA / CuteDSL).  On the
Ascend NPU the two kernels that test exercises do not exist, so this port swaps
them for their A2 counterparts while keeping the workload, the seed, the four
configurations (SISO/MIMO x is_outproj_norm) and every assertion of the original
suite:

    upstream                                      Ascend NPU (this file)
    --------------------------------------------  ------------------------------------------
    Mamba3.forward  MIMO  -> mamba3_mimo (cute)   -> mamba3_mimo (tilelang_ascend)
    Mamba3.forward  SISO  -> mamba3_siso_combined -> unchanged (triton-ascend)
    Mamba3.step()         -> mamba3_step_fn (cute)-> mamba3_step_fn (Ascend C, A2 kernel)

The swap is a two-line import on the module globals of
``mamba_ssm.modules.mamba3`` (``mamba3_mimo_combined`` / ``mamba3_step_fn``), the
same way ``tests/ops/triton/test_mamba3_siso_e2e_npu.py`` installs its golden.
No production source is modified.

One contract difference found while porting, and **deliberately not patched in this
file**: the A2 kernel takes one dtype code for ``A`` / ``dt`` / ``trap`` / ``D``
(derived from ``A``), whereas the CUDA step path feeds ``trap`` in the activation
dtype (bf16 here, because ``Mamba3.step`` computes ``sigmoid`` on the bf16
projection).  A bf16 ``trap`` next to an fp32 ``A`` was silently misread: ~13% of
the output elements off by more than 0.1, measured max|err| = 1.15 on a case whose
golden matches at 3e-8 once the scalar side is fp32.  That is fixed in the operator
wrapper now -- ``mamba3_step_fn`` normalises the four tensors to fp32 and the C++
validator rejects a mixed group -- so this file installs the operator as-is and a
regression there fails the suite instead of being papered over by an adapter.

Usage:
    # correctness (inside the NPU container, from the repo root)
    MAMBA3_STEP_DEVICE=npu MAMBA3_DISABLE_TILELANG=1 ASCEND_RT_VISIBLE_DEVICES=0 \
        python -m pytest -q -s -p no:cacheprovider tests/ops/ascendc/test_mamba3_mimo_step.py

    # benchmark (single step, no assertions)
    MAMBA3_DISABLE_TILELANG=1 python tests/ops/ascendc/test_mamba3_mimo_step.py

Prerequisites:
    * ``torch_npu`` and an Ascend device;
    * ``tilelang`` built with the ``ascend910`` backend (MIMO forward);
    * the Ascend C step operator built into
      ``mamba_ssm/ops/ascendc/build/libmamba3_decode_step_ops.so``
      (``bash mamba_ssm/ops/ascendc/build.sh``, or point ``MAMBA3_DECODE_STEP_LIB``
      at an existing ``.so``).

Remove the -s flag for less verbose output.

Debug knobs (defaults are the upstream workload; only for quick iteration):
    MAMBA3_STEP_BATCH, MAMBA3_STEP_SEQLEN
"""
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
import torch
from torch import Tensor


# TileLang's CUDA MIMO path cannot be built on Ascend; skip the import so that
# ``mamba_ssm.modules.mamba3`` comes up with ``mamba3_mimo_combined = None``
# (patched to the A2 kernel below).
os.environ.setdefault("MAMBA3_DISABLE_TILELANG", "1")

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


# ---------------------------------------------------------------------------
# Device / environment gating
# ---------------------------------------------------------------------------
# The whole module is an NPU suite: report one clean skip instead of an import
# error on a CUDA/CPU-only checkout.
torch_npu = pytest.importorskip(
    "torch_npu", reason="Ascend NPU (torch_npu) is required for mamba3 step tests"
)

BATCH = int(os.environ.get("MAMBA3_STEP_BATCH", 128))
SEQLEN = int(os.environ.get("MAMBA3_STEP_SEQLEN", 32))
NHEADS = 64
HDIM = 64
DSTATE = 128
DTYPE = torch.bfloat16
DEVICE = os.environ.get("MAMBA3_STEP_DEVICE", "npu")
RTOL = 0.1
ATOL = 0.1

_SKIP_REASON = None
if DEVICE == "npu" and not torch.npu.is_available():
    _SKIP_REASON = "MAMBA3_STEP_DEVICE=npu but no Ascend device is visible"
elif DEVICE == "npu":
    torch.npu.set_device(0)


@dataclass(frozen=True)
class VariantConfig:
    """Captures the SISO/MIMO knobs that used to be mutable globals."""
    mimo_dim: int
    use_tilelang: bool
    is_mimo: bool


SISO = VariantConfig(mimo_dim=1, use_tilelang=False, is_mimo=False)
MIMO = VariantConfig(mimo_dim=4, use_tilelang=True, is_mimo=True)


def _npu_kernels_available() -> Optional[str]:
    """Return a skip reason when the A2 kernels cannot be used, else ``None``."""
    try:  # MIMO forward is a tilelang kernel; the Ascend backend is a separate build.
        import tilelang.ascend910.language  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"tilelang with the ascend910 backend is required (import failed: {exc!r})"

    import mamba_ssm.ops.ascendc.mamba3_step as ascendc_step

    lib = ascendc_step._lib_path()
    if not lib.is_file():
        return (
            f"Ascend C step operator not built: {lib} is missing; run "
            f"`bash {Path(ascendc_step.__file__).resolve().parent / 'build.sh'}` "
            f"or set MAMBA3_DECODE_STEP_LIB"
        )
    return None


def _install_npu_kernels() -> None:
    """Point the Mamba3 module at the A2 forward/step kernels.

    ``Mamba3.forward`` dispatches to a module-global ``mamba3_mimo_combined`` and
    ``Mamba3.step`` to ``mamba3_step_fn``; rebinding those two names is the whole
    port.  The SISO forward keeps using ``mamba3_siso_combined`` (triton-ascend),
    which already runs on NPU.
    """
    import mamba_ssm.modules.mamba3 as m3mod
    from mamba_ssm.ops.ascendc.mamba3_step import mamba3_step_fn as ascendc_step_fn
    from mamba_ssm.ops.tilelang_ascend.mamba3.mamba3_mimo import mamba3_mimo as a2_mimo

    m3mod.mamba3_mimo_combined = a2_mimo
    # No dtype adapter on purpose: ``mamba3_step_fn`` normalises the scalar side
    # (A/dt/trap/D) to fp32 itself (and the C++ validator rejects a mixed group),
    # so upcasting ``dt``/``trap`` here would hide a regression of that normalisation.
    m3mod.mamba3_step_fn = ascendc_step_fn


def _require_npu_kernels() -> None:
    if _SKIP_REASON is not None:
        pytest.skip(_SKIP_REASON)
    reason = _npu_kernels_available()
    if reason is not None:
        pytest.skip(reason)
    _install_npu_kernels()


def _require_npu_kernels_or_exit() -> None:
    """Same gate as ``_require_npu_kernels`` but usable from ``python test.py``."""
    if _SKIP_REASON is not None:
        raise SystemExit(_SKIP_REASON)
    reason = _npu_kernels_available()
    if reason is not None:
        raise SystemExit(reason)
    _install_npu_kernels()


def _mamba3_cls():
    from mamba_ssm.modules.mamba3 import Mamba3

    return Mamba3


@pytest.fixture(scope="module", autouse=True)
def _kernel_deps() -> None:
    _require_npu_kernels()


def _seed(value: int = 42) -> None:
    torch.manual_seed(value)
    torch.npu.manual_seed_all(value)


# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/forward_step.py#L31
@dataclass
class InferenceParams:
    """Inference parameters used to store context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    new_key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


@dataclass
class RunOutputs:
    config_label: str
    split: int
    out_fwd_fp32: Tensor
    outputs_step: Tensor
    prefix_out: Tensor
    outputs_mixed: Tensor


def _case_config(variant: VariantConfig, *, is_outproj_norm: bool) -> dict:
    d_model = NHEADS * HDIM // 2
    return {
        "d_model": d_model,
        "d_state": DSTATE,
        "headdim": HDIM,
        "is_mimo": variant.is_mimo,
        "mimo_rank": variant.mimo_dim,
        "chunk_size": 64 // variant.mimo_dim,
        "dtype": DTYPE,
        "device": DEVICE,
        "layer_idx": 0,
        "use_tilelang": variant.use_tilelang,
        "is_outproj_norm": is_outproj_norm,
    }


def _diff_stats(actual: Tensor, expected: Tensor) -> str:
    diff = (actual.float() - expected.float()).abs()
    return f"max_abs={diff.max().item():.6e}, mean_abs={diff.mean().item():.6e}"


def _assert_close(
    actual: Tensor,
    expected: Tensor,
    *,
    label: str,
    cfg: str,
    step: Optional[int] = None,
) -> None:
    try:
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=RTOL,
            atol=ATOL,
        )
    except AssertionError as err:
        location = f", step={step}" if step is not None else ""
        stats = _diff_stats(actual, expected)
        raise AssertionError(
            f"{label} assertion failed for {cfg}{location} ({stats})"
        ) from err


def _run_case(variant: VariantConfig, *, is_outproj_norm: bool) -> RunOutputs:
    Mamba3 = _mamba3_cls()
    cfg = _case_config(variant, is_outproj_norm=is_outproj_norm)
    config_label = (
        f"use_tilelang={cfg['use_tilelang']}, "
        f"is_outproj_norm={cfg['is_outproj_norm']}, "
        f"batch={BATCH}, seqlen={SEQLEN}, "
        f"nheads={NHEADS}, hdim={HDIM}, dstate={DSTATE}, mimo_dim={variant.mimo_dim}"
    )

    _seed(42)
    model_fwd = Mamba3(**cfg)
    model_fwd.eval()

    cfg_fp32 = {**cfg, "dtype": torch.float32}
    _seed(42)
    model_fwd_fp32 = Mamba3(**cfg_fp32)
    model_fwd_fp32.eval()
    model_fwd_fp32.load_state_dict(
        {k: v.float() for k, v in model_fwd.state_dict().items()},
        strict=False,
    )

    _seed(42)
    model_step = Mamba3(**cfg)
    model_step.eval()
    model_step.load_state_dict(model_fwd.state_dict(), strict=False)

    _seed(42)
    model_mix = Mamba3(**cfg)
    model_mix.eval()
    model_mix.load_state_dict(model_fwd.state_dict(), strict=False)

    _seed(42)
    u = torch.randn(BATCH, SEQLEN, cfg["d_model"], device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out_fwd_fp32 = model_fwd_fp32(u.float())

        state = model_step.allocate_inference_cache(BATCH, 1, device=DEVICE, dtype=DTYPE)
        outputs_step = []
        for t in range(SEQLEN):
            out_step, nxt_angle_state, state_out, nxt_k_state, nxt_v_state = model_step.step(
                u[:, t], *state
            )
            state = (nxt_angle_state, state_out, nxt_k_state, nxt_v_state)
            outputs_step.append(out_step)
        outputs_step = torch.stack(outputs_step, dim=1)

        split = SEQLEN // 2
        assert 0 < split < SEQLEN
        inference_params = InferenceParams(max_seqlen=SEQLEN, max_batch_size=BATCH)
        prefix_out = model_mix(u[:, :split], inference_params=inference_params)
        state = inference_params.key_value_memory_dict[model_mix.layer_idx]
        mixed_suffix = []
        for t in range(split, SEQLEN):
            out_step, nxt_angle_state, state_out, nxt_k_state, nxt_v_state = model_mix.step(
                u[:, t], *state
            )
            state = (nxt_angle_state, state_out, nxt_k_state, nxt_v_state)
            mixed_suffix.append(out_step)
        outputs_mixed = torch.cat([prefix_out, torch.stack(mixed_suffix, dim=1)], dim=1)

    return RunOutputs(
        config_label=config_label,
        split=split,
        out_fwd_fp32=out_fwd_fp32,
        outputs_step=outputs_step,
        prefix_out=prefix_out,
        outputs_mixed=outputs_mixed,
    )


@pytest.mark.parametrize("variant", [pytest.param(SISO, id="siso"), pytest.param(MIMO, id="mimo")])
@pytest.mark.parametrize(
    "is_outproj_norm",
    [
        pytest.param(False, id="outproj_norm_false"),
        pytest.param(True, id="outproj_norm_true"),
    ],
)
def test_step_matches_forward_fp32(variant: VariantConfig, is_outproj_norm: bool) -> None:
    outputs = _run_case(variant, is_outproj_norm=is_outproj_norm)

    for t in range(SEQLEN):
        _assert_close(
            outputs.outputs_step[:, t],
            outputs.out_fwd_fp32[:, t],
            label="pure-step",
            cfg=outputs.config_label,
            step=t,
        )

    _assert_close(
        outputs.prefix_out,
        outputs.out_fwd_fp32[:, :outputs.split],
        label="mixed-prefix",
        cfg=outputs.config_label,
    )

    for t in range(outputs.split, SEQLEN):
        _assert_close(
            outputs.outputs_mixed[:, t],
            outputs.out_fwd_fp32[:, t],
            label="mixed-suffix",
            cfg=outputs.config_label,
            step=t,
        )


def _bench_ms(fn, *, warmup: int = 5, rep: int = 30) -> float:
    """Median single-step latency in milliseconds (NPU event timing)."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / rep


def run_step_benchmark(variant: VariantConfig, *, is_outproj_norm: bool) -> None:
    _require_npu_kernels()
    Mamba3 = _mamba3_cls()

    cfg = _case_config(variant, is_outproj_norm=is_outproj_norm)
    rotate_str = "halved" if variant.use_tilelang else "pairwise"

    _seed(42)
    model_step = Mamba3(**cfg)
    model_step.eval()

    state_bm = model_step.allocate_inference_cache(BATCH, 1, device=DEVICE, dtype=DTYPE)
    u_step_bm = torch.randn(BATCH, cfg["d_model"], device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        model_step.step(u_step_bm, *state_bm)

    def full_step_fn():
        out, _, _, _, _ = model_step.step(u_step_bm, *state_bm)
        return out

    ms_full = _bench_ms(full_step_fn)

    dtype_size = torch.tensor([], dtype=DTYPE).element_size()
    state_dtype_size = 4
    num_rope_angles = model_step.num_rope_angles

    bytes_read = (
        BATCH * cfg["d_model"] * dtype_size
        + BATCH * NHEADS * HDIM * DSTATE * state_dtype_size
        + BATCH * NHEADS * num_rope_angles * state_dtype_size
        + BATCH * variant.mimo_dim * NHEADS * DSTATE * dtype_size
        + BATCH * NHEADS * HDIM * dtype_size
    )
    bytes_write = (
        BATCH * NHEADS * HDIM * dtype_size
        + BATCH * NHEADS * HDIM * DSTATE * state_dtype_size
        + BATCH * NHEADS * num_rope_angles * state_dtype_size
        + BATCH * variant.mimo_dim * NHEADS * DSTATE * dtype_size
        + BATCH * NHEADS * HDIM * dtype_size
    )

    total_bytes = bytes_read + bytes_write
    bw = total_bytes / (ms_full * 1e-3) / 1e9

    print("\n" + "=" * 70)
    print(
        "Benchmark: Mamba3.step() "
        f"(rotation={rotate_str}, is_outproj_norm={is_outproj_norm})"
    )
    print("=" * 70)
    print(
        f"  batch={BATCH}, d_model={cfg['d_model']}, nheads={NHEADS}, "
        f"hdim={HDIM}, dstate={DSTATE}, mimo_dim={variant.mimo_dim}"
    )
    print(f"  Time per step: {ms_full:.4f} ms")
    print(
        "  Memory I/O:    "
        f"{total_bytes / 1e6:.2f} MB "
        f"(Read: {bytes_read / 1e6:.2f} MB, Write: {bytes_write / 1e6:.2f} MB)"
    )
    print(f"  Bandwidth:     {bw:.1f} GB/s")


if __name__ == "__main__":
    _require_npu_kernels_or_exit()
    print("Running SISO benchmarks...")
    run_step_benchmark(SISO, is_outproj_norm=False)
    run_step_benchmark(SISO, is_outproj_norm=True)

    print("Running MIMO benchmarks...")
    run_step_benchmark(MIMO, is_outproj_norm=False)
    run_step_benchmark(MIMO, is_outproj_norm=True)
