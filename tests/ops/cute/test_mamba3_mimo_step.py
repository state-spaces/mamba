"""
Mamba-3 MIMO Step Function Tests

Copyright (c) 2026, Dao AI Lab, Goombalab

Pytest coverage for Mamba3.step() and mixed forward/step decoding.

Usage:
pytest -q -s -p no:warnings tests/ops/cute/test_mamba3_mimo_step.py  # For correctness tests
python tests/ops/cute/test_mamba3_mimo_step.py  # For benchmark

Remove the -s flag for less verbose output.
"""
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
import torch
from torch import Tensor


warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)





BATCH = 128
SEQLEN = 32
NHEADS = 64
HDIM = 64
DSTATE = 128
DTYPE = torch.bfloat16
DEVICE = "cuda"
RTOL = 0.1
ATOL = 0.1

@dataclass(frozen=True)
class VariantConfig:
    """Captures the SISO/MIMO knobs that used to be mutable globals."""
    mimo_dim: int
    use_tilelang: bool
    is_mimo: bool


SISO = VariantConfig(mimo_dim=1, use_tilelang=False, is_mimo=False)
MIMO = VariantConfig(mimo_dim=4, use_tilelang=True, is_mimo=True)


def _require_cuda_and_kernel_deps() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for mamba3 step tests")
    pytest.importorskip("tilelang")
    pytest.importorskip("triton")


def _mamba3_cls():
    from mamba_ssm.modules.mamba3 import Mamba3

    return Mamba3


@pytest.fixture(scope="module", autouse=True)
def _kernel_deps() -> None:
    _require_cuda_and_kernel_deps()


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

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_fwd = Mamba3(**cfg)
    model_fwd.eval()

    cfg_fp32 = {**cfg, "dtype": torch.float32}
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_fwd_fp32 = Mamba3(**cfg_fp32)
    model_fwd_fp32.eval()
    model_fwd_fp32.load_state_dict(
        {k: v.float() for k, v in model_fwd.state_dict().items()},
        strict=False,
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_step = Mamba3(**cfg)
    model_step.eval()
    model_step.load_state_dict(model_fwd.state_dict(), strict=False)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_mix = Mamba3(**cfg)
    model_mix.eval()
    model_mix.load_state_dict(model_fwd.state_dict(), strict=False)

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


def run_step_benchmark(variant: VariantConfig, *, is_outproj_norm: bool) -> None:
    _require_cuda_and_kernel_deps()
    from triton.testing import do_bench_cudagraph
    Mamba3 = _mamba3_cls()

    cfg = _case_config(variant, is_outproj_norm=is_outproj_norm)
    rotate_str = "halved" if variant.use_tilelang else "pairwise"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_step = Mamba3(**cfg)
    model_step.eval()

    state_bm = model_step.allocate_inference_cache(BATCH, 1, device=DEVICE, dtype=DTYPE)
    u_step_bm = torch.randn(BATCH, cfg["d_model"], device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        model_step.step(u_step_bm, *state_bm)

    def full_step_fn():
        out, _, _, _, _ = model_step.step(u_step_bm, *state_bm)
        return out

    ms_full = do_bench_cudagraph(full_step_fn, rep=30)

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
    print("Running SISO benchmarks...")
    run_step_benchmark(SISO, is_outproj_norm=False)
    run_step_benchmark(SISO, is_outproj_norm=True)

    print("Running MIMO benchmarks...")
    run_step_benchmark(MIMO, is_outproj_norm=False)
    run_step_benchmark(MIMO, is_outproj_norm=True)