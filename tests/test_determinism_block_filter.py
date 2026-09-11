# Copyright (c) 2026, Tri Dao, Albert Gu.
import os

import pytest
import triton

from mamba_ssm.utils import determinism
from mamba_ssm.utils.determinism import _filter_configs_by_block_sizes, autotune_configs

CUMSUM = [triton.Config({"BLOCK_SIZE_H": h}) for h in (1, 2, 4, 8, 16, 32, 64)]
BMM = [
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=2),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
]


def config_kwargs(configs):
    return [c.kwargs for c in configs]


@pytest.fixture
def deterministic(monkeypatch):
    for k in list(os.environ):
        if k.startswith("TRITON_AUTOTUNE_BLOCK"):
            monkeypatch.delenv(k)
    monkeypatch.delenv("TRITON_CACHE_AUTOTUNING", raising=False)
    # isolate the module's global state: the override and the one-shot warning flag
    monkeypatch.setattr(determinism, "_deterministic_override", True)
    monkeypatch.setattr(determinism, "_autotune_warning_issued", False)
    return monkeypatch


def test_head_tile_pins_the_cumsum_kernel(deterministic):
    deterministic.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_H", "2")
    assert config_kwargs(_filter_configs_by_block_sizes(CUMSUM)) == [{"BLOCK_SIZE_H": 2}]
    assert config_kwargs(autotune_configs(CUMSUM)) == [{"BLOCK_SIZE_H": 2}]


def test_head_tile_is_ignored_by_kernels_without_that_key(deterministic):
    deterministic.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_H", "2")
    assert _filter_configs_by_block_sizes(BMM) is None
    # the bmm kernel keeps the cheapest-config rule
    assert config_kwargs(autotune_configs(BMM)) == [{"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}]


def test_filters_apply_per_kernel_and_are_anded(deterministic):
    deterministic.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_M", "64")
    deterministic.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_H", "2")
    assert config_kwargs(autotune_configs(CUMSUM)) == [{"BLOCK_SIZE_H": 2}]
    # only M applies to the bmm kernel: first config with M=64 in list order
    assert config_kwargs(autotune_configs(BMM)) == [{"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}]
    deterministic.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_N", "32")
    assert config_kwargs(autotune_configs(BMM)) == [{"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}]


def test_no_matching_config_falls_back_to_the_cheapest(deterministic):
    deterministic.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_H", "3")
    assert _filter_configs_by_block_sizes(CUMSUM) is None
    assert config_kwargs(autotune_configs(CUMSUM)) == [{"BLOCK_SIZE_H": 1}]


def test_env_has_no_effect_outside_deterministic_mode(deterministic):
    deterministic.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_H", "2")
    deterministic.setattr(determinism, "_deterministic_override", False)
    assert autotune_configs(CUMSUM) is CUMSUM


@pytest.mark.skipif(not determinism.TRITON_HAS_CACHE_RESULTS, reason="cached autotuning needs Triton >= 3.4")
def test_cached_autotuning_keeps_all_configs(deterministic):
    deterministic.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_H", "2")
    deterministic.setenv("TRITON_CACHE_AUTOTUNING", "1")
    assert autotune_configs(CUMSUM) is CUMSUM


def test_non_integer_value_for_an_unrelated_key_is_ignored(deterministic):
    deterministic.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_H", "auto")
    assert _filter_configs_by_block_sizes(BMM) is None
    assert config_kwargs(autotune_configs(BMM)) == [{"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}]


@pytest.mark.parametrize("value", ["auto", ""])
def test_non_integer_value_for_an_applicable_key_is_an_error(deterministic, value):
    deterministic.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_H", value)
    with pytest.raises(ValueError):
        _filter_configs_by_block_sizes(CUMSUM)
