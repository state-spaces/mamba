"""Dispatch tests for ``mamba_ssm.ops.backend``.

Hardware-free and dependency-free on purpose: these pin the *selection* logic
only -- the environment override, the TileLang kill switch, which package each
family maps to, and what happens when an optional family is missing -- because
that is what decides which numeric path a call lands on.

``backend.py`` is loaded straight from its file instead of through
``mamba_ssm.ops``: ``mamba_ssm/__init__.py`` imports the whole model stack
(torch + triton + transformers), so a normal import would make these tests
require a complete install -- exactly the situation in which dispatch matters
most.  Every other suite in this tree imports the installed package, so the real
import path stays covered.

Run from the repository root::

    python -m pytest -q tests/ops/test_backend_dispatch.py
"""
import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_PATH = _REPO_ROOT / "mamba_ssm" / "ops" / "backend.py"


def _load_backend():
    spec = importlib.util.spec_from_file_location("_mamba3_backend_under_test", _BACKEND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backend = _load_backend()


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    """``backend()`` is memoized for the process; each test picks its own env."""
    backend.backend.cache_clear()
    yield
    backend.backend.cache_clear()


def test_backend_module_is_located():
    """Guard against the test silently passing against a stale/stub file."""
    assert _BACKEND_PATH.is_file()
    assert hasattr(backend, "_EXPORTS")


def test_forced_cuda_selects_the_upstream_packages(monkeypatch):
    monkeypatch.setenv(backend.BACKEND_ENV, "cuda")
    monkeypatch.delenv(backend.DISABLE_TILELANG_ENV, raising=False)

    assert backend.backend() == "cuda"
    assert backend.is_ascend() is False
    assert backend.triton_package() == "mamba_ssm.ops.triton"
    assert backend.ops_package() == "mamba_ssm.ops.triton"  # legacy alias
    assert backend.tilelang_package() == "mamba_ssm.ops.tilelang"
    assert backend.step_module() == "mamba_ssm.ops.cute.mamba3.mamba3_step_fn"


def test_forced_ascend_selects_the_npu_packages(monkeypatch):
    monkeypatch.setenv(backend.BACKEND_ENV, "ascend")
    monkeypatch.delenv(backend.DISABLE_TILELANG_ENV, raising=False)

    assert backend.backend() == "ascend"
    assert backend.is_ascend() is True
    assert backend.triton_package() == "mamba_ssm.ops.triton_ascend"
    assert backend.tilelang_package() == "mamba_ssm.ops.tilelang_ascend"
    assert backend.step_module() == "mamba_ssm.ops.ascendc.mamba3_step"


@pytest.mark.parametrize("value", ["cuda", "gpu", "nvidia", "CUDA", " Gpu "])
def test_cuda_aliases(monkeypatch, value):
    monkeypatch.setenv(backend.BACKEND_ENV, value)
    assert backend.backend() == "cuda"


@pytest.mark.parametrize("value", ["ascend", "npu", "cann", "NPU"])
def test_ascend_aliases(monkeypatch, value):
    monkeypatch.setenv(backend.BACKEND_ENV, value)
    assert backend.backend() == "ascend"


def test_typo_in_backend_env_raises(monkeypatch):
    monkeypatch.setenv(backend.BACKEND_ENV, "ascnd")
    with pytest.raises(ValueError, match=backend.BACKEND_ENV):
        backend.backend()


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_tilelang_kill_switch(monkeypatch, value):
    monkeypatch.setenv(backend.BACKEND_ENV, "ascend")
    monkeypatch.setenv(backend.DISABLE_TILELANG_ENV, value)
    assert backend.tilelang_package() is None


def test_tilelang_kill_switch_is_backend_independent(monkeypatch):
    monkeypatch.setenv(backend.DISABLE_TILELANG_ENV, "1")
    monkeypatch.setenv(backend.BACKEND_ENV, "cuda")
    assert backend.tilelang_package() is None
    backend.backend.cache_clear()
    monkeypatch.setenv(backend.BACKEND_ENV, "ascend")
    assert backend.tilelang_package() is None


@pytest.mark.parametrize("value", ["0", "false", "off", ""])
def test_falsey_switch_leaves_tilelang_enabled(monkeypatch, value):
    monkeypatch.setenv(backend.BACKEND_ENV, "cuda")
    monkeypatch.setenv(backend.DISABLE_TILELANG_ENV, value)
    assert backend.tilelang_package() == "mamba_ssm.ops.tilelang"


def test_every_model_entry_name_is_exported():
    """``modules/mamba3.py`` must be able to take all three families from here."""
    for name in (
        "RMSNormGated",
        "mamba3_siso_combined",
        "mamba3_mimo",
        "mamba3_step_fn",
        "apply_rotary_qk_inference_fwd",
    ):
        assert name in backend.__all__


def test_disabled_tilelang_resolves_to_none_without_importing(monkeypatch):
    monkeypatch.setenv(backend.BACKEND_ENV, "ascend")
    monkeypatch.setenv(backend.DISABLE_TILELANG_ENV, "1")
    assert backend._resolve("mamba3_mimo") is None


def test_missing_optional_family_resolves_to_none(monkeypatch):
    monkeypatch.setenv(backend.BACKEND_ENV, "cuda")
    monkeypatch.setitem(
        backend._STEP_MODULES, "cuda", ("mamba_ssm.ops.definitely_not_installed", "mamba3_step_fn")
    )
    assert backend._resolve("mamba3_step_fn") is None


def test_present_optional_family_returns_the_attribute(monkeypatch):
    import json

    monkeypatch.setenv(backend.BACKEND_ENV, "cuda")
    monkeypatch.setitem(backend._STEP_MODULES, "cuda", ("json", "dumps"))
    assert backend._resolve("mamba3_step_fn") is json.dumps


def test_missing_required_family_still_raises(monkeypatch):
    monkeypatch.setenv(backend.BACKEND_ENV, "cuda")
    monkeypatch.setitem(
        backend._FAMILY_PACKAGES["triton"], "cuda", "mamba_ssm.ops.definitely_not_installed"
    )
    with pytest.raises(ImportError):
        backend._resolve("mamba3_siso_combined")


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        backend.no_such_operator  # noqa: B018
