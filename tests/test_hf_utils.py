"""Tests for Hugging Face checkpoint loading helpers."""

import importlib.util
import json
import pickle
from pathlib import Path

import pytest
import torch

_hf_path = Path(__file__).resolve().parents[1] / "mamba_ssm" / "utils" / "hf.py"
_spec = importlib.util.spec_from_file_location("mamba_ssm.utils.hf", _hf_path)
hf_utils = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(hf_utils)


def test_load_config_hf_reads_json(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    payload = {"d_model": 128, "n_layer": 2}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        hf_utils,
        "cached_file",
        lambda model_name, filename, **_kwargs: str(config_path),
    )

    assert hf_utils.load_config_hf("dummy/model") == payload


def test_load_config_hf_missing_file(monkeypatch):
    monkeypatch.setattr(
        hf_utils,
        "cached_file",
        lambda model_name, filename, **_kwargs: None,
    )

    with pytest.raises(FileNotFoundError, match="config.json"):
        hf_utils.load_config_hf("missing/model")


def test_load_state_dict_hf_missing_weights(monkeypatch):
    monkeypatch.setattr(
        hf_utils,
        "cached_file",
        lambda model_name, filename, **_kwargs: None,
    )

    with pytest.raises(FileNotFoundError, match="pytorch_model.bin"):
        hf_utils.load_state_dict_hf("missing/model")


def test_load_state_dict_hf_uses_weights_only_true(tmp_path, monkeypatch):
    weights_path = tmp_path / "pytorch_model.bin"
    tensor = torch.randn(2, 2)
    torch.save({"weight": tensor}, weights_path)

    captured = {}

    def fake_torch_load(path, *, weights_only=None, **kwargs):
        captured.update(kwargs, weights_only=weights_only)
        return {"weight": tensor.clone()}

    monkeypatch.setattr(
        hf_utils,
        "cached_file",
        lambda model_name, filename, **_kwargs: str(weights_path),
    )
    monkeypatch.setattr(hf_utils.torch, "load", fake_torch_load)

    state_dict = hf_utils.load_state_dict_hf("dummy/model")
    assert torch.allclose(state_dict["weight"], tensor)
    assert captured["weights_only"] is True


@pytest.mark.parametrize("dtype", [None, torch.float16, torch.float64])
def test_load_state_dict_hf_converts_dtype_and_device(tmp_path, monkeypatch, dtype):
    weights_path = tmp_path / "pytorch_model.bin"
    tensor = torch.randn(2, 2)
    torch.save({"weight": tensor}, weights_path)
    monkeypatch.setattr(hf_utils, "cached_file", lambda *args, **kwargs: str(weights_path))

    state_dict = hf_utils.load_state_dict_hf("dummy/model", device="cpu", dtype=dtype)

    expected = tensor if dtype is None else tensor.to(dtype=dtype)
    torch.testing.assert_close(state_dict["weight"], expected)
    assert state_dict["weight"].device == torch.device("cpu")


def test_load_state_dict_hf_rejects_non_tensor_pickle(tmp_path, monkeypatch):
    weights_path = tmp_path / "pytorch_model.bin"
    torch.save({"weight": object()}, weights_path)
    monkeypatch.setattr(hf_utils, "cached_file", lambda *args, **kwargs: str(weights_path))

    with pytest.raises(pickle.UnpicklingError, match="[Ww]eights only|[Ww]eights_only"):
        hf_utils.load_state_dict_hf("dummy/model")
