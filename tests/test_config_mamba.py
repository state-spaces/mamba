"""Tests for MambaConfig defaults and serialization."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

_config_path = Path(__file__).resolve().parents[1] / "mamba_ssm" / "models" / "config_mamba.py"
_spec = importlib.util.spec_from_file_location("mamba_ssm.models.config_mamba", _config_path)
config_mamba = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(config_mamba)
MambaConfig = config_mamba.MambaConfig


def test_mamba_config_defaults_are_independent():
    first = MambaConfig()
    second = MambaConfig()
    first.ssm_cfg["layer"] = "Mamba3"
    first.attn_layer_idx.append(0)
    assert second.ssm_cfg == {}
    assert second.attn_layer_idx == []


def test_mamba_config_round_trips_through_json():
    config = MambaConfig(d_model=512, n_layer=8, vocab_size=32000, ssm_cfg={"layer": "Mamba2"})
    payload = json.loads(json.dumps(asdict(config)))
    restored = MambaConfig(**payload)
    assert restored.d_model == 512
    assert restored.n_layer == 8
    assert restored.ssm_cfg == {"layer": "Mamba2"}


def test_pad_vocab_size_multiple_rounds_up():
    config = MambaConfig(vocab_size=50277, pad_vocab_size_multiple=8)
    padded = config.vocab_size
    if padded % config.pad_vocab_size_multiple != 0:
        padded += config.pad_vocab_size_multiple - (padded % config.pad_vocab_size_multiple)
    assert padded == 50280
    assert padded % 8 == 0
