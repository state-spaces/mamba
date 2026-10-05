import importlib.util
from pathlib import Path
import sys

import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parents[1]
generation = load_module("generation_reset", ROOT / "mamba_ssm/utils/generation.py")
mha = load_module("mha_reset", ROOT / "mamba_ssm/modules/mha.py")


def test_reset_restarts_kv_cache_at_first_batch_slot():
    cache = torch.zeros(4, 6, 2, 1, 1)
    params = generation.InferenceParams(
        max_seqlen=6,
        max_batch_size=4,
        seqlen_offset=2,
        batch_size_offset=2,
        key_value_memory_dict={0: (cache, None)},
        lengths_per_sample=torch.full((4,), 2, dtype=torch.int32),
    )
    params.reset(max_seqlen=6, max_batch_size=4)
    values = torch.ones(2, 1, 2, 1, 1)
    actual = mha._update_kv_cache(values, params, 0)
    torch.testing.assert_close(actual, values)
    torch.testing.assert_close(cache[:2, :1], values)
    assert torch.count_nonzero(cache[2:]) == 0
    assert params.seqlen_offset == 0
    assert torch.count_nonzero(params.lengths_per_sample) == 0
