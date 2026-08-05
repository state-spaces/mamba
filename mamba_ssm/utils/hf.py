import json

import torch

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file


def load_config_hf(model_name):
    resolved_archive_file = cached_file(
        model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False
    )
    if resolved_archive_file is None:
        raise FileNotFoundError(
            f"Could not find {CONFIG_NAME} for Hugging Face model {model_name!r}."
        )
    with open(resolved_archive_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state_dict_hf(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(
        model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
    )
    if resolved_archive_file is None:
        raise FileNotFoundError(
            f"Could not find {WEIGHTS_NAME} for Hugging Face model {model_name!r}."
        )
    load_kwargs = {"map_location": mapped_device}
    # PyTorch 2.6+ defaults weights_only=True; HF checkpoints need the full pickle.
    if "weights_only" in torch.load.__code__.co_varnames:
        load_kwargs["weights_only"] = False
    return torch.load(resolved_archive_file, **load_kwargs)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict
