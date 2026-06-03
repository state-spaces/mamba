import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

class MambaDNA(nn.Module):
    """
    Wrapper around the official MambaLMHeadModel for DNA sequence modeling.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.model = MambaLMHeadModel(config)
        
    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        return self.model(input_ids, position_ids=position_ids, inference_params=inference_params, num_last_tokens=num_last_tokens)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        return MambaLMHeadModel.from_pretrained(pretrained_model_name, device=device, dtype=dtype, **kwargs)
