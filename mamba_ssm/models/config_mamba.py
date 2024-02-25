from dataclasses import dataclass, field
from typing import Union


@dataclass
class MambaConfig:

    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    bidirectional: bool = False
    bidirectional_strategy: Union[str, None] = None
