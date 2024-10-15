from dataclasses import dataclass, field


@dataclass
class MambaConfig:
    def __init__(self, d_model=2560, d_intermediate=0, n_layer=64, vocab_size=50277,
                 ssm_cfg=None, attn_layer_idx=None, attn_cfg=None, rms_norm=True,
                 residual_in_fp32=True, fused_add_norm=True, pad_vocab_size_multiple=8,
                 tie_embeddings=True, **kwargs):
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg or {}
        self.attn_layer_idx = attn_layer_idx or []
        self.attn_cfg = attn_cfg or {}
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_embeddings = tie_embeddings
        # Ignore any unexpected keyword arguments
        for key, value in kwargs.items():
            print(f"Warning: Ignoring unexpected argument '{key}' with value '{value}'")
