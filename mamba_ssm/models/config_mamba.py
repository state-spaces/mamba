class MambaConfig:
    def __init__(
        self,
        d_model: 2560,
        n_layer: 64,
        vocab_size: 50277,
        ssm_cfg: {},
        rms_norm: True,
        residual_in_fp32: True,
        fused_add_norm: True,
        pad_vocab_size_multiple: 8
    ):
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
