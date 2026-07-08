"""
og_model.py — OGStatefulMambaHybrid1D

Reference model: Hybrid1DBackbone + Mamba3(expand=1, CUDA FP32).
Used as the golden reference for ANE parity tests.

Do NOT import from tests/ane/StatefulMobileNet/mamba/model.py to avoid
circular issues — Hybrid1DBackbone is duplicated here verbatim.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow import of mamba_ssm from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mamba_ssm.modules.mamba3 import Mamba3

try:
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class InferenceParams:
        max_seqlen: int; max_batch_size: int; seqlen_offset: int = 0
        batch_size_offset: int = 0
        key_value_memory_dict: dict = field(default_factory=dict)
        lengths_per_sample = None


class Hybrid1DBackbone(nn.Module):
    """Exact copy of Hybrid1DBackbone from model.py — do not diverge."""
    def __init__(self, in_channels=3, hidden_dim=256, output_dim=512, seq_length=224):
        super().__init__()
        self.seq_length = seq_length
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=5, stride=2, padding=2, bias=True)
        self.silu1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2, bias=True)
        self.silu2 = nn.SiLU(inplace=True)
        conv_out_length = seq_length // 4
        self.fc1 = nn.Linear(hidden_dim * 2 * conv_out_length, hidden_dim * 2)
        self.silu3 = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)
        self.silu4 = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.silu1(self.conv1(x))
        x = self.silu2(self.conv2(x))
        x = x.flatten(1)
        x = self.silu3(self.fc1(x))
        x = self.silu4(self.fc2(x))
        return x


class OGStatefulMambaHybrid1D(nn.Module):
    """
    Golden reference: same Hybrid1DBackbone as StatefulMambaHybrid1D,
    but uses Mamba3(expand=1) from mamba_ssm instead of MambaBlock.

    expand=1 → d_inner=512, nheads=8, num_rope_angles=16 — exact match to MambaBlock.

    Autoregressive inference: forward() called with seqlen=1 and
    seqlen_offset kept at 0 permanently. Requires mamba3.py fix (Input_States).
    """
    def __init__(self, num_classes=1000, backbone_in_channels=3,
                 backbone_hidden_dim=256, backbone_output_dim=512,
                 seq_length=224, mamba_d_state=64, mamba_headdim=64,
                 device=None, dtype=torch.float32):
        super().__init__()
        self.backbone = Hybrid1DBackbone(
            in_channels=backbone_in_channels,
            hidden_dim=backbone_hidden_dim,
            output_dim=backbone_output_dim,
            seq_length=seq_length,
        )
        self.mamba = Mamba3(
            d_model=backbone_output_dim,
            d_state=mamba_d_state,
            expand=1,                  # d_inner = backbone_output_dim
            headdim=mamba_headdim,
            ngroups=1,
            rope_fraction=0.5,
            is_mimo=False,
            is_outproj_norm=False,
            layer_idx=0,
            device=device,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(backbone_output_dim, num_classes)
        self._inf_params = None

    def reset_state(self):
        self._inf_params = InferenceParams(max_seqlen=4096, max_batch_size=1)
        self._inf_params.seqlen_offset = 0  # NEVER increment — always use forward path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, 3, seq_length) — one frame at a time."""
        if self._inf_params is None:
            self.reset_state()
        features = self.backbone(x)                              # (1, backbone_output_dim)
        mamba_in = features.unsqueeze(1)                         # (1, 1, backbone_output_dim)
        mamba_out = self.mamba(mamba_in, inference_params=self._inf_params)
        mamba_out = mamba_out.squeeze(1)                         # (1, backbone_output_dim)
        logits = self.classifier(self.dropout(mamba_out))
        return logits

    def load_from_stateful(self, src) -> None:
        """
        Copy weights from StatefulMambaHybrid1D into this model.
        src.mamba is a MambaBlock; self.mamba is a Mamba3(expand=1).
        All backbone and classifier weights are copied directly.
        The only reshape needed is B_bias / C_bias layout.
        """
        with torch.no_grad():
            # Backbone (same class, same shapes)
            self.backbone.load_state_dict(src.backbone.state_dict())
            # Classifier
            self.classifier.load_state_dict(src.classifier.state_dict())
            # Mamba3 ← MambaBlock (direct copies)
            self.mamba.in_proj.weight.copy_(src.mamba.in_proj.weight)
            self.mamba.out_proj.weight.copy_(src.mamba.out_proj.weight)
            self.mamba.dt_bias.copy_(src.mamba.dt_bias)
            self.mamba.D.copy_(src.mamba.D)
            self.mamba.B_norm.weight.copy_(src.mamba.B_norm.weight)
            self.mamba.C_norm.weight.copy_(src.mamba.C_norm.weight)
            # B_bias: MambaBlock (1,8,64) → Mamba3 (8,1,64)
            self.mamba.B_bias.copy_(src.mamba.B_bias.squeeze(0).unsqueeze(1))
            # C_bias: same reshape
            self.mamba.C_bias.copy_(src.mamba.C_bias.squeeze(0).unsqueeze(1))


if __name__ == '__main__':
    # Smoke test
    from mamba_ane.models.hybrid1d import StatefulMambaHybrid1D
    torch.manual_seed(42)
    ref = StatefulMambaHybrid1D().cuda().float().eval()
    og = OGStatefulMambaHybrid1D().cuda().float().eval()
    og.load_from_stateful(ref)
    x = torch.randn(1, 3, 224, device='cuda')
    with torch.no_grad():
        out_ref = ref(x)
        out_og  = og(x)
    print(f"ref logits norm: {out_ref.norm():.4f}")
    print(f"og  logits norm: {out_og.norm():.4f}")
    print("load_from_stateful: OK")
