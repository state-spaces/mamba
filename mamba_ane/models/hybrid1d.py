"""
mamba_ane/models/hybrid1d.py — Hybrid1D models combining Conv1D backbone with MambaBlock.
"""

import torch
import torch.nn as nn
from mamba_ane.modules.mamba3 import MambaBlock


class Hybrid1DBackbone(nn.Module):
    """
    Conv1D feature extraction → Linear projection. SiLU activations only.

    Two stride-2 Conv1D layers reduce seq_length by 4x, then two Linear
    layers project to output_dim.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 512,
        seq_length: int = 224,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=5, stride=2, padding=2, bias=True)
        self.silu1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2, bias=True)
        self.silu2 = nn.SiLU(inplace=True)
        conv_out_length = seq_length // 4
        self.fc1   = nn.Linear(hidden_dim * 2 * conv_out_length, hidden_dim * 2)
        self.silu3 = nn.SiLU(inplace=True)
        self.fc2   = nn.Linear(hidden_dim * 2, output_dim)
        self.silu4 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, seq_length) -> (B, output_dim)"""
        x = self.silu1(self.conv1(x))
        x = self.silu2(self.conv2(x))
        x = x.flatten(1)
        x = self.silu3(self.fc1(x))
        return self.silu4(self.fc2(x))


class StatefulMambaHybrid1D(nn.Module):
    """
    Hybrid1DBackbone -> MambaBlock -> Classifier.

    State buffers (angle_state, ssm_state, k_state, v_state inside MambaBlock)
    persist across inferences via ct.StateType in CoreML export.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        backbone_in_channels: int = 3,
        backbone_hidden_dim: int = 256,
        backbone_output_dim: int = 512,
        seq_length: int = 224,
        ema_alpha: float = 0.1,
        mamba_d_state: int = 64,
        mamba_headdim: int = 64,
        mamba_num_heads: int = 8,
    ):
        super().__init__()
        self.backbone = Hybrid1DBackbone(
            in_channels=backbone_in_channels,
            hidden_dim=backbone_hidden_dim,
            output_dim=backbone_output_dim,
            seq_length=seq_length,
        )
        self.mamba = MambaBlock(
            d_model=backbone_output_dim,
            d_state=mamba_d_state,
            headdim=mamba_headdim,
            num_heads=mamba_num_heads,
            ema_alpha=ema_alpha,
        )
        self.dropout    = nn.Dropout(0.2)
        self.classifier = nn.Linear(backbone_output_dim, num_classes)
        self.feature_dim = backbone_output_dim
        self.ema_alpha   = ema_alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, 3, seq_length) -> (1, num_classes)"""
        features  = self.backbone(x)
        mamba_out = self.mamba(features)
        return self.classifier(self.dropout(mamba_out))


if __name__ == "__main__":
    print("=== StatefulMambaHybrid1D — PyTorch sanity check ===\n")
    model = StatefulMambaHybrid1D(
        num_classes=1000, backbone_in_channels=3, backbone_hidden_dim=256,
        backbone_output_dim=512, seq_length=224, ema_alpha=0.1,
        mamba_d_state=64, mamba_headdim=64, mamba_num_heads=8,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")
    print(f"Feature dim: {model.feature_dim}")
    print(f"Mamba states:")
    print(f"  angle_state: {model.mamba.angle_state.shape}")
    print(f"  ssm_state:   {model.mamba.ssm_state.shape}")
    print(f"  k_state:     {model.mamba.k_state.shape}")
    print(f"  v_state:     {model.mamba.v_state.shape}\n")
    print("Simulating 5 frames:")
    for i in range(5):
        x = torch.randn(1, 3, 224)
        with torch.no_grad():
            logits = model(x)
        print(f"  Frame {i}: {x.shape} -> {logits.shape} v")
    print("\nv Forward pass OK")
