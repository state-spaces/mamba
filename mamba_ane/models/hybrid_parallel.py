# mamba_ane/models/hybrid_parallel.py
"""
Hybrid parallel models: sequence-preserving Conv1D backbone + Mamba3ParallelPortable.

ParallelHybridBackbone:    (B, C_in, L) → (B, L, d_model)   [stride-1 Conv1D, SiLU]
StatefulMambaParallelHybrid:  full ANE-exportable model (no state buffers)
OGStatefulMambaParallelHybrid: golden reference using Mamba3 (CUDA), for GPU parity only
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

from mamba_ane.modules.mamba3_parallel import Mamba3ParallelPortable


class ParallelHybridBackbone(nn.Module):
    """
    Sequence-preserving Conv1D feature extractor.

    Two stride-1 same-padding Conv1d layers → SiLU.  The sequence length L is
    unchanged, which is required for Mamba3ParallelPortable to receive (B, L, d_model).
    At least one Conv1d ensures the ANE workload threshold is met.

    Input:  (B, C_in, L)
    Output: (B, L, d_model)
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim:  int = 64,
        d_model:     int = 256,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2, bias=True)
        self.act1  = nn.SiLU()
        self.conv2 = nn.Conv1d(hidden_dim,  d_model,    kernel_size=3, padding=1, bias=True)
        self.act2  = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, L) → (B, L, d_model)"""
        x = self.act1(self.conv1(x))    # (B, hidden_dim, L)
        x = self.act2(self.conv2(x))    # (B, d_model,    L)
        return x.transpose(-1, -2)      # (B, L, d_model)


class StatefulMambaParallelHybrid(nn.Module):
    """
    ParallelHybridBackbone → Mamba3ParallelPortable → mean-pool → Linear.

    Stateless: no register_buffer, no ct.StateType.
    Input:  (B, C_in, seq_length) — seq_length must be fixed at export time.
    Output: (B, num_classes)
    """
    def __init__(
        self,
        num_classes:  int   = 1000,
        in_channels:  int   = 3,
        hidden_dim:   int   = 64,
        d_model:      int   = 256,
        d_state:      int   = 64,
        headdim:      int   = 64,
    ):
        super().__init__()
        self.backbone   = ParallelHybridBackbone(in_channels, hidden_dim, d_model)
        self.mamba      = Mamba3ParallelPortable(d_model=d_model, d_state=d_state, headdim=headdim)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, L) → (B, num_classes)"""
        feat     = self.backbone(x)         # (B, L, d_model)
        mamba_out = self.mamba(feat)         # (B, L, d_model)
        pooled   = mamba_out.mean(dim=1)    # (B, d_model)
        return self.classifier(pooled)       # (B, num_classes)


# ---------------------------------------------------------------------------
# OG reference model — import mamba_ssm only when available (GPU only)
# ---------------------------------------------------------------------------

def _build_og_model(d_model, d_state, headdim, num_classes, in_channels, hidden_dim,
                    device=None, dtype=torch.float32):
    """Factory that imports mamba_ssm lazily (not available on dorian-mac)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mamba_ssm.modules.mamba3 import Mamba3

    class OGStatefulMambaParallelHybrid(nn.Module):
        """
        Golden reference for GPU parity tests.
        Same topology as StatefulMambaParallelHybrid but uses Mamba3 (Triton) internally.
        NOT for CoreML export.
        """
        def __init__(self):
            super().__init__()
            self.backbone   = ParallelHybridBackbone(in_channels, hidden_dim, d_model)
            self.mamba      = Mamba3(
                d_model=d_model, d_state=d_state, expand=2, headdim=headdim,
                ngroups=1, rope_fraction=0.5, is_mimo=False, is_outproj_norm=False,
                layer_idx=0, device=device, dtype=dtype,
            )
            self.classifier = nn.Linear(d_model, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat      = self.backbone(x)
            mamba_out = self.mamba(feat)          # (B, L, d_model)
            pooled    = mamba_out.mean(dim=1)
            return self.classifier(pooled)

        def load_from_portable(self, src: StatefulMambaParallelHybrid) -> None:
            """Copy weights from portable model into this OG reference."""
            with torch.no_grad():
                self.backbone.load_state_dict(src.backbone.state_dict())
                self.classifier.load_state_dict(src.classifier.state_dict())
                self.mamba.in_proj.weight.copy_(src.mamba.in_proj.weight)
                self.mamba.out_proj.weight.copy_(src.mamba.out_proj.weight)
                self.mamba.dt_bias.copy_(src.mamba.dt_bias)
                self.mamba.D.copy_(src.mamba.D)
                self.mamba.B_norm.weight.copy_(src.mamba.B_norm.weight)
                self.mamba.C_norm.weight.copy_(src.mamba.C_norm.weight)
                # portable: (nheads, d_state) → Mamba3: (nheads, 1, d_state)
                self.mamba.B_bias.copy_(src.mamba.B_bias.unsqueeze(1))
                self.mamba.C_bias.copy_(src.mamba.C_bias.unsqueeze(1))

    return OGStatefulMambaParallelHybrid()
