import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple

# Assuming you placed this in mamba_ane/modules/stvmamba_modules.py
from mamba_ane.modules.stvmamba_modules import STSS_ANE 
from mamba_ane.models.hybrid_parallel import ParallelHybridBackbone

# ==============================================================================
# ANE-Safe Unpatchify (Output Layer for Tri-planes)
# ==============================================================================
class LinearUnpatchifyANE(nn.Module):
    """
    Simplified, CoreML-friendly linear unpatchify layer.
    Maps (B, L, D) -> (B, C, H, W) where L = grid_h * grid_w.
    """
    def __init__(
        self,
        model_dim: int,
        cond_dim: int,         # Dimension of Action/IMU conditioning
        out_channels: int,     # Usually 64 for ICANSII Tri-planes
        patch_size: int = 16,  # 16x16 patches
        grid_h: int = 8,       # 8x8 grid = 64 sequence length
        grid_w: int = 8,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        
        # Projection outputs all pixels for a single patch
        output_dim = out_channels * (patch_size * patch_size)
        
        self.norm = nn.LayerNorm(model_dim, eps=1e-5) # LayerNorm is highly ANE-optimized
        
        # AdaLN Modulation (Action/IMU conditioning)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * model_dim),
        )
        self.proj = nn.Linear(model_dim, output_dim)

        # Final smoothing using 2D Conv (ANE loves Conv2D)
        self.final_smooth = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # Initialize modulation and smoothing to zero-impact initially
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.dirac_(self.final_smooth.weight) # Identity init
        nn.init.zeros_(self.final_smooth.bias)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, model_dim)
        cond_emb: (B, cond_dim) - the Action/IMU warp context
        """
        # 1. AdaLN Modulation
        x = self.norm(x)
        scale, shift = self.modulation(cond_emb).chunk(2, dim=-1)
        
        # Explicit unsqueeze for broadcasting on ANE
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        x = x * (1.0 + scale) + shift
        
        # 2. Linear Projection
        x = self.proj(x) # (B, L, out_channels * patch_h * patch_w)
        
        # 3. Rearrange to 2D Spatial Grid
        # CoreML compiles this to standard memory strides perfectly.
        x = rearrange(
            x,
            'b (h w) (c ph pw) -> b c (h ph) (w pw)',
            h=self.grid_h, w=self.grid_w, 
            ph=self.patch_size, pw=self.patch_size, 
            c=self.out_channels
        )

        # 4. Final spatial smooth
        x = self.final_smooth(x)
        
        return x

# ==============================================================================
# Full ICANSII STVMamba Hybrid Predictor
# ==============================================================================
class STVMambaHybridANE(nn.Module):
    """
    End-to-end Tri-plane predictor:
    Folded Buffer -> Conv1D Backbone -> STSS (4-way Mamba) -> Unpatchify -> Tri-plane
    """
    def __init__(
        self,
        in_channels: int = 320,  # e.g., 5 frames * 64 channels
        d_model: int = 256,      # Multiple of 64 for ANE
        cond_dim: int = 16,      # Size of IMU/Action vector
        out_channels: int = 64,  # Channels of predicted Tri-plane
        patch_size: int = 16,
        grid_size: int = 8,      # 8x8 = seq_len 64
        **mamba_kwargs
    ):
        super().__init__()
        
        # 1. Temporal Fold Backbone (STDS Conv equivalent)
        self.backbone = ParallelHybridBackbone(
            in_channels=in_channels, 
            hidden_dim=d_model // 2, 
            d_model=d_model
        )
        
        # 2. 4-way Spatial-Temporal Selective Scan
        self.stss = STSS_ANE(
            d_model=d_model, 
            grid_h=grid_size, 
            grid_w=grid_size, 
            **mamba_kwargs
        )
        
        # 3. Decoder
        self.decoder = LinearUnpatchifyANE(
            model_dim=d_model,
            cond_dim=cond_dim,
            out_channels=out_channels,
            patch_size=patch_size,
            grid_h=grid_size,
            grid_w=grid_size
        )

    def forward(self, triplane_buffer: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        triplane_buffer: (B=1, T*C, L) — The folded input from Phase 1
        cond_emb: (B=1, cond_dim) — IMU/Action conditioning
        
        Returns:
        next_triplane: (B=1, C, H, W) — The predicted spatial occupancy
        """
        # (B, T*C, L) -> (B, L, d_model)
        feat = self.backbone(triplane_buffer)
        
        # (B, L, d_model) -> (B, L, d_model)
        mamba_out = self.stss(feat)
        
        # (B, L, d_model) -> (B, C, H, W)
        pred_plane = self.decoder(mamba_out, cond_emb)
        
        return pred_plane