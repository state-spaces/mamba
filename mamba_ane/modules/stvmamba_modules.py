import torch
import torch.nn as nn
from einops import rearrange
# Import your new parallel block
from mamba_ane.modules.mamba3_parallel import Mamba3ParallelPortable

class STSS_ANE(nn.Module):
    """
    ANE-native 4-direction Spatial-Temporal Selective Scan.
    Expects input shape: (B, L, D) where B=1 and L = grid_h * grid_w.
    """
    def __init__(self, d_model: int, grid_h: int = 8, grid_w: int = 8, **mamba_kwargs):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.d_model = d_model
        
        # 4 independent parallel Mamba blocks
        self.scans = nn.ModuleList([
            Mamba3ParallelPortable(d_model=d_model, **mamba_kwargs)
            for _ in range(4)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) - For ANE, B must be 1.
        
        # 1. Row Forward (Standard sequence)
        scan0 = x
        
        # 2. Row Reverse (Flip sequence)
        scan1 = torch.flip(x, dims=[1])
        
        # 3. Col Forward (Transpose spatial grid)
        scan2 = rearrange(x, 'b (h w) c -> b (w h) c', h=self.grid_h, w=self.grid_w)
        
        # 4. Col Reverse (Transpose then flip)
        scan3 = torch.flip(scan2, dims=[1])

        # Execute the 4 Mambas (Sequential in Python, but fast on ANE)
        out0 = self.scans[0](scan0)
        out1 = self.scans[1](scan1)
        out2 = self.scans[2](scan2)
        out3 = self.scans[3](scan3)

        # Un-flip and un-transpose the outputs to align back to the original grid
        out1 = torch.flip(out1, dims=[1])
        out2 = rearrange(out2, 'b (w h) c -> b (h w) c', h=self.grid_h, w=self.grid_w)
        out3 = rearrange(torch.flip(out3, dims=[1]), 'b (w h) c -> b (h w) c', h=self.grid_h, w=self.grid_w)

        # Sum the 4 directions
        return out0 + out1 + out2 + out3