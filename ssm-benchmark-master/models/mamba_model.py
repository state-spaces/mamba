import torch
from torch import nn
from simple_mamba.mamba import MambaBlock as MambaLayer
from simple_mamba.mamba import MambaConfig

from .common import MATCH


class GLU(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        out = self.linear(x)
        return out[:, :, :x.shape[2]] * torch.sigmoid(out[:, :, x.shape[2]:])


class MambaBlock(torch.nn.Module):
    def __init__(self, hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm, ssm_type):
        super().__init__()
        mamba_config = MambaConfig(ssm_type=ssm_type,
                                   n_layers=1,  # doesn't effect the model because we only use MambaBlock, but is required #TODO: refactor
                                   d_model=hidden_dim,
                                   d_state=state_dim,
                                   d_conv=conv_dim,
                                   expand_factor=expansion
        )
        print(mamba_config)
        self.mamba = MambaLayer(mamba_config)
        if glu:
            self.glu = GLU(hidden_dim)
        else:
            self.glu = None
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if norm in ["layer"]:
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm in ["batch"]:
            # TODO: add batch norm
            raise RuntimeError("dimensions don't agree for batch norm to work")
            self.norm = nn.BatchNorm1d(hidden_dim)
        self.prenorm = prenorm

    def forward(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(self.activation(x))
        if self.glu is not None:
            x = self.glu(x)
        x = self.dropout(x)
        x = x + skip
        if not self.prenorm:
            x = self.norm(x)
        return x


class Mamba(torch.nn.Module):
    def __init__(self, num_blocks, input_dim, output_dim, hidden_dim, state_dim, conv_dim, expansion, dropout, glu,
                 norm, prenorm, dual, pooling="mean", ssm_type="S6-Real"):
        super().__init__()
        self.linear_encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[MambaBlock(hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm, ssm_type) for _ in
              range(num_blocks)])
        self.linear_decoder = nn.Linear(hidden_dim, output_dim)
        self.pooling = pooling
        self.softmax = nn.LogSoftmax(dim=1)
        self.dual = dual
        if dual:
            self.match = MATCH(output_dim * 2, output_dim)

    def forward(self, x):
        x = self.linear_encoder(x)
        x = self.blocks(x)
        if self.pooling in ["mean"]:
            x = torch.mean(x, dim=1)
        elif self.pooling in ["max"]:
            x = torch.max(x, dim=1)[0]
        elif self.pooling in ["last"]:
            x = x[:, -1, :]
        else:
            x = x  # no pooling
        x = self.linear_decoder(x)
        if self.dual:
            (x1, x2) = torch.split(x, int(x.shape[0] / 2))
            x = self.match(torch.concatenate((x1, x2), dim=1))
        return torch.softmax(x, dim=1)