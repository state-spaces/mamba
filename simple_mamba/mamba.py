import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_mamba.pscan import pscan
from s4 import SSMKernelDiag, FFTConv

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison.

- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


@dataclass
class MambaConfig:
    discretizationA: str
    discretizationB: str
    initA_imag: str
    initA_real: str
    param_A_imag: str
    dt_is_selective: str
    channel_sharing: str
    ssm_type: str
    d_model: int  #  D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  #  N in paper/comments
    expand_factor: int = 2  #  E in paper/comments
    d_conv: int = 4
    A_imag_using_weight_decay: str = True

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  #  "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True  #  use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        # self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        #  x : (B, L, D)

        #  y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        # x = self.norm_f(x)

        return x

    def step(self, x, caches):
        #  x : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        #  y : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        #  x : (B, L, D)

        #  output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs: (B, ED, d_conv-1)

        #  output : (B, D)
        #  cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        #  projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        if config.ssm_type == "S6-Real":
            #  projects x to input-dependent Δ, B, C
            self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

            #  projects Δ from dt_rank to d_inner
            self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

            #  dt initialization
            #  dt weights
            dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
            if config.dt_init == "constant":
                nn.init.constant_(self.dt_proj.weight, dt_init_std)
            elif config.dt_init == "random":
                nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # dt bias
            dt = torch.exp(
                torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(
                    config.dt_min)
            ).clamp(min=config.dt_init_floor)
            inv_dt = dt + torch.log(
                -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)
            # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            #  todo : explain why removed

            # S4D real initialization
            A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.A_log = nn.Parameter(
                torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
            self.D = nn.Parameter(torch.ones(config.d_inner))

        elif config.ssm_type == "S6-Real-complex-bias":
            assert self.config.channel_sharing

            self.BC_dims = config.d_state * config.d_inner

            #  projects x to input-dependent Δ, B, C
            self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * self.BC_dims, bias=False)

            self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * self.BC_dims, bias=False)
            self.C_bias_real = torch.randn(config.d_inner, config.d_state, dtype=torch.float)
            self.C_bias_imag = torch.randn(config.d_inner, config.d_state, dtype=torch.float)
            self.B_bias_real = torch.ones(config.d_inner, config.d_state, dtype=torch.float)
            self.B_bias_imag = torch.zeros(config.d_inner, config.d_state, dtype=torch.float)
            self.C_bias_real = nn.Parameter(self.B_bias_real)
            self.C_bias_imag = nn.Parameter(self.B_bias_imag)
            self.B_bias_real = nn.Parameter(self.C_bias_real)
            self.B_bias_imag = nn.Parameter(self.C_bias_imag)

            #  dt initialization
            #  dt weights
            inv_dt = torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(
                config.dt_min)
            self.inv_dt = nn.Parameter(inv_dt)

            A = 1/2 + 0*torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.log_A_real = nn.Parameter(
                torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
            A_imag = math.pi * torch.arange(config.d_state).repeat(config.d_inner, 1)
            A_imag[0] += 1e-4
            self.A_imag = nn.Parameter(A_imag)

            self.D = nn.Parameter(torch.randn(config.d_inner))

        elif config.ssm_type == "S6-Complex":
            #  projects x to input-dependent Δ, B, C
            if config.channel_sharing == "False":
                self.BC_dims = config.d_state * config.d_inner
            elif config.channel_sharing == "True":
                self.BC_dims = config.d_state
            else:
                raise NotImplementedError
            self.x_proj_real = nn.Linear(config.d_inner, config.dt_rank + 2 * self.BC_dims, config.bias)
            self.x_proj_complex = nn.Linear(config.d_inner, 2 * self.BC_dims, config.bias)


            # # init the imaginary part of x_proj to 0
            # self.x_proj.weight.imag.data.zero_()

            #  projects Δ from dt_rank to d_inner
            if config.dt_is_selective == "True":
                self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True, dtype=torch.float)

                #  dt initialization
                #  dt weights
                dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
                if config.dt_init == "constant":
                    nn.init.constant_(self.dt_proj.weight, dt_init_std)
                elif config.dt_init == "random":
                    nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
                else:
                    raise NotImplementedError

                # dt bias
                dt = torch.exp(
                    torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(
                        config.dt_min)
                ).clamp(min=config.dt_init_floor)
                inv_dt = dt + torch.log(
                    -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
                with torch.no_grad():
                    self.dt_proj.bias.copy_(inv_dt)
                self.dt_proj.bias._no_reinit = True  # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            elif config.dt_is_selective == "False":
                inv_dt = torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(
                        config.dt_min)
                self.inv_dt = nn.Parameter(inv_dt)
            else:
                raise NotImplementedError


            if config.initA_real == "S4":
                log_A_real = torch.log(0.5 * torch.ones(config.d_inner, config.d_state))
            elif config.initA_real == "S6":
                A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
                log_A_real = torch.log(A)
            else:
                raise NotImplementedError

            if config.initA_imag == "uniform":
                A_imag = 2*math.pi * torch.linspace(0, 1, config.d_state).repeat(config.d_inner, 1)
            elif config.initA_imag == "uniform_small":
                A_imag = 2 * math.pi * torch.linspace(0, 0.01, config.d_state).repeat(config.d_inner, 1)
            elif config.initA_imag == "zero":
                A_imag = 0 * torch.linspace(0, 0.01, config.d_state).repeat(config.d_inner, 1)
            elif config.initA_imag == "rand":
                A_imag = (torch.rand(log_A_real.shape) * 2 * torch.pi) - torch.pi
            elif config.initA_imag == "rand_small":
                A_imag = ((torch.rand(log_A_real.shape) * 2 * torch.pi) - torch.pi)/10
            elif config.initA_imag == "S4":
                A_imag = math.pi * torch.arange(config.d_state).repeat(config.d_inner, 1)
            else:
                raise NotImplementedError

            self.log_A_real = nn.Parameter(log_A_real)
            self.log_A_real._no_weight_decay = True
            self.A_imag = nn.Parameter(A_imag)
            if config.param_A_imag == "fixed":
                self.A_imag.requires_grad = False
            elif config.param_A_imag == "normal":
                pass
            else:
                raise NotImplementedError

            if config.A_imag_using_weight_decay == "True":
                pass
            elif config.A_imag_using_weight_decay == "False":
                self.A_imag._no_weight_decay = True
            else:
                raise NotImplementedError

            # # initialize A to be complex but with imaginary part 0
            # A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            # self.log_A_real = nn.Parameter(
            #     torch.log(A))
            # self.A_imag = nn.Parameter(torch.zeros_like(A))

            # D does not need to be complex since it is multiplied by x, and we take real part of the output
            self.D = nn.Parameter(torch.randn(config.d_inner))

        elif config.ssm_type == "S4D-Complex":
            self.ssm_kernel = FFTConv(d_model=config.d_inner,
                                      d_state=config.d_state,
                                      activation='id',
                                      transposed=False,
                                      mode='s4d',
                                      is_real=False,
                                      shared=config.channel_sharing,
                                      init="diag-lin")
        elif config.ssm_type == "S4D-Real":
            self.ssm_kernel = FFTConv(d_model=config.d_inner,
                                      d_state=config.d_state,
                                      activation='id',
                                      transposed=False,
                                      mode='s4d',
                                      is_real=True)
        elif config.ssm_type == "conv":
            self.ssm_kernel = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                        kernel_size=config.d_state, bias=False,
                                        groups=config.d_inner,
                                        padding=config.d_state - 1)
        else:
            print("type",config.ssm_type)
            raise NotImplementedError

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        #  x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  #  (B, L, ED), (B, L, ED)

        #  x branch
        x = x.transpose(1, 2)  #  (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  #  (B, L, ED)

        x = F.silu(x)
        x = x*0+1
        y = self.ssm(x)

        #  z branch
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)  #  (B, L, D)

        return output

    def ssm(self, x):
        #  x : (B, L, ED)

        #  y : (B, L, ED)
        if self.config.ssm_type == "S6-Real-complex-bias":
            D = self.D
            A = -torch.exp(self.log_A_real) - 1j * self.A_imag  # (ED, N)


            deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)

            _, B, C = torch.split(deltaBC, [self.config.dt_rank, self.BC_dims, self.BC_dims],
                                      dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)

            b, l, ed = x.shape
            B = B.reshape(b, l, ed, self.config.d_state)
            C = C.reshape(b, l, ed, self.config.d_state)
            B_bias = self.B_bias_real + 1j * self.B_bias_imag
            C_bias = self.C_bias_real + 1j * self.C_bias_imag
            B_bias = B_bias.unsqueeze(0).unsqueeze(0) # (1, 1, L, N)
            C_bias = C_bias.unsqueeze(0).unsqueeze(0) # (1, 1, L, N)
            if self.config.A_imag_using_weight_decay:
                B = B*0
                C = C*0
            B = B + B_bias
            C = C + C_bias

            delta_new = torch.exp(self.inv_dt)
            delta = delta_new.unsqueeze(0).unsqueeze(0)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)
            # y = self.selective_scan_seq(x, delta, A, B, C, D)

            return y.real

        elif self.config.ssm_type == "S6-Real":
            A = -torch.exp(self.A_log.float())  # (ED, N)
            D = self.D.float()
            #  TODO remove .float()

            deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)

            delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                      dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
            delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

            return y

        elif self.config.ssm_type == "S6-Complex":
            A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (ED, N)
            D = self.D

            if self.config.initA_imag == "zero" and self.config.param_A_imag == "fixed":
                if torch.any(A.imag != 0):
                    print("zeros did learn something on fixed")
                    raise

            deltaBC_real = self.x_proj_real(x) #  (B, L, dt_rank+2*N)
            delta, B_real, C_real = torch.split(deltaBC_real, [self.config.dt_rank, self.BC_dims, self.BC_dims],
                                      dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
            BC_complex = self.x_proj_complex(x)
            B_imag, C_imag = torch.split(BC_complex,
                                                [self.BC_dims, self.BC_dims],
                                                dim=-1)

            B = B_real + 1j * B_imag
            C = C_real + 1j * C_imag
            if not self.config.channel_sharing:
                b, l, ed = x.shape
                B = B.reshape(b, l, ed, self.config.d_state)
                C = C.reshape(b, l, ed, self.config.d_state)

            if self.config.dt_is_selective == "True":
                delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)
            elif self.config.dt_is_selective == "False":
                #delta = torch.zeros(delta.shape) + torch.exp(self.inv_dt)
                delta_new = torch.exp(self.inv_dt)
                delta = torch.zeros([B.shape[0], B.shape[1], A.shape[0]], device=A.device) + delta_new #  (B, L, ED)
            else:
                raise NotImplementedError

            if self.config.pscan and self.config.ssm_type != "S6-Real-complex-bias":
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

            return y.real
        elif self.config.ssm_type == "S4D-Complex" or self.config.ssm_type == "S4D-Real":
            return self.ssm_kernel(x)[0]
        elif self.config.ssm_type == "conv":
            x_bdl = x.transpose(-1, -2)
            L = x_bdl.size(-1)
            out = self.ssm_kernel(x_bdl)[:, :, :L]
            return out.transpose(-1, -2)
        else:
            raise NotImplementedError

    def selective_scan(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)
        if self.config.discretizationA == "yuval_disc" and (self.config.ssm_type == "S6-Complex" or self.config.ssm_type == "S6-Real-complex-bias"):
            deltaA = torch.exp(delta.unsqueeze(-1) * A.real +1j * A.imag)
        elif self.config.discretizationA == "normal":
            deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        else:
            print("disc",self.config.discretizationA)
            raise NotImplementedError

        if self.config.channel_sharing and self.config.ssm_type != "S6-Real-complex-bias":
            B = B.unsqueeze(2)

        if self.config.discretizationB == "s6":
            deltaB = delta.unsqueeze(-1) * B  #  (B, L, ED, N)
        elif self.config.discretizationB == "zoh":
            #deltaB = B * torch.exp(delta.unsqueeze(-1) * A - 1.) / A  #  (B, L, ED, N)
            deltaB = B * (torch.exp(delta.unsqueeze(-1) * A) - 1.) / A
            # C = C * (torch.exp(dtA)-1.) / A
        else:
            raise NotImplementedError

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)
        if self.config.ssm_type == "S6-Real-complex-bias":
            deltaA = deltaA.expand_as(BX)
        hs = pscan(deltaA, BX)

        if self.config.channel_sharing and self.config.ssm_type != "S6-Real-complex-bias":
            C = C.unsqueeze(2)
        y = (hs * C).sum(dim=3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)


        # if self.config.ssm_type != "S6-Real-complex-bias":
        #     y = y + D * x
        y = y + D.unsqueeze(0).unsqueeze(0)*x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        _, L, _ = x.shape

        # deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        # deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        if self.config.discretizationA == "yuval_disc" and (
                self.config.ssm_type == "S6-Complex" or self.config.ssm_type == "S6-Real-complex-bias"):
            deltaA = torch.exp(delta.unsqueeze(-1) * A.real + 1j * A.imag)
        elif self.config.discretizationA == "normal":
            deltaA = torch.exp(delta.unsqueeze(-1) * A)  #

        if self.config.discretizationB == "s6":
            deltaB = delta.unsqueeze(-1) * B  #  (B, L, ED, N)

        elif self.config.discretizationB == "zoh":
            # deltaB = B * torch.exp(delta.unsqueeze(-1) * A - 1.) / A  #  (B, L, ED, N)
            deltaB = B * (torch.exp(delta.unsqueeze(-1) * A) - 1.) / A


        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)
        if self.config.ssm_type == "S6-Real-complex-bias":
            deltaA = deltaA.expand_as(BX)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  #  (B, L, ED, N)

        if self.config.ssm_type == "S6-Real-complex-bias":
            y = (hs * C).sum(dim=3)
        else:
            y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D.unsqueeze(0).unsqueeze(0) * x

        return y

    #  -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs : (B, ED, d_conv-1)

        #  y : (B, D)
        #  cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  #  (B, ED), (B, ED)

        #  x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]  #  (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  #  (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        #  x : (B, ED)
        #  h : (B, ED, N)

        #  y : (B, ED)
        #  h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        #  TODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  #  (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)

        h = deltaA * h + BX  #  (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  #  (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        #  todo : pq h.squeeze(1) ??
        return y, h.squeeze(1)


#  taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
