"""Standalone version of Structured State Space sequence model (S4)."""

from collections import defaultdict
from typing import Optional, Mapping, Tuple, Union
import logging
from functools import partial
import math
import numpy as np
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from einops import rearrange, repeat

# Function aliases
contract = torch.einsum

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
log = get_logger(__name__)

"""Structured matrix kernels"""

# Try CUDA extension
try:
    from extensions.kernels.cauchy import cauchy_mult as cauchy_cuda
    from extensions.kernels.vandermonde import log_vandermonde_cuda
    has_cuda_extension = True
    log.info("CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) found.")
except:
    log.warning(
        "CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) not found. Install by going to extensions/kernels/ and running `python setup.py install`, for improved speed and memory efficiency. Note that the kernel changed for state-spaces 4.0 and must be recompiled."
    )
    has_cuda_extension = False

# Try pykeops
try:
    import pykeops
    from pykeops.torch import Genred
    has_pykeops = True
    log.info("Pykeops installation found.")

    def _broadcast_dims(*tensors):
        max_dim = max([len(tensor.shape) for tensor in tensors])
        tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
        return tensors

    def cauchy_keops(v, z, w):
        expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
        expr_denom = 'ComplexMult(z-w, z-Conj(w))'

        cauchy_mult = Genred(
            f'ComplexDivide({expr_num}, {expr_denom})',
            [
                'v = Vj(2)',
                'z = Vi(2)',
                'w = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        v, z, w = _broadcast_dims(v, z, w)
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

        r = 2*cauchy_mult(v, z, w, backend='GPU')
        return _r2c(r)

    def log_vandermonde_keops(v, x, L):
        expr = 'ComplexMult(v, ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'v = Vj(2)',
                'x = Vj(2)',
                'l = Vi(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        v, x, l = _broadcast_dims(v, x, l)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(v, x, l, backend='GPU')
        return 2*_r2c(r).real

    def log_vandermonde_transpose_keops(u, v, x, L):
        """
        u: ... H L
        v: ... H N
        x: ... H N
        Returns: ... H N

        V = Vandermonde(a, L) : (H N L)
        contract_L(V * u * v)
        """
        expr = 'ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'u = Vj(2)',
                'v = Vi(2)',
                'x = Vi(2)',
                'l = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        u, v, x, l = _broadcast_dims(u, v, x, l)
        u = _c2r(u)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(u, v, x, l, backend='GPU')
        return _r2c(r)

except ImportError:
    has_pykeops = False
    if not has_cuda_extension:
        log.warning(
            "Falling back on slow Cauchy and Vandermonde kernel. Install at least one of pykeops or the CUDA extension for better speed and memory efficiency."
        )

# Fallback versions
def cauchy_naive(v, z, w):
    """
    v: (..., N)
    z: (..., L)
    w: (..., N)
    returns: (..., L) \sum v/(z-w)
    """
    v = _conj(v)
    w = _conj(w)
    cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1)) # (... N L)
    return torch.sum(cauchy_matrix, dim=-2)

def log_vandermonde_naive(v, x, L, conj=True):
    """
    v: (..., N)
    x: (..., N)
    returns: (..., L) \sum v x^l
    """
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = contract('... n, ... n l -> ... l', v, vandermonde_matrix) # (... L)
    return 2*vandermonde_prod.real

def log_vandermonde_transpose_naive(u, v, x, L):
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = contract('... l, ... n, ... n l -> ... n', u.to(x), v.to(x), vandermonde_matrix) # (... L)
    return vandermonde_prod



""" Simple nn.Module components """

def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

def LinearActivation(
        d_input, d_output, bias=True,
        transposed=False,
        activation=None,
        activate=False, # Apply activation as part of this module
        **kwargs,
    ):
    """Returns a linear nn.Module with control over axes order, initialization, and activation."""

    # Construct core module
    linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    if activation is not None and activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X

"""Misc functional utilities"""

def power(L, A, v=None):
    """Compute A^L and the scan sum_i A^i v_i.

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        if v is None:
            powers = [powers[-1] @ powers[-1]]
        else:
            powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


"""HiPPO utilities"""

def transition(measure, N, **measure_args):
    """A, B transition matrices for different measures.

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Legendre (translated)
    if measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A

        # Halve again for timescale correctness
        A *= 0.5
        B *= 0.5
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure in ['fourier', 'fout']:
        freqs = np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**.5
        B[0] = 1

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=torch.float):
    """Return low-rank matrix L such that A + L is normal."""

    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(.5+torch.arange(N, dtype=dtype)).unsqueeze(0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = torch.sqrt(1+2*torch.arange(N, dtype=dtype)) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0) # (2 N)
        P *= 2**(-0.5) # Halve the rank correct just like the original matrix was halved
    elif measure in ['fourier', 'fout']:
        P = torch.zeros(N)
        P[0::2] = 2**.5
        P[0] = 1
        P = P.unsqueeze(0)
    else: raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank-d, N, dtype=dtype)], dim=0) # (rank N)
    return P

def nplr(measure, N, rank=1, dtype=torch.float, diagonalize_precision=True, B_clip=2.0):
    """Constructs NPLR form of HiPPO matrices.

    Returns w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B

    measure: Name of HiPPO method.
    N: Size of recurrent A matrix (also known as `d_state` elsewhere).
    dtype: Single or double precision.
    diagonalize_precision: Calculate diagonalization in double precision.
    B_clip: Clip values of B, can help with stability. None for no clipping.
    """

    assert dtype == torch.float or dtype == torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype) # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0] # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype) # (r N)
    AP = A + torch.sum(P.unsqueeze(-2)*P.unsqueeze(-1), dim=-3)

    # We require AP to be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    if (err := torch.sum((_A - _A[0,0]*torch.eye(N))**2) / N) > 1e-5: # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
        print("WARNING: HiPPO matrix not skew symmetric", err)


    # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
    # Imaginary part can use eigh instead of eig
    W_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision: AP = AP.to(torch.double)
    # w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
    W_im, V = torch.linalg.eigh(AP*-1j) # (..., N) (..., N, N)
    if diagonalize_precision: W_im, V = W_im.to(cdtype), V.to(cdtype)
    W = W_re + 1j * W_im
    # Check: V W V^{-1} = A
    # print("check", V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2))


    # Only keep half of each conjugate pair
    _, idx = torch.sort(W.imag)
    W_sorted = W[idx]
    V_sorted = V[:, idx]

    # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
    # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
    V = V_sorted[:, :N//2]
    W = W_sorted[:N//2]  # Only keep negative imaginary components
    assert W[-2].abs() > 1e-4, "Only 1 zero eigenvalue allowed in diagonal part of A"
    if W[-1].abs() < 1e-4:
        V[:, -1] = 0.
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j

    _AP = V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2)
    if ((err := torch.sum((2*_AP.real-AP)**2)/N) > 1e-5):
        print("Warning: Diagonalization of A matrix not numerically precise - error", err)
    # print("check", V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2))

    V_inv = V.conj().transpose(-1, -2)

    # C = initial_C(measure, N, dtype=dtype)
    B = contract('ij, j -> i', V_inv, B.to(V)) # V^* B
    # C = contract('ij, j -> i', V_inv, C.to(V)) # V^* C
    P = contract('ij, ...j -> ...i', V_inv, P.to(V)) # V^* P

    if B_clip is not None:
        B = B.real + 1j*torch.clamp(B.imag, min=-B_clip, max=B_clip)

    # W represents the imaginary part of the DPLR form: A = W - PP^*
    # Downstream classes just call this A for simplicity,
    # which is also more consistent with the diagonal case
    return W, P, B, V

def dplr(
    init='hippo',
    N=64, rank=1, H=1,
    dtype=torch.float,
    real_random=False,
    real_scale=1.0,
    imag_random=False,
    imag_scale=1.0,
    B_random=False,
    B_init='constant',
    B_scale=1.0,
    P_scale=1.0,
    normalize=False,
):
    """Directly construct a DPLR matrix.

    Args:
    - init: (str) ['rand', 'lin', inv', 'real', 'hippo'] Choices for initialization of A.
          Most of these affect the imaginary part of A, except for 'real'.
    - real_random: (bool) Initialize A.real in -U[0, 1]. Otherwise, initialize to -1/2.
    - real_scale: (float) Scaling factor of real part of A.
    - imag_random: (bool) Initialize A.imag randomly.
    - imag_scale: (bool) Scaling factor of imaginary part of A.
    - B_init: (str) ['constant' | 'random' | 'alternating' | 'unit-cw' | 'unit-ccw' | 'hippo']
          Choices for initialization of B.
    - B_scale: (float) Scaling factor for B
    - P_scale: (float) Scaling factor for P
    - normalize: (bool) Apply an automatic normalization factor on B
    """
    assert dtype == torch.float or dtype == torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble

    pi = torch.tensor(math.pi)

    # Construct real part of diagonal A (must be non-negative)
    if real_random:
        real_part = torch.rand(H, N//2)
    else:
        real_part = .5 * torch.ones(H, N//2)
    real_part = real_scale * real_part

    # Construct imaginary part of diagonal A (must be non-negative)
    if imag_random:
        imag_part = N//2 * torch.rand(H, N//2)
    else:
        imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)

    if init in ['random', 'rand']:
        imag_part = torch.exp(torch.randn(H, N//2))
    elif init == 'real':
        imag_part = 0 * imag_part
        if real_random:
            real_part = torch.rand(H, N//2) * N//2
        else:
            # This is the S4D-Real method described in the S4D paper
            # The A matrix is diag(-1, -2, ..., -N), which are the eigenvalues of the HiPPO matrix
            real_part = 1 + repeat(torch.arange(N//2), 'n -> h n', h=H)
    elif init in ['linear', 'lin']:
        imag_part = pi * imag_part
    elif init in ['inverse', 'inv']: # Based on asymptotics of the default HiPPO matrix
        imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    elif init in ['inverse2', 'inv2']:
        imag_part = 1/pi * N * (N/(1+imag_part)-1)
    elif init in ['quadratic', 'quad']:
        imag_part = 1/pi * (1+2*imag_part)**2
    elif init in ['legs', 'hippo']:
        A, _, _, _ = nplr('legs', N)
        imag_part = -A.imag  # Positive
    else: raise NotImplementedError
    imag_part = imag_scale * imag_part

    # Construct diagonal A
    A = -real_part - 1j * imag_part  # Force negative real and imag
    assert torch.all(A.real < 1e-4) and torch.all(A.imag <= 0.0)  # Allow some tolerance for numerical precision on real part

    # Initialize B
    if B_random:
        log.warning("'B_random' is deprecated in favor of B_init='random' and will be deprecated in a future version.")
    if init in ['legs', 'hippo']:
        log.info(f'Initializing with S4D-LegS and ignoring argument {B_init=}')
        # Special initialization using the HiPPO B matrix
        # Note that theory (from S4D paper) says that B should be halved
        # to match DPLR but we drop this 0.5 factor for simplicity
        _, P, B, _ = nplr('legs', N, B_clip=2.0)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    elif B_init == 'constant':
        B = torch.ones(H, N//2, dtype=dtype)
    elif B_init == 'random':
        B = torch.randn(H, N//2, dtype=dtype)
    elif B_init == 'alternating':  # Seems to track 'constant' exactly for some reason
        B = torch.ones(H, N//4, 2, dtype=dtype)
        B[:, :, 1] *= -1
        B = B.view(H, N//2)
    elif B_init == 'unit-cw':
        z = torch.tensor(torch.exp(-2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    elif B_init == 'unit-ccw':
        z = torch.tensor(torch.exp(2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    else: raise NotImplementedError
    B *= B_scale

    # Experimental feature that appeared in earlier versions of HTTYH (not extensively tested)
    # Seems more principled for normalization theoretically, but seemed to hurt on PathX
    if normalize:
        norm = -B/A # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
        B = B / zeta**.5

    # Initialize P
    if B_init in ['legs', 'hippo']:
        # P constructed earlier
        P = repeat(P, 'r n -> r h n', h=H).clone().contiguous()
    else:
        P = torch.randn(rank, H, N//2, dtype=dtype)
        P = P * P_scale

    # Initialize V (only used in testing)
    V = torch.eye(N, dtype=dtype)[:, :N//2]
    V = repeat(V, 'n m -> h n m', h=H)

    return A, P, B, V

def ssm(init, N, R, H, **ssm_args):
    """Dispatcher to create single SSM initialization

    N: state size
    R: rank (for DPLR parameterization)
    H: number of independent SSM copies
    """

    if init.startswith("diag") or init.startswith("dplr"):
        if init.startswith("diag"):
            ssm_args["P_scale"] = 0.0
        args = init[4:].split("-")
        assert args[0] == ""
        if len(args) > 1:
            ssm_args["init"] = args[1]
        A, P, B, V = dplr(N=N, rank=R, H=H, **ssm_args)
    else:
        A, P, B, V = nplr(init, N, R, **ssm_args)
        A = repeat(A, 'n -> s n', s=H)
        P = repeat(P, 'r n -> r s n', s=H)
        B = repeat(B, 'n -> s n', s=H)
        V = repeat(V, 'n m -> s n m', s=H)
    return A, P, B, V

combinations = {
    'hippo': ['legs', 'fourier'],
    'diag': ['diag-inv', 'diag-lin'],
    'all': ['legs', 'fourier', 'diag-inv', 'diag-lin'],
}

def combination(inits, N, R, S, **ssm_args):
    if isinstance(inits, str):
        inits = combinations[inits] if inits in combinations else [inits]

    assert S % len(inits) == 0, f"{S} independent trainable SSM copies must be multiple of {len(inits)} different inits"
    A, P, B, V = zip(
        *[ssm(init, N, R, S // len(inits), **ssm_args) for init in inits]
    )
    A = torch.cat(A, dim=0) # (S N)
    P = torch.cat(P, dim=1) # (R S N)
    B = torch.cat(B, dim=0) # (S N)
    V = torch.cat(V, dim=0) # (S N N)
    return A, P, B, V


"""SSM convolution kernels"""

def inv_transform(param, transform='none'):
    """Initialize a (positive) parameter under a transform."""
    param = torch.clamp(param, min=1e-4)
    if transform == 'none':
        return param
    elif transform == 'exp':
        return torch.log(param) # Some of the HiPPO methods have real part 0
    elif transform == 'relu':
        return param
    elif transform == 'sigmoid':
        return torch.logit(param)
    elif transform == 'softplus':
        return torch.log(torch.exp(param)-1)
    else: raise NotImplementedError

def param_transform(param, transform='none'):
    """Get a (positive) parameter under a transform."""
    if transform == 'none':
        p = param
    elif transform == 'exp':
        p = torch.exp(param)
    elif transform == 'relu':
        # JAX version seems to NaN if you allow 0's, although this code was fine without it
        p = F.relu(param)+1e-4
    elif transform == 'sigmoid':
        p = F.sigmoid(param)
    elif transform == 'softplus':
        p = F.softplus(param)
    else: raise NotImplementedError
    return p

class Kernel(nn.Module):
    """Interface for modules that produce convolution kernels.

    A main distinction between these and normal Modules is that the forward pass
    does not take inputs. It is a mapping from parameters to a tensor that can
    be used in other modules, in particular as a convolution kernel.

    Because of the unusual parameterization, these kernels may often want special
    hyperparameter settings on their parameters. The `register` method provides
    an easy interface for controlling this, and is intended to be used with an
    optimizer hook that can be found in train.py or example.py.

    This class also defines an interface for interacting with kernels *statefully*,
    in particular for state space models (SSMs). This interface handles the setting
    when a model can be converted from a "CNN" into an "RNN".
    _setup_step()
    step()
    default_state()
    forward_state()

    See ConvKernel for the simplest instantiation of this interface.
    """

    def __init__(
        self,
        d_model: int = 0,
        channels: int = 1,
        l_max: Optional[int] = None,
        lr: Union[float, Optional[Mapping]] = None,
        wd: Union[float, Optional[Mapping]] = 0.0,
        verbose: bool = True,
        **kwargs,
    ):
        """General interface.

        d_model (H): Model dimension, or number of independent convolution kernels created.
        channels (C): Extra dimension in the returned output (see .forward()).
            - One interpretation is that it expands the input dimension giving it C separate "heads" per feature.
              That is convolving by this kernel maps shape (B L D) -> (B L C D)
            - This is also used to implement a particular form of bidirectionality in an efficient way.
            - In general for making a more powerful model, instead of increasing C
              it is recommended to set channels=1 and adjust H to control parameters instead.
        l_max (L): Maximum kernel length (optional). If unspecified, most Kernel instantiations
            will return kernels of arbitrary length as passed into .forward().
        lr: Optional dictionary specifying special hyperparameters for .register().
            Passing in a number (e.g. 0.001) sets attributes of SSM parameters (A, B, dt).
            A custom optimizer hook is needed to configure the optimizer to set the learning rates appropriately for these parameters.
        wd: Same as lr, but for weight decay.
        """
        super().__init__()
        assert d_model > 0
        self.H = self.d_model = d_model
        self.L = self.l_max = l_max
        self.channels = channels
        self.lr = lr
        self.wd = wd
        self.verbose = verbose

        # Add a catch-all **kwargs to make it easier to change kernels
        # without manually moving other options passed in the config.
        # Good to log these just so it's explicit.
        if self.verbose and len(kwargs) > 0:
            log.info(f"{type(self)} extra kwargs: {kwargs}")

        # Logic for registering parameters
        # Case 1: lr: None | float
        #   All params should have this lr (None means inherit from global lr)
        # Case 2: lr: dict
        #   Specified params should have that lr, all others should be None
        if self.lr is None or isinstance(self.lr, float):
            self.lr_dict = defaultdict(lambda: self.lr)
        else:
            self.lr_dict = defaultdict(lambda: None)
            self.lr_dict.update(self.lr)

        # Same logic for weight decay
        # (but is always just set to 0.0 and hasn't been ablated)
        if self.wd is None or isinstance(self.wd, float):
            self.wd_dict = defaultdict(lambda: self.wd)
        else:
            self.wd_dict = defaultdict(lambda: None)
            self.wd_dict.update(self.wd)

    def forward(self, state=None, rate=1.0, L=None):
        """General interface to generate a global convolution kernel.

        state: Initial state for recurrent updates.
            E.g. for SSMs, this should have shape (B, H, N) (batch, d_model, d_state).
        rate: Relative sampling rate.
        L: Target kernel length.

        Returns:
          - (C, H, L) (channels, d_model, l_kernel) The convolution kernel.
          - (B, H, L) (batch, d_model, l_kernel)
              Extra information for how the state affects the output of convolving by kernel.
        """
        raise NotImplementedError

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

    def _setup_step(self, **kwargs):
        """Convert a model into a recurrent mode for autoregressive inference."""
        raise NotImplementedError

    def step(self, x, state, **kwargs):
        """Step the model for one timestep with input x and recurrent state."""
        raise NotImplementedError

    def default_state(self, *args, **kwargs):
        """Return a default initial state."""
        raise NotImplementedError

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence, i.e. computes the state after passing chunk through the kernel."""
        raise NotImplementedError

    @property
    def d_state(self):
        """Implement this for interfaces that want to interact with a stateful layer (i.e. SSMs).

        Currently the only codepath that might use this is the StateDecoder, which is not used.
        """
        raise NotImplementedError

    @property
    def state_to_tensor(self):
        """Same as d_state, only needed for niche codepaths involving recurrent state."""
        raise NotImplementedError

class SSMKernel(Kernel):
    """Parent class for different SSM parameterizations.

    This class is abstract and only defines some initializations and flags that are common to all SSM variants.
    It is instantiated by subclasses SSMKernel{Dense,Real,Diag,DPLR}.

    Options:
    d_state (N): State size (dimensionality of parameters A, B, C). Generally shouldn't need to be adjusted and doens't affect speed much for most kernels (e.g. S4, S4D).
    deterministic: Use a deterministic initialization for dt, A, B, C.
        Useful for debugging as well as constructing a simple exponential decay kernel (e.g. used in S4ND image->video inflation).

    dt_min, dt_max: min and max values for the step size dt
    dt_tie: Keep dt tied across the N dimensions of the state. Although this theoretically makes more sense, models such as S5 and Mega have found slightly improvements by setting it to False.
    dt_transform: Transform function for parameterization of dt (default 'softplus', used to be 'exp')

    rank: Rank of low-rank correction for DPLR mode. Needs to be increased for init "legt".
    n_ssm: Number of independent trainable (A, B) SSMs, e.g.
        `n_ssm=1` means all A/B parameters are tied across the H different instantiations of C.
        `n_ssm=None` means all H SSMs are completely independent.
        Generally, changing this option can save parameters but doesn't affect performance or speed much.
        This parameter must divide H.
    init: Options for initialization of (A, B). For DPLR mode, recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin).
    init_args: Extra arguments passed into initialization function (see dplr.py for options).
    """

    def init_dt(self):
        # Generate dt
        if self.deterministic:  # Meant for debugging
            assert self.dt_tie, "Deterministic dt initialization is tied"
            assert self.dt_transform == 'exp', "Deterministic dt transform should be 'exp' for simplicity"
            inv_dt = torch.exp(torch.linspace(math.log(self.dt_min), math.log(self.dt_max), self.H)).unsqueeze(-1) # (H 1)
        else:
            shape = (self.H, 1) if self.dt_tie else (self.H, self.N//2)
            # Initialize log dt
            inv_dt = torch.rand(*shape, dtype=self.dtype) * (
                math.log(self.dt_max) - math.log(self.dt_min)
            ) + math.log(self.dt_min)
            if self.dt_transform != 'exp':
                inv_dt = inv_transform(torch.exp(inv_dt), self.dt_transform)

        return inv_dt

    def init_ssm_real(self):
        """Returns (dense, real) (A, B, C) parameters for init options."""
        # Generate A, B
        A, B = transition(self.init, self.N)
        A = torch.as_tensor(A, dtype=self.dtype)
        B = torch.as_tensor(B, dtype=self.dtype)[:, 0]
        B = repeat(B, 'n -> v n', v=self.n_ssm).clone().contiguous()
        A = repeat(A, 'n m -> v n m', v=self.n_ssm).clone().contiguous()

        # Generate C
        if self.deterministic:
            C = torch.zeros(self.channels, self.H, self.N, dtype=self.dtype)
            C[..., :1] = 1.0
        else:
            C = torch.randn(self.channels, self.H, self.N, dtype=self.dtype)

        return A, B, C

    def init_ssm_dplr(self):
        """Returns DPLR (A, P, B, C) parameters for init options."""
        A, P, B, V = combination(self.init, self.N, self.rank, self.n_ssm, **self.init_args)

        # Broadcast C to have H channels
        if self.deterministic:
            C = torch.zeros(self.channels, self.n_ssm, self.N, dtype=self.cdtype)
            C[:, :, :1] = 1.
            C = contract('hmn, chn -> chm', V.conj().transpose(-1, -2), C) # V^* C
            C = repeat(C, 'c t n -> c (v t) n', v=self.H // C.size(-2)).clone().contiguous()
        else:
            C = torch.randn(self.channels, self.H, self.N//2, dtype=self.cdtype)

        # Broadcast other parameters to have n_ssm copies
        assert self.n_ssm % B.size(-2) == 0 \
                and self.n_ssm % P.size(-2) == 0 \
                and self.n_ssm % A.size(-2) == 0

        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
        A = repeat(A, 't n -> (v t) n', v=self.n_ssm // A.size(-2)).clone().contiguous()

        # Because these complex parameterizations assume conjugate symmetry,
        # halve the value of self.N for convenience
        self.N //= 2

        return A, P, B, C

    def __init__(
        self,
        # General Kernel arguments for parent class
        d_model: int = 0,
        channels: int = 1,
        l_max: Optional[int] = None,
        lr: Union[float, Optional[Mapping]] = None,
        wd: Union[float, Optional[Mapping]] = 0.0,
        verbose: bool = True,
        # SSM arguments
        d_state: int = 64,
        deterministic: bool = False,
        # dt options
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_tie: bool = True,
        dt_transform: str = 'exp',
        # (A, B, C) options
        rank: int = 1,
        n_ssm: Optional[int] = None,
        measure: Optional[str] = None,
        init: Optional[str] = "legs",
        # Extra hyperparameters for initialization
        **init_args,
    ):
        super().__init__(d_model=d_model, channels=channels, l_max=l_max, lr=lr, wd=wd, verbose=verbose)
        self.N = d_state
        self.dtype, self.cdtype = torch.float, torch.cfloat
        self.deterministic = deterministic
        # dt options
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_tie = dt_tie
        self.dt_transform = dt_transform
        # SSM options (A, B, C)
        self.rank = rank
        self.n_ssm = n_ssm if n_ssm is not None else self.H
        if measure is not None:
            log.warning("Warning: 'measure' option changed to 'init' and will be removed in a future version.")
            assert init is None, "'measure' and 'init' cannot both be passed into SSMKernel"
            init, measure = measure, init
        self.init = init
        self.init_args = init_args

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        This is a generic version of this functionality that works for SSMs.
        It is currently used by SSMKernelDense and SSMKernelDPLR.
        This is a suboptimal implementation; it is recommended to use SSMKernelDiag
        if this functionality is desired.

        state: (B, H, N)
        u: (B, H, L)

        Returns: (B, H, N)
        """

        # Construct dA, dB matrices
        dA, dB = self._setup_state() # (H N N) (H N)

        conj = state.size(-1) != dA.size(-1)
        if conj: state = _conj(state)

        v = contract('h n, b h l -> b h n l', dB, u.flip(-1))
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj: next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def _setup_state(self):
        """Register dA and dB to module."""
        raise NotImplementedError

    @property
    def d_state(self):
        """d_state and state_to_tensor are used by specific decoders.

        These were used in earlier versions and should not be needed in general.
        """
        return self.H * self.N

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class SSMKernelDiag(SSMKernel):
    """SSM kernel using diagonal state matrix (S4D model).

    Options:
    disc: ['zoh' | 'bilinear' | 'dss'] Discretization options.
    dt_fast:  (experimental) Parameterize inv_dt under sinh function.
        (Ohno et al. "Fast Saturating Gate for Learning Long Time Scales with RNNs")
    real_transform, imag_transform: ['none' | 'exp' | 'relu' | 'sigmoid' | 'softplus']
        Parameterize the real/imag parts of the diagonal of A under this function.
    bandlimit: Mask high frequencies of the kernel (indices corresponding to
        diagonal elements with large imaginary part). Introduced in S4ND paper.
    backend: ['cuda' | 'keops' | 'naive'] Options for Vandermonde/Cauchy kernel (in order of efficiency).
    is_real : Real-valued SSM; can be interpreted as EMA.
    """

    def __init__(
        self,
        disc: str = 'zoh',  # Change to 'bilinear' to match S4, but should make little difference either way
        dt_fast: bool = False,
        real_transform: str = 'exp',
        imag_transform: str = 'none',
        bandlimit: Optional[float] = None,
        backend: str = 'cuda',
        is_real: bool = False,
        **kwargs,
    ):
        # Special case: for real-valued, d_state semantics change
        if is_real and 'd_state' in kwargs:
            kwargs['d_state'] = kwargs['d_state'] * 2
        super().__init__(**kwargs)
        self.disc = disc
        self.dt_fast = dt_fast
        self.real_transform = real_transform
        self.imag_transform = imag_transform
        self.bandlimit = bandlimit
        self.backend = backend
        self.is_real = is_real

        # Initialize dt, A, B, C
        inv_dt = self.init_dt()
        A, P, B, C = self.init_ssm_dplr()
        # Note that in the Diag case, P will be ignored
        # The DPLR case subclasses this and uses P
        self.register_params(A, B, C, inv_dt, P)

    def register_params(self, A, B, C, inv_dt, P):
        """Process the initialization into form of trainable parameters.

        A: (S, N) diagonal matrix
        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        assert self.backend in ['cuda', 'keops', 'naive']

        if self.dt_fast: inv_dt = torch.asinh(inv_dt)

        # Rank of low-rank correction
        assert self.H == inv_dt.size(0)
        assert self.N == A.size(-1) == B.size(-1) == C.size(-1)
        assert self.n_ssm == A.size(-2) == B.size(-2) # Number of independent SSMs trained
        self.repeat = self.H // A.size(0)

        # Check that diagonal part has negative real and imag part
        # (allow some tolerance for numerical precision on real part
        # since it may be constructed by a diagonalization)
        assert torch.all(A.real < 1e-4) and torch.all(A.imag <= 0.0)

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (C, H, N)  # TODO originally this was only in DPLR, check safe for Diag
        B = B.unsqueeze(0) # (1, H, N)
        assert self.channels == C.shape[0]

        # Register dt
        self.register("inv_dt", inv_dt, self.lr_dict['dt'], self.wd_dict['dt'])
        # Register ABC
        if self.is_real:
            self.register("C", C.real, self.lr_dict['C'], None)
            self.register("B", B.real, self.lr_dict['B'], self.wd_dict['B'])
            self.register("A_real", inv_transform(-A.real, self.real_transform), self.lr_dict['A'], self.wd_dict['A'])
        else:
            self.register("C", _c2r(_resolve_conj(C)), self.lr_dict['C'], None)
            self.register("B", _c2r(B), self.lr_dict['B'], self.wd_dict['B'])
            self.register("A_real", inv_transform(-A.real, self.real_transform), self.lr_dict['A'], self.wd_dict['A'])
            self.register("A_imag", inv_transform(-A.imag, self.imag_transform), self.lr_dict['A'], self.wd_dict['A'])

    def _get_params(self, rate=1.0):
        """Process the internal parameters."""

        # (S N) where S=n_ssm
        if self.is_real:
            A = -param_transform(self.A_real, self.real_transform)
            B = self.B # (1 S N)
            C = self.C # (C H N)
        else:
            A = -param_transform(self.A_real, self.real_transform) - 1j * param_transform(self.A_imag, self.imag_transform)
            B = _r2c(self.B) # (1 S N)
            C = _r2c(self.C) # (C H N)

        if self.dt_fast: inv_dt = torch.sinh(self.inv_dt)
        else: inv_dt = self.inv_dt
        dt = param_transform(inv_dt, self.dt_transform) * rate # (H N)

        if self.bandlimit is not None:
            freqs = dt / rate * A.imag.abs() / (2*math.pi) # (H N)
            mask = torch.where(freqs < self.bandlimit * .5, 1, 0)
            C = C * mask

        # Incorporate dt into A and B
        A = repeat(A, 't n -> (v t) n', v=self.repeat)  # (H N)
        B = repeat(B, 'b t n -> b (v t) n', v=self.repeat)  # (1 H N)

        # TODO: The downstream algorithm should only need to access dt*A
        # However the current DPLR kernel still uses dt and A separately
        # Once that is fixed, this should return dtA instead of dt and A
        dtA = dt * A  # (H N)

        return dt, A, B, C

    def forward(self, L, state=None, rate=1.0):
        """See Kernel.forward() for argument documentation."""

        dt, A, B, C = self._get_params(rate)
        dtA = dt * A

        # Augment B with state
        if state is not None:
            s = state / dt
            if self.disc == 'bilinear':
                s = s * (1. + dtA/2)
            elif self.disc == 'zoh':
                s = s * dtA * dtA.exp() / (dtA.exp() - 1.)
            B = torch.cat([s, B], dim=-3) # (1+B H N)


        # Combine B and C
        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)

        # Dispatch which Vandermonde kernel to use
        if has_cuda_extension and C.dtype == torch.cfloat and C.device.type == 'cuda' and self.backend == 'cuda':
            log_vandermonde = log_vandermonde_cuda
        elif has_pykeops and self.backend in ['cuda', 'keops']:
            log_vandermonde = log_vandermonde_keops
        else:
            log_vandermonde = log_vandermonde_naive

        # Main kernel
        if self.disc == 'zoh':
            # Power up
            C = C * (torch.exp(dtA)-1.) / A
            K = log_vandermonde(C, dtA, L) # (H L)
        elif self.disc == 'bilinear':
            C = C * (1. - dtA/2).reciprocal() * dt # or * dtA / A
            dA = (1. + dtA/2) / (1. - dtA/2)
            K = log_vandermonde(C, dA.log(), L)
        elif self.disc == 'dss':
            # Implementation from DSS meant for case when real eigenvalues can be positive
            P = dtA.unsqueeze(-1) * torch.arange(L, device=C.device) # [H N L]
            A_gt_0 = A.real > 0                                      # [N]
            if A_gt_0.any():
                with torch.no_grad():
                    P_max = dtA * (A_gt_0 * (L-1))                   # [H N]
                P = P - P_max.unsqueeze(-1)                          # [H N L]
            S = P.exp()                                              # [H N L]

            dtA_neg = dtA * (1 - 2*A_gt_0)                           # [H N]
            num = dtA_neg.exp() - 1                                  # [H N]
            den = (dtA_neg * L).exp() - 1                            # [H N]

            # Inline reciprocal function for DSS logic
            x = den * A
            x_conj = _resolve_conj(x)
            r = x_conj / (x*x_conj + 1e-7)

            C = C * num * r             # [C H N]
            K = contract('chn,hnl->chl', C, S).float()
        else: raise ValueError(f"Discretization {self.disc} not supported")

        K = K.view(-1, self.channels, self.H, L) # (1+B C H L)

        if state is not None:
            K_state = K[:-1, :, :, :] # (B C H L)
        else:
            K_state = None
        K = K[-1, :, :, :] # (C H L)

        return K, K_state

    def _setup_step(self):
        """Set up dA, dB, dC discretized parameters for stepping."""

        dt, A, B, C, = self._get_params()
        # Incorporate dt into A
        dtA = dt * A  # (H N)

        if self.disc == 'zoh':
            self.dA = torch.exp(dtA) # (H N)
            self.dB = B * (torch.exp(dtA)-1.) / A # (C H N)
        elif self.disc == 'bilinear':
            self.dA = (1. + dtA/2) / (1. - dtA/2)
            self.dB = B * (1. - dtA/2).reciprocal() * dt # or * dtA / A
        self.dB = rearrange(self.dB, '1 h n -> h n')
        self.dC = C

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2*y.real, next_state

    def forward_state(self, u, state):
        """Pass the state forward through an entire sequence."""
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).to(self.dA).contiguous() # (B H L)
        # Dispatch which Vandermonde kernel to use
        if has_pykeops and self.backend in ['cuda', 'keops']:
            log_vandermonde_transpose = log_vandermonde_transpose_keops
        else:
            log_vandermonde_transpose = log_vandermonde_transpose_naive
        v = log_vandermonde_transpose(u, self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state


class SSMKernelDPLR(SSMKernelDiag):
    """SSM kernel for diagonal + low rank (DPLR) state matrices, corresponding to the original S4 model."""

    @torch.no_grad()
    def _setup_C(self, L):
        """Construct C~ from C.

        Two modes are supported: go directly to length L if self.l_kernel is 1, or length is doubled
        """

        if self.l_kernel.item() == 0:
            if self.verbose: log.info(f"S4: Initializing kernel to length {L}")
            double_length = False
        elif L > self.l_kernel.item(): # 2*int(self.l_kernel) == L:
            if self.verbose: log.info(f"S4: Doubling length from L = {self.l_kernel.item()} to {2*self.l_kernel.item()}")
            double_length = True
            L = self.l_kernel.item() # Convenience for the math below
        else: return

        C = _r2c(self.C)
        dA, _ = self._setup_state()
        dA_L = power(L, dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
        if double_length: prod = -prod # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again
        self.C.copy_(_c2r(C_))

        self.l_kernel = 2*self.l_kernel if double_length else self.l_kernel+L # Preserve type/device

    def _omega(self, L, dtype, device, cache=True):
        """Calculate (and cache) FFT nodes.

        This also caches a version of the nodes "unprocessed" with the bilinear transform.
        This method should be called everytime the internal length self.l_kernel changes.
        """

        # Use cached if available
        if cache and hasattr(self, 'omega') and self.omega.size(-1) == L//2+1:
            return self.omega, self.z

        omega = torch.tensor(
            np.exp(-2j * np.pi / (L)), dtype=dtype, device=device
        )  # \omega_{2L}
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)

        # Cache if necessary
        if cache:
            self.omega = omega
            self.z = z
        return omega, z


    def register_params(self, A, B, C, inv_dt, P):
        """Process the initialization into form of trainable parameters.

        The SSM state matrix is represented by diag_embed(A) - PP^*
        Note that the A notation here is slightly overloaded:
        normally A refers to the full SSM state matrix (DPLR in this case)
        but here we're using it to refer to the diagonal part of the matrix.
        This is to make variable names compatible with the SSMKernelDiag class (DSS/S4D)
        and is a much simpler variable name (e.g. as opposed to Lambda).

        A: (S, N) diagonal part
        P: (R, S, N) low-rank part
        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        R (or rank): rank of low-rank part
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        # Print out kernel lengths; it can be tricky to make sure the length logic is correct
        if self.verbose:
            log.info(f"Constructing S4 (H, N, L) = ({self.H}, {self.N}, {self.l_max})")

        # Register the basic params for diagonal SSM (A, B, C, dt)
        super().register_params(A, B, C, inv_dt, P)

        # Check shapes
        assert self.rank == P.shape[-3]
        assert self.N == P.size(-1)
        assert self.n_ssm == P.size(-2)

        self.register('P', _c2r(P), self.lr_dict['A'], self.wd_dict['A'])

        # Track the current kernel length this is "attuned" to
        self.register_buffer('l_kernel', torch.tensor(0))

    def _get_params(self, rate=1.0):
        dt, A, B, C = super()._get_params(rate=rate)
        P = _r2c(self.P)  # (R S N)
        P = repeat(P, 'r t n -> r (v t) n', v=self.repeat)  # (R H N)
        Q = P.conj()

        return dt, A, B, C, P, Q

    def forward(self, state=None, rate=1.0, L=None):
        """See Kernel.forward() for argument documentation."""

        # Initialize C~ if necessary (done in forward pass so it's on the correct device)
        if self.l_kernel.item() == 0 and self.l_max is not None and self.l_max > 0:
            self._setup_C(self.l_max)

        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.l_kernel, while we are asked to provide a kernel of length L at (relative) frequency rate
        if L is None:
            L = round(self.l_kernel.item() / rate)

        # Increase the internal length if needed
        continuous_L = round(rate*L)
        while continuous_L > self.l_kernel.item():
            self._setup_C(continuous_L)
        discrete_L = round(self.l_kernel.item()/rate)

        dt, A, B, C, P, Q = self._get_params(rate)

        # Get FFT nodes of right length
        omega, z = self._omega(discrete_L, dtype=A.dtype, device=A.device, cache=(rate==1.0))

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.size(-1) == self.N else state # (B H N)
            sA = (
                s * _conj(A) # (B H N)
                - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
            s = s / dt + sA / 2
            s = s[..., :self.N]

            B = torch.cat([s, B], dim=-3)  # (B+1, H, N)

        # Incorporate dt into A
        A = A * dt  # (H N)

        # Stack B and p, C and q for convenient batching
        B = torch.cat([B, P], dim=-3) # (B+1+R, H, N)
        C = torch.cat([C, Q], dim=-3) # (C+R, H, N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)  # (B+1+R, C+R, H, N)
        v = v * dt  # Incorporate dt into B

        # Dispatch which Cauchy kernel to use
        if has_cuda_extension and z.dtype == torch.cfloat and z.device.type == 'cuda' and self.backend == 'cuda':
            cauchy_mult = cauchy_cuda
        elif has_pykeops and self.backend in ['cuda', 'keops']:
            cauchy_mult = cauchy_keops
        else:
            cauchy_mult = cauchy_naive
        # Calculate resolvent at omega
        r = cauchy_mult(v, z, A)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - torch.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f, n=discrete_L)  # (B+1, C, H, L)

        # # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (B, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :] # (C H L)

        return k_B, k_state

    @torch.no_grad()
    def double_length(self):
        self._setup_C(2*self.l_kernel)

    @torch.no_grad()
    def _check(self):
        """Check if A, B, C parameters and vanilla SSMKernel construction can be recovered"""

        # assert self.l_kernel > 0, "Set up module first"

        K = self.forward(L=self.l_max)[0]

        self._setup_step()
        K_ = krylov(self.l_max, self.dA, self.dB, self.dC)

        diff = K - K_
        print("checking DPLR Kernel construction", torch.sum(diff ** 2))

    @torch.no_grad()
    def _setup_linear(self):
        """Preprocessing that allows fast linear-time (in state dimension) stepping."""
        dt, A, B, C, P, Q = self._get_params()

        # Prepare Linear stepping
        D = (2.0 / dt - A).reciprocal()  # (H, N)
        R = (torch.eye(self.rank, dtype=A.dtype, device=A.device) + 2*contract('r h n, h n, s h n -> h r s', Q, D, P).real) # (H R R)
        Q_D = rearrange(Q*D, 'r h n -> h r n')
        try:
            R = torch.linalg.solve(R, Q_D) # (H R N)
        except:
            R = torch.tensor(np.linalg.solve(R.to(Q_D).contiguous().detach().cpu(), Q_D.contiguous().detach().cpu())).to(Q_D)
        R = rearrange(R, 'h r n -> r h n')

        self.step_params = {
            "D": D, # (H N)
            "R": R, # (R H N)
            "P": P, # (R H N)
            "Q": Q, # (R H N)
            "B": B, # (1 H N)
            "E": 2.0 / dt + A, # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations.
        Perhaps a fused CUDA kernel implementation would be much faster.

        u: (H) Input
        state: (H, N/2) State with conjugate pairs. Optionally, the state can have last dimension N.

        Returns: same shape as state
        """
        C = _r2c(self.C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None: # Special case used to find dB
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

        step_params = self.step_params.copy()
        if state.size(-1) == self.N: # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
        else:
            assert state.size(-1) == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y) # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (R H N)
        P = step_params["P"]  # (R H N)
        Q = step_params["Q"]  # (R H N)
        B = step_params["B"]  # (1 H N)

        new_state = E * state - contract_fn(P, Q, state) # (B H N)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _setup_state(self):
        """Construct dA and dB for discretized state equation."""

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C) # Just returns a view that we use for finding dtype/device

        state = torch.eye(2*self.N, dtype=C.dtype, device=C.device).unsqueeze(-2) # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")

        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        dB = rearrange(dB, '1 h n -> h n') # (H N)
        return dA, dB

    def _step_state(self, u, state):
        """Must be called after self.default_state() is used to construct an initial state!"""
        next_state = (torch.einsum(self.state_contraction, self.dA, state)
                     + torch.einsum(self.input_contraction, self.dB, u))
        return next_state

    def _setup_step(self, mode='dense'):
        """Set up dA, dB, dC discretized parameters for stepping."""
        self.dA, self.dB = self._setup_state()

        # Calculate original C
        C = _conj(_r2c(self.C)) # (H C N)
        if self.l_kernel.item() == 0:
            dC = C
        else:
            # self.C represents C_tilde
            dA_L = power(self.l_kernel.item(), self.dA)
            I = torch.eye(self.dA.size(-1)).to(dA_L)

            dC = torch.linalg.solve(
                I - dA_L.transpose(-1, -2),
                C.unsqueeze(-1),
            ).squeeze(-1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode
        if mode == 'linear':
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2*self.dC[:, :, :self.N]
        elif mode == 'diagonal':
            # Eigendecomposition of the A matrix
            L, V = torch.linalg.eig(self.dA)
            V_inv = torch.linalg.inv(V)
            # Check that the eigendedecomposition is correct
            if self.verbose:
                print("Diagonalization error:", torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA))

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
            self.dC = contract('h n m, c h n -> c h m', V, self.dC)

        elif mode == 'dense':
            pass
        else: raise NotImplementedError("DPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.size(-1)
        H = C.size(-2)

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        step_mode = getattr(self, "_step_mode", "dense")  # Used in default_state, which is called without _setup_step() in forward_state()
        if step_mode != 'linear':
            N *= 2

            if step_mode == 'diagonal':
                self.state_contraction = "h n, ... h n -> ... h n"
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = "h m n, ... h n -> ... h m"

            self.input_contraction = "h n, ... h -> ... h n"

        self.output_contraction = "c h n, ... h n -> ... c h"

        state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        """Must have called self._setup_step() and created state with self.default_state() before calling this."""

        if self._step_mode == 'linear':
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = torch.einsum(self.output_contraction, self.dC, new_state)
        return y.real, new_state

    def forward_state(self, *args, **kwargs):
        # Dispatch directly to generic state forwarding
        # instead of using the Diag version

        # TODO design pattern is ugly. Can be fixed with an intermediate
        # subclass above Diag/DPLR that has the shared logic (parameter construction)
        # but not the state/step logic.
        # Fine to keep like this for now since we want Diag to be the standard
        # instead of having too many layers of subclassing.

        return SSMKernel.forward_state(self, *args, **kwargs)

kernel_registry = {
    's4d': SSMKernelDiag,
    'diag': SSMKernelDiag,
    's4': SSMKernelDPLR,
    'nplr': SSMKernelDPLR,
    'dplr': SSMKernelDPLR,
}

class FFTConv(nn.Module):
    """Implements an FFT Convolution around a convolution kernel.

    d_model (H): Model dimension (in CNN terminology, this would be "channels").
    l_max (L): The maximum kernel length. Set l_max=None to always use a global kernel.
    channels: Can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this; instead, increase d_model for larger models.
    bidirectional: If True, convolution kernel will be two-sided.
    activation: Activation after the full convolution.
    transposed, dropout, tie_dropout: More general model options, see SequenceModule.
    mode: Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D. Other options can be found in the kernel registry.

    kernel_args: See the class .kernel.SSMKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode", "init", "dt_min", "dt_max", "lr"
    """

    def __init__(
        self,
        d_model,
        l_max=None,
        channels=1,
        swap_channels=False,
        bidirectional=False,
        activation='gelu', # Activation after layer
        transposed=True,
        dropout=0.0,
        tie_dropout=False,
        drop_kernel=0.0,
        mode='dplr',
        kernel=None,
        **kernel_args,  # Arguments passed into inner convolution kernel
    ):
        super().__init__()
        self.d_model = d_model
        self.L = self.l_max = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.swap_channels = swap_channels


        if activation is not None and activation.startswith('glu'):
            channels *= 2
        self.activation = Activation(activation, dim=1 if self.transposed else -1)

        self.D = nn.Parameter(torch.randn(channels, self.d_model))

        if self.bidirectional:
            channels *= 2

        # Inner convolution kernel
        if mode is not None:
            assert kernel is None, "Pass either mode or kernel but not both"
            # log.info(
            #     "Argument 'mode' is deprecated and renamed to 'kernel',"
            #     "and will be removed in a future version."
            # )
            kernel, mode = mode, kernel
        kernel_cls = kernel_registry[kernel]
        self.kernel = kernel_cls(
            d_model=self.d_model,
            l_max=self.l_max,
            channels=channels,
            **kernel_args,
        )

        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_kernel = nn.Dropout(drop_kernel) if drop_kernel > 0.0 else nn.Identity()

    def forward(self, x, state=None, rate=1.0, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B D L) if self.transposed else (B L D)
        """

        # Always work with (B D L) dimension in this module
        if not self.transposed: x = x.transpose(-1, -2)
        L = x.size(-1)

        # Compute SS Kernel
        l_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state =  self.kernel(L=l_kernel, rate=rate, state=state) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0))
            # The above has an off-by-one in the reverse direction
            # This is a deliberate choice since the off-by-one should not affect any applications
            # This can be amended which may be very slightly slower
            # k = F.pad(k0, (0, L)) \
            #         + F.pad(k1[..., 1:].flip(-1), (L+1, 0)) \
            #         + F.pad(k1[..., :1], (0, l_kernel+L-1))

        # Kernel dropout
        k = self.drop_kernel(k)

        # In principle, we could pad to l_kernel+L-1 instead of l_kernel+L, but we choose the latter for
        # equational simplicity. Additionally, we have not experimented to compare the efficiency of the two.
        k_f = torch.fft.rfft(k, n=l_kernel+L) # (C H L)
        x_f = torch.fft.rfft(x, n=l_kernel+L) # (B H L)
        y_f = contract('bhl,chl->bchl', x_f, k_f)
        y = torch.fft.irfft(y_f, n=l_kernel+L)[..., :L] # (B C H L)


        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', x, self.D)

        # Compute state update
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state #
            next_state = self.kernel.forward_state(x, state)
        else:
            next_state = None


        # Reshape to flatten channels
        if self.swap_channels:
            y = rearrange(y, 'b c h l -> b (h c) l')
        else:
            y = rearrange(y, 'b c h l -> b (c h) l')

        y = self.drop(y)  # DropoutNd better with transposed=True

        if not self.transposed: y = y.transpose(-1, -2)
        y = self.activation(y)

        return y, next_state


    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, x, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        y, next_state = self.kernel.step(x, state) # (B C H)
        y = y + x.unsqueeze(-2) * self.D
        y = rearrange(y, 'b c h -> b (c h)')
        y = self.activation(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.kernel.default_state(*batch_shape)

    @property
    def d_output(self):
        return self.d_model * self.channels


class S4Block(nn.Module):
    """General block design wrapping an inner layer. Currently only layer=FFTConv is supported, but easy to incorporate others.

    Arguments:
    - bottleneck: Reduce dimension of inner layer (e.g. used in GSS).
    - gate: Add multiplicative gating (e.g. used in GSS), which is essentially a multiplicative instead of additive residual branch.
    - gate_act: Activation function to apply on the gate residual branch.
    - mult_act: Activation function to apply after gate multiplication (e.g. GELU in GSS).
    - final_act: Activation function to apply after final linear layer. 'id' for no activation, None for no linear layer at all.

    - initializer: Initializer on final linear layer.
    - weight_norm: Weight normalization on final linear layer.
    - dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

    - transposed: Choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=model dimension]

    Other options are all experimental and should not need to be configured.
    """

    def __init__(
        self,
        d_model,
        bottleneck=None,
        gate=None,
        gate_act=None,
        mult_act=None,
        final_act='glu',
        postact=None,
        initializer=None,
        weight_norm=False,
        dropout=0.0,
        tie_dropout=False,
        transposed=True,
        **layer_args,  # Arguments into inner layer (e.g. FFTConv)
    ):
        super().__init__()

        self.d_model = d_model
        self.transposed = transposed

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.d_model = self.d_model // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.d_model,
                transposed=False,
                activation=None,
                activate=False,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=False,
                activation=gate_act,
                activate=True,
            )
            if self.layer.d_output != self.d_model * gate:
                self.output_gate = LinearActivation(
                    self.d_model*self.channels,
                    self.d_model * gate,
                    transposed=False,
                    activation=None,
                    activate=False,
                )

        # Currently this module only uses FFTConv for its inner module
        # But the options here are all agnostic to the inner block
        # If other types of inner layers are desired, it is easy
        # to add an option to swap a different module in
        self.layer = FFTConv(d_model, transposed=False, dropout=dropout, tie_dropout=tie_dropout, **layer_args)

        # Pointwise operations

        # Activation after (optional) multiplication by gate branch
        self.mult_activation = Activation(mult_act)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is not None:
            assert final_act is None
            log.warning("Warning: 'postact' option changed to 'final_act' and will be removed in a future version.")
            final_act, postact = postact, final_act
        if final_act is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_model*gate if gate is not None else self.layer.d_output,
                self.d_model,
                transposed=False,
                activation=final_act,
                activate=True,
            )



    def forward(self, x, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as x
        """
        if self.transposed: x = rearrange(x, 'b d ... -> b ... d')
        L = x.size(1)

        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=x.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, x.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device)[:, None] < lengths[:, None, None], 1., 0.)
            x = x * mask

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)

        y, state = self.layer(x, **kwargs)


        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)

        if self.transposed: y = rearrange(y, 'b d ... -> b ... d')

        return y, state

    def setup_step(self, **kwargs):
        self.layer.setup_step(**kwargs)

    def step(self, x, state):
        """Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)
        y, next_state = self.layer.step(x, state) # (B C H)
        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.layer.default_state(*batch_shape)

    @property
    def d_output(self):
        return self.d_model
