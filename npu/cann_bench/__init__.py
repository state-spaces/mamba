import torch

try:
    from . import _C  # noqa: F401  (triggers TORCH_LIBRARY static init)
except ImportError as e:
    raise ImportError("Cannot import _C. Please install the cann_bench package.") from e


def selective_scan_fwd(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    return torch.ops.cann_bench.selective_scan_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)


def selective_scan_bwd(u, delta, A, B, C, dout, D=None, z=None, delta_bias=None, x=None,
                       out=None, dz=None, delta_softplus=False, recompute_out_z=False):
    return torch.ops.cann_bench.selective_scan_bwd(
        u, delta, A, B, C, dout, D, z, delta_bias, x, out, dz, delta_softplus, recompute_out_z)
