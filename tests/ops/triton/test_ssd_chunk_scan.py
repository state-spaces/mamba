import torch

from mamba_ssm.ops.triton.ssd_chunk_scan import (
    _chunk_scan_fwd,
    _chunk_scan_fwd_kernel,
)


def test_chunk_scan_fwd_every_config_matches(monkeypatch):
    """The autotuner picks a config by speed only, so every config in the list
    must give the same finite output. On sm90, configs that pipeline a wgmma
    with `cb` in registers wrote NaN into whole rows."""
    from triton.runtime.errors import OutOfResources

    torch.manual_seed(0)
    batch, seqlen = 1, 181
    chunk_size, nchunks = 128, 2
    nheads, headdim, ngroups, dstate = 64, 64, 8, 128
    dev = "cuda"

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.bfloat16, device=dev) * 0.5
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.bfloat16, device=dev) * 0.5
    cb = torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size, device=dev)
    dt = torch.rand(batch, nheads, nchunks, chunk_size, device=dev) * 0.1
    dA_cumsum = -torch.rand(batch, nheads, nchunks, chunk_size, device=dev).cumsum(-1) * 0.05
    states = torch.randn(batch, nchunks, nheads, headdim, dstate, device=dev) * 0.1
    D = torch.randn(nheads, headdim, device=dev) * 0.1

    ref = None
    for config in list(_chunk_scan_fwd_kernel.configs):
        monkeypatch.setattr(_chunk_scan_fwd_kernel, "configs", [config])
        for _ in range(10):
            try:
                out, _ = _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D=D)
            except OutOfResources:
                # The autotuner skips configs that do not fit this device.
                break
            assert torch.isfinite(out).all(), f"non-finite output with {config}"
            if ref is None:
                ref = out
            torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
