import torch

from mamba_ssm.ops.triton.layer_norm import rms_norm_fn

from overflow_test_utils import assert_isfinite, bwd_row_start_max, gpu_memory_skipif


@gpu_memory_skipif(21)
def test_rms_norm_large_row_count_no_overflow() -> None:
    # int64 to avoid int32 overflow, see https://github.com/state-spaces/mamba/issues/1015.
    device = 'cuda'
    torch.manual_seed(0)
    N = 32_768  # hard cap: layer_norm.py raises if group_size/N exceeds 64KB / dtype_size
    M = 80_000  # (sm_count - 1) * ceil(M / sm_count) * N still clears 2**31 - 1 with margin (asserted below)

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    assert bwd_row_start_max(M, sm_count) * N > 2**31 - 1, (
        f"test parameters too small to overflow the backward kernel on this GPU "
        f"(sm_count={sm_count}): increase N"
    )

    x = torch.randn(M, N, dtype=torch.bfloat16, device=device, requires_grad=True)
    weight = torch.randn(N, dtype=torch.float32, device=device, requires_grad=True)

    out = rms_norm_fn(x, weight, bias=None)
    torch.cuda.synchronize(device)
    assert out.shape == (M, N)
    assert_isfinite(out)

    out.sum().backward()
    torch.cuda.synchronize(device)
    assert x.grad is not None
    assert_isfinite(x.grad)
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
