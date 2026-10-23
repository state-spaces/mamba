import pytest
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def fn_broadcast_1d(output_ptr, x_ptr, XS: tl.constexpr, YS: tl.constexpr):
    xidx = tl.arange(0, XS)[None, :]
    base = tl.load(x_ptr + xidx)
    out = base.broadcast_to((YS, XS))
    oidx = tl.arange(0, YS)[:, None] * XS + tl.arange(0, XS)[None, :]
    tl.store(output_ptr + oidx, out)

@pytest.mark.parametrize("shape", [(1,), (2,), (4,)])
@pytest.mark.parametrize("dtype", [torch.int32])
def test_npu_1d(shape, dtype):
    XS = shape[0]
    YS = 4
    x = torch.randint(-1000, 1000, (XS,), dtype=dtype, device="npu")
    std = torch.broadcast_to(x, (YS, XS))
    output = torch.randint(-1000, 1000, (YS, XS), dtype=dtype, device="npu")
    fn_broadcast_1d[(1,)](output, x, XS, YS)
    assert torch.allclose(std, output)
