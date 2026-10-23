#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selective_scan（fwd + bwd）NPU 算子最小示例。

流程：挂 NPU shim → selective_scan_fn 前向（shim 直映射 torch.ops.cann_bench.selective_scan_fwd）
→ autograd 反向（触发 torch.ops.cann_bench.selective_scan_bwd）→ 与 CPU 参考逐项对比
max_err → 打印 PASS/FAIL（退出码 0/1）。

用法：
    cd <mamba 仓根目录>            # 保证 import mamba_ssm 可用
    python npu/examples/quickstart_npu.py [--device 0]
"""
import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int,
                        default=int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")))
    args = parser.parse_args()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)

    import torch
    import torch_npu  # noqa: F401  显式引入 torch_npu（NPU 后端注册）

    torch.manual_seed(0)
    torch.npu.manual_seed(0)

    # shim 须先于 mamba_ssm import（模块导入期执行 triton 兼容 polyfill）。
    # 追加（而非插入头部）到 sys.path：避免源码树 npu/cann_bench/ 遮蔽已安装的
    # cann_bench wheel（算子 .so 只存在于安装后的包内）。
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    import npu_shim_selective_scan
    import mamba_ssm.ops.selective_scan_interface as ssi
    npu_shim_selective_scan.install_npu_shim()  # selective_scan_cuda → NPU 算子
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

    assert isinstance(ssi.selective_scan_cuda, npu_shim_selective_scan._SelectiveScanCudaNpuShim), \
        "NPU shim 未生效"

    dev = "npu"
    b, d, l, dstate = 2, 16, 64, 16
    u = torch.randn(b, d, l, device=dev, requires_grad=True)
    delta = torch.randn(b, d, l, device=dev, requires_grad=True)
    A = (-torch.exp(torch.rand(d, dstate, device=dev) * 15 + 1)).requires_grad_(True)  # A < 0（先取负再挂梯度，保证 A 为叶节点）
    B = torch.randn(b, dstate, l, device=dev, requires_grad=True)   # 静态 B（3 维）
    C = torch.randn(b, dstate, l, device=dev, requires_grad=True)   # 静态 C（3 维）
    D = torch.randn(d, device=dev, requires_grad=True)
    z = torch.randn(b, d, l, device=dev, requires_grad=True)
    delta_bias = torch.randn(d, device=dev, requires_grad=True)

    # 前向 + 反向（autograd 自动触发 NPU bwd 算子）
    out = selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus=True)
    out.sum().backward()
    torch.npu.synchronize()

    # CPU 参考复算（fp64 升维，参考数值误差远小于 fp32 判定门限）
    def to_cpu(t):
        return t.detach().double().cpu()

    cpu_leaves = [to_cpu(t).clone().requires_grad_(True)
                  for t in (u, delta, A, B, C, D, z, delta_bias)]
    out_ref = selective_scan_ref(*cpu_leaves[:7], delta_bias=cpu_leaves[7],
                                 delta_softplus=True)
    out_ref.sum().backward()

    def max_err(a, g):
        return (a.double().cpu() - g).abs().max().item()

    names = ["out", "du", "ddelta", "dA", "dB", "dC", "dD", "dz", "ddelta_bias"]
    npu_vals = [out, u.grad, delta.grad, A.grad, B.grad, C.grad, D.grad, z.grad, delta_bias.grad]
    ref_vals = [out_ref] + [t.grad for t in cpu_leaves]

    # 判定门限：|a-g|max <= atol + rtol*|g|max（fp32 算子 vs fp64 参考，小 shape 实测余量充足）
    atol, rtol = 1e-3, 1e-2
    ok = True
    print(f"shape(b,d,l;ds)=({b},{d},{l};{dstate})  dtype=fp32  device=npu:{args.device}")
    for name, a, g in zip(names, npu_vals, ref_vals):
        err = max_err(a, g)
        ref = g.abs().max().item()
        bound = atol + rtol * ref
        status = "PASS" if err <= bound else "FAIL"
        ok &= (err <= bound)
        print(f"  {name:<12s} max_err={err:.3e}  (bound={bound:.3e})  {status}")
    n_fwd = npu_shim_selective_scan._FWD_CALL["n"]
    n_bwd = npu_shim_selective_scan._BWD_CALL["n"]
    print(f"shim 调用计数: fwd={n_fwd}, bwd={n_bwd}")
    print(f"RESULT: {'PASS' if ok and n_fwd >= 1 and n_bwd >= 1 else 'FAIL'}")
    sys.exit(0 if ok and n_fwd >= 1 and n_bwd >= 1 else 1)
