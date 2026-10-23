#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selective_scan NPU shim 模块（自 npu_run_selective_scan.py [1] 节抽出复用，逻辑未改动）。

供 benchmarks/benchmark_generation_mamba_simple.py 等 NPU 驱动脚本复用：
    import npu_shim_selective_scan  # import 时完成 triton 兼容 polyfill（须先于 import mamba_ssm）
    npu_shim_selective_scan.install_npu_shim()  # 挂载 selective_scan_cuda NPU shim（prefill 走 NPU 算子）

组成
----
[1] triton 兼容 polyfill：模块 import 时执行。triton 3.2.0 无 set_allocator
    （triton ≥3.3 API）；mamba3 的 triton kernel 模块在 import 期顶层无条件
    调用之（mamba3_siso_fwd/step/bwd），导致 import mamba_ssm 失败。Mamba-1
    不经过 mamba3 kernel，polyfill 为 no-op 无副作用（详见问题记录文档）。
[2] NPU shim：运行时替换 ``mamba_ssm.ops.selective_scan_interface.selective_scan_cuda``
    模块属性（forward/backward 内按全局名解析，patch 生效）。``.fwd`` 参数直映射
    ``torch.ops.cann_bench.selective_scan_fwd``；返回值元数与 CUDA pybind 严格一致：
    z=None → (out, x)，z 非 None → (out, x, out_z)。``.bwd`` 参数序与 CUDA pybind
    ``bwd(u, delta, A, B, C, D, z, delta_bias, dout, x, out, dz, delta_softplus,
    recompute_out_z)`` 逐位对齐（内部重排为 NPU 算子的 dout 必选前置序），返回值
    元数同样严格对齐 CUDA 动态行为：7 元基础梯度，has_z 追加 dz，recompute_out_z
    追加 out_z。其它属性访问报清晰错误。
[3] decode 步进 triton kernel 兼容开关：``disable_triton_selective_state_update()``
    将 mamba_simple 内 import 期绑定的 selective_state_update 置 None，
    令 Mamba.step 自动回退纯 torch ref 步进（仅当 triton_ascend 编译不可用时使用）。
"""

import os

import torch

# torch.ops.cann_bench.selective_scan_fwd/bwd 句柄缓存（首次调用时 import cann_bench 后解析）
_NPU_OP = None
_NPU_OP_BWD = None
# 诊断开关初始化（模块 import 时读取环境变量）：
#   NPU_SHIM_TRACE=1       打印每次 fwd/bwd 的调用序号与 shape（默认关）
#   NPU_SHIM_SYNC=1        每次 fwd/bwd 后强制 device 同步（异步 aivec 异常即时定位，默认关）
#   NPU_SHIM_DUMP=<dir>    每次 fwd/bwd 输入 CPU 副本落盘 <dir>/{fwd,bwd}{n}.pt（默认关）
_DIAG = {"trace": os.environ.get("NPU_SHIM_TRACE") == "1",
         "sync": os.environ.get("NPU_SHIM_SYNC") == "1",
         "dump_dir": os.environ.get("NPU_SHIM_DUMP") or None}
# 调用计数（问题定位辅助）
_FWD_CALL = {"n": 0}
_BWD_CALL = {"n": 0}


def _trace_op(tag, counter, **tensors):
    """fwd/bwd 通用诊断：调用计数 + 可选 dump/trace（开关见 _DIAG 注释）。"""
    counter["n"] += 1
    if not _DIAG["trace"] and _DIAG["dump_dir"] is None:
        return
    n = counter["n"]
    parts = " ".join(
        f"{k}={'None' if v is None else '(' + ', '.join(map(str, v.shape)) + ')' + ':' + str(v.dtype).split('.')[-1]}"
        for k, v in tensors.items())
    if _DIAG["dump_dir"] is not None:
        os.makedirs(_DIAG["dump_dir"], exist_ok=True)
        torch.save({k: None if v is None else v.detach().cpu()
                    for k, v in tensors.items()},
                   os.path.join(_DIAG["dump_dir"], f"{tag}{n}.pt"))
    if _DIAG["trace"]:
        print(f"[shim] {tag} #{n} {parts}", flush=True)


def ensure_triton_compat():
    """[1] triton 兼容 polyfill（须在 import mamba_ssm 之前调用）。"""
    import triton
    if not hasattr(triton, "set_allocator"):
        triton.set_allocator = lambda *args, **kwargs: None


class _SelectiveScanCudaNpuShim:
    """selective_scan_cuda 的 NPU 适配 shim（real 变体前向 + 反向，同 wheel 配套）。

    - fwd 参数与 CUDA pybind ``fwd(u, delta, A, B, C, D, z, delta_bias,
      delta_softplus)`` 逐位对应，直映射 torch.ops.cann_bench.selective_scan_fwd；
    - bwd 参数与 CUDA pybind ``bwd(u, delta, A, B, C, D, z, delta_bias, dout, x,
      out, dz, delta_softplus, recompute_out_z)`` 逐位对应（调用点见
      selective_scan_interface.py 的 SelectiveScanFn.backward / MambaInnerFn.backward），
      内部重排为 NPU 算子的 ``bwd(u, delta, A, B, C, dout, D, z, delta_bias, x,
      out, dz, delta_softplus, recompute_out_z)``（dout 必选前置）；
    - NPU 算子要求全连续输入，而上游仅保证末维 stride==1（如 z 为 xz 切片时
      非全连续），故 shim 内统一 .contiguous() 兜底（数值无影响）；
    - fwd 返回元数严格对齐 CUDA：z=None → (out, x)；z 非 None → (out, x, out_z)。
      bwd 返回元数严格对齐 CUDA 动态行为：恒有 7 元基础梯度
      (du, ddelta, dA, dB, dC, dD, ddelta_bias)，has_z 追加 dz（第 8 位），
      recompute_out_z 追加 out_z（第 9 位）——NPU 算子恒返 9 元组（缺省槽位为
      空 tensor），shim 按 CUDA 语义裁剪。
    """

    def fwd(self, u, delta, A, B, C, D=None, z=None, delta_bias=None,
            delta_softplus=False):
        # torch.ops 命名空间惰性解析：须先 import cann_bench 触发 .so 加载与
        # torch.library 注册，torch.ops.cann_bench.selective_scan_fwd 才可见。
        # 首次调用时解析并缓存算子句柄。
        global _NPU_OP
        if _NPU_OP is None:
            import cann_bench  # noqa: F401
            _NPU_OP = torch.ops.cann_bench.selective_scan_fwd
        _trace_op("fwd", _FWD_CALL, u=u, delta=delta, A=A, B=B, C=C, D=D,
                  z=z, delta_bias=delta_bias)
        u = u.contiguous()
        delta = delta.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        if z is not None:
            z = z.contiguous()
        if delta_bias is not None:
            delta_bias = delta_bias.contiguous()
        out, x, out_z = _NPU_OP(
            u, delta, A, B, C, D, z, delta_bias, bool(delta_softplus))
        if _DIAG["sync"]:
            torch.npu.synchronize()
            print(f"[shim] fwd sync OK", flush=True)
        if z is None:
            return out, x
        return out, x, out_z

    def bwd(self, u, delta, A, B, C, D, z, delta_bias, dout, x, out, dz,
            delta_softplus, recompute_out_z=False):
        """CUDA pybind bwd 序 → NPU 算子 bwd 序，返回元数对齐 CUDA 动态行为。

        - has_z=False 时 out=None（CUDA 语义：out 仅在 has_z 时消费），dz 与
          recompute_out_z 被 NPU plugin 静默忽略（对齐 CUDA 参考实现 if (has_z) 块）；
        - x 必须是同一 fwd 返回的真实 chunk 复合前缀（(a,h) 交错，fp32）——
          mamba 侧 ctx.save_for_backward 已保存 fwd 的 x，此处直接透传，严禁伪造；
        - dz 为预分配输出缓冲（提供时原地写入并原样返回），上游 SelectiveScanFn
          传 None、MambaInnerFn 传预分配 dz，均直传。
        """
        global _NPU_OP_BWD
        if _NPU_OP_BWD is None:
            import cann_bench  # noqa: F401  触发 .so 加载与 torch.library 注册
            _NPU_OP_BWD = torch.ops.cann_bench.selective_scan_bwd
        has_z = z is not None
        _trace_op("bwd", _BWD_CALL, u=u, delta=delta, A=A, B=B, C=C, D=D,
                  z=z, delta_bias=delta_bias, dout=dout, x=x, out=out, dz=dz)
        # 连续性兜底（同 fwd：NPU 算子要求全连续；dout/x/out/dz 亦同）
        u = u.contiguous()
        delta = delta.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        dout = dout.contiguous()
        if D is not None:
            D = D.contiguous()
        if z is not None:
            z = z.contiguous()
        if delta_bias is not None:
            delta_bias = delta_bias.contiguous()
        if x is not None:
            x = x.contiguous()
        if out is not None:
            out = out.contiguous()
        if dz is not None:
            dz = dz.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, dz_out, out_z = _NPU_OP_BWD(
            u, delta, A, B, C, dout, D, z, delta_bias, x, out, dz,
            bool(delta_softplus), bool(recompute_out_z))
        if _DIAG["sync"]:
            torch.npu.synchronize()
            print(f"[shim] bwd sync OK", flush=True)
        # NPU 恒返 9 元组；按 CUDA pybind 动态行为裁剪：
        #   7 元基础 + has_z → dz + recompute_out_z → out_z
        # （recompute 的 has_z 门控由 NPU plugin 内部完成：无 z 时 out_z 为空 tensor，
        #   与 CUDA 无 z 且 recompute 时返回 undefined out_z 无现实调用点分歧）
        result = [du, ddelta, dA, dB, dC, dD, ddelta_bias]
        if has_z:
            result.append(dz_out)
        if recompute_out_z:
            result.append(out_z)
        return tuple(result)

    def __getattr__(self, name):
        raise AttributeError(
            f"NPU shim 未提供 selective_scan_cuda.{name} "
            f"（已迁移 real 变体 fwd/bwd；complex 变体与其它入口未迁移）")


def install_npu_shim():
    """[2] 挂载 NPU shim（须在 import mamba_ssm 之后、模型推理之前调用）。"""
    import mamba_ssm.ops.selective_scan_interface as ssi
    ssi.selective_scan_cuda = _SelectiveScanCudaNpuShim()
    return ssi.selective_scan_cuda


def disable_triton_selective_state_update():
    """[3] 令 Mamba.step 回退纯 torch ref 步进（triton_ascend 编译不可用时的绕行开关）。

    mamba_simple.selective_state_update 为 import 期按名绑定（同问题 11），
    patch 模块属性即可生效。
    """
    import mamba_ssm.modules.mamba_simple as mamba_simple_mod
    mamba_simple_mod.selective_state_update = None


def enable_triton_cuda_ctx_compat():
    """[4] mamba_ssm 各 triton kernel（layer_norm / selective_state_update / ssd_*）
    内部均以 ``with torch.cuda.device(x.device.index):`` 包裹 kernel 启动；CPU-torch +
    torch_npu 环境下该上下文直接抛 RuntimeError（"PyTorch was compiled without CUDA
    support"）。patch 为 torch.npu.device 等价上下文，使 triton kernel 启动路径在
    NPU 上可用（triton_ascend 后端编译实测可行）。
    """
    torch.cuda.device = torch.npu.device


# [1] 模块 import 时立即执行 triton 兼容 polyfill（置于文件末尾确保函数已定义）：
# 本模块必须在 import mamba_ssm 之前被 import，polyfill 才能先于 mamba3 kernel
# 的 import 期 set_allocator 调用生效。
ensure_triton_compat()
