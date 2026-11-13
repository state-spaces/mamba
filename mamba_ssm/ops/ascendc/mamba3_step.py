# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""mamba3_decode_step 的 Python 入口（Ascend C kernel 直调）。

与参考实现 ``mamba_ssm.ops.cute.mamba3.mamba3_step_fn.mamba3_step_fn`` 同名同签名同副作用：
就地写 ``state_out``（缺省即 ``state``）与 ``out``，返回 ``None``。

任务书 ``docs/plan/mamba3_decode_step_task_doc.md``：
  - §2.4 参数表 / §2.5 实现约束：全部参数合法性由本入口在 host 侧校验并抛 ``ValueError``；
  - §2.6 精度见 ``SELFTEST.md``。

PLUT-189 D3（下发侧第三块，语义等价，与 CATLASS 路线 PLUT-181 同一做法）：§2.4/§2.5 的
**同一套**参数校验从「Python 逐项属性访问」改成「一次 C++ 调用」
（``Mamba3DecodeStepHostValidator::check`` → ``op_extension/mamba3_decode_step_validate.cpp``）。
约束与做法：
  * 校验项、判定顺序、拒绝时机（host 侧、下发 kernel 之前）与原实现相同；
  * **每次调用都完整跑一遍**，不缓存结果、不抽样、不异步；
  * 拒绝路径**重放**下面的原逐项 Python 实现（``_mamba3_step_fn_python``，逐行未改），
    因此异常类型（``ValueError``）与文案与原实现**逐字一致**——重放只是一次「再校验」，
    不是把校验从通过路径上省掉：通过/拒绝两侧每次调用都真跑了全部判据；
  * 张量类型判定仍由 C++ 侧的 pybind11 签名兜住（非 ``torch.Tensor`` 的参数原实现在读取
    ``.shape`` 时会先抛其它异常，这类输入的原始行为由同一份原实现兜住）。

用法::

    from mamba_ssm.ops.ascendc import mamba3_step_fn
    mamba3_step_fn(state, Bstate, Xstate, A, B, C, D, x, dt, trap, xproj,
                   outproj=outproj, state_out=None, out=out, z=z, zproj=zproj, tile_D=64)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

__all__ = ["mamba3_step_fn", "SUPPORTED_DTYPES", "SUPPORTED_TILE_D"]

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SUPPORTED_TILE_D = (32, 64, 128)

# A / dt / trap / D 四项在 tiling 里共用同一个 dtype 编码（host 侧取自 A），kernel 按该宽度读
# 这四个 buffer，所以它们必须同 dtype；kernel 内部一律按 fp32 计算，故入口把半精度归一到 fp32。
_HALF_DTYPES = (torch.float16, torch.bfloat16)

_OP_NAME = "mamba3_decode_step"
_LIB_ENV = "MAMBA3_DECODE_STEP_LIB"
_EXT_MODULE = "mamba3_decode_step_ext"
_lib_handle = None
_VALIDATOR = None
_STEP = None


def _lib_path() -> Path:
    env = os.environ.get(_LIB_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent / "build" / "libmamba3_decode_step_ops.so"


def _load_op():
    """加载 libmamba3_decode_step_ops.so（进程内只加载一次）。"""
    global _lib_handle
    if _lib_handle is not None:
        return _lib_handle
    path = _lib_path()
    if not path.is_file():
        raise RuntimeError(
            f"{_OP_NAME}: 未找到算子动态库 {path}；请先执行 "
            f"`bash {Path(__file__).resolve().parent / 'build.sh'}` 编译，"
            f"或通过环境变量 {_LIB_ENV} 指定 .so 路径"
        )
    torch.ops.load_library(str(path))
    _lib_handle = getattr(torch.ops.npu, _OP_NAME)
    return _lib_handle


def _validator():
    """取 C++ 侧参数校验函数（进程内只取一次；每次调用仍逐项跑全部判据）。

    走 pybind11 直调（``spec_from_file_location`` 加载同一个 .so 里的
    ``mamba3_decode_step_ext.validate``）：不装箱成 IValue、不查 dispatcher schema。
    与 CATLASS 侧 PLUT-181 的口径一致（那份实测见其报告；两边的模块名不同，
    避免两条路线的 .so 在同一进程加载时互相顶替）。

    PLUT-215：模块加载时顺带取 ``step``（单跨边界入口，校验 + 下发一次调用完成）。
    旧 .so 没有该符号时保持 ``None``，调用方回落到「validate + op 派发」两步路径。
    """
    global _VALIDATOR, _STEP
    if _VALIDATOR is None:
        path = _lib_path()
        if not path.is_file():
            raise RuntimeError(
                f"{_OP_NAME}: 未找到算子动态库 {path}；请先执行 "
                f"`bash {Path(__file__).resolve().parent / 'build.sh'}` 编译，"
                f"或通过环境变量 {_LIB_ENV} 指定 .so 路径"
            )
        import importlib.util

        spec = importlib.util.spec_from_file_location(_EXT_MODULE, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"{_OP_NAME}: 无法加载校验扩展 {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _VALIDATOR = module.validate
        _STEP = getattr(module, "step", None)
    return _VALIDATOR


def _check_tensor(name: str, tensor: torch.Tensor, rank: int, shape, need_npu: bool = True) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{_OP_NAME}: {name} 必须是 torch.Tensor，实际 {type(tensor)}")
    if tensor.dim() != rank:
        raise ValueError(f"{_OP_NAME}: {name} 的秩必须为 {rank}，实际 {tensor.dim()}（shape={tuple(tensor.shape)}）")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{_OP_NAME}: {name} 的形状必须为 {tuple(shape)}，实际 {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise ValueError(f"{_OP_NAME}: {name} 的 numel 不能为 0（空 tensor 为非法输入）")
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"{_OP_NAME}: {name} 的 dtype 必须属于 {[str(d) for d in SUPPORTED_DTYPES]}，实际 {tensor.dtype}"
        )
    if need_npu and tensor.device.type != "npu":
        raise ValueError(f"{_OP_NAME}: {name} 必须在 NPU 上，实际 device={tensor.device}")


def _check_dtype_group(name: str, tensors) -> torch.dtype:
    dtypes = {t.dtype for t in tensors}
    if len(dtypes) != 1:
        raise ValueError(f"{_OP_NAME}: dtype 分组不一致：{name} 必须同 dtype，实际 {[str(d) for d in dtypes]}")
    return dtypes.pop()


def _storage_range(tensor: torch.Tensor):
    storage = tensor.untyped_storage()
    base = storage.data_ptr()
    return base, base + storage.nbytes()


def _check_alias(name: str, out_tensor: torch.Tensor, in_tensor: torch.Tensor) -> bool:
    """返回是否同址。部分重叠返回参数错误（任务书 §2.4 state_out 行、§7 第 3 条）。"""
    if out_tensor.data_ptr() == in_tensor.data_ptr():
        return True
    o0, o1 = _storage_range(out_tensor)
    i0, i1 = _storage_range(in_tensor)
    if o0 < i1 and i0 < o1:
        raise ValueError(f"{_OP_NAME}: {name} 与输入张量存在部分重叠；只允许同址或完全独立")
    return False


def _as_innermost_contiguous(name: str, tensor: torch.Tensor, is_output: bool = False) -> torch.Tensor:
    """kernel 需按 stride 读取；仅要求最后一维 stride == 1。

    非最后一维的 stride 由 kernel 直接支持（含 B/C 的 head 轴 stride 0 视图，不物化副本）；
    最后一维非连续时输入物化为连续副本，输出张量不接受该情形（避免静默丢弃写回）。
    """
    if tensor.stride(-1) == 1:
        return tensor
    if is_output:
        raise ValueError(f"{_OP_NAME}: {name} 的最后一维 stride 必须为 1，实际 {tensor.stride(-1)}")
    return tensor.contiguous()


def _mamba3_step_fn_python(
    state: torch.Tensor,
    Bstate: torch.Tensor,
    Xstate: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    xproj: torch.Tensor,
    outproj: Optional[torch.Tensor] = None,
    state_out: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    zproj: Optional[torch.Tensor] = None,
    tile_D: int = 64,
    num_warps: int = 2,
) -> None:
    """PLUT-189 之前的原逐项实现（逐行保留，作为「拒绝路径重放 / 非张量参数」的同一份判据）。

    这份实现没有被删除，也不允许被改成「只留几条」：C++ 判定拒绝时由它复现原异常，
    非 ``torch.Tensor`` 参数由它兜住原始行为。它跑的就是 §2.4/§2.5 全量校验。
    """
    # ---- 标量属性 ----
    if isinstance(tile_D, bool) or not isinstance(tile_D, int) or tile_D not in SUPPORTED_TILE_D:
        raise ValueError(f"{_OP_NAME}: tile_D 必须属于 {SUPPORTED_TILE_D}，实际 {tile_D!r}")
    if isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps <= 0:
        raise ValueError(f"{_OP_NAME}: num_warps 必须为正整数（兼容参数，接收后忽略），实际 {num_warps!r}")

    if out is None:
        raise ValueError(f"{_OP_NAME}: out 必须由调用方预分配（不支持内部创建输出）")
    if (z is None) != (zproj is None):
        raise ValueError(f"{_OP_NAME}: z 与 zproj 必须同时为 None 或同时存在")

    rank_state = 4
    _check_tensor("state", state, rank_state, None)
    Bsz, H, Dd, N = state.shape
    R = Bstate.shape[1]

    if outproj is None and tuple(out.shape) != (Bsz, R, H, Dd):
        raise ValueError(
            f"{_OP_NAME}: outproj 为 None 时 out 必须是 [B,R,H,D]={(Bsz, R, H, Dd)}，实际 {tuple(out.shape)}"
        )
    if outproj is not None and tuple(out.shape) != (Bsz, H, Dd):
        raise ValueError(
            f"{_OP_NAME}: outproj 非空时 out 必须是 [B,H,D]={(Bsz, H, Dd)}，实际 {tuple(out.shape)}"
        )

    if Dd % tile_D != 0:
        raise ValueError(f"{_OP_NAME}: D({Dd}) 必须能被 tile_D({tile_D}) 整除")
    if R < 1:
        raise ValueError(f"{_OP_NAME}: R(=Bstate.shape[1]) 必须 >= 1，实际 {R}")
    if N % 8 != 0:
        raise ValueError(f"{_OP_NAME}: N(=d_state={N}) 必须为 8 的倍数（参考实现的向量化读写前置约束）")

    # ---- 秩与形状 ----
    _check_tensor("Bstate", Bstate, 4, (Bsz, R, H, N))
    _check_tensor("Xstate", Xstate, 3, (Bsz, H, Dd))
    _check_tensor("A", A, 2, (Bsz, H))
    _check_tensor("B", B, 4, (Bsz, R, H, N))
    _check_tensor("C", C, 4, (Bsz, R, H, N))
    _check_tensor("D", D, 1, (H,))
    _check_tensor("x", x, 3, (Bsz, H, Dd))
    _check_tensor("dt", dt, 2, (Bsz, H))
    _check_tensor("trap", trap, 2, (Bsz, H))
    _check_tensor("xproj", xproj, 3, (R, H, Dd))
    if outproj is not None:
        _check_tensor("outproj", outproj, 3, (R, H, Dd))
        _check_tensor("out", out, 3, (Bsz, H, Dd))
    else:
        _check_tensor("out", out, 4, (Bsz, R, H, Dd))
    if z is not None:
        _check_tensor("z", z, 3, (Bsz, H, Dd))
        _check_tensor("zproj", zproj, 3, (R, H, Dd))

    if state_out is None:
        state_out = state
        inplace = True
    else:
        _check_tensor("state_out", state_out, 4, (Bsz, H, Dd, N))
        inplace = _check_alias("state_out", state_out, state)
    if inplace and state_out.dtype != state.dtype:
        raise ValueError(f"{_OP_NAME}: state_out 与 state 同址时 dtype 必须一致")

    # ---- dtype 分组（任务书 §2.4 dtype：五组，组内一致、组间独立） ----
    _check_dtype_group("state/state_out", (state, state_out))
    _check_dtype_group("B/Bstate/C", (B, Bstate, C))
    _check_dtype_group("x/Xstate/z/out", tuple(t for t in (x, Xstate, z, out) if t is not None))
    _check_dtype_group("xproj/outproj/zproj", tuple(t for t in (xproj, outproj, zproj) if t is not None))
    # 第五组：四项在 tiling 里共用一个 scalar dtype 编码，混搭会被 kernel 按另一宽度读成垃圾值。
    _check_dtype_group("A/dt/trap/D", (A, dt, trap, D))

    # ---- 最后一维连续（kernel 按 stride 读取外层维度） ----
    state_in = _as_innermost_contiguous("state", state)
    state_out_t = _as_innermost_contiguous("state_out", state_out, is_output=True)
    Bstate = _as_innermost_contiguous("Bstate", Bstate)
    Xstate = _as_innermost_contiguous("Xstate", Xstate)
    A = _as_innermost_contiguous("A", A)
    B = _as_innermost_contiguous("B", B)
    C = _as_innermost_contiguous("C", C)
    D = _as_innermost_contiguous("D", D)
    x = _as_innermost_contiguous("x", x)
    dt = _as_innermost_contiguous("dt", dt)
    trap = _as_innermost_contiguous("trap", trap)
    xproj = _as_innermost_contiguous("xproj", xproj)
    if outproj is not None:
        outproj = _as_innermost_contiguous("outproj", outproj)
    if z is not None:
        z = _as_innermost_contiguous("z", z)
        zproj = _as_innermost_contiguous("zproj", zproj)
    out = _as_innermost_contiguous("out", out, is_output=True)

    if inplace and state_in.data_ptr() != state_out_t.data_ptr():
        # state 被物化（最后一维非连续）时不能静默丢掉就地更新
        raise ValueError(
            f"{_OP_NAME}: state 最后一维非连续且 state_out 与 state 同址，无法就地更新；"
            f"请显式传入独立的 state_out"
        )

    op = _load_op()
    op(state_in, Bstate, Xstate, A, B, C, D, x, dt, trap, xproj, outproj, z, zproj, state_out_t, out, int(tile_D))
    return None


def mamba3_step_fn(
    state: torch.Tensor,
    Bstate: torch.Tensor,
    Xstate: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    xproj: torch.Tensor,
    outproj: Optional[torch.Tensor] = None,
    state_out: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    zproj: Optional[torch.Tensor] = None,
    tile_D: int = 64,
    num_warps: int = 2,
) -> None:
    """单 token Mamba-3 SSM 递推（语义等同参考实现 ``mamba3_step_fn``）。"""
    # ---- 标量属性：与原实现同一判定、同一文案（Python 侧保留，开销 ~0.3 µs） ----
    if isinstance(tile_D, bool) or not isinstance(tile_D, int) or tile_D not in SUPPORTED_TILE_D:
        raise ValueError(f"{_OP_NAME}: tile_D 必须属于 {SUPPORTED_TILE_D}，实际 {tile_D!r}")
    if isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps <= 0:
        raise ValueError(f"{_OP_NAME}: num_warps 必须为正整数（兼容参数，接收后忽略），实际 {num_warps!r}")
    if out is None:
        raise ValueError(f"{_OP_NAME}: out 必须由调用方预分配（不支持内部创建输出）")
    if (z is None) != (zproj is None):
        raise ValueError(f"{_OP_NAME}: z 与 zproj 必须同时为 None 或同时存在")

    # ---- 标量侧四项归一到 fp32（kernel 内部计算 dtype） ----
    # 模块路径给的是 fp32 的 A 配 bf16 的 dt/trap（两者都由 bf16 激活算出），而四项在 tiling 里
    # 共用一个 dtype 编码且取自 A：原样下发时 kernel 会按 fp32 的宽度读两个 bf16 buffer，读出垃圾值
    # 且不报错。这里统一到 fp32，调用方（含 Mamba3.step）无需自己 upcast；非 Tensor 实参不在此
    # 处理，交给下面的原逐项实现复现改动前的原始异常。
    _scalars = (A, dt, trap, D)
    if all(isinstance(t, torch.Tensor) for t in _scalars) and any(
            t.dtype in _HALF_DTYPES for t in _scalars):
        A, dt, trap, D = tuple(
            t.to(torch.float32) if t.dtype in _HALF_DTYPES else t for t in _scalars)

    # ---- 快速路径：张量侧全部判据（§2.4/§2.5）下沉 C++，**每次调用真跑一遍** ----
    # 非 torch.Tensor 的实参由 pybind11 抛 TypeError（零成本 try：不改解释器状态），
    # 此时交回原逐项实现，由它复现改动前对这类输入的原始行为（异常类型/文案/时机）。
    #
    # PLUT-215：默认走单跨边界入口 ``step``（同一调用内完成全部判据 + 下发），
    # 相比旧的「validate 一次 + torch 派发一次」少一次跨边界、少一遍 17 个张量拆箱；
    # 判据、判定顺序、拒绝时机、下发实参、kernel 语义均不变。
    entry = _STEP
    if entry is None:
        _validator()  # 加载扩展；顺带填充 _VALIDATOR / _STEP
        entry = _STEP
    if entry is not None:
        try:
            rc = entry(
                state, Bstate, Xstate, A, B, C, D, x, dt, trap, xproj,
                outproj, state_out, out, z, zproj, tile_D,
            )
        except TypeError:
            return _mamba3_step_fn_python(
                state, Bstate, Xstate, A, B, C, D, x, dt, trap, xproj,
                outproj=outproj, state_out=state_out, out=out, z=z, zproj=zproj,
                tile_D=tile_D, num_warps=num_warps,
            )
        if rc == 0:
            return None
        # 拒绝路径：重放同一份原逐项实现，保证异常类型/文案/拒绝时机与改动前逐字一致。
        # 注意这不是「跳过校验」：上面 C++ 已经跑完全部判据，这里再跑一遍只为构造原异常。
        _mamba3_step_fn_python(
            state, Bstate, Xstate, A, B, C, D, x, dt, trap, xproj,
            outproj=outproj, state_out=state_out, out=out, z=z, zproj=zproj,
            tile_D=tile_D, num_warps=num_warps,
        )
        raise RuntimeError(f"{_OP_NAME}: 内部校验口径不一致（C++ rc={rc}，Python 逐项校验未拒绝）")

    # ---- 兼容路径（旧 .so 无 step 符号）：validate 一次 + torch 派发一次 ----
    validate = _VALIDATOR
    try:
        rc, resolved = validate(
            state, Bstate, Xstate, A, B, C, D, x, dt, trap, xproj,
            outproj, state_out, out, z, zproj, tile_D,
        )
    except TypeError:
        return _mamba3_step_fn_python(
            state, Bstate, Xstate, A, B, C, D, x, dt, trap, xproj,
            outproj=outproj, state_out=state_out, out=out, z=z, zproj=zproj,
            tile_D=tile_D, num_warps=num_warps,
        )

    if rc == 0:
        if resolved is None:
            # 常见路径：没有输入需要物化，直接下发（实参就是调用方给的那批对象）。
            # state_out 缺省即 state（任务书 §2.4 state_out 行）；z 与 zproj 同时缺省时
            # kernel 侧按 hasZ=false 走无门控分支，与改动前完全一致。
            _load_op()(
                state, Bstate, Xstate, A, B, C, D, x, dt, trap, xproj, outproj, z, zproj,
                state if state_out is None else state_out, out, tile_D,
            )
        else:
            # 少数路径：末维非连续的输入已被物化成连续副本，用回传的实参下发
            _load_op()(*resolved, tile_D)
        return None
    # 拒绝路径：重放同一份原逐项实现，保证异常类型/文案/拒绝时机与改动前逐字一致。
    # 注意这不是「跳过校验」：上面 C++ 已经跑完全部判据，这里再跑一遍只为构造原异常。
    _mamba3_step_fn_python(
        state, Bstate, Xstate, A, B, C, D, x, dt, trap, xproj,
        outproj=outproj, state_out=state_out, out=out, z=z, zproj=zproj,
        tile_D=tile_D, num_warps=num_warps,
    )
    # 原实现未拒绝 => C++ 判定与 §2.4/§2.5 不一致（实现缺陷），必须显形而不是静默放行
    raise RuntimeError(f"{_OP_NAME}: 内部校验口径不一致（C++ rc={rc}，Python 逐项校验未拒绝）")
