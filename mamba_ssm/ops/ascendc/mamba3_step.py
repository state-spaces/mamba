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

_OP_NAME = "mamba3_decode_step"
_LIB_ENV = "MAMBA3_DECODE_STEP_LIB"
_lib_handle = None


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
