# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""mamba3_siso_step 的 Ascend C 版实现（SISO decode 的单 token 递推）。

源实现（被替换的 Triton 版本）：``mamba_ssm.ops.triton.mamba3.mamba3_siso_step.mamba3_siso_step``。
本模块**同名、同签名、同返回结构**（``(Out, [Angle_State, SSM_State, K_State, V_State])``），
调用方只替换导入即可运行::

    from mamba_ssm.ops.ascendc.mamba3_siso_step import mamba3_siso_step

实现分两段（与 MIMO decode 链路一致的做法）：

1. **rotary + angle_state 更新**：复用 MIMO decode 已在用的 Triton 算子
   ``apply_rotary_qk_inference_fwd``（``mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py``）。
   源 Triton ``mamba3_siso_step`` 是把这一步 fuse 在 step kernel 里的；Ascend C 的
   ``mamba3_step_fn``（``mamba_ssm/ops/ascendc/mamba3_step.py``）不含 rotary，故按 MIMO
   生产链路的同一做法把它前置一次。
2. **状态递推 + 输出**：调用 Ascend C 算子 ``mamba3_step_fn``，用 ``R=1``（mimo_rank=1）退化
   成 SISO 的语义。

参数映射（SISO 记号 → ``mamba3_step_fn`` 实参）::

    state   = Input_SSM_State                     (B, H, Dv, Dq)
    Bstate  = Input_K_State                       (B, H, Dq)      -> (B, 1, H, Dq)
    B       = k_rot（本模块第一段算出的旋转后 K）  -> (B, 1, H, Dq)
    C       = q_rot（本模块第一段算出的旋转后 Q）  -> (B, 1, H, Dq)
    x       = V                                    (B, H, Dv)
    Xstate  = Input_V_State                        (B, H, Dv)
    A       = ADT / DT     （见下方 α 口径）        (B, H)
    dt      = DT                                   (B, H)
    trap    = sigmoid(Trap)（见下方 trap 口径）     (B, H)
    D       = D（None 时传 0）                     (H,)
    xproj/outproj/zproj = 全 1 的 (1, H, Dv)
    z       = Z（None 时不传）
    out     = 预分配 (B, H, Dv) fp32 缓冲，最后按 V.dtype 收窄

两处**口径对齐**（源实现与 ``mamba3_step_fn`` 的约定不同，必须在 glue 层换算）：

* **α**：源实现 ``alpha = exp(ADT)``（ADT 已是 A·dt 的乘积）；``mamba3_step_fn`` 内部是
  ``alpha = exp(A[b,h] * dt[b,h])``（A 为原始负特征值、dt 为步长）。因此传 ``A = ADT / DT``、
  ``dt = DT``。fp32 除法 + 乘法各一次舍入，相对误差 ~1e-7，远低于各 dtype 的判定门槛。
  ``DT`` 恒为正（生产链路是 ``softplus`` 输出），不做额外值域校验。
* **trap**：源实现收的是 **pre-sigmoid** 的投影（kernel 内自己 sigmoid）；``mamba3_step_fn``
  收的是 **post-sigmoid** 的梯形系数（MIMO 链路在 ``Mamba3._preprocess`` 里先 sigmoid 再传）。
  因此这里先 ``torch.sigmoid(Trap)`` 再传；sigmoid 在 ``Trap`` 的原 dtype（bf16）上算，
  与 ``mamba3_siso_step_ref`` 的 ``trap = torch.sigmoid(trap)`` 口径一致。

数值口径（与 CANN 上 Triton 版 ``PRECISE_FP32`` 路径逐项对齐）：

* Q/K 先 upcast 到 fp32 再做 bias 相加与旋转，旋转结果保持 fp32 送进 ``mamba3_step_fn``
  （源 kernel 在 CANN 上正是保留 fp32，见 ``NPU_MIGRATION_mamba3_siso.md`` §12.3：
  bf16 舍入会在 GQA 的相消点上放大成单点离群）；
* ``state``/``x``/``Xstate``/``z``/``out`` 走 fp32 计算，只在 ``Out`` 落盘时收窄到 ``V.dtype``
  （源 kernel 同样是 fp32 计算 + store 时收窄）；
* ``angle_state`` 按源实现做 ``mod 2π`` 归一（rotary 算子只做累加、不取模；cos/sin 与取模无关，
  但取模保证状态量级不随步数线性增长）。

不原地改写入参：返回的四个 state 都是新张量（与源 Triton 版一致——它也是新分配
``Output_Angle_State`` / ``Output_SSM_State`` / ``Output_K_State``）。``V_State`` 按源实现
原样返回输入 ``V``。

限制（与 ``mamba3_step_fn`` 的接受面一致）：

* ``headdim_v`` 必须能被 ``tile_D ∈ {32,64,128}`` 之一整除（算子要求 ``D % tile_D == 0``）；
* ``d_state``（= ``headdim_qk``）必须是 8 的倍数；
* 只在 NPU 上可用（``mamba3_step_fn`` 为 Ascend C 直调算子）。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

__all__ = ["mamba3_siso_step", "pick_tile_d"]

_TWO_PI = 2.0 * math.pi
_TILE_D_CHOICES = (128, 64, 32)

# 全 1 的投影/门控权重缓存：内容不变量，只在 kernel 侧按 stride 读，不复用会有每 token
# 三次 `torch.ones` 的额外下发。key = (heads, headdim_v, device.type, device.index, dtype)。
_ONES_CACHE: dict = {}
_ONES_CACHE_MAX = 16


def pick_tile_d(headdim_v: int) -> int:
    """给 ``mamba3_step_fn`` 选 D 轴切分粒度（算子只接受 32/64/128，且需整除 D）。"""
    for tile_d in _TILE_D_CHOICES:
        if headdim_v % tile_d == 0:
            return tile_d
    raise ValueError(
        f"mamba3_siso_step: headdim_v({headdim_v}) 必须能被 {_TILE_D_CHOICES} 之一整除"
        f"（mamba3_step_fn 的 tile_D 约束）"
    )


def _ones_proj(heads: int, headdim_v: int, device: torch.device) -> torch.Tensor:
    """``(1, H, Dv)`` 的 fp32 全 1 张量（xproj / outproj / zproj 共用，进程内按形状缓存）。"""
    key = (heads, headdim_v, device.type, device.index)
    tensor = _ONES_CACHE.get(key)
    if tensor is None:
        if len(_ONES_CACHE) >= _ONES_CACHE_MAX:
            _ONES_CACHE.clear()
        tensor = torch.ones((1, heads, headdim_v), device=device, dtype=torch.float32)
        _ONES_CACHE[key] = tensor
    return tensor


def _rotary_fwd():
    """惰性取 MIMO decode 用的 Triton rotary 算子（避免 import 期硬依赖 triton）。"""
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import apply_rotary_qk_inference_fwd

    return apply_rotary_qk_inference_fwd


def _step_fn():
    """惰性取 Ascend C 的 ``mamba3_step_fn``（其 import 不依赖 triton/CANN，保持同一写法）。"""
    from mamba_ssm.ops.ascendc.mamba3_step import mamba3_step_fn

    return mamba3_step_fn


def mamba3_siso_step(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Input_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    """单 token 的 Mamba-3 SISO 递推（Ascend C 版，签名/返回与 Triton 版一致）。

    Inputs:
        Q, K:           (batch, nheads_qk, headdim_qk)
        V:              (batch, nheads, headdim_v)
        ADT, DT, Trap:  (batch, nheads)；Trap 为 **pre-sigmoid** 的投影
        Q_bias, K_bias: (nheads, headdim_qk)
        Angles:         (batch, nheads, headdim_angles)，raw 角度投影（不做 cumsum）
        D:              (nheads,)，None 时不加 D 直连项
        Z:              (batch, nheads, headdim_v)，None 时不做 SiLU 门控
        Input_States:   (Angle_State, SSM_State, K_State, V_State)，形状见源实现的文档

    Returns:
        Out:    (batch, nheads, headdim_v)，dtype 同 ``V``
        States: [Angle_State, SSM_State, K_State, V_State]（均为新张量，不改写入参）
    """
    # ---- 形状校验：与 Triton 版逐条一致（同一批 assert，含异常类型） ----
    batch, nheads_qk, headdim_qk = Q.shape
    _, nheads, headdim_v = V.shape
    device = Q.device

    assert Q.shape == K.shape, f"Q and K shape mismatch: {Q.shape} vs {K.shape}"
    assert nheads % nheads_qk == 0, f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})"
    assert ADT.shape == (batch, nheads), f"ADT shape mismatch: expected {(batch, nheads)}, got {ADT.shape}"
    assert DT.shape == (batch, nheads), f"DT shape mismatch: expected {(batch, nheads)}, got {DT.shape}"
    assert Trap.shape == (batch, nheads), f"Trap shape mismatch: expected {(batch, nheads)}, got {Trap.shape}"
    assert Q_bias.shape == (nheads, headdim_qk), (
        f"Q_bias shape mismatch: expected {(nheads, headdim_qk)}, got {Q_bias.shape}"
    )
    assert K_bias.shape == (nheads, headdim_qk), (
        f"K_bias shape mismatch: expected {(nheads, headdim_qk)}, got {K_bias.shape}"
    )
    headdim_angles = Angles.shape[-1]
    assert headdim_angles <= headdim_qk // 2 and headdim_angles % 2 == 0, (
        f"headdim_angles ({headdim_angles}) must be <= headdim_qk // 2 ({headdim_qk // 2}) and even."
    )
    assert Angles.shape == (batch, nheads, headdim_angles), (
        f"Angles shape mismatch: expected {(batch, nheads, headdim_angles)}, got {Angles.shape}"
    )
    if D is not None:
        assert D.shape == (nheads,), f"D shape mismatch: expected {(nheads,)}, got {D.shape}"
    if Z is not None:
        assert Z.shape == (batch, nheads, headdim_v), (
            f"Z shape mismatch: expected {(batch, nheads, headdim_v)}, got {Z.shape}"
        )

    assert Input_States is not None, "Input_States must be provided: (Angle_State, SSM_State, K_State, V_State)"
    Input_Angle_State, Input_SSM_State, Input_K_State, Input_V_State = Input_States
    assert Input_Angle_State.shape == (batch, nheads, headdim_angles), (
        f"Input_Angle_State shape mismatch: expected {(batch, nheads, headdim_angles)}, "
        f"got {Input_Angle_State.shape}"
    )
    assert Input_SSM_State.shape == (batch, nheads, headdim_v, headdim_qk), (
        f"Input_SSM_State shape mismatch: expected {(batch, nheads, headdim_v, headdim_qk)}, "
        f"got {Input_SSM_State.shape}"
    )
    assert Input_K_State.shape == (batch, nheads, headdim_qk), (
        f"Input_K_State shape mismatch: expected {(batch, nheads, headdim_qk)}, got {Input_K_State.shape}"
    )
    assert Input_V_State.shape == (batch, nheads, headdim_v), (
        f"Input_V_State shape mismatch: expected {(batch, nheads, headdim_v)}, got {Input_V_State.shape}"
    )

    tile_d = pick_tile_d(headdim_v)
    gqa = nheads // nheads_qk

    # ---- 第一段：rotary + angle_state ----
    # Q/K 按 value head 展开（value head h 用 qk head h // gqa，与源 kernel 的
    # head_idx_qk = pid_head // (nheads // nheads_qk) 以及 ref 的 repeat 展开一致），
    # 并 upcast 到 fp32：bias 相加与旋转都在 fp32 上做（CANN 上 Triton 版的 PRECISE_FP32 口径）。
    q4 = Q.to(torch.float32).repeat_interleave(gqa, dim=1).unsqueeze(1)  # (B, 1, H, Dq)
    k4 = K.to(torch.float32).repeat_interleave(gqa, dim=1).unsqueeze(1)  # (B, 1, H, Dq)
    bias_q = Q_bias.to(torch.float32).unsqueeze(0)  # (1, H, Dq)
    bias_k = K_bias.to(torch.float32).unsqueeze(0)  # (1, H, Dq)

    dt_f32 = DT.to(torch.float32)
    angle_state = Input_Angle_State.to(torch.float32).contiguous()  # (B, H, A)
    angle_proj = Angles.to(torch.float32).contiguous()  # (B, H, A)

    q_rot, k_rot, angle_new = _rotary_fwd()(
        q4,
        k4,
        angle_state,
        angle_proj,
        dt_f32,
        bias_q=bias_q,
        bias_k=bias_k,
        inplace=False,
        conjugate=False,
        rotate_pairwise=True,  # 源实现是 (x0,x1),(x2,x3)… 的相邻配对
    )
    q_rot = q_rot.squeeze(1)  # (B, H, Dq) fp32
    k_rot = k_rot.squeeze(1)  # (B, H, Dq) fp32

    # 角度状态按源实现归一：angle -= 2π * floor(angle / 2π)
    angle_out = angle_new - _TWO_PI * torch.floor(angle_new / _TWO_PI)

    # ---- 第二段：Ascend C mamba3_step_fn（R=1） ----
    f32 = torch.float32
    # α 口径：算子内部算 exp(A*dt)，源实现是 exp(ADT) => A = ADT / DT
    a_eff = ADT.to(f32) / dt_f32
    # trap 口径：算子要 post-sigmoid；源实现收 pre-sigmoid 并在 kernel 内 sigmoid
    trap_eff = torch.sigmoid(Trap).to(f32)

    if D is not None:
        d_eff = D.to(f32)
    else:
        # 无 D 直连项：传 0，算子的 `out += D * x * xproj` 逐元素得 0
        d_eff = torch.zeros((nheads,), device=device, dtype=f32)

    proj = _ones_proj(nheads, headdim_v, device)  # (1, H, Dv) fp32 全 1

    state = Input_SSM_State
    # 不改写入参：源 Triton 版同样是新分配 state_out
    state_out = torch.empty_like(state)
    x_f32 = V.to(f32)
    x_state_f32 = Input_V_State.to(f32)
    z_f32 = None if Z is None else Z.to(f32)
    out_f32 = torch.empty((batch, nheads, headdim_v), device=device, dtype=f32)

    bstate = Input_K_State.to(f32).unsqueeze(1)  # (B, 1, H, Dq)
    b_rot = k_rot.unsqueeze(1)  # (B, 1, H, Dq)
    c_rot = q_rot.unsqueeze(1)  # (B, 1, H, Dq)

    _step_fn()(
        state,
        bstate,
        x_state_f32,
        a_eff,
        b_rot,
        c_rot,
        d_eff,
        x_f32,
        dt_f32,
        trap_eff,
        proj,  # xproj
        outproj=proj,
        state_out=state_out,
        out=out_f32,
        z=z_f32,
        zproj=None if z_f32 is None else proj,
        tile_D=tile_d,
    )

    out = out_f32.to(V.dtype)
    # V_State 按源实现原样返回输入 V（既不拷贝也不算新值）
    return out, [angle_out, state_out, k_rot, V]
