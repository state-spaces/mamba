# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""mamba3_decode_step 的 CPU golden（float64 真值，再按输出 dtype 收窄）。

语义与 ``mamba_ssm.ops.cute.mamba3.mamba3_step_fn.selective_state_update_fused_ref_v2``
（mamba3_step_fn.py:741-813）逐行一致，仅把张量轴名换成 mamba3_step_fn 的 (B,H,D,N)/(B,R,H,N)。
任务书 §3.2 第 3 条：输入转 float64、标量按 float64 计算，再按输出 dtype 收窄落盘。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def mamba3_decode_step_ref(
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
    z: Optional[torch.Tensor] = None,
    zproj: Optional[torch.Tensor] = None,
    compute_dtype: torch.dtype = torch.float64,
):
    """返回 (out, new_state)，dtype 分别为 out / state 的存储 dtype。"""
    og_dtype = state.dtype
    out_dtype = x.dtype

    A_f = A.to(compute_dtype)
    dt_f = dt.to(compute_dtype)
    trap_f = trap.to(compute_dtype)
    D_f = D.to(compute_dtype)
    x_f = x.to(compute_dtype)
    xst_f = Xstate.to(compute_dtype)
    B_f = B.to(compute_dtype)
    C_f = C.to(compute_dtype)
    Bst_f = Bstate.to(compute_dtype)
    Xp_f = xproj.to(compute_dtype)
    st_f = state.to(compute_dtype)

    alpha = torch.exp(A_f * dt_f)                       # (B,H)
    beta = (1.0 - trap_f) * dt_f * alpha                # (B,H)
    gamma = trap_f * dt_f                               # (B,H)

    x_vals = x_f[:, None, :, :] * Xp_f[None, :, :, :]      # (B,R,H,D)
    xs_vals = xst_f[:, None, :, :] * Xp_f[None, :, :, :]   # (B,R,H,D)

    xBt_state = torch.einsum("brhd,brhn->bhdn", x_vals * gamma[:, None, :, None], B_f)
    xBt_prev = torch.einsum("brhd,brhn->bhdn", xs_vals * beta[:, None, :, None], Bst_f)

    new_state = st_f * alpha[:, :, None, None] + xBt_state + xBt_prev

    out_r = torch.einsum("bhdn,brhn->brhd", new_state, C_f)
    out_r = out_r + x_vals * D_f[None, None, :, None]

    # 门控只在 outproj 非空时生效：任务书 §2.1 第 4 条 / §7 第 1 条明确
    # "outproj 为空且 z 非空时按参考实现丢弃门控"，对应 mamba3_step_fn.py:533-563
    # 只在该分支写 out_r。
    if z is not None and outproj is not None:
        z_f = z.to(compute_dtype)
        Zp_f = zproj.to(compute_dtype)
        z_vals = z_f[:, None, :, :] * Zp_f[None, :, :, :]
        out_r = out_r * z_vals * torch.sigmoid(z_vals)

    if outproj is not None:
        Op_f = outproj.to(compute_dtype)
        out = torch.einsum("brhd,rhd->bhd", out_r, Op_f)
    else:
        out = out_r

    return out.to(out_dtype), new_state.to(og_dtype)
