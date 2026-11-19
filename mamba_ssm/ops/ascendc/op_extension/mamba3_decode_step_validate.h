/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAMBA3_DECODE_STEP_VALIDATE_H
#define MAMBA3_DECODE_STEP_VALIDATE_H

#include <cstdint>
#include <tuple>
#include <vector>

#include <c10/util/Optional.h>
#include <torch/extension.h>

namespace ascend_kernel {

// PLUT-189 D3：mamba3_decode_step（路线 A / 纯 Ascend C）参数合法性校验的 C++ 实现。
//
// 语义契约（与 mamba_ssm/ops/ascendc/mamba3_step.py 里逐项 Python 实现逐条对齐）：
//   * 每次调用都完整跑一遍，不做缓存、不做抽样、不异步；
//   * 校验项、判定顺序、拒绝时机（host 侧、下发 kernel 之前）与 Python 原实现相同；
//   * 通过路径不再跑 Python 逐项校验；被拒绝时 Python 侧**重放**同一份原逐项实现，
//     以取得逐字一致的异常类型（ValueError）与文案。
//
// 返回 (rc, resolved)：
//   rc       0 = 通过；> 0 = 第 N 项判据拒绝（仅用于回溯，不参与判定口径）。
//   resolved 仅在**有输入被物化成连续副本**时给出（16 槽，顺序 = op 入参顺序）；
//            否则为 None——避免每次调用白包 16 个 at::Tensor 回 Python。
using ValidationResult = std::tuple<int64_t, c10::optional<std::vector<c10::optional<at::Tensor>>>>;

// 绑定方式：pybind11 直调（不走 dispatcher、不走 torch 自定义类），与 CATLASS 侧
// PLUT-181 的结论一致；本类名刻意与 CATLASS 侧校验类不同名，避免两条路线的 .so
// 在同一进程加载时发生符号互相顶替。
struct Mamba3DecodeStepHostValidator {
    ValidationResult check(at::Tensor state, at::Tensor bstate, at::Tensor xstate, at::Tensor a, at::Tensor b,
                           at::Tensor c, at::Tensor d, at::Tensor x, at::Tensor dt, at::Tensor trap,
                           at::Tensor xproj, c10::optional<at::Tensor> outproj,
                           c10::optional<at::Tensor> state_out, c10::optional<at::Tensor> out,
                           c10::optional<at::Tensor> z, c10::optional<at::Tensor> zproj, int64_t tile_d) const;
};

// pybind11 入口（Python 侧用 importlib 直接加载本 .so，取 validate 函数）。
// 模块名必须与 Python 侧 spec 名一致：mamba3_decode_step_ext。
ValidationResult ValidateArgsEntry(at::Tensor state, at::Tensor bstate, at::Tensor xstate, at::Tensor a, at::Tensor b,
                                   at::Tensor c, at::Tensor d, at::Tensor x, at::Tensor dt, at::Tensor trap,
                                   at::Tensor xproj, c10::optional<at::Tensor> outproj,
                                   c10::optional<at::Tensor> state_out, c10::optional<at::Tensor> out,
                                   c10::optional<at::Tensor> z, c10::optional<at::Tensor> zproj, int64_t tile_d);

}  // namespace ascend_kernel

#endif  // MAMBA3_DECODE_STEP_VALIDATE_H
