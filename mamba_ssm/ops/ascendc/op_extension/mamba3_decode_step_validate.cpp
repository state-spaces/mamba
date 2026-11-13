/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// PLUT-189 D3：mamba3_decode_step（路线 A / 纯 Ascend C）参数合法性校验（host 侧）
//
// 任务书 §2.4 / §2.5 要求：dtype 分组一致性、各张量秩与形状、numel==0、tile_D 取值、
// D 与 tile_D 的整除关系、R>=1、N 为 8 的倍数、state_out 与 state 的部分重叠，
// **全部在 host 侧校验并返回参数错误**。
//
// 本文件把这套校验从 Python 逐项搬到 C++，目的是把每次调用的 16 个张量 × 5~6 项
// Python 层属性访问换成一次 C++ 调用（判定集合与顺序逐条不变）。
// 它**不是**校验缓存、也不是抽样、更不是异步：
//   * 每个参数、每一项判据每次调用都真跑（见 ValidationRc 逐项返回码）；
//   * 拒绝时机不变（仍在 host 侧、下发 kernel 之前，本文件不碰 kernel 与流）。
// 异常文案与类型由 Python 入口在「拒绝」时重放原逐项实现取得（逐字一致），
// 通过路径不再跑 Python 逐项校验（否则校验跑两遍）。
//
// 与 CATLASS 侧（PLUT-181）的口径差异只有「本路线原本就没有的那条判据」：
// 路线 A 的 Python 入口没有 D % 8 这一条（它只要求 N % 8），因此这里也不加。
// ============================================================================

#include "mamba3_decode_step_validate.h"

#include <utility>

#include "ops.h"

namespace ascend_kernel {
namespace {

// 逐项返回码：0 = 通过；其余 = 第一次触发拒绝的判据（仅用于回溯，不参与判定口径）。
// 与 mamba_ssm/ops/ascendc/mamba3_step.py 原逐项实现的判定顺序一一对应，便于审核逐条对拍。
enum ValidationRc : int64_t {
    kRcOk = 0,
    kRcState = 1,                   // state 秩 4 / numel==0 / dtype / device
    kRcBstateRank = 2,              // Bstate 秩 < 2（原实现在此读 shape[1]）
    kRcOutMissing = 3,              // out 未预分配（Python 入口已先判，此处防御）
    kRcOutShapeNoOutproj = 4,       // outproj 为 None 时 out 必须 [B,R,H,D]
    kRcOutShapeWithOutproj = 5,     // outproj 非空时 out 必须 [B,H,D]
    kRcTileDivides = 6,             // D 必须能被 tile_D 整除
    kRcRankPositive = 7,            // R >= 1
    kRcStateAligned = 8,            // N 必须为 8 的倍数
    kRcBstate = 9,                  // Bstate 秩/形状/numel/dtype/device
    kRcXstate = 10,
    kRcA = 11,
    kRcB = 12,
    kRcC = 13,
    kRcD = 14,
    kRcX = 15,
    kRcDt = 16,
    kRcTrap = 17,
    kRcXproj = 18,
    kRcOutproj = 19,
    kRcOut = 20,
    kRcZ = 21,
    kRcZproj = 22,
    kRcStateOut = 23,
    kRcStateOutOverlap = 24,        // state_out 与 state 部分重叠
    kRcStateOutDtype = 25,          // state_out 与 state 同址时 dtype 必须一致
    kRcDtypeState = 26,             // state/state_out 组内 dtype 不一致
    kRcDtypeBC = 27,                // B/Bstate/C 组内 dtype 不一致
    kRcDtypeXOut = 28,              // x/Xstate/z/out 组内 dtype 不一致
    kRcDtypeProj = 29,              // xproj/outproj/zproj 组内 dtype 不一致
    kRcStateIn = 30,                // state 末维物化（输入方向，不会因此拒绝）
    kRcStateOutStride = 31,         // state_out 末维 stride 必须为 1
    kRcOutStride = 32,              // out 末维 stride 必须为 1
    kRcStateInplaceNonContig = 33,  // state 末维非连续且与 state_out 同址
    kRcDtypeScalar = 34,            // A/dt/trap/D 组内 dtype 不一致（tiling 里共用 scalar 编码）
};

bool SupportedDtype(const at::Tensor &t)
{
    const auto st = t.scalar_type();
    return st == at::kHalf || st == at::kBFloat16 || st == at::kFloat;
}

bool OnNpu(const at::Tensor &t)
{
    return t.device().is_privateuseone();
}

// 与 Python ``_check_tensor`` 同义：秩、numel、dtype、device（形状由带形状的重载再判）。
bool OkTensor(const at::Tensor &t, int64_t rank)
{
    if (t.dim() != rank) {
        return false;
    }
    if (t.numel() == 0) {
        return false;
    }
    if (!SupportedDtype(t)) {
        return false;
    }
    return OnNpu(t);
}

bool OkTensor(const at::Tensor &t, int64_t rank, std::initializer_list<int64_t> shape)
{
    if (!OkTensor(t, rank)) {
        return false;
    }
    return t.sizes().equals(at::IntArrayRef(shape));
}

// 与 Python ``_storage_range`` 同义：整块 storage 的 [base, base+nbytes)。
std::pair<uintptr_t, uintptr_t> StorageRange(const at::Tensor &t)
{
    const c10::Storage &storage = t.storage();
    const auto base = reinterpret_cast<uintptr_t>(storage.data_ptr().get());
    return {base, base + static_cast<uintptr_t>(storage.nbytes())};
}

// 与 Python ``_check_alias`` 同义：同址 / 独立 / 部分重叠（部分重叠由调用方拒绝）。
enum AliasKind { kAliasSame = 0, kAliasIndependent = 1, kAliasPartial = 2 };

AliasKind CheckAlias(const at::Tensor &outTensor, const at::Tensor &inTensor)
{
    if (outTensor.data_ptr() == inTensor.data_ptr()) {
        return kAliasSame;
    }
    const auto o = StorageRange(outTensor);
    const auto i = StorageRange(inTensor);
    if (o.first < i.second && i.first < o.second) {
        return kAliasPartial;
    }
    return kAliasIndependent;
}

bool SameDtype(std::initializer_list<const at::Tensor *> tensors)
{
    const at::Tensor *first = nullptr;
    for (const at::Tensor *t : tensors) {
        if (t == nullptr) {
            continue;
        }
        if (first == nullptr) {
            first = t;
            continue;
        }
        if (t->scalar_type() != first->scalar_type()) {
            return false;
        }
    }
    return true;
}

// 与 Python ``_as_innermost_contiguous`` 同义；输出张量末维非连续时返回 false（调用方拒绝）。
bool InnermostContiguous(const at::Tensor &t, at::Tensor &out, bool isOutput)
{
    if (t.stride(-1) == 1) {
        out = t;
        return true;
    }
    if (isOutput) {
        return false;
    }
    out = t.contiguous();
    return true;
}

}  // namespace

std::tuple<int64_t, c10::optional<std::vector<c10::optional<at::Tensor>>>> Mamba3DecodeStepHostValidator::check(
    at::Tensor state, at::Tensor bstate, at::Tensor xstate, at::Tensor a, at::Tensor b, at::Tensor c, at::Tensor d,
    at::Tensor x, at::Tensor dt, at::Tensor trap, at::Tensor xproj, c10::optional<at::Tensor> outproj,
    c10::optional<at::Tensor> state_out, c10::optional<at::Tensor> out, c10::optional<at::Tensor> z,
    c10::optional<at::Tensor> zproj, int64_t tile_d) const
{
    using Rc = int64_t;
    const auto fail = [](Rc rc) { return ValidationResult{rc, c10::nullopt}; };

    // ---- 1) state：§2.4 state 行（秩 4 / numel!=0 / dtype / device） ----
    if (!OkTensor(state, 4)) {
        return fail(kRcState);
    }
    const int64_t batch = state.size(0);
    const int64_t heads = state.size(1);
    const int64_t headDim = state.size(2);
    const int64_t dState = state.size(3);
    // ---- 2) 原实现在这里读 Bstate.shape[1]（秩 < 2 时抛 IndexError，由 Python 重放复现） ----
    if (bstate.dim() < 2) {
        return fail(kRcBstateRank);
    }
    const int64_t rank = bstate.size(1);

    // ---- 3) out 形状分支必须与 outproj 是否为空一致（§2.4 out 行） ----
    if (!out.has_value()) {
        return fail(kRcOutMissing);
    }
    const bool hasOutproj = outproj.has_value();
    if (!hasOutproj) {
        if (!out->sizes().equals(at::IntArrayRef({batch, rank, heads, headDim}))) {
            return fail(kRcOutShapeNoOutproj);
        }
    } else {
        if (!out->sizes().equals(at::IntArrayRef({batch, heads, headDim}))) {
            return fail(kRcOutShapeWithOutproj);
        }
    }

    // ---- 4) tile_D 整除 / R>=1 / N 的 8 对齐（§2.4 tile_D 行 + 实现前置约束） ----
    if (headDim % tile_d != 0) {
        return fail(kRcTileDivides);
    }
    if (rank < 1) {
        return fail(kRcRankPositive);
    }
    if (dState % 8 != 0) {
        return fail(kRcStateAligned);
    }

    // ---- 5) 其余张量的秩与形状（§2.4 逐行，顺序与原实现一致） ----
    if (!OkTensor(bstate, 4, {batch, rank, heads, dState})) {
        return fail(kRcBstate);
    }
    if (!OkTensor(xstate, 3, {batch, heads, headDim})) {
        return fail(kRcXstate);
    }
    if (!OkTensor(a, 2, {batch, heads})) {
        return fail(kRcA);
    }
    if (!OkTensor(b, 4, {batch, rank, heads, dState})) {
        return fail(kRcB);
    }
    if (!OkTensor(c, 4, {batch, rank, heads, dState})) {
        return fail(kRcC);
    }
    if (!OkTensor(d, 1, {heads})) {
        return fail(kRcD);
    }
    if (!OkTensor(x, 3, {batch, heads, headDim})) {
        return fail(kRcX);
    }
    if (!OkTensor(dt, 2, {batch, heads})) {
        return fail(kRcDt);
    }
    if (!OkTensor(trap, 2, {batch, heads})) {
        return fail(kRcTrap);
    }
    if (!OkTensor(xproj, 3, {rank, heads, headDim})) {
        return fail(kRcXproj);
    }
    if (hasOutproj && !OkTensor(*outproj, 3, {rank, heads, headDim})) {
        return fail(kRcOutproj);
    }
    const int64_t outRank = hasOutproj ? 3 : 4;
    if (!OkTensor(*out, outRank)) {
        return fail(kRcOut);
    }
    if (z.has_value() && !OkTensor(*z, 3, {batch, heads, headDim})) {
        return fail(kRcZ);
    }
    // 路线 A 的 Python 入口先判「z 与 zproj 必须同时存在或同时为 None」，因此到这里
    // zproj 非空 <=> z 非空；仍按 zproj 自己那一行校验形状/dtype（与 Python 同规则）。
    if (zproj.has_value() && !OkTensor(*zproj, 3, {rank, heads, headDim})) {
        return fail(kRcZproj);
    }

    // ---- 6) state_out：缺省即 state（原地）；显式传入时判形状 / 部分重叠 / dtype ----
    bool inplace = false;
    if (!state_out.has_value()) {
        inplace = true;  // 原实现：state_out = state
    } else {
        if (!OkTensor(*state_out, 4, {batch, heads, headDim, dState})) {
            return fail(kRcStateOut);
        }
        const AliasKind alias = CheckAlias(*state_out, state);
        if (alias == kAliasPartial) {
            return fail(kRcStateOutOverlap);
        }
        inplace = (alias == kAliasSame);
    }
    // 原实现：inplace 且 state_out.dtype != state.dtype -> 参数错误；
    // state_out 缺省时它就是 state 本身，dtype 必然一致，所以只在显式传入时可能失败。
    if (inplace && state_out.has_value() && state_out->scalar_type() != state.scalar_type()) {
        return fail(kRcStateOutDtype);
    }

    // ---- 7) dtype 分组（§2.5：五组，组内一致、组间独立） ----
    const at::Tensor *stateOutPtr = state_out.has_value() ? &(*state_out) : &state;
    if (!SameDtype({&state, stateOutPtr})) {
        return fail(kRcDtypeState);
    }
    if (!SameDtype({&b, &bstate, &c})) {
        return fail(kRcDtypeBC);
    }
    if (!SameDtype({&x, &xstate, z.has_value() ? &(*z) : nullptr, &(*out)})) {
        return fail(kRcDtypeXOut);
    }
    if (!SameDtype({&xproj, hasOutproj ? &(*outproj) : nullptr, zproj.has_value() ? &(*zproj) : nullptr})) {
        return fail(kRcDtypeProj);
    }
    // A / dt / trap / D 在 tiling 里共用同一个 dtype 编码（host 侧取自 A，见
    // mamba3_decode_step_torch.cpp 的 dtypeScalar），kernel 按该宽度读四个 buffer：组内不一致时
    // 其中一方会被按另一方的元素宽度读成垃圾值（状态相对 L2 误差 6e-1 量级，且不抛异常），
    // 所以必须在这里显式拒绝，而不是放行到 kernel。
    if (!SameDtype({&a, &dt, &trap, &d})) {
        return fail(kRcDtypeScalar);
    }

    // ---- 8) 末维连续（与原实现同一顺序：state -> state_out -> 其余输入 -> out） ----
    at::Tensor stateIn;
    if (!InnermostContiguous(state, stateIn, false)) {
        return fail(kRcStateIn);  // 输入方向不会走到（isOutput=false）
    }
    // state_out 缺省时按原实现等价于「state_out = state 后仍走输出方向判定」：
    // state 末维非连续 = 参数错误（不是静默就地写回）。
    at::Tensor stateOutT;
    if (!InnermostContiguous(state_out.has_value() ? *state_out : state, stateOutT, true)) {
        return fail(kRcStateOutStride);
    }
    at::Tensor bstateT, xstateT, aT, bT, cT, dT, xT, dtT, trapT, xprojT;
    if (!InnermostContiguous(bstate, bstateT, false) || !InnermostContiguous(xstate, xstateT, false) ||
        !InnermostContiguous(a, aT, false) || !InnermostContiguous(b, bT, false) ||
        !InnermostContiguous(c, cT, false) || !InnermostContiguous(d, dT, false) ||
        !InnermostContiguous(x, xT, false) || !InnermostContiguous(dt, dtT, false) ||
        !InnermostContiguous(trap, trapT, false) || !InnermostContiguous(xproj, xprojT, false)) {
        return fail(kRcBstate);  // 输入方向不会走到（isOutput=false）
    }
    c10::optional<at::Tensor> outprojT;
    if (hasOutproj) {
        at::Tensor tmp;
        if (!InnermostContiguous(*outproj, tmp, false)) {
            return fail(kRcOutproj);
        }
        outprojT = tmp;
    }
    c10::optional<at::Tensor> zT, zprojT;
    if (z.has_value()) {
        at::Tensor zTmp, zprojTmp;
        if (!InnermostContiguous(*z, zTmp, false)) {
            return fail(kRcZ);
        }
        if (!InnermostContiguous(*zproj, zprojTmp, false)) {
            return fail(kRcZproj);
        }
        zT = zTmp;
        zprojT = zprojTmp;
    }
    at::Tensor outT;
    if (!InnermostContiguous(*out, outT, true)) {
        return fail(kRcOutStride);
    }

    // ---- 9) state 末维非连续且与 state_out 同址 -> 无法就地更新 ----
    if (inplace && stateIn.data_ptr() != stateOutT.data_ptr()) {
        return fail(kRcStateInplaceNonContig);
    }

    // 无物化（绝大多数调用）就不回传张量：pybind11 每包一个 at::Tensor 约 1.5 µs，
    // 回传 16 个 = 24 µs，而它们与调用方手里的实参是同一批对象。
    const bool materialized =
        !stateIn.is_same(state) || !bstateT.is_same(bstate) || !xstateT.is_same(xstate) || !aT.is_same(a) ||
        !bT.is_same(b) || !cT.is_same(c) || !dT.is_same(d) || !xT.is_same(x) || !dtT.is_same(dt) ||
        !trapT.is_same(trap) || !xprojT.is_same(xproj) || (hasOutproj && !outprojT->is_same(*outproj)) ||
        (z.has_value() && (!zT->is_same(*z) || !zprojT->is_same(*zproj)));
    if (!materialized) {
        return ValidationResult{kRcOk, c10::nullopt};
    }

    std::vector<c10::optional<at::Tensor>> resolved(16);
    resolved[0] = stateIn;
    resolved[1] = bstateT;
    resolved[2] = xstateT;
    resolved[3] = aT;
    resolved[4] = bT;
    resolved[5] = cT;
    resolved[6] = dT;
    resolved[7] = xT;
    resolved[8] = dtT;
    resolved[9] = trapT;
    resolved[10] = xprojT;
    resolved[11] = outprojT;
    resolved[12] = zT;
    resolved[13] = zprojT;
    resolved[14] = stateOutT;
    resolved[15] = outT;
    return ValidationResult{kRcOk, std::move(resolved)};
}

ValidationResult ValidateArgsEntry(at::Tensor state, at::Tensor bstate, at::Tensor xstate, at::Tensor a, at::Tensor b,
                                  at::Tensor c, at::Tensor d, at::Tensor x, at::Tensor dt, at::Tensor trap,
                                  at::Tensor xproj, c10::optional<at::Tensor> outproj,
                                  c10::optional<at::Tensor> state_out, c10::optional<at::Tensor> out,
                                  c10::optional<at::Tensor> z, c10::optional<at::Tensor> zproj, int64_t tile_d)
{
    return Mamba3DecodeStepHostValidator{}.check(state, bstate, xstate, a, b, c, d, x, dt, trap, xproj, outproj,
                                                state_out, out, z, zproj, tile_d);
}

// ---------------------------------------------------------------------------
// PLUT-215 单跨边界入口：同一次 pybind11 调用内完成「全部张量侧判据 + 下发」。
//
// 与旧两步路径（``validate`` 一次 + ``torch.ops.npu.mamba3_decode_step`` 一次）的差别只在
// 跨边界次数：判据集合、判定顺序、拒绝时机、下发实参、kernel 语义逐条不变。
//   * 通过：校验通过后按 check() 的「物化决议」直接下发，返回 0；
//   * 拒绝：返回 check() 的逐项返回码，**不下发**，由 Python 入口重放原逐项实现取原文案；
//   * 非 torch.Tensor 实参：pybind11 抛 TypeError，Python 入口同样交回原逐项实现。
// 旧 ``validate`` 绑定保留不动（离线对拍与旧 .so 兼容用），本函数不缓存任何判定结果。
// ---------------------------------------------------------------------------
int64_t StepEntry(at::Tensor state, at::Tensor bstate, at::Tensor xstate, at::Tensor a, at::Tensor b, at::Tensor c,
                  at::Tensor d, at::Tensor x, at::Tensor dt, at::Tensor trap, at::Tensor xproj,
                  c10::optional<at::Tensor> outproj, c10::optional<at::Tensor> state_out, c10::optional<at::Tensor> out,
                  c10::optional<at::Tensor> z, c10::optional<at::Tensor> zproj, int64_t tile_d)
{
    ValidationResult res = Mamba3DecodeStepHostValidator{}.check(state, bstate, xstate, a, b, c, d, x, dt, trap, xproj,
                                                               outproj, state_out, out, z, zproj, tile_d);
    const int64_t rc = std::get<0>(res);
    if (rc != 0) {
        return rc;  // 拒绝路径：不下发，异常类型/文案由 Python 侧重放原实现取得
    }
    const c10::optional<std::vector<c10::optional<at::Tensor>>> &resolved = std::get<1>(res);
    if (!resolved.has_value()) {
        // 绝大多数调用：无输入需要物化，实参就是调用方给的那批对象
        mamba3_decode_step_torch(state, bstate, xstate, a, b, c, d, x, dt, trap, xproj, outproj, z, zproj,
                                 state_out.has_value() ? *state_out : state, *out, tile_d);
        return 0;
    }
    const std::vector<c10::optional<at::Tensor>> &v = *resolved;  // 槽位见 check() 末尾的 resolved[0..15]
    mamba3_decode_step_torch(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7], *v[8], *v[9], *v[10],
                             v[11], v[12], v[13], *v[14], *v[15], tile_d);
    return 0;
}

}  // namespace ascend_kernel

// ---------------------------------------------------------------------------
// pybind11 入口：Python 侧用 importlib 直接加载同一个 .so 并取 ``validate``。
// 不走 torch 自定义类：ScriptMethod 会把实参逐个装箱成 IValue 再按 schema 反解
// （CATLASS 侧 PLUT-181 实测 50.9 µs/call，比它替代掉的 Python 逐项校验还贵）。
// ---------------------------------------------------------------------------
PYBIND11_MODULE(mamba3_decode_step_ext, m)
{
    m.def("validate", &ascend_kernel::ValidateArgsEntry, "mamba3_decode_step 参数合法性校验（每次调用全量执行）");
    m.def("step", &ascend_kernel::StepEntry,
          "mamba3_decode_step 单跨边界入口：全量校验 + 下发（每次调用都跑全部判据）");
    m.def("launch_mode", &ascend_kernel::SelfLaunchMode,
          "当前线程的下发形态：1 = 自管 aclrtLaunchKernelWithHostArgs，0 = 生成桩（只读诊断）");
}
