/**
 * selective_scan_fwd - torch.library 注册 + Meta + NPU impl（g++ 编译）
 * schema 契约来源：需求文档 1.1-需求分析.md 2.2 节（模式 B 自定契约，对齐 CUDA pybind fwd）
 * 校验清单：2.2-开发方案设计.md 第 7 节（含 CP2.2 增强校验 S-E-15/16，先于 tiling 计算）
 */
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cstddef>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../op_kernel/selective_scan_launch.h"

namespace cann_bench {

// ---------- schema 注册（逐字对齐需求文档 2.2 接口原型） ----------
TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("selective_scan_fwd(Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C, "
          "Tensor? D=None, Tensor? z=None, Tensor? delta_bias=None, "
          "bool delta_softplus=False) -> (Tensor out, Tensor x, Tensor out_z)");
}

// ---------- 内部工具：形状/dtype 校验 ----------
namespace {

constexpr int64_t SSF_CHUNK_T = 2048;   // x 契约 chunk 宽度（与 kernel 侧处理单元 TILE_T=1024 解耦：chunk=2×tile，R6 绕行）
constexpr int64_t SSF_MAX_DSTATE = 256; // 方案 4 节：dstate ∈ [1,256]

// 记录一次调用的解析结果，供 meta/impl 复用
struct SsfShapeInfo {
    int64_t batch = 0;
    int64_t dim = 0;
    int64_t seqlen = 0;
    int64_t dstate = 0;
    int64_t nGroups = 1;     // B/C 分组数（静态 B/C 时 = dim）
    int64_t dimPerGroup = 1; // dim / nGroups
    int64_t nChunks = 0;
    bool hasD = false;
    bool hasZ = false;
    bool hasBias = false;
    bool softplus = false;
    bool varB = false;
    bool varC = false;
};

[[noreturn]] void ssf_error(const std::string &msg)
{
    TORCH_CHECK(false, "selective_scan_fwd: ", msg);
}

void ssf_check_contiguous(const torch::Tensor &t, const char *name)
{
    TORCH_CHECK(t.is_contiguous(), "selective_scan_fwd: ", name, " must be contiguous");
}

torch::ScalarType ssf_input_dtype(const torch::Tensor &u, const torch::Tensor &delta,
                                  const torch::Tensor &z, const torch::Tensor &B,
                                  const torch::Tensor &C, bool varB, bool varC)
{
    auto dt = u.scalar_type();
    TORCH_CHECK(dt == torch::kFloat32 || dt == torch::kFloat16 || dt == torch::kBFloat16,
                "selective_scan_fwd: u dtype must be float32/float16/bfloat16, got ", dt);
    TORCH_CHECK(delta.scalar_type() == dt, "selective_scan_fwd: delta dtype must match u");
    if (z.defined()) {
        TORCH_CHECK(z.scalar_type() == dt, "selective_scan_fwd: z dtype must match u");
    }
    if (varB) {
        TORCH_CHECK(B.scalar_type() == dt, "selective_scan_fwd: variable B dtype must match u");
    }
    if (varC) {
        TORCH_CHECK(C.scalar_type() == dt, "selective_scan_fwd: variable C dtype must match u");
    }
    return dt;
}

// 解析并校验全部输入；校验顺序按方案第 7 节（S-E-15/16 最先、先于 tiling）
SsfShapeInfo ssf_parse(const torch::Tensor &u, const torch::Tensor &delta, const torch::Tensor &A,
                       const torch::Tensor &B, const torch::Tensor &C, const torch::Tensor &D,
                       const torch::Tensor &z, const torch::Tensor &delta_bias, bool delta_softplus)
{
    SsfShapeInfo info{};

    // ---- 形状类校验最前：S-E-15 / S-E-16（CP2.2 增强校验，先于 tiling 拦截除零） ----
    TORCH_CHECK(u.dim() == 3, "selective_scan_fwd: u must be [batch, dim, seqlen]");
    ssf_check_contiguous(u, "u");
    info.batch = u.size(0);
    info.dim = u.size(1);
    info.seqlen = u.size(2);
    if (info.batch == 0) {
        ssf_error("batch must be positive, got 0");
    }
    if (info.dim == 0) {
        ssf_error("dim must be positive, got 0");
    }
    // seqlen == 0 合法（2.1 §4.2 S-B-06：空 tensor，out (b,d,0) / x (b,d,0,2·dstate)）

    // ---- delta ----
    TORCH_CHECK(delta.dim() == 3 && delta.size(0) == info.batch && delta.size(1) == info.dim &&
                    delta.size(2) == info.seqlen,
                "selective_scan_fwd: delta must have shape [batch, dim, seqlen] same as u");
    ssf_check_contiguous(delta, "delta");

    // ---- A: [dim, dstate] fp32 ----
    TORCH_CHECK(A.dim() == 2 && A.size(0) == info.dim,
                "selective_scan_fwd: A must have shape [dim, dstate]");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "selective_scan_fwd: A dtype must be float32");
    ssf_check_contiguous(A, "A");
    info.dstate = A.size(1);
    TORCH_CHECK(info.dstate >= 1 && info.dstate <= SSF_MAX_DSTATE,
                "selective_scan_fwd: dstate must be in [1, 256], got ", info.dstate);

    // ---- B/C: 静态 [dim, dstate] fp32 或变量 [batch, n_groups, dstate, seqlen] input_t ----
    //     B/C 变体互相独立（2.1 §3：四组合均合法）
    TORCH_CHECK(B.dim() == 2 || B.dim() == 4,
                "selective_scan_fwd: B must be [dim, dstate] (static) or "
                "[batch, n_groups, dstate, seqlen] (variable), got ndim=",
                B.dim());
    TORCH_CHECK(C.dim() == 2 || C.dim() == 4,
                "selective_scan_fwd: C must be [dim, dstate] (static) or "
                "[batch, n_groups, dstate, seqlen] (variable), got ndim=",
                C.dim());
    info.varB = (B.dim() == 4);
    info.varC = (C.dim() == 4);
    if (info.varB) {
        info.nGroups = B.size(1);
        TORCH_CHECK(B.size(0) == info.batch && B.size(2) == info.dstate && B.size(3) == info.seqlen,
                    "selective_scan_fwd: variable B must have shape [batch, n_groups, dstate, seqlen]");
    } else {
        info.nGroups = info.dim;   // 静态 B 按 dim 逐通道，无分组
        TORCH_CHECK(B.size(0) == info.dim && B.size(1) == info.dstate,
                    "selective_scan_fwd: static B must have shape [dim, dstate]");
        TORCH_CHECK(B.scalar_type() == torch::kFloat32, "selective_scan_fwd: static B dtype must be float32");
    }
    if (info.varC) {
        if (info.varB) {
            // 双变量：核内共享一份 dimPerGroup 行索引，要求分组数一致
            TORCH_CHECK(C.size(1) == info.nGroups,
                        "selective_scan_fwd: variable C n_groups (", C.size(1),
                        ") must match variable B n_groups (", info.nGroups, ")");
        } else {
            info.nGroups = C.size(1);
        }
        TORCH_CHECK(C.size(0) == info.batch && C.size(2) == info.dstate && C.size(3) == info.seqlen,
                    "selective_scan_fwd: variable C must have shape [batch, n_groups, dstate, seqlen]");
    } else {
        TORCH_CHECK(C.size(0) == info.dim && C.size(1) == info.dstate,
                    "selective_scan_fwd: static C must have shape [dim, dstate]");
        TORCH_CHECK(C.scalar_type() == torch::kFloat32, "selective_scan_fwd: static C dtype must be float32");
    }
    TORCH_CHECK(info.dim % info.nGroups == 0,
                "selective_scan_fwd: dim must be divisible by n_groups (dim=", info.dim,
                ", n_groups=", info.nGroups, ")");
    info.dimPerGroup = info.dim / info.nGroups;
    ssf_check_contiguous(B, "B");
    ssf_check_contiguous(C, "C");

    // ---- D / delta_bias: [dim] fp32 ----
    info.hasD = D.defined();
    if (info.hasD) {
        TORCH_CHECK(D.dim() == 1 && D.size(0) == info.dim,
                    "selective_scan_fwd: D must have shape [dim]");
        TORCH_CHECK(D.scalar_type() == torch::kFloat32, "selective_scan_fwd: D dtype must be float32");
        ssf_check_contiguous(D, "D");
    }
    info.hasBias = delta_bias.defined();
    if (info.hasBias) {
        TORCH_CHECK(delta_bias.dim() == 1 && delta_bias.size(0) == info.dim,
                    "selective_scan_fwd: delta_bias must have shape [dim]");
        TORCH_CHECK(delta_bias.scalar_type() == torch::kFloat32,
                    "selective_scan_fwd: delta_bias dtype must be float32");
        ssf_check_contiguous(delta_bias, "delta_bias");
    }

    // ---- z: [batch, dim, seqlen] ----
    info.hasZ = z.defined();
    if (info.hasZ) {
        TORCH_CHECK(z.dim() == 3 && z.size(0) == info.batch && z.size(1) == info.dim &&
                        z.size(2) == info.seqlen,
                    "selective_scan_fwd: z must have shape [batch, dim, seqlen]");
        ssf_check_contiguous(z, "z");
    }
    info.softplus = delta_softplus;

    // ---- dtype 分派校验（u 为基准，变量 B/C/z 同 u） ----
    ssf_input_dtype(u, delta, z, B, C, info.varB, info.varC);

    info.nChunks = (info.seqlen + SSF_CHUNK_T - 1) / SSF_CHUNK_T;
    return info;
}

uint32_t ssf_flags(const SsfShapeInfo &info)
{
    uint32_t flags = 0;
    if (info.hasD)     flags |= SSF_FLAG_HAS_D;
    if (info.hasZ)     flags |= SSF_FLAG_HAS_Z;
    if (info.hasBias)  flags |= SSF_FLAG_HAS_BIAS;
    if (info.softplus) flags |= SSF_FLAG_SOFTPLUS;
    if (info.varB)     flags |= SSF_FLAG_VAR_B;
    if (info.varC)     flags |= SSF_FLAG_VAR_C;
    return flags;
}

// kernel 侧 InitBuffer 的 UB 需求镜像（字节）：与 selective_scan_kernel.cpp Init 的分配清单逐项一致
// （R6 修订：TILE_T=1024/PAD=512；3.4 HANG-01 修复后 z 弃 TQue 改 zStageBuf_ TBuf 单缓冲）
// 逐项算式（TILE_T=1024，dts=sizeof(INPUT_T)，inBytes=TILE_T·dts，DS=dstate 静态上界 256）：
//   inQu_/inQd_        : 2 队列 × 深度 2 × inBytes = 4·inBytes
//   u32/d32/du32/outAcc/tmpA : 5 × TILE_T × 4（恒 fp32 TBuf）
//   a0/a1/b0/b1        : 4 × (TILE_T+PAD) × 4（含左填充，PAD=512）
//   gatherIdx          : TILE_T × 4
//   xBuild             : 2 × DS × 4（静态上限）
//   aRow               : DS × 4（静态上限）
//   sTmp               : 64
//   hasBias → biasRow  : dim × 4（dav-2201 绕行：bias 每 pair 4B 单元素 DataCopyPad 触发
//                        MTE 时序缺陷，改每核一次整行块搬运；dim 为运行时值）
//   hasZ  → zStageBuf_ : 1 × inBytes（TBuf 单缓冲；无 inQz_ 队列）
//   varB  → inQvarB_(深度 2, 2·inBytes) + varB32(TILE_T×4)；否则 bRow = DS×4
//   varC  → inQvarC_(深度 2, 2·inBytes) + varC32(TILE_T×4)；否则 cRow = DS×4
// fp32 全特性（hasZ+varB+varC）上界：
//   4·4096 + 5·4096 + 4·6144 + 4096 + 2·256·4 + 1024 + 64 + 4096 + (2·4096+4096)×2
//   = 16384 + 20480 + 24576 + 4096 + 2048 + 1024 + 64 + 4096 + 12288 + 12288 = 97344 B ≪ 192KB
int64_t ssf_ub_require_bytes(const SsfShapeInfo &info, at::ScalarType dtype)
{
    constexpr int64_t T = 1024;    // 处理单元 TILE_T（kernel 侧固定常量）
    constexpr int64_t PAD = T / 2; // 扫描缓冲左填充
    const int64_t dts = (dtype == at::kFloat) ? 4 : 2;
    const int64_t inBytes = T * dts;
    int64_t need = 0;
    need += 4 * inBytes;                      // inQu_/inQd_：2 队列 × 深度 2 = 4×inBytes
    need += 5 * T * 4;                        // u32/d32/du32/outAcc/tmpA（恒 fp32）
    need += 4 * (T + PAD) * 4;                // a0/a1/b0/b1（含左填充）
    need += T * 4;                            // gatherIdx
    need += 2 * SSF_MAX_DSTATE * 4;           // xBuild（静态上限，dstate ≤ 256）
    need += SSF_MAX_DSTATE * 4;               // aRow（静态上限）
    need += 64;                               // sTmp
    if (info.hasBias) {
        need += info.dim * 4;                 // biasRow（问题 6 绕行：bias 每核一次整行块搬运，dim 为运行时值）
    }
    if (info.hasZ) {
        need += inBytes;                      // zStageBuf_（TBuf 单缓冲，HANG-01 修复后弃 TQue）
    }
    if (info.varB) {
        need += 2 * inBytes + T * 4;          // inQvarB_（深度 2）+ varB32
    } else {
        need += SSF_MAX_DSTATE * 4;           // bRow（静态上限）
    }
    if (info.varC) {
        need += 2 * inBytes + T * 4;          // inQvarC_（深度 2）+ varC32
    } else {
        need += SSF_MAX_DSTATE * 4;           // cRow（静态上限）
    }
    return need;
}

// B1 红线：UB 容量运行时查询准入校验（不硬编码容量）
void ssf_check_ub_capacity(const SsfShapeInfo &info, at::ScalarType dtype)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const int64_t need = ssf_ub_require_bytes(info, dtype);
    TORCH_CHECK(need <= static_cast<int64_t>(ubSize),
                "selective_scan_fwd: UB requirement ", need, " bytes (dtype=", dtype,
                ", dstate=", info.dstate, ", flags shape variant) exceeds device UB ", ubSize, " bytes");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ssf_meta_impl(
    const torch::Tensor &u, int64_t nChunks, int64_t dstate, bool hasZ)
{
    auto inputOptions = u.options();
    auto out = torch::empty_like(u);
    // x: [batch, dim, n_chunks, 2*dstate] fp32
    auto x = torch::empty({u.size(0), u.size(1), nChunks, 2 * dstate},
                          inputOptions.dtype(torch::kFloat32));
    // z=None 时 out_z 返回 shape {0} 空 tensor（2.2 §2.1 已决策口径）
    torch::Tensor outZ = hasZ ? torch::empty_like(u) : torch::empty({0}, inputOptions);
    return std::make_tuple(out, x, outZ);
}

void ssf_launch(const torch::Tensor &u, const torch::Tensor &delta, const torch::Tensor &A,
                const torch::Tensor &B, const torch::Tensor &C, const torch::Tensor &D,
                const torch::Tensor &z, const torch::Tensor &delta_bias,
                torch::Tensor &out, torch::Tensor &x, torch::Tensor &outZ,
                const SsfShapeInfo &info)
{
    int64_t numPairs = info.batch * info.dim;
    // seqlen=0（nChunks=0）：合法空输出，跳过 kernel launch（meta 已产出空 out/x/out_z）
    if (info.seqlen == 0 || info.nChunks == 0) {
        return;
    }
    int64_t usedCores = 0;
    int64_t pairsPerCore = 0;
    std::tie(usedCores, pairsPerCore) = calc_selective_scan_tiling_params(info.batch, info.dim);
    uint32_t flags = ssf_flags(info);
    const uint32_t numBlocks =
        static_cast<uint32_t>((numPairs + pairsPerCore - 1) / pairsPerCore);

    // dav-2201 绕行（问题 6，方案 V）：delta_bias 的 GM 原张量不可被本 kernel 以任何引擎
    // 访问（MTE2 DataCopyPad / 标量 GetValue 均触发 aivec 507035，地址组合依赖、与数据/
    // 指令流无关，详见 doc/mamba仓NPU跑测问题与解决方案.md 问题 6）；V 引擎读原地址实测
    // 安全。plugin 以纯 V 算子 mul 1.0 生成安全副本 bias_safe（读原地址、写新分配地址），
    // kernel 读 bias_safe，原 bias 张量对 kernel 完全不可见。
    // bias_safe 为 launch 期首个分配（先于 tiling 缓冲）：优先复用 bias 上游临时量的
    // 释放块，落点与各场景 bias 的历史安全区一致
    torch::Tensor biasSafe;
    if (info.hasBias) {
        biasSafe = delta_bias * 1.0f;  // V 引擎 mul：读原 bias 地址（安全），落新分配地址
    }

    // 标量参数打包为 GM tiling 缓冲（直调 launch 链路对中段标量参数有损坏，见 launch.h）
    SsfTilingData td{};
    td.batch = info.batch;
    td.dim = info.dim;
    td.seqlen = info.seqlen;
    td.dstate = info.dstate;
    td.dimPerGroup = info.dimPerGroup;
    td.nChunks = info.nChunks;
    td.numPairs = numPairs;
    td.pairsPerCore = pairsPerCore;
    td.flags = flags;
    torch::Tensor tilingTensor = torch::empty(
        {static_cast<int64_t>(sizeof(SsfTilingData))}, u.options().dtype(torch::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(SsfTilingData),
                &td, sizeof(SsfTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    auto uPtr = (GM_ADDR)u.data_ptr();
    auto deltaPtr = (GM_ADDR)delta.data_ptr();
    auto aPtr = (GM_ADDR)A.data_ptr();
    auto bPtr = (GM_ADDR)B.data_ptr();
    auto cPtr = (GM_ADDR)C.data_ptr();
    auto dPtr = info.hasD ? (GM_ADDR)D.data_ptr() : (GM_ADDR)nullptr;
    auto zPtr = info.hasZ ? (GM_ADDR)z.data_ptr() : (GM_ADDR)nullptr;
    auto biasPtr = info.hasBias ? (GM_ADDR)biasSafe.data_ptr() : (GM_ADDR)nullptr;
    auto outPtr = (GM_ADDR)out.data_ptr();
    auto xPtr = (GM_ADDR)x.data_ptr();
    auto outZPtr = info.hasZ ? (GM_ADDR)outZ.data_ptr() : (GM_ADDR)nullptr;
    auto tilingPtr = (GM_ADDR)tilingTensor.data_ptr();
    // stream(true)：返回前清 task queue，确保直调 kernel 晚于队列中既有 NPU 操作（反模式规避）
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    auto acl_call = [=]() -> int {
        auto dtype = u.scalar_type();
        if (dtype == torch::kFloat32) {
            launch_selective_scan_fwd_float(numBlocks, uPtr, deltaPtr, aPtr, bPtr, cPtr, dPtr,
                zPtr, biasPtr, outPtr, xPtr, outZPtr, tilingPtr, stream);
        } else if (dtype == torch::kFloat16) {
            launch_selective_scan_fwd_half(numBlocks, uPtr, deltaPtr, aPtr, bPtr, cPtr, dPtr,
                zPtr, biasPtr, outPtr, xPtr, outZPtr, tilingPtr, stream);
        } else if (dtype == torch::kBFloat16) {
            launch_selective_scan_fwd_bfloat16(numBlocks, uPtr, deltaPtr, aPtr, bPtr, cPtr, dPtr,
                zPtr, biasPtr, outPtr, xPtr, outZPtr, tilingPtr, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("SelectiveScanFwd", acl_call);
}

} // namespace

// ---------- Meta 函数（输出 shape/dtype 推导） ----------
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> selective_scan_fwd_meta(
    const torch::Tensor &u, const torch::Tensor &delta, const torch::Tensor &A,
    const torch::Tensor &B, const torch::Tensor &C, c10::optional<torch::Tensor> D,
    c10::optional<torch::Tensor> z, c10::optional<torch::Tensor> delta_bias, bool delta_softplus)
{
    // meta 阶段只做 shape/dtype 推导所需的解析（不触发设备侧校验）
    SsfShapeInfo info = ssf_parse(u, delta, A, B, C,
                                  D.has_value() ? D.value() : torch::Tensor(),
                                  z.has_value() ? z.value() : torch::Tensor(),
                                  delta_bias.has_value() ? delta_bias.value() : torch::Tensor(),
                                  delta_softplus);
    auto r = ssf_meta_impl(u, info.nChunks, info.dstate, info.hasZ);
    return r;
}

// ---------- NPU 实现（校验 + tiling + launch kernel） ----------
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> selective_scan_fwd_npu(
    const torch::Tensor &u, const torch::Tensor &delta, const torch::Tensor &A,
    const torch::Tensor &B, const torch::Tensor &C, c10::optional<torch::Tensor> D,
    c10::optional<torch::Tensor> z, c10::optional<torch::Tensor> delta_bias, bool delta_softplus)
{
    const c10::OptionalDeviceGuard guard(u.device());
    SsfShapeInfo info = ssf_parse(u, delta, A, B, C,
                                  D.has_value() ? D.value() : torch::Tensor(),
                                  z.has_value() ? z.value() : torch::Tensor(),
                                  delta_bias.has_value() ? delta_bias.value() : torch::Tensor(),
                                  delta_softplus);
    // UB 容量准入（B1 红线：运行时查询；仅设备侧执行，Meta 阶段不查）
    ssf_check_ub_capacity(info, u.scalar_type());
    auto r = ssf_meta_impl(u, info.nChunks, info.dstate, info.hasZ);
    ssf_launch(u, delta, A, B, C,
               D.has_value() ? D.value() : torch::Tensor(),
               z.has_value() ? z.value() : torch::Tensor(),
               delta_bias.has_value() ? delta_bias.value() : torch::Tensor(),
               std::get<0>(r), std::get<1>(r), std::get<2>(r), info);
    return r;
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("selective_scan_fwd", selective_scan_fwd_meta);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("selective_scan_fwd", selective_scan_fwd_npu);
}

} // namespace cann_bench
