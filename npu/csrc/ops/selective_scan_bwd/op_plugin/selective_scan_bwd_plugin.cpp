/**
 * selective_scan_bwd - torch.library 注册 + Meta + NPU impl（g++ 编译）
 * schema 契约来源：2.2-开发方案设计.md §2.1（14 参入参 → 9 元组输出，对齐 golden 反传语义）
 * 校验清单：2.2-开发方案设计.md 第 7 节（校验顺序逐条落地，先于 tiling 计算）
 * 输出装配：du/ddelta/dz/out_z 直出；dA/dB/dC/dD/ddelta_bias 经 fp32 pad（zeros_ 初始化，
 *   kernel 侧 SetAtomicAdd 跨核跨 pair 累加）→ narrow/select 视图变换为最终契约 shape。
 */
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../op_kernel/selective_scan_bwd_launch.h"

namespace cann_bench {

// ---------- schema 注册（逐字对齐 2.2 §2.1 接口原型） ----------
TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("selective_scan_bwd(Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C, Tensor dout, "
          "Tensor? D=None, Tensor? z=None, Tensor? delta_bias=None, Tensor? x=None, "
          "Tensor? out=None, Tensor? dz=None, "
          "bool delta_softplus=False, bool recompute_out_z=False) -> "
          "(Tensor du, Tensor ddelta, Tensor dA, Tensor dB, Tensor dC, "
          "Tensor dD, Tensor ddelta_bias, Tensor dz, Tensor out_z)");
}

namespace {

constexpr int64_t SSB_CHUNK_T = 2048;   // x 契约 chunk 宽度（kernel 侧 TILE_T=1024，chunk=2×tile）
constexpr int64_t SSB_MAX_DSTATE = 256; // 方案约束：dstate ∈ [1,256]
constexpr int64_t SSB_TILE_T = 1024;    // kernel 处理单元（UB 需求镜像用）

struct SsbShapeInfo {
    int64_t batch = 0;
    int64_t dim = 0;
    int64_t seqlen = 0;
    int64_t dstate = 0;
    int64_t nGroups = 1;    // dim / dimPerGroup（变量 B/C 分组数；静态按 dim 逐通道）
    int64_t dimPerGroup = 1;
    int64_t nChunks = 0;
    bool hasD = false;
    bool hasZ = false;
    bool hasBias = false;
    bool hasX = false;
    bool softplus = false;
    bool recompute = false;
    bool varB = false;
    bool varC = false;
};

[[noreturn]] void ssb_error(const std::string &msg)
{
    TORCH_CHECK(false, "selective_scan_bwd: ", msg);
}

void ssb_check_contiguous(const torch::Tensor &t, const char *name)
{
    TORCH_CHECK(t.is_contiguous(), "selective_scan_bwd: ", name, " must be contiguous");
}

// dtype 分派：u 为基准，delta/dout/z/变量 B/C 同 u（2.2 §2.1）
void ssb_input_dtype(const torch::Tensor &u, const torch::Tensor &delta, const torch::Tensor &dout,
                     const torch::Tensor &z, const torch::Tensor &B, const torch::Tensor &C,
                     bool varB, bool varC)
{
    const auto dt = u.scalar_type();
    TORCH_CHECK(dt == torch::kFloat32 || dt == torch::kFloat16 || dt == torch::kBFloat16,
                "selective_scan_bwd: u dtype must be float32/float16/bfloat16, got ", dt);
    TORCH_CHECK(delta.scalar_type() == dt, "selective_scan_bwd: delta dtype must match u");
    TORCH_CHECK(dout.scalar_type() == dt, "selective_scan_bwd: dout dtype must match u");
    if (z.defined()) {
        TORCH_CHECK(z.scalar_type() == dt, "selective_scan_bwd: z dtype must match u");
    }
    if (varB) {
        TORCH_CHECK(B.scalar_type() == dt, "selective_scan_bwd: variable B dtype must match u");
    }
    if (varC) {
        TORCH_CHECK(C.scalar_type() == dt, "selective_scan_bwd: variable C dtype must match u");
    }
}

// ---------- 解析并校验全部输入（顺序按方案第 7 节，先于 tiling） ----------
SsbShapeInfo ssb_parse(const torch::Tensor &u, const torch::Tensor &delta, const torch::Tensor &A,
                       const torch::Tensor &B, const torch::Tensor &C, const torch::Tensor &dout,
                       const torch::Tensor &D, const torch::Tensor &z, const torch::Tensor &delta_bias,
                       const torch::Tensor &x, const torch::Tensor &out, const torch::Tensor &dz,
                       bool delta_softplus, bool recompute_out_z)
{
    SsbShapeInfo info{};

    // ---- 形状类校验最前：u 为 3 维 [batch, dim, seqlen]，batch/dim > 0（先于 tiling 拦截除零） ----
    TORCH_CHECK(u.dim() == 3, "selective_scan_bwd: u must have 3 dims [batch, dim, seqlen]");
    ssb_check_contiguous(u, "u");
    info.batch = u.size(0);
    info.dim = u.size(1);
    info.seqlen = u.size(2);
    if (info.batch <= 0) {
        ssb_error("batch must be positive, got 0");
    }
    if (info.dim <= 0) {
        ssb_error("dim must be positive, got 0");
    }
    // seqlen == 0 合法（空 tensor，全部梯度输出为空/零）

    // ---- delta / dout：同 u 形状 ----
    TORCH_CHECK(delta.dim() == 3 && delta.size(0) == info.batch && delta.size(1) == info.dim &&
                    delta.size(2) == info.seqlen,
                "selective_scan_bwd: delta must have shape [batch, dim, seqlen] same as u");
    ssb_check_contiguous(delta, "delta");
    TORCH_CHECK(dout.dim() == 3 && dout.size(0) == info.batch && dout.size(1) == info.dim &&
                    dout.size(2) == info.seqlen,
                "selective_scan_bwd: dout must have shape [batch, dim, seqlen] same as u");
    ssb_check_contiguous(dout, "dout");

    // ---- A: [dim, dstate] fp32 ----
    TORCH_CHECK(A.dim() == 2 && A.size(0) == info.dim,
                "selective_scan_bwd: A must have shape [dim, dstate]");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "selective_scan_bwd: A dtype must be float32");
    ssb_check_contiguous(A, "A");
    info.dstate = A.size(1);
    TORCH_CHECK(info.dstate >= 1 && info.dstate <= SSB_MAX_DSTATE,
                "selective_scan_bwd: dstate must be in [1, 256], got ", info.dstate);

    // ---- B/C: 静态 [dim, dstate] fp32 或变量 [batch, n_groups, dstate, seqlen] input_t ----
    TORCH_CHECK(B.dim() == 2 || B.dim() == 4,
                "selective_scan_bwd: B must be [dim, dstate] (static) or "
                "[batch, n_groups, dstate, seqlen] (variable), got ndim=",
                B.dim());
    TORCH_CHECK(C.dim() == 2 || C.dim() == 4,
                "selective_scan_bwd: C must be [dim, dstate] (static) or "
                "[batch, n_groups, dstate, seqlen] (variable), got ndim=",
                C.dim());
    info.varB = (B.dim() == 4);
    info.varC = (C.dim() == 4);
    if (info.varB) {
        info.nGroups = B.size(1);
        TORCH_CHECK(B.size(0) == info.batch && B.size(2) == info.dstate && B.size(3) == info.seqlen,
                    "selective_scan_bwd: variable B must have shape [batch, n_groups, dstate, seqlen]");
    } else {
        info.nGroups = info.dim;  // 静态 B 按 dim 逐通道，无分组
        TORCH_CHECK(B.size(0) == info.dim && B.size(1) == info.dstate,
                    "selective_scan_bwd: static B must have shape [dim, dstate]");
        TORCH_CHECK(B.scalar_type() == torch::kFloat32, "selective_scan_bwd: static B dtype must be float32");
    }
    if (info.varC) {
        if (info.varB) {
            TORCH_CHECK(C.size(1) == info.nGroups,
                        "selective_scan_bwd: variable C n_groups (", C.size(1),
                        ") must match variable B n_groups (", info.nGroups, ")");
        } else {
            info.nGroups = C.size(1);
        }
        TORCH_CHECK(C.size(0) == info.batch && C.size(2) == info.dstate && C.size(3) == info.seqlen,
                    "selective_scan_bwd: variable C must have shape [batch, n_groups, dstate, seqlen]");
    } else {
        TORCH_CHECK(C.size(0) == info.dim && C.size(1) == info.dstate,
                    "selective_scan_bwd: static C must have shape [dim, dstate]");
        TORCH_CHECK(C.scalar_type() == torch::kFloat32, "selective_scan_bwd: static C dtype must be float32");
    }
    TORCH_CHECK(info.dim % info.nGroups == 0,
                "selective_scan_bwd: dim must be divisible by n_groups (dim=", info.dim,
                ", n_groups=", info.nGroups, ")");
    info.dimPerGroup = info.dim / info.nGroups;
    ssb_check_contiguous(B, "B");
    ssb_check_contiguous(C, "C");

    // ---- D / delta_bias: [dim] fp32 ----
    info.hasD = D.defined();
    if (info.hasD) {
        TORCH_CHECK(D.dim() == 1 && D.size(0) == info.dim,
                    "selective_scan_bwd: D must have shape [dim]");
        TORCH_CHECK(D.scalar_type() == torch::kFloat32, "selective_scan_bwd: D dtype must be float32");
        ssb_check_contiguous(D, "D");
    }
    info.hasBias = delta_bias.defined();
    if (info.hasBias) {
        TORCH_CHECK(delta_bias.dim() == 1 && delta_bias.size(0) == info.dim,
                    "selective_scan_bwd: delta_bias must have shape [dim]");
        TORCH_CHECK(delta_bias.scalar_type() == torch::kFloat32,
                    "selective_scan_bwd: delta_bias dtype must be float32");
        ssb_check_contiguous(delta_bias, "delta_bias");
    }

    // ---- z / out / dz（z 链依赖：has_z ⇒ out 必须提供且同形） ----
    info.hasZ = z.defined();
    if (info.hasZ) {
        TORCH_CHECK(z.dim() == 3 && z.size(0) == info.batch && z.size(1) == info.dim &&
                        z.size(2) == info.seqlen,
                    "selective_scan_bwd: z must have shape [batch, dim, seqlen]");
        ssb_check_contiguous(z, "z");
        TORCH_CHECK(out.defined(),
                    "selective_scan_bwd: out (fwd pre-gate output) is required when z is provided");
        TORCH_CHECK(out.dim() == 3 && out.size(0) == info.batch && out.size(1) == info.dim &&
                        out.size(2) == info.seqlen,
                    "selective_scan_bwd: out must have shape [batch, dim, seqlen]");
        ssb_check_contiguous(out, "out");
        if (dz.defined()) {
            TORCH_CHECK(dz.dim() == 3 && dz.size(0) == info.batch && dz.size(1) == info.dim &&
                            dz.size(2) == info.seqlen,
                        "selective_scan_bwd: dz must have shape [batch, dim, seqlen] same as z");
            TORCH_CHECK(dz.scalar_type() == z.scalar_type(),
                        "selective_scan_bwd: dz dtype must match z");
            ssb_check_contiguous(dz, "dz");
        }
    } else {
        // 无 z：dz / recompute_out_z 入参忽略（S-NW-03 忽略语义，非报错）。
        // 权威依据：CUDA 参考 selective_scan.cpp 的 z/out/dz/recompute 块整体位于
        // if (has_z) 内——dz 提供而无 z 时参考实现不读、不校验、不报错，结果元组
        // 亦不含 dz；recompute_out_z 无 z 时 out_z 同样不产出。本侧按 C3 固定
        // 9 元组缺省项 {0} 契约返回空 tensor（下方返回构造已处理）。
        // dz 的 shape/dtype 校验仅在 has_z 分支执行（S-EW-24/25 语义不变）。
    }
    info.softplus = delta_softplus;
    // recompute 旗标门控：无 z 时忽略（S-NW-03），防止 kernel 侧触碰未分配的
    // outZ32Buf_（InitBuffer 位于 hasZ_ 块内）；has_z 时语义不变。
    info.recompute = recompute_out_z && info.hasZ;

    // ---- x: n_chunks>1 ⇒ 必须提供，fp32 [batch, dim, n_chunks, 2*dstate] ----
    info.nChunks = (info.seqlen + SSB_CHUNK_T - 1) / SSB_CHUNK_T;
    info.hasX = (info.nChunks > 1);
    if (info.hasX) {
        TORCH_CHECK(x.defined(),
                    "selective_scan_bwd: x (fwd per-chunk prefix states) is required when n_chunks>1");
        TORCH_CHECK(x.dim() == 4 && x.size(0) == info.batch && x.size(1) == info.dim &&
                        x.size(2) == info.nChunks && x.size(3) == 2 * info.dstate,
                    "selective_scan_bwd: x must have shape [batch, dim, n_chunks, 2*dstate]");
        TORCH_CHECK(x.scalar_type() == torch::kFloat32, "selective_scan_bwd: x dtype must be float32");
        ssb_check_contiguous(x, "x");
    }

    // ---- dtype 分派校验（u 为基准，变量 B/C/z 同 u） ----
    ssb_input_dtype(u, delta, dout, z, B, C, info.varB, info.varC);
    return info;
}

uint32_t ssb_flags(const SsbShapeInfo &info)
{
    uint32_t flags = 0;
    if (info.hasD)      flags |= SSB_FLAG_HAS_D;
    if (info.hasZ)      flags |= SSB_FLAG_HAS_Z;
    if (info.hasBias)   flags |= SSB_FLAG_HAS_BIAS;
    if (info.softplus)  flags |= SSB_FLAG_SOFTPLUS;
    if (info.varB)      flags |= SSB_FLAG_VAR_B;
    if (info.varC)      flags |= SSB_FLAG_VAR_C;
    if (info.recompute) flags |= SSB_FLAG_RECOMPUTE_OUT_Z;
    if (info.hasX)      flags |= SSB_FLAG_HAS_X;
    return flags;
}

// kernel 侧 InitBuffer 的 UB 需求镜像（字节）：与 selective_scan_bwd_kernel.cpp Init 分配清单
// 逐项一致（TILE_T=1024；gate 复用 h0Buf_ 不计独立缓冲；dz 存出批量后移不增缓冲）。
// 逐项算式（T=TILE_T，dts=sizeof(INPUT_T)，inBytes=T·dts，DS=dstate 静态上界 256）：
//   11 × T×4（恒 fp32 TBuf：u32/u32T0/d32/d32T0/rawDelta/rawDeltaT0/deltaU32/dout32/du32/ddelta32/tmpA）
//   4 × 2048×4（sa0/sa1/sb0/sb1 扫描缓冲，含双侧 pad）
//   2 × T×4（h0/b0k；σ gate 复用 h0）
//   T×4（gatherIdx）+ 3×DS×4（aRow/bRow/cRow）+ 64（sTmp）+ 2×DS×4（xPrefix，hasX）+ DS×4（dx0）
//   3×32（reduceDstA/B/C）+ 4×DS×4+64（rowVec，VECOUT）+ T×4（rowStage，VECOUT）
//   T×dts（doutStage）+ hasZ→2×T×dts（z/outStage）+ 3×T×4（z32/out32/dz32）+ recompute→T×4（outZ32）
//   4×inBytes（inQu/inQd 深度2）+ varB/varC→各 4×inBytes + 各 T×4
// fp32 全特性上界 = 190688 B ≈ 186.2KB ≤ 192KB
int64_t ssb_ub_require_bytes(const SsbShapeInfo &info, at::ScalarType dtype)
{
    const int64_t T = SSB_TILE_T;
    const int64_t dts = (dtype == at::kFloat) ? 4 : 2;
    const int64_t inBytes = T * dts;
    int64_t need = 0;
    need += 11 * T * 4;                 // 恒 fp32 TBuf（11 块）
    need += 4 * 2048 * 4;               // 扫描缓冲 ×4（2048 lanes）
    need += 2 * T * 4;                  // h0/b0k（gate 复用 h0）
    if (info.varB) need += T * 4;       // varB32
    if (info.varC) need += T * 4;       // varC32
    need += T * dts;                    // doutStage（INPUT_T 暂存）
    if (info.hasZ) {
        need += 2 * T * dts;            // zStage/outStage
        need += 3 * T * 4;              // z32/out32/dz32
        if (info.recompute) need += T * 4;  // outZ32
    }
    need += T * 4;                      // gatherIdx
    need += 3 * SSB_MAX_DSTATE * 4;     // aRow/bRow/cRow（静态上限）
    need += 64;                         // sTmp（D@0/bias@8 两槽）
    if (info.hasX) need += 2 * SSB_MAX_DSTATE * 4;  // xPrefix
    need += SSB_MAX_DSTATE * 4;         // dx0
    need += 3 * 32;                     // reduceDstA/B/C
    need += 4 * SSB_MAX_DSTATE * 4 + 64;  // rowVec（VECOUT，[4×dsp)+16 pack8 布局）
    need += T * 4;                      // rowStage（VECOUT 原子行源）
    need += 4 * inBytes;                // inQu/inQd：2 队列 × 深度 2
    if (info.varB) need += 4 * inBytes; // inQvarB
    if (info.varC) need += 4 * inBytes; // inQvarC
    return need;
}

// B1 红线：UB 容量运行时查询准入校验（不硬编码容量）
void ssb_check_ub_capacity(const SsbShapeInfo &info, at::ScalarType dtype)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const int64_t need = ssb_ub_require_bytes(info, dtype);
    TORCH_CHECK(need <= static_cast<int64_t>(ubSize),
                "selective_scan_bwd: UB requirement ", need, " bytes (dtype=", dtype,
                ", dstate=", info.dstate, ", flags shape variant) exceeds device UB ", ubSize, " bytes");
}

// ---------- 输出 shape/dtype 推导（仅 Meta 阶段占位用；NPU 实现在 npu_impl 内装配） ----------
// 9 元组：du, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z
using SsbOutTuple = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                               torch::Tensor>;

SsbOutTuple ssb_outputs_meta(const SsbShapeInfo &info, const torch::Tensor &u,
                             const torch::Tensor &B, const torch::Tensor &C,
                             const torch::Tensor &dz)
{
    auto inputOptions = u.options();
    auto fp32Options = inputOptions.dtype(torch::kFloat32);

    // du/ddelta：与 u 同形同 dtype
    torch::Tensor du = torch::empty_like(u);
    torch::Tensor ddelta = torch::empty_like(u);
    // dA：[dim, dstate] fp32
    torch::Tensor dA = torch::empty({info.dim, info.dstate}, fp32Options);
    // dB/dC：静态 [dim, dstate] fp32；变量 [batch, n_groups, dstate, seqlen] 同 B/C dtype
    torch::Tensor dB;
    torch::Tensor dC;
    if (info.varB) {
        dB = torch::empty({info.batch, info.nGroups, info.dstate, info.seqlen},
                          inputOptions.dtype(B.scalar_type()));
    } else {
        dB = torch::empty({info.dim, info.dstate}, fp32Options);
    }
    if (info.varC) {
        dC = torch::empty({info.batch, info.nGroups, info.dstate, info.seqlen},
                          inputOptions.dtype(C.scalar_type()));
    } else {
        dC = torch::empty({info.dim, info.dstate}, fp32Options);
    }
    // dD/ddelta_bias：[dim] fp32；D/delta_bias 未提供时按 {0} 空 tensor 契约（9 元组固定返回）
    torch::Tensor dD = info.hasD ? torch::empty({info.dim}, fp32Options)
                                 : torch::empty({0}, fp32Options);
    torch::Tensor dBias = info.hasBias ? torch::empty({info.dim}, fp32Options)
                                       : torch::empty({0}, fp32Options);
    // dz：提供则原地写返回，否则 empty_like；!hasZ 返回 {0}
    torch::Tensor dzOut;
    if (info.hasZ) {
        dzOut = dz.defined() ? dz : torch::empty_like(u);
    } else {
        dzOut = torch::empty({0}, inputOptions);
    }
    // out_z：hasZ 且 recompute 才产出，否则 {0}
    torch::Tensor outZ;
    if (info.hasZ && info.recompute) {
        outZ = torch::empty_like(u);
    } else {
        outZ = torch::empty({0}, inputOptions);
    }
    return std::make_tuple(std::move(du), std::move(ddelta), std::move(dA), std::move(dB),
                           std::move(dC), std::move(dD), std::move(dBias), std::move(dzOut),
                           std::move(outZ));
}

// ---------- 直调 launch（16 全指针 + GM tiling 标量包） ----------
void ssb_launch(const torch::Tensor &u, const torch::Tensor &delta, const torch::Tensor &A,
                const torch::Tensor &B, const torch::Tensor &C, const torch::Tensor &dout,
                const torch::Tensor &D, const torch::Tensor &z, const torch::Tensor &delta_bias,
                const torch::Tensor &x, const torch::Tensor &out, const torch::Tensor &dz,
                torch::Tensor &outZ, torch::Tensor &du, torch::Tensor &ddelta,
                torch::Tensor &dAPad, torch::Tensor &dBPad, torch::Tensor &dCPad,
                torch::Tensor &dDPad, torch::Tensor &dBiasPad, const SsbShapeInfo &info)
{
    const int64_t numPairs = info.batch * info.dim;
    // seqlen=0（nChunks=0）：合法空输出，跳过 kernel launch（pad 已 zeros_，结果恒零正确）
    if (info.seqlen == 0 || info.nChunks == 0) {
        return;
    }
    int64_t usedCores = 0;
    int64_t pairsPerCore = 0;
    int64_t dstatePad = 0;
    int64_t seqlenPad = 0;
    std::tie(usedCores, pairsPerCore, std::ignore, dstatePad, seqlenPad) =
        calc_selective_scan_bwd_tiling_params(info.batch, info.dim, info.seqlen, info.dstate);
    (void)usedCores;
    const uint32_t numBlocks =
        static_cast<uint32_t>((numPairs + pairsPerCore - 1) / pairsPerCore);

    // 标量参数打包为 GM tiling 缓冲（直调链路对 >11 参签名中段标量有损坏，见 launch.h）
    SsbTilingData td{};
    td.batch = info.batch;
    td.dim = info.dim;
    td.seqlen = info.seqlen;
    td.dstate = info.dstate;
    td.dimPerGroup = info.dimPerGroup;
    td.nChunks = info.nChunks;
    td.numPairs = numPairs;
    td.pairsPerCore = pairsPerCore;
    td.dstatePad = dstatePad;
    td.seqlenPad = seqlenPad;
    td.flags = ssb_flags(info);
    td.dAPadPtr = reinterpret_cast<int64_t>(dAPad.data_ptr());
    td.dBPadPtr = reinterpret_cast<int64_t>(dBPad.data_ptr());
    td.dCPadPtr = reinterpret_cast<int64_t>(dCPad.data_ptr());
    td.dDPadPtr = reinterpret_cast<int64_t>(dDPad.data_ptr());
    td.dbiasPadPtr = reinterpret_cast<int64_t>(dBiasPad.data_ptr());
    torch::Tensor tilingTensor = torch::empty(
        {static_cast<int64_t>(sizeof(SsbTilingData))}, u.options().dtype(torch::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(SsbTilingData),
                &td, sizeof(SsbTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    auto uPtr = (GM_ADDR)u.data_ptr();
    auto deltaPtr = (GM_ADDR)delta.data_ptr();
    auto aPtr = (GM_ADDR)A.data_ptr();
    auto bPtr = (GM_ADDR)B.data_ptr();
    auto cPtr = (GM_ADDR)C.data_ptr();
    auto doutPtr = (GM_ADDR)dout.data_ptr();
    auto dPtr = info.hasD ? (GM_ADDR)D.data_ptr() : (GM_ADDR)nullptr;
    auto zPtr = info.hasZ ? (GM_ADDR)z.data_ptr() : (GM_ADDR)nullptr;
    auto biasPtr = info.hasBias ? (GM_ADDR)delta_bias.data_ptr() : (GM_ADDR)nullptr;
    auto xPtr = info.hasX ? (GM_ADDR)x.data_ptr() : (GM_ADDR)nullptr;
    auto outPtr = info.hasZ ? (GM_ADDR)out.data_ptr() : (GM_ADDR)nullptr;
    auto dzPtr = info.hasZ ? (GM_ADDR)dz.data_ptr() : (GM_ADDR)nullptr;
    auto outZPtr = (info.hasZ && info.recompute) ? (GM_ADDR)outZ.data_ptr() : (GM_ADDR)nullptr;
    auto duPtr = (GM_ADDR)du.data_ptr();
    auto ddeltaPtr = (GM_ADDR)ddelta.data_ptr();
    auto tilingPtr = (GM_ADDR)tilingTensor.data_ptr();
    // stream(true)：返回前清 task queue，确保直调 kernel 晚于队列中既有 NPU 操作（反模式规避）
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    auto acl_call = [=]() -> int {
        auto dtype = u.scalar_type();
        if (dtype == torch::kFloat32) {
            launch_selective_scan_bwd_float(numBlocks, uPtr, deltaPtr, aPtr, bPtr, cPtr, doutPtr,
                dPtr, zPtr, biasPtr, xPtr, outPtr, dzPtr, outZPtr, duPtr, ddeltaPtr, tilingPtr, stream);
        } else if (dtype == torch::kFloat16) {
            launch_selective_scan_bwd_half(numBlocks, uPtr, deltaPtr, aPtr, bPtr, cPtr, doutPtr,
                dPtr, zPtr, biasPtr, xPtr, outPtr, dzPtr, outZPtr, duPtr, ddeltaPtr, tilingPtr, stream);
        } else if (dtype == torch::kBFloat16) {
            launch_selective_scan_bwd_bfloat16(numBlocks, uPtr, deltaPtr, aPtr, bPtr, cPtr, doutPtr,
                dPtr, zPtr, biasPtr, xPtr, outPtr, dzPtr, outZPtr, duPtr, ddeltaPtr, tilingPtr, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("SelectiveScanBwd", acl_call);
}

} // namespace

// ---------- Meta 函数（输出 shape/dtype 推导） ----------
static SsbOutTuple selective_scan_bwd_meta(
    const torch::Tensor &u, const torch::Tensor &delta, const torch::Tensor &A,
    const torch::Tensor &B, const torch::Tensor &C, const torch::Tensor &dout,
    c10::optional<torch::Tensor> D, c10::optional<torch::Tensor> z,
    c10::optional<torch::Tensor> delta_bias, c10::optional<torch::Tensor> x,
    c10::optional<torch::Tensor> out, c10::optional<torch::Tensor> dz,
    bool delta_softplus, bool recompute_out_z)
{
    // meta 阶段只做 shape/dtype 推导所需的解析（不触发设备侧校验）
    SsbShapeInfo info = ssb_parse(u, delta, A, B, C, dout,
                                  D.has_value() ? D.value() : torch::Tensor(),
                                  z.has_value() ? z.value() : torch::Tensor(),
                                  delta_bias.has_value() ? delta_bias.value() : torch::Tensor(),
                                  x.has_value() ? x.value() : torch::Tensor(),
                                  out.has_value() ? out.value() : torch::Tensor(),
                                  dz.has_value() ? dz.value() : torch::Tensor(),
                                  delta_softplus, recompute_out_z);
    return ssb_outputs_meta(info, u, B, C,
                            dz.has_value() ? dz.value() : torch::Tensor());
}

// ---------- NPU 实现（校验 + UB 准入 + pad 分配 + launch + pad→final 变换） ----------
static SsbOutTuple selective_scan_bwd_npu(
    const torch::Tensor &u, const torch::Tensor &delta, const torch::Tensor &A,
    const torch::Tensor &B, const torch::Tensor &C, const torch::Tensor &dout,
    c10::optional<torch::Tensor> D, c10::optional<torch::Tensor> z,
    c10::optional<torch::Tensor> delta_bias, c10::optional<torch::Tensor> x,
    c10::optional<torch::Tensor> out, c10::optional<torch::Tensor> dz,
    bool delta_softplus, bool recompute_out_z)
{
    const c10::OptionalDeviceGuard guard(u.device());
    const torch::Tensor &zT = z.has_value() ? z.value() : torch::Tensor();
    const torch::Tensor &dzT = dz.has_value() ? dz.value() : torch::Tensor();
    SsbShapeInfo info = ssb_parse(u, delta, A, B, C, dout,
                                  D.has_value() ? D.value() : torch::Tensor(),
                                  zT,
                                  delta_bias.has_value() ? delta_bias.value() : torch::Tensor(),
                                  x.has_value() ? x.value() : torch::Tensor(),
                                  out.has_value() ? out.value() : torch::Tensor(),
                                  dzT,
                                  delta_softplus, recompute_out_z);
    // UB 容量准入（B1 红线：运行时查询；仅设备侧执行，Meta 阶段不查）
    ssb_check_ub_capacity(info, u.scalar_type());

    auto inputOptions = u.options();
    auto fp32Options = inputOptions.dtype(torch::kFloat32);
    const int64_t dsp = (info.dstate + 7) / 8 * 8;   // dstate_pad
    const int64_t slp = (info.seqlen + 7) / 8 * 8;   // seqlen_pad

    // du/ddelta/dz/out_z 直出
    torch::Tensor du = torch::empty_like(u);
    torch::Tensor ddelta = torch::empty_like(u);
    torch::Tensor dzOut = info.hasZ ? (dzT.defined() ? dzT : torch::empty_like(u))
                                    : torch::empty({0}, inputOptions);
    torch::Tensor outZ = (info.hasZ && info.recompute) ? torch::empty_like(u)
                                                       : torch::empty({0}, inputOptions);

    // 归约 pad：zeros_ 初始化（kernel 侧 SetAtomicAdd 跨核/跨 pair 累加；pad lane 恒 0）
    torch::Tensor dAPad = torch::zeros({info.dim, dsp}, fp32Options);
    torch::Tensor dDPad = torch::zeros({info.dim, 8}, fp32Options);
    torch::Tensor dBiasPad = torch::zeros({info.dim, 8}, fp32Options);
    torch::Tensor dBPad;
    torch::Tensor dCPad;
    if (info.varB) {
        dBPad = torch::zeros({info.batch, info.nGroups, info.dstate, slp}, fp32Options);
    } else {
        dBPad = torch::zeros({info.dim, dsp}, fp32Options);
    }
    if (info.varC) {
        dCPad = torch::zeros({info.batch, info.nGroups, info.dstate, slp}, fp32Options);
    } else {
        dCPad = torch::zeros({info.dim, dsp}, fp32Options);
    }

    // launch（pad 指针经 tiling 传入）
    ssb_launch(u, delta, A, B, C, dout,
               D.has_value() ? D.value() : torch::Tensor(),
               zT,
               delta_bias.has_value() ? delta_bias.value() : torch::Tensor(),
               x.has_value() ? x.value() : torch::Tensor(),
               out.has_value() ? out.value() : torch::Tensor(),
               dzOut, outZ, du, ddelta,
               dAPad, dBPad, dCPad, dDPad, dBiasPad, info);

    // pad → 最终契约 shape 变换（流上异步，晚于 kernel 排队）
    torch::Tensor dA = dAPad.narrow(1, 0, info.dstate).contiguous();
    torch::Tensor dB;
    torch::Tensor dC;
    if (info.varB) {
        dB = dBPad.narrow(3, 0, info.seqlen).to(B.scalar_type()).contiguous();
    } else {
        dB = dBPad.narrow(1, 0, info.dstate).contiguous();
    }
    if (info.varC) {
        dC = dCPad.narrow(3, 0, info.seqlen).to(C.scalar_type()).contiguous();
    } else {
        dC = dCPad.narrow(1, 0, info.dstate).contiguous();
    }
    torch::Tensor dD = info.hasD ? dDPad.select(1, 0)        // lane0 视图
                                 : torch::empty({0}, fp32Options);  // {0} 空契约
    torch::Tensor dBias = info.hasBias ? dBiasPad.select(1, 0)  // lane0 视图
                                       : torch::empty({0}, fp32Options);  // {0} 空契约

    return std::make_tuple(std::move(du), std::move(ddelta), std::move(dA), std::move(dB),
                           std::move(dC), std::move(dD), std::move(dBias), std::move(dzOut),
                           std::move(outZ));
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("selective_scan_bwd", selective_scan_bwd_meta);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("selective_scan_bwd", selective_scan_bwd_npu);
}

} // namespace cann_bench
