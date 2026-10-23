#ifndef SELECTIVE_SCAN_BWD_LAUNCH_H
#define SELECTIVE_SCAN_BWD_LAUNCH_H
#include <cstdint>
#include <tuple>
#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// 特性标志位掩码（kernel / plugin 共享；plugin 计算后传入 kernel）
// 位定义与 fwd SsfTilingData 同构扩展：低 6 位同 fwd 语义，新增 bit6/bit7
static constexpr uint32_t SSB_FLAG_HAS_D        = 1u << 0;  // D 跳连提供
static constexpr uint32_t SSB_FLAG_HAS_Z        = 1u << 1;  // z 门控提供（dz 必写；out 入参必须提供）
static constexpr uint32_t SSB_FLAG_HAS_BIAS     = 1u << 2;  // delta_bias 提供
static constexpr uint32_t SSB_FLAG_SOFTPLUS     = 1u << 3;  // delta_softplus=true
static constexpr uint32_t SSB_FLAG_VAR_B        = 1u << 4;  // 变量 B
static constexpr uint32_t SSB_FLAG_VAR_C        = 1u << 5;  // 变量 C
static constexpr uint32_t SSB_FLAG_RECOMPUTE_OUT_Z = 1u << 6; // 复算 out_z = out·silu(z)（前置：has_z）
static constexpr uint32_t SSB_FLAG_HAS_X        = 1u << 7;  // x 前缀提供（n_chunks>1；kernel 侧 c>0 才读）

// 标量参数聚合结构（kernel / plugin 共享，纯 C++ 语法）。
// 经 GM tiling 缓冲传入 kernel：直调 launch 链路（<<<>>> 降级的 eager-launch）
// 实测对 >11 参签名存在中段标量参数损坏（指针区完好），故标量统一走 GM tiling，
// kernel 签名仅保留指针参数（16 个全指针，全部落在实测完好的参数区）。
// 归约 GM pad 缓冲指针也经此结构传入（kernel 侧 SetGlobalBuffer 重绑）。
// 布局（fp64 对齐，sizeof==128）：
//   [0..79)   10 × int64 形状/切分
//   [80]      uint32 flags；[84] uint32 pad（对齐填充）
//   [88..127) 5 × int64 归约缓冲 GM 指针
struct SsbTilingData {
    int64_t batch;        // 0
    int64_t dim;          // 8
    int64_t seqlen;       // 16
    int64_t dstate;       // 24
    int64_t dimPerGroup;  // 32
    int64_t nChunks;      // 40
    int64_t numPairs;     // 48
    int64_t pairsPerCore; // 56
    int64_t dstatePad;    // 64
    int64_t seqlenPad;    // 72
    uint32_t flags;       // 80
    uint32_t pad0;        // 84
    int64_t dAPadPtr;     // 88  dA_pad [dim, dstate_pad] fp32（原子加目标）
    int64_t dBPadPtr;     // 96  dB_pad：静态 [dim, dstate_pad] / 变量 [b,ng,dstate,seqlen_pad] fp32
    int64_t dCPadPtr;     // 104 dC_pad：同 dB_pad 形态
    int64_t dDPadPtr;     // 112 dD_pad [dim, 8] fp32（每 d 占 8 lane，lane0 有效）
    int64_t dbiasPadPtr;  // 120 ddelta_bias_pad [dim, 8] fp32
};

// Tiling 参数计算（kernel 侧定义，plugin 调用；B1 红线：核数运行时查询，不硬编码）
// 返回 (usedCores, pairsPerCore, nChunks, dstate_pad, seqlen_pad)
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>
calc_selective_scan_bwd_tiling_params(int64_t batch, int64_t dim, int64_t seqlen, int64_t dstate);

// GM 指针顺序（16 个，全指针）：u, delta, A, B, C, dout, D, z, delta_bias,
//                               x, out, dz, out_z, du, ddelta, tiling
// blockDim 由 plugin 按 ceil(numPairs/pairsPerCore) 计算后显式传入
extern "C" {
void launch_selective_scan_bwd_float(uint32_t numBlocks,
    GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR dout,
    GM_ADDR D, GM_ADDR z, GM_ADDR delta_bias,
    GM_ADDR x, GM_ADDR out, GM_ADDR dz, GM_ADDR out_z,
    GM_ADDR du, GM_ADDR ddelta, GM_ADDR tiling, void* stream);
void launch_selective_scan_bwd_half(uint32_t numBlocks,
    GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR dout,
    GM_ADDR D, GM_ADDR z, GM_ADDR delta_bias,
    GM_ADDR x, GM_ADDR out, GM_ADDR dz, GM_ADDR out_z,
    GM_ADDR du, GM_ADDR ddelta, GM_ADDR tiling, void* stream);
void launch_selective_scan_bwd_bfloat16(uint32_t numBlocks,
    GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR dout,
    GM_ADDR D, GM_ADDR z, GM_ADDR delta_bias,
    GM_ADDR x, GM_ADDR out, GM_ADDR dz, GM_ADDR out_z,
    GM_ADDR du, GM_ADDR ddelta, GM_ADDR tiling, void* stream);
}
#endif
