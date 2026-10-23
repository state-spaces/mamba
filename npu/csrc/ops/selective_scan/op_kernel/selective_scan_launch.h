#ifndef SELECTIVE_SCAN_LAUNCH_H
#define SELECTIVE_SCAN_LAUNCH_H
#include <cstdint>
#include <tuple>
#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// 特性标志位掩码（kernel / plugin 共享；plugin 计算后传入 kernel）
static constexpr uint32_t SSF_FLAG_HAS_D     = 1u << 0;  // D 跳连提供
static constexpr uint32_t SSF_FLAG_HAS_Z     = 1u << 1;  // z 门控提供（需写 out_z）
static constexpr uint32_t SSF_FLAG_HAS_BIAS  = 1u << 2;  // delta_bias 提供
static constexpr uint32_t SSF_FLAG_SOFTPLUS  = 1u << 3;  // delta_softplus=true
static constexpr uint32_t SSF_FLAG_VAR_B     = 1u << 4;  // 变量 B
static constexpr uint32_t SSF_FLAG_VAR_C     = 1u << 5;  // 变量 C

// 标量参数聚合结构（kernel / plugin 共享，纯 C++ 语法）。
// 经 GM tiling 缓冲传入 kernel：直调 launch 链路（<<<>>> 降级的 eager-launch）
// 实测对 >11 参签名存在中段标量参数损坏（指针区完好），故标量统一走 GM tiling，
// kernel 签名仅保留指针参数（全部落在实测完好的参数区）。
//
// dav-2201 绕行（问题 6）：delta_bias 的 GM 原张量不可被本 kernel 以任何引擎访问
//（MTE2 DataCopyPad / 标量 GetValue 均触发 aivec 507035，地址组合依赖、与数据/
// 指令流无关；详见 doc/mamba仓NPU跑测问题与解决方案.md 问题 6）。传给 kernel 的
// delta_bias 指针须为 plugin 侧纯 V 算子生成的安全副本（见 plugin ssf_launch），
// 原 bias 张量对 kernel 完全不可见。
struct SsfTilingData {
    int64_t batch;
    int64_t dim;
    int64_t seqlen;
    int64_t dstate;
    int64_t dimPerGroup;
    int64_t nChunks;
    int64_t numPairs;
    int64_t pairsPerCore;
    uint32_t flags;
};

// Tiling 参数计算（kernel 侧定义，plugin 调用；返回 usedCores, pairsPerCore）
std::tuple<int64_t, int64_t> calc_selective_scan_tiling_params(int64_t batch, int64_t dim);

// GM 指针顺序：u, delta, A, B, C, D, z, delta_bias, out, x, out_z, tiling（12 个）
// blockDim 由 plugin 按 ceil(numPairs/pairsPerCore) 计算后显式传入
extern "C" {
void launch_selective_scan_fwd_float(uint32_t numBlocks,
    GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C,
    GM_ADDR D, GM_ADDR z, GM_ADDR delta_bias,
    GM_ADDR out, GM_ADDR x, GM_ADDR out_z, GM_ADDR tiling, void* stream);
void launch_selective_scan_fwd_half(uint32_t numBlocks,
    GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C,
    GM_ADDR D, GM_ADDR z, GM_ADDR delta_bias,
    GM_ADDR out, GM_ADDR x, GM_ADDR out_z, GM_ADDR tiling, void* stream);
void launch_selective_scan_fwd_bfloat16(uint32_t numBlocks,
    GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C,
    GM_ADDR D, GM_ADDR z, GM_ADDR delta_bias,
    GM_ADDR out, GM_ADDR x, GM_ADDR out_z, GM_ADDR tiling, void* stream);
}
#endif
