/**
 * selective_scan_fwd 算子 - Kernel + Tiling + Launch
 * 编译：bisheng --npu-arch=dav-2201 -xasc -fPIC
 *
 * 方案：.cannbot/selective_scan/2.2-开发方案设计.md（round 2 修订版）
 *  - MemBase SIMD；x 契约 chunk=2048 与处理单元 TILE_T=1024 解耦（R6 平台绕行：
 *    vgather count>1024 间歇损坏，禁止任何使 Gather count>1024 的改动，见方案 6.1/§10）
 *  - 每 chunk 由 ≤2 个 TILE_T=1024 tile 串行完成，chunk-carry 逐 tile 携带运行前缀；
 *    out/out_z 按 tile 写出，x 仅在 2048 边界（每 chunk 末）写出
 *  - Hillis-Steele 倍增扫描（10 步，d=1..512；count=tileValidLen≤1024）
 *  - 绕行：d∈{1,2,4} 用 Gather 移位（srcBaseAddr 字节基址）、d≥8 用 operator[] 偏移视图
 *  - softplus 等价式 Δ=Max(Δ, Log(Exp(Min(Δ,20))+1))；a_t=Exp(Δ·A)（dav-2201 Exp/Log 均为 base-e 实测）
 *  - 队列深度统一 2（R6 修订后 UB 预算宽松，fp32 varB/C 深度 1 分支取消；UB 兜底预案为 varB/C 降深 1）
 *
 * 性能优化项落地（tile 粒度，R6 修订后回修）：O1（u/delta/z/varB/C tile 级双缓冲）、
 * O2（段级公共量每 tile 一次算 + outAcc 每 tile 重置、常驻跨 s 累加）、
 * O3（A/静态B/C 每 pair 仅载入一次 + xBuild 跨 tile 携带仅 chunk 末单次写出 +
 * Gather 索引每核一次）、O4（扫描步 d≥tileValidLen 跳过 + 越界 tile 整体跳过）、
 * O5（varB/C 队列统一深度 2，状态行 s 顶预取）。
 * 性能迭代（标量维，逐轮效果与负结果见 .cannbot/selective_scan/性能迭代记录.md）：
 * R1 = O6 乘加融合——隔离实验定稿：仅 varC 输出累加融合为 MulAddDst（outAcc += h·C 单条
 * API，消中间 tmpA 往返）；前缀衔接与静态 C 输出实测 Axpy 在 fp32 劣化，维持 Muls+Add；
 * 扫描步 Mul+Add 的 3→2 融合经分析不可达（MulAddDst 需 dst 预置，ping-pong 下等价于原 2 条；
 * 原地 FMA 有跨 repeat 读旧值冒险，禁用）。
 * R2 = O7 结构性标量链优化——①扫描起始侧固定 (a0,b0)，删除 per-s ping-pong 侧交替
 * （5a/5b 每 s 全量重写起始侧数据区、pad 区一次性初始化，数值中性）；②相位拆分：
 * dd∈{1,2,4} Gather 步直线化 + dd≥8 对齐步 2 步展开（ScanStepGather/ScanStepAligned 助手，
 * cur→next 侧序实参硬编码，操作序列与原实现逐指令一致 = 位级等价）；③tile 级不变量外提
 * （a0D/a1D/b0D/b1D 数据区视图 + bRowV/cRowV/varB32V/varC32V 条件 Get，P4/P9）；
 * ④totalSteps 每 tile 一算，终态 curIsA0 单次选择 aFin/bFin——消 12 assignments/step 的
 * ping-pong 交换与 8 selects/s（标量链热点，静态审计估算 100-500 ops/s）。
 */
#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "selective_scan_launch.h"

namespace {

constexpr int64_t TILE_T = 1024;     // 处理单元宽度（R6 绕行决策：Gather count≤1024 实测稳健域，方案 6.1，固定常量）
constexpr int64_t PAD_LANES = 512;   // 扫描缓冲左填充 = TILE_T/2
constexpr int64_t BUF_T = TILE_T + PAD_LANES;
constexpr int64_t CHUNK_T = 2048;    // x 契约 chunk 宽度（与 TILE_T 解耦：chunk = 2×tile，方案 2.c，契约不变）
constexpr int64_t TILES_PER_CHUNK = CHUNK_T / TILE_T;  // = 2
constexpr int64_t SSF_DSTATE_MAX = 256;  // dstate 上界（方案 §接口约束，plugin 校验兜底）
// 注：dav-2201 的 Exp/Log 均为 base-e 实测语义，无需 log2e 预乘常量
// R6 禁改约束（方案 §10）：任何回修/优化/兜底不得使 Gather count 超过 1024（含 TILE_T 回大）

}  // namespace

template <typename INPUT_T>
class SelectiveScanFwdKernel {
public:
    __aicore__ inline SelectiveScanFwdKernel() {}

    __aicore__ inline void Init(GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                GM_ADDR D, GM_ADDR z, GM_ADDR deltaBias,
                                GM_ADDR out, GM_ADDR x, GM_ADDR outZ,
                                const __gm__ SsfTilingData *tp)
    {
        // 标量参数经 GM tiling 缓冲传入（直调 launch 链路对中段标量参数有损坏，见 launch.h）
        batch_ = tp->batch; dim_ = tp->dim; seqlen_ = tp->seqlen; dstate_ = tp->dstate;
        dimPerGroup_ = tp->dimPerGroup; nChunks_ = tp->nChunks; numPairs_ = tp->numPairs;
        pairsPerCore_ = tp->pairsPerCore; flags_ = tp->flags;
        hasD_ = (flags_ & SSF_FLAG_HAS_D) != 0;
        hasZ_ = (flags_ & SSF_FLAG_HAS_Z) != 0;
        hasBias_ = (flags_ & SSF_FLAG_HAS_BIAS) != 0;
        softplus_ = (flags_ & SSF_FLAG_SOFTPLUS) != 0;
        varB_ = (flags_ & SSF_FLAG_VAR_B) != 0;
        varC_ = (flags_ & SSF_FLAG_VAR_C) != 0;

        // 1. GlobalTensor 绑定
        uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(u));
        deltaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(delta));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A));
        // B/C 按各自变体独立绑定（四组合均合法；错位绑定会使静态行拷贝读到未绑定 GM → MTE 故障）
        if (varB_) {
            varBGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(B));
        } else {
            bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(B));
        }
        if (varC_) {
            varCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(C));
        } else {
            cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(C));
        }
        if (hasD_) {
            dGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(D));
        }
        if (hasZ_) {
            zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(z));
            outZGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(outZ));
        }
        if (hasBias_) {
            biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(deltaBias));
        }
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(out));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));

        // 2. InitBuffer——分配顺序说明：小缓冲（标量访问 + 行缓冲）与扫描缓冲靠前，
        //    队列与纯向量 scratch 靠后；mssanitizer 的 UB 模型对高地址（≥~120KB）访问
        //    存在误报，低地址布局可减少误报干扰。
        const uint32_t inBytes = static_cast<uint32_t>(TILE_T * sizeof(INPUT_T));
        pipe_.InitBuffer(xBuildBuf_, 2 * SSF_DSTATE_MAX * sizeof(float));
        pipe_.InitBuffer(aRowBuf_, SSF_DSTATE_MAX * sizeof(float));
        if (!varB_) {
            pipe_.InitBuffer(bRowBuf_, SSF_DSTATE_MAX * sizeof(float));
        }
        if (!varC_) {
            pipe_.InitBuffer(cRowBuf_, SSF_DSTATE_MAX * sizeof(float));
        }
        pipe_.InitBuffer(sTmpBuf_, 64);  // D 单元素标量中转（bias 已改整行块搬运，见 Process）
        if (hasBias_) {
            // dav-2201 绕行（问题 6）：每 pair 的 4B 单元素 DataCopyPad 触发 MTE 时序缺陷
            //（fixp_error=mte 镜像证据；7_noD 崩 / 7_noBias 过），bias 改每核一次整行块搬运
            pipe_.InitBuffer(biasRowBuf_, static_cast<uint32_t>(dim_ * sizeof(float)));
        }
        pipe_.InitBuffer(gatherIdxBuf_, TILE_T * sizeof(uint32_t));
        pipe_.InitBuffer(a0Buf_, BUF_T * sizeof(float));
        pipe_.InitBuffer(a1Buf_, BUF_T * sizeof(float));
        pipe_.InitBuffer(b0Buf_, BUF_T * sizeof(float));
        pipe_.InitBuffer(b1Buf_, BUF_T * sizeof(float));
        // R3-O9（跨 s 软流水）：5a/5b 提前发射侧的独立起始缓冲——**两套交替**
        //（aPreA/bPreA + aPreB/bPreB）：s 轮扫描 cur 读 X 组时，emitAB(s+1) 写 Y 组，
        // s+1 轮反转。单套会在 V 按序发射下被 emitAB(s+1) 先写覆盖（扫描读到 s+1
        // 的结果，实测 63/80 FAIL 根因），双套彻底解耦读/写时序
        pipe_.InitBuffer(aPreABuf_, BUF_T * sizeof(float));
        pipe_.InitBuffer(bPreABuf_, BUF_T * sizeof(float));
        pipe_.InitBuffer(aPreBBuf_, BUF_T * sizeof(float));
        pipe_.InitBuffer(bPreBBuf_, BUF_T * sizeof(float));
        pipe_.InitBuffer(inQu_, 2, inBytes);      // O1: chunk 级双缓冲
        pipe_.InitBuffer(inQd_, 2, inBytes);      // O1
        if (hasZ_) {
            pipe_.InitBuffer(zStageBuf_, TILE_T * sizeof(INPUT_T));  // z 内联搬入暂存（弃 TQue，S-C）
        }
        if (varB_) {
            pipe_.InitBuffer(inQvarB_, 2, inBytes);  // 队列深度统一 2（R6 修订：fp32 深度 1 分支取消）
        }
        if (varC_) {
            pipe_.InitBuffer(inQvarC_, 2, inBytes);  // 同上
        }
        pipe_.InitBuffer(u32Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(d32Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(du32Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(tmpABuf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(outAccBuf_, TILE_T * sizeof(float));
        if (varB_) {
            pipe_.InitBuffer(varB32Buf_, TILE_T * sizeof(float));
        }
        if (varC_) {
            pipe_.InitBuffer(varC32Buf_, TILE_T * sizeof(float));
        }
        // 2.5 事件 ID 经 TPipe 独占分配（对齐官方 InitOutput 范式；硬编码 ID 有与
        //     TQue 内部 FetchEventID 动态分配冲突的风险）
        evV2Mte3_ = static_cast<uint8_t>(pipe_.AllocEventID<AscendC::HardEvent::V_MTE3>());
        evMte3V_ = static_cast<uint8_t>(pipe_.AllocEventID<AscendC::HardEvent::MTE3_V>());
        evV2S_ = static_cast<uint8_t>(pipe_.AllocEventID<AscendC::HardEvent::V_S>());
        if (hasZ_) {
            evMte2V_ = static_cast<uint8_t>(pipe_.AllocEventID<AscendC::HardEvent::MTE2_V>());
        }

        // 3. 扫描缓冲填充区中性元初始化（一次写永久生效：数据区写入只落 [PAD, PAD+count)）
        //    a 侧 pad=1.0、b 侧 pad=0.0 —— Hillis-Steele 左边界 i<dd 时移位读命中填充区，保持恒等
        //    dav-2201 绕行（问题 6）：高频标量 UB 写（SetValue）在标量/向量 pipe 交错时存在
        //    时序型停驻风险（0xcc8 "scalar 访问内部 buffer 越界"），纯初始化写全部改向量 Duplicate
        auto a0v = a0Buf_.Get<float>();
        auto a1v = a1Buf_.Get<float>();
        auto b0v = b0Buf_.Get<float>();
        auto b1v = b1Buf_.Get<float>();
        AscendC::Duplicate(a0v, 1.0f, static_cast<uint32_t>(PAD_LANES));
        AscendC::Duplicate(a1v, 1.0f, static_cast<uint32_t>(PAD_LANES));
        AscendC::Duplicate(b0v, 0.0f, static_cast<uint32_t>(PAD_LANES));
        AscendC::Duplicate(b1v, 0.0f, static_cast<uint32_t>(PAD_LANES));
        // R3-O9：pre 双套 pad 区同语义初始化（数据区每 s 全量重写覆盖遗留值）
        auto aPreAv = aPreABuf_.Get<float>();
        auto bPreAv = bPreABuf_.Get<float>();
        auto aPreBv = aPreBBuf_.Get<float>();
        auto bPreBv = bPreBBuf_.Get<float>();
        AscendC::Duplicate(aPreAv, 1.0f, static_cast<uint32_t>(PAD_LANES));
        AscendC::Duplicate(bPreAv, 0.0f, static_cast<uint32_t>(PAD_LANES));
        AscendC::Duplicate(aPreBv, 1.0f, static_cast<uint32_t>(PAD_LANES));
        AscendC::Duplicate(bPreBv, 0.0f, static_cast<uint32_t>(PAD_LANES));
        // 4. Gather 字节偏移索引 idx[i]=i·4（每核一次，跨 pair/跨 chunk 复用，O3）
        //    dav-2201 绕行：TILE_T=1024 次标量写 → Arange 等差生成（内部 8 次标量 + 向量扩展）。
        //    经 int32 视图写入（官方支持类型），值域 [0,4092] 全正，位模式与 uint32 逐位一致
        auto idx = gatherIdxBuf_.Get<int32_t>();
        AscendC::Arange(idx, 0, 4, static_cast<int32_t>(TILE_T));
        AscendC::PipeBarrier<PIPE_ALL>();  // 每核一次的构建汇聚点（标量写 → 后续 V 侧读）
    }

    __aicore__ inline void Process()
    {
        const int64_t coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t start = coreIdx * pairsPerCore_;
        const int64_t end = (start + pairsPerCore_ < numPairs_) ? (start + pairsPerCore_) : numPairs_;
        if (hasBias_ && start < end) {
            // dav-2201 绕行（问题 6）：bias 每核一次整行块搬运（dim·4B，GM 侧整行无越界），
            // 替代每 pair 的 4B 单元素 DataCopyPad；MTE2 内序 barrier 后 pair 内标量回读
            //（PIPE_MTE2 屏障在 dav-2201 AIV 上真实发射——elide 守卫仅覆盖 3003/3113 与
            // AIC 侧 PIPE_V，见 SyncVToScalar 注释；且首对 tile 的 TQue DeQue 内部事件
            // 等待先于 bias 回读，MTE2 队列保序使本读必然晚于搬入完成）
            AscendC::DataCopyExtParams biasCp{1, static_cast<uint32_t>(dim_ * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(biasRowBuf_.Get<float>(), biasGm_[0], biasCp,
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        for (int64_t p = start; p < end; ++p) {
            ProcessPair(p);
        }
    }

private:
    // ---------- pair 级：一条 (b,d) 序列的全部 chunk ----------
    __aicore__ inline void ProcessPair(int64_t p)
    {
        const int64_t b = p / dim_;
        const int64_t d = p - b * dim_;
        pairUBase_ = p * seqlen_;            // u/delta/out 行基址（元素）
        pairXBase_ = p * nChunks_ * 2 * dstate_;  // x 行基址（元素，fp32）

        // 运行前缀存储初始化：(A_run, B_run) 的单位元语义经 gt==0 首 tile 分支承载
        //（见 ProcessTile 5d），xBuild 仅需全 0 起始（B_run 加法单位元 + gt==0 时
        // h=P·B_run+Q 的 B_run=0）。不用位图 mask 多 repeat 写偶 lane 1.0：
        // float 位图 mask 每 repeat 仅低 32 lane 生效，ds 大时 A_run 槽成片漏写
        //（L0-F-14 实测 mr=0.79），且 repeat 数×stride 存在 UB 越界写风险
        auto xBuild = xBuildBuf_.Get<float>();
        const int64_t xLanes = dstate_ * 2;
        AscendC::Duplicate(xBuild, 0.0f, static_cast<uint32_t>(xLanes));

        // O3: A / 静态B / 静态C 行每 pair 仅载入一次（dstate fp32 ≤ 1KB；禁用 GlobalTensor::GetValue 红线 B2）
        auto aRow = aRowBuf_.Get<float>();
        AscendC::DataCopyExtParams rowParams{1, static_cast<uint32_t>(dstate_ * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(aRow, aGm_[d * dstate_], rowParams,
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        if (!varB_) {
            AscendC::DataCopyPad(bRowBuf_.Get<float>(), bGm_[d * dstate_], rowParams,
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        if (!varC_) {
            AscendC::DataCopyPad(cRowBuf_.Get<float>(), cGm_[d * dstate_], rowParams,
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        }

        // D 单元素标量中转（每 pair 一次；bias 已改 Process 开头整行搬运绕行，见问题 6）
        if (hasD_) {
            AscendC::DataCopyExtParams oneParams{1, sizeof(float), 0, 0, 0};
            AscendC::DataCopyPad(sTmpBuf_.Get<float>()[0], dGm_[d], oneParams,
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        AscendC::PipeBarrier<PIPE_MTE2>();  // MTE2 搬入完成 → 标量 GetValue 回读（真实发射，同 Process 内 bias 站点注释）

        // O1: tile 级双缓冲——首 tile 搬入提前发射，主循环内发射下一 tile
        // chunk=2048 与 TILE_T=1024 解耦（方案 2.c）：每 chunk 由 ≤2 个 tile 串行完成，
        // chunk-carry 经 xBuild 逐 tile 携带，x 仅在 chunk 末（2048 边界）写出
        const int64_t totalTiles = (seqlen_ + TILE_T - 1) / TILE_T;
        CopyInTile(0);
        int64_t gt = 0;  // 全局 tile 序号（u/delta/out 行内偏移 = gt·TILE_T）
        for (int64_t c = 0; c < nChunks_; ++c) {
            const int64_t rest = seqlen_ - c * CHUNK_T;
            const int64_t validLenChunk = (rest < CHUNK_T) ? rest : CHUNK_T;
            const int64_t tiles = (validLenChunk + TILE_T - 1) / TILE_T;  // ≤2；越界 tile 整体跳过（O4）
            for (int64_t t = 0; t < tiles; ++t) {
                const int64_t rem = validLenChunk - t * TILE_T;
                const int64_t tileValidLen = (rem < TILE_T) ? rem : TILE_T;
                if (gt + 1 < totalTiles) {
                    CopyInTile(gt + 1);  // O1: 与本 tile 计算重叠的搬入
                }
                ProcessTile(c, gt, tileValidLen, t == tiles - 1, b, d);
                ++gt;
            }
        }
    }

    // ---------- tile 级搬入（O1 提前发射；尾块 DataCopyPad 按 tileValidLen 搬入，UB 侧由 padValue 补齐） ----------
    __aicore__ inline void CopyInTile(int64_t gt)
    {
        const int64_t off = pairUBase_ + gt * TILE_T;
        // GM 侧 blockLen 必须等于本 tile 有效字节数：seqlen 非整块时按 TILE_T 搬入会越界读 GM（MTE 异常）
        const int64_t valid = seqlen_ - gt * TILE_T;
        const uint32_t bytes = static_cast<uint32_t>((valid < TILE_T ? valid : TILE_T) * sizeof(INPUT_T));
        AscendC::DataCopyExtParams cp{1, bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<INPUT_T> pad{true, 0, 0, 0};

        auto uLocal = inQu_.AllocTensor<INPUT_T>();
        AscendC::DataCopyPad(uLocal, uGm_[off], cp, pad);
        inQu_.EnQue(uLocal);

        auto dLocal = inQd_.AllocTensor<INPUT_T>();
        AscendC::DataCopyPad(dLocal, deltaGm_[off], cp, pad);
        inQd_.EnQue(dLocal);
        // z 不在 O1 预取队列（事件池容量约束，见 inQz_ 声明处说明），改 step7 内联搬入
    }

    // ---------- 变量 B/C 状态行搬入（s 级，tile 粒度；队列深度统一 2，O5） ----------
    template <typename Q>
    __aicore__ inline void CopyInVarRow(Q &q, AscendC::GlobalTensor<INPUT_T> &gm,
                                        int64_t b, int64_t d, int64_t s, int64_t gt)
    {
        const int64_t g = d / dimPerGroup_;
        const int64_t off = ((b * (dim_ / dimPerGroup_) + g) * dstate_ + s) * seqlen_ + gt * TILE_T;
        // 同 CopyInTile：GM 侧 blockLen 按本 tile 有效字节数，防 seqlen 非整块时越界读
        const int64_t valid = seqlen_ - gt * TILE_T;
        const uint32_t bytes = static_cast<uint32_t>((valid < TILE_T ? valid : TILE_T) * sizeof(INPUT_T));
        AscendC::DataCopyExtParams cp{1, bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<INPUT_T> pad{true, 0, 0, 0};
        auto local = q.template AllocTensor<INPUT_T>();
        AscendC::DataCopyPad(local, gm[off], cp, pad);
        q.EnQue(local);
    }

    // ---------- tile 级主计算（方案 2.a 核内流程 1–8 步；chunk=2×tile 串行，R6 绕行） ----------
    // c: x 契约 chunk 序号（x 列偏移用）；gt: 全局 tile 序号（u/delta/out 行内偏移用）；
    // tileValidLen: 本 tile 有效步数 ≤1024（R6：Gather count 上限）；chunkEnd: 本 tile 是否 chunk 末（x 写出点）
    __aicore__ inline void ProcessTile(int64_t c, int64_t gt, int64_t tileValidLen, bool chunkEnd,
                                       int64_t b, int64_t d)
    {
        const int32_t count = static_cast<int32_t>(tileValidLen);
        const int64_t tileUOff = pairUBase_ + gt * TILE_T;

        if (gt % TILES_PER_CHUNK == 1) {
            // 前驱为 chunk 内非末 tile（其尾部已 SetFlag<MTE3_V>）：本 tile V 侧复用 du32/tmpA 前
            // 必须等其 out/out_z 的 MTE3 读取完成（SYNC-02/06：PipeBarrier<PIPE_MTE3> 不跨 PIPE）
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
        }

        // ---- 1. CopyIn 消费 + Cast 到 fp32（A/B/C 固定 fp32、x 输出 fp32 契约） ----
        auto uLocal = inQu_.DeQue<INPUT_T>();
        auto dLocal = inQd_.DeQue<INPUT_T>();
        auto u32 = u32Buf_.Get<float>();
        auto d32 = d32Buf_.Get<float>();
        InputCast(u32, uLocal, count);
        InputCast(d32, dLocal, count);
        inQu_.FreeTensor(uLocal);
        inQd_.FreeTensor(dLocal);

        // ---- 2. delta_bias（hasBias；经整行块搬运的 biasRow，问题 6 绕行） ----
        if (hasBias_) {
            const float biasVal = biasRowBuf_.Get<float>().GetValue(static_cast<uint32_t>(d));
            AscendC::Adds(d32, d32, biasVal, count);
        }

        // ---- 3. softplus 等价式（softplus 标志）：Δ = Max(Δ, Log(Exp(Min(Δ,20))+1)) ----
        auto tmpA = tmpABuf_.Get<float>();
        auto b0full = b0Buf_.Get<float>();
        if (softplus_) {
            // S-V-07 修复：钳位值写独立 scratch（du32），不原地 Min 覆盖 d32——
            // 原实现 Min(d32,d32,tmpA) 破坏原始 Δ，Δ>20 时外层 Max 取钳位值
            // 20.0000000021 而非原值，softplus(x)=Max(x,·) 的全域等价失效。
            // 复用核对：du32 在步骤 4 被 Mul(du32,d32,u32) 整体覆写，与本块无读依赖；
            // du32Buf_/d32Buf_/tmpABuf_ 三 TBuf 互不重叠，无别名风险。
            auto dClamp = du32Buf_.Get<float>();
            AscendC::Duplicate(tmpA, 20.0f, count);
            AscendC::Min(dClamp, d32, tmpA, count);       // 钳位 = Min(Δ,20)，d32 保留原始 Δ
            AscendC::Exp(tmpA, dClamp, count);            // e^min(Δ,20)（≤20 无溢出）
            AscendC::Adds(tmpA, tmpA, 1.0f, count);       // 原地 +1
            AscendC::Log(b0full[PAD_LANES], tmpA, count); // ln(1+e^min(Δ,20))（dst=b0 数据区，与 src 不重叠）
            AscendC::Max(d32, d32, b0full[PAD_LANES], count); // 原地 Max（src0=原始 Δ，全域等价）
        }

        // ---- 4. 段级公共量（O2：s 循环外提）：Δu 与 D 跳连初值 ----
        auto du32 = du32Buf_.Get<float>();
        auto outAcc = outAccBuf_.Get<float>();
        AscendC::Mul(du32, d32, u32, count);
        if (hasD_) {
            const float dVal = sTmpBuf_.Get<float>().GetValue(0);
            AscendC::Muls(outAcc, u32, dVal, count);
        } else {
            AscendC::Duplicate(outAcc, 0.0f, count);
        }

        // u32 在 Δu/D 跳连后不再使用；hasZ 时其 TBuf 复用为 z32（3.2 复用说明）
        auto z32 = u32;

        // ---- 5. 逐状态扫描 + 前缀衔接 + 输出累加 ----
        auto xBuild = xBuildBuf_.Get<float>();
        auto aRow = aRowBuf_.Get<float>();
        auto a0 = a0Buf_.Get<float>();
        auto a1 = a1Buf_.Get<float>();
        auto b1 = b1Buf_.Get<float>();
        auto idx = gatherIdxBuf_.Get<uint32_t>();

        // 变量 B/C 状态行流水（O5：队列深度统一 2，s 循环顶预取；行偏移按 tile 粒度）
        if (varB_) {
            CopyInVarRow(inQvarB_, varBGm_, b, d, 0, gt);
        }
        if (varC_) {
            CopyInVarRow(inQvarC_, varCGm_, b, d, 0, gt);
        }

        // R2 标量链优化：tile 级不变量外提（P4 视图复用 / P9 标量减少）——
        //  - 数据区视图 a0D/a1D/b0D/b1D 一次构造：起始侧固定 (a0,b0) 后不再 per-s 翻转；
        //  - bRowV/cRowV/varB32V/varC32V 条件 Get（对应 InitBuffer 条件分配，不可无条件 Get），
        //    消 per-s 重复 Get 句柄构造；
        //  - totalSteps 与原 dd 循环步数定义一致（dd<TILE_T && dd<tileValidLen 的迭代次数），
        //    终态侧 curIsA0 每 tile 一算，aFin/bFin 单次选择——消 per-s 8 次 selects 与
        //    每步 6 组 ping-pong 交换赋值（12 assignments/step 的标量链热点）。
        auto a0D = a0[PAD_LANES];   // 数据区视图（2048B 对齐）
        auto a1D = a1[PAD_LANES];
        auto b0D = b0full[PAD_LANES];
        auto b1D = b1[PAD_LANES];
        // R3-O9：提前发射侧起始缓冲视图（双套 A/B 按 s 奇偶交替：s 轮扫描 cur 读
        // X 组时 emitAB(s+1) 写 Y 组；pad 区初始化 1.0/0.0，同 a0/b0 语义）
        auto aPreA = aPreABuf_.Get<float>();
        auto bPreA = bPreABuf_.Get<float>();
        auto aPreB = aPreBBuf_.Get<float>();
        auto bPreB = bPreBBuf_.Get<float>();
        auto aPreAD = aPreA[PAD_LANES];
        auto bPreAD = bPreA[PAD_LANES];
        auto aPreBD = aPreB[PAD_LANES];
        auto bPreBD = bPreB[PAD_LANES];
        AscendC::LocalTensor<float> bRowV, cRowV, varB32V, varC32V;
        if (!varB_) {
            bRowV = bRowBuf_.Get<float>();
        }
        if (!varC_) {
            cRowV = cRowBuf_.Get<float>();
        }
        if (varB_) {
            varB32V = varB32Buf_.Get<float>();
        }
        if (varC_) {
            varC32V = varC32Buf_.Get<float>();
        }
        const bool runD1 = tileValidLen > 1;   // A 相 Gather 步守卫（= 原循环 dd=1/2/4 条件）
        const bool runD2 = tileValidLen > 2;
        const bool runD4 = tileValidLen > 4;
        int64_t totalSteps = 0;
        for (int64_t dd = 1; dd < TILE_T && dd < tileValidLen; dd <<= 1) {
            ++totalSteps;
        }
        const bool curIsA0 = (totalSteps & 1) == 0;
        auto aFin = curIsA0 ? a0D : a1D;   // 终态扫描结果视图（tile 内 s 不变）
        auto bFin = curIsA0 ? b0D : b1D;

        // R3-O9（跨 s 软流水）：5a+5b 的输出落独立 aPreD/bPreD（与 a0/b0/a1/b1
        // ping-pong 区、5c 读写集不相交），故 s+1 的 5a/5b 可在本 s 标量段前发射，
        // 填充 SyncVToScalar 强制 V 排空 + 标量回读段的 V idle。
        // 指令序列与原 5a/5b 逐条一致（仅目标缓冲改名）= 位级等价。
        // 注：bisheng 不支持 __aicore__ 函数内 lambda（判为 __host__），5a/5b 以
        // 直排代码块在两处发射点复用（循环前 s=0 + 循环内 s+1）。
        // 5a. a_cur = Exp(Δ · A[s])（实测 dav-2201 Exp 为 base-e；数值防御钳 -87.3
        //     防 denormal（问题 6 定稿），证据与容差论证见历史记录）
        // 5b. b_cur = B_t · Δu（静态标量 / 变量行）
        // —— s=0 发射（原 s=0 轮开头位置语义不变）
        {
            const float Aval = aRow.GetValue(static_cast<uint32_t>(0));
            AscendC::Muls(tmpA, d32, Aval, count);
            AscendC::Maxs(tmpA, tmpA, -87.3f, count);
            AscendC::Exp(aPreAD, tmpA, count);
            if (varB_) {
                auto vB = inQvarB_.template DeQue<INPUT_T>();
                InputCast(varB32V, vB, count);
                inQvarB_.FreeTensor(vB);
                AscendC::Mul(bPreAD, varB32V, du32, count);
            } else {
                const float bVal = bRowV.GetValue(static_cast<uint32_t>(0));
                AscendC::Muls(bPreAD, du32, bVal, count);
            }
        }

        for (int64_t s = 0; s < dstate_; ++s) {
            // 状态行提前发射（O5：深度 2，隐藏 MTE 延迟）
            if (varB_ && s + 1 < dstate_) {
                CopyInVarRow(inQvarB_, varBGm_, b, d, s + 1, gt);
            }
            if (varC_ && s + 1 < dstate_) {
                CopyInVarRow(inQvarC_, varCGm_, b, d, s + 1, gt);
            }

            // R3-O9：s+1 的 5a/5b 提前发射，写 side(s+1) 组（与本 s 扫描 cur 读的
            // side(s) 组分离，双套交替解耦读写时序；依赖 d32/du32 恒定 + aRow/bRowV
            // 常驻；发射后 V 流水在本 s 标量段仍具在飞指令）
            if (s + 1 < dstate_) {
                auto aW = ((s + 1) & 1) == 0 ? aPreAD : aPreBD;
                auto bW = ((s + 1) & 1) == 0 ? bPreAD : bPreBD;
                const float Aval = aRow.GetValue(static_cast<uint32_t>(s + 1));
                AscendC::Muls(tmpA, d32, Aval, count);
                AscendC::Maxs(tmpA, tmpA, -87.3f, count);
                AscendC::Exp(aW, tmpA, count);
                if (varB_) {
                    auto vB = inQvarB_.template DeQue<INPUT_T>();
                    InputCast(varB32V, vB, count);
                    inQvarB_.FreeTensor(vB);
                    AscendC::Mul(bW, varB32V, du32, count);
                } else {
                    const float bVal = bRowV.GetValue(static_cast<uint32_t>(s + 1));
                    AscendC::Muls(bW, du32, bVal, count);
                }
            }

            // 5c. Hillis-Steele 倍增扫描（10 步上限，d=1..512；O4: d ≥ tileValidLen 的步跳过）
            //     R6 禁改约束（方案 6.1/§10）：d∈{1,2,4} Gather 步 count=tileValidLen ≤1024
            //     R2 相位拆分 + R3-O9：起始 cur 侧改 (preA/preB[s])（本 s 的 5a/5b 产出），
            //     后续 ping-pong 侧序与原实现完全一致（a0/b0 ↔ a1/b1），操作序列逐指令一致
            {
                const bool curA = (s & 1) == 0;
                auto aCurPreF = curA ? aPreA : aPreB;   // full 视图（Gather 源含 pad 区）
                auto bCurPreF = curA ? bPreA : bPreB;
                auto aCurPreD = curA ? aPreAD : aPreBD; // 数据区视图
                auto bCurPreD = curA ? bPreAD : bPreBD;
                if (runD1) {
                    ScanStepGather(1, count, aCurPreF, bCurPreF, aCurPreD, bCurPreD, a1D, b1D, idx);
                }
                if (runD2) {
                    ScanStepGather(2, count, a1, b1, a1D, b1D, a0D, b0D, idx);
                }
                if (runD4) {
                    ScanStepGather(4, count, a0, b0full, a0D, b0D, a1D, b1D, idx);
                }
                int64_t dd = 8;
                while (dd < TILE_T && dd < tileValidLen) {
                    ScanStepAligned(dd, count, a1, b1, a1D, b1D, a0D, b0D);
                    const int64_t dd2 = dd << 1;
                    // 偶步内层守卫：与原循环逐步条件一致（dd2 可能已越过 tileValidLen）
                    if (dd2 < TILE_T && dd2 < tileValidLen) {
                        ScanStepAligned(dd2, count, a0, b0full, a0D, b0D, a1D, b1D);
                    }
                    dd = dd2 << 1;
                }
            }

            // 5d. 前缀衔接 + 运行前缀标量更新
            // 终态侧选择（R3-O9 修正）：totalSteps≥1 时扫描在 ping-pong 侧（a0/b0 ↔
            // a1/b1）进行，终态 = curIsA0 决定的固定侧（tile 内 s 不变，R2）；**totalSteps==0
            // （tileValidLen≤1）扫描零步，终态 = 本 s 的 pre 侧（随 s 奇偶交替）**——
            // 原 aFin 固定 ping-pong 侧的选择在零步场景读错缓冲（L0-F-16 根因）
            auto aFin = totalSteps == 0 ? (((s & 1) == 0) ? aPreAD : aPreBD)
                                        : (curIsA0 ? a0D : a1D);
            auto bFin = totalSteps == 0 ? (((s & 1) == 0) ? bPreAD : bPreBD)
                                        : (curIsA0 ? b0D : b1D);
            // F-BW18：原 PipeBarrier<PIPE_V> 对 V→标量回读无跨 PIPE 排序保证（见
            // SyncVToScalar 注释），改 V_S 事件对——覆盖本 s 轮扫描链（aFin/bFin）、
            // Exp/Muls 等全部在飞 V 指令完成后才 GetValue 回读。
            SyncVToScalar();
            const float pLast = aFin.GetValue(static_cast<uint32_t>(count - 1));
            const float qLast = bFin.GetValue(static_cast<uint32_t>(count - 1));
            const float bRunOld = xBuild.GetValue(static_cast<uint32_t>(2 * s + 1));
            // h = P_i·B_run + Q_i —— O6/R1 隔离实验定稿：前缀衔接维持 Muls+Add
            //（Axpy 实测 fp32 劣化 +2.3%，归因见性能迭代记录）
            AscendC::Muls(aFin, aFin, bRunOld, count);
            AscendC::Add(aFin, aFin, bFin, count);
            // 输出累加（O2: outAcc 常驻 UB 跨全 s 累加，无 per-s 搬运；每 tile 重置）
            // O6 乘加融合（R1 定稿）：varC 输出累加融合为 MulAddDst 单条 API（消中间
            // tmpA 往返）；静态 C 实测 Axpy 在 fp32（T-05）劣化 +2.2%，维持 Muls+Add
            if (varC_) {
                auto vC = inQvarC_.template DeQue<INPUT_T>();
                InputCast(varC32V, vC, count);
                inQvarC_.FreeTensor(vC);
                AscendC::MulAddDst(outAcc, varC32V, aFin, count);
            } else {
                const float cVal = cRowV.GetValue(static_cast<uint32_t>(s));
                AscendC::Muls(tmpA, aFin, cVal, count);
                AscendC::Add(outAcc, outAcc, tmpA, count);
            }
            // 运行前缀更新：(A,B)∘(P,Q) = (P·A, P·B + Q)，写入 xBuild
            // （chunk-carry 逐 tile 携带：chunk 内第 2 个 tile 完成后即为 2048-chunk 复合前缀）
            // gt==0 全序列首 tile：无前导前缀，(1,0)∘(P,Q) = (P,Q) 直接落值——
            // A_run 乘法单位元 1.0 的语义由此分支承载（xBuild init 仅全 0）
            if (gt == 0) {
                xBuild.SetValue(static_cast<uint32_t>(2 * s), pLast);
                xBuild.SetValue(static_cast<uint32_t>(2 * s + 1), qLast);
            } else {
                const float aRunOld = xBuild.GetValue(static_cast<uint32_t>(2 * s));
                xBuild.SetValue(static_cast<uint32_t>(2 * s), aRunOld * pLast);
                xBuild.SetValue(static_cast<uint32_t>(2 * s + 1), pLast * bRunOld + qLast);
            }
        }

        // ---- 6. out 写出（tile 粒度）：cast 目的 = du32（s 循环后空闲，z 链不碰 → 无 MTE3 读竞态） ----
        //     fp32 直出同型拷贝（Muls ×1.0），fp16/bf16 Cast(CAST_RINT)
        auto outCast = du32Buf_.Get<INPUT_T>();
        OutputCast(outCast, outAcc, count);
        // V→MTE3 事件同步（SYNC-02/06 红线：PipeBarrier<PIPE_V> 只保证 V-V 排序、不跨 PIPE，
        // MTE3 读 du32 前必须以事件等待 V 侧 cast 完成，否则可能读到陈旧值）
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
        AscendC::DataCopyExtParams outCp{1, static_cast<uint32_t>(count * sizeof(INPUT_T)), 0, 0, 0};
        AscendC::DataCopyPad(outGm_[tileUOff], outCast, outCp);

        // ---- 7. z 门控（tile 粒度）：先写 out，再 outAcc ← outAcc · silu(z) = outAcc · z · sigmoid(z) → 写 out_z ----
        if (hasZ_) {
            // z 内联搬入（TBuf + 手工 MTE2_V 事件；弃 TQue 规避 V_MTE2 事件池占用超限挂死，
            // 见队列声明处根因说明。z 消费点本就在 step7，内联仅损失 ≤2us/ tile 预取重叠）
            auto zStage = zStageBuf_.Get<INPUT_T>();
            const int64_t zValid = seqlen_ - gt * TILE_T;
            AscendC::DataCopyExtParams zCp{1, static_cast<uint32_t>((zValid < TILE_T ? zValid : TILE_T) * sizeof(INPUT_T)), 0, 0, 0};
            AscendC::DataCopyPad(zStage, zGm_[tileUOff], zCp, AscendC::DataCopyPadExtParams<INPUT_T>{true, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            InputCast(z32, zStage, count);

            auto b0d = b0full[PAD_LANES];               // b 侧扫描后空闲（作 e^{-z} 缓冲）
            AscendC::Muls(tmpA, z32, -1.0f, count);     // -z（tmpA 已被 Exp 消费完）
            AscendC::Exp(b0d, tmpA, count);             // e^{-z}（Exp dst≠src）
            AscendC::Adds(b0d, b0d, 1.0f, count);
            AscendC::Duplicate(d32, 1.0f, count);       // d32 在 s 循环后空闲（作全 1 向量）
            AscendC::Div(d32, d32, b0d, count);         // sigmoid(z)
            AscendC::Mul(d32, d32, z32, count);         // silu(z) = z · sigmoid(z)
            AscendC::Mul(outAcc, outAcc, d32, count);   // outAcc = out · silu(z)
            // out_z cast 目的 = tmpA（Exp 后空闲）
            auto outZCast = tmpABuf_.Get<INPUT_T>();
            OutputCast(outZCast, outAcc, count);
            // V→MTE3 事件同步（同 step 6：tmpA 的 MTE3 读前必须事件等待 V 侧 cast 完成；
            // 否则 MTE3 可能读到 z 门控首条 Muls(tmpA,-z) 的中间值——实测 out_z=−z 竞态复现）
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
            AscendC::DataCopyPad(outZGm_[tileUOff], outZCast, outCp);

        }

        // ---- 8. x 写出（仅 chunk 末 = 2048 边界；O3：chunk=2 tile 下严禁 per-tile 写 x）：
        //     xBuild 经 chunk 内 ≤2 个 tile 连续携带，此刻恰为 2048-chunk 复合前缀 (A_run_s, B_run_s) 交错序列 ----
        if (chunkEnd) {
            AscendC::PipeBarrier<PIPE_ALL>();  // 汇聚：标量 SetValue(xBuild) 可见 + 本 chunk out/out_z 的 MTE3 完成（du32/tmpA 复用安全）
            AscendC::DataCopyExtParams xCp{1, static_cast<uint32_t>(2 * dstate_ * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(xGm_[pairXBase_ + c * 2 * dstate_], xBuild, xCp);
            AscendC::PipeBarrier<PIPE_ALL>();  // x MTE3 完成后再进下一 chunk（xBuild 将被标量重写）
        } else {
            // 非 chunk 末 tile：置 MTE3→V 事件（SYNC-02/06 红线：PipeBarrier<PIPE_MTE3> 不跨 PIPE，
            // 下一 tile V 侧复用 du32/tmpA 前须经 WaitFlag 等本 tile out/out_z 的 MTE3 读取完成）
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
        }
    }

    // ---------- 扫描单步（性能迭代 R2 新增助手：ping-pong 侧序由调用点实参硬编码） ----------
    // cur=(aCur,aCurD/bCurD) → next=(aNextD,bNextD) 一步 Hillis-Steele 复合：
    //   b_next[i] = b_cur[i-dd]·a_cur[i] + b_cur[i]；a_next[i] = a_cur[i-dd]·a_cur[i]
    // 操作序列与原实现逐指令一致（位级等价）；每步末对产出 a 侧做下界钳制
    //（数值防御，问题 6 定稿保留）。Gather 路径（dd∈{1,2,4}，R6 绕行）源为 full
    // 视图（含 pad 区）；对齐路径（dd≥8，dd%8==0 → 32B 对齐）源为 full 视图 [PAD-dd]。
    __aicore__ inline void ScanStepGather(int64_t dd, int32_t count,
        const AscendC::LocalTensor<float> &aCur, const AscendC::LocalTensor<float> &bCur,
        const AscendC::LocalTensor<float> &aCurD, const AscendC::LocalTensor<float> &bCurD,
        const AscendC::LocalTensor<float> &aNextD, const AscendC::LocalTensor<float> &bNextD,
        const AscendC::LocalTensor<uint32_t> &idx)
    {
        const uint32_t base = static_cast<uint32_t>((PAD_LANES - dd) * sizeof(float));
        // R3-O8（V 域双发射编排）：b/a 两条独立依赖链交错排列——原实现 b 链 3 条连发
        // （Gather→Mul→Add 原地 RAW 串行）后 a 链 3 条连发，相邻指令同链依赖使 VALU 双
        // 发射窗口空转；交错后相邻指令分属不同链（无依赖），发射口可并行。链内顺序保持
        //（数值位级等价），链间无写别名（ping-pong 两侧独立 buffer）。
        AscendC::Gather(bNextD, bCur, idx, base, count);
        AscendC::Gather(aNextD, aCur, idx, base, count);
        AscendC::Mul(bNextD, bNextD, aCurD, count);   // a·b_shift（原地 dst=src0）
        AscendC::Mul(aNextD, aNextD, aCurD, count);   // a·a_shift
        AscendC::Add(bNextD, bNextD, bCurD, count);   // + b（原地累加）
        AscendC::Maxs(aNextD, aNextD, 1.1754944e-38f, count);
    }

    __aicore__ inline void ScanStepAligned(int64_t dd, int32_t count,
        const AscendC::LocalTensor<float> &aCur, const AscendC::LocalTensor<float> &bCur,
        const AscendC::LocalTensor<float> &aCurD, const AscendC::LocalTensor<float> &bCurD,
        const AscendC::LocalTensor<float> &aNextD, const AscendC::LocalTensor<float> &bNextD)
    {
        auto shiftB = bCur[PAD_LANES - dd];
        auto shiftA = aCur[PAD_LANES - dd];
        AscendC::Mul(bNextD, shiftB, aCurD, count);
        AscendC::Add(bNextD, bNextD, bCurD, count);
        AscendC::Mul(aNextD, aCurD, shiftA, count);
        AscendC::Maxs(aNextD, aNextD, 1.1754944e-38f, count);
    }

    // V→标量真同步（F-BW18 同族修复，对齐 selective_scan_bwd 的 BW-18 方案）：
    // PipeBarrier<PIPE_V> 不能用于 V→标量回读同步——①dav-2201 编译宏实为 __NPU_ARCH__=2201
    // （bisheng -E -dM 实测；kernel_reg.h 2201 分支仅 AIC 侧 elide，AIV 发射真实屏障指令），
    // 但 ②pipe_barrier 语义仅限管线内排序（本项目 SYNC-02/06 红线：只保证 V-V 排序、不跨
    // PIPE），对后续标量序上的 GetValue 回读无 V→S 排序保证（bwd 侧设备实测：裸 V 结果
    // 回读间歇陈旧值 + 507034 挂死族，改 V_S 事件对后消解）。此处按官方归约实现同款
    // V_S 事件对：SetFlag 位于标量序中全部已发射 V 指令之后，WaitFlag 等待 V 队列排空，
    // 其后 GetValue 读到的一定是已完成的 V 结果（同时覆盖 pair 首 Duplicate(xBuild) V 写
    // → 首 s 轮 xBuild.GetValue 的隐性跨 PIPE 读）。
    __aicore__ inline void SyncVToScalar()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(evV2S_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(evV2S_);
    }

    // 输入 cast：fp32 同型直拷（Muls ×1.0；dav-2201 不支持 float→float 的 Cast CAST_NONE，静默无操作）；
    // fp16/bf16 Cast(CAST_NONE) 升精度
    __aicore__ inline void InputCast(const AscendC::LocalTensor<float> &dst,
                                     const AscendC::LocalTensor<INPUT_T> &src, int32_t count)
    {
        if constexpr (std::is_same<INPUT_T, float>::value) {
            AscendC::Muls(dst, src, 1.0f, count);
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
        }
    }

    // 输出 cast：fp32 直出（out 与 x 同为 fp32 契约，向量拷贝一次）；fp16/bf16 Cast(CAST_RINT)
    __aicore__ inline void OutputCast(const AscendC::LocalTensor<INPUT_T> &dst,
                                      const AscendC::LocalTensor<float> &src, int32_t count)
    {
        if constexpr (std::is_same<INPUT_T, float>::value) {
            AscendC::Muls(dst, src, 1.0f, count);
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, count);
        }
    }

    // 事件 ID（SYNC-04/07：同 id Set/Wait 1:1 配对；经 TPipe AllocEventID 独占分配）
    uint8_t evV2Mte3_ = 0;  // V→MTE3：out/out_z 存出前等 V 侧算完
    uint8_t evMte3V_ = 0;   // MTE3→V：下一 tile V 侧复用 du32/tmpA 前等存出完成
    uint8_t evV2S_ = 0;     // V→S：扫描结果标量回读前真同步（PipeBarrier<PIPE_V> 无跨 PIPE 语义，见 SyncVToScalar 注释）
    uint8_t evMte2V_ = 0;   // MTE2→V：z 内联搬入（TBuf）后等 MTE2 完成

    AscendC::TPipe pipe_;
    // GM
    AscendC::GlobalTensor<INPUT_T> uGm_, deltaGm_, zGm_, varBGm_, varCGm_, outGm_, outZGm_;
    AscendC::GlobalTensor<float> aGm_, bGm_, cGm_, dGm_, biasGm_, xGm_;
    // 队列（tile 级双缓冲，O1；u/delta 双缓冲）。z 弃用 TQue（见 zStageBuf_ 说明），
    // 全核 VECIN 队列数控制为 4：dav-2201 上 TQueBind::FreeTensor 每缓冲释放占用一个
    // V_MTE2 事件 ID（QUE_MAX_EVENT=8，仅当该缓冲被重新分配时才回收）——每 pair 恰
    // 2 tile 时各队列 2 缓冲均只释放一次、不再复用，5 队列（u/delta/z/varB/varC）
    // 共 10 个占用 > 8，AllocEventID 越界返回 → SetFlag 非法 ID → 向量核确定性挂死
    // （P-24 定位结论：V/V + z + 同 chunk ≥2 tile 触发，与 dtype/ds/D 无关）；
    // 4 队列 = 8 个占用，恰满不越界。
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQu_, inQd_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zStageBuf_;  // z 经此暂存：step7 内联搬入 + 手工 MTE2_V 事件
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQvarB_, inQvarC_;
    // 常驻计算缓冲
    AscendC::TBuf<AscendC::TPosition::VECCALC> u32Buf_, d32Buf_, du32Buf_, outAccBuf_, tmpABuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> a0Buf_, a1Buf_, b0Buf_, b1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aPreABuf_, bPreABuf_, aPreBBuf_, bPreBBuf_;  // R3-O9: 提前发射侧起始缓冲（双套交替）
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIdxBuf_, xBuildBuf_, aRowBuf_, sTmpBuf_, biasRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bRowBuf_, cRowBuf_;  // 静态 B/C 行（每 pair 一次载入，O3）
    AscendC::TBuf<AscendC::TPosition::VECCALC> varB32Buf_, varC32Buf_;
    // 标量
    int64_t batch_ = 0, dim_ = 0, seqlen_ = 0, dstate_ = 0, dimPerGroup_ = 0;
    int64_t nChunks_ = 0, numPairs_ = 0, pairsPerCore_ = 0;
    int64_t pairUBase_ = 0, pairXBase_ = 0;
    uint32_t flags_ = 0;
    bool hasD_ = false, hasZ_ = false, hasBias_ = false, softplus_ = false, varB_ = false, varC_ = false;
};

// Kernel 函数（<<<>>> launch 语法，bisheng -xasc 模式有效）
// 标量参数经 GM tiling 缓冲传入（签名仅指针参数，见 selective_scan_launch.h）
template <typename INPUT_T>
__global__ __aicore__ __vector__ void selective_scan_fwd_kernel(
    GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C,
    GM_ADDR D, GM_ADDR z, GM_ADDR delta_bias,
    GM_ADDR out, GM_ADDR x, GM_ADDR out_z, GM_ADDR tiling)
{
    SelectiveScanFwdKernel<INPUT_T> op;
    op.Init(u, delta, A, B, C, D, z, delta_bias, out, x, out_z,
            reinterpret_cast<const __gm__ SsfTilingData *>(tiling));
    op.Process();
}

// Tiling 计算（返回 usedCores, pairsPerCore；B1 红线：核数运行时查询，不硬编码）
std::tuple<int64_t, int64_t> calc_selective_scan_tiling_params(int64_t batch, int64_t dim)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t aivNum = static_cast<int64_t>(ascendcPlatform->GetCoreNumAiv());
    if (aivNum <= 0) aivNum = 1;
    int64_t numPairs = batch * dim;
    int64_t usedCores = std::min(aivNum, numPairs);
    if (usedCores < 1) usedCores = 1;
    int64_t pairsPerCore = (numPairs + usedCores - 1) / usedCores;
    return std::make_tuple(usedCores, pairsPerCore);
}

#define SELECTIVE_SCAN_LAUNCH_IMPL(DTYPE, SUFFIX)                                          \
    extern "C" void launch_selective_scan_fwd_##SUFFIX(                                    \
        uint32_t numBlocks,                                                                \
        GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C,                         \
        GM_ADDR D, GM_ADDR z, GM_ADDR delta_bias,                                          \
        GM_ADDR out, GM_ADDR x, GM_ADDR out_z, GM_ADDR tiling, void* stream)               \
    {                                                                                      \
        selective_scan_fwd_kernel<DTYPE><<<numBlocks, nullptr, stream>>>(                  \
            u, delta, A, B, C, D, z, delta_bias, out, x, out_z, tiling);                   \
    }

SELECTIVE_SCAN_LAUNCH_IMPL(float, float)
SELECTIVE_SCAN_LAUNCH_IMPL(half, half)
SELECTIVE_SCAN_LAUNCH_IMPL(bfloat16_t, bfloat16)
