// selective_scan_bwd_kernel.cpp —— SelectiveScan 反向算子（迭代 2，DAV_2201/A2）
//
// 语义（对齐 test/bwd/golden.py 手写 fp64 反传 + fwd kernel 实测）：
//   fwd:  h_t = a_t·h_{t-1} + b_t,  a_t = Exp(-Δ_t·A_s)（Exp base-e 实测）,
//         b̃_t = Δ_t·u_t·B_{t,s},  y_t = C_{t,s}·h_t,  out_t = y_t + D_s·u_t（pre-gate）,
//         out_z_t = out_t·silu(z_t),  silu(z) = z·σ(z)（fwd L472-478：Div σ 后 Mul z32）
//   bwd:  dx_t = w_t + a_{t+1}·dx_{t+1}（序列末 dx=0），w_t = dout'_t·C·(H 变换) 按组合展开；
//         du_t = dx_t·Δ_t·B_t(或·B_t) + D 项 Axpy(dout')；
//         ddelta_t = dx_t·u_t + dx_t·A·a_prev（a_prev = a_t·h̃_{t−1}，softplus 后 ×invF）；
//         dA_s = Σ_t dx_t·Δ_t·a_prev；dD_s = Σ_t dout'_t·u_t；ddelta_bias = Σ_t ddelta_t；
//         dz_t = dout_t·out_pre_t·σ(z)·(1+z·(1−σ(z)))（out_pre = fwd 量化 out 入参，已认可）；
//         dout'_t = dout_t·silu(z_t)；out_z 复算 = out_t·silu(z_t)。
//
// 实现要点（与 2.2 设计 §4/§5/§8 逐条对应）：
//   - 后缀扫描 seed 置于数据索引 count'=ceil8(tvl)（绝对 lane 512+count'）；满 tile 时
//     seed lane=1536、a 侧恒等 1.0 由 Init 填充（尾块 kill 写不触及 [1536,2048)）。
//   - 尾块 kill 写：a 侧 [1,0×7]、b 侧 [0×8] 于 lane 512+count'——af[count'-1] 经 Gather
//     读 kill lane 取 1（边界 a 已折入 seedW=aFirstSrc·bPost）。尾块 ⟺ 全局末 tile ⟹ seedW=0，
//     kill 即中和；伪 lane（tvl 非整 8）的 a_pad≠1 污染仅进入 afFin×seedW=0，无害。
//   - lane-0 修正：仅 count' 为 2 的幂（此时扫描窗=[0,count') 恰不含 seed lane）时
//     dx_0 += afFin[0]·seedW；非幂次时窗口 2^k ≥ count' 已含 seed，无需修正。
//   - Gather 小步长 base=(PAD±d)·4（设计稿 (PAD·d) 笔误已修正）；d≥8 用 operator[] 偏移视图。
//   - a_prev 乘积式 = a_t·h̃_{t−1}（Exp+Gather 左移+Mul+lane0 标量补，4 ops）：
//     弃用 Sub=h̃−b̃ 减法式（h̃≫b̃ 时灾难性核销，方案 L30 两式等价）；b̃ 原 b0k 双写
//     随之废止，b̃ 单写扫描 cur；du 的 D 跳连项 Axpy 移出 s 循环（s 无关，每 tile 一次）。
//   - ddelta 累加实测 5 ops/状态（设计 §8 预算 3 漏计运行和 Add，偏差已记录）。
//   - t0 缓冲复用（O3）：visit A 产出 u32_t0/d32_t0/invF_t0，T0-full 直接复用零重算；
//     rawDeltaT0 载入 raw→softplus→invF 同缓冲覆写（raw 用后即死）。
//   - 原子行源缓冲为 VECOUT 位（rowStage/rowVec，连续 DataCopy UB→GM 硬件约束）；
//     计算缓冲窄化 cast 目的取空闲非扫描缓冲（tmpA/h0/b0k/deltaU32 视图），不做同缓冲
//     原地窄化；扫描缓冲 sa/sb 严禁作 cast 目的——左 pad [0,512) 为 Init 恒等区（a=1/b=0），
//     cast 踩 pad 即污染后续 visit 前缀扫描（2-tile T0 全 lane 垃圾根因，fix_G）。
//   - 事件纪律：evV2Mte3_（V→MTE3 存出前）、evMte3V_（仅有 MTE3 存出的 visit 末放行 /
//     有存出前驱的 visit 首 Waiting——visit A 纯计算不 Set，消除空 MTE3 队列纯信号，
//     空 MTE3 队列的 SetFlag 偶发延迟 flush 会致下游 Wait 死等）、
//     evMte2V_（dout/z/out 批量内联 1 Set/Wait）、evRow_（rowStage 原子行
//     1:1；Process 起始一次放行，跨 pair 由上一 pair MaterializeAtomic 末 Set 供货，
//     visit 内 var 行与 pair 末静态行共用 token 串行）。Process 末平衡回收两链残留。
//     AllocEventID 独占分配。
//   - 4 条 VECIN 队列（u/delta/varB/varC）×深度 2=8 恰满 V_MTE2 事件池（fwd P-24 结论），
//     dout/z/out/x 前缀一律 TBuf+手工事件，禁止再增队列。
//   - 性能迭代 R1（标量链直线化，fwd R2 同款迁移）：三处 Hillis-Steele 扫描
//     （VisitA 前缀 / VisitFull 前缀 / VisitFull 后缀）删除 per-s ping-pong 侧交替与
//     每步 8 次交换赋值——起始侧固定 (sa0,sb0)（数值中性：prep 每 s 全量重写起始侧数据区，
//     pad/kill/seed 区两侧对称且写双份，上轮遗留值在扫描读取前被覆盖）；A 相 dd∈{1,2,4}
//     Gather 步直线化（runD1/2/4 守卫），B 相 dd≥8 对齐步 2 步展开（实参硬编码角色）；
//     终态视图 curIsA0 每 tile 一算单次选择（aFin/bFin 与死侧 aDead/bDead/bDeadF）。
//     后缀扫描起始侧本就复位固定（原 L788 复位块的直线化等价），复位块随之删除。
//     消费链视图重绑定：h̃ 垫高复制落死侧（bDead/bDeadF 作 Gather 源），指令序列逐条不变。
//     P4/P9 外提：bRowV/cRowV/varB32V 条件 Get 与扫描数据区视图出 s 循环。

#include <cstdint>
#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "selective_scan_bwd_launch.h"


namespace {
constexpr int64_t TILE_T = 1024;          // R6：单 tile 步数上限（Gather count ≤1024）
constexpr int64_t PAD_LANES = 512;        // 扫描缓冲单侧 pad lane 数
constexpr int64_t SB_LEN = 2048;          // 扫描缓冲全长（512 pad + 1024 数据 + 512 pad）
constexpr int64_t CHUNK_T = 2048;         // x 契约 chunk 步数
constexpr int64_t TILES_PER_CHUNK = 2;    // 每 chunk tile 数上限
constexpr int64_t DSTATE_MAX = 256;       // dstate 上限（需求校验）
constexpr float SOFTPLUS_CLAMP = 20.0f;   // softplus 钳位（fwd 同款）
// full visit 类型：tile1（fresh prep，hRun=P0·hRun0+Q0，seed=chunk 携带，末端写 bPost2_）
//                / T0（复用 t0 副本，hRun=hRun0，seed=tile1 边界，末端写 chunk 携带）
//                / single（fresh prep，hRun=hRun0，seed=chunk 携带，末端写 chunk 携带）
constexpr int KIND_TILE1 = 0;
constexpr int KIND_T0 = 1;
constexpr int KIND_SINGLE = 2;
}  // namespace

template <typename INPUT_T>
class SelectiveScanBwdKernel {
public:
    __aicore__ inline SelectiveScanBwdKernel() {}

    __aicore__ inline void Init(GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                GM_ADDR dout, GM_ADDR D, GM_ADDR z, GM_ADDR delta_bias,
                                GM_ADDR x, GM_ADDR out, GM_ADDR dz, GM_ADDR outZ,
                                GM_ADDR du, GM_ADDR ddelta, const __gm__ SsbTilingData *tp)
    {
        batch_ = tp->batch; dim_ = tp->dim; seqlen_ = tp->seqlen; dstate_ = tp->dstate;
        dimPerGroup_ = tp->dimPerGroup; nChunks_ = tp->nChunks; numPairs_ = tp->numPairs;
        pairsPerCore_ = tp->pairsPerCore; dstatePad_ = tp->dstatePad; seqlenPad_ = tp->seqlenPad;
        flags_ = tp->flags;
        hasD_ = (flags_ & SSB_FLAG_HAS_D) != 0;
        hasZ_ = (flags_ & SSB_FLAG_HAS_Z) != 0;
        hasBias_ = (flags_ & SSB_FLAG_HAS_BIAS) != 0;
        softplus_ = (flags_ & SSB_FLAG_SOFTPLUS) != 0;
        varB_ = (flags_ & SSB_FLAG_VAR_B) != 0;
        varC_ = (flags_ & SSB_FLAG_VAR_C) != 0;
        recompute_ = (flags_ & SSB_FLAG_RECOMPUTE_OUT_Z) != 0;
        hasX_ = (flags_ & SSB_FLAG_HAS_X) != 0;
        hasBCReduce_ = !varB_ || !varC_;  // dBC 归约组合：静静/静B变C/变B静C（变变无）

        uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(u));
        deltaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(delta));
        doutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(dout));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A));
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
        if (hasD_) dGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(D));
        if (hasBias_) biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(delta_bias));
        if (hasX_) xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
        if (hasZ_) {
            zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(z));
            outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(out));
            dzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(dz));
            if (recompute_) outZGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(outZ));
        }
        duGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(du));
        ddeltaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(ddelta));
        dAPadGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(static_cast<intptr_t>(tp->dAPadPtr)));
        dBPadGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(static_cast<intptr_t>(tp->dBPadPtr)));
        dCPadGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(static_cast<intptr_t>(tp->dCPadPtr)));
        dDPadGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(static_cast<intptr_t>(tp->dDPadPtr)));
        dbiasPadGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(static_cast<intptr_t>(tp->dbiasPadPtr)));

        // 总 tile 数（载入预取的越界判据）
        const int64_t restLast = seqlen_ - (nChunks_ - 1) * CHUNK_T;
        const int64_t tilesLast = (restLast + TILE_T - 1) / TILE_T;
        totalTiles_ = (nChunks_ - 1) * TILES_PER_CHUNK + tilesLast;

        // 1. 缓冲规划（见文件头；队列深度统一 2=R6 修订）
        const int64_t inBytes = TILE_T * sizeof(INPUT_T);
        pipe_.InitBuffer(inQu_, 2, inBytes);       // O1
        pipe_.InitBuffer(inQd_, 2, inBytes);       // O1
        if (varB_) pipe_.InitBuffer(inQvarB_, 2, inBytes);   // O5
        if (varC_) pipe_.InitBuffer(inQvarC_, 2, inBytes);   // O5
        pipe_.InitBuffer(u32Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(u32T0Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(d32Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(d32T0Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(rawDeltaBuf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(rawDeltaT0Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(deltaU32Buf_, TILE_T * sizeof(float));  // O2: Δu 每 tile 一次
        pipe_.InitBuffer(dout32Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(du32Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(ddelta32Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(tmpABuf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(sa0Buf_, SB_LEN * sizeof(float));
        pipe_.InitBuffer(sa1Buf_, SB_LEN * sizeof(float));
        pipe_.InitBuffer(sb0Buf_, SB_LEN * sizeof(float));
        pipe_.InitBuffer(sb1Buf_, SB_LEN * sizeof(float));
        pipe_.InitBuffer(h0Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(b0kBuf_, TILE_T * sizeof(float));
        if (varB_) pipe_.InitBuffer(varB32Buf_, TILE_T * sizeof(float));
        if (varC_) pipe_.InitBuffer(varC32Buf_, TILE_T * sizeof(float));
        pipe_.InitBuffer(doutStageBuf_, TILE_T * sizeof(INPUT_T));
        if (hasZ_) {
            pipe_.InitBuffer(zStageBuf_, TILE_T * sizeof(INPUT_T));
            pipe_.InitBuffer(outStageBuf_, TILE_T * sizeof(INPUT_T));
            pipe_.InitBuffer(z32Buf_, TILE_T * sizeof(float));
            pipe_.InitBuffer(out32Buf_, TILE_T * sizeof(float));
            // gate(σ) 复用 h0Buf_（z 链属 prep 段，h0 首写于状态循环，生命周期不交叠）
            pipe_.InitBuffer(dz32Buf_, TILE_T * sizeof(float));
            if (recompute_) pipe_.InitBuffer(outZ32Buf_, TILE_T * sizeof(float));
        }
        pipe_.InitBuffer(gatherIdxBuf_, TILE_T * sizeof(uint32_t));
        pipe_.InitBuffer(aRowBuf_, DSTATE_MAX * sizeof(float));
        if (!varB_) pipe_.InitBuffer(bRowBuf_, DSTATE_MAX * sizeof(float));
        if (!varC_) pipe_.InitBuffer(cRowBuf_, DSTATE_MAX * sizeof(float));
        pipe_.InitBuffer(sTmpBuf_, 64);            // D@lane0 / bias@lane8；ReduceSum tmp（910b 方式一不消费）
        if (hasX_) pipe_.InitBuffer(xPrefixBuf_, 2 * DSTATE_MAX * sizeof(float));
        pipe_.InitBuffer(dx0Buf_, DSTATE_MAX * sizeof(float));
        pipe_.InitBuffer(reduceDstABuf_, 32);
        pipe_.InitBuffer(reduceDstBBuf_, 32);
        pipe_.InitBuffer(reduceDstCBuf_, 32);
        pipe_.InitBuffer(rowVecBuf_, 4 * DSTATE_MAX * sizeof(float) + 64);  // VECOUT：静态行+pack8
        pipe_.InitBuffer(rowStageBuf_, TILE_T * sizeof(float));             // VECOUT：变量原子行

        // 2. 事件 ID 经 TPipe 独占分配（SYNC-04/07：同 id Set/Wait 1:1）
        evV2Mte3_ = static_cast<uint8_t>(pipe_.AllocEventID<AscendC::HardEvent::V_MTE3>());
        evMte3V_ = static_cast<uint8_t>(pipe_.AllocEventID<AscendC::HardEvent::MTE3_V>());
        evMte2V_ = static_cast<uint8_t>(pipe_.AllocEventID<AscendC::HardEvent::MTE2_V>());
        evRow_ = static_cast<uint8_t>(pipe_.AllocEventID<AscendC::HardEvent::MTE3_V>());
        evV2S_ = static_cast<uint8_t>(pipe_.AllocEventID<AscendC::HardEvent::V_S>());
        evS2V_ = static_cast<uint8_t>(pipe_.AllocEventID<AscendC::HardEvent::S_V>());

        // 3. 扫描缓冲中性元一次性填充：a 侧全 1（左 pad=前缀恒等；右 pad=后缀 af 恒等+
        //    满tile seed-a 占位）、b 侧全 0（左 pad=前缀恒等；右 pad=后缀 wf 恒等）。
        //    数据区 [512,1536) 每 tile 由 prep/扫描覆写；尾块 kill 写只落 [512+cnt',+8)。
        auto sa0v = sa0Buf_.Get<float>();
        auto sa1v = sa1Buf_.Get<float>();
        auto sb0v = sb0Buf_.Get<float>();
        auto sb1v = sb1Buf_.Get<float>();
        AscendC::Duplicate(sa0v, 1.0f, static_cast<int32_t>(SB_LEN));
        AscendC::Duplicate(sa1v, 1.0f, static_cast<int32_t>(SB_LEN));
        AscendC::Duplicate(sb0v, 0.0f, static_cast<int32_t>(SB_LEN));
        AscendC::Duplicate(sb1v, 0.0f, static_cast<int32_t>(SB_LEN));
        // 4. Gather 字节偏移索引 idx[i]=i·4（每核一次，跨 pair/跨 chunk 复用，O3）
        auto idx = gatherIdxBuf_.Get<uint32_t>();
        for (int64_t i = 0; i < TILE_T; ++i) {
            idx.SetValue(static_cast<uint32_t>(i), static_cast<uint32_t>(i * 4));
        }
        AscendC::PipeBarrier<PIPE_ALL>();  // 每核一次的构建汇聚点（标量写/Duplicate → V 侧读）
        prevVisitHadMte3_ = false;
    }

    // BW-18：V→标量真同步。勘误（fwd 审计后）：本仓宏实测 dav-2201 ↔ __NPU_ARCH__=2201
    // （非 3113），AIV 上 PipeBarrier<PIPE_V> 经 kernel_reg.h PipeBarrierImpl 真实发射
    // pipe_barrier 指令（elide 守卫 3003/3113 仅覆盖他架构，2201 分支仅 AIC 侧 elide）；
    // 但 pipe_barrier 仅有管线内排序语义，不向标量回读提供 V→S 跨 PIPE 序——旧版屏障
    // 对 V→标量序无保证（效果同未同步）。此处改用官方归约实现同款 V_S 事件对：
    // SetFlag 在标量序上位于全部已发射 V 指令之后，Wait 等待 V 队列排空，
    // 其后 GetValue 读到的一定是已完成的 V 结果。
    __aicore__ inline void SyncVToScalar()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(evV2S_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(evV2S_);
    }

    // BW-18：标量 UB 写 → 后续 V 侧读（prep Exp / Gather 等）。官方归约实现同款 S_V
    // 事件对：标量写冲刷至 UB 后才继续标量序（后续 V 指令发射必见）。
    __aicore__ inline void SyncScalarToV()
    {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(evS2V_);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(evS2V_);
    }

    // ---------- 扫描单步助手（性能迭代 R1 新增：ping-pong 侧序由调用点实参硬编码） ----------
    // 前缀方向（左移 dd）：cur=(aCur,aCurD/bCurD) → next=(aNextD,bNextD) 一步 Hillis-Steele 复合：
    //   b_next[i] = b_cur[i-dd]·a_cur[i] + b_cur[i]；a_next[i] = a_cur[i-dd]·a_cur[i]
    // 操作序列与原实现逐指令一致（位级等价）；与 fwd R2 助手同构但**无 Maxs 下界钳制**
    //（bwd 原扫描步无该防御，R1 不新增数值路径）。Gather 路径（dd∈{1,2,4}，R6 绕行约束）
    // 源为 full 视图（含 pad 区）；对齐路径（dd≥8）源为 full 视图 [PAD-dd] 偏移视图。
    __aicore__ inline void ScanStepGather(int64_t dd, int32_t count,
        const AscendC::LocalTensor<float> &aCur, const AscendC::LocalTensor<float> &bCur,
        const AscendC::LocalTensor<float> &aCurD, const AscendC::LocalTensor<float> &bCurD,
        const AscendC::LocalTensor<float> &aNextD, const AscendC::LocalTensor<float> &bNextD,
        const AscendC::LocalTensor<uint32_t> &idx)
    {
        const uint32_t base = static_cast<uint32_t>((PAD_LANES - dd) * sizeof(float));
        AscendC::Gather(bNextD, bCur, idx, base, count);
        AscendC::Mul(bNextD, bNextD, aCurD, count);
        AscendC::Add(bNextD, bNextD, bCurD, count);
        AscendC::Gather(aNextD, aCur, idx, base, count);
        AscendC::Mul(aNextD, aNextD, aCurD, count);
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
    }

    // 后缀方向（右移 dd）：b_next[i] = b_cur[i+dd]·a_cur[i] + b_cur[i]；a_next 同构。
    // 读窗 [PAD+dd, PAD+dd+cnt) 伸入右 pad/kill/seed lane——两侧 Init 对称（a=1/b=0）、
    // kill/seed 写双份，任一侧作 cur 语义一致（固定起始侧数值中性的组成部分）。
    __aicore__ inline void SufScanStepGather(int64_t dd, int32_t count,
        const AscendC::LocalTensor<float> &aCur, const AscendC::LocalTensor<float> &bCur,
        const AscendC::LocalTensor<float> &aCurD, const AscendC::LocalTensor<float> &bCurD,
        const AscendC::LocalTensor<float> &aNextD, const AscendC::LocalTensor<float> &bNextD,
        const AscendC::LocalTensor<uint32_t> &idx)
    {
        const uint32_t base = static_cast<uint32_t>((PAD_LANES + dd) * sizeof(float));
        AscendC::Gather(bNextD, bCur, idx, base, count);
        AscendC::Mul(bNextD, bNextD, aCurD, count);
        AscendC::Add(bNextD, bNextD, bCurD, count);
        AscendC::Gather(aNextD, aCur, idx, base, count);
        AscendC::Mul(aNextD, aNextD, aCurD, count);
    }

    __aicore__ inline void SufScanStepAligned(int64_t dd, int32_t count,
        const AscendC::LocalTensor<float> &aCur, const AscendC::LocalTensor<float> &bCur,
        const AscendC::LocalTensor<float> &aCurD, const AscendC::LocalTensor<float> &bCurD,
        const AscendC::LocalTensor<float> &aNextD, const AscendC::LocalTensor<float> &bNextD)
    {
        auto shiftB = bCur[PAD_LANES + dd];
        auto shiftA = aCur[PAD_LANES + dd];
        AscendC::Mul(bNextD, shiftB, aCurD, count);
        AscendC::Add(bNextD, bNextD, bCurD, count);
        AscendC::Mul(aNextD, aCurD, shiftA, count);
    }

    __aicore__ inline void Process()
    {
        if (numPairs_ <= 0) return;
        // rowStage 原子行事件初始放行（此后每行 Wait→写→原子拷→Set 严格 1:1，
        // 跨 pair 由上一 pair 末行 Set 供货，计数全局守恒）
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evRow_);
        const int64_t coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t start = coreIdx * pairsPerCore_;
        const int64_t end = (start + pairsPerCore_ < numPairs_) ? (start + pairsPerCore_) : numPairs_;
        for (int64_t p = start; p < end; ++p) {
            ProcessPair(p);
        }
        // 计数平衡回收：末 visit 的 evMte3V_ 放行、末 pair 的 evRow_ token 归还
        //（避免跨 launch 残留旧计数导致下次 launch 的 Wait 提前通过；
        //  evMte3V_ 仅在确有存出 visit 时才有 +1 供给，条件对齐）
        if (prevVisitHadMte3_) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evRow_);
    }

private:
    // ---------- tile 级搬入（O1 提前发射；GM 侧 blockLen=有效字节数防越界读） ----------
    __aicore__ inline void CopyInTile(int64_t gt)
    {
        const int64_t off = pairUBase_ + gt * TILE_T;
        const int64_t valid = seqlen_ - gt * TILE_T;
        const int64_t tvl = (valid < TILE_T ? valid : TILE_T);
        const uint32_t bytes = static_cast<uint32_t>(tvl * sizeof(INPUT_T));
        // 尾块 VECIN pad 区 [tvl, cnt) 必须显式补零：DataCopyPad 只搬 blockLen 有效
        // 字节，padValue 仅在 rightPadding>0 时生效；否则 pad 区为 UB 残留
        // （实测为上一 tile 头部值），a_pad=Exp(Δ_pad·A)≠1、b̃_pad≠0 破坏缝隙中性。
        const uint8_t padR = static_cast<uint8_t>(((tvl + 7) / 8) * 8 - tvl);
        AscendC::DataCopyExtParams cp{1, bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<INPUT_T> pad{true, 0, padR, 0};

        auto uLocal = inQu_.AllocTensor<INPUT_T>();
        AscendC::DataCopyPad(uLocal, uGm_[off], cp, pad);
        inQu_.EnQue(uLocal);

        auto dLocal = inQd_.AllocTensor<INPUT_T>();
        AscendC::DataCopyPad(dLocal, deltaGm_[off], cp, pad);
        inQd_.EnQue(dLocal);
    }

    // ---------- 变量 B/C 状态行搬入（O5：队列深度 2，s 循环顶预取；GM 偏移用未 pad seqlen） ----------
    __aicore__ inline void CopyInVarRow(AscendC::TQue<AscendC::TPosition::VECIN, 2> &q,
                                        AscendC::GlobalTensor<INPUT_T> &gm,
                                        int64_t b, int64_t d, int64_t s, int64_t gt)
    {
        const int64_t g = d / dimPerGroup_;
        const int64_t off = ((b * (dim_ / dimPerGroup_) + g) * dstate_ + s) * seqlen_ + gt * TILE_T;
        const int64_t valid = seqlen_ - gt * TILE_T;
        const int64_t tvl = (valid < TILE_T ? valid : TILE_T);
        const uint32_t bytes = static_cast<uint32_t>(tvl * sizeof(INPUT_T));
        const uint8_t padR = static_cast<uint8_t>(((tvl + 7) / 8) * 8 - tvl);  // 尾块 pad 补零（同 CopyInTile）
        AscendC::DataCopyExtParams cp{1, bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<INPUT_T> pad{true, 0, padR, 0};
        auto local = q.template AllocTensor<INPUT_T>();
        AscendC::DataCopyPad(local, gm[off], cp, pad);
        q.EnQue(local);
    }

    // ---------- 输入/输出 cast（fwd 同款：fp32 Muls×1.0 同型保持；fp16/bf16 Cast） ----------
    __aicore__ inline void InputCast(const AscendC::LocalTensor<float> &dst,
                                     const AscendC::LocalTensor<INPUT_T> &src, int32_t count)
    {
        if constexpr (std::is_same<INPUT_T, float>::value) {
            AscendC::Muls(dst, src, 1.0f, count);
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
        }
    }

    __aicore__ inline void OutputCast(const AscendC::LocalTensor<INPUT_T> &dst,
                                      const AscendC::LocalTensor<float> &src, int32_t count)
    {
        if constexpr (std::is_same<INPUT_T, float>::value) {
            AscendC::Muls(dst, src, 1.0f, count);
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, count);
        }
    }

    // ---------- pair 级：一条 (b,d) 序列的全部 chunk（降序=反向扫描，携带右→左） ----------
    __aicore__ inline void ProcessPair(int64_t p)
    {
        const int64_t b = p / dim_;
        const int64_t d = p - b * dim_;
        pairUBase_ = p * seqlen_;
        pairXBase_ = p * nChunks_ * 2 * dstate_;

        // 归约累加器清零 + 反向携带初始化（右边界=序列末：bPost=0；aFirst 恒等=1）
        for (int64_t s = 0; s < dstate_; ++s) {
            dAAcc_[s] = 0.0f;
            dBCAcc_[s] = 0.0f;
            bPost_[s] = 0.0f;
            bPost2_[s] = 0.0f;
            aFirstCarry_[s] = 1.0f;
            aFirst2_[s] = 1.0f;
            P0_[s] = 0.0f;
            Q0_[s] = 0.0f;
            hRun0_[s] = 0.0f;
        }
        dDAcc_ = 0.0f;
        dBiasAcc_ = 0.0f;

        // O3: A / 静态 B/C 行 + D/bias 标量中转（每 pair 仅载入一次；禁 GetValue 红线 B2）
        AscendC::DataCopyExtParams rowParams{1, static_cast<uint32_t>(dstate_ * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(aRowBuf_.Get<float>(), aGm_[d * dstate_], rowParams,
                             AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        if (!varB_) {
            AscendC::DataCopyPad(bRowBuf_.Get<float>(), bGm_[d * dstate_], rowParams,
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        if (!varC_) {
            AscendC::DataCopyPad(cRowBuf_.Get<float>(), cGm_[d * dstate_], rowParams,
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        if (hasD_) {
            AscendC::DataCopyExtParams oneParams{1, sizeof(float), 0, 0, 0};
            AscendC::DataCopyPad(sTmpBuf_.Get<float>()[0], dGm_[d], oneParams,
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        if (hasBias_) {
            AscendC::DataCopyExtParams oneParams{1, sizeof(float), 0, 0, 0};
            AscendC::DataCopyPad(sTmpBuf_.Get<float>()[8], biasGm_[d], oneParams,
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        AscendC::PipeBarrier<PIPE_MTE2>();  // MTE2 完成 → 标量 GetValue 回读
        if (hasD_) dVal_ = sTmpBuf_.Get<float>().GetValue(0);
        if (hasBias_) biasVal_ = sTmpBuf_.Get<float>().GetValue(8);

        // chunk 降序主循环（O4：越界 tile 整体跳过由 tiles 计算天然保证）
        for (int64_t c = nChunks_ - 1; c >= 0; --c) {
            const int64_t rest = seqlen_ - c * CHUNK_T;
            const int64_t validLenChunk = (rest < CHUNK_T) ? rest : CHUNK_T;
            const int64_t tiles = (validLenChunk + TILE_T - 1) / TILE_T;
            // x 前缀（chunk 左入口运行前缀 B 分量）：c>0 读 x[c-1]，c=0 恒 0
            if (c > 0) {
                const int64_t xOff = pairXBase_ + (c - 1) * 2 * dstate_;
                AscendC::DataCopyExtParams xCp{1, static_cast<uint32_t>(2 * dstate_ * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(xPrefixBuf_.Get<float>(), xGm_[xOff], xCp,
                                     AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
                AscendC::PipeBarrier<PIPE_MTE2>();
                auto xp = xPrefixBuf_.Get<float>();
                for (int64_t s = 0; s < dstate_; ++s) {
                    const float hTrue = xp.GetValue(static_cast<uint32_t>(2 * s + 1));
                    if (varB_) {
                        hRun0_[s] = hTrue;  // 变 B 全真域，无需换算
                    } else {
                        // 静 B：x 存真域 h，缩放域入口 hRun = h/B
                        //（B=0 ⇒ b̃=0 ⇒ x 前缀恒 0，invB 安全置 0，全梯度代数自洽归零）
                        const float Bv = bRowBuf_.Get<float>().GetValue(static_cast<uint32_t>(s));
                        const float invB = (Bv == 0.0f) ? 0.0f : 1.0f / Bv;
                        hRun0_[s] = hTrue * invB;
                    }
                }
            } else {
                for (int64_t s = 0; s < dstate_; ++s) hRun0_[s] = 0.0f;
            }
            if (tiles == 2) {
                CopyInTile(c * TILES_PER_CHUNK);              // visit A 的 u/delta
                VisitAPrefix(c * TILES_PER_CHUNK, b, d);
                VisitFull(c * TILES_PER_CHUNK + 1, b, d, validLenChunk - TILE_T, KIND_TILE1);
                VisitFull(c * TILES_PER_CHUNK, b, d, TILE_T, KIND_T0);
            } else {
                CopyInTile(c * TILES_PER_CHUNK);
                VisitFull(c * TILES_PER_CHUNK, b, d, validLenChunk, KIND_SINGLE);
            }
        }
        MaterializeAtomic(d);
    }

    // ---------- visit A（2-tile chunk 的 tile0 预备 visit）：
    //   产出 t0 副本（u32_t0/d32_t0/invF_t0/Δu32_t0）+ 逐状态前缀折叠 P0/Q0；无 dout/无输出 ----------
    __aicore__ inline void VisitAPrefix(int64_t gt, int64_t b, int64_t d)
    {
        const int64_t tvl = TILE_T;  // 2-tile chunk 的 tile0 恒满
        const int32_t cnt = static_cast<int32_t>(TILE_T);
        const int64_t tileUOff = pairUBase_ + gt * TILE_T;
        (void)tileUOff;

        if (prevVisitHadMte3_) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
        }

        // 消费 tile0 的 u/delta（ProcessPair 已发射）
        auto uLocal = inQu_.DeQue<INPUT_T>();
        auto dLocal = inQd_.DeQue<INPUT_T>();
        auto u32T0 = u32T0Buf_.Get<float>();
        auto rawT0 = rawDeltaT0Buf_.Get<float>();
        InputCast(u32T0, uLocal, cnt);
        InputCast(rawT0, dLocal, cnt);
        inQu_.FreeTensor(uLocal);
        inQd_.FreeTensor(dLocal);
        if (hasBias_) AscendC::Adds(rawT0, rawT0, biasVal_, cnt);
        // O1：发射 tile1 的 u/delta（本 visit 计算期间飞行）
        CopyInTile(gt + 1);

        auto d32T0 = d32T0Buf_.Get<float>();
        auto tmpA = tmpABuf_.Get<float>();
        if (softplus_) {
            // softplus（fwd 等价式，钳位 scratch=du32 本 visit 未启用，S-V-07 同款）
            auto clampV = du32Buf_.Get<float>();
            auto logD = sb1Buf_.Get<float>()[PAD_LANES];
            AscendC::Duplicate(tmpA, SOFTPLUS_CLAMP, cnt);
            AscendC::Min(clampV, rawT0, tmpA, cnt);
            AscendC::Exp(tmpA, clampV, cnt);
            AscendC::Adds(tmpA, tmpA, 1.0f, cnt);
            AscendC::Log(logD, tmpA, cnt);
            AscendC::Max(d32T0, rawT0, logD, cnt);
            // invF = 1/(1+e^{-raw})，写回 rawT0（raw 用后即死，同缓冲覆写）
            auto dd32 = ddelta32Buf_.Get<float>();
            AscendC::Muls(tmpA, rawT0, -1.0f, cnt);
            AscendC::Exp(dd32, tmpA, cnt);
            AscendC::Adds(dd32, dd32, 1.0f, cnt);
            AscendC::Duplicate(tmpA, 1.0f, cnt);
            AscendC::Div(rawT0, tmpA, dd32, cnt);
        } else {
            AscendC::Muls(d32T0, rawT0, 1.0f, cnt);
        }
        // O2: Δu32_t0 每 tile 一次（T0-full 复用，免第二份缓冲）
        AscendC::Mul(deltaU32Buf_.Get<float>(), d32T0, u32T0, cnt);

        if (varB_) CopyInVarRow(inQvarB_, varBGm_, b, d, 0, gt);

        auto aRow = aRowBuf_.Get<float>();
        auto sa0 = sa0Buf_.Get<float>();
        auto sa1 = sa1Buf_.Get<float>();
        auto sb0 = sb0Buf_.Get<float>();
        auto sb1 = sb1Buf_.Get<float>();
        auto idx = gatherIdxBuf_.Get<uint32_t>();
        auto deltaU32 = deltaU32Buf_.Get<float>();
        AscendC::LocalTensor<float> varB32V;
        if (varB_) {
            varB32V = varB32Buf_.Get<float>();  // R1：per-s 双 Get 外提（对应条件 InitBuffer）
        }
        // R1 标量链优化（fwd R2 同款）：起始侧固定 (sa0,sb0)——prep 每 s 全量重写起始侧
        // 数据区（Exp/b̃ count=cnt 覆盖全部有效 lane），pad/kill 区两侧对称，上轮遗留值在
        // 扫描读取前被覆盖，删除侧交替数值中性；终态视图 curIsA0 每 tile 一算（本 visit
        // tvl=TILE_T 恒 10 步，编译期折叠）——消 per-s 8 次 selects 与每步 8 次交换赋值。
        auto a0D = sa0[PAD_LANES];
        auto a1D = sa1[PAD_LANES];
        auto b0D = sb0[PAD_LANES];
        auto b1D = sb1[PAD_LANES];
        int64_t totalSteps = 0;
        for (int64_t dd = 1; dd < TILE_T && dd < tvl; dd <<= 1) {
            ++totalSteps;
        }
        const bool curIsA0 = (totalSteps & 1) == 0;
        auto aFin = curIsA0 ? a0D : a1D;   // 终态扫描结果视图（visit 内 s 不变）
        auto bFin = curIsA0 ? b0D : b1D;
        for (int64_t s = 0; s < dstate_; ++s) {
            if (varB_ && s + 1 < dstate_) CopyInVarRow(inQvarB_, varBGm_, b, d, s + 1, gt);

            // prep：a = Exp(Δ·A)（固定写 sa0）；b̃ = Δu（缩放域，静 B——2.a 统一式要求 h̃ 同缩放，
            //       B 因子在 w=dx 域与 dB/dC 派生式逐项消去；visit A 无输出，免 b̃ 写）
            const float Aval = aRow.GetValue(static_cast<uint32_t>(s));
            AscendC::Muls(tmpA, d32T0, Aval, cnt);
            AscendC::Exp(a0D, tmpA, cnt);
            if (varB_) {
                auto vB = inQvarB_.DeQue<INPUT_T>();
                InputCast(varB32V, vB, cnt);
                inQvarB_.FreeTensor(vB);
                AscendC::Mul(b0D, deltaU32, varB32V, cnt);
            } else {
                AscendC::Muls(b0D, deltaU32, 1.0f, cnt);
            }

            // 10 步前缀扫描（fwd 同款 Hillis-Steele，d≥tvl 跳过——tvl=1024 全跑）
            // R1 相位拆分：A 相 dd∈{1,2,4} 直线化（守卫 = 原循环 dd=1/2/4 条件的等价标量形式）；
            // B 相 dd≥8 对齐步 2 步展开——A 相恰 3 步后 cur=(a1,b1)，每对步角色固定（奇 j 读
            // (a1,b1) 写 (a0,b0)，偶 j 读 (a0,b0) 写 (a1,b1)），ping-pong 侧序由实参硬编码，
            // 操作序列与原实现逐指令一致（位级等价）。
            if (tvl > 1) {
                ScanStepGather(1, cnt, sa0, sb0, a0D, b0D, a1D, b1D, idx);
            }
            if (tvl > 2) {
                ScanStepGather(2, cnt, sa1, sb1, a1D, b1D, a0D, b0D, idx);
            }
            if (tvl > 4) {
                ScanStepGather(4, cnt, sa0, sb0, a0D, b0D, a1D, b1D, idx);
            }
            int64_t dd = 8;
            while (dd < TILE_T && dd < tvl) {
                ScanStepAligned(dd, cnt, sa1, sb1, a1D, b1D, a0D, b0D);
                const int64_t dd2 = dd << 1;
                // 偶步内层守卫：与原循环逐步条件一致（dd2 可能已越过 tvl）
                if (dd2 < TILE_T && dd2 < tvl) {
                    ScanStepAligned(dd2, cnt, sa0, sb0, a0D, b0D, a1D, b1D);
                }
                dd = dd2 << 1;
            }
            // P0/Q0 捕获（lane cnt'-1；aFin/bFin：R1 终态视图，visit 内 s 不变）。
            // 红线：V 活跃循环中禁止标量 SetValue 写 UB——910B 实测标量写与紧随的
            // V 指令无屏障交错会偶发死锁（2-tile hang 根因，G/G1 二分定位）。
            // 直接读入标量数组（纯标量侧，无 UB 写）。
            // BW-18：PipeBarrier<PIPE_V> 仅有管线内排序、无 V→S 跨 PIPE 序，V→标量回读改 V_S 事件对。
            SyncVToScalar();
            P0_[s] = aFin.GetValue(static_cast<uint32_t>(cnt - 1));
            Q0_[s] = bFin.GetValue(static_cast<uint32_t>(cnt - 1));
        }
        // visit A 为纯计算（无任何 MTE3 存出）——不发 SetFlag 放行信号：
        // 空 MTE3 指令队列中的 SetFlag 执行时机不确定（偶发延迟到后续 MTE3 指令
        // 入队才 flush），下游 Wait 会偶发死等（2-tile hang 根因）。
        // visit A 与下一 visit 间的 V 缓冲复用由同 V 队列 FIFO 程序序天然保证。
        prevVisitHadMte3_ = false;
    }

    // ---------- 满访问（tile1 / T0 / single 三态合一）----------
    //   KIND_TILE1:  fresh prep；hRun=P0·hRun0+Q0；seed=chunk 携带；末端写 bPost2_/aFirst2_
    //   KIND_T0:     复用 t0 副本（O3 零重 prep，仅重发 Δu32 1 op）；hRun=hRun0；seed=aFirst2·bPost2
    //   KIND_SINGLE: fresh prep；hRun=hRun0；seed=chunk 携带；末端写 chunk 携带
    __aicore__ inline void VisitFull(int64_t gt, int64_t b, int64_t d, int64_t tvl, int kind)
    {
        const int32_t cnt = static_cast<int32_t>(((tvl + 7) / 8) * 8);
        const int64_t tileUOff = pairUBase_ + gt * TILE_T;
        const uint32_t outBytes = static_cast<uint32_t>(tvl * sizeof(INPUT_T));
        AscendC::DataCopyExtParams outCp{1, outBytes, 0, 0, 0};
        // 尾块 pad 补零（同 CopyInTile）：dout pad=0 保证 w_pad=0 缝隙中性
        const uint8_t padR = static_cast<uint8_t>(cnt - tvl);
        AscendC::DataCopyPadExtParams<INPUT_T> inPad{true, 0, padR, 0};

        // 跨 visit 缓冲复用保护（SYNC-02/06）：上一 visit 的 MTE3 存出完成后才能复用 V 缓冲
        //（仅当上一 visit 确有存出；visit A 纯计算不置位，与其后 visit 无需事件）
        if (prevVisitHadMte3_) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
        }

        auto tmpA = tmpABuf_.Get<float>();
        const bool useT0 = (kind == KIND_T0);
        auto u32 = useT0 ? u32T0Buf_.Get<float>() : u32Buf_.Get<float>();
        auto d32 = useT0 ? d32T0Buf_.Get<float>() : d32Buf_.Get<float>();
        auto invF = useT0 ? rawDeltaT0Buf_.Get<float>() : rawDeltaBuf_.Get<float>();

        if (kind == KIND_T0) {
            // O3：t0 副本常驻复用（u32_t0/d32_t0/invF_t0 由 visit A 产出），零重复 MTE2/cast
            AscendC::Mul(deltaU32Buf_.Get<float>(), d32, u32, cnt);  // O2: Δu32 重发（1 op）
        } else {
            auto uLocal = inQu_.DeQue<INPUT_T>();
            auto dLocal = inQd_.DeQue<INPUT_T>();
            auto u32v = u32Buf_.Get<float>();
            auto raw = rawDeltaBuf_.Get<float>();
            InputCast(u32v, uLocal, cnt);
            InputCast(raw, dLocal, cnt);
            inQu_.FreeTensor(uLocal);
            inQd_.FreeTensor(dLocal);
            if (hasBias_) AscendC::Adds(raw, raw, biasVal_, cnt);
            // 原 O1 跨 tile 预发射（CIT(gt+1)）已废止（BW-16 根因修复）：chunk 降序处理
            // 下 gt+1 属于后处理的 chunk+1，该陈旧副本滞留 FIFO 队列头部，使后续 chunk
            // 的 VisitA/TILE1 DeQue 错位拿到错误 tile 的 u/delta（l=6144 实测 chunk0
            // 全坏、l=8192 队列满死锁）；跨 chunk tile0 由该 chunk 自身 CIT 发射，
            // 不再预取。chunk 内 tile1 预取保留（VisitA L401，方向正确）。

            if (softplus_) {
                // softplus（fwd 等价式；钳位 scratch=du32 此时尚未启用，S-V-07 同款）
                auto d32v = d32Buf_.Get<float>();
                auto clampV = du32Buf_.Get<float>();
                auto logD = sb1Buf_.Get<float>()[PAD_LANES];
                AscendC::Duplicate(tmpA, SOFTPLUS_CLAMP, cnt);
                AscendC::Min(clampV, raw, tmpA, cnt);
                AscendC::Exp(tmpA, clampV, cnt);
                AscendC::Adds(tmpA, tmpA, 1.0f, cnt);
                AscendC::Log(logD, tmpA, cnt);
                AscendC::Max(d32v, raw, logD, cnt);
                // invF = 1/(1+e^{-raw})，同缓冲覆写（raw 用后即死）
                auto dd32 = ddelta32Buf_.Get<float>();
                AscendC::Muls(tmpA, raw, -1.0f, cnt);
                AscendC::Exp(dd32, tmpA, cnt);
                AscendC::Adds(dd32, dd32, 1.0f, cnt);
                AscendC::Duplicate(tmpA, 1.0f, cnt);
                AscendC::Div(raw, tmpA, dd32, cnt);
            } else {
                AscendC::Muls(d32Buf_.Get<float>(), raw, 1.0f, cnt);
            }
            AscendC::Mul(deltaU32Buf_.Get<float>(), d32Buf_.Get<float>(), u32v, cnt);  // O2
        }

        // dout (+z/out) 批量内联搬入 → 1 次 MTE2_V 事件（弃 TQue 保 4 队列事件池恰满）
        auto dout32 = dout32Buf_.Get<float>();
        AscendC::DataCopyPad(doutStageBuf_.Get<INPUT_T>(), doutGm_[tileUOff], outCp, inPad);
        if (hasZ_) {
            AscendC::DataCopyPad(zStageBuf_.Get<INPUT_T>(), zGm_[tileUOff], outCp, inPad);
            AscendC::DataCopyPad(outStageBuf_.Get<INPUT_T>(), outGm_[tileUOff], outCp, inPad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
        InputCast(dout32, doutStageBuf_.Get<INPUT_T>(), cnt);
        if (hasZ_) {
            InputCast(z32Buf_.Get<float>(), zStageBuf_.Get<INPUT_T>(), cnt);
            InputCast(out32Buf_.Get<float>(), outStageBuf_.Get<INPUT_T>(), cnt);
        }

        // z 链（O2：s 无关段外提）：dz 因子、dout' 因子、out_z 复算
        if (hasZ_) {
            auto z32 = z32Buf_.Get<float>();
            auto gate = h0Buf_.Get<float>();  // σ 复用 h0（z 链属 prep 段，h0 首写于状态循环，生命周期不交叠）
            auto dz32 = dz32Buf_.Get<float>();
            // σ(z) = 1/(1+e^{-z})（fwd 同款 Div 路线，base-e 实测）
            AscendC::Muls(tmpA, z32, -1.0f, cnt);
            AscendC::Exp(tmpA, tmpA, cnt);
            AscendC::Adds(tmpA, tmpA, 1.0f, cnt);
            AscendC::Duplicate(gate, 1.0f, cnt);
            AscendC::Div(gate, gate, tmpA, cnt);
            // silu'(z) = σ·(1+z·(1−σ))
            AscendC::Muls(tmpA, gate, -1.0f, cnt);
            AscendC::Adds(tmpA, tmpA, 1.0f, cnt);
            AscendC::Mul(tmpA, tmpA, z32, cnt);
            AscendC::Adds(tmpA, tmpA, 1.0f, cnt);
            AscendC::Mul(tmpA, tmpA, gate, cnt);
            // dz = dout·out_pre·silu'(z)（out_pre=fwd 量化 out 入参）；存出后移至 visit 末批量
            AscendC::Mul(dz32, dout32, out32Buf_.Get<float>(), cnt);
            AscendC::Mul(dz32, dz32, tmpA, cnt);
            // dout' = dout·silu(z)；recompute 时顺带 out_z = out·silu(z)
            AscendC::Mul(tmpA, z32, gate, cnt);
            if (recompute_) {
                AscendC::Mul(outZ32Buf_.Get<float>(), out32Buf_.Get<float>(), tmpA, cnt);
            }
            AscendC::Mul(dout32, dout32, tmpA, cnt);
        }

        // 累加器清零（du/ddelta 跨状态累加，每 visit 重置）
        auto du32 = du32Buf_.Get<float>();
        auto ddelta32 = ddelta32Buf_.Get<float>();
        AscendC::Duplicate(du32, 0.0f, cnt);
        AscendC::Duplicate(ddelta32, 0.0f, cnt);

        // D 跳连归约（s 无关，prep 段；state0 组回读）
        if (hasD_) {
            AscendC::Mul(tmpA, dout32, u32, cnt);
            AscendC::ReduceSum(reduceDstABuf_.Get<float>(), tmpA, sTmpBuf_.Get<float>(),
                               static_cast<int32_t>(tvl));
        }

        // 扫描右边界中和（kill 写；尾块与满 tile 均需）：af 构造每次读 [PAD_LANES+cnt]
        // 作 af[cnt-1]，逆扫描边界项=seedW·…·af[cnt-1]，该 lane 必须恒 1.0（a(cnt) 因子
        // 已折入 seedW=aFirstSrc·bPost）；满 tile 原跳过使 af[cnt-1] 落在未初始化 UB
        // （BW-12 满 tile 坏/尾块绿的分界根因）。
        // kill lane = 512+cnt' 随 cnt' 任意对齐，Duplicate 偏移视图要求 32B 对齐 →
        // 改标量 SetValue（8 lane 窗口 [kill, kill+8)）。
        // BW-18/G-G1：标量写 UB 前必须排空在飞 V 指令——visit 首 Wait(evMte3V_) 只覆盖
        // 上一 visit，本 visit 的 cast 链 / Δu Duplicate / D 跳连归约均可能未完成；
        // 且 PipeBarrier<PIPE_V> 仅有管线内排序、无 V→S 跨 PIPE 序（不能排空在飞 V），无屏障交错即
        // G/G1 红线偶发死锁（全核同 PC 停归约 V_S 等待）与在飞 V 结果损坏。
        // 此处块顶 V_S 事件对真排空，块尾 S_V 事件对放行本 visit V 侧读。
        {
            SyncVToScalar();
            const uint32_t kill = static_cast<uint32_t>(PAD_LANES + cnt);
            auto sa0k = sa0Buf_.Get<float>();
            auto sa1k = sa1Buf_.Get<float>();
            auto sb0k = sb0Buf_.Get<float>();
            auto sb1k = sb1Buf_.Get<float>();
            sa0k.SetValue(kill, 1.0f);
            sa1k.SetValue(kill, 1.0f);
            for (uint32_t k = 1; k < 8; ++k) {
                sa0k.SetValue(kill + k, 0.0f);
                sa1k.SetValue(kill + k, 0.0f);
            }
            for (uint32_t k = 0; k < 8; ++k) {
                sb0k.SetValue(kill + k, 0.0f);
                sb1k.SetValue(kill + k, 0.0f);
            }
            // 尾块 d32 pad 区 [tvl, cnt) 显式清零（满 tile 时循环体空跑）：raw pad=0（载入
            // rightPadding）经 bias Adds / softplus 变换后非零（bias / softplus(0)=log2）→
            // 若不清零，a_pad=Exp(Δ_pad·A)≠1 破坏缝隙中性。
            // 仅尾块有净效果；T0 恒满 tile 且活动 d32 为 d32T0Buf_，此写不触及。
            // （在飞 V 已由块顶 SyncVToScalar 排空，此处直接标量写。）
            auto d32k = d32Buf_.Get<float>();
            for (int32_t k = static_cast<int32_t>(tvl); k < cnt; ++k) {
                d32k.SetValue(static_cast<uint32_t>(k), 0.0f);
            }
            SyncScalarToV();  // BW-18：标量 kill/d32 写 → 本 visit V 侧（prep Exp / Gather）读
        }

        auto aRow = aRowBuf_.Get<float>();
        auto sa0 = sa0Buf_.Get<float>();
        auto sa1 = sa1Buf_.Get<float>();
        auto sb0 = sb0Buf_.Get<float>();
        auto sb1 = sb1Buf_.Get<float>();
        auto idx = gatherIdxBuf_.Get<uint32_t>();
        auto h0 = h0Buf_.Get<float>();
        auto deltaU32 = deltaU32Buf_.Get<float>();
        auto dx0v = dx0Buf_.Get<float>();
        auto redA = reduceDstABuf_.Get<float>();
        auto redB = reduceDstBBuf_.Get<float>();
        auto varB32 = varB_ ? varB32Buf_.Get<float>() : redB;
        auto varC32 = varC_ ? varC32Buf_.Get<float>() : redB;
        float *bPostSrc = (kind == KIND_T0) ? bPost2_ : bPost_;
        float *aFirstSrc = (kind == KIND_T0) ? aFirst2_ : aFirstCarry_;
        float *bPostTgt = (kind == KIND_TILE1) ? bPost2_ : bPost_;
        float *aFirstTgt = (kind == KIND_TILE1) ? aFirst2_ : aFirstCarry_;

        if (varB_) CopyInVarRow(inQvarB_, varBGm_, b, d, 0, gt);
        if (varC_) CopyInVarRow(inQvarC_, varCGm_, b, d, 0, gt);

        // R1 标量链优化（fwd R2 同款，tile 级不变量外提 / P4 视图复用 / P9 消热循环构造）：
        //  - 数据区视图 a0D/a1D/b0D/b1D 一次构造：起始侧固定 (sa0,sb0) 后不再 per-s 翻转
        //   （数值中性：prep 每 s 全量重写起始侧数据区 count=cnt 覆盖全部有效 lane，pad/kill/
        //    seed 区两侧对称且写双份，上轮遗留值在扫描读取前被覆盖）；
        //  - bRowV/cRowV 条件 Get（对应 InitBuffer 条件分配，不可无条件 Get），消 per-s Get；
        //  - totalSteps 与原 dd 循环步数定义一致（dd<TILE_T && dd<tvl 的迭代次数），前缀/
        //    后缀两扫描共用同一步数与终态奇偶：curIsA0 每 tile 一算，aFin/bFin（终态结果侧）
        //    与 aDead/bDead/bDeadF（对侧死缓冲，消费链 h̃ 垫高/Gather 源）单次选择——
        //    消 per-s 8 次 selects、每步 8 次交换赋值与循环尾复位块的 6 次赋值。
        auto a0D = sa0[PAD_LANES];
        auto a1D = sa1[PAD_LANES];
        auto b0D = sb0[PAD_LANES];
        auto b1D = sb1[PAD_LANES];
        AscendC::LocalTensor<float> bRowV, cRowV;
        if (!varB_) {
            bRowV = bRowBuf_.Get<float>();
        }
        if (!varC_) {
            cRowV = cRowBuf_.Get<float>();
        }
        const bool runD1 = tvl > 1;   // A 相 Gather 步守卫（= 原循环 dd=1/2/4 条件）
        const bool runD2 = tvl > 2;
        const bool runD4 = tvl > 4;
        int64_t totalSteps = 0;
        for (int64_t dd0 = 1; dd0 < TILE_T && dd0 < tvl; dd0 <<= 1) {
            ++totalSteps;
        }
        const bool curIsA0 = (totalSteps & 1) == 0;
        auto aFin = curIsA0 ? a0D : a1D;    // 终态扫描结果视图（tile 内 s 不变；前缀/后缀共用）
        auto bFin = curIsA0 ? b0D : b1D;
        auto aDead = curIsA0 ? a1D : a0D;   // 对侧死缓冲（后缀扫描已消费，消费链 scratch）
        auto bDead = curIsA0 ? b1D : b0D;   // h̃ 垫高复制落点（Gather 源需含 1 个 pad 位布局）
        auto bDeadF = curIsA0 ? sb1 : sb0;  // bDead 的 full 视图（h̃_{t-1} Gather 源）
        const bool fullTile = (tvl == TILE_T);          // seed 写门控外提（tile 内不变）
        const bool pow2Cnt = (cnt & (cnt - 1)) == 0;    // lane-0 修正门控外提（tile 内不变）

        for (int64_t s = 0; s < dstate_; ++s) {
            if (varB_ && s + 1 < dstate_) CopyInVarRow(inQvarB_, varBGm_, b, d, s + 1, gt);
            if (varC_ && s + 1 < dstate_) CopyInVarRow(inQvarC_, varCGm_, b, d, s + 1, gt);

            // ---- prep：a=Exp(Δ·A)（固定写 sa0）；b̃ 单写扫描 cur=sb0（a_prev 已改乘积式，原 b0k 双写废止） ----
            const float Aval = aRow.GetValue(static_cast<uint32_t>(s));
            const float Bval = varB_ ? 0.0f : bRowV.GetValue(static_cast<uint32_t>(s));
            const float Cval = varC_ ? 0.0f : cRowV.GetValue(static_cast<uint32_t>(s));
            AscendC::Muls(tmpA, d32, Aval, cnt);
            AscendC::Exp(a0D, tmpA, cnt);
            if (varB_) {
                auto vB = inQvarB_.DeQue<INPUT_T>();
                InputCast(varB32, vB, cnt);
                inQvarB_.FreeTensor(vB);
                AscendC::Mul(b0D, deltaU32, varB32, cnt);  // b̃ → 扫描 cur
            } else {
                // 静 B：b̃ 缩放域 =Δu（h̃ 随之缩放；B 因子在 w 域与 dB/dC 派生式逐项消去）
                AscendC::Muls(b0D, deltaU32, 1.0f, cnt);
            }

            // 状态组屏障：prep V 完成 → aFirst/上状态归约标量回读
            // BW-18：PipeBarrier<PIPE_V> 仅有管线内排序、无 V→S 跨 PIPE 序，V→标量回读改 V_S 事件对。
            SyncVToScalar();
            const float aFirst = a0D.GetValue(0);  // lane 512 = a_0（a0D 视图 lane 0；prep 固定写 sa0）
            aFirstTgt[s] = aFirst;
            if (s > 0) {
                dAAcc_[s - 1] += redA.GetValue(0);
                if (hasBCReduce_) dBCAcc_[s - 1] += redB.GetValue(0);
                bPostTgt[s - 1] = dx0v.GetValue(0);  // 单槽 carry：上一状态 dx（屏障见状态组开头）
            }
            if (s == 0 && hasD_) dDAcc_ += redA.GetValue(0);

            // ---- 前缀扫描（fwd 同款 Hillis-Steele；R1 相位拆分，侧序实参硬编码）----
            if (runD1) {
                ScanStepGather(1, cnt, sa0, sb0, a0D, b0D, a1D, b1D, idx);
            }
            if (runD2) {
                ScanStepGather(2, cnt, sa1, sb1, a1D, b1D, a0D, b0D, idx);
            }
            if (runD4) {
                ScanStepGather(4, cnt, sa0, sb0, a0D, b0D, a1D, b1D, idx);
            }
            int64_t dd = 8;
            while (dd < TILE_T && dd < tvl) {
                ScanStepAligned(dd, cnt, sa1, sb1, a1D, b1D, a0D, b0D);
                const int64_t dd2 = dd << 1;
                // 偶步内层守卫：与原循环逐步条件一致（dd2 可能已越过 tvl）
                if (dd2 < TILE_T && dd2 < tvl) {
                    ScanStepAligned(dd2, cnt, sa0, sb0, a0D, b0D, a1D, b1D);
                }
                dd = dd2 << 1;
            }

            // ---- 前缀衔接 + h̃ 复排（h = aFold·hRun + bFold；aFin/bFin：R1 终态视图）----
            const float hRun = (kind == KIND_TILE1) ? (P0_[s] * hRun0_[s] + Q0_[s]) : hRun0_[s];
            if (hRun == 0.0f) {
                AscendC::Muls(h0, bFin, 1.0f, cnt);
            } else {
                AscendC::Muls(h0, aFin, hRun, cnt);
                AscendC::Add(h0, h0, bFin, cnt);
            }

            // ---- 后缀扫描起点（R1）：直线化后侧序由实参硬编码、终态由 curIsA0 承载，
            // 原强制复位块（ping-pong 落点奇偶漂移防御，3.1 根因 B）的语义已由固定角色
            // 等价覆盖——af 固定重建于 sa0、w 固定写入 sb0、后缀扫描固定自 (sa0,sb0) 起。----
            // ---- w 构造（按 B/C 组合；前缀折叠已消费，sb0 可安全覆写）----
            if (varC_) {
                auto vC = inQvarC_.DeQue<INPUT_T>();
                InputCast(varC32, vC, cnt);
                inQvarC_.FreeTensor(vC);
            }
            if (!varB_ && !varC_) {
                AscendC::Muls(b0D, dout32, Bval * Cval, cnt);
            } else if (!varB_ && varC_) {
                AscendC::Mul(b0D, dout32, varC32, cnt);
                AscendC::Muls(b0D, b0D, Bval, cnt);
            } else if (varB_ && !varC_) {
                AscendC::Muls(b0D, dout32, Cval, cnt);
            } else {
                AscendC::Mul(b0D, dout32, varC32, cnt);
            }
            const float bPostSeed = bPostSrc[s];
            // seed 边界项的 a 因子必须取“被携带侧首 lane a”（头注释 KIND_* 契约：
            // T0=aFirst2·bPost2、TILE1/SINGLE=跨 chunk aFirstCarry·bPost），而非本 tile
            // a_0——af[cnt-1] 被 kill 中和为 1 后，逆扫描边界项=seedW·Π_{i..cnt-2}af，
            // a(cnt) 因子只能折入 seedW；误用本 tile a_0 时，凡携带非零的 tile 其整条
            // dx 链被污染（BW-12 实锤：T0 恒坏、TILE1 随跨 chunk 携带非零而坏）。
            const float seedW = aFirstSrc[s] * bPostSeed;
            if (fullTile) {  // 满 tile：seed 写 lane 512+cnt'=1536（尾块 seedW≡0 由 kill 零写承担）
                const uint32_t seedLane = static_cast<uint32_t>(PAD_LANES + cnt);
                AscendC::Duplicate(sb0[seedLane], seedW, 1);
                AscendC::Duplicate(sb1[seedLane], seedW, 1);
            }

            // ---- 后缀预备：af[i]=a[i+1]（Gather 读 kill/seed-a lane 使 af[count'-1]=1）----
            AscendC::Exp(sa1[PAD_LANES], tmpA, cnt);
            AscendC::Gather(sa0[PAD_LANES], sa1, idx,
                            static_cast<uint32_t>((PAD_LANES + 1) * sizeof(float)), cnt);
            // ---- 后缀扫描（右移 Hillis-Steele；R1：起始侧固定 (sa0,sb0)，侧序实参硬编码，
            //      终态由 curIsA0 承载——与原动态 ping-pong 逐指令等价）----
            if (runD1) {
                SufScanStepGather(1, cnt, sa0, sb0, a0D, b0D, a1D, b1D, idx);
            }
            if (runD2) {
                SufScanStepGather(2, cnt, sa1, sb1, a1D, b1D, a0D, b0D, idx);
            }
            if (runD4) {
                SufScanStepGather(4, cnt, sa0, sb0, a0D, b0D, a1D, b1D, idx);
            }
            dd = 8;
            while (dd < TILE_T && dd < tvl) {
                SufScanStepAligned(dd, cnt, sa1, sb1, a1D, b1D, a0D, b0D);
                const int64_t dd2 = dd << 1;
                // 偶步内层守卫：与原循环逐步条件一致（dd2 可能已越过 tvl）
                if (dd2 < TILE_T && dd2 < tvl) {
                    SufScanStepAligned(dd2, cnt, sa0, sb0, a0D, b0D, a1D, b1D);
                }
                dd = dd2 << 1;
            }
            // dx = bFin（后缀 W 折叠终值，R1 终态视图）；afFin = aFin[0]
            if (pow2Cnt) {  // count' 为 2 的幂：扫描窗恰不含 seed lane → lane-0 修正（门控外提）
                AscendC::Muls(tmpA, aFin, seedW, 1);
                AscendC::Add(bFin, bFin, tmpA, 1);
            }
            AscendC::Muls(dx0v, bFin, 1.0f, 1);  // dx 标量单槽 carry（dst 首址恒 32B 对齐；消费先于下轮覆写，见 659/831）

            // ---- 消费链 + 归约 + 变量行原子写 ----
            if (!varB_) {
                // 静态 B：du += dx·Δ；ddelta += dx·u + dx·A·a_prev（5 ops/状态，含运行和 Add）
                // a_prev = a_t·h̃_{t−1}（乘积式；方案 L30 恒等式 a_prev=h̃−b̃=a_t·h̃_{t−1}）。
                // 弃用 Sub 减法式：h̃~1e3 时 fp32 绝对误差 ~1e-4，a_t·h̃_{t−1} 真值 ~1e-4
                // 被完全核销（BW-14/15 实测 dA 偏 -6.7e-4）；乘积路径相对精度 ~1e-7。
                // R1：h̃ 复制垫高落死侧 bDead/bDeadF（原动态 bNext 同语义——终态对侧），
                // a_t 覆写终态 aFin（afFin 已被 lane-0 修正消费）、h̃_{t-1} Gather 源 bDeadF。
                AscendC::Muls(bDead, h0, 1.0f, cnt);              // h̃ 垫高复制 → 死侧（Gather 源）
                AscendC::Exp(aFin, tmpA, cnt);                    // a_t = exp(Δ·A)（tmpA=Δ·A 跨段存续）
                AscendC::Gather(aDead, bDeadF, idx,
                                static_cast<uint32_t>((PAD_LANES - 1) * sizeof(float)), cnt);  // h̃_{t−1} 左移 1 lane
                AscendC::Mul(aFin, aFin, aDead, cnt);             // a_prev = a_t·h̃_{t−1}
                AscendC::Duplicate(aFin, aFirst * hRun, 1);       // lane0 = a_0·hRun（精确标量补）
                AscendC::Mul(bDead, bFin, d32, cnt);              // dx·Δ（h̃ 复制已消费，死侧转作链 temp）
                AscendC::Add(du32, du32, bDead, cnt);
                AscendC::Mul(bDead, bDead, aFin, cnt);            // dx·Δ·a_prev
                AscendC::ReduceSum(redA, bDead, sTmpBuf_.Get<float>(), static_cast<int32_t>(tvl));
                AscendC::Muls(aFin, aFin, Aval, cnt);             // A·a_prev
                AscendC::Mul(tmpA, bFin, u32, cnt);               // dx·u
                AscendC::Mul(bDead, bFin, aFin, cnt);             // dx·A·a_prev
                AscendC::Add(tmpA, tmpA, ddelta32, cnt);
                AscendC::Add(ddelta32, tmpA, bDead, cnt);
            } else {
                // 变量 B：du += dx·B·Δ；ddelta += dx·B·u + dx·A·a_prev；dB 变量行 = dx·Δ·u
                // a_prev 乘积式（同静态 B 路径注释：消除 h̃−b̃ 减法核销）
                // R1：视图重绑定同静态 B 路径（h̃ 落死侧 / a_t 覆写 aFin / Gather 源 bDeadF）。
                AscendC::Muls(bDead, h0, 1.0f, cnt);              // h̃ 垫高复制 → 死侧
                AscendC::Exp(aFin, tmpA, cnt);                    // a_t = exp(Δ·A)
                AscendC::Gather(aDead, bDeadF, idx,
                                static_cast<uint32_t>((PAD_LANES - 1) * sizeof(float)), cnt);  // h̃_{t−1}
                AscendC::Mul(aFin, aFin, aDead, cnt);             // a_prev
                AscendC::Duplicate(aFin, aFirst * hRun, 1);       // lane0 精确补
                AscendC::Mul(tmpA, bFin, varB32, cnt);            // dx·B
                AscendC::Mul(bDead, tmpA, u32, cnt);              // dx·B·u（h̃ 复制已消费）
                AscendC::Mul(tmpA, tmpA, d32, cnt);               // dx·B·Δ
                AscendC::Add(du32, du32, tmpA, cnt);
                AscendC::Mul(tmpA, bFin, d32, cnt);               // dx·Δ（前值已消费，复用）
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evRow_);
                AscendC::Mul(rowStageBuf_.Get<float>(), tmpA, u32, cnt);  // dB 行
                AscendC::Mul(tmpA, tmpA, aFin, cnt);              // dx·Δ·a_prev
                AscendC::ReduceSum(redA, tmpA, sTmpBuf_.Get<float>(), static_cast<int32_t>(tvl));
                AscendC::Muls(aFin, aFin, Aval, cnt);
                AscendC::Mul(tmpA, bFin, aFin, cnt);              // dx·A·a_prev
                AscendC::Add(bDead, bDead, ddelta32, cnt);
                AscendC::Add(ddelta32, bDead, tmpA, cnt);
                // dB 变量行原子写（VECOUT 源；count=cnt·4B 恒 32B 对齐）
                const int64_t rowOff = ((b * (dim_ / dimPerGroup_) + d / dimPerGroup_) * dstate_ + s) * seqlenPad_
                                       + gt * TILE_T;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
                AscendC::SetAtomicAdd<float>();
                AscendC::DataCopy(dBPadGm_[rowOff], rowStageBuf_.Get<float>(), cnt);
                AscendC::DisableDmaAtomic();
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evRow_);
            }
            // dBC 归约 / dC 行（按组合）
            if (!varC_) {
                // 静静 / 变B静C：dBC = Σ dout'·h̃（dB/dC 静态行 pair 末派生/直写）
                AscendC::Mul(tmpA, dout32, h0, cnt);
                AscendC::ReduceSum(redB, tmpA, sTmpBuf_.Get<float>(), static_cast<int32_t>(tvl));
            } else if (!varB_) {
                // 静B变C：dC 变量行 = dout'·B·h̃；dBC = Σ dout'·C_t·h̃（dB 静态行 pair 末直写）
                AscendC::Mul(tmpA, dout32, h0, cnt);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evRow_);
                AscendC::Muls(rowStageBuf_.Get<float>(), tmpA, Bval, cnt);
                AscendC::Mul(tmpA, tmpA, varC32, cnt);
                AscendC::ReduceSum(redB, tmpA, sTmpBuf_.Get<float>(), static_cast<int32_t>(tvl));
                const int64_t rowOff = ((b * (dim_ / dimPerGroup_) + d / dimPerGroup_) * dstate_ + s) * seqlenPad_
                                       + gt * TILE_T;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
                AscendC::SetAtomicAdd<float>();
                AscendC::DataCopy(dCPadGm_[rowOff], rowStageBuf_.Get<float>(), cnt);
                AscendC::DisableDmaAtomic();
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evRow_);
            } else {
                // 变变：dC 变量行 = dout'·h̃（1 op 直行）
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evRow_);
                AscendC::Mul(rowStageBuf_.Get<float>(), dout32, h0, cnt);
                const int64_t rowOff = ((b * (dim_ / dimPerGroup_) + d / dimPerGroup_) * dstate_ + s) * seqlenPad_
                                       + gt * TILE_T;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
                AscendC::SetAtomicAdd<float>();
                AscendC::DataCopy(dCPadGm_[rowOff], rowStageBuf_.Get<float>(), cnt);
                AscendC::DisableDmaAtomic();
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evRow_);
            }
        }

        // du 的 D 跳连项（s 无关，每 tile 恰一次）：原置于 s 循环内按状态重复累加，
        // ds>1 时 D·dout' 被 double-count（BW-07 实测 du got≈2×exp）。V 流保序：
        // 位于循环内末次 du 累加之后、输出 cast 之前；dout32/dVal_ 生命周期覆盖至此。
        if (hasD_) {
            AscendC::Axpy(du32, dout32, dVal_, cnt);  // du += D·dout'（MemBase 4 参 Axpy）
        }

        // ---- 循环后：末端提取 + invF/bias + 输出存出 ----
        SyncVToScalar();  // BW-18：V_S 事件对（原 PIPE_V 屏障无 V→S 跨管线排序）
        dAAcc_[dstate_ - 1] += redA.GetValue(0);
        if (hasBCReduce_) dBCAcc_[dstate_ - 1] += redB.GetValue(0);
        bPostTgt[dstate_ - 1] = dx0v.GetValue(0);  // 单槽 carry：末状态 dx（barrier 行 826 已保证）
        if (softplus_) {
            AscendC::Mul(ddelta32, ddelta32, invF, cnt);
        }
        if (hasBias_) {
            AscendC::ReduceSum(reduceDstCBuf_.Get<float>(), ddelta32, sTmpBuf_.Get<float>(),
                               static_cast<int32_t>(tvl));
        }
        SyncVToScalar();  // BW-18：归约回读（归约内部自带 V_S，此处统一显式同步）
        if (hasBias_) dBiasAcc_ += reduceDstCBuf_.Get<float>().GetValue(0);

        // 输出 cast（窄化目的取空闲缓冲跨缓冲视图，不做同缓冲原地窄化——fwd 无此前例）。
        // 红线：cast 目的禁用扫描缓冲 sa0/sa1/sb0/sb1——其左 pad [0,512) 是 Init 一次填充的
        // 前缀恒等区（a=1/b=0），而 cast 自字节 0 起写（fp16 时 cnt 元素恰覆盖整个 pad）；
        // TILE1/SINGLE cast 污染 pad 后，同 chunk T0 的前缀扫描 Gather base=(PAD_LANES-dd)/
        // shift 视图读到 ddelta/dz 垃圾值（小 dd 命中高 pad lane、无 A-窗衰减 → 全 lane 污染）
        // → T0 的 bFold/aFold 垃圾 → h̃ 垃圾 → ddelta/dB/dC/dA 全坏、du（后缀链，只读右侧）
        // 免疫——BW-12「T0 恒坏、T1 净」根因。跨 chunk 时还会污染后续 visit A 的 P0/Q0。
        auto duCast = tmpABuf_.Get<INPUT_T>();
        OutputCast(duCast, du32, cnt);
        auto ddCast = h0Buf_.Get<INPUT_T>();        // h0 循环后死缓冲（末状态 h̃ 已被 dBC 消费）
        OutputCast(ddCast, ddelta32, cnt);
        auto dzCast = deltaU32Buf_.Get<INPUT_T>();  // deltaU32 循环后死缓冲；下 visit 重发（T0 O2/TILE1 prep）前有 visit 首 MTE3_V Wait 守护
        if (hasZ_) {
            OutputCast(dzCast, dz32Buf_.Get<float>(), cnt);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
        AscendC::DataCopyPad(duGm_[tileUOff], duCast, outCp);
        AscendC::DataCopyPad(ddeltaGm_[tileUOff], ddCast, outCp);
        if (hasZ_) {
            AscendC::DataCopyPad(dzGm_[tileUOff], dzCast, outCp);  // dz 存出批量后移（竞态规避）
        }
        if (recompute_) {
            auto ozCast = b0kBuf_.Get<INPUT_T>();
            OutputCast(ozCast, outZ32Buf_.Get<float>(), cnt);
            AscendC::DataCopyPad(outZGm_[tileUOff], ozCast, outCp);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);  // 本 visit 有存出，末尾放行供下一 visit / 下一 pair 首 Wait
        prevVisitHadMte3_ = true;
    }

    // ---------- pair 末静态行装配 + 原子写出（dA/静态dB/静态dC/dD/dbias；单次 SetAtomicAdd 批量）----------
    __aicore__ inline void MaterializeAtomic(int64_t d)
    {
        auto rv = rowVecBuf_.Get<float>();
        const int64_t dsp = dstatePad_;
        // 跨 pair rowVec 保护：等上一 pair（或 Process 起始放行）的原子行存出完成，
        // 才能 V 写 rowVec（与 var 行共用 token，全局 1:1 守恒）
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evRow_);
        for (int64_t s = 0; s < dsp; ++s) {
            rv.SetValue(static_cast<uint32_t>(s), s < dstate_ ? dAAcc_[s] : 0.0f);
        }
        if (!varB_ && !varC_) {
            // 静静：dBCrow 暂存 → 末尾 2 Mul 派生 dBRow=dBC·C、dCRow=dBC·B（各 1 Mul）
            for (int64_t s = 0; s < dsp; ++s) {
                rv.SetValue(static_cast<uint32_t>(3 * dsp + s), s < dstate_ ? dBCAcc_[s] : 0.0f);
            }
        } else if (!varB_ && varC_) {
            // 静B变C：dB 静态行 = dBC 直写
            for (int64_t s = 0; s < dsp; ++s) {
                rv.SetValue(static_cast<uint32_t>(dsp + s), s < dstate_ ? dBCAcc_[s] : 0.0f);
            }
        } else if (varB_ && !varC_) {
            // 变B静C：dC 静态行 = dBC 直写
            for (int64_t s = 0; s < dsp; ++s) {
                rv.SetValue(static_cast<uint32_t>(2 * dsp + s), s < dstate_ ? dBCAcc_[s] : 0.0f);
            }
        }
        if (hasD_) {
            rv.SetValue(static_cast<uint32_t>(4 * dsp), dDAcc_);
            for (int64_t i = 1; i < 8; ++i) rv.SetValue(static_cast<uint32_t>(4 * dsp + i), 0.0f);
        }
        if (hasBias_) {
            rv.SetValue(static_cast<uint32_t>(4 * dsp + 8), dBiasAcc_);
            for (int64_t i = 9; i < 16; ++i) rv.SetValue(static_cast<uint32_t>(4 * dsp + i), 0.0f);
        }
        AscendC::PipeBarrier<PIPE_ALL>();  // SetValue 批次可见 → V 侧 Mul
        if (!varB_ && !varC_) {
            auto bRow = bRowBuf_.Get<float>();
            auto cRow = cRowBuf_.Get<float>();
            AscendC::Mul(rv[dsp], rv[3 * dsp], cRow, static_cast<int32_t>(dstate_));
            AscendC::Mul(rv[2 * dsp], rv[3 * dsp], bRow, static_cast<int32_t>(dstate_));
            for (int64_t s = dstate_; s < dsp; ++s) {
                rv.SetValue(static_cast<uint32_t>(dsp + s), 0.0f);
                rv.SetValue(static_cast<uint32_t>(2 * dsp + s), 0.0f);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV2Mte3_);
        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopy(dAPadGm_[d * dstatePad_], rv, static_cast<int32_t>(dstatePad_));
        if (!varB_) {
            AscendC::DataCopy(dBPadGm_[d * dstatePad_], rv[dsp], static_cast<int32_t>(dstatePad_));
        }
        if (!varC_) {
            AscendC::DataCopy(dCPadGm_[d * dstatePad_], rv[2 * dsp], static_cast<int32_t>(dstatePad_));
        }
        if (hasD_) {
            AscendC::DataCopy(dDPadGm_[d * 8], rv[4 * dsp], 8);
        }
        if (hasBias_) {
            AscendC::DataCopy(dbiasPadGm_[d * 8], rv[4 * dsp + 8], 8);
        }
        AscendC::DisableDmaAtomic();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evRow_);  // 原子行存出完成，归还 token（下一 pair / 下一 var 行供货）
    }
    // 事件 ID（SYNC-04/07：同 id Set/Wait 1:1 配对；经 TPipe AllocEventID 独占分配）
    uint8_t evV2Mte3_ = 0;  // V→MTE3：存出前等 V 侧算完
    uint8_t evMte3V_ = 0;   // MTE3→V：仅有存出的 visit 末放行 / 有存出前驱的 visit 首 Waiting
    uint8_t evMte2V_ = 0;   // MTE2→V：dout/z/out 批量内联搬入（TBuf+手工事件，弃 TQue）
    uint8_t evRow_ = 0;     // MTE3→V：rowStage 原子行 token（var 行与 MaterializeAtomic 共用，全局 1:1）
    uint8_t evV2S_ = 0;     // V→S：V 结果标量回读前真同步（PipeBarrier<PIPE_V> 无 V→S 跨 PIPE 序，仅 intra-pipe）
    uint8_t evS2V_ = 0;     // S→V：标量 UB 写 → 本 visit V 侧读（官方归约实现同款 S_V 事件对）

    AscendC::TPipe pipe_;
    // GM
    AscendC::GlobalTensor<INPUT_T> uGm_, deltaGm_, doutGm_, varBGm_, varCGm_;
    AscendC::GlobalTensor<INPUT_T> zGm_, outGm_, dzGm_, outZGm_, duGm_, ddeltaGm_;
    AscendC::GlobalTensor<float> aGm_, bGm_, cGm_, dGm_, biasGm_, xGm_;
    AscendC::GlobalTensor<float> dAPadGm_, dBPadGm_, dCPadGm_, dDPadGm_, dbiasPadGm_;
    // 队列（4 条 VECIN × 深度 2 = 8 恰满 V_MTE2 事件池，禁止再增——见文件头 P-24 结论）
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQu_, inQd_, inQvarB_, inQvarC_;
    // TBuf（VECCALC）
    AscendC::TBuf<AscendC::TPosition::VECCALC> u32Buf_, u32T0Buf_, d32Buf_, d32T0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawDeltaBuf_, rawDeltaT0Buf_, deltaU32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dout32Buf_, du32Buf_, ddelta32Buf_, tmpABuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sa0Buf_, sa1Buf_, sb0Buf_, sb1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> h0Buf_, b0kBuf_, varB32Buf_, varC32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> doutStageBuf_, zStageBuf_, outStageBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> z32Buf_, out32Buf_, dz32Buf_, outZ32Buf_;  // gate 复用 h0Buf_
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIdxBuf_, aRowBuf_, bRowBuf_, cRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sTmpBuf_, xPrefixBuf_, dx0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceDstABuf_, reduceDstBBuf_, reduceDstCBuf_;
    // TBuf（VECOUT：连续 DataCopy UB→GM 源位置硬件约束）
    AscendC::TBuf<AscendC::TPosition::VECOUT> rowVecBuf_, rowStageBuf_;
    // tiling / flags
    int64_t batch_ = 0, dim_ = 0, seqlen_ = 0, dstate_ = 0, dimPerGroup_ = 1;
    int64_t nChunks_ = 0, numPairs_ = 0, pairsPerCore_ = 0;
    int64_t dstatePad_ = 0, seqlenPad_ = 0, totalTiles_ = 0;
    int64_t pairUBase_ = 0, pairXBase_ = 0;
    uint32_t flags_ = 0;
    bool hasD_ = false, hasZ_ = false, hasBias_ = false, softplus_ = false;
    bool varB_ = false, varC_ = false, recompute_ = false, hasX_ = false, hasBCReduce_ = false;
    bool prevVisitHadMte3_ = false;  // 上一 visit 是否有 MTE3 存出（决定下一 visit 首 Wait）

    float dVal_ = 0.0f, biasVal_ = 0.0f;
    // pair 级标量累加/反向携带（dstate ≤ 256，标量域驻留）
    float dAAcc_[DSTATE_MAX];
    float dBCAcc_[DSTATE_MAX];
    float bPost_[DSTATE_MAX];
    float bPost2_[DSTATE_MAX];
    float aFirstCarry_[DSTATE_MAX];
    float aFirst2_[DSTATE_MAX];
    float P0_[DSTATE_MAX];
    float Q0_[DSTATE_MAX];
    float hRun0_[DSTATE_MAX];
    float dDAcc_ = 0.0f;
    float dBiasAcc_ = 0.0f;
};

// Tiling 计算（返回 usedCores, pairsPerCore, nChunks, dstatePad, seqlenPad；B1 红线：核数运行时查询）
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>
calc_selective_scan_bwd_tiling_params(int64_t batch, int64_t dim, int64_t seqlen, int64_t dstate)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t aivNum = static_cast<int64_t>(ascendcPlatform->GetCoreNumAiv());
    if (aivNum <= 0) aivNum = 1;
    int64_t numPairs = batch * dim;
    int64_t usedCores = std::min(aivNum, numPairs);
    if (usedCores < 1) usedCores = 1;
    int64_t pairsPerCore = (numPairs + usedCores - 1) / usedCores;
    int64_t nChunks = (seqlen + CHUNK_T - 1) / CHUNK_T;
    int64_t dstatePad = (dstate + 7) / 8 * 8;
    int64_t seqlenPad = (seqlen + 7) / 8 * 8;
    return std::make_tuple(usedCores, pairsPerCore, nChunks, dstatePad, seqlenPad);
}

// Kernel 函数（<<<>>> launch 语法；标量经 GM tiling 传入，16 参全指针签名）
template <typename INPUT_T>
__global__ __aicore__ __vector__ void selective_scan_bwd_kernel(
    GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR dout,
    GM_ADDR D, GM_ADDR z, GM_ADDR delta_bias,
    GM_ADDR x, GM_ADDR out, GM_ADDR dz, GM_ADDR out_z,
    GM_ADDR du, GM_ADDR ddelta, GM_ADDR tiling)
{
    SelectiveScanBwdKernel<INPUT_T> op;
    op.Init(u, delta, A, B, C, dout, D, z, delta_bias, x, out, dz, out_z, du, ddelta,
            reinterpret_cast<const __gm__ SsbTilingData *>(tiling));
    op.Process();
}

#define SELECTIVE_SCAN_BWD_LAUNCH_IMPL(DTYPE, SUFFIX)                                      \
    extern "C" void launch_selective_scan_bwd_##SUFFIX(                                    \
        uint32_t numBlocks,                                                                \
        GM_ADDR u, GM_ADDR delta, GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR dout,           \
        GM_ADDR D, GM_ADDR z, GM_ADDR delta_bias,                                          \
        GM_ADDR x, GM_ADDR out, GM_ADDR dz, GM_ADDR out_z,                                 \
        GM_ADDR du, GM_ADDR ddelta, GM_ADDR tiling, void* stream)                          \
    {                                                                                      \
        selective_scan_bwd_kernel<DTYPE><<<numBlocks, nullptr, stream>>>(                  \
            u, delta, A, B, C, dout, D, z, delta_bias, x, out, dz, out_z, du, ddelta, tiling); \
    }

SELECTIVE_SCAN_BWD_LAUNCH_IMPL(float, float)
SELECTIVE_SCAN_BWD_LAUNCH_IMPL(half, half)
SELECTIVE_SCAN_BWD_LAUNCH_IMPL(bfloat16_t, bfloat16)
