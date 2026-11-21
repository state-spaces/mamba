// Mamba-3 MIMO varlen backward, Blackwell tcgen05 edition (sm_100 / sm_103), Olala configuration: R=4, N=128,
// P=64, 32 rotary phases, one Q/K group, reduceO + Z gate + D skip, bf16 activations.
//
// Chunk-parallel backward over 32-token sub-blocks (128 MIMO rows, row = token * R + rank). With, per head,
// qhat / khat = rotated (q / k + bias) rows, PsiV = v * mimo_v, dO = dy * mimo_o * silu(z * mimo_z),
// l = cumsum of ADT within the sub-block (inclusive), f = gamma + shifted gamma, W[t][s] = exp(l_t - l_s) f_s (s < t),
// gamma_t (s = t), h_b = state entering sub-block b, D_b = gradient of the loss w.r.t. the state leaving it:
//
//   MODE_LOCAL  Sloc_b = khat^T (PsiV * exp(l_end - l) f),  dloc_b = qhat^T (dO * exp(l))       -> h, D (Triton scan)
//               (epilogue one item late, bf16 tiles staged in smem and written by TMA stores)
//   MODE_FUSED  per (sub-block, head), both sides on the same operands:
//     q side  (rows t):  G = qhat khat^T, dP = dO PsiV^T;  O1 = qhat h_b, O2 = (G * W) PsiV,
//             dqhat = (dO * exp(l)) h_b^T + (dP * W) khat
//             -> o = exp(l) O1 + O2 + D PsiV -> the gate gradients (dz, dmimo_o, dmimo_z); dq (rotation undone,
//                summed over the heads); the q part of the phase gradient; sum_{s<t} W A + exp(l_t) O1 . dO
//                (A = G * dP) and the diagonal (gamma) gradient sum_{s=t} A
//     kv side (rows s):  GT = khat qhat^T, dPT = PsiV dO^T;  kD = khat D_b, dPsi2 = (GT * W^T) dO,
//             dkhat = (PsiV * exp(l_end - l) f) D_b^T + (dPT * W^T) qhat
//             -> dPsiV = exp(l_end - l) f kD + dPsi2 + D dO -> dv, dmimo_v, dD; dk (summed over the heads); the k
//                part of the phase gradient; the trapezoid factor gradient sum_{t>s} exp(l_t - l_s) A
//                + exp(l_end - l_s) kD . PsiV
// The masked matrices go back to tensor memory as bf16 (over their own fp32 columns) and feed the second products as
// the A operand, so the transposed products are computed directly (no column reductions, no shared-memory staging
// of 128 x 128 tiles). The kv side also yields <h_{b+1}, D_b> = exp(tot) <h_b, D_b> + <Sloc_b, D_b> for dADT.
// Per-head reductions (bias, mimo and D gradients) are summed in a fixed order into per-(CTA, head) accumulators
// owned by the CTA (deterministic).
//
// One persistent CTA per SM (16 warps), work = sub-blocks (CTA-strided), the heads of a sub-block one after the other
// (the Q / K rows of a sub-block are read from L2 for every head). Per (sub-block, head): TMA of the raw rows (Q, K,
// V, dy, z, phases, both state tiles; the next head's rows are prefetched as soon as their buffers free up), all
// warps build the bf16 operands in place (bias + rotation, PsiV, dO and the scaled copies, token tables), one thread
// in each of warps 0 / 4 / 8 / 12 issues an independent UMMA chain (M = 128, fp32 accumulators in TMEM; G and O1 start
// as soon as qhat / khat are built, the kv side's first products as soon as the q side has read its columns), all
// warps run the epilogues (warp w: TMEM lanes 32 (w % 4), column group w / 4; reductions over lanes are
// reduce-scatters). The kernel is bound by the CUDA-core epilogue / operand work, not by the tensor cores.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace {
constexpr int R = 4, N = 128, P = 64, NQ = 32, C = 32, ROWS = C * R;
constexpr int THREADS = 512, QC = THREADS / 128;  // QC column groups in the epilogues
constexpr int TILE = ROWS * 64;   // one 128-row x 64-col bf16 tile (128B rows, SW128)
constexpr int TILE_B = TILE * 2;  // 16 KB
constexpr int CP = C + 1, NQP = NQ + 1, PP = P + 8;  // padded strides (bank-conflict-free columns / ranks)
constexpr int MODE_LOCAL = 0, MODE_FUSED = 1, SIDE_Q = 0, SIDE_KV = 1;
// tensor memory columns
// G | dP (fp32; the masked bf16 G*W / dP*W are written back over their first 64 columns) | O1 | O2 | dq (dk)
constexpr int TM_A = 0, TM_B = 128, TM_O1 = 256, TM_O2 = 320, TM_D = 384;

struct __align__(1024) Smem {
  __nv_bfloat16 q[2 * TILE];   // Q -> qhat   [n half][row][64]
  __nv_bfloat16 k[2 * TILE];   // K -> khat
  __nv_bfloat16 pv[TILE];      // PsiV [row][p]           (LOCAL: PsiV * exp(l_end - l) f)
  __nv_bfloat16 dob[TILE];     // dO [row][p]             (LOCAL: dO * exp(l))
  __nv_bfloat16 xq[TILE];      // dO * exp(l) (q side); then the epilogue reduction buffers (Red)
  __nv_bfloat16 xk[TILE];      // PsiV * exp(l_end - l) f (kv side)
  __nv_bfloat16 sth[TILE];     // h_b, bf16 [n][p]
  __nv_bfloat16 std_[TILE];    // D_b, bf16 [n][p]
  __nv_bfloat16 v[2][C * P], dy[2][C * P], z[2][C * P];  // raw [tok][p] (SW128), two-deep (next head prefetched)
  alignas(128) float angr[C * NQ];                      // raw phases [tok][j] (TMA)
  alignas(16) float cs[C * NQP];  // cos table [tok][j]
  alignas(16) float sn[C * NQP];  // sin table [tok][j]
  float tab1[C * CP];             // W[t][s] = exp(l_t - l_s) f_s (s < t), gamma_t (s = t)
  float tab2[C * CP];             // dec[s][t] = exp(l_t - l_s) (s < t)
  // per-head constants and token scalars, two-deep (the next head's are prefetched with cp.async)
  alignas(16) __nv_bfloat16 qb[2][R * N], kb[2][R * N];
  alignas(16) float mv[2][R * PP], mo[2][R * PP], mz[2][R * PP];
  alignas(16) float ell[2][C], ff[2][C], gam[2][C], sc[2][4];  // sc: tot, D skip
  alignas(16) float ainb[2][NQ];                                // base phase of the sub-block's forward chunk
  alignas(16) float red[3][QC][ROWS];
  float xr[THREADS / 32 + 4];  // <h_{b+1}, D_b> partials: per warp (exp(tot) <h_b, D_b>), per lane quarter (<Sloc_b, D_b>)
  unsigned long long tA, tB[2], m1[2], m2a[2], m2b[2];  // Q / K / state rows, V / dy / z / phases, products (per side)
  uint32_t tmem_base;
};
struct Red {  // epilogue reduction buffers, overlaid on Smem::xq once the q-side products are done with it
  float bias[4][R][N];         // per-warp-quarter partial bias gradients (dq / dk)
  float g1[4][R][P], g2[4][R][P];  // q: dmimo_o, dmimo_z partials; kv: dmimo_v partials
};
static_assert(sizeof(Red) <= TILE_B, "Red must fit in one tile");
struct Maps { CUtensorMap q, k, v, dy, z, ang, sth, std_; };

__device__ __forceinline__ int swz(int row, int col) {  // 128B swizzle, 64-element rows
  return row * 64 + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7);
}
__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(void* mbar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(count));
}
__device__ __forceinline__ void mbar_wait(void* mbar, unsigned parity) {
  unsigned done = 0;
  while (!done) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(smem_u32(mbar)), "r"(parity) : "memory");
  }
}
__device__ __forceinline__ void mbar_arrive_expect_tx(void* mbar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ uint64_t udesc(const void* p, uint32_t lbo, uint32_t sbo) {
  uint64_t d = (uint64_t)((smem_u32(p) & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
__device__ __forceinline__ uint64_t kdesc(const void* p) { return udesc(p, 16, 1024); }      // K-major SW128
__device__ __forceinline__ uint64_t mdesc(const void* p) { return udesc(p, TILE_B, 1024); }  // MN-major SW128
__host__ __device__ constexpr uint32_t idesc(int M, int Nn, int a_mn, int b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) |
         ((uint32_t)(Nn >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
__device__ __forceinline__ void umma(uint32_t tmem_d, uint64_t a, uint64_t b, uint32_t id, uint32_t accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               :: "r"(tmem_d), "l"(a), "l"(b), "r"(id), "r"(accum) : "memory");
}
__device__ __forceinline__ void umma_ts(uint32_t tmem_d, uint32_t tmem_a, uint64_t b, uint32_t id, uint32_t accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
               :: "r"(tmem_d), "r"(tmem_a), "l"(b), "r"(id), "r"(accum) : "memory");
}
__device__ __forceinline__ void umma_commit(void* mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" :: "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float* v) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float* v) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]));
}
__device__ __forceinline__ void tmem_ld8(uint32_t taddr, float* v) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
__device__ __forceinline__ void tmem_st8(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]));
}
template <int W> __device__ __forceinline__ void tmem_ldw(uint32_t taddr, float* v) {
  if constexpr (W == 32) tmem_ld32(taddr, v);
  else if constexpr (W == 16) tmem_ld16(taddr, v);
  else { static_assert(W == 8); tmem_ld8(taddr, v); }
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
template <int W> __device__ __forceinline__ void tmem_stw(uint32_t taddr, const uint32_t* r) {
  if constexpr (W == 16) tmem_st16(taddr, r);
  else { static_assert(W == 8); tmem_st8(taddr, r); }
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
union U8 { uint4 u; __nv_bfloat16 h[8]; __nv_bfloat162 h2[4]; };
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, int c0, int c1, void* mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tma_store2d(const CUtensorMap* m, int c0, int c1, const void* src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
               :: "l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1), "r"(smem_u32(src)) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void tma3d(void* dst, const CUtensorMap* m, int c0, int c1, int c2, void* mbar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp4z(void* dst, const void* src, bool ok) {  // zero-fill when !ok
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(smem_u32(dst)), "l"(src), "r"(ok ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void red_add(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ float silu_f(float x) { return __fdividef(x, 1.f + __expf(-x)); }
// Reduce-scatter step over the lanes `lane ^ MASK`: v[0 .. NV/2) <- the pair sums of the half this lane keeps (the
// upper half v[NV/2 ..) if lane & MASK); half the shuffles of a full butterfly per step.
template <int NV, int MASK>
__device__ __forceinline__ void rs_step(float* v, int lane) {
  const bool up = lane & MASK;
#pragma unroll
  for (int i = 0; i < NV / 2; ++i) {
    const float send = up ? v[i] : v[i + NV / 2], keep = up ? v[i + NV / 2] : v[i];
    v[i] = keep + __shfl_xor_sync(0xffffffffu, send, MASK);
  }
}

constexpr int NACC = R * N + 2 * R * P + 4;  // per (CTA, head) accumulators: bias grad | mimo grads | dD
struct Args {
  const __nv_bfloat16 *qbias, *kbias;  // [H, R, N] bf16
  const float *mv, *mo, *mz, *Dskip, *ell, *ff, *gam, *tot;
  const int *sb_start, *sb_len;
  const float* ain;                    // [*, H, NQ] phase entering the forward chunk of a sub-block (ang is relative)
  const int* sb_chunk;                 // [NB] that chunk
  __nv_bfloat16 *sloc, *dloc;          // LOCAL: [NB, H, N, P] (bf16)
  __nv_bfloat16 *dq, *dk;              // summed over the heads  [S, R, N]
  __nv_bfloat16 *dz, *dv;              // [S, H, P]
  float* dth;                          // phase gradient (q + k parts)   [S, H, NQ]
  float *dlrow, *dgam, *dfac;          // [H, S]
  float *accq, *acck;                  // [gridDim, H, NACC], zero-initialised, owned by the CTA
  float* cross;                        // [NB, H] <h_{b+1}, D_b> = exp(tot_b) <h_b, D_b> + <Sloc_b, D_b>
  int S, H, NB;
  unsigned long long* prof;            // optional phase stamps (block 0, first 64 items)
};

#define STAMP(k) do { if (a.prof && blockIdx.x == 0 && tid == 0 && it < 64) a.prof[it * 16 + (k)] = gtime(); } while (0)
template <int MODE>
__global__ void __launch_bounds__(THREADS, 1) bwd_tc_kernel(const __grid_constant__ Maps maps, const Args a) {
  extern __shared__ unsigned char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023));
  Red& RD = *reinterpret_cast<Red*>(S.xq);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, H = a.H, NB = a.NB;
  if (blockIdx.x >= NB) return;
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(smem_u32(&S.tmem_base)), "n"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
  }
  if (tid == 0) {
    mbar_init(&S.tA, 1); mbar_init(&S.tB[0], 1); mbar_init(&S.tB[1], 1);
    // the first products of a side are issued as independent chains by up to 4 threads (warps 0 / 4 / 8 / 12)
    mbar_init(&S.m1[0], MODE == MODE_LOCAL ? 2 : 4); mbar_init(&S.m1[1], 3);
    for (int i = 0; i < 2; ++i) { mbar_init(&S.m2a[i], 1); mbar_init(&S.m2b[i], 1); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = S.tmem_base;
  constexpr unsigned TXA = 4 * TILE_B + (MODE == MODE_LOCAL ? 0 : 2 * TILE_B), TXB = 3 * C * P * 2 + C * NQ * 4;
  // Work: sub-block b (CTA-strided), the heads one after the other. The loads of (b, h + 1) overlap (b, h): V / dy / z /
  // phases as soon as the operands of (b, h) are built (V / dy / z two-deep: the epilogue still reads them), Q / K /
  // state rows as soon as the last products are done (the epilogue keeps the qhat / khat values it needs in registers).
  auto issue_a = [&](int bb, int hh) {
    const int tt = a.sb_start[bb];
    mbar_arrive_expect_tx(&S.tA, TXA);
    tma2d(&S.q[0], &maps.q, 0, tt * R, &S.tA);
    tma2d(&S.q[TILE], &maps.q, 64, tt * R, &S.tA);
    tma2d(&S.k[0], &maps.k, 0, tt * R, &S.tA);
    tma2d(&S.k[TILE], &maps.k, 64, tt * R, &S.tA);
    if (MODE != MODE_LOCAL) {
      tma2d(S.sth, &maps.sth, 0, (bb * H + hh) * N, &S.tA);
      tma2d(S.std_, &maps.std_, 0, (bb * H + hh) * N, &S.tA);
    }
  };
  auto issue_b = [&](int bb, int hh, int rb) {
    const int tt = a.sb_start[bb];
    mbar_arrive_expect_tx(&S.tB[rb], TXB);
    tma3d(S.v[rb], &maps.v, 0, hh, tt, &S.tB[rb]);
    tma3d(S.dy[rb], &maps.dy, 0, hh, tt, &S.tB[rb]);
    tma3d(S.z[rb], &maps.z, 0, hh, tt, &S.tB[rb]);
    tma3d(S.angr, &maps.ang, 0, hh, tt, &S.tB[rb]);
  };
  auto prefetch_consts = [&](int bb, int hh, int cb) {  // head hh's constants, sub-block bb's token scalars
    for (int j = tid; j < R * N / 8; j += THREADS) {
      cp16(&S.qb[cb][8 * j], a.qbias + hh * R * N + 8 * j);
      cp16(&S.kb[cb][8 * j], a.kbias + hh * R * N + 8 * j);
    }
    for (int j = tid; j < R * P / 4; j += THREADS) {
      const int jj = (j >> 4) * PP + 4 * (j & 15);
      cp16(&S.mv[cb][jj], a.mv + hh * R * P + 4 * j);
      cp16(&S.mo[cb][jj], a.mo + hh * R * P + 4 * j);
      cp16(&S.mz[cb][jj], a.mz + hh * R * P + 4 * j);
    }
    if (tid < 3 * C) {
      const int w = tid / C, i = tid % C, tt = a.sb_start[bb];
      const bool ok = i < a.sb_len[bb];
      const float* src = w == 0 ? a.ell : w == 1 ? a.ff : a.gam;
      float* dst = w == 0 ? S.ell[cb] : w == 1 ? S.ff[cb] : S.gam[cb];
      cp4z(dst + i, src + (ok ? (long)hh * a.S + tt + i : 0), ok);
    } else if (tid == 3 * C) {
      cp4z(&S.sc[cb][0], a.tot + bb * H + hh, true);
    } else if (tid == 3 * C + 1) {
      cp4z(&S.sc[cb][1], a.Dskip + hh, true);
    } else if (tid >= 3 * C + 2 && tid < 3 * C + 2 + NQ / 4) {
      const int j = tid - (3 * C + 2);
      cp16(&S.ainb[cb][4 * j], a.ain + ((long)a.sb_chunk[bb] * H + hh) * NQ + 4 * j);
    }
    cp_commit();
  };
  if (tid == 0) { issue_a(blockIdx.x, 0); issue_b(blockIdx.x, 0, 0); }
  prefetch_consts(blockIdx.x, 0, 0);
  // epilogue geometry: TMEM lanes 32 (warp % 4), column group warp / 4
  const int q4 = warp & 3, cq = warp >> 2, row = 32 * q4 + lane, tok = row >> 2, rank = row & 3;
  const uint32_t tl = tmem + ((uint32_t)(32 * q4) << 16);
  // producer geometry (rotation units): fixed rank / column piece per thread
  const int cr = tid & 3, rr = (tid >> 3) & 3, cA = 8 * cr, cB = 32 + 8 * cr;
  auto unit_row = [](int u) { return (u >> 5) * 8 + ((u >> 2) & 1) * 4 + ((u >> 3) & 3); };
  constexpr int JW = NQ / QC;   // rotated pairs per thread in the dq (dk) epilogue
  constexpr int OW = P / QC;    // O / PsiV columns per thread

  auto issue_m1 = [&](auto side_tag, unsigned parts) {  // all threads: chain `part` (bit of `parts`) issued by thread 128 * part
    constexpr bool QS = decltype(side_tag)::value == SIDE_Q;
    const int part = tid >> 7;
    if ((tid & 127) != 0 || part >= (QS ? 4 : 3) || !((parts >> part) & 1)) return;
    tc_fence_after();
    const __nv_bfloat16* X = QS ? S.q : S.k;
    const __nv_bfloat16* Y = QS ? S.k : S.q;
    const __nv_bfloat16* RA = QS ? S.dob : S.pv;
    const __nv_bfloat16* RB = QS ? S.pv : S.dob;
    const __nv_bfloat16* ST = QS ? S.sth : S.std_;
    constexpr uint32_t ID = idesc(128, 128, 0, 0), ID_O = idesc(128, 64, 0, 1);
    if (part == 0) {
#pragma unroll
      for (int kk = 0; kk < 8; ++kk)  // G = qhat khat^T  |  GT
        umma(tmem + TM_A, kdesc(&X[(kk >> 2) * TILE + (kk & 3) * 16]), kdesc(&Y[(kk >> 2) * TILE + (kk & 3) * 16]), ID, kk > 0);
    } else if (part == 1) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) umma(tmem + TM_B, kdesc(&RA[kk * 16]), kdesc(&RB[kk * 16]), ID, kk > 0);  // dP | dPT
    } else if (part == 2) {
#pragma unroll
      for (int kk = 0; kk < 8; ++kk)  // O1 = qhat h  |  kD = khat D
        umma(tmem + TM_O1, kdesc(&X[(kk >> 2) * TILE + (kk & 3) * 16]), mdesc(&ST[kk * 16 * 64]), ID_O, kk > 0);
    } else {  // dq: (dO exp(l)) h^T
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) umma(tmem + TM_D, kdesc(&S.xq[kk * 16]), kdesc(&S.sth[kk * 16]), ID, kk > 0);
    }
    umma_commit(&S.m1[QS ? SIDE_Q : SIDE_KV]);
  };

  // LOCAL epilogue: row = state row n; bf16 tiles staged in sth / std_ (unused by LOCAL, SW128 like the TMA maps),
  // written by TMA stores issued after the next barrier (the previous stores' reads are waited at the item start)
  auto local_epi = [&]() {
    float x[OW], y[OW];
    tmem_ldw<OW>(tl + TM_O1 + OW * cq, x);
    tmem_ldw<OW>(tl + TM_O2 + OW * cq, y);
    tmem_wait_ld();
#pragma unroll
    for (int e = 0; e < OW; e += 8) {
      *reinterpret_cast<uint4*>(&S.sth[swz(row, OW * cq + e)]) = make_uint4(pack_bf16(x[e], x[e + 1]), pack_bf16(x[e + 2], x[e + 3]),
                                                                pack_bf16(x[e + 4], x[e + 5]), pack_bf16(x[e + 6], x[e + 7]));
      *reinterpret_cast<uint4*>(&S.std_[swz(row, OW * cq + e)]) = make_uint4(pack_bf16(y[e], y[e + 1]), pack_bf16(y[e + 2], y[e + 3]),
                                                                 pack_bf16(y[e + 4], y[e + 5]), pack_bf16(y[e + 6], y[e + 7]));
    }
  };
  auto local_store = [&](int pb, int ph) {  // thread 0, after a barrier that follows local_epi + fence_async
    tma_store2d(&maps.sth, 0, (pb * H + ph) * N, S.sth);
    tma_store2d(&maps.std_, 0, (pb * H + ph) * N, S.std_);
    bulk_commit();
  };
  int it = 0, pb_ = 0, ph_ = 0;
  for (int b = blockIdx.x; b < NB; b += gridDim.x) {
    const int t0 = a.sb_start[b], nv = a.sb_len[b];
    const bool okr = tok < nv;
    const long grow = (long)t0 * R + row;  // global MIMO row
    float dxq[4][JW], dxk[4][JW];          // dq, dk of this thread's columns, summed over the heads
#pragma unroll
    for (int x = 0; x < 4; ++x)
#pragma unroll
      for (int e = 0; e < JW; ++e) { dxq[x][e] = 0.f; dxk[x][e] = 0.f; }
    for (int h = 0; h < H; ++h, ++it) {
      float dthk0 = 0.f, dthk1 = 0.f;  // q part of this thread's phase gradient pairs
      const int rb = it & 1;
      int nb_ = b, nh_ = h + 1;
      if (nh_ == H) { nb_ = b + gridDim.x; nh_ = 0; }
      const bool has_next = nb_ < NB;
      STAMP(0);
      // ---------------------------------------------------------------- per-head constants and token scalars
      const int cbf = it & 1;
      cp_wait_all();
      if (MODE == MODE_LOCAL && tid == 0) bulk_wait_read();  // staging tiles free
      __syncthreads();                                   // this head's constants visible; the other buffer is free
      if (has_next) prefetch_consts(nb_, nh_, cbf ^ 1);
      const float* ellv = S.ell[cbf];
      const float* ffv = S.ff[cbf];
      const float* gamv = S.gam[cbf];
      const float* mvv = S.mv[cbf];
      const float* mov = S.mo[cbf];
      const float* mzv = S.mz[cbf];
      const float totv = S.sc[cbf][0], dskv = S.sc[cbf][1];
      mbar_wait(&S.tB[rb], (it >> 1) & 1);
      STAMP(1);
      // ---------------------------------------------------------------- cos / sin table, token tables
      for (int idx = tid; idx < C * NQ; idx += THREADS) {
        float th = S.angr[idx] + S.ainb[cbf][idx & (NQ - 1)];
        th = th - 6.283185307179586f * floorf(th * 0.15915494309189535f);
        float sv, cv;
        __sincosf(th, &sv, &cv);
        S.cs[(idx >> 5) * NQP + (idx & 31)] = cv;
        S.sn[(idx >> 5) * NQP + (idx & 31)] = sv;
      }
      if (MODE != MODE_LOCAL) {
        for (int idx = tid; idx < C * C; idx += THREADS) {
          const int i = idx >> 5, j = idx & 31;
          const bool valid = i < nv && j < nv;
          {  // W[t][s]
            const int t = i, s = j;
            S.tab1[i * CP + j] = !valid ? 0.f : s < t ? __expf(fminf(ellv[t] - ellv[s], 0.f)) * ffv[s] : s == t ? gamv[t] : 0.f;
          }
          {  // dec[s][t]
            const int s = i, t = j;
            S.tab2[i * CP + j] = (valid && s < t) ? __expf(fminf(ellv[t] - ellv[s], 0.f)) : 0.f;
          }
        }
      }
      __syncthreads();
      if (MODE == MODE_LOCAL && it > 0) {  // the previous item's products are done: its Q / K buffers take this item's
        STAMP(5);
        mbar_wait(&S.m1[0], (it - 1) & 1);  // rows, its epilogue runs under that TMA (and its MMAs ran under this
        tc_fence_after();                   // item's setup)
        STAMP(6);
        if (tid == 0) issue_a(b, h);
        local_epi();
        STAMP(7);
        tc_fence_before();
      }
      mbar_wait(&S.tA, it & 1);
      STAMP(2);
      // ---------------------------------------------------------------- operands
      {  // qhat, khat in place (bias in bf16, rotation of the (n, n + 64) pairs, n < 32)
        U8 bq[4], bk[4];
        bq[0].u = *reinterpret_cast<const uint4*>(&S.qb[cbf][rr * N + cA]);
        bq[1].u = *reinterpret_cast<const uint4*>(&S.qb[cbf][rr * N + cA + 64]);
        bq[2].u = *reinterpret_cast<const uint4*>(&S.qb[cbf][rr * N + cB]);
        bq[3].u = *reinterpret_cast<const uint4*>(&S.qb[cbf][rr * N + cB + 64]);
        bk[0].u = *reinterpret_cast<const uint4*>(&S.kb[cbf][rr * N + cA]);
        bk[1].u = *reinterpret_cast<const uint4*>(&S.kb[cbf][rr * N + cA + 64]);
        bk[2].u = *reinterpret_cast<const uint4*>(&S.kb[cbf][rr * N + cB]);
        bk[3].u = *reinterpret_cast<const uint4*>(&S.kb[cbf][rr * N + cB + 64]);
#pragma unroll 1
        for (int u = tid; u < ROWS * 4; u += THREADS) {
          const int r_ = unit_row(u), tk = r_ >> 2;
          const bool ok = tk < nv;
          const int oA = swz(r_, cA), oB = swz(r_, cB);
          U8 qv[4], kv[4];
          qv[0].u = *reinterpret_cast<const uint4*>(&S.q[oA]);
          qv[1].u = *reinterpret_cast<const uint4*>(&S.q[TILE + oA]);
          qv[2].u = *reinterpret_cast<const uint4*>(&S.q[oB]);
          qv[3].u = *reinterpret_cast<const uint4*>(&S.q[TILE + oB]);
          kv[0].u = *reinterpret_cast<const uint4*>(&S.k[oA]);
          kv[1].u = *reinterpret_cast<const uint4*>(&S.k[TILE + oA]);
          kv[2].u = *reinterpret_cast<const uint4*>(&S.k[oB]);
          kv[3].u = *reinterpret_cast<const uint4*>(&S.k[TILE + oB]);
#pragma unroll
          for (int x = 0; x < 4; ++x)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              qv[x].h2[j] = __hadd2(qv[x].h2[j], bq[x].h2[j]);
              kv[x].h2[j] = __hadd2(kv[x].h2[j], bk[x].h2[j]);
            }
          const float* cp = &S.cs[tk * NQP + cA];
          const float* sp = &S.sn[tk * NQP + cA];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float c_a = cp[2 * j], c_b = cp[2 * j + 1], s_a = sp[2 * j], s_b = sp[2 * j + 1];
            const float2 x0 = __bfloat1622float2(qv[0].h2[j]), x1 = __bfloat1622float2(qv[1].h2[j]);
            const float2 y0 = __bfloat1622float2(kv[0].h2[j]), y1 = __bfloat1622float2(kv[1].h2[j]);
            qv[0].h2[j] = __floats2bfloat162_rn(c_a * x0.x - s_a * x1.x, c_b * x0.y - s_b * x1.y);
            qv[1].h2[j] = __floats2bfloat162_rn(s_a * x0.x + c_a * x1.x, s_b * x0.y + c_b * x1.y);
            kv[0].h2[j] = __floats2bfloat162_rn(c_a * y0.x - s_a * y1.x, c_b * y0.y - s_b * y1.y);
            kv[1].h2[j] = __floats2bfloat162_rn(s_a * y0.x + c_a * y1.x, s_b * y0.y + c_b * y1.y);
          }
          if (!ok) {
#pragma unroll
            for (int x = 0; x < 4; ++x) { qv[x].u = make_uint4(0, 0, 0, 0); kv[x].u = make_uint4(0, 0, 0, 0); }
          }
          *reinterpret_cast<uint4*>(&S.q[oA]) = qv[0].u;
          *reinterpret_cast<uint4*>(&S.q[TILE + oA]) = qv[1].u;
          *reinterpret_cast<uint4*>(&S.q[oB]) = qv[2].u;
          *reinterpret_cast<uint4*>(&S.q[TILE + oB]) = qv[3].u;
          *reinterpret_cast<uint4*>(&S.k[oA]) = kv[0].u;
          *reinterpret_cast<uint4*>(&S.k[TILE + oA]) = kv[1].u;
          *reinterpret_cast<uint4*>(&S.k[oB]) = kv[2].u;
          *reinterpret_cast<uint4*>(&S.k[TILE + oB]) = kv[3].u;
        }
      }
      STAMP(14);
      if (MODE != MODE_LOCAL) {  // G = qhat khat^T and O1 = qhat h run while PsiV / dO are built
        fence_async();
        tc_fence_before();
        __syncthreads();
        issue_m1(std::integral_constant<int, SIDE_Q>{}, 0b0101u);
      }
      STAMP(15);
      // PsiV, dO and the scaled copies: (row, 16-byte piece); the rank and piece of a thread are the same in both
      // iterations (THREADS is a multiple of 32 rows x 8 pieces), so the per-head constants are read once
      static_assert(THREADS % 32 == 0 && (ROWS * 8) % THREADS == 0, "fixed rank / piece per thread");
      float cmv[8], cmo[8], cmz[8];
      {
        const int rk = (tid >> 3) & 3, p0 = 8 * (tid & 7);
        *reinterpret_cast<float4*>(cmv) = *reinterpret_cast<const float4*>(&mvv[rk * PP + p0]);
        *reinterpret_cast<float4*>(cmv + 4) = *reinterpret_cast<const float4*>(&mvv[rk * PP + p0 + 4]);
        *reinterpret_cast<float4*>(cmo) = *reinterpret_cast<const float4*>(&mov[rk * PP + p0]);
        *reinterpret_cast<float4*>(cmo + 4) = *reinterpret_cast<const float4*>(&mov[rk * PP + p0 + 4]);
        *reinterpret_cast<float4*>(cmz) = *reinterpret_cast<const float4*>(&mzv[rk * PP + p0]);
        *reinterpret_cast<float4*>(cmz + 4) = *reinterpret_cast<const float4*>(&mzv[rk * PP + p0 + 4]);
      }
#pragma unroll
      for (int u = tid; u < ROWS * 8; u += THREADS) {
        const int r_ = u >> 3, ci = u & 7, tk = r_ >> 2, p0 = 8 * ci;
        const bool ok = tk < nv;
        U8 vv, dd, zz, o1, o2, o3, o4;
        vv.u = *reinterpret_cast<const uint4*>(&S.v[rb][swz(tk, p0)]);
        dd.u = *reinterpret_cast<const uint4*>(&S.dy[rb][swz(tk, p0)]);
        zz.u = *reinterpret_cast<const uint4*>(&S.z[rb][swz(tk, p0)]);
        const float eq = ok ? __expf(ellv[tk]) : 0.f;
        const float ek = ok ? __expf(totv - ellv[tk]) * ffv[tk] : 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float ps = ok ? __bfloat162float(vv.h[e]) * cmv[e] : 0.f;
          const float zm = __bfloat162float(zz.h[e]) * cmz[e];
          const float dO = ok ? __bfloat162float(dd.h[e]) * cmo[e] * silu_f(zm) : 0.f;
          if (MODE == MODE_LOCAL) {
            o1.h[e] = __float2bfloat16(ps * ek);
            o2.h[e] = __float2bfloat16(dO * eq);
          } else {
            o1.h[e] = __float2bfloat16(ps);
            o2.h[e] = __float2bfloat16(dO);
            o3.h[e] = __float2bfloat16(dO * eq);
            o4.h[e] = __float2bfloat16(ps * ek);
          }
        }
        *reinterpret_cast<uint4*>(&S.pv[swz(r_, p0)]) = o1.u;
        *reinterpret_cast<uint4*>(&S.dob[swz(r_, p0)]) = o2.u;
        if (MODE != MODE_LOCAL) {
          *reinterpret_cast<uint4*>(&S.xq[swz(r_, p0)]) = o3.u;
          *reinterpret_cast<uint4*>(&S.xk[swz(r_, p0)]) = o4.u;
        }
      }
      fence_async();
      tc_fence_before();
      __syncthreads();
      STAMP(3);
      if (tid == 0 && has_next) issue_b(nb_, nh_, rb ^ 1);
      if (MODE == MODE_LOCAL) {
        constexpr uint32_t ID = idesc(128, 64, 1, 1);
        if (tid == 0) {  // Sloc chain (and the TMA stores of the previous item's states)
          if (it > 0) local_store(pb_, ph_);
          tc_fence_after();
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) umma(tmem + TM_O1, mdesc(&S.k[kk * 16 * 64]), mdesc(&S.pv[kk * 16 * 64]), ID, kk > 0);
          umma_commit(&S.m1[0]);
        } else if (tid == 128) {  // dloc chain, issued in parallel
          tc_fence_after();
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) umma(tmem + TM_O2, mdesc(&S.q[kk * 16 * 64]), mdesc(&S.dob[kk * 16 * 64]), ID, kk > 0);
          umma_commit(&S.m1[0]);
        }
        pb_ = b; ph_ = h;  // epilogue deferred to the next item (or the end)
        continue;
      }
      // ---------------------------------------------------------------- the two sides
      auto side = [&](auto side_tag, float (&dxa)[4][JW]) {
        constexpr int SD = decltype(side_tag)::value;
        constexpr bool QS = SD == SIDE_Q;
        const __nv_bfloat16* X = QS ? S.q : S.k;       // rows of this side: qhat (q side) / khat (kv side)
        const __nv_bfloat16* Y = QS ? S.k : S.q;
        const __nv_bfloat16* RA = QS ? S.dob : S.pv;    // dP = dO PsiV^T  |  dPT = PsiV dO^T
        const __nv_bfloat16* RB = QS ? S.pv : S.dob;
        const __nv_bfloat16* ST = QS ? S.sth : S.std_;  // h_b  |  D_b
        const __nv_bfloat16* XE = QS ? S.xq : S.xk;     // dO exp(l)  |  PsiV exp(l_end - l) f
        // ---------------------------------------------------------- first products: G / GT, dP / dPT, O1 / kD (and
        // the state part of dq) -- the kv side's are issued by the q side once its TMEM columns are free
        if (QS) issue_m1(side_tag, 0b1010u);  // dP, dq state part (G, O1 went during the operand build)
        mbar_wait(&S.m1[SD], it & 1);
        tc_fence_after();
        STAMP(4 + 5 * SD);
        // ---------------------------------------------------------- masks: (G or GT) * W -> bf16 in TMEM
        {
          constexpr int GW = 128 / QC;  // columns per thread
          float acc1 = 0.f, acc2 = 0.f;
          const int cb = GW * cq;
          float g[GW], d[GW];
          tmem_ldw<GW>(tl + TM_A + cb, g);
          tmem_ldw<GW>(tl + TM_B + cb, d);
          tmem_wait_ld();
          tc_fence_before();
          __syncthreads();  // every column group read before the bf16 copies overwrite the first 64 columns
          tc_fence_after();
          uint32_t pg[GW / 2], pd[GW / 2];
#pragma unroll
          for (int e = 0; e < GW; e += 2) {
            const int ts = (cb + e) >> 2;  // both columns of the pair belong to the same token
            const float A0 = g[e] * d[e], A1 = g[e + 1] * d[e + 1];
            float w;
            if (QS) {
              w = S.tab1[tok * CP + ts];             // W[t = row][s = col]
              if (ts < tok) acc1 += (A0 + A1) * w;
              else if (ts == tok) acc2 += A0 + A1;
            } else {
              w = S.tab1[ts * CP + tok];             // W[t = col][s = row]
              acc1 += (A0 + A1) * S.tab2[tok * CP + ts];
            }
            pg[e >> 1] = pack_bf16(g[e] * w, g[e + 1] * w);
            pd[e >> 1] = pack_bf16(d[e] * w, d[e + 1] * w);
          }
          tmem_stw<GW / 2>(tl + TM_A + (cb >> 1), pg);
          tmem_stw<GW / 2>(tl + TM_B + (cb >> 1), pd);
          S.red[0][cq][row] = acc1;
          S.red[1][cq][row] = acc2;
          tmem_wait_st();
        }
        tc_fence_before();
        __syncthreads();
        STAMP(5 + 5 * SD);
        // ---------------------------------------------------------- second products
        constexpr uint32_t ID_O = idesc(128, 64, 0, 1), ID_X = idesc(128, 128, 0, 1), ID_I = idesc(128, 128, 0, 0);
        if (tid == 0) {  // O2 chain
          tc_fence_after();
#pragma unroll
          for (int kk = 0; kk < 8; ++kk)  // O2 = (G W) PsiV  |  dPsi2 = (GT W^T) dO
            umma_ts(tmem + TM_O2, tmem + TM_A + 8 * kk, mdesc(&RB[kk * 16 * 64]), ID_O, kk > 0);
          umma_commit(&S.m2a[SD]);
        } else if (tid == 128) {  // dq (dk) chain, issued in parallel by another warp
          tc_fence_after();
          if (!QS) {  // dk: (PsiV exp(l_end - l) f) D^T (the q side's state part of dq went with its first products)
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) umma(tmem + TM_D, kdesc(&XE[kk * 16]), kdesc(&ST[kk * 16]), ID_I, kk > 0);
          }
#pragma unroll
          for (int kk = 0; kk < 8; ++kk)  // dq += (dP W) khat  |  dk += (dPT W^T) qhat
            umma_ts(tmem + TM_D, tmem + TM_B + 8 * kk, mdesc(&Y[kk * 16 * 64]), ID_X, 1);
          umma_commit(&S.m2b[SD]);
        }
        mbar_wait(&S.m2a[SD], it & 1);
        tc_fence_after();
        STAMP(6 + 5 * SD);
        // ---------------------------------------------------------- outputs
        float sc3 = 0.f;
        float g1[OW], g2[OW], dz[OW];
        float pdl = 0.f;
        {  // O1 / O2 columns p in [OW cq, OW cq + OW): q: o -> gate gradients; kv: dPsiV -> dv, dmimo_v, dD
          float o1[OW], o2[OW];
          tmem_ldw<OW>(tl + TM_O1 + OW * cq, o1);
          tmem_ldw<OW>(tl + TM_O2 + OW * cq, o2);
          if (!QS) {  // exp(tot) <h_b, D_b>: 16 elements per thread (same positions in both tiles), summed per warp
            static_assert(N * P == THREADS * 16, "16 state elements per thread");
            U8 h0, h1, d0, d1;
            h0.u = reinterpret_cast<const uint4*>(S.sth)[2 * tid];
            h1.u = reinterpret_cast<const uint4*>(S.sth)[2 * tid + 1];
            d0.u = reinterpret_cast<const uint4*>(S.std_)[2 * tid];
            d1.u = reinterpret_cast<const uint4*>(S.std_)[2 * tid + 1];
            float hd = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e)
              hd += __bfloat162float(h0.h[e]) * __bfloat162float(d0.h[e]) + __bfloat162float(h1.h[e]) * __bfloat162float(d1.h[e]);
#pragma unroll
            for (int m = 1; m <= 16; m <<= 1) hd += __shfl_xor_sync(0xffffffffu, hd, m);
            if (lane == 0) S.xr[warp] = __expf(totv) * hd;
          }
          tmem_wait_ld();
          const float eq = okr ? __expf(ellv[tok]) : 0.f;
          const float ek0 = okr ? __expf(totv - ellv[tok]) : 0.f, ek = ek0 * ffv[tok];
#pragma unroll
          for (int ch = 0; ch < OW / 8; ++ch) {
            const int p0 = OW * cq + 8 * ch;
            U8 pv, dd, ry, rz, rv;
            pv.u = *reinterpret_cast<const uint4*>(&S.pv[swz(row, p0)]);
            dd.u = *reinterpret_cast<const uint4*>(&S.dob[swz(row, p0)]);
            float cmo[8], cmz[8], cmv[8];
            *reinterpret_cast<float4*>(cmo) = *reinterpret_cast<const float4*>(&mov[rank * PP + p0]);
            *reinterpret_cast<float4*>(cmo + 4) = *reinterpret_cast<const float4*>(&mov[rank * PP + p0 + 4]);
            *reinterpret_cast<float4*>(cmz) = *reinterpret_cast<const float4*>(&mzv[rank * PP + p0]);
            *reinterpret_cast<float4*>(cmz + 4) = *reinterpret_cast<const float4*>(&mzv[rank * PP + p0 + 4]);
            *reinterpret_cast<float4*>(cmv) = *reinterpret_cast<const float4*>(&mvv[rank * PP + p0]);
            *reinterpret_cast<float4*>(cmv + 4) = *reinterpret_cast<const float4*>(&mvv[rank * PP + p0 + 4]);
            if (QS) {
              ry.u = *reinterpret_cast<const uint4*>(&S.dy[rb][swz(tok, p0)]);
              rz.u = *reinterpret_cast<const uint4*>(&S.z[rb][swz(tok, p0)]);
            } else {
              rv.u = *reinterpret_cast<const uint4*>(&S.v[rb][swz(tok, p0)]);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              const int i = 8 * ch + e;
              const float ps = __bfloat162float(pv.h[e]), dO = __bfloat162float(dd.h[e]);
              if (QS) {
                const float o = eq * o1[i] + o2[i] + dskv * ps;   // pre-gate output (recomputed)
                sc3 += eq * o1[i] * dO;                          // inter part of the decay row sum
                const float y = okr ? __bfloat162float(ry.h[e]) : 0.f, zv = __bfloat162float(rz.h[e]);
                const float zz = zv * cmz[e], sg = 1.f / (1.f + __expf(-zz));
                const float dzz = y * cmo[e] * o * sg * (1.f + zz * (1.f - sg));
                g1[i] = y * zz * sg * o;   // dmimo_o
                g2[i] = dzz * zv;          // dmimo_z
                dz[i] = dzz * cmz[e];      // dz (summed over the ranks below)
              } else {
                const float dps = okr ? ek * o1[i] + o2[i] + dskv * dO : 0.f;  // dPsiV
                sc3 += ek0 * o1[i] * ps;                                     // kD . PsiV (factor gradient)
                const float vv = okr ? __bfloat162float(rv.h[e]) : 0.f;
                g1[i] = dps * vv;          // dmimo_v
                g2[i] = 0.f;
                dz[i] = dps * cmv[e];      // dv (summed over the ranks below)
                pdl += dO * ps;            // dD
              }
            }
          }
        }
        // per-token sums over the ranks (lanes 4m .. 4m + 3), reduce-scattered: dz / dv columns
        // OW cq + 8 (rank & 1) + 4 (rank >> 1) + [0, 4) on every lane of the token
        static_assert(OW == 16, "reduce-scatter layout");
        rs_step<16, 1>(dz, lane);
        rs_step<8, 2>(dz, lane);
        if (okr) {
          uint2* dst = reinterpret_cast<uint2*>((QS ? a.dz : a.dv) + ((long)(t0 + tok) * H + h) * P + OW * cq
                                                + 8 * (rank & 1) + 4 * (rank >> 1));
          *dst = make_uint2(pack_bf16(dz[0], dz[1]), pack_bf16(dz[2], dz[3]));
        }
        // sums over the tokens of the warp (lanes of equal rank), reduce-scattered: column
        // OW cq + 8 b4 + 4 b8 + 2 b16 + [0, 2) (b_m = lane & m)
        rs_step<16, 4>(g1, lane); rs_step<8, 8>(g1, lane); rs_step<4, 16>(g1, lane);
        if (QS) { rs_step<16, 4>(g2, lane); rs_step<8, 8>(g2, lane); rs_step<4, 16>(g2, lane); }
        const int gcol = OW * cq + 8 * ((lane >> 2) & 1) + 4 * ((lane >> 3) & 1) + 2 * ((lane >> 4) & 1);
        if (!QS) {
#pragma unroll
          for (int m = 1; m <= 16; m <<= 1) pdl += __shfl_xor_sync(0xffffffffu, pdl, m);
        }
        STAMP(7 + 5 * SD);
        // qhat (khat) values of this thread's rotated pairs, for the phase gradient (read before the next head's
        // Q / K rows are requested)
        float xh0[JW], xh1[JW];
        {
          static_assert(JW == 8, "one 16-byte piece");
          U8 u0, u1;
          u0.u = *reinterpret_cast<const uint4*>(&X[swz(row, JW * cq)]);
          u1.u = *reinterpret_cast<const uint4*>(&X[TILE + swz(row, JW * cq)]);
#pragma unroll
          for (int e = 0; e < JW; ++e) { xh0[e] = __bfloat162float(u0.h[e]); xh1[e] = __bfloat162float(u1.h[e]); }
        }
        mbar_wait(&S.m2b[SD], it & 1);
        tc_fence_after();
        tc_fence_before();
        __syncthreads();
        if (QS) {  // the q side is done with G / dP / O1 / O2: the kv side's first products run under its dq epilogue
          issue_m1(std::integral_constant<int, SIDE_KV>{}, 0b111u);
        } else {   // Q / K / state tiles are free after the kv products and the reads above
          if (tid == 0 && has_next) issue_a(nb_, nh_);
        }
        tc_fence_after();
        // the reduction buffers overlay xq, free once the q-side products are done (m2b of the q side, waited above)
        *reinterpret_cast<float2*>(&RD.g1[q4][rank][gcol]) = make_float2(g1[0], g1[1]);
        if (QS) *reinterpret_cast<float2*>(&RD.g2[q4][rank][gcol]) = make_float2(g2[0], g2[1]);
        if (!QS && lane == 0) S.red[1][cq][32 * q4] = pdl;  // dD partial of this warp (red[1] is unused by kv)
        {  // dq (dk): rotated pairs j in [JW cq, JW cq + JW) and the plain columns 32 + JW cq.., 96 + JW cq..
          float d0[JW], d1[JW], p0v[JW], p1v[JW];
          tmem_ldw<JW>(tl + TM_D + JW * cq, d0);
          tmem_ldw<JW>(tl + TM_D + 64 + JW * cq, d1);
          tmem_ldw<JW>(tl + TM_D + 32 + JW * cq, p0v);
          tmem_ldw<JW>(tl + TM_D + 96 + JW * cq, p1v);
          tmem_wait_ld();
          float dth[JW], bx[4][JW];
#pragma unroll
          for (int e = 0; e < JW; ++e) {
            const int j = JW * cq + e;
            const float c = S.cs[tok * NQP + j], s = S.sn[tok * NQP + j];
            const float a0 = d0[e], a1 = d1[e];
            bx[0][e] = okr ? c * a0 + s * a1 : 0.f;
            bx[1][e] = okr ? c * a1 - s * a0 : 0.f;
            bx[2][e] = okr ? p0v[e] : 0.f;
            bx[3][e] = okr ? p1v[e] : 0.f;
            dth[e] = a1 * xh0[e] - a0 * xh1[e];
#pragma unroll
            for (int x = 0; x < 4; ++x) dxa[x][e] += bx[x][e];
          }
          static_assert(JW == 8, "reduce-scatter layout");
          rs_step<8, 1>(dth, lane);  // over the ranks: pairs JW cq + 4 (rank & 1) + 2 (rank >> 1) + [0, 2)
          rs_step<4, 2>(dth, lane);
          if (QS) {  // q part kept in registers, written with the k part
            dthk0 = dth[0]; dthk1 = dth[1];
          } else if (okr) {
            float* dst = a.dth + ((long)(t0 + tok) * H + h) * NQ + JW * cq + 4 * (rank & 1) + 2 * (rank >> 1);
            *reinterpret_cast<float2*>(dst) = make_float2(dth[0] + dthk0, dth[1] + dthk1);
          }
          // bias gradient of this head: sum over the tokens (lanes of equal rank, reduce-scattered), then the warp
          // quarters. Flat index f = 8 x + e -> this lane keeps f = 16 b4 + 8 b8 + 4 b16 + [0, 4): x = f >> 3
          float* bf = &bx[0][0];
          rs_step<32, 4>(bf, lane); rs_step<16, 8>(bf, lane); rs_step<8, 16>(bf, lane);
          {
            const int f0 = 16 * ((lane >> 2) & 1) + 8 * ((lane >> 3) & 1) + 4 * ((lane >> 4) & 1);
            const int x = f0 >> 3, j = JW * cq + (f0 & 7);
            const int col = (x == 0 ? 0 : x == 1 ? 64 : x == 2 ? 32 : 96) + j;
            *reinterpret_cast<float4*>(&RD.bias[q4][rank][col]) = make_float4(bf[0], bf[1], bf[2], bf[3]);
          }
        }
        S.red[2][cq][row] = sc3;
        STAMP(8 + 5 * SD);
        __syncthreads();
        if (cq == 0) {  // per-token scalars (sum over the column groups, then over the ranks)
          float v1 = 0.f, v2 = 0.f;
#pragma unroll
          for (int g = 0; g < QC; ++g) {
            v1 += S.red[0][g][row] + S.red[2][g][row];  // q: dl row part (intra + inter); kv: dfactor (intra + inter)
            v2 += S.red[1][g][row];
          }
#pragma unroll
          for (int m = 1; m <= 2; m <<= 1) {
            v1 += __shfl_xor_sync(0xffffffffu, v1, m);
            v2 += __shfl_xor_sync(0xffffffffu, v2, m);
          }
          if (okr && rank == 0) {
            const long o = (long)h * a.S + t0 + tok;
            if (QS) { a.dlrow[o] = v1; a.dgam[o] = v2; }
            else a.dfac[o] = v1;
          }
          if (!QS) {  // <Sloc_b, D_b> = sum_s f_s exp(l_end - l_s) kD_s . PsiV_s (the inter part of dfactor, times f)
            float xc = 0.f;
#pragma unroll
            for (int g = 0; g < QC; ++g) xc += S.red[2][g][row];
            xc = okr ? xc * ffv[tok] : 0.f;
#pragma unroll
            for (int m = 1; m <= 16; m <<= 1) xc += __shfl_xor_sync(0xffffffffu, xc, m);
            if (lane == 0) S.xr[THREADS / 32 + q4] = xc;
          }
        }
        {  // per (CTA, head) accumulators: fixed summation order, one owning thread per address (deterministic)
          float* acc = (QS ? a.accq : a.acck) + ((long)blockIdx.x * H + h) * NACC;
          for (int j = tid; j < R * N; j += THREADS) {
            const int r_ = j >> 7, n = j & 127;
            red_add(acc + j, ((RD.bias[0][r_][n] + RD.bias[1][r_][n]) + RD.bias[2][r_][n]) + RD.bias[3][r_][n]);
          }
          for (int j = tid; j < R * P; j += THREADS) {
            const int r_ = j >> 6, p = j & 63;
            red_add(acc + R * N + j, ((RD.g1[0][r_][p] + RD.g1[1][r_][p]) + RD.g1[2][r_][p]) + RD.g1[3][r_][p]);
            if (QS) red_add(acc + R * N + R * P + j, ((RD.g2[0][r_][p] + RD.g2[1][r_][p]) + RD.g2[2][r_][p]) + RD.g2[3][r_][p]);
          }
          if (!QS && tid == 0) {
            float s = 0.f;
            for (int g = 0; g < QC; ++g)
              for (int w = 0; w < 4; ++w) s += S.red[1][g][32 * w];
            red_add(acc + R * N + 2 * R * P, s);
          }
        }
        tc_fence_before();
        __syncthreads();
        if (!QS && tid == 0) {  // fixed order (deterministic)
          float c = 0.f;
#pragma unroll
          for (int i = 0; i < THREADS / 32 + 4; ++i) c += S.xr[i];
          a.cross[(long)b * H + h] = c;
        }
      };
      side(std::integral_constant<int, SIDE_Q>{}, dxq);
      side(std::integral_constant<int, SIDE_KV>{}, dxk);
    }
    if (MODE != MODE_LOCAL && okr) {  // dq, dk summed over the heads
#pragma unroll
      for (int sd = 0; sd < 2; ++sd) {
        __nv_bfloat16* dst = (sd == 0 ? a.dq : a.dk) + grow * N + JW * cq;
        float (&dxa)[4][JW] = sd == 0 ? dxq : dxk;
#pragma unroll
        for (int x = 0; x < JW / 8; ++x) {
          uint4 u[4];
#pragma unroll
          for (int g = 0; g < 4; ++g)
            u[g] = make_uint4(pack_bf16(dxa[g][8 * x], dxa[g][8 * x + 1]), pack_bf16(dxa[g][8 * x + 2], dxa[g][8 * x + 3]),
                              pack_bf16(dxa[g][8 * x + 4], dxa[g][8 * x + 5]), pack_bf16(dxa[g][8 * x + 6], dxa[g][8 * x + 7]));
          *reinterpret_cast<uint4*>(dst + 8 * x) = u[0];
          *reinterpret_cast<uint4*>(dst + 64 + 8 * x) = u[1];
          *reinterpret_cast<uint4*>(dst + 32 + 8 * x) = u[2];
          *reinterpret_cast<uint4*>(dst + 96 + 8 * x) = u[3];
        }
      }
    }
  }
  if (MODE == MODE_LOCAL && it > 0) {
    if (tid == 0) bulk_wait_read();
    __syncthreads();
    mbar_wait(&S.m1[0], (it - 1) & 1);
    tc_fence_after();
    local_epi();
    fence_async();
    __syncthreads();
    if (tid == 0) { local_store(pb_, ph_); bulk_wait(); }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tmem), "n"(512));
  }
}
}  // namespace

// ---------------------------------------------------------------- host
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn get_encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qres;
    void* p = nullptr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qres);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<EncodeTiledFn>(p);
  }
  return fn;
}
static CUtensorMap make_map(const void* base, CUtensorMapDataType dt, int rank, const cuuint64_t* dims, const cuuint64_t* strides,
                            const cuuint32_t* box, CUtensorMapSwizzle sw) {
  CUtensorMap m;
  const cuuint32_t estr[3] = {1, 1, 1};
  const CUresult r = get_encode_fn()(&m, dt, rank, const_cast<void*>(base), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
static CUtensorMap state_map(const torch::Tensor& st) {
  TORCH_CHECK(st.is_contiguous() && st.scalar_type() == torch::kBFloat16 && st.size(-1) == P && st.size(-2) == N);
  const cuuint64_t d[2] = {(cuuint64_t)P, (cuuint64_t)(st.numel() / P)}, s[1] = {(cuuint64_t)P * 2};
  const cuuint32_t box[2] = {64, ROWS};
  return make_map(st.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d, s, box, CU_TENSOR_MAP_SWIZZLE_128B);
}

static Maps build_maps(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor DY, torch::Tensor Z,
                       torch::Tensor ang, const torch::Tensor* hs, const torch::Tensor* ds) {
  const int T = V.size(0), H = V.size(1);
  TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && Q.size(-1) == N && Q.numel() == (long)T * R * N);
  for (auto* x : {&V, &DY, &Z})
    TORCH_CHECK(x->size(0) == T && x->size(1) == H && x->size(2) == P && x->stride(2) == 1 && x->stride(1) == P &&
                x->stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(x->data_ptr()) & 15) == 0);
  TORCH_CHECK(ang.is_contiguous() && ang.size(-1) == NQ && ang.numel() == (long)T * H * NQ);
  Maps M;
  {
    const cuuint64_t d[2] = {(cuuint64_t)N, (cuuint64_t)T * R}, s[1] = {(cuuint64_t)N * 2};
    const cuuint32_t box[2] = {64, ROWS};
    M.q = make_map(Q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d, s, box, CU_TENSOR_MAP_SWIZZLE_128B);
    M.k = make_map(K.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d, s, box, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  {
    const cuuint32_t box[3] = {P, 1, C};
    const cuuint64_t d[3] = {(cuuint64_t)P, (cuuint64_t)H, (cuuint64_t)T};
    const cuuint64_t sv[2] = {(cuuint64_t)P * 2, (cuuint64_t)V.stride(0) * 2};
    const cuuint64_t sd[2] = {(cuuint64_t)P * 2, (cuuint64_t)DY.stride(0) * 2};
    const cuuint64_t sz[2] = {(cuuint64_t)P * 2, (cuuint64_t)Z.stride(0) * 2};
    M.v = make_map(V.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d, sv, box, CU_TENSOR_MAP_SWIZZLE_128B);
    M.dy = make_map(DY.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d, sd, box, CU_TENSOR_MAP_SWIZZLE_128B);
    M.z = make_map(Z.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, d, sz, box, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  {
    const cuuint64_t d[3] = {(cuuint64_t)NQ, (cuuint64_t)H, (cuuint64_t)T}, s[2] = {(cuuint64_t)NQ * 4, (cuuint64_t)H * NQ * 4};
    const cuuint32_t box[3] = {NQ, 1, C};
    M.ang = make_map(ang.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, d, s, box, CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  M.sth = hs ? state_map(*hs) : M.q;
  M.std_ = ds ? state_map(*ds) : M.q;
  return M;
}

template <int MODE>
static void launch(const Maps& M, const Args& a, int dev) {
  const int smem = (int)sizeof(Smem) + 1024;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(bwd_tc_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                "bwd_tc: shared memory request too large: ", smem);
    attr = true;
  }
  static int nsm = 0;
  if (!nsm) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  if (a.NB == 0) return;
  const int grid = std::min(a.NB, nsm);
  bwd_tc_kernel<MODE><<<grid, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(M, a);
}

static unsigned long long* g_prof = nullptr;
void set_prof(torch::Tensor t) { g_prof = t.numel() ? reinterpret_cast<unsigned long long*>(t.data_ptr()) : nullptr; }

static Args base_args(torch::Tensor V, torch::Tensor qb, torch::Tensor kb, torch::Tensor mv, torch::Tensor mo, torch::Tensor mz,
                      torch::Tensor Dskip, torch::Tensor ell, torch::Tensor ff, torch::Tensor gam, torch::Tensor tot,
                      torch::Tensor sb_start, torch::Tensor sb_len, torch::Tensor ain, torch::Tensor sb_chunk, int64_t NB) {
  TORCH_CHECK(ain.is_contiguous() && ain.scalar_type() == torch::kFloat && sb_chunk.scalar_type() == torch::kInt);
  TORCH_CHECK(qb.scalar_type() == torch::kBFloat16 && kb.scalar_type() == torch::kBFloat16 && qb.is_contiguous() && kb.is_contiguous());
  Args a{};
  a.qbias = reinterpret_cast<const __nv_bfloat16*>(qb.data_ptr()); a.kbias = reinterpret_cast<const __nv_bfloat16*>(kb.data_ptr());
  a.mv = mv.data_ptr<float>(); a.mo = mo.data_ptr<float>();
  a.mz = mz.data_ptr<float>(); a.Dskip = Dskip.data_ptr<float>(); a.ell = ell.data_ptr<float>(); a.ff = ff.data_ptr<float>();
  a.gam = gam.data_ptr<float>(); a.tot = tot.data_ptr<float>(); a.sb_start = sb_start.data_ptr<int>();
  a.sb_len = sb_len.data_ptr<int>(); a.S = V.size(0); a.H = V.size(1); a.NB = (int)NB; a.prof = g_prof;
  a.ain = ain.data_ptr<float>(); a.sb_chunk = sb_chunk.data_ptr<int>();
  return a;
}

void bwd_local(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor DY, torch::Tensor Z, torch::Tensor ang,
               torch::Tensor qb, torch::Tensor kb, torch::Tensor mv, torch::Tensor mo, torch::Tensor mz, torch::Tensor Dskip,
               torch::Tensor ell, torch::Tensor ff, torch::Tensor gam, torch::Tensor tot, torch::Tensor sb_start,
               torch::Tensor sb_len, torch::Tensor ain, torch::Tensor sb_chunk, torch::Tensor sloc, torch::Tensor dloc,
               int64_t NB) {
  Args a = base_args(V, qb, kb, mv, mo, mz, Dskip, ell, ff, gam, tot, sb_start, sb_len, ain, sb_chunk, NB);
  a.sloc = reinterpret_cast<__nv_bfloat16*>(sloc.data_ptr()); a.dloc = reinterpret_cast<__nv_bfloat16*>(dloc.data_ptr());
  launch<MODE_LOCAL>(build_maps(Q, K, V, DY, Z, ang, &sloc, &dloc), a, V.get_device());  // state maps: TMA stores
}

void bwd_main(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor DY, torch::Tensor Z, torch::Tensor ang,
              torch::Tensor qb, torch::Tensor kb, torch::Tensor mv, torch::Tensor mo, torch::Tensor mz, torch::Tensor Dskip,
              torch::Tensor ell, torch::Tensor ff, torch::Tensor gam, torch::Tensor tot, torch::Tensor sb_start,
              torch::Tensor sb_len, torch::Tensor ain, torch::Tensor sb_chunk, torch::Tensor hs, torch::Tensor ds, torch::Tensor dq, torch::Tensor dk, torch::Tensor dz,
              torch::Tensor dv, torch::Tensor dth, torch::Tensor dlrow, torch::Tensor dgam,
              torch::Tensor dfac, torch::Tensor accq, torch::Tensor acck, torch::Tensor cross, int64_t NB) {
  Args a = base_args(V, qb, kb, mv, mo, mz, Dskip, ell, ff, gam, tot, sb_start, sb_len, ain, sb_chunk, NB);
  a.dq = reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()); a.dk = reinterpret_cast<__nv_bfloat16*>(dk.data_ptr());
  a.dz = reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()); a.dv = reinterpret_cast<__nv_bfloat16*>(dv.data_ptr());
  a.dth = dth.data_ptr<float>();
  a.dlrow = dlrow.data_ptr<float>(); a.dgam = dgam.data_ptr<float>(); a.dfac = dfac.data_ptr<float>();
  a.accq = accq.data_ptr<float>(); a.acck = acck.data_ptr<float>(); a.cross = cross.data_ptr<float>();
  launch<MODE_FUSED>(build_maps(Q, K, V, DY, Z, ang, &hs, &ds), a, V.get_device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("local", &bwd_local);
  m.def("main", &bwd_main);
  m.def("set_prof", &set_prof);
  m.def("nacc", []() { return NACC; });
  m.def("smem", []() { return (int64_t)sizeof(Smem) + 1024; });
  m.def("grid", [](int64_t NB) { int nsm = 0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0); return std::min<int64_t>(NB, nsm); });
}
