// Mamba-3 MIMO varlen forward, hand-written CUDA (sm_90, wgmma) for the Olala configuration: R=4, N=128, P=64,
// chunk 16, one Q/K group, reduceO + Z gate + D skip (no fused RMSNorm), 32 rotary angles, bf16 activations.
// Drop-in for the two passes of the TileLang ``mamba_mimo_fwd`` kernel (see fwd.py):
//
//   scan        (full pass, two-level Pass C / stock): outputs of every block (a run of 16-token sub-blocks of
//               one sequence) from the state entering it (INIT_STATE, or zero). Per sub-block (64 MIMO rows,
//               row = token * R + rank):
//                 o   = exp(da_cs) * (qhat @ S)
//                 G   = qhat @ (khat * ts)^T, masked: earlier token -> exp(da_cs_i - da_cs_j), same token -> gamma/ts
//                 o  += G @ PsiV + D * PsiV ;  out[t] = sum_r o[t, r] * phi_r * silu(z_t * zeta_r)
//                 S   = exp(da_cs_last) * S + (khat * ts * exp(da_cs_rev))^T @ PsiV
//   chunk_state (state-only pass, two-level Pass A): the state each block produces from zero,
//                 C[n][p] = sum_j khat_j^T (PsiV_j * ts_j * exp(dtot - dacs_j)), dacs = block-level log-decay cumsum.
//
// Same math and bf16 operand roundings as the TileLang kernel except where noted (G and the state update share
// the khat * ts * exp(da_cs_rev) rows; Pass A weights PsiV instead of khat). Warp-specialized and persistent over
// (block, head) items: two producer warpgroups stage the raw sub-block with cp.async and build the bf16 operands
// (bias, rotation, trap scale, PsiV, gate) into a two-deep smem ring; consumer warpgroups run the GEMMs with wgmma
// (hand-built 128B-swizzle descriptors), the fp32 state in registers. Adapted from the vLLM Olala prefill kernels
// (jgcb00/vllm, olala/csrc/mamba3_scan_prefill.cu).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int R = 4, N = 128, P = 64, NQ = 32, C = 16, ROWS = C * R;  // 64 rows per sub-block
constexpr int THREADS = 512;  // warpgroups 0, 1: consumers (one P half each); 2, 3: producers (32 rows each)
constexpr int TILE = 64 * 64;  // one 128B-swizzled 64 x 64 bf16 tile
// padded row strides of the producer's per-token / per-rank tables (rows land on different banks)
// k rows carry e_j = exp(max(da_cs_rev_j, -LOG_EMIN)) so that the state update needs no second PsiV; the G mask
// divides it back out (the clamp keeps 1 / e_j finite; it only changes contributions already below e^-80)
constexpr float LOG_EMIN = 80.f;
constexpr int LTH = NQ + 4, LBN = N + 8, LPB = P + 8, LPF = P + 4;
enum { BAR_FULL = 1, BAR_EMPTY = 3, BAR_CONS = 5, BAR_PROD = 7, BAR_G = 8 };

struct __align__(1024) Stage {
  __nv_bfloat16 q[2 * TILE];  // qhat [row][n], two n-halves (A of qhat @ S and of G, K-major)
  __nv_bfloat16 k[2 * TILE];  // khat * ts * e_j [row][n] (B of G, K-major; A of the state update, MN-major)
  __nv_bfloat16 pv[TILE];     // PsiV [row][p] (B, MN-major)
  __nv_bfloat16 gz[TILE];     // phi_r * silu(z * zeta_r)
  float coef[C * C];          // intra-block mask per (token i, token j), divided by e_j
  float erow[C];              // exp(da_cs)
  float dec;                  // exp(da_cs) of the last token
};
struct __align__(128) Raw {   // cp.async staging of one sub-block (rows past the chunk end are zero)
  __nv_bfloat16 q[ROWS * N], k[ROWS * N];
  __nv_bfloat16 v[C * P], z[C * P];
  float th[C * NQ];
  float dcs[C], dcsr[C], ts[C], gam[C];
};
struct Smem {
  Stage stage[2];
  __nv_bfloat16 S[2 * TILE];  // state, bf16 [n][p] (B of qhat @ S, MN-major)
  __nv_bfloat16 G[2][TILE];   // masked qhat @ k^T, bf16 [row][j] (A of G @ PsiV, K-major); each consumer fills
                              // its 32 columns j
  Raw raw[2];
  float tc[C * LTH], tsn[C * LTH];
  __nv_bfloat16 os[2][4][4 * 32];
  __nv_bfloat16 qb[R * LBN], kb[R * LBN], psi[R * LPB];
  float phi[R * LPF], zeta[R * LPF];
  float a0[NQ];
};

__device__ __forceinline__ int swz(int row, int col) {  // 128B swizzle, 64-element rows
  return row * 64 + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7);
}
__device__ __forceinline__ int sw(int row, int col, int ld) {
  return row * ld + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7);
}
__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint64_t gdesc(const void* p) {  // SW128 operand descriptor, 1024 B between 8-row groups
  const uint64_t a = smem_u32(p);
  return ((a & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
#define WG_ACC32(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), \
  "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
  "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), \
  "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
#define WG_D32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
template <int TA, int TB, int ACC = 1>
__device__ __forceinline__ void wgmma_ss(float* d, uint64_t da, uint64_t db) {  // d (+)= A @ B, m64n64k16
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %36, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " WG_D32 ", %32, %33, p, 1, 1, %34, %35;\n}\n"
               : WG_ACC32(d) : "l"(da), "l"(db), "n"(TA), "n"(TB), "n"(ACC));
}
#define WG_ACC16(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), \
  "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
#define WG_D16 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}"
template <int TA, int TB, int ACC = 1>
__device__ __forceinline__ void wgmma32_ss(float* d, uint64_t da, uint64_t db) {  // d (+)= A @ B, m64n32k16
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %20, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 " WG_D16 ", %16, %17, p, 1, 1, %18, %19;\n}\n"
               : WG_ACC16(d) : "l"(da), "l"(db), "n"(TA), "n"(TB), "n"(ACC));
}
template <int TB>
__device__ __forceinline__ void wgmma32_rs(float* d, const uint32_t* a, uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 " WG_D16 ", {%16,%17,%18,%19}, %20, p, 1, 1, %21;\n}\n"
               : WG_ACC16(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(TB));
}
template <int TB>
__device__ __forceinline__ void wgmma_rs(float* d, const uint32_t* a, uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " WG_D32 ", {%32,%33,%34,%35}, %36, p, 1, 1, %37;\n}\n"
               : WG_ACC32(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(TB));
}
template <int n>
__device__ __forceinline__ void reg_fence(float* r) {  // keeps the compiler from moving accumulator writes across wgmma
#pragma unroll
  for (int i = 0; i < n; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_u32(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp4(void* dst, const void* src, bool ok) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(smem_u32(dst)), "l"(src), "r"(ok ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
union U8 { uint4 u; __nv_bfloat16 h[8]; __nv_bfloat162 h2[4]; };

// Raw inputs of sub-block [t0, t0 + nv) into a staging buffer; pt = producer thread index (0..255).
__device__ __forceinline__ void issue_raw(Raw& rw, int pt, int t0, int nv, int h, int T, int H,
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, const __nv_bfloat16* Z,
    const float* thl, const float* dcs16, const float* dcsr16, const float* ts_g, const float* gam_g) {
  const long qbase = (long)t0 * R * N;
#pragma unroll
  for (int i = 0; i < ROWS * N / 8 / 256; ++i) {
    const int idx = pt + 256 * i, row = idx >> 4, ch = idx & 15;
    const bool ok = (row >> 2) < nv;
    const long off = ok ? qbase + (long)row * N + 8 * ch : 0;
    cp16(&rw.q[sw(row, 8 * ch, N)], Q + off, ok);
    cp16(&rw.k[sw(row, 8 * ch, N)], K + off, ok);
  }
  if (pt < C * 8) {
    const int tok = pt >> 3, ch = pt & 7;
    const bool ok = tok < nv;
    const long off = ok ? ((long)(t0 + tok) * H + h) * P + 8 * ch : 0;
    cp16(&rw.v[sw(tok, 8 * ch, P)], V + off, ok);
    cp16(&rw.z[sw(tok, 8 * ch, P)], Z + off, ok);
    const long toff = ok ? ((long)(t0 + tok) * H + h) * NQ + 4 * ch : 0;
    cp16(&rw.th[tok * NQ + 4 * ch], thl + toff, ok);
  }
  if (pt < 4 * C) {
    const int a = pt >> 4, j = pt & 15;
    const bool ok = j < nv;
    const long off = ok ? (long)h * T + t0 + j : 0;
    const float* src = a == 0 ? dcs16 : a == 1 ? dcsr16 : a == 2 ? ts_g : gam_g;
    float* dst = a == 0 ? rw.dcs : a == 1 ? rw.dcsr : a == 2 ? rw.ts : rw.gam;
    cp4(dst + j, src + off, ok);
  }
  cp_commit();
}

// Per-head producer constants (biases, Psi, Phi, Zeta, the chunk's phase offset): 8 values per producer thread,
// fetched into registers ahead of time and written to smem when the previous work item is done with them.
constexpr int NCONST = 2 * R * N + 3 * R * P + NQ;
__device__ __forceinline__ void load_consts(float* v, int pt, int c, int h, int H, const float* qbias, const float* kbias,
                                            const float* phi, const float* zeta, const float* psi, const float* ain) {
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int j = pt + 256 * k;
    float x = 0.f;
    if (j < R * N) x = qbias[h * R * N + j];
    else if (j < 2 * R * N) x = kbias[h * R * N + j - R * N];
    else if (j < 2 * R * N + R * P) x = phi[h * R * P + j - 2 * R * N];
    else if (j < 2 * R * N + 2 * R * P) x = zeta[h * R * P + j - 2 * R * N - R * P];
    else if (j < 2 * R * N + 3 * R * P) x = psi[h * R * P + j - 2 * R * N - 2 * R * P];
    else if (j < NCONST) x = ain ? ain[(c * H + h) * NQ + j - 2 * R * N - 3 * R * P] : 0.f;
    v[k] = x;
  }
}
template <typename Sm>
__device__ __forceinline__ void store_consts(Sm& S, const float* v, int pt);

__global__ void __launch_bounds__(THREADS, 1) mamba3_scan_prefill_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,  // (T*R, N)
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ Z,  // (T, H, P)
    const float* __restrict__ qbias, const float* __restrict__ kbias,          // (H, R, N)
    const float* __restrict__ psi, const float* __restrict__ phi, const float* __restrict__ zeta,  // (H, R, P)
    const float* __restrict__ Dskip,                                           // (H,)
    const float* __restrict__ thl, const float* __restrict__ ain,              // (T, H, NQ) phases, (NC, H, NQ) offset or null
    const float* __restrict__ dcs16, const float* __restrict__ dcsr16,         // (H, T)
    const float* __restrict__ ts_g, const float* __restrict__ gam_g,           // (H, T)
    const int* __restrict__ ch_start, const int* __restrict__ ch_len, const int* __restrict__ ch_last,
    const float* __restrict__ sin_t,                                           // (NC, H, N, P) or null
    __nv_bfloat16* __restrict__ out,                                           // (T, H, P)
    __nv_bfloat16* __restrict__ fin_k,                                         // (NC, R, H, N), chunks ending a sequence
    int T, int H, int NC) {
  extern __shared__ unsigned char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023));
  const int tid = threadIdx.x, nitems = NC * H;  // persistent: work item = chunk * H + head, strided by gridDim.x
  int total = 0;                                  // sub-blocks this CTA processes
  for (int item = blockIdx.x; item < nitems; item += gridDim.x) total += (ch_len[item / H] + C - 1) / C;

  if (tid >= 256) {
    // ================= producer =================
    const int pt = tid - 256;
    const int wq = (pt >> 5) & 3, row = 32 * (pt >> 7) + (pt & 31), tok = row >> 2, rr = row & 3;  // wq: column quarter
    int item = blockIdx.x;
    if (item >= nitems) return;
    int c = item / H, h = item % H;
    issue_raw(S.raw[0], pt, ch_start[c], min(C, ch_len[c]), h, T, H, Q, K, V, Z, thl, dcs16, dcsr16, ts_g, gam_g);
    float pre[8];
    load_consts(pre, pt, c, h, H, qbias, kbias, phi, zeta, psi, ain);
    store_consts(S, pre, pt);
    int gi = 0;
    for (; item < nitems; item += gridDim.x) {
     c = item / H;
     h = item % H;
     const int cs = ch_start[c], cl = ch_len[c], last_seq = ch_last ? ch_last[c] : -1, nsb = (cl + C - 1) / C;
     const int nitem = item + gridDim.x, nc = nitem < nitems ? nitem / H : 0, nh = nitem % H;
     for (int it = 0; it < nsb; ++it, ++gi) {
      const int s = gi & 1, t0 = cs + it * C, nv = min(C, cl - it * C);
      Raw& rw = S.raw[gi & 1];
      Stage& sg = S.stage[s];
      const bool item_end = it + 1 == nsb, more = !item_end || nitem < nitems;
      if (more) {
        if (!item_end)
          issue_raw(S.raw[(gi + 1) & 1], pt, t0 + C, min(C, cl - (it + 1) * C), h, T, H, Q, K, V, Z, thl, dcs16, dcsr16,
                    ts_g, gam_g);
        else
          issue_raw(S.raw[(gi + 1) & 1], pt, ch_start[nc], min(C, ch_len[nc]), nh, T, H, Q, K, V, Z, thl, dcs16, dcsr16,
                    ts_g, gam_g);
        cp_wait<1>();
      } else {
        cp_wait<0>();
      }
      if (item_end && nitem < nitems) load_consts(pre, pt, nc, nh, H, qbias, kbias, phi, zeta, psi, ain);
      bar_sync(BAR_PROD, 256);
      if (gi >= 2) bar_sync(BAR_EMPTY + s, THREADS);  // the consumer is done with this stage
      // ---- rotary table, mask, row scales
#pragma unroll
      for (int i = 0; i < C * NQ / 256; ++i) {
        const int idx = pt + 256 * i;
        float th = rw.th[idx] + S.a0[idx & (NQ - 1)];
        th = th - 6.283185307179586f * floorf(th * 0.15915494309189535f);
        const int ti = (idx / NQ) * LTH + (idx & (NQ - 1));
        __sincosf(th, &S.tsn[ti], &S.tc[ti]);
      }
#pragma unroll
      for (int i = 0; i < C * C / 256; ++i) {
        const int idx = pt + 256 * i, ti = idx >> 4, tj = idx & 15;
        const float di = rw.dcs[ti], dj = rw.dcs[tj];
        const float tsi = ti < nv ? rbf(rw.ts[ti]) : 0.f;  // the trap scale is held in bf16
        const float dg = (ti < nv && tsi > 0.f) ? rw.gam[ti] / tsi : 0.f;
        const float lej = fmaxf(rw.dcsr[tj], -LOG_EMIN);  // log e_j
        sg.coef[idx] = ti > tj ? __expf(fminf(di - dj, 0.f) - lej) : (ti == tj ? dg * __expf(-lej) : 0.f);
      }
      if (pt < C) {
        sg.erow[pt] = __expf(rw.dcs[pt]);
        if (pt == nv - 1) sg.dec = __expf(rw.dcs[pt]);
      }
      bar_sync(BAR_PROD, 256);
      // ---- qhat, khat * ts (bias, rotation of the (n, n + 64) pairs for n < NQ)
      {
        // khat * (ts * e_j) in one bf16 product (ts itself is held in bf16)
        const __nv_bfloat162 tse2 =
            __float2bfloat162_rn(tok < nv ? rbf(rw.ts[tok]) * __expf(fmaxf(rw.dcsr[tok], -LOG_EMIN)) : 0.f);
        const bool last_tok = (last_seq >= 0) && (it == nsb - 1) && (tok == nv - 1);
        __nv_bfloat16* fk = fin_k + ((long)(c * R + rr) * H + h) * N;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int c0 = 16 * wq + 8 * i, c1 = 64 + c0;  // rotated for wq < 2
          U8 q0, q1, k0, k1, bq0, bq1, bk0, bk1;
          q0.u = *reinterpret_cast<const uint4*>(&rw.q[sw(row, c0, N)]);
          q1.u = *reinterpret_cast<const uint4*>(&rw.q[sw(row, c1, N)]);
          k0.u = *reinterpret_cast<const uint4*>(&rw.k[sw(row, c0, N)]);
          k1.u = *reinterpret_cast<const uint4*>(&rw.k[sw(row, c1, N)]);
          bq0.u = *reinterpret_cast<const uint4*>(&S.qb[rr * LBN + c0]);
          bq1.u = *reinterpret_cast<const uint4*>(&S.qb[rr * LBN + c1]);
          bk0.u = *reinterpret_cast<const uint4*>(&S.kb[rr * LBN + c0]);
          bk1.u = *reinterpret_cast<const uint4*>(&S.kb[rr * LBN + c1]);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            q0.h2[j] = __hadd2(q0.h2[j], bq0.h2[j]);
            q1.h2[j] = __hadd2(q1.h2[j], bq1.h2[j]);
            k0.h2[j] = __hadd2(k0.h2[j], bk0.h2[j]);
            k1.h2[j] = __hadd2(k1.h2[j], bk1.h2[j]);
          }
          if (wq < 2) {
            const float4* cp = reinterpret_cast<const float4*>(&S.tc[tok * LTH + c0]);
            const float4* sp = reinterpret_cast<const float4*>(&S.tsn[tok * LTH + c0]);
            const float4 ca = cp[0], cb = cp[1], sa = sp[0], sb = sp[1];
            const float cn[8] = {ca.x, ca.y, ca.z, ca.w, cb.x, cb.y, cb.z, cb.w};
            const float sn[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float2 x0 = __bfloat1622float2(q0.h2[j]), x1 = __bfloat1622float2(q1.h2[j]);
              const float2 y0 = __bfloat1622float2(k0.h2[j]), y1 = __bfloat1622float2(k1.h2[j]);
              const float c_a = cn[2 * j], c_b = cn[2 * j + 1], s_a = sn[2 * j], s_b = sn[2 * j + 1];
              q0.h2[j] = __floats2bfloat162_rn(c_a * x0.x - s_a * x1.x, c_b * x0.y - s_b * x1.y);
              q1.h2[j] = __floats2bfloat162_rn(s_a * x0.x + c_a * x1.x, s_b * x0.y + c_b * x1.y);
              k0.h2[j] = __floats2bfloat162_rn(c_a * y0.x - s_a * y1.x, c_b * y0.y - s_b * y1.y);
              k1.h2[j] = __floats2bfloat162_rn(s_a * y0.x + c_a * y1.x, s_b * y0.y + c_b * y1.y);
            }
          }
          if (last_tok) {
            *reinterpret_cast<uint4*>(fk + c0) = k0.u;
            *reinterpret_cast<uint4*>(fk + c1) = k1.u;
          }
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            k0.h2[j] = __hmul2(k0.h2[j], tse2);
            k1.h2[j] = __hmul2(k1.h2[j], tse2);
          }
          *reinterpret_cast<uint4*>(&sg.q[swz(row, c0)]) = q0.u;          // c0 < 64: first tile
          *reinterpret_cast<uint4*>(&sg.q[TILE + swz(row, c0)]) = q1.u;   // c1 = 64 + c0: second tile
          *reinterpret_cast<uint4*>(&sg.k[swz(row, c0)]) = k0.u;
          *reinterpret_cast<uint4*>(&sg.k[TILE + swz(row, c0)]) = k1.u;
        }
      }
      // ---- PsiV, phi * silu(z * zeta)
      {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int col = 16 * wq + 8 * i;
          U8 va, za, ps, z8;
          va.u = *reinterpret_cast<const uint4*>(&rw.v[sw(tok, col, P)]);
          za.u = *reinterpret_cast<const uint4*>(&rw.z[sw(tok, col, P)]);
          ps.u = *reinterpret_cast<const uint4*>(&S.psi[rr * LPB + col]);
          const float4* zp = reinterpret_cast<const float4*>(&S.zeta[rr * LPF + col]);
          const float4* fp = reinterpret_cast<const float4*>(&S.phi[rr * LPF + col]);
          const float4 za4 = zp[0], zb4 = zp[1], fa4 = fp[0], fb4 = fp[1];
          const float zt[8] = {za4.x, za4.y, za4.z, za4.w, zb4.x, zb4.y, zb4.z, zb4.w};
          const float ft[8] = {fa4.x, fa4.y, fa4.z, fa4.w, fb4.x, fb4.y, fb4.z, fb4.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            va.h2[j] = __hmul2(va.h2[j], ps.h2[j]);
            const float2 z = __bfloat1622float2(za.h2[j]);
            const float g0 = 0.5f * z.x * zt[2 * j], g1 = 0.5f * z.y * zt[2 * j + 1];
            const float s0 = rbf(g0 * tanh_approx(g0) + g0), s1 = rbf(g1 * tanh_approx(g1) + g1);  // silu in bf16
            z8.h2[j] = __floats2bfloat162_rn(ft[2 * j] * s0, ft[2 * j + 1] * s1);
          }
          *reinterpret_cast<uint4*>(&sg.pv[swz(row, col)]) = va.u;
          *reinterpret_cast<uint4*>(&sg.gz[swz(row, col)]) = z8.u;
        }
      }
      fence_async();
      bar_sync(BAR_PROD, 256);  // done with this raw buffer, the tables and the head's constants
      if (item_end && nitem < nitems) store_consts(S, pre, pt);
      bar_arrive(BAR_FULL + s, THREADS);
     }
    }
    return;
  }

  // ================= consumer (P columns pc .. pc + 31) =================
  const int cw = tid >> 7, pc = 32 * cw, lane = tid & 31, w = (tid >> 5) & 3, g_ = lane >> 2, cq = lane & 3;
  const int ti_lo = 4 * w + (g_ >> 2), ti_hi = ti_lo + 2;  // tokens of rows 16 w + g_ and + 8
  int gi = 0;
  for (int item = blockIdx.x; item < nitems; item += gridDim.x) {
  const int c = item / H, h = item % H;
  const int cs = ch_start[c], cl = ch_len[c], nsb = (cl + C - 1) / C;
  const float dsk = Dskip[h];
  float st[2][16];  // state rows n = 64 mh + 16 w + g_ (+8), cols p = pc + 8 nt + 2 cq (+1)
  {
    const float* sb = sin_t ? sin_t + (long)(c * H + h) * N * P : nullptr;  // [N][P] fp32, or a zero state
#pragma unroll
    for (int mh = 0; mh < 2; ++mh)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          st[mh][4 * nt + e] = sb ? sb[(64 * mh + 16 * w + g_ + (e >> 1) * 8) * P + pc + 8 * nt + 2 * cq + (e & 1)] : 0.f;
    bar_sync(BAR_CONS + cw, 128);  // the previous item's last qhat @ S is complete everywhere
#pragma unroll
    for (int mh = 0; mh < 2; ++mh)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int eh = 0; eh < 2; ++eh)
          *reinterpret_cast<uint32_t*>(&S.S[swz(64 * mh + 16 * w + g_ + 8 * eh, pc + 8 * nt + 2 * cq)]) =
              pack_bf16(st[mh][4 * nt + 2 * eh], st[mh][4 * nt + 2 * eh + 1]);
    fence_async();
    bar_sync(BAR_CONS + cw, 128);
  }
  for (int it = 0; it < nsb; ++it, ++gi) {
    const int s = gi & 1, t0 = cs + it * C, nv = min(C, cl - it * C);
    Stage& sg = S.stage[s];
    bar_sync(BAR_FULL + s, THREADS);
    float o[16], ga[16];
    wg_fence();
    wgmma32_ss<0, 1, 0>(o, gdesc(&sg.q[0]), gdesc(&S.S[pc]));
#pragma unroll
    for (int kk = 1; kk < 8; ++kk)
      wgmma32_ss<0, 1>(o, gdesc(&sg.q[(kk >> 2) * TILE + (kk & 3) * 16]), gdesc(&S.S[kk * 16 * 64 + pc]));
    // this consumer's half of G: columns j = pc .. pc + 31 (k rows pc .. pc + 31)
    wgmma32_ss<0, 0, 0>(ga, gdesc(&sg.q[0]), gdesc(&sg.k[pc * 64]));
#pragma unroll
    for (int kk = 1; kk < 8; ++kk)
      wgmma32_ss<0, 0>(ga, gdesc(&sg.q[(kk >> 2) * TILE + (kk & 3) * 16]), gdesc(&sg.k[(kk >> 2) * TILE + pc * 64 + (kk & 3) * 16]));
    wg_commit();
    wg_wait0();
    reg_fence<16>(o);
    reg_fence<16>(ga);
    const float elo = sg.erow[ti_lo], ehi = sg.erow[ti_hi];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      o[4 * nt] *= elo; o[4 * nt + 1] *= elo; o[4 * nt + 2] *= ehi; o[4 * nt + 3] *= ehi;
    }
    __nv_bfloat16* gs = S.G[gi & 1];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int j0 = pc + 8 * nt + 2 * cq, tj = j0 >> 2;
      const float clo = sg.coef[ti_lo * C + tj], chi = sg.coef[ti_hi * C + tj];
      *reinterpret_cast<uint32_t*>(&gs[swz(16 * w + g_, j0)]) = pack_bf16(ga[4 * nt] * clo, ga[4 * nt + 1] * clo);
      *reinterpret_cast<uint32_t*>(&gs[swz(16 * w + g_ + 8, j0)]) = pack_bf16(ga[4 * nt + 2] * chi, ga[4 * nt + 3] * chi);
    }
    fence_async();
    // the chunk's final state is not needed (the Triton phases own it); the update still runs on the last
    // sub-block so that every wgmma sits on a uniform path
    const bool upd = it + 1 < nsb;
    {
      const float dec = sg.dec;
#pragma unroll
      for (int mh = 0; mh < 2; ++mh)
#pragma unroll
        for (int i = 0; i < 16; ++i) st[mh][i] *= dec;
    }
    bar_sync(BAR_G, 256);  // both halves of G are in smem
    reg_fence<16>(o);
    reg_fence<16>(st[0]);
    reg_fence<16>(st[1]);
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) wgmma32_ss<0, 1>(o, gdesc(&gs[kk * 16]), gdesc(&sg.pv[kk * 16 * 64 + pc]));
    {
#pragma unroll
      for (int mh = 0; mh < 2; ++mh)
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          wgmma32_ss<1, 1>(st[mh], gdesc(&sg.k[mh * TILE + kk * 16 * 64]), gdesc(&sg.pv[kk * 16 * 64 + pc]));
    }
    wg_commit();
    wg_wait0();
    reg_fence<16>(o);
    reg_fence<16>(st[0]);
    reg_fence<16>(st[1]);
    // D skip, gate, sum over the R ranks (rows r..r+3 sit in lanes 4 apart); the warp's 4 tokens are staged
    // in smem and stored as 16-byte pieces
    {
      const int rank = g_ & 3;
      __nv_bfloat16* os_w = S.os[cw][w];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int p0 = pc + 8 * nt + 2 * cq;
#pragma unroll
        for (int eh = 0; eh < 2; ++eh) {
          const int rw = 16 * w + g_ + 8 * eh;
          const float2 pv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sg.pv[swz(rw, p0)]));
          const float2 gz = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sg.gz[swz(rw, p0)]));
          float v0 = (o[4 * nt + 2 * eh] + dsk * pv.x) * gz.x;
          float v1 = (o[4 * nt + 2 * eh + 1] + dsk * pv.y) * gz.y;
          v0 += __shfl_xor_sync(0xffffffffu, v0, 4);
          v1 += __shfl_xor_sync(0xffffffffu, v1, 4);
          v0 += __shfl_xor_sync(0xffffffffu, v0, 8);
          v1 += __shfl_xor_sync(0xffffffffu, v1, 8);
          if (rank == 0)
            *reinterpret_cast<__nv_bfloat162*>(&os_w[((g_ >> 2) + 2 * eh) * 32 + 8 * nt + 2 * cq]) =
                __floats2bfloat162_rn(v0, v1);
        }
      }
      __syncwarp();
      const int tl = (lane >> 2) & 3, col = 8 * (lane & 3), tk = 4 * w + tl;
      if (lane < 16 && tk < nv)
        *reinterpret_cast<uint4*>(out + ((long)(t0 + tk) * H + h) * P + pc + col) =
            *reinterpret_cast<const uint4*>(&os_w[tl * 32 + col]);
      __syncwarp();
    }
    if (gi + 2 < total) bar_arrive(BAR_EMPTY + s, THREADS);
    if (upd) {
      bar_sync(BAR_CONS + cw, 128);  // every warp's qhat @ S is complete before S is overwritten
#pragma unroll
      for (int mh = 0; mh < 2; ++mh)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int eh = 0; eh < 2; ++eh)
            *reinterpret_cast<uint32_t*>(&S.S[swz(64 * mh + 16 * w + g_ + 8 * eh, pc + 8 * nt + 2 * cq)]) =
                pack_bf16(st[mh][4 * nt + 2 * eh], st[mh][4 * nt + 2 * eh + 1]);
      fence_async();
      bar_sync(BAR_CONS + cw, 128);
    }
  }
  }
}

template <typename Sm>
__device__ __forceinline__ void store_consts(Sm& S, const float* v, int pt) {
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    int j = pt + 256 * k;
    const float x = v[k];
    if (j < R * N) S.qb[(j / N) * LBN + j % N] = __float2bfloat16(x);
    else if ((j -= R * N) < R * N) S.kb[(j / N) * LBN + j % N] = __float2bfloat16(x);
    else if ((j -= R * N) < R * P) S.phi[(j / P) * LPF + j % P] = rbf(x);
    else if ((j -= R * P) < R * P) S.zeta[(j / P) * LPF + j % P] = x;
    else if ((j -= R * P) < R * P) S.psi[(j / P) * LPB + j % P] = __float2bfloat16(x);
    else if ((j -= R * P) < NQ) S.a0[j] = x;
  }
}

// ======================================================================================================
// Chunk-local states for the Triton state pass (replaces _ssd_chunk_state): per 128-token chunk and head,
//   Sloc[n][p] = sum_j khat_j^T (PsiV_j * ts_j * exp(dtot - dacs_j))      (fp32, layout (NC, H, N, P))
// with khat rotated by the chunk-local phase thl (the state pass applies the chunk's phase offset).
// Warpgroup 0 accumulates with wgmma (fp32 state in registers); warpgroups 1, 2 stage and prepare the operands
// (same layout and two-deep ring as the scan kernel).
namespace cs {
constexpr int THREADS = 384;
constexpr int CC = 32, CROWS = CC * R;  // 32-token steps (128 rows): half the synchronization per token
struct __align__(1024) Stage {
  __nv_bfloat16 k[2][2 * TILE];  // khat, [row half][n half][64 rows][64] (A, MN-major)
  __nv_bfloat16 pv[2 * TILE];    // PsiV * w [row][p] (B, MN-major)
};
struct __align__(128) Raw {
  __nv_bfloat16 k[CROWS * N];
  __nv_bfloat16 v[CC * P];
  float th[CC * NQ];
  float dacs[CC], ts[CC];
};
struct Smem {
  Stage stage[2];
  Raw raw[2];
  float tc[CC * LTH], tsn[CC * LTH];
  float w[CC];
  __nv_bfloat16 kb[R * LBN];
  float psi[R * LPF];
};

__device__ __forceinline__ void issue_raw(Raw& rw, int pt, int t0, int nv, int h, int T, int H, const __nv_bfloat16* K,
                                          const __nv_bfloat16* V, const float* thl, const float* dacs, const float* ts_g) {
  const long qbase = (long)t0 * R * N;
#pragma unroll
  for (int i = 0; i < CROWS * N / 8 / 256; ++i) {
    const int idx = pt + 256 * i, row = idx >> 4, ch = idx & 15;
    const bool ok = (row >> 2) < nv;
    cp16(&rw.k[sw(row, 8 * ch, N)], K + (ok ? qbase + (long)row * N + 8 * ch : 0), ok);
  }
  {
    const int tok = pt >> 3, ch = pt & 7;  // 256 threads = 32 tokens x 8 pieces
    const bool ok = tok < nv;
    cp16(&rw.v[sw(tok, 8 * ch, P)], V + (ok ? ((long)(t0 + tok) * H + h) * P + 8 * ch : 0), ok);
    cp16(&rw.th[tok * NQ + 4 * ch], thl + (ok ? ((long)(t0 + tok) * H + h) * NQ + 4 * ch : 0), ok);
  }
  if (pt < 2 * CC) {
    const int a = pt >> 5, j = pt & 31;
    const bool ok = j < nv;
    cp4((a ? rw.ts : rw.dacs) + j, (a ? ts_g : dacs) + (ok ? (long)h * T + t0 + j : 0), ok);
  }
  cp_commit();
}
constexpr int NCONST = R * N + R * P;  // kbias, psi: 3 values per producer thread
__device__ __forceinline__ void load_consts(float* v, int pt, int h, const float* kbias, const float* psi) {
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int j = pt + 256 * k;
    v[k] = j < R * N ? kbias[h * R * N + j] : psi[h * R * P + j - R * N];
  }
}
__device__ __forceinline__ void store_consts(Smem& S, const float* v, int pt) {
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    int j = pt + 256 * k;
    if (j < R * N) S.kb[(j / N) * LBN + j % N] = __float2bfloat16(v[k]);
    else if ((j -= R * N) < R * P) S.psi[(j / P) * LPF + j % P] = v[k];
  }
}

__global__ void __launch_bounds__(THREADS, 1) chunk_state_kernel(
    const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V, const float* __restrict__ kbias,
    const float* __restrict__ psi, const float* __restrict__ thl, const float* __restrict__ dacs,
    const float* __restrict__ ts_g, const float* __restrict__ dtot, const int* __restrict__ ch_start,
    const int* __restrict__ ch_len, float* __restrict__ sloc, int T, int H, int NC) {
  extern __shared__ unsigned char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023));
  const int tid = threadIdx.x, nitems = NC * H;
  int total = 0;
  for (int item = blockIdx.x; item < nitems; item += gridDim.x) total += (ch_len[item / H] + CC - 1) / CC;

  if (tid >= 128) {
    // ---------------- producers ----------------
    const int pt = tid - 128;
    const int wq = (pt >> 5) & 3, row0 = 32 * (pt >> 7) + (pt & 31), rr = row0 & 3;
    int item = blockIdx.x;
    if (item >= nitems) return;
    issue_raw(S.raw[0], pt, ch_start[item / H], min(CC, ch_len[item / H]), item % H, T, H, K, V, thl, dacs, ts_g);
    float pre[3];
    load_consts(pre, pt, item % H, kbias, psi);
    store_consts(S, pre, pt);
    int gi = 0;
    for (; item < nitems; item += gridDim.x) {
      const int c = item / H, h = item % H;
      const int cs = ch_start[c], cl = ch_len[c], nsb = (cl + CC - 1) / CC;
      const float dl = dtot[c * H + h];
      const int nitem = item + gridDim.x, nc = nitem < nitems ? nitem / H : 0, nh = nitem % H;
      for (int it = 0; it < nsb; ++it, ++gi) {
        const int s = gi & 1, t0 = cs + it * CC, nv = min(CC, cl - it * CC);
        Raw& rw = S.raw[gi & 1];
        Stage& sg = S.stage[s];
        const bool item_end = it + 1 == nsb, more = !item_end || nitem < nitems;
        if (more) {
          if (!item_end)
            issue_raw(S.raw[(gi + 1) & 1], pt, t0 + CC, min(CC, cl - (it + 1) * CC), h, T, H, K, V, thl, dacs, ts_g);
          else
            issue_raw(S.raw[(gi + 1) & 1], pt, ch_start[nc], min(CC, ch_len[nc]), nh, T, H, K, V, thl, dacs, ts_g);
          cp_wait<1>();
        } else {
          cp_wait<0>();
        }
        if (item_end && nitem < nitems) load_consts(pre, pt, nh, kbias, psi);
        bar_sync(BAR_PROD, 256);
        if (gi >= 2) bar_sync(BAR_EMPTY + s, THREADS);
#pragma unroll
        for (int i = 0; i < CC * NQ / 256; ++i) {
          const int idx = pt + 256 * i;
          float th = rw.th[idx];
          th = th - 6.283185307179586f * floorf(th * 0.15915494309189535f);
          const int ti = (idx / NQ) * LTH + (idx & (NQ - 1));
          __sincosf(th, &S.tsn[ti], &S.tc[ti]);
        }
        if (pt < CC) S.w[pt] = pt < nv ? rw.ts[pt] * __expf(dl - rw.dacs[pt]) : 0.f;
        bar_sync(BAR_PROD, 256);
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int row = row0 + 64 * rh, tok = row >> 2;
          __nv_bfloat16* sk = sg.k[rh];
          const float wt = S.w[tok];
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int c0 = 16 * wq + 8 * i, c1 = 64 + c0;  // rotated for wq < 2
            U8 k0, k1, bk0, bk1;
            k0.u = *reinterpret_cast<const uint4*>(&rw.k[sw(row, c0, N)]);
            k1.u = *reinterpret_cast<const uint4*>(&rw.k[sw(row, c1, N)]);
            bk0.u = *reinterpret_cast<const uint4*>(&S.kb[rr * LBN + c0]);
            bk1.u = *reinterpret_cast<const uint4*>(&S.kb[rr * LBN + c1]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              k0.h2[j] = __hadd2(k0.h2[j], bk0.h2[j]);
              k1.h2[j] = __hadd2(k1.h2[j], bk1.h2[j]);
            }
            if (wq < 2) {
              const float4* cp = reinterpret_cast<const float4*>(&S.tc[tok * LTH + c0]);
              const float4* sp = reinterpret_cast<const float4*>(&S.tsn[tok * LTH + c0]);
              const float4 ca = cp[0], cb = cp[1], sa = sp[0], sb = sp[1];
              const float cn[8] = {ca.x, ca.y, ca.z, ca.w, cb.x, cb.y, cb.z, cb.w};
              const float sn[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                const float2 y0 = __bfloat1622float2(k0.h2[j]), y1 = __bfloat1622float2(k1.h2[j]);
                const float c_a = cn[2 * j], c_b = cn[2 * j + 1], s_a = sn[2 * j], s_b = sn[2 * j + 1];
                k0.h2[j] = __floats2bfloat162_rn(c_a * y0.x - s_a * y1.x, c_b * y0.y - s_b * y1.y);
                k1.h2[j] = __floats2bfloat162_rn(s_a * y0.x + c_a * y1.x, s_b * y0.y + c_b * y1.y);
              }
            }
            *reinterpret_cast<uint4*>(&sk[swz(row0, c0)]) = k0.u;
            *reinterpret_cast<uint4*>(&sk[TILE + swz(row0, c0)]) = k1.u;
          }
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int col = 16 * wq + 8 * i;
            U8 va;
            va.u = *reinterpret_cast<const uint4*>(&rw.v[sw(tok, col, P)]);
            const float4* pp = reinterpret_cast<const float4*>(&S.psi[rr * LPF + col]);
            const float4 pa = pp[0], pb = pp[1];
            const float ps[8] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float2 x = __bfloat1622float2(va.h2[j]);
              va.h2[j] = __floats2bfloat162_rn(x.x * ps[2 * j] * wt, x.y * ps[2 * j + 1] * wt);
            }
            *reinterpret_cast<uint4*>(&sg.pv[swz(row, col)]) = va.u;  // 128 rows: one tile per 64
          }
        }
        fence_async();
        bar_sync(BAR_PROD, 256);
        if (item_end && nitem < nitems) store_consts(S, pre, pt);
        bar_arrive(BAR_FULL + s, THREADS);
      }
    }
    return;
  }
  // ---------------- consumer ----------------
  const int lane = tid & 31, w = tid >> 5, g_ = lane >> 2, cq = lane & 3;
  int gi = 0;
  for (int item = blockIdx.x; item < nitems; item += gridDim.x) {
    const int c = item / H, h = item % H, nsb = (ch_len[c] + CC - 1) / CC;
    float st[2][32];
#pragma unroll
    for (int mh = 0; mh < 2; ++mh)
#pragma unroll
      for (int i = 0; i < 32; ++i) st[mh][i] = 0.f;
    for (int it = 0; it < nsb; ++it, ++gi) {
      const int s = gi & 1;
      Stage& sg = S.stage[s];
      bar_sync(BAR_FULL + s, THREADS);
      reg_fence<32>(st[0]);
      reg_fence<32>(st[1]);
      wg_fence();
#pragma unroll
      for (int mh = 0; mh < 2; ++mh)
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
          wgmma_ss<1, 1>(st[mh], gdesc(&sg.k[kk >> 2][mh * TILE + (kk & 3) * 16 * 64]), gdesc(&sg.pv[kk * 16 * 64]));
      wg_commit();
      wg_wait0();
      reg_fence<32>(st[0]);
      reg_fence<32>(st[1]);
      if (gi + 2 < total) bar_arrive(BAR_EMPTY + s, THREADS);
    }
    float* dst = sloc + (long)(c * H + h) * N * P;
#pragma unroll
    for (int mh = 0; mh < 2; ++mh)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int eh = 0; eh < 2; ++eh)
          *reinterpret_cast<float2*>(&dst[(64 * mh + 16 * w + g_ + 8 * eh) * P + 8 * nt + 2 * cq]) =
              make_float2(st[mh][4 * nt + 2 * eh], st[mh][4 * nt + 2 * eh + 1]);
  }
}
}  // namespace cs

}  // namespace

// Optional tensors: ain (phase offset per item, else 0), ch_last + fin_k (final K rows, else not written),
// sin_t (state entering every item, [NC][H][N][P] fp32, else zero).
void mamba3_scan_prefill(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor Z,
                         torch::Tensor qbias, torch::Tensor kbias, torch::Tensor psi, torch::Tensor phi, torch::Tensor zeta,
                         torch::Tensor Dskip, torch::Tensor thl, c10::optional<torch::Tensor> ain, torch::Tensor dcs16,
                         torch::Tensor dcsr16, torch::Tensor ts, torch::Tensor gam, torch::Tensor ch_start, torch::Tensor ch_len,
                         c10::optional<torch::Tensor> ch_last, c10::optional<torch::Tensor> sin_t, torch::Tensor out,
                         c10::optional<torch::Tensor> fin_k, int64_t NC) {
  const int T = V.size(0), H = V.size(1);
  TORCH_CHECK(V.size(2) == P && Q.size(-1) == N && Q.is_contiguous() && K.is_contiguous() && V.is_contiguous() && Z.is_contiguous());
  TORCH_CHECK(out.is_contiguous() && thl.is_contiguous() && (!sin_t || sin_t->is_contiguous()));
  TORCH_CHECK(ch_last.has_value() == fin_k.has_value());
  const int smem = (int)sizeof(Smem) + 1024;
  static_assert(sizeof(Smem) + 1024 <= 232448, "shared memory");
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(mamba3_scan_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  static int nsm = 0;
  if (!nsm) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, out.get_device());
  const int grid = (int)std::min<int64_t>(NC * H, nsm);
  if (grid == 0) return;
  mamba3_scan_prefill_kernel<<<grid, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(Z.data_ptr()),
      qbias.data_ptr<float>(), kbias.data_ptr<float>(), psi.data_ptr<float>(), phi.data_ptr<float>(), zeta.data_ptr<float>(),
      Dskip.data_ptr<float>(), thl.data_ptr<float>(), ain ? ain->data_ptr<float>() : nullptr, dcs16.data_ptr<float>(),
      dcsr16.data_ptr<float>(), ts.data_ptr<float>(), gam.data_ptr<float>(), ch_start.data_ptr<int>(), ch_len.data_ptr<int>(),
      ch_last ? ch_last->data_ptr<int>() : nullptr, sin_t ? sin_t->data_ptr<float>() : nullptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      fin_k ? reinterpret_cast<__nv_bfloat16*>(fin_k->data_ptr()) : nullptr, T, H, (int)NC);
}

void mamba3_chunk_state(torch::Tensor K, torch::Tensor V, torch::Tensor kbias, torch::Tensor psi, torch::Tensor thl,
                        torch::Tensor dacs, torch::Tensor ts, torch::Tensor dtot, torch::Tensor ch_start, torch::Tensor ch_len,
                        torch::Tensor sloc, int64_t NC) {
  const int T = V.size(0), H = V.size(1);
  TORCH_CHECK(V.size(2) == P && K.size(-1) == N && K.is_contiguous() && V.is_contiguous() && thl.is_contiguous());
  const int smem = (int)sizeof(cs::Smem) + 1024;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(cs::chunk_state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  static int nsm = 0;
  if (!nsm) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, sloc.get_device());
  const int grid = (int)std::min<int64_t>(NC * H, nsm);
  if (grid == 0) return;
  cs::chunk_state_kernel<<<grid, cs::THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
      kbias.data_ptr<float>(), psi.data_ptr<float>(), thl.data_ptr<float>(), dacs.data_ptr<float>(), ts.data_ptr<float>(),
      dtot.data_ptr<float>(), ch_start.data_ptr<int>(), ch_len.data_ptr<int>(), sloc.data_ptr<float>(), T, H, (int)NC);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scan", &mamba3_scan_prefill);
  m.def("chunk_state", &mamba3_chunk_state);
}
