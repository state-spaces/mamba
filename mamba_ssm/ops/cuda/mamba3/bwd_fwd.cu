// Mamba-3 MIMO varlen backward ("bwd_fwd" pass), hand-written CUDA for the Olala configuration:
// R=4, N=128, P=64, chunk 16 (64 fused rows), one Q/K group, reduceO + Z gate + D skip (no fused RMSNorm),
// 32 rotary angles, bf16 activations / bf16 cached states. Drop-in for the TileLang mamba_mimo_bwd_fwd kernel
// (same outputs: STATES, QK_DOT, DZ, DMIMO_O, DMIMO_Z, FINAL_STATE; same two-level / init-state / state-only
// modes); see that kernel for the math.
//
// Full pass: one CTA (8 warps, one CTA per SM) per (head, chunk-block); chunks walk forward with the state
// (N x P, fp32) carried in registers (warp w owns rows 16w..16w+15). GEMM / epilogue warp tile: 32 rows x
// (16 state columns | 16 intra-chunk score columns), so each q fragment feeds both products; four warps also
// compute one 16x16 same-token q.k block each. All GEMMs are mma.sync m16n8k16 on XOR-swizzled bf16 tiles; the
// next chunk's raw inputs are staged with cp.async while the current chunk computes. Element-wise steps that are
// bf16 x bf16 in TileLang use packed bf16x2 arithmetic (one rounding of the exact result, same values). The kernel
// is shared-memory-bandwidth bound, so per-thread constants (Phi, Zeta columns) live in registers.
// dMIMO_O / dMIMO_Z are reduced in a fixed order (bitwise deterministic).
// State-only pass (two-level Pass A): a separate lean kernel, three CTAs per SM.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "tile.cuh"

namespace m3 {
namespace {
constexpr int C = 16, R = 4, F = C * R, N = 128, P = 64, NA = 32, THREADS = 256;

struct __align__(128) Smem {
  __nv_bfloat16 q[F * N];      // rotated q (bias added)            Tile<128>
  __nv_bfloat16 ks[F * N];     // rotated k * trap_scale            Tile<128>
  __nv_bfloat16 kst[F * N];    // ks * exp(dA_cs_rev)               Tile<128>
  __nv_bfloat16 qpre[F * N];   // q + bias, pre-rotation            Tile<128>
  __nv_bfloat16 kpre[F * N];   // k + bias, pre-rotation            Tile<128>
  __nv_bfloat16 st[N * P];     // bf16 state entering the chunk     Tile<64>
  __nv_bfloat16 psiv[F * P];   // PsiV                              Tile<64>
  __nv_bfloat16 msk[F * F];    // masked, decayed q.ks^T            Tile<64>
  float gamma[C], exp_cs[C], eseg[C * C];
  __align__(16) __nv_bfloat16 qkd[C * R * R];
  float scal[1];
  __align__(16) __nv_bfloat16 qb[R * N], kb[R * N], psi[R * P];
  __align__(16) __nv_bfloat16 dout_c[C * P], z_c[C * P];
  // cp.async staging of the NEXT chunk's raw inputs
  __align__(128) __nv_bfloat16 st_q[F * N], st_k[F * N];
  __align__(16) __nv_bfloat16 st_dout[C * P], st_z[C * P], st_v[C * P];
  __align__(16) float st_ang[C * NA], st_seg[C * C];
};

struct Args {
  const __nv_bfloat16 *dout, *q, *k, *v, *z, *trap;
  const float *q_bias, *k_bias, *mimo_v, *mimo_o, *mimo_z, *angles, *da_cs, *da_cs_rev, *dt, *Dp, *segsum, *init_state;
  __nv_bfloat16 *states, *dz, *qkdot;
  float *dmimo_o, *dmimo_z, *final_state;
  const int *cu, *blk_c0, *blk_nch, *blk_s0, *blk_len, *blk_ch0;
  int S, H, NS, max_nchunks, nblk_dim;
  bool blocked, has_init, state_only;
};

__device__ __forceinline__ void cpa16(void* sdst, const void* g, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(su(sdst)), "l"(g), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cpa_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int j = 0; j < 4; ++j) { const float2 v = __bfloat1622float2(p[j]); f[2 * j] = v.x; f[2 * j + 1] = v.y; }
}
__device__ __forceinline__ float tanh_fast(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigm_fast(float x) { return __fdividef(1.f, 1.f + __expf(-x)); }

__global__ void __launch_bounds__(THREADS, 1) bwd_fwd_kernel(const Args a) {
  extern __shared__ __align__(128) char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const Tile<128> Tq{s.q}, Tks{s.ks}, Tkst{s.kst}, Tqpre{s.qpre}, Tkpre{s.kpre};
  const Tile<64> Tst{s.st}, Tpsiv{s.psiv}, Tmsk{s.msk};
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int h = blockIdx.x, blk = blockIdx.y, H = a.H;
  const bool so = a.state_only;

  // ---- segment / block bounds (same rules as the TileLang kernel)
  int seg_s0, seg_ch0, seg_len, blk_c0, blk_nch;
  if (a.blocked) { seg_s0 = a.blk_s0[blk]; seg_ch0 = a.blk_ch0[blk]; seg_len = a.blk_len[blk]; }
  else if (a.NS > 1) { seg_s0 = a.cu[blk]; seg_ch0 = seg_s0 / C + blk; seg_len = a.cu[blk + 1] - seg_s0; }
  else { seg_s0 = 0; seg_ch0 = 0; seg_len = a.S; }
  const int seg_end = seg_s0 + seg_len, tail = seg_len % C;
  const int full_nchunks = seg_len / C + (tail > 0 ? 1 : 0);
  if (a.blocked) { blk_c0 = a.blk_c0[blk]; blk_nch = a.blk_nch[blk]; } else { blk_c0 = 0; blk_nch = full_nchunks; }

  for (int i = t; i < R * N; i += THREADS) {
    const int r = i / N, n = i % N;
    s.qb[i] = __float2bfloat16(a.q_bias[((size_t)h * R + r) * N + n]);
    s.kb[i] = __float2bfloat16(a.k_bias[((size_t)h * R + r) * N + n]);
  }
  for (int i = t; i < R * P; i += THREADS) {
    const size_t o = (size_t)h * R * P + i;
    s.psi[i] = __float2bfloat16(a.mimo_v[o]);
  }
  const int g = lane >> 2, qd = lane & 3, r_own = g & 3;
  const int rb = 32 * (warp & 1), cg = warp >> 1, pcg = 16 * cg;   // GEMM / epilogue warp tile

  // state rows n = 16*warp + acc_row(0, e), all P columns
  float S_[1][8][4];
  if (a.has_init) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        S_[0][nt][e] = a.init_state[(((size_t)blk * H + h) * N + 16 * warp + acc_row(0, e)) * P + acc_col(nt, e)];
  } else {
    zero(S_);
  }
  float dPhi[2][2], dZeta[2][2];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) { dPhi[nt][0] = dPhi[nt][1] = dZeta[nt][0] = dZeta[nt][1] = 0.f; }
  const float Dh = __ldg(a.Dp + h);
  // epilogue constants of this thread's (rank, columns): fixed for the whole kernel
  __nv_bfloat162 phi_r[2]; float2 zeta_r[2];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    const size_t o = ((size_t)h * R + r_own) * P + pcg + nt * 8 + 2 * qd;
    phi_r[nt] = __floats2bfloat162_rn(a.mimo_o[o], a.mimo_o[o + 1]);
    zeta_r[nt] = make_float2(a.mimo_z[o], a.mimo_z[o + 1]);
  }

  auto issue_inputs = [&](int ci_n) {
    const int c0n = seg_s0 + ci_n * C, gch = seg_ch0 + ci_n;
#pragma unroll
    for (int j = 0; j < 4; ++j) {                     // q, k: 1024 x 16 B each
      const int i = t + THREADS * j, row = i >> 4, c8 = (i & 15) * 8;
      const bool ok = c0n + (row >> 2) < a.S;
      const size_t gb = ((size_t)c0n * R + row) * N + c8;
      if (!so) cpa16(s.st_q + row * N + c8, a.q + (ok ? gb : 0), ok);
      cpa16(s.st_k + row * N + c8, a.k + (ok ? gb : 0), ok);
    }
    if (t < 128) {                                     // dout, z: 128 x 16 B each
      const int cs = t >> 3, p8 = (t & 7) * 8; const bool ok = c0n + cs < a.S;
      const size_t base = ok ? ((size_t)(c0n + cs) * H + h) * P + p8 : 0;
      if (!so) { cpa16(s.st_dout + cs * P + p8, a.dout + base, ok); cpa16(s.st_z + cs * P + p8, a.z + base, ok); }
    } else {                                           // v, angles
      const int u = t - 128, cs = u >> 3, p8 = (u & 7) * 8; const bool ok = c0n + cs < a.S;
      cpa16(s.st_v + cs * P + p8, a.v + (ok ? ((size_t)(c0n + cs) * H + h) * P + p8 : 0), ok);
      const int a4 = (u & 7) * 4;
      cpa16(s.st_ang + cs * NA + a4, a.angles + (ok ? ((size_t)(c0n + cs) * H + h) * NA + a4 : 0), ok);
    }
    if (!so && t < 64) cpa16(s.st_seg + t * 4, a.segsum + ((size_t)h * a.max_nchunks + gch) * C * C + t * 4, true);
    cpa_commit();
  };
  float rs_dt = 0.f, rs_tr = 0.f, rs_dts = 0.f, rs_trs = 0.f, rs_er = 0.f, rs_ec = 0.f;
  auto load_scalars = [&](int ci_n) {
    const int c0n = seg_s0 + ci_n * C;
    {   // every thread: its token t >> 4; tokens past S read as 0 (TileLang's guarded loads)
      const int tk = t >> 4;
      const size_t o = (size_t)h * a.S + c0n + tk;
      const bool ok = c0n + tk < a.S, ok1 = c0n + tk + 1 < a.S;
      rs_dt = ok ? __ldg(a.dt + o) : 0.f; rs_tr = ok ? bf(a.trap[o]) : 0.f;
      rs_er = ok ? __ldg(a.da_cs_rev + o) : 0.f; rs_ec = ok ? __ldg(a.da_cs + o) : 0.f;
      rs_dts = ok1 ? __ldg(a.dt + o + 1) : 0.f; rs_trs = ok1 ? bf(a.trap[o + 1]) : 0.f;
    }
  };
  if (blk_nch > 0) { issue_inputs(blk_c0); load_scalars(blk_c0); }

  for (int it = 0; it < blk_nch; ++it) {
    const int ci = blk_c0 + it;                 // segment-relative chunk index
    const int c0 = seg_s0 + ci * C;
    const int gchunk = seg_ch0 + ci;
    const bool last = ci == full_nchunks - 1;
    const int eff_tail = (last && tail > 0) ? tail : C;
    const int da_end = eff_tail < C ? seg_end - 1 : c0 + C - 1;

    cpa_wait_all();
    __syncthreads();   // staged inputs landed; previous chunk fully consumed every tile
    // per-token scalars, computed by each of the token's 16 threads (no extra barrier); smem copies for later phases
    __nv_bfloat162 ts2; float er;
    {
      const int tk = t >> 4;
      const float gam = rbf(rs_dt) * sigm(rs_tr);
      const float sg = (!last || tk + 1 < eff_tail) ? rbf(rbf(rs_dts) * sigm(-rs_trs)) : 0.f;
      ts2 = __bfloat162bfloat162(__float2bfloat16(gam + sg)); er = __expf(rs_er);
      if ((t & 15) == 0) {
        const float ec = __expf(rs_ec);
        s.gamma[tk] = gam; s.exp_cs[tk] = ec;
        if (c0 + tk == da_end) s.scal[0] = ec;
      }
    }
    if (!so) s.eseg[t] = __expf(s.st_seg[t]);
    if (it + 1 < blk_nch) load_scalars(ci + 1);
    if (!so) store_acc(S_, Tst, 16 * warp);        // bf16 state entering this chunk
    // ---- PsiV (rows cs*R + r); dout / z copied out of the staging buffers.
    // bf16 x bf16 products / sums use packed bf16x2 arithmetic: one rounding of the exact result, as TileLang's fragments.
    {
      const int u = t & 127, rh = t >> 7, cs = u >> 3, p0 = (u & 7) * 8;
      const uint4 rv = *reinterpret_cast<const uint4*>(s.st_v + cs * P + p0);
      if (!so) {
        if (rh == 0) *reinterpret_cast<uint4*>(s.dout_c + cs * P + p0) = *reinterpret_cast<const uint4*>(s.st_dout + cs * P + p0);
        else *reinterpret_cast<uint4*>(s.z_c + cs * P + p0) = *reinterpret_cast<const uint4*>(s.st_z + cs * P + p0);
      }
      const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&rv);
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int r = 2 * rh + rr;
        const uint4 ps = *reinterpret_cast<const uint4*>(s.psi + r * P + p0);
        const __nv_bfloat162* ps2 = reinterpret_cast<const __nv_bfloat162*>(&ps);
        uint4 o; __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) o2[j] = __hmul2(v2[j], ps2[j]);
        *reinterpret_cast<uint4*>(Tpsiv.p + Tpsiv.off(cs * R + r, p0)) = o;
      }
    }
    // ---- q / k (+bias): qpre, kpre; rotated q; ks = rot(k) * trap_scale; kst = ks * exp(dA_cs_rev).
    // Thread = (row, c): rotary 8-column group c (n0 = 8c, paired with n0 + 64) and the plain group (n0 + 32, n0 + 96).
    {
      const int row = t >> 2, c = t & 3, cs = row >> 2, r = row & 3;
      const bool ok = c0 + cs < a.S;
      const __nv_bfloat162 z2 = __floats2bfloat162_rn(0.f, 0.f);
      float cnv[8], snv[8];
      {
        const float4* ap = reinterpret_cast<const float4*>(s.st_ang + cs * NA + c * 8);
        const float4 a0 = ap[0], a1 = ap[1];
        const float an[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) __sincosf(an[e], &snv[e], &cnv[e]);
      }
      auto rot = [&](const __nv_bfloat162* x1, const __nv_bfloat162* x2, __nv_bfloat162* y1, __nv_bfloat162* y2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 a1 = __bfloat1622float2(x1[j]), a2 = __bfloat1622float2(x2[j]);
          const float c_0 = cnv[2 * j], c_1 = cnv[2 * j + 1], s_0 = snv[2 * j], s_1 = snv[2 * j + 1];
          y1[j] = __floats2bfloat162_rn(c_0 * a1.x - s_0 * a2.x, c_1 * a1.y - s_1 * a2.y);
          y2[j] = __floats2bfloat162_rn(s_0 * a1.x + c_0 * a2.x, s_1 * a1.y + c_1 * a2.y);
        }
      };
#pragma unroll
      for (int grp = 0; grp < 2; ++grp) {
        const int n0 = c * 8 + grp * 32;
        uint4 P1, P2, O1, O2;
        __nv_bfloat162 *p1 = reinterpret_cast<__nv_bfloat162*>(&P1), *p2 = reinterpret_cast<__nv_bfloat162*>(&P2);
        __nv_bfloat162 *o1 = reinterpret_cast<__nv_bfloat162*>(&O1), *o2 = reinterpret_cast<__nv_bfloat162*>(&O2);
        if (!so) {
          const uint4 X1 = *reinterpret_cast<const uint4*>(s.st_q + row * N + n0), X2 = *reinterpret_cast<const uint4*>(s.st_q + row * N + n0 + 64);
          const uint4 B1 = *reinterpret_cast<const uint4*>(s.qb + r * N + n0), B2 = *reinterpret_cast<const uint4*>(s.qb + r * N + n0 + 64);
          const __nv_bfloat162 *x1 = reinterpret_cast<const __nv_bfloat162*>(&X1), *x2 = reinterpret_cast<const __nv_bfloat162*>(&X2);
          const __nv_bfloat162 *b1 = reinterpret_cast<const __nv_bfloat162*>(&B1), *b2 = reinterpret_cast<const __nv_bfloat162*>(&B2);
#pragma unroll
          for (int j = 0; j < 4; ++j) { p1[j] = ok ? __hadd2(x1[j], b1[j]) : z2; p2[j] = ok ? __hadd2(x2[j], b2[j]) : z2; }
          if (grp == 0) rot(p1, p2, o1, o2); else { O1 = P1; O2 = P2; }
          *reinterpret_cast<uint4*>(Tq.p + Tq.off(row, n0)) = O1;
          *reinterpret_cast<uint4*>(Tq.p + Tq.off(row, n0 + 64)) = O2;
          *reinterpret_cast<uint4*>(Tqpre.p + Tqpre.off(row, n0)) = P1;
          *reinterpret_cast<uint4*>(Tqpre.p + Tqpre.off(row, n0 + 64)) = P2;
        }
        {
          const uint4 X1 = *reinterpret_cast<const uint4*>(s.st_k + row * N + n0), X2 = *reinterpret_cast<const uint4*>(s.st_k + row * N + n0 + 64);
          const uint4 B1 = *reinterpret_cast<const uint4*>(s.kb + r * N + n0), B2 = *reinterpret_cast<const uint4*>(s.kb + r * N + n0 + 64);
          const __nv_bfloat162 *x1 = reinterpret_cast<const __nv_bfloat162*>(&X1), *x2 = reinterpret_cast<const __nv_bfloat162*>(&X2);
          const __nv_bfloat162 *b1 = reinterpret_cast<const __nv_bfloat162*>(&B1), *b2 = reinterpret_cast<const __nv_bfloat162*>(&B2);
#pragma unroll
          for (int j = 0; j < 4; ++j) { p1[j] = ok ? __hadd2(x1[j], b1[j]) : z2; p2[j] = ok ? __hadd2(x2[j], b2[j]) : z2; }
        }
        if (grp == 0) rot(p1, p2, o1, o2); else { O1 = P1; O2 = P2; }
        uint4 K1, K2;
        __nv_bfloat162 *k1 = reinterpret_cast<__nv_bfloat162*>(&K1), *k2 = reinterpret_cast<__nv_bfloat162*>(&K2);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          o1[j] = __hmul2(o1[j], ts2); o2[j] = __hmul2(o2[j], ts2);
          const float2 a1 = __bfloat1622float2(o1[j]), a2 = __bfloat1622float2(o2[j]);
          k1[j] = __floats2bfloat162_rn(a1.x * er, a1.y * er); k2[j] = __floats2bfloat162_rn(a2.x * er, a2.y * er);
        }
        if (!so) {
          *reinterpret_cast<uint4*>(Tkpre.p + Tkpre.off(row, n0)) = P1;
          *reinterpret_cast<uint4*>(Tkpre.p + Tkpre.off(row, n0 + 64)) = P2;
          *reinterpret_cast<uint4*>(Tks.p + Tks.off(row, n0)) = O1;
          *reinterpret_cast<uint4*>(Tks.p + Tks.off(row, n0 + 64)) = O2;
        }
        *reinterpret_cast<uint4*>(Tkst.p + Tkst.off(row, n0)) = K1;
        *reinterpret_cast<uint4*>(Tkst.p + Tkst.off(row, n0 + 64)) = K2;
      }
    }
    __syncthreads();   // staging consumed -> refill it with the next chunk
    if (it + 1 < blk_nch) issue_inputs(ci + 1);

    if (!so) {
      // warp tile: rows rb..rb+32 x (16 state columns pcg.. | 16 F columns pcg.. of q.ks^T); one shared A fragment per k-step.
      // Warps cg < 2 also compute the full 16x16 diagonal block (same-token q.k, pre-rotation) of rows rb + 16 cg.
      float acc[2][2][4], lqk[2][2][4], d4[2][4];
      zero(acc); zero(lqk);
#pragma unroll
      for (int e = 0; e < 4; ++e) { d4[0][e] = 0.f; d4[1][e] = 0.f; }
      const bool dw = cg < 2;
      const int dm = rb + 16 * cg;
#pragma unroll
      for (int k0 = 0; k0 < N; k0 += 16) {
        uint32_t aq[2][4], b1[4], b2[4];
        load_a<false>(aq[0], Tq, rb, k0); load_a<false>(aq[1], Tq, rb + 16, k0);
        load_b<true>(b1, Tst, pcg, k0);
        load_b<false>(b2, Tks, pcg, k0);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma(acc[mt][0], aq[mt], b1[0], b1[1]); mma(acc[mt][1], aq[mt], b1[2], b1[3]);
          mma(lqk[mt][0], aq[mt], b2[0], b2[1]); mma(lqk[mt][1], aq[mt], b2[2], b2[3]);
        }
        if (dw) {
          uint32_t ap[4], bd[4];
          load_a<false>(ap, Tqpre, dm, k0);
          load_b<false>(bd, Tkpre, dm, k0);
          mma(d4[0], ap, bd[0], bd[1]); mma(d4[1], ap, bd[2], bd[3]);
        }
      }
      if (dw) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int row = dm + acc_row(0, e), col = dm + acc_col(nt, e), cs = row >> 2;
            if ((col >> 2) == cs) s.qkd[(cs * R + (row & 3)) * R + (col & 3)] = __float2bfloat16(d4[nt][e]);
          }
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int row = rb + mt * 16 + g + 8 * hf, ti = row >> 2;
          const float ec = s.exp_cs[ti];
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            const int col = pcg + nt * 8 + 2 * qd, tj = col >> 2;
            acc[mt][nt][2 * hf] *= ec; acc[mt][nt][2 * hf + 1] *= ec;
            const float es = ti > tj ? s.eseg[ti * C + tj] : 0.f;
            *Tmsk.at2(row, col) = __floats2bfloat162_rn(lqk[mt][nt][2 * hf] * es, lqk[mt][nt][2 * hf + 1] * es);
          }
        }
      __syncthreads();   // full masked rows; qkd visible
      gemm<2, 16, F, false, true>(acc, Tmsk, rb, Tpsiv, pcg);             // += masked . PsiV
      if (t < C * R && (t >> 2) < eff_tail)   // QK_DOT rows (token, ro): 4 bf16 each
        *reinterpret_cast<uint2*>(a.qkdot + ((size_t)h * a.S + c0) * R * R + t * R) = *reinterpret_cast<const uint2*>(s.qkd + t * R);
      // ---- epilogue: + diag + D * PsiV -> out_pre; gate, dPhi, dPhiO, dZ, dZeta
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int row = rb + mt * 16 + g + 8 * hf, cs = row >> 2, ro = row & 3;
        const float gam = s.gamma[cs];
        const bool valid = cs < eff_tail;
        __nv_bfloat162 qk2[R];
#pragma unroll
        for (int ri = 0; ri < R; ++ri) qk2[ri] = __bfloat162bfloat162(s.qkd[(cs * R + ro) * R + ri]);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int col = pcg + nt * 8 + 2 * qd;
          __nv_bfloat162 dg2 = __floats2bfloat162_rn(0.f, 0.f);
#pragma unroll
          for (int ri = 0; ri < R; ++ri) dg2 = __hadd2(dg2, __hmul2(qk2[ri], *Tpsiv.at2(cs * R + ri, col)));
          const float2 dgf = __bfloat1622float2(dg2);
          const float2 dgg = __bfloat1622float2(__floats2bfloat162_rn(dgf.x * gam, dgf.y * gam));
          const float2 own = __bfloat1622float2(*Tpsiv.at2(row, col));
          const __nv_bfloat162 op2 = __floats2bfloat162_rn(acc[mt][nt][2 * hf] + dgg.x + Dh * own.x, acc[mt][nt][2 * hf + 1] + dgg.y + Dh * own.y);
          const __nv_bfloat162 do2 = valid ? *reinterpret_cast<const __nv_bfloat162*>(s.dout_c + cs * P + col) : __floats2bfloat162_rn(0.f, 0.f);
          const __nv_bfloat162 dpo2 = __hmul2(__hmul2(do2, phi_r[nt]), op2);
          const float2 opf = __bfloat1622float2(op2), dof = __bfloat1622float2(do2), dpf = __bfloat1622float2(dpo2);
          const float2 zz2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(s.z_c + cs * P + col));
          const float2 ze2 = zeta_r[nt];
          const float opv[2] = {opf.x, opf.y}, dov[2] = {dof.x, dof.y}, dpv[2] = {dpf.x, dpf.y}, zv[2] = {zz2.x, zz2.y}, zev[2] = {ze2.x, ze2.y};
          float dzr[2];
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const float zz = zv[j] * zev[j], tmp = zz * 0.5f;
            dPhi[nt][j] += (tmp * tanh_fast(tmp) + tmp) * opv[j] * dov[j];
            const float sz = sigm_fast(zz);
            const float dzz = dpv[j] * sz * (1.f + zz * (1.f - sz));
            dZeta[nt][j] += dzz * zv[j];
            dzr[j] = dzz * zev[j];
          }
          // dZ[cs, p] = sum over the 4 ranks (lanes 4 apart), bf16-rounded after each add like the TileLang fragment
          __nv_bfloat162 x2 = __floats2bfloat162_rn(0.f, 0.f);
#pragma unroll
          for (int r = 0; r < R; ++r) {
            const int src = (lane & 0x13) | (r << 2);
            const float2 xf = __bfloat1622float2(x2);
            x2 = __floats2bfloat162_rn(xf.x + __shfl_sync(0xffffffffu, dzr[0], src), xf.y + __shfl_sync(0xffffffffu, dzr[1], src));
          }
          if (r_own == 0 && valid) *reinterpret_cast<__nv_bfloat162*>(a.dz + ((size_t)(c0 + cs) * H + h) * P + col) = x2;
        }
      }
      // STATES[chunk] = bf16 state entering the chunk (coalesced from the swizzled tile)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = t + THREADS * j, n = i >> 3, c8 = (i & 7) * 8;
        *reinterpret_cast<uint4*>(a.states + (((size_t)h * a.max_nchunks + gchunk) * N + n) * P + c8) =
            *reinterpret_cast<const uint4*>(Tst.chunk(n, c8));
      }
    }
    // ---- state update: S = S * exp(dA_cs[end]) + kst^T . PsiV
    {
      const float sc = s.scal[0];
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) S_[0][nt][e] *= sc;
      gemm<1, P, F, true, true>(S_, Tkst, 16 * warp, Tpsiv);
    }
  }

  if (a.blocked) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        a.final_state[(((size_t)blk * H + h) * N + 16 * warp + acc_row(0, e)) * P + acc_col(nt, e)] = S_[0][nt][e];
  }
  if (!so) {
    __syncthreads();
    // deterministic dMIMO_O / dMIMO_Z: one writer per (row block, r, p), then a fixed-order sum over the 2 row blocks
    float* red = reinterpret_cast<float*>(s.q);   // [2 quantities][2 row blocks][R * P]
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float u = dPhi[nt][j], w = dZeta[nt][j];
        u += __shfl_xor_sync(0xffffffffu, u, 16);
        w += __shfl_xor_sync(0xffffffffu, w, 16);
        const int i = (warp & 1) * R * P + r_own * P + pcg + nt * 8 + 2 * qd + j;
        if ((lane & 16) == 0) { red[i] = u; red[2 * R * P + i] = w; }
      }
    __syncthreads();
    for (int i = t; i < R * P; i += THREADS) {
      const size_t o = ((size_t)h * a.nblk_dim + blk) * R * P + i;
      a.dmimo_o[o] = red[i] + red[R * P + i];
      a.dmimo_z[o] = red[2 * R * P + i] + red[3 * R * P + i];
    }
  }
}

// ---- state-only pass (two-level Pass A): only the recurrence S = S * exp(dA_cs[end]) + kst^T . PsiV.
// Small footprint (~46 KB smem, <= 85 registers) so three CTAs share an SM and hide each other's latency.
struct __align__(128) SmemA {
  __nv_bfloat16 kst[F * N];    // Tile<128>
  __nv_bfloat16 psiv[F * P];   // Tile<64>
  __align__(16) __nv_bfloat16 kb[R * N], psi[R * P];
  __align__(128) __nv_bfloat16 st_k[F * N];
  __align__(16) __nv_bfloat16 st_v[C * P];
  __align__(16) float st_ang[C * NA];
  float scal[1];
};

__global__ void __launch_bounds__(THREADS, 3) bwd_fwd_state_kernel(const Args a) {
  extern __shared__ __align__(128) char smem_raw[];
  SmemA& s = *reinterpret_cast<SmemA*>(smem_raw);
  const Tile<128> Tkst{s.kst};
  const Tile<64> Tpsiv{s.psiv};
  const int t = threadIdx.x, warp = t >> 5;
  const int h = blockIdx.x, blk = blockIdx.y, H = a.H;
  int seg_s0, seg_len, blk_c0, blk_nch;
  if (a.blocked) { seg_s0 = a.blk_s0[blk]; seg_len = a.blk_len[blk]; }
  else if (a.NS > 1) { seg_s0 = a.cu[blk]; seg_len = a.cu[blk + 1] - seg_s0; }
  else { seg_s0 = 0; seg_len = a.S; }
  const int seg_end = seg_s0 + seg_len, tail = seg_len % C;
  const int full_nchunks = seg_len / C + (tail > 0 ? 1 : 0);
  if (a.blocked) { blk_c0 = a.blk_c0[blk]; blk_nch = a.blk_nch[blk]; } else { blk_c0 = 0; blk_nch = full_nchunks; }
  for (int i = t; i < R * N; i += THREADS) s.kb[i] = __float2bfloat16(a.k_bias[(size_t)h * R * N + i]);
  for (int i = t; i < R * P; i += THREADS) s.psi[i] = __float2bfloat16(a.mimo_v[(size_t)h * R * P + i]);
  float S_[1][8][4];
  if (a.has_init) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        S_[0][nt][e] = a.init_state[(((size_t)blk * H + h) * N + 16 * warp + acc_row(0, e)) * P + acc_col(nt, e)];
  } else {
    zero(S_);
  }
  auto issue_inputs = [&](int ci_n) {
    const int c0n = seg_s0 + ci_n * C;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = t + THREADS * j, row = i >> 4, c8 = (i & 15) * 8;
      const bool ok = c0n + (row >> 2) < a.S;
      cpa16(s.st_k + row * N + c8, a.k + (ok ? ((size_t)c0n * R + row) * N + c8 : 0), ok);
    }
    if (t < 128) {
      const int cs = t >> 3, p8 = (t & 7) * 8; const bool ok = c0n + cs < a.S;
      cpa16(s.st_v + cs * P + p8, a.v + (ok ? ((size_t)(c0n + cs) * H + h) * P + p8 : 0), ok);
      const int a4 = (t & 7) * 4;
      cpa16(s.st_ang + cs * NA + a4, a.angles + (ok ? ((size_t)(c0n + cs) * H + h) * NA + a4 : 0), ok);
    }
    cpa_commit();
  };
  float rs_dt = 0.f, rs_tr = 0.f, rs_dts = 0.f, rs_trs = 0.f, rs_er = 0.f, rs_ec = 0.f;
  auto load_scalars = [&](int ci_n) {
    const int c0n = seg_s0 + ci_n * C, tk = t >> 4;
    const size_t o = (size_t)h * a.S + c0n + tk;
    const bool ok = c0n + tk < a.S, ok1 = c0n + tk + 1 < a.S;
    rs_dt = ok ? __ldg(a.dt + o) : 0.f; rs_tr = ok ? bf(a.trap[o]) : 0.f;
    rs_er = ok ? __ldg(a.da_cs_rev + o) : 0.f; rs_ec = ok ? __ldg(a.da_cs + o) : 0.f;
    rs_dts = ok1 ? __ldg(a.dt + o + 1) : 0.f; rs_trs = ok1 ? bf(a.trap[o + 1]) : 0.f;
  };
  if (blk_nch > 0) { issue_inputs(blk_c0); load_scalars(blk_c0); }
  for (int it = 0; it < blk_nch; ++it) {
    const int ci = blk_c0 + it, c0 = seg_s0 + ci * C;
    const bool last = ci == full_nchunks - 1;
    const int eff_tail = (last && tail > 0) ? tail : C;
    const int da_end = eff_tail < C ? seg_end - 1 : c0 + C - 1;
    cpa_wait_all();
    __syncthreads();
    const int row = t >> 2, c = t & 3, cs = row >> 2, r = row & 3;
    __nv_bfloat162 ts2; float er;
    {
      const float gam = rbf(rs_dt) * sigm(rs_tr);
      const float sg = (!last || cs + 1 < eff_tail) ? rbf(rbf(rs_dts) * sigm(-rs_trs)) : 0.f;
      ts2 = __bfloat162bfloat162(__float2bfloat16(gam + sg)); er = __expf(rs_er);
      if ((t & 15) == 0 && c0 + cs == da_end) s.scal[0] = __expf(rs_ec);
    }
    if (it + 1 < blk_nch) load_scalars(ci + 1);
    {   // PsiV
      const int u = t & 127, rh = t >> 7, vc = u >> 3, p0 = (u & 7) * 8;
      const uint4 rv = *reinterpret_cast<const uint4*>(s.st_v + vc * P + p0);
      const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&rv);
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int rk = 2 * rh + rr;
        const uint4 ps = *reinterpret_cast<const uint4*>(s.psi + rk * P + p0);
        const __nv_bfloat162* ps2 = reinterpret_cast<const __nv_bfloat162*>(&ps);
        uint4 o; __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) o2[j] = __hmul2(v2[j], ps2[j]);
        *reinterpret_cast<uint4*>(Tpsiv.p + Tpsiv.off(vc * R + rk, p0)) = o;
      }
    }
    {   // kst = rot(k + bias) * trap_scale * exp(dA_cs_rev)
      const bool ok = c0 + cs < a.S;
      const __nv_bfloat162 z2 = __floats2bfloat162_rn(0.f, 0.f);
#pragma unroll
      for (int grp = 0; grp < 2; ++grp) {
        const int n0 = c * 8 + grp * 32;
        const uint4 X1 = *reinterpret_cast<const uint4*>(s.st_k + row * N + n0), X2 = *reinterpret_cast<const uint4*>(s.st_k + row * N + n0 + 64);
        const uint4 B1 = *reinterpret_cast<const uint4*>(s.kb + r * N + n0), B2 = *reinterpret_cast<const uint4*>(s.kb + r * N + n0 + 64);
        const __nv_bfloat162 *x1 = reinterpret_cast<const __nv_bfloat162*>(&X1), *x2 = reinterpret_cast<const __nv_bfloat162*>(&X2);
        const __nv_bfloat162 *b1 = reinterpret_cast<const __nv_bfloat162*>(&B1), *b2 = reinterpret_cast<const __nv_bfloat162*>(&B2);
        float an[8];
        if (grp == 0) {
          const float4* ap = reinterpret_cast<const float4*>(s.st_ang + cs * NA + c * 8);
          const float4 a0 = ap[0], a1 = ap[1];
          an[0] = a0.x; an[1] = a0.y; an[2] = a0.z; an[3] = a0.w; an[4] = a1.x; an[5] = a1.y; an[6] = a1.z; an[7] = a1.w;
        }
        uint4 K1, K2;
        __nv_bfloat162 *k1 = reinterpret_cast<__nv_bfloat162*>(&K1), *k2 = reinterpret_cast<__nv_bfloat162*>(&K2);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          __nv_bfloat162 p1 = ok ? __hadd2(x1[j], b1[j]) : z2, p2 = ok ? __hadd2(x2[j], b2[j]) : z2;
          if (grp == 0) {
            float s0, c_0, s1, c_1;
            __sincosf(an[2 * j], &s0, &c_0); __sincosf(an[2 * j + 1], &s1, &c_1);
            const float2 a1 = __bfloat1622float2(p1), a2 = __bfloat1622float2(p2);
            p1 = __floats2bfloat162_rn(c_0 * a1.x - s0 * a2.x, c_1 * a1.y - s1 * a2.y);
            p2 = __floats2bfloat162_rn(s0 * a1.x + c_0 * a2.x, s1 * a1.y + c_1 * a2.y);
          }
          const float2 y1 = __bfloat1622float2(__hmul2(p1, ts2)), y2 = __bfloat1622float2(__hmul2(p2, ts2));
          k1[j] = __floats2bfloat162_rn(y1.x * er, y1.y * er); k2[j] = __floats2bfloat162_rn(y2.x * er, y2.y * er);
        }
        *reinterpret_cast<uint4*>(Tkst.p + Tkst.off(row, n0)) = K1;
        *reinterpret_cast<uint4*>(Tkst.p + Tkst.off(row, n0 + 64)) = K2;
      }
    }
    __syncthreads();
    if (it + 1 < blk_nch) issue_inputs(ci + 1);
    const float sc = s.scal[0];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) S_[0][nt][e] *= sc;
    gemm<1, P, F, true, true>(S_, Tkst, 16 * warp, Tpsiv);
  }
  if (a.blocked) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        a.final_state[(((size_t)blk * H + h) * N + 16 * warp + acc_row(0, e)) * P + acc_col(nt, e)] = S_[0][nt][e];
  }
}
}  // namespace
}  // namespace m3

#define BF(x) reinterpret_cast<const __nv_bfloat16*>((x).data_ptr())
#define BFW(x) reinterpret_cast<__nv_bfloat16*>((x).data_ptr())
// Argument order = the TileLang kernel's, minus the unused RMSNorm buffers and NS_ANCHOR / BLK_SEG.
void bwd_fwd(torch::Tensor dout, torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor q_bias, torch::Tensor k_bias,
             torch::Tensor mimo_v, torch::Tensor mimo_o, torch::Tensor dmimo_o, torch::Tensor states, torch::Tensor z,
             torch::Tensor mimo_z, torch::Tensor dz, torch::Tensor dmimo_z, torch::Tensor angles, torch::Tensor da_cs,
             torch::Tensor da_cs_rev, torch::Tensor dt, torch::Tensor trap, torch::Tensor Dp, torch::Tensor qk_dot,
             torch::Tensor segsum, torch::Tensor cu, torch::Tensor blk_seg, torch::Tensor blk_c0, torch::Tensor blk_nch,
             torch::Tensor blk_s0, torch::Tensor blk_len, torch::Tensor blk_ch0, torch::Tensor init_state,
             torch::Tensor final_state, bool blocked, bool has_init, bool state_only) {
  using namespace m3;
  Args a;
  a.dout = BF(dout); a.q = BF(q); a.k = BF(k); a.v = BF(v); a.z = BF(z); a.trap = BF(trap);
  a.q_bias = q_bias.data_ptr<float>(); a.k_bias = k_bias.data_ptr<float>(); a.mimo_v = mimo_v.data_ptr<float>();
  a.mimo_o = mimo_o.data_ptr<float>(); a.mimo_z = mimo_z.data_ptr<float>(); a.angles = angles.data_ptr<float>();
  a.da_cs = da_cs.data_ptr<float>(); a.da_cs_rev = da_cs_rev.data_ptr<float>(); a.dt = dt.data_ptr<float>();
  a.Dp = Dp.data_ptr<float>(); a.segsum = segsum.data_ptr<float>(); a.init_state = init_state.data_ptr<float>();
  a.states = BFW(states); a.dz = BFW(dz); a.qkdot = BFW(qk_dot);
  a.dmimo_o = dmimo_o.data_ptr<float>(); a.dmimo_z = dmimo_z.data_ptr<float>(); a.final_state = final_state.data_ptr<float>();
  a.cu = cu.data_ptr<int>(); a.blk_c0 = blk_c0.data_ptr<int>(); a.blk_nch = blk_nch.data_ptr<int>();
  a.blk_s0 = blk_s0.data_ptr<int>(); a.blk_len = blk_len.data_ptr<int>(); a.blk_ch0 = blk_ch0.data_ptr<int>();
  a.S = q.size(1); a.H = v.size(2); a.NS = cu.numel() - 1; a.max_nchunks = states.size(2);
  a.nblk_dim = dmimo_o.size(2);
  a.blocked = blocked; a.has_init = has_init; a.state_only = state_only;
  TORCH_CHECK(q.size(2) == R && q.size(3) == 1 && q.size(4) == N && v.size(3) == P && angles.size(3) == NA, "Olala shapes only");
  const int grid_y = blocked ? blk_seg.numel() : a.NS;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(bwd_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem));
    cudaFuncSetAttribute(bwd_fwd_state_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(SmemA));
    attr = true;
  }
  if (state_only)
    bwd_fwd_state_kernel<<<dim3(a.H, grid_y), THREADS, sizeof(SmemA), c10::cuda::getCurrentCUDAStream()>>>(a);
  else
    bwd_fwd_kernel<<<dim3(a.H, grid_y), THREADS, sizeof(Smem), c10::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
int smem_bytes() { return sizeof(m3::Smem); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("bwd_fwd", &bwd_fwd); m.def("smem", &smem_bytes); }
