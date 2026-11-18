// Mamba-3 MIMO varlen backward ("bwd_bwd" pass), hand-written CUDA for the Olala configuration:
// R=4, N=128, P=64, chunk 16 (64 fused rows), one Q/K group, reduceO + Z gate + D skip, 32 rotary angles,
// bf16 activations / bf16 cached states. Drop-in for the TileLang mamba_mimo_bwd_bwd kernel (same
// arguments, same outputs, same two-level / init-state / state-only modes); see that kernel for the math.
//
// One CTA (8 warps, 255 registers, ~190 KB smem: one CTA per SM) per (head, chunk-block); chunks walk in
// reverse with dstates (N x P, fp32) carried in registers. Warp = (16-row slab, column half): F x P / F x F
// products split P / F in halves, F x N products split N as [0,32)+[64,96) | [32,64)+[96,128) so the rotary
// pairs (n, n+64) stay in one thread; dstates rows split 8 x 16. All GEMMs are mma.sync m16n8k16 on
// XOR-swizzled bf16 tiles. The same-token dqk_from_diag blocks (4x4 per token) are applied on tensor cores
// as a block-diagonal matrix split into bf16 hi + lo. The next chunk's raw inputs and this chunk's STATES
// are staged with cp.async while the current chunk computes. Per-token sums whose terms span the two column
// halves are accumulated with exactly two atomicAdds onto zero-initialised outputs (order-independent).
#include <torch/extension.h>
#include <cstdlib>
#include <cstdio>
#include <c10/cuda/CUDAStream.h>
#include "tile.cuh"

namespace m3 {
namespace {
constexpr int C = 16, R = 4, F = C * R, N = 128, P = 64, NA = 32, THREADS = 256;

struct __align__(128) Smem {
  __nv_bfloat16 q[F * N];      // rotated q (bias added)                  Tile<128>
  __nv_bfloat16 ks[F * N];     // rotated k * trap_scale                  Tile<128>
  __nv_bfloat16 kpre[F * N];   // k + bias, pre-rotation                  Tile<128>
  __nv_bfloat16 ds[N * P];     // dstates (bf16 copy) / STATES (phase 5)  Tile<64>
  __nv_bfloat16 qpre[F * N];   // q + bias pre-rotation                  Tile<128>
  __nv_bfloat16 m1hi[F * F], m1lo[F * F];   // same-token blocks of (dqk_from_diag * gamma)^T, hi/lo bf16 split
  __nv_bfloat16 dphio[F * P];  // dPhiO                                   Tile<64>
  __nv_bfloat16 psiv[F * P];   // PsiV; later dPhiO * exp(dA_cs)          Tile<64>
  __nv_bfloat16 dkq[F * F];    // lkq masked, then dk_intra masked        Tile<64>
  float cosv[C * NA], sinv[C * NA];
  float gamma[C], exp_rev[C], exp_cs[C];
  __nv_bfloat16 gamma_b[C], tscale_b[C];
  __nv_bfloat16 qkdot[C * R * R];
  float red[8], red2[8];
  float scal[2];               // exp(dA_cs[end]), ddA
  // per-head constants, bf16-rounded like the TileLang fragments
  __align__(16) __nv_bfloat16 qb[R * N], kb[R * N], phi[R * P], zeta[R * P], psi[R * P];
  __align__(16) __nv_bfloat16 vraw[C * P];   // v rows of the chunk
  // cp.async staging of the NEXT chunk's raw inputs, and this chunk's STATES (read in place by phase 5)
  __align__(128) __nv_bfloat16 st_q[F * N], st_k[F * N];          // raw rows, linear
  __align__(16) __nv_bfloat16 st_dout[C * P], st_z[C * P], st_v[C * P];
  __align__(16) float st_ang[C * NA], st_seg[C * C];
  __align__(16) __nv_bfloat16 st_qkd[C * R * R];
  __align__(16) float st_dt[C + 4], st_er[C], st_ec[C];
  __align__(16) __nv_bfloat16 st_tr[C + 8];
  __align__(128) __nv_bfloat16 states[N * P];                      // Tile<64> (swizzled), this chunk
  float eseg[C * C];            // exp(SEGSUM[chunk])[tj][ti]
};

struct Args {
  const __nv_bfloat16 *dout, *q, *k, *v, *z, *trap, *qkdot, *states;
  const float *q_bias, *k_bias, *mimo_v, *mimo_o, *mimo_z, *angles, *da_cs, *da_cs_rev, *dt, *Dp, *segsum, *init_state;
  __nv_bfloat16 *dk, *dv, *dq;
  float *dmimo_v, *dfactor, *dgamma_diag, *dangles, *dd, *dda, *dssda, *dda_cs_rev, *dda_cs, *final_state;
  const int *cu, *blk_seg, *blk_c0, *blk_nch, *blk_s0, *blk_len, *blk_ch0;
  int S, H, NS, max_nchunks, nblk_dim;
  bool blocked, has_init, state_only;
};

__device__ __forceinline__ void cpa16(void* sdst, const void* g, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(su(sdst)), "l"(g), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cpa4(void* sdst, const void* g, bool ok) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(su(sdst)), "l"(g), "r"(ok ? 4 : 0));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cpa_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }
__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int j = 0; j < 4; ++j) { const float2 v = __bfloat1622float2(p[j]); f[2 * j] = v.x; f[2 * j + 1] = v.y; }
}
__device__ __forceinline__ float tanh_fast(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

__global__ void __launch_bounds__(THREADS, 1) bwd_bwd_kernel(const Args a) {
  extern __shared__ __align__(128) char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const Tile<128> Tq{s.q}, Tks{s.ks}, Tkpre{s.kpre};
  const Tile<64> Tds{s.ds}, Tdphio{s.dphio}, Tpsiv{s.psiv}, Tdkq{s.dkq}, Tm1h{s.m1hi}, Tm1l{s.m1lo};
  const Tile<128> Tqpre{s.qpre};
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int h = blockIdx.x, blk = blockIdx.y, H = a.H;

  // ---- segment / block bounds (same rules as the TileLang kernel)
  int seg_s0, seg_ch0, seg_len, blk_c0, blk_nch;
  if (a.blocked) { seg_s0 = a.blk_s0[blk]; seg_ch0 = a.blk_ch0[blk]; seg_len = a.blk_len[blk]; }
  else if (a.NS > 1) { seg_s0 = a.cu[blk]; seg_ch0 = seg_s0 / C + blk; seg_len = a.cu[blk + 1] - seg_s0; }
  else { seg_s0 = 0; seg_ch0 = 0; seg_len = a.S; }
  const int seg_end = seg_s0 + seg_len, tail = seg_len % C;
  const int full_nchunks = seg_len / C + (tail > 0 ? 1 : 0);
  if (a.blocked) { blk_c0 = a.blk_c0[blk]; blk_nch = a.blk_nch[blk]; } else { blk_c0 = 0; blk_nch = full_nchunks; }

  // per-head constants (bf16-rounded like the TileLang fragments)
  for (int i = t; i < R * N; i += THREADS) {
    const int r = i / N, n = i % N;
    s.qb[i] = __float2bfloat16(a.q_bias[((size_t)h * R + r) * N + n]);
    s.kb[i] = __float2bfloat16(a.k_bias[((size_t)h * R + r) * N + n]);
  }
  for (int i = t; i < R * P; i += THREADS) {
    const int r = i / P, p = i % P;
    s.phi[i] = __float2bfloat16(a.mimo_o[((size_t)h * R + r) * P + p]);
    s.zeta[i] = __float2bfloat16(a.mimo_z[((size_t)h * R + r) * P + p]);
    s.psi[i] = __float2bfloat16(a.mimo_v[((size_t)h * R + r) * P + p]);
  }
  // ---- ownership: warp = (row slab, column half)
  const int g = lane >> 2, qd = lane & 3;
  const int r_own = g & 3;
  const int slab = warp & 3, hv = warp >> 2, m0 = 16 * slab;
  const int pc0 = 32 * hv;            // P / F column half
  const int nlo = 32 * hv, nhi = 64 + 32 * hv;   // N column halves (rotary pairs n, n+64 together; rotary only in hv 0)

  // dstates rows n = 16*warp + g + 8*(e>>1), all P columns
  float dS[1][8][4];
  if (a.has_init) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        dS[0][nt][e] = a.init_state[(((size_t)blk * H + h) * N + 16 * warp + acc_row(0, e)) * P + acc_col(nt, e)];
  } else {
    zero(dS);
  }
  store_acc(dS, Tds, 16 * warp);
  float dPsi[4][2];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) { dPsi[nt][0] = 0.f; dPsi[nt][1] = 0.f; }
  float dD = 0.f;
  const float Dh = __ldg(a.Dp + h);

  const Tile<64> Tst{s.states};
  // async staging of a chunk's raw inputs (group "B"); scalars go through registers
  auto issue_inputs = [&](int ci_n) {
    const int c0n = seg_s0 + ci_n * C, gch = seg_ch0 + ci_n;
#pragma unroll
    for (int j = 0; j < 4; ++j) {                     // q, k: 1024 x 16 B each
      const int i = t + THREADS * j, row = i >> 4, c8 = (i & 15) * 8;
      const bool ok = c0n + (row >> 2) < a.S;
      const size_t gb = ((size_t)c0n * R + row) * N + c8;
      cpa16(s.st_q + row * N + c8, a.q + (ok ? gb : 0), ok);
      if (!a.state_only) cpa16(s.st_k + row * N + c8, a.k + (ok ? gb : 0), ok);
    }
    if (t < 128) {                                     // dout, z: 128 x 16 B each
      const int cs = t >> 3, p8 = (t & 7) * 8; const bool ok = c0n + cs < a.S;
      const size_t base = ok ? ((size_t)(c0n + cs) * H + h) * P + p8 : 0;
      cpa16(s.st_dout + cs * P + p8, a.dout + base, ok);
      cpa16(s.st_z + cs * P + p8, a.z + base, ok);
    } else {                                           // v, angles
      const int u = t - 128, cs = u >> 3, p8 = (u & 7) * 8; const bool ok = c0n + cs < a.S;
      cpa16(s.st_v + cs * P + p8, a.v + (ok ? ((size_t)(c0n + cs) * H + h) * P + p8 : 0), ok);
      const int ac = u >> 3, a4 = (u & 7) * 4; const bool aok = c0n + ac < a.S;
      cpa16(s.st_ang + ac * NA + a4, a.angles + (aok ? ((size_t)(c0n + ac) * H + h) * NA + a4 : 0), aok);
    }
    if (!a.state_only) {
      if (t < 64) cpa16(s.st_seg + t * 4, a.segsum + ((size_t)h * a.max_nchunks + gch) * C * C + t * 4, true);
      else if (t < 96) { const int u = t - 64; const bool ok = c0n + (u >> 1) < a.S;
        cpa16(s.st_qkd + u * 8, a.qkdot + (ok ? ((size_t)h * a.S + c0n) * R * R + u * 8 : 0), ok); }
    }
    cpa_commit();
  };
  float rs_dt = 0.f, rs_tr = 0.f, rs_dts = 0.f, rs_trs = 0.f, rs_er = 0.f, rs_ec = 0.f;
  auto load_scalars = [&](int ci_n) {
    const int c0n = seg_s0 + ci_n * C;
    if (t < C) {   // tokens past S read as 0 (TileLang's guarded loads); past the last head's row they would be garbage
      const size_t o = (size_t)h * a.S + c0n + t;
      const bool ok = c0n + t < a.S, ok1 = c0n + t + 1 < a.S;
      rs_dt = ok ? __ldg(a.dt + o) : 0.f; rs_tr = ok ? bf(a.trap[o]) : 0.f;
      rs_er = ok ? __ldg(a.da_cs_rev + o) : 0.f; rs_ec = ok ? __ldg(a.da_cs + o) : 0.f;
      rs_dts = ok1 ? __ldg(a.dt + o + 1) : 0.f; rs_trs = ok1 ? bf(a.trap[o + 1]) : 0.f;
    }
  };
  if (blk_nch > 0) { issue_inputs(blk_c0 + blk_nch - 1); load_scalars(blk_c0 + blk_nch - 1); }

  for (int it = 0; it < blk_nch; ++it) {
    const int ci = blk_c0 + blk_nch - 1 - it;                 // segment-relative chunk index
    const int c0 = seg_s0 + ci * C;                            // first token
    const int gchunk = seg_ch0 + ci;
    const bool last = ci == full_nchunks - 1;
    const int eff_tail = (last && tail > 0) ? tail : C;
    const int da_end = eff_tail < C ? seg_end - 1 : c0 + C - 1;

    cpa_wait_all();
    __syncthreads();   // staged inputs landed; previous chunk fully consumed every tile
    // this chunk's STATES (group "A"), consumed in phase 5
    if (!a.state_only) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = t + THREADS * j, n = i >> 3, c8 = (i & 7) * 8;
        cpa16(Tst.p + Tst.off(n, c8), a.states + (((size_t)h * a.max_nchunks + gchunk) * N + n) * P + c8, true);
      }
    }
    cpa_commit();
    if (t < C) {
      const float dtb = rbf(rs_dt), gam = dtb * sigm(rs_tr);
      const float sg = (!last || t + 1 < eff_tail) ? rbf(rbf(rs_dts) * sigm(-rs_trs)) : 0.f;
      s.gamma[t] = gam; s.gamma_b[t] = __float2bfloat16(gam); s.tscale_b[t] = __float2bfloat16(gam + sg);
      s.exp_rev[t] = __expf(rs_er); s.exp_cs[t] = __expf(rs_ec);
      if (c0 + t == da_end) s.scal[0] = __expf(rs_ec);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) { float sn, cn; __sincosf(s.st_ang[t + THREADS * j], &sn, &cn); s.cosv[t + THREADS * j] = cn; s.sinv[t + THREADS * j] = sn; }
    if (!a.state_only) { s.eseg[t] = __expf(s.st_seg[t]); s.qkdot[t] = s.st_qkd[t]; }
    if (it + 1 < blk_nch) load_scalars(ci - 1);   // registers, consumed next chunk
    __syncthreads();
    // ---- dout / z / v -> dPhiO, PsiV (rows cs*R + r), raw v: thread = (cs, 8 p) x ranks {2*rh, 2*rh+1}
    {
      const int u = t & 127, rh = t >> 7, cs = u >> 3, p0 = (u & 7) * 8;
      const uint4 rd = *reinterpret_cast<const uint4*>(s.st_dout + cs * P + p0);
      const uint4 rz = *reinterpret_cast<const uint4*>(s.st_z + cs * P + p0);
      const uint4 rv = *reinterpret_cast<const uint4*>(s.st_v + cs * P + p0);
      const __nv_bfloat162* pd = reinterpret_cast<const __nv_bfloat162*>(&rd);
      const __nv_bfloat162* pz = reinterpret_cast<const __nv_bfloat162*>(&rz);
      const __nv_bfloat162* pv = reinterpret_cast<const __nv_bfloat162*>(&rv);
      float dof[8], zf[8], vf[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 d2 = __bfloat1622float2(pd[j]), z2 = __bfloat1622float2(pz[j]), v2 = __bfloat1622float2(pv[j]);
        dof[2 * j] = d2.x; dof[2 * j + 1] = d2.y; zf[2 * j] = z2.x; zf[2 * j + 1] = z2.y; vf[2 * j] = v2.x; vf[2 * j + 1] = v2.y;
      }
      if (rh == 0) *reinterpret_cast<uint4*>(s.vraw + cs * P + p0) = rv;
      const bool valid = cs < eff_tail;
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int r = 2 * rh + rr;
        __align__(16) __nv_bfloat16 o[8], pv8[8];
        float phv[8], zev[8], psv[8];
        unpack8(*reinterpret_cast<const uint4*>(s.phi + r * P + p0), phv);
        unpack8(*reinterpret_cast<const uint4*>(s.zeta + r * P + p0), zev);
        unpack8(*reinterpret_cast<const uint4*>(s.psi + r * P + p0), psv);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float x = valid ? rbf(dof[j] * phv[j]) : 0.f;
          const float tmp = zf[j] * zev[j] * 0.5f;
          x = rbf(x * (tmp * tanh_fast(tmp) + tmp));
          const float psi = psv[j];
          dD += x * vf[j] * psi;
          o[j] = __float2bfloat16(x);
          pv8[j] = __float2bfloat16(vf[j] * psi);
        }
        *reinterpret_cast<uint4*>(Tdphio.p + Tdphio.off(cs * R + r, p0)) = *reinterpret_cast<const uint4*>(o);
        if (!a.state_only) *reinterpret_cast<uint4*>(Tpsiv.p + Tpsiv.off(cs * R + r, p0)) = *reinterpret_cast<const uint4*>(pv8);
      }
    }
    // ---- q / k (+bias) -> qpre, rotated q; kpre, rotated k * trap_scale.
    // Thread = (row, c in 0..3): rotary 8-column group c (n0 = 8c, paired with n0 + 64) and the plain group c + 4
    // (n0 + 32, n0 + 96), so every warp runs the same control flow.
    {
      const int row = t >> 2, c = t & 3, cs = row >> 2, r = row & 3;
      const bool ok = c0 + cs < a.S;
      const float ts = bf(s.tscale_b[cs]);
      float cnv[8], snv[8];
      {
        const float4* cp = reinterpret_cast<const float4*>(s.cosv + cs * NA + c * 8);
        const float4* sp = reinterpret_cast<const float4*>(s.sinv + cs * NA + c * 8);
        const float4 c0v = cp[0], c1v = cp[1], s0v = sp[0], s1v = sp[1];
        cnv[0] = c0v.x; cnv[1] = c0v.y; cnv[2] = c0v.z; cnv[3] = c0v.w; cnv[4] = c1v.x; cnv[5] = c1v.y; cnv[6] = c1v.z; cnv[7] = c1v.w;
        snv[0] = s0v.x; snv[1] = s0v.y; snv[2] = s0v.z; snv[3] = s0v.w; snv[4] = s1v.x; snv[5] = s1v.y; snv[6] = s1v.z; snv[7] = s1v.w;
      }
#pragma unroll
      for (int grp = 0; grp < 2; ++grp) {          // 0: rotary pair (n0, n0+64); 1: plain pair (n0+32, n0+96)
        const int n0 = c * 8 + grp * 32;
        float q1v[8], q2v[8], qb1[8], qb2[8], k1v[8], k2v[8], kb1[8], kb2[8];
        unpack8(*reinterpret_cast<const uint4*>(s.st_q + row * N + n0), q1v);
        unpack8(*reinterpret_cast<const uint4*>(s.st_q + row * N + n0 + 64), q2v);
        unpack8(*reinterpret_cast<const uint4*>(s.qb + r * N + n0), qb1);
        unpack8(*reinterpret_cast<const uint4*>(s.qb + r * N + n0 + 64), qb2);
        if (!a.state_only) {
          unpack8(*reinterpret_cast<const uint4*>(s.st_k + row * N + n0), k1v);
          unpack8(*reinterpret_cast<const uint4*>(s.st_k + row * N + n0 + 64), k2v);
          unpack8(*reinterpret_cast<const uint4*>(s.kb + r * N + n0), kb1);
          unpack8(*reinterpret_cast<const uint4*>(s.kb + r * N + n0 + 64), kb2);
        }
        __align__(16) __nv_bfloat16 oq1[8], oq2[8], oqp1[8], oqp2[8], ok1[8], ok2[8], okp1[8], okp2[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float q1 = ok ? rbf(q1v[e] + qb1[e]) : 0.f, q2 = ok ? rbf(q2v[e] + qb2[e]) : 0.f;
          oqp1[e] = __float2bfloat16(q1); oqp2[e] = __float2bfloat16(q2);
          if (grp == 0) { oq1[e] = __float2bfloat16(cnv[e] * q1 - snv[e] * q2); oq2[e] = __float2bfloat16(snv[e] * q1 + cnv[e] * q2); }
          else { oq1[e] = oqp1[e]; oq2[e] = oqp2[e]; }
          if (!a.state_only) {
            const float k1 = ok ? rbf(k1v[e] + kb1[e]) : 0.f, k2 = ok ? rbf(k2v[e] + kb2[e]) : 0.f;
            okp1[e] = __float2bfloat16(k1); okp2[e] = __float2bfloat16(k2);
            if (grp == 0) {
              ok1[e] = __float2bfloat16(rbf(cnv[e] * k1 - snv[e] * k2) * ts); ok2[e] = __float2bfloat16(rbf(snv[e] * k1 + cnv[e] * k2) * ts);
            } else {
              ok1[e] = __float2bfloat16(k1 * ts); ok2[e] = __float2bfloat16(k2 * ts);
            }
          }
        }
        *reinterpret_cast<uint4*>(Tq.p + Tq.off(row, n0)) = *reinterpret_cast<const uint4*>(oq1);
        *reinterpret_cast<uint4*>(Tq.p + Tq.off(row, n0 + 64)) = *reinterpret_cast<const uint4*>(oq2);
        *reinterpret_cast<uint4*>(Tqpre.p + Tqpre.off(row, n0)) = *reinterpret_cast<const uint4*>(oqp1);
        *reinterpret_cast<uint4*>(Tqpre.p + Tqpre.off(row, n0 + 64)) = *reinterpret_cast<const uint4*>(oqp2);
        if (!a.state_only) {
          *reinterpret_cast<uint4*>(Tkpre.p + Tkpre.off(row, n0)) = *reinterpret_cast<const uint4*>(okp1);
          *reinterpret_cast<uint4*>(Tkpre.p + Tkpre.off(row, n0 + 64)) = *reinterpret_cast<const uint4*>(okp2);
          *reinterpret_cast<uint4*>(Tks.p + Tks.off(row, n0)) = *reinterpret_cast<const uint4*>(ok1);
          *reinterpret_cast<uint4*>(Tks.p + Tks.off(row, n0 + 64)) = *reinterpret_cast<const uint4*>(ok2);
        }
      }
    }
    __syncthreads();   // staging consumed -> refill it with the next chunk
    if (it + 1 < blk_nch) issue_inputs(ci - 1);


    if (!a.state_only) {
      // ================= phase 2: dPsiV, lkq, diag, dV, dPsi (cols pc0 .. pc0+32) =================
      float acc[1][4][4];
      zero(acc);
      gemm<1, 32, N, false, true>(acc, Tks, m0, Tds, pc0);                 // k_s . dS        [F,N]x[N,P]
      float lkq[1][4][4];
      zero(lkq);
      gemm<1, 32, N, false, false>(lkq, Tks, m0, Tq, pc0);                 // k_s . q^T       [F,N]x[N,F]
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int row = m0 + g + 8 * hf, ti = row >> 2;
        const float er = s.exp_rev[ti];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int col = pc0 + nt * 8 + 2 * qd, tj = col >> 2;   // both columns share tj
          acc[0][nt][2 * hf] *= er; acc[0][nt][2 * hf + 1] *= er;
          const float es = ti < tj ? s.eseg[tj * C + ti] : 0.f;
          *Tdkq.at2(row, col) = __floats2bfloat162_rn(lkq[0][nt][2 * hf] * es, lkq[0][nt][2 * hf + 1] * es);
          lkq[0][nt][2 * hf] = rbf(lkq[0][nt][2 * hf]); lkq[0][nt][2 * hf + 1] = rbf(lkq[0][nt][2 * hf + 1]);
        }
      }
      __syncthreads();   // both halves of each slab's lkq_m row
      gemm<1, 32, F, false, true>(acc, Tdkq, m0, Tdphio, pc0);            // += lkq_m . dPhiO
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int row = m0 + g + 8 * hf, cs = row >> 2, rin = row & 3;
        const float gb = bf(s.gamma_b[cs]);
        float wq[R];
#pragma unroll
        for (int ro = 0; ro < R; ++ro) wq[ro] = bf(s.qkdot[(cs * R + ro) * R + rin]) * gb;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int col = pc0 + acc_col(nt, 0);
          const float2 own = __bfloat1622float2(*Tdphio.at2(row, col));
          float x0 = acc[0][nt][2 * hf] + own.x * Dh, x1 = acc[0][nt][2 * hf + 1] + own.y * Dh;
#pragma unroll
          for (int ro = 0; ro < R; ++ro) {
            const float2 d = __bfloat1622float2(*Tdphio.at2(cs * R + ro, col));
            x0 += d.x * wq[ro]; x1 += d.y * wq[ro];
          }
          acc[0][nt][2 * hf] = rbf(x0); acc[0][nt][2 * hf + 1] = rbf(x1);   // dPsiV_combined (bf16)
        }
      }
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int col = pc0 + nt * 8 + 2 * qd + j;
          const float psi = bf(s.psi[r_own * P + col]);
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const int cs = (m0 + g + 8 * hf) >> 2;
            const float x = acc[0][nt][2 * hf + j];
            dPsi[nt][j] += x * bf(s.vraw[cs * P + col]);
            float dv = x * psi;
            dv += __shfl_xor_sync(0xffffffffu, dv, 4);
            dv += __shfl_xor_sync(0xffffffffu, dv, 8);
            if (r_own == 0 && cs < eff_tail) a.dv[((size_t)(c0 + cs) * H + h) * P + col] = __float2bfloat16(dv);
          }
        }
      // ================= phase 3: same-token blocks of dqk_from_diag, dgamma (the warp owning the diag columns) ==
      if (hv == (slab >> 1)) {
        float dq4[1][2][4];
        zero(dq4);
#pragma unroll
        for (int k0 = 0; k0 < P; k0 += 16) {
          uint32_t af[4], bfr[4];
          load_a<false>(af, Tdphio, m0, k0);
          load_b<false>(bfr, Tpsiv, m0, k0);
          mma(dq4[0][0], af, bfr[0], bfr[1]);
          mma(dq4[0][1], af, bfr[2], bfr[3]);
        }
        float dg[2] = {0.f, 0.f};
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int row = m0 + acc_row(0, e), col = m0 + acc_col(nt, e), cs = row >> 2;
            float m = 0.f;
            if ((col >> 2) == cs) {
              const int ro = row & 3, ri = col & 3;
              dg[e >> 1] += bf(s.qkdot[(cs * R + ro) * R + ri]) * dq4[0][nt][e];
              m = dq4[0][nt][e] * s.gamma[cs];
            }
            const __nv_bfloat16 mh = __float2bfloat16(m);
            Tm1h.at(col, row) = mh;                                   // M1[(t, ri)][(t, ro)]
            Tm1l.at(col, row) = __float2bfloat16(m - __bfloat162float(mh));
          }
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          float v = dg[hf];
          v += __shfl_xor_sync(0xffffffffu, v, 1); v += __shfl_xor_sync(0xffffffffu, v, 2);
          v += __shfl_xor_sync(0xffffffffu, v, 4); v += __shfl_xor_sync(0xffffffffu, v, 8);
          const int cs = (m0 + g + 8 * hf) >> 2;
          if ((lane & 15) == 0 && cs < eff_tail) a.dgamma_diag[(size_t)h * a.S + c0 + cs] = v;
        }
      }
      __syncthreads();   // lkq_m fully consumed (G3) before dkq overwrites it; dqk visible
      // ================= phase 4a: dk_intra, DSSDA, dkq (cols pc0 ..) =================
      {
        float dki[1][4][4];
        zero(dki);
        gemm<1, 32, P, false, false>(dki, Tpsiv, m0, Tdphio, pc0);         // PsiV . dPhiO^T
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          float v[2];
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            v[hf] = rbf(lkq[0][nt][2 * hf] * dki[0][nt][2 * hf]) + rbf(lkq[0][nt][2 * hf + 1] * dki[0][nt][2 * hf + 1]);
            v[hf] += __shfl_xor_sync(0xffffffffu, v[hf], 1);
            v[hf] += __shfl_xor_sync(0xffffffffu, v[hf], 4);
            v[hf] += __shfl_xor_sync(0xffffffffu, v[hf], 8);
          }
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const int ti = (m0 + g + 8 * hf) >> 2, tj = (pc0 >> 2) + 2 * nt + (qd >> 1);
            if ((lane & 13) == 0) a.dssda[(((size_t)h * a.max_nchunks + gchunk) * C + ti) * C + tj] = v[hf];
          }
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const int row = m0 + g + 8 * hf, col = pc0 + nt * 8 + 2 * qd, ti = row >> 2, tj = col >> 2;
            const float es = ti < tj ? s.eseg[tj * C + ti] : 0.f;
            *Tdkq.at2(row, col) = __floats2bfloat162_rn(dki[0][nt][2 * hf] * es, dki[0][nt][2 * hf + 1] * es);
          }
        }
      }
      __syncthreads();   // full dkq rows for G7 (and dkq^T for G9)
      // ================= phase 4b: dK (cols nlo.., nhi..) =================
      float dangle[4][2][2];
      {
        float dkl[1][4][4], dkh[1][4][4];
        zero(dkl); zero(dkh);
        gemm<1, 32, P, false, false>(dkl, Tpsiv, m0, Tds, nlo);            // PsiV . dS^T
        gemm<1, 32, P, false, false>(dkh, Tpsiv, m0, Tds, nhi);
        float red[2] = {0.f, 0.f};
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int row = m0 + g + 8 * hf;
          const float er = s.exp_rev[row >> 2];
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const int col = nt * 8 + 2 * qd;
            const float2 kl = __bfloat1622float2(*Tks.at2(row, nlo + col)), kh = __bfloat1622float2(*Tks.at2(row, nhi + col));
            float* L = &dkl[0][nt][2 * hf]; float* Hh = &dkh[0][nt][2 * hf];
            red[hf] += kl.x * L[0] + kl.y * L[1] + kh.x * Hh[0] + kh.y * Hh[1];
            L[0] = rbf(L[0] * er); L[1] = rbf(L[1] * er); Hh[0] = rbf(Hh[0] * er); Hh[1] = rbf(Hh[1] * er);
          }
        }
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          float v = red[hf];
          v += __shfl_xor_sync(0xffffffffu, v, 1); v += __shfl_xor_sync(0xffffffffu, v, 2);
          v += __shfl_xor_sync(0xffffffffu, v, 4); v += __shfl_xor_sync(0xffffffffu, v, 8);
          const int cs = (m0 + g + 8 * hf) >> 2;
          if ((lane & 15) == 0 && cs < eff_tail) atomicAdd(&a.dda_cs_rev[(size_t)h * a.S + c0 + cs], v);
        }
        gemm<1, 32, F, false, true>(dkl, Tdkq, m0, Tq, nlo);               // += dkq . q
        gemm<1, 32, F, false, true>(dkh, Tdkq, m0, Tq, nhi);
        float df[2] = {0.f, 0.f};
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int row = m0 + g + 8 * hf, cs = row >> 2;
          const float ts = bf(s.tscale_b[cs]);
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const int n = nlo + nt * 8 + 2 * qd;
            const float2 k1 = __bfloat1622float2(*Tkpre.at2(row, n)), k2 = __bfloat1622float2(*Tkpre.at2(row, n + 64));
            float* L = &dkl[0][nt][2 * hf]; float* Hh = &dkh[0][nt][2 * hf];
            if (hv == 0) {   // rotary columns: rotate k for dfactor, then inverse-rotate dk and form dangle_dk
              const float2 cn = *reinterpret_cast<const float2*>(&s.cosv[cs * NA + n]);
              const float2 sn = *reinterpret_cast<const float2*>(&s.sinv[cs * NA + n]);
              const float c_[2] = {cn.x, cn.y}, s_[2] = {sn.x, sn.y}, a1[2] = {k1.x, k1.y}, a2[2] = {k2.x, k2.y};
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                const float kp1 = rbf(c_[j] * a1[j] - s_[j] * a2[j]), kp2 = rbf(s_[j] * a1[j] + c_[j] * a2[j]);
                df[hf] += kp1 * L[j] + kp2 * Hh[j];
                const float d1 = rbf(L[j] * ts), d2 = rbf(Hh[j] * ts);
                dangle[nt][hf][j] = d1 * (-a1[j] * s_[j] - a2[j] * c_[j]) + d2 * (a1[j] * c_[j] - a2[j] * s_[j]);
                L[j] = rbf(c_[j] * d1 + s_[j] * d2);
                Hh[j] = rbf(-s_[j] * d1 + c_[j] * d2);
              }
            } else {
              df[hf] += k1.x * L[0] + k1.y * L[1] + k2.x * Hh[0] + k2.y * Hh[1];
              L[0] = rbf(L[0] * ts); L[1] = rbf(L[1] * ts); Hh[0] = rbf(Hh[0] * ts); Hh[1] = rbf(Hh[1] * ts);
            }
          }
        }
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          float v = df[hf];
          v += __shfl_xor_sync(0xffffffffu, v, 1); v += __shfl_xor_sync(0xffffffffu, v, 2);
          v += __shfl_xor_sync(0xffffffffu, v, 4); v += __shfl_xor_sync(0xffffffffu, v, 8);
          const int cs = (m0 + g + 8 * hf) >> 2;
          if ((lane & 15) == 0 && cs < eff_tail) atomicAdd(&a.dfactor[(size_t)h * a.S + c0 + cs], v);
        }
        // + same-token diag term on tensor cores: dk[(t,ri), n] += sum_ro M1[(t,ri),(t,ro)] qpre[(t,ro), n]
        {
          uint32_t ah[4], al[4];
          load_a<false>(ah, Tm1h, m0, m0); load_a<false>(al, Tm1l, m0, m0);
#pragma unroll
          for (int nt = 0; nt < 4; nt += 2) {
            uint32_t bl[4], bh[4];
            load_b<true>(bl, Tqpre, nlo + nt * 8, m0); load_b<true>(bh, Tqpre, nhi + nt * 8, m0);
            mma(dkl[0][nt], ah, bl[0], bl[1]); mma(dkl[0][nt + 1], ah, bl[2], bl[3]);
            mma(dkh[0][nt], ah, bh[0], bh[1]); mma(dkh[0][nt + 1], ah, bh[2], bh[3]);
            mma(dkl[0][nt], al, bl[0], bl[1]); mma(dkl[0][nt + 1], al, bl[2], bl[3]);
            mma(dkh[0][nt], al, bh[0], bh[1]); mma(dkh[0][nt + 1], al, bh[2], bh[3]);
          }
        }
#pragma unroll
        for (int part = 0; part < 2; ++part)
#pragma unroll
          for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              const int row = m0 + g + 8 * hf, cs = row >> 2, ri = row & 3;
              const int n = (part ? nhi : nlo) + nt * 8 + 2 * qd;
              const float x0 = part ? dkh[0][nt][2 * hf] : dkl[0][nt][2 * hf];
              const float x1 = part ? dkh[0][nt][2 * hf + 1] : dkl[0][nt][2 * hf + 1];
              if (cs < eff_tail)
                *reinterpret_cast<__nv_bfloat162*>(a.dk + (((size_t)(c0 + cs) * R + ri) * H + h) * N + n) = __floats2bfloat162_rn(x0, x1);
            }
      }
      // ================= phase 5: STATES (staged), ddA, dQ, dAngles =================
      asm volatile("cp.async.wait_group 1;\n");   // group A (STATES) done; the next chunk's inputs may be in flight
      __syncthreads();
      {
        float part = 0.f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) part += bf(Tst.at(16 * warp + acc_row(0, e), acc_col(nt, e))) * dS[0][nt][e];
#pragma unroll
        for (int o = 16; o; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
        if (lane == 0) s.red[warp] = part;
      }
      {
        float dql[1][4][4], dqh[1][4][4];
        zero(dql); zero(dqh);
        gemm<1, 32, P, false, false>(dql, Tdphio, m0, Tst, nlo);           // dPhiO . STATES^T
        gemm<1, 32, P, false, false>(dqh, Tdphio, m0, Tst, nhi);
        float red[2] = {0.f, 0.f};
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int row = m0 + g + 8 * hf;
          const float ec = s.exp_cs[row >> 2];
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const int col = nt * 8 + 2 * qd;
            const float2 ql = __bfloat1622float2(*Tq.at2(row, nlo + col)), qh = __bfloat1622float2(*Tq.at2(row, nhi + col));
            float* L = &dql[0][nt][2 * hf]; float* Hh = &dqh[0][nt][2 * hf];
            red[hf] += ql.x * L[0] + ql.y * L[1] + qh.x * Hh[0] + qh.y * Hh[1];
            L[0] = rbf(L[0] * ec); L[1] = rbf(L[1] * ec); Hh[0] = rbf(Hh[0] * ec); Hh[1] = rbf(Hh[1] * ec);
          }
        }
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          float v = red[hf];
          v += __shfl_xor_sync(0xffffffffu, v, 1); v += __shfl_xor_sync(0xffffffffu, v, 2);
          v += __shfl_xor_sync(0xffffffffu, v, 4); v += __shfl_xor_sync(0xffffffffu, v, 8);
          const int cs = (m0 + g + 8 * hf) >> 2;
          if ((lane & 15) == 0 && cs < eff_tail) atomicAdd(&a.dda_cs[(size_t)h * a.S + c0 + cs], v);
        }
        // dq += dkq^T . k_s
#pragma unroll
        for (int k0 = 0; k0 < F; k0 += 16) {
          uint32_t af[4];
          load_a<true>(af, Tdkq, m0, k0);
#pragma unroll
          for (int nt = 0; nt < 4; nt += 2) {
            uint32_t bl[4], bh[4];
            load_b<true>(bl, Tks, nlo + nt * 8, k0);
            load_b<true>(bh, Tks, nhi + nt * 8, k0);
            mma(dql[0][nt], af, bl[0], bl[1]); mma(dql[0][nt + 1], af, bl[2], bl[3]);
            mma(dqh[0][nt], af, bh[0], bh[1]); mma(dqh[0][nt + 1], af, bh[2], bh[3]);
          }
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) { dql[0][nt][e] = rbf(dql[0][nt][e]); dqh[0][nt][e] = rbf(dqh[0][nt][e]); }
        if (hv == 0) {   // inverse rotary of dq + dangles
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const int row = m0 + g + 8 * hf, cs = row >> 2;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
              const int n = nt * 8 + 2 * qd;
              const float2 q1 = __bfloat1622float2(*Tqpre.at2(row, n)), q2 = __bfloat1622float2(*Tqpre.at2(row, n + 64));
              const float2 cn = *reinterpret_cast<const float2*>(&s.cosv[cs * NA + n]);
              const float2 sn = *reinterpret_cast<const float2*>(&s.sinv[cs * NA + n]);
              const float c_[2] = {cn.x, cn.y}, s_[2] = {sn.x, sn.y}, a1[2] = {q1.x, q1.y}, a2[2] = {q2.x, q2.y};
              float da[2];
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                const float d1 = dql[0][nt][2 * hf + j], d2 = dqh[0][nt][2 * hf + j];
                da[j] = dangle[nt][hf][j] + d1 * (-a1[j] * s_[j] - a2[j] * c_[j]) + d2 * (a1[j] * c_[j] - a2[j] * s_[j]);
                dql[0][nt][2 * hf + j] = rbf(c_[j] * d1 + s_[j] * d2);
                dqh[0][nt][2 * hf + j] = rbf(-s_[j] * d1 + c_[j] * d2);
              }
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                da[j] += __shfl_xor_sync(0xffffffffu, da[j], 4);
                da[j] += __shfl_xor_sync(0xffffffffu, da[j], 8);
              }
              if (r_own == 0 && cs < eff_tail)
                *reinterpret_cast<float2*>(a.dangles + ((size_t)(c0 + cs) * H + h) * NA + n) = make_float2(da[0], da[1]);
            }
          }
        }
        {
          uint32_t ah[4], al[4];
          load_a<true>(ah, Tm1h, m0, m0); load_a<true>(al, Tm1l, m0, m0);
#pragma unroll
          for (int nt = 0; nt < 4; nt += 2) {
            uint32_t bl[4], bh[4];
            load_b<true>(bl, Tkpre, nlo + nt * 8, m0); load_b<true>(bh, Tkpre, nhi + nt * 8, m0);
            mma(dql[0][nt], ah, bl[0], bl[1]); mma(dql[0][nt + 1], ah, bl[2], bl[3]);
            mma(dqh[0][nt], ah, bh[0], bh[1]); mma(dqh[0][nt + 1], ah, bh[2], bh[3]);
            mma(dql[0][nt], al, bl[0], bl[1]); mma(dql[0][nt + 1], al, bl[2], bl[3]);
            mma(dqh[0][nt], al, bh[0], bh[1]); mma(dqh[0][nt + 1], al, bh[2], bh[3]);
          }
        }
#pragma unroll
        for (int part = 0; part < 2; ++part)
#pragma unroll
          for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              const int row = m0 + g + 8 * hf, cs = row >> 2, ro = row & 3;
              const int n = (part ? nhi : nlo) + nt * 8 + 2 * qd;
              const float x0 = part ? dqh[0][nt][2 * hf] : dql[0][nt][2 * hf];
              const float x1 = part ? dqh[0][nt][2 * hf + 1] : dql[0][nt][2 * hf + 1];
              if (cs < eff_tail)
                *reinterpret_cast<__nv_bfloat162*>(a.dq + (((size_t)(c0 + cs) * R + ro) * H + h) * N + n) = __floats2bfloat162_rn(x0, x1);
            }
      }
    }
    // ================= phase 6: dstates update =================
    for (int i = t; i < F * (P / 2); i += THREADS) {
      const int row = i / (P / 2), p = (i % (P / 2)) * 2;
      const float2 x = __bfloat1622float2(*Tdphio.at2(row, p));
      const float sc = s.exp_cs[row >> 2];
      *Tpsiv.at2(row, p) = __floats2bfloat162_rn(x.x * sc, x.y * sc);
    }
    __syncthreads();   // dPhiO_s ready; every G8 done with the staged STATES; s.red complete
    if (!a.state_only && t < eff_tail) {
      const float dda = (s.red[0] + s.red[1] + s.red[2] + s.red[3] + s.red[4] + s.red[5] + s.red[6] + s.red[7]) * s.scal[0];
      a.dda[(size_t)h * a.S + c0 + t] = dda;
    }
    {
      const float sc = s.scal[0];
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) dS[0][nt][e] *= sc;
      gemm<1, P, F, true, true>(dS, Tq, 16 * warp, Tpsiv);                 // += q^T . dPhiO_s
      store_acc(dS, Tds, 16 * warp);
    }
  }

  if (a.blocked) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        a.final_state[(((size_t)blk * H + h) * N + 16 * warp + acc_row(0, e)) * P + acc_col(nt, e)] = dS[0][nt][e];
  }
  if (!a.state_only) {
    __syncthreads();
    // deterministic dMIMO_V: one writer per (slab, r, p), then a fixed-order sum over the 4 slabs
    float* red = reinterpret_cast<float*>(s.q);   // [4 slabs][R * P]
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float v = dPsi[nt][j];
        v += __shfl_xor_sync(0xffffffffu, v, 16);
        if ((lane & 16) == 0) red[slab * R * P + r_own * P + pc0 + nt * 8 + 2 * qd + j] = v;
      }
    float dd = dD;
#pragma unroll
    for (int o = 16; o; o >>= 1) dd += __shfl_xor_sync(0xffffffffu, dd, o);
    if (lane == 0) s.red2[warp] = dd;
    __syncthreads();
    for (int i = t; i < R * P; i += THREADS)
      a.dmimo_v[((size_t)h * a.nblk_dim + blk) * R * P + i] = ((red[i] + red[R * P + i]) + red[2 * R * P + i]) + red[3 * R * P + i];
    if (t == 0) {
      float s8 = 0.f;
      for (int w = 0; w < 8; ++w) s8 += s.red2[w];
      a.dd[(size_t)h * a.nblk_dim + blk] = s8;
    }
  }
}
}  // namespace
}  // namespace m3

#define BF(x) reinterpret_cast<const __nv_bfloat16*>((x).data_ptr())
#define BFW(x) reinterpret_cast<__nv_bfloat16*>((x).data_ptr())
void bwd_bwd(torch::Tensor dout, torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor q_bias, torch::Tensor k_bias,
             torch::Tensor mimo_v, torch::Tensor mimo_o, torch::Tensor dk, torch::Tensor dv, torch::Tensor dmimo_v, torch::Tensor states,
             torch::Tensor dq, torch::Tensor z, torch::Tensor mimo_z, torch::Tensor angles, torch::Tensor da_cs, torch::Tensor da_cs_rev,
             torch::Tensor dt, torch::Tensor trap, torch::Tensor dfactor, torch::Tensor dgamma_diag, torch::Tensor dangles, torch::Tensor Dp,
             torch::Tensor dd, torch::Tensor qk_dot, torch::Tensor dda, torch::Tensor dssda, torch::Tensor dda_cs_rev, torch::Tensor dda_cs,
             torch::Tensor segsum, torch::Tensor cu, torch::Tensor blk_seg, torch::Tensor blk_c0, torch::Tensor blk_nch, torch::Tensor blk_s0,
             torch::Tensor blk_len, torch::Tensor blk_ch0, torch::Tensor init_state, torch::Tensor final_state,
             bool blocked, bool has_init, bool state_only) {
  using namespace m3;
  Args a;
  a.dout = BF(dout); a.q = BF(q); a.k = BF(k); a.v = BF(v); a.z = BF(z); a.trap = BF(trap); a.qkdot = BF(qk_dot); a.states = BF(states);
  a.q_bias = q_bias.data_ptr<float>(); a.k_bias = k_bias.data_ptr<float>(); a.mimo_v = mimo_v.data_ptr<float>(); a.mimo_o = mimo_o.data_ptr<float>();
  a.mimo_z = mimo_z.data_ptr<float>(); a.angles = angles.data_ptr<float>(); a.da_cs = da_cs.data_ptr<float>(); a.da_cs_rev = da_cs_rev.data_ptr<float>();
  a.dt = dt.data_ptr<float>(); a.Dp = Dp.data_ptr<float>(); a.segsum = segsum.data_ptr<float>(); a.init_state = init_state.data_ptr<float>();
  a.dk = BFW(dk); a.dv = BFW(dv); a.dq = BFW(dq);
  a.dmimo_v = dmimo_v.data_ptr<float>(); a.dfactor = dfactor.data_ptr<float>(); a.dgamma_diag = dgamma_diag.data_ptr<float>();
  a.dangles = dangles.data_ptr<float>(); a.dd = dd.data_ptr<float>(); a.dda = dda.data_ptr<float>(); a.dssda = dssda.data_ptr<float>();
  a.dda_cs_rev = dda_cs_rev.data_ptr<float>(); a.dda_cs = dda_cs.data_ptr<float>(); a.final_state = final_state.data_ptr<float>();
  a.cu = cu.data_ptr<int>(); a.blk_seg = blk_seg.data_ptr<int>(); a.blk_c0 = blk_c0.data_ptr<int>(); a.blk_nch = blk_nch.data_ptr<int>();
  a.blk_s0 = blk_s0.data_ptr<int>(); a.blk_len = blk_len.data_ptr<int>(); a.blk_ch0 = blk_ch0.data_ptr<int>();
  a.S = q.size(1); a.H = v.size(2); a.NS = cu.numel() - 1; a.max_nchunks = states.size(2);
  a.nblk_dim = dmimo_v.size(2);
  a.blocked = blocked; a.has_init = has_init; a.state_only = state_only;
  TORCH_CHECK(q.size(2) == R && q.size(3) == 1 && q.size(4) == N && v.size(3) == P && angles.size(3) == NA, "Olala shapes only");
  const int grid_y = blocked ? blk_seg.numel() : a.NS;
  const int smem = sizeof(Smem);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(bwd_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  bwd_bwd_kernel<<<dim3(a.H, grid_y), THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
int smem_bytes() { return sizeof(m3::Smem); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("bwd_bwd", &bwd_bwd); m.def("smem", &smem_bytes); }
