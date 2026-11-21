// Shared-memory bf16 tiles with a 16-byte-chunk XOR swizzle, and mma.sync m16n8k16 GEMM helpers
// for one warp owning a 16-row (or 32-row) slab of the output. Used by the Mamba-3 MIMO bwd kernel.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace m3 {

__device__ __forceinline__ unsigned su(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

// row-major [ROWS x COLS] bf16, COLS in {64, 128}: chunk (16 B = 8 elems) index XORed with (row & 7)
template <int COLS>
struct Tile {
  static_assert(COLS == 64 || COLS == 128, "swizzle needs >= 8 chunks per row");
  __nv_bfloat16* p;
  __device__ __forceinline__ int off(int r, int c) const { return r * COLS + ((((c >> 3) ^ (r & 7))) << 3) + (c & 7); }
  __device__ __forceinline__ __nv_bfloat16& at(int r, int c) const { return p[off(r, c)]; }
  __device__ __forceinline__ __nv_bfloat162* at2(int r, int c) const { return reinterpret_cast<__nv_bfloat162*>(p + off(r, c)); }  // c even
  __device__ __forceinline__ const void* chunk(int r, int c8) const { return p + off(r, c8); }                         // c8 % 8 == 0
};

__device__ __forceinline__ void ldm4(uint32_t* r, const void* a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(su(a)));
}
__device__ __forceinline__ void ldm4t(uint32_t* r, const void* a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(su(a)));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A fragment (m16 x k16) at rows [m0, m0+16), k [k0, k0+16).
//   A_T = false: A stored row-major [m][k] in tile X.
//   A_T = true : A = X^T, X stored [k][m]  (ldmatrix.trans).
template <bool A_T, int C>
__device__ __forceinline__ void load_a(uint32_t* a, const Tile<C>& X, int m0, int k0) {
  const int lane = threadIdx.x & 31;
  if (!A_T) {
    ldm4(a, X.chunk(m0 + (lane & 15), k0 + ((lane >> 4) << 3)));
  } else {
    const int mi = lane >> 3;
    ldm4t(a, X.chunk(k0 + ((mi >> 1) << 3) + (lane & 7), m0 + ((mi & 1) << 3)));
  }
}
// B fragments for two n8 tiles [n0, n0+16) x k [k0, k0+16): b[0..1] -> n-tile n0, b[2..3] -> n0+8.
//   B_T = false: B^T stored, Y [n][k] row-major (ldmatrix).
//   B_T = true : B stored [k][n] row-major (ldmatrix.trans).
template <bool B_T, int C>
__device__ __forceinline__ void load_b(uint32_t* b, const Tile<C>& Y, int n0, int k0) {
  const int lane = threadIdx.x & 31, mi = lane >> 3;
  if (!B_T) ldm4(b, Y.chunk(n0 + ((mi >> 1) << 3) + (lane & 7), k0 + ((mi & 1) << 3)));
  else      ldm4t(b, Y.chunk(k0 + ((mi & 1) << 3) + (lane & 7), n0 + ((mi >> 1) << 3)));
}

// acc[MT][NN/8][4] += A[MT*16 rows starting at m0, K] * B[K, NN]
template <int MT, int NN, int K, bool A_T, bool B_T, int CA, int CB>
__device__ __forceinline__ void gemm(float (&acc)[MT][NN / 8][4], const Tile<CA>& A, int m0, const Tile<CB>& B, int n0 = 0) {
#pragma unroll
  for (int k0 = 0; k0 < K; k0 += 16) {
    uint32_t a[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) load_a<A_T>(a[mt], A, m0 + mt * 16, k0);
#pragma unroll
    for (int nt = 0; nt < NN / 8; nt += 2) {
      uint32_t b[4];
      load_b<B_T>(b, B, n0 + nt * 8, k0);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) { mma(acc[mt][nt], a[mt], b[0], b[1]); mma(acc[mt][nt + 1], a[mt], b[2], b[3]); }
    }
  }
}

template <int MT, int NT>
__device__ __forceinline__ void zero(float (&acc)[MT][NT][4]) {
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
}

// Accumulator element coordinates: tile (mt, nt), element e in 0..3 -> (row, col) relative to the warp slab.
__device__ __forceinline__ int acc_row(int mt, int e) { return mt * 16 + ((threadIdx.x & 31) >> 2) + ((e >> 1) << 3); }
__device__ __forceinline__ int acc_col(int nt, int e) { return nt * 8 + 2 * (threadIdx.x & 3) + (e & 1); }

// store accumulator (rows m0 + slab) to a tile as bf16
template <int MT, int NT, int C>
__device__ __forceinline__ void store_acc(const float (&acc)[MT][NT][4], const Tile<C>& D, int m0) {
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        *D.at2(m0 + acc_row(mt, 2 * h), acc_col(nt, 0)) = __floats2bfloat162_rn(acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1]);
}

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16(v)); }   // round through bf16

}  // namespace m3
