# Copyright (c) 2025, Tri Dao.
# Modified to use tvm-ffi and fake tensors instead of dlpack.
# Modified to optionally update state in place (state_out=None) or write to separate state_out.

import math
from typing import Optional, Type, Literal, List

import torch
import torch.nn.functional as F
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, Float16, BFloat16, Boolean, const_expr

from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))

def select(a: cute.Tensor, mode: List[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))



def get_gmem_tiled_copy(dtype: Type[cutlass.Numeric], major_mode_size: int, num_threads: int, is_async: bool = True):
    num_copy_bits = math.gcd(major_mode_size, 128 // dtype.width) * dtype.width
    copy_elems = num_copy_bits // dtype.width
    copy_op = cute.nvgpu.cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    gmem_threads_per_row = major_mode_size // copy_elems
    assert num_threads % gmem_threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // gmem_threads_per_row, gmem_threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


class Mamba3Step():
    def __init__(self, tile_D: int, dstate: int, mimo: int = 1, num_warps: int = 4, remove_gate: bool = False, remove_outproj: bool = False):
        assert num_warps >= 2
        assert dstate % 8 == 0, "dstate must be multiple of 8" # for vectorized load /store
        self.tile_D = tile_D
        self.dstate = dstate
        self.mimo = mimo
        self.num_warps = num_warps
        self.remove_gate = remove_gate
        self.remove_outproj = remove_outproj

    def _setup_smem_layouts(self):
        self.sState_layout = cute.make_ordered_layout((self.tile_D, self.dstate), order=(1, 0))
        # We don't need any swizzling for Bstate, B, C
        self.sBC_layout = cute.make_ordered_layout((self.mimo, self.dstate), order=(1, 0))
        # We don't need any swizzling for Xproj, Zproj, Outproj
        self.sProj_layout = cute.make_ordered_layout((self.mimo, self.tile_D), order=(1, 0))

    def _setup_gmem_tiled_copy(self, ):
        num_threads = self.num_warps * cute.arch.WARP_SIZE
        self.gmem_tiled_copy_state = get_gmem_tiled_copy(self.dtype, self.dstate, num_threads)
        self.gmem_tiled_copy_BC = get_gmem_tiled_copy(self.b_dtype, self.dstate, num_threads)
        self.gmem_tiled_copy_Proj = get_gmem_tiled_copy(self.proj_dtype, self.tile_D, num_threads)
        # Gmem tiled copy for X, Z
        # e.g. for tile_D = 64, we only want each thread loading 2 values
        copy_elems_x = const_expr(min(4, cute.ceil_div(self.tile_D, cute.arch.WARP_SIZE)))
        num_copy_bits_x = copy_elems_x * self.x_dtype.width
        copy_atom_load_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.x_dtype, num_bits_per_copy=num_copy_bits_x
        )
        gmem_threads_per_row = self.tile_D // copy_elems_x
        assert cute.arch.WARP_SIZE >= gmem_threads_per_row   # Only 1 warp loads X, Z
        self.gmem_tiled_copy_X = cute.make_tiled_copy_tv(
            copy_atom_load_x, cute.make_layout(self.tile_D // copy_elems_x), cute.make_layout(copy_elems_x)
        )


    @cute.jit
    def __call__(
        # B: batch size, H: num heads, D: head dim, N: dstate, R: mimo
        self,
        mState: cute.Tensor,  # (B, H, D, N)
        mBstate: cute.Tensor,  # (B, R, H, N)
        mXstate: cute.Tensor,  # (B, H, D)
        mA: cute.Tensor,  # (B, H)
        mB: cute.Tensor,  # (B, R, H, N)
        mC: cute.Tensor,  # (B, R, H, N)
        mD: cute.Tensor,  # (H)
        mX: cute.Tensor,  # (B, H, D)
        mDt: cute.Tensor,  # (B, H)
        mTrap: cute.Tensor,  # (B, H)
        mXproj: cute.Tensor,  # (R, H, D)
        mOutproj: Optional[cute.Tensor],  # (R, H, D), None if remove_outproj
        mStateOut: cute.Tensor,  # (B, H, D, N) — same as mState for in-place, or separate
        mOut: cute.Tensor,  # (B, H, D) or (B, R, H, D) if remove_outproj
        mZ: Optional[cute.Tensor],  # (B, H, D), None if remove_gate
        mZproj: Optional[cute.Tensor],  # (R, H, D), None if remove_gate
        stream: cuda.CUstream,
    ):
        self.dtype = mState.element_type
        self.b_dtype = mB.element_type
        self.proj_dtype = mXproj.element_type
        self.x_dtype = mX.element_type
        assert mStateOut.element_type == self.dtype
        assert mBstate.element_type == mB.element_type == mC.element_type
        if const_expr(mOutproj is not None):
            assert mXproj.element_type == mOutproj.element_type
        if const_expr(mZ is not None):
            assert mXproj.element_type == mZproj.element_type
            assert mZ.element_type == self.x_dtype

        self._setup_smem_layouts()
        self._setup_gmem_tiled_copy()

        # TV layout, this is the most important step as it decides which elements in B, C, State
        # each thread will load from smem
        num_threads = self.num_warps * cute.arch.WARP_SIZE
        # TODO: these need to be adjusted based on dstate and tile_D
        assert self.dstate in [32, 64, 128]
        # TODO: This is not optimal for dstate=32 and 64, just to get sth quick to run
        vecsize_dstate = 4 if self.dstate == 128 else 2 if self.dstate == 64 else 1
        threads_per_dstate = self.dstate // vecsize_dstate
        assert cute.arch.WARP_SIZE % threads_per_dstate == 0
        num_groups = num_threads // threads_per_dstate
        assert self.tile_D % num_groups == 0
        lanes_per_D = self.tile_D // num_groups
        copy_atom_state_s2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mState.element_type, num_bits_per_copy=vecsize_dstate * mState.element_type.width
        )
        tiled_copy_state_s2r = cute.make_tiled_copy_tv(
            copy_atom_state_s2r,
            cute.make_ordered_layout((num_groups, threads_per_dstate), order=(1, 0)),
            cute.make_ordered_layout((lanes_per_D, vecsize_dstate), order=(1, 0)),
        )
        copy_atom_B_s2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mB.element_type, num_bits_per_copy=vecsize_dstate * mB.element_type.width
        )
        tiled_copy_B_s2r = cute.make_tiled_copy_tv(
            copy_atom_B_s2r,
            cute.make_ordered_layout((1, threads_per_dstate), order=(1, 0)),
            cute.make_ordered_layout((1, vecsize_dstate), order=(1, 0)),
        )

        self.buffer_align_bytes = 1024

        sZproj_size = cute.cosize(self.sProj_layout) if not self.remove_gate else 0
        sOutproj_size = cute.cosize(self.sProj_layout) if not self.remove_outproj else 0

        @cute.struct
        class SharedStorage:
            sX: cute.struct.Align[cute.struct.MemRange[Float32, self.tile_D], 128]
            sXgamma: cute.struct.Align[cute.struct.MemRange[Float32, self.tile_D], 128]
            sXstate: cute.struct.Align[cute.struct.MemRange[Float32, self.tile_D], 128]
            sState: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sState_layout)],
                self.buffer_align_bytes,
            ]
            sBstate: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.sBC_layout)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.sBC_layout)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.sBC_layout)],
                self.buffer_align_bytes,
            ]
            sXproj: cute.struct.Align[
                cute.struct.MemRange[self.proj_dtype, cute.cosize(self.sProj_layout)],
                self.buffer_align_bytes,
            ]
            sZproj: cute.struct.Align[
                cute.struct.MemRange[self.proj_dtype, sZproj_size],
                self.buffer_align_bytes,
            ]
            sOutproj: cute.struct.Align[
                cute.struct.MemRange[self.proj_dtype, sOutproj_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            mState,
            mBstate,
            mXstate,
            mA,
            mB,
            mC,
            mD,
            mX,
            mDt,
            mTrap,
            mXproj,
            mOutproj,
            mStateOut,
            mOut,
            mZ,
            mZproj,
            self.sState_layout,
            self.sBC_layout,
            self.sProj_layout,
            self.gmem_tiled_copy_state,
            self.gmem_tiled_copy_BC,
            self.gmem_tiled_copy_Proj,
            self.gmem_tiled_copy_X,
            tiled_copy_state_s2r,
            tiled_copy_B_s2r,
            vecsize_dstate,
        ).launch(
            # grid: (d, h, b)
            grid=[cute.ceil_div(mState.shape[2], self.tile_D), mState.shape[1], mState.shape[0]],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mState: cute.Tensor,  # (B, H, D, N)
        mBstate: cute.Tensor,  # (B, R, H, N)
        mXstate: cute.Tensor,  # (B, H, D)
        mA: cute.Tensor,  # (B, H)
        mB: cute.Tensor,  # (B, R, H, N)
        mC: cute.Tensor,  # (B, R, H, N)
        mD: cute.Tensor,  # (H)
        mX: cute.Tensor,  # (B, H, D)
        mDt: cute.Tensor,  # (B, H)
        mTrap: cute.Tensor,  # (B, H)
        mXproj: cute.Tensor,  # (R, H, D)
        mOutproj: Optional[cute.Tensor],  # (R, H, D), None if remove_outproj
        mStateOut: cute.Tensor,  # (B, H, D, N)
        mOut: cute.Tensor,  # (B, H, D) or (B, R, H, D) if remove_outproj
        mZ: Optional[cute.Tensor],  # (B, H, D), None if remove_gate
        mZproj: Optional[cute.Tensor],  # (R, H, D), None if remove_gate
        sState_layout: cute.Layout | cute.ComposedLayout,
        sBC_layout: cute.Layout | cute.ComposedLayout,
        sProj_layout: cute.Layout | cute.ComposedLayout,
        gmem_tiled_copy_state: cute.TiledCopy,
        gmem_tiled_copy_BC: cute.TiledCopy,
        gmem_tiled_copy_Proj: cute.TiledCopy,
        gmem_tiled_copy_X: cute.TiledCopy,
        tiled_copy_state_s2r: cute.TiledCopy,
        tiled_copy_B_s2r: cute.TiledCopy,
        vecsize_dstate: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidd, bidh, bidb = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        limit_d = mState.shape[2]

        # ///////////////////////////////////////////////////////////////////////////////
        #  Slice for CTA
        # ///////////////////////////////////////////////////////////////////////////////
        # (tile_D, N)
        gState, gStateOut = [
            cute.local_tile(t[bidb, bidh, None, None], (self.tile_D, self.dstate), (bidd, 0))
            for t in (mState, mStateOut)
        ]
        # (R, N)
        gBstate, gB, gC = [
            cute.local_tile(t[bidb, None, bidh, None], (self.mimo, self.dstate), (0, 0))
            for t in (mBstate, mB, mC)
        ]
        # (tile_D,)
        gXstate, gX = [
            cute.local_tile(t[bidb, bidh, None], (self.tile_D,), (bidd,))
            for t in (mXstate, mX)
        ]
        if const_expr(mOutproj is not None):
            # Output is (B, H, D), outproj reduces MIMO rank
            gOut = cute.local_tile(mOut[bidb, bidh, None], (self.tile_D,), (bidd,))
            gXproj = cute.local_tile(mXproj[None, bidh, None], (self.mimo, self.tile_D), (0, bidd))
            gOutproj = cute.local_tile(mOutproj[None, bidh, None], (self.mimo, self.tile_D), (0, bidd))
        else:
            # Output is (B, R, H, D), no outproj reduction
            gXproj = cute.local_tile(mXproj[None, bidh, None], (self.mimo, self.tile_D), (0, bidd))
            gOutproj = None
        if const_expr(mZ is not None):
            gZ = cute.local_tile(mZ[bidb, bidh, None], (self.tile_D,), (bidd,))
            gZproj = cute.local_tile(mZproj[None, bidh, None], (self.mimo, self.tile_D), (0, bidd))

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensors
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sState = storage.sState.get_tensor(sState_layout)
        sBstate = storage.sBstate.get_tensor(sBC_layout)
        sB = storage.sB.get_tensor(sBC_layout)
        sC = storage.sC.get_tensor(sBC_layout)
        sXproj = storage.sXproj.get_tensor(sProj_layout)
        sZproj = storage.sZproj.get_tensor(sProj_layout) if const_expr(mZ is not None) else None
        sOutproj = storage.sOutproj.get_tensor(sProj_layout) if const_expr(mOutproj is not None) else None
        sXstate = storage.sXstate.get_tensor(cute.make_layout(self.tile_D))
        sX = storage.sX.get_tensor(cute.make_layout(self.tile_D))
        sXgamma = storage.sXgamma.get_tensor(cute.make_layout(self.tile_D))

        # ///////////////////////////////////////////////////////////////////////////////
        #  Partitioning using copy atoms
        # ///////////////////////////////////////////////////////////////////////////////
        gmem_thr_copy_state = gmem_tiled_copy_state.get_slice(tidx)
        # copying states from r2g uses the same tiled copy as s2r
        gmem_thr_copy_StateOut = tiled_copy_state_s2r.get_slice(tidx)
        gmem_thr_copy_BC = gmem_tiled_copy_BC.get_slice(tidx)
        gmem_thr_copy_Proj = gmem_tiled_copy_Proj.get_slice(tidx)
        gmem_thr_copy_X = gmem_tiled_copy_X.get_slice(lane_idx)  # Only 1 warp loads X, Z

        tSgS = gmem_thr_copy_state.partition_S(gState)
        tSsS_g2s = gmem_thr_copy_state.partition_D(sState)
        tSgSOut = gmem_thr_copy_StateOut.partition_D(gStateOut)
        tBCgBstate = gmem_thr_copy_BC.partition_S(gBstate)
        tBCsBstate = gmem_thr_copy_BC.partition_D(sBstate)
        tBCgB = gmem_thr_copy_BC.partition_S(gB)
        tBCsB = gmem_thr_copy_BC.partition_D(sB)
        tBCgC = gmem_thr_copy_BC.partition_S(gC)
        tBCsC = gmem_thr_copy_BC.partition_D(sC)
        tPgXproj = gmem_thr_copy_Proj.partition_S(gXproj)
        tPsXproj = gmem_thr_copy_Proj.partition_D(sXproj)
        if const_expr(mZ is not None):
            tPgZproj = gmem_thr_copy_Proj.partition_S(gZproj)
            tPsZproj = gmem_thr_copy_Proj.partition_D(sZproj)
        if const_expr(mOutproj is not None):
            tPgOutproj = gmem_thr_copy_Proj.partition_S(gOutproj)
            tPsOutproj = gmem_thr_copy_Proj.partition_D(sOutproj)
        tXgX = gmem_thr_copy_X.partition_S(gX)
        tXsX = gmem_thr_copy_X.partition_D(sX)
        tXsXgamma = gmem_thr_copy_X.partition_D(sXgamma)
        tXgXstate = gmem_thr_copy_X.partition_S(gXstate)
        tXsXstate = gmem_thr_copy_X.partition_D(sXstate)

        # Idk why this order of threads_per_dstate and num_groups are reversed
        threads_per_dstate, num_groups = tiled_copy_state_s2r.layout_tv_tiled[0].shape
        lanes_per_D = self.tile_D // num_groups

        # For bound checking
        cS = cute.make_identity_tensor((self.tile_D, self.dstate))
        tScS = gmem_thr_copy_state.partition_S(cS)
        cBC = cute.make_identity_tensor((self.mimo, self.dstate))
        tBCcBC = gmem_thr_copy_BC.partition_S(cBC)
        cProj = cute.make_identity_tensor((self.mimo, self.tile_D))
        tPcProj = gmem_thr_copy_Proj.partition_S(cProj)

        A_val = Float32(mA[bidb, bidh])
        dt_val = Float32(mDt[bidb, bidh])
        trap_val = Float32(mTrap[bidb, bidh])

        # Load X and Xstate, these are small so we want to kick them off first
        tXrX = cute.make_rmem_tensor_like(tXgX)
        tXrXstate = cute.make_rmem_tensor_like(tXgXstate)
        copy_elems_x = cute.size(tXgX.shape[0][0])
        assert cute.size(tXgX.shape) == copy_elems_x  # Only 1 load instruction
        num_loads_X = const_expr(self.tile_D // copy_elems_x)
        need_bound_check_X = const_expr(cute.arch.WARP_SIZE > num_loads_X)
        if warp_idx == 0:
            if not need_bound_check_X or lane_idx < num_loads_X:
                cute.copy(gmem_tiled_copy_X, tXgX, tXrX)
        if warp_idx == 1:
            if not need_bound_check_X or lane_idx < num_loads_X:
                cute.copy(gmem_tiled_copy_X, tXgXstate, tXrXstate)

        # Load Bstate, B, Xproj need bound checking
        for m in cutlass.range(cute.size(tBCcBC.shape[1]), unroll_full=True):
            if tBCcBC[0, m, 0][0] < self.mimo:
                cute.copy(gmem_tiled_copy_BC, tBCgBstate[None, m, None], tBCsBstate[None, m, None])
                cute.copy(gmem_tiled_copy_BC, tBCgB[None, m, None], tBCsB[None, m, None])
        for m in cutlass.range(cute.size(tPcProj.shape[1]), unroll_full=True):
            if tPcProj[0, m, 0][0] < self.mimo:
                cute.copy(gmem_tiled_copy_Proj, tPgXproj[None, m, None], tPsXproj[None, m, None])
        cute.arch.cp_async_commit_group()

        # Load State, not doing any bound check for now
        cute.copy(gmem_tiled_copy_state, tSgS, tSsS_g2s)
        cute.arch.cp_async_commit_group()

        alpha_val = cute.math.exp(A_val * dt_val, fastmath=True)
        # Transform X and Xstate by multiplying with gamma and beta, then write to smem
        if warp_idx == 0:
            tXrX_f32 = cute.make_rmem_tensor_like(tXrX, Float32)
            tXrX_f32.store(tXrX.load().to(Float32))
            if not need_bound_check_X or lane_idx < num_loads_X:
                cute.autovec_copy(tXrX_f32, tXsX)
            gamma_val = trap_val * dt_val
            tXrX_f32.store(tXrX_f32.load() * gamma_val)
            if not need_bound_check_X or lane_idx < num_loads_X:
                cute.autovec_copy(tXrX_f32, tXsXgamma)
        if warp_idx == 1:
            beta_val = (1.0 - trap_val) * dt_val * alpha_val
            tXrXstate_f32 = cute.make_rmem_tensor_like(tXgXstate, Float32)
            tXrXstate_f32.store(tXrXstate.load().to(Float32) * beta_val)
            if not need_bound_check_X or lane_idx < num_loads_X:
                cute.autovec_copy(tXrXstate_f32, tXsXstate)

        # Load C, need bound checking
        for m in cutlass.range(cute.size(tBCcBC.shape[1]), unroll_full=True):
            if tBCcBC[0, m, 0][0] < self.mimo:
                cute.copy(gmem_tiled_copy_BC, tBCgC[None, m, None], tBCsC[None, m, None])
        cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(2)  # B, Bstate, Xproj are done loading
        cute.arch.sync_threads()
        # Load B, Bstate, Xproj from smem
        smem_thr_copy_B = tiled_copy_B_s2r.get_slice(tidx % threads_per_dstate)
        # ((vecsize_dstate, 1), mimo, 1) -> ((vecsize_dstate, 1), mimo)
        tSsB = smem_thr_copy_B.partition_S(sB)[None, None, 0]
        tSsBstate = smem_thr_copy_B.partition_S(sBstate)[None, None, 0]
        tSrB = cute.make_rmem_tensor_like(tSsB)
        tSrBstate = cute.make_rmem_tensor_like(tSsBstate)
        cute.autovec_copy(tSsB, tSrB)
        cute.autovec_copy(tSsBstate, tSrBstate)
        tSrB_f32 = cute.make_rmem_tensor_like(tSrB, Float32)
        tSrB_f32.store(tSrB.load().to(Float32))
        tSrBstate_f32 = cute.make_rmem_tensor_like(tSrBstate, Float32)
        tSrBstate_f32.store(tSrBstate.load().to(Float32))
        # Loading x and xstate, at most 1 val per thread
        x_val = Float32(0.0)
        if lane_idx < lanes_per_D:
            # TODO: should this be warp_idx or group_idx?
            x_val = sXgamma[warp_idx * lanes_per_D + lane_idx]
        x_state_val = Float32(0.0)
        if lane_idx < lanes_per_D:
            x_state_val = sXstate[warp_idx * lanes_per_D + lane_idx]

        new_state = cute.make_rmem_tensor((vecsize_dstate, lanes_per_D), Float32)
        for r in cutlass.range_constexpr(self.mimo):
            x_proj_val = Float32(0.0)
            if lane_idx < lanes_per_D:
                x_proj_val = Float32(sXproj[r, warp_idx * lanes_per_D + lane_idx])
            x_gamma_x_proj_val = x_val * x_proj_val
            x_state_x_proj_val = x_state_val * x_proj_val
            for d in cutlass.range(lanes_per_D, unroll_full=True):
                xg = cute.arch.shuffle_sync(x_gamma_x_proj_val, d)
                xb = cute.arch.shuffle_sync(x_state_x_proj_val, d)
                for v in cutlass.range(vecsize_dstate, unroll_full=True):
                    if const_expr(r == 0):
                        new_state[v, d] = xg * tSrB_f32[v, r]
                    else:
                        new_state[v, d] += xg * tSrB_f32[v, r]
                    new_state[v, d] += xb * tSrBstate_f32[v, r]

        cute.arch.cp_async_wait_group(1)  # state is done loading
        cute.arch.sync_threads()
        thr_copy_state_s2r = tiled_copy_state_s2r.get_slice(tidx)
        # ((vecsize_state, lanes_per_D), 1, 1)
        tSsS = thr_copy_state_s2r.partition_S(sState)
        tSrS = cute.make_rmem_tensor_like(tSsS)
        cute.autovec_copy(tSsS, tSrS)

        # ((vecsize_state, lanes_per_D), 1, 1)
        # tSrS_f32 = cute.make_rmem_tensor_like(tSrS, Float32)
        tSrS_f32 = cute.make_rmem_tensor(((vecsize_dstate, 1), lanes_per_D, 1), Float32)
        assert cute.size(tSrS.shape) == cute.size(tSrS_f32.shape)
        tSrS_f32.store(tSrS.load().to(Float32))
        for v in cutlass.range(cute.size(tSrS_f32), unroll_full=True):
            tSrS_f32[v] = tSrS_f32[v] * alpha_val + new_state[v]
        tSrS.store(tSrS_f32.load().to(self.dtype))

        # Load Z from gmem -> rmem, it's small, at most 1 val per thread
        if const_expr(mZ is not None):
            z_val = Float32(0.0)
            if lane_idx < lanes_per_D:
                z_val = Float32(gZ[warp_idx * lanes_per_D + lane_idx])
        # Load Zproj and Outproj, need bound checking
        for m in cutlass.range(cute.size(tPcProj.shape[1]), unroll_full=True):
            if tPcProj[0, m, 0][0] < self.mimo:
                if const_expr(mZ is not None):
                    cute.copy(gmem_tiled_copy_Proj, tPgZproj[None, m, None], tPsZproj[None, m, None])
                if const_expr(mOutproj is not None):
                    cute.copy(gmem_tiled_copy_Proj, tPgOutproj[None, m, None], tPsOutproj[None, m, None])
        cute.arch.cp_async_commit_group()

        # Write state back to StateOut (may be same memory as State for in-place)
        cute.copy(tiled_copy_state_s2r, tSrS, tSgSOut)

        # Do state @ C
        cute.arch.cp_async_wait_group(1)  # C is done loading
        cute.arch.sync_threads()
        # ((vecsize_dstate, 1), mimo, 1) -> ((vecsize_dstate, 1), 1, mimo)
        tSsC = select(smem_thr_copy_B.partition_S(sC), mode=[0, 2, 1])
        tSrC = cute.make_rmem_tensor_like(tSsC)
        cute.autovec_copy(tSsC, tSrC)
        tSrC_f32 = cute.make_rmem_tensor_like(tSrC, Float32)
        tSrC_f32.store(tSrC.load().to(Float32))
        out_expanded = cute.make_rmem_tensor((lanes_per_D, self.mimo), Float32)
        # tSrS_f32 has shape ((vecsize_dstate, 1), lanes_per_D, 1)
        # tSrC has shape ((vecsize_dstate, 1), mimo)
        out_expanded.store(
            (tSrS_f32.load() * tSrC_f32.load()).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, None))
        )
        assert lanes_per_D <= threads_per_dstate
        for d in cutlass.range(lanes_per_D, unroll_full=True):
            for r in cutlass.range(self.mimo, unroll_full=True):
                out_expanded[d, r] += cute.arch.shuffle_sync_bfly(out_expanded[d, r], offset=16)
        for i in cutlass.range_constexpr(int(math.log2(lanes_per_D))):
            step = 1 << (int(math.log2(lanes_per_D)) - 1 - i)
            should_swap = not Boolean(lane_idx & step)
            for j in cutlass.range_constexpr(step):
                for r in cutlass.range(self.mimo, unroll_full=True):
                    lower, upper = out_expanded[j, r], out_expanded[j + step, r]
                    out_expanded[j, r] = upper if should_swap else lower
                    out_expanded[j + step, r] = lower if should_swap else upper
                    shfl_val = cute.arch.shuffle_sync_bfly(out_expanded[j, r], offset=step)
                    out_expanded[j, r] = shfl_val + out_expanded[j + step, r]
        # After this, the out values are just out_expanded[0, None]
        out = out_expanded[0, None]  # (mimo,)

        # Add D * x * x_proj to out
        D_val = Float32(mD[bidh])
        x_val = Float32(0.0)
        if lane_idx < lanes_per_D:
            x_val = sX[warp_idx * lanes_per_D + lane_idx]
        for r in cutlass.range_constexpr(self.mimo):
            x_proj_val = Float32(0.0)
            if lane_idx < lanes_per_D:
                x_proj_val = Float32(sXproj[r, warp_idx * lanes_per_D + lane_idx])
            out[r] += D_val * x_val * x_proj_val

        cute.arch.cp_async_wait_group(0)  # Zproj and Outproj are done loading
        cute.arch.sync_threads()

        if const_expr(mOutproj is not None):
            # Gate: z_r * sigmoid(z_r)
            if const_expr(mZ is not None):
                for r in cutlass.range_constexpr(self.mimo):
                    z_proj_val = Float32(0.0)
                    if lane_idx < lanes_per_D:
                        z_proj_val = Float32(sZproj[r, warp_idx * lanes_per_D + lane_idx])
                    z_r_half = 0.5 * (z_val * z_proj_val)
                    z_r_silu = z_r_half * cute.math.tanh(z_r_half, fastmath=True) + z_r_half
                    out[r] *= z_r_silu

            # Final projection along mimo dim
            out_val = Float32(0.0)
            for r in cutlass.range_constexpr(self.mimo):
                out_proj_val = Float32(0.0)
                if lane_idx < lanes_per_D:
                    out_proj_val = Float32(sOutproj[r, warp_idx * lanes_per_D + lane_idx])
                if const_expr(r == 0):
                    out_val = out[r] * out_proj_val
                else:
                    out_val += out[r] * out_proj_val

            # Write final output (B, H, D)
            if lane_idx < lanes_per_D:
                gOut[warp_idx * lanes_per_D + lane_idx] = out_val.to(mOut.element_type)
        else:
            # No outproj: write per-rank output (B, R, H, D)
            for r in cutlass.range_constexpr(self.mimo):
                gOut_r = cute.local_tile(mOut[bidb, r, bidh, None], (self.tile_D,), (bidd,))
                if lane_idx < lanes_per_D:
                    gOut_r[warp_idx * lanes_per_D + lane_idx] = out[r].to(mOut.element_type)


def mamba3_step_fn(
    # B: batch size, H: num heads, D: head dim, N: dstate, R: mimo
    state: Tensor,  # (B, H, D, N) — updated in place if state_out is None
    Bstate: Tensor,  # (B, R, H, N)
    Xstate: Tensor,  # (B, H, D)
    A: Tensor,  # (B, H)
    B: Tensor,  # (B, R, H, N)
    C: Tensor,  # (B, R, H, N)
    D: Tensor,  # (H)
    x: Tensor,  # (B, H, D)
    dt: Tensor,  # (B, H)
    trap: Tensor,  # (B, H)
    xproj: Tensor,  # (R, H, D)
    outproj: Optional[Tensor] = None,  # (R, H, D), None if remove_outproj
    state_out: Optional[Tensor] = None,  # (B, H, D, N), None for in-place update
    out: Tensor = None,  # (B, H, D) or (B, R, H, D) if remove_outproj
    z: Optional[Tensor] = None,  # (B, H, D), None if remove_gate
    zproj: Optional[Tensor] = None,  # (R, H, D), None if remove_gate
    tile_D: int = 64,
    num_warps: int = 2,
) -> None:
    has_z = z is not None
    has_outproj = outproj is not None
    inplace = state_out is None
    batch, nheads, hdim, dstate = state.shape
    mimo = Bstate.shape[1]
    assert state.shape == (batch, nheads, hdim, dstate)
    assert Bstate.shape == (batch, mimo, nheads, dstate)
    assert Xstate.shape == (batch, nheads, hdim)
    assert A.shape == (batch, nheads)
    assert B.shape == (batch, mimo, nheads, dstate)
    assert C.shape == (batch, mimo, nheads, dstate)
    assert D.shape == (nheads,)
    assert x.shape == (batch, nheads, hdim)
    if has_z:
        assert z.shape == (batch, nheads, hdim)
        assert zproj is not None
        assert zproj.shape == (mimo, nheads, hdim)
    assert dt.shape == (batch, nheads)
    assert trap.shape == (batch, nheads)
    assert xproj.shape == (mimo, nheads, hdim)
    xproj = xproj.contiguous()
    if has_outproj:
        assert outproj.shape == (mimo, nheads, hdim)
        assert out.shape == (batch, nheads, hdim)
    else:
        assert out.shape == (batch, mimo, nheads, hdim)

    # Use state itself as output target when in-place
    if inplace:
        state_out = state
    else:
        assert state_out.shape == (batch, nheads, hdim, dstate)

    required_tensors = [state, Bstate, Xstate, A, B, C, D, x, dt, trap, xproj, state_out, out]
    if has_outproj:
        required_tensors.append(outproj)
    if has_z:
        required_tensors.extend([z, zproj])
    assert all(t.is_cuda for t in required_tensors)
    assert state.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"

    # Map torch dtypes to cutlass dtypes
    state_cute_dtype = torch2cute_dtype_map[state.dtype]
    b_cute_dtype = torch2cute_dtype_map[Bstate.dtype]
    x_cute_dtype = torch2cute_dtype_map[x.dtype]
    proj_cute_dtype = torch2cute_dtype_map[xproj.dtype]
    a_cute_dtype = torch2cute_dtype_map[A.dtype]
    d_cute_dtype = torch2cute_dtype_map[D.dtype]
    dt_cute_dtype = torch2cute_dtype_map[dt.dtype]
    trap_cute_dtype = torch2cute_dtype_map[trap.dtype]

    compile_key = (
        tile_D,
        num_warps,
        dstate,
        hdim,
        mimo,
        state.dtype,
        Bstate.dtype,
        xproj.dtype,
        A.dtype,
        D.dtype,
        dt.dtype,
        trap.dtype,
        has_z,
        has_outproj,
    )
    if compile_key not in mamba3_step_fn.compile_cache:
        mamba3_step_op = Mamba3Step(tile_D, dstate, mimo, num_warps, remove_gate=not has_z, remove_outproj=not has_outproj)

        # Create symbolic dimensions for batch and nheads
        batch_sym = cute.sym_int()
        nheads_sym = cute.sym_int()

        # Divisibility for strides (128-bit alignment)
        div_state = 128 // state_cute_dtype.width
        div_b = 128 // b_cute_dtype.width
        div_x = 128 // x_cute_dtype.width
        div_proj = 128 // proj_cute_dtype.width
        div_a = 128 // a_cute_dtype.width
        div_d = 128 // d_cute_dtype.width
        div_dt = 128 // dt_cute_dtype.width
        div_trap = 128 // trap_cute_dtype.width

        # Create fake tensors with symbolic batch/nheads dimensions
        state_fake = make_fake_tensor(state_cute_dtype, (batch_sym, nheads_sym, hdim, dstate), div_state)
        Bstate_fake = make_fake_tensor(b_cute_dtype, (batch_sym, mimo, nheads_sym, dstate), div_b)
        Xstate_fake = make_fake_tensor(x_cute_dtype, (batch_sym, nheads_sym, hdim), div_x)
        A_fake = make_fake_tensor(a_cute_dtype, (batch_sym, nheads_sym), div_a)
        B_fake = make_fake_tensor(b_cute_dtype, (batch_sym, mimo, nheads_sym, dstate), div_b)
        C_fake = make_fake_tensor(b_cute_dtype, (batch_sym, mimo, nheads_sym, dstate), div_b)
        D_fake = make_fake_tensor(d_cute_dtype, (nheads_sym,), div_d)
        x_fake = make_fake_tensor(x_cute_dtype, (batch_sym, nheads_sym, hdim), div_x)
        dt_fake = make_fake_tensor(dt_cute_dtype, (batch_sym, nheads_sym), div_dt)
        trap_fake = make_fake_tensor(trap_cute_dtype, (batch_sym, nheads_sym), div_trap)
        xproj_fake = make_fake_tensor(proj_cute_dtype, (mimo, nheads_sym, hdim), div_proj)
        outproj_fake = make_fake_tensor(proj_cute_dtype, (mimo, nheads_sym, hdim), div_proj) if has_outproj else None
        state_out_fake = make_fake_tensor(state_cute_dtype, (batch_sym, nheads_sym, hdim, dstate), div_state)
        if has_outproj:
            out_fake = make_fake_tensor(x_cute_dtype, (batch_sym, nheads_sym, hdim), div_x)
        else:
            out_fake = make_fake_tensor(x_cute_dtype, (batch_sym, mimo, nheads_sym, hdim), div_x)
        z_fake = make_fake_tensor(x_cute_dtype, (batch_sym, nheads_sym, hdim), div_x) if has_z else None
        zproj_fake = make_fake_tensor(proj_cute_dtype, (mimo, nheads_sym, hdim), div_proj) if has_z else None

        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        mamba3_step_fn.compile_cache[compile_key] = cute.compile(
            mamba3_step_op,
            state_fake,
            Bstate_fake,
            Xstate_fake,
            A_fake,
            B_fake,
            C_fake,
            D_fake,
            x_fake,
            dt_fake,
            trap_fake,
            xproj_fake,
            outproj_fake,
            state_out_fake,
            out_fake,
            z_fake,
            zproj_fake,
            fake_stream,
            options="--enable-tvm-ffi",
        )

    # Call with real PyTorch tensors directly (no dlpack conversion needed)
    # When inplace, state_out is state (set above)
    mamba3_step_fn.compile_cache[compile_key](
        state,
        Bstate,
        Xstate,
        A,
        B,
        C,
        D,
        x,
        dt,
        trap,
        xproj,
        outproj,
        state_out,
        out,
        z,
        zproj,
    )


mamba3_step_fn.compile_cache = {}


def selective_state_update_fused_ref_v2(
    state, A, B, C, xproj, x, zproj, z, dt, B_state, x_state, trap, D, outproj,
    compute_dtype=torch.float32
):
    """
    Reference to match the new fused kernel API.

    Shapes:
        state:   (B, N, H, S)
        A:       (B, N)
        B:       (B, R, N, S)
        C:       (B, R, N, S)
        xproj:   (R, N, H)
        x:       (B, N, H)
        zproj:   (R, N, H)
        z:       (B, N, H)
        dt:      (B, N)
        B_state: (B, R, N, S)
        x_state: (B, N, H)
        trap:    (B, N)
        D:       (N,)
        outproj: (R, N, H)

    Returns:
        out:       (B, N, H)
        new_state: (B, N, H, S)
    """
    Bsz, N, H, S = state.shape
    R = B.shape[1]

    # Dtypes for numerics (match kernel's fp32 accum)
    og_dtype = state.dtype
    A_f    = A.to(compute_dtype)       # (B, N)
    dt_f   = dt.to(compute_dtype)      # (B, N)
    trap_f = trap.to(compute_dtype)    # (B, N)
    D_f    = D.to(compute_dtype)       # (N,)
    x_f    = x.to(compute_dtype)       # (B, N, H)
    xst_f  = x_state.to(compute_dtype) # (B, N, H)
    B_f    = B.to(compute_dtype)       # (B, R, N, S)
    C_f    = C.to(compute_dtype)       # (B, R, N, S)
    Bst_f  = B_state.to(compute_dtype) # (B, R, N, S)
    Xp_f   = xproj.to(compute_dtype)   # (R, N, H)
    st_f   = state.to(compute_dtype)   # (B, N, H, S)

    alpha = torch.exp(A_f * dt_f)                 # (B, N)
    beta  = (1.0 - trap_f) * dt_f * alpha        # (B, N)
    gamma = trap_f * dt_f                         # (B, N)

    x_vals   = (x_f[:, None, :, :] * Xp_f[None, :, :, :])    # (B, R, N, H)
    xs_vals  = (xst_f[:, None, :, :] * Xp_f[None, :, :, :])  # (B, R, N, H)

    xBt_state = torch.einsum('brnh,brns->bnhs', x_vals * gamma.unsqueeze(-1).unsqueeze(1),  B_f)
    xBt_prev  = torch.einsum('brnh,brns->bnhs', xs_vals * beta.unsqueeze(-1).unsqueeze(1), Bst_f)

    new_state = st_f * alpha[:, :, None, None] + xBt_state + xBt_prev   # (B, N, H, S)

    out_r = torch.einsum('bnhs,brns->brnh', new_state, C_f)  # (B, R, N, H)

    out_r = out_r + (x_vals * D_f[None, :, None])            # (B, R, N, H)

    if z is not None:
        z_f    = z.to(compute_dtype)       # (B, N, H)
        Zp_f   = zproj.to(compute_dtype)   # (R, N, H)
        z_vals = (z_f[:, None, :, :] * Zp_f[None, :, :, :])      # (B, R, N, H)
        out_r  = out_r * z_vals * torch.sigmoid(z_vals)          # (B, R, N, H)

    if outproj is not None:
        Op_f   = outproj.to(compute_dtype) # (R, N, H)
        out = torch.einsum('brnh,rnh->bnh', out_r, Op_f)         # (B, N, H)
    else:
        out = out_r                                               # (B, R, N, H)

    return out.to(og_dtype), new_state.to(og_dtype)


def _bytes_of(t):
    return t.numel() * t.element_size()


if __name__ == "__main__":
    torch.manual_seed(1357)
    batch, nheads, hdim, dstate, mimo = 128, 64, 64, 128, 4
    device = torch.device("cuda:0")
    dtype_state = torch.float32
    dtype = torch.float32
    state = torch.randn(batch, nheads, hdim, dstate, device=device, dtype=dtype_state)
    Bstate = torch.randn(batch, mimo, nheads, dstate, device=device, dtype=dtype)
    Xstate = torch.randn(batch, nheads, hdim, device=device, dtype=dtype)
    A = -F.softplus(torch.randn(batch, nheads, device=device, dtype=torch.float32))
    B = torch.randn(batch, mimo, nheads, dstate, device=device, dtype=dtype)
    C = torch.randn(batch, mimo, nheads, dstate, device=device, dtype=dtype)
    D = torch.randn(nheads, device=device, dtype=dtype)
    x = torch.randn(batch, nheads, hdim, device=device, dtype=dtype)
    z = torch.randn(batch, nheads, hdim, device=device, dtype=dtype)
    dt = torch.exp(torch.rand(nheads, device=device) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    dt = F.softplus(torch.randn(batch, nheads, device=device) + dt_bias)  # (B, H)
    trap = torch.sigmoid(torch.randn(batch, nheads, device=device, dtype=torch.float32))
    xproj = torch.randn(mimo, nheads, hdim, device=device, dtype=dtype)
    zproj = torch.randn(mimo, nheads, hdim, device=device, dtype=dtype)
    outproj = torch.randn(mimo, nheads, hdim, device=device, dtype=dtype)
    out = torch.zeros_like(x)

    # =========================================================================
    # Test 1: Out-of-place (explicit state_out)
    # =========================================================================
    print("=== Out-of-place test ===")
    state_out = torch.zeros_like(state)
    fn_oop = lambda: mamba3_step_fn(
        state,
        Bstate,
        Xstate,
        A,
        B,
        C,
        D,
        x,
        dt,
        trap,
        xproj,
        outproj,
        state_out,
        out,
        z=z,
        zproj=zproj,
        tile_D=64,
        num_warps=4,
    )

    fn_oop()
    out_ref, state_out_ref = selective_state_update_fused_ref_v2(state, A, B, C, xproj, x, zproj, z, dt, Bstate, Xstate, trap, D, outproj, compute_dtype=torch.float64)
    out_pt, state_out_pt = selective_state_update_fused_ref_v2(state, A, B, C, xproj, x, zproj, z, dt, Bstate, Xstate, trap, D, outproj, compute_dtype=torch.float32)
    print(f"state_out vs ref (f64): {(state_out - state_out_ref).abs().max()}")
    print(f"state_out_pt vs ref (f64): {(state_out_pt - state_out_ref).abs().max()}")
    print(f"out vs ref (f64): {(out - out_ref).abs().max()}")
    print(f"out_pt vs ref (f64): {(out_pt - out_ref).abs().max()}")

    # =========================================================================
    # Test 2: In-place (state_out=None)
    # =========================================================================
    print("\n=== In-place test ===")
    # Fresh state for in-place test
    state_ip = state.clone()
    out_ip = torch.zeros_like(x)
    fn_ip = lambda: mamba3_step_fn(
        state_ip,
        Bstate,
        Xstate,
        A,
        B,
        C,
        D,
        x,
        dt,
        trap,
        xproj,
        outproj,
        None,  # state_out=None -> in-place
        out_ip,
        z=z,
        zproj=zproj,
        tile_D=64,
        num_warps=4,
    )

    fn_ip()
    # state_ip was updated in place, compare against same reference
    print(f"state (in-place) vs ref (f64): {(state_ip - state_out_ref).abs().max()}")
    print(f"out (in-place) vs ref (f64): {(out_ip - out_ref).abs().max()}")
    # Verify in-place and out-of-place produce identical results
    print(f"state in-place vs out-of-place: {(state_ip - state_out).abs().max()}")
    print(f"out in-place vs out-of-place: {(out_ip - out).abs().max()}")

    # =========================================================================
    # Benchmark (out-of-place)
    # =========================================================================
    read_bytes = (
        _bytes_of(state) + _bytes_of(A) + _bytes_of(B)
        + _bytes_of(C)
        + _bytes_of(xproj) + _bytes_of(x)
        + _bytes_of(zproj) + _bytes_of(z)
        + _bytes_of(dt) + _bytes_of(Bstate) + _bytes_of(Xstate)
        + _bytes_of(trap) + _bytes_of(D) + _bytes_of(outproj)
    )
    out_bytes       = _bytes_of(out)
    new_state_bytes = _bytes_of(state)
    total_bytes = read_bytes + out_bytes + new_state_bytes

    from triton.testing import do_bench_cudagraph
    ms = do_bench_cudagraph(fn_oop, rep=30)
    bandwidth = (total_bytes) / ms * 1e-6
    print(f"\nMamba3 step (out-of-place): {ms:.3f} ms, {bandwidth:.1f} GB/s")