# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# CMake 构建配置 - mamba3_decode_step（Ascend C kernel 直调）
#
# 目录结构:
#   op_kernel/     - NPU 计算层（tiling.h + kernel.asc）
#   op_extension/  - PyTorch 接入层（torch.cpp + register.cpp + ops.h）
#
# 产物: build/libmamba3_decode_step_ops.so
# 加载: torch.ops.load_library("build/libmamba3_decode_step_ops.so")
#       torch.ops.npu.mamba3_decode_step(...)
#
# 目标平台: Atlas A2 系列（Ascend910B3, dav-2201）
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(mamba3_decode_step LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

add_library(mamba3_decode_step_ops SHARED
    op_kernel/mamba3_decode_step_kernel.asc
    op_extension/mamba3_decode_step_torch.cpp
    op_extension/mamba3_decode_step_validate.cpp
    op_extension/register.cpp
)

target_include_directories(mamba3_decode_step_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${ACL_INCLUDE_DIR}
)

target_link_libraries(mamba3_decode_step_ops PRIVATE
    torch_npu
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
    ascendcl
    ascendc_runtime
)

target_link_directories(mamba3_decode_step_ops PRIVATE
    ${ACL_LIB_DIR}
    "$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64"
    ${TORCH_NPU_LIBRARIES}
)

target_compile_options(mamba3_decode_step_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# PLUT-215 H0：host 侧优化开关。
#
# 事实（本轮实测，构建目录 `build/CMakeFiles/*/flags.make`）：此前 CMAKE_BUILD_TYPE 为空，
# CMake 不会补任何 -O 开关，CXX 编译选项只有 `-fPIC`——也就是说 pybind 入口、§2.4/§2.5
# 的全部判据、tiling 填充这些**每次调用都跑的 host 代码一直是 -O0 编出来的**。
# 目标情形里 host 侧是唯一的瓶颈（perf-4/5 的 kernel 只有 ~6 µs，wall 却是 60+ µs），
# 这条与算法无关、与 kernel 无关，只是把 host 侧按正常发布形态编出来。
#
# 只对 CXX 生效：ASC（device 侧 kernel 与生成桩）编译选项保持原样、一个字节都不动，
# 因此 kernel 二进制、tiling 语义、精度判据均不受影响。
target_compile_options(mamba3_decode_step_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
