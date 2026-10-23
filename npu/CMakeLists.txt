cmake_minimum_required(VERSION 3.16)
project(cann_bench_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ascend)
include(python)
include(torch)
include(torch_npu)
include(func)

# ---- 收集算子源文件（csrc/ops/<op> 各自 register_direct_launch_op）----
set(ALL_KERNEL_SRCS "")
set(ALL_PLUGIN_SRCS "")
set(ALL_KERNEL_INCLUDE_DIRS "")
set(ALL_PLUGIN_INCLUDE_DIRS "")
add_subdirectory(csrc/ops)
# ops 级 PARENT_SCOPE 已回传一级；此处兜底再取一次（若 add_subdirectory 直接改写本域变量）
set(ALL_KERNEL_SRCS ${ALL_KERNEL_SRCS})
set(ALL_PLUGIN_SRCS ${ALL_PLUGIN_SRCS})

message(STATUS "ALL_KERNEL_SRCS: ${ALL_KERNEL_SRCS}")
message(STATUS "ALL_PLUGIN_SRCS: ${ALL_PLUGIN_SRCS}")

if(NOT ALL_KERNEL_SRCS)
    message(FATAL_ERROR "no kernel sources registered under csrc/ops/")
endif()

# ---- NPU_ARCH → bisheng --npu-arch 映射 ----
set(NPU_ARCH "ascend910b" CACHE STRING "ascend910b | ascend910_93 | ascend950")
if(NPU_ARCH STREQUAL "ascend910b" OR NPU_ARCH STREQUAL "ascend910_93")
    set(BISHENG_NPU_ARCH "dav-2201")
elseif(NPU_ARCH STREQUAL "ascend950")
    set(BISHENG_NPU_ARCH "dav-3510")
else()
    message(FATAL_ERROR "unsupported NPU_ARCH: ${NPU_ARCH}")
endif()
message(STATUS "BISHENG_NPU_ARCH: ${BISHENG_NPU_ARCH}")

# ---- bisheng 编译 kernel（custom command 直调，CMake 4.x 不支持中途切换编译器）----
set(_SAVED_CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})
# include 目录预展开为 -I 旗标列表（多算子并存时列表含多项，
# 不可放入 $<BOOL:/$<JOIN:> 生成表达式——`;` 会把表达式拆裂为多个参数）
set(_KINC_FLAGS "")
foreach(_KINC ${ALL_KERNEL_INCLUDE_DIRS})
    list(APPEND _KINC_FLAGS "-I${_KINC}")
endforeach()
set(ALL_KERNEL_OBJS "")
foreach(_KSRC ${ALL_KERNEL_SRCS})
    get_filename_component(_KNAME ${_KSRC} NAME_WE)
    set(_KOBJ "${CMAKE_CURRENT_BINARY_DIR}/kernels/${_KNAME}.o")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/kernels")
    add_custom_command(
        OUTPUT ${_KOBJ}
        COMMAND ${BISHENG_COMPILER} --npu-arch=${BISHENG_NPU_ARCH} -xasc -fPIC -O2 -g -c ${_KSRC} -o ${_KOBJ}
                ${_KINC_FLAGS}
                -I${ASCENDC_INC}/basic_api
                -I${ASCENDC_INC}/lib
                -I${ASCENDC_INC}/kernel_pkg
                -I${ASCENDC_INC}
                -I${ASCEND_CANN_INC}
        DEPENDS ${_KSRC}
        COMMENT "bisheng kernel: ${_KNAME}"
        VERBATIM)
    list(APPEND ALL_KERNEL_OBJS ${_KOBJ})
endforeach()
add_custom_target(all_kernels DEPENDS ${ALL_KERNEL_OBJS})

# ---- g++ 编译 plugin ----
set(CMAKE_CXX_COMPILER ${_SAVED_CMAKE_CXX_COMPILER})
add_library(all_plugins_obj OBJECT ${ALL_PLUGIN_SRCS})
target_include_directories(all_plugins_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIR}
    ${ASCEND_CANN_INC}
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/include"
    ${PYTHON_INCLUDE_DIR}
    ${ALL_PLUGIN_INCLUDE_DIRS})
target_compile_options(all_plugins_obj PRIVATE -O3 -fPIC -Wno-deprecated-declarations)

# ---- 合并为 _C.abi3.so，复制到 cann_bench/ ----
add_custom_target(ensure_kernels ALL DEPENDS ${ALL_KERNEL_OBJS})
add_library(_C SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extension.cpp
    ${ALL_KERNEL_OBJS}
    $<TARGET_OBJECTS:all_plugins_obj>)
add_dependencies(_C all_kernels)
set_target_properties(_C PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_C.abi3"
    POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(_C PRIVATE TORCH_EXTENSION_NAME=_C)
target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${ASCENDC_INC}
    ${ASCEND_CANN_INC})
target_link_libraries(_C PRIVATE
    ${TORCH_LIBRARY} ${TORCH_CPU_LIBRARY} ${C10_LIBRARY} ${TORCH_PYTHON_LIBRARY}
    ${TORCH_NPU_LIBRARY}
    # tiling_api 静态库提供 PlatformAscendCManager 静态成员（kernel.o host 侧引用）；
    # platform 提供 fe::PlatFormInfos/PlatformInfoManager；unified_dlog 提供 DlogRecord；
    # ascendc_runtime 静态库提供 ReportAscendProf（bisheng 生成代码的打点桩）
    tiling_api ascendc_runtime platform unified_dlog c_sec mmpa
    ascendcl)

# 产物复制到 cann_bench/ 包目录（wheel 打包读取）
add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:_C>
        ${CMAKE_CURRENT_SOURCE_DIR}/cann_bench/_C.abi3.so)
