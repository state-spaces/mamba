# selective_scan_bwd 算子自注册（迭代 2：反向算子；零改动接入，镜像 fwd 8 行模式）
set(SELECTIVE_SCAN_BWD_KERNEL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/selective_scan_bwd_kernel.cpp)
set(SELECTIVE_SCAN_BWD_PLUGIN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/selective_scan_bwd_plugin.cpp)

register_direct_launch_op(
    "${SELECTIVE_SCAN_BWD_KERNEL_SRCS}" op_kernel
    "${SELECTIVE_SCAN_BWD_PLUGIN_SRCS}" op_kernel
)
