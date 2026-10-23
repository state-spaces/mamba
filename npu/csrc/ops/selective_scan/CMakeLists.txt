# selective_scan 算子自注册
set(SELECTIVE_SCAN_KERNEL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/selective_scan_kernel.cpp)
set(SELECTIVE_SCAN_PLUGIN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/selective_scan_plugin.cpp)

register_direct_launch_op(
    "${SELECTIVE_SCAN_KERNEL_SRCS}" op_kernel
    "${SELECTIVE_SCAN_PLUGIN_SRCS}" op_kernel
)
