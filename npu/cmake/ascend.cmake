# ascend.cmake — CANN 路径与 bisheng 编译器发现
# 环境信息来源：CANN 9.1.0 安装于 /usr/local/Ascend/cann-9.1.0（环境信息.md 唯一来源）

if(NOT DEFINED ASCEND_HOME_PATH OR ASCEND_HOME_PATH STREQUAL "")
    if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
        # 兜底：扫描 /usr/local/Ascend 下的 cann-* 安装
        file(GLOB _ASCEND_CANDS "/usr/local/Ascend/cann-*" "/usr/local/Ascend/ascend-toolkit/*")
        foreach(_c ${_ASCEND_CANDS})
            if(IS_DIRECTORY "${_c}" AND EXISTS "${_c}/aarch64-linux/bin/bisheng")
                set(ASCEND_HOME_PATH "${_c}")
                break()
            endif()
        endforeach()
    endif()
endif()

if(NOT DEFINED ASCEND_HOME_PATH OR ASCEND_HOME_PATH STREQUAL "")
    message(FATAL_ERROR "CANN toolkit not found: set ASCEND_HOME_PATH or install CANN under /usr/local/Ascend")
endif()
message(STATUS "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")

# bisheng 编译器（kernel 编译用）
find_program(BISHENG_COMPILER
    NAMES bisheng
    HINTS "${ASCEND_HOME_PATH}/aarch64-linux/bin" "${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin"
    NO_DEFAULT_PATH
)
if(NOT BISHENG_COMPILER)
    message(FATAL_ERROR "bisheng compiler not found under ${ASCEND_HOME_PATH}")
endif()
message(STATUS "BISHENG: ${BISHENG_COMPILER}")

# host 侧 CANN 头文件（kernel/launch 声明用）与库
set(ASCEND_CANN_INC "${ASCEND_HOME_PATH}/aarch64-linux/include")
set(ASCENDC_INC "${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include")
set(ASCEND_LIB_DIR "${ASCEND_HOME_PATH}/aarch64-linux/lib64")
set(ASCEND_DEVLIB_DIR "${ASCEND_HOME_PATH}/aarch64-linux/devlib")
if(NOT EXISTS "${ASCEND_LIB_DIR}")
    set(ASCEND_LIB_DIR "${ASCEND_HOME_PATH}/lib64")
endif()

link_directories(${ASCEND_LIB_DIR} ${ASCEND_DEVLIB_DIR})
