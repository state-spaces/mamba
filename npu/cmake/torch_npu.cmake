# torch_npu.cmake — torch_npu 头文件/库发现
if(NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "torch_npu.cmake requires PYTHON_EXECUTABLE (include python.cmake first)")
endif()

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE _TORCH_NPU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _TORCH_NPU_RES
    ERROR_QUIET
)
if(NOT _TORCH_NPU_RES EQUAL 0)
    message(FATAL_ERROR "import torch_npu failed via ${PYTHON_EXECUTABLE}")
endif()
string(STRIP "${_TORCH_NPU_ROOT}" TORCH_NPU_ROOT)
string(REPLACE "\n" "" TORCH_NPU_ROOT "${TORCH_NPU_ROOT}")
message(STATUS "TORCH_NPU_ROOT: ${TORCH_NPU_ROOT}")

set(TORCH_NPU_INCLUDE_DIR "${TORCH_NPU_ROOT}/include")
if(NOT EXISTS "${TORCH_NPU_INCLUDE_DIR}/torch_npu/csrc/core/npu/NPUStream.h")
    message(FATAL_ERROR "torch_npu include incomplete: ${TORCH_NPU_INCLUDE_DIR}")
endif()

find_library(TORCH_NPU_LIBRARY torch_npu HINTS "${TORCH_NPU_ROOT}/lib" NO_DEFAULT_PATH)
message(STATUS "TORCH_NPU_LIBRARY: ${TORCH_NPU_LIBRARY}")
