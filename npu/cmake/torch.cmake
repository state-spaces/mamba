# torch.cmake — torch 头文件/库发现（经 python import 定位）
if(NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "torch.cmake requires PYTHON_EXECUTABLE (include python.cmake first)")
endif()

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import torch, os; print(os.path.dirname(torch.__file__)); print(torch.__version__)"
    OUTPUT_VARIABLE _TORCH_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _TORCH_RES
)
if(NOT _TORCH_RES EQUAL 0)
    message(FATAL_ERROR "import torch failed via ${PYTHON_EXECUTABLE}")
endif()

string(REPLACE "\n" ";" _TORCH_LIST "${_TORCH_INFO}")
list(GET _TORCH_LIST 0 TORCH_ROOT)
list(GET _TORCH_LIST 1 TORCH_VERSION_STR)
string(STRIP "${TORCH_ROOT}" TORCH_ROOT)
string(STRIP "${TORCH_VERSION_STR}" TORCH_VERSION_STR)
message(STATUS "TORCH_ROOT: ${TORCH_ROOT} (torch ${TORCH_VERSION_STR})")

set(TORCH_INCLUDE_DIRS
    "${TORCH_ROOT}/include"
    "${TORCH_ROOT}/include/torch/csrc/api/include"
)
foreach(_d ${TORCH_INCLUDE_DIRS})
    if(NOT EXISTS "${_d}")
        message(FATAL_ERROR "torch include dir missing: ${_d}")
    endif()
endforeach()

find_library(TORCH_LIBRARY torch HINTS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
find_library(TORCH_CPU_LIBRARY torch_cpu HINTS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
find_library(C10_LIBRARY c10 HINTS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
find_library(TORCH_PYTHON_LIBRARY torch_python HINTS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
message(STATUS "torch libs: ${TORCH_LIBRARY} ${TORCH_CPU_LIBRARY} ${C10_LIBRARY} ${TORCH_PYTHON_LIBRARY}")
