# python.cmake — Python3 与头文件发现
set(PYTHON_EXECUTABLE "" CACHE FILEPATH "python3 executable path")

if(NOT PYTHON_EXECUTABLE)
    find_program(PYTHON_EXECUTABLE NAMES python3 python)
endif()
if(NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "python3 not found")
endif()
message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import sysconfig, sys; print(sysconfig.get_paths()['include']); print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
    OUTPUT_VARIABLE _PY_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _PY_RES
)
if(NOT _PY_RES EQUAL 0)
    message(FATAL_ERROR "failed to query python sysconfig via ${PYTHON_EXECUTABLE}")
endif()

string(REPLACE "\n" ";" _PY_LIST "${_PY_INFO}")
list(GET _PY_LIST 0 PYTHON_INCLUDE_DIR)
list(GET _PY_LIST 1 PYTHON_VERSION_STR)
string(STRIP "${PYTHON_INCLUDE_DIR}" PYTHON_INCLUDE_DIR)
string(STRIP "${PYTHON_VERSION_STR}" PYTHON_VERSION_STR)
message(STATUS "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR} (python ${PYTHON_VERSION_STR})")

if(NOT EXISTS "${PYTHON_INCLUDE_DIR}/Python.h")
    message(FATAL_ERROR "Python.h not found under ${PYTHON_INCLUDE_DIR} (install python3-dev)")
endif()

find_library(PYTHON_LIBRARY
    NAMES python${PYTHON_VERSION_STR}
    HINTS "${PYTHON_INCLUDE_DIR}/../lib"
)
message(STATUS "PYTHON_LIBRARY: ${PYTHON_LIBRARY}")
