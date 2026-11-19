# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ============================================================================
# Ascend C Kernel 直调构建脚本 - mamba3_decode_step
#
# 用法：
#   bash build.sh                # 配置 + 编译，产物 build/libmamba3_decode_step_ops.so
#   bash build.sh --clean        # 先清空 build/ 再编译
#
# 依赖：已 source CANN set_env.sh（ASCEND_HOME_PATH 可用）、cmake>=3.16、python 带 torch/torch_npu
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ERROR: ASCEND_HOME_PATH 未设置，请先 source CANN 的 set_env.sh" >&2
    exit 1
fi

if [ "${1:-}" = "--clean" ]; then
    rm -rf build
fi

mkdir -p build
cd build
cmake .. -DPython3_EXECUTABLE="$(command -v "${PYTHON_BIN}")"
make -j"$(nproc)"

echo "=== 构建完成：${SCRIPT_DIR}/build/libmamba3_decode_step_ops.so ==="
