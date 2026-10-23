#!/bin/bash
# selective_scan (fwd + bwd) AscendC 算子 wheel 构建入口。
#
# 流程：bisheng 编译 kernel + g++ 编译 plugin/extension（cmake 驱动）→ 打包 cann_bench wheel。
# 产物：dist/cann_bench-<version>-cp38-abi3-linux_<arch>.whl
# 用法：
#   bash build.sh                  # 自动探测 SoC（ascend910b / ascend910_93 / ascend950）
#   bash build.sh --soc=ascend910b # 显式指定 SoC
#   bash build.sh --install        # 构建后 pip 安装 wheel（--no-deps 强制重装）
#
# 依赖：CANN toolkit（ASCEND_HOME_PATH 或 /usr/local/Ascend/cann-*）、python3 + torch + torch_npu、
#       cmake ≥3.16、setuptools/wheel。详见 README.md。
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

detect_soc_version() {
    local torch_soc=$(python3 -c "import torch, torch_npu; print(torch.npu.get_device_name(0))" 2>/dev/null)
    if [ -n "${torch_soc}" ]; then
        case "${torch_soc}" in
            Ascend910B*)     echo "ascend910b" ; return ;;
            Ascend910_93*)   echo "ascend910_93" ; return ;;
            Ascend950*)      echo "ascend950" ; return ;;
        esac
    fi
    local npu_name=$(npu-smi info 2>/dev/null | grep -oP 'Ascend\S+' | head -1)
    case "${npu_name}" in
        Ascend910B1|Ascend910B2|Ascend910B3|Ascend910B4) echo "ascend910b" ;;
        Ascend910_93*)  echo "ascend910_93" ;;
        Ascend950*)     echo "ascend950" ;;
        *)              echo "" ;;
    esac
}

SOC_VERSION=""
INSTALL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --soc=*) SOC_VERSION="${1#*=}"; shift ;;
        --install) INSTALL=true; shift ;;
        *) shift ;;
    esac
done

if [ -z "${SOC_VERSION}" ]; then
    SOC_VERSION=$(detect_soc_version)
    if [ -z "${SOC_VERSION}" ]; then
        echo "[ERROR] Cannot detect SoC version. Use --soc=<soc_version>."
        echo "Supported: ascend910b, ascend910_93, ascend950"
        exit 1
    fi
    echo "[INFO] Auto-detected SoC: ${SOC_VERSION}"
fi
export NPU_ARCH="${SOC_VERSION}"

echo "=== Building cann_bench wheel (selective_scan fwd + bwd) ==="
echo "NPU_ARCH: ${NPU_ARCH}"

BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/dist"
rm -rf "${BUILD_DIR}" "${DIST_DIR}"
mkdir -p "${BUILD_DIR}" "${DIST_DIR}"

JOBS=$(nproc)
[ "${JOBS}" -lt 1 ] && JOBS=1

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$(command -v python3)"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

python3 "${SCRIPT_DIR}/setup.py" bdist_wheel --dist-dir "${DIST_DIR}"

if [[ "${INSTALL}" == "true" ]]; then
    echo "=== Installing wheel package ==="
    pip install "${DIST_DIR}"/cann_bench*.whl --force-reinstall --no-deps
fi
echo "=== Build complete ==="
ls -la "${DIST_DIR}"
