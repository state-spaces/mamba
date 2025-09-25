#!/bin/bash

set -eoxu pipefail

# We want setuptools >= 49.6.0 otherwise we can't compile the extension if system CUDA version is 11.7 and pytorch cuda version is 11.6
# https://github.com/pytorch/pytorch/blob/664058fa83f1d8eede5d66418abff6e20bd76ca8/torch/utils/cpp_extension.py#L810
# However this still fails so I am using a newer version of setuptools
pip install setuptools==68.0.0
pip install ninja packaging wheel
export PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/nvidia/lib64:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Limit MAX_JOBS otherwise the github runner goes OOM
export MAX_JOBS=2 
export MAMBA_FORCE_BUILD="TRUE" 
export MAMBA_FORCE_CXX11_ABI=$CXX11_ABI 

# 5h timeout since GH allows max 6h and we want some buffer
EXIT_CODE=0
timeout 5h python setup.py bdist_wheel --dist-dir=dist || EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
MATRIX_TORCH_VERSION=$(echo "$MATRIX_TORCH_VERSION" | tr '+' '_')
tmpname=cu${WHEEL_CUDA_VERSION}torch${MATRIX_TORCH_VERSION}cxx11abi$CXX11_ABI
wheel_name=$(ls dist/*whl | xargs -n 1 basename | sed "s/-/+$tmpname-/2")
ls dist/*whl |xargs -I {} mv {} dist/${wheel_name}
echo "wheel_name=${wheel_name}" >> $GITHUB_ENV
fi

echo $EXIT_CODE