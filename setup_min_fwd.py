from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

generator_flag = []

cc_flag = ["-DBUILD_PYTHON_PACKAGE"]

extra_compile_args = {
      "cxx": ["-O3", "-std=c++17"] + generator_flag,
      "nvcc": [
            "-O3",
            "-std=c++17",
            f"--offload-arch={os.getenv('HIP_ARCHITECTURES', 'native')}",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-DCK_FMHA_FWD_FAST_EXP2=1",
            "-fgpu-flush-denormals-to-zero",
      ]
      + generator_flag
      + cc_flag,
}

setup(name='my_mamba_fwd',
      ext_modules=[cpp_extension.CUDAExtension('my_mamba_fwd', 
                                              ["csrc/selective_scan/selective_scan.cu",
                                                "csrc/selective_scan/selective_scan_fwd_fp32.cu"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
