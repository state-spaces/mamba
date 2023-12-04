# Copyright (c) 2023, Albert Gu, Tri Dao.
import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
import shutil

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "mamba_ssm"

BASE_WHEEL_URL = "https://github.com/state-spaces/mamba/releases/download/{tag_name}/{wheel_name}"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("MAMBA_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("MAMBA_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("MAMBA_FORCE_CXX11_ABI", "FALSE") == "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none(PACKAGE_NAME)
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.6"):
            raise RuntimeError(
                f"{PACKAGE_NAME} is only supported on CUDA 11.6 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )

    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    ext_modules.append(
        CUDAExtension(
            name="selective_scan_cuda",
            sources=[
                "csrc/selective_scan/selective_scan.cpp",
                "csrc/selective_scan/selective_scan_fwd_fp32.cu",
                "csrc/selective_scan/selective_scan_fwd_fp16.cu",
                "csrc/selective_scan/selective_scan_fwd_bf16.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_complex.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        "--ptxas-options=-v",
                        "-lineinfo",
                    ]
                    + cc_flag
                ),
            },
            include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
        )
    )


def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("MAMBA_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


def get_wheel_url():
    # Determine the version numbers that will be used to determine the correct wheel
    # We're using the CUDA version used to build torch, not the one currently installed
    # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
    torch_cuda_version = parse(torch.version.cuda)
    torch_version_raw = parse(torch.__version__)
    # For CUDA 11, we only compile for CUDA 11.8, and for CUDA 12 we only compile for CUDA 12.2
    # to save CI time. Minor versions should be compatible.
    torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.2")
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    mamba_ssm_version = get_package_version()
    # cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
    cuda_version = f"{torch_cuda_version.major}{torch_cuda_version.minor}"
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    # Determine wheel URL based on CUDA version, torch version, python version and OS
    wheel_filename = f"{PACKAGE_NAME}-{mamba_ssm_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    wheel_url = BASE_WHEEL_URL.format(
        tag_name=f"v{mamba_ssm_version}", wheel_name=wheel_filename
    )
    return wheel_url, wheel_filename


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            shutil.move(wheel_filename, wheel_path)
        except urllib.error.HTTPError:
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "mamba_ssm.egg-info",
        )
    ),
    author="Tri Dao, Albert Gu",
    author_email="tri@tridao.me, agu@cs.cmu.edu",
    description="Mamba state-space model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/state-spaces/mamba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
        "einops",
        "triton",
        "transformers",
        "causal_conv1d",
    ],
)
