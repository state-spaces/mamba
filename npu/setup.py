import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel

PACKAGE_NAME = "cann_bench"
VERSION = "1.0.0"


class CMakeBuildExt(build_ext):
    """cmake 产物已由 scripts/build_wheel.sh 构建完成；此处仅把 _C 扩展拷入包目录。

    拷贝后命名 _C.abi3.so 以匹配 cp38-abi3 wheel 标签（torch 扩展不参与 limited
    API 编译，标签约定与产物命名对齐 cann-bench 提交契约）。
    """

    def build_extension(self, ext):
        build_lib = self.build_lib
        pkg_dir = os.path.join(build_lib, PACKAGE_NAME)
        os.makedirs(pkg_dir, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "_C.abi3.so")
        if not os.path.exists(src):
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "_C.so")
        if not os.path.exists(src):
            raise RuntimeError(f"cmake artifact not found: expected build/_C.abi3.so (run scripts/build_wheel.sh first)")
        dst = os.path.join(pkg_dir, "_C.abi3.so")
        shutil.copy2(src, dst)


class BDistWheelAbi3(bdist_wheel):
    def initialize_options(self):
        super().initialize_options()
        self.py_limited_api = "cp38"


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=[PACKAGE_NAME],
    package_data={PACKAGE_NAME: ["_C.abi3.so"]},
    include_package_data=True,
    ext_modules=[Extension(f"{PACKAGE_NAME}._C", sources=[])],
    cmdclass={"build_ext": CMakeBuildExt, "bdist_wheel": BDistWheelAbi3},
    python_requires=">=3.8",
    install_requires=[],
    zip_safe=False,
)
