# Copyright (c) 2024, Tri Dao, Albert Gu.

import os
import sys
import urllib.error
from unittest.mock import patch
import pytest
import torch


def test_setup_cxx_standard_selection():
    """Verify setup.py selects C++20 for PyTorch >= 2.13 and C++17 for older versions (#1088)."""
    setup_path = os.path.join(os.path.dirname(__file__), "..", "setup.py")
    with open(setup_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Test PyTorch 2.14
    with patch.object(torch, "__version__", "2.14.0"), patch.dict(
        os.environ, {"MAMBA_KEEP_CUDA_BUILD": "TRUE"}
    ), patch.object(sys, "argv", ["setup.py", "--name"]):
        g = {"__file__": setup_path, "__name__": "__main__", "sys": sys}
        exec(code, g)
        assert g["cxx_standard"] == "c++20"

    # Test PyTorch 2.12
    with patch.object(torch, "__version__", "2.12.0"), patch.dict(
        os.environ, {"MAMBA_KEEP_CUDA_BUILD": "TRUE"}
    ), patch.object(sys, "argv", ["setup.py", "--name"]):
        g = {"__file__": setup_path, "__name__": "__main__", "sys": sys}
        exec(code, g)
        assert g["cxx_standard"] == "c++17"


def test_setup_with_cuda_home_none():
    """Verify setup.py executes safely when CUDA_HOME is None without NameError or unassigned variable crashes."""
    setup_path = os.path.join(os.path.dirname(__file__), "..", "setup.py")
    with open(setup_path, "r", encoding="utf-8") as f:
        code = f.read()

    with patch("torch.utils.cpp_extension.CUDA_HOME", None), patch.dict(
        os.environ, {"MAMBA_KEEP_CUDA_BUILD": "TRUE"}
    ), patch.object(sys, "argv", ["setup.py", "--name"]):
        g = {"__file__": setup_path, "__name__": "__main__", "sys": sys}
        exec(code, g)

        assert g["ext_modules"] == []
        assert "bdist_wheel" in g["cmdclass"]
        assert "build_ext" not in g["cmdclass"]


def test_setup_cpu_torch_clear_error():
    """Verify get_wheel_url provides a clear error message when PyTorch has no CUDA support."""
    setup_path = os.path.join(os.path.dirname(__file__), "..", "setup.py")
    with open(setup_path, "r", encoding="utf-8") as f:
        code = f.read()

    with patch.object(torch.version, "cuda", None), patch.object(
        sys, "argv", ["setup.py", "--name"]
    ):
        g = {"__file__": setup_path, "__name__": "__main__", "sys": sys}
        exec(code, g)

        with pytest.raises(RuntimeError, match="PyTorch was built without CUDA support"):
            g["get_wheel_url"]()


def test_cached_wheels_command_force_build_without_compiler():
    """Verify FORCE_BUILD fails with an informative error if no compiler is available."""
    setup_path = os.path.join(os.path.dirname(__file__), "..", "setup.py")
    with open(setup_path, "r", encoding="utf-8") as f:
        code = f.read()

    with patch("torch.utils.cpp_extension.CUDA_HOME", None), patch.dict(
        os.environ, {"MAMBA_FORCE_BUILD": "TRUE", "MAMBA_KEEP_CUDA_BUILD": "TRUE"}
    ), patch.object(sys, "argv", ["setup.py", "--name"]):
        g = {"__file__": setup_path, "__name__": "__main__", "sys": sys}
        exec(code, g)

        from setuptools.dist import Distribution

        dist = Distribution({"name": "mamba_ssm"})
        cmd = g["cmdclass"]["bdist_wheel"](dist)

        with pytest.raises(RuntimeError, match="no compiler was found"):
            cmd.run()


def test_cached_wheels_command_missing_wheel_without_compiler():
    """Verify missing precompiled wheel fails with an informative error if no compiler is available."""
    setup_path = os.path.join(os.path.dirname(__file__), "..", "setup.py")
    with open(setup_path, "r", encoding="utf-8") as f:
        code = f.read()

    with patch("torch.utils.cpp_extension.CUDA_HOME", None), patch.dict(
        os.environ, {"MAMBA_KEEP_CUDA_BUILD": "TRUE"}
    ), patch.object(sys, "argv", ["setup.py", "--name"]), patch(
        "urllib.request.urlretrieve",
        side_effect=urllib.error.HTTPError("url", 404, "Not Found", {}, None),
    ):
        g = {"__file__": setup_path, "__name__": "__main__", "sys": sys}
        exec(code, g)

        from setuptools.dist import Distribution

        dist = Distribution({"name": "mamba_ssm"})
        cmd = g["cmdclass"]["bdist_wheel"](dist)

        with pytest.raises(RuntimeError, match="cannot build from source because no compiler was found"):
            cmd.run()
