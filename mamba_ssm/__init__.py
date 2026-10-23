__version__ = "2.3.2.post1"

# Lazy exports (PEP 562): importing the package must not eagerly import optional
# backends. On Ascend NPU (torch_npu + triton-ascend) the mamba3 stack is
# unavailable (tilelang / triton.set_allocator), and an eager mamba3 import in
# the parent package would also block every mamba2 entry point
# (`import mamba_ssm.modules.mamba2`, `mamba_ssm.ops.triton.ssd_combined`, ...).
import importlib

__all__ = [
    "selective_scan_fn",
    "mamba_inner_fn",
    "Mamba",
    "Mamba2",
    "Mamba3",
    "MambaLMHeadModel",
]

_LAZY_IMPORTS = {
    "selective_scan_fn": "mamba_ssm.ops.selective_scan_interface",
    "mamba_inner_fn": "mamba_ssm.ops.selective_scan_interface",
    "Mamba": "mamba_ssm.modules.mamba_simple",
    "Mamba2": "mamba_ssm.modules.mamba2",
    "Mamba3": "mamba_ssm.modules.mamba3",
    "MambaLMHeadModel": "mamba_ssm.models.mixer_seq_simple",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
