from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "2.3.2.post1"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2

if TYPE_CHECKING:
    from mamba_ssm.modules.mamba3 import Mamba3
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

__all__ = [
    "__version__",
    "selective_scan_fn",
    "mamba_inner_fn",
    "Mamba",
    "Mamba2",
    "Mamba3",
    "MambaLMHeadModel",
]

_LAZY_EXPORTS = {
    "Mamba3": ("mamba_ssm.modules.mamba3", "Mamba3"),
    "MambaLMHeadModel": ("mamba_ssm.models.mixer_seq_simple", "MambaLMHeadModel"),
}


def __getattr__(name: str) -> Any:
    """Load optional Mamba-3 surfaces only when explicitly requested.

    Importing an unrelated submodule such as a Mamba-2 Triton SSD operator must
    not initialize TileLang/TVM through the Mamba-3 module.
    """

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
