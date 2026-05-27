# Monkey patch Triton to prevent AttributeError on versions lacking set_allocator
try:
    import triton
    if not hasattr(triton, "set_allocator"):
        def _dummy_set_allocator(*args, **kwargs):
            pass
        triton.set_allocator = _dummy_set_allocator
except ImportError:
    pass

__version__ = "2.3.2.post1"


from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
