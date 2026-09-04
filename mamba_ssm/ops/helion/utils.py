import os
from functools import wraps
from pathlib import Path
from typing import Optional, Sequence

import helion

def save_helion_config(config: helion.Config, kernel_name: str) -> None:
    """Save an autotuned Helion config as ``<kernel_name>.json``.

    The output directory is read from the ``MAMBA_HELION_CONFIG_DIR``
    environment variable, which must be set. Overwrites any existing file, which
    is how a forced re-autotune (``MAMBA_HELION_AUTOTUNE=2``) replaces a config.
    """
    config_dir = os.environ.get("MAMBA_HELION_CONFIG_DIR")
    assert config_dir, "MAMBA_HELION_CONFIG_DIR environment variable is not set"
    path = Path(config_dir) / f"{kernel_name}.json"
    config.save(path)
    print(f"[helion] saved config for {kernel_name} to {path}")


def load_helion_config(kernel_name: str) -> Optional[helion.Config]:
    """Load a saved Helion config for ``<kernel_name>.json``.

    The directory is read from the ``MAMBA_HELION_CONFIG_DIR`` environment
    variable, which must be set. Returns ``None`` -- telling the caller to
    autotune -- in two cases driven by ``MAMBA_HELION_AUTOTUNE``:

    * ``=1`` (missing-only): no config file exists yet. An existing file is
      loaded and reused, so a crashed multi-kernel run resumes where it left off.
    * ``=2`` (force): always re-autotune and overwrite, even if a config file
      already exists. Use this to re-tune for a new GPU / Helion version.

    Otherwise a missing config is a hard error: autotuning is an offline,
    single-process step and must never run at runtime (e.g. under CUDA graph
    capture or distributed jobs).
    """
    config_dir = os.environ.get("MAMBA_HELION_CONFIG_DIR")
    assert config_dir, "MAMBA_HELION_CONFIG_DIR environment variable is not set"
    autotune = os.environ.get("MAMBA_HELION_AUTOTUNE")
    path = Path(config_dir) / f"{kernel_name}.json"
    if autotune == "2":
        # Force: re-autotune and overwrite any existing config.
        return None
    if not path.is_file():
        assert autotune == "1", (
            f"[helion] config for {kernel_name} not found at {path}; "
            f"run offline with MAMBA_HELION_AUTOTUNE=1 to generate it"
        )
        return None
    config = helion.Config.load(path)
    print(f"[helion] loaded config for {kernel_name} from {path}")
    return config


def ensure_helion_config(kernel: helion.Kernel, args: Sequence[object]) -> None:
    """Make sure ``kernel`` has a config, loading from disk or autotuning.

    Uses ``kernel.name`` as the config filename, so callers don't repeat the
    kernel name as a string. On first call, load ``<kernel.name>.json`` and
    install it as the kernel's single config so every specialization (e.g.
    forward and backward calls with slightly different inputs) reuses it. A
    missing config is a hard error unless ``MAMBA_HELION_AUTOTUNE`` is set, in
    which case it autotunes with the example ``args`` and saves the result (an
    offline, single-process step only): ``=1`` only fills in missing configs,
    ``=2`` forces a re-autotune and overwrites any existing one. No-op once a
    config is set.
    """
    if kernel.configs:
        return
    config = load_helion_config(kernel.name)
    if config is None:
        config = kernel.autotune(args)
        save_helion_config(config, kernel.name)
    kernel.configs = [config]


def json_cached_autotune(kernel):
    """Wrap a ``helion.Kernel`` to lazily resolve its config on first call: load
    ``<kernel.name>.json`` from ``MAMBA_HELION_CONFIG_DIR``, or autotune and save
    that JSON when none exists (``MAMBA_HELION_AUTOTUNE=1``) or when a re-autotune
    is forced (``MAMBA_HELION_AUTOTUNE=2``, overwriting any existing config).

    The autotune arguments come from ``kernel.normalize_args`` on the actual call, so
    no call site has to restate them. The wrapper stays callable exactly like the
    kernel and exposes it as ``.kernel``.
    """

    @wraps(kernel)
    def wrapper(*args, **kwargs):
        if not kernel.configs:
            ensure_helion_config(kernel, kernel.normalize_args(*args, **kwargs))
        # Triton's allocator is a ContextVar, so it does not cross thread boundaries:
        # PyTorch autograd runs backward kernels on its own worker thread, which sees
        # the default NullAllocator rather than whatever the forward thread installed.
        # Helion only emits its own set_triton_allocator() at the top level of the
        # generated module (so it runs once, on the compiling thread) and only for
        # kernels taking tensor descriptors. Any kernel needing global scratch --
        # TMA descriptors, or a persistent kernel's workspace -- would then fail at
        # launch. Calling it here is idempotent: it no-ops unless the allocator is
        # still NullAllocator.
        helion.runtime.set_triton_allocator()
        return kernel(*args, **kwargs)

    wrapper.kernel = kernel
    return wrapper
