"""Optional Helion kernels for Mamba SSM.

The existing Triton kernels remain the default. Set ``MAMBA_USE_HELION=1`` and
point ``MAMBA_HELION_CONFIG_DIR`` at matching pre-generated configs to opt in.
Generate missing configs with::

    python benchmarks/helion/autotune_mamba_mixer.py
"""
