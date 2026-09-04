# b200-helion1.4-triton3.6-nemotron3-nano-bf16-seq8k

Example Helion configs, generated with:

```sh
MAMBA_HELION_CONFIG_DIR=$PWD/benchmarks/helion/configs/b200-helion1.4-triton3.6-nemotron3-nano-bf16-seq8k \
    python benchmarks/helion/autotune_mamba_mixer.py
```

i.e. the harness defaults: Nemotron3 Nano dims (hidden 2688, 64 heads x 64 head
dim, state 128, 8 groups, chunk 128), seq_len 8192, batch 1, bf16, unpacked path.

Environment: NVIDIA B200, helion 1.4.0, triton 3.6.0,
torch 2.11.0a0+a6c236b9fd.nv26.03.

A config is specialized to all of the above, so reuse it only on a matching
setup; otherwise re-autotune into a directory of your own.
