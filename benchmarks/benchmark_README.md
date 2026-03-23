# Benchmarks

## Overview

| Script | Purpose |
|---|---|
| `benchmark_text_generation_latency.py` | How fast can the model write text? |
| `benchmark_speed_mamba123.py` | Which Mamba generation computes fastest? |

## benchmark_text_generation_latency.py

Loads a pretrained model (Mamba or Transformer), feeds a prompt, generates tokens, and measures the total time.

- **Measures:** End-to-end latency — prompt processing + token-by-token decoding (ms)
- **Models:** One pretrained model at a time (e.g. `state-spaces/mamba-2.8b`)
- **Includes backward?** No (inference only)
- **Use case:** Evaluating deployment/serving performance

```sh
python benchmarks/benchmark_text_generation_latency.py \
    --model-name "state-spaces/mamba-2.8b" \
    --prompt "My cat wrote all this CUDA code for a new language model and" \
    --topp 0.9 --temperature 0.7
```

## benchmark_speed_mamba123.py

Creates Mamba1, Mamba2, and Mamba3 modules with identical settings (same d_model, sequence length, batch size, dtype), runs the same tensor through each, and times forward + backward passes.

- **Measures:** Raw computation speed — forward pass (ms) + backward pass (ms) + VRAM (GB)
- **Models:** Mamba1, Mamba2, Mamba3 side by side (random weights, same config)
- **Includes backward?** Yes
- **Use case:** Comparing architecture efficiency for research/training

```sh
python benchmarks/benchmark_speed_mamba123.py
```

**Note:** Mamba3 MIMO mode requires >100KB shared memory per SM (available on H100/sm_90+). On consumer GPUs like RTX 4080 (sm_89), use SISO mode instead.
