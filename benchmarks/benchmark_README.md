# Benchmarks

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
