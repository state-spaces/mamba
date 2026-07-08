# Mamba ANE

ANE-native implementation of Mamba-3 for Apple Silicon (M-series). This package provides a pure PyTorch port of Mamba-3 optimized for inference on Neural Engine accelerators via CoreML.

## Quick Start

### Installation

Create a dedicated conda environment with Python 3.12:

```bash
conda create -n ane_export python=3.12
conda activate ane_export
pip install -r requirements.txt
```

Alternatively, use a virtual environment:

```bash
python3.12 -m venv ane_export
source ane_export/bin/activate
pip install -r requirements.txt
```

### Model Understanding

For a detailed walkthrough of the Mamba-3 SISO (Single Input Single Output) architecture and how it's adapted for ANE, see:

📖 **[Mamba-3 SISO Interactive Visualization](./docs/mamba3_siso_viz.html)**

This HTML guide includes:
- **7-stage pipeline visualization** showing the forward pass flow
- **Side-by-side code comparison** between original Mamba3 (Triton kernels) and ANE-native PyTorch
- **Tensor shape tracking** across all stages
- **Key architectural differences** for ANE compatibility (RoPE, SSM state, attention patterns)
- **Interactive visualizers** for understanding state flow and computation

Open in any modern browser (Chrome, Safari, Firefox).

## Architecture

### Overview

The ANE implementation replaces Triton/CUDA kernels with pure PyTorch operations compatible with Core ML's Neural Engine compute units. Key modules:

- **`mamba_ane/modules/mamba3.py`** — Core Mamba-3 block with ANE-compatible SSM
- **`mamba_ane/models/hybrid1d.py`** — StatefulMambaHybrid1D model wrapper

### Design Principles

1. **Pure PyTorch** — No Triton/CuteDSL; uses standard PyTorch ops
2. **Stateful** — Maintains hidden state for streaming inference
3. **FP32 Default** — Full precision; FP16 supported for CoreML export
4. **Shape-Friendly** — Tensors designed for BNHD (batch, nheads, height, depth) layout

## Parity Testing

Numerical equivalence is verified across:

### GPU → ANE Parity (FP32)
Original Mamba-3 (CUDA) vs ANE PyTorch implementation.

**Result**: ✅ **PASS**
- Max absolute error: 6.01e-06 (tolerance: < 0.01)
- Cosine similarity: 1.000000 (tolerance: > 0.999)
- Consistency across 64 measurement frames

See: [`tests/parity_tests/parity_report_gpu.md`](../tests/parity_tests/parity_report_gpu.md)

### FP32 → FP16 Precision
ANE FP32 model → ANE FP16 quantization.

**Result**: ✅ **PASS**
- Max absolute error: 3.05e-05 (tolerance: < 0.01)
- Cosine similarity: 1.000000 (tolerance: > 0.999)

See: [`tests/parity_tests/parity_report_gpu.md`](../tests/parity_tests/parity_report_gpu.md)

### PyTorch → CoreML (CPU_AND_NE) Parity
ANE PyTorch export (FP32) → CoreML execution on Apple Silicon (FP16)

**Result**: ✅ **PASS**
- Max absolute error: 2.10e-04 (tolerance: < 0.03)
- Cosine similarity: 0.999998 (tolerance: > 0.999)
- Consistency across 64 measurement frames
- Compute units: `CPU_AND_NE` (ANE acceleration enabled)

See: [`tests/parity_tests/parity_report_ane.md`](../tests/parity_tests/parity_report_ane.md)

## Usage

### Export to CoreML

```python
from mamba_ane.modules.hybrid1d import StatefulMambaHybrid1D
import coremltools as ct

# Load weights (or train from scratch)
model = StatefulMambaHybrid1D(d_model=512, num_heads=8, d_state=64)

# Trace to CoreML
traced = ct.models.neural_network.NeuralNetworkBuilder(
    inputs=[...],
    outputs=[...]
)
model.to_coreml(traced, compute_units=ct.ComputeUnit.CPU_AND_NE)
```

### Inference

```python
model.eval()
with torch.no_grad():
    output, state = model(input_ids, state)
```

## Configuration

Model config in `mamba_ane/configs/`:
- **d_model** — Hidden dimension (default: 512)
- **num_heads** — Attention heads (default: 8)
- **d_state** — State dimension (default: 64)
- **seq_length** — Maximum sequence length (default: 2048)

## Testing

Run all parity tests:

```bash
python -m pytest tests/parity_tests/ -v
```

Specific test suites:
- GPU parity: `python tests/parity_tests/test_impl_gpu.py`
- ANE parity: `python tests/parity_tests/test_ane_mac.py`

See: [`tests/parity_tests/`](../tests/parity_tests/)

## References

- **Mamba-3 Original**: https://github.com/state-spaces/mamba/
- **StatefulMobileNet**: https://github.com/Dorianhgn/StatefulMobileNet - *A step by step implementation of mamba operations in torch to make the `Mamba3` model work on ANE.*
- **ANE Documentation**: https://developer.apple.com/documentation/coreml/neural_engine
- **CoreML Tools**: https://github.com/apple/coremltools

## Status

- ✅ Architecture ported to pure PyTorch
- ✅ GPU/ANE numerical parity verified
- ✅ CoreML export working
- ✅ FP16 quantization tested
- 🟡 `Mamba3 SISO → ANE Pytorch → CoreML` model weights conversion.
- 🟡 Production benchmarking on iPhone 17 Pro ANE
