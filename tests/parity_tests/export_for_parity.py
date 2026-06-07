"""
export_for_parity.py — Export StatefulMambaHybrid1D with a fixed seed and save weights.

Runs on dorian-mac (needs coremltools). Creates:
  outputs/StatefulMambaHybrid1D_seq224_c1000_alpha0.1.mlpackage  (overwritten)
  outputs/StatefulMambaHybrid1D_seq224_c1000_alpha0.1.pt          (state_dict)

Usage: python tests/parity_tests/export_for_parity.py
"""
import sys, os, time
from pathlib import Path
import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mamba_ane.models.hybrid1d import StatefulMambaHybrid1D

SEED        = 42
NUM_CLASSES = 1000
SEQ_LENGTH  = 224
EMA_ALPHA   = 0.1
OUT_DIR     = Path(__file__).resolve().parents[2] / 'outputs'
STEM        = f"StatefulMambaHybrid1D_seq{SEQ_LENGTH}_c{NUM_CLASSES}_alpha{EMA_ALPHA}"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MLPACKAGE_PATH = OUT_DIR / f"{STEM}.mlpackage"
WEIGHTS_PATH   = OUT_DIR / f"{STEM}.pt"

# ── Build model with fixed seed ───────────────────────────────────────────────
print(f"[1/4] Building model (seed={SEED})")
torch.manual_seed(SEED)
model = StatefulMambaHybrid1D(
    num_classes=NUM_CLASSES,
    backbone_in_channels=3,
    backbone_hidden_dim=256,
    backbone_output_dim=512,
    seq_length=SEQ_LENGTH,
    ema_alpha=EMA_ALPHA,
    mamba_d_state=64,
    mamba_headdim=64,
    mamba_num_heads=8,
).eval()
print(f"      Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ── Save state_dict ───────────────────────────────────────────────────────────
print(f"[2/4] Saving weights → {WEIGHTS_PATH}")
torch.save(model.state_dict(), WEIGHTS_PATH)

# ── TorchScript trace ─────────────────────────────────────────────────────────
print(f"[3/4] Tracing + CoreML conversion")
example_input = torch.rand(1, 3, SEQ_LENGTH)
with torch.no_grad():
    traced = torch.jit.trace(model, (example_input,))

states_list = [
    ct.StateType(wrapped_type=ct.TensorType(shape=model.mamba.angle_state.shape), name="mamba.angle_state"),
    ct.StateType(wrapped_type=ct.TensorType(shape=model.mamba.k_state.shape),     name="mamba.k_state"),
    ct.StateType(wrapped_type=ct.TensorType(shape=model.mamba.v_state.shape),     name="mamba.v_state"),
    ct.StateType(wrapped_type=ct.TensorType(shape=model.mamba.ssm_state.shape),   name="mamba.ssm_state"),
]

t0 = time.time()
mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[ct.TensorType(name="x", shape=(1, 3, SEQ_LENGTH), dtype=np.float32)],
    outputs=[ct.TensorType(name="logits", dtype=np.float32)],
    states=states_list,
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT16,
)
print(f"      Conversion OK ({time.time()-t0:.1f}s)")

# ── Save mlpackage ────────────────────────────────────────────────────────────
print(f"[4/4] Saving mlpackage → {MLPACKAGE_PATH}")
mlmodel.save(str(MLPACKAGE_PATH))

# ── Quick sanity check: 3 frames, same model vs CoreML ───────────────────────
print("\nSanity check (3 frames, both starting from zero state):")
model.mamba.angle_state.zero_()
model.mamba.k_state.zero_()
model.mamba.v_state.zero_()
model.mamba.ssm_state.zero_()
coreml_state = mlmodel.make_state()
torch.manual_seed(0)
for i in range(3):
    x = torch.rand(1, 3, SEQ_LENGTH)
    with torch.no_grad():
        pt_out = model(x).numpy()
    cml_out = mlmodel.predict({"x": x.numpy()}, state=coreml_state)["logits"]
    diff = np.abs(pt_out - cml_out).max()
    print(f"  Frame {i}: max_abs={diff:.2e}")

print("\nDone. Run test_ane_mac.py to get the full parity report.")
