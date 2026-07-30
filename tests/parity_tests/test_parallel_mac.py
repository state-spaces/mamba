# tests/parity_tests/test_parallel_mac.py
"""
Mac parity test: PyTorch FP16 (MPS) vs CoreML CPU_AND_NE.

Requires:
  - outputs/golden_parallel.pt  (produced by test_parallel_gpu.py on pc_ia)
  - coremltools 9.0 (ane_export conda env)
  - Apple Silicon Mac (dorian-mac)

Steps:
  1. Load golden_parallel.pt (weights + reference outputs)
  2. Run StatefulMambaParallelHybrid in FP16 on MPS; compare vs FP32 golden
  3. Export to CoreML via export_parallel.py logic
  4. Run CoreML (CPU_AND_NE); compare vs FP32 golden
  5. Run 64 inputs through CoreML to check for NaN accumulation
  6. Write parity_report_parallel.md

Run: python tests/parity_tests/test_parallel_mac.py
"""
import sys, time
import numpy as np
import torch
import coremltools as ct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mamba_ane.models.hybrid_parallel import StatefulMambaParallelHybrid
from parity_lib import compute_metrics, aggregate_metrics, check_thresholds, generate_report

GOLDEN_PATH = Path(__file__).resolve().parents[2] / 'outputs' / 'golden_parallel.pt'
OUT_DIR     = Path(__file__).resolve().parents[2] / 'outputs'

TOL_MPS    = {'max_abs': 1e-2,  'cosine_sim': 0.9999}   # PyTorch FP16 vs FP32 golden
TOL_COREML = {'max_abs': 3e-2,  'cosine_sim': 0.999}    # CoreML vs FP32 golden

assert GOLDEN_PATH.exists(), (
    f"golden_parallel.pt not found at {GOLDEN_PATH}\n"
    "Run test_parallel_gpu.py on pc_ia first, then rsync the outputs/ directory."
)

print(f"Loading golden from {GOLDEN_PATH}")
golden = torch.load(GOLDEN_PATH, map_location='cpu')
cfg    = golden['config']

SEQ_LENGTH  = cfg['seq_length']   # 256
D_MODEL     = cfg['d_model']      # 256
IN_CHANNELS = cfg['in_channels']  # 3
NUM_CLASSES = cfg['num_classes']  # 10
N_SAMPLES   = golden['inputs'].shape[0]  # 64

print(f"Config: d_model={D_MODEL}, seq_length={SEQ_LENGTH}, "
      f"n_samples={N_SAMPLES}, num_classes={NUM_CLASSES}")

# ── Load portable model ───────────────────────────────────────────────────────
print("\n[1/5] Loading StatefulMambaParallelHybrid from golden weights")
pt_model = StatefulMambaParallelHybrid(
    num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, d_model=D_MODEL
).eval()
pt_model.load_state_dict(golden['portable_weights'])

# ── Step 2: FP16 on MPS ───────────────────────────────────────────────────────
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"\n[2/5] PyTorch FP16 on {DEVICE}")
pt_fp16 = StatefulMambaParallelHybrid(
    num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, d_model=D_MODEL
).to(DEVICE).half().eval()
with torch.no_grad():
    for (_, p16), (_, p32) in zip(
        pt_fp16.named_parameters(), pt_model.named_parameters()
    ):
        p16.copy_(p32.half())

outs_fp16 = []
with torch.no_grad():
    for i in range(N_SAMPLES):
        x = golden['inputs'][i].to(DEVICE, torch.float16)
        outs_fp16.append(pt_fp16(x).cpu().float())

ref_fp32 = [golden['outputs_og_fp32'][i].float() for i in range(N_SAMPLES)]

per_mps = [compute_metrics(outs_fp16[i], ref_fp32[i]) for i in range(N_SAMPLES)]
agg_mps = aggregate_metrics(per_mps)
thr_mps = check_thresholds(agg_mps, TOL_MPS)
st = 'PASS' if thr_mps['pass'] else 'FAIL'
print(f"  FP16 MPS: {st}  max_abs={agg_mps['max_abs']:.3e}  "
      f"cosine_sim_min={agg_mps['cosine_sim_min']:.6f}")
if not thr_mps['pass']:
    for f in thr_mps['failed_checks']: print(f"  x {f}")

# ── Step 3: Export to CoreML ─────────────────────────────────────────────────
print(f"\n[3/5] Exporting to CoreML (seq={SEQ_LENGTH})")
model_cpu = pt_model.float().cpu().eval()
input_shape = (1, IN_CHANNELS, SEQ_LENGTH)
with torch.no_grad():
    traced = torch.jit.trace(model_cpu, (torch.rand(*input_shape),))

t0 = time.time()
mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[ct.TensorType(name="x", shape=input_shape, dtype=np.float32)],
    outputs=[ct.TensorType(name="logits", dtype=np.float32)],
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT16,
)
elapsed = time.time() - t0
print(f"      Conversion OK ({elapsed:.1f}s)")

stem = f"StatefulMambaParallelHybrid_seq{SEQ_LENGTH}_c{NUM_CLASSES}"
mlpackage_path = OUT_DIR / f"{stem}.mlpackage"
mlmodel.save(str(mlpackage_path))
print(f"      Saved → {mlpackage_path}")

# ── Step 4: CoreML CPU_AND_NE parity ─────────────────────────────────────────
print(f"\n[4/5] CoreML CPU_AND_NE parity ({N_SAMPLES} inputs)")
cml_model = ct.models.MLModel(
    str(mlpackage_path), compute_units=ct.ComputeUnit.CPU_AND_NE
)

outs_cml = []
for i in range(N_SAMPLES):
    x_np = golden['inputs'][i].numpy().astype(np.float32)
    result = cml_model.predict({"x": x_np})
    outs_cml.append(torch.from_numpy(result["logits"]).float())
    if (i + 1) % 16 == 0:
        print(f"  {i+1}/{N_SAMPLES} done")

per_cml = [compute_metrics(outs_cml[i], ref_fp32[i]) for i in range(N_SAMPLES)]
agg_cml = aggregate_metrics(per_cml)
thr_cml = check_thresholds(agg_cml, TOL_COREML)
st = 'PASS' if thr_cml['pass'] else 'FAIL'
print(f"  CoreML CPU_AND_NE: {st}  max_abs={agg_cml['max_abs']:.3e}  "
      f"cosine_sim_min={agg_cml['cosine_sim_min']:.6f}")
if not thr_cml['pass']:
    for f in thr_cml['failed_checks']: print(f"  x {f}")

# ── Step 5: NaN check (64 inputs) ─────────────────────────────────────────────
print(f"\n[5/5] NaN accumulation check (64 inputs)")
nan_found = False
for i in range(min(64, N_SAMPLES)):
    x_np = golden['inputs'][i].numpy().astype(np.float32)
    result = cml_model.predict({"x": x_np})
    if np.isnan(result["logits"]).any():
        print(f"  NaN at input {i}!")
        nan_found = True
        break
if not nan_found:
    print("  No NaN detected  OK")

# ── Write report ─────────────────────────────────────────────────────────────
cfg_str = (f"d_model={D_MODEL}, d_state=64, headdim=64, "
           f"seq_length={SEQ_LENGTH}, num_classes={NUM_CLASSES}")
seq_str = f"{N_SAMPLES} independent inputs, seed={cfg['seed']}, compute_units=CPU_AND_NE"

comparisons = [
    dict(name='MPS_FP16',    ref_label='OG_FP32 (golden)',  test_label=f'Portable_FP16 ({DEVICE})',
         tol=TOL_MPS,    agg=agg_mps, threshold=thr_mps,
         per_step_max_abs=[m['max_abs'] for m in per_mps]),
    dict(name='CoreML_ANE',  ref_label='OG_FP32 (golden)',  test_label='CoreML CPU_AND_NE',
         tol=TOL_COREML, agg=agg_cml, threshold=thr_cml,
         per_step_max_abs=[m['max_abs'] for m in per_cml]),
]

report_path = Path(__file__).parent / 'parity_report_parallel.md'
report = generate_report(
    'Parallel ANE Parity Report: StatefulMambaParallelHybrid', comparisons, cfg_str, seq_str)
report_path.write_text(report)
print(f"\nReport written to {report_path}")

all_pass = thr_mps['pass'] and thr_cml['pass'] and not nan_found
if not all_pass:
    sys.exit(1)
print("\nAll checks PASS")
