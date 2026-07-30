"""
test_ane_mac.py — ANE parity: pytorch FP32 vs CoreML CPU_AND_NE.
Runs on dorian-mac only (requires coremltools + Apple Silicon).

Usage: python tests/parity_tests/test_ane_mac.py
"""
import sys, torch, numpy as np
from pathlib import Path
import coremltools as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for mamba_ane
sys.path.insert(0, str(Path(__file__).resolve().parent))       # for parity_lib local import

from mamba_ane.models.hybrid1d import StatefulMambaHybrid1D
from parity_lib import compute_metrics, aggregate_metrics, check_thresholds, generate_report

# ── Config ──────────────────────────────────────────────────────────────────
SEED        = 42
N_WARMUP    = 32
N_MEASURE   = 64
SEQ_LENGTH  = 224
N_TOTAL     = N_WARMUP + N_MEASURE

MLPACKAGE   = Path(__file__).resolve().parents[2] / 'outputs' / \
              'StatefulMambaHybrid1D_seq224_c1000_alpha0.1.mlpackage'
WEIGHTS     = Path(__file__).resolve().parents[2] / 'outputs' / \
              'StatefulMambaHybrid1D_seq224_c1000_alpha0.1.pt'

TOL_ANE = {'max_abs': 3e-2, 'cosine_sim': 0.999}

# ── Load models ──────────────────────────────────────────────────────────────

print(f"Loading .mlpackage from {MLPACKAGE}")
assert MLPACKAGE.exists(), f"mlpackage not found: {MLPACKAGE}"
assert WEIGHTS.exists(), f"weights not found: {WEIGHTS} — run export_for_parity.py first"

mlmodel = ct.models.MLModel(str(MLPACKAGE),
                             compute_units=ct.ComputeUnit.CPU_AND_NE)
print("CoreML model loaded (CPU_AND_NE)")

pt_model = StatefulMambaHybrid1D().float().eval()
pt_model.load_state_dict(torch.load(WEIGHTS, map_location='cpu'))
print(f"PyTorch weights loaded from {WEIGHTS.name}")

# Reset pytorch state buffers
for buf in ['angle_state', 'ssm_state', 'k_state', 'v_state']:
    getattr(pt_model.mamba, buf).zero_()

# CoreML state
coreml_state = mlmodel.make_state()

# ── Generate frames ───────────────────────────────────────────────────────────

g = torch.Generator(); g.manual_seed(SEED)
frames = torch.randn(N_TOTAL, 1, 3, SEQ_LENGTH, generator=g, dtype=torch.float32)

# ── Run both models ───────────────────────────────────────────────────────────

print(f"Running {N_TOTAL} frames (warmup={N_WARMUP}, measure={N_MEASURE})...")

outs_pt     = []
outs_coreml = []

with torch.no_grad():
    for t in range(N_TOTAL):
        x = frames[t]                                  # (1, 3, 224)
        x_np = x.numpy().astype(np.float32)

        out_pt = pt_model(x).cpu().float()
        outs_pt.append(out_pt)

        out_cml = mlmodel.predict({'x': x_np}, state=coreml_state)['logits']
        outs_coreml.append(torch.from_numpy(out_cml).float())

        if (t + 1) % 16 == 0:
            print(f"  {t+1}/{N_TOTAL} frames done")

# ── Metrics ───────────────────────────────────────────────────────────────────

per_step = [compute_metrics(outs_coreml[N_WARMUP + i], outs_pt[N_WARMUP + i])
            for i in range(N_MEASURE)]
agg  = aggregate_metrics(per_step)
thr  = check_thresholds(agg, TOL_ANE)
hist = [m['max_abs'] for m in per_step]

status = 'PASS' if thr['pass'] else 'FAIL'
print(f"\nANE (CPU_AND_NE): {status}")
print(f"  max_abs={agg['max_abs']:.3e}  cosine_sim_min={agg['cosine_sim_min']:.6f}")
if not thr['pass']:
    for f in thr['failed_checks']: print(f"  x {f}")

# ── Report ────────────────────────────────────────────────────────────────────

cfg_str = f"d_model=512, d_state=64, headdim=64, num_heads=8, seq_length={SEQ_LENGTH}"
seq_str = f"warmup={N_WARMUP} + measure={N_MEASURE} frames, seed={SEED}, compute_units=CPU_AND_NE"

comparisons = [
    dict(name='ANE', ref_label='pytorch FP32 (CPU)', test_label='CoreML CPU_AND_NE',
         tol=TOL_ANE, agg=agg, threshold=thr, per_step_max_abs=hist),
]

report_path = Path(__file__).parent / 'parity_report_ane.md'
report_path.write_text(generate_report(
    'ANE Parity Report: StatefulMambaHybrid1D', comparisons, cfg_str, seq_str))
print(f"\nReport written to {report_path}")
