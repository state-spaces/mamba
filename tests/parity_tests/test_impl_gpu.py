"""
test_impl_gpu.py — GPU parity: OG_FP32 vs ANE_FP32, then ANE_FP32 vs ANE_FP16.
Runs on pc_ia (RTX 4090). Requires mamba3.py fix (Task 1).

Usage: python tests/parity_tests/test_impl_gpu.py
"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mamba_ane.models.hybrid1d import StatefulMambaHybrid1D
from og_model import OGStatefulMambaHybrid1D
from parity_lib import compute_metrics, aggregate_metrics, check_thresholds, generate_report

# ── Config ──────────────────────────────────────────────────────────────────
SEED        = 42
N_WARMUP    = 32
N_MEASURE   = 64
SEQ_LENGTH  = 224
BATCH       = 1
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOL_IMPL = {'max_abs': 1e-2, 'cosine_sim': 0.999}
TOL_PREC = {'max_abs': 1e-2, 'cosine_sim': 0.999}

# ── Helpers ─────────────────────────────────────────────────────────────────

def make_frames(n, seed=SEED, dtype=torch.float32):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.randn(n, BATCH, 3, SEQ_LENGTH, generator=g, dtype=torch.float32).to(DEVICE, dtype)


def run_model(model, frames):
    """Run model frame by frame. Returns list of logit tensors."""
    outs = []
    with torch.no_grad():
        for t in range(len(frames)):
            out = model(frames[t])
            outs.append(out.detach().cpu().float())
    return outs


def reset_model(model):
    """Reset any accumulated state before a run."""
    if hasattr(model, 'reset_state'):
        model.reset_state()
    elif hasattr(model, 'mamba'):
        for buf in ['angle_state', 'ssm_state', 'k_state', 'v_state']:
            b = getattr(model.mamba, buf, None)
            if b is not None: b.zero_()

# ── Build reference model ────────────────────────────────────────────────────

torch.manual_seed(SEED)
ref_fp32 = StatefulMambaHybrid1D().to(DEVICE).float().eval()

# OG model gets exact same weights via mapper
og_fp32 = OGStatefulMambaHybrid1D().to(DEVICE).float().eval()
og_fp32.load_from_stateful(ref_fp32)

# FP16 model gets same weights (copied from fp32 then cast)
ref_fp16 = StatefulMambaHybrid1D().to(DEVICE).half().eval()
with torch.no_grad():
    for (n1, p1), (n2, p2) in zip(ref_fp16.named_parameters(), ref_fp32.named_parameters()):
        p1.copy_(p2.half())

# ── All frames ───────────────────────────────────────────────────────────────

N_TOTAL = N_WARMUP + N_MEASURE
frames_fp32 = make_frames(N_TOTAL, dtype=torch.float32)
frames_fp16 = make_frames(N_TOTAL, dtype=torch.float16)

# ── Run models ───────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")
print(f"Warmup: {N_WARMUP} frames  Measure: {N_MEASURE} frames")

print("\n[1/2] OG_FP32 vs ANE_FP32 (implementation gap)...")
reset_model(og_fp32); outs_og   = run_model(og_fp32,   frames_fp32)
reset_model(ref_fp32); outs_fp32 = run_model(ref_fp32, frames_fp32)

print("[2/2] ANE_FP32 vs ANE_FP16 (precision gap)...")
reset_model(ref_fp32); outs_fp32b = run_model(ref_fp32, frames_fp32)
reset_model(ref_fp16); outs_fp16  = run_model(ref_fp16, frames_fp16)

# ── Metrics (measurement frames only) ────────────────────────────────────────

def measure(outs_ref, outs_test):
    per_step = [compute_metrics(outs_test[N_WARMUP + i], outs_ref[N_WARMUP + i])
                for i in range(N_MEASURE)]
    return aggregate_metrics(per_step), [m['max_abs'] for m in per_step]

agg_impl, hist_impl = measure(outs_og,    outs_fp32)
agg_prec, hist_prec = measure(outs_fp32b, outs_fp16)

thr_impl = check_thresholds(agg_impl, TOL_IMPL)
thr_prec = check_thresholds(agg_prec, TOL_PREC)

# ── Print summary ─────────────────────────────────────────────────────────────

for label, agg, thr in [('IMPL (OG→ANE FP32)', agg_impl, thr_impl),
                         ('PREC (FP32→FP16)',   agg_prec, thr_prec)]:
    status = 'PASS' if thr['pass'] else 'FAIL'
    print(f"\n{label}: {status}")
    print(f"  max_abs={agg['max_abs']:.3e}  cosine_sim_min={agg['cosine_sim_min']:.6f}")
    if not thr['pass']:
        for f in thr['failed_checks']: print(f"  x {f}")

# ── Write report ──────────────────────────────────────────────────────────────

cfg_str = f"d_model=512, d_state=64, expand=1, headdim=64, num_heads=8, seq_length={SEQ_LENGTH}"
seq_str = f"warmup={N_WARMUP} + measure={N_MEASURE} frames, seed={SEED}, batch={BATCH}"

comparisons = [
    dict(name='IMPL', ref_label='OG_FP32 (Mamba3 CUDA)', test_label='ANE_FP32 (MambaBlock)',
         tol=TOL_IMPL, agg=agg_impl, threshold=thr_impl, per_step_max_abs=hist_impl),
    dict(name='PREC', ref_label='ANE_FP32', test_label='ANE_FP16',
         tol=TOL_PREC, agg=agg_prec, threshold=thr_prec, per_step_max_abs=hist_prec),
]

report_path = Path(__file__).parent / 'parity_report_gpu.md'
report_path.write_text(generate_report(
    'GPU Parity Report: StatefulMambaHybrid1D', comparisons, cfg_str, seq_str))
print(f"\nReport written to {report_path}")
