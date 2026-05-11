# tests/parity_tests/test_parallel_gpu.py
"""
GPU parity: OGStatefulMambaParallelHybrid (Mamba3 CUDA) vs
            StatefulMambaParallelHybrid (Mamba3ParallelPortable).

Two comparisons:
  IMPL: OG_FP32  vs  Portable_FP32  (algorithm gap)
  PREC: Port_FP32  vs  Port_FP16  (precision gap)

Plus raw module comparison at L in {64, 256, 1024}.

Saves: outputs/golden_parallel.pt
  keys: config, weights, inputs_{L}, golden_{L} for L in {64, 256, 1024}

Run: python tests/parity_tests/test_parallel_gpu.py
Requires: CUDA + mamba_ssm (Triton kernels)
"""
import sys, os, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mamba_ane.modules.mamba3_parallel import Mamba3ParallelPortable
from mamba_ane.models.hybrid_parallel import (
    StatefulMambaParallelHybrid, _build_og_model
)
from parity_lib import compute_metrics, aggregate_metrics, check_thresholds, generate_report

SEED        = 42
N_SAMPLES   = 64          # number of independent (B=1, L=256, 3) inputs to measure
SEQ_LENGTH  = 256         # primary sequence length for full-model comparison
D_MODEL     = 256
IN_CHANNELS = 3
NUM_CLASSES = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR     = Path(__file__).resolve().parents[2] / 'outputs'

TOL_IMPL = {'max_abs': 1e-4,  'mean_abs': 1e-5,  'cosine_sim': 0.99999}
TOL_PREC = {'max_abs': 1e-2,  'cosine_sim': 0.999}

OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)

# ── Build portable model (FP32) ──────────────────────────────────────────────
print(f"Device: {DEVICE}")
print("[1/5] Building StatefulMambaParallelHybrid (portable, FP32)")
portable_fp32 = StatefulMambaParallelHybrid(
    num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, d_model=D_MODEL
).to(DEVICE).float().eval()
print(f"      Parameters: {sum(p.numel() for p in portable_fp32.parameters())/1e6:.2f}M")

# ── Build OG model, copy weights ─────────────────────────────────────────────
print("[2/5] Building OGStatefulMambaParallelHybrid and copying weights")
og_fp32 = _build_og_model(
    d_model=D_MODEL, d_state=64, headdim=64, num_classes=NUM_CLASSES,
    in_channels=IN_CHANNELS, hidden_dim=64, device=DEVICE, dtype=torch.float32
).to(DEVICE).float().eval()
og_fp32.load_from_portable(portable_fp32)

# ── FP16 portable ─────────────────────────────────────────────────────────────
portable_fp16 = StatefulMambaParallelHybrid(
    num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, d_model=D_MODEL
).to(DEVICE).half().eval()
with torch.no_grad():
    for (_, p16), (_, p32) in zip(
        portable_fp16.named_parameters(), portable_fp32.named_parameters()
    ):
        p16.copy_(p32.half())

# ── Generate input samples ────────────────────────────────────────────────────
print("[3/5] Generating input samples")
g = torch.Generator(device='cpu'); g.manual_seed(SEED)
# shape: (N_SAMPLES, 1, IN_CHANNELS, SEQ_LENGTH) — batch=1
inputs_all = torch.randn(N_SAMPLES, 1, IN_CHANNELS, SEQ_LENGTH, generator=g)

def run_samples(model, inputs, dtype=torch.float32):
    outs = []
    with torch.no_grad():
        for i in range(len(inputs)):
            x = inputs[i].to(DEVICE, dtype)
            outs.append(model(x).cpu().float())
    return outs

print("[4/5] Running all models")
outs_og   = run_samples(og_fp32,       inputs_all, torch.float32)
outs_fp32 = run_samples(portable_fp32, inputs_all, torch.float32)
outs_fp16 = run_samples(portable_fp16, inputs_all, torch.float16)

# ── Raw module L-sweep ────────────────────────────────────────────────────────
print("[5/5] Raw module L-sweep (direct Mamba3 vs Mamba3ParallelPortable)")
from mamba_ssm.modules.mamba3 import Mamba3

raw_results = {}
for L in [64, 256, 1024]:
    torch.manual_seed(SEED)
    og_raw = Mamba3(d_model=D_MODEL, d_state=64, expand=2, headdim=64,
                    ngroups=1, rope_fraction=0.5, is_mimo=False, is_outproj_norm=False,
                    layer_idx=0, dtype=torch.float32).to(DEVICE).float().eval()
    portable_raw = Mamba3ParallelPortable(d_model=D_MODEL).to(DEVICE).float().eval()
    portable_raw.load_from_original(og_raw)

    u = torch.randn(1, L, D_MODEL, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        ref = og_raw(u)
        tst = portable_raw(u)
    m = compute_metrics(tst, ref)
    raw_results[L] = m
    status = 'PASS' if m['max_abs'] < 1e-4 and m['cosine_sim'] > 0.99999 else 'FAIL'
    print(f"  L={L:4d}: max_abs={m['max_abs']:.2e}  cos={m['cosine_sim']:.7f}  {status}")

# ── Full model metrics ────────────────────────────────────────────────────────
per_impl = [compute_metrics(outs_fp32[i], outs_og[i])   for i in range(N_SAMPLES)]
per_prec = [compute_metrics(outs_fp16[i], outs_fp32[i]) for i in range(N_SAMPLES)]

agg_impl = aggregate_metrics(per_impl)
agg_prec = aggregate_metrics(per_prec)
thr_impl = check_thresholds(agg_impl, TOL_IMPL)
thr_prec = check_thresholds(agg_prec, TOL_PREC)

for label, agg, thr in [('IMPL (OG→Portable FP32)', agg_impl, thr_impl),
                         ('PREC (FP32→FP16)',        agg_prec, thr_prec)]:
    st = 'PASS' if thr['pass'] else 'FAIL'
    print(f"\n{label}: {st}")
    print(f"  max_abs={agg['max_abs']:.3e}  cosine_sim_min={agg['cosine_sim_min']:.6f}")
    if not thr['pass']:
        for f in thr['failed_checks']: print(f"  x {f}")

# ── Save golden ──────────────────────────────────────────────────────────────
print("\nSaving golden_parallel.pt ...")
golden = {
    'config': dict(d_model=D_MODEL, d_state=64, expand=2, headdim=64,
                   rope_fraction=0.5, num_classes=NUM_CLASSES,
                   in_channels=IN_CHANNELS, seq_length=SEQ_LENGTH, seed=SEED),
    'portable_weights': portable_fp32.state_dict(),
    'inputs': inputs_all,                                  # (N, 1, C, L)
    'outputs_og_fp32':  torch.stack(outs_og),              # (N, 1, num_classes)
    'outputs_fp32':     torch.stack(outs_fp32),
    'outputs_fp16':     torch.stack(outs_fp16),
}
for L in [64, 256, 1024]:
    u = torch.randn(1, L, D_MODEL, device=DEVICE, generator=torch.Generator(device=DEVICE).manual_seed(SEED))
    from mamba_ssm.modules.mamba3 import Mamba3 as _Mamba3
    og_raw = _Mamba3(d_model=D_MODEL, d_state=64, expand=2, headdim=64,
                     ngroups=1, rope_fraction=0.5, is_mimo=False,
                     is_outproj_norm=False, layer_idx=0).to(DEVICE).float().eval()
    portable_raw = Mamba3ParallelPortable(d_model=D_MODEL).to(DEVICE).float().eval()
    portable_raw.load_from_original(og_raw)
    with torch.no_grad():
        golden[f'raw_input_{L}']  = u.cpu()
        golden[f'raw_golden_{L}'] = og_raw(u).cpu()

torch.save(golden, OUT_DIR / 'golden_parallel.pt')
print(f"  Saved to {OUT_DIR / 'golden_parallel.pt'}")

# ── Write report ─────────────────────────────────────────────────────────────
cfg_str = (f"d_model={D_MODEL}, d_state=64, expand=2, headdim=64, "
           f"rope_fraction=0.5, seq_length={SEQ_LENGTH}")
seq_str = f"{N_SAMPLES} independent (B=1, C=3, L={SEQ_LENGTH}) inputs, seed={SEED}"

comparisons = [
    dict(name='IMPL', ref_label='OG_FP32 (Mamba3 CUDA)', test_label='Portable_FP32',
         tol=TOL_IMPL, agg=agg_impl, threshold=thr_impl,
         per_step_max_abs=[m['max_abs'] for m in per_impl]),
    dict(name='PREC', ref_label='Portable_FP32', test_label='Portable_FP16',
         tol=TOL_PREC, agg=agg_prec, threshold=thr_prec,
         per_step_max_abs=[m['max_abs'] for m in per_prec]),
]

raw_section = "\n## Raw module L-sweep (Mamba3 CUDA vs Mamba3ParallelPortable)\n\n"
raw_section += "| L | max_abs | cosine_sim | Status |\n|---|---------|------------|--------|\n"
for L, m in raw_results.items():
    st = '✓' if m['max_abs'] < 1e-4 and m['cosine_sim'] > 0.99999 else '✗'
    raw_section += f"| {L} | {m['max_abs']:.2e} | {m['cosine_sim']:.7f} | {st} |\n"

report_path = Path(__file__).parent / 'parity_report_parallel.md'
body = generate_report(
    'GPU Parity Report: StatefulMambaParallelHybrid', comparisons, cfg_str, seq_str)
report_path.write_text(body + raw_section)
print(f"\nReport written to {report_path}")

# Hard exit code for CI
if not (thr_impl['pass'] and thr_prec['pass']):
    sys.exit(1)
