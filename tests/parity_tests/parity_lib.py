"""
parity_lib.py — Shared parity metrics and report generation.
No model imports. Used by test_impl_gpu.py and test_ane_mac.py.
"""
import math
from datetime import datetime


def compute_metrics(test, ref):
    """max_abs, mean_abs, cosine_sim between test and ref tensors."""
    diff = (test.float() - ref.float()).abs()
    a = test.float().flatten()
    b = ref.float().flatten()
    mask = (a != 0) | (b != 0)
    if mask.sum() == 0:
        cos = 1.0
    else:
        a, b = a[mask], b[mask]
        dot = (a * b).sum()
        cos = (dot / (a.norm() * b.norm()).clamp(min=1e-12)).item()
    return {'max_abs': diff.max().item(), 'mean_abs': diff.mean().item(), 'cosine_sim': cos}


def aggregate_metrics(per_step):
    """
    per_step: list of dicts {max_abs, mean_abs, cosine_sim} — one per measurement frame.
    Returns {max_abs (worst), mean_abs_avg, cosine_sim_min}.
    """
    return {
        'max_abs':        max(m['max_abs']    for m in per_step),
        'mean_abs_avg':   sum(m['mean_abs']   for m in per_step) / len(per_step),
        'cosine_sim_min': min(m['cosine_sim'] for m in per_step),
    }


def check_thresholds(agg, tol):
    """tol: dict with optional keys max_abs, mean_abs, cosine_sim."""
    fails = []
    if 'max_abs'    in tol and agg['max_abs']        > tol['max_abs']:
        fails.append(f"max_abs={agg['max_abs']:.2e} > {tol['max_abs']:.2e}")
    if 'mean_abs'   in tol and agg['mean_abs_avg']   > tol['mean_abs']:
        fails.append(f"mean_abs_avg={agg['mean_abs_avg']:.2e} > {tol['mean_abs']:.2e}")
    if 'cosine_sim' in tol and agg['cosine_sim_min'] < tol['cosine_sim']:
        fails.append(f"cosine_sim_min={agg['cosine_sim_min']:.6f} < {tol['cosine_sim']}")
    return {'pass': len(fails) == 0, 'failed_checks': fails}


def ascii_histogram(values, n_bins=8, width=30):
    if not values: return "(no data)"
    lo, hi = min(values), max(values)
    if lo == hi: return f"All = {lo:.3e}"
    bins = [0] * n_bins
    for v in values:
        bins[min(int((v - lo) / (hi - lo) * n_bins), n_bins - 1)] += 1
    mx = max(bins) or 1
    lines = []
    for i, c in enumerate(bins):
        low  = lo + (hi - lo) * i / n_bins
        high = lo + (hi - lo) * (i + 1) / n_bins
        lines.append(f"[{low:.2e},{high:.2e}) {c:4d}  {'#' * int(c/mx*width)}")
    return '\n'.join(lines)


def generate_report(title, comparisons, config_str, seq_str):
    """
    comparisons: list of dicts:
      {name, ref_label, test_label, tol, agg, threshold, per_step_max_abs}
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [f"# {title}", f"\nGenerated: {ts}",
             f"Config: {config_str}", f"Sequence: {seq_str}", ""]

    for c in comparisons:
        agg = c['agg']
        thr = c['threshold']
        status = "**PASS**" if thr['pass'] else "**FAIL**"
        lines.append(f"## {c['name']}: `{c['ref_label']}` → `{c['test_label']}`")
        lines.append(f"Tolerances: max_abs < {c['tol'].get('max_abs','—')}, "
                     f"cosine_sim > {c['tol'].get('cosine_sim','—')}")
        lines.append("")
        lines.append("| Metric | Value | Status |")
        lines.append("|--------|-------|--------|")
        lines.append(f"| max_abs (worst frame)     | {agg['max_abs']:.3e}    | {status} |")
        lines.append(f"| mean_abs_avg              | {agg['mean_abs_avg']:.3e} |  |")
        lines.append(f"| cosine_sim_min            | {agg['cosine_sim_min']:.6f} |  |")
        lines.append("")
        if thr['failed_checks']:
            lines.append("**Failed checks:**")
            for f in thr['failed_checks']:
                lines.append(f"- {f}")
            lines.append("")
        lines.append("### max_abs distribution across measurement frames")
        lines.append("```")
        lines.append(ascii_histogram(c['per_step_max_abs']))
        lines.append("```")
        lines.append("")

    return '\n'.join(lines) + '\n'
