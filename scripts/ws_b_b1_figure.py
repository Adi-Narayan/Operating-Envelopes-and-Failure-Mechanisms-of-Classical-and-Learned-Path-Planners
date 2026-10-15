"""
WS-B, B1 summary and figure: fraction of actions at the bound and of pre-tanh outputs with |u| > 3
against checkpoint episode, collapsed and learned seeds separately, per learner; critic values at the
bound vs zero action vs the Monte Carlo return.

    python scripts/ws_b_b1_figure.py [--out ../figures]  -> results/v2/ws_b/b1_summary.json, collapse_mechanism.pdf
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump  # noqa: E402

MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 7.5})
GROUPS = [("td3", "TD3, static training"), ("td3dyn", "TD3, moving obstacles"), ("sacdyn", "SAC, moving obstacles")]


def load(g):
    p = os.path.join(ROOT, "results", "v2", "ws_b", "b1_diagnostics", f"{g}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(ROOT), "figures"))
    a = ap.parse_args()
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(190 * MM, 58 * MM), sharey=True)
    for ax, (g, title) in zip(axes, GROUPS):
        D = load(g)
        if D is None:
            ax.set_title(title + "\n(not available)", fontsize=8)
            continue
        runs = {}
        for k, r in D.items():
            if r["episode"] is None:
                continue
            runs.setdefault(r["run"], []).append((r["episode"], r["frac_action_at_bound"], r["frac_abs_u_gt3"], r["collapsed_final"]))
        for run, pts in sorted(runs.items()):
            pts.sort()
            ep = [p[0] for p in pts]
            col = "#9c3b30" if pts[0][3] else "#1f5fa8"
            ax.plot(ep, [p[1] for p in pts], "-o", color=col, ms=2.5, lw=1, alpha=0.85)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("checkpoint episode")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        rows = list(D.values())
        coll = [r for r in rows if r["collapsed_final"]]
        lrn = [r for r in rows if r["collapsed_final"] is False]
        f = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs else None
        summary[g] = {"n_checkpoints": len(rows),
                      "collapsed_runs": {k: f(coll, k) for k in ("frac_action_at_bound", "frac_abs_u_gt3", "mean_Q_bound", "mean_Q_zero", "mean_Q_policy", "mean_G", "frac_Q_bound_gt_Q_zero")},
                      "learned_runs": {k: f(lrn, k) for k in ("frac_action_at_bound", "frac_abs_u_gt3", "mean_Q_bound", "mean_Q_zero", "mean_Q_policy", "mean_G", "frac_Q_bound_gt_Q_zero")},
                      "first_checkpoint_bound_fraction_collapsed": [min(v)[1] for v in runs.values() if v and v[0][3]],
                      "first_checkpoint_bound_fraction_learned": [min(v)[1] for v in runs.values() if v and v[0][3] is False]}
    axes[0].set_ylabel("fraction of actions at the bound")
    axes[0].plot([], [], "-o", color="#9c3b30", ms=2.5, label="collapsed seeds")
    axes[0].plot([], [], "-o", color="#1f5fa8", ms=2.5, label="seeds that learned")
    axes[0].legend(fontsize=6.5, loc="center right")
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "collapse_mechanism.pdf"))
    fig.savefig(os.path.join(a.out, "collapse_mechanism.png"), dpi=220)
    arc = load("arc")
    if arc:
        summary["arc"] = {k: {kk: v[kk] for kk in ("frac_action_at_bound", "frac_abs_u_gt3", "mean_Q_bound", "mean_Q_zero", "mean_G")}
                          for k, v in arc.items()}
    dump(os.path.join(ROOT, "results", "v2", "ws_b"), "b1_summary.json", summary)
    print(json.dumps({g: {k: v for k, v in s.items() if k in ("collapsed_runs", "learned_runs")} for g, s in summary.items() if g != "arc"}, indent=1))


if __name__ == "__main__":
    main()
