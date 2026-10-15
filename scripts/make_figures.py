"""
WS-A: regenerate the paper's figures from the result files as vector PDFs (Arial, embedded fonts,
sized for RAS columns: single 90 mm, double 190 mm), plus PNG previews.

    python scripts/make_figures.py [--out ../figures]      (run scripts/make_numbers.py first)

Figures produced here:
  envelope_heatmap   success (%) of every planner on every envelope axis (results/numbers.json)
The experiment figures (trajectories, training curves, speed envelope, ...) are produced by
experiments/figures.py, which this script calls so that one command regenerates everything.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import EXP, ROOT  # noqa: E402

MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def heatmap(out):
    H = json.load(open(os.path.join(ROOT, "results", "numbers.json")))["heatmap"]
    rows, axes, cells = H["rows"], H["axes"], H["cells"]
    M = np.full((len(rows), len(axes)), np.nan)
    for i, r in enumerate(rows):
        for j, (a, _) in enumerate(axes):
            if a in cells.get(r, {}):
                M[i, j] = cells[r][a]["mean"]
    fig, ax = plt.subplots(figsize=(150 * MM, (14 + 7.2 * len(rows)) * MM))
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i, r in enumerate(rows):
        for j, (a, _) in enumerate(axes):
            c = cells.get(r, {}).get(a)
            if c is None:
                ax.text(j, i, "not run", ha="center", va="center", fontsize=6, color="#808080")
                continue
            col = "black" if c["mean"] > 55 else "white"
            ax.text(j, i - (0.13 if c["n_seeds"] > 1 else 0), f"{c['mean']:.0f}", ha="center", va="center",
                    fontsize=8, color=col, fontweight="bold")
            if c["n_seeds"] > 1:
                ax.text(j, i + 0.27, f"{c['min']:.0f}-{c['max']:.0f}", ha="center", va="center", fontsize=5.5, color=col)
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels([lab for _, lab in axes], fontsize=6.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(axes)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("success (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, out, "envelope_heatmap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(ROOT), "figures"))
    ap.add_argument("--skip-experiment-figures", action="store_true")
    a = ap.parse_args()
    heatmap(a.out)
    if not a.skip_experiment_figures:
        subprocess.check_call([sys.executable, os.path.join(EXP, "figures.py"), "--tag", "final_corridor_a"], cwd=EXP)
        m = {"env2d_apf": "fig1_static_final_corridor_a", "dyn_examples": "fig_dyn_examples", "baselines_seeds": "fig_baselines_seeds",
             "baselines_training": "fig_baselines_training", "speed_envelope": "fig_speed_envelope", "arc_training": "fig5_arc_training",
             "theta_sweep": "fig6_theta_sweep", "env3d": "fig_env3d", "apf_arc3d": "fig_apf_in_arc3d_stage2"}
        for p, o in m.items():
            for e in (".pdf", ".png"):
                shutil.copy2(os.path.join(EXP, "out", "figures", o + e), os.path.join(a.out, p + e))
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
