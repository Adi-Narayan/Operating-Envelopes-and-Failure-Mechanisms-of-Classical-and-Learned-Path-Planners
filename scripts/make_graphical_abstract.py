"""
Graphical abstract for the RAS submission, one panel per contribution:
  (1) the same paired environments order the planners differently on different axes (results/numbers.json);
  (2) an APF stall: the normalised force field around a stable node on random layout 16, the two-point
      oscillation of APF and the path of APF-D (beta = 0.85), both rerun here with the evaluation settings;
  (3) collapsed seeds per learner and training condition (results/numbers.json, Table 6 of the paper);
  (4) total successes of SAC alone, the arbiter (D1) and APF-D on SAC's rays, over the 10 SAC seeds and
      the seven heatmap axes (results/numbers.json: arbTotSac, arbTotDone, arbTotCapf).

    python scripts/make_graphical_abstract.py     -> results/v2/media/graphical_abstract.{pdf,png,tif}

Elsevier: landscape, at least 1328 x 531 px, readable when shown at 13 x 5 cm. The figure is drawn at
twice that size with fonts of 11 pt or more, so they are at least 5.5 pt at the displayed size.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import media_common as M          # noqa: E402
import harness as H               # noqa: E402
import apf_variants as V          # noqa: E402
import apf_damped as Dmp          # noqa: E402
from random_layouts import make_layout  # noqa: E402

CM = 1 / 2.54
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 12, "axes.titlesize": 13.5, "axes.titleweight": "bold"})
STALL_LAYOUT = 16


def _clean(ax, left=True):
    for side in ("top", "right") + (() if left else ("left",)):
        ax.spines[side].set_visible(False)
    if not left:
        ax.set_yticks([])


def panel_rankings(ax, num):
    cells = num["heatmap"]["cells"]
    axes = [("indist", "In\ndistr."), ("r158", "Fast\nobstacles"), ("noise5", "Heavy\nnoise"),
            ("lay_static", "New static\nlayouts")]
    rows = [("APF", "APF"), ("DWA-CV", "DWA-CV"), ("SAC moving", "SAC"), ("Arbiter D1", "Arbiter")]
    x = np.arange(len(axes))
    for key, lab in rows:
        y = [cells[key][a]["mean"] for a, _ in axes]
        ax.plot(x, y, "-o", color=M.COL[lab], lw=2.2, ms=6, label=lab, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in axes], fontsize=10.5)
    ax.set_xlim(-0.3, len(axes) - 0.7)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Success (%)", fontsize=11.5)
    ax.grid(axis="y", color="#e3e3e3", lw=0.8)
    _clean(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=10.5, frameon=False, ncol=2,
              handlelength=1.2, columnspacing=0.7, borderaxespad=0.1)
    ax.set_title("1  Rankings change", loc="left")


def panel_stall(ax):
    ob = make_layout(STALL_LAYOUT)
    c = json.load(open(os.path.join(M.ROOT, "results", "v2", "ws_c", "c1_classify", "result.json")))["stalls"][str(STALL_LAYOUT)]
    eq = np.array(c["equilibrium"])
    apf = V.run(H.StaticTimeline(ob), 0, sensing="full", turn_cap=False, rate="apf")
    apfd = Dmp.run(H.StaticTimeline(ob), 0, momentum=0.85)
    assert apf["outcome"] == "timeout" and apfd["outcome"] == "success"
    x0, x1, y0, y1 = eq[0] - 95, eq[0] + 75, eq[1] - 100, eq[1] + 70
    gx, gy = np.meshgrid(np.linspace(x0, x1, 46), np.linspace(y0, y1, 46))
    cfg = H.dyn.APFConfig()
    U, Vv = np.zeros_like(gx), np.zeros_like(gy)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            q = np.array([gx[i, j], gy[i, j]])
            if H.clearance(q, ob) < 1.0:
                U[i, j] = Vv[i, j] = np.nan
                continue
            f = H.dyn.F_total(q, H.GOAL.astype(float), ob, cfg)[0]
            n = np.linalg.norm(f)
            U[i, j], Vv[i, j] = f / n if n > 0 else (0, 0)
    ax.streamplot(gx, gy, U, Vv, color="#c3ccd5", density=0.9, linewidth=0.7, arrowsize=0.8, zorder=1)
    M.draw_obstacles(ax, ob, z=2)
    ax.plot(*apfd["path"].T, color=M.COL["APF-D"], lw=2.6, zorder=4, label="APF + momentum")
    ax.plot(*apf["path"].T, color=M.COL["APF"], lw=2.6, zorder=5, label="APF")
    ax.plot(*eq, "x", color="#d62728", ms=12, mew=2.8, zorder=7)
    ax.text(eq[0] - 8, eq[1] - 4, "sink:\nAPF\noscillates", ha="right", va="top", fontsize=10.5, color="#7f1d1d",
            zorder=8, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#999999")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=10.5, frameon=False,
              handlelength=1.2, columnspacing=0.8)
    ax.set_title("2  APF stalls at sinks", loc="left")
    return apf, apfd


def panel_collapse(ax, num):
    n = num["numbers"]
    # TD3 static 10/10 and TD3 moving 5/10: Table 6 of the paper (results/v2/ws_b); penalty and SAC: numbers.json
    bars = [("TD3\nstatic", 10), ("TD3\nmoving", 5), ("TD3 +\npen.", int(n["bFourCollapsed"])), ("SAC", 0)]
    cols = ["#7f1d1d", "#c0392b", M.COL["APF-D"], "#8a8a8a"]
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], color=cols, width=0.66, zorder=3)
    for xi, (_, v) in zip(x, bars):
        ax.text(xi, v + 0.3, f"{v}/10", ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=10)
    ax.set_ylim(0, 13.5)
    _clean(ax, left=False)
    ax.text(2.5, 6.0, "tanh output\nsaturates early;\npre-tanh penalty\nfixes it", ha="center", va="center",
            fontsize=10.5, color="#333333")
    ax.set_title("3  TD3 collapses", loc="left")
    ax.text(0.0, 0.99, "collapsed seeds", transform=ax.transAxes, fontsize=10.5, color="#555555", va="top")


def panel_arbiter(ax, num):
    n = num["numbers"]
    t = {k: int(n[m].replace(",", "")) for k, m in (("sac", "arbTotSac"), ("d1", "arbTotDone"), ("capf", "arbTotCapf"))}
    vals = [("SAC", t["sac"], M.COL["SAC"]), ("Arbiter", t["d1"], M.COL["Arbiter"]), ("APF-D", t["capf"], M.COL["APF-D"])]
    x = np.arange(len(vals))
    ax.bar(x, [v[1] for v in vals], color=[v[2] for v in vals], width=0.66, zorder=3)
    for xi, (_, v, _) in zip(x, vals):
        ax.text(xi, v + 70, f"{v:,}", ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([v[0] for v in vals], fontsize=10.5)
    ax.set_ylim(0, 5400)
    _clean(ax, left=False)
    ax.set_title("4  Arbiter falls short", loc="left")
    ax.text(0.0, 0.99, "successes of 4,500 runs", transform=ax.transAxes, fontsize=10.5, color="#555555", va="top")


def main():
    num = M.numbers()
    fig = plt.figure(figsize=(26.6 * CM, 10.6 * CM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.1, 1.05, 0.95], left=0.06, right=0.99, top=0.74,
                          bottom=0.16, wspace=0.3)
    panel_rankings(fig.add_subplot(gs[0]), num)
    panel_stall(fig.add_subplot(gs[1]))
    panel_collapse(fig.add_subplot(gs[2]), num)
    panel_arbiter(fig.add_subplot(gs[3]), num)
    fig.text(0.5, 0.965, "Classical and learned path planners under one paired, seeded protocol",
             ha="center", va="top", fontsize=15, fontweight="bold", color="#1b1b1b")
    fig.text(0.5, 0.885, "APF, DWA-CV, SAC, TD3 and an envelope-derived arbiter on identical obstacle motion",
             ha="center", va="top", fontsize=11.5, color="#444444")
    os.makedirs(M.MEDIA, exist_ok=True)
    base = os.path.join(M.MEDIA, "graphical_abstract")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("graphical abstract ->", base + ".{pdf,png,tif}")


if __name__ == "__main__":
    main()
