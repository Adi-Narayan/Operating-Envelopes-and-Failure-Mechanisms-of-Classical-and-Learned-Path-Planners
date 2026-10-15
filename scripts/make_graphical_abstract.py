"""
Graphical abstract for the RAS submission, one panel per contribution:
  (1) the same paired environments order the planners differently on different axes (results/numbers.json);
  (2) an APF stall: the normalised force field around a stable node on random layout 16, the two-point
      oscillation of APF and the path of APF-D (beta = 0.85), both rerun here with the evaluation settings;
  (3) collapsed seeds out of 10, moving-obstacle training: TD3, TD3 with the pre-tanh penalty, SAC
      (Table 6 of the paper; the penalty count from results/numbers.json);
  (4) total successes of SAC alone, the arbiter (D1) and APF-D on SAC's rays, over the 10 SAC seeds and
      the seven heatmap axes (results/numbers.json: arbTotSac, arbTotDone, arbTotCapf).

    python scripts/make_graphical_abstract.py     -> results/v2/media/graphical_abstract.{pdf,png,tif}

Elsevier: at least 1328 x 531 px at 300 dpi in the ratio 500:200, Arial, no synopsis text or heading; the
image is reduced to 200 px high in the table of contents, so the labels are large and few.
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
                     "pdf.fonttype": 42, "font.size": 15, "axes.titlesize": 17.5, "axes.titleweight": "bold"})
STALL_LAYOUT = 16


def _clean(ax, left=True):
    for side in ("top", "right") + (() if left else ("left",)):
        ax.spines[side].set_visible(False)
    if not left:
        ax.set_yticks([])


def panel_rankings(ax, num):
    cells = num["heatmap"]["cells"]
    axes = [("indist", "In\ndistr."), ("r158", "Fast\nobst."), ("noise5", "Heavy\nnoise"), ("lay_static", "New\nlayouts")]
    rows = [("APF", "APF"), ("DWA-CV", "DWA-CV"), ("SAC moving", "SAC"), ("Arbiter D1", "Arbiter")]
    x = np.arange(len(axes))
    for key, lab in rows:
        y = [cells[key][a]["mean"] for a, _ in axes]
        ax.plot(x, y, "-o", color=M.COL[lab], lw=3.0, ms=8, label=lab, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in axes], fontsize=15)
    ax.set_xlim(-0.25, len(axes) - 0.75)
    ax.set_ylim(0, 142)                         # the band above 100 holds the legend
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylabel("Success (%)", fontsize=16)
    ax.yaxis.set_label_coords(-0.17, 0.36)
    ax.grid(axis="y", color="#e3e3e3", lw=0.8)
    _clean(ax)
    ax.spines["left"].set_bounds(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=14, frameon=False, ncol=2,
              handlelength=0.9, columnspacing=0.8, handletextpad=0.35, borderaxespad=0.0)
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
    ax.streamplot(gx, gy, U, Vv, color="#c3ccd5", density=0.8, linewidth=0.7, arrowsize=0.8, zorder=1)
    M.draw_obstacles(ax, ob, z=2)
    ax.plot(*apfd["path"].T, color=M.COL["APF-D"], lw=3.4, zorder=4)
    ax.plot(*apf["path"].T, color=M.COL["APF"], lw=3.4, zorder=5)
    ax.plot(*eq, "x", color="#d62728", ms=15, mew=3.4, zorder=7)
    box = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9)
    ax.text(eq[0] - 10, eq[1] - 2, "APF\nstalls", ha="right", va="top", fontsize=15, fontweight="bold",
            color=M.COL["APF"], zorder=8, bbox=box)
    ax.text(x0 + 4, y1 - 4, "APF +\nmomentum", ha="left", va="top", fontsize=15, fontweight="bold",
            color=M.COL["APF-D"], zorder=8, bbox=box)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#999999")
    ax.set_title("2  APF stalls", loc="left")
    return apf, apfd


def panel_collapse(ax, num):
    n = num["numbers"]
    # moving-obstacle training: TD3 5/10 (Table 6 of the paper); TD3 + penalty from numbers.json; SAC 0/10 (Table 6)
    bars = [("TD3", 5), ("TD3 +\npenalty", int(n["bFourCollapsed"])), ("SAC", 0)]
    cols = ["#c0392b", M.COL["APF-D"], "#8a8a8a"]
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], color=cols, width=0.62, zorder=3)
    for xi, (_, v) in zip(x, bars):
        ax.text(xi, v + 0.2, f"{v}/10", ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=15)
    ax.set_ylim(0, 8.2)
    _clean(ax, left=False)
    ax.text(1.5, 3.4, "tanh output\nsaturates;\npenalty\nfixes it", ha="center", va="center", fontsize=14.5,
            color="#333333")
    ax.set_title("3  TD3 collapses", loc="left")
    ax.text(0.0, 0.99, "collapsed seeds", transform=ax.transAxes, fontsize=15, color="#555555", va="top")


def panel_arbiter(ax, num):
    n = num["numbers"]
    t = {k: int(n[m].replace(",", "")) for k, m in (("sac", "arbTotSac"), ("d1", "arbTotDone"), ("capf", "arbTotCapf"))}
    vals = [("SAC", t["sac"], M.COL["SAC"]), ("Arbiter", t["d1"], M.COL["Arbiter"]), ("APF-D", t["capf"], M.COL["APF-D"])]
    x = np.arange(len(vals))
    ax.bar(x, [v[1] for v in vals], color=[v[2] for v in vals], width=0.62, zorder=3)
    for xi, (_, v, _) in zip(x, vals):
        ax.text(xi, v + 70, f"{v:,}", ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([v[0] for v in vals], fontsize=15)
    ax.set_ylim(0, 5600)
    _clean(ax, left=False)
    ax.set_title("4  Arbiter < APF-D", loc="left")
    ax.text(0.0, 0.99, "successes of 4,500", transform=ax.transAxes, fontsize=15, color="#555555", va="top")


def main():
    num = M.numbers()
    fig = plt.figure(figsize=(26.6 * CM, 10.6 * CM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.0, 0.9, 1.25], left=0.075, right=0.985, top=0.87,
                          bottom=0.17, wspace=0.28)
    panel_rankings(fig.add_subplot(gs[0]), num)
    panel_stall(fig.add_subplot(gs[1]))
    panel_collapse(fig.add_subplot(gs[2]), num)
    panel_arbiter(fig.add_subplot(gs[3]), num)
    os.makedirs(M.MEDIA, exist_ok=True)
    base = os.path.join(M.MEDIA, "graphical_abstract")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("graphical abstract ->", base + ".{pdf,png,tif}")


if __name__ == "__main__":
    main()
