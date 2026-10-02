"""
Regenerate the paper's 2D figures under the corrected protocol.

    python figures.py --tag final_corridor_a
Reads out/<tag>.json, out/<tag>_static_paths.npz, out/<tag>_dynamic_paths.npz,
out/arc_models/seed*/train_summary.json; writes out/figures/*.pdf (vector, fonts embedded, used by the
paper) and a 200 dpi *.png preview of each.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import matplotlib.ticker                              # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
import numpy as np                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H   # noqa: E402

OUT = os.path.join(HERE, "out")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
APF_C, ARC_C = "#1f5fa8", "#c2452d"
# Elsevier artwork guidance: Arial/Helvetica lettering, vector files with embedded (TrueType) fonts.
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, name, dpi=200, **kw):
    """Write <name>.pdf (vector) and <name>.png (preview)."""
    base = os.path.join(FIG, os.path.splitext(name)[0])
    fig.savefig(base + ".pdf", **kw)
    fig.savefig(base + ".png", dpi=dpi, **kw)
    plt.close(fig)


def draw_obs(ax, obstacles, **kw):
    for o in obstacles:
        style = dict(facecolor="#cfd4da", edgecolor="#4a5058", lw=1.2, alpha=0.9)
        if o.get("outlier"):
            style.update(facecolor="#f2d0cc", edgecolor="#9c3b30")
        style.update(kw)
        t = o["type"]
        if t == "circle":
            cx, cy, r = o["params"]
            ax.add_patch(Circle((cx, cy), r, **style))
        elif t in ("rectangle", "square"):
            cx, cy, hw, hh = o["params"]
            ax.add_patch(Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, **style))
        else:
            ax.add_patch(Polygon(o["params"], closed=True, **style))


def frame(ax, title=None):
    ax.set_xlim(0, 500); ax.set_ylim(0, 500); ax.set_aspect("equal")
    ax.plot(*H.START, "o", color="k", ms=6); ax.plot(*H.GOAL, "*", color="#2a8a3e", ms=14)
    ax.add_patch(Circle(H.GOAL, H.GOAL_RADIUS, fill=False, ls="--", color="#2a8a3e", lw=1))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title, fontsize=10)


def fig1(sp, obstacles, res=None):
    """APF path plus every ARC static run (among the evaluated models) that reached the goal."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    draw_obs(ax, obstacles)
    ax.plot(sp["apf"][:, 0], sp["apf"][:, 1], color=APF_C, lw=2, label="APF")
    if res is not None:
        drawn = False
        for i, (m, r) in enumerate(zip(res["models"], res["static"]["arc_runs"])):
            if r["success"] and f"arc{i}" in sp.files and "arc_best" in m and "seed" in m and "urban" not in m:
                ax.plot(sp[f"arc{i}"][:, 0], sp[f"arc{i}"][:, 1], color=ARC_C, lw=2,
                        label=None if drawn else "ARC (successful arc_best policies)")
                drawn = True
    frame(ax); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); save(fig, f"fig1_static_{TAG}.png")


def draw_pair(ax, dp, seed, title, obstacles):
    tl = H.ObstacleTimeline(seed, obstacles)
    a, r = dp[f"apf_{seed}"], dp[f"arc_{seed}"]
    draw_obs(ax, tl.at(0.0), alpha=0.35)
    t_end = (len(r) - 1) * H.ARC_DT
    draw_obs(ax, tl.at(t_end), facecolor="none", ls="--", alpha=1.0)
    ax.plot(a[:, 0], a[:, 1], color=APF_C, lw=2, label="APF")
    ax.plot(r[:, 0], r[:, 1], color=ARC_C, lw=2, label="ARC")
    ax.plot(*r[-1], "x", color=ARC_C, ms=9, mew=2)
    frame(ax, title); ax.legend(loc="upper left", fontsize=8)


def fig_pair(dp, seed, name, title, obstacles):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    draw_pair(ax, dp, seed, title, obstacles)
    fig.tight_layout(); save(fig, name)


def fig5():
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "arc_models", "seed*", "train_summary.json")))]
    runs = [r for r in runs if r["episodes"] >= 300]
    if not runs:
        return
    keys = [("collision_rate", "Collision rate"), ("episode_rewards", "Episode reward"), ("success_rate", "Success rate")]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True)
    for ax, (k, lab) in zip(axes, keys):
        M = np.array([r[k] for r in runs])
        ep = np.arange(1, M.shape[1] + 1)
        for row in M:
            ax.plot(ep, row, color="0.75", lw=0.6)
        ax.plot(ep, M.mean(0), color=ARC_C, lw=1.6, label=f"mean of {len(runs)} seeds")
        ax.set_ylabel(lab); ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8); axes[-1].set_xlabel("Episode")
    fig.tight_layout(); save(fig, "fig5_arc_training.png")


def fig6():
    sta = H.sta
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    draw_obs(ax, H.OBSTACLES)
    cmap = plt.get_cmap("viridis")
    thetas = list(range(0, 91, 15))
    for i, th in enumerate(thetas):
        cfg = sta.APFConfig(); cfg.theta = np.deg2rad(th)
        ok, p, _, _ = sta.simulate_static_apf(H.START.copy(), H.GOAL.copy(), sta.ENVIRONMENTS[1]["obstacles"], cfg,
                                             rng=np.random.default_rng(0))
        eff = 100 * H.D_STRAIGHT / sta.path_length(p)
        ax.plot(p[:, 0], p[:, 1], color=cmap(i / (len(thetas) - 1)), lw=1.6, label=f"θ={th}°  η={eff:.1f}%")
    frame(ax); ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); save(fig, "fig6_theta_sweep.png")


TD3_C, SAC_C = "#7a3fa0", "#2a8a3e"
METH = {"corridor": ("ARC (DDPG, static training)", ARC_C, "s"), "td3dyn": ("TD3, moving-obstacle training", TD3_C, "^"),
        "sacdyn": ("SAC, moving-obstacle training", SAC_C, "D")}


def run_of(path):
    """(method label as in report.ident, seed) for a checkpoint path."""
    import re
    p = path.replace("\\", "/")
    m = re.search(r"(env_apf_urban_|td3dyn_|td3_|sacdyn_|sac_)?seed(\d+)", p)
    pre = (m.group(1) or "").rstrip("_")
    return {"env_apf_urban": "urban", "": "corridor"}.get(pre, pre), int(m.group(2))


def speed_rows():
    """{speed: {'ratio', 'apf' (% of 50), (method, seed): % success}} using every speed_sweep_corridor*.json;
    each (method, seed) keeps its largest-sample file (50 environments where available, else the 10-env check)."""
    rows, n_of = {}, {}
    for f in sorted(glob.glob(os.path.join(OUT, "speed_sweep_corridor*.json"))):
        r = json.load(open(f))
        for v, d in r["speeds"].items():
            n = len(d["apf"])
            e = rows.setdefault(float(v), {"ratio": d["ratio"]})
            if n == 50:
                e["apf"] = 100 * d["apf_success"] / 50
            for m, k in zip(r["models"], d["arc_success"]):
                key = run_of(m)
                if n >= n_of.get((key, v), 0):
                    e[key] = 100 * k / n
                    n_of[(key, v)] = n
    return rows


def fig_speed():
    rows = speed_rows()
    if not rows:
        return
    v = sorted(rows)
    r = np.array([rows[x]["ratio"] for x in v])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(r, [rows[x]["apf"] for x in v], "o-", color=APF_C, lw=2, label="APF")
    for meth, (lab, col, mk) in METH.items():
        seeds = sorted({k[1] for x in v for k in rows[x] if isinstance(k, tuple) and k[0] == meth})
        if not seeds:
            continue
        M = np.array([[rows[x].get((meth, s), np.nan) for x in v] for s in seeds])
        for row in M:
            ax.plot(r, row, "-", color=col, alpha=0.22, lw=0.7)
        m = np.nanmean(M, 0)       # no SD band: seed outcomes are bimodal (collapsed vs learned)
        ax.plot(r, m, mk + "-", color=col, lw=2, ms=5, label=f"{lab} (mean of {len(seeds)} seed{'s' if len(seeds) != 1 else ''})")
    ax.axvline(1.0, color="0.5", ls=":", lw=1)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([0.2, 0.3, 0.5, 1.0, 1.5]); ax.set_xticklabels(["0.2", "0.3", "0.5", "1.0", "1.5"])
    ax.set_xlabel("obstacle-to-agent speed ratio  r = v_obs,max / v_agent")
    ax.set_ylabel("success over 50 paired environments (%)")
    ax.set_ylim(-3, 103); ax.grid(alpha=0.25)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=7.5, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1)); save(fig, "fig_speed_envelope.png")


def fig_baselines_seeds():
    """Dynamic-suite success per training seed (best checkpoint) for every learned method; collapsed seeds hollow."""
    rep = json.load(open(os.path.join(OUT, "report.json")))
    col = json.load(open(os.path.join(OUT, "collapse.json"))) if os.path.exists(os.path.join(OUT, "collapse.json")) else {}
    B = rep.get("baselines", {}).get("corridor", {})
    groups = [("ARC\n(static training)", rep["corridor"]["arc"]["corridor/best"]["dynamic"]["per_seed"], "seed", ARC_C),
              ("TD3\n(static training)", B.get("td3/best", {}).get("dynamic", {}).get("per_seed", {}), "td3_seed", "#9c6fb8"),
              ("TD3\n(moving obstacles)", B.get("td3dyn/best", {}).get("dynamic", {}).get("per_seed", {}), "td3dyn_seed", TD3_C),
              ("SAC\n(moving obstacles)", B.get("sacdyn/best", {}).get("dynamic", {}).get("per_seed", {}), "sacdyn_seed", SAC_C)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(100, color=APF_C, lw=2, label="APF (50/50)")
    rng = np.random.default_rng(0)
    for i, (lab, per, pre, c) in enumerate(groups):
        for s, d in per.items():
            y = 2 * d["success"]
            collapsed = col.get(f"{pre}{s}", {}).get("collapsed", False)
            ax.scatter(i + rng.uniform(-0.18, 0.18), y, s=36, facecolor="none" if collapsed else c, edgecolor=c, lw=1.4, zorder=3)
        if per:
            ys = [2 * d["success"] for d in per.values()]
            ax.plot([i - 0.3, i + 0.3], [np.mean(ys)] * 2, color="k", lw=1.5)
    ax.scatter([], [], facecolor="0.4", edgecolor="0.4", s=36, label="training seed")
    ax.scatter([], [], facecolor="none", edgecolor="0.4", s=36, lw=1.4, label="collapsed seed (circles at start)")
    ax.plot([], [], color="k", lw=1.5, label="mean over seeds")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("success over 50 paired\ndynamic environments (%)"); ax.set_ylim(-5, 105)
    ax.grid(alpha=0.2, axis="y"); ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); save(fig, "fig_baselines_seeds.png")


def fig_baselines_training():
    """Episode reward during training for TD3 (static), TD3 (moving obstacles) and SAC (moving obstacles);
    collapsed seeds drawn dashed."""
    col = json.load(open(os.path.join(OUT, "collapse.json"))) if os.path.exists(os.path.join(OUT, "collapse.json")) else {}
    panels = [("TD3, static training", "td3_seed", "#9c6fb8"), ("TD3, moving-obstacle training", "td3dyn_seed", TD3_C),
              ("SAC, moving-obstacle training", "sacdyn_seed", SAC_C)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, (title, pre, c) in zip(axes, panels):
        n_col = 0
        for s in range(10):
            p = os.path.join(OUT, "arc_models", f"{pre}{s}", "train_summary.json")
            if not os.path.exists(p):
                continue
            er = np.array(json.load(open(p))["episode_rewards"])
            sm = np.convolve(er, np.ones(10) / 10, mode="valid")
            collapsed = col.get(f"{pre}{s}", {}).get("collapsed", False)
            n_col += collapsed
            ax.plot(np.arange(10, 10 + len(sm)), sm, color="0.55" if collapsed else c, lw=1.0, ls="--" if collapsed else "-")
        ax.set_title(f"{title}\n({n_col} of 10 seeds collapsed)", fontsize=9); ax.set_xlabel("episode"); ax.grid(alpha=0.2)
    axes[0].set_ylabel("episode reward\n(10-episode moving mean)")
    fig.tight_layout(); save(fig, "fig_baselines_training.png")


def fig_dyn_examples(dp, panels, obstacles):
    """The paper's two-panel figure: the shared-success and ARC-collision examples side by side.
    panels: [(seed, title), (seed, title)]."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
    for ax, (seed, title), lab in zip(axes, panels, ("(a)", "(b)")):
        draw_pair(ax, dp, seed, f"{lab} {title}", obstacles)
    fig.tight_layout(pad=0.4); save(fig, "fig_dyn_examples.png")


def fig_arc3d():
    p = os.path.join(OUT, "apf_in_arc3d_paths.npz")
    if not os.path.exists(p):
        return
    import apf_in_arc3d as A
    path = np.load(p)["s2_calm"]
    fig = plt.figure(figsize=(6.2, 5.6)); ax = fig.add_subplot(projection="3d")
    for o in A.CFG["CURRICULUM"][2]["obstacles"]:
        t = o["type"]; c = o["center"]
        if t == "sphere":
            u, w = np.mgrid[0:2 * np.pi:16j, 0:np.pi:10j]
            ax.plot_wireframe(c[0] + o["radius"] * np.cos(u) * np.sin(w), c[1] + o["radius"] * np.sin(u) * np.sin(w),
                              c[2] + o["radius"] * np.cos(w), color="0.55", lw=0.4)
        elif t == "box":
            h = o["half"]
            for dx in (-1, 1):
                for dy in (-1, 1):
                    ax.plot([c[0] + dx * h[0]] * 2, [c[1] + dy * h[1]] * 2, [c[2] - h[2], c[2] + h[2]], color="0.45", lw=0.6)
            for dz in (-1, 1):
                xs = c[0] + h[0] * np.array([-1, 1, 1, -1, -1]); ys = c[1] + h[1] * np.array([-1, -1, 1, 1, -1])
                ax.plot(xs, ys, [c[2] + dz * h[2]] * 5, color="0.45", lw=0.6)
        elif t == "cylinder":
            th = np.linspace(0, 2 * np.pi, 24)
            for z in (c[2], c[2] + o["height"]):
                ax.plot(c[0] + o["radius"] * np.cos(th), c[1] + o["radius"] * np.sin(th), [z] * 24, color="0.45", lw=0.6)
            for a_ in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                ax.plot([c[0] + o["radius"] * np.cos(a_)] * 2, [c[1] + o["radius"] * np.sin(a_)] * 2,
                        [c[2], c[2] + o["height"]], color="0.45", lw=0.6)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color=APF_C, lw=2, label="APF")
    ax.scatter(*A.START, color="k", s=20); ax.scatter(*A.GOAL, color="#2a8a3e", marker="*", s=120)
    ax.set_xlim(0, 500); ax.set_ylim(0, 500); ax.set_zlim(0, 500)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)"); ax.legend(fontsize=8)
    ax.set_title("APF in the 3D ARC arena, stage 2 (23 obstacles)", fontsize=10)
    fig.tight_layout(); save(fig, "fig_apf_in_arc3d_stage2.png")


def fig_env3d():
    """The paper's own 3D environment (volumetric Corridor Gauntlet) with the APF static path."""
    import run_apf3d as A3
    core = A3.core
    cfg = core.APFConfig()
    ok, path, _ = core.simulate_static_apf(A3.START.copy(), A3.GOAL.copy(), core.STATIC_OBSTACLES_3D, cfg,
                                           rng=np.random.default_rng(0))
    fig = plt.figure(figsize=(6.2, 5.6)); ax = fig.add_subplot(projection="3d")
    for o in core.STATIC_OBSTACLES_3D:
        col = "#9c3b30" if o.get("outlier") else "0.5"
        if o["type"] == "sphere":
            cx, cy, cz, r = o["params"]
            u, w = np.mgrid[0:2 * np.pi:16j, 0:np.pi:10j]
            ax.plot_wireframe(cx + r * np.cos(u) * np.sin(w), cy + r * np.sin(u) * np.sin(w), cz + r * np.cos(w),
                              color=col, lw=0.4)
        else:
            cx, cy, cz, hx, hy, hz = o["params"]
            for dx in (-1, 1):
                for dy in (-1, 1):
                    ax.plot([cx + dx * hx] * 2, [cy + dy * hy] * 2, [cz - hz, cz + hz], color=col, lw=0.7)
            for dz in (-1, 1):
                ax.plot(cx + hx * np.array([-1, 1, 1, -1, -1]), cy + hy * np.array([-1, -1, 1, 1, -1]),
                        [cz + dz * hz] * 5, color=col, lw=0.7)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color=APF_C, lw=2, label="APF (static)")
    ax.scatter(*A3.START, color="k", s=20); ax.scatter(*A3.GOAL, color="#2a8a3e", marker="*", s=120)
    ax.set_xlim(0, 500); ax.set_ylim(0, 500); ax.set_zlim(0, 500)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)"); ax.legend(fontsize=8)
    fig.tight_layout(); save(fig, "fig_env3d.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="final_corridor_a"); args = ap.parse_args()
    TAG = args.tag
    res = json.load(open(os.path.join(OUT, f"{args.tag}.json")))
    sp = np.load(os.path.join(OUT, f"{args.tag}_static_paths.npz"))
    dp = np.load(os.path.join(OUT, f"{args.tag}_dynamic_paths.npz"))
    obstacles = H.LAYOUTS[res.get("layout", "corridor")]
    fig1(sp, obstacles, res)
    apf = res["dynamic"]["apf_runs"]; arc = res["dynamic"]["arc_runs_per_model"][res["models"].index(res["primary"])]
    shared = [s for s in range(len(apf)) if apf[s]["success"] and arc[s]["success"]]
    arc_coll = [s for s in range(len(apf)) if apf[s]["success"] and arc[s]["outcome"] == "collision"]
    panels = []
    if shared:   # the shared-success environment with median ARC-APF efficiency gap
        gaps = sorted(shared, key=lambda s: arc[s]["efficiency"] - apf[s]["efficiency"])
        s = gaps[len(gaps) // 2]
        panels.append((s, f"Environment {s}: both planners reach the goal"))
        fig_pair(dp, s, f"fig3_shared_success_{args.tag}.png", *panels[-1][1:], obstacles)
    if arc_coll:  # the ARC collision closest to the median collision step
        steps = sorted(arc_coll, key=lambda s: arc[s]["steps"])
        s = steps[len(steps) // 2]
        panels.append((s, f"Environment {s}: APF reaches the goal, ARC collides"))
        fig_pair(dp, s, f"fig4_arc_collision_{args.tag}.png", *panels[-1][1:], obstacles)
    fig5(); fig6(); fig_speed(); fig_arc3d(); fig_env3d(); fig_baselines_seeds(); fig_baselines_training()
    if len(panels) == 2:
        fig_dyn_examples(dp, panels, obstacles)
    print("figures ->", FIG, os.listdir(FIG))
