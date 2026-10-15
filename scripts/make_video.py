"""
Supplementary simulation video (MP4, 1920 x 1080, 30 fps), rendered from reruns of stored evaluation runs.

Scene 1: paired dynamic environment 37 at 18 m/s (r = 0.26). APF (planner seed 1037), DWA-CV (selected
         weights), SAC (moving-obstacle seed 0, reward-selected checkpoint) and the arbiter D1 (the same SAC
         policy with APF-D on its rays, selected thresholds) on the identical obstacle timeline. The
         environment was chosen to show differing outcomes: SAC collides, the other three reach the goal.
Scene 2: random static layout 16. APF (planner seed 0) stalls at a stable node of its rotated force field
         and oscillates between two points; APF-D (beta = 0.85) reaches the goal.

Every run is checked against the stored outcome (results/v2/ws_d/d4_eval, results/v2/ws_c) before
rendering; the script stops if any differs.

    python scripts/make_video.py   -> results/v2/media/simulation_video.mp4 (720p), simulation_video_still.png
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import media_common as M          # noqa: E402
from v2common import OUT          # noqa: E402
import harness as H               # noqa: E402
import arbiter as A               # noqa: E402
import dwa                        # noqa: E402
import apf_variants as V          # noqa: E402
import apf_damped as Dmp          # noqa: E402
from random_layouts import make_layout  # noqa: E402

FPS = 30
W, HGT = 1920, 1080
ENV, SAC_SEED, LAYOUT = 37, 0, 16
V2 = os.path.join(M.ROOT, "results", "v2")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
OUTCOME = {"success": ("reached the goal", M.COL["goal"]), "collision": ("collision", "#d62728"),
           "out_of_bounds": ("left the arena", "#d62728"), "timeout": ("stalled", "#d62728")}


# ------------------------------------------------------------------ simulation
def simulate_scene1():
    H.set_ground_speed(18.0)
    cc = json.load(open(os.path.join(V2, "ws_d", "c_config.json")))
    th = json.load(open(os.path.join(V2, "ws_d", "d3_selection", "result.json")))["selected"]
    model = os.path.join(OUT, "arc_models", f"sacdyn_seed{SAC_SEED}", "models", "arc_best.pth")
    w = json.load(open(os.path.join(OUT, "dwa_tune.json")))["selected"]["DWA-CV"]["weights"]
    runs = {}
    r = H.run_apf(H.ObstacleTimeline(ENV), 1000 + ENV)
    runs["APF"] = dict(path=r["path"], dt=H.APF_DT, outcome=r["outcome"])
    r = dwa.run(H.ObstacleTimeline(ENV), predict=True, weights=w, record=True)
    runs["DWA-CV"] = dict(path=r["path"], dt=H.APF_DT, outcome=r["outcome"])
    r = A.run(H.ObstacleTimeline(ENV), A.Learned(model), A.APFRays(momentum=cc["momentum"]), "apf_rays",
              -np.inf, np.inf, c_seed=1000 + ENV)                       # SAC alone, as in the D4 evaluation
    runs["SAC"] = dict(path=r["path"], dt=H.ARC_DT, outcome=r["outcome"])
    r = A.run(H.ObstacleTimeline(ENV), A.Learned(model), A.APFRays(momentum=cc["momentum"]), "apf_rays",
              th["d_on"], th["rho_on"], d_off=th["d_off"], rho_off=th["rho_off"], c_seed=1000 + ENV)
    tick = A.TICKS["L"]
    assert A.TICKS["apf_rays"] == tick
    sw = sorted((k, d) for d, k, *_ in r["switch_log"])
    modes, active, j = [], "L", 0
    for i in range(len(r["path"]) - 1):
        while j < len(sw) and sw[j][0] <= i * tick:
            active = "C" if sw[j][1] == "L->C" else "L"
            j += 1
        modes.append(active)
    runs["Arbiter"] = dict(path=r["path"], dt=H.ARC_DT, outcome=r["outcome"], modes=modes,
                           frac_c=r["frac_time_C"], switches=r["switches"])
    stored = {"SAC": json.load(open(os.path.join(V2, "ws_d", "d4_eval", f"sac_sac{SAC_SEED}.json"))),
              "Arbiter": json.load(open(os.path.join(V2, "ws_d", "d4_eval", f"d1_sac{SAC_SEED}.json"))),
              "DWA-CV": json.load(open(os.path.join(V2, "ws_d", "d4_eval", "dwacv.json")))}
    for k, d in stored.items():
        assert d["evaluation"]["dynamic"][ENV]["outcome"] == runs[k]["outcome"], (k, runs[k]["outcome"])
    assert runs["APF"]["outcome"] == "success"
    return runs


def simulate_scene2():
    ob = make_layout(LAYOUT)
    a = V.run(H.StaticTimeline(ob), 0, sensing="full", turn_cap=False, rate="apf")
    d = Dmp.run(H.StaticTimeline(ob), 0, momentum=0.85)
    st = json.load(open(os.path.join(V2, "ws_c", "apfd_eval", "layouts.json")))["static"][LAYOUT]["outcome"]
    c = json.load(open(os.path.join(V2, "ws_c", "c1_classify", "result.json")))["stalls"][str(LAYOUT)]
    assert a["outcome"] == "timeout" and d["outcome"] == st == "success" and c["type"] == "stable node"
    eq = np.array(c["equilibrium"])
    t_stall = H.APF_DT * int(np.argmax(np.linalg.norm(a["path"] - eq, axis=1) < 2.0))
    return ob, eq, {"APF": dict(path=a["path"], dt=H.APF_DT, outcome="timeout", t_stall=t_stall),
                    "APF-D": dict(path=d["path"], dt=H.APF_DT, outcome="success")}


# ------------------------------------------------------------------ drawing helpers
def pos_at(run, t):
    p, dt = run["path"], run["dt"]
    u = t / dt
    i = min(int(u), len(p) - 1)
    if i >= len(p) - 1:
        return p[-1], len(p) - 1
    f = u - i
    return p[i] * (1 - f) + p[i + 1] * f, i


def end_time(run):
    return (len(run["path"]) - 1) * run["dt"]


class ObstacleArtists:
    def __init__(self, ax, obstacles):
        self.arts = M.draw_obstacles(ax, obstacles, z=2)

    def update(self, obstacles):
        for a, o in zip(self.arts, obstacles):
            t, p = o["type"], o["params"]
            if t == "circle":
                a.set_center((p[0], p[1]))
            elif t in ("rectangle", "square"):
                a.set_xy((p[0] - p[2], p[1] - p[3]))
            else:
                a.set_xy(np.asarray(p))


def panel(fig, rect, name, obstacles, colour):
    ax = fig.add_axes(rect)
    M.arena(ax)
    M.mark_start_goal(ax, size=1.3)
    oa = ObstacleArtists(ax, obstacles)
    trail, = ax.plot([], [], color=colour, lw=2.4, zorder=4, solid_capstyle="round")
    dot, = ax.plot([], [], "o", color=colour, ms=10, mec="white", mew=1.5, zorder=8)
    ax.set_title(name, fontsize=20, fontweight="bold", color=colour, loc="left", pad=6)
    badge = ax.text(0.98, 0.03, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=17, fontweight="bold",
                    zorder=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9))
    return dict(ax=ax, oa=oa, trail=trail, dot=dot, badge=badge)


def text_card(writer, lines, seconds):
    fig = plt.figure(figsize=(W / 100, HGT / 100), dpi=100)
    fig.patch.set_facecolor("white")
    y = 0.62
    for txt, size, weight, col in lines:
        fig.text(0.5, y, txt, ha="center", va="center", fontsize=size, fontweight=weight, color=col, wrap=True)
        y -= 0.05 + size / 400
    emit(writer, fig, int(seconds * FPS))
    plt.close(fig)


def emit(writer, fig, n=1):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    for _ in range(n):
        writer.send(buf)


# ------------------------------------------------------------------ scenes
def scene1(writer, runs, num):
    tl = H.ObstacleTimeline(ENV)
    fig = plt.figure(figsize=(W / 100, HGT / 100), dpi=100)
    fig.patch.set_facecolor("white")
    s = 0.44
    rects = {"APF": [0.02, 0.50, s * HGT / W, s], "DWA-CV": [0.27, 0.50, s * HGT / W, s],
             "SAC": [0.02, 0.03, s * HGT / W, s], "Arbiter": [0.27, 0.03, s * HGT / W, s]}
    names = {"APF": "APF", "DWA-CV": "DWA-CV", "SAC": f"SAC (seed {SAC_SEED})", "Arbiter": "Arbiter: SAC + APF-D"}
    P = {k: panel(fig, rects[k], names[k], tl.at(0.0), M.COL[k]) for k in runs}
    arb = P["Arbiter"]
    arb["trail"].set_visible(False)
    lc = LineCollection([], linewidths=2.4, zorder=4)
    arb["ax"].add_collection(lc)
    seg_col = {"L": M.COL["SAC"], "C": M.COL["APF-D"]}
    cells = num["heatmap"]["cells"]
    x0 = 0.53
    fig.text(x0, 0.93, "Same environment, same obstacle motion", fontsize=30, fontweight="bold", va="top")
    fig.text(x0, 0.86, f"Paired dynamic environment {ENV}: agents at 18 m/s, fastest obstacle\n"
             "4.75 m/s (r = 0.26). Every planner faces the identical seeded obstacle timeline.",
             fontsize=17, va="top", color="#333333", linespacing=1.5)
    clock = fig.text(x0, 0.73, "", fontsize=26, va="top", family="monospace")
    status = {k: fig.text(x0, 0.63 - 0.06 * i, "", fontsize=19, va="top") for i, k in enumerate(runs)}
    fig.text(x0, 0.37, "Arbiter path colour: red while SAC is in control, teal while APF-D is in control.\n"
             "Control passes to APF-D when an obstacle is close or closing fast.", fontsize=16, va="top",
             color="#333333", linespacing=1.5)
    fig.text(x0, 0.20, "This environment was chosen to show differing outcomes. Over all 50 paired\n"
             f"environments: APF {cells['APF']['indist']['mean']:.0f}%, DWA-CV {cells['DWA-CV']['indist']['mean']:.0f}%, "
             f"SAC {cells['SAC moving']['indist']['mean']:.0f}% (mean of 10 seeds), "
             f"arbiter {cells['Arbiter D1']['indist']['mean']:.0f}%.\nPlayback at twice real time.",
             fontsize=15, va="top", color="#555555", linespacing=1.5)
    T = max(end_time(r) for r in runs.values())
    speed = 2.0
    nfr = int(math.ceil(T / speed * FPS)) + 1
    for f in range(nfr + int(2.5 * FPS)):
        t = min(f * speed / FPS, T)
        obs = tl.at(t)
        clock.set_text(f"t = {t:5.1f} s")
        for k, r in runs.items():
            te = end_time(r)
            q, i = pos_at(r, min(t, te))
            pp = np.vstack([r["path"][:i + 1], q])
            P[k]["oa"].update(tl.at(min(t, te)))
            if k == "Arbiter":
                segs = [pp[j:j + 2] for j in range(len(pp) - 1)]
                lc.set_segments(segs)
                lc.set_color([seg_col[r["modes"][min(j, len(r["modes"]) - 1)]] for j in range(len(segs))])
                P[k]["dot"].set_color(seg_col[r["modes"][min(i, len(r["modes"]) - 1)]])
            else:
                P[k]["trail"].set_data(pp[:, 0], pp[:, 1])
            P[k]["dot"].set_data([q[0]], [q[1]])
            if t >= te:
                lab, col = OUTCOME[r["outcome"]]
                P[k]["badge"].set_text(f"{lab[0].upper() + lab[1:]} at {te:.1f} s")
                P[k]["badge"].set_color(col)
                status[k].set_text(f"{k}: {lab} ({te:.1f} s)")
                status[k].set_color(col)
            else:
                status[k].set_text(f"{k}: running")
                status[k].set_color("#333333")
        del obs
        emit(writer, fig)
    plt.close(fig)


def scene2(writer, ob, eq, runs):
    fig = plt.figure(figsize=(W / 100, HGT / 100), dpi=100)
    fig.patch.set_facecolor("white")
    s = 0.70
    w = s * HGT / W
    left = (1.0 - 2 * w - 0.04) / 2
    rects = {"APF": [left, 0.03, w, s], "APF-D": [left + w + 0.04, 0.03, w, s]}
    names = {"APF": "APF", "APF-D": "APF-D (heading momentum, beta = 0.85)"}
    P = {k: panel(fig, rects[k], names[k], ob, M.COL[k]) for k in runs}
    fig.text(left, 0.965, f"Free-space stall: unseen static layout {LAYOUT}", fontsize=30, fontweight="bold", va="top")
    fig.text(left, 0.905, "APF settles at a stable sink of its rotated force field and oscillates between two points "
             "one 1.8 m step apart.\nHeading momentum prevents the reversal, and APF-D passes the same point. Over "
             "the 100 unseen static\nlayouts, APF stalls on 32 and APF-D removes 20 of them. Playback at four times "
             "real time.", fontsize=16, va="top", color="#333333", linespacing=1.45)
    clock = fig.text(left + 2 * w + 0.04, 0.965, "", fontsize=26, va="top", ha="right", family="monospace")
    sink = None
    T = end_time(runs["APF-D"]) + 6.0
    speed = 4.0
    nfr = int(math.ceil(T / speed * FPS)) + 1
    for f in range(nfr + int(2.0 * FPS)):
        t = min(f * speed / FPS, T)
        clock.set_text(f"t = {t:5.1f} s")
        for k, r in runs.items():
            te = min(end_time(r), T)
            q, i = pos_at(r, min(t, te))
            pp = np.vstack([r["path"][:i + 1], q])
            P[k]["trail"].set_data(pp[:, 0], pp[:, 1])
            P[k]["dot"].set_data([q[0]], [q[1]])
        if t >= runs["APF"]["t_stall"] + 4.0:
            if sink is None:
                sink = P["APF"]["ax"].plot(*eq, "x", color="#d62728", ms=18, mew=3.5, zorder=9)[0]
            P["APF"]["badge"].set_text("Stalled: two-point oscillation\n(run times out at 600 s)")
            P["APF"]["badge"].set_color("#d62728")
        if t >= end_time(runs["APF-D"]):
            P["APF-D"]["badge"].set_text(f"Reached the goal at {end_time(runs['APF-D']):.1f} s")
            P["APF-D"]["badge"].set_color(M.COL["goal"])
        emit(writer, fig)
    plt.close(fig)


def main():
    import imageio_ffmpeg
    num = M.numbers()
    print("simulating scene 1 ...", flush=True)
    r1 = simulate_scene1()
    for k, r in r1.items():
        print(f"  {k}: {r['outcome']} at {end_time(r):.1f} s", flush=True)
    print("simulating scene 2 ...", flush=True)
    ob, eq, r2 = simulate_scene2()
    os.makedirs(M.MEDIA, exist_ok=True)
    out = os.path.join(M.MEDIA, "simulation_video_master.tmp.mp4")     # 1080p master, removed after encoding
    writer = imageio_ffmpeg.write_frames(out, (W, HGT), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                         quality=8, macro_block_size=8)
    writer.send(None)
    title = "Operating Envelopes and Failure Mechanisms of Classical and Learned Path Planners"
    text_card(writer, [(title, 28, "bold", "#1b1b1b"),
                       ("Simulation examples from the paired evaluation protocol", 20, "normal", "#444444")], 3.5)
    text_card(writer, [("1  Same obstacle motion, different outcomes", 30, "bold", "#1b1b1b"),
                       ("APF, DWA-CV, SAC and the envelope-derived arbiter on one paired environment", 20, "normal",
                        "#444444")], 2.5)
    print("rendering scene 1 ...", flush=True)
    scene1(writer, r1, num)
    text_card(writer, [("2  Why APF stalls in free space", 30, "bold", "#1b1b1b"),
                       ("APF and APF-D on one unseen static layout", 20, "normal", "#444444")], 2.5)
    print("rendering scene 2 ...", flush=True)
    scene2(writer, ob, eq, r2)
    text_card(writer, [("Every run in this video is a rerun of a stored evaluation run", 22, "bold", "#1b1b1b"),
                       ("Reproduced by scripts/make_video.py in the code repository", 18, "normal", "#444444")], 3.0)
    writer.close()
    finalise(out, imageio_ffmpeg.get_ffmpeg_exe())


def finalise(master, ffmpeg):
    """Elsevier media specifications: MP4 with H.264 + AAC, at most 720p, at least 15 fps, bit rate at least
    260 kbps (750 kbps preferred), at most 5 minutes and 150 MB; a still frame serves as the thumbnail.
    The 1080p master is re-encoded to 720p at a constant 1 Mbit/s with a silent AAC track."""
    import subprocess
    out = os.path.join(M.MEDIA, "simulation_video.mp4")
    tmp = os.path.join(M.MEDIA, "simulation_video_720p.tmp.mp4")
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", master, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                    "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-preset", "slow", "-profile:v", "high",
                    "-pix_fmt", "yuv420p", "-b:v", "1000k", "-minrate", "1000k", "-maxrate", "1000k", "-bufsize", "2000k",
                    "-x264-params", "nal-hrd=cbr", "-r", str(FPS),
                    "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, out)
    os.remove(master)
    still = os.path.join(M.MEDIA, "simulation_video_still.png")
    subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", "30", "-i", out, "-frames:v", "1", still], check=True)
    print("video ->", out, "\nstill ->", still, flush=True)


if __name__ == "__main__":
    main()
