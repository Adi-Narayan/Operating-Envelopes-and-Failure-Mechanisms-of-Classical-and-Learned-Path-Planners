"""
WS-B: evaluation of one B4 (fixed TD3) or B5 (static SAC) training run, as pre-registered in prereg.md
("Evaluation of B4 and B5 policies"). Same run calls, seeds and rules as the paper's scripts
(eval_checkpoints.py + select_oracle.py, collapse_diagnostic.py, evaluate.py, speed_sweep.py,
robustness_suite.py); outputs go to results/v2 instead of experiments/out so nothing stored is
overwritten.

    python scripts/ws_b_eval.py <b4|b5> <seed>   -> results/v2/ws_b/eval_<b4|b5>/<run>.json

Per run:
  checkpoints : every saved checkpoint on the static custom environment, deterministic and from
                10 random headings (rng 0); oracle = select_oracle.py rule
  collapse    : collapse_diagnostic.py rule on the reward-selected checkpoint (arc_best)
  for each of arc_best (primary), ep300 and the oracle checkpoint, every axis:
    static custom, 50 dynamic environments (18 m/s), speed sweep, noise L1-L3, 100 random layouts
    (static and dynamic), second layout (static and dynamic)
'axes' holds the primary checkpoint's run lists under the heatmap axis names.
"""
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                            # noqa: E402
from random_layouts import make_layout                         # noqa: E402

GROUPS = {"b4": "td3fix_seed", "b5": "sacstatic_seed"}
SPEEDS = [3.0, 4.0, 4.75, 6.0, 9.0, 12.0, 18.0, 27.0]
NOISE = {"L1": (1.0, 2.0), "L2": (2.5, 5.0), "L3": (5.0, 10.0)}
CAP_DEG = 9.0
KEEP = ("outcome", "steps", "path_length", "efficiency", "min_clearance")


def brief(r):
    return {k: (float(r[k]) if isinstance(r[k], (np.floating, float)) else r[k]) for k in KEEP if k in r}


def ckpt_episode(name):
    m = re.search(r"ep(\d+)", name)
    return int(m.group(1)) if m else -1


def scan_checkpoints(model_dir):
    """eval_checkpoints.py protocol on the custom environment; select_oracle.py rule."""
    tl = H.StaticTimeline(H.LAYOUTS["corridor"])
    heads = np.random.default_rng(0).uniform(-math.pi, math.pi, size=10)
    rows = {}
    for f in sorted(os.listdir(os.path.join(model_dir, "models"))):
        if not f.endswith(".pth"):
            continue
        P = H.ARCPlanner(os.path.join(model_dir, "models", f))
        det = P.run(tl, static=True)
        rnd = [P.run(tl, heading0=float(h), static=True) for h in heads]
        rows[f] = {"deterministic": brief(det), "random_heading_success": sum(r["outcome"] == "success" for r in rnd),
                   "random_heading_outcomes": [r["outcome"] for r in rnd]}
    score = lambda f: (rows[f]["random_heading_success"], rows[f]["deterministic"]["outcome"] == "success", ckpt_episode(f))
    oracle = max(rows, key=score)
    return rows, oracle


def collapse(model_dir):
    """collapse_diagnostic.py rule on arc_best."""
    t = json.load(open(os.path.join(model_dir, "train_summary.json")))
    er = np.asarray(t["episode_rewards"], float)
    r = H.ARCPlanner(os.path.join(model_dir, "models", "arc_best.pth")).run(H.StaticTimeline(H.LAYOUTS["corridor"]), static=True)
    p = np.asarray(r["path"])
    hd = np.unwrap(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0])))
    turn = np.rad2deg(np.diff(hd))
    at_cap = float(np.mean(np.abs(np.abs(turn) - CAP_DEG) < 0.05)) if len(turn) else float("nan")
    return {"reward_first50": float(er[:50].mean()), "reward_last50": float(er[-50:].mean()), "reward_max": float(er.max()),
            "best_ckpt_outcome": r["outcome"], "frac_steps_at_cap": at_cap, "collapsed": bool(at_cap >= 0.99)}


def static_run(P, obstacles):
    P._static_cache = None            # ARCPlanner caches static geometry; reset per layout (as robustness_suite.py)
    return brief(P.run(H.StaticTimeline(obstacles), static=True))


def all_axes(path):
    P = H.ARCPlanner(path)
    urban = H.LAYOUTS["urban"]
    H.set_ground_speed(18.0)
    out = {"static_custom": static_run(P, H.LAYOUTS["corridor"]),
           "dynamic": [brief(P.run(H.ObstacleTimeline(s))) for s in range(50)],
           "noise": {lev: [brief(P.run(H.ObstacleTimeline(s), range_noise=rn, heading_noise=math.radians(hd), noise_seed=50_000 + s))
                           for s in range(50)] for lev, (rn, hd) in NOISE.items()},
           "layouts_static": [static_run(P, make_layout(k)) for k in range(100)],
           "layouts_dynamic": [brief(P.run(H.ObstacleTimeline(10_000 + k, make_layout(k)))) for k in range(100)],
           "second_static": static_run(P, urban),
           "second_dynamic": [brief(P.run(H.ObstacleTimeline(s, urban))) for s in range(50)],
           "speed": {}}
    for v in SPEEDS:
        H.set_ground_speed(v)
        out["speed"][str(v)] = [brief(P.run(H.ObstacleTimeline(s))) for s in range(50)]
    H.set_ground_speed(18.0)
    return out


def main():
    group, seed = sys.argv[1], int(sys.argv[2])
    run = f"{GROUPS[group]}{seed}"
    model_dir = os.path.join(ROOT, "results", "v2", "ws_b", group, run)
    assert os.path.exists(os.path.join(model_dir, "train_summary.json")), f"{run} has not finished training"
    t0 = now()
    rows, oracle = scan_checkpoints(model_dir)
    res = {"run": run, "checkpoints": rows, "oracle_checkpoint": oracle, "collapse": collapse(model_dir), "evaluations": {}}
    for label, f in (("arc_best", "arc_best.pth"), ("ep300", "arc_checkpoint_ep300.pth"), ("oracle", oracle)):
        if label == "oracle" and f in ("arc_best.pth", "arc_checkpoint_ep300.pth"):
            res["evaluations"]["oracle"] = {"same_as": f}
            continue
        res["evaluations"][label] = all_axes(os.path.join(model_dir, "models", f))
        print(run, label, "done", flush=True)
    e = res["evaluations"]["arc_best"]
    res["axes"] = {"indist": e["dynamic"], "r100": e["speed"]["4.75"], "r158": e["speed"]["3.0"], "noise5": e["noise"]["L3"],
                   "lay_static": e["layouts_static"], "lay_dynamic": e["layouts_dynamic"], "second_dyn": e["second_dynamic"]}
    d = outdir("ws_b", f"eval_{group}")
    dump(d, f"{run}.json", res)
    write_manifest(d, f"scripts/ws_b_eval.py {group} {seed}", {"protocol": "prereg.md, Evaluation of B4 and B5 policies"},
                   {"see": "notes/seed_map.md (frozen evaluation sets)"}, t0, name=f"manifest_{run}.json")
    print(run, json.dumps(res["collapse"]), {k: sum(r["outcome"] == "success" for r in v) for k, v in res["axes"].items()}, flush=True)


if __name__ == "__main__":
    main()
