"""
Every evaluation of the paper, repeated for the Dynamic Window Approach (dwa.py), plain (DWA)
and with constant-velocity obstacle prediction (DWA-CV), with the weights selected by
dwa_tune.py (out/dwa_tune.json). Same environments and seeds as for APF and the learned
planners:

  static    : the static custom environment and the static second layout
  dynamic   : the 50 paired dynamic environments of the custom layout (seeds 0-49), and of the
              second layout, at 18 m/s
  speed     : the 50 custom-layout environments at the speeds of speed_sweep.py
  noise     : the 50 custom-layout environments at noise levels L1-L3 of robustness_suite.py
              (noise seed 50_000 + s); L0 is the dynamic run above
  layouts   : the 100 random layouts, static and dynamic (motion seed 10_000 + k)

    python dwa_eval.py --workers 4          -> out/dwa_eval.json
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SPEEDS = [3.0, 4.0, 4.75, 6.0, 9.0, 12.0, 27.0]           # 18 m/s is the main dynamic run
NOISE = {"L1": (1.0, 2.0), "L2": (2.5, 5.0), "L3": (5.0, 10.0)}
KEEP = ("outcome", "steps", "path_length", "efficiency", "min_clearance", "mean_speed", "time_s",
        "no_admissible", "ms_per_step")


def brief(r):
    return {k: (float(r[k]) if isinstance(r[k], float) else r[k]) for k in KEEP}


def task(t):
    kind, var, arg = t
    import harness as H
    import dwa
    from random_layouts import make_layout
    w = json.load(open(os.path.join(HERE, "out", "dwa_tune.json")))["selected"][var]["weights"]
    pred = var == "DWA-CV"
    H.set_ground_speed(18.0)
    if kind == "static":
        r = dwa.run(H.StaticTimeline(H.LAYOUTS[arg]), predict=pred, weights=w)
    elif kind == "dynamic":
        layout, s = arg
        r = dwa.run(H.ObstacleTimeline(s, H.LAYOUTS[layout]), predict=pred, weights=w)
    elif kind == "speed":
        v, s = arg
        H.set_ground_speed(v)
        r = dwa.run(H.ObstacleTimeline(s), predict=pred, weights=w)
    elif kind == "noise":
        lev, s = arg
        rn, hd = NOISE[lev]
        r = dwa.run(H.ObstacleTimeline(s), predict=pred, weights=w, range_noise=rn,
                    heading_noise=math.radians(hd), noise_seed=50_000 + s)
    else:                                                    # layouts
        part, k = arg
        obst = make_layout(k)
        tl = H.StaticTimeline(obst) if part == "static" else H.ObstacleTimeline(10_000 + k, obst)
        r = dwa.run(tl, predict=pred, weights=w)
    return t, brief(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tasks = []
    for var in ("DWA", "DWA-CV"):
        tasks += [("static", var, lay) for lay in ("corridor", "urban")]
        tasks += [("dynamic", var, (lay, s)) for lay in ("corridor", "urban") for s in range(50)]
        tasks += [("speed", var, (v, s)) for v in SPEEDS for s in range(50)]
        tasks += [("noise", var, (lev, s)) for lev in NOISE for s in range(50)]
        tasks += [("layouts", var, (part, k)) for part in ("static", "dynamic") for k in range(100)]
    # longest first (static layouts can run to the full budget), so the pool drains evenly
    tasks.sort(key=lambda t: 0 if t[0] in ("layouts", "static") else 1)
    res = {"weights": json.load(open(os.path.join(HERE, "out", "dwa_tune.json")))["selected"]}
    out_path = os.path.join(HERE, "out", "dwa_eval.json")
    with Pool(a.workers) as pool:
        for n, ((kind, var, arg), r) in enumerate(pool.imap_unordered(task, tasks, chunksize=2), 1):
            V = res.setdefault(var, {})
            if kind == "static":
                V.setdefault("static", {})[arg] = r
            elif kind == "dynamic":
                V.setdefault("dynamic", {}).setdefault(arg[0], {})[str(arg[1])] = r
            elif kind == "speed":
                V.setdefault("speed", {}).setdefault(str(arg[0]), {})[str(arg[1])] = r
            elif kind == "noise":
                V.setdefault("noise", {}).setdefault(arg[0], {})[str(arg[1])] = r
            else:
                V.setdefault("layouts", {}).setdefault(arg[0], {})[str(arg[1])] = r
            if n % 100 == 0 or n == len(tasks):
                print(f"{n}/{len(tasks)}", flush=True)
                with open(out_path, "w") as f:
                    json.dump(res, f, indent=1)
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
