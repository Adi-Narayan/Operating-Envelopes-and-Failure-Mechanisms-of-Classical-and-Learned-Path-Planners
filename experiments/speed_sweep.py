"""
Operating-envelope sweep: success vs. obstacle-to-agent speed ratio.

Both planners fly at the same ground speed v; obstacle speeds are fixed by the
motion model (max 4.75 m/s), so the ratio r = 4.75 / v is swept by varying v.
Same 50 seeded environments at every speed.

    python speed_sweep.py --layout corridor --models out/arc_models/seed*/models/arc_best.pth
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H   # noqa: E402

V_MAX_OBS = 4.75
SPEEDS = [3.0, 4.0, 4.75, 6.0, 9.0, 12.0, 18.0, 27.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--layout", default="corridor", choices=sorted(H.LAYOUTS))
    ap.add_argument("--envs", type=int, default=50)
    ap.add_argument("--speeds", type=float, nargs="+", default=SPEEDS)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    models = []
    for m in args.models:
        models += sorted(glob.glob(m)) if any(c in m for c in "*?[") else [m]
    tag = args.tag or f"speed_sweep_{args.layout}"
    obstacles = H.LAYOUTS[args.layout]
    planners = [H.ARCPlanner(m) for m in models]
    res = {"layout": args.layout, "models": models, "v_max_obstacle": V_MAX_OBS, "speeds": {}}
    for v in args.speeds:
        H.set_ground_speed(v)
        apf, arc = [], [[] for _ in models]
        for s in range(args.envs):
            tl = H.ObstacleTimeline(s, obstacles)
            apf.append(H.run_apf(tl, 1000 + s)["outcome"])
            for j, p in enumerate(planners):
                arc[j].append(p.run(tl)["outcome"])
        res["speeds"][str(v)] = {"ratio": V_MAX_OBS / v, "apf": apf, "arc": arc,
                                 "apf_success": apf.count("success"),
                                 "arc_success": [a.count("success") for a in arc]}
        print(f"v={v:5.2f} r={V_MAX_OBS / v:4.2f}  APF {apf.count('success')}/{args.envs}  ARC {[a.count('success') for a in arc]}", flush=True)
        with open(os.path.join(HERE, "out", f"{tag}.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
