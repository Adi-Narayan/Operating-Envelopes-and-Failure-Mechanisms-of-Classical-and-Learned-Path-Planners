"""
Tests 2 and 5 (both planners, same protocol as Table 4, 18 m/s):

  noise   : the 50 paired Corridor Gauntlet environments under four noise levels
            L0 (0 m, 0 deg), L1 (1 m, 2 deg), L2 (2.5 m, 5 deg), L3 (5 m, 10 deg)
            = (SD of perceived obstacle distance, SD of executed heading)
  layouts : 100 random layouts (random_layouts.py), static (deterministic) and dynamic
            (one obstacle-motion seed per layout, 10_000 + k)

APF variants: the paper's APF (full|nocap|apf) and APF under all of ARC's constraints
(rays|cap|arc). Learned planners: any list of checkpoints (ARC, TD3, SAC).

    python robustness_suite.py --test noise   --models <ckpts> --tag noise_arc
    python robustness_suite.py --test layouts --models <ckpts> --tag layouts_arc
    (--no-apf to skip the APF variants when adding more learned models later)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H             # noqa: E402
import apf_variants as V        # noqa: E402
from random_layouts import make_layout  # noqa: E402

NOISE_LEVELS = {"L0": (0.0, 0.0), "L1": (1.0, 2.0), "L2": (2.5, 5.0), "L3": (5.0, 10.0)}
APF_VARIANTS = [dict(sensing="full", turn_cap=False, rate="apf"), dict(sensing="rays", turn_cap=True, rate="arc")]


def brief(r):
    return {k: (float(r[k]) if isinstance(r[k], (np.floating, float)) else r[k])
            for k in ("outcome", "steps", "path_length", "efficiency", "min_clearance")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["noise", "layouts"], required=True)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--no-apf", action="store_true")
    ap.add_argument("--n-layouts", type=int, default=100)
    ap.add_argument("--n-envs", type=int, default=50, help="noise test: number of paired dynamic environments")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    models = []
    for m in args.models:
        models += sorted(glob.glob(m)) if any(c in m for c in "*?[") else [m]
    planners = [(m.replace("\\", "/"), H.ARCPlanner(m)) for m in models]
    apf = [] if args.no_apf else APF_VARIANTS
    out_path = os.path.join(HERE, "out", f"{args.tag}.json")
    res = {"test": args.test, "models": [m for m, _ in planners], "apf_variants": [V.label(v) for v in apf], "results": {}}

    def save():
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1, default=float)

    if args.test == "noise":
        for lev, (rn, hd) in NOISE_LEVELS.items():
            hn = np.deg2rad(hd)
            R = res["results"].setdefault(lev, {"range_noise_m": rn, "heading_noise_deg": hd, "runs": {}})
            for s in range(args.n_envs):
                tl = H.ObstacleTimeline(s)
                for v in apf:
                    R["runs"].setdefault("APF " + V.label(v), []).append(
                        brief(V.run(tl, 1000 + s, range_noise=rn, heading_noise=hn, noise_seed=50_000 + s, **v)))
                for m, P in planners:
                    R["runs"].setdefault(m, []).append(
                        brief(P.run(tl, range_noise=rn, heading_noise=hn, noise_seed=50_000 + s)))
            summ = {k: sum(r["outcome"] == "success" for r in v) for k, v in R["runs"].items()}
            R["success"] = summ
            print(lev, summ, flush=True)
            save()
    else:
        for k in range(args.n_layouts):
            obst = make_layout(k)
            row = res["results"].setdefault(str(k), {"static": {}, "dynamic": {}})
            st, dy = H.StaticTimeline(obst), H.ObstacleTimeline(10_000 + k, obst)
            for v in apf:
                row["static"]["APF " + V.label(v)] = brief(V.run(st, 0, **v))
                row["dynamic"]["APF " + V.label(v)] = brief(V.run(dy, 20_000 + k, **v))
            for m, P in planners:
                P._static_cache = None            # different layout each time
                row["static"][m] = brief(P.run(st, static=True))
                row["dynamic"][m] = brief(P.run(dy))
            if k % 10 == 9:
                print(f"layout {k + 1}/{args.n_layouts}", flush=True)
                save()
        tot = {}
        for part in ("static", "dynamic"):
            keys = res["results"]["0"][part].keys()
            tot[part] = {kk: sum(res["results"][str(k)][part][kk]["outcome"] == "success" for k in range(args.n_layouts))
                         for kk in keys}
        res["success"] = tot
        print(json.dumps(tot, indent=1))
        save()
    print("SUITE_DONE", args.tag)


if __name__ == "__main__":
    main()
