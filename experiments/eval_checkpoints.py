"""
Static evaluation of every saved ARC checkpoint (ep50..ep300 and arc_best) on a layout,
from the fixed start (35, 35):
  * deterministic, initial heading toward the goal (the paper protocol)
  * 10 random initial headings (seeded), i.e. the start distribution used in training

    python eval_checkpoints.py --layout corridor --dirs out/arc_models/seed0 out/arc_models/seed2
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--layout", default="corridor", choices=sorted(H.LAYOUTS))
    ap.add_argument("--headings", type=int, default=10)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tl = H.StaticTimeline(H.LAYOUTS[args.layout])
    heads = np.random.default_rng(0).uniform(-math.pi, math.pi, size=args.headings)
    res = {"layout": args.layout, "headings_rad": heads.tolist(), "models": {}}
    for d in args.dirs:
        dd = sorted(glob.glob(d)) if any(c in d for c in "*?[") else [d]
        for dir_ in dd:
            ckpts = sorted(glob.glob(os.path.join(dir_, "**", "*.pth"), recursive=True),
                           key=lambda p: (("best" in p), int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0)))
            for c in ckpts:
                P = H.ARCPlanner(c)
                det = P.run(tl, static=True)
                rnd = [P.run(tl, heading0=float(h), static=True) for h in heads]
                ok = [r for r in rnd if r["success"]]
                key = os.path.relpath(c, HERE)
                res["models"][key] = {
                    "deterministic": {k: det[k] for k in ("outcome", "steps", "path_length", "efficiency", "min_clearance")},
                    "random_heading_success": len(ok),
                    "random_heading_outcomes": [r["outcome"] for r in rnd],
                    "random_heading_eff_success": H.mean_sd([r["efficiency"] for r in ok]),
                    "random_heading_L_success": H.mean_sd([r["path_length"] for r in ok]),
                }
                print(f"{key}: det={det['outcome']} ({det['steps']} st, {det['path_length']:.0f} m) | "
                      f"random-heading success {len(ok)}/{len(rnd)}", flush=True)
    tag = args.tag or f"checkpoints_{args.layout}"
    with open(os.path.join(HERE, "out", f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
