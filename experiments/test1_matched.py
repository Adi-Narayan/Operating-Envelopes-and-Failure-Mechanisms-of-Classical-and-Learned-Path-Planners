"""
Test 1: does APF's robustness survive when its advantages over ARC are removed?

12 APF variants (apf_variants.VARIANTS): sensing {full, range, rays} x turn-rate cap
{off, on} x step/decision rate {APF 1.8 m @ 10 Hz, ARC 3.0 m @ 6 Hz}, all at 18 m/s.
  * static: the five 2D layouts (one deterministic run each; planner seed 0)
  * dynamic: the 50 paired Corridor Gauntlet environments of Table 4 (seeds 0..49,
    planner seed 1000 + s, identical obstacle timelines for every variant)

    python test1_matched.py        ->  out/test1_matched.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H          # noqa: E402
import apf_variants as V     # noqa: E402

sta = H.sta


def strip(r):
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items() if k != "path"}


def main():
    res = {"variants": [V.label(v) for v in V.VARIANTS], "static": {}, "dynamic": {V.label(v): [] for v in V.VARIANTS}}
    for env_id in range(1, 6):
        name = sta.ENVIRONMENTS[env_id]["name"]
        tl = H.StaticTimeline(sta.ENVIRONMENTS[env_id]["obstacles"])
        res["static"][name] = {V.label(v): strip(V.run(tl, 0, **v)) for v in V.VARIANTS}
        print(name, {k: v["outcome"][:4] for k, v in res["static"][name].items()}, flush=True)
    for s in range(50):
        tl = H.ObstacleTimeline(s)
        row = []
        for v in V.VARIANTS:
            r = V.run(tl, 1000 + s, **v)
            res["dynamic"][V.label(v)].append(strip(r))
            row.append(r["outcome"][:4])
        print(f"env {s:2d}", " ".join(row), flush=True)
        with open(os.path.join(HERE, "out", "test1_matched.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)
    summary = {}
    for lab, runs in res["dynamic"].items():
        ok = [r for r in runs if r["success"]]
        summary[lab] = {"success": len(ok), "collision": sum(r["outcome"] == "collision" for r in runs),
                        "timeout": sum(r["outcome"] == "timeout" for r in runs),
                        "eff_success": H.mean_sd([r["efficiency"] for r in ok])}
    res["dynamic_summary"] = summary
    with open(os.path.join(HERE, "out", "test1_matched.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    for lab, d in summary.items():
        print(f"{lab:22s} success {d['success']:2d}/50 coll {d['collision']:2d} timeout {d['timeout']:2d}  eff {d['eff_success']}")
    print("TEST1_DONE")


if __name__ == "__main__":
    main()
