"""
Where and why the paper's APF (full sensing, no turn cap, 10 Hz) fails on the 100 random static
layouts of random_layouts.py (the failures of out/layouts_apf_arc.json).

For every layout it reruns the deterministic static APF (planner seed 0, identical to
robustness_suite.py) and, for each failure, records the end point, its distance to the goal,
the number of distinct positions among the last 6 steps (2 = a two-point oscillation), the
magnitudes of the attractive, summed repulsive and total force at the end point, and whether
another planner in out/layouts_apf_arc.json / out/layouts_learned*.json reaches the goal on that
layout (i.e. whether the layout is solvable).

    python apf_traps.py  ->  out/apf_traps.json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.modules["tensorflow"] = None
import harness as H            # noqa: E402
import apf_variants as V       # noqa: E402
from random_layouts import make_layout  # noqa: E402

sta = H.sta


def main():
    others = {}
    for f in glob.glob(os.path.join(HERE, "out", "layouts_*.json")):
        if "factorial" in f or "collapsed10" in f:
            continue
        r = json.load(open(f))
        for k, row in r["results"].items():
            for m, d in row["static"].items():
                if m != "APF full|nocap|apf" and d["outcome"] == "success":
                    others.setdefault(int(k), []).append(m.replace("\\", "/"))
    cfg = sta.APFConfig()
    res = {"failures": {}}
    for k in range(100):
        obst = make_layout(k)
        r = V.run(H.StaticTimeline(obst), 0, sensing="full", turn_cap=False, rate="apf")
        if r["outcome"] == "success":
            continue
        p = np.asarray(r["path"])
        q = p[-1]
        _, fa, fr = sta.F_total(q, H.GOAL.astype(float), obst, cfg)
        res["failures"][k] = {"outcome": r["outcome"], "end": q.tolist(), "dist_to_goal": float(np.linalg.norm(q - H.GOAL)),
                              "distinct_last6": int(len({tuple(np.round(x, 6)) for x in p[-6:]})),
                              "F_att": float(np.linalg.norm(fa)), "F_rep": float(np.linalg.norm(fr)),
                              "F_total": float(np.linalg.norm(fa + fr)), "min_clearance": float(r["min_clearance"]),
                              "solved_by_another_planner": sorted(set(others.get(k, [])))}
    F = res["failures"]
    d = [v["dist_to_goal"] for v in F.values()]
    res["summary"] = {"n_failures": len(F), "outcomes": sorted({v["outcome"] for v in F.values()}),
                      "dist_to_goal_min_median_max": [min(d), float(np.median(d)), max(d)] if d else None,
                      "two_point_oscillations": sum(v["distinct_last6"] == 2 for v in F.values()),
                      "solvable": sum(bool(v["solved_by_another_planner"]) for v in F.values()),
                      "F_total_over_F_att_max": max(v["F_total"] / v["F_att"] for v in F.values()) if F else None}
    json.dump(res, open(os.path.join(HERE, "out", "apf_traps.json"), "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))
    for k, v in sorted(F.items())[:5]:
        print(k, {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "solved_by_another_planner"})


if __name__ == "__main__":
    main()
