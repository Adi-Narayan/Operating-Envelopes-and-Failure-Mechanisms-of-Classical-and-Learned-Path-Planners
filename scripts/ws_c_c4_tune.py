"""
WS-C, C4 tuning runs for one candidate on the held-out sets of prereg.md (never the evaluation sets).

    python scripts/ws_c_c4_tune.py <variant>   -> results/v2/ws_c/c4_tuning/<variant>.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                      # noqa: E402
import apf_damped as D                                   # noqa: E402
from random_layouts import make_layout                   # noqa: E402

KEEP = ("outcome", "steps", "path_length", "efficiency", "min_clearance", "escape_steps", "pushouts", "capped_steps", "osc_triggers")


def brief(r):
    return {k: (float(r[k]) if isinstance(r[k], float) else r[k]) for k in KEEP if k in r}


def main():
    v = sys.argv[1]
    kw = D.VARIANTS[v]
    t0 = now()
    H.set_ground_speed(18.0)
    res = {"variant": v, "kwargs": kw, "static_layouts": [], "dynamic_layouts": [], "heldout_seeds": []}
    for j in range(50):
        obst = make_layout(500_000 + j)
        res["static_layouts"].append(brief(D.run(H.StaticTimeline(obst), 0, **kw)))
        res["dynamic_layouts"].append(brief(D.run(H.ObstacleTimeline(510_000 + j, obst), 520_000 + j, **kw)))
    for s in range(50):
        res["heldout_seeds"].append(brief(D.run(H.ObstacleTimeline(500_000 + s), 501_000 + s, **kw)))
    res["success"] = {k: sum(r["outcome"] == "success" for r in res[k]) for k in ("static_layouts", "dynamic_layouts", "heldout_seeds")}
    print(v, res["success"], flush=True)
    d = outdir("ws_c", "c4_tuning")
    dump(d, f"{v}.json", res)
    write_manifest(d, f"scripts/ws_c_c4_tune.py {v}", {"variants": D.VARIANTS, "speed": 18.0},
                   {"tuning_layouts": "500000-500049", "tuning_layout_motion": "510000-510049",
                    "tuning_layout_planner": "0 (static), 520000-520049 (dynamic)",
                    "heldout_motion": "500000-500049", "heldout_planner": "501000-501049"}, t0)


if __name__ == "__main__":
    main()
