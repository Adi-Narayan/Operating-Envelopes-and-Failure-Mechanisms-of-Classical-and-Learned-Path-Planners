"""
WS-C, C3: does the rotation cause the free-space traps? APF (paper gains, full sensing, no turn cap,
10 Hz, planner seed 0) on the 100 random static layouts at theta = 0, 5, ..., 90 deg.
Characterisation only: the paper's APF stays at theta = 25 deg.

    python scripts/ws_c_theta_sweep.py  -> results/v2/ws_c/c3_theta_sweep/
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                      # noqa: E402
from random_layouts import make_layout                   # noqa: E402

THETAS = list(range(0, 91, 5))


def main():
    t0 = now()
    d = outdir("ws_c", "c3_theta_sweep")
    layouts = [make_layout(k) for k in range(100)]
    res = {"thetas": THETAS, "runs": {}}
    for th in THETAS:
        cfg = H.dyn.APFConfig()
        cfg.theta = np.deg2rad(th)
        rows = []
        for k, obst in enumerate(layouts):
            r = H.run_apf(H.StaticTimeline(obst), 0, cfg=cfg)
            p = r["path"]
            two_cycle = bool(len(p) > 8 and np.linalg.norm(p[-1] - p[-3]) < 1e-6 and np.linalg.norm(p[-2] - p[-4]) < 1e-6
                             and np.linalg.norm(p[-1] - p[-2]) > 1e-6)
            rows.append({"layout": k, "outcome": r["outcome"], "steps": r["steps"], "path_length": r["path_length"],
                         "two_cycle_at_end": two_cycle and r["outcome"] == "timeout"})
        res["runs"][str(th)] = rows
        n = {o: sum(x["outcome"] == o for x in rows) for o in ("success", "timeout", "collision")}
        print(f"theta {th:2d}: {n}  two-cycles {sum(x['two_cycle_at_end'] for x in rows)}", flush=True)
        dump(d, "result.json", res)
    res["summary"] = {str(th): {"success": sum(x["outcome"] == "success" for x in res["runs"][str(th)]),
                                "timeout": sum(x["outcome"] == "timeout" for x in res["runs"][str(th)]),
                                "collision": sum(x["outcome"] == "collision" for x in res["runs"][str(th)]),
                                "two_cycles": sum(x["two_cycle_at_end"] for x in res["runs"][str(th)])} for th in THETAS}
    dump(d, "result.json", res)
    write_manifest(d, "scripts/ws_c_theta_sweep.py", {"thetas_deg": THETAS, "gains": [5.0, 0.85], "sensing": "full",
                                                      "turn_cap": False, "rate": "10 Hz, 1.8 m"},
                   {"layout_generator": "0-99", "planner_seed": 0}, t0)


if __name__ == "__main__":
    main()
