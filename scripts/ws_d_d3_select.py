"""
WS-D, D3: apply the pre-registered selection rule (prereg.md, WS-D D3) to the tuning runs.
Objective: mean success over the five held-out sets and the 10 SAC seeds; ties -> fewer switches per run.

    python scripts/ws_d_d3_select.py  -> results/v2/ws_d/d3_selection/result.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump, now, outdir, write_manifest  # noqa: E402

D3 = os.path.join(ROOT, "results", "v2", "ws_d", "d3_tuning")
SETS = ("heldout", "layouts_static", "layouts_dynamic", "noise", "r100")


def main():
    t0 = now()
    rows = {}
    for g in range(9):
        files = [os.path.join(D3, f"g{g}_sac{s}.json") for s in range(10)]
        missing = [f for f in files if not os.path.exists(f)]
        assert not missing, f"grid point {g}: {len(missing)} runs missing"
        R = [json.load(open(f)) for f in files]
        runs = [r for x in R for k in SETS for r in x["sets"][k]]
        rows[g] = {"d_on": R[0]["d_on"], "rho_on": R[0]["rho_on"],
                   "mean_success": float(np.mean([r["outcome"] == "success" for r in runs])),
                   "success_by_set": {k: int(sum(x["success"][k] for x in R)) for k in SETS},
                   "success_by_seed": [int(sum(x["success"].values())) for x in R],
                   "mean_switches": float(np.mean([r["switches"] for r in runs])),
                   "mean_frac_time_C": float(np.mean([r["frac_time_C"] for r in runs])),
                   "chatter_per_run": float(np.mean([r["chatter"] for r in runs]))}
    best = max(rows, key=lambda g: (rows[g]["mean_success"], -rows[g]["mean_switches"]))
    top = [g for g in rows if rows[g]["mean_success"] == rows[best]["mean_success"]]
    res = {"rule": "prereg.md WS-D D3: max mean success over five sets x 10 SAC seeds; ties -> fewer switches",
           "grid": rows, "selected_index": best, "tied_on_success": top,
           "selected": {"d_on": rows[best]["d_on"], "rho_on": rows[best]["rho_on"],
                        "d_off": rows[best]["d_on"] + 5.0, "rho_off": 0.8 * rows[best]["rho_on"]},
           "selected_on_grid_edge": {"d_on": best // 3 in (0, 2), "rho_on": best % 3 in (0, 2)}}
    d = outdir("ws_d", "d3_selection")
    dump(d, "result.json", res)
    write_manifest(d, "scripts/ws_d_d3_select.py", {"rule": res["rule"]}, {"inputs": "results/v2/ws_d/d3_tuning"}, t0)
    for g, r in rows.items():
        print(g, round(r["d_on"], 2), round(r["rho_on"], 3), round(r["mean_success"], 4), r["success_by_set"], round(r["mean_switches"], 2))
    print("SELECTED", best, res["selected"], "ties:", top)


if __name__ == "__main__":
    main()
