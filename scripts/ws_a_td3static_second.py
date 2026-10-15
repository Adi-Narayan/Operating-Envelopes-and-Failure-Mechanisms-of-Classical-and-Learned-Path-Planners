"""
WS-A: the missing heatmap cell. TD3 (static training), reward-selected checkpoint of each of the 10
seeds, on the second layout's 50 dynamic environments (evaluate.py --layout urban protocol, 18 m/s).

    python scripts/ws_a_td3static_second.py  -> results/v2/ws_a/td3static_second/result.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import OUT, dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                           # noqa: E402

KEEP = ("outcome", "steps", "path_length", "efficiency", "min_clearance")


def main():
    t0 = now()
    H.set_ground_speed(18.0)
    urban = H.LAYOUTS["urban"]
    res = {}
    for s in range(10):
        P = H.ARCPlanner(os.path.join(OUT, "arc_models", f"td3_seed{s}", "models", "arc_best.pth"))
        runs = [{k: r[k] for k in KEEP} for r in (P.run(H.ObstacleTimeline(e, urban)) for e in range(50))]
        res[f"td3_seed{s}"] = {"success": sum(r["outcome"] == "success" for r in runs), "runs": runs}
        print(s, res[f"td3_seed{s}"]["success"], flush=True)
    d = outdir("ws_a", "td3static_second")
    dump(d, "result.json", {"per_seed": res, "axes_note": "second_dyn cell for the TD3 static row"})
    write_manifest(d, "scripts/ws_a_td3static_second.py", {"checkpoint": "arc_best", "layout": "urban", "speed": 18.0},
                   {"motion": "0-49 (frozen second-layout set)"}, t0)


if __name__ == "__main__":
    main()
