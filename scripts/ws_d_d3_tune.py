"""
WS-D, D3: one grid point x one SAC seed on the five held-out tuning sets (prereg.md).

    python scripts/ws_d_d3_tune.py <grid index 0-8> <sac seed 0-9>
        -> results/v2/ws_d/d3_tuning/g<grid>_sac<seed>.json
Reads the thresholds grid from results/v2/ws_d/d2_thresholds/result.json and the fallback
configuration from results/v2/ws_d/c_config.json (written when the C4 rule has been applied).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import OUT, ROOT, dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                                 # noqa: E402
import arbiter as A                                                 # noqa: E402
from random_layouts import make_layout                              # noqa: E402

V2 = os.path.join(ROOT, "results", "v2", "ws_d")


def make_c(cc):
    return A.APFRays(momentum=cc.get("momentum"), osc_cap_K=cc.get("osc_cap_K"))


def brief(r):
    return {k: r[k] for k in ("outcome", "steps", "frac_time_C", "switches", "chatter")}


def main():
    g, seed = int(sys.argv[1]), int(sys.argv[2])
    t0 = now()
    th = json.load(open(os.path.join(V2, "d2_thresholds", "result.json")))["grid"]
    d_on, rho_on = th["d_on"][g // 3], th["rho_on"][g % 3]
    cc = json.load(open(os.path.join(V2, "c_config.json")))
    L = A.Learned(os.path.join(OUT, "arc_models", f"sacdyn_seed{seed}", "models", "arc_best.pth"))
    C = make_c(cc)
    run = lambda tl, cs: brief(A.run(tl, L, C, "apf_rays", d_on, rho_on, c_seed=cs))
    res = {"grid_index": g, "d_on": d_on, "rho_on": rho_on, "sac_seed": seed, "c_config": cc, "sets": {}}
    H.set_ground_speed(18.0)
    res["sets"]["heldout"] = [run(H.ObstacleTimeline(500_000 + s), 501_000 + s) for s in range(50)]
    res["sets"]["layouts_static"] = [run(H.StaticTimeline(make_layout(500_000 + j)), 0) for j in range(50)]
    res["sets"]["layouts_dynamic"] = [run(H.ObstacleTimeline(510_000 + j, make_layout(500_000 + j)), 520_000 + j) for j in range(50)]
    res["sets"]["noise"] = [brief(A.run(H.ObstacleTimeline(500_000 + s), L, C, "apf_rays", d_on, rho_on, c_seed=501_000 + s,
                                        range_noise=2.5, heading_noise=math.radians(5.0), noise_seed=502_000 + s))
                            for s in range(50)]
    H.set_ground_speed(4.75)
    res["sets"]["r100"] = [run(H.ObstacleTimeline(500_000 + s), 501_000 + s) for s in range(50)]
    H.set_ground_speed(18.0)
    res["success"] = {k: sum(r["outcome"] == "success" for r in v) for k, v in res["sets"].items()}
    print(g, seed, res["success"], flush=True)
    d = outdir("ws_d", "d3_tuning")
    dump(d, f"g{g}_sac{seed}.json", res)
    write_manifest(d, "scripts/ws_d_d3_tune.py", {"rule": "prereg.md WS-D D3", "c_config": cc},
                   {"heldout_motion": "500000-500049", "tuning_layouts": "500000-500049", "tuning_layout_motion": "510000-510049"}, t0)


if __name__ == "__main__":
    main()
