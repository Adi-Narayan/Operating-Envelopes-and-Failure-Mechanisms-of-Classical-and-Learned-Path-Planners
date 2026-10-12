"""
WS-D, D4: evaluation of the arbiter once on every axis with the frozen evaluation sets (prereg.md, WS-D D4).
Thresholds from results/v2/ws_d/d3_selection; fallback C from results/v2/ws_d/c_config.json.

    python scripts/ws_d_d4_eval.py d1 <sac seed>     arbiter, C = ray-APF (matched information)
    python scripts/ws_d_d4_eval.py d2 <sac seed>     arbiter, C = DWA-CV (privileged), D1's thresholds
    python scripts/ws_d_d4_eval.py sac <sac seed>    SAC alone (the arbiter with switching disabled)
    python scripts/ws_d_d4_eval.py capf              ray-APF alone
    python scripts/ws_d_d4_eval.py dwacv             DWA-CV alone
        -> results/v2/ws_d/d4_eval/<mode>[_sac<seed>].json

Planner seeds for APF as in the paper's scripts: 1000 + s (custom and second layout, dynamic, speed,
noise), 0 (static), 20000 + k (random layouts, dynamic); noise seed 50000 + s; layout motion 10000 + k.
'axes' holds the run lists under the heatmap axis names.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import OUT, ROOT, dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                                 # noqa: E402
import arbiter as A                                                 # noqa: E402
from random_layouts import make_layout                              # noqa: E402

V2 = os.path.join(ROOT, "results", "v2", "ws_d")
SPEEDS = [3.0, 4.0, 4.75, 6.0, 9.0, 12.0, 18.0, 27.0]
NOISE = {"L1": (1.0, 2.0), "L2": (2.5, 5.0), "L3": (5.0, 10.0)}
KEEP = ("outcome", "steps", "path_length", "efficiency", "min_clearance", "frac_time_C", "switches", "chatter", "ms_per_step",
        "switch_log")


def brief(r):
    return {k: (float(r[k]) if isinstance(r[k], (np.floating, float)) else r[k]) for k in KEEP if k in r}   # switch_log: where switches occur (Adi's item 9)


def build(mode, seed):
    cc = json.load(open(os.path.join(V2, "c_config.json")))
    L = A.Learned(os.path.join(OUT, "arc_models", f"sacdyn_seed{seed}", "models", "arc_best.pth")) if seed is not None else None
    if mode in ("d1", "capf"):
        C, kind = A.APFRays(momentum=cc.get("momentum"), osc_cap_K=cc.get("osc_cap_K")), "apf_rays"
    else:
        w = json.load(open(os.path.join(OUT, "dwa_tune.json")))["selected"]["DWA-CV"]["weights"]
        C, kind = A.DWACV(w), "dwa_cv"
    if mode in ("d1", "d2"):
        sel = json.load(open(os.path.join(V2, "d3_selection", "result.json")))["selected"]
        th = (sel["d_on"], sel["rho_on"], sel["d_off"], sel["rho_off"])
        on = True
    elif mode == "sac":
        th, on = (-np.inf, np.inf, None, None), True
    else:
        th, on = (0.0, 0.0, None, None), False
    cfg = {"mode": mode, "sac_seed": seed, "c_kind": kind, "c_config": cc if kind == "apf_rays" else "DWA-CV (dwa_tune.json)",
           "thresholds": dict(zip(("d_on", "rho_on", "d_off", "rho_off"), th)) if mode in ("d1", "d2") else None}

    def go(tl, c_seed, **nz):
        return brief(A.run(tl, L, C, kind, th[0], th[1], d_off=th[2], rho_off=th[3], c_seed=c_seed, l_enabled=on, **nz))
    return go, cfg


def main():
    mode = sys.argv[1]
    seed = int(sys.argv[2]) if mode in ("d1", "d2", "sac") else None
    t0 = now()
    go, cfg = build(mode, seed)
    urban = H.LAYOUTS["urban"]
    H.set_ground_speed(18.0)
    ev = {"static_custom": go(H.StaticTimeline(), 0),
          "dynamic": [go(H.ObstacleTimeline(s), 1000 + s) for s in range(50)],
          "noise": {lev: [go(H.ObstacleTimeline(s), 1000 + s, range_noise=rn, heading_noise=math.radians(hd), noise_seed=50_000 + s)
                          for s in range(50)] for lev, (rn, hd) in NOISE.items()},
          "layouts_static": [go(H.StaticTimeline(make_layout(k)), 0) for k in range(100)],
          "layouts_dynamic": [go(H.ObstacleTimeline(10_000 + k, make_layout(k)), 20_000 + k) for k in range(100)],
          "second_static": go(H.StaticTimeline(urban), 0),
          "second_dynamic": [go(H.ObstacleTimeline(s, urban), 1000 + s) for s in range(50)],
          "speed": {}}
    for v in SPEEDS:
        H.set_ground_speed(v)
        ev["speed"][str(v)] = [go(H.ObstacleTimeline(s), 1000 + s) for s in range(50)]
    H.set_ground_speed(18.0)
    res = {"config": cfg, "evaluation": ev,
           "axes": {"indist": ev["dynamic"], "r100": ev["speed"]["4.75"], "r158": ev["speed"]["3.0"], "noise5": ev["noise"]["L3"],
                    "lay_static": ev["layouts_static"], "lay_dynamic": ev["layouts_dynamic"], "second_dyn": ev["second_dynamic"]}}
    name = mode if seed is None else f"{mode}_sac{seed}"
    d = outdir("ws_d", "d4_eval")
    dump(d, f"{name}.json", res)
    write_manifest(d, f"scripts/ws_d_d4_eval.py {' '.join(sys.argv[1:])}", cfg, {"see": "notes/seed_map.md (frozen evaluation sets)"},
                   t0, name=f"manifest_{name}.json")
    print(name, {k: sum(r["outcome"] == "success" for r in v) for k, v in res["axes"].items()}, flush=True)


if __name__ == "__main__":
    main()
