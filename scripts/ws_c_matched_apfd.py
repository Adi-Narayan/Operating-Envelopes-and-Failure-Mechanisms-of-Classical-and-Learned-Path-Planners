"""
WS-C, post hoc (prereg.md, entry of 2026-10-14): APF-D under all three of the learned agent's constraints
(nine 50 m rays, 6 Hz with 3.0 m steps, hard 9 deg per step turn cap) on every envelope axis, beside
matched APF without momentum on the same runs; and the step types of APF-D on the rays without the cap
(the arbiter's C alone), whose escape and push-out steps are not limited by the momentum.

    python scripts/ws_c_matched_apfd.py  -> results/v2/ws_c/matched_apfd/result.json

Seeds as in experiments/robustness_suite.py and scripts/ws_d_d4_eval.py: planner 1000 + s (paired
dynamic, speed, noise, second layout), 0 (static), 20000 + k (random layouts, dynamic); noise seed
50000 + s; layout motion 10000 + k. Axes and levels as in scripts/ws_d_d4_eval.py.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                           # noqa: E402
import apf_variants as V                                      # noqa: E402
import arbiter as A                                           # noqa: E402
from random_layouts import make_layout                        # noqa: E402

V2 = os.path.join(ROOT, "results", "v2", "ws_d")
NOISE = {"L1": (1.0, 2.0), "L2": (2.5, 5.0), "L3": (5.0, 10.0)}
SPEEDS = {"r100": 4.75, "r158": 3.0}
KEEP = ("outcome", "steps", "path_length", "efficiency", "escape_steps", "pushouts")


def axes(go):
    urban = H.LAYOUTS["urban"]
    H.set_ground_speed(18.0)
    ev = {"indist": [go(H.ObstacleTimeline(s), 1000 + s) for s in range(50)],
          "noise": {lev: [go(H.ObstacleTimeline(s), 1000 + s, range_noise=rn, heading_noise=math.radians(hd), noise_seed=50_000 + s)
                          for s in range(50)] for lev, (rn, hd) in NOISE.items()},
          "lay_static": [go(H.StaticTimeline(make_layout(k)), 0) for k in range(100)],
          "lay_dynamic": [go(H.ObstacleTimeline(10_000 + k, make_layout(k)), 20_000 + k) for k in range(100)],
          "second_dyn": [go(H.ObstacleTimeline(s, urban), 1000 + s) for s in range(50)],
          "static_custom": go(H.StaticTimeline(), 0)}
    for ax, v in SPEEDS.items():
        H.set_ground_speed(v)
        ev[ax] = [go(H.ObstacleTimeline(s), 1000 + s) for s in range(50)]
    H.set_ground_speed(18.0)
    ev["noise5"] = ev["noise"]["L3"]
    return ev


def summary(ev):
    out = {ax: sum(r["outcome"] == "success" for r in ev[ax]) for ax in ("indist", "r100", "r158", "noise5", "lay_static", "lay_dynamic", "second_dyn")}
    out.update({f"noise_{k}": sum(r["outcome"] == "success" for r in v) for k, v in ev["noise"].items()})
    out["static_custom"] = ev["static_custom"]["outcome"]
    return out


def main():
    t0 = now()
    res = {}
    for name, mom in (("matched_apf", None), ("matched_apfd", 0.85)):
        def go(tl, seed, **nz):
            r = V.run(tl, seed, sensing="rays", turn_cap=True, rate="arc", momentum=mom, **nz)
            return {k: (float(r[k]) if isinstance(r[k], (np.floating, float)) else r[k]) for k in KEEP if k in r}
        ev = axes(go)
        res[name] = {"summary": summary(ev), "runs": ev}
        print(name, res[name]["summary"], flush=True)
    # C alone (APF-D on the rays, no cap) with step-type counts
    cc = json.load(open(os.path.join(V2, "c_config.json")))
    C = A.APFRays(momentum=cc.get("momentum"), osc_cap_K=cc.get("osc_cap_K"))

    def go_c(tl, seed, **nz):
        r = A.run(tl, None, C, "apf_rays", 0.0, 0.0, c_seed=seed, l_enabled=False, **nz)
        return {"outcome": r["outcome"], "steps": r["steps"], **dict(C.counts)}
    ev = axes(go_c)
    stored = json.load(open(os.path.join(V2, "d4_eval", "capf.json")))["axes"]
    same = {ax: [a["outcome"] for a in ev[ax]] == [b["outcome"] for b in stored[ax]] for ax in stored}
    flat = [r for ax in ("indist", "r100", "r158", "noise5", "lay_static", "lay_dynamic", "second_dyn") for r in ev[ax]]
    tot = {k: int(sum(r[k] for r in flat)) for k in ("decisions", "jitter", "escape", "pushout")}
    fails = [r for r in flat if r["outcome"] != "success"]
    res["c_alone_steps"] = {"summary": summary(ev), "reproduces_d4_capf": same, "totals": tot,
                            "runs_with_escape": int(sum(r["escape"] > 0 for r in flat)), "n_runs": len(flat),
                            "escape_per_100_decisions": 100.0 * tot["escape"] / tot["decisions"],
                            "pushout_per_100_decisions": 100.0 * tot["pushout"] / tot["decisions"],
                            "successes_with_escape": int(sum(r["escape"] > 0 for r in flat if r["outcome"] == "success")),
                            "failures": len(fails), "runs": ev}
    print("C alone", res["c_alone_steps"]["summary"], same, tot, flush=True)
    d = outdir("ws_c", "matched_apfd")
    dump(d, "result.json", res)
    write_manifest(d, "scripts/ws_c_matched_apfd.py", {"momentum": 0.85, "cap_deg_per_step": 9.0, "rate": "6 Hz, 3.0 m",
                                                       "sensing": "nine 50 m rays"}, {"see": "docstring"}, t0)


if __name__ == "__main__":
    main()
