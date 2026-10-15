"""
WS-C, C4: evaluate the selected damped APF (APF-D, results/v2/ws_c/c4_selection) once on every axis
with the frozen evaluation sets and APF's own seeds (notes/seed_map.md).

    python scripts/ws_c_apfd_eval.py <part>   part in: static, dynamic, speed, noise, layouts, second
        -> results/v2/ws_c/apfd_eval/<part>.json
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                            # noqa: E402
import apf_damped as D                                         # noqa: E402
from random_layouts import make_layout                         # noqa: E402

KEEP = ("outcome", "steps", "path_length", "efficiency", "min_clearance", "escape_steps", "pushouts")
SPEEDS = [3.0, 4.0, 4.75, 6.0, 9.0, 12.0, 18.0, 27.0]
NOISE = {"L1": (1.0, 2.0), "L2": (2.5, 5.0), "L3": (5.0, 10.0)}


def brief(r):
    return {k: (float(r[k]) if isinstance(r[k], float) else r[k]) for k in KEEP if k in r}


def main():
    part = sys.argv[1]
    sel = json.load(open(os.path.join(ROOT, "results", "v2", "ws_c", "c4_selection", "result.json")))["selected"]
    kw = D.VARIANTS[sel]
    t0 = now()
    H.set_ground_speed(18.0)
    res = {"variant": sel, "kwargs": kw}
    if part == "static":
        res["custom"] = brief(D.run(H.StaticTimeline(), 0, **kw))
        res["second"] = brief(D.run(H.StaticTimeline(H.LAYOUTS["urban"]), 0, **kw))
        import static_apf as sta
        res["five_layouts"] = {e["name"]: brief(D.run(H.StaticTimeline(e["obstacles"]), 0, **kw))
                               for e in (sta.ENVIRONMENTS.values() if isinstance(sta.ENVIRONMENTS, dict) else sta.ENVIRONMENTS)}
    elif part == "dynamic":
        res["runs"] = [brief(D.run(H.ObstacleTimeline(s), 1000 + s, **kw)) for s in range(50)]
    elif part == "second":
        res["runs"] = [brief(D.run(H.ObstacleTimeline(s, H.LAYOUTS["urban"]), 1000 + s, **kw)) for s in range(50)]
    elif part == "speed":
        res["speeds"] = {}
        for v in SPEEDS:
            H.set_ground_speed(v)
            res["speeds"][str(v)] = [brief(D.run(H.ObstacleTimeline(s), 1000 + s, **kw)) for s in range(50)]
        H.set_ground_speed(18.0)
    elif part == "noise":
        res["levels"] = {lev: [brief(D.run(H.ObstacleTimeline(s), 1000 + s, range_noise=rn, heading_noise=math.radians(hd),
                                           noise_seed=50_000 + s, **kw)) for s in range(50)]
                         for lev, (rn, hd) in NOISE.items()}
    elif part == "layouts":
        res["static"] = [brief(D.run(H.StaticTimeline(make_layout(k)), 0, **kw)) for k in range(100)]
        res["dynamic"] = [brief(D.run(H.ObstacleTimeline(10_000 + k, make_layout(k)), 20_000 + k, **kw)) for k in range(100)]
    d = outdir("ws_c", "apfd_eval")
    dump(d, f"{part}.json", res)
    write_manifest(d, f"scripts/ws_c_apfd_eval.py {part}", {"variant": sel, "kwargs": kw},
                   {"see": "notes/seed_map.md (frozen evaluation sets, APF planner seeds)"}, t0)
    print("done", part)


if __name__ == "__main__":
    main()
