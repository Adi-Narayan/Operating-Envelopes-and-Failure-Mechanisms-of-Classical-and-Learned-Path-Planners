"""Detail for the 3.4 audit: obstacle-state differences between the two timelines (agent 18 m/s with
baseline obstacle speeds, agent 9 m/s with obstacle speeds x0.5) and how APF's path deviation grows,
for the environments where APF paths differ by more than 1e-6 m.

    python scripts/verify_scaleinv_detail.py  -> results/v2/verification/scale_invariance/detail.json
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H                                      # noqa: E402
from verify_section3 import scaled_timeline               # noqa: E402


def obstacle_dev(tl_a, tl_b, ticks):
    dev = 0.0
    for k in range(ticks):
        for oa, ob in zip(tl_a.states[k], tl_b.states[k]):
            pa, pb = np.asarray(H.dyn.obs_centroid(oa)), np.asarray(H.dyn.obs_centroid(ob))
            dev = max(dev, float(np.abs(pa - pb).max()))
    return dev


out = {}
for s in (5, 9):
    H.set_ground_speed(18.0)
    ta = H.ObstacleTimeline(s)
    a = H.run_apf(ta, 1000 + s)
    H.set_ground_speed(9.0)
    tb = scaled_timeline(s, 0.5)
    b = H.run_apf(tb, 1000 + s)
    H.set_ground_speed(18.0)
    pa, pb = np.asarray(a["path"]), np.asarray(b["path"])
    d = np.abs(pa - pb).max(axis=1)
    first = {thr: int(np.argmax(d > thr)) if (d > thr).any() else None for thr in (1e-9, 1e-6, 1e-3)}
    ticks = min(len(ta.states), len(tb.states))
    out[str(s)] = {"steps": a["steps"], "escape_steps": [a.get("escape_steps"), b.get("escape_steps")],
                   "pushouts": [a.get("pushouts"), b.get("pushouts")],
                   "max_obstacle_centroid_dev_m": obstacle_dev(ta, tb, ticks),
                   "first_step_path_dev_exceeds": first, "final_dev_m": float(d[-1]), "max_dev_m": float(d.max())}
    print(s, out[str(s)], flush=True)
p = os.path.join(ROOT, "results", "v2", "verification", "scale_invariance", "detail.json")
json.dump(out, open(p, "w"), indent=1)
print("wrote", p)
