"""
Timing for the planners added in v2, in ONE idle session together with APF (instructions.md compute
rules: timing is measured with the machine otherwise idle, and compared only within a session).

For each planner, the 50 paired dynamic environments at 18 m/s, three repeats: planning time per
decision (run time minus obstacle-timeline generation, divided by decisions) and per 100 m of path,
median over runs, and the ratio to APF in the same session.
Planners: APF, APF-D, APF-D under the learned agent's rays (C alone), SAC alone (seed 0), arbiter D1
and D2 (seed 0, thresholds from results/v2/ws_d/d3_selection).

    python scripts/ws_x_timing.py  -> results/v2/ws_x/timing/result.json   (refuses to run while training runs)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import OUT, ROOT, dump, now, outdir, training_active, write_manifest  # noqa: E402
import harness as H                                                                  # noqa: E402
import apf_damped as D                                                               # noqa: E402
import arbiter as A                                                                  # noqa: E402

REPEATS = 3


def per(r):
    steps = max(r["steps"], 1)
    return r["ms_per_step"], r["ms_per_step"] * steps / max(r["path_length"], 1e-9) * 100.0


def main():
    if training_active():
        sys.exit("training is running; timing needs an idle machine")
    t0 = now()
    V2 = os.path.join(ROOT, "results", "v2", "ws_d")
    sel = json.load(open(os.path.join(V2, "d3_selection", "result.json")))["selected"]
    cc = json.load(open(os.path.join(V2, "c_config.json")))
    w = json.load(open(os.path.join(OUT, "dwa_tune.json")))["selected"]["DWA-CV"]["weights"]
    L = A.Learned(os.path.join(OUT, "arc_models", "sacdyn_seed0", "models", "arc_best.pth"))
    C = A.APFRays(momentum=cc.get("momentum"), osc_cap_K=cc.get("osc_cap_K"))
    th = (sel["d_on"], sel["rho_on"], sel["d_off"], sel["rho_off"])
    planners = {
        "APF": lambda tl, s: H.run_apf(tl, 1000 + s),
        "APF-D": lambda tl, s: D.run(tl, 1000 + s, momentum=0.85),
        "APF-D, rays (C alone)": lambda tl, s: A.run(tl, None, C, "apf_rays", 0, 0, c_seed=1000 + s, l_enabled=False),
        "SAC alone": lambda tl, s: A.run(tl, L, C, "apf_rays", -np.inf, np.inf),
        "Arbiter D1": lambda tl, s: A.run(tl, L, C, "apf_rays", th[0], th[1], d_off=th[2], rho_off=th[3], c_seed=1000 + s),
        "Arbiter D2": lambda tl, s: A.run(tl, L, A.DWACV(w), "dwa_cv", th[0], th[1], d_off=th[2], rho_off=th[3]),
    }
    H.set_ground_speed(18.0)
    res = {}
    for name, f in planners.items():
        ms, ms100 = [], []
        for _ in range(REPEATS):
            for s in range(50):
                r = f(H.ObstacleTimeline(s), s)
                a, b = per(r)
                ms.append(a)
                ms100.append(b)
        res[name] = {"ms_per_decision_median": float(np.median(ms)), "ms_per_100m_median": float(np.median(ms100))}
        print(name, res[name], flush=True)
    base = res["APF"]
    for name, r in res.items():
        r["ratio_to_apf_per_decision"] = r["ms_per_decision_median"] / base["ms_per_decision_median"]
        r["ratio_to_apf_per_100m"] = r["ms_per_100m_median"] / base["ms_per_100m_median"]
    d = outdir("ws_x", "timing")
    dump(d, "result.json", res)
    write_manifest(d, "scripts/ws_x_timing.py", {"repeats": REPEATS, "speed": 18.0}, {"motion": "0-49"}, t0, busy=False)


if __name__ == "__main__":
    main()
