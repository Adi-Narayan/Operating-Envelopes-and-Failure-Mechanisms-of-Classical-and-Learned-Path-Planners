"""
3D APF evaluation (Table 7). The simulation core is taken verbatim from the
heredoc in apf_3d/create_project_unified.sh, so that script stays the single
source of truth.

    python run_apf3d.py            # corrected protocol, 10 static + 10 dynamic seeded runs
"""
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def load_core():
    sh = open(os.path.join(ROOT, "apf_3d", "create_project_unified.sh"), encoding="utf8").read().replace("\r\n", "\n")
    m = re.search(r"cat << 'EOF' > \"\$OUTDIR/simulation/core.py\"\n(.*?)\nEOF\n", sh, re.S)
    path = os.path.join(OUT, "_apf3d_core.py")
    with open(path, "w", encoding="utf8") as f:
        f.write(m.group(1))
    spec = importlib.util.spec_from_file_location("apf3d_core", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


core = load_core()
START = np.array([35.0, 35.0, 35.0])
GOAL = np.array([465.0, 465.0, 465.0])
D = float(np.linalg.norm(GOAL - START))
N = 10


def one(mode, seed, goal_tol=None):
    cfg = core.APFConfig()
    if goal_tol is not None:
        cfg.goal_tol = goal_tol
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if mode == "static":
        ok, path, info = core.simulate_static_apf(START.copy(), GOAL.copy(), core.STATIC_OBSTACLES_3D, cfg, rng=rng)
    else:
        dyns = core.assign_velocities_3d(core.STATIC_OBSTACLES_3D, rng)
        ok, path, _, info = core.simulate_dynamic_apf(START.copy(), GOAL.copy(), dyns, cfg, rng=rng)
    ms = (time.perf_counter() - t0) * 1000
    L = core.path_length(path)
    return {"success": bool(ok), "outcome": info["outcome"], "steps": len(path) - 1, "L": L,
            "eff": 100 * D / L, "time_ms": ms, "escape_steps": info["escape_steps"],
            "pushouts": info["pushouts"], "min_clearance": float(info["min_clearance"]),
            "end": path[-1].tolist()}


def summarise(rows):
    def ms(k):
        x = np.array([r[k] for r in rows], float)
        return [float(x.mean()), float(x.std(ddof=1))]
    return {"success": sum(r["success"] for r in rows), "n": len(rows),
            "outcomes": [r["outcome"] for r in rows],
            "steps": ms("steps"), "L": ms("L"), "eff": ms("eff"), "time_ms": ms("time_ms"),
            "ms_per_step": float(np.mean([r["time_ms"] / r["steps"] for r in rows])),
            "escape_steps_total": sum(r["escape_steps"] for r in rows),
            "pushouts_total": sum(r["pushouts"] for r in rows),
            "min_clearance": min(r["min_clearance"] for r in rows),
            "L_minus_1p8_steps_max": max(r["L"] - 1.8 * r["steps"] for r in rows)}


if __name__ == "__main__":
    res = {"d_straight": D}
    # original protocol replica (goal_tol 18 + final segment to the goal centre) for the ratio check
    o = one("static", 0, goal_tol=18.0)
    snap = float(np.linalg.norm(np.array(o["end"]) - GOAL))
    res["original_protocol_static_seed0"] = {"steps_orig": o["steps"] + 1, "L_orig": o["L"] + snap,
                                             "snap_segment": snap, "escape_steps": o["escape_steps"],
                                             "pushouts": o["pushouts"]}
    for mode in ("static", "dynamic"):
        rows = [one(mode, s) for s in range(N)]
        res[mode] = summarise(rows)
        res[mode + "_runs"] = rows
    with open(os.path.join(OUT, "apf3d.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_runs")}, indent=1))
