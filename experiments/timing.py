"""
Timing measurements (run on an otherwise idle machine).

  per-step planner cost, measured by replaying a recorded successful path:
      APF : F_total at q, normalise, candidate step, 6 m margin check
      ARC : observation construction (9 ray casts) + LSTM/actor forward pass
  full-run wall time (static 2D, both planners; 3D static/dynamic APF)
  steps per 100 m and compute per 100 m (Eqs. 10-11)

    python timing.py --models out/arc_models/seed*/models/arc_best.pth
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H   # noqa: E402

sta = H.sta
REPEATS = 10


def apf_step_cost(path, cfg, obstacles):
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for q in path[:-1]:
            Ft, _, _ = sta.F_total(q, H.GOAL, obstacles, cfg)
            sd = Ft / (np.linalg.norm(Ft) + 1e-12)
            sta.in_collision(q + cfg.step * sd, obstacles, margin=sta.COLLISION_MARGIN)
        ts.append((time.perf_counter() - t0) * 1000 / (len(path) - 1))
    return float(np.median(ts)), ts


def arc_step_cost(planner, path):
    env = planner.env
    env.obstacles = H._to_shapely(H.OBSTACLES)
    env.target_pos = H.GOAL.astype(np.float32)
    hist = planner.ObservationHistory(planner.hist_len, planner.obs_dim)
    ts = []
    for _ in range(REPEATS):
        env.uav_pos = path[0].astype(np.float32).copy()
        hist.reset(env._get_observation())
        t0 = time.perf_counter()
        for k in range(len(path) - 1):
            d = path[k + 1] - path[k]
            env.uav_pos = path[k].astype(np.float32).copy()
            env.uav_heading = math.atan2(d[1], d[0])
            hist.append(env._get_observation())
            planner.agent.select_action(hist.get_tensor(planner.agent.device), None, noise_scale=0.0)
        ts.append((time.perf_counter() - t0) * 1000 / (len(path) - 1))
    return float(np.median(ts)), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="timing", help="output name: out/<out>.json")
    ap.add_argument("--no-3d", action="store_true", help="skip the 3D APF wall times")
    args = ap.parse_args()
    models = []
    for m in args.models:
        models += sorted(glob.glob(m)) if any(c in m for c in "*?[") else [m]
    res = {}
    static = H.StaticTimeline()
    cfg = sta.APFConfig()

    # full runs (these include the shared per-step clearance check of the evaluation)
    apf_runs = [H.run_apf(static, s) for s in range(REPEATS)]
    res["apf_static_run_ms"] = H.mean_sd([r["time_ms"] for r in apf_runs])
    apf_path = apf_runs[0]["path"]
    res["apf_step_ms"], res["apf_step_ms_all"] = apf_step_cost(apf_path, cfg, sta.ENVIRONMENTS[1]["obstacles"])
    res["apf_steps_per_100m"] = 100 * apf_runs[0]["steps"] / apf_runs[0]["path_length"]

    arc = []
    for m in models:
        p = H.ARCPlanner(m)
        runs = [p.run(static, static=True) for _ in range(3)]
        r = runs[0]
        entry = {"model": m, "outcome": r["outcome"], "run_ms": H.mean_sd([x["time_ms"] for x in runs])}
        if r["success"]:
            entry["step_ms"], _ = arc_step_cost(p, r["path"])
            entry["steps_per_100m"] = 100 * r["steps"] / r["path_length"]
        arc.append(entry)
    res["arc"] = arc
    ok = [a for a in arc if "step_ms" in a]
    if ok:
        res["arc_step_ms"] = H.mean_sd([a["step_ms"] for a in ok])
        res["arc_run_ms_successful"] = H.mean_sd([a["run_ms"][0] for a in ok])
        res["arc_steps_per_100m"] = H.mean_sd([a["steps_per_100m"] for a in ok])
        res["compute_per_100m_ms"] = {"apf": res["apf_steps_per_100m"] * res["apf_step_ms"],
                                      "arc": res["arc_steps_per_100m"][0] * res["arc_step_ms"][0]}

    # 3D APF wall times
    if not args.no_3d:
        import run_apf3d as A3
        for mode in ("static", "dynamic"):
            rows = [A3.one(mode, s) for s in range(REPEATS)]
            res[f"apf3d_{mode}_run_ms"] = H.mean_sd([r["time_ms"] for r in rows])
            res[f"apf3d_{mode}_ms_per_step"] = float(np.mean([r["time_ms"] / r["steps"] for r in rows]))

    with open(os.path.join(HERE, "out", f"{args.out}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_all")}, indent=1, default=float))


if __name__ == "__main__":
    main()
