"""
Static head-to-head (Table 3) and paired 50-environment dynamic comparison (Table 4).

    python evaluate.py --models out/arc_models/seed*/models/arc_best.pth [--primary 0] [--envs 50]

Path metrics are only ever aggregated over runs that reached the goal:
  * Table 3 : over successful runs of each planner (both start from the same fixed
              start and use d_straight = 608.1 m)
  * Table 4B: over the environments that BOTH planners solve (paired), never pooled
              over collided/timed-out runs
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H   # noqa: E402

OUT = os.path.join(HERE, "out")


def strip(r):
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items() if k != "path"}


def agg(rows):
    ok = [r for r in rows if r["success"]]
    d = {"n": len(rows), "success": len(ok),
         "collision": sum(r["outcome"] == "collision" for r in rows),
         "timeout": sum(r["outcome"] == "timeout" for r in rows),
         "out_of_bounds": sum(r["outcome"] == "out_of_bounds" for r in rows)}
    for k in ("steps", "path_length", "efficiency", "time_ms", "ms_per_step"):
        d[k] = H.mean_sd([r[k] for r in ok])
    return d


def panel_b(apf_rows, arc_rows):
    shared = [i for i, (a, b) in enumerate(zip(apf_rows, arc_rows)) if a["success"] and b["success"]]
    out = {"n_shared": len(shared), "env_seeds": shared}
    for name, rows in (("apf", apf_rows), ("arc", arc_rows)):
        sub = [rows[i] for i in shared]
        out[name] = {k: H.mean_sd([r[k] for r in sub]) for k in ("path_length", "efficiency", "time_ms", "steps")}
    out["arc_more_efficient"] = sum(arc_rows[i]["efficiency"] > apf_rows[i]["efficiency"] for i in shared)
    out["apf_more_efficient"] = sum(apf_rows[i]["efficiency"] > arc_rows[i]["efficiency"] for i in shared)
    arc_ok = {i for i, r in enumerate(arc_rows) if r["success"]}
    apf_ok = {i for i, r in enumerate(apf_rows) if r["success"]}
    out["arc_only"] = sorted(arc_ok - apf_ok)
    out["apf_only"] = sorted(apf_ok - arc_ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--primary", type=int, default=0, help="index into --models of the primary model")
    ap.add_argument("--envs", type=int, default=50)
    ap.add_argument("--apf-static-runs", type=int, default=10)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--layout", default="corridor", choices=sorted(H.LAYOUTS))
    ap.add_argument("--speed", type=float, default=18.0, help="common ground speed of both planners (m/s)")
    args = ap.parse_args()
    H.set_ground_speed(args.speed)
    models = []
    for m in args.models:
        models += sorted(glob.glob(m)) if any(c in m for c in "*?[") else [m]

    obstacles = H.LAYOUTS[args.layout]
    res = {"models": models, "primary": models[args.primary], "layout": args.layout, "protocol": {
        "start": H.START.tolist(), "goal": H.GOAL.tolist(), "d_straight": H.D_STRAIGHT,
        "goal_radius": H.GOAL_RADIUS, "collision_radius": H.COLLISION_RADIUS,
        "ground_speed_mps": H.GROUND_SPEED, "apf_dt": H.APF_DT, "arc_dt": H.ARC_DT,
        "max_distance_m": H.MAX_DISTANCE, "apf_max_steps": H.APF_MAX_STEPS, "arc_max_steps": H.ARC_MAX_STEPS}}

    # ---------------- static head-to-head
    static = H.StaticTimeline(obstacles)
    apf_s = [H.run_apf(static, s) for s in range(args.apf_static_runs)]
    planners = [H.ARCPlanner(m) for m in models]
    arc_s = [p.run(static, static=True) for p in planners]
    res["static"] = {"apf": agg(apf_s), "arc_over_models": agg(arc_s),
                     "apf_runs": [strip(r) for r in apf_s], "arc_runs": [strip(r) for r in arc_s]}
    np.savez_compressed(os.path.join(OUT, f"{args.tag}_static_paths.npz"),
                        apf=apf_s[0]["path"], **{f"arc{i}": r["path"] for i, r in enumerate(arc_s)})
    print("STATIC  APF:", {k: v for k, v in agg(apf_s).items()})
    print("STATIC  ARC:", {k: v for k, v in agg(arc_s).items()})
    for m, r in zip(models, arc_s):
        print(f"   {os.path.basename(os.path.dirname(os.path.dirname(m)))}: {r['outcome']} steps={r['steps']} L={r['path_length']:.1f} eff={r['efficiency']:.1f}")

    # ---------------- paired dynamic suite
    apf_d, arc_d = [], [[] for _ in models]
    keep_paths = {}
    for s in range(args.envs):
        tl = H.ObstacleTimeline(s, obstacles)
        a = H.run_apf(tl, 1000 + s)
        apf_d.append(a)
        for j, p in enumerate(planners):
            arc_d[j].append(p.run(tl))
        keep_paths[f"apf_{s}"] = a["path"]
        keep_paths[f"arc_{s}"] = arc_d[args.primary][-1]["path"]
        print(f"env {s:2d}: APF {a['outcome'][:4]} {a['efficiency']:5.1f} | ARC " +
              " ".join(f"{arc_d[j][-1]['outcome'][:4]}" for j in range(len(models))), flush=True)
    np.savez_compressed(os.path.join(OUT, f"{args.tag}_dynamic_paths.npz"), **keep_paths)

    res["dynamic"] = {
        "apf": agg(apf_d),
        "apf_runs": [strip(r) for r in apf_d],
        "arc_per_model": [agg(rows) for rows in arc_d],
        "arc_runs_per_model": [[strip(r) for r in rows] for rows in arc_d],
        "panel_b_primary": panel_b(apf_d, arc_d[args.primary]),
        "panel_b_per_model": [panel_b(apf_d, rows) for rows in arc_d],
    }
    sr = np.array([x["success"] for x in res["dynamic"]["arc_per_model"]], float)
    res["dynamic"]["arc_success_across_models"] = {"mean": sr.mean(), "sd": sr.std(ddof=1) if len(sr) > 1 else 0.0,
                                                   "min": sr.min(), "max": sr.max()}
    with open(os.path.join(OUT, f"{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("DYNAMIC APF:", res["dynamic"]["apf"])
    for j, x in enumerate(res["dynamic"]["arc_per_model"]):
        print(f"DYNAMIC ARC[{j}]:", {k: x[k] for k in ("success", "collision", "timeout", "out_of_bounds")})
    print("PANEL B (primary):", json.dumps(res["dynamic"]["panel_b_primary"], default=float))


if __name__ == "__main__":
    main()
