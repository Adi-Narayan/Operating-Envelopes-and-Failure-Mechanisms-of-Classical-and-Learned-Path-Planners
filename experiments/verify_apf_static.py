"""
Static 2D APF verification (no timing here; see timing.py).

  1. Reproduce the ORIGINAL protocol numbers (goal_tol = 18 m, final segment to
     the goal centre appended and counted as a step) and check what actually
     makes L exceed 1.8 x steps.
  2. Corrected protocol (goal radius 10 m shared with ARC, no appended segment):
     10 seeded runs on each of the five layouts (Table 3 APF column, Table 6).
  3. Is the 0.8 m distance clamp ever active?
  4. Only the ratio k_r / k_a matters for the trajectory.
  5. Rotation-angle sweep 0..90 deg on every layout (Sec. 7.4 / Fig. 6 and the
     Sec. 8.2 claim that ~45 deg would clear Narrow Passage).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H   # noqa: E402

sta = H.sta
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
START, GOAL = H.START.copy(), H.GOAL.copy()
N_RUNS = 10


def cfg_with(**kw):
    c = sta.APFConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def run(env_id, seed, cfg=None):
    cfg = cfg or sta.APFConfig()
    obs = sta.ENVIRONMENTS[env_id]["obstacles"]
    ok, path, _, info = sta.simulate_static_apf(START.copy(), GOAL.copy(), obs, cfg,
                                                rng=np.random.default_rng(seed))
    L = sta.path_length(path)
    return {"success": bool(ok), "outcome": info["outcome"], "steps": len(path) - 1,
            "L": L, "eff": 100 * H.D_STRAIGHT / L, "escape_steps": info["escape_steps"],
            "pushouts": info["pushouts"], "min_clearance": float(info["min_clearance"]),
            "end": path[-1].tolist(), "path": path}


res = {}

# 1. original protocol -----------------------------------------------------------
orig = []
for seed in range(N_RUNS):
    r = run(1, seed, cfg_with(goal_tol=18.0))
    snap = float(np.linalg.norm(np.array(r["end"]) - GOAL))
    orig.append({"steps_orig": r["steps"] + 1, "L_orig": r["L"] + snap, "snap_segment": snap,
                 "escape_steps": r["escape_steps"], "pushouts": r["pushouts"]})
st = np.array([o["steps_orig"] for o in orig]); L = np.array([o["L_orig"] for o in orig])
res["original_protocol_custom_env"] = {
    "steps_mean": st.mean(), "steps_sd": st.std(ddof=1), "L_mean": L.mean(), "L_sd": L.std(ddof=1),
    "eff_mean": float(np.mean(100 * H.D_STRAIGHT / L)),
    "snap_segment_mean": float(np.mean([o["snap_segment"] for o in orig])),
    "escape_steps_total": int(sum(o["escape_steps"] for o in orig)),
    "pushouts_total": int(sum(o["pushouts"] for o in orig)),
    "L_over_steps": float(np.mean(L / st)),
}
orig5 = {}
for env_id in range(1, 6):
    r = run(env_id, 0, cfg_with(goal_tol=18.0))
    snap = float(np.linalg.norm(np.array(r["end"]) - GOAL)) if r["success"] else 0.0
    orig5[sta.ENVIRONMENTS[env_id]["name"]] = {
        "success": r["success"], "steps_orig": r["steps"] + (1 if r["success"] else 0),
        "L_orig": r["L"] + snap, "snap": snap, "escape_steps": r["escape_steps"],
        "pushouts": r["pushouts"]}
res["original_protocol_5_layouts_seed0"] = orig5

# 2. corrected protocol, 10 runs x 5 layouts -----------------------------------------
sta.CLAMP_HITS[0] = 0
layouts = {}
for env_id in range(1, 6):
    rs = [run(env_id, s) for s in range(N_RUNS)]
    ok = [r for r in rs if r["success"]]
    def ms(key, rows):
        return H.mean_sd([r[key] for r in rows]) if rows else (float("nan"), float("nan"))
    layouts[sta.ENVIRONMENTS[env_id]["name"]] = {
        "n_obstacles": len(sta.ENVIRONMENTS[env_id]["obstacles"]),
        "types": [o["type"] for o in sta.ENVIRONMENTS[env_id]["obstacles"]],
        "success": len(ok), "n": N_RUNS,
        "outcomes": [r["outcome"] for r in rs],
        "steps": ms("steps", rs), "L": ms("L", rs), "eff": ms("eff", rs),
        "escape_steps_total": sum(r["escape_steps"] for r in rs),
        "pushouts_total": sum(r["pushouts"] for r in rs),
        "min_clearance": min(r["min_clearance"] for r in rs),
        "L_minus_1p8_steps_max": max(r["L"] - 1.8 * r["steps"] for r in rs),
    }
res["corrected_protocol_layouts"] = layouts
res["clamp_hits_corrected_protocol"] = sta.CLAMP_HITS[0]

# 3. clamp under the original (no-termination) behaviour: min raw rho seen by F_rep
min_rho = [np.inf]
_orig_get = sta.get_distance
def _spy(p, obs):
    out = _orig_get(p, obs)
    min_rho[0] = min(min_rho[0], out[0])
    return out
sta.get_distance = _spy
for env_id in range(1, 6):
    run(env_id, 0, cfg_with(goal_tol=18.0))
sta.get_distance = _orig_get
res["min_raw_rho_any_distance_query_5_layouts"] = float(min_rho[0])

# 4. gain-ratio invariance --------------------------------------------------------
pairs = [(5.0, 0.85), (2.5, 0.425), (1.0, 0.17)]
gr = {}
for ka, kr in pairs:
    r = run(1, 0, cfg_with(k_a=ka, k_r=kr))
    gr[f"ka={ka},kr={kr}"] = {"steps": r["steps"], "L": r["L"], "end": r["end"]}
p0 = run(1, 0)["path"]; p1 = run(1, 0, cfg_with(k_a=2.5, k_r=0.425))["path"]
n = min(len(p0), len(p1))
gr["max_pointwise_dev_5.0/0.85_vs_2.5/0.425_m"] = float(np.max(np.linalg.norm(p0[:n] - p1[:n], axis=1)))
res["gain_ratio_invariance"] = gr

# 5. rotation sweep --------------------------------------------------------------
sweep = {}
for env_id in range(1, 6):
    name = sta.ENVIRONMENTS[env_id]["name"]
    sweep[name] = []
    for th in range(0, 91, 5):
        r = run(env_id, 0, cfg_with(theta=np.deg2rad(th)))
        sweep[name].append({"theta": th, "success": r["success"], "outcome": r["outcome"],
                            "steps": r["steps"], "L": round(r["L"], 1), "eff": round(r["eff"], 1)})
res["theta_sweep_seed0"] = sweep

with open(os.path.join(OUT, "verify_apf_static.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)

# ---- print summary
print(json.dumps({k: v for k, v in res.items() if k != "theta_sweep_seed0"}, indent=1, default=lambda x: round(float(x), 3)))
for name, rows in sweep.items():
    print(name)
    print("   " + " ".join(f"{r['theta']:>2}:{'S' if r['success'] else r['outcome'][0].upper()}{r['eff']:5.1f}" for r in rows))
