"""
APF inside ARC's 3D arena (arc_3d/arc_3d.py, 500 x 500 x 500 m, curriculum stages 0-2).

The arena geometry, SDF kernels (sphere / box / cylinder) and wind model are exec'd
verbatim from arc_3d.py (importing that file would start a training run). The APF
planner is the unchanged 3D formulation of Sec. 3.6 (k_a = 5.0, k_r = 0.85,
theta = 25 deg about z, s = 1.8 m, 6 m margin, 5 escape attempts + push-out); only the
distance kernel is swapped for ARC's exact SDFs, whose gradient ARC also uses.
Success / collision follow ARC's own definitions for this arena:
  success   : within GOAL_RADIUS = 20 m of the goal
  collision : SDF < COLLISION_M = 1.75 m
Each stage is run 10 times without wind and 10 times with ARC's stochastic wind
(the same per-step displacement ARC experiences: wind * 0.05 * WORLD_SCALE).

    python apf_in_arc3d.py
"""
import json
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = open(os.path.join(ROOT, "arc_3d", "arc_3d.py"), encoding="utf8").read().replace("\r\n", "\n")


def _grab(start_pat, end_pat):
    i = SRC.index(start_pat)
    j = SRC.index(end_pat, i) + len(end_pat)
    return SRC[i:j]


NS = {"np": np}
exec(_grab("ARC3D_CONFIG = {", '"OBSTACLE_SPEED_SCALE": 0.06,\n}'), NS)
exec(_grab("# ── SDF primitives", "        return self.wind.copy()\n"), NS)
CFG = NS["ARC3D_CONFIG"]
obs_sdf, obs_grad, SOLID = NS["obs_sdf"], NS["obs_sdf_gradient"], NS["_SOLID_TYPES"]
WindModel = NS["WindModel"]
WORLD_SCALE = float(re.search(r"WORLD_SCALE\s*=\s*([0-9.]+)", SRC).group(1))
COLLISION_M = float(re.search(r"COLLISION_M\s*=\s*([0-9.]+)\s*\*\s*WORLD_SCALE", SRC).group(1)) * WORLD_SCALE
ARC_MAX_STEPS = int(re.search(r"MAX_STEPS\s*=\s*(\d+)", SRC).group(1))
ARC_STEP_MAX = float(re.search(r"STEP_SIZE\s*=\s*([0-9.]+)\s*\*\s*WORLD_SCALE", SRC).group(1)) * WORLD_SCALE

BOUNDS = CFG["ENV_BOUNDS"]
START = np.array(CFG["START"], float)
GOAL = np.array(CFG["GOAL"], float)
GOAL_RADIUS = float(CFG["GOAL_RADIUS"])
D_STRAIGHT = float(np.linalg.norm(GOAL - START))

# APF (Sec. 3) -----------------------------------------------------------------
K_A, K_R, THETA, POW, STEP = 5.0, 0.85, np.deg2rad(25), 2, 1.8
MARGIN, MAX_STEPS, RHO_CLAMP = 6.0, 6000, 0.8
c, s = np.cos(THETA), np.sin(THETA)
RZ = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def dist(p, o):
    return max(obs_sdf(p, o), 0.0)


def clearance(p, obs):
    return min(obs_sdf(p, o) for o in obs)


def force(q, obs, stats):
    f = -K_A * (q - GOAL)
    rg = np.linalg.norm(q - GOAL) + 1e-12
    for o in obs:
        rho = dist(q, o)
        if rho < RHO_CLAMP:
            stats["clamp_hits"] += 1
        rho = max(rho, RHO_CLAMP)
        f += RZ @ (K_R * (10.0 / rho ** 2) * rg ** POW * obs_grad(q, o))
    return f


def in_margin(q, obs):
    return any(dist(q, o) <= MARGIN for o in obs)


def run(stage, seed, wind):
    obs = [o for o in CFG["CURRICULUM"][stage]["obstacles"] if o["type"] in SOLID]
    rng = np.random.default_rng(seed)
    wm = WindModel(np.random.default_rng(10_000 + seed))
    wm.reset()
    lo, hi = np.array(BOUNDS[0::2]), np.array(BOUNDS[1::2])
    q = START.copy(); path = [q.copy()]
    bias = rng.normal(0, 0.06, size=3)
    st = {"escape_steps": 0, "pushouts": 0, "clamp_hits": 0}
    min_c = clearance(q, obs); outcome = "timeout"
    t0 = time.perf_counter()
    for _ in range(MAX_STEPS):
        if np.linalg.norm(q - GOAL) <= GOAL_RADIUS:
            outcome = "success"; break
        F = force(q, obs, st) + bias
        sd = F / (np.linalg.norm(F) + 1e-12)
        qn = np.clip(q + STEP * sd, lo, hi)
        k = 0
        while in_margin(qn, obs) and k < 5:
            perp = rng.normal(0, 1, 3); perp -= perp.dot(sd) * sd; perp /= np.linalg.norm(perp) + 1e-12
            qn = np.clip(q + STEP * 1.8 * perp + rng.normal(0, 0.15, 3), lo, hi); k += 1
        st["escape_steps"] += bool(k)
        if in_margin(qn, obs):
            d, o = min((dist(qn, o), o) for o in obs)
            qn = qn + (MARGIN - d + 3.0) * obs_grad(qn, o); st["pushouts"] += 1
        if wind:
            qn = np.clip(qn + wm.step() * 0.05 * WORLD_SCALE, lo, hi)
        q = qn; path.append(q.copy())
        cl = clearance(q, obs); min_c = min(min_c, cl)
        if cl < COLLISION_M:
            outcome = "collision"; break
    ms = (time.perf_counter() - t0) * 1000
    path = np.array(path)
    L = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return {"stage": stage, "seed": seed, "wind": wind, "outcome": outcome, "success": outcome == "success",
            "steps": len(path) - 1, "L": L, "eff": 100 * D_STRAIGHT / L if L else float("nan"),
            "time_ms": ms, "min_clearance": float(min_c), **st}, path


def summ(rows):
    ok = [r for r in rows if r["success"]]
    ms = lambda k: [float(np.mean([r[k] for r in ok])), float(np.std([r[k] for r in ok], ddof=1))] if len(ok) > 1 else \
        ([float(ok[0][k]), 0.0] if ok else [float("nan")] * 2)
    return {"n": len(rows), "success": len(ok), "collision": sum(r["outcome"] == "collision" for r in rows),
            "timeout": sum(r["outcome"] == "timeout" for r in rows),
            "steps": ms("steps"), "L": ms("L"), "eff": ms("eff"), "time_ms": ms("time_ms"),
            "min_clearance": min(r["min_clearance"] for r in rows),
            "escape_steps": sum(r["escape_steps"] for r in rows), "pushouts": sum(r["pushouts"] for r in rows),
            "clamp_hits": sum(r["clamp_hits"] for r in rows)}


if __name__ == "__main__":
    res = {"arena": {"bounds": BOUNDS, "start": START.tolist(), "goal": GOAL.tolist(), "goal_radius": GOAL_RADIUS,
                     "collision_margin": COLLISION_M, "d_straight": D_STRAIGHT,
                     "arc_max_steps": ARC_MAX_STEPS, "arc_max_step_m": ARC_STEP_MAX,
                     "n_solid": {st: sum(o["type"] in SOLID for o in CFG["CURRICULUM"][st]["obstacles"]) for st in (0, 1, 2)},
                     "types_stage2": [o["type"] for o in CFG["CURRICULUM"][2]["obstacles"]]}}
    paths = {}
    for stage in (0, 1, 2):
        for wind in (False, True):
            rows = []
            for seed in range(10):
                r, p = run(stage, seed, wind); rows.append(r)
                if seed == 0:
                    paths[f"s{stage}_{'wind' if wind else 'calm'}"] = p
            key = f"stage{stage}_{'wind' if wind else 'calm'}"
            res[key] = summ(rows); res[key + "_runs"] = rows
            print(key, {k: v for k, v in res[key].items()}, flush=True)
    np.savez_compressed(os.path.join(HERE, "out", "apf_in_arc3d_paths.npz"), **paths)
    with open(os.path.join(HERE, "out", "apf_in_arc3d.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res["arena"], default=float))
