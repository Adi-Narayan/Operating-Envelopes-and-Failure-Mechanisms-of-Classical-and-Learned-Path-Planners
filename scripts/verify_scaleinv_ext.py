"""
E1 follow-up (Adi's review, item 8): extend the scale-invariance check of verify_section3.py scaleinv.

(A) agent 18 m/s with the baseline obstacle speeds against (B) agent 9 m/s with every obstacle velocity
x0.5 (same random draws, so r is equal), for APF, SAC (sacdyn seed 0) and DWA-CV, each without noise and
with noise level L2 (2.5 m, 5 deg, noise seed 50000 + s). Event counts of every obstacle branch the proof
relies on (goal push-away, wall reflection, obstacle separation) are logged per run, so each branch is
seen to fire; motion seeds are added until every branch has fired in at least one compared run.

    python scripts/verify_scaleinv_ext.py  -> results/v2/verification/scale_invariance_ext/result.json
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import OUT, dump, now, outdir, training_active, write_manifest  # noqa: E402
import harness as H                                                           # noqa: E402
import dwa                                                                    # noqa: E402

apf = H.dyn
SAC = os.path.join(OUT, "arc_models", "sacdyn_seed0", "models", "arc_best.pth")
EV = {"push_away": 0, "wall": 0, "separation": 0}

_step, _coll = apf.DynObstacle.step, apf.DynObstacle.handle_obs_collision


def step(self, dt, goal=None):
    nxt = np.array([self.cx + self.vel[0] * dt, self.cy + self.vel[1] * dt])   # position the branch tests
    if goal is not None and np.linalg.norm(nxt - goal) < apf.GOAL_PROTECTION_RADIUS:
        EV["push_away"] += 1
    v0 = self.vel.copy()
    out = _step(self, dt, goal)
    if np.any(np.sign(self.vel) != np.sign(v0)) and np.linalg.norm(self.vel) > 0:
        EV["wall"] += 1
    return out


def coll(self, other):
    a, b = apf.get_aabb(self.obs), apf.get_aabb(other.obs)
    # count material separations only: after a separation, pairs sit at contact, and overlaps of
    # order 1e-9 m flip with rounding (first run of this script counted those, see PROGRESS.md)
    if min(a[1] - b[0], b[1] - a[0], a[3] - b[2], b[3] - a[2]) > 1e-6:
        EV["separation"] += 1
    return _coll(self, other)


apf.DynObstacle.step, apf.DynObstacle.handle_obs_collision = step, coll


def scaled(seed, f):
    tl = H.ObstacleTimeline(seed)
    for d in tl.dyns:
        d.vel = d.vel * f
    return tl


def compare(x, y):
    same = len(x["path"]) == len(y["path"])
    dev = float(np.abs(np.asarray(x["path"]) - np.asarray(y["path"])).max()) if same else float("inf")
    return {"outcome_A": x["outcome"], "outcome_B": y["outcome"], "steps_A": x["steps"], "steps_B": y["steps"],
            "max_dev_m": dev, "bit_identical": same and bool(np.array_equal(np.asarray(x["path"]), np.asarray(y["path"])))}


def main():
    t0, busy = now(), training_active()
    P = H.ARCPlanner(SAC)
    w = json.load(open(os.path.join(OUT, "dwa_tune.json")))["selected"]["DWA-CV"]["weights"]
    planners = {
        "APF": lambda tl, s, nz: H.run_apf(tl, 1000 + s) if not nz else __import__("apf_variants").run(
            tl, 1000 + s, range_noise=2.5, heading_noise=math.radians(5.0), noise_seed=50_000 + s),
        "SAC": lambda tl, s, nz: P.run(tl, **({"range_noise": 2.5, "heading_noise": math.radians(5.0), "noise_seed": 50_000 + s} if nz else {})),
        "DWA-CV": lambda tl, s, nz: dwa.run(tl, predict=True, weights=w, record=True,
                                            **({"range_noise": 2.5, "heading_noise": math.radians(5.0), "noise_seed": 50_000 + s} if nz else {})),
    }
    rows, fired, s = [], {k: 0 for k in EV}, 0
    while s < 10 or (min(fired["wall"], fired["separation"]) == 0 and s < 50):   # push-away: timeline-only check below
        for name, f in planners.items():
            for nz in (False, True):
                for k in EV:
                    EV[k] = 0
                H.set_ground_speed(18.0)
                x = f(H.ObstacleTimeline(s), s, nz)
                ev_a = dict(EV)
                for k in EV:
                    EV[k] = 0
                H.set_ground_speed(9.0)
                y = f(scaled(s, 0.5), s, nz)
                ev_b = dict(EV)
                row = {"env": s, "planner": name, "noise": "L2" if nz else "none", **compare(x, y), "events_A": ev_a, "events_B": ev_b}
                rows.append(row)
                for k in EV:
                    fired[k] += ev_a[k] > 0
                print(s, name, row["noise"], row["outcome_A"], row["outcome_B"], round(row["max_dev_m"], 6), ev_a, flush=True)
        s += 1
    H.set_ground_speed(18.0)
    summ = {}
    for name in planners:
        for nz in ("none", "L2"):
            rr = [r for r in rows if r["planner"] == name and r["noise"] == nz]
            summ[f"{name} {nz}"] = {"n": len(rr), "same_outcome": sum(r["outcome_A"] == r["outcome_B"] for r in rr),
                                    "same_steps": sum(r["steps_A"] == r["steps_B"] for r in rr),
                                    "bit_identical": sum(r["bit_identical"] for r in rr), "max_dev_m": max(r["max_dev_m"] for r in rr)}
    summ["runs_with_event"] = fired
    summ["events_equal_A_B"] = all(r["events_A"] == r["events_B"] for r in rows)
    # timeline-only coverage of the goal push-away branch, which no planner run reaches: long timelines
    # (20000 ticks) under both speed conditions, until the branch has fired in at least five seeds
    tl_rows, seed = [], 0
    while sum(r["events_A"]["push_away"] > 0 for r in tl_rows) < 5 and seed < 200:
        out = {}
        for tag, v, f in (("A", 18.0, 1.0), ("B", 9.0, 0.5)):
            for k in EV:
                EV[k] = 0
            H.set_ground_speed(v)
            tl = scaled(seed, f)
            tl.at(20000 * tl.dt)
            out[tag] = (tl, dict(EV))
        (ta, ea), (tb, eb) = out["A"], out["B"]
        dev = max(float(np.max(np.abs(np.asarray(oa["params"], float) - np.asarray(ob["params"], float))))
                  for sa, sb in zip(ta.states, tb.states) for oa, ob in zip(sa, sb))
        tl_rows.append({"seed": seed, "events_A": ea, "events_B": eb, "max_obstacle_dev_m": dev})
        print("timeline", seed, ea, eb, dev, flush=True)
        seed += 1
    H.set_ground_speed(18.0)
    summ["timeline_only"] = {"seeds": len(tl_rows), "with_push_away": sum(r["events_A"]["push_away"] > 0 for r in tl_rows),
                             "events_equal": all(r["events_A"] == r["events_B"] for r in tl_rows),
                             "max_obstacle_dev_m": max(r["max_obstacle_dev_m"] for r in tl_rows)}
    rows.extend({"timeline_only": True, **r} for r in tl_rows)
    d = outdir("verification", "scale_invariance_ext2")
    dump(d, "result.json", {"summary": summ, "rows": rows})
    write_manifest(d, "scripts/verify_scaleinv_ext.py", {"A": "18 m/s", "B": "9 m/s, obstacle velocities x0.5", "noise_L2": [2.5, 5.0]},
                   {"motion": f"0-{s - 1}", "apf_planner": "1000+s", "noise": "50000+s"}, t0, busy)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
