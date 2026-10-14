"""
WS-C follow-up (Adi's review of Appendix E): exact stability of every node two-cycle. For each stable-node
stall, the stalled run is reproduced, the planner's step map T(q) = q + Ds (F(q) + b)/|F(q) + b| is
composed twice, and the eigenvalues of the Jacobian of T o T at the last cycle point (central differences)
give the cycle's true multipliers. Also: whether the run is an exact period-2 orbit, the cycle's offset
from the fast eigenline in eigen-coordinates, and the linearised multiplier with x0 measured in
eigen-coordinates (m = 1 - rho (1 - rho) / (a (1 - a)), a = x0 / (x0 + |x1|)).

    python scripts/ws_c_cycle_multipliers.py  -> results/v2/ws_c/cycle_multipliers/result.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                           # noqa: E402
import apf_variants as V                                      # noqa: E402
from random_layouts import make_layout                        # noqa: E402
from ws_c_c2_followup import F, jac, newton, dyn, DS          # noqa: E402


def step(q, obst, cfg, b):
    f = F(q, obst, cfg, b)
    return q + DS * f / np.linalg.norm(f)


def two_step_jac(q, obst, cfg, b, h=1e-6):
    T2 = lambda x: step(step(x, obst, cfg, b), obst, cfg, b)
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = h
        J[:, j] = (T2(q + e) - T2(q - e)) / (2 * h)
    return J


def main():
    t0 = now()
    cfg = dyn.APFConfig()
    b = np.random.default_rng(0).normal(0, 0.06, size=2)
    c2 = json.load(open(os.path.join(ROOT, "results", "v2", "ws_c", "c2_followup", "result.json")))["stalls"]
    rows = {}
    for k, s in c2.items():
        if s["type"] != "stable node":
            continue
        obst = make_layout(int(k))
        r = V.run(H.StaticTimeline(obst), 0, sensing="full", turn_cap=False, rate="apf")
        p = np.asarray(r["path"])
        q_eq, _ = newton(0.5 * (p[-1] + p[-2]), obst, cfg, b)
        ev, evec = np.linalg.eig(jac(q_eq, obst, cfg, b))
        ev, evec = np.real(ev), np.real(evec)
        fast, slow = int(np.argmax(np.abs(ev))), int(np.argmin(np.abs(ev)))
        Binv = np.linalg.inv(np.column_stack([evec[:, fast], evec[:, slow]]))
        xy = (Binv @ (p[-2:] - q_eq).T).T
        xs = sorted(xy[:, 0])
        a = xs[1] / (xs[1] - xs[0])
        rho = abs(ev[slow]) / abs(ev[fast])
        mu = np.linalg.eigvals(two_step_jac(p[-1], obst, cfg, b))
        neutral, trans = sorted(mu, key=lambda z: abs(abs(z) - 1.0))   # one multiplier is 1: shifts along the line
        map_err = float(np.linalg.norm(step(step(p[-1], obst, cfg, b), obst, cfg, b) - p[-1]))
        rows[k] = {"exact_period2_m": float(np.max(np.linalg.norm(p[-20:] - p[-22:-2], axis=1))),
                   "map_reproduces_cycle_m": map_err,
                   "multipliers": [[float(np.real(z)), float(np.imag(z))] for z in mu],
                   "max_abs_multiplier": float(np.max(np.abs(mu))),
                   "neutral_multiplier": float(abs(neutral)), "transverse_multiplier": float(np.real(trans)),
                   "offset_slow_coord": float(np.max(np.abs(xy[:, 1]))),
                   "rho": float(rho), "a_eigen": float(a), "m_linear_eigen": float(1 - rho * (1 - rho) / (a * (1 - a))),
                   "m_linear_c2": s.get("m")}
        print(k, {kk: rows[k][kk] for kk in ("map_reproduces_cycle_m", "max_abs_multiplier", "m_linear_eigen", "m_linear_c2")}, flush=True)
    summ = {"nodes": len(rows),
            "all_true_multipliers_inside_unit_circle": all(v["max_abs_multiplier"] < 1 for v in rows.values()),
            "max_abs_true_multiplier": max(v["max_abs_multiplier"] for v in rows.values()),
            "max_abs_transverse": max(abs(v["transverse_multiplier"]) for v in rows.values()),
            "neutral_dev_max": max(abs(v["neutral_multiplier"] - 1) for v in rows.values()),
            "transverse_89": rows["89"]["transverse_multiplier"] if "89" in rows else None,
            "linear_eigen_abs_m_below_1": sum(abs(v["m_linear_eigen"]) < 1 for v in rows.values()),
            "max_map_error_m": max(v["map_reproduces_cycle_m"] for v in rows.values()),
            "max_period2_dev_m": max(v["exact_period2_m"] for v in rows.values())}
    d = outdir("ws_c", "cycle_multipliers")
    dump(d, "result.json", {"summary": summ, "cycles": rows})
    write_manifest(d, "scripts/ws_c_cycle_multipliers.py", {"Ds": DS, "fd_step": 1e-6}, {"planner": 0}, t0)
    print(summ)


if __name__ == "__main__":
    main()
