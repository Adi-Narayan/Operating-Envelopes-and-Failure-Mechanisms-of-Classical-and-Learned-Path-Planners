"""
WS-C, C1: classify APF's free-space stall points on the 100 random static layouts.

For each layout on which the paper's APF (theta = 25 deg, planner seed 0) times out:
  1. rerun it and find the period P of the terminal oscillation (smallest P <= 12 with
     |q_end - q_{end-P}| < 1e-6 m over the last 3P steps) and its midpoint (mean of the last P points);
  2. Newton iterations on F_total(q) = 0 from the midpoint, with the exact force the planner uses
     (dynamic_apf.apf.F_total: Eqs. 1-3, including the 0.8 m clamp);
  3. the numerical Jacobian at the equilibrium (central differences, h = 1e-4 m), classified for the
     flow dq/dt = F(q) by its eigenvalues: stable node, stable focus, saddle, unstable node/focus;
  4. curl (dFy/dx - dFx/dy) and divergence at the equilibrium, normalised by |F_att| there / distance,
     so their size can be compared with the attraction's own gradient (-2 k_a, divergence of F_att).

    python scripts/ws_c_classify_stalls.py  -> results/v2/ws_c/c1_classify/
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                      # noqa: E402
import apf_variants as V                                 # noqa: E402
from random_layouts import make_layout                   # noqa: E402

dyn = H.dyn
GOAL = H.GOAL.astype(float)


def F(q, obst, cfg):
    return dyn.F_total(np.asarray(q, float), GOAL, obst, cfg)[0]


def jac(q, obst, cfg, h=1e-4):
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = h
        J[:, j] = (F(q + e, obst, cfg) - F(q - e, obst, cfg)) / (2 * h)
    return J


def newton(q0, obst, cfg, iters=60):
    q = np.asarray(q0, float).copy()
    f0 = np.linalg.norm(F(q, obst, cfg))
    for _ in range(iters):
        f = F(q, obst, cfg)
        if np.linalg.norm(f) < 1e-9 * max(f0, 1.0):
            break
        J = jac(q, obst, cfg)
        try:
            dq = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            return q, False
        step = 1.0
        while step > 1e-6:                      # backtracking on |F|
            qn = q + step * dq
            if np.linalg.norm(F(qn, obst, cfg)) < np.linalg.norm(f):
                break
            step *= 0.5
        q = qn
    return q, bool(np.linalg.norm(F(q, obst, cfg)) < 1e-6 * max(f0, 1.0))


def classify(ev):
    re, im = ev.real, ev.imag
    if np.all(np.abs(im) > 1e-9):
        return "stable focus" if re[0] < 0 else ("unstable focus" if re[0] > 0 else "centre")
    if re[0] < 0 and re[1] < 0:
        return "stable node"
    if re[0] > 0 and re[1] > 0:
        return "unstable node"
    return "saddle"


def period(p, max_p=12, tol=1e-6):
    for P in range(1, max_p + 1):
        if len(p) > 4 * P and all(np.linalg.norm(p[-1 - i] - p[-1 - i - P]) < tol for i in range(3 * P)):
            return P
    return None


def main():
    t0 = now()
    cfg = dyn.APFConfig()
    rows = {}
    for k in range(100):
        obst = make_layout(k)
        r = V.run(H.StaticTimeline(obst), 0, sensing="full", turn_cap=False, rate="apf")
        if r["outcome"] == "success":
            continue
        p = np.asarray(r["path"])
        P = period(p)
        mid = p[-(P or 2):].mean(axis=0)
        qs, ok = newton(mid, obst, cfg)
        J = jac(qs, obst, cfg)
        ev = np.linalg.eigvals(J)
        ev = ev[np.argsort(ev.real)]
        fa = dyn.F_att(qs, GOAL, cfg.k_a)
        clear = H.clearance(qs, obst)
        lo, hi = np.array(dyn.BOUNDS[0::2]), np.array(dyn.BOUNDS[1::2])
        at_wall = bool(P == 1 and (np.any(np.abs(p[-1] - lo) < 1e-9) or np.any(np.abs(p[-1] - hi) < 1e-9)))
        if at_wall:     # pinned against the arena boundary: the step is clipped back every time
            ok = False
        rows[k] = {"outcome": r["outcome"], "period": P, "arena_boundary_trap": at_wall, "midpoint": mid.tolist(), "equilibrium": qs.tolist(),
                   "newton_converged": ok, "dist_midpoint_to_eq": float(np.linalg.norm(qs - mid)),
                   "eq_to_last_point": float(np.linalg.norm(qs - p[-1])),
                   "eigenvalues": [[float(e.real), float(e.imag)] for e in ev], "type": "arena-boundary trap" if at_wall else (classify(ev) if ok else "no equilibrium found"),
                   "curl": float(J[1, 0] - J[0, 1]), "divergence": float(np.trace(J)),
                   "curl_over_div": float((J[1, 0] - J[0, 1]) / np.trace(J)) if np.trace(J) != 0 else None,
                   "F_att_at_eq": float(np.linalg.norm(fa)), "clearance_at_eq": clear,
                   "dist_goal_eq": float(np.linalg.norm(qs - GOAL)),
                   "last_steps_heading_flip_deg": float(np.degrees(np.arccos(np.clip(
                       np.dot(p[-1] - p[-2], p[-2] - p[-3]) / (np.linalg.norm(p[-1] - p[-2]) * np.linalg.norm(p[-2] - p[-3]) + 1e-12), -1, 1))))}
        print(k, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in rows[k].items()
                  if kk in ("period", "type", "dist_midpoint_to_eq", "curl", "divergence", "newton_converged")}, flush=True)
    types = {}
    for v in rows.values():
        types[v["type"]] = types.get(v["type"], 0) + 1
    periods = {}
    for v in rows.values():
        periods[str(v["period"])] = periods.get(str(v["period"]), 0) + 1
    summary = {"n_stalls": len(rows), "types": types, "periods": periods,
               "newton_converged": sum(v["newton_converged"] for v in rows.values()),
               "max_dist_midpoint_to_eq_m": max(v["dist_midpoint_to_eq"] for v in rows.values()),
               "eq_within_half_step_of_last_point": sum(v["eq_to_last_point"] <= 0.9 + 1e-9 for v in rows.values())}
    print(summary, flush=True)
    d = outdir("ws_c", "c1_classify")
    dump(d, "result.json", {"summary": summary, "stalls": rows})
    write_manifest(d, "scripts/ws_c_classify_stalls.py", {"theta_deg": 25, "k_a": cfg.k_a, "k_r": cfg.k_r, "step_m": cfg.step,
                                                          "jacobian_h_m": 1e-4}, {"layout_generator": "0-99", "planner_seed": 0}, t0)


if __name__ == "__main__":
    main()
