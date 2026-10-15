"""
C2 follow-ups from Adi's review (items 3-6), on the 31 interior stalls of results/v2/ws_c/c1_classify.

  3  Corollary 3 check: divergence of F at every equilibrium is negative; det J decides sink vs saddle.
  4  Shifted equilibrium: the step follows F + b (b drawn once per run; planner seed 0), so the fixed point
     of the motion is F(q) + b = 0. Newton from the c1 equilibrium; |J^-1 b| (first-order shift) and the
     exact shift; orbit midpoints re-measured from the shifted point.
  5  Proposition 1 check (nodes): eigen-decomposition of J at the shifted point; angle between the
     oscillation (last two positions) and the fast eigenvector; x0 = position of one cycle point along
     the eigenline; rho = |lambda_slow| / |lambda_fast|; transverse two-step multiplier
     m = (1 - rho Ds / x0)(1 - rho Ds / (Ds - x0)); |m| < 1 expected.
  6  Nearest saddle of F + b to each sink (Newton from a grid of starts within 80 m, zeros outside
     obstacles with det J < 0), against the turning diameters of the 54 deg/s cap (38.2 m) and of APF-D
     (max turn arcsin((1 - beta) / beta) = 10.2 deg per 1.8 m step, 20.3 m); and whether
     "nearest saddle closer than the diameter" predicts which stalls each fix removes (cap: APF with
     the always-on cap, full sensing, 10 Hz, rerun here per layout; APF-D: results/v2/ws_c/apfd_eval).

    python scripts/ws_c_c2_followup.py  -> results/v2/ws_c/c2_followup/result.json
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                            # noqa: E402
import apf_variants as V                                       # noqa: E402
from random_layouts import make_layout                         # noqa: E402

dyn = H.dyn
GOAL = H.GOAL.astype(float)
DS = 1.8
BETA = 0.85
CAP_DEG = 54.0 * 0.1                                   # 54 deg/s at 10 Hz
diam = lambda c_deg: 2 * DS / (2 * math.sin(math.radians(c_deg) / 2))
D_CAP = diam(CAP_DEG)
C_APFD = math.degrees(math.asin((1 - BETA) / BETA))
D_APFD = diam(C_APFD)


def F(q, obst, cfg, b):
    return dyn.F_total(np.asarray(q, float), GOAL, obst, cfg)[0] + b


def jac(q, obst, cfg, b, h=1e-4):
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = h
        J[:, j] = (F(q + e, obst, cfg, b) - F(q - e, obst, cfg, b)) / (2 * h)
    return J


def newton(q0, obst, cfg, b, iters=80):
    q = np.asarray(q0, float).copy()
    for _ in range(iters):
        f = F(q, obst, cfg, b)
        if np.linalg.norm(f) < 1e-9:
            return q, True
        try:
            dq = np.linalg.solve(jac(q, obst, cfg, b), -f)
        except np.linalg.LinAlgError:
            return q, False
        n = np.linalg.norm(dq)
        q = q + (dq if n < 5.0 else dq * 5.0 / n)
        if not (0 <= q[0] <= 500 and 0 <= q[1] <= 500):
            return q, False
    return q, np.linalg.norm(F(q, obst, cfg, b)) < 1e-6


def saddles_near(q_sink, obst, cfg, b, radius=80.0, step=8.0):
    found = []
    g = np.arange(-radius, radius + 1e-9, step)
    for dx in g:
        for dy in g:
            if dx * dx + dy * dy > radius * radius:
                continue
            q, ok = newton(q_sink + np.array([dx, dy]), obst, cfg, b, iters=40)
            if not ok or H.clearance(q, obst) <= 0.8:
                continue
            if np.linalg.det(jac(q, obst, cfg, b)) < 0 and all(np.linalg.norm(q - p) > 0.05 for p in found):
                found.append(q)
    return found


def main():
    t0 = now()
    cfg = dyn.APFConfig()
    b = np.random.default_rng(0).normal(0, 0.06, size=2)       # planner seed 0, as in apf_variants.run
    c1 = json.load(open(os.path.join(ROOT, "results", "v2", "ws_c", "c1_classify", "result.json")))["stalls"]
    apfd = json.load(open(os.path.join(ROOT, "results", "v2", "ws_c", "apfd_eval", "layouts.json")))["static"]
    rows = {}
    for k, s in c1.items():
        if s["arena_boundary_trap"] or not s["newton_converged"]:
            continue
        obst = make_layout(int(k))
        q_old = np.asarray(s["equilibrium"])
        J0 = jac(q_old, obst, cfg, np.zeros(2))
        lin_shift = float(np.linalg.norm(np.linalg.solve(J0, b)))
        q_eq, ok = newton(q_old, obst, cfg, b)
        J = jac(q_eq, obst, cfg, b)
        ev, evec = np.linalg.eig(J)
        r = V.run(H.StaticTimeline(obst), 0, sensing="full", turn_cap=False, rate="apf")
        p = np.asarray(r["path"])
        row = {"type": s["type"], "bias_norm": float(np.linalg.norm(b)), "shift_linear": lin_shift,
               "shift_exact": float(np.linalg.norm(q_eq - q_old)), "shifted_converged": bool(ok),
               "div": float(np.trace(J)), "det": float(np.linalg.det(J)),
               "eigenvalues": [[float(np.real(x)), float(np.imag(x))] for x in ev]}
        if s["type"] == "stable node" and np.all(np.isreal(ev)):
            ev, evec = np.real(ev), np.real(evec)
            fast, slow = int(np.argmax(np.abs(ev))), int(np.argmin(np.abs(ev)))
            osc = p[-1] - p[-2]
            cosang = abs(np.dot(osc, evec[:, fast])) / (np.linalg.norm(osc) * np.linalg.norm(evec[:, fast]))
            mid = 0.5 * (p[-1] + p[-2])
            # coordinates of the two cycle points along the fast eigenline, relative to the shifted equilibrium
            e = evec[:, fast] / np.linalg.norm(evec[:, fast])
            xs = sorted([float(np.dot(p[-1] - q_eq, e)), float(np.dot(p[-2] - q_eq, e))])
            x0 = xs[1]                                     # the point on the positive side
            rho = abs(ev[slow]) / abs(ev[fast])
            m = (1 - rho * DS / x0) * (1 - rho * DS / (DS - x0)) if 0 < x0 < DS else float("nan")
            row.update({"angle_osc_fast_deg": float(math.degrees(math.acos(min(1.0, cosang)))),
                        "mid_to_shifted_eq": float(np.linalg.norm(mid - q_eq)), "mid_to_unshifted_eq": float(np.linalg.norm(mid - q_old)),
                        "x0": x0, "x_other": xs[0], "rho": float(rho), "m": float(m)})
        sad = saddles_near(q_eq, obst, cfg, b)
        dsad = min((float(np.linalg.norm(q - q_eq)) for q in sad), default=float("inf"))
        cap = V.run(H.StaticTimeline(obst), 0, sensing="full", turn_cap=True, rate="apf")["outcome"]
        row.update({"nearest_saddle_m": dsad, "n_saddles_found": len(sad), "cap_outcome": cap,
                    "apfd_outcome": apfd[int(k)]["outcome"],
                    "pred_cap_escapes": dsad < D_CAP, "pred_apfd_escapes": dsad < D_APFD})
        rows[k] = row
        print(k, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items() if kk != "eigenvalues"}, flush=True)

    def table(pred_key, out_key):
        t = {"pred_escape_and_removed": 0, "pred_escape_not_removed": 0, "pred_stay_removed": 0, "pred_stay_not_removed": 0}
        for r in rows.values():
            rem = r[out_key] == "success"
            t[("pred_escape" if r[pred_key] else "pred_stay") + ("_and_removed" if rem and r[pred_key] else "_removed" if rem else "_not_removed")] += 1
        return t
    nodes = [r for r in rows.values() if "m" in r]
    summ = {"bias": b.tolist(), "turn_diameter_cap_m": D_CAP, "turn_deg_apfd": C_APFD, "turn_diameter_apfd_m": D_APFD,
            "all_div_negative": all(r["div"] < 0 for r in rows.values()),
            "det_sign": {"positive": sum(r["det"] > 0 for r in rows.values()), "negative": sum(r["det"] < 0 for r in rows.values())},
            "shift_linear_range": [min(r["shift_linear"] for r in rows.values()), max(r["shift_linear"] for r in rows.values())],
            "shift_exact_range": [min(r["shift_exact"] for r in rows.values()), max(r["shift_exact"] for r in rows.values())],
            "nodes": len(nodes),
            "angle_osc_fast_deg_range": [min(r["angle_osc_fast_deg"] for r in nodes), max(r["angle_osc_fast_deg"] for r in nodes)],
            "mid_to_shifted_eq_range": [min(r["mid_to_shifted_eq"] for r in nodes), max(r["mid_to_shifted_eq"] for r in nodes)],
            "abs_m_max": max(abs(r["m"]) for r in nodes), "all_abs_m_lt1": all(abs(r["m"]) < 1 for r in nodes),
            "rho_range": [min(r["rho"] for r in nodes), max(r["rho"] for r in nodes)],
            "cap": table("pred_cap_escapes", "cap_outcome"), "apfd": table("pred_apfd_escapes", "apfd_outcome"),
            "cap_outcomes": {o: sum(r["cap_outcome"] == o for r in rows.values()) for o in ("success", "timeout", "collision")},
            "apfd_outcomes": {o: sum(r["apfd_outcome"] == o for r in rows.values()) for o in ("success", "timeout", "collision")}}
    d = outdir("ws_c", "c2_followup")
    dump(d, "result.json", {"summary": summ, "stalls": rows})
    write_manifest(d, "scripts/ws_c_c2_followup.py", {"beta": BETA, "cap_deg_per_step": CAP_DEG, "saddle_search_radius_m": 80},
                   {"layouts": "the 31 interior stalls of c1", "planner_seed": 0}, t0)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
