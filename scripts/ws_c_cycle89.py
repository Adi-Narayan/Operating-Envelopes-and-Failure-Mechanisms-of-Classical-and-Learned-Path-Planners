"""
WS-C follow-up (Adi's review of Appendix E): the one node stall whose two-cycle has a linearised transverse
multiplier |m| > 1 (layout 89). Reruns the stalled run and decomposes the last 600 positions in the
eigenbasis of the Jacobian at the shifted equilibrium: coordinate x along the fast eigenvector, y along
the slow one. Reports the smallest period (up to 64) of the terminal orbit at several tolerances, the
envelope of |y| over time, and the multiplier the run itself shows (ratio of successive |y| at even steps).

    python scripts/ws_c_cycle89.py [layout]  -> results/v2/ws_c/cycle89/result.json
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                    # noqa: E402
import apf_variants as V                               # noqa: E402
from random_layouts import make_layout                 # noqa: E402
from ws_c_c2_followup import jac, newton, dyn          # noqa: E402

TAIL = 600


def smallest_period(p, tol, max_p=64, span=3):
    for ell in range(1, max_p + 1):
        if len(p) > span * ell and np.max(np.linalg.norm(p[-span * ell:] - p[-span * ell - ell:-ell], axis=1)) < tol:
            return ell
    return None


def main():
    t0 = now()
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 89
    obst = make_layout(k)
    cfg = dyn.APFConfig()
    b = np.random.default_rng(0).normal(0, 0.06, size=2)
    r = V.run(H.StaticTimeline(obst), 0, sensing="full", turn_cap=False, rate="apf")
    p = np.asarray(r["path"])
    mid = 0.5 * (p[-1] + p[-2])
    q_eq, ok = newton(mid, obst, cfg, b)
    J = jac(q_eq, obst, cfg, b)
    ev, evec = np.linalg.eig(J)
    ev, evec = np.real(ev), np.real(evec)
    fast, slow = int(np.argmax(np.abs(ev))), int(np.argmin(np.abs(ev)))
    Binv = np.linalg.inv(np.column_stack([evec[:, fast], evec[:, slow]]))
    tail = p[-TAIL:]
    xy = (Binv @ (tail - q_eq).T).T                     # columns: x (fast), y (slow), eigenvector units
    x, y = xy[:, 0], xy[:, 1]
    periods = {tol: smallest_period(p, float(tol)) for tol in ("1e-12", "1e-9", "1e-6", "1e-3")}
    ay = np.abs(y)
    even = ay[::2]
    ratios = even[1:] / np.maximum(even[:-1], 1e-300)
    rho = abs(ev[slow]) / abs(ev[fast])
    DS = 1.8
    xs = sorted([float(x[-1]), float(x[-2])])
    a = xs[1] / DS if 0 < xs[1] < DS else float("nan")
    m_lin = 1 - rho * (1 - rho) / (a * (1 - a))
    res = {"layout": k, "outcome": r["outcome"], "steps": r["steps"], "newton_converged": bool(ok),
           "eigenvalues": ev.tolist(), "rho": float(rho), "a": a, "m_linear": float(m_lin),
           "periods_by_tol": periods, "x_last4": x[-4:].tolist(), "y_last8": y[-8:].tolist(),
           "abs_y_max_by_window": {f"{i}-{i + 100}": float(ay[i:i + 100].max()) for i in range(0, TAIL, 100)},
           "two_step_y_ratio_median_last100": float(np.median(ratios[-50:])),
           "x_max_dev_from_2cycle_last100": float(np.max(np.abs(x[-100:] - np.tile(x[-2:], 50)))),
           "jitter_or_escape": {"escape_steps": r.get("escape_steps"), "pushouts": r.get("pushouts")}}
    d = outdir("ws_c", "cycle89")
    dump(d, "result.json", res)
    write_manifest(d, "scripts/ws_c_cycle89.py", {"tail": TAIL, "layout": k}, {"planner": 0}, t0)
    for kk, v in res.items():
        print(kk, v)


if __name__ == "__main__":
    main()
