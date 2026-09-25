"""
15 x 15 (k_a, k_r) sweep on the custom environment with the composite score of
Eq. (9): S = L + 0.2 R + 1000 [collision] + 500 [no goal], theta fixed at 25 deg.

Two grids are swept:
  doc   : k_a in [1, 5], k_r in [0.01, 0.50]  (grid of APF_Methodology.docx)
  ext   : k_a in [1, 5], k_r in [0.15, 1.55]  (step 0.1, contains the reported k_r = 0.85)
Wall time for each 225-combination sweep is recorded.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H   # noqa: E402

sta = H.sta
OBS = sta.ENVIRONMENTS[1]["obstacles"]


def roughness(path):
    d = np.diff(path, axis=0)
    psi = np.arctan2(d[:, 1], d[:, 0])
    dpsi = np.arctan2(np.sin(np.diff(psi)), np.cos(np.diff(psi)))
    return float(np.sum(dpsi ** 2))


def score(ka, kr, seed=0):
    cfg = sta.APFConfig()
    cfg.k_a, cfg.k_r = ka, kr
    ok, path, _, info = sta.simulate_static_apf(H.START.copy(), H.GOAL.copy(), OBS, cfg,
                                                rng=np.random.default_rng(seed))
    L = sta.path_length(path)
    R = roughness(path)
    S = L + 0.2 * R + 1000.0 * (info["outcome"] == "collision") + 500.0 * (not ok)
    return {"ka": ka, "kr": kr, "success": bool(ok), "outcome": info["outcome"], "steps": len(path) - 1,
            "L": L, "R": R, "S": S}


def sweep(kas, krs):
    t0 = time.perf_counter()
    rows = [score(float(a), float(r)) for a in kas for r in krs]
    wall = time.perf_counter() - t0
    ok = sorted([r for r in rows if r["success"]], key=lambda r: r["S"])
    return {"wall_s": wall, "n": len(rows), "n_success": len(ok),
            "n_timeout": sum(r["outcome"] == "timeout" for r in rows),
            "best": ok[0] if ok else None, "top10": ok[:10], "rows": rows}


if __name__ == "__main__":
    kas = np.linspace(1.0, 5.0, 15)
    res = {"doc": sweep(kas, np.linspace(0.01, 0.50, 15)),
           "ext": sweep(kas, np.round(np.linspace(0.15, 1.55, 15), 4))}
    ref = score(5.0, 0.85)
    res["reported_5.0_0.85"] = ref
    ext_sorted = sorted([r for r in res["ext"]["rows"] if r["success"]], key=lambda r: r["S"])
    res["rank_of_reported_in_ext"] = next((i + 1 for i, r in enumerate(ext_sorted)
                                           if abs(r["ka"] - 5.0) < 1e-9 and abs(r["kr"] - 0.85) < 1e-9), None)
    with open(os.path.join(HERE, "out", "gain_sweep.json"), "w") as f:
        json.dump(res, f, indent=1)
    for g in ("doc", "ext"):
        r = res[g]
        print(f"{g}: wall {r['wall_s']:.1f}s  success {r['n_success']}/{r['n']}  timeouts {r['n_timeout']}")
        for t in r["top10"][:5]:
            print(f"    ka={t['ka']:.3f} kr={t['kr']:.3f} kr/ka={t['kr']/t['ka']:.4f} S={t['S']:.2f} L={t['L']:.1f} R={t['R']:.2f}")
    print("reported (5.0, 0.85):", {k: round(v, 2) if isinstance(v, float) else v for k, v in ref.items()},
          "rank in ext grid:", res["rank_of_reported_in_ext"])
