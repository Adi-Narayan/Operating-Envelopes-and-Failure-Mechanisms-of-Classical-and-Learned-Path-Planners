"""
WS-D, D2: initial thresholds from SAC alone on the held-out motion seeds (prereg.md rule).
For every SAC collision: d_min at the learned observation 12 steps (2 s) before the collision, and the
largest v_c over the 12 observations before it. d_on = median of the former, rho_on = median of the
latter, over all collisions of the 10 SAC seeds. The same quantities over successful runs (taken at
matching positions from the end of the run) are reported as a false-alarm view.

    python scripts/ws_d_d2_thresholds.py  -> results/v2/ws_d/d2_thresholds/
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump, now, outdir, write_manifest  # noqa: E402

LEAD = 12


def main():
    t0 = now()
    src = os.path.join(ROOT, "results", "v2", "ws_d", "d2_signals")
    d_at, vmax, d_ok, v_ok, n_coll, n_succ = [], [], [], [], 0, 0
    for s in range(10):
        data = json.load(open(os.path.join(src, f"sac{s}.json")))
        for r in data["runs"]:
            sig = r["signals"]
            if len(sig) <= LEAD:
                continue
            window = sig[-(LEAD + 1):-1]            # the 12 learned observations before the final one
            if r["outcome"] == "collision":
                n_coll += 1
                d_at.append(window[0][1])
                vmax.append(max(v for _, _, v in window))
            elif r["outcome"] == "success":
                n_succ += 1
                for i in range(LEAD, len(sig), LEAD):  # every 2 s along successful runs
                    w = sig[i - LEAD:i]
                    d_ok.append(w[0][1])
                    v_ok.append(max(v for _, _, v in w))
    d_on, rho_on = float(np.median(d_at)), float(np.median(vmax))
    out = {"n_collisions": n_coll, "n_successes": n_succ, "d_on": d_on, "rho_on": rho_on,
           "d_off": d_on + 5.0, "rho_off": 0.8 * rho_on,
           "collision_d_min_at_minus2s": {"p25": float(np.percentile(d_at, 25)), "median": d_on, "p75": float(np.percentile(d_at, 75))},
           "collision_vc_max_2s": {"p25": float(np.percentile(vmax, 25)), "median": rho_on, "p75": float(np.percentile(vmax, 75))},
           "success_windows_triggering_fraction": float(np.mean([(d < d_on) or (v > rho_on) for d, v in zip(d_ok, v_ok)])) if d_ok else None,
           "grid": {"d_on": [d_on - 5.0, d_on, d_on + 5.0], "rho_on": [0.75 * rho_on, rho_on, 1.25 * rho_on]}}
    print(json.dumps(out, indent=1))
    d = outdir("ws_d", "d2_thresholds")
    dump(d, "result.json", out)
    write_manifest(d, "scripts/ws_d_d2_thresholds.py", {"rule": "prereg.md WS-D D2", "lead_steps": LEAD},
                   {"heldout_motion": "500000-500049", "sac_seeds": "0-9"}, t0)


if __name__ == "__main__":
    main()
