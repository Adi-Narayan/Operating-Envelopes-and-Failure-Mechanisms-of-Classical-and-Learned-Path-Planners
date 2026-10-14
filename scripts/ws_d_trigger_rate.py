"""
WS-D: how often the switching trigger fires along SAC's own successful runs, at the D2 thresholds and at
the D3-selected thresholds, with the 2-s window definition of scripts/ws_d_d2_thresholds.py (held-out
motion seeds 500000-500049, the ten SAC seeds; no evaluation data).

    python scripts/ws_d_trigger_rate.py  -> results/v2/ws_d/trigger_rate/result.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump, now, outdir, write_manifest  # noqa: E402
from ws_d_d2_thresholds import LEAD                            # noqa: E402

V2 = os.path.join(ROOT, "results", "v2", "ws_d")


def main():
    t0 = now()
    d_ok, v_ok = [], []
    for s in range(10):
        for r in json.load(open(os.path.join(V2, "d2_signals", f"sac{s}.json")))["runs"]:
            sig = r["signals"]
            if r["outcome"] != "success" or len(sig) <= LEAD:
                continue
            for i in range(LEAD, len(sig), LEAD):
                w = sig[i - LEAD:i]
                d_ok.append(w[0][1])
                v_ok.append(max(v for _, _, v in w))
    d_ok, v_ok = np.array(d_ok), np.array(v_ok)
    d2 = json.load(open(os.path.join(V2, "d2_thresholds", "result.json")))
    sel = json.load(open(os.path.join(V2, "d3_selection", "result.json")))["selected"]
    res = {"windows": int(len(d_ok))}
    for name, (d_on, rho_on) in (("d2", (d2["d_on"], d2["rho_on"])), ("selected", (sel["d_on"], sel["rho_on"]))):
        res[name] = {"d_on": d_on, "rho_on": rho_on,
                     "fraction_triggering": float(np.mean((d_ok < d_on) | (v_ok > rho_on))),
                     "fraction_by_distance": float(np.mean(d_ok < d_on)), "fraction_by_closing": float(np.mean(v_ok > rho_on))}
    d = outdir("ws_d", "trigger_rate")
    dump(d, "result.json", res)
    write_manifest(d, "scripts/ws_d_trigger_rate.py", {"lead_steps": LEAD}, {"heldout_motion": "500000-500049", "sac_seeds": "0-9"}, t0)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
