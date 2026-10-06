"""
Per-step planning cost of DWA and DWA-CV, measured as timing.py measures APF and ARC: on an
otherwise idle machine, one process, obstacle-timeline generation excluded. Static custom
environment (ten repeats) and the first ten dynamic environments, with the APF per-step
cost measured in the same session for reference.

    python dwa_timing.py          -> out/timing_dwa.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H   # noqa: E402
import dwa            # noqa: E402


def main():
    W = json.load(open(os.path.join(HERE, "out", "dwa_tune.json")))["selected"]
    res = {}
    for var in ("DWA", "DWA-CV"):
        w = W[var]["weights"]
        st = [dwa.run(H.StaticTimeline(), predict=var == "DWA-CV", weights=w) for _ in range(10)]
        dy = [dwa.run(H.ObstacleTimeline(s), predict=var == "DWA-CV", weights=w) for s in range(10)]
        res[var] = {"static_ms_per_step": [float(np.mean([r["ms_per_step"] for r in st])), float(np.std([r["ms_per_step"] for r in st], ddof=1))],
                    "static_run_ms": [float(np.mean([r["time_ms"] for r in st])), float(np.std([r["time_ms"] for r in st], ddof=1))],
                    "static_outcome": st[0]["outcome"], "static_steps": st[0]["steps"], "static_L": st[0]["path_length"],
                    "dynamic_ms_per_step": float(np.mean([r["ms_per_step"] for r in dy]))}
        print(var, res[var], flush=True)
    ap = [H.run_apf(H.StaticTimeline(), 0) for _ in range(10)]
    res["APF_same_session"] = {"static_ms_per_step": float(np.mean([r["ms_per_step"] for r in ap])),
                               "static_run_ms": float(np.mean([r["time_ms"] for r in ap]))}
    print(res["APF_same_session"])
    json.dump(res, open(os.path.join(HERE, "out", "timing_dwa.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
