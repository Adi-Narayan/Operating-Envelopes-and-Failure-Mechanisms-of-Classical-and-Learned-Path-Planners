"""
WS-D, D2: switching signals of SAC alone on the held-out motion seeds 500000-500049 (prereg.md),
for one SAC seed. The threshold rule is applied by scripts/ws_d_d2_thresholds.py.

    python scripts/ws_d_d2_signals.py <sac seed>  -> results/v2/ws_d/d2_signals/sac<seed>.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import OUT, dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                           # noqa: E402
import arbiter as A                                           # noqa: E402


def main():
    seed = int(sys.argv[1])
    t0 = now()
    H.set_ground_speed(18.0)
    L = A.Learned(os.path.join(OUT, "arc_models", f"sacdyn_seed{seed}", "models", "arc_best.pth"))
    runs = []
    for s in range(500_000, 500_050):
        r = A.run(H.ObstacleTimeline(s), L, A.APFRays(), "apf_rays", -np.inf, np.inf, log_signals=True)
        runs.append({"motion_seed": s, "outcome": r["outcome"], "end_tick": r["end_tick"],
                     "signals": [[int(k), float(d), float(v)] for k, d, v, _ in r["signals"]]})
    print(seed, {o: sum(x["outcome"] == o for x in runs) for o in ("success", "collision", "timeout", "out_of_bounds")}, flush=True)
    d = outdir("ws_d", "d2_signals")
    dump(d, f"sac{seed}.json", {"sac_seed": seed, "runs": runs})
    write_manifest(d, f"scripts/ws_d_d2_signals.py {seed}", {"L": f"sacdyn_seed{seed}/arc_best", "mode": "L alone"},
                   {"heldout_motion": "500000-500049"}, t0)


if __name__ == "__main__":
    main()
