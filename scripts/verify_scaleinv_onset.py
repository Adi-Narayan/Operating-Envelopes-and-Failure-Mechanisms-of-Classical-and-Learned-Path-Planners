"""
E1 follow-up: in long obstacle timelines (20000 ticks, longer than any evaluated run needs), the first tick at
which the timeline at 18 m/s and the one at 9 m/s with obstacle velocities x0.5 differ by more than
1e-6 m, 1e-3 m and 0.1 m (motion seeds 0-6, those of the timeline-only check in verify_scaleinv_ext.py).

    python scripts/verify_scaleinv_onset.py  -> results/v2/verification/scale_invariance_onset/result.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                      # noqa: E402

TICKS = 20000


def main():
    t0 = now()
    res = {}
    for seed in range(7):
        H.set_ground_speed(18.0)
        A = H.ObstacleTimeline(seed)
        H.set_ground_speed(9.0)
        B = H.ObstacleTimeline(seed)
        for d in B.dyns:
            d.vel = d.vel * 0.5
        A.at(TICKS * A.dt)
        B.at(TICKS * B.dt)
        first, mx = {}, 0.0
        for k in range(TICKS + 1):
            dev = max(float(np.max(np.abs(np.asarray(a["params"], float) - np.asarray(b["params"], float))))
                      for a, b in zip(A.states[k], B.states[k]))
            mx = max(mx, dev)
            for th in ("1e-6", "1e-3", "0.1"):
                if dev > float(th) and th not in first:
                    first[th] = k
        res[str(seed)] = {"first_tick_over": first, "max_dev_m": mx, "bit_identical": mx == 0.0}
        print(seed, res[str(seed)], flush=True)
    H.set_ground_speed(18.0)
    summ = {"seeds": len(res), "bit_identical": sum(r["bit_identical"] for r in res.values()),
            "onsets_1e-6": sorted(r["first_tick_over"]["1e-6"] for r in res.values() if "1e-6" in r["first_tick_over"])}
    d = outdir("verification", "scale_invariance_onset")
    dump(d, "result.json", {"summary": summ, "seeds": res, "ticks": TICKS})
    write_manifest(d, "scripts/verify_scaleinv_onset.py", {"ticks": TICKS}, {"motion": "0-6"}, t0)
    print(summ)


if __name__ == "__main__":
    main()
