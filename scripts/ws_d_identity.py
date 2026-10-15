"""WS-D check: the arbiter with switching disabled reproduces each component alone, path for path.
  L alone  (d_on = -inf, rho_on = inf)  vs harness.ARCPlanner.run      (SAC moving seed 0)
  C alone  apf_rays                     vs apf_variants.run(rays, nocap, arc)
  C alone  dwa_cv                       vs dwa.run(predict=True)
on motion seeds 0-9 of the custom environment, 18 m/s.

    python scripts/ws_d_identity.py  -> results/v2/ws_d/identity/
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import OUT, dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                           # noqa: E402
import apf_variants as V                                      # noqa: E402
import dwa                                                    # noqa: E402
import arbiter as A                                           # noqa: E402

SAC0 = os.path.join(OUT, "arc_models", "sacdyn_seed0", "models", "arc_best.pth")


def same(a, b):
    pa, pb = np.asarray(a["path"], float), np.asarray(b["path"], float)
    return a["outcome"] == b["outcome"] and pa.shape == pb.shape and bool(np.array_equal(pa, pb))


def main():
    t0 = now()
    H.set_ground_speed(18.0)
    L = A.Learned(SAC0)
    w = json.load(open(os.path.join(OUT, "dwa_tune.json")))["selected"]["DWA-CV"]["weights"]
    rows = []
    for s in range(10):
        r_l = A.run(H.ObstacleTimeline(s), L, A.APFRays(), "apf_rays", -np.inf, np.inf)
        ref_l = L.P.run(H.ObstacleTimeline(s))
        r_c = A.run(H.ObstacleTimeline(s), None, A.APFRays(), "apf_rays", 0, 0, c_seed=1000 + s, l_enabled=False)
        ref_c = V.run(H.ObstacleTimeline(s), 1000 + s, sensing="rays", turn_cap=False, rate="arc")
        r_d = A.run(H.ObstacleTimeline(s), None, A.DWACV(w), "dwa_cv", 0, 0, l_enabled=False)
        ref_d = dwa.run(H.ObstacleTimeline(s), predict=True, weights=w, record=True)
        nz = dict(range_noise=2.5, heading_noise=np.radians(5.0), noise_seed=502_000 + s)
        n_l = A.run(H.ObstacleTimeline(s), L, A.APFRays(), "apf_rays", -np.inf, np.inf, **nz)
        nref_l = L.P.run(H.ObstacleTimeline(s), **nz)
        n_c = A.run(H.ObstacleTimeline(s), None, A.APFRays(), "apf_rays", 0, 0, c_seed=1000 + s, l_enabled=False, **nz)
        nref_c = V.run(H.ObstacleTimeline(s), 1000 + s, sensing="rays", turn_cap=False, rate="arc", **nz)
        n_d = A.run(H.ObstacleTimeline(s), None, A.DWACV(w), "dwa_cv", 0, 0, l_enabled=False, **nz)
        nref_d = dwa.run(H.ObstacleTimeline(s), predict=True, weights=w, record=True, **nz)
        rows.append({"seed": s, "L_alone": same(r_l, ref_l), "apf_rays_alone": same(r_c, ref_c), "dwa_cv_alone": same(r_d, ref_d),
                     "L_alone_noise": same(n_l, nref_l), "apf_rays_alone_noise": same(n_c, nref_c), "dwa_cv_alone_noise": same(n_d, nref_d),
                     "outcomes": [r_l["outcome"], r_c["outcome"], r_d["outcome"]]})
        print(rows[-1], flush=True)
    ok = all(all(v for k, v in r.items() if k not in ("seed", "outcomes")) for r in rows)
    print("ALL_IDENTICAL", ok)
    d = outdir("ws_d", "identity")
    dump(d, "result.json", {"rows": rows, "all_identical": ok})
    write_manifest(d, "scripts/ws_d_identity.py", {"L": "sacdyn_seed0/arc_best", "dwa_cv_weights": w}, {"motion": "0-9"}, t0)


if __name__ == "__main__":
    main()
