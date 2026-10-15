"""WS-C, C4 check: experiments/apf_damped.py with damping off reproduces harness.run_apf, and its
always-on cap reproduces apf_variants.run(turn_cap=True) (full sensing, 10 Hz), path for path.

    python scripts/ws_c_apfd_identity.py  -> results/v2/ws_c/c4_identity/
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import dump, now, outdir, write_manifest  # noqa: E402
import harness as H                                      # noqa: E402
import apf_variants as V                                 # noqa: E402
import apf_damped as D                                   # noqa: E402
from random_layouts import make_layout                   # noqa: E402


def same(a, b):
    pa, pb = np.asarray(a["path"]), np.asarray(b["path"])
    return a["outcome"] == b["outcome"] and pa.shape == pb.shape and bool(np.array_equal(pa, pb))


def main():
    t0 = now()
    rows = []
    for s in range(10):
        tl = H.ObstacleTimeline(s)
        rows.append({"case": f"dynamic seed {s}", "apf_vs_run_apf": same(D.run(tl, 1000 + s), H.run_apf(tl, 1000 + s)),
                     "cap_vs_apf_variants": same(D.run(tl, 1000 + s, always_cap=True),
                                                 V.run(tl, 1000 + s, sensing="full", turn_cap=True, rate="apf"))})
    for k in (0, 1, 2, 3, 8, 15):
        st = H.StaticTimeline(make_layout(k))
        rows.append({"case": f"random layout {k} (static)", "apf_vs_run_apf": same(D.run(st, 0), H.run_apf(st, 0)),
                     "cap_vs_apf_variants": same(D.run(st, 0, always_cap=True),
                                                 V.run(st, 0, sensing="full", turn_cap=True, rate="apf"))})
    for r in rows:
        print(r, flush=True)
    ok = all(r["apf_vs_run_apf"] and r["cap_vs_apf_variants"] for r in rows)
    print("ALL_IDENTICAL", ok)
    d = outdir("ws_c", "c4_identity")
    dump(d, "result.json", {"rows": rows, "all_identical": ok})
    write_manifest(d, "scripts/ws_c_apfd_identity.py", {"variants": ["apf (no damping)", "always-on cap"]},
                   {"motion": "0-9 (planner 1000+s)", "layouts": [0, 1, 2, 3, 8, 15]}, t0)


if __name__ == "__main__":
    main()
