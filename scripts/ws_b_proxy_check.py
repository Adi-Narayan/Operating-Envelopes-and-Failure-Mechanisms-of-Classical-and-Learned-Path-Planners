"""WS-B, B3 proxy validation: does collapse status at the episode-50 checkpoint predict the final
collapse status (out/collapse.json, retained checkpoint) of the existing TD3 runs?

Collapse is evaluated exactly as experiments/collapse_diagnostic.py does: one deterministic run
on the static custom environment from q0 heading towards the goal; collapsed if >= 99% of the
executed heading changes are at the 9 deg bound.

    python scripts/ws_b_proxy_check.py   -> results/v2/ws_b/proxy_check/
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import FULL_CKPT, OUT, dump, now, outdir, write_manifest  # noqa: E402
import json                                                             # noqa: E402
import harness as H                                                     # noqa: E402

CAP_DEG = 9.0


def frac_at_cap(model_path):
    r = H.ARCPlanner(model_path).run(H.StaticTimeline(H.LAYOUTS["corridor"]), static=True)
    p = np.asarray(r["path"])
    hd = np.unwrap(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0])))
    turn = np.rad2deg(np.diff(hd))
    return float(np.mean(np.abs(np.abs(turn) - CAP_DEG) < 0.05)) if len(turn) else float("nan"), r["outcome"]


def main():
    t0 = now()
    final = json.load(open(os.path.join(OUT, "collapse.json")))
    res = {}
    for group in ("td3dyn", "td3"):
        rows = []
        for s in range(10):
            run = f"{group}_seed{s}"
            ck = os.path.join(FULL_CKPT, run, "models", "arc_checkpoint_ep50.pth")
            f50, outc = frac_at_cap(ck)
            rows.append({"run": run, "frac_at_cap_ep50": f50, "collapsed_ep50": f50 >= 0.99, "outcome_ep50": outc,
                         "collapsed_final": final[run]["collapsed"], "frac_at_cap_final": final[run]["frac_steps_at_cap"]})
            print(rows[-1], flush=True)
        agree = sum(r["collapsed_ep50"] == r["collapsed_final"] for r in rows)
        res[group] = {"rows": rows, "agree": agree, "n": len(rows), "criterion_9_of_10": agree >= 9}
        print(group, "agree", agree, "/ 10", flush=True)
    d = outdir("ws_b", "proxy_check")
    dump(d, "result.json", res)
    write_manifest(d, "scripts/ws_b_proxy_check.py", {"checkpoint": "arc_checkpoint_ep50.pth (full, checkpoints_full)",
                                                      "collapse_rule": ">= 0.99 of heading changes at 9 deg, static custom env"},
                   {"runs": "td3dyn_seed0-9, td3_seed0-9"}, t0)


if __name__ == "__main__":
    main()
