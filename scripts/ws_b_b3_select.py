"""
WS-B, B3: collapse status of every B3 run's episode-50 checkpoint (prereg.md outcome measure) and the
pre-registered B4 selection rule.

    python scripts/ws_b_b3_select.py  -> results/v2/ws_b/b3_selection/
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump, now, outdir, write_manifest   # noqa: E402
from ws_b_proxy_check import frac_at_cap                        # noqa: E402

CONDS = ["a", "b", "c", "d"]
SEEDS = [10, 11, 12]
TIE_ORDER = ["d", "b", "c"]          # smallest change from published TD3 first (prereg.md)
FLAGS = {"a": [], "b": ["--gamma", "0.9"], "c": ["--pretanh-penalty", "0.002"], "d": ["--exploration-noise", "0.3"]}


def main():
    t0 = now()
    rows, counts = [], {}
    for c in CONDS:
        n = 0
        for s in SEEDS:
            ck = os.path.join(ROOT, "results", "v2", "ws_b", "b3", f"b3{c}_seed{s}", "models", "arc_checkpoint_ep50.pth")
            f, outc = frac_at_cap(ck)
            collapsed = f >= 0.99
            n += collapsed
            log = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.dirname(ck)), "train_log.jsonl"))]
            rows.append({"condition": c, "seed": s, "frac_at_cap_ep50": f, "collapsed": collapsed, "static_outcome": outc,
                         "reward_last10": sum(r["reward"] for r in log[-10:]) / 10,
                         "u2_last10": sum(r.get("u2", 0) for r in log[-10:]) / 10,
                         "u_sat_last10": sum(r.get("u_sat", 0) for r in log[-10:]) / 10})
            print(rows[-1], flush=True)
        counts[c] = n
    cand = min(TIE_ORDER, key=lambda c: (counts[c], TIE_ORDER.index(c)))
    run_b4 = counts[cand] < counts["a"]
    out = {"rows": rows, "collapses": counts, "candidate": cand, "run_b4": run_b4,
           "b4_flags": FLAGS[cand] if run_b4 else None}
    print("COLLAPSES", counts, "CANDIDATE", cand, "RUN_B4", run_b4)
    d = outdir("ws_b", "b3_selection")
    dump(d, "result.json", out)
    write_manifest(d, "scripts/ws_b_b3_select.py", {"rule": "prereg.md WS-B B3", "tie_order": TIE_ORDER},
                   {"learner_seeds": SEEDS}, t0)


if __name__ == "__main__":
    main()
