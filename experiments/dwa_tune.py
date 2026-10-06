"""
Select the DWA objective weights (alpha: heading, beta: clearance, gamma: velocity) for each
variant (DWA, DWA-CV) on a tuning set disjoint from every evaluation environment:
20 held-out obstacle-motion seeds (1000-1019) on the custom layout at 18 m/s, plus the static
custom environment. Selection: most successes over the 21 tuning runs, ties broken by mean
efficiency over the successful runs.

    python dwa_tune.py --workers 4          -> out/dwa_tune.json
"""
import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ALPHAS = [0.5, 1.0, 2.0]
BETAS = [0.1, 0.2, 0.5, 1.0]
GAMMAS = [0.2, 0.5, 1.0]
TUNE_SEEDS = list(range(1000, 1020))


def task(args):
    predict, w, env = args
    import harness as H
    import dwa
    tl = H.StaticTimeline() if env == "static" else H.ObstacleTimeline(env)
    r = dwa.run(tl, predict=predict, weights=w, max_periods=3000)
    return predict, tuple(sorted(w.items())), env, r["outcome"], r["efficiency"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    grid = [dict(alpha=x, beta=y, gamma=z) for x, y, z in itertools.product(ALPHAS, BETAS, GAMMAS)]
    tasks = [(p, w, e) for p in (False, True) for w in grid for e in ["static"] + TUNE_SEEDS]
    res = {}
    with Pool(a.workers) as pool:
        for n, (p, wk, e, outc, eff) in enumerate(pool.imap_unordered(task, tasks, chunksize=4), 1):
            res.setdefault(("DWA-CV" if p else "DWA", wk), []).append({"env": e, "outcome": outc, "efficiency": eff})
            if n % 200 == 0:
                print(f"{n}/{len(tasks)}", flush=True)
    table, best = [], {}
    for (var, wk), runs in res.items():
        succ = [r for r in runs if r["outcome"] == "success"]
        row = {"variant": var, "weights": dict(wk), "success": len(succ),
               "static_success": any(r["env"] == "static" and r["outcome"] == "success" for r in runs),
               "mean_eff": sum(r["efficiency"] for r in succ) / len(succ) if succ else 0.0}
        table.append(row)
        key = (row["success"], row["mean_eff"])
        if var not in best or key > (best[var]["success"], best[var]["mean_eff"]):
            best[var] = row
    table.sort(key=lambda r: (r["variant"], -r["success"], -r["mean_eff"]))
    with open(os.path.join(HERE, "out", "dwa_tune.json"), "w") as f:
        json.dump({"tuning_seeds": TUNE_SEEDS, "grid": {"alpha": ALPHAS, "beta": BETAS, "gamma": GAMMAS},
                   "selected": best, "table": table}, f, indent=1)
    for var, row in best.items():
        print("SELECTED", var, row)
    print("TUNE_DONE")


if __name__ == "__main__":
    main()
