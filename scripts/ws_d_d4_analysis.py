"""
WS-D, D4 analysis: the arbiter against its components and the oracle reference, per SAC seed and axis
(prereg.md, WS-D D4), and the handover-bug sensitivity check (prereg.md, WS-D deviation of 2026-10-13).

Comparisons (paired exact McNemar per seed and axis, Holm within each family of 10 seeds x 7 axes):
    d1_vs_sac, d1_vs_capf          D1 as registered against SAC alone and ray-APF-D alone
    d1fix_vs_sac, d1fix_vs_capf    D1 with the handover reset against the same components
    d1fix_vs_d1                    effect of the fix
    d2_vs_sac, d2_vs_dwacv         D2 against SAC alone and DWA-CV alone
Oracle reference: an environment counts if either component succeeds.
Also: controller in charge at each failure, switches, chattering, time in C, and where switches occur
relative to the goal (Adi's item 9).

    python scripts/ws_d_d4_analysis.py  -> results/v2/ws_d/d4_analysis/result.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from v2common import ROOT, dump, now, outdir, write_manifest  # noqa: E402
from stats_v2 import holm                                      # noqa: E402
from stats_utils import mcnemar_exact                          # noqa: E402

D = os.path.join(ROOT, "results", "v2", "ws_d", "d4_eval")
AXES = ["indist", "r100", "r158", "noise5", "lay_static", "lay_dynamic", "second_dyn"]
SEEDS = range(10)
FAMILIES = {"d1_vs_sac": ("d1", "sac"), "d1_vs_capf": ("d1", "capf"), "d1fix_vs_sac": ("d1fix", "sac"),
            "d1fix_vs_capf": ("d1fix", "capf"), "d1fix_vs_d1": ("d1fix", "d1"),
            "d2_vs_sac": ("d2", "sac"), "d2_vs_dwacv": ("d2", "dwacv")}


def load(name):
    return json.load(open(os.path.join(D, name + ".json")))["axes"]


def ok(runs):
    return np.array([r["outcome"] == "success" for r in runs])


def controller_at_end(r):
    log = r.get("switch_log") or []
    return "C" if log and log[-1][0] == "L->C" else "L"


def main():
    t0 = now()
    comp = {"capf": load("capf"), "dwacv": load("dwacv")}
    arb = {m: {s: load(f"{m}_sac{s}") for s in SEEDS} for m in ("d1", "d1fix", "d2", "sac")}

    def runs(mode, s, ax):
        return comp[mode][ax] if mode in comp else arb[mode][s][ax]

    succ = {m: {ax: [int(ok(runs(m, s, ax)).sum()) for s in SEEDS] for ax in AXES} for m in ("d1", "d1fix", "d2", "sac")}
    succ.update({m: {ax: int(ok(comp[m][ax]).sum()) for ax in AXES} for m in comp})
    n = {ax: len(comp["capf"][ax]) for ax in AXES}
    oracle = {k: {ax: [int((ok(arb["sac"][s][ax]) | ok(comp[c][ax])).sum()) for s in SEEDS] for ax in AXES}
              for k, c in (("d1", "capf"), ("d2", "dwacv"))}

    fam = {}
    for f, (a, b) in FAMILIES.items():
        cells = [(s, ax) + mcnemar_exact(ok(runs(a, s, ax)), ok(runs(b, s, ax))) for s in SEEDS for ax in AXES]
        adj = holm([c[4] for c in cells])
        rows = [{"seed": s, "axis": ax, "a_only": ao, "b_only": bo, "p": p, "p_holm": pa}
                for (s, ax, ao, bo, p), pa in zip(cells, adj)]
        sig = [r for r in rows if r["p_holm"] < 0.05]
        fam[f] = {"a": a, "b": b, "cells": rows,
                  "sig_a_better": sum(r["a_only"] > r["b_only"] for r in sig),
                  "sig_b_better": sum(r["b_only"] > r["a_only"] for r in sig),
                  "cells_a_ahead": sum(r["a_only"] > r["b_only"] for r in rows),
                  "cells_b_ahead": sum(r["b_only"] > r["a_only"] for r in rows),
                  "pooled": {ax: {"a_only": sum(r["a_only"] for r in rows if r["axis"] == ax),
                                  "b_only": sum(r["b_only"] for r in rows if r["axis"] == ax)} for ax in AXES}}

    beh = {}
    for m in ("d1", "d1fix", "d2"):
        allr = [r for s in SEEDS for ax in AXES for r in arb[m][s][ax]]
        fails = [r for r in allr if r["outcome"] != "success"]
        by = {}
        for r in fails:
            key = f"{r['outcome']}|{controller_at_end(r)}"
            by[key] = by.get(key, 0) + 1
        sw = [e for r in allr for e in (r.get("switch_log") or [])]
        lc = np.array([e[2] for e in sw if e[0] == "L->C"])
        cl = np.array([e[2] for e in sw if e[0] == "C->L"])
        first = np.array([r["switch_log"][0][2] for r in allr if r.get("switch_log")])
        reasons = {}
        for e in sw:
            if e[0] == "C->L":
                reasons[e[5]] = reasons.get(e[5], 0) + 1
        steps, nsw = np.array([r["steps"] for r in allr]), np.array([r["switches"] for r in allr])
        beh[m] = {"runs": len(allr), "failures_by_outcome_and_controller": by,
                  "switches_per_run": float(nsw.mean()), "chatter_per_run": float(np.mean([r["chatter"] for r in allr])),
                  "frac_time_C": float(np.mean([r["frac_time_C"] for r in allr])),
                  "decisions_per_switch": float(steps.sum() / max(nsw.sum(), 1)),
                  "runs_without_switch": int(np.sum(nsw == 0)),
                  "goal_dist_L_to_C": {"p25": float(np.percentile(lc, 25)), "median": float(np.median(lc)), "p75": float(np.percentile(lc, 75)),
                                       "frac_within_50m": float(np.mean(lc < 50)), "frac_within_100m": float(np.mean(lc < 100))},
                  "goal_dist_C_to_L": {"median": float(np.median(cl)), "frac_within_50m": float(np.mean(cl < 50))},
                  "first_switch_goal_dist_median": float(np.median(first)),
                  "runs_switching_within_50m_of_goal": int(sum(any(e[2] < 50 for e in (r.get("switch_log") or [])) for r in allr)),
                  "c_to_l_reasons": reasons}

    res = {"n_per_axis": n, "successes": succ, "oracle": oracle, "families": fam, "behaviour": beh,
           "note": "successes per SAC seed (list) or single (components without L); McNemar a_only = runs only a solved"}
    d = outdir("ws_d", "d4_analysis")
    dump(d, "result.json", res)
    write_manifest(d, "scripts/ws_d_d4_analysis.py", {"families": FAMILIES, "axes": AXES}, {"sac": "0-9"}, t0)
    for ax in AXES:
        print(f"{ax:12s} n={n[ax]:3d} capf={succ['capf'][ax]:3d} dwacv={succ['dwacv'][ax]:3d} | "
              f"sac {succ['sac'][ax]} | d1 {succ['d1'][ax]} | d1fix {succ['d1fix'][ax]} | d2 {succ['d2'][ax]} | "
              f"orac1 {oracle['d1'][ax]}")
    for f, v in fam.items():
        print(f, "Holm-sig a better:", v["sig_a_better"], "b better:", v["sig_b_better"],
              "| cells a ahead", v["cells_a_ahead"], "b ahead", v["cells_b_ahead"])
    for m, v in beh.items():
        print(m, json.dumps({k: v[k] for k in v if k != "c_to_l_reasons"}))


if __name__ == "__main__":
    main()
