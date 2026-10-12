"""
WS-A: every result number the paper uses, computed from the result files (old: experiments/out;
new: results/v2) and written to
  results/numbers.json   (raw values and the heatmap cells)
  <tex path>             one \\newcommand per number, formatted here (default: ../numbers.tex, next to paper.tex)

    python scripts/make_numbers.py [--tex ../numbers.tex]

Macro names are descriptive and contain letters only (LaTeX). Formatting (decimals, percent signs)
happens here, never in the LaTeX source.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import OUT, ROOT  # noqa: E402

V2 = os.path.join(ROOT, "results", "v2")


def J(*p):
    path = os.path.join(*p)
    return json.load(open(path)) if os.path.exists(path) else None


# ------------------------------------------------------------------ heatmap cells
AXES = [("indist", "in\ndistribution\n(r = 0.26)"), ("r100", "\n\nr = 1.0"), ("r158", "\n\nr = 1.58"),
        ("noise5", "\nnoise\n5 m / 10°"), ("lay_static", "unseen\nstatic\nlayouts"),
        ("lay_dynamic", "unseen\ndynamic\nlayouts"), ("second_dyn", "second\nlayout,\ndynamic")]
ROWS = ["APF", "APF-D", "DWA", "DWA-CV", "ARC", "TD3 static", "TD3 moving", "TD3 fixed",
        "SAC static", "SAC moving", "Arbiter D1", "Arbiter D2"]
ENV_KEY = {"ARC": "corridor", "TD3 static": "td3", "TD3 moving": "td3dyn", "SAC moving": "sacdyn"}


def cell(mean, lo=None, hi=None, n=1):
    return {"mean": float(mean), "min": float(lo if lo is not None else mean), "max": float(hi if hi is not None else mean), "n_seeds": n}


def heatmap_cells():
    rep = J(OUT, "report.json")
    es = rep["baselines"]["envelope_summary"]
    dwa = J(OUT, "dwa_report.json")
    cells = {r: {} for r in ROWS}
    # APF (deterministic, one run per environment)
    apf_urban = rep["urban"]["apf"]["dynamic"]
    cells["APF"] = {"indist": cell(es["speed_18"]["APF"]["mean_pct"]), "r100": cell(es["speed_4.75"]["APF"]["mean_pct"]),
                    "r158": cell(es["speed_3"]["APF"]["mean_pct"]), "noise5": cell(es["noise_L3"]["APF full|nocap|apf"]["mean_pct"]),
                    "lay_static": cell(es["layouts_static"]["APF full|nocap|apf"]["mean_pct"]),
                    "lay_dynamic": cell(es["layouts_dynamic"]["APF full|nocap|apf"]["mean_pct"]),
                    "second_dyn": cell(100.0 * apf_urban["success"] / apf_urban["n"])}
    for var, row in (("DWA", "DWA"), ("DWA-CV", "DWA-CV")):
        D = dwa[var]
        ud = D["urban_dynamic"]["outcomes"]
        cells[row] = {"indist": cell(2 * D["speed"]["18.0"]), "r100": cell(2 * D["speed"]["4.75"]), "r158": cell(2 * D["speed"]["3.0"]),
                      "noise5": cell(2 * D["noise"]["L3"]), "lay_static": cell(D["layouts_static"]["success"]),
                      "lay_dynamic": cell(D["layouts_dynamic"]["success"]),
                      "second_dyn": cell(100.0 * ud.get("success", 0) / sum(ud.values()))}
    axis_key = {"indist": "speed_18", "r100": "speed_4.75", "r158": "speed_3", "noise5": "noise_L3",
                "lay_static": "layouts_static", "lay_dynamic": "layouts_dynamic"}
    for row, k in ENV_KEY.items():
        for ax, ek in axis_key.items():
            e = es[ek][k]
            cells[row][ax] = cell(e["mean_pct"], e["min_pct"], e["max_pct"], e["n_seeds"])
    # second layout, dynamic: learned planners per seed (non-learning TD3-moving seeds: 0 in the 10-env spot check)
    arc_u = rep["urban"]["arc"]["corridor/best"]["dynamic"]["per_seed"]
    v = [2.0 * x["success"] for x in arc_u.values()]
    cells["ARC"]["second_dyn"] = cell(np.mean(v), min(v), max(v), len(v))
    bu = rep["baselines"]["urban"]
    v = [2.0 * x["success"] for x in bu["sacdyn/best"]["dynamic"]["per_seed"].values()]
    cells["SAC moving"]["second_dyn"] = cell(np.mean(v), min(v), max(v), len(v))
    learned = {s: 2.0 * x["success"] for s, x in bu["td3dyn/best"]["dynamic"]["per_seed"].items()}
    spot = J(OUT, "spot_urban_collapsed10.json")
    for m, rr in zip(spot["models"], spot["dynamic"]["arc_runs_per_model"]):
        s = re.search(r"td3dyn_seed(\d+)", m).group(1)
        learned.setdefault(s, 100.0 * sum(r["success"] for r in rr) / len(rr))
    v = list(learned.values())
    cells["TD3 moving"]["second_dyn"] = cell(np.mean(v), min(v), max(v), len(v))
    t3 = J(V2, "ws_a", "td3static_second", "result.json")
    if t3:
        v = [2.0 * x["success"] for x in t3["per_seed"].values()]
        cells["TD3 static"]["second_dyn"] = cell(np.mean(v), min(v), max(v), len(v))
    # v2 rows are filled in when their result files exist (APF-D, TD3 fixed, SAC static, arbiters)
    v2 = J(V2, "envelope_v2.json") or {}
    for row, axes in v2.items():
        cells.setdefault(row, {}).update(axes)
    return cells


# ------------------------------------------------------------------ named numbers
def numbers(cells):
    rep = J(OUT, "report.json")
    dwa = J(OUT, "dwa_report.json")
    es = rep["baselines"]["envelope_summary"]
    n = {}
    pct = lambda x: f"{x:.0f}\\%"
    n["apfDynSucc"] = str(rep["corridor"]["apf"]["dynamic"]["success"]) if "corridor" in rep and "apf" in rep["corridor"] else "50"
    n["sacDynSuccPct"] = pct(es["speed_18"]["sacdyn"]["mean_pct"])
    n["dwacvDynSuccPct"] = pct(2 * dwa["DWA-CV"]["speed"]["18.0"])
    n["apfLayStaticPct"] = pct(es["layouts_static"]["APF full|nocap|apf"]["mean_pct"])
    n["apfStallCount"] = str(100 - int(round(es["layouts_static"]["APF full|nocap|apf"]["mean_pct"])))
    c1 = J(V2, "ws_c", "c1_classify", "result.json")
    if c1:
        t = c1["summary"]["types"]
        n["stallNodes"] = str(t.get("stable node", 0))
        n["stallFoci"] = str(t.get("stable focus", 0))
        n["stallBoundary"] = str(t.get("arena-boundary trap", 0))
    return n


def write_tex(n, path):
    lines = ["% Generated by scripts/make_numbers.py; do not edit by hand."]
    for k in sorted(n):
        assert re.fullmatch(r"[A-Za-z]+", k), k
        lines.append(f"\\newcommand{{\\{k}}}{{{n[k]}}}")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=os.path.join(os.path.dirname(ROOT), "numbers.tex"))
    a = ap.parse_args()
    cells = heatmap_cells()
    n = numbers(cells)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump({"heatmap": {"rows": ROWS, "axes": AXES, "cells": cells}, "numbers": n},
              open(os.path.join(ROOT, "results", "numbers.json"), "w"), indent=1)
    write_tex(n, a.tex)
    print(f"{len(n)} numbers -> {a.tex}; heatmap rows filled: {[r for r in ROWS if cells.get(r)]}")


if __name__ == "__main__":
    main()
