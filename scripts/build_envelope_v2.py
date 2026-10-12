"""
WS-A: heatmap rows for the planners added in v2, computed from their result files. A row is written
only when all of its result files exist; missing rows are listed and skipped.

    python scripts/build_envelope_v2.py   -> results/v2/envelope_v2.json, results/v2/ws_c/summary.json

Rows: APF-D (results/v2/ws_c/apfd_eval), TD3 fixed and SAC static (results/v2/ws_b/eval_b4, eval_b5),
Arbiter D1 and Arbiter D2 (results/v2/ws_d/d4_eval). Axes as in scripts/make_numbers.py.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump  # noqa: E402

V2 = os.path.join(ROOT, "results", "v2")


def J(*p):
    path = os.path.join(*p)
    return json.load(open(path)) if os.path.exists(path) else None


def pct(runs):
    return 100.0 * sum(r["outcome"] == "success" for r in runs) / len(runs)


def cell(vals):
    vals = [float(v) for v in vals]
    return {"mean": float(np.mean(vals)), "min": min(vals), "max": max(vals), "n_seeds": len(vals)}


def apfd_row():
    d = os.path.join(V2, "ws_c", "apfd_eval")
    parts = {p: J(d, f"{p}.json") for p in ("dynamic", "speed", "noise", "layouts", "second", "static")}
    if any(v is None for v in parts.values()):
        return None, None
    row = {"indist": cell([pct(parts["dynamic"]["runs"])]),
           "r100": cell([pct(parts["speed"]["speeds"]["4.75"])]),
           "r158": cell([pct(parts["speed"]["speeds"]["3.0"])]),
           "noise5": cell([pct(parts["noise"]["levels"]["L3"])]),
           "lay_static": cell([pct(parts["layouts"]["static"])]),
           "lay_dynamic": cell([pct(parts["layouts"]["dynamic"])]),
           "second_dyn": cell([pct(parts["second"]["runs"])])}
    st = parts["static"]
    summ = {"variant": parts["dynamic"]["variant"], "kwargs": parts["dynamic"]["kwargs"],
            "static_custom": st["custom"]["outcome"], "static_second": st["second"]["outcome"],
            "static_five_layouts": {k: v["outcome"] for k, v in st["five_layouts"].items()},
            "dynamic_success": sum(r["outcome"] == "success" for r in parts["dynamic"]["runs"]),
            "dynamic_efficiency_mean": float(np.mean([r["efficiency"] for r in parts["dynamic"]["runs"] if r["outcome"] == "success"])),
            "speed_success": {v: sum(r["outcome"] == "success" for r in rr) for v, rr in parts["speed"]["speeds"].items()},
            "noise_success": {k: sum(r["outcome"] == "success" for r in rr) for k, rr in parts["noise"]["levels"].items()},
            "layouts_static": {o: sum(r["outcome"] == o for r in parts["layouts"]["static"]) for o in ("success", "timeout", "collision")},
            "layouts_dynamic": {o: sum(r["outcome"] == o for r in parts["layouts"]["dynamic"]) for o in ("success", "timeout", "collision")},
            "second_dynamic_success": sum(r["outcome"] == "success" for r in parts["second"]["runs"])}
    return row, summ


def per_seed_row(dirname, prefix):
    """One result file per seed, <prefix><k>.json, each with per-axis run lists under 'axes'."""
    d = os.path.join(V2, *dirname)
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.startswith(prefix) and f.endswith(".json"))
    if not files:
        return None
    per = [J(d, f)["axes"] for f in files]
    axes = per[0].keys()
    return {ax: cell([pct(p[ax]) for p in per]) for ax in axes}


def main():
    env, missing = {}, []
    row, summ = apfd_row()
    if row:
        env["APF-D"] = row
    else:
        missing.append("APF-D")
    for name, dirname, prefix in (("TD3 fixed", ("ws_b", "eval_b4"), "td3fix_seed"),
                                  ("SAC static", ("ws_b", "eval_b5"), "sacstatic_seed"),
                                  ("Arbiter D1", ("ws_d", "d4_eval"), "d1_sac"),
                                  ("Arbiter D2", ("ws_d", "d4_eval"), "d2_sac")):
        r = per_seed_row(dirname, prefix)
        if r:
            env[name] = r
        else:
            missing.append(name)
    dump(V2, "envelope_v2.json", env)
    c3 = J(V2, "ws_c", "c3_theta_sweep", "result.json")
    c1 = J(V2, "ws_c", "c1_classify", "result.json")
    dump(os.path.join(V2, "ws_c"), "summary.json",
         {"c1_types": c1["summary"]["types"] if c1 else None,
          "c3_by_theta": c3["summary"] if c3 else None,
          "apfd": summ})
    print("rows written:", list(env), "missing:", missing)
    if summ:
        print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
