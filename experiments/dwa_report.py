"""
Numbers for the paper's DWA rows, from out/dwa_eval.json against the stored APF and SAC
results (same environments and seeds).

    python dwa_report.py            -> out/dwa_report.json (and a printed summary)
"""
import glob
import json
import os

import numpy as np

from stats_utils import mcnemar_exact, wilcoxon_paired, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
D_STRAIGHT = 430 * 2 ** 0.5


def J(name):
    return json.load(open(os.path.join(OUT, name)))


def runs_of(block):
    return [block[str(s)] for s in range(len(block))]


def outcomes(runs):
    o = [r["outcome"] for r in runs]
    return {k: o.count(k) for k in sorted(set(o))}


def paired(a_runs, b_runs):
    """a vs b on the same environments: McNemar on success, efficiency on shared successes."""
    a_ok = [r["outcome"] == "success" for r in a_runs]
    b_ok = [r["outcome"] == "success" for r in b_runs]
    a_only, b_only, p = mcnemar_exact(a_ok, b_ok)
    sh = [i for i in range(len(a_runs)) if a_ok[i] and b_ok[i]]
    ea = [a_runs[i]["efficiency"] for i in sh]
    eb = [b_runs[i]["efficiency"] for i in sh]
    la = [a_runs[i]["path_length"] for i in sh]
    lb = [b_runs[i]["path_length"] for i in sh]
    n, _, pw = wilcoxon_paired(ea, eb)
    return {"a_only": a_only, "b_only": b_only, "mcnemar_p": p, "shared": len(sh),
            "eff_a": [float(np.mean(ea)), float(np.std(ea, ddof=1))] if len(ea) > 1 else None,
            "eff_b": [float(np.mean(eb)), float(np.std(eb, ddof=1))] if len(eb) > 1 else None,
            "len_a": float(np.mean(la)) if la else None, "len_b": float(np.mean(lb)) if lb else None,
            "a_more_efficient": int(sum(x > y for x, y in zip(ea, eb))), "wilcoxon_p": pw}


def sac_best_runs():
    """{seed: dynamic runs of the sacdyn arc_best policy} from the final_corridor_sacdyn_* files."""
    out = {}
    for f in sorted(glob.glob(os.path.join(OUT, "final_corridor_sacdyn_*.json"))):
        d = json.load(open(f))
        for m, runs in zip(d["models"], d["dynamic"]["arc_runs_per_model"]):
            if m.endswith("arc_best.pth"):
                out[m.split("/")[2]] = runs
    return out


def main():
    E = J("dwa_eval.json")
    apf_c = J("final_corridor_a.json")["dynamic"]["apf_runs"]
    apf_u = J("final_urban_a.json")["dynamic"]["apf_runs"]
    sac = sac_best_runs()
    noise_apf = J("noise_apf_arc.json")["results"]
    lay_apf = J("layouts_apf_arc.json")["results"]
    rep = {"weights": {v: E["weights"][v]["weights"] for v in ("DWA", "DWA-CV")}}
    for var in ("DWA", "DWA-CV"):
        V = E[var]
        R = rep[var] = {}
        st = V["static"]["corridor"]
        R["static_corridor"] = {k: st[k] for k in ("outcome", "path_length", "efficiency", "steps", "mean_speed", "time_s")}
        R["static_corridor"]["steps_per_100m"] = 100 * st["steps"] / st["path_length"]
        R["static_urban"] = V["static"]["urban"]["outcome"]
        dc = runs_of(V["dynamic"]["corridor"])
        du = runs_of(V["dynamic"]["urban"])
        succ = [r for r in dc if r["outcome"] == "success"]
        R["dynamic"] = {"outcomes": outcomes(dc), "success": len(succ), "wilson": wilson(len(succ), 50),
                        "mean_speed_success": float(np.mean([r["mean_speed"] for r in succ])) if succ else None,
                        "time_s_success": [float(np.min([r["time_s"] for r in succ])), float(np.mean([r["time_s"] for r in succ])),
                                           float(np.max([r["time_s"] for r in succ]))] if succ else None,
                        "eff_success": [float(np.mean([r["efficiency"] for r in succ])), float(np.std([r["efficiency"] for r in succ], ddof=1))] if len(succ) > 1 else None,
                        "min_clearance_success": float(np.min([r["min_clearance"] for r in succ])) if succ else None,
                        "no_admissible_runs": int(sum(r["no_admissible"] > 0 for r in dc)),
                        "failed_envs": [s for s, r in enumerate(dc) if r["outcome"] != "success"]}
        R["vs_apf"] = paired(dc, apf_c)
        R["apf_time_s_on_dwa_successes"] = float(np.mean([apf_c[s]["steps"] * 0.1 for s, r in enumerate(dc) if r["outcome"] == "success"])) if succ else None
        R["vs_sac"] = {seed: paired(dc, runs) for seed, runs in sorted(sac.items())}
        pooled_a = [x for runs in sac.values() for x in dc]
        pooled_b = [x for runs in sac.values() for x in runs]
        R["vs_sac_pooled"] = paired(pooled_a, pooled_b)
        R["urban_dynamic"] = {"outcomes": outcomes(du), "vs_apf": paired(du, apf_u)}
        sp = {"18.0": len(succ)}
        for v, blk in V["speed"].items():
            sp[v] = sum(r["outcome"] == "success" for r in runs_of(blk))
        R["speed"] = {v: sp[v] for v in sorted(sp, key=float)}
        R["noise"] = {"L0": len(succ)}
        for lev, blk in V["noise"].items():
            R["noise"][lev] = sum(r["outcome"] == "success" for r in runs_of(blk))
        R["noise_outcomes"] = {lev: outcomes(runs_of(blk)) for lev, blk in V["noise"].items()}
        for part in ("static", "dynamic"):
            rr = runs_of(V["layouts"][part])
            apf_ok = [lay_apf[str(k)][part]["APF full|nocap|apf"]["outcome"] == "success" for k in range(100)]
            ok = [r["outcome"] == "success" for r in rr]
            R[f"layouts_{part}"] = {"success": int(sum(ok)), "outcomes": outcomes(rr),
                                    "solves_where_apf_fails": int(sum(o and not a for o, a in zip(ok, apf_ok))),
                                    "apf_fails": int(sum(not a for a in apf_ok)),
                                    "fails_where_apf_succeeds": int(sum(a and not o for o, a in zip(ok, apf_ok)))}
        R["ms_per_step_eval_mean"] = float(np.mean([r["ms_per_step"] for r in dc]))
    with open(os.path.join(OUT, "dwa_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
