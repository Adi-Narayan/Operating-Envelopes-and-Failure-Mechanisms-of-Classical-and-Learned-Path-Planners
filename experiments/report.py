"""
Merge all evaluation outputs into the paper's tables (markdown + JSON).

    python report.py            ->  out/report.md, out/report.json

Model variants per trained seed:
  best   : arc_best.pth, the pipeline's own selection by training reward (primary, pre-declared)
  final  : arc_checkpoint_ep300.pth
  oracle : checkpoint with the most successes on its own static layout (out/oracle_checkpoints.json);
           selected on the evaluation layout, so an optimistic upper bound
"""
import glob
import json
import os
import re

import numpy as np

from stats_utils import bootstrap_mean_ci, mcnemar_exact, wilcoxon_paired, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
LAYOUT_NAME = {"corridor": "Corridor Gauntlet", "urban": "ARC original layout (env_apf_urban)"}


def jload(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ident(path):
    p = path.replace("\\", "/")
    if "models_original" in p:
        train, seed = "urban", "orig"
    else:
        m = re.search(r"(env_apf_urban_)?seed(\d+)", p)
        train, seed = ("urban" if m.group(1) else "corridor"), m.group(2)
    if p.endswith("arc_best.pth"):
        var = "best"
    elif p.endswith("ep300.pth"):
        var = "final"
    else:
        var = "ckpt"
    return train, seed, var


def fmt(ms, d=1):
    m, s = ms
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "–"
    return f"{m:.{d}f} ± {s:.{d}f}"


def merge_evals(layout):
    """Combine final_<layout>_a/_b json files: APF (identical in each) + ARC runs keyed by model path."""
    files = sorted(glob.glob(os.path.join(OUT, f"final_{layout}_*.json")))
    if not files:
        return None
    base = json.load(open(files[0]))
    models, static_runs, dyn_runs = [], {}, {}
    for f in files:
        r = json.load(open(f))
        for i, m in enumerate(r["models"]):
            m = m.replace("\\", "/")
            if m in static_runs:
                continue
            models.append(m)
            static_runs[m] = r["static"]["arc_runs"][i]
            dyn_runs[m] = r["dynamic"]["arc_runs_per_model"][i]
        # sanity: APF identical across files
        a0 = [x["outcome"] for x in base["dynamic"]["apf_runs"]]
        assert a0 == [x["outcome"] for x in r["dynamic"]["apf_runs"]], "APF runs differ between files"
    return {"base": base, "models": models, "static": static_runs, "dynamic": dyn_runs}


def oracle_paths():
    o = jload("oracle_checkpoints.json") or {}
    return {v.replace("\\", "/").split("experiments/")[-1] for v in o.values()}


def variant_of(m, oracles):
    tr, seed, var = ident(m)
    key = m.split("experiments/")[-1]
    if key in oracles or m in oracles:
        var = "oracle" if var == "ckpt" else var
        # an oracle that coincides with best/final keeps both roles
    return tr, seed, var


def select(E, train, variant, oracles):
    out = []
    for m in E["models"]:
        tr, seed, var = ident(m)
        is_oracle = (m.split("experiments/")[-1] in oracles) or (m in oracles)
        if tr != train:
            continue
        if variant == "oracle" and is_oracle:
            out.append((seed, m))
        elif variant != "oracle" and var == variant:
            out.append((seed, m))
    # an oracle may coincide with a best/final file that appears only once in the list
    if variant == "oracle":
        have = {s for s, _ in out}
        for m in E["models"]:
            tr, seed, var = ident(m)
            if tr == train and seed not in have and (m.split("experiments/")[-1] in oracles or m in oracles):
                out.append((seed, m))
    return sorted(out, key=lambda x: (x[0] == "orig", x[0]))


def static_block(E, train, variant, oracles):
    rows = select(E, train, variant, oracles)
    rs = [E["static"][m] for _, m in rows if _ != "orig"]
    ok = [r for r in rs if r["success"]]
    ms = lambda k: ([float(np.mean([r[k] for r in ok])), float(np.std([r[k] for r in ok], ddof=1)) if len(ok) > 1 else 0.0]
                    if ok else [float("nan"), float("nan")])
    return {"n_seeds": len(rs), "success": len(ok), "outcomes": [r["outcome"] for r in rs],
            "steps": ms("steps"), "L": ms("path_length"), "eff": ms("efficiency"),
            "per_seed": {s: {k: E["static"][m][k] for k in ("outcome", "steps", "path_length", "efficiency", "min_clearance")}
                         for s, m in rows}}


def dynamic_block(E, train, variant, oracles):
    apf = E["base"]["dynamic"]["apf_runs"]
    apf_ok = np.array([r["success"] for r in apf])
    rows = [(s, m) for s, m in select(E, train, variant, oracles) if s != "orig"]
    per, pooled_apf_eff, pooled_arc_eff = {}, [], []
    succ = []
    for s, m in rows:
        arc = E["dynamic"][m]
        ok = np.array([r["success"] for r in arc])
        shared = np.where(ok & apf_ok)[0]
        a_only, b_only, p = mcnemar_exact(apf_ok, ok)
        ea = [apf[i]["efficiency"] for i in shared]; eb = [arc[i]["efficiency"] for i in shared]
        n, W, pw = wilcoxon_paired(eb, ea)
        pooled_apf_eff += ea; pooled_arc_eff += eb
        succ.append(int(ok.sum()))
        per[s] = {"success": int(ok.sum()), "collision": sum(r["outcome"] == "collision" for r in arc),
                  "timeout": sum(r["outcome"] == "timeout" for r in arc),
                  "out_of_bounds": sum(r["outcome"] == "out_of_bounds" for r in arc),
                  "success_ci95": wilson(int(ok.sum()), len(arc)),
                  "mcnemar_apf_only": a_only, "mcnemar_arc_only": b_only, "mcnemar_p": p,
                  "n_shared": int(len(shared)),
                  "shared_eff_apf": [float(np.mean(ea)), float(np.std(ea, ddof=1))] if len(ea) > 1 else [float(np.mean(ea)) if ea else float("nan"), 0.0],
                  "shared_eff_arc": [float(np.mean(eb)), float(np.std(eb, ddof=1))] if len(eb) > 1 else [float(np.mean(eb)) if eb else float("nan"), 0.0],
                  "shared_L_apf": [float(np.mean([apf[i]["path_length"] for i in shared]))] if len(shared) else [float("nan")],
                  "shared_L_arc": [float(np.mean([arc[i]["path_length"] for i in shared]))] if len(shared) else [float("nan")],
                  "arc_more_efficient": int(sum(b > a for a, b in zip(ea, eb))),
                  "wilcoxon_p": pw}
    n_pairs, Wp, pwp = wilcoxon_paired(pooled_arc_eff, pooled_apf_eff)
    return {"per_seed": per, "success_mean_sd": [float(np.mean(succ)), float(np.std(succ, ddof=1))] if len(succ) > 1 else [float(np.mean(succ)) if succ else float("nan"), 0.0],
            "success_min_max": [min(succ), max(succ)] if succ else None,
            "pooled_shared_pairs": n_pairs,
            "pooled_shared_eff_apf": [float(np.mean(pooled_apf_eff)), float(np.std(pooled_apf_eff, ddof=1))] if len(pooled_apf_eff) > 1 else None,
            "pooled_shared_eff_arc": [float(np.mean(pooled_arc_eff)), float(np.std(pooled_arc_eff, ddof=1))] if len(pooled_arc_eff) > 1 else None,
            "pooled_shared_eff_arc_ci95": bootstrap_mean_ci(pooled_arc_eff),
            "pooled_shared_eff_apf_ci95": bootstrap_mean_ci(pooled_apf_eff),
            "pooled_wilcoxon_p": pwp}


def apf_blocks(E):
    b = E["base"]
    st = b["static"]["apf"]
    dyn = b["dynamic"]["apf"]
    apf_runs = b["dynamic"]["apf_runs"]
    return {"static": st, "dynamic": dyn, "dynamic_success_ci95": wilson(dyn["success"], dyn["n"]),
            "dynamic_failures": [(i, r["outcome"], round(r["final_dist_to_goal"], 1)) for i, r in enumerate(apf_runs) if not r["success"]],
            "dynamic_min_clearance": min(r["min_clearance"] for r in apf_runs),
            "dynamic_escape_steps": sum(r.get("escape_steps", 0) for r in apf_runs),
            "dynamic_pushouts": sum(r.get("pushouts", 0) for r in apf_runs)}


def speed_table():
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, "speed_sweep_corridor*.json"))):
        r = json.load(open(f))
        for v, d in r["speeds"].items():
            e = rows.setdefault(float(v), {"ratio": d["ratio"], "apf": d["apf_success"], "arc": {}})
            assert e["apf"] == d["apf_success"]
            for m, k in zip(r["models"], d["arc_success"]):
                e["arc"][m.replace("\\", "/")] = k
    return dict(sorted(rows.items()))


def main():
    oracles = oracle_paths()
    rep, md = {}, ["# Results report (generated by experiments/report.py)", ""]
    evals = {L: merge_evals(L) for L in ("corridor", "urban")}

    for L, E in evals.items():
        if E is None:
            continue
        rep[L] = {"apf": apf_blocks(E), "arc": {}}
        for train in ("corridor", "urban"):
            for var in ("best", "final", "oracle"):
                rep[L]["arc"][f"{train}/{var}"] = {"static": static_block(E, train, var, oracles),
                                                   "dynamic": dynamic_block(E, train, var, oracles)}
        # original model
        for var in ("best", "final", "oracle"):
            rows = [m for s, m in select(E, "urban", var, oracles) if s == "orig"]
            if rows:
                m = rows[0]
                arc = E["dynamic"][m]
                rep[L]["arc"][f"original/{var}"] = {"static": E["static"][m],
                                                    "dynamic_success": sum(r["success"] for r in arc)}

    # ---- markdown
    if "corridor" in rep:
        C = rep["corridor"]
        a = C["apf"]
        md += ["## Table 3 — static head-to-head, Corridor Gauntlet (ARC trained on it)", "",
               "| planner | success | steps | L (m) | η (%) |", "|---|---|---|---|---|",
               f"| APF (10 runs) | {a['static']['success']}/{a['static']['n']} | {fmt(a['static']['steps'])} | {fmt(a['static']['path_length'])} | {fmt(a['static']['efficiency'])} |"]
        for var, lab in (("best", "ARC arc_best (primary)"), ("final", "ARC final ep300"), ("oracle", "ARC best checkpoint by eval (upper bound)")):
            s = C["arc"][f"corridor/{var}"]["static"]
            md.append(f"| {lab}, {s['n_seeds']} seeds | {s['success']}/{s['n_seeds']} | {fmt(s['steps'])} | {fmt(s['L'])} | {fmt(s['eff'])} |")
        md += ["", "ARC per seed (arc_best): " + "; ".join(f"s{k}: {v['outcome']} {v['steps']} st {v['path_length']:.0f} m"
                                                          for k, v in C["arc"]["corridor/best"]["static"]["per_seed"].items()), ""]

        md += ["## Table 4 — paired 50-environment dynamic suite, Corridor Gauntlet, both planners at 18 m/s", "",
               f"APF: {a['dynamic']['success']}/50 success (95% CI {a['dynamic_success_ci95'][0]:.0f}–{a['dynamic_success_ci95'][1]:.0f}%), "
               f"{a['dynamic']['collision']} collisions, {a['dynamic']['timeout']} timeouts; failures {a['dynamic_failures']}; "
               f"min clearance {a['dynamic_min_clearance']:.2f} m; escapes {a['dynamic_escape_steps']}, push-outs {a['dynamic_pushouts']}.", "",
               "| ARC model | success | collision | timeout | OOB | McNemar (APF-only / ARC-only, p) | shared | η shared APF | η shared ARC | ARC more efficient | Wilcoxon p |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for var in ("best", "final", "oracle"):
            D = C["arc"][f"corridor/{var}"]["dynamic"]
            for s, d in D["per_seed"].items():
                md.append(f"| corridor s{s} {var} | {d['success']}/50 | {d['collision']} | {d['timeout']} | {d['out_of_bounds']} | "
                          f"{d['mcnemar_apf_only']} / {d['mcnemar_arc_only']}, p={d['mcnemar_p']:.2g} | {d['n_shared']} | "
                          f"{fmt(d['shared_eff_apf'])} | {fmt(d['shared_eff_arc'])} | {d['arc_more_efficient']}/{d['n_shared']} | "
                          f"{d['wilcoxon_p']:.2g} |")
            md.append(f"| **{var}: across seeds** | {fmt(D['success_mean_sd'])} (min–max {D['success_min_max']}) | | | | | "
                      f"{D['pooled_shared_pairs']} pairs | {fmt(D['pooled_shared_eff_apf'] or [float('nan')]*2)} | "
                      f"{fmt(D['pooled_shared_eff_arc'] or [float('nan')]*2)} | | {D['pooled_wilcoxon_p']:.2g} |")
        md.append("")

    # transfer matrix
    md += ["## Transfer matrix (ARC, arc_best / oracle; 5 seeds per training layout; both planners at 18 m/s)", "",
           "| trained on → evaluated on | static det. success (best) | static (oracle) | dynamic success mean ± SD (best) | dynamic (oracle) | APF static / dynamic on that layout |",
           "|---|---|---|---|---|---|"]
    for train in ("corridor", "urban"):
        for L in ("corridor", "urban"):
            if L not in rep:
                continue
            b = rep[L]["arc"][f"{train}/best"]; o = rep[L]["arc"][f"{train}/oracle"]
            a = rep[L]["apf"]
            md.append(f"| {train} → {L} | {b['static']['success']}/{b['static']['n_seeds']} | {o['static']['success']}/{o['static']['n_seeds']} | "
                      f"{fmt(b['dynamic']['success_mean_sd'])}/50 | {fmt(o['dynamic']['success_mean_sd'])}/50 | "
                      f"{a['static']['success']}/{a['static']['n']} / {a['dynamic']['success']}/50 |")
    md.append("")

    # speed sweep
    sp = speed_table()
    if sp:
        rep["speed_sweep_corridor"] = sp
        md += ["## Speed-ratio sweep, Corridor Gauntlet (ARC arc_best, corridor-trained seeds)", "",
               "| agent speed (m/s) | r = v_obs,max / v | APF success | ARC success per seed | ARC mean ± SD |", "|---|---|---|---|---|"]
        for v, e in sp.items():
            ks = [e["arc"][m] for m in sorted(e["arc"])]
            md.append(f"| {v:g} | {e['ratio']:.2f} | {e['apf']}/50 | {ks} | {np.mean(ks):.1f} ± {np.std(ks, ddof=1) if len(ks) > 1 else 0:.1f} |")
        md.append("")

    # APF-only tables
    va = jload("verify_apf_static.json")
    if va:
        md += ["## Table 6 — APF on five static layouts (corrected protocol, 10 runs each)", "",
               "| layout | obstacles | success | steps | L (m) | η (%) |", "|---|---|---|---|---|---|"]
        for name, d in va["corrected_protocol_layouts"].items():
            md.append(f"| {name} | {d['n_obstacles']} | {d['success']}/{d['n']} | {d['steps'][0]:.0f} | {d['L'][0]:.1f} | {d['eff'][0]:.1f} |")
        md.append("")
    a3 = jload("apf3d.json")
    if a3:
        md += ["## Table 7 — APF, paper's 3D environment (10 runs each)", "",
               "| | success | steps | L (m) | η (%) |", "|---|---|---|---|---|"]
        for k in ("static", "dynamic"):
            d = a3[k]
            md.append(f"| 3D {k} | {d['success']}/{d['n']} | {fmt(d['steps'])} | {fmt(d['L'])} | {fmt(d['eff'])} |")
        md.append("")
    ai = jload("apf_in_arc3d.json")
    if ai:
        md += ["## Table 7b — APF inside ARC's 3D arena (10 runs each)", "",
               "| stage | solid obstacles | wind | success | steps | L (m) | η (%) | min clearance (m) |", "|---|---|---|---|---|---|---|---|"]
        for st in (0, 1, 2):
            for w in ("calm", "wind"):
                d = ai[f"stage{st}_{w}"]
                md.append(f"| {st} | {ai['arena']['n_solid'][str(st)]} | {w} | {d['success']}/{d['n']} | {fmt(d['steps'])} | {fmt(d['L'])} | {fmt(d['eff'])} | {d['min_clearance']:.1f} |")
        md.append("")
    tm = jload("timing.json")
    if tm:
        rep["timing"] = tm
        md += ["## Timing (idle machine, AC power)", "", "```", json.dumps({k: v for k, v in tm.items() if not k.endswith("_all") and k != "arc"}, indent=1, default=float), "```", ""]

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
