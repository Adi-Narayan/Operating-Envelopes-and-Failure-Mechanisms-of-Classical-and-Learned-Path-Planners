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
        # td3_/td3dyn_/sac_ baselines get their own "train" label so they never mix with ARC seeds
        m = re.search(r"(env_apf_urban_|td3dyn_|td3_|sacdyn_|sac_)?seed(\d+)", p)
        pre = (m.group(1) or "").rstrip("_")
        train, seed = {"env_apf_urban": "urban", "": "corridor"}.get(pre, pre), m.group(2)
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


METHOD = {"corridor": "ARC", "urban": "ARC (second layout)", "td3": "TD3", "td3dyn": "TD3-dyn", "sacdyn": "SAC-dyn"}
BASELINES = ("td3", "td3dyn", "sacdyn")


def method_of(path):
    return ident(path)[0]


def speed_table(n_envs=50, trains=("corridor",)):
    """Speed sweep over the files with n_envs environments; models filtered by training label."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, "speed_sweep_corridor*.json"))):
        r = json.load(open(f))
        if len(next(iter(r["speeds"].values()))["apf"]) != n_envs:
            continue
        for v, d in r["speeds"].items():
            e = rows.setdefault(float(v), {"ratio": d["ratio"], "apf": d["apf_success"], "arc": {}})
            assert e["apf"] == d["apf_success"], f"APF differs between speed-sweep files at {v} m/s ({f})"
            for m, k in zip(r["models"], d["arc_success"]):
                if method_of(m) in trains:
                    e["arc"][m.replace("\\", "/")] = k
    return dict(sorted(rows.items()))


def exact_perm_test(a, b):
    """Two-sided exact permutation test on the difference of means (all C(n_a+n_b, n_a) splits)."""
    from itertools import combinations
    x = np.array(list(a) + list(b), float)
    n, na = len(x), len(a)
    obs = abs(np.mean(a) - np.mean(b))
    tot, cnt = 0, 0
    for idx in combinations(range(n), na):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        tot += 1
        cnt += abs(x[m].mean() - x[~m].mean()) >= obs - 1e-12
    return cnt / tot


def seed_summary(xs):
    xs = [float(v) for v in xs]
    return {"per_seed": xs, "mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
            "median": float(np.median(xs)), "min": min(xs), "max": max(xs),
            "mean_ci95_bootstrap_over_seeds": bootstrap_mean_ci(xs)}


def noise_tables():
    """Success per model and noise level, from every out/noise_*.json (50- and 10-environment files kept apart)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(OUT, "noise_*.json"))):
        r = json.load(open(f))
        for lev, R in r["results"].items():
            n = len(next(iter(R["runs"].values())))
            for m, runs in R["runs"].items():
                key = m.replace("\\", "/")
                meth = "APF " + key[4:] if key.startswith("APF ") else method_of(key)
                d = out.setdefault(n, {}).setdefault(meth, {}).setdefault(key, {})
                d[lev] = sum(x["outcome"] == "success" for x in runs)
    return out


def layout_tables():
    """Random-layout success per model (static / dynamic), from every out/layouts_*.json except the APF factorial."""
    out = {}
    for f in sorted(glob.glob(os.path.join(OUT, "layouts_*.json"))):
        if "factorial" in f:
            continue
        r = json.load(open(f))
        n = len(r["results"])
        for part in ("static", "dynamic"):
            for m in r["results"]["0"][part]:
                key = m.replace("\\", "/")
                meth = "APF " + key[4:] if key.startswith("APF ") else method_of(key)
                k = sum(r["results"][str(i)][part][m]["outcome"] == "success" for i in range(n))
                out.setdefault(n, {}).setdefault(part, {}).setdefault(meth, {})[key] = k
    return out


def envelope_summary(nt, lt, sp):
    """Per method: % success averaged over ALL its training seeds, for every noise level, both random-layout
    parts and every speed. A seed's 50-environment (100-layout) result is used where it exists, otherwise its
    10-environment spot check (the non-learning TD3 seeds)."""
    def run_id(key):
        return key.split("/")[-3] if "/" in key else key

    out = {}
    per = {}                                           # (test, method) -> {run: %}
    for n in sorted(nt):                               # smaller n first, overwritten by larger n
        for meth, models in nt[n].items():
            for m, d in models.items():
                for lev, k in d.items():
                    per.setdefault((f"noise_{lev}", meth), {})[run_id(m)] = 100 * k / n
    for n in sorted(lt):
        for part in ("static", "dynamic"):
            for meth, models in lt[n].get(part, {}).items():
                for m, k in models.items():
                    per.setdefault((f"layouts_{part}", meth), {})[run_id(m)] = 100 * k / n
    for n in sorted(sp):
        for v, e in sp[n].items():
            if n == 50:
                per.setdefault((f"speed_{v:g}", "APF"), {})["APF"] = 100 * e["apf"] / n
            for m, k in e["arc"].items():
                per.setdefault((f"speed_{v:g}", method_of(m)), {})[run_id(m)] = 100 * k / n
    for (test, meth), runs in per.items():
        xs = list(runs.values())
        out.setdefault(test, {})[meth] = {"n_seeds": len(xs), "mean_pct": float(np.mean(xs)),
                                          "sd_pct": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
                                          "min_pct": float(min(xs)), "max_pct": float(max(xs))}
    return out


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

    # learned baselines (TD3, TD3-dyn, SAC-dyn) on the same protocol; kept under their own key
    base = {}
    for L, E in evals.items():
        if E is None:
            continue
        for train in BASELINES:
            if not any(method_of(m) == train for m in E["models"]):
                continue
            for var in ("best", "final", "oracle"):
                base.setdefault(L, {})[f"{train}/{var}"] = {"static": static_block(E, train, var, oracles),
                                                           "dynamic": dynamic_block(E, train, var, oracles)}
    col = jload("collapse.json")
    if col:
        base["collapse"] = {}
        for lab, pre in (("corridor", "seed"), ("td3", "td3_seed"), ("td3dyn", "td3dyn_seed"), ("sacdyn", "sacdyn_seed")):
            runs = {k: v for k, v in col.items() if k.startswith(pre) and k[len(pre):].isdigit()}
            base["collapse"][lab] = {"n": len(runs), "collapsed": sorted(int(k[len(pre):]) for k, v in runs.items() if v["collapsed"]),
                                     "devices": sorted({v["device"] for v in runs.values()})}
        c = base["collapse"]
        if c.get("td3dyn", {}).get("n") and c.get("sacdyn", {}).get("n"):
            from scipy.stats import fisher_exact
            t, s = c["td3dyn"], c["sacdyn"]
            base["collapse"]["fisher_td3dyn_vs_sacdyn_p"] = float(fisher_exact(
                [[len(t["collapsed"]), t["n"] - len(t["collapsed"])], [len(s["collapsed"]), s["n"] - len(s["collapsed"])]])[1])
    if "corridor" in base:
        cmp = {}
        for var in ("best", "final", "oracle"):
            per = {lab: [d["success"] for _, d in sorted(blk["dynamic"]["per_seed"].items(), key=lambda x: int(x[0]))]
                   for lab, blk in [("corridor", rep["corridor"]["arc"][f"corridor/{var}"])]
                   + [(t, base["corridor"][f"{t}/{var}"]) for t in BASELINES if f"{t}/{var}" in base["corridor"]]}
            cmp[var] = {lab: seed_summary(xs) for lab, xs in per.items()}
            if "sacdyn" in per and "td3dyn" in per:
                cmp[var]["perm_p_sacdyn_vs_td3dyn"] = exact_perm_test(per["sacdyn"], per["td3dyn"])
            if "td3dyn" in per and "td3" in per:
                cmp[var]["perm_p_td3dyn_vs_td3"] = exact_perm_test(per["td3dyn"], per["td3"])
        base["corridor_seed_comparison"] = cmp
    nt, lt = noise_tables(), layout_tables()
    base["noise"], base["layouts"] = nt, lt
    sp_all = {n: speed_table(n, trains=("corridor",) + BASELINES) for n in (50, 10)}
    base["speed_sweep"] = {n: t for n, t in sp_all.items() if t}
    base["envelope_summary"] = envelope_summary(nt, lt, base["speed_sweep"])
    rep["baselines"] = base

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

    # ---- learned baselines (markdown)
    B = rep["baselines"]
    if B.get("collapse"):
        md += ["## Learned baselines — training collapse (corridor-trained; policy at the ±9° action bound on ≥99% of steps)", "",
               "| method | seeds | collapsed seeds | devices |", "|---|---|---|---|"]
        for lab in ("corridor", "td3", "td3dyn", "sacdyn"):
            c = B["collapse"].get(lab)
            if c and c["n"]:
                md.append(f"| {METHOD[lab]} | {c['n']} | {len(c['collapsed'])} {c['collapsed']} | {', '.join(c['devices'])} |")
        if "fisher_td3dyn_vs_sacdyn_p" in B["collapse"]:
            md.append(f"\nFisher exact, collapse TD3-dyn vs SAC-dyn: p = {B['collapse']['fisher_td3dyn_vs_sacdyn_p']:.3g}")
        md.append("")
    for L in ("corridor", "urban"):
        if L not in B:
            continue
        md += [f"## Learned baselines — Table 3/4 protocol on the {L} layout (per seed; APF as in Table 4)", "",
               "| method / ckpt | static success | dynamic success per seed | mean ± SD | pooled | shared pairs | η APF (shared) | η method (shared) | Wilcoxon p |",
               "|---|---|---|---|---|---|---|---|---|"]
        for t in BASELINES:
            for var in ("best", "final", "oracle"):
                blk = B[L].get(f"{t}/{var}")
                if not blk or not blk["dynamic"]["per_seed"]:
                    continue
                D, S = blk["dynamic"], blk["static"]
                ks = [d["success"] for _, d in sorted(D["per_seed"].items(), key=lambda x: int(x[0]))]
                md.append(f"| {METHOD[t]} {var} | {S['success']}/{S['n_seeds']} | {ks} | {fmt(D['success_mean_sd'])} | {sum(ks)}/{50 * len(ks)} | "
                          f"{D['pooled_shared_pairs']} | {fmt(D['pooled_shared_eff_apf'] or [float('nan')] * 2)} | "
                          f"{fmt(D['pooled_shared_eff_arc'] or [float('nan')] * 2)} | {D['pooled_wilcoxon_p']:.2g} |")
        md.append("")
    if B.get("corridor_seed_comparison"):
        md += ["## Per-seed dynamic success (corridor suite): method comparison over training seeds", "",
               "| ckpt | method | n seeds | mean ± SD | median | min–max | bootstrap 95% CI of mean |", "|---|---|---|---|---|---|---|"]
        for var, cm in B["corridor_seed_comparison"].items():
            for lab, s in cm.items():
                if isinstance(s, dict):
                    md.append(f"| {var} | {METHOD[lab]} | {len(s['per_seed'])} | {s['mean']:.1f} ± {s['sd']:.1f} | {s['median']:.1f} | "
                              f"{s['min']:.0f}–{s['max']:.0f} | {s['mean_ci95_bootstrap_over_seeds'][0]:.1f}–{s['mean_ci95_bootstrap_over_seeds'][1]:.1f} |")
            md.append(f"| {var} | exact permutation p: SAC-dyn vs TD3-dyn = {cm.get('perm_p_sacdyn_vs_td3dyn', float('nan')):.3g}; "
                      f"TD3-dyn vs TD3 = {cm.get('perm_p_td3dyn_vs_td3', float('nan')):.3g} | | | | | |")
        md.append("")
    es = B.get("envelope_summary", {})
    if es:
        md += ["## Envelope summary: % success, mean ± SD over ALL training seeds of each method", "",
               "(50-env / 100-layout result per seed where available, else the 10-env spot check)", "",
               "| test | " + " | ".join(es[next(iter(es))].keys()) + " |", "|---" * (1 + len(es[next(iter(es))])) + "|"]
        meths = sorted({m for t in es.values() for m in t})
        md[-2] = "| test | " + " | ".join(METHOD.get(m, m) for m in meths) + " |"
        md[-1] = "|---" * (1 + len(meths)) + "|"
        for test in sorted(es):
            md.append(f"| {test} | " + " | ".join(
                f"{es[test][m]['mean_pct']:.1f} ± {es[test][m]['sd_pct']:.1f} (n={es[test][m]['n_seeds']})" if m in es[test] else ""
                for m in meths) + " |")
        md.append("")
    for n, T in sorted(B.get("noise", {}).items(), reverse=True):
        md += [f"## Noise robustness, {n} paired dynamic environments (success per model; L0–L3 as in robustness_suite.py)", "",
               "| method | model | L0 | L1 | L2 | L3 |", "|---|---|---|---|---|---|"]
        for meth, models in T.items():
            for m, d in models.items():
                md.append(f"| {METHOD.get(meth, meth)} | {m.split('/')[-3] if '/' in m else m} | " + " | ".join(str(d.get(l, '')) for l in ("L0", "L1", "L2", "L3")) + " |")
        md.append("")
    for n, T in sorted(B.get("layouts", {}).items(), reverse=True):
        md += [f"## Random layouts, {n} layouts (success per model)", "", "| method | model | static | dynamic |", "|---|---|---|---|"]
        for meth in T.get("static", {}):
            for m, k in T["static"][meth].items():
                md.append(f"| {METHOD.get(meth, meth)} | {m.split('/')[-3] if '/' in m else m} | {k}/{n} | {T['dynamic'][meth].get(m, '')}/{n} |")
        md.append("")
    for n, sp2 in sorted(B.get("speed_sweep", {}).items(), reverse=True):
        md += [f"## Speed sweep, {n} environments per speed (success per seed, by method)", "",
               "| v (m/s) | r | APF | " + " | ".join(METHOD[t] for t in ("corridor",) + BASELINES) + " |", "|---" * 7 + "|"]
        for v, e in sp2.items():
            cells = []
            for t in ("corridor",) + BASELINES:
                ks = [k for m, k in sorted(e["arc"].items()) if method_of(m) == t]
                cells.append(f"{ks}" if ks else "")
            md.append(f"| {v:g} | {e['ratio']:.2f} | {e['apf']}/{n} | " + " | ".join(cells) + " |")
        md.append("")

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
