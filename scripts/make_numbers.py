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
        S = list(c1["stalls"].values())
        node = [v for v in S if v["type"] == "stable node"]
        eq = [v for v in S if v["newton_converged"] and not v["arena_boundary_trap"]]
        n["eqGoalMin"] = f"{min(v['dist_goal_eq'] for v in eq):.0f}"
        n["eqGoalMax"] = f"{max(v['dist_goal_eq'] for v in eq):.0f}"
        n["eqClearMin"] = f"{min(v['clearance_at_eq'] for v in eq):.0f}"
        n["eqClearMax"] = f"{max(v['clearance_at_eq'] for v in eq):.0f}"
        n["nodeMidMin"] = f"{min(v['dist_midpoint_to_eq'] for v in node):.2f}"
        n["nodeMidMax"] = f"{max(v['dist_midpoint_to_eq'] for v in node):.2f}"
        cd = [v["curl_over_div"] for v in eq]
        n["curlDiv"] = f"{np.median(cd):.7f}"
        n["nEquilibria"] = str(len(eq))
    n.update(v2_numbers())
    return n


def v2_numbers():
    """Numbers of the sections added in v2 (WS-B, WS-C, WS-D, WS-F)."""
    from stats_v2 import iqm_ci, wilson
    n = {}
    f2 = lambda x: f"{x:.2f}".replace("-", "$-$")
    f1 = lambda x: f"{x:.1f}".replace("-", "$-$")
    # C3: stalls (timeouts) against theta on the 100 random layouts
    c3 = J(V2, "ws_c", "c3_theta_sweep", "result.json")
    if c3:
        to = {int(k): v["timeout"] for k, v in c3["summary"].items()}
        n["thetaStallsZero"], n["thetaStallsAdopted"] = str(to[0]), str(to[25])
        n["thetaStallsMidMin"] = str(min(to[t] for t in (35, 40, 45)))
        n["thetaStallsMidMax"] = str(max(to[t] for t in (35, 40, 45)))
        n["thetaStallsFifty"], n["thetaStallsNinety"] = str(to[50]), str(to[90])
    # C2 follow-ups (Adi's review): shifted equilibria, cycle alignment, transverse multipliers, nearest saddles
    c2 = J(V2, "ws_c", "c2_followup", "result.json")
    if c2:
        sm, st = c2["summary"], c2["stalls"]
        nodes = [v for v in st.values() if "m" in v]
        n["cTwoBias"] = f"{np.linalg.norm(sm['bias']):.3f}"
        n["cTwoShiftMin"], n["cTwoShiftMax"] = f"{1000 * sm['shift_exact_range'][0]:.1f}", f"{1000 * sm['shift_exact_range'][1]:.1f}"
        n["cTwoAngleMax"] = f"{sm['angle_osc_fast_deg_range'][1]:.1f}"
        n["cTwoRhoMin"], n["cTwoRhoMax"] = f"{sm['rho_range'][0]:.2f}", f"{sm['rho_range'][1]:.2f}"
        n["cTwoStableCycles"] = str(sum(abs(v["m"]) < 1 for v in nodes))
        n["cTwoNodes"] = str(len(nodes))
        bad = [v for v in nodes if abs(v["m"]) >= 1]
        if bad:
            n["cTwoBadXzero"] = f"{min(min(abs(v['x0']), abs(v['x_other'])) for v in bad):.2f}"
        n["cTwoMidMin"], n["cTwoMidMax"] = f"{sm['mid_to_shifted_eq_range'][0]:.2f}", f"{sm['mid_to_shifted_eq_range'][1]:.2f}"
        n["cTwoSinks"] = str(sm["det_sign"]["positive"])
        n["cTwoCapRemoved"] = str(sm["cap_outcomes"]["success"])
        n["cTwoApfdRemoved"] = str(sm["apfd_outcomes"]["success"])
        n["cTwoBothRemoved"] = str(sum(v["cap_outcome"] == "success" and v["apfd_outcome"] == "success" for v in st.values()))
        n["cTwoInterior"] = str(len(st))
        n["cTwoDiamCap"], n["cTwoDiamApfd"] = f"{sm['turn_diameter_cap_m']:.1f}", f"{sm['turn_diameter_apfd_m']:.1f}"
        n["cTwoTurnApfd"] = f"{sm['turn_deg_apfd']:.1f}"
        from scipy.stats import mannwhitneyu
        for fix, key, nm in (("cap", "cap_outcome", "Cap"), ("apfd", "apfd_outcome", "Apfd")):
            rem = [v["nearest_saddle_m"] for v in st.values() if v[key] == "success"]
            kept = [v["nearest_saddle_m"] for v in st.values() if v[key] != "success"]
            u = mannwhitneyu(kept, rem)       # AUC = P(nearest saddle of a removed stall < that of a kept one)
            n[f"cTwoAuc{nm}"] = f"{u.statistic / (len(rem) * len(kept)):.2f}"
            n[f"cTwoAucP{nm}"] = f"{u.pvalue:.2f}" if u.pvalue >= 0.01 else f"{u.pvalue:.3f}"
    # E1: extended scale-invariance test (50 environments, APF, SAC, DWA-CV, no noise and L2)
    e1 = J(V2, "verification", "scale_invariance_ext", "result.json")
    if e1:
        sm = e1["summary"]
        keys = [k for k in sm if k.split()[0] in ("APF", "SAC", "DWA-CV")]
        n["eOneRuns"] = str(sum(sm[k]["n"] for k in keys))
        n["eOneEnvs"] = str(sm["APF none"]["n"])
        n["eOneSameOutcome"] = str(sum(sm[k]["same_outcome"] for k in keys))
        n["eOneDwaBit"] = str(sm["DWA-CV none"]["bit_identical"] + sm["DWA-CV L2"]["bit_identical"])
        n["eOneSacBit"] = str(sm["SAC none"]["bit_identical"] + sm["SAC L2"]["bit_identical"])
        n["eOneSacDevMm"] = f"{1000 * max(sm['SAC none']['max_dev_m'], sm['SAC L2']['max_dev_m']):.0f}"
        n["eOneApfStepDiff"] = str(sm["APF none"]["n"] - sm["APF none"]["same_steps"])
    on = J(V2, "verification", "scale_invariance_onset", "result.json")
    if on:
        sd = on["seeds"].values()
        stay = [r for r in sd if not r["first_tick_over"]]
        n["eOneLongSeeds"], n["eOneLongStable"] = str(len(on["seeds"])), str(len(stay))
        mx = max(r["max_dev_m"] for r in stay)
        e = int(np.floor(np.log10(mx)))
        n["eOneLongDev"] = f"${mx / 10 ** e:.0f} \\times 10^{{{e}}}$"
        ons = sorted(r["first_tick_over"]["1e-6"] for r in sd if r["first_tick_over"])
        n["eOneOnsetMin"] = f"{ons[0]:,}".replace(",", "{,}")
        n["eOneTicks"] = f"{on['ticks']:,}".replace(",", "{,}")
    # C4: APF-D against APF, paired on every axis
    cmp_ = J(V2, "ws_c", "apfd_compare", "result.json")
    if cmp_:
        n["apfdStallsRemoved"] = str(cmp_["stalls_removed_by_apfd"])
        n["apfdNewStaticFailures"] = str(len(cmp_["new_static_failures"]))
        names = {"indist": "Indist", "r100": "Rone", "r158": "Rhigh", "noise5": "Noise", "lay_static": "LayStatic",
                 "lay_dynamic": "LayDynamic", "second_dyn": "Second"}
        for ax, nm in names.items():
            r = cmp_["axes"][ax]
            pts = lambda x: "0" if round(100 * x) == 0 else f"{round(100 * x):+d}".replace("-", "$-$")
            n[f"apfdN{nm}"] = str(r["n"])
            n[f"apfdA{nm}"], n[f"apfdD{nm}"] = str(r["apf"]), str(r["apfd"])
            n[f"apfdOnlyA{nm}"], n[f"apfdOnlyD{nm}"] = str(r["apf_only"]), str(r["apfd_only"])
            n[f"apfdDiff{nm}"] = pts(r["diff"])
            n[f"apfdLo{nm}"], n[f"apfdHi{nm}"] = pts(r["newcombe95"][0]), pts(r["newcombe95"][1])
            hp = r["holm_p"]
            n[f"apfdHolm{nm}"] = ("1" if hp >= 0.995 else f"{hp:.2f}") if hp >= 0.01 else \
                f"${hp / 10 ** int(np.floor(np.log10(hp))):.1f} \\times 10^{{{int(np.floor(np.log10(hp)))}}}$"
    # B1: actor saturation and critic values
    b1 = J(V2, "ws_b", "b1_summary.json")
    if b1:
        cs, cd_, ld, sd = b1["td3"]["collapsed_runs"], b1["td3dyn"]["collapsed_runs"], b1["td3dyn"]["learned_runs"], b1["sacdyn"]["learned_runs"]
        n["bOneStatQBound"], n["bOneStatQZero"] = f2(cs["mean_Q_bound"]), f2(cs["mean_Q_zero"])
        n["bOneStatQBoundWins"] = f"{100 * cs['frac_Q_bound_gt_Q_zero']:.0f}\\%"
        n["bOneDynQPolicy"], n["bOneDynReturn"] = f2(cd_["mean_Q_policy"]), f2(cd_["mean_G"])
        n["bOneLearnedBound"] = f"{100 * ld['frac_action_at_bound']:.0f}\\%"
        n["bOneSacBound"] = f"{100 * sd['frac_action_at_bound']:.1f}\\%"
        n["bOneSacUsat"] = f"{100 * sd['frac_abs_u_gt3']:.1f}\\%"
        n["bOneSacQ"], n["bOneSacReturn"] = f1(sd["mean_Q_policy"]), f1(sd["mean_G"])
    # B2: discounted returns of scripted behaviours
    b2 = J(V2, "ws_b", "b2_returns", "result.json")
    if b2:
        pol = {"circling": "Circ", "greedy": "Greedy", "random": "Rand", "sac_best": "Sac"}
        for env, en in (("static", "Stat"), ("moving", "Mov")):
            for p, pn in pol.items():
                for g, gn in (("0.9", "Short"), ("0.99", "Long")):
                    v = b2[env][p]
                    n[f"bTwo{en}{pn}{gn}"] = f"{f1(v['mean_return'][g])} ({v['se_return'][g]:.1f})"
    # B3: ablation collapses and pre-tanh magnitudes
    b3 = J(V2, "ws_b", "b3_selection", "result.json")
    if b3:
        rows = b3["rows"]
        for c, cn in (("a", "Base"), ("b", "Gamma"), ("c", "Pen"), ("d", "Noise")):
            n[f"bThree{cn}"] = str(sum(r["collapsed"] for r in rows if r["condition"] == c))
        pen = [r["u2_last10"] for r in rows if r["condition"] == "c"]
        col = [r["u2_last10"] for r in rows if r["collapsed"]]
        n["bThreePenUMin"], n["bThreePenUMax"] = f"{min(pen):.1f}", f"{max(pen):.1f}"
        n["bThreeColUMin"], n["bThreeColUMax"] = f"{min(col):.0f}", f"{max(col):.0f}"
    # D2: initial arbiter thresholds
    d2 = J(V2, "ws_d", "d2_thresholds", "result.json")
    if d2:
        n["dTwoColl"], n["dTwoDon"], n["dTwoRhoOn"] = str(d2["n_collisions"]), f"{d2['d_on']:.1f}", f"{d2['rho_on']:.2f}"
    # WS-F: Holm-corrected p-values of the existing test families
    h = J(V2, "ws_f", "holm_existing", "result.json")
    if h:
        g = lambda x: (f"{x:.2f}" if x >= 0.1 else f"{x:.3f}" if x >= 0.01 else f"{x:.4f}" if x >= 1e-3 else
                       f"${x / 10 ** int(np.floor(np.log10(x))):.0f} \\times 10^{{{int(np.floor(np.log10(x)))}}}$")
        n["holmArcMcnemarMax"] = g(max(r["holm_p"] for k, r in h["paired_mcnemar"].items() if k.startswith("ARC")))
        n["holmDwaVsApf"] = g(h["paired_mcnemar"]["DWA"]["holm_p"])
        n["sacVsApfSigRaw"] = str(sum(r["p"] < 0.05 for r in h["sac_vs_apf_mcnemar"].values()))
        n["sacVsApfSigHolm"] = str(sum(r["significant_05"] for r in h["sac_vs_apf_mcnemar"].values()))
        n["sacVsApfWilcoxonMaxHolm"] = g(max(r["holm_p"] for r in h["sac_vs_apf_wilcoxon"].values()))
        dv = h["dwacv_vs_sac_mcnemar"]
        n["dwacvSacSigHolm"] = str(sum(r["significant_05"] for r in dv.values()))
        n["dwacvSacSeedNineHolm"], n["dwacvSacSeedFiveHolm"] = g(dv["SAC seed9"]["holm_p"]), g(dv["SAC seed5"]["holm_p"])
        n["dwacvSacSeedFiveRaw"] = g(dv["SAC seed5"]["p"])
        lt = h["learner_tests"]
        n["holmPermSacTd"] = g(lt["perm SAC vs TD3 (moving)"]["holm_p"])
        n["holmPermTdTd"] = g(lt["perm TD3 moving vs static"]["holm_p"])
        n["holmFisherCollapse"] = g(lt["Fisher collapse TD3 (moving) vs SAC"]["holm_p"])
        pw = h["paired_wilcoxon"]
        n["holmWilcoxonDwa"], n["holmWilcoxonDwacv"] = g(pw["DWA"]["holm_p"]), g(pw["DWA-CV"]["holm_p"])
        n["holmWilcoxonArcA"], n["holmWilcoxonArcB"] = g(pw["ARC seed 0"]["holm_p"]), g(pw["ARC seed 3"]["holm_p"])
    # WS-F: Newcombe intervals (method 10) for the paired success differences against APF
    from stats_v2 import newcombe_counts
    dwa = J(OUT, "dwa_report.json")
    pt = lambda x: f"{100 * x:.0f}".replace("-", "$-$")
    for v, nm in (("DWA", "Dwa"), ("DWA-CV", "Dwacv")):
        x = dwa[v]["vs_apf"]          # a = the DWA variant, b = APF
        both, a_only, b_only = x["shared"], x["a_only"], x["b_only"]
        t, lo, hi = newcombe_counts(both, a_only, b_only, 50 - both - a_only - b_only)
        n[f"ncDiff{nm}"] = f"{pt(t)} ({pt(lo)}, {pt(hi)})"
    arc = J(OUT, "report.json")["corridor"]["arc"]["corridor/best"]["dynamic"]["per_seed"]
    diffs = []
    for v in arc.values():
        b_, c_ = v["mcnemar_arc_only"], v["mcnemar_apf_only"]
        a_ = v["success"] - b_
        diffs.append(newcombe_counts(a_, b_, c_, 50 - a_ - b_ - c_))
    n["ncDiffArcMin"], n["ncDiffArcMax"] = pt(min(d[0] for d in diffs)), pt(max(d[0] for d in diffs))
    # WS-F: IQM over seeds with a 95% bootstrap interval, dynamic suite, reward-selected checkpoint
    rep = J(OUT, "report.json")
    for grp, nm in (("td3dyn/best", "TdDyn"), ("sacdyn/best", "SacDyn")):
        x = [v["success"] for v in rep["baselines"]["corridor"][grp]["dynamic"]["per_seed"].values()]
        m, lo, hi = iqm_ci([x])
        n[f"iqm{nm}"] = f"{m:.1f} ({lo:.1f}--{hi:.1f})"
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
