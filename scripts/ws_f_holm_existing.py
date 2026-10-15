"""
WS-F: Holm correction, within each family of tests the paper reports, of the p-values already stored
by experiments/report.py (report.json) and experiments/dwa_report.py (dwa_report.json). No test is
recomputed; only the adjustment is new. Families:
  paired_mcnemar   Table (paired dynamic suite): DWA, DWA-CV and the five ARC policies vs APF (7 tests)
  paired_wilcoxon  same table, efficiency on shared successes (policies with at least one shared pair)
  sac_vs_apf_mcnemar / sac_vs_apf_wilcoxon   ten SAC (moving) seeds vs APF
  dwacv_vs_sac_mcnemar                        DWA-CV vs each SAC seed
  learner_tests    seed-level tests of the TD3/SAC table: two permutation tests and Fisher's test

    python scripts/ws_f_holm_existing.py  -> results/v2/ws_f/holm_existing/result.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import OUT, dump, now, outdir, write_manifest  # noqa: E402
from stats_v2 import holm                                    # noqa: E402
import json                                                   # noqa: E402


def fam(tests):
    names = list(tests)
    adj = holm([tests[k] for k in names])
    return {k: {"p": tests[k], "holm_p": a, "significant_05": a < 0.05} for k, a in zip(names, adj)}


def main():
    t0 = now()
    rep = json.load(open(os.path.join(OUT, "report.json")))
    dwa = json.load(open(os.path.join(OUT, "dwa_report.json")))
    arc = rep["corridor"]["arc"]["corridor/best"]["dynamic"]["per_seed"]
    sac = rep["baselines"]["corridor"]["sacdyn/best"]["dynamic"]["per_seed"]
    res = {}
    t = {"DWA": dwa["DWA"]["vs_apf"]["mcnemar_p"], "DWA-CV": dwa["DWA-CV"]["vs_apf"]["mcnemar_p"]}
    t.update({f"ARC seed {s}": v["mcnemar_p"] for s, v in arc.items()})
    res["paired_mcnemar"] = fam(t)
    t = {"DWA": dwa["DWA"]["vs_apf"]["wilcoxon_p"], "DWA-CV": dwa["DWA-CV"]["vs_apf"]["wilcoxon_p"]}
    t.update({f"ARC seed {s}": v["wilcoxon_p"] for s, v in arc.items() if v.get("wilcoxon_p") is not None and v["wilcoxon_p"] == v["wilcoxon_p"]})   # NaN: too few pairs for a test
    res["paired_wilcoxon"] = fam(t)
    res["sac_vs_apf_mcnemar"] = fam({f"SAC seed {s}": v["mcnemar_p"] for s, v in sac.items()})
    res["sac_vs_apf_wilcoxon"] = fam({f"SAC seed {s}": v["wilcoxon_p"] for s, v in sac.items()})
    res["dwacv_vs_sac_mcnemar"] = fam({k.replace("sacdyn_", "SAC "): v["mcnemar_p"] for k, v in dwa["DWA-CV"]["vs_sac"].items()})
    cmp_ = rep["baselines"]["corridor_seed_comparison"]["best"]
    res["learner_tests"] = fam({"perm SAC vs TD3 (moving)": cmp_["perm_p_sacdyn_vs_td3dyn"],
                                "perm TD3 moving vs static": cmp_["perm_p_td3dyn_vs_td3"],
                                "Fisher collapse TD3 (moving) vs SAC": rep["baselines"]["collapse"]["fisher_td3dyn_vs_sacdyn_p"]})
    d = outdir("ws_f", "holm_existing")
    dump(d, "result.json", res)
    write_manifest(d, "scripts/ws_f_holm_existing.py", {"inputs": ["experiments/out/report.json", "experiments/out/dwa_report.json"]}, {}, t0)
    for f, rows in res.items():
        print(f"== {f}: {sum(r['significant_05'] for r in rows.values())} of {len(rows)} significant after Holm")
        for k, r in rows.items():
            print(f"   {k:38s} p={r['p']:.3g}  holm={r['holm_p']:.3g}")


if __name__ == "__main__":
    main()
