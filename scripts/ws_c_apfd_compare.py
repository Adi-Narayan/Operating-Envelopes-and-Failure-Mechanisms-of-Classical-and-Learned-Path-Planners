"""
WS-C/WS-F: APF-D against APF, paired by environment, on every heatmap axis: successes, exact McNemar p,
Newcombe (method 10) 95% interval of the success difference, Holm correction over the family of the
seven axes; which of APF's static-layout stalls APF-D removes and which new failures it adds.

    python scripts/ws_c_apfd_compare.py  -> results/v2/ws_c/apfd_compare/result.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import OUT, ROOT, dump, now, outdir, write_manifest  # noqa: E402
from stats_v2 import holm, newcombe_paired                        # noqa: E402
from stats_utils import mcnemar_exact                              # noqa: E402

APF = "APF full|nocap|apf"


def ok(x):
    return (x if isinstance(x, str) else x["outcome"]) == "success"


def J(*p):
    return json.load(open(os.path.join(*p)))


def apf_axes():
    sp = J(OUT, "speed_sweep_corridor_s0234.json")["speeds"]
    noise = J(OUT, "noise_apf_arc.json")["results"]
    lay = J(OUT, "layouts_apf_arc.json")["results"]
    return {"indist": [ok(r) for r in J(OUT, "final_corridor_a.json")["dynamic"]["apf_runs"]],
            "r100": [ok(r) for r in sp["4.75"]["apf"]],
            "r158": [ok(r) for r in sp["3.0"]["apf"]],
            "noise5": [ok(r) for r in noise["L3"]["runs"][APF]],
            "lay_static": [ok(lay[str(k)]["static"][APF]) for k in range(100)],
            "lay_dynamic": [ok(lay[str(k)]["dynamic"][APF]) for k in range(100)],
            "second_dyn": [ok(r) for r in J(OUT, "final_urban_a.json")["dynamic"]["apf_runs"]]}


def apfd_axes():
    d = os.path.join(ROOT, "results", "v2", "ws_c", "apfd_eval")
    sp = J(d, "speed.json")["speeds"]
    lay = J(d, "layouts.json")
    return {"indist": [ok(r) for r in J(d, "dynamic.json")["runs"]],
            "r100": [ok(r) for r in sp["4.75"]],
            "r158": [ok(r) for r in sp["3.0"]],
            "noise5": [ok(r) for r in J(d, "noise.json")["levels"]["L3"]],
            "lay_static": [ok(r) for r in lay["static"]],
            "lay_dynamic": [ok(r) for r in lay["dynamic"]],
            "second_dyn": [ok(r) for r in J(d, "second.json")["runs"]]}


def main():
    t0 = now()
    a, b = apf_axes(), apfd_axes()
    rows = {}
    for ax in a:
        x, y = a[ax], b[ax]
        assert len(x) == len(y), ax
        diff, lo, hi = newcombe_paired(y, x)                     # APF-D minus APF
        rows[ax] = {"n": len(x), "apf": sum(x), "apfd": sum(y),
                    "apfd_only": sum(1 for p, q in zip(x, y) if q and not p),
                    "apf_only": sum(1 for p, q in zip(x, y) if p and not q),
                    "diff": diff, "newcombe95": [lo, hi], "mcnemar_p": mcnemar_exact(y, x)[2]}
    adj = holm([rows[ax]["mcnemar_p"] for ax in rows])
    for ax, p in zip(rows, adj):
        rows[ax]["holm_p"] = p
    stalls = [k for k, s in enumerate(a["lay_static"]) if not s]
    res = {"family": "the seven heatmap axes, APF-D vs APF (Holm within this family)", "axes": rows,
           "apf_static_stalls": len(stalls),
           "stalls_removed_by_apfd": sum(1 for k in stalls if b["lay_static"][k]),
           "new_static_failures": [k for k, s in enumerate(b["lay_static"]) if not s and a["lay_static"][k]]}
    d = outdir("ws_c", "apfd_compare")
    dump(d, "result.json", res)
    write_manifest(d, "scripts/ws_c_apfd_compare.py", {"stats": "exact McNemar, Newcombe method 10, Holm"}, {}, t0)
    for ax, r in rows.items():
        print(f"{ax:12s} APF {r['apf']:3d} APF-D {r['apfd']:3d} (+{r['apfd_only']} -{r['apf_only']}) "
              f"diff {r['diff']:+.2f} [{r['newcombe95'][0]:+.2f}, {r['newcombe95'][1]:+.2f}] p {r['mcnemar_p']:.3g} holm {r['holm_p']:.3g}")
    print("stalls", res["apf_static_stalls"], "removed", res["stalls_removed_by_apfd"], "new failures", res["new_static_failures"])


if __name__ == "__main__":
    main()
