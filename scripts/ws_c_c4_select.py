"""
WS-C, C4: apply the pre-registered selection rule (prereg.md) to results/v2/ws_c/c4_tuning/.

    python scripts/ws_c_c4_select.py  -> results/v2/ws_c/c4_selection/
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import ROOT, dump, now, outdir, write_manifest  # noqa: E402

ORDER = ["apf", "osc10", "osc30", "mom050", "mom070", "mom085", "cap"]     # smallest change first


def main():
    t0 = now()
    src = os.path.join(ROOT, "results", "v2", "ws_c", "c4_tuning")
    R = {v: json.load(open(os.path.join(src, f"{v}.json"))) for v in ORDER}
    base_dyn = R["apf"]["success"]["heldout_seeds"]
    table = []
    for v in ORDER:
        s = R[v]["success"]
        lost = sum(1 for a, b in zip(R["apf"]["heldout_seeds"], R[v]["heldout_seeds"])
                   if a["outcome"] == "success" and b["outcome"] != "success")
        table.append({"variant": v, "static_tuning": s["static_layouts"], "dynamic_tuning": s["dynamic_layouts"],
                      "heldout_dynamic": s["heldout_seeds"], "heldout_lost_vs_apf": lost, "feasible": lost <= 1})
    feas = [r for r in table if r["feasible"]]
    best = max(r["static_tuning"] for r in feas)
    selected = next(r["variant"] for r in table if r["feasible"] and r["static_tuning"] == best)   # ORDER breaks ties
    out = {"table": table, "apf_heldout_dynamic": base_dyn, "selected": selected,
           "carried_forward": selected != "apf"}
    for r in table:
        print(r)
    print("SELECTED", selected)
    d = outdir("ws_c", "c4_selection")
    dump(d, "result.json", out)
    write_manifest(d, "scripts/ws_c_c4_select.py", {"rule": "prereg.md WS-C C4", "order": ORDER}, {"see": "c4_tuning manifest"}, t0)


if __name__ == "__main__":
    main()
