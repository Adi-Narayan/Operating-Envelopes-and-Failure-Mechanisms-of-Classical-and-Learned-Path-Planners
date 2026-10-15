"""
Section 3 verification pass (instructions.md): reproduce reported numbers from the existing
code (3.1) and test that outcomes depend on speeds only through r (3.4).

    python scripts/verify_section3.py reproduce   -> results/v2/verification/reproduce/
    python scripts/verify_section3.py scaleinv    -> results/v2/verification/scale_invariance/

Each output folder gets result.json and manifest.json (git commit, config, seeds, start/end
time, whether the machine was also training). Nothing under experiments/out is written.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import datetime
import json
import math
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
OUT = os.path.join(EXP, "out")
sys.path.insert(0, EXP)
sys.modules["tensorflow"] = None
import harness as H            # noqa: E402
import apf_variants as V       # noqa: E402
import dwa                     # noqa: E402
from random_layouts import make_layout  # noqa: E402

SAC_SEED = "sacdyn_seed0"
SAC_MODEL = os.path.join(OUT, "arc_models", SAC_SEED, "models", "arc_best.pth")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def git_head():
    return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()


def training_active():
    """True if any training job (train_arc_seed.py) is running on this machine."""
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                       "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                                       "ForEach-Object { $_.CommandLine }"], text=True)
        return "train_arc_seed" in out
    except Exception:
        return None


def write(folder, result, config, seeds, t_start, busy):
    d = os.path.join(ROOT, "results", "v2", "verification", folder)
    os.makedirs(d, exist_ok=True)
    json.dump(result, open(os.path.join(d, "result.json"), "w"), indent=1, default=float)
    json.dump({"git_commit": git_head(), "script": "scripts/verify_section3.py " + folder,
               "config": config, "seeds": seeds, "start": t_start, "end": now(),
               "machine_also_training": busy, "threads_per_process": 1}, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    print("wrote", d, flush=True)


def same_run(a, b, keys=("outcome", "steps", "path_length")):
    """Compare a fresh run with a stored one: outcome and steps exactly, path length to 1e-6 m."""
    for k in keys:
        if k == "path_length":
            if not math.isclose(float(a[k]), float(b[k]), rel_tol=0, abs_tol=1e-6):
                return False
        elif a[k] != b[k]:
            return False
    return True


# ------------------------------------------------------------------ 3.1
def reproduce():
    t0, busy = now(), training_active()
    H.set_ground_speed(18.0)
    res = {}

    # (a) APF, 50 dynamic environments, 18 m/s (stored: final_corridor_a.json)
    stored = json.load(open(os.path.join(OUT, "final_corridor_a.json")))["dynamic"]["apf_runs"]
    runs = [H.run_apf(H.ObstacleTimeline(s), 1000 + s) for s in range(50)]
    succ = [r for r in runs if r["success"]]
    res["apf_dynamic"] = {
        "success": len(succ), "mean_eta": float(np.mean([r["efficiency"] for r in succ])),
        "paper": {"success": 50, "mean_eta": 72.7},
        "per_env_match_stored": [same_run(r, s) for r, s in zip(runs, stored)],
        "max_abs_dL_vs_stored": float(max(abs(r["path_length"] - s["path_length"]) for r, s in zip(runs, stored)))}
    res["apf_dynamic"]["all_match"] = all(res["apf_dynamic"]["per_env_match_stored"])
    print("APF dynamic", {k: v for k, v in res["apf_dynamic"].items() if k != "per_env_match_stored"}, flush=True)

    # (b) one SAC moving-obstacle seed (stored: final_corridor_sacdyn_*.json, per-seed report)
    stored_sac = None
    for f in sorted(os.listdir(OUT)):
        if f.startswith("final_corridor_sacdyn_") and f.endswith(".json"):
            d = json.load(open(os.path.join(OUT, f)))
            for m, rr in zip(d["models"], d["dynamic"]["arc_runs_per_model"]):
                if m.replace("\\", "/").endswith(f"{SAC_SEED}/models/arc_best.pth"):
                    stored_sac = rr
    rep = json.load(open(os.path.join(OUT, "report.json")))
    per_seed = rep["baselines"]["corridor"]["sacdyn/best"]["dynamic"]["per_seed"]
    report_count = per_seed[SAC_SEED.replace("sacdyn_seed", "")]["success"]
    P = H.ARCPlanner(SAC_MODEL)
    sac_runs = [P.run(H.ObstacleTimeline(s)) for s in range(50)]
    res["sac_dynamic"] = {
        "model": os.path.relpath(SAC_MODEL, ROOT).replace("\\", "/"),
        "success": sum(r["success"] for r in sac_runs),
        "report_per_seed_success": report_count,
        "per_env_match_stored": [same_run(r, s) for r, s in zip(sac_runs, stored_sac)],
        "max_abs_dL_vs_stored": float(max(abs(r["path_length"] - s["path_length"]) for r, s in zip(sac_runs, stored_sac)))}
    res["sac_dynamic"]["all_match"] = all(res["sac_dynamic"]["per_env_match_stored"])
    print("SAC dynamic", {k: v for k, v in res["sac_dynamic"].items() if k != "per_env_match_stored"}, flush=True)

    # (c) APF, 100 random static layouts (stored: layouts_apf_arc.json)
    lay = json.load(open(os.path.join(OUT, "layouts_apf_arc.json")))["results"]
    v = dict(sensing="full", turn_cap=False, rate="apf")
    lruns = [V.run(H.StaticTimeline(make_layout(k)), 0, **v) for k in range(100)]
    lst = [lay[str(k)]["static"]["APF full|nocap|apf"] for k in range(100)]
    fails = [r["outcome"] for r in lruns if r["outcome"] != "success"]
    res["apf_layouts_static"] = {
        "success": sum(r["outcome"] == "success" for r in lruns), "failures": {o: fails.count(o) for o in set(fails)},
        "paper": {"success": 68, "failures": {"timeout": 32}},
        "per_layout_match_stored": [same_run(r, s) for r, s in zip(lruns, lst)]}
    res["apf_layouts_static"]["all_match"] = all(res["apf_layouts_static"]["per_layout_match_stored"])
    print("APF layouts", {k: v for k, v in res["apf_layouts_static"].items() if k != "per_layout_match_stored"}, flush=True)

    # (d) DWA-CV, dynamic suite (stored: dwa_eval.json, weights from dwa_tune.json)
    w = json.load(open(os.path.join(OUT, "dwa_tune.json")))["selected"]["DWA-CV"]["weights"]
    stored_dwa = json.load(open(os.path.join(OUT, "dwa_eval.json")))["DWA-CV"]["dynamic"]["corridor"]
    druns = [dwa.run(H.ObstacleTimeline(s), predict=True, weights=w) for s in range(50)]
    res["dwacv_dynamic"] = {
        "weights": w, "success": sum(r["success"] for r in druns), "paper": {"success": 47},
        "per_env_match_stored": [same_run(r, stored_dwa[str(s)]) for s, r in enumerate(druns)]}
    res["dwacv_dynamic"]["all_match"] = all(res["dwacv_dynamic"]["per_env_match_stored"])
    print("DWA-CV dynamic", {k: v for k, v in res["dwacv_dynamic"].items() if k != "per_env_match_stored"}, flush=True)

    res["verdict"] = {
        "apf_dynamic": res["apf_dynamic"]["success"] == 50 and round(res["apf_dynamic"]["mean_eta"], 1) == 72.7 and res["apf_dynamic"]["all_match"],
        "sac_dynamic": res["sac_dynamic"]["success"] == report_count and res["sac_dynamic"]["all_match"],
        "apf_layouts_static": res["apf_layouts_static"]["success"] == 68 and res["apf_layouts_static"]["failures"] == {"timeout": 32} and res["apf_layouts_static"]["all_match"],
        "dwacv_dynamic": res["dwacv_dynamic"]["success"] == 47 and res["dwacv_dynamic"]["all_match"]}
    print("VERDICT", res["verdict"], flush=True)
    write("reproduce", res, {"ground_speed": 18.0, "apf_planner_seed": "1000+s (dynamic), 0 (static layouts)",
                             "sac_model": res["sac_dynamic"]["model"], "dwa_cv_weights": w},
          {"motion_seeds": "0-49", "layout_generator_seeds": "0-99"}, t0, busy)


# ------------------------------------------------------------------ 3.4
def scaled_timeline(seed, factor):
    """Motion seed `seed` with every obstacle velocity multiplied by `factor` (same random draws)."""
    tl = H.ObstacleTimeline(seed)
    for d in tl.dyns:
        d.vel = d.vel * factor
    return tl


def scaleinv():
    t0, busy = now(), training_active()
    P = H.ARCPlanner(SAC_MODEL)
    envs = list(range(10))
    res = {"envs": envs, "apf": [], "sac": []}
    for s in envs:
        H.set_ground_speed(18.0)
        a18 = H.run_apf(H.ObstacleTimeline(s), 1000 + s)
        s18 = P.run(H.ObstacleTimeline(s))
        H.set_ground_speed(9.0)
        a9 = H.run_apf(scaled_timeline(s, 0.5), 1000 + s)
        s9 = P.run(scaled_timeline(s, 0.5))
        for name, x, y in (("apf", a18, a9), ("sac", s18, s9)):
            same_len = len(x["path"]) == len(y["path"])
            dev = float(np.abs(np.asarray(x["path"]) - np.asarray(y["path"])).max()) if same_len else float("inf")
            res[name].append({"env": s, "outcome_18": x["outcome"], "outcome_9_half": y["outcome"],
                              "steps_18": x["steps"], "steps_9_half": y["steps"], "max_pointwise_dev_m": dev,
                              "bit_identical": same_len and bool(np.array_equal(np.asarray(x["path"]), np.asarray(y["path"])))})
        print(s, {n: (res[n][-1]["outcome_18"], res[n][-1]["outcome_9_half"], res[n][-1]["max_pointwise_dev_m"]) for n in ("apf", "sac")}, flush=True)
    H.set_ground_speed(18.0)
    for n in ("apf", "sac"):
        res[f"{n}_summary"] = {"same_outcome_all": all(r["outcome_18"] == r["outcome_9_half"] for r in res[n]),
                               "same_steps_all": all(r["steps_18"] == r["steps_9_half"] for r in res[n]),
                               "max_dev_m": max(r["max_pointwise_dev_m"] for r in res[n]),
                               "bit_identical_envs": sum(r["bit_identical"] for r in res[n])}
    print("SUMMARY", res["apf_summary"], res["sac_summary"], flush=True)
    write("scale_invariance", res, {"A": "agent 18 m/s, baseline obstacle speeds", "B": "agent 9 m/s, obstacle speeds x0.5",
                                    "sac_model": os.path.relpath(SAC_MODEL, ROOT).replace("\\", "/")},
          {"motion_seeds": envs, "apf_planner_seed": "1000+s"}, t0, busy)


if __name__ == "__main__":
    {"reproduce": reproduce, "scaleinv": scaleinv}[sys.argv[1]]()
