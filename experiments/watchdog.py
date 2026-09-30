"""
Read-only health check for the training queues. Every 60 s it verifies that
  * every started-but-unfinished TD3/SAC run (STARTED marker, no train_summary.json)
    still has a live train_arc_seed.py process
  * each queue runner is alive (or has logged ALL_JOBS_DONE)
  * each running job's log was written within STALL_MIN minutes
  * available RAM stays above LOW_RAM_GB and the GPU sensor below HOT_C
and exits with a report as soon as anything changes (job finished, job died, stall,
runner down, low RAM), so the caller can react and re-arm it. It never kills or starts
anything.

    python watchdog.py
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "out", "arc_models")
LOGS = os.path.join(HERE, "out", "logs")
STALL_MIN, LOW_RAM_GB, PERIOD = 25, 0.35, 60
HOT_C = 90   # GPU sensor, idle now: shares the laptop's cooling with the CPU, so it is a heat proxy
RUNNERS = {  # label -> (fingerprint in the command line, runner log)
    "CPU runner A (--parallel 5)": ("--parallel 5 --min-free-gb 0.8", "queue.log"),
    "CPU runner B (_q2)": ("--log-suffix _q2", "queue2.log"),
    "SAC CPU runner (_sac)": ("--log-suffix _sac", "queue_sac.log"),
    "TD3 seed-2 CPU retrain runner (_cpu)": ("--log-suffix _cpu", "queue_td3s2cpu.log"),
}
RUN_DIR = re.compile(r"^(td3|sac)(dyn)?_seed(\d+)$")


def procs():
    """python processes only, keyed by the script they execute (argv[1]), so shells whose
    command text merely mentions a script are never mistaken for it"""
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            c = p.info["cmdline"] or []
            if (p.info["name"] or "").lower().startswith("python") and len(c) > 1:
                out.append((p.info["pid"], " ".join(c), os.path.basename(c[1])))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def runs():
    """started-but-unfinished and finished TD3/SAC runs: {dir: (algo, dyn, seed, finished)}"""
    res = {}
    for d in os.listdir(MODELS):
        m = RUN_DIR.match(d)
        p = os.path.join(MODELS, d)
        if not m or not os.path.exists(os.path.join(p, "STARTED")) or os.path.exists(os.path.join(p, "SKIPPED")):
            continue
        res[d] = (m.group(1), bool(m.group(2)), int(m.group(3)), os.path.exists(os.path.join(p, "train_summary.json")))
    return res


def job_log(algo, dyn, seed):
    fs = [f for f in glob.glob(os.path.join(LOGS, f"{algo}{'dyn' if dyn else ''}_300ep_seed{seed}*.log")) if "_killed" not in f]
    return max(fs, key=os.path.getmtime) if fs else None


def check():
    ps = procs()
    train = [c for _, c, script in ps if script == "train_arc_seed.py"]
    state, problems = {}, []
    pf = os.path.join(LOGS, "paused.json")        # runs suspended on purpose by pause_jobs.py
    paused = json.load(open(pf)) if os.path.exists(pf) else {}
    rf = os.path.join(LOGS, "resumed.json")       # no stall check within STALL_MIN of a resume
    resumed = json.load(open(rf)) if os.path.exists(rf) else {}
    for d, (algo, dyn, seed, fin) in sorted(runs().items()):
        if fin:
            state[d] = "done"
            continue
        want = f"--algo {algo} {'--dynamic ' if dyn else ''}--seed {seed}"
        alive = any(want in c and (dyn or "--dynamic" not in c) for c in train)
        log = job_log(algo, dyn, seed)
        age = (time.time() - os.path.getmtime(log)) / 60 if log else None
        if alive and d in paused:
            state[d] = "paused"
        elif not alive:
            state[d] = "DIED"
            problems.append(f"{d}: process gone without train_summary.json (log {os.path.basename(log) if log else '?'})")
        elif age is not None and age > STALL_MIN and time.time() - resumed.get(d, 0) > STALL_MIN * 60:
            state[d] = "STALLED"
            problems.append(f"{d}: alive but log not written for {age:.0f} min")
        else:
            state[d] = "running"
    for lab, (fp, logname) in RUNNERS.items():
        alive = any(script == "run_queue.py" and fp in c for _, c, script in ps)
        lp = os.path.join(LOGS, logname)
        finished = os.path.exists(lp) and "ALL_JOBS_DONE" in open(lp, errors="replace").read()
        state[lab] = "alive" if alive else ("finished" if finished else "DOWN")
        if state[lab] == "DOWN":
            problems.append(f"{lab}: runner process not found and no ALL_JOBS_DONE in {logname}")
    try:
        temp = int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=20).stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        temp = None
    state["hot"] = temp is not None and temp >= HOT_C
    if state["hot"]:
        problems.append(f"GPU sensor {temp} C >= {HOT_C} C (laptop running hot)")
    avail = psutil.virtual_memory().available / 2**30
    state["low_ram"] = avail < LOW_RAM_GB
    if state["low_ram"]:
        problems.append(f"available RAM {avail:.2f} GB < {LOW_RAM_GB} GB")
    return state, problems, avail, len(train)


def main():
    base, probs, avail, n = check()
    stamp = lambda: time.strftime("%a %H:%M:%S")
    print(f"{stamp()} watchdog armed: {n} training processes, "
          f"{sum(v == 'running' for v in base.values())} runs running, "
          f"{sum(v == 'done' for v in base.values())} done, RAM available {avail:.2f} GB", flush=True)
    if probs:                                     # report pre-existing problems immediately
        print("PROBLEMS:\n  " + "\n  ".join(probs), flush=True)
        sys.exit(2)
    while True:
        time.sleep(PERIOD)
        st, probs, avail, n = check()
        changes = [f"{k}: {base.get(k, 'new')} -> {v}" for k, v in st.items() if base.get(k) != v]
        if changes or probs:
            print(f"{stamp()} CHANGE ({n} training processes, RAM available {avail:.2f} GB)", flush=True)
            for c in changes:
                print("  " + c, flush=True)
            if probs:
                print("PROBLEMS:\n  " + "\n  ".join(probs), flush=True)
            sys.exit(2 if probs else 0)


if __name__ == "__main__":
    main()
