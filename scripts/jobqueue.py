"""
Resumable job queue for v2 experiments (instructions.md compute rules).

    python scripts/jobqueue.py run   --jobs <jobs.json> [--max-parallel 4] [--min-free-gb 0.7]
    python scripts/jobqueue.py status --jobs <jobs.json>

jobs.json: {"name": "<queue name>", "jobs": [{"name": ..., "cmd": [...], "priority": "normal"|"below",
            "clean": ["<path removed before a (re)start>", ...]}, ...]}

* State lives in results/v2/queue/<queue name>/: <job>.done (exit code 0), <job>.failed, <job>.log.
* On restart finished jobs are skipped; a job that was running when the queue died is restarted
  from scratch after its "clean" paths are removed, so a crash loses at most one job per worker.
* A job starts only while fewer than --max-parallel run and free RAM is at least --min-free-gb.
* One thread per process (OMP/MKL/OPENBLAS = 1); "below" jobs run at below-normal priority.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_gb():
    return psutil.virtual_memory().available / 2 ** 30


def load(path):
    spec = json.load(open(path))
    state = os.path.join(ROOT, "results", "v2", "queue", spec["name"])
    os.makedirs(state, exist_ok=True)
    return spec, state


def status(spec, state):
    rows = []
    for j in spec["jobs"]:
        n = j["name"]
        s = "done" if os.path.exists(os.path.join(state, n + ".done")) else \
            "failed" if os.path.exists(os.path.join(state, n + ".failed")) else \
            "running" if os.path.exists(os.path.join(state, n + ".running")) else "pending"
        rows.append((n, s))
    return rows


def run(spec, state, max_parallel, min_free, poll):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    pending = []
    for j in spec["jobs"]:
        n = j["name"]
        if os.path.exists(os.path.join(state, n + ".done")):
            continue
        for stale in (".running", ".failed"):
            p = os.path.join(state, n + stale)
            if os.path.exists(p):
                os.remove(p)
        pending.append(j)
    running = {}
    print(f"[{time.strftime('%H:%M:%S')}] queue {spec['name']}: {len(pending)} to run", flush=True)
    while pending or running:
        for n, (proc, logf) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            logf.close()
            os.remove(os.path.join(state, n + ".running"))
            open(os.path.join(state, n + (".done" if rc == 0 else ".failed")), "w").write(f"exit {rc} at {time.ctime()}\n")
            print(f"[{time.strftime('%H:%M:%S')}] {'DONE' if rc == 0 else 'FAILED'} {n} (exit {rc})", flush=True)
            del running[n]
        while pending and len(running) < max_parallel and free_gb() >= min_free:
            j = pending.pop(0)
            for p in j.get("clean", []):
                p = os.path.join(ROOT, p)
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.exists(p):
                    os.remove(p)
            logf = open(os.path.join(state, j["name"] + ".log"), "w")
            proc = subprocess.Popen(j["cmd"], cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
            if j.get("priority") == "below":
                try:
                    psutil.Process(proc.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except Exception:
                    pass
            open(os.path.join(state, j["name"] + ".running"), "w").write(f"pid {proc.pid} started {time.ctime()}\n")
            running[j["name"]] = (proc, logf)
            print(f"[{time.strftime('%H:%M:%S')}] START {j['name']} pid={proc.pid} free={free_gb():.2f} GB", flush=True)
            time.sleep(5)       # let memory use settle before the next launch decision
        time.sleep(poll)
    print(f"[{time.strftime('%H:%M:%S')}] QUEUE_DONE {spec['name']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "status"])
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=0.7)
    ap.add_argument("--poll", type=float, default=20)
    a = ap.parse_args()
    spec, state = load(a.jobs)
    if a.action == "status":
        for n, s in status(spec, state):
            print(f"{s:8s} {n}")
    else:
        run(spec, state, a.max_parallel, a.min_free_gb, a.poll)


if __name__ == "__main__":
    main()
