"""
Suspend / resume training processes without losing progress (frees RAM for other programs).

    python pause_jobs.py pause  "sac --dynamic --seed 3" "td3 --seed 9"
    python pause_jobs.py resume            # resumes everything listed in out/logs/paused.json
    python pause_jobs.py list

A job is matched on the arguments after --algo, and only python processes whose argv[1] is
train_arc_seed.py are considered (a shell whose command line mentions a job is never matched).
Pausing also trims the process's working set, so Windows hands the RAM back right away; the
pages are read back from the page file on resume. out/logs/paused.json tells watchdog.py that
these runs are paused rather than stalled.
"""
import ctypes
import json
import os
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "out", "logs", "paused.json")


def training():
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            c = p.info["cmdline"] or []
            if (p.info["name"] or "").lower().startswith("python") and len(c) > 1 and c[1].endswith("train_arc_seed.py"):
                out.append((p, " ".join(c[c.index("--algo") + 1:])))
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            pass
    return out


def run_dir(args):
    a = args.split()
    algo, seed, dyn = a[0], a[a.index("--seed") + 1], "--dynamic" in a
    return f"{algo}{'dyn' if dyn else ''}_seed{seed}"


def trim(pid):
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x0400 | 0x0100, False, pid)       # QUERY_INFORMATION | SET_QUOTA
    ok = bool(h) and bool(ctypes.windll.psapi.EmptyWorkingSet(h))
    if h:
        k32.CloseHandle(h)
    return ok


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def main():
    cmd, want = sys.argv[1], sys.argv[2:]
    state = load()
    procs = training()
    if cmd == "list":
        for p, a in procs:
            print(p.pid, p.status(), f"{p.memory_info().rss / 2**20:6.0f} MB", a, "(PAUSED)" if run_dir(a) in state else "")
        return
    if cmd == "pause":
        before = psutil.virtual_memory().available
        for w in want:
            hits = [(p, a) for p, a in procs if a.startswith(w) and (a == w or a[len(w)] == " ")]
            if len(hits) != 1:
                print(f"'{w}': expected exactly one matching job, found {len(hits)} -> skipped")
                continue
            p, a = hits[0]
            rss = p.memory_info().rss / 2**20
            p.suspend()
            print(f"paused pid {p.pid} ({a}); working set {rss:.0f} MB, trimmed: {trim(p.pid)}")
            state[run_dir(a)] = p.pid
        json.dump(state, open(STATE, "w"), indent=1)
        print(f"RAM available: {before / 2**30:.2f} -> {psutil.virtual_memory().available / 2**30:.2f} GB")
    elif cmd == "resume":
        rf = os.path.join(HERE, "out", "logs", "resumed.json")   # watchdog grace period after a resume
        resumed = json.load(open(rf)) if os.path.exists(rf) else {}
        for d, pid in list(state.items()):
            try:
                psutil.Process(pid).resume()
                resumed[d] = time.time()
                print(f"resumed pid {pid} ({d})")
            except psutil.NoSuchProcess:
                print(f"pid {pid} ({d}) no longer exists")
            state.pop(d)
        json.dump(state, open(STATE, "w"), indent=1)
        json.dump(resumed, open(rf, "w"), indent=1)


if __name__ == "__main__":
    main()
