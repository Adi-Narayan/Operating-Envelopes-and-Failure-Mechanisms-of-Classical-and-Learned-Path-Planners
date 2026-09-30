"""
Run a list of training jobs with at most N concurrent processes (one CPU thread each).

    python run_queue.py --parallel 5

Jobs are defined in JOBS below; each writes its log to out/logs/<name>.log and is
skipped by train_arc_seed.py if its output folder already has a STARTED marker.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

JOBS = (
    # priority 1: TD3 trained WITH moving obstacles (10 seeds) and TD3 on the static
    # layout (10 seeds), interleaved so both have results early
    [["--algo", "td3", "--dynamic", "--seed", str(s)] for s in range(0, 5)]
    + [["--algo", "td3", "--seed", str(s)] for s in range(0, 5)]
    + [["--algo", "td3", "--dynamic", "--seed", str(s)] for s in range(5, 10)]
    + [["--algo", "td3", "--seed", str(s)] for s in range(5, 10)]
    # SAC on the static layout: dropped (folders carry SKIPPED markers)
    + [["--algo", "sac", "--seed", str(s)] for s in range(5)]
    # SAC trained WITH moving obstacles (10 seeds, CPU), run by a separate runner
    # (--only sacdyn --global-cap 9) that fills CPU slots the TD3 runners leave free
    + [["--algo", "sac", "--dynamic", "--seed", str(s)] for s in range(10)]
)


def name(args):
    dyn = "dyn" if "--dynamic" in args else ""
    a = [x for x in args if x != "--dynamic"]
    d = dict(zip(a[::2], a[1::2]))
    return f"{d['--algo']}{dyn}_{d.get('--episodes', '300')}ep_seed{d['--seed']}"


def out_dir(args):
    from train_arc_seed import run_name
    a = [x for x in args if x != "--dynamic"]
    d = dict(zip(a[::2], a[1::2]))
    return os.path.join(HERE, "out", "arc_models", run_name(d["--algo"], "corridor_gauntlet", int(d["--seed"]),
                                                            int(d.get("--episodes", 300)), "--dynamic" in args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=5, help="maximum concurrent training processes")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--pythonpath", default=None, help="prepended to PYTHONPATH of the jobs (e.g. a CUDA torch build)")
    ap.add_argument("--log-suffix", default="", help="appended to log names (for a second concurrent runner)")
    ap.add_argument("--min-free-gb", type=float, default=0.8,
                    help="only start a new process while at least this much RAM is available (each uses ~0.3 GB)")
    ap.add_argument("--only", default=None, help="run only jobs whose name starts with this (e.g. sacdyn)")
    ap.add_argument("--global-cap", type=int, default=None,
                    help="only start a job while fewer than this many training processes run machine-wide "
                         "(lets this runner fill slots other runners leave free)")
    a = ap.parse_args()
    import psutil

    def n_training():
        # only python processes executing the script (not shells whose command text mentions it)
        n = 0
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                c = p.info["cmdline"] or []
                n += (p.info["name"] or "").lower().startswith("python") and len(c) > 1 \
                    and c[1].endswith("train_arc_seed.py")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return n

    os.makedirs(os.path.join(HERE, "out", "logs"), exist_ok=True)
    pending, running = [j for j in JOBS if a.only is None or name(j).startswith(a.only)], []
    while pending or running:
        running = [(p, n) for p, n in running if p.poll() is None]
        while (pending and len(running) < a.parallel
               and psutil.virtual_memory().available / 2**30 >= a.min_free_gb
               and (a.global_cap is None or n_training() < a.global_cap)):
            job = pending.pop(0)
            if os.path.exists(os.path.join(out_dir(job), "STARTED")):
                continue                      # already running or finished (e.g. by another runner)
            log = open(os.path.join(HERE, "out", "logs", name(job) + a.log_suffix + ".log"), "w")
            env = dict(os.environ)
            if a.pythonpath:
                env["PYTHONPATH"] = a.pythonpath + os.pathsep + env.get("PYTHONPATH", "")
            p = subprocess.Popen([sys.executable, os.path.join(HERE, "train_arc_seed.py")] + job + ["--device", a.device],
                                 stdout=log, stderr=subprocess.STDOUT, cwd=HERE, env=env)
            running.append((p, name(job)))
            print(time.strftime("%H:%M:%S"), "started", name(job), flush=True)
            time.sleep(30)   # let the new process allocate before re-checking free RAM
        time.sleep(20)
    print(time.strftime("%H:%M:%S"), "ALL_JOBS_DONE", flush=True)


if __name__ == "__main__":
    main()
