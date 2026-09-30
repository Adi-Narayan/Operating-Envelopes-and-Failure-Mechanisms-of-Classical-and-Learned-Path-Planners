"""
Time left for every training job in run_queue.JOBS (running, queued and done), plus any
evaluation jobs listed in out/logs/eval_tasks.json.

  * running : rate (s/episode) from the job's latest tqdm line -> remaining = (300 - n) * rate
  * queued  : simulated on SLOTS machine-wide CPU slots in queue order; each queued job is
              assumed to take 300 episodes at the mean current rate of running jobs of the same
              kind (TD3 moving-obstacle, TD3 static, SAC), so estimates tighten as jobs start.

    python progress.py              # print once
    python progress.py --watch 600  # redraw every 10 min (also writes out/logs/progress.txt)
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.modules["tensorflow"] = None
from run_queue import JOBS, name, out_dir  # noqa: E402

LOGS = os.path.join(HERE, "out", "logs")
SLOTS = 9                                        # machine-wide cap: TD3 runners (5 + 2) + SAC runner fills the rest
TQDM = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)(s/it|it/s)")


def hms(s):
    s = int(max(s, 0))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m"


def secs(t):
    p = [int(x) for x in t.split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(p)))


def last_tqdm(job):
    best = None
    for f in glob.glob(os.path.join(LOGS, name(job) + "*.log")):
        if "_killed" in f:                           # logs of runs that were terminated and restarted
            continue
        with open(f, "rb") as fh:
            fh.seek(max(0, os.path.getsize(f) - 4000))
            txt = fh.read().decode("utf-8", "replace")
        hits = TQDM.findall(txt.replace("\r", "\n"))
        if hits and (best is None or os.path.getmtime(f) > best[0]):
            best = (os.path.getmtime(f), hits[-1])
    if not best:
        return None
    n, tot, el, _, r, unit = best[1]
    rate = float(r) if unit == "s/it" else 1.0 / max(float(r), 1e-9)
    return int(n), int(tot), secs(el), rate


def status(job):
    d = out_dir(job)
    kind = "sac" if name(job).startswith("sacdyn") else ("dyn" if "--dynamic" in job else "static")
    if os.path.exists(os.path.join(d, "SKIPPED")):
        return dict(job=name(job), kind=kind, state="dropped")
    if os.path.exists(os.path.join(d, "train_summary.json")):
        w = json.load(open(os.path.join(d, "train_summary.json"))).get("wall_clock_s", 0)
        return dict(job=name(job), kind=kind, state="done", wall=w)
    if os.path.exists(os.path.join(d, "STARTED")):
        t = last_tqdm(job)
        if t:
            n, tot, el, rate = t
            return dict(job=name(job), kind=kind, state="running", n=n, tot=tot, elapsed=el, rate=rate,
                        left=(tot - n) * rate)
        return dict(job=name(job), kind=kind, state="running", n=0, tot=300, elapsed=0, rate=None, left=None)
    return dict(job=name(job), kind=kind, state="queued")


def eval_rows(now):
    spec = os.path.join(LOGS, "eval_tasks.json")
    rows = []
    for t in (json.load(open(spec)) if os.path.exists(spec) else []):
        f = os.path.join(LOGS, t["log"])
        if not os.path.exists(f):
            rows.append(dict(job=t["name"], state="queued"))
            continue
        txt = open(f, encoding="utf-8", errors="replace").read()
        n, start = txt.count(t["unit_marker"]), t["started"]
        if t.get("done_marker") and t["done_marker"] in txt or n >= t["total"]:
            rows.append(dict(job=t["name"], state="done", n=n, tot=t["total"]))
        elif "Traceback" in txt:
            rows.append(dict(job=t["name"], state="ERROR", n=n, tot=t["total"]))
        else:
            el = now - start
            left = el / n * (t["total"] - n) if n else None
            rows.append(dict(job=t["name"], state="running", n=n, tot=t["total"], elapsed=el, left=left))
    return rows


def bar(n, tot, w=20):
    k = int(w * n / tot) if tot else 0
    return "[" + "#" * k + "." * (w - k) + "]"


def render():
    now = time.time()
    rows = [status(j) for j in JOBS]
    run = [r for r in rows if r["state"] == "running"]
    rate = {}
    for kind in ("dyn", "static", "sac"):
        rs = [r["rate"] for r in run if r["kind"] == kind and r["rate"]]
        done = [r["wall"] / 300 for r in rows if r["kind"] == kind and r["state"] == "done"]
        rate[kind] = sum(rs) / len(rs) if rs else (sum(done) / len(done) if done else None)
    for kind in ("dyn", "static"):                 # TD3 kinds fall back to each other
        rate[kind] = rate[kind] or rate["static" if kind == "dyn" else "dyn"] or 150.0
    rate["sac"] = rate["sac"] or rate["dyn"]       # SAC before its first job reports a rate
    # one CPU pool of SLOTS processes, filled in queue order (TD3 first, then SAC)
    est = lambda r: r["left"] if r.get("left") is not None else 300 * rate[r["kind"]]
    free = sorted(est(r) for r in run) + [0.0] * max(0, SLOTS - len(run))
    for r in rows:
        if r["state"] == "running":
            r["finish"] = est(r)
        elif r["state"] == "queued":
            free.sort()
            start = free.pop(0)
            dur = 300 * rate[r["kind"]]
            r.update(start_in=start, left=start + dur, dur=dur, finish=start + dur)
            free.append(start + dur)
    clock = lambda s: (dt.datetime.now() + dt.timedelta(seconds=s)).strftime("%a %H:%M")
    out = [f"Training progress  (updated {dt.datetime.now():%a %H:%M:%S})", "",
           f"{'job':24s} {'state':8s} {'progress':22s} {'ep':>7s} {'elapsed':>8s} {'s/ep':>6s} {'left':>8s}  finishes"]
    unk = lambda r: r.get("finish") is None
    for r in rows:
        s = r["state"]
        if r["kind"] == "sac" and r["job"].endswith("seed0"):
            out.append("--- SAC moving-obstacle (CPU; fills slots TD3 leaves free) ---")
        if s == "done":
            out.append(f"{r['job']:24s} {'done':8s} {bar(1, 1):22s} {'300/300':>7s} {hms(r['wall']):>8s} "
                       f"{r['wall'] / 300:6.0f} {'-':>8s}  -")
        elif s == "running":
            rt = f"{r['rate']:6.0f}" if r["rate"] else f"{'?':>6s}"
            left = f"{'?':>8s}  ?" if unk(r) else f"{hms(r['finish']):>8s}  {clock(r['finish'])}"
            out.append(f"{r['job']:24s} {'running':8s} {bar(r['n'], r['tot']):22s} {str(r['n']) + '/' + str(r['tot']):>7s} "
                       f"{hms(r['elapsed']):>8s} {rt} {left}")
        elif s == "queued":
            out.append(f"{r['job']:24s} {'queued':8s} {'starts in ' + hms(r['start_in']):22s} {'0/300':>7s} {'-':>8s} "
                       f"{rate[r['kind']]:6.0f} {hms(r['finish']):>8s}  {clock(r['finish'])}  (est.)")
        else:
            out.append(f"{r['job']:24s} {'dropped':8s}")
    out.append("")
    for lab, lane in (("TD3 (CPU)", lambda r: r["kind"] != "sac"), ("SAC moving-obstacle (CPU)", lambda r: r["kind"] == "sac")):
        rr = [r for r in rows if lane(r) and r["state"] != "dropped"]
        fins = [r["finish"] for r in rr if r["state"] in ("running", "queued")]
        tail = "all done" if not fins else f"all done in ~{hms(max(fins))} ({clock(max(fins))})"
        out.append(f"{lab:27s}: {sum(r['state'] == 'done' for r in rr)}/{len(rr)} done, "
                   f"{sum(r['state'] == 'running' for r in rr)} running, {sum(r['state'] == 'queued' for r in rr)} queued  ->  {tail}")
    out.append("Queued estimates assume the current per-episode rate (SAC: TD3's until a SAC job reports).")
    ev = eval_rows(now)
    if ev:
        out += ["", "Evaluation jobs"]
        for r in ev:
            if r["state"] == "running":
                left = hms(r["left"]) if r["left"] is not None else "?"
                out.append(f"  {r['job']:40s} running {bar(r['n'], r['tot'])} {r['n']}/{r['tot']}  left ~{left}")
            else:
                out.append(f"  {r['job']:40s} {r['state']} {r.get('n', '')}/{r.get('tot', '')}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="redraw every N seconds")
    a = ap.parse_args()
    while True:
        txt = render()
        with open(os.path.join(LOGS, "progress.txt"), "w") as f:
            f.write(txt + "\n")
        if a.watch:
            os.system("cls" if os.name == "nt" else "clear")   # constant string, no user input
        print(txt, flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
