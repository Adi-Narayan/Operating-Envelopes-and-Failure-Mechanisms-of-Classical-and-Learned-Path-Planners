"""Shared helpers for v2 experiments (instructions.md ground rules): output folders under
results/v2/<workstream>/<experiment>/, a manifest.json per experiment, single-threaded numerics."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import datetime
import json
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
OUT = os.path.join(EXP, "out")
FULL_CKPT = os.path.join(os.path.dirname(ROOT), "checkpoints_full")   # full TD3/SAC checkpoints (outside the repo)
if EXP not in sys.path:
    sys.path.insert(0, EXP)
sys.modules.setdefault("tensorflow", None)


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def training_active():
    """True if a training job (train_arc_seed.py) is running on this machine."""
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                       "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                                       "ForEach-Object { $_.CommandLine }"], text=True)
        return out.count("train_arc_seed") > 0
    except Exception:
        return None


def outdir(workstream, experiment):
    d = os.path.join(ROOT, "results", "v2", workstream, experiment)
    os.makedirs(d, exist_ok=True)
    return d


def write_manifest(d, script, config, seeds, start, busy=None, extra=None):
    m = {"git_commit": git_head(), "script": script, "config": config, "seeds": seeds,
         "start": start, "end": now(), "machine_also_training": training_active() if busy is None else busy,
         "threads_per_process": 1}
    if extra:
        m.update(extra)
    json.dump(m, open(os.path.join(d, "manifest.json"), "w"), indent=1, default=str)


def dump(d, name, obj):
    json.dump(obj, open(os.path.join(d, name), "w"), indent=1, default=float)
