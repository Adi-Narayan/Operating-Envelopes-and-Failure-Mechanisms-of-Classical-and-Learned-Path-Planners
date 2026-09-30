"""
Training-collapse diagnostic for every 2D learned policy trained on the Corridor Gauntlet
(ARC seeds 0-4, TD3 static 0-9, TD3-dyn 0-9, SAC-dyn 0-9).

For each run: mean training reward over the first and last 50 episodes, and one deterministic run
of the retained checkpoint (arc_best) on the static Corridor from q0 with the heading toward the
goal, recording the fraction of steps whose executed heading change is at the action bound
(|dpsi| = 9 deg = pi/20). A run is 'collapsed' when that fraction is >= 0.99: the actor outputs the
bound on every step and the agent circles at the start (the training rewards are reported alongside).

    python collapse_diagnostic.py  ->  out/collapse.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.modules["tensorflow"] = None
import harness as H  # noqa: E402

RUNS = ([f"seed{s}" for s in range(5)] + [f"td3_seed{s}" for s in range(10)]
        + [f"td3dyn_seed{s}" for s in range(10)] + [f"sacdyn_seed{s}" for s in range(10)])
CAP_DEG = 9.0


def main():
    tl = H.StaticTimeline(H.LAYOUTS["corridor"])
    res = {}
    for run in RUNS:
        d = os.path.join(HERE, "out", "arc_models", run)
        if not os.path.exists(os.path.join(d, "train_summary.json")):
            print(run, "not trained yet; skipped", flush=True)
            continue
        t = json.load(open(os.path.join(d, "train_summary.json")))
        er = np.asarray(t["episode_rewards"], float)
        r = H.ARCPlanner(os.path.join(d, "models", "arc_best.pth")).run(tl, static=True)
        p = np.asarray(r["path"])
        hd = np.unwrap(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0])))
        turn = np.rad2deg(np.diff(hd))
        at_cap = float(np.mean(np.abs(np.abs(turn) - CAP_DEG) < 0.05)) if len(turn) else float("nan")
        res[run] = {"device": t.get("device", "cpu"), "dynamic_training": bool(t.get("dynamic_training", False)),
                    "algo": t.get("algo", "arc"), "wall_clock_h": t["wall_clock_s"] / 3600,
                    "reward_first50": float(er[:50].mean()), "reward_last50": float(er[-50:].mean()),
                    "reward_max": float(er.max()), "best_ckpt_outcome": r["outcome"],
                    "mean_turn_deg": float(turn.mean()) if len(turn) else float("nan"), "frac_steps_at_cap": at_cap,
                    "collapsed": bool(at_cap >= 0.99)}
        print(run, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[run].items()}, flush=True)
    json.dump(res, open(os.path.join(HERE, "out", "collapse.json"), "w"), indent=1)
    print("COLLAPSE_DONE", flush=True)


if __name__ == "__main__":
    main()
