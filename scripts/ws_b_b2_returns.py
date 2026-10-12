"""
WS-B, B2: discounted returns of scripted policies under the exact training reward and
exploring-start distribution (no training).

Policies (normalised action a in [-1, 1], heading change a * pi/20 per step, as in training):
  circling : a = +1 every step (constant maximum turn)
  greedy   : turn towards the goal bearing, clipped to the bound, ignoring obstacles
  random   : a ~ U(-1, 1) each step
  sac_best : the deterministic mean action of the SAC checkpoint with the most dynamic-suite
             successes (sacdyn_seed3, arc_best, 49/50)
Environments: the static custom environment (ARC / TD3-static / B5 training) and the
moving-obstacle training environment (TD3-moving / SAC training).
Each policy runs from the same 100 exploring starts (np.random seeded 0-99 before each reset,
which draws the start, the heading and, in the moving environment, the motion seed), to the end
of the rollout (goal, collision, leaving the arena, or 500 steps). Returns at gamma = 0.9 and
0.99 come from the same reward sequence.

    python scripts/ws_b_b2_returns.py  -> results/v2/ws_b/b2_returns/
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import EXP, OUT, dump, now, outdir, write_manifest  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(EXP), "arc_2d", "code"))
import torch                                                    # noqa: E402
torch.set_num_threads(1)

N_STARTS, GAMMAS = 100, (0.9, 0.99)
SAC_BEST = os.path.join(OUT, "arc_models", "sacdyn_seed3", "models", "arc_best.pth")


def make_env(dynamic):
    from config import ENV_CONFIG, ENVIRONMENTS
    from environment import UAVEnvironment
    env = UAVEnvironment(ENVIRONMENTS["corridor_gauntlet"], ENV_CONFIG)
    if dynamic:
        from dyn_train_env import make_dynamic
        make_dynamic(env)
    return env, ENV_CONFIG


def policies():
    from baselines import load_agent
    from config import NETWORK_CONFIG
    from replay_buffer import ObservationHistory
    sac = load_agent(SAC_BEST)

    def circling(env, hist, rng):
        return 1.0

    def greedy(env, hist, rng):
        g = env.target_pos - env.uav_pos
        bearing = math.atan2(g[1], g[0])
        d = math.atan2(math.sin(bearing - env.uav_heading), math.cos(bearing - env.uav_heading))
        return float(np.clip(d / env.action_range, -1, 1))

    def random_(env, hist, rng):
        return float(rng.uniform(-1, 1))

    def sac_best(env, hist, rng):
        a, _ = sac.select_action(hist.get_tensor(sac.device), None, 0.0)
        return float(a[0])

    return {"circling": circling, "greedy": greedy, "random": random_, "sac_best": sac_best}, \
        lambda: ObservationHistory(NETWORK_CONFIG["history_length"], 3 + 9)


def rollout(env, cfg, pol, new_hist, start):
    np.random.seed(start)
    obs = env.reset()
    hist = new_hist()
    hist.reset(obs)
    rng = np.random.default_rng(1000 + start)        # only the random policy uses it
    rewards, info, step, done = [], {}, 0, False
    while not done and step < cfg["max_steps"]:
        a = pol(env, hist, rng)
        obs, r, done, info = env.step(np.array([a], dtype=np.float32)[0])
        hist.append(obs)
        rewards.append(float(r))
        step += 1
    p = env.uav_pos
    oob = bool(p[0] < 0 or p[0] > env.width or p[1] < 0 or p[1] > env.height)
    outcome = "success" if info.get("success") else "collision" if info.get("collision") else \
        "out_of_bounds" if oob else "timeout"
    rets = {str(g): float(sum(r * g ** t for t, r in enumerate(rewards))) for g in GAMMAS}
    return {"start": start, "steps": step, "outcome": outcome, "undiscounted": float(sum(rewards)), "returns": rets}


def main():
    t0 = now()
    pols, new_hist = policies()
    res = {}
    for dynamic in (False, True):
        env, cfg = make_env(dynamic)
        key = "moving" if dynamic else "static"
        res[key] = {}
        for name, pol in pols.items():
            runs = [rollout(env, cfg, pol, new_hist, s) for s in range(N_STARTS)]
            out = [r["outcome"] for r in runs]
            res[key][name] = {
                "mean_return": {g: float(np.mean([r["returns"][g] for r in runs])) for g in map(str, GAMMAS)},
                "se_return": {g: float(np.std([r["returns"][g] for r in runs], ddof=1) / math.sqrt(len(runs))) for g in map(str, GAMMAS)},
                "outcomes": {o: out.count(o) for o in sorted(set(out))},
                "mean_steps": float(np.mean([r["steps"] for r in runs])), "runs": runs}
            print(key, name, {g: round(v, 2) for g, v in res[key][name]["mean_return"].items()}, res[key][name]["outcomes"], flush=True)
    d = outdir("ws_b", "b2_returns")
    dump(d, "result.json", res)
    write_manifest(d, "scripts/ws_b_b2_returns.py", {"gammas": GAMMAS, "sac_best": "sacdyn_seed3/arc_best",
                                                    "reward": "training reward (arc_2d/code/environment.py)",
                                                    "max_steps": 500}, {"starts": "np.random.seed(0-99)", "random_policy": "default_rng(1000+start)"}, t0)


if __name__ == "__main__":
    main()
