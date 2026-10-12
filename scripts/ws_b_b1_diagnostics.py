"""
WS-B, B1: actor saturation and critic values on existing checkpoints (no training).

For every checkpoint of a learner group (TD3 static, TD3 moving, SAC moving: all saved 50-episode
checkpoints and arc_best, from the full checkpoints; ARC: the checkpoints still stored, i.e.
arc_best, the best-evaluating checkpoint and episode 300, because ARC's other checkpoints were not
kept), the deterministic policy is rolled out from 20 exploring starts (np.random.seed(0-19)) in its
own training environment (static custom environment, or the moving-obstacle training environment).
Over every visited state:
  * pre-tanh actor output u: mean |u|, fraction with |u| > 3, fraction of actions at the bound (|a| >= 0.999);
  * critic values (TD3, SAC: min of the twin critics; ARC: its critic) at the bound (max over a = +1, -1),
    at zero action and at the policy's action, against the Monte Carlo return-to-go G_t of the same
    rollout with the learner's own discount (TD3/SAC 0.99, ARC 0.9).

    python scripts/ws_b_b1_diagnostics.py <td3|td3dyn|sacdyn|arc>  -> results/v2/ws_b/b1_diagnostics/<group>.json
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import EXP, FULL_CKPT, OUT, dump, now, outdir, write_manifest  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(EXP), "arc_2d", "code"))
import torch                                                                 # noqa: E402
torch.set_num_threads(1)

N_ROLL = 20


def checkpoints(group):
    if group == "arc":
        rows = []
        for s in range(5):
            for f in sorted(glob.glob(os.path.join(OUT, "arc_models", f"seed{s}", "models", "*.pth"))):
                rows.append((f"seed{s}", f))
        return rows
    rows = []
    for s in range(10):
        for f in sorted(glob.glob(os.path.join(FULL_CKPT, f"{group}_seed{s}", "models", "*.pth"))):
            rows.append((f"{group}_seed{s}", f))
    return rows


def episode_of(path):
    m = re.search(r"ep(\d+)\.pth$", path)
    return int(m.group(1)) if m else None


class Wrapped:
    """Uniform access: pre-tanh output u, policy action, and a critic value for TD3, SAC and ARC."""

    def __init__(self, group, path):
        self.group = group
        if group == "arc":
            from config import NETWORK_CONFIG, TRAIN_CONFIG
            from networks import ARCAgent
            self.agent = ARCAgent(12, 1, {**NETWORK_CONFIG, **TRAIN_CONFIG}, torch.device("cpu"))
            self.agent.load(path)
            self.net = self.agent.policy_net.eval()
            self.gamma = 0.9
        else:
            from baselines import load_agent
            self.agent = load_agent(path)
            self.gamma = 0.99

    @torch.no_grad()
    def pre(self, H):
        if self.group == "arc":
            z, _ = self.net.get_lstm_output(H)
            a = self.net.actor
            return a.fc3(torch.relu(a.fc2(torch.relu(a.fc1(z)))))
        if self.agent.algo == "sac":
            return self.agent.actor(H)[0]
        return self.agent.actor.pre(H)

    @torch.no_grad()
    def q(self, H, a):
        if self.group == "arc":
            z, _ = self.net.get_lstm_output(H)
            return self.net.critic(z, a)
        return torch.min(self.agent.q1(H, a), self.agent.q2(H, a))

    @torch.no_grad()
    def act(self, h):
        return float(torch.tanh(self.pre(h.unsqueeze(0)))[0, 0])


def make_env(dynamic):
    from config import ENV_CONFIG, ENVIRONMENTS
    from environment import UAVEnvironment
    env = UAVEnvironment(ENVIRONMENTS["corridor_gauntlet"], ENV_CONFIG)
    if dynamic:
        from dyn_train_env import make_dynamic
        make_dynamic(env)
    return env, ENV_CONFIG


def evaluate(W, env, cfg):
    from replay_buffer import ObservationHistory
    Hs, G, outcomes = [], [], []
    for i in range(N_ROLL):
        np.random.seed(i)
        obs = env.reset()
        hist = ObservationHistory(5, 12)
        hist.reset(obs)
        states, rewards, info, done, step = [], [], {}, False, 0
        while not done and step < cfg["max_steps"]:
            h = hist.get_tensor(torch.device("cpu"))
            h = h[0] if h.dim() == 3 else h
            states.append(h.clone())
            a = W.act(h)
            obs, r, done, info = env.step(np.float32(a))
            hist.append(obs)
            rewards.append(float(r))
            step += 1
        g, ret = 0.0, []
        for r in reversed(rewards):
            g = r + W.gamma * g
            ret.append(g)
        G += ret[::-1]
        Hs += states
        p = env.uav_pos
        oob = p[0] < 0 or p[0] > env.width or p[1] < 0 or p[1] > env.height
        outcomes.append("success" if info.get("success") else "collision" if info.get("collision") else
                        "out_of_bounds" if oob else "timeout")
    H = torch.stack(Hs)
    G = np.asarray(G)
    u = W.pre(H)[:, 0].numpy()
    a = np.tanh(u)
    n = len(H)
    qp = W.q(H, torch.tensor(a[:, None], dtype=torch.float32))[:, 0].numpy()
    qpos = W.q(H, torch.ones(n, 1))[:, 0].numpy()
    qneg = W.q(H, -torch.ones(n, 1))[:, 0].numpy()
    q0 = W.q(H, torch.zeros(n, 1))[:, 0].numpy()
    qb = np.maximum(qpos, qneg)
    return {"n_states": int(n), "outcomes": {o: outcomes.count(o) for o in sorted(set(outcomes))},
            "mean_abs_u": float(np.mean(np.abs(u))), "frac_abs_u_gt3": float(np.mean(np.abs(u) > 3)),
            "frac_action_at_bound": float(np.mean(np.abs(a) >= 0.999)),
            "mean_Q_bound": float(qb.mean()), "mean_Q_zero": float(q0.mean()), "mean_Q_policy": float(qp.mean()),
            "frac_Q_bound_gt_Q_zero": float(np.mean(qb > q0)), "mean_G": float(G.mean()),
            "mean_Q_policy_minus_G": float(np.mean(qp - G)),
            "corr_Q_policy_G": float(np.corrcoef(qp, G)[0, 1]) if np.std(qp) > 0 and np.std(G) > 0 else None}


def main():
    group = sys.argv[1]
    t0 = now()
    collapse = json.load(open(os.path.join(OUT, "collapse.json")))
    dynamic = group in ("td3dyn", "sacdyn")
    env, cfg = make_env(dynamic)
    res = {}
    for run, path in checkpoints(group):
        W = Wrapped(group, path)
        r = evaluate(W, env, cfg)
        r.update({"run": run, "checkpoint": os.path.basename(path), "episode": episode_of(path),
                  "collapsed_final": collapse.get(run, {}).get("collapsed")})
        res[f"{run}/{os.path.basename(path)}"] = r
        print(run, os.path.basename(path), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                             if k in ("frac_action_at_bound", "frac_abs_u_gt3", "mean_Q_bound", "mean_Q_zero",
                                                      "mean_Q_policy", "mean_G", "outcomes")}, flush=True)
        d = outdir("ws_b", "b1_diagnostics")
        dump(d, f"{group}.json", res)
    write_manifest(d, f"scripts/ws_b_b1_diagnostics.py {group}",
                   {"group": group, "rollouts_per_checkpoint": N_ROLL, "environment": "moving" if dynamic else "static",
                    "gamma": 0.9 if group == "arc" else 0.99,
                    "checkpoints": "full checkpoints (checkpoints_full) for TD3/SAC; stored arc_best/best/ep300 for ARC"},
                   {"starts": "np.random.seed(0-19)"}, t0)


if __name__ == "__main__":
    main()
