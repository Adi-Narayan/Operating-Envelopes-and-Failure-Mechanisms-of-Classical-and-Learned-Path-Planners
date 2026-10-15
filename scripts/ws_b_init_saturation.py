"""
WS-B follow-up (Adi's review, item 1): is the TD3 actor's pre-tanh output saturated at initialisation,
before any gradient update, or does saturation develop during training?

Set-up identical to experiments/train_arc_seed.py (TD3, moving obstacles, seed k, BASELINE_TRAIN_CONFIG).
The first 64 transitions in the replay buffer (states visited by the untrained actor plus exploration
noise) are frozen as a probe batch, read directly from the buffer so that no random number is consumed.
Before update number 0, 1, 2, 5, 10, 20, 50, 100, 200, 500 and 1000 the pre-tanh output u = actor.pre(h)
is evaluated on that probe batch without gradient: E[u^2], fraction |u| > 3, mean |u|. Training then
continues unchanged for one episode.

    python scripts/ws_b_init_saturation.py <seed>   -> results/v2/ws_b/init_saturation/seed<k>.json
"""
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import EXP, ROOT, dump, now, outdir, write_manifest  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "arc_2d", "code"))
import torch  # noqa: E402

PROBE_AT = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def main():
    seed = int(sys.argv[1])
    t0 = now()
    d = outdir("ws_b", "init_saturation")
    work = os.path.join(d, f"work_seed{seed}")
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    import config
    config.MODEL_CONFIG["save_dir"] = os.path.join(work, "models")
    config.TB_CONFIG["log_dir"] = os.path.join(work, "runs")
    from baselines import BASELINE_TRAIN_CONFIG, ALGOS
    config.TRAIN_CONFIG.update(BASELINE_TRAIN_CONFIG)
    from train import ARCTrainer
    from replay_buffer import ReplayBuffer
    trainer = ARCTrainer(env_name="corridor_gauntlet", device=torch.device("cpu"), collect_metrics=True)
    agent_cfg = {**config.NETWORK_CONFIG, **config.TRAIN_CONFIG}
    trainer.agent = ALGOS["td3"](trainer.obs_dim, trainer.action_dim, agent_cfg, torch.device("cpu"))
    trainer.replay_buffer = ReplayBuffer(capacity=config.TRAIN_CONFIG["memory_capacity"], obs_dim=trainer.obs_dim,
                                         history_length=config.NETWORK_CONFIG["history_length"], device=torch.device("cpu"))
    from dyn_train_env import make_dynamic
    make_dynamic(trainer.env)

    agent, buf = trainer.agent, trainer.replay_buffer
    rows, state = [], {"n": 0, "probe": None}
    orig_update = agent.update

    def probe_stats():
        with torch.no_grad():
            u = agent.actor.pre(state["probe"]).squeeze(-1)
        return {"E_u2": float(u.pow(2).mean()), "frac_abs_u_gt3": float((u.abs() > 3).float().mean()),
                "mean_abs_u": float(u.abs().mean()), "max_abs_u": float(u.abs().max())}

    def update(*a, **k):
        if state["probe"] is None:
            hist = np.stack([np.asarray(e["history"], np.float32) for e in list(buf.buffer)[:64]])
            state["probe"] = torch.as_tensor(hist)
        if state["n"] in PROBE_AT:
            rows.append({"update": state["n"], **probe_stats()})
            print(rows[-1], flush=True)
        state["n"] += 1
        return orig_update(*a, **k)

    agent.update = update
    t = time.time()
    trainer.train(num_episodes=1)
    res = {"seed": seed, "algo": "td3", "dynamic": True, "probe": "first 64 buffer transitions (untrained actor + exploration noise)",
           "rows": rows, "updates_in_episode_0": state["n"], "wall_s": time.time() - t}
    dump(d, f"seed{seed}.json", res)
    write_manifest(d, f"scripts/ws_b_init_saturation.py {seed}", {"config": "BASELINE_TRAIN_CONFIG, TD3, moving obstacles"},
                   {"seed": seed}, t0, name=f"manifest_seed{seed}.json")


if __name__ == "__main__":
    main()
