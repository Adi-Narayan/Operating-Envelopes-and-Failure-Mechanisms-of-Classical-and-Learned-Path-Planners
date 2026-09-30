"""
Train one 2D learned policy on a layout (default: the shared Corridor Gauntlet).

    python train_arc_seed.py --seed 0 [--episodes 300] [--algo arc|td3|sac] [--env corridor_gauntlet]

--algo arc (default) is the unmodified arc_2d/code/train.py pipeline (DDPG + LSTM,
RMSprop, 4000-transition FIFO buffer, batch 32, gamma 0.9). --algo td3 / sac swap in
the agents of baselines.py with standard hyperparameters (baselines.BASELINE_TRAIN_CONFIG)
and keep everything else identical: environment, observation, action, reward,
5 exploring-start rollouts per episode, one update per environment step, episode budget.

Output folder: experiments/out/arc_models/<name>/ where <name> is
  seed<N>                      arc, corridor, 300 episodes (the paper's primary runs)
  <env>_seed<N>                arc on another layout
  <algo>_seed<N>               td3/sac on the corridor
  <algo>dyn_seed<N>            trained with moving obstacles (--dynamic)
  arc<E>ep_seed<N>             arc with a non-default episode budget E
(checkpoints, TensorBoard events, per-episode metrics CSV, train_summary.json).
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "arc_2d", "code"))
sys.path.insert(0, HERE)
# TensorBoard's SummaryWriter tries to import TensorFlow when it is installed; on
# this machine that import is broken (pandas/numpy ABI mismatch) and is not
# needed, so force TensorBoard's built-in stub.
sys.modules["tensorflow"] = None

import numpy as np   # noqa: E402
import torch         # noqa: E402


def run_name(algo, env, seed, episodes, dynamic=False):
    prefix = "" if env == "corridor_gauntlet" else f"{env}_"
    if algo != "arc" or dynamic:
        prefix += f"{algo}{'dyn' if dynamic else ''}_"
    elif episodes != 300:
        prefix += f"arc{episodes}ep_"
    return f"{prefix}seed{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--env", default="corridor_gauntlet")
    ap.add_argument("--algo", default="arc", choices=["arc", "td3", "sac"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dynamic", action="store_true",
                    help="train with moving obstacles (dyn_train_env.py; corridor layout only)")
    ap.add_argument("--name", default=None, help="override the output folder name (e.g. for timing benchmarks)")
    args = ap.parse_args()
    if args.dynamic:
        assert args.env == "corridor_gauntlet"

    out = os.path.join(HERE, "out", "arc_models",
                       args.name or run_name(args.algo, args.env, args.seed, args.episodes, args.dynamic))
    os.makedirs(out, exist_ok=True)
    marker = os.path.join(out, "STARTED")
    if os.path.exists(marker):
        print(f"{out} already started/finished elsewhere; skipping")
        return
    open(marker, "w").close()
    os.chdir(out)

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    import config
    config.MODEL_CONFIG["save_dir"] = os.path.join(out, "models")
    config.TB_CONFIG["log_dir"] = os.path.join(out, "runs")
    if args.algo != "arc":
        from baselines import BASELINE_TRAIN_CONFIG
        config.TRAIN_CONFIG.update(BASELINE_TRAIN_CONFIG)   # same dict object train.py reads
    from train import ARCTrainer

    t0 = time.time()
    device = torch.device(args.device)
    trainer = ARCTrainer(env_name=args.env, device=device, collect_metrics=True)
    if args.algo != "arc":
        from baselines import ALGOS
        from replay_buffer import ReplayBuffer
        agent_cfg = {**config.NETWORK_CONFIG, **config.TRAIN_CONFIG}
        trainer.agent = ALGOS[args.algo](trainer.obs_dim, trainer.action_dim, agent_cfg, device)
        trainer.replay_buffer = ReplayBuffer(capacity=config.TRAIN_CONFIG["memory_capacity"], obs_dim=trainer.obs_dim,
                                             history_length=config.NETWORK_CONFIG["history_length"],
                                             device=device)
    if args.dynamic:
        from dyn_train_env import make_dynamic
        make_dynamic(trainer.env)
    trainer.train(num_episodes=args.episodes)
    wall = time.time() - t0

    with open(os.path.join(out, "train_summary.json"), "w") as f:
        json.dump({
            "seed": args.seed, "env": args.env, "algo": args.algo, "dynamic_training": args.dynamic, "episodes": args.episodes,
            "wall_clock_s": wall, "torch_threads": args.threads, "device": args.device,
            "train_config": {k: v for k, v in config.TRAIN_CONFIG.items() if isinstance(v, (int, float, str))},
            "episode_rewards": [float(x) for x in trainer.episode_rewards],
            "success_rate": [float(x) for x in trainer.success_rate],
            "collision_rate": [float(x) for x in trainer.collision_rate],
            "episode_steps": [float(x) for x in trainer.episode_steps],
            "best_avg_reward": float(trainer.best_avg_reward),
        }, f)
    print(f"DONE {os.path.basename(out)} wall={wall/60:.2f} min")


if __name__ == "__main__":
    main()
