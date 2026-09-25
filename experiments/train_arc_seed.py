"""
Train one 2D ARC policy on the shared Corridor Gauntlet layout.

    python train_arc_seed.py --seed 0 [--episodes 300]

Outputs go to experiments/out/arc_models/seed<N>/ (arc_best.pth, checkpoints,
TensorBoard events, per-episode training metrics CSV, wall-clock time).
Everything except the environment and the seeding is the unmodified
arc_2d/code/train.py pipeline (300 episodes, 5 exploring-start UAVs, DDPG+LSTM).
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "arc_2d", "code"))
# TensorBoard's SummaryWriter tries to import TensorFlow when it is installed; on
# this machine that import is broken (pandas/numpy ABI mismatch) and is not
# needed, so force TensorBoard's built-in stub.
sys.modules["tensorflow"] = None

import numpy as np   # noqa: E402
import torch         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--env", default="corridor_gauntlet")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    name = f"seed{args.seed}" if args.env == "corridor_gauntlet" else f"{args.env}_seed{args.seed}"
    out = os.path.join(HERE, "out", "arc_models", name)
    os.makedirs(out, exist_ok=True)
    marker = os.path.join(out, "STARTED")
    if os.path.exists(marker):
        print(f"seed {args.seed} already started/finished elsewhere; skipping")
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
    from train import ARCTrainer

    t0 = time.time()
    trainer = ARCTrainer(env_name=args.env, device=torch.device("cpu"), collect_metrics=True)
    trainer.train(num_episodes=args.episodes)
    wall = time.time() - t0

    with open(os.path.join(out, "train_summary.json"), "w") as f:
        json.dump({
            "seed": args.seed, "env": args.env, "episodes": args.episodes,
            "wall_clock_s": wall, "torch_threads": args.threads,
            "episode_rewards": [float(x) for x in trainer.episode_rewards],
            "success_rate": [float(x) for x in trainer.success_rate],
            "collision_rate": [float(x) for x in trainer.collision_rate],
            "episode_steps": [float(x) for x in trainer.episode_steps],
            "best_avg_reward": float(trainer.best_avg_reward),
        }, f)
    print(f"DONE seed={args.seed} wall={wall/60:.2f} min")


if __name__ == "__main__":
    main()
