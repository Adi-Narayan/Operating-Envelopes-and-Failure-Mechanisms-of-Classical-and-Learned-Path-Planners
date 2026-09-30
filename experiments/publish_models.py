"""
Prepare the TD3/SAC model folders for publication.

For every run out/arc_models/{td3,td3dyn,sacdyn}_seed<N>/:
  * copy the full checkpoints (actor + critics + targets + optimisers) to <FULL>/<run>/models/
    (outside the repository; nothing is deleted without a copy);
  * keep in the repository only the checkpoints the paper evaluates -- arc_best.pth,
    arc_checkpoint_ep300.pth and the oracle named in out/oracle_checkpoints.json --
    rewritten as actor-only files (baselines.export_actor);
  * check that each actor-only file reproduces the full checkpoint's actions exactly.

    python publish_models.py --full ../../checkpoints_full
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baselines import export_actor, load_agent  # noqa: E402

RUN = re.compile(r"^(td3|td3dyn|sacdyn)_seed\d+$")


def same_actions(a, b, n=64):
    torch.manual_seed(0)
    h = torch.rand(n, 5, 12)
    with torch.no_grad():
        return all(torch.equal(torch.as_tensor(a.select_action(h[i], None, 0.0)[0]),
                               torch.as_tensor(b.select_action(h[i], None, 0.0)[0])) for i in range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="where to keep the full checkpoints (outside the repository)")
    a = ap.parse_args()
    oracle = {k: os.path.basename(v) for k, v in json.load(open(os.path.join(HERE, "out", "oracle_checkpoints.json"))).items()}
    root = os.path.join(HERE, "out", "arc_models")
    for run in sorted(d for d in os.listdir(root) if RUN.match(d)):
        mdir = os.path.join(root, run, "models")
        full = os.path.join(a.full, run, "models")
        os.makedirs(full, exist_ok=True)
        for f in glob.glob(os.path.join(mdir, "*.pth")):
            dst = os.path.join(full, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
        keep = {"arc_best.pth", "arc_checkpoint_ep300.pth", oracle[run]}
        for f in sorted(glob.glob(os.path.join(mdir, "*.pth"))):
            name = os.path.basename(f)
            src_full = os.path.join(full, name)
            assert os.path.getsize(src_full) == os.path.getsize(f) or torch.load(f, weights_only=True).get("actor_only")
            if name in keep:
                tmp = f + ".tmp"
                export_actor(src_full, tmp)
                assert same_actions(load_agent(src_full), load_agent(tmp)), f"{run}/{name}: actions differ"
                os.replace(tmp, f)
            else:
                os.remove(f)                    # a copy is in <FULL>
        print(run, "kept", sorted(keep), flush=True)


if __name__ == "__main__":
    main()
