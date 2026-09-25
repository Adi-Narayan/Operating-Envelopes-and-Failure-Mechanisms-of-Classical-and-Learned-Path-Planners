"""
Alternative head-to-head on ARC's ORIGINAL training layout (arc_2d config 'env_apf_urban'),
using the ORIGINAL trained model from models.zip, under the corrected protocol.

APF needs no training, so running it on ARC's layout is the cheapest way to make the
static comparison like-for-like while keeping the original ARC model.

    python urban_head_to_head.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H   # noqa: E402
from config import make_triangle  # noqa: E402

# env_apf_urban in APF obstacle format (same geometry as arc_2d/code/config.py)
URBAN = [
    {"type": "rectangle", "params": (150.0, 300.0, 30.0, 75.0), "label": "B1", "outlier": False},
    {"type": "rectangle", "params": (400.0, 250.0, 30.0, 75.0), "label": "B2", "outlier": False},
    {"type": "square", "params": (250.0, 130.0, 30.0, 30.0), "label": "B3", "outlier": False},
    {"type": "square", "params": (250.0, 370.0, 30.0, 30.0), "label": "B4", "outlier": False},
    {"type": "triangle", "params": make_triangle(150, 110, 32.5, 0.0), "label": "V1", "outlier": False},
    {"type": "triangle", "params": make_triangle(250, 250, 32.5, 0.5), "label": "V2", "outlier": False},
    {"type": "triangle", "params": make_triangle(360, 390, 32.5, 1.0), "label": "V3", "outlier": False},
    {"type": "circle", "params": (400.0, 140.0, 25.0), "label": "C1", "outlier": False},
    {"type": "circle", "params": (40.0, 460.0, 32.5), "label": "OL1", "outlier": True},
    {"type": "circle", "params": (460.0, 40.0, 32.5), "label": "OL2", "outlier": True},
]


def strip(r):
    return {k: v for k, v in r.items() if k != "path"}


if __name__ == "__main__":
    model = os.path.join(H.ROOT, "arc_2d", "models_original", "arc_best.pth")
    # geometry check against the ARC config definition
    from config import ENVIRONMENTS
    ref = H._to_shapely(URBAN)
    from environment import UAVEnvironment
    from config import ENV_CONFIG
    arc_geo = UAVEnvironment(ENVIRONMENTS["env_apf_urban"], ENV_CONFIG).obstacles
    symdiff = [round(a.symmetric_difference(b).area, 3) for a, b in zip(ref, arc_geo)]

    tl = H.StaticTimeline(URBAN)
    apf = [H.run_apf(tl, s) for s in range(10)]
    P = H.ARCPlanner(model)
    arc_det = P.run(tl, static=True)                              # heading toward goal
    rng = np.random.default_rng(0)
    heads = rng.uniform(-math.pi, math.pi, size=10)                # original tester's random heading
    arc_rand = [P.run(tl, heading0=float(h), static=True) for h in heads]

    def agg(rows):
        ok = [r for r in rows if r["success"]]
        return {"n": len(rows), "success": len(ok), "outcomes": [r["outcome"] for r in rows],
                "steps": H.mean_sd([r["steps"] for r in ok]), "L": H.mean_sd([r["path_length"] for r in ok]),
                "eff": H.mean_sd([r["efficiency"] for r in ok])}

    res = {"geometry_symdiff_vs_config": symdiff, "apf": agg(apf), "arc_heading_to_goal": strip(arc_det),
           "arc_random_heading": agg(arc_rand), "arc_random_heading_runs": [strip(r) for r in arc_rand],
           "headings_rad": heads.tolist()}
    np.savez_compressed(os.path.join(H.ROOT, "experiments", "out", "urban_paths.npz"),
                        apf=apf[0]["path"], arc=arc_det["path"])
    with open(os.path.join(H.ROOT, "experiments", "out", "urban_head_to_head.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("arc_random_heading_runs", "headings_rad")},
                     indent=1, default=float))
