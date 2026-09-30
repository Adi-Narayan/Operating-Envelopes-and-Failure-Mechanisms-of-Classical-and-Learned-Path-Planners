"""
Procedurally generated layouts with the Corridor Gauntlet's primitive mix:
2 rectangles (60 x 150 m, random orientation), 2 squares (60 x 60 m), 3 triangles
(circumradius 32.5 m, random rotation), 1 circle (r = 25 m) and 2 "outlier" circles
(r = 32.5 m, stationary in the dynamic suite), placed uniformly in [40, 460]^2 with
>= 25 m gaps between obstacle bounding circles and >= 70 m clearance from start and goal.
Layout k is generated from seed k (deterministic).
"""
import math

import numpy as np

from harness import GOAL, START

SPECS = (
    [("rectangle", None)] * 2 + [("square", None)] * 2 + [("triangle", None)] * 3
    + [("circle", False), ("circle", True), ("circle", True)]
)
GAP, END_CLEAR = 25.0, 70.0


def _tri(cx, cy, size, ang):
    a = np.deg2rad([90, 210, 330]) + ang
    return [(cx + size * math.cos(t), cy + size * math.sin(t)) for t in a]


def make_layout(seed):
    rng = np.random.default_rng(seed)
    placed, circles = [], []           # circles: (centre, bounding radius) for spacing checks
    for i, (kind, outlier) in enumerate(SPECS):
        for _ in range(5000):
            c = rng.uniform(40, 460, size=2)
            if kind == "rectangle":
                hw, hh = (30.0, 75.0) if rng.random() < 0.5 else (75.0, 30.0)
                R, obs = math.hypot(hw, hh), {"type": "rectangle", "params": (c[0], c[1], hw, hh)}
            elif kind == "square":
                R, obs = math.hypot(30, 30), {"type": "square", "params": (c[0], c[1], 30.0, 30.0)}
            elif kind == "triangle":
                R, obs = 32.5, {"type": "triangle", "params": _tri(c[0], c[1], 32.5, rng.uniform(0, 2 * math.pi))}
            else:
                r = 32.5 if outlier else 25.0
                R, obs = r, {"type": "circle", "params": (c[0], c[1], r)}
            if min(np.linalg.norm(c - START), np.linalg.norm(c - GOAL)) < R + END_CLEAR:
                continue
            if any(np.linalg.norm(c - c2) < R + R2 + GAP for c2, R2 in circles):
                continue
            obs.update({"label": f"{kind[0].upper()}{i}", "outlier": bool(outlier)})
            placed.append(obs)
            circles.append((c, R))
            break
        else:
            raise RuntimeError(f"could not place obstacle {i} for layout seed {seed}")
    return placed
