"""
Shared helpers for the graphical abstract and the simulation video (scripts/make_graphical_abstract.py,
scripts/make_video.py): obstacle drawing, planner runs that return time-stamped paths, and the colours.
"""
import json
import os
import sys

import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2common import EXP, ROOT  # noqa: E402,F401
import harness as H              # noqa: E402

MEDIA = os.path.join(ROOT, "results", "v2", "media")
COL = {"APF": "#1f4e9c", "APF-D": "#2a9d8f", "DWA-CV": "#e07b00", "SAC": "#c0392b", "Arbiter": "#6a3d9a",
       "obst": "#9aa5b1", "obst_edge": "#5f6b78", "outlier": "#c9d3dd", "goal": "#1a9850", "start": "#222222"}


def draw_obstacles(ax, obstacles, alpha=1.0, z=1):
    arts = []
    for o in obstacles:
        fc = COL["outlier"] if o.get("outlier") else COL["obst"]
        kw = dict(facecolor=fc, edgecolor=COL["obst_edge"], lw=0.6, alpha=alpha, zorder=z)
        t, p = o["type"], o["params"]
        if t == "circle":
            a = Circle((p[0], p[1]), p[2], **kw)
        elif t in ("rectangle", "square"):
            a = Rectangle((p[0] - p[2], p[1] - p[3]), 2 * p[2], 2 * p[3], **kw)
        else:
            a = Polygon(np.asarray(p), closed=True, **kw)
        ax.add_patch(a)
        arts.append(a)
    return arts


def arena(ax, ticks=False):
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_aspect("equal")
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#888888")
        s.set_linewidth(0.6)


def mark_start_goal(ax, size=1.0):
    ax.add_patch(Circle(tuple(H.GOAL), H.GOAL_RADIUS, facecolor="none", edgecolor=COL["goal"], lw=1.0 * size,
                        ls="--", zorder=5))
    ax.plot(*H.START, "o", color=COL["start"], ms=4 * size, zorder=6)
    ax.plot(*H.GOAL, "*", color=COL["goal"], ms=9 * size, zorder=6)


def numbers():
    return json.load(open(os.path.join(ROOT, "results", "numbers.json")))
