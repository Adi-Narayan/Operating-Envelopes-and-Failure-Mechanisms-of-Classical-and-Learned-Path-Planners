"""
Shared evaluation harness for the APF-vs-ARC comparisons.

One world model is used for both planners:
  * geometry   : the Corridor Gauntlet layout (dynamic_apf.STATIC_OBSTACLES,
                 identical to static_apf env 1 and arc_2d 'corridor_gauntlet')
  * motion     : the APF dynamic motion model (dynamic_apf.DynObstacle), integrated
                 on a fixed 1/30 s grid and shared through an ObstacleTimeline, so
                 APF (0.1 s per step) and ARC (1/6 s per step) see the identical
                 obstacle motion sequence
  * speed      : both agents move at 18 m/s (APF 1.8 m per 0.1 s, ARC 3.0 m per 1/6 s)
  * collision  : exact clearance to any obstacle boundary < 5 m (ARC's d_min_obstacle)
  * goal       : within 10 m of the goal (ARC's d_min_target)
  * budget     : 10.8 km of travel (APF 6000 steps, ARC 3600 steps)
  * metrics    : L = arc length of the executed path, efficiency = d_straight / L
                 with d_straight = |goal - start| = 608.1 m for every run
"""
import copy
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("dynamic_apf", "static_apf", os.path.join("arc_2d", "code")):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

_cwd = os.getcwd()
os.chdir(os.path.join(ROOT, "experiments"))    # apf.py / static_apf.py create ./results on import
import apf as dyn                                # noqa: E402
import static_apf as sta                         # noqa: E402
os.chdir(_cwd)

START = np.array([35.0, 35.0])
GOAL = np.array([465.0, 465.0])
D_STRAIGHT = float(np.linalg.norm(GOAL - START))
GOAL_RADIUS = 10.0
COLLISION_RADIUS = 5.0
GROUND_SPEED = 18.0                  # m/s, both planners
APF_DT = 0.1                         # s per APF step (1.8 m)
ARC_STEP = 3.0                       # m per ARC step
ARC_DT = ARC_STEP / GROUND_SPEED     # 1/6 s per ARC step
TIMELINE_DT = 1.0 / 30.0             # common integration grid (0.1 = 3 ticks, 1/6 = 5 ticks)


def set_ground_speed(v):
    """Both planners move at v m/s. The obstacle grid scales with 1/v so that the
    number of integration ticks per planner step (APF 3, ARC 5) stays fixed."""
    global GROUND_SPEED, APF_DT, ARC_DT, TIMELINE_DT
    GROUND_SPEED = float(v)
    APF_DT = 1.8 / GROUND_SPEED
    ARC_DT = ARC_STEP / GROUND_SPEED
    TIMELINE_DT = APF_DT / 3.0


MAX_DISTANCE = 10800.0
APF_MAX_STEPS = int(round(MAX_DISTANCE / 1.8))    # 6000
ARC_MAX_STEPS = int(round(MAX_DISTANCE / ARC_STEP))  # 3600

OBSTACLES = dyn.STATIC_OBSTACLES   # Corridor Gauntlet (APF custom environment)


def _urban():
    from config import make_triangle
    # arc_2d config 'env_apf_urban' (ARC's original training layout) in APF obstacle format
    return [
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


LAYOUTS = {"corridor": OBSTACLES, "urban": _urban()}


def clearance(q, obstacles):
    return min(dyn.get_distance(np.asarray(q, float), o)[0] for o in obstacles)


class StaticTimeline:
    def __init__(self, obstacles=OBSTACLES):
        self.obstacles = obstacles

    def at(self, t):
        return self.obstacles


class ObstacleTimeline:
    """Obstacle states on a fixed time grid, generated lazily from one seed."""

    def __init__(self, seed, obstacles=OBSTACLES, dt=None):
        self.seed = seed
        self.dt = TIMELINE_DT if dt is None else dt
        rng = np.random.default_rng(seed)
        self.dyns, self.vel_table = dyn.assign_velocities(obstacles, rng, verbose=False)
        self.states = [[d.obs for d in self.dyns]]
        self.gen_time = 0.0   # seconds spent integrating obstacles; excluded from planner timings

    def _advance(self):
        for d in self.dyns:
            d.step(self.dt, goal=GOAL)
        n = len(self.dyns)
        for i in range(n):
            for j in range(i + 1, n):
                self.dyns[i].handle_obs_collision(self.dyns[j])
        # _build_obs creates new dicts, so these references are immutable snapshots
        self.states.append([d.obs for d in self.dyns])

    def at(self, t):
        k = int(round(t / self.dt))
        if len(self.states) <= k:
            g0 = time.perf_counter()
            while len(self.states) <= k:
                self._advance()
            self.gen_time += time.perf_counter() - g0
        return self.states[k]


def _gen_time(timeline):
    return getattr(timeline, "gen_time", 0.0)


def _summ(path, outcome, t0, info=None, gen_s=0.0):
    path = np.asarray(path, float)
    L = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
    steps = len(path) - 1
    ms = (time.perf_counter() - t0 - gen_s) * 1000.0
    out = {
        "outcome": outcome,
        "success": outcome == "success",
        "steps": steps,
        "path_length": L,
        "efficiency": 100.0 * D_STRAIGHT / L if L > 0 else float("nan"),
        "time_ms": ms,
        "ms_per_step": ms / max(steps, 1),
        "final_dist_to_goal": float(np.linalg.norm(path[-1] - GOAL)),
        "path": path,
    }
    if info:
        out.update({k: v for k, v in info.items() if k != "outcome"})
    return out


# ---------------------------------------------------------------- APF
def run_apf(timeline, planner_seed, cfg=None, max_steps=APF_MAX_STEPS):
    cfg = cfg or dyn.APFConfig()
    assert abs(cfg.goal_tol - GOAL_RADIUS) < 1e-9
    rng = np.random.default_rng(planner_seed)
    g0 = _gen_time(timeline)
    t0 = time.perf_counter()
    ok, path, _, _, info = dyn.simulate_dynamic_apf(
        START.copy(), GOAL.copy(), None, cfg, dt=APF_DT, rng=rng,
        timeline=timeline, max_steps=max_steps, record_history=False)
    return _summ(path, info["outcome"], t0, info, _gen_time(timeline) - g0)


# ---------------------------------------------------------------- ARC
def _to_shapely(obstacles):
    from shapely.geometry import Point, Polygon, box
    geoms = []
    for o in obstacles:
        t = o["type"]
        if t == "circle":
            cx, cy, r = o["params"]
            geoms.append(Point(cx, cy).buffer(r))
        elif t in ("rectangle", "square"):
            cx, cy, hw, hh = o["params"]
            geoms.append(box(cx - hw, cy - hh, cx + hw, cy + hh))
        else:
            geoms.append(Polygon(o["params"]))
    return geoms


class ARCPlanner:
    """Wraps a trained 2D ARC model; observations/kinematics follow UAVEnvironment exactly."""

    def __init__(self, model_path):
        import torch
        from config import ENV_CONFIG, ENVIRONMENTS, NETWORK_CONFIG, TRAIN_CONFIG
        from environment import UAVEnvironment
        from networks import ARCAgent
        from replay_buffer import ObservationHistory
        torch.set_num_threads(1)
        self.torch = torch
        self.ObservationHistory = ObservationHistory
        self.hist_len = NETWORK_CONFIG["history_length"]
        self.obs_dim = 3 + ENV_CONFIG["num_sensors"]
        self.env = UAVEnvironment(ENVIRONMENTS["corridor_gauntlet"], ENV_CONFIG)
        self.speed = ENV_CONFIG["uav_speed"] * ENV_CONFIG["sampling_interval"]  # 3.0 m per step
        assert abs(self.speed - ARC_STEP) < 1e-9
        from baselines import load_agent
        agent = load_agent(model_path)            # TD3 / SAC checkpoints carry an "algo" key
        if agent is None:
            agent = ARCAgent(self.obs_dim, 1, {**NETWORK_CONFIG, **TRAIN_CONFIG}, torch.device("cpu"))
            agent.load(model_path)
            agent.policy_net.eval()
        self.agent = agent
        self.algo = getattr(agent, "algo", None) or "arc"
        self.model_path = model_path
        self._static_cache = None

    def _geoms(self, obstacles, static):
        if static:
            if self._static_cache is None:
                self._static_cache = _to_shapely(obstacles)
            return self._static_cache
        return _to_shapely(obstacles)

    def run(self, timeline, heading0=None, max_steps=ARC_MAX_STEPS, static=False,
            range_noise=0.0, heading_noise=0.0, noise_seed=0):
        """range_noise: SD (m) of Gaussian noise on each ray reading that hits an obstacle;
        heading_noise: SD (rad) of Gaussian noise on the executed heading. Collisions are
        judged on the true state (exact clearance and the noise-free ray readings)."""
        env = self.env
        nrng = np.random.default_rng(noise_seed)

        def perceive(ob):
            if range_noise <= 0:
                return ob
            ob = ob.copy()
            d = ob[3:] * env.sensor_range
            hit = d < env.sensor_range - 1e-6
            d[hit] = np.clip(d[hit] + nrng.normal(0.0, range_noise, size=int(hit.sum())), 0.1, env.sensor_range)
            ob[3:] = d / env.sensor_range
            return ob

        g0 = _gen_time(timeline)
        t0 = time.perf_counter()
        env.uav_pos = START.astype(np.float32).copy()
        env.target_pos = GOAL.astype(np.float32).copy()
        g = GOAL - START
        env.uav_heading = float(math.atan2(g[1], g[0]) if heading0 is None else heading0)
        env.obstacles = self._geoms(timeline.at(0.0), static)
        hist = self.ObservationHistory(self.hist_len, self.obs_dim)
        hist.reset(perceive(env._get_observation()))
        path = [env.uav_pos.astype(float).copy()]
        min_c = clearance(env.uav_pos, timeline.at(0.0))
        outcome = "timeout"
        for k in range(max_steps):
            a, _ = self.agent.select_action(hist.get_tensor(self.agent.device), None, noise_scale=0.0)
            turn = float(a[0]) * env.action_range
            if heading_noise > 0:
                turn += nrng.normal(0.0, heading_noise)     # actuation noise on the executed heading
            env.uav_heading = env._normalize_angle(env.uav_heading + turn)
            env.uav_pos[0] += self.speed * np.cos(env.uav_heading)
            env.uav_pos[1] += self.speed * np.sin(env.uav_heading)
            path.append(env.uav_pos.astype(float).copy())
            obs_now = timeline.at((k + 1) * ARC_DT)
            env.obstacles = self._geoms(obs_now, static)
            ob = env._get_observation()
            hist.append(perceive(ob))
            p = env.uav_pos
            if p[0] < 0 or p[0] > env.width or p[1] < 0 or p[1] > env.height:
                outcome = "out_of_bounds"
                break
            c = clearance(p, obs_now)
            min_c = min(min_c, c)
            native_hit = bool(np.any(ob[3:] * env.sensor_range < env.d_min_obstacle))
            if c < COLLISION_RADIUS or native_hit:
                outcome = "collision"
                break
            if np.linalg.norm(p - GOAL) < GOAL_RADIUS:
                outcome = "success"
                break
        return _summ(path, outcome, t0, {"min_clearance": min_c}, _gen_time(timeline) - g0)


def mean_sd(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
