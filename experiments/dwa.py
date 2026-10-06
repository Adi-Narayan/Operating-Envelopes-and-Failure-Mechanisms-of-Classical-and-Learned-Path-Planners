"""
Dynamic Window Approach (Fox, Burgard and Thrun, 1997) on the harness protocol: the
velocity-space classical baseline for the dynamic suite.

Agent      unicycle with state (q, psi, v, w); speed 0 <= v <= v_max = the ground speed of
           the run (18 m/s in the paired suite), so DWA never moves faster than APF or the
           learned planners, but unlike them it may slow down and stop.
Limits     linear acceleration A_MAX, turn rate W_MAX, angular acceleration ALPHA_MAX. All
           limits and the rollout horizon scale with the ground speed so that the planner's
           geometry (minimum turning radius 11.5 m, braking distance 18 m, look-ahead 54 m)
           is the same at every speed, as APF's 1.8 m step is.
Control    one decision per 1.8 m of travel at v_max (0.1 s at 18 m/s, APF's rate). Each
           decision samples N_V x N_W velocity pairs over the dynamic window, rolls each out
           at constant (v, w) for the horizon, discards inadmissible ones (rollout clearance
           below the 6 m margin APF also uses, a speed that cannot brake before the nearest
           obstacle, or leaving the arena) and maximises
               G = alpha * heading + beta * clearance + gamma * velocity
           with each term normalised over the admissible set, as in Fox et al. If no pair is
           admissible, the pair with the largest rollout clearance is executed.
Sensing    APF's: exact geometry of every obstacle at the current time. With predict=True
           (DWA-CV) every obstacle is also translated along the rollout with its velocity,
           estimated from its displacement over the last control period (constant-velocity
           prediction; no knowledge of future bounces).
Evaluation harness protocol: exact clearance < 5 m after a control step is a collision,
           within 10 m of the goal is a success, the time budget equals APF's (6000 control
           periods, 600 s at 18 m/s), shared obstacle timeline. Noise as apf_variants.run:
           one Gaussian sample per obstacle per decision on every perceived distance, and a
           Gaussian rotation of the executed displacement.
"""
import math
import time

import numpy as np

import harness as H

dyn = H.dyn
V_REF = 18.0                         # m/s at which the limits below are stated
W_MAX_REF = math.pi / 2              # 90 deg/s  -> minimum turning radius 18 / (pi/2) = 11.5 m
A_MAX_REF = 9.0                      # m/s^2     -> braking distance 18^2 / (2 * 9) = 18 m
ALPHA_MAX_REF = 2 * math.pi          # rad/s^2
HORIZON_REF = 3.0                    # s         -> 54 m look-ahead at 18 m/s
MARGIN = 6.0                         # m, APF's collision margin
CLEAR_CAP = 30.0                     # m, clearance term saturates here
N_V, N_W = 7, 21
WEIGHTS = {"alpha": 1.0, "beta": 0.5, "gamma": 0.2}   # replaced by the values selected in dwa_tune.py


# ------------------------------------------------------------------ vectorised distance kernels
def _dist_circle(P, o):
    cx, cy, r = o["params"]
    return np.maximum(np.hypot(P[..., 0] - cx, P[..., 1] - cy) - r, 0.0)


def _dist_box(P, o):
    cx, cy, hw, hh = o["params"]
    dx = np.maximum(np.abs(P[..., 0] - cx) - hw, 0.0)
    dy = np.maximum(np.abs(P[..., 1] - cy) - hh, 0.0)
    return np.hypot(dx, dy)


def _dist_tri(P, o):
    V = np.asarray(o["params"], float)
    best = None
    sign = []
    for i in range(3):
        a, b = V[i], V[(i + 1) % 3]
        ab = b - a
        ap = P - a
        t = np.clip((ap[..., 0] * ab[0] + ap[..., 1] * ab[1]) / (ab @ ab), 0.0, 1.0)
        d = np.hypot(ap[..., 0] - t * ab[0], ap[..., 1] - t * ab[1])
        best = d if best is None else np.minimum(best, d)
        sign.append(ab[0] * ap[..., 1] - ab[1] * ap[..., 0])
    s = np.stack(sign)
    inside = np.all(s >= 0, axis=0) | np.all(s <= 0, axis=0)
    return np.where(inside, 0.0, best)


def dist(P, o):
    """Distance from points P[..., 2] to the boundary of obstacle o (0 inside), as dyn.get_distance."""
    t = o["type"]
    if t == "circle":
        return _dist_circle(P, o)
    if t in ("rectangle", "square"):
        return _dist_box(P, o)
    return _dist_tri(P, o)


def _centroid(o):
    return dyn.obs_centroid(o)


# ------------------------------------------------------------------ planner
def run(timeline, predict=False, weights=None, range_noise=0.0, heading_noise=0.0, noise_seed=0,
        max_periods=None, record=False):
    w8 = {**WEIGHTS, **(weights or {})}
    v_max = H.GROUND_SPEED
    s = v_max / V_REF                                    # limits scale with the ground speed
    w_max, a_max, al_max = W_MAX_REF * s, A_MAX_REF * s * s, ALPHA_MAX_REF * s * s
    dt = 1.8 / v_max                                     # = H.APF_DT
    n_h = int(round(HORIZON_REF / (dt * s)))             # rollout steps (30)
    max_periods = max_periods or H.APF_MAX_STEPS
    lo, hi = np.array([dyn.BOUNDS[0], dyn.BOUNDS[2]]), np.array([dyn.BOUNDS[1], dyn.BOUNDS[3]])
    nrng = np.random.default_rng(noise_seed)
    goal = H.GOAL.copy()
    q = H.START.copy()
    psi = math.atan2(goal[1] - q[1], goal[0] - q[0])
    v, w = 0.0, 0.0
    g0 = H._gen_time(timeline)
    t0 = time.perf_counter()
    path = [q.copy()]
    speeds = []
    info = {"min_clearance": H.clearance(q, timeline.at(0.0)), "no_admissible": 0}
    outcome = "timeout"
    tau = (np.arange(1, n_h + 1) * dt)                   # rollout times
    prev = None
    for k in range(max_periods):
        if np.linalg.norm(q - goal) <= H.GOAL_RADIUS:
            outcome = "success"
            break
        obs = timeline.at(k * dt)
        # ---- dynamic window
        vs = np.linspace(max(0.0, v - a_max * dt), min(v_max, v + a_max * dt), N_V)
        ws = np.linspace(max(-w_max, w - al_max * dt), min(w_max, w + al_max * dt), N_W)
        VV, WW = np.meshgrid(vs, ws, indexing="ij")
        VV, WW = VV.ravel(), WW.ravel()                  # (C,)
        # ---- rollouts at constant (v, w): psi_j = psi + w * tau_j, Euler position update
        ang = psi + WW[:, None] * tau[None, :]           # (C, n_h)
        step = (VV * dt)[:, None]
        X = q[0] + np.cumsum(step * np.cos(ang), axis=1)
        Y = q[1] + np.cumsum(step * np.sin(ang), axis=1)
        P = np.stack([X, Y], axis=-1)                    # (C, n_h, 2)
        # ---- clearance along each rollout
        eps = nrng.normal(0.0, range_noise, size=len(obs)) if range_noise > 0 else np.zeros(len(obs))
        clr = np.full(P.shape[:2], np.inf)
        for i, o in enumerate(obs):
            Q = P
            if predict and prev is not None:
                vel = (_centroid(o) - _centroid(prev[i])) / dt
                Q = P - vel[None, None, :] * tau[None, :, None]
            clr = np.minimum(clr, np.maximum(dist(Q, o) + eps[i], 0.0))
        cmin = clr.min(axis=1)                            # (C,)
        inb = np.all((P >= lo) & (P <= hi), axis=(1, 2))
        brake_ok = VV <= np.sqrt(2.0 * a_max * np.maximum(cmin - MARGIN, 0.0)) + 1e-9
        adm = (cmin > MARGIN) & inb & brake_ok
        if adm.any():
            end = P[:, -1, :]
            to_goal = np.arctan2(goal[1] - end[:, 1], goal[0] - end[:, 0])
            dpsi = np.abs(np.arctan2(np.sin(to_goal - ang[:, -1]), np.cos(to_goal - ang[:, -1])))
            terms = []
            for x in (math.pi - dpsi, np.minimum(cmin, CLEAR_CAP), VV):
                xa = x[adm]
                rng_ = xa.max() - xa.min()
                terms.append((x - xa.min()) / rng_ if rng_ > 1e-12 else np.zeros_like(x))
            G = w8["alpha"] * terms[0] + w8["beta"] * terms[1] + w8["gamma"] * terms[2]
            G[~adm] = -np.inf
            j = int(np.argmax(G))
        else:
            info["no_admissible"] += 1
            j = int(np.argmax(np.where(inb, cmin, -np.inf)))
        v, w = float(VV[j]), float(WW[j])
        prev = obs
        # ---- execute one control period
        psi_n = psi + w * dt
        d = v * dt * np.array([math.cos(psi_n), math.sin(psi_n)])
        if heading_noise > 0 and v > 0:
            a = nrng.normal(0.0, heading_noise)
            c_, s_ = math.cos(a), math.sin(a)
            d = np.array([c_ * d[0] - s_ * d[1], s_ * d[0] + c_ * d[1]])
            psi_n += a
        psi = math.atan2(math.sin(psi_n), math.cos(psi_n))
        q = np.clip(q + d, lo, hi)
        path.append(q.copy())
        speeds.append(v)
        c = H.clearance(q, timeline.at((k + 1) * dt))
        info["min_clearance"] = min(info["min_clearance"], c)
        if c < H.COLLISION_RADIUS:
            outcome = "collision"
            break
    info["mean_speed"] = float(np.mean(speeds)) if speeds else 0.0
    info["time_s"] = len(speeds) * dt
    out = H._summ(path, outcome, t0, info, H._gen_time(timeline) - g0)
    out["periods"] = len(speeds)
    if not record:
        out.pop("path")
    return out
