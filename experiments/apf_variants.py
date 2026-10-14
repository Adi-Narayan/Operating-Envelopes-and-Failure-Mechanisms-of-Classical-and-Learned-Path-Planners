"""
APF with the three planner-side asymmetries against ARC switchable one at a time.

  sensing   'full' : exact distance to every obstacle (the paper's APF)
            'range': exact geometry in every direction, but only obstacles within the
                     50 m sensor range (isolates the effect of range from ray discreteness
                     and the forward-only field of view)
            'rays' : only ARC's sensor: 9 rays over the forward half-plane (22.5 deg
                     spacing), 50 m range. Each ray hit is a point obstacle for the
                     repulsive term and for the 6 m margin check; obstacles no ray
                     hits are invisible. APF then has the same obstacle information
                     as ARC (plus the goal position, which ARC also observes).
  turn_cap  None   : the heading may change arbitrarily between steps (the paper's APF)
            True   : heading change limited to ARC's turn rate, 9 deg per 1/6 s =
                     54 deg/s, i.e. the same minimum turning radius at equal speed
                     (5.4 deg per 1.8 m step, 9 deg per 3.0 m step). With the cap on,
                     margin violations are resolved by choosing among headings
                     reachable within the cap (the escape and push-out moves of the
                     unconstrained planner would violate it).
  rate      'apf'  : 1.8 m steps, 10 Hz at 18 m/s (the paper's APF)
            'arc'  : 3.0 m steps, 6 Hz at 18 m/s (ARC's step length and decision rate)

Evaluation is the harness protocol: exact clearance < 5 m after a step is a
collision, success within 10 m of the goal, 10.8 km travel budget, shared obstacle
timeline. With all switches off this reproduces harness.run_apf exactly.
"""
import math
import time

import numpy as np

import harness as H

dyn = H.dyn
RAY_RANGE, N_RAYS = 50.0, 9
ARC_TURN_RATE = (math.pi / 20) / (3.0 / H.GROUND_SPEED)   # rad/s at 18 m/s


def _rays(q, heading, geoms):
    """Hit points of ARC's 9 forward rays (list of (distance, hit_xy))."""
    from shapely.geometry import LineString
    hits = []
    for i in range(N_RAYS):
        ang = heading + (i - (N_RAYS - 1) / 2) * math.pi / (N_RAYS - 1)
        end = (q[0] + RAY_RANGE * math.cos(ang), q[1] + RAY_RANGE * math.sin(ang))
        ray = LineString([tuple(q), end])
        best = None
        for g in geoms:
            if not ray.intersects(g):
                continue
            inter = ray.intersection(g)
            pts = []
            if inter.geom_type == "Point":
                pts = [inter.coords[0]]
            elif hasattr(inter, "geoms"):
                for gg in inter.geoms:
                    pts += list(gg.coords)
            else:
                pts = list(inter.coords)
            for p in pts:
                d = math.hypot(p[0] - q[0], p[1] - q[1])
                if best is None or d < best[0]:
                    best = (d, np.array(p, float))
        if best is not None:
            hits.append(best)
    return hits


def _rep_points(q, goal, hits, cfg):
    R = dyn.rotation_matrix(cfg.theta)
    rho_goal = np.linalg.norm(q - goal) + 1e-12
    f = np.zeros(2)
    for d, p in hits:
        v = q - p
        n = np.linalg.norm(v)
        grad = v / n if n > 1e-12 else np.zeros(2)
        rho = max(d, 0.8)
        f += R @ (cfg.k_r * (10.0 / rho ** 2) * rho_goal ** cfg.goal_factor_pow * grad)
    return f


def run(timeline, planner_seed, sensing="full", turn_cap=False, rate="apf", max_distance=H.MAX_DISTANCE,
        range_noise=0.0, heading_noise=0.0, noise_seed=None, momentum=None):
    """range_noise: SD (m) of Gaussian noise on every perceived obstacle distance, one sample per
    obstacle (or ray) per step. heading_noise: SD (rad) of Gaussian noise on the executed heading.
    Collision and success are always judged on the true state.
    momentum: heading momentum beta (APF-D) in the turn-capped branch only; None (default) leaves the
    planner unchanged. The desired heading is that of beta * last executed direction + (1 - beta) * force
    direction, then clipped to the cap (post-hoc matched-constraints check, prereg.md 2026-10-14)."""
    d_prev = None
    cfg = dyn.APFConfig()
    nrng = np.random.default_rng(planner_seed + 7_000_000 if noise_seed is None else noise_seed)
    R_theta = dyn.rotation_matrix(cfg.theta)
    step = 1.8 if rate == "apf" else 3.0
    dt = step / H.GROUND_SPEED
    cap = ARC_TURN_RATE * dt if turn_cap else None
    max_steps = int(round(max_distance / step))
    rng = np.random.default_rng(planner_seed)
    goal, q = H.GOAL.copy(), H.START.copy()
    g0 = H._gen_time(timeline)
    t0 = time.perf_counter()
    bias = rng.normal(0, 0.06, size=2)
    heading = math.atan2(goal[1] - q[1], goal[0] - q[0])
    path = [q.copy()]
    info = {"escape_steps": 0, "pushouts": 0, "min_clearance": H.clearance(q, timeline.at(0.0))}
    outcome = "timeout"
    lo, hi = np.array([H.dyn.BOUNDS[0], H.dyn.BOUNDS[2]]), np.array([H.dyn.BOUNDS[1], H.dyn.BOUNDS[3]])

    for k in range(max_steps):
        obs = timeline.at(k * dt)
        if np.linalg.norm(q - goal) <= cfg.goal_tol:
            outcome = "success"
            break
        if sensing in ("full", "range"):
            if sensing == "range":   # exact geometry, all directions, but only within the 50 m sensor range
                obs = [o for o in obs if dyn.get_distance(q, o)[0] <= RAY_RANGE]
            if range_noise > 0:
                eps = nrng.normal(0.0, range_noise, size=len(obs))
                nd = lambda x, i, o: max(dyn.get_distance(x, o)[0] + eps[i], 0.0)
                rho_goal = np.linalg.norm(q - goal) + 1e-12
                Ft = dyn.F_att(q, goal, cfg.k_a)
                for i, o in enumerate(obs):
                    grad = dyn.get_distance(q, o)[1]
                    rho = max(nd(q, i, o), 0.8)
                    Ft = Ft + R_theta @ (cfg.k_r * (10.0 / rho ** 2) * rho_goal ** cfg.goal_factor_pow * grad)
                margin_bad = lambda x, obs=obs: any(nd(x, i, o) <= dyn.COLLISION_MARGIN for i, o in enumerate(obs))
                sensed_clear = lambda x, obs=obs: min((nd(x, i, o) for i, o in enumerate(obs)), default=RAY_RANGE)
            else:
                if obs:
                    Ft, _, _ = dyn.F_total(q, goal, obs, cfg)
                else:
                    Ft = dyn.F_att(q, goal, cfg.k_a)
                margin_bad = lambda x, obs=obs: dyn.in_collision(x, obs, margin=dyn.COLLISION_MARGIN)
                sensed_clear = lambda x, obs=obs: H.clearance(x, obs) if obs else RAY_RANGE
        else:
            hits = _rays(q, heading, H._to_shapely(obs))
            if range_noise > 0:
                noisy = []
                for d, p in hits:
                    dn = max(d + nrng.normal(0.0, range_noise), 0.0)
                    noisy.append((dn, q + (p - q) * (dn / d) if d > 1e-9 else p))
                hits = noisy
            Ft = dyn.F_att(q, goal, cfg.k_a) + _rep_points(q, goal, hits, cfg)
            pts = [p for _, p in hits]
            margin_bad = lambda x: any(np.linalg.norm(x - p) <= dyn.COLLISION_MARGIN for p in pts)
        Fb = Ft + bias
        nF = np.linalg.norm(Fb)

        if cap is None:
            # unconstrained planner, identical to dynamic_apf.simulate_dynamic_apf
            if nF < cfg.stuck_eps:
                sd = rng.normal(0, 1, size=2); sd /= np.linalg.norm(sd) + 1e-12
                q_next = q + cfg.jitter * sd
            else:
                sd = Fb / nF
                q_next = q + step * sd
            q_next = np.clip(q_next, lo, hi)
            tries = 0
            while margin_bad(q_next) and tries < 5:
                perp = np.array([-sd[1], sd[0]]); perp /= np.linalg.norm(perp) + 1e-12
                q_next = np.clip(q + step * 1.8 * perp + rng.normal(0, 0.15, size=2), lo, hi)
                tries += 1
            info["escape_steps"] += bool(tries)
            if margin_bad(q_next):
                if sensing in ("full", "range"):
                    min_rho, best_grad = float("inf"), np.zeros(2)
                    for i, o in enumerate(obs):
                        rho, grad, _ = dyn.get_distance(q_next, o)
                        if range_noise > 0:
                            rho = max(rho + eps[i], 0.0)
                        if rho < min_rho:
                            min_rho, best_grad = rho, grad
                else:
                    d_, p_ = min(((np.linalg.norm(q_next - p), p) for p in pts), key=lambda x: x[0])
                    min_rho, best_grad = d_, (q_next - p_) / (d_ + 1e-12)
                if min_rho < dyn.COLLISION_MARGIN:
                    q_next = q_next + (dyn.COLLISION_MARGIN - min_rho + 3.0) * best_grad
                    info["pushouts"] += 1
            if np.linalg.norm(q_next - q) > 1e-9:
                heading = math.atan2(q_next[1] - q[1], q_next[0] - q[0])
        else:
            desired = math.atan2(Fb[1], Fb[0]) if nF >= cfg.stuck_eps else heading
            if momentum is not None and d_prev is not None and nF >= cfg.stuck_eps:
                mv = momentum * d_prev + (1 - momentum) * Fb / nF
                if np.linalg.norm(mv) > 1e-12:
                    desired = math.atan2(mv[1], mv[0])
            delta = math.atan2(math.sin(desired - heading), math.cos(desired - heading))
            cand = [heading + max(-cap, min(cap, delta))]
            cand_q = [np.clip(q + step * np.array([math.cos(c), math.sin(c)]), lo, hi) for c in cand]
            if margin_bad(cand_q[0]):
                info["escape_steps"] += 1
                # choose among reachable headings the one with the largest sensed clearance
                for c in (heading - cap, heading, heading + cap):
                    cand.append(c)
                    cand_q.append(np.clip(q + step * np.array([math.cos(c), math.sin(c)]), lo, hi))
                if sensing in ("full", "range"):
                    score = [sensed_clear(x) for x in cand_q]
                else:
                    score = [min((np.linalg.norm(x - p) for p in pts), default=RAY_RANGE) for x in cand_q]
                i = int(np.argmax(score))
            else:
                i = 0
            heading, q_next = cand[i], cand_q[i]

        if heading_noise > 0 and np.linalg.norm(q_next - q) > 1e-9:
            # actuation noise: the executed displacement deviates from the commanded one
            a = nrng.normal(0.0, heading_noise)
            c_, s_ = math.cos(a), math.sin(a)
            dvec = q_next - q
            q_next = np.clip(q + np.array([c_ * dvec[0] - s_ * dvec[1], s_ * dvec[0] + c_ * dvec[1]]), lo, hi)
            heading = math.atan2(q_next[1] - q[1], q_next[0] - q[0])

        if np.linalg.norm(q_next - q) > 1e-9:
            d_prev = (q_next - q) / np.linalg.norm(q_next - q)
        q = q_next
        path.append(q.copy())
        c = H.clearance(q, timeline.at((k + 1) * dt))
        info["min_clearance"] = min(info["min_clearance"], c)
        if c < H.COLLISION_RADIUS:
            outcome = "collision"
            break
    return H._summ(path, outcome, t0, info, H._gen_time(timeline) - g0)


VARIANTS = [
    {"sensing": s, "turn_cap": tc, "rate": r}
    for s in ("full", "range", "rays") for tc in (False, True) for r in ("apf", "arc")
]


def label(v):
    return f"{v['sensing']}|{'cap' if v['turn_cap'] else 'nocap'}|{v['rate']}"
