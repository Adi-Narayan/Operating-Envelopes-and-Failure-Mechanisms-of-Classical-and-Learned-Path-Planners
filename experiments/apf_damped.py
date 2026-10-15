"""
Damped APF variants (instructions.md WS-C, C4). Full sensing (exact geometry of every obstacle),
1.8 m steps at 10 Hz (at 18 m/s), the paper's gains and theta = 25 deg; harness protocol for
evaluation (exact clearance < 5 m after a step is a collision, 10 m goal radius, 6000 steps).

  momentum=beta     : direction d_t = normalise(beta * d_{t-1} + (1 - beta) * f_t), f_t the unit
                      (biased) force direction; the step is taken along d_t.
  osc_cap_K=K       : if the heading changes by more than 150 deg on two consecutive steps, the
                      54 deg/s turn cap (5.4 deg per 1.8 m step) is applied for the next K steps.
  always_cap=True   : the existing always-on 54 deg/s cap (apf_variants.run(turn_cap=True)).

With momentum=None, osc_cap_K=None and always_cap=False this is the paper's APF and reproduces
harness.run_apf (checked by scripts/ws_c_apfd_identity.py).
Margin handling follows apf_variants.run: unconstrained steps use escape attempts and the
push-out; capped steps choose among headings reachable within the cap.
"""
import math
import time

import numpy as np

import harness as H

dyn = H.dyn
CAP_RAD_PER_S = math.radians(54.0)
OSC_DEG = 150.0


def run(timeline, planner_seed, momentum=None, osc_cap_K=None, always_cap=False, max_distance=H.MAX_DISTANCE,
        range_noise=0.0, heading_noise=0.0, noise_seed=None):
    cfg = dyn.APFConfig()
    nrng = np.random.default_rng(planner_seed + 7_000_000 if noise_seed is None else noise_seed)
    R_theta = dyn.rotation_matrix(cfg.theta)
    step = 1.8
    dt = step / H.GROUND_SPEED
    cap = CAP_RAD_PER_S * dt
    max_steps = int(round(max_distance / step))
    rng = np.random.default_rng(planner_seed)
    goal, q = H.GOAL.copy(), H.START.copy()
    g0 = H._gen_time(timeline)
    t0 = time.perf_counter()
    bias = rng.normal(0, 0.06, size=2)
    heading = math.atan2(goal[1] - q[1], goal[0] - q[0])
    d_prev = None
    path = [q.copy()]
    info = {"escape_steps": 0, "pushouts": 0, "min_clearance": H.clearance(q, timeline.at(0.0)),
            "capped_steps": 0, "osc_triggers": 0}
    outcome = "timeout"
    lo, hi = np.array([dyn.BOUNDS[0], dyn.BOUNDS[2]]), np.array([dyn.BOUNDS[1], dyn.BOUNDS[3]])
    big_turns, cap_left = 0, 0

    for k in range(max_steps):
        obs = timeline.at(k * dt)
        if np.linalg.norm(q - goal) <= cfg.goal_tol:
            outcome = "success"
            break
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
            sensed_clear = lambda x, obs=obs: min((nd(x, i, o) for i, o in enumerate(obs)), default=50.0)
        else:
            eps = None
            Ft = dyn.F_total(q, goal, obs, cfg)[0] if obs else dyn.F_att(q, goal, cfg.k_a)
            margin_bad = lambda x, obs=obs: dyn.in_collision(x, obs, margin=dyn.COLLISION_MARGIN)
            sensed_clear = lambda x, obs=obs: H.clearance(x, obs) if obs else 50.0
        Fb = Ft + bias
        nF = np.linalg.norm(Fb)
        capped = always_cap or cap_left > 0

        if not capped:
            if nF < cfg.stuck_eps:
                sd = rng.normal(0, 1, size=2); sd /= np.linalg.norm(sd) + 1e-12
                q_next = q + cfg.jitter * sd
            else:
                sd = Fb / nF
                if momentum is not None and d_prev is not None:
                    m = momentum * d_prev + (1.0 - momentum) * sd
                    nm = np.linalg.norm(m)
                    sd = m / nm if nm > 1e-12 else sd
                q_next = q + step * sd
            q_next = np.clip(q_next, lo, hi)
            tries = 0
            while margin_bad(q_next) and tries < 5:
                perp = np.array([-sd[1], sd[0]]); perp /= np.linalg.norm(perp) + 1e-12
                q_next = np.clip(q + step * 1.8 * perp + rng.normal(0, 0.15, size=2), lo, hi)
                tries += 1
            info["escape_steps"] += bool(tries)
            if margin_bad(q_next):
                min_rho, best_grad = float("inf"), np.zeros(2)
                for i, o in enumerate(obs):
                    rho, grad, _ = dyn.get_distance(q_next, o)
                    if eps is not None:
                        rho = max(rho + eps[i], 0.0)
                    if rho < min_rho:
                        min_rho, best_grad = rho, grad
                if min_rho < dyn.COLLISION_MARGIN:
                    q_next = q_next + (dyn.COLLISION_MARGIN - min_rho + 3.0) * best_grad
                    info["pushouts"] += 1
            new_heading = math.atan2(q_next[1] - q[1], q_next[0] - q[0]) if np.linalg.norm(q_next - q) > 1e-9 else heading
        else:
            info["capped_steps"] += 1
            desired = math.atan2(Fb[1], Fb[0]) if nF >= cfg.stuck_eps else heading
            if momentum is not None and d_prev is not None and nF >= cfg.stuck_eps:
                m = momentum * d_prev + (1.0 - momentum) * (Fb / nF)
                if np.linalg.norm(m) > 1e-12:
                    desired = math.atan2(m[1], m[0])
            delta = math.atan2(math.sin(desired - heading), math.cos(desired - heading))
            cand = [heading + max(-cap, min(cap, delta))]
            cand_q = [np.clip(q + step * np.array([math.cos(c), math.sin(c)]), lo, hi) for c in cand]
            if margin_bad(cand_q[0]):
                info["escape_steps"] += 1
                for c in (heading - cap, heading, heading + cap):
                    cand.append(c)
                    cand_q.append(np.clip(q + step * np.array([math.cos(c), math.sin(c)]), lo, hi))
                i = int(np.argmax([sensed_clear(x) for x in cand_q]))
            else:
                i = 0
            new_heading, q_next = cand[i], cand_q[i]
            if cap_left > 0:
                cap_left -= 1

        if heading_noise > 0 and np.linalg.norm(q_next - q) > 1e-9:
            a = nrng.normal(0.0, heading_noise)
            c_, s_ = math.cos(a), math.sin(a)
            dvec = q_next - q
            q_next = np.clip(q + np.array([c_ * dvec[0] - s_ * dvec[1], s_ * dvec[0] + c_ * dvec[1]]), lo, hi)
            new_heading = math.atan2(q_next[1] - q[1], q_next[0] - q[0])

        turn = abs(math.degrees(math.atan2(math.sin(new_heading - heading), math.cos(new_heading - heading))))
        if osc_cap_K is not None and not always_cap:
            big_turns = big_turns + 1 if turn > OSC_DEG else 0
            if big_turns >= 2 and cap_left == 0:
                cap_left, big_turns = osc_cap_K, 0
                info["osc_triggers"] += 1
        disp = q_next - q
        if np.linalg.norm(disp) > 1e-9:
            d_prev = disp / np.linalg.norm(disp)
        heading = new_heading
        q = q_next
        path.append(q.copy())
        c = H.clearance(q, timeline.at((k + 1) * dt))
        info["min_clearance"] = min(info["min_clearance"], c)
        if c < H.COLLISION_RADIUS:
            outcome = "collision"
            break
    return H._summ(path, outcome, t0, info, H._gen_time(timeline) - g0)


VARIANTS = {
    "apf": {},
    "osc10": {"osc_cap_K": 10},
    "osc30": {"osc_cap_K": 30},
    "mom050": {"momentum": 0.5},
    "mom070": {"momentum": 0.7},
    "mom085": {"momentum": 0.85},
    "cap": {"always_cap": True},
}
