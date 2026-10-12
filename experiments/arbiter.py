"""
Envelope-derived arbiter (instructions.md WS-D): a learned default planner L and a classical
fallback C, switched at the active planner's decision instants on the shared obstacle timeline.

L : a learned policy (ARC/TD3/SAC checkpoint), 3.0 m per decision, as harness.ARCPlanner.
C : 'apf_rays' - APF under ARC's nine-ray sensing, 3.0 m steps at 6 Hz (apf_variants.run with
    sensing='rays', rate='arc', no turn cap), optionally with momentum / an oscillation-triggered
    cap (WS-C variants); or 'dwa_cv' - DWA with constant-velocity prediction (dwa.py), exact geometry
    (privileged information).

Signals (from L's own ray observation, computed at every learned step):
  d_min : minimum ray range (m);
  v_c   : closing speed of the minimum-range ray over the last two learned observations, divided by
          the agent speed, i.e. (d_prev - d_now) / (3.0 m); 0 when no ray hits an obstacle.
Rule: L -> C when d_min < d_on or v_c > rho_on.
      C -> L when d_min > d_off and v_c < rho_off and C has been active for at least 1 s, or
      (stall guard) C's goal progress over its last W = 20 decisions is below 10% of the distance it
      travelled in them.
L's LSTM keeps receiving an observation at every learned step (every 5 ticks of the 1/30 s grid at
18 m/s) while C is active. Every decision is checked against the obstacles at the time it ends
(exact clearance < 5 m; for L also any ray < 5 m and leaving the arena), as in each planner's own
protocol. Time budget 600 s at 18 m/s (10.8 km).

With d_on = -inf and rho_on = inf the arbiter is L alone and reproduces harness.ARCPlanner.run;
with L disabled it is C alone and reproduces apf_variants.run / dwa.run (scripts/ws_d_identity.py).
"""
import math
import time

import numpy as np

import harness as H
import apf_variants as V
import dwa as DW

dyn = H.dyn
TICKS = {"L": 5, "apf_rays": 5, "dwa_cv": 3}          # ticks of TIMELINE_DT per decision (at any speed)
W_STALL, STALL_FRAC = 20, 0.10


class Learned:
    def __init__(self, model_path):
        self.P = H.ARCPlanner(model_path)
        self.env = self.P.env

    def reset(self):
        self.hist = self.P.ObservationHistory(self.P.hist_len, self.P.obs_dim)
        self.started = False

    def observe(self, q32, psi, obs_list, nrng=None, range_noise=0.0):
        env = self.env
        env.uav_pos = q32
        env.uav_heading = psi
        env.target_pos = H.GOAL.astype(np.float32).copy()
        env.obstacles = H._to_shapely(obs_list)
        raw = env._get_observation()
        ob = raw
        if range_noise > 0:                       # as harness.ARCPlanner.run's perceive()
            ob = raw.copy()
            d = ob[3:] * env.sensor_range
            hit = d < env.sensor_range - 1e-6
            d[hit] = np.clip(d[hit] + nrng.normal(0.0, range_noise, size=int(hit.sum())), 0.1, env.sensor_range)
            ob[3:] = d / env.sensor_range
        if not self.started:
            self.hist.reset(ob)
            self.started = True
        else:
            self.hist.append(ob)
        return raw, ob

    def act(self, psi, nrng=None, heading_noise=0.0):
        a, _ = self.P.agent.select_action(self.hist.get_tensor(self.P.agent.device), None, noise_scale=0.0)
        turn = float(a[0]) * self.env.action_range
        if heading_noise > 0:
            turn += nrng.normal(0.0, heading_noise)
        return self.env._normalize_angle(psi + turn)


class APFRays:
    """One decision of APF under ARC's nine rays (apf_variants.run, sensing='rays', no cap, 3.0 m)."""

    def __init__(self, momentum=None, osc_cap_K=None):
        self.cfg = dyn.APFConfig()
        self.step = 3.0
        self.momentum, self.osc_cap_K = momentum, osc_cap_K
        self.lo, self.hi = np.array([dyn.BOUNDS[0], dyn.BOUNDS[2]]), np.array([dyn.BOUNDS[1], dyn.BOUNDS[3]])

    def reset(self, rng):
        self.rng = rng
        self.bias = rng.normal(0, 0.06, size=2)
        self.d_prev, self.big_turns, self.cap_left = None, 0, 0

    def decide(self, q, heading, obs, nrng=None, range_noise=0.0, heading_noise=0.0):
        cfg, goal, step, rng = self.cfg, H.GOAL.copy(), self.step, self.rng
        hits = V._rays(q, heading, H._to_shapely(obs))
        if range_noise > 0:
            noisy = []
            for d, p in hits:
                dn = max(d + nrng.normal(0.0, range_noise), 0.0)
                noisy.append((dn, q + (p - q) * (dn / d) if d > 1e-9 else p))
            hits = noisy
        Ft = dyn.F_att(q, goal, cfg.k_a) + V._rep_points(q, goal, hits, cfg)
        pts = [p for _, p in hits]
        margin_bad = lambda x: any(np.linalg.norm(x - p) <= dyn.COLLISION_MARGIN for p in pts)
        Fb = Ft + self.bias
        nF = np.linalg.norm(Fb)
        cap = math.radians(9.0)
        if self.cap_left > 0:                                  # oscillation-triggered cap (WS-C variant)
            self.cap_left -= 1
            desired = math.atan2(Fb[1], Fb[0]) if nF >= cfg.stuck_eps else heading
            delta = math.atan2(math.sin(desired - heading), math.cos(desired - heading))
            cand = [heading + max(-cap, min(cap, delta)), heading - cap, heading, heading + cap]
            cq = [np.clip(q + step * np.array([math.cos(c), math.sin(c)]), self.lo, self.hi) for c in cand]
            i = 0 if not margin_bad(cq[0]) else int(np.argmax([min((np.linalg.norm(x - p) for p in pts), default=50.0) for x in cq]))
            new_h, q_next = cand[i], cq[i]
        else:
            if nF < cfg.stuck_eps:
                sd = rng.normal(0, 1, size=2); sd /= np.linalg.norm(sd) + 1e-12
                q_next = q + cfg.jitter * sd
            else:
                sd = Fb / nF
                if self.momentum is not None and self.d_prev is not None:
                    m = self.momentum * self.d_prev + (1 - self.momentum) * sd
                    sd = m / np.linalg.norm(m) if np.linalg.norm(m) > 1e-12 else sd
                q_next = q + step * sd
            q_next = np.clip(q_next, self.lo, self.hi)
            tries = 0
            while margin_bad(q_next) and tries < 5:
                perp = np.array([-sd[1], sd[0]]); perp /= np.linalg.norm(perp) + 1e-12
                q_next = np.clip(q + step * 1.8 * perp + rng.normal(0, 0.15, size=2), self.lo, self.hi)
                tries += 1
            if margin_bad(q_next):
                d_, p_ = min(((np.linalg.norm(q_next - p), p) for p in pts), key=lambda x: x[0])
                if d_ < dyn.COLLISION_MARGIN:
                    q_next = q_next + (dyn.COLLISION_MARGIN - d_ + 3.0) * ((q_next - p_) / (d_ + 1e-12))
            new_h = math.atan2(q_next[1] - q[1], q_next[0] - q[0]) if np.linalg.norm(q_next - q) > 1e-9 else heading
        if heading_noise > 0 and np.linalg.norm(q_next - q) > 1e-9:
            a = nrng.normal(0.0, heading_noise)
            c_, s_ = math.cos(a), math.sin(a)
            dvec = q_next - q
            q_next = np.clip(q + np.array([c_ * dvec[0] - s_ * dvec[1], s_ * dvec[0] + c_ * dvec[1]]), self.lo, self.hi)
            new_h = math.atan2(q_next[1] - q[1], q_next[0] - q[0])
        turn = abs(math.degrees(math.atan2(math.sin(new_h - heading), math.cos(new_h - heading))))
        if self.osc_cap_K is not None:
            self.big_turns = self.big_turns + 1 if turn > 150.0 else 0
            if self.big_turns >= 2 and self.cap_left == 0:
                self.cap_left, self.big_turns = self.osc_cap_K, 0
        disp = q_next - q
        if np.linalg.norm(disp) > 1e-9:
            self.d_prev = disp / np.linalg.norm(disp)
        return q_next, new_h


class DWACV:
    """One decision of DWA-CV (dwa.run, predict=True) with persistent (v, w) and the last obstacles."""

    def __init__(self, weights):
        self.w8 = {**DW.WEIGHTS, **weights}

    def reset(self, v=0.0, w=0.0):
        self.v, self.w, self.prev = v, w, None

    def decide(self, q, psi, obs, nrng=None, range_noise=0.0, heading_noise=0.0):
        w8 = self.w8
        v_max = H.GROUND_SPEED
        s = v_max / DW.V_REF
        w_max, a_max, al_max = DW.W_MAX_REF * s, DW.A_MAX_REF * s * s, DW.ALPHA_MAX_REF * s * s
        dt = 1.8 / v_max
        n_h = int(round(DW.HORIZON_REF / (dt * s)))
        lo, hi = np.array([dyn.BOUNDS[0], dyn.BOUNDS[2]]), np.array([dyn.BOUNDS[1], dyn.BOUNDS[3]])
        goal = H.GOAL
        tau = np.arange(1, n_h + 1) * dt
        vs = np.linspace(max(0.0, self.v - a_max * dt), min(v_max, self.v + a_max * dt), DW.N_V)
        ws = np.linspace(max(-w_max, self.w - al_max * dt), min(w_max, self.w + al_max * dt), DW.N_W)
        VV, WW = np.meshgrid(vs, ws, indexing="ij")
        VV, WW = VV.ravel(), WW.ravel()
        ang = psi + WW[:, None] * tau[None, :]
        stp = (VV * dt)[:, None]
        X = q[0] + np.cumsum(stp * np.cos(ang), axis=1)
        Y = q[1] + np.cumsum(stp * np.sin(ang), axis=1)
        P = np.stack([X, Y], axis=-1)
        eps = nrng.normal(0.0, range_noise, size=len(obs)) if range_noise > 0 else np.zeros(len(obs))
        clr = np.full(P.shape[:2], np.inf)
        for i, o in enumerate(obs):
            Q = P
            if self.prev is not None:
                vel = (DW._centroid(o) - DW._centroid(self.prev[i])) / dt
                Q = P - vel[None, None, :] * tau[None, :, None]
            clr = np.minimum(clr, np.maximum(DW.dist(Q, o) + eps[i], 0.0))
        cmin = clr.min(axis=1)
        inb = np.all((P >= lo) & (P <= hi), axis=(1, 2))
        brake_ok = VV <= np.sqrt(2.0 * a_max * np.maximum(cmin - DW.MARGIN, 0.0)) + 1e-9
        adm = (cmin > DW.MARGIN) & inb & brake_ok
        if adm.any():
            end = P[:, -1, :]
            to_goal = np.arctan2(goal[1] - end[:, 1], goal[0] - end[:, 0])
            dpsi = np.abs(np.arctan2(np.sin(to_goal - ang[:, -1]), np.cos(to_goal - ang[:, -1])))
            terms = []
            for x in (math.pi - dpsi, np.minimum(cmin, DW.CLEAR_CAP), VV):
                xa = x[adm]
                r_ = xa.max() - xa.min()
                terms.append((x - xa.min()) / r_ if r_ > 1e-12 else np.zeros_like(x))
            G = w8["alpha"] * terms[0] + w8["beta"] * terms[1] + w8["gamma"] * terms[2]
            G[~adm] = -np.inf
            j = int(np.argmax(G))
        else:
            j = int(np.argmax(np.where(inb, cmin, -np.inf)))
        self.v, self.w = float(VV[j]), float(WW[j])
        self.prev = obs
        psi_n = psi + self.w * dt
        d = self.v * dt * np.array([math.cos(psi_n), math.sin(psi_n)])
        if heading_noise > 0 and self.v > 0:
            a = nrng.normal(0.0, heading_noise)
            c_, s_ = math.cos(a), math.sin(a)
            d = np.array([c_ * d[0] - s_ * d[1], s_ * d[0] + c_ * d[1]])
            psi_n += a
        return np.clip(q + d, lo, hi), math.atan2(math.sin(psi_n), math.cos(psi_n))


def run(timeline, L, C, c_kind, d_on, rho_on, d_off=None, rho_off=None, c_seed=0, l_enabled=True,
        max_time_s=None, log_signals=False, range_noise=0.0, heading_noise=0.0, noise_seed=0):
    """One arbitrated run. c_kind in {'apf_rays', 'dwa_cv'}; l_enabled=False gives C alone."""
    d_off = d_on + 5.0 if d_off is None else d_off
    rho_off = 0.8 * rho_on if rho_off is None else rho_off
    tick = H.TIMELINE_DT
    max_time_s = max_time_s or H.MAX_DISTANCE / H.GROUND_SPEED
    max_ticks = int(round(max_time_s / tick))
    g = H.GOAL - H.START
    psi = float(math.atan2(g[1], g[0]))
    q32 = H.START.astype(np.float32).copy()                 # L integrates in float32, as ARCPlanner does
    q = q32.astype(float)
    if l_enabled:
        L.reset()
    if c_kind == "apf_rays":
        C.reset(np.random.default_rng(c_seed))
    else:
        C.reset(v=H.GROUND_SPEED if l_enabled else 0.0)
    active = "L" if l_enabled else "C"
    next_dec = 0
    c_since, c_hist = None, []                              # tick C became active; C positions for the stall guard
    d_prev_min, sig = None, (50.0, 0.0)
    path, mode_ticks = [q.copy()], {"L": 0, "C": 0}
    switches, switch_ticks = 0, []
    switch_log = []                                         # (direction, tick, distance to goal, d_min, v_c, reason)
    info = {"min_clearance": H.clearance(q, timeline.at(0.0)), "decisions": {"L": 0, "C": 0}}
    outcome, t0, g0 = "timeout", time.perf_counter(), H._gen_time(timeline)
    last_learned_obs_tick = -10 ** 9
    sig_log = []
    nrng = np.random.default_rng(noise_seed)
    nz = {"nrng": nrng, "range_noise": range_noise}
    for k in range(max_ticks + 1):
        obs = timeline.at(k * tick)
        learned_instant = l_enabled and ((active == "L" and k == next_dec) or (active == "C" and k - last_learned_obs_tick >= TICKS["L"]))
        ob = raw = None
        if learned_instant:
            raw, ob = L.observe(q32 if active == "L" else q.astype(np.float32), psi, obs, **nz)
            last_learned_obs_tick = k
            rays = np.asarray(ob[3:], float) * L.env.sensor_range
            i_min = int(np.argmin(rays))
            d_now = float(rays[i_min])
            v_c = 0.0
            if d_now < L.env.sensor_range - 1e-6 and d_prev_min is not None:
                v_c = (d_prev_min[i_min] - d_now) / 3.0
            d_prev_min = rays
            sig = (d_now, v_c)
            if log_signals:
                sig_log.append((k, d_now, v_c, active))
        if k != next_dec:
            continue
        if k > 0:                                           # checks for the state reached at this decision instant
            c = H.clearance(q, obs)
            info["min_clearance"] = min(info["min_clearance"], c)
            if active == "L":
                p = q32
                if p[0] < 0 or p[0] > L.env.width or p[1] < 0 or p[1] > L.env.height:
                    outcome = "out_of_bounds"; break
                native = raw is not None and bool(np.any(np.asarray(raw[3:]) * L.env.sensor_range < L.env.d_min_obstacle))
                if c < H.COLLISION_RADIUS or native:
                    outcome = "collision"; break
            elif c < H.COLLISION_RADIUS:
                outcome = "collision"; break
            dg = np.linalg.norm(q - H.GOAL)
            if (dg < H.GOAL_RADIUS) if active == "L" else (dg <= H.GOAL_RADIUS):   # each planner's own goal test
                outcome = "success"; break
        if k >= max_ticks:
            break
        # ---- arbitration (only at the active planner's decision instants)
        if l_enabled:
            d_min, v_c = sig
            if active == "L" and (d_min < d_on or v_c > rho_on):
                active, switches, c_since, c_hist = "C", switches + 1, k, [q.copy()]
                switch_ticks.append(k)
                switch_log.append(("L->C", k, float(np.linalg.norm(q - H.GOAL)), float(d_min), float(v_c), "trigger"))
                if c_kind == "dwa_cv":
                    C.reset(v=H.GROUND_SPEED)
            elif active == "C":
                held = (k - c_since) * tick >= 1.0
                stall = False
                if len(c_hist) > W_STALL:
                    win = c_hist[-(W_STALL + 1):]
                    travelled = sum(np.linalg.norm(win[i + 1] - win[i]) for i in range(W_STALL))
                    progress = np.linalg.norm(win[0] - H.GOAL) - np.linalg.norm(win[-1] - H.GOAL)
                    stall = travelled > 0 and progress < STALL_FRAC * travelled
                if (held and d_min > d_off and v_c < rho_off) or stall:
                    active, switches = "L", switches + 1
                    switch_ticks.append(k)
                    switch_log.append(("C->L", k, float(np.linalg.norm(q - H.GOAL)), float(d_min), float(v_c),
                                       "stall_guard" if stall and not (held and d_min > d_off and v_c < rho_off) else "release"))
                    q32 = q.astype(np.float32)
                    if ob is None:                          # L needs a fresh observation to act on
                        raw, ob = L.observe(q32, psi, obs, **nz)
                        last_learned_obs_tick = k
        # ---- decision of the active planner
        if active == "L":
            psi = L.act(psi, nrng, heading_noise)
            q32 = q32.copy()
            q32[0] += 3.0 * np.cos(psi)
            q32[1] += 3.0 * np.sin(psi)
            q = q32.astype(float)
            info["decisions"]["L"] += 1
            per = TICKS["L"]
        else:
            q, psi = C.decide(q, psi, obs, nrng, range_noise, heading_noise)
            q32 = q.astype(np.float32)
            c_hist.append(q.copy())
            info["decisions"]["C"] += 1
            per = TICKS[c_kind]
        mode_ticks["L" if active == "L" else "C"] += per
        path.append(q.copy())
        next_dec = k + per
    tot = max(1, mode_ticks["L"] + mode_ticks["C"])
    info.update({"frac_time_C": mode_ticks["C"] / tot, "switches": switches, "switch_log": switch_log,
                 "chatter": int(sum(1 for a, b in zip(switch_ticks, switch_ticks[1:]) if (b - a) * tick < 1.0))})
    if log_signals:
        info["signals"] = sig_log
        info["end_tick"] = k
    return H._summ(path, outcome, t0, info, H._gen_time(timeline) - g0)
