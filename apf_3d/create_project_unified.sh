#!/usr/bin/env bash
set -e

OUTDIR="/home/anshul/Desktop/thrustMIT/apf_unified"
mkdir -p "$OUTDIR/templates"
mkdir -p "$OUTDIR/simulation"
mkdir -p "$OUTDIR/static/results"

# ── Install dependencies ───────────────────────────────────────────────────────
echo "Installing dependencies into myenv..."
conda run -n myenv pip install flask numpy matplotlib plotly psutil openpyxl pillow --quiet
echo "Dependencies installed."

# ── requirements.txt ─────────────────────────────────────────────────────────
cat << 'EOF' > "$OUTDIR/requirements.txt"
flask>=3.0.0
numpy>=1.26.0
matplotlib>=3.8.0
plotly>=5.20.0
psutil>=5.9.0
openpyxl>=3.1.0
pillow>=10.0.0
EOF

# ── README.md ─────────────────────────────────────────────────────────────────
cat << 'EOF' > "$OUTDIR/README.md"
# APF Unified Simulation

APF (Artificial Potential Field) path planning with **3D Static** and **3D Dynamic** modes.

## How to Run

```bash
conda activate myenv
cd /home/anshul/Desktop/thrustMIT/apf_unified
python app.py
```

Open your browser at http://127.0.0.1:5000

## Features
- 3D Static mode: fixed obstacles, animated path playback with Play/Pause
- 3D Dynamic mode: moving obstacles, animated path playback with Play/Pause
- All run metrics appended to `metrics_log.xlsx` (never overwritten)
EOF

# ── simulation/__init__.py ────────────────────────────────────────────────────
touch "$OUTDIR/simulation/__init__.py"

# ── simulation/core.py ────────────────────────────────────────────────────────
cat << 'EOF' > "$OUTDIR/simulation/core.py"
"""
APF core simulation logic — 3D Static and 3D Dynamic modes.
"""

import copy
import numpy as np

DT = 0.1
BASE_SPEED_SCALE = 1.25
BOUNDS_3D = (0.0, 500.0, 0.0, 500.0, 0.0, 500.0)
COLLISION_MARGIN = 6.0
GOAL_PROTECTION_RADIUS = 50.0
MAX_STEPS = 6000


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clamp_vec(v, bounds):
    out = v.copy()
    for i in range(len(v)):
        out[i] = clamp(v[i], bounds[2 * i], bounds[2 * i + 1])
    return out


# ── 3D distance functions ─────────────────────────────────────────────────────

def dist_point_to_sphere_3d(p, center, radius):
    v = p - center
    d = np.linalg.norm(v)
    grad = v / d if d > 1e-12 else np.array([1.0, 0.0, 0.0])
    rho = max(d - radius, 0.0)
    closest = center + grad * radius
    return rho, grad, closest


def dist_point_to_aabb_3d(p, center, half_extents):
    lo = center - half_extents
    hi = center + half_extents
    closest = np.clip(p, lo, hi)
    v = p - closest
    rho = np.linalg.norm(v)
    grad = v / rho if rho > 1e-12 else np.array([0.0, 0.0, 1.0])
    return max(rho, 0.0), grad, closest


def get_distance_3d(p, obs):
    t = obs['type']
    if t == 'sphere':
        cx, cy, cz, r = obs['params']
        return dist_point_to_sphere_3d(p, np.array([cx, cy, cz], float), r)
    elif t == 'box':
        cx, cy, cz, hx, hy, hz = obs['params']
        return dist_point_to_aabb_3d(
            p, np.array([cx, cy, cz], float), np.array([hx, hy, hz], float)
        )


# ── obstacle AABB helper ──────────────────────────────────────────────────────

def get_aabb_3d(obs):
    t = obs['type']
    if t == 'sphere':
        cx, cy, cz, r = obs['params']
        return cx-r, cx+r, cy-r, cy+r, cz-r, cz+r
    elif t == 'box':
        cx, cy, cz, hx, hy, hz = obs['params']
        return cx-hx, cx+hx, cy-hy, cy+hy, cz-hz, cz+hz


# ── Static obstacle definitions ───────────────────────────────────────────────

STATIC_OBSTACLES_3D = [
    {"type": "box",    "params": (180.0, 280.0, 250.0, 30.0, 75.0, 30.0), "label": "B1", "outlier": False},
    {"type": "box",    "params": (370.0, 270.0, 250.0, 30.0, 75.0, 30.0), "label": "B2", "outlier": False},
    {"type": "box",    "params": (230.0, 180.0, 200.0, 30.0, 30.0, 30.0), "label": "B3", "outlier": False},
    {"type": "box",    "params": (270.0, 320.0, 300.0, 30.0, 30.0, 30.0), "label": "B4", "outlier": False},
    {"type": "sphere", "params": (185.0, 155.0, 150.0, 32.5),              "label": "S1", "outlier": False},
    {"type": "sphere", "params": (260.0, 240.0, 250.0, 32.5),              "label": "S2", "outlier": False},
    {"type": "sphere", "params": (325.0, 355.0, 350.0, 32.5),              "label": "S3", "outlier": False},
    {"type": "sphere", "params": (365.0, 175.0, 200.0, 25.0),              "label": "C2", "outlier": False},
    {"type": "sphere", "params": (70.0,  430.0, 430.0, 32.5),              "label": "OL1", "outlier": True},
    {"type": "sphere", "params": (430.0, 70.0,  70.0,  32.5),              "label": "OL2", "outlier": True},
]


# ── DynObstacle ───────────────────────────────────────────────────────────────

def obs_centroid_3d(obs):
    return np.array(obs['params'][:3], dtype=float)


class DynObstacle:
    def __init__(self, obs_dict, vel):
        self.base  = copy.deepcopy(obs_dict)
        self.label = obs_dict['label']
        self.otype = obs_dict['type']
        self.vel   = np.array(vel, dtype=float)
        self.pos   = obs_centroid_3d(obs_dict).copy()
        self._build_obs()

    def _build_obs(self):
        bp = self.base['params']
        if self.otype == 'sphere':
            _, _, _, r = bp
            self.obs = {**self.base, 'params': (self.pos[0], self.pos[1], self.pos[2], r)}
        elif self.otype == 'box':
            _, _, _, hx, hy, hz = bp
            self.obs = {**self.base, 'params': (self.pos[0], self.pos[1], self.pos[2], hx, hy, hz)}

    def step(self, dt, goal=None):
        self.pos += self.vel * dt
        self._build_obs()

        if goal is not None:
            g_dist = np.linalg.norm(self.pos - goal)
            if g_dist < GOAL_PROTECTION_RADIUS:
                dir_away = self.pos - goal
                dir_away /= np.linalg.norm(dir_away) + 1e-12
                push = (GOAL_PROTECTION_RADIUS - g_dist + 10.0) * dir_away
                self.pos += push * 0.6

        aabb  = get_aabb_3d(self.obs)
        speed = np.linalg.norm(self.vel)
        for i in range(3):
            lo, hi     = BOUNDS_3D[2*i], BOUNDS_3D[2*i+1]
            obs_lo, obs_hi = aabb[2*i], aabb[2*i+1]
            if obs_lo < lo:
                self.pos[i] += lo - obs_lo
                self.vel[i]  = abs(self.vel[i])
            if obs_hi > hi:
                self.pos[i] -= obs_hi - hi
                self.vel[i]  = -abs(self.vel[i])
        current_speed = np.linalg.norm(self.vel)
        if current_speed > 1e-8:
            self.vel = self.vel * (speed / current_speed)
        self._build_obs()

    def handle_obs_collision(self, other):
        aabb1 = get_aabb_3d(self.obs)
        aabb2 = get_aabb_3d(other.obs)
        overlaps = []
        for i in range(3):
            lo1, hi1 = aabb1[2*i], aabb1[2*i+1]
            lo2, hi2 = aabb2[2*i], aabb2[2*i+1]
            if hi1 <= lo2 or hi2 <= lo1:
                return
            overlaps.append(min(hi1-lo2, hi2-lo1))
        min_idx = int(np.argmin(overlaps))
        min_ol  = overlaps[min_idx]
        n = np.zeros(3)
        c1 = (aabb1[2*min_idx] + aabb1[2*min_idx+1]) / 2
        c2 = (aabb2[2*min_idx] + aabb2[2*min_idx+1]) / 2
        n[min_idx] = 1.0 if c1 <= c2 else -1.0
        push = min_ol * 0.501
        self.pos -= n * push
        other.pos += n * push
        v1n  = np.dot(self.vel, n)
        v2n  = np.dot(other.vel, n)
        sp1  = np.linalg.norm(self.vel)
        sp2  = np.linalg.norm(other.vel)
        self.vel  += (v2n - v1n) * n
        other.vel += (v1n - v2n) * n
        if sp1 > 1e-8:
            self.vel  *= sp1 / (np.linalg.norm(self.vel)  + 1e-12)
        if sp2 > 1e-8:
            other.vel *= sp2 / (np.linalg.norm(other.vel) + 1e-12)
        self._build_obs()
        other._build_obs()


def assign_velocities_3d(obstacles_list, rng):
    speed_groups = {
        'sphere': BASE_SPEED_SCALE * 2.9,
        'box':    BASE_SPEED_SCALE * 1.8,
    }
    dyns = []
    for obs in obstacles_list:
        if obs.get('outlier', False):
            vel = np.zeros(3)
        else:
            group_max = speed_groups.get(obs['type'], BASE_SPEED_SCALE)
            speed     = rng.uniform(0.7 * group_max, group_max)
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction) + 1e-12
            vel = speed * direction
        dyns.append(DynObstacle(obs, vel))
    return dyns


# ── APF functions ─────────────────────────────────────────────────────────────

class APFConfig:
    def __init__(self):
        self.k_a          = 5.0
        self.k_r          = 0.85
        self.theta        = np.deg2rad(25)
        self.goal_factor_pow = 2
        self.step         = 1.8
        self.goal_tol     = 18.0
        self.stuck_eps    = 1e-6
        self.jitter       = 0.35


def F_att(q, q_goal, k_a):
    return -k_a * (q - q_goal)


def F_rep(q, q_goal, obs, k_r, theta, goal_pow):
    rho, grad_rho, _ = get_distance_3d(q, obs)
    rho      = max(rho, 0.8)
    rho_goal = np.linalg.norm(q - q_goal) + 1e-12
    mag      = k_r * (10.0 / rho ** 2) * (rho_goal ** goal_pow)
    F_base   = mag * grad_rho
    c, s     = np.cos(theta), np.sin(theta)
    Rz       = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)
    return Rz @ F_base


def F_total(q, q_goal, obstacles, cfg):
    fa = F_att(q, q_goal, cfg.k_a)
    fr = np.zeros(3)
    for obs in obstacles:
        fr += F_rep(q, q_goal, obs, cfg.k_r, cfg.theta, cfg.goal_factor_pow)
    return fa + fr, fa, fr


def in_collision(q, obstacles, margin):
    for obs in obstacles:
        rho, _, _ = get_distance_3d(q, obs)
        if rho <= margin:
            return True
    return False


def path_length(path):
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))


# ── Static APF loop ───────────────────────────────────────────────────────────

def simulate_static_apf(start, goal, obstacles, cfg, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    q     = start.copy()
    path  = [q.copy()]
    step_count = 0
    bias  = rng.normal(0, 0.06, size=3)

    while step_count < MAX_STEPS:
        if np.linalg.norm(q - goal) <= cfg.goal_tol:
            path.append(goal.copy())
            return True, np.array(path)

        Ft, _, _ = F_total(q, goal, obstacles, cfg)
        Ft_b = Ft + bias
        nFt  = np.linalg.norm(Ft_b)

        if nFt < cfg.stuck_eps:
            sd    = rng.normal(0, 1, size=3)
            sd   /= np.linalg.norm(sd) + 1e-12
            q_next = q + cfg.jitter * sd
        else:
            sd     = Ft_b / nFt
            q_next = q + cfg.step * sd

        q_next = clamp_vec(q_next, BOUNDS_3D)

        escape_attempts = 0
        while in_collision(q_next, obstacles, COLLISION_MARGIN) and escape_attempts < 5:
            perp  = rng.normal(0, 1, size=3)
            perp -= np.dot(perp, sd) * sd
            perp /= np.linalg.norm(perp) + 1e-12
            q_next = q + cfg.step * 1.8 * perp + rng.normal(0, 0.15, size=3)
            q_next = clamp_vec(q_next, BOUNDS_3D)
            escape_attempts += 1

        if in_collision(q_next, obstacles, COLLISION_MARGIN):
            min_rho, best_grad = float('inf'), np.zeros(3)
            for obs in obstacles:
                rho, grad, _ = get_distance_3d(q_next, obs)
                if rho < min_rho:
                    min_rho, best_grad = rho, grad
            if min_rho < COLLISION_MARGIN:
                q_next += (COLLISION_MARGIN - min_rho + 3.0) * best_grad

        q = q_next
        path.append(q.copy())
        step_count += 1

    return False, np.array(path)


# ── Dynamic APF loop ──────────────────────────────────────────────────────────

def simulate_dynamic_apf(start, goal, dyn_obstacles, cfg, dt=DT, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    q     = start.copy()
    path  = [q.copy()]
    obs_history = [copy.deepcopy([d.obs for d in dyn_obstacles])]
    step_count  = 0
    bias  = rng.normal(0, 0.06, size=3)

    while step_count < MAX_STEPS:
        for d in dyn_obstacles:
            d.step(dt, goal=goal)
        for i in range(len(dyn_obstacles)):
            for j in range(i+1, len(dyn_obstacles)):
                dyn_obstacles[i].handle_obs_collision(dyn_obstacles[j])

        current_obs = [d.obs for d in dyn_obstacles]

        if np.linalg.norm(q - goal) <= cfg.goal_tol:
            path.append(goal.copy())
            obs_history.append(copy.deepcopy(current_obs))
            return True, np.array(path), obs_history

        Ft, _, _ = F_total(q, goal, current_obs, cfg)
        Ft_b = Ft + bias
        nFt  = np.linalg.norm(Ft_b)

        if nFt < cfg.stuck_eps:
            sd    = rng.normal(0, 1, size=3)
            sd   /= np.linalg.norm(sd) + 1e-12
            q_next = q + cfg.jitter * sd
        else:
            sd     = Ft_b / nFt
            q_next = q + cfg.step * sd

        q_next = clamp_vec(q_next, BOUNDS_3D)

        escape_attempts = 0
        while in_collision(q_next, current_obs, COLLISION_MARGIN) and escape_attempts < 5:
            perp  = rng.normal(0, 1, size=3)
            perp -= np.dot(perp, sd) * sd
            perp /= np.linalg.norm(perp) + 1e-12
            q_next = q + cfg.step * 1.8 * perp + rng.normal(0, 0.15, size=3)
            q_next = clamp_vec(q_next, BOUNDS_3D)
            escape_attempts += 1

        if in_collision(q_next, current_obs, COLLISION_MARGIN):
            min_rho, best_grad = float('inf'), np.zeros(3)
            for obs in current_obs:
                rho, grad, _ = get_distance_3d(q_next, obs)
                if rho < min_rho:
                    min_rho, best_grad = rho, grad
            if min_rho < COLLISION_MARGIN:
                q_next += (COLLISION_MARGIN - min_rho + 3.0) * best_grad

        q = q_next
        path.append(q.copy())
        obs_history.append(copy.deepcopy(current_obs))
        step_count += 1

    return False, np.array(path), obs_history
EOF

# ── simulation/visualization.py ───────────────────────────────────────────────
cat << 'EOF' > "$OUTDIR/simulation/visualization.py"
"""
Visualization: animated Plotly 3D for both Static and Dynamic modes.
Features play/pause controls and a scrubber slider.
"""

import numpy as np
import plotly.graph_objects as go

DARK_BG    = '#080c14'
DARK_PANEL = '#0d1220'
DARK_GRID  = '#1a2a4a'


# ── Mesh helpers ──────────────────────────────────────────────────────────────

def _sphere_mesh(cx, cy, cz, r, n=18):
    phi   = np.linspace(0, np.pi, n)
    theta = np.linspace(0, 2*np.pi, n)
    ph, th = np.meshgrid(phi, theta)
    xs = (cx + r * np.sin(ph) * np.cos(th)).ravel().tolist()
    ys = (cy + r * np.sin(ph) * np.sin(th)).ravel().tolist()
    zs = (cz + r * np.cos(ph)).ravel().tolist()
    ii, jj, kk = [], [], []
    for row in range(n-1):
        for col in range(n-1):
            a = row*n+col; b = a+1; c = a+n; d = c+1
            ii += [a, a]; jj += [b, c]; kk += [c, d]
    return xs, ys, zs, ii, jj, kk


def _box_mesh(cx, cy, cz, hx, hy, hz):
    verts = [
        (cx-hx,cy-hy,cz-hz),(cx+hx,cy-hy,cz-hz),
        (cx+hx,cy+hy,cz-hz),(cx-hx,cy+hy,cz-hz),
        (cx-hx,cy-hy,cz+hz),(cx+hx,cy-hy,cz+hz),
        (cx+hx,cy+hy,cz+hz),(cx-hx,cy+hy,cz+hz),
    ]
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (2,3,7),(2,7,6),(1,2,6),(1,6,5),(0,3,7),(0,7,4)]
    ii = [f[0] for f in faces]; jj = [f[1] for f in faces]; kk = [f[2] for f in faces]
    return xs, ys, zs, ii, jj, kk


def _box_lines(cx, cy, cz, hx, hy, hz):
    corners = [
        (cx-hx,cy-hy,cz-hz),(cx+hx,cy-hy,cz-hz),
        (cx+hx,cy+hy,cz-hz),(cx-hx,cy+hy,cz-hz),
        (cx-hx,cy-hy,cz+hz),(cx+hx,cy-hy,cz+hz),
        (cx+hx,cy+hy,cz+hz),(cx-hx,cy+hy,cz+hz),
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]
    return xs, ys, zs


def _axis_style(label):
    return dict(
        title=dict(text=label, font=dict(color='#4a6a9a', size=10, family='monospace')),
        range=[0, 500],
        backgroundcolor=DARK_PANEL,
        gridcolor=DARK_GRID,
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        tickfont=dict(color='#4a6a9a', size=9, family='monospace'),
        showspikes=False,
    )


def _path_colors(n):
    t = np.linspace(0, 1, max(n, 2))
    return [f'rgb({int(ti*123)},{int(212-ti*165)},255)' for ti in t]


# ── Static obstacles as fixed traces ─────────────────────────────────────────

def _static_obs_traces(obstacles):
    traces = []
    for obs in obstacles:
        is_out = obs.get('outlier', False)
        if obs['type'] == 'sphere':
            cx, cy, cz, r = obs['params']
            c_solid = 'rgba(220,50,50,0.72)' if is_out else 'rgba(0,160,255,0.72)'
            c_wire  = '#ff3333' if is_out else '#00d4ff'
            xs, ys, zs, ii, jj, kk = _sphere_mesh(cx, cy, cz, r)
            traces.append(go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                                    color=c_solid, opacity=0.55, showscale=False,
                                    name=obs.get('label',''), showlegend=True,
                                    lighting=dict(ambient=0.5, diffuse=0.7, specular=0.5)))
            theta = np.linspace(0, 2*np.pi, 40)
            traces.append(go.Scatter3d(
                x=(cx+r*np.cos(theta)).tolist(), y=(cy+r*np.sin(theta)).tolist(), z=[cz]*40,
                mode='lines', line=dict(color=c_wire, width=2),
                showlegend=False, hoverinfo='skip'))
        elif obs['type'] == 'box':
            cx, cy, cz, hx, hy, hz = obs['params']
            c_solid = 'rgba(200,30,30,0.55)' if is_out else 'rgba(80,60,200,0.55)'
            c_wire  = '#ff3333' if is_out else '#7b5fff'
            bxs, bys, bzs, bi, bj, bk = _box_mesh(cx, cy, cz, hx, hy, hz)
            traces.append(go.Mesh3d(x=bxs, y=bys, z=bzs, i=bi, j=bj, k=bk,
                                    color=c_solid, opacity=0.5, showscale=False,
                                    name=obs.get('label',''), showlegend=True,
                                    lighting=dict(ambient=0.5, diffuse=0.7, specular=0.4)))
            lx, ly, lz = _box_lines(cx, cy, cz, hx, hy, hz)
            traces.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines',
                                       line=dict(color=c_wire, width=2),
                                       showlegend=False, hoverinfo='skip'))
    return traces


# ── Per-frame dynamic obstacle traces ────────────────────────────────────────

def _dyn_obs_traces(obs_list):
    traces = []
    for obs in obs_list:
        is_out = obs.get('outlier', False)
        if obs['type'] == 'sphere':
            cx, cy, cz, r = obs['params']
            c_solid = 'rgba(220,50,50,0.72)' if is_out else 'rgba(0,160,255,0.72)'
            c_wire  = '#ff3333' if is_out else '#00d4ff'
            xs, ys, zs, ii, jj, kk = _sphere_mesh(cx, cy, cz, r)
            traces.append(go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                                    color=c_solid, opacity=0.55, showscale=False,
                                    showlegend=False,
                                    lighting=dict(ambient=0.5, diffuse=0.7, specular=0.5)))
            theta = np.linspace(0, 2*np.pi, 40)
            traces.append(go.Scatter3d(
                x=(cx+r*np.cos(theta)).tolist(), y=(cy+r*np.sin(theta)).tolist(), z=[cz]*40,
                mode='lines', line=dict(color=c_wire, width=2),
                showlegend=False, hoverinfo='skip'))
        elif obs['type'] == 'box':
            cx, cy, cz, hx, hy, hz = obs['params']
            c_solid = 'rgba(200,30,30,0.55)' if is_out else 'rgba(80,60,200,0.55)'
            c_wire  = '#ff3333' if is_out else '#7b5fff'
            bxs, bys, bzs, bi, bj, bk = _box_mesh(cx, cy, cz, hx, hy, hz)
            traces.append(go.Mesh3d(x=bxs, y=bys, z=bzs, i=bi, j=bj, k=bk,
                                    color=c_solid, opacity=0.5, showscale=False,
                                    showlegend=False,
                                    lighting=dict(ambient=0.5, diffuse=0.7, specular=0.4)))
            lx, ly, lz = _box_lines(cx, cy, cz, hx, hy, hz)
            traces.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines',
                                       line=dict(color=c_wire, width=2),
                                       showlegend=False, hoverinfo='skip'))
    return traces


def _make_layout(title_text, n_frames):
    return dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        margin=dict(l=0, r=0, t=55, b=10),
        height=700,
        title=dict(
            text=f'<b>{title_text}</b>',
            font=dict(color='#00d4ff', size=14, family='monospace'),
            x=0.5, y=0.97,
        ),
        legend=dict(
            bgcolor='rgba(11,16,32,0.8)', bordercolor='#162040', borderwidth=1,
            font=dict(color='#c0d0f0', size=10, family='monospace'),
        ),
        scene=dict(
            xaxis=_axis_style('X'), yaxis=_axis_style('Y'), zaxis=_axis_style('Z'),
            aspectmode='cube',
            bgcolor=DARK_PANEL,
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.3), center=dict(x=0, y=0, z=-0.1)),
        ),
        updatemenus=[dict(
            type='buttons', showactive=False,
            y=0.02, x=0.5, xanchor='center', yanchor='bottom',
            bgcolor='#111527', bordercolor='#1e2a4a',
            font=dict(color='#00d4ff', family='monospace', size=11),
            buttons=[
                dict(label='▶  PLAY', method='animate',
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True, mode='immediate')]),
                dict(label='⏸  PAUSE', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
            ],
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='Frame: ', font=dict(color='#4a6a9a', family='monospace', size=10)),
            pad=dict(t=5, b=10),
            bgcolor='#0d1220', bordercolor='#1e2a4a', tickcolor='#1e2a4a',
            font=dict(color='#4a6a9a', family='monospace', size=8),
            steps=[dict(
                method='animate',
                args=[[str(i)], dict(mode='immediate',
                                     frame=dict(duration=80, redraw=True),
                                     transition=dict(duration=0))],
                label=str(i),
            ) for i in range(n_frames)],
        )],
    )


# ── PUBLIC: Static 3D animated figure ────────────────────────────────────────

def create_plotly_3d_static(path, obstacles, start, goal, run_num):
    """
    Animate the agent moving along the pre-computed path.
    Obstacles stay fixed. Returns HTML string.
    """
    n_total = len(path)
    n_frames = min(80, n_total)
    frame_indices = np.linspace(0, n_total-1, n_frames, dtype=int)

    # Fixed traces: bounding box, start, goal, obstacles
    fixed = []

    # Bounding box
    lx, ly, lz = _box_lines(250, 250, 250, 250, 250, 250)
    fixed.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines',
                              line=dict(color='#1e2a4a', width=1),
                              showlegend=False, hoverinfo='skip', name='Bounds'))

    # Start
    sx, sy, sz, si, sj, sk = _sphere_mesh(start[0], start[1], start[2], 8, n=10)
    fixed.append(go.Mesh3d(x=sx, y=sy, z=sz, i=si, j=sj, k=sk,
                           color='#00e58a', opacity=0.9, showscale=False,
                           name='Start', showlegend=True,
                           lighting=dict(ambient=0.6, diffuse=0.9, specular=0.7)))
    fixed.append(go.Scatter3d(x=[start[0]], y=[start[1]], z=[start[2]+20],
                              mode='text', text=['S'],
                              textfont=dict(color='#00e58a', size=12, family='monospace'),
                              showlegend=False, hoverinfo='skip'))

    # Goal
    gx, gy, gz, gi, gj, gk = _sphere_mesh(goal[0], goal[1], goal[2], 10, n=12)
    fixed.append(go.Mesh3d(x=gx, y=gy, z=gz, i=gi, j=gj, k=gk,
                           color='#ffd700', opacity=0.9, showscale=False,
                           name='Goal', showlegend=True,
                           lighting=dict(ambient=0.6, diffuse=0.9, specular=0.7)))
    fixed.append(go.Scatter3d(x=[goal[0]], y=[goal[1]], z=[goal[2]+22],
                              mode='text', text=['G'],
                              textfont=dict(color='#ffd700', size=12, family='monospace'),
                              showlegend=False, hoverinfo='skip'))

    # Full path ghost (static, faint)
    cols_all = _path_colors(len(path))
    fixed.append(go.Scatter3d(
        x=path[:,0].tolist(), y=path[:,1].tolist(), z=path[:,2].tolist(),
        mode='lines',
        line=dict(color=cols_all, width=2),
        opacity=0.18,
        showlegend=False, hoverinfo='skip', name='Full Path',
    ))

    # Obstacles (fixed)
    fixed += _static_obs_traces(obstacles)
    n_fixed = len(fixed)

    # Build per-frame dynamic traces: path trail + agent sphere + glow
    def _frame_dyn(fi):
        fp = path[:fi+1]
        dyn = []
        # Gradient path trail
        if len(fp) >= 2:
            cols = _path_colors(len(fp))
            dyn.append(go.Scatter3d(
                x=fp[:,0].tolist(), y=fp[:,1].tolist(), z=fp[:,2].tolist(),
                mode='lines+markers',
                line=dict(color=cols, width=5),
                marker=dict(size=1.5, color=cols),
                showlegend=False, hoverinfo='skip',
            ))
        else:
            dyn.append(go.Scatter3d(x=[], y=[], z=[], mode='lines',
                                    showlegend=False, hoverinfo='skip'))
        # Agent
        pos = fp[-1]
        ax, ay, az, ai, aj, ak = _sphere_mesh(pos[0], pos[1], pos[2], 7, n=10)
        dyn.append(go.Mesh3d(x=ax, y=ay, z=az, i=ai, j=aj, k=ak,
                             color='#ff4d6d', opacity=1.0, showscale=False,
                             name='Agent', showlegend=False,
                             lighting=dict(ambient=0.6, diffuse=0.8, specular=0.6)))
        # Glow
        gx2, gy2, gz2, gi2, gj2, gk2 = _sphere_mesh(pos[0], pos[1], pos[2], 15, n=10)
        dyn.append(go.Mesh3d(x=gx2, y=gy2, z=gz2, i=gi2, j=gj2, k=gk2,
                             color='#ff4d6d', opacity=0.10, showscale=False,
                             showlegend=False, hoverinfo='skip'))
        return dyn

    first_dyn = _frame_dyn(frame_indices[0])

    frames = []
    for fi_idx, fi in enumerate(frame_indices):
        dyn = _frame_dyn(fi)
        frames.append(go.Frame(
            data=dyn,
            traces=list(range(n_fixed, n_fixed + len(dyn))),
            name=str(fi_idx),
        ))

    fig = go.Figure(data=fixed + first_dyn, frames=frames)
    fig.update_layout(**_make_layout(f'RUN {run_num:02d} · 3D STATIC APF — PATH PLAYBACK', len(frames)))
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# ── PUBLIC: Dynamic 3D animated figure ───────────────────────────────────────

def create_plotly_3d_dynamic(path, obs_history, start, goal, run_num):
    """
    Animate agent + moving obstacles simultaneously.
    Returns HTML string.
    """
    n_total = len(path)
    n_frames = min(60, n_total)
    frame_indices = np.linspace(0, n_total-1, n_frames, dtype=int)

    # Fixed traces
    fixed = []
    lx, ly, lz = _box_lines(250, 250, 250, 250, 250, 250)
    fixed.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines',
                              line=dict(color='#1e2a4a', width=1),
                              showlegend=False, hoverinfo='skip'))
    sx, sy, sz, si, sj, sk = _sphere_mesh(start[0], start[1], start[2], 8, n=10)
    fixed.append(go.Mesh3d(x=sx, y=sy, z=sz, i=si, j=sj, k=sk,
                           color='#00e58a', opacity=0.9, showscale=False,
                           name='Start', showlegend=True,
                           lighting=dict(ambient=0.6, diffuse=0.9, specular=0.7)))
    fixed.append(go.Scatter3d(x=[start[0]], y=[start[1]], z=[start[2]+20],
                              mode='text', text=['S'],
                              textfont=dict(color='#00e58a', size=12, family='monospace'),
                              showlegend=False, hoverinfo='skip'))
    gx, gy, gz, gi, gj, gk = _sphere_mesh(goal[0], goal[1], goal[2], 10, n=12)
    fixed.append(go.Mesh3d(x=gx, y=gy, z=gz, i=gi, j=gj, k=gk,
                           color='#ffd700', opacity=0.9, showscale=False,
                           name='Goal', showlegend=True,
                           lighting=dict(ambient=0.6, diffuse=0.9, specular=0.7)))
    fixed.append(go.Scatter3d(x=[goal[0]], y=[goal[1]], z=[goal[2]+22],
                              mode='text', text=['G'],
                              textfont=dict(color='#ffd700', size=12, family='monospace'),
                              showlegend=False, hoverinfo='skip'))
    n_fixed = len(fixed)

    def _frame_dyn(fi):
        fp  = path[:fi+1]
        fob = obs_history[fi]
        dyn = []
        # Path trail
        if len(fp) >= 2:
            cols = _path_colors(len(fp))
            dyn.append(go.Scatter3d(
                x=fp[:,0].tolist(), y=fp[:,1].tolist(), z=fp[:,2].tolist(),
                mode='lines+markers',
                line=dict(color=cols, width=5),
                marker=dict(size=1.5, color=cols),
                showlegend=False, hoverinfo='skip',
            ))
        else:
            dyn.append(go.Scatter3d(x=[], y=[], z=[], mode='lines',
                                    showlegend=False, hoverinfo='skip'))
        # Agent
        pos = fp[-1]
        ax, ay, az, ai, aj, ak = _sphere_mesh(pos[0], pos[1], pos[2], 7, n=10)
        dyn.append(go.Mesh3d(x=ax, y=ay, z=az, i=ai, j=aj, k=ak,
                             color='#ff4d6d', opacity=1.0, showscale=False,
                             showlegend=False,
                             lighting=dict(ambient=0.6, diffuse=0.8, specular=0.6)))
        gx2, gy2, gz2, gi2, gj2, gk2 = _sphere_mesh(pos[0], pos[1], pos[2], 15, n=10)
        dyn.append(go.Mesh3d(x=gx2, y=gy2, z=gz2, i=gi2, j=gj2, k=gk2,
                             color='#ff4d6d', opacity=0.10, showscale=False,
                             showlegend=False, hoverinfo='skip'))
        # Dynamic obstacles
        dyn += _dyn_obs_traces(fob)
        return dyn

    first_dyn = _frame_dyn(frame_indices[0])

    frames = []
    for fi_idx, fi in enumerate(frame_indices):
        dyn = _frame_dyn(fi)
        frames.append(go.Frame(
            data=dyn,
            traces=list(range(n_fixed, n_fixed + len(dyn))),
            name=str(fi_idx),
        ))

    fig = go.Figure(data=fixed + first_dyn, frames=frames)
    fig.update_layout(**_make_layout(f'RUN {run_num:02d} · 3D DYNAMIC APF — PATH PLAYBACK', len(frames)))
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
EOF

# ── simulation/metrics.py ─────────────────────────────────────────────────────
cat << 'EOF' > "$OUTDIR/simulation/metrics.py"
"""
Appends run metrics to metrics_log.xlsx — never overwrites existing data.
"""

import os
import datetime
import openpyxl

EXCEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'metrics_log.xlsx')

HEADERS = [
    'Timestamp', 'Mode', 'Run #',
    'Success', 'Steps', 'Computation Time (ms)',
    'Avg CPU (%)', 'Peak RAM (GB)', 'System Power (W)',
    'Path Length (m)', 'Efficiency (%)',
]


def _get_or_create_workbook():
    if os.path.exists(EXCEL_PATH):
        wb = openpyxl.load_workbook(EXCEL_PATH)
        ws = wb.active
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'APF Metrics'
        # Style the header row
        from openpyxl.styles import Font, PatternFill, Alignment
        header_fill = PatternFill(start_color='0D1220', end_color='0D1220', fill_type='solid')
        header_font = Font(bold=True, color='00D4FF', name='Consolas')
        for col_idx, h in enumerate(HEADERS, start=1):
            cell = ws.cell(row=1, column=col_idx, value=h)
            cell.fill  = header_fill
            cell.font  = header_font
            cell.alignment = Alignment(horizontal='center')
        # Column widths
        widths = [20, 12, 8, 10, 8, 22, 12, 14, 18, 16, 14]
        for col_idx, w in enumerate(widths, start=1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = w
    return wb, wb.active


def append_run(result):
    """Append a single run result dict to the Excel log."""
    wb, ws = _get_or_create_workbook()
    row = [
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        result.get('mode', ''),
        result.get('run', ''),
        'TRUE' if result.get('success') else 'FALSE',
        result.get('steps', ''),
        result.get('time_ms', ''),
        result.get('avg_cpu', ''),
        result.get('peak_ram_gb', ''),
        result.get('power_w', ''),
        result.get('path_length', ''),
        result.get('efficiency', ''),
    ]
    ws.append(row)
    wb.save(EXCEL_PATH)


def append_runs(results):
    """Append multiple run results in one save."""
    wb, ws = _get_or_create_workbook()
    for result in results:
        row = [
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            result.get('mode', ''),
            result.get('run', ''),
            'TRUE' if result.get('success') else 'FALSE',
            result.get('steps', ''),
            result.get('time_ms', ''),
            result.get('avg_cpu', ''),
            result.get('peak_ram_gb', ''),
            result.get('power_w', ''),
            result.get('path_length', ''),
            result.get('efficiency', ''),
        ]
        ws.append(row)
    wb.save(EXCEL_PATH)
EOF

# ── app.py ────────────────────────────────────────────────────────────────────
cat << 'EOF' > "$OUTDIR/app.py"
"""
Flask app — APF Unified (3D Static + 3D Dynamic).
Metrics appended to metrics_log.xlsx after every run.
"""

import os
import time
import threading
import numpy as np
import psutil
from flask import Flask, render_template, request

from simulation.core import (
    APFConfig, STATIC_OBSTACLES_3D,
    assign_velocities_3d,
    simulate_static_apf, simulate_dynamic_apf,
    path_length, MAX_STEPS,
)
from simulation.visualization import create_plotly_3d_static, create_plotly_3d_dynamic
from simulation.metrics import append_runs

app = Flask(__name__)
app.secret_key = os.urandom(24)

START_3D = np.array([35.0, 35.0, 35.0])
GOAL_3D  = np.array([465.0, 465.0, 465.0])


def _collect_cpu_and_ram(stop_event, cpu_samples, ram_samples):
    proc = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_samples.append(psutil.cpu_percent(interval=None))
        ram_samples.append(proc.memory_info().rss / (1024 ** 3))
        time.sleep(0.05)


def run_single(mode, run_num, rng):
    cfg = APFConfig()
    cpu_samples, ram_samples = [], []
    stop_evt = threading.Event()
    monitor = threading.Thread(target=_collect_cpu_and_ram,
                               args=(stop_evt, cpu_samples, ram_samples), daemon=True)
    monitor.start()

    t0 = time.perf_counter()

    if mode == '3d_static':
        success, path = simulate_static_apf(
            START_3D.copy(), GOAL_3D.copy(), STATIC_OBSTACLES_3D, cfg, rng=rng
        )
        obs_history = None
    else:  # 3d_dynamic
        dyn_obs = assign_velocities_3d(STATIC_OBSTACLES_3D, rng)
        success, path, obs_history = simulate_dynamic_apf(
            START_3D.copy(), GOAL_3D.copy(), dyn_obs, cfg, rng=rng
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    stop_evt.set()
    monitor.join(timeout=1)

    avg_cpu   = round(float(np.mean(cpu_samples)) if cpu_samples else 0.0, 1)
    peak_ram  = round(float(max(ram_samples)) if ram_samples else 0.0, 3)
    power_w   = round(avg_cpu * 0.65, 1)

    steps = max(len(path) - 1, 1)
    L     = path_length(path)
    straight = np.linalg.norm(GOAL_3D - START_3D)
    efficiency = round(straight / L * 100, 1) if L > 0 else 0.0

    plotly_html = None
    if success:
        if mode == '3d_static':
            plotly_html = create_plotly_3d_static(path, STATIC_OBSTACLES_3D, START_3D, GOAL_3D, run_num)
        else:
            plotly_html = create_plotly_3d_dynamic(path, obs_history, START_3D, GOAL_3D, run_num)

    return {
        'run':        run_num,
        'mode':       mode,
        'success':    success,
        'steps':      steps,
        'time_ms':    round(elapsed_ms, 1),
        'avg_cpu':    avg_cpu,
        'peak_ram_gb': peak_ram,
        'power_w':    power_w,
        'path_length': round(L, 2),
        'efficiency': efficiency,
        'plotly_html': plotly_html,
    }


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/run', methods=['POST'])
def run_simulation():
    mode     = request.form.get('mode', '3d_static')
    num_runs = int(request.form.get('num_runs', 1))
    num_runs = min(max(num_runs, 1), 10)

    results    = []
    seed_base  = int(time.time() * 1000) % 10_000_000
    for i in range(1, num_runs + 1):
        rng = np.random.default_rng(seed_base + i * 7919)
        results.append(run_single(mode, i, rng))

    # Append all results to Excel
    append_runs(results)

    success_count = sum(1 for r in results if r['success'])
    summary = {
        'mode':          mode,
        'num_runs':      num_runs,
        'success_count': success_count,
        'success_rate':  round(success_count / num_runs * 100, 1),
        'avg_steps':     round(sum(r['steps'] for r in results) / num_runs, 1),
        'avg_time_ms':   round(sum(r['time_ms'] for r in results) / num_runs, 1),
        'avg_efficiency':round(sum(r['efficiency'] for r in results) / num_runs, 1),
        'avg_cpu':       round(sum(r['avg_cpu'] for r in results) / num_runs, 1),
        'peak_ram_gb':   round(max(r['peak_ram_gb'] for r in results), 3),
    }
    return render_template('results.html', results=results, summary=summary)


if __name__ == '__main__':
    app.run(debug=True)
EOF

# ── templates/index.html ──────────────────────────────────────────────────────
cat << 'EOF' > "$OUTDIR/templates/index.html"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>APF Unified Simulation</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:     #060a12; --panel: #0b1020; --border: #162040;
      --accent: #00d4ff; --accent2: #7b2fff; --glow: #ff4d6d;
      --text:   #c0d0f0; --muted:  #3a5070;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg); color: var(--text);
      font-family: 'Syne', sans-serif; min-height: 100vh;
      display: flex; align-items: center; justify-content: center;
      overflow: hidden; position: relative;
    }
    #canvas { position: fixed; inset: 0; pointer-events: none; z-index: 0; }
    .blob { position: fixed; border-radius: 50%; filter: blur(80px); pointer-events: none; z-index: 0; }
    .blob-1 { width:500px;height:400px;top:-100px;left:-120px;
               background:radial-gradient(circle,rgba(0,212,255,0.09) 0%,transparent 70%); }
    .blob-2 { width:400px;height:500px;bottom:-80px;right:-100px;
               background:radial-gradient(circle,rgba(123,47,255,0.10) 0%,transparent 70%); }

    .card-shell {
      position:relative; z-index:1; width:100%; max-width:520px; margin:1rem;
      background:rgba(11,16,32,0.92); border:1px solid var(--border);
      border-radius:20px; padding:2.8rem 3rem; backdrop-filter:blur(20px);
      box-shadow:0 0 0 1px rgba(0,212,255,0.04),0 20px 60px rgba(0,0,0,0.6),inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .card-shell::before {
      content:''; position:absolute; top:0;left:10%;right:10%;height:1px;
      background:linear-gradient(90deg,transparent,var(--accent),var(--accent2),transparent);
    }
    .card-shell::after {
      content:''; position:absolute; inset:0; border-radius:20px;
      background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,212,255,0.012) 3px,rgba(0,212,255,0.012) 4px);
      pointer-events:none;
    }

    .brand { font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.22em;
             color:var(--muted); text-transform:uppercase; margin-bottom:0.6rem;
             display:flex; align-items:center; gap:0.5rem; }
    .brand::before { content:''; display:inline-block; width:20px;height:1px;
                     background:var(--accent); opacity:0.5; }
    h1 { font-size:2.4rem; font-weight:800; line-height:1.05; letter-spacing:-0.02em;
         background:linear-gradient(135deg,#00d4ff 0%,#7b2fff 55%,#ff4d6d 100%);
         -webkit-background-clip:text; -webkit-text-fill-color:transparent;
         background-clip:text; margin-bottom:0.4rem; }
    .subtitle { color:var(--muted); font-size:0.82rem; font-family:'Space Mono',monospace;
                letter-spacing:0.04em; margin-bottom:2.2rem; }

    .field-label { font-size:0.65rem; font-family:'Space Mono',monospace; font-weight:700;
                   letter-spacing:0.18em; text-transform:uppercase; color:var(--muted);
                   margin-bottom:0.65rem; display:block; }

    /* Mode cards — 2 columns */
    .mode-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; margin-bottom:1.6rem; }
    .mode-card input[type="radio"] { display:none; }
    .mode-card label {
      display:flex; flex-direction:column; align-items:center; gap:0.45rem;
      padding:1.2rem 0.5rem; border:1px solid var(--border); border-radius:12px;
      cursor:pointer; transition:all 0.25s cubic-bezier(0.4,0,0.2,1);
      background:rgba(255,255,255,0.015); font-size:0.78rem; font-weight:700;
      letter-spacing:0.08em; text-transform:uppercase; font-family:'Space Mono',monospace;
      position:relative; overflow:hidden;
    }
    .mode-card label .icon { font-size:1.8rem; line-height:1; filter:grayscale(0.4); transition:filter 0.25s; }
    .mode-card label .desc { font-size:0.6rem; color:var(--muted); font-weight:400;
                              letter-spacing:0.06em; transition:color 0.25s; }
    .mode-card input:checked + label {
      border-color:var(--accent); background:rgba(0,212,255,0.06); color:var(--accent);
      box-shadow:0 0 24px rgba(0,212,255,0.12),inset 0 0 20px rgba(0,212,255,0.03);
    }
    .mode-card input:checked + label .icon { filter:none; }
    .mode-card input:checked + label .desc { color:rgba(0,212,255,0.6); }
    .mode-card label:hover { border-color:rgba(0,212,255,0.3); background:rgba(0,212,255,0.03); }

    /* Runs */
    .runs-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; margin-bottom:2rem; }
    .runs-card input[type="radio"] { display:none; }
    .runs-card label {
      display:flex; flex-direction:column; align-items:center; justify-content:center; gap:0.2rem;
      padding:0.9rem; border:1px solid var(--border); border-radius:10px; cursor:pointer;
      transition:all 0.25s; background:rgba(255,255,255,0.015);
      font-family:'Space Mono',monospace;
    }
    .runs-card label .runs-num { font-size:1.6rem; font-weight:700; line-height:1; color:var(--text); }
    .runs-card label .runs-lbl { font-size:0.58rem; letter-spacing:0.14em; color:var(--muted); text-transform:uppercase; }
    .runs-card input:checked + label { border-color:var(--accent2); background:rgba(123,47,255,0.08);
      box-shadow:0 0 24px rgba(123,47,255,0.12),inset 0 0 20px rgba(123,47,255,0.03); }
    .runs-card input:checked + label .runs-num { color:#b89dff; }
    .runs-card input:checked + label .runs-lbl { color:rgba(184,157,255,0.6); }
    .runs-card label:hover { border-color:rgba(123,47,255,0.3); }

    .btn-run {
      width:100%; padding:1rem; border:none; border-radius:12px;
      background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%);
      color:#fff; font-family:'Space Mono',monospace; font-size:0.85rem; font-weight:700;
      letter-spacing:0.14em; text-transform:uppercase; cursor:pointer; transition:all 0.2s;
      position:relative; overflow:hidden;
      box-shadow:0 4px 20px rgba(0,212,255,0.2),0 4px 40px rgba(123,47,255,0.15);
    }
    .btn-run::after { content:''; position:absolute; inset:0;
      background:linear-gradient(135deg,rgba(255,255,255,0.12) 0%,transparent 60%); }
    .btn-run:hover { transform:translateY(-2px);
      box-shadow:0 8px 30px rgba(0,212,255,0.3),0 8px 50px rgba(123,47,255,0.2); }
    .btn-run:active { transform:translateY(0); }

    .loading-overlay {
      display:none; position:fixed; inset:0; background:rgba(6,10,18,0.9); z-index:999;
      align-items:center; justify-content:center; flex-direction:column; gap:1.5rem;
      backdrop-filter:blur(8px);
    }
    .loading-overlay.active { display:flex; }
    .loader-ring { position:relative; width:64px; height:64px; }
    .loader-ring::before,.loader-ring::after {
      content:''; position:absolute; inset:0; border-radius:50%; border:2px solid transparent;
    }
    .loader-ring::before { border-top-color:var(--accent); animation:spin 0.9s linear infinite; }
    .loader-ring::after  { border-bottom-color:var(--accent2); animation:spin 1.4s linear infinite reverse; }
    @keyframes spin { to { transform:rotate(360deg); } }
    .loading-text { font-family:'Space Mono',monospace; font-size:0.75rem; color:var(--accent);
                    letter-spacing:0.2em; text-transform:uppercase; animation:pulse 1.5s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
    .divider { height:1px; background:linear-gradient(90deg,transparent,var(--border),transparent); margin:1.8rem 0; }

    .excel-note {
      margin-top: 1.2rem;
      font-family: 'Space Mono', monospace;
      font-size: 0.58rem;
      color: var(--muted);
      letter-spacing: 0.1em;
      text-align: center;
    }
    .excel-note span { color: #00e58a; }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>

  <div class="loading-overlay" id="loadingOverlay">
    <div class="loader-ring"></div>
    <div class="loading-text">Running Simulation…</div>
  </div>

  <div class="card-shell">
    <div class="brand">Artificial Potential Field</div>
    <h1>APF Sim<br>Engine</h1>
    <p class="subtitle">// path planning · 3D static & dynamic · play/pause</p>

    <form method="POST" action="/run" id="simForm">
      <span class="field-label">Simulation Mode</span>
      <div class="mode-grid">
        <div class="mode-card">
          <input type="radio" name="mode" id="mode_3ds" value="3d_static" checked>
          <label for="mode_3ds">
            <span class="icon">🧊</span>
            3D Static
            <span class="desc">fixed obstacles</span>
          </label>
        </div>
        <div class="mode-card">
          <input type="radio" name="mode" id="mode_3dd" value="3d_dynamic">
          <label for="mode_3dd">
            <span class="icon">🌀</span>
            3D Dynamic
            <span class="desc">moving obstacles</span>
          </label>
        </div>
      </div>

      <span class="field-label">Number of Runs</span>
      <div class="runs-grid">
        <div class="runs-card">
          <input type="radio" name="num_runs" id="runs1" value="1" checked>
          <label for="runs1">
            <span class="runs-num">1</span>
            <span class="runs-lbl">Run</span>
          </label>
        </div>
        <div class="runs-card">
          <input type="radio" name="num_runs" id="runs10" value="10">
          <label for="runs10">
            <span class="runs-num">10</span>
            <span class="runs-lbl">Runs</span>
          </label>
        </div>
      </div>

      <button type="submit" class="btn-run" onclick="showLoading()">
        ▶ &nbsp; Run Simulation
      </button>
    </form>

    <div class="excel-note">
      metrics auto-saved to <span>metrics_log.xlsx</span> after every run
    </div>
  </div>

  <script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let W, H, particles = [];
    function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
    resize(); window.addEventListener('resize', resize);
    const COLORS = ['#00d4ff','#7b2fff','#ff4d6d','#00e58a'];
    for (let i = 0; i < 60; i++) {
      particles.push({ x:Math.random()*window.innerWidth, y:Math.random()*window.innerHeight,
        r:Math.random()*1.2+0.4, vx:(Math.random()-0.5)*0.3, vy:(Math.random()-0.5)*0.3,
        color:COLORS[Math.floor(Math.random()*COLORS.length)], alpha:Math.random()*0.5+0.1 });
    }
    function draw() {
      ctx.clearRect(0,0,W,H);
      for (let i=0;i<particles.length;i++) for (let j=i+1;j<particles.length;j++) {
        const dx=particles[i].x-particles[j].x, dy=particles[i].y-particles[j].y, d=Math.sqrt(dx*dx+dy*dy);
        if (d<120) { ctx.beginPath(); ctx.strokeStyle=`rgba(0,212,255,${0.06*(1-d/120)})`; ctx.lineWidth=0.5;
          ctx.moveTo(particles[i].x,particles[i].y); ctx.lineTo(particles[j].x,particles[j].y); ctx.stroke(); }
      }
      for (const p of particles) {
        ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
        ctx.fillStyle=p.color.replace(')' ,`,${p.alpha})`).replace('rgb','rgba'); ctx.fill();
        p.x+=p.vx; p.y+=p.vy;
        if(p.x<0)p.x=W; if(p.x>W)p.x=0; if(p.y<0)p.y=H; if(p.y>H)p.y=0;
      }
      requestAnimationFrame(draw);
    }
    draw();
    function showLoading() { document.getElementById('loadingOverlay').classList.add('active'); }
  </script>
</body>
</html>
EOF

# ── templates/results.html ────────────────────────────────────────────────────
cat << 'EOF' > "$OUTDIR/templates/results.html"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>APF Results</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:#060a12; --panel:#0b1020; --panel2:#0e1428; --border:#162040;
      --accent:#00d4ff; --accent2:#7b2fff; --success:#00e58a; --fail:#ff4d6d;
      --text:#c0d0f0; --muted:#3a5070;
    }
    *, *::before, *::after { box-sizing:border-box; }
    body { background:var(--bg); color:var(--text); font-family:'Syne',sans-serif;
           min-height:100vh; padding:2.5rem 1rem 5rem; }
    body::before { content:''; position:fixed; inset:0;
      background:radial-gradient(ellipse 70% 50% at 5% 5%,rgba(0,212,255,0.04) 0%,transparent 55%),
                 radial-gradient(ellipse 60% 60% at 95% 95%,rgba(123,47,255,0.05) 0%,transparent 55%);
      pointer-events:none; z-index:0; }
    .container { max-width:1200px; position:relative; z-index:1; }

    .page-header { display:flex; align-items:center; gap:1rem; flex-wrap:wrap;
                   margin-bottom:2.5rem; padding-bottom:1.2rem; border-bottom:1px solid var(--border); }
    .page-header h1 { font-size:1.9rem; font-weight:800; letter-spacing:-0.02em;
      background:linear-gradient(135deg,var(--accent),var(--accent2));
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
    .badge-mode { font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.18em;
      padding:0.3rem 0.8rem; border-radius:20px; background:rgba(0,212,255,0.08);
      color:var(--accent); border:1px solid rgba(0,212,255,0.2); text-transform:uppercase; }
    .back-link { margin-left:auto; color:var(--muted); text-decoration:none; font-size:0.78rem;
      font-family:'Space Mono',monospace; font-weight:700; letter-spacing:0.06em;
      border:1px solid var(--border); padding:0.35rem 0.8rem; border-radius:8px; transition:color 0.2s; }
    .back-link:hover { color:var(--accent); border-color:rgba(0,212,255,0.3); }

    .excel-saved { display:inline-flex; align-items:center; gap:0.5rem;
      font-family:'Space Mono',monospace; font-size:0.62rem; color:#00e58a;
      border:1px solid rgba(0,229,138,0.25); padding:0.3rem 0.8rem; border-radius:20px;
      background:rgba(0,229,138,0.07); }

    .summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                    gap:0.85rem; margin-bottom:2.5rem; }
    .stat-card { background:var(--panel); border:1px solid var(--border); border-radius:14px;
      padding:1.3rem 1rem; text-align:center; position:relative; overflow:hidden;
      transition:border-color 0.2s,box-shadow 0.2s; }
    .stat-card:hover { border-color:rgba(0,212,255,0.25); box-shadow:0 0 20px rgba(0,212,255,0.05); }
    .stat-card::before { content:''; position:absolute; top:0;left:15%;right:15%;height:1px;
      background:linear-gradient(90deg,transparent,var(--accent),transparent); opacity:0.35; }
    .stat-value { font-size:1.8rem; font-weight:800; line-height:1; margin-bottom:0.4rem;
                  font-family:'Space Mono',monospace; }
    .stat-value.c-success { color:var(--success); }
    .stat-value.c-accent  { color:var(--accent); }
    .stat-value.c-purple  { color:#b89dff; }
    .stat-label { font-size:0.6rem; font-family:'Space Mono',monospace; letter-spacing:0.16em;
                  text-transform:uppercase; color:var(--muted); font-weight:700; }

    .section-title { font-size:0.62rem; font-family:'Space Mono',monospace; letter-spacing:0.22em;
      text-transform:uppercase; color:var(--muted); margin-bottom:1rem;
      display:flex; align-items:center; gap:0.6rem; }
    .section-title::after { content:''; flex:1; height:1px;
      background:linear-gradient(90deg,var(--border),transparent); }

    .table-wrap { background:var(--panel); border:1px solid var(--border); border-radius:14px;
                  overflow:hidden; margin-bottom:3rem; }
    table { width:100%; border-collapse:collapse; }
    thead th { background:var(--panel2); padding:0.8rem 1.2rem; font-size:0.62rem;
      font-family:'Space Mono',monospace; font-weight:700; letter-spacing:0.14em;
      text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--border); }
    tbody tr { border-bottom:1px solid rgba(22,32,64,0.7); transition:background 0.15s; }
    tbody tr:last-child { border-bottom:none; }
    tbody tr:hover { background:rgba(0,212,255,0.025); }
    tbody td { padding:0.8rem 1.2rem; font-size:0.85rem; }
    .badge-success { background:rgba(0,229,138,0.1); color:var(--success);
      border:1px solid rgba(0,229,138,0.22); border-radius:20px; padding:0.2rem 0.65rem;
      font-size:0.62rem; font-family:'Space Mono',monospace; font-weight:700; letter-spacing:0.08em; }
    .badge-fail { background:rgba(255,77,109,0.09); color:var(--fail);
      border:1px solid rgba(255,77,109,0.18); border-radius:20px; padding:0.2rem 0.65rem;
      font-size:0.62rem; font-family:'Space Mono',monospace; font-weight:700; letter-spacing:0.08em; }
    .mono { font-family:'Space Mono',monospace; font-size:0.8rem; }

    .viz-card { background:var(--panel); border:1px solid var(--border); border-radius:16px;
                overflow:hidden; margin-bottom:2rem; }
    .viz-header { background:var(--panel2); padding:0.75rem 1.2rem; font-size:0.62rem;
      font-family:'Space Mono',monospace; letter-spacing:0.18em; color:var(--muted);
      border-bottom:1px solid var(--border); display:flex; align-items:center; gap:0.6rem; }
    .viz-dot { width:7px;height:7px;border-radius:50%;background:var(--accent);
               box-shadow:0 0 8px var(--accent); animation:glow-pulse 2s ease-in-out infinite; }
    @keyframes glow-pulse { 0%,100%{box-shadow:0 0 6px var(--accent);opacity:1}
                             50%{box-shadow:0 0 14px var(--accent);opacity:0.7} }
    .viz-type-badge { margin-left:auto; font-size:0.58rem; padding:0.15rem 0.55rem; border-radius:10px;
      background:rgba(0,212,255,0.08); border:1px solid rgba(0,212,255,0.15);
      color:var(--accent); letter-spacing:0.12em; }
    .viz-body { padding:1.2rem; }
    .plotly-wrap { border-radius:10px; overflow:hidden; border:1px solid var(--border); background:var(--bg); }
  </style>
</head>
<body>
<div class="container mx-auto">

  <div class="page-header">
    <h1>Simulation Results</h1>
    <span class="badge-mode">{{ summary.mode | upper | replace('_', ' ') }}</span>
    <span class="excel-saved">✓ saved to metrics_log.xlsx</span>
    <a href="/" class="back-link">← New Run</a>
  </div>

  <!-- Summary cards -->
  <div class="summary-grid">
    <div class="stat-card">
      <div class="stat-value c-success">{{ summary.success_count }}/{{ summary.num_runs }}</div>
      <div class="stat-label">Successes</div>
    </div>
    <div class="stat-card">
      <div class="stat-value c-accent">{{ summary.success_rate }}%</div>
      <div class="stat-label">Success Rate</div>
    </div>
    <div class="stat-card">
      <div class="stat-value c-purple">{{ summary.avg_steps }}</div>
      <div class="stat-label">Avg Steps</div>
    </div>
    <div class="stat-card">
      <div class="stat-value c-accent">{{ summary.avg_time_ms }}</div>
      <div class="stat-label">Avg Time (ms)</div>
    </div>
    <div class="stat-card">
      <div class="stat-value c-purple">{{ summary.avg_efficiency }}%</div>
      <div class="stat-label">Avg Efficiency</div>
    </div>
    <div class="stat-card">
      <div class="stat-value c-accent">{{ summary.avg_cpu }}</div>
      <div class="stat-label">Avg CPU (%)</div>
    </div>
    <div class="stat-card">
      <div class="stat-value c-purple">{{ summary.peak_ram_gb }}</div>
      <div class="stat-label">Peak RAM (GB)</div>
    </div>
  </div>

  <!-- Metrics table -->
  <div class="section-title">Run Metrics</div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Status</th><th>Steps</th><th>Time (ms)</th>
          <th>CPU (%)</th><th>RAM (GB)</th><th>Power (W)</th>
          <th>Path Length</th><th>Efficiency %</th>
        </tr>
      </thead>
      <tbody>
        {% for r in results %}
        <tr>
          <td class="mono">{{ "%02d" % r.run }}</td>
          <td>{% if r.success %}<span class="badge-success">SUCCESS</span>
              {% else %}<span class="badge-fail">FAILED</span>{% endif %}</td>
          <td class="mono">{{ r.steps }}</td>
          <td class="mono">{{ r.time_ms }}</td>
          <td class="mono">{{ r.avg_cpu }}</td>
          <td class="mono">{{ r.peak_ram_gb }}</td>
          <td class="mono">{{ r.power_w }}</td>
          <td class="mono">{{ r.path_length }}</td>
          <td class="mono">{{ r.efficiency }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Visualizations with play/pause -->
  {% set viz_runs = results | selectattr('plotly_html') | list %}
  {% if viz_runs %}
  <div class="section-title">Path Playback (Play / Pause / Scrub)</div>
  {% for r in viz_runs %}
  <div class="viz-card">
    <div class="viz-header">
      <div class="viz-dot"></div>
      RUN {{ "%02d" % r.run }} — {{ r.mode | upper | replace('_', ' ') }} · 3D PATH ANIMATION
      <span class="viz-type-badge">PLOTLY · ANIMATED</span>
    </div>
    <div class="viz-body">
      <div class="plotly-wrap">{{ r.plotly_html | safe }}</div>
    </div>
  </div>
  {% endfor %}
  {% endif %}

</div>
</body>
</html>
EOF

echo ""
echo "========================================================"
echo "  Project created at: $OUTDIR"
echo "========================================================"
echo ""
echo "To run:"
echo "  conda activate myenv"
echo "  cd $OUTDIR"
echo "  python app.py"
echo ""
echo "Then open: http://127.0.0.1:5000"
echo ""
echo "Metrics are saved/appended to: $OUTDIR/metrics_log.xlsx"
