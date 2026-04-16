"""
APF Static Environment — 5 Different Environments
Obstacles are STATIC (no movement)
Metrics: Steps, Time (ms), Avg CPU %, Peak RAM (GB), System Power (W)
Outputs: GIF (per env, if success), path PNG (per env), force heatmap PNG (per env)
"""

import os, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
from matplotlib.animation import FuncAnimation
import openpyxl
from openpyxl.styles import Font, Border, Side
import psutil

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
DT = 0.1
BOUNDS = (0.0, 500.0, 0.0, 500.0)
COLLISION_MARGIN = 6.0
MAX_STEPS = 6000

# =========================================================
# UTILITIES
# =========================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# =========================================================
# OBSTACLE GEOMETRY
# =========================================================
def dist_point_to_aabb(p, center, half_w, half_h):
    cx, cy = center
    qx = clamp(p[0], cx - half_w, cx + half_w)
    qy = clamp(p[1], cy - half_h, cy + half_h)
    closest = np.array([qx, qy], dtype=float)
    v = p - closest
    rho = np.linalg.norm(v)
    grad_rho = v / rho if rho > 1e-12 else np.array([0.0, 0.0])
    return max(rho, 0.0), grad_rho, closest

def dist_point_to_circle(p, center, radius):
    v = p - center
    d = np.linalg.norm(v)
    grad_rho = v / d if d > 1e-12 else np.array([1.0, 0.0])
    rho = max(d - radius, 0.0)
    closest = center + grad_rho * radius
    return rho, grad_rho, closest

def dist_point_to_triangle(p, vertices):
    def closest_on_segment(a, b, pt):
        ab = b - a
        t = np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-12)
        return a + clamp(t, 0.0, 1.0) * ab

    def point_in_triangle(pt, v0, v1, v2):
        def sign(pa, pb, pc):
            return (pa[0]-pc[0])*(pb[1]-pc[1]) - (pb[0]-pc[0])*(pa[1]-pc[1])
        d1 = sign(pt, v0, v1)
        d2 = sign(pt, v1, v2)
        d3 = sign(pt, v2, v0)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    v0, v1, v2 = [np.array(v, dtype=float) for v in vertices]
    best_dist, best_closest = np.inf, p.copy()
    for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
        c = closest_on_segment(a, b, p)
        d = np.linalg.norm(p - c)
        if d < best_dist:
            best_dist, best_closest = d, c
    if point_in_triangle(p, v0, v1, v2):
        grad = p - best_closest
        n = np.linalg.norm(grad)
        return 0.0, grad / (n + 1e-12), best_closest
    v = p - best_closest
    rho = np.linalg.norm(v)
    return rho, v / (rho + 1e-12), best_closest

def get_distance(p, obs):
    t = obs['type']
    if t == 'circle':
        cx, cy, r = obs['params']
        return dist_point_to_circle(p, np.array([cx, cy], float), r)
    elif t in ('square', 'rectangle'):
        cx, cy, hw, hh = obs['params']
        return dist_point_to_aabb(p, np.array([cx, cy], float), hw, hh)
    elif t == 'triangle':
        return dist_point_to_triangle(p, obs['params'])

# =========================================================
# OBSTACLE HELPER
# =========================================================
def make_triangle(cx, cy, size, angle_offset=0.0):
    angles = np.deg2rad([90, 210, 330]) + angle_offset
    return [(cx + size*np.cos(a), cy + size*np.sin(a)) for a in angles]

# =========================================================
# 5 STATIC ENVIRONMENTS
# =========================================================
ENVIRONMENTS = {
    1: {
        "name": "Corridor Gauntlet",
        "obstacles": [
            {"type": "rectangle", "params": (180.0, 280.0, 30.0, 75.0), "label": "B1"},
            {"type": "rectangle", "params": (370.0, 270.0, 30.0, 75.0), "label": "B2"},
            {"type": "square",    "params": (230.0, 180.0, 30.0, 30.0), "label": "B3"},
            {"type": "square",    "params": (270.0, 320.0, 30.0, 30.0), "label": "B4"},
            {"type": "triangle",  "params": make_triangle(185.0, 155.0, 32.5, 0.0), "label": "V1"},
            {"type": "triangle",  "params": make_triangle(260.0, 240.0, 32.5, 0.5), "label": "V2"},
            {"type": "triangle",  "params": make_triangle(325.0, 355.0, 32.5, 1.0), "label": "V3"},
            {"type": "circle",    "params": (365.0, 175.0, 25.0),  "label": "C1"},
            {"type": "circle",    "params": (70.0,  430.0, 32.5),  "label": "C2"},
            {"type": "circle",    "params": (430.0,  70.0, 32.5),  "label": "C3"},
        ]
    },
    2: {
        "name": "Dense Forest",
        "obstacles": [
            {"type": "circle",   "params": (100.0, 100.0, 20.0), "label": "C1"},
            {"type": "circle",   "params": (200.0, 150.0, 25.0), "label": "C2"},
            {"type": "circle",   "params": (300.0, 100.0, 20.0), "label": "C3"},
            {"type": "circle",   "params": (400.0, 150.0, 22.0), "label": "C4"},
            {"type": "circle",   "params": (150.0, 250.0, 18.0), "label": "C5"},
            {"type": "circle",   "params": (250.0, 300.0, 30.0), "label": "C6"},
            {"type": "circle",   "params": (350.0, 250.0, 20.0), "label": "C7"},
            {"type": "circle",   "params": (450.0, 300.0, 25.0), "label": "C8"},
            {"type": "circle",   "params": (100.0, 380.0, 22.0), "label": "C9"},
            {"type": "circle",   "params": (200.0, 420.0, 28.0), "label": "C10"},
            {"type": "circle",   "params": (320.0, 400.0, 20.0), "label": "C11"},
            {"type": "circle",   "params": (420.0, 420.0, 18.0), "label": "C12"},
        ]
    },
    3: {
        "name": "Wall Maze",
        "obstacles": [
            {"type": "rectangle", "params": (150.0, 200.0, 8.0,  80.0), "label": "W1"},
            {"type": "rectangle", "params": (250.0, 300.0, 8.0,  80.0), "label": "W2"},
            {"type": "rectangle", "params": (350.0, 200.0, 8.0,  80.0), "label": "W3"},
            {"type": "rectangle", "params": (200.0, 130.0, 60.0,  8.0), "label": "W4"},
            {"type": "rectangle", "params": (300.0, 370.0, 60.0,  8.0), "label": "W5"},
            {"type": "square",    "params": (420.0, 200.0, 25.0, 25.0), "label": "S1"},
            {"type": "square",    "params": ( 80.0, 300.0, 25.0, 25.0), "label": "S2"},
            {"type": "triangle",  "params": make_triangle(250.0, 250.0, 35.0, 0.0), "label": "T1"},
        ]
    },
    4: {
        "name": "Scattered Mixed",
        "obstacles": [
            {"type": "circle",    "params": (130.0, 130.0, 30.0), "label": "C1"},
            {"type": "circle",    "params": (370.0, 370.0, 30.0), "label": "C2"},
            {"type": "rectangle", "params": (250.0, 150.0, 40.0, 20.0), "label": "R1"},
            {"type": "rectangle", "params": (250.0, 350.0, 40.0, 20.0), "label": "R2"},
            {"type": "rectangle", "params": (150.0, 250.0, 20.0, 40.0), "label": "R3"},
            {"type": "rectangle", "params": (350.0, 250.0, 20.0, 40.0), "label": "R4"},
            {"type": "triangle",  "params": make_triangle(250.0, 250.0, 40.0, 0.0),   "label": "T1"},
            {"type": "triangle",  "params": make_triangle(130.0, 370.0, 28.0, 1.0),   "label": "T2"},
            {"type": "triangle",  "params": make_triangle(370.0, 130.0, 28.0, 2.0),   "label": "T3"},
            {"type": "square",    "params": ( 80.0,  80.0, 22.0, 22.0), "label": "S1"},
            {"type": "square",    "params": (420.0, 420.0, 22.0, 22.0), "label": "S2"},
        ]
    },
    5: {
        "name": "Narrow Passage",
        "obstacles": [
            {"type": "rectangle", "params": (100.0, 250.0, 80.0,  12.0), "label": "TL"},
            {"type": "rectangle", "params": (400.0, 250.0, 80.0,  12.0), "label": "TR"},
            {"type": "rectangle", "params": (250.0, 100.0, 12.0,  80.0), "label": "TT"},
            {"type": "rectangle", "params": (250.0, 400.0, 12.0,  80.0), "label": "TB"},
            {"type": "circle",    "params": (250.0, 250.0, 45.0), "label": "CC"},
            {"type": "circle",    "params": (160.0, 160.0, 22.0), "label": "C1"},
            {"type": "circle",    "params": (340.0, 160.0, 22.0), "label": "C2"},
            {"type": "circle",    "params": (160.0, 340.0, 22.0), "label": "C3"},
            {"type": "circle",    "params": (340.0, 340.0, 22.0), "label": "C4"},
            {"type": "triangle",  "params": make_triangle(250.0, 160.0, 25.0, 0.0), "label": "V1"},
            {"type": "triangle",  "params": make_triangle(250.0, 340.0, 25.0, 1.5), "label": "V2"},
        ]
    },
}

# =========================================================
# APF CONFIG & FUNCTIONS
# =========================================================
class APFConfig:
    def __init__(self):
        self.k_a = 5.0
        self.k_r = 0.85
        self.theta = np.deg2rad(25)
        self.goal_factor_pow = 2
        self.step = 1.8
        self.goal_tol = 18.0
        self.stuck_eps = 1e-6
        self.jitter = 0.35

def F_att(q, q_goal, k_a):
    return -k_a * (q - q_goal)

def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)

def F_rep_global(q, q_goal, obs, k_r, theta, goal_pow=2):
    rho, grad_rho, _ = get_distance(q, obs)
    rho = max(rho, 0.8)
    rho_goal = np.linalg.norm(q - q_goal) + 1e-12
    mag = k_r * (10.0 / rho**2) * (rho_goal**goal_pow)
    F_base = mag * grad_rho
    return rotation_matrix(theta) @ F_base

def F_total(q, q_goal, obstacles, cfg):
    fa = F_att(q, q_goal, cfg.k_a)
    fr = np.zeros(2, dtype=float)
    for obs in obstacles:
        fr += F_rep_global(q, q_goal, obs, cfg.k_r, cfg.theta, cfg.goal_factor_pow)
    return fa + fr, fa, fr

def in_collision(q, obstacles, margin=0.0):
    for obs in obstacles:
        rho, _, _ = get_distance(q, obs)
        if rho <= margin:
            return True
    return False

def path_length(path):
    if len(path) < 2: return 0.0
    return float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))

# =========================================================
# STATIC SIMULATION
# =========================================================
def simulate_static_apf(start, goal, obstacles, cfg, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    q = start.copy()
    path = [q.copy()]
    t_sim = 0.0
    step_count = 0
    bias = rng.normal(0, 0.06, size=2)

    while step_count < MAX_STEPS:
        t_sim += DT

        if np.linalg.norm(q - goal) <= cfg.goal_tol:
            path.append(goal.copy())
            return True, np.array(path), t_sim

        Ft, _, _ = F_total(q, goal, obstacles, cfg)
        Ft_b = Ft + bias
        nFt = np.linalg.norm(Ft_b)

        if nFt < cfg.stuck_eps:
            sd = rng.normal(0, 1, size=2)
            sd /= np.linalg.norm(sd) + 1e-12
            q_next = q + cfg.jitter * sd
        else:
            sd = Ft_b / nFt
            q_next = q + cfg.step * sd

        q_next[0] = clamp(q_next[0], *BOUNDS[:2])
        q_next[1] = clamp(q_next[1], *BOUNDS[2:])

        escape_attempts = 0
        while in_collision(q_next, obstacles, margin=COLLISION_MARGIN) and escape_attempts < 5:
            perp = np.array([-sd[1], sd[0]])
            perp /= np.linalg.norm(perp) + 1e-12
            q_next = q + cfg.step * 1.8 * perp + rng.normal(0, 0.15, size=2)
            q_next[0] = clamp(q_next[0], *BOUNDS[:2])
            q_next[1] = clamp(q_next[1], *BOUNDS[2:])
            escape_attempts += 1

        if in_collision(q_next, obstacles, margin=COLLISION_MARGIN):
            min_rho = float('inf')
            best_grad = np.array([0.0, 0.0])
            for obs in obstacles:
                rho, grad, _ = get_distance(q_next, obs)
                if rho < min_rho:
                    min_rho = rho
                    best_grad = grad
            if min_rho < COLLISION_MARGIN:
                q_next += (COLLISION_MARGIN - min_rho + 3.0) * best_grad

        q = q_next
        path.append(q.copy())
        step_count += 1

    return False, np.array(path), t_sim

# =========================================================
# GIF GENERATION
# =========================================================
def draw_obstacles(ax, obs_list):
    for obs in obs_list:
        t = obs['type']
        if t == 'circle':
            cx, cy, r = obs['params']
            ax.add_patch(Circle((cx, cy), r, facecolor="lightblue", edgecolor="blue", alpha=0.7, lw=2))
        elif t in ('square', 'rectangle'):
            cx, cy, hw, hh = obs['params']
            ax.add_patch(Rectangle((cx-hw, cy-hh), 2*hw, 2*hh, facecolor="lightgray", edgecolor="dimgray", alpha=0.7, lw=2))
        elif t == 'triangle':
            verts = obs['params']
            ax.add_patch(MplPolygon(verts, closed=True, facecolor="lightyellow", edgecolor="goldenrod", alpha=0.7, lw=2))

def create_gif(path, obstacles, start, goal, env_num, env_name):
    try:
        out_path = f"results/static_apf_env{env_num:02d}.gif"
        fig, ax = plt.subplots(figsize=(8, 8))
        frame_skip = max(1, len(path) // 150)
        path_s = path[::frame_skip]

        def update(frame):
            ax.clear()
            ax.set_xlim(*BOUNDS[:2])
            ax.set_ylim(*BOUNDS[2:])
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            ax.plot(start[0], start[1], "bo", ms=8, label="Start")
            ax.plot(goal[0], goal[1], "g*", ms=14, label="Goal")
            draw_obstacles(ax, obstacles)
            curr = np.array(path_s[:frame+1])
            ax.plot(curr[:,0], curr[:,1], "b-", lw=2.5, alpha=0.85)
            if len(curr) > 0:
                ax.plot(curr[-1,0], curr[-1,1], "ro", ms=7)
            ax.set_title(f"Env {env_num}: {env_name} | Static APF")
            ax.legend(loc="upper left", fontsize=8)
            return []

        anim = FuncAnimation(fig, update, frames=len(path_s), blit=False)
        anim.save(out_path, writer='pillow', fps=20)
        plt.close()
        print(f"  GIF saved: {out_path}")
        return True
    except Exception as e:
        print(f"  GIF failed for Env {env_num}: {e}")
        return False

def save_static_plot(path, obstacles, start, goal, env_num, env_name):
    try:
        out_path = f"results/static_apf_env{env_num:02d}_path.png"
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(*BOUNDS[:2])
        ax.set_ylim(*BOUNDS[2:])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        draw_obstacles(ax, obstacles)
        ax.plot(path[:,0], path[:,1], "b-", lw=2.0, alpha=0.85, label="Path")
        ax.plot(start[0], start[1], "bo", ms=10, label="Start")
        ax.plot(goal[0], goal[1], "g*", ms=14, label="Goal")
        ax.plot(path[-1,0], path[-1,1], "ro", ms=8, label="End")
        ax.set_title(f"Env {env_num}: {env_name} | Static APF")
        ax.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close()
        print(f"  Plot saved: {out_path}")
    except Exception as e:
        print(f"  Plot failed for Env {env_num}: {e}")

# =========================================================
# FORCE FIELD HEATMAPS  (separate PNG per env, 3 cols each)
# =========================================================
def compute_force_grids(obstacles, goal, cfg, resolution=80):
    """
    Returns three 2-D magnitude grids (resolution x resolution):
      att_mag  — attractive force magnitude   log1p-scaled
      rep_mag  — repulsive force magnitude    log1p-scaled
      tot_mag  — total (net) force magnitude  log1p-scaled
    Also returns meshgrid arrays X, Y.
    """
    xs = np.linspace(BOUNDS[0], BOUNDS[1], resolution)
    ys = np.linspace(BOUNDS[2], BOUNDS[3], resolution)
    X, Y = np.meshgrid(xs, ys)

    att_mag = np.zeros((resolution, resolution))
    rep_mag = np.zeros((resolution, resolution))
    tot_mag = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            q  = np.array([X[i, j], Y[i, j]])
            fa = F_att(q, goal, cfg.k_a)
            fr = np.zeros(2)
            for obs in obstacles:
                fr += F_rep_global(q, goal, obs, cfg.k_r, cfg.theta, cfg.goal_factor_pow)
            ft = fa + fr
            att_mag[i, j] = np.linalg.norm(fa)
            rep_mag[i, j] = np.linalg.norm(fr)
            tot_mag[i, j] = np.linalg.norm(ft)

    # log-compress to handle the extreme dynamic range near obstacle surfaces
    return X, Y, np.log1p(att_mag), np.log1p(rep_mag), np.log1p(tot_mag)


def _overlay_obstacles_on_heatmap(ax, obstacles, alpha=0.50):
    """Draw obstacle shapes as white filled patches over a heatmap."""
    for obs in obstacles:
        t = obs['type']
        if t == 'circle':
            cx, cy, r = obs['params']
            ax.add_patch(Circle((cx, cy), r,
                                facecolor='white', edgecolor='white',
                                alpha=alpha, lw=1.5, zorder=3))
        elif t in ('square', 'rectangle'):
            cx, cy, hw, hh = obs['params']
            ax.add_patch(Rectangle((cx-hw, cy-hh), 2*hw, 2*hh,
                                   facecolor='white', edgecolor='white',
                                   alpha=alpha, lw=1.5, zorder=3))
        elif t == 'triangle':
            verts = obs['params']
            ax.add_patch(MplPolygon(verts, closed=True,
                                    facecolor='white', edgecolor='white',
                                    alpha=alpha, lw=1.5, zorder=3))


def save_force_heatmap_single(edata, goal, cfg, resolution=90):
    """
    Save a separate force-field heatmap PNG for a single environment.
    Layout: 1 row x 3 columns (Attractive | Repulsive | Total).

    edata : dict with keys 'env_id', 'name', 'obstacles', 'path'
    """
    obstacles = edata["obstacles"]
    env_id    = edata["env_id"]
    env_name  = edata["name"]
    path      = edata.get("path", None)

    col_titles = [
        "Attractive Force  |F_att|",
        "Repulsive Force  |F_rep|",
        "Total (Net) Force  |F_tot|",
    ]
    col_cmaps = ["YlOrRd", "Blues", "RdYlGn_r"]

    print(f"  Computing force grids for Env {env_id}: {env_name} …")
    X, Y, att_mag, rep_mag, tot_mag = compute_force_grids(
        obstacles, goal, cfg, resolution=resolution
    )
    grids = [att_mag, rep_mag, tot_mag]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5.5),
        gridspec_kw={"wspace": 0.30},
    )

    for col_idx in range(3):
        ax   = axes[col_idx]
        grid = grids[col_idx]
        cmap = col_cmaps[col_idx]

        im = ax.imshow(
            grid,
            origin="lower",
            extent=[BOUNDS[0], BOUNDS[1], BOUNDS[2], BOUNDS[3]],
            cmap=cmap,
            interpolation="bilinear",
            aspect="equal",
        )
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log(1 + |F|)", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # obstacle outlines
        _overlay_obstacles_on_heatmap(ax, obstacles, alpha=0.45)

        # actual path taken by the robot
        if path is not None and len(path) > 1:
            ax.plot(path[:, 0], path[:, 1],
                    color="white", lw=1.3, alpha=0.75, zorder=4,
                    label="Path")

        # start / goal markers
        ax.plot(35,  35,  marker="^", color="cyan",  ms=7,
                zorder=5, lw=0, label="Start")
        ax.plot(465, 465, marker="*", color="lime",  ms=9,
                zorder=5, lw=0, label="Goal")

        ax.set_xlim(BOUNDS[0], BOUNDS[1])
        ax.set_ylim(BOUNDS[2], BOUNDS[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(col_titles[col_idx], fontsize=11, fontweight="bold", pad=7)

        # small legend only on first column
        if col_idx == 0:
            ax.legend(loc="lower right", fontsize=6.5,
                      framealpha=0.6, markerscale=0.85)

    fig.suptitle(
        f"APF Force Field Heatmaps — Env {env_id}: {env_name}\n"
        "(log-scaled magnitudes  |  white shapes = obstacles  |  white line = robot path)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    out_path = f"results/static_apf_force_heatmap_env{env_id:02d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap PNG saved: {out_path}")


def save_force_heatmaps(all_env_data, goal, cfg, resolution=90):
    """Save one heatmap PNG per environment."""
    for edata in all_env_data:
        save_force_heatmap_single(edata, goal, cfg, resolution=resolution)

# =========================================================
# POWER MONITORING
# =========================================================
def get_system_power():
    rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    try:
        if os.path.exists(rapl_path):
            with open(rapl_path) as f: e1 = int(f.read().strip())
            time.sleep(0.1)
            with open(rapl_path) as f: e2 = int(f.read().strip())
            watts = round(((e2 - e1) / 1_000_000) / 0.1, 1)
            if watts > 0:
                return watts
    except Exception:
        pass

    try:
        for bat in os.listdir("/sys/class/power_supply/"):
            if bat.startswith(("BAT", "battery")):
                path = f"/sys/class/power_supply/{bat}/power_now"
                if os.path.exists(path):
                    with open(path) as f:
                        power_uw = int(f.read().strip())
                    if power_uw > 0:
                        return round(power_uw / 1_000_000, 1)
                curr_path = f"/sys/class/power_supply/{bat}/current_now"
                volt_path = f"/sys/class/power_supply/{bat}/voltage_now"
                if os.path.exists(curr_path) and os.path.exists(volt_path):
                    with open(curr_path) as f: current_ua = int(f.read().strip())
                    with open(volt_path) as f: voltage_uv = int(f.read().strip())
                    power_w = (current_ua * voltage_uv) / 1_000_000_000_000
                    if power_w > 0:
                        return round(power_w, 1)
    except Exception:
        pass

    try:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        TDP_W = 15.0
        idle_W = TDP_W * 0.10
        return round(idle_W + (TDP_W - idle_W) * (cpu_pct / 100.0), 1)
    except:
        pass

    return "N/A"

def get_resources():
    cpu = psutil.cpu_percent(interval=0.1)
    ram_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    sys_power = get_system_power()
    return round(cpu, 1), round(ram_gb, 2), sys_power

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    START = np.array([35.0, 35.0])
    GOAL  = np.array([465.0, 465.0])
    cfg   = APFConfig()
    rng   = np.random.default_rng(42)

    print("=== APF Static Environment — 5 Environments ===")
    print("Metrics: Steps | Time (ms) | Avg CPU % | Peak RAM (GB) | System Power (W)\n")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Static_APF_5_Envs"

    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )

    row = 1
    ws.cell(row, 1, "APF Static — 5 Environments").font = Font(bold=True, size=14)
    row += 3

    summary_data = []
    all_env_data = []   # collected for the heatmap pass after all runs

    for env_id, env in ENVIRONMENTS.items():
        env_name  = env["name"]
        obstacles = env["obstacles"]

        print(f"--- Environment {env_id:02d}: {env_name} ---")
        print(f"  Obstacles: {len(obstacles)}")

        cpu_start, ram_start, sys_start = get_resources()
        t_start = time.perf_counter()

        success, path, final_t = simulate_static_apf(
            START.copy(), GOAL.copy(), obstacles, cfg, rng=rng
        )

        sim_time_ms = (time.perf_counter() - t_start) * 1000
        cpu_end, peak_ram, sys_end = get_resources()

        avg_cpu    = round((cpu_start + cpu_end) / 2, 1)
        steps      = max(len(path) - 1, 1)
        L          = path_length(path)
        straight   = np.linalg.norm(GOAL - START)
        efficiency = round(straight / L * 100, 1) if L > 0 else 0.0
        power_str  = f"{sys_end} W" if isinstance(sys_end, (int, float)) else sys_end

        print(f"  Result   → {'SUCCESS' if success else 'FAILED'}")
        print(f"  Steps    → {steps}")
        print(f"  Time     → {sim_time_ms:.1f} ms")
        print(f"  CPU      → {avg_cpu} %")
        print(f"  RAM      → {peak_ram:.2f} GB")
        print(f"  Power    → {power_str}")
        print(f"  PathLen  → {L:.2f} m   Efficiency: {efficiency} %\n")

        if success:
            create_gif(path, obstacles, START, GOAL, env_id, env_name)
        save_static_plot(path, obstacles, START, GOAL, env_id, env_name)

        # store for heatmap generation
        all_env_data.append({
            "env_id":    env_id,
            "name":      env_name,
            "obstacles": obstacles,
            "path":      path,
        })

        summary_data.append({
            "Env":            env_id,
            "Name":           env_name,
            "Num_Obstacles":  len(obstacles),
            "Success":        success,
            "Steps":          steps,
            "Sim_Time_ms":    round(sim_time_ms, 1),
            "Avg_CPU_%":      avg_cpu,
            "Peak_RAM_GB":    peak_ram,
            "System_Power_W": power_str,
            "Path_Length":    round(L, 2),
            "Efficiency_%":   efficiency,
        })

        # Per-environment Excel block
        ws.cell(row, 1, f"ENVIRONMENT {env_id:02d} — {env_name}").font = Font(bold=True, size=12)
        row += 1
        metrics = [
            ["Environment Name",        env_name],
            ["Number of Obstacles",     len(obstacles)],
            ["Success",                 "Yes" if success else "No"],
            ["Steps Taken",             steps],
            ["Computation Time (ms)",   round(sim_time_ms, 1)],
            ["Avg CPU (%)",             avg_cpu],
            ["Peak RAM (GB)",           peak_ram],
            ["System Power (W)",        power_str],
            ["Path Length (m)",         round(L, 2)],
            ["Efficiency (%)",          efficiency],
        ]
        for label, value in metrics:
            ws.cell(row, 1, label).font = Font(bold=True)
            ws.cell(row, 2, value)
            row += 1
        row += 2

    # -------------------------------------------------------
    # Force field heatmaps — separate PNG per environment
    # -------------------------------------------------------
    print("\n--- Generating force field heatmaps (separate PNG per env) ---")
    save_force_heatmaps(all_env_data, GOAL, cfg, resolution=90)

    # -------------------------------------------------------
    # Summary Table in Excel
    # -------------------------------------------------------
    row += 1
    ws.cell(row, 1, "SUMMARY — ALL 5 ENVIRONMENTS").font = Font(bold=True, size=14)
    row += 2

    headers = ["Env", "Name", "# Obstacles", "Success", "Steps",
               "Sim Time (ms)", "Avg CPU %", "Peak RAM (GB)",
               "System Power (W)", "Path Length (m)", "Efficiency (%)"]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row, col, h)
        cell.font = Font(bold=True)
        cell.border = thin_border
    row += 1

    success_count = sum(1 for d in summary_data if d["Success"])
    for d in summary_data:
        values = [
            d["Env"], d["Name"], d["Num_Obstacles"],
            "Yes" if d["Success"] else "No",
            d["Steps"], d["Sim_Time_ms"], d["Avg_CPU_%"],
            d["Peak_RAM_GB"], d["System_Power_W"],
            d["Path_Length"], d["Efficiency_%"],
        ]
        for col, val in enumerate(values, 1):
            ws.cell(row, col, val).border = thin_border
        row += 1

    row += 2
    ws.cell(row, 1,
        f"Success Rate: {success_count}/5 ({success_count/5*100:.1f}%)"
    ).font = Font(bold=True)

    col_widths = [6, 22, 14, 10, 8, 16, 12, 16, 18, 18, 14]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w

    out_xlsx = "results/static_apf_5_environments.xlsx"
    wb.save(out_xlsx)

    print("\n" + "=" * 70)
    print("ALL 5 ENVIRONMENTS COMPLETED!")
    print(f"Excel     → {out_xlsx}")
    print("GIFs      → results/static_apf_env##.gif  (successful runs)")
    print("Plots     → results/static_apf_env##_path.png")
    print("Heatmaps  → results/static_apf_force_heatmap_env##.png  (one per env)")