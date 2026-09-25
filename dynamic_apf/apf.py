"""
APF Dynamic Environment — 50 Seeded Runs (seed = run index - 1)
Obstacles bounce on collision but KEEP CONSTANT SPEED
Metrics: Steps, Time (ms), Avg CPU %, Peak RAM (GB), System Power (W)
"""

import os, time, copy, warnings, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
BASE_SPEED_SCALE = 1.25
BOUNDS = (0.0, 500.0, 0.0, 500.0)
COLLISION_MARGIN = 6.0   # planner-side safety margin (escape / push-out trigger)
COLLISION_RADIUS = 5.0   # evaluation-side collision criterion, shared with ARC (d_min_obstacle)
GOAL_RADIUS = 10.0       # shared with ARC (d_min_target)
GOAL_PROTECTION_RADIUS = 50.0
MAX_STEPS = 6000

# =========================================================
# UTILITIES
# =========================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def angle_wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))

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

def obs_centroid(obs):
    if obs['type'] == 'triangle':
        return np.array(obs['params'], dtype=float).mean(axis=0)
    return np.array(obs['params'][:2], dtype=float)

def get_aabb(obs):
    t = obs['type']
    if t == 'circle':
        cx, cy, r = obs['params']
        return cx-r, cx+r, cy-r, cy+r
    elif t in ('square', 'rectangle'):
        cx, cy, hw, hh = obs['params']
        return cx-hw, cx+hw, cy-hh, cy+hh
    elif t == 'triangle':
        xs = [v[0] for v in obs['params']]
        ys = [v[1] for v in obs['params']]
        return min(xs), max(xs), min(ys), max(ys)

# =========================================================
# STATIC OBSTACLES
# =========================================================
def make_triangle(cx, cy, size, angle_offset=0.0):
    angles = np.deg2rad([90, 210, 330]) + angle_offset
    return [(cx + size*np.cos(a), cy + size*np.sin(a)) for a in angles]

STATIC_OBSTACLES = [
    {"type": "rectangle", "params": (180.0, 280.0, 30.0, 75.0), "label": "B1", "outlier": False},
    {"type": "rectangle", "params": (370.0, 270.0, 30.0, 75.0), "label": "B2", "outlier": False},
    {"type": "square", "params": (230.0, 180.0, 30.0, 30.0), "label": "B3", "outlier": False},
    {"type": "square", "params": (270.0, 320.0, 30.0, 30.0), "label": "B4", "outlier": False},
    {"type": "triangle", "params": make_triangle(185.0, 155.0, 32.5, 0.0), "label": "V1", "outlier": False},
    {"type": "triangle", "params": make_triangle(260.0, 240.0, 32.5, 0.5), "label": "V2", "outlier": False},
    {"type": "triangle", "params": make_triangle(325.0, 355.0, 32.5, 1.0), "label": "V3", "outlier": False},
    {"type": "circle", "params": (365.0, 175.0, 25.0), "label": "C2", "outlier": False},
    {"type": "circle", "params": (70.0, 430.0, 32.5), "label": "OL1", "outlier": True},
    {"type": "circle", "params": (430.0, 70.0, 32.5), "label": "OL2", "outlier": True},
]

# =========================================================
# DYNAMIC OBSTACLE
# =========================================================
class DynObstacle:
    def __init__(self, obs_dict, vx: float, vy: float):
        self.base = copy.deepcopy(obs_dict)
        self.label = obs_dict["label"]
        self.otype = obs_dict["type"]
        self.vel = np.array([vx, vy], dtype=float)
        c = obs_centroid(obs_dict)
        self.cx = float(c[0])
        self.cy = float(c[1])
        self._build_obs()

    def _build_obs(self):
        t = self.otype
        bp = self.base["params"]
        if t == 'circle':
            _, _, r = bp
            self.obs = {**self.base, "params": (self.cx, self.cy, r)}
        elif t in ('square', 'rectangle'):
            _, _, hw, hh = bp
            self.obs = {**self.base, "params": (self.cx, self.cy, hw, hh)}
        elif t == 'triangle':
            bc = obs_centroid(self.base)
            off = np.array([self.cx - bc[0], self.cy - bc[1]])
            new_verts = [(v[0]+off[0], v[1]+off[1]) for v in bp]
            self.obs = {**self.base, "params": new_verts}

    def step(self, dt: float, goal=None):
        self.cx += self.vel[0] * dt
        self.cy += self.vel[1] * dt
        self._build_obs()
        if goal is not None:
            g_dist = np.linalg.norm(np.array([self.cx, self.cy]) - goal)
            if g_dist < GOAL_PROTECTION_RADIUS:
                dir_away = np.array([self.cx, self.cy]) - goal
                dir_away /= np.linalg.norm(dir_away) + 1e-12
                push = (GOAL_PROTECTION_RADIUS - g_dist + 10.0) * dir_away
                self.cx += push[0] * 0.6
                self.cy += push[1] * 0.6

        minx, maxx, miny, maxy = get_aabb(self.obs)
        speed = np.linalg.norm(self.vel)
        xmin, xmax, ymin, ymax = BOUNDS
        if minx < xmin: self.cx += (xmin - minx); self.vel[0] = abs(self.vel[0])
        if maxx > xmax: self.cx -= (maxx - xmax); self.vel[0] = -abs(self.vel[0])
        if miny < ymin: self.cy += (ymin - miny); self.vel[1] = abs(self.vel[1])
        if maxy > ymax: self.cy -= (maxy - ymax); self.vel[1] = -abs(self.vel[1])

        current_speed = np.linalg.norm(self.vel)
        if current_speed > 1e-8:
            self.vel = self.vel * (speed / current_speed)
        self._build_obs()

    def handle_obs_collision(self, other):
        minx1, maxx1, miny1, maxy1 = get_aabb(self.obs)
        minx2, maxx2, miny2, maxy2 = get_aabb(other.obs)
        if maxx1 > minx2 and minx1 < maxx2 and maxy1 > miny2 and miny1 < maxy2:
            ol_l = maxx1 - minx2
            ol_r = maxx2 - minx1
            ol_t = maxy1 - miny2
            ol_b = maxy2 - miny1
            min_ol = min(ol_l, ol_r, ol_t, ol_b)
            push_buffer = 0.501
            if min_ol == ol_l:
                nx, ny = -1.0, 0.0
                self.cx -= ol_l * push_buffer
                other.cx += ol_l * push_buffer
            elif min_ol == ol_r:
                nx, ny = 1.0, 0.0
                self.cx += ol_r * push_buffer
                other.cx -= ol_r * push_buffer
            elif min_ol == ol_t:
                nx, ny = 0.0, -1.0
                self.cy -= ol_t * push_buffer
                other.cy += ol_t * push_buffer
            else:
                nx, ny = 0.0, 1.0
                self.cy += ol_b * push_buffer
                other.cy -= ol_b * push_buffer

            v1 = self.vel.copy()
            v2 = other.vel.copy()
            speed1 = np.linalg.norm(v1)
            speed2 = np.linalg.norm(v2)
            v1n = v1[0]*nx + v1[1]*ny
            v2n = v2[0]*nx + v2[1]*ny
            self.vel[0] += (v2n - v1n) * nx
            self.vel[1] += (v2n - v1n) * ny
            other.vel[0] += (v1n - v2n) * nx
            other.vel[1] += (v1n - v2n) * ny

            if speed1 > 1e-8: self.vel *= speed1 / (np.linalg.norm(self.vel) + 1e-12)
            if speed2 > 1e-8: other.vel *= speed2 / (np.linalg.norm(other.vel) + 1e-12)

            self._build_obs()
            other._build_obs()

# =========================================================
# VELOCITY ASSIGNMENT
# =========================================================
def assign_velocities(obstacles_list, rng, verbose=True):
    # Speeds are in m per simulated second; obstacles move vel * dt per step
    # (dt = 0.1 s for APF), i.e. at most 0.475 m per APF step.
    dyns = []
    vel_table = []
    if verbose:
        print(" Initial velocities:")
    speed_groups = {
        'rectangle': BASE_SPEED_SCALE * 1.4,
        'square': BASE_SPEED_SCALE * 2.4,
        'triangle': BASE_SPEED_SCALE * 3.8,
        'circle': BASE_SPEED_SCALE * 2.9
    }
    for obs in obstacles_list:
        if obs.get('outlier', False):
            vx = vy = speed = 0.0
        else:
            otype = obs["type"]
            group_max = speed_groups.get(otype, BASE_SPEED_SCALE)
            speed = rng.uniform(0.7 * group_max, group_max)
            angle = rng.uniform(0, 2 * np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
        dyns.append(DynObstacle(obs, vx, vy))
        vel_table.append((obs["label"], round(vx, 3), round(vy, 3), round(speed, 3)))

    for label, vx, vy, sp in (vel_table if verbose else []):
        print(f" {label:4} | vx={vx:8.3f} vy={vy:8.3f} speed={sp:6.3f}")
    return dyns, vel_table

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
        self.goal_tol = GOAL_RADIUS
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

def clearance(q, obstacles):
    return min(get_distance(q, obs)[0] for obs in obstacles)

# =========================================================
# SIMULATION
# =========================================================
def simulate_dynamic_apf(start, goal, dyn_obstacles, cfg, dt=DT, rng=None, timeline=None,
                         max_steps=MAX_STEPS, record_history=True):
    """Returns (success, path, obs_history, t_sim, info).

    If `timeline` is given, obstacle states come from timeline.at(t) instead of
    stepping `dyn_obstacles`, so that different planners can be run against the
    identical obstacle motion sequence. The path ends at the first position
    inside the goal radius (no final segment to the goal centre is appended).
    A collision is recorded whenever the exact clearance from the agent to any
    obstacle drops below COLLISION_RADIUS, whether the agent moved into the
    obstacle or the obstacle moved into the agent.
    """
    if rng is None:
        rng = np.random.default_rng()
    q = start.copy()
    path = [q.copy()]
    obs0 = timeline.at(0.0) if timeline is not None else [d.obs for d in dyn_obstacles]
    obs_history = [copy.deepcopy(obs0)] if record_history else []
    t_sim = 0.0
    step_count = 0
    bias = rng.normal(0, 0.06, size=2)
    info = {'escape_steps': 0, 'pushouts': 0, 'min_clearance': clearance(q, obs0)}

    def _finish(outcome, success):
        info['outcome'] = outcome
        return success, np.array(path), obs_history, t_sim, info

    while step_count < max_steps:
        if timeline is not None:
            # plan on the world at t_k; the post-move check below uses t_{k+1}
            current_obs = timeline.at(step_count * dt)
        else:
            for d in dyn_obstacles:
                d.step(dt, goal=goal)
            for i in range(len(dyn_obstacles)):
                for j in range(i + 1, len(dyn_obstacles)):
                    dyn_obstacles[i].handle_obs_collision(dyn_obstacles[j])
            current_obs = [d.obs for d in dyn_obstacles]

            c = clearance(q, current_obs)      # obstacle moved into the agent?
            info['min_clearance'] = min(info['min_clearance'], c)
            if c < COLLISION_RADIUS:
                return _finish('collision', False)

        t_sim += dt

        if np.linalg.norm(q - goal) <= cfg.goal_tol:
            return _finish('success', True)

        Ft, _, _ = F_total(q, goal, current_obs, cfg)
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
        while in_collision(q_next, current_obs, margin=COLLISION_MARGIN) and escape_attempts < 5:
            perp = np.array([-sd[1], sd[0]])
            perp /= np.linalg.norm(perp) + 1e-12
            q_next = q + cfg.step * 1.8 * perp + rng.normal(0, 0.15, size=2)
            q_next[0] = clamp(q_next[0], *BOUNDS[:2])
            q_next[1] = clamp(q_next[1], *BOUNDS[2:])
            escape_attempts += 1
        if escape_attempts:
            info['escape_steps'] += 1

        if in_collision(q_next, current_obs, margin=COLLISION_MARGIN):
            min_rho = float('inf')
            best_grad = np.array([0.0, 0.0])
            for obs in current_obs:
                rho, grad, _ = get_distance(q_next, obs)
                if rho < min_rho:
                    min_rho = rho
                    best_grad = grad
            if min_rho < COLLISION_MARGIN:
                q_next += (COLLISION_MARGIN - min_rho + 3.0) * best_grad
                info['pushouts'] += 1

        q = q_next
        path.append(q.copy())
        if record_history:
            obs_history.append(copy.deepcopy(current_obs))
        step_count += 1

        post_obs = timeline.at(step_count * dt) if timeline is not None else current_obs
        c = clearance(q, post_obs)             # agent moved into an obstacle (or vice versa)?
        info['min_clearance'] = min(info['min_clearance'], c)
        if c < COLLISION_RADIUS:
            return _finish('collision', False)

    return _finish('timeout', False)

# =========================================================
# GIF GENERATION
# =========================================================
def draw_obstacles(ax, obs_list):
    for obs in obs_list:
        t = obs['type']
        is_out = obs.get('outlier', False)
        if t == 'circle':
            cx, cy, r = obs['params']
            fc = "#ffcccc" if is_out else "lightblue"
            ec = "#cc0000" if is_out else "blue"
            ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor=ec, alpha=0.7, lw=2))
        elif t in ('square', 'rectangle'):
            cx, cy, hw, hh = obs['params']
            ax.add_patch(Rectangle((cx-hw, cy-hh), 2*hw, 2*hh, facecolor="lightgray", edgecolor="dimgray", alpha=0.7, lw=2))
        elif t == 'triangle':
            verts = obs['params']
            ax.add_patch(MplPolygon(verts, closed=True, facecolor="lightyellow", edgecolor="goldenrod", alpha=0.7, lw=2))

def create_gif(path, obs_history, start, goal, run_num):
    try:
        out_path = f"results/dynamic_apf_test{run_num:02d}.gif"
        fig, ax = plt.subplots(figsize=(8, 8))
        frame_skip = 3
        path_s = path[::frame_skip]
        obs_s = obs_history[::frame_skip]

        def update(frame):
            ax.clear()
            ax.set_xlim(*BOUNDS[:2])
            ax.set_ylim(*BOUNDS[2:])
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            ax.plot(start[0], start[1], "bo", ms=8)
            ax.plot(goal[0], goal[1], "g*", ms=14)
            draw_obstacles(ax, obs_s[frame])
            curr = np.array(path_s[:frame+1])
            ax.plot(curr[:,0], curr[:,1], "b-", lw=2.5, alpha=0.85)
            if len(curr) > 0:
                ax.plot(curr[-1,0], curr[-1,1], "ro", ms=7)
            ax.set_title(f"Run {run_num:02d} | Constant Speed Bounce")
            return []

        anim = FuncAnimation(fig, update, frames=len(path_s), blit=False)
        anim.save(out_path, writer='pillow', fps=20)
        plt.close()
        print(f" GIF saved: {out_path}")
        return True
    except Exception as e:
        print(f" GIF failed for Run {run_num:02d}: {e}")
        return False

# =========================================================
# FIXED POWER MONITORING
# =========================================================
def get_system_power():
    # 1. Try RAPL (Intel/AMD, may need sudo)
    rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    try:
        if os.path.exists(rapl_path):
            with open(rapl_path) as f: e1 = int(f.read().strip())
            time.sleep(0.1)
            with open(rapl_path) as f: e2 = int(f.read().strip())
            watts = round(((e2 - e1) / 1_000_000) / 0.1, 1)
            if watts > 0:
                return watts
    except Exception as e:
        print(f"  [RAPL failed: {e}]")

    # 2. Battery discharge power (only useful when unplugged)
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
    except Exception as e:
        print(f"  [Battery fallback failed: {e}]")

    # 3. CPU-utilization estimate (works plugged in, always non-zero under load)
    try:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        TDP_W = 15.0  # ← Set this to your CPU's TDP (e.g. 15W for mobile, 65W for desktop)
        idle_W = TDP_W * 0.10
        estimated = idle_W + (TDP_W - idle_W) * (cpu_pct / 100.0)
        return round(estimated, 1)
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
    GOAL = np.array([465.0, 465.0])
    cfg = APFConfig()
    N_RUNS = 50

    print(f"=== APF Dynamic Environment — {N_RUNS} Seeded Runs ===")
    print("Metrics: Steps | Time (ms) | Avg CPU % | Peak RAM (GB) | System Power (W)\n")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Dynamic_APF_50_Runs"

    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                         top=Side(style='thin'), bottom=Side(style='thin'))

    row = 1
    ws.cell(row, 1, f"APF Dynamic {N_RUNS} Runs").font = Font(bold=True, size=14)
    row += 3

    summary_data = []

    for run in range(1, N_RUNS + 1):
        run_seed = run - 1   # fixed and reproducible; same seeds as experiments/paired_dynamic.py
        rng = np.random.default_rng(run_seed)
        random.seed(run_seed)

        print(f"--- Starting Run {run:02d} ---")
        dyn_obstacles, _ = assign_velocities(STATIC_OBSTACLES, rng)

        cpu_start, ram_start, sys_start = get_resources()
        t_start = time.perf_counter()

        success, path, obs_history, final_t, _ = simulate_dynamic_apf(
            START, GOAL, dyn_obstacles, cfg, dt=DT, rng=rng
        )

        sim_time_ms = (time.perf_counter() - t_start) * 1000
        cpu_end, peak_ram, sys_end = get_resources()

        avg_cpu = round((cpu_start + cpu_end) / 2, 1)

        steps = max(len(path) - 1, 1)
        L = path_length(path)
        efficiency = (np.linalg.norm(GOAL - START) / L * 100) if L > 0 else 0.0

        power_str = f"{sys_end} W" if isinstance(sys_end, (int, float)) else sys_end

        print(f"Run {run:02d} → Steps: {steps} | Time: {sim_time_ms:.1f}ms | "
              f"CPU: {avg_cpu}% | RAM: {peak_ram:.2f}GB | System Power: {power_str}")

        if success:
            create_gif(path, obs_history, START, GOAL, run)

        summary_data.append({
            "Run": run,
            "Success": success,
            "Steps": steps,
            "Sim_Time_ms": round(sim_time_ms, 1),
            "Avg_CPU_%": avg_cpu,
            "Peak_RAM_GB": peak_ram,
            "System_Power_W": power_str,
            "Path_Length": round(L, 2),
            "Efficiency_%": round(efficiency, 1),
            "Seed": run_seed
        })

        # Write to Excel
        ws.cell(row, 1, f"TEST RUN {run:02d}").font = Font(bold=True, size=12)
        row += 1
        metrics = [
            ["Random Seed", run_seed],
            ["Success", success],
            ["Steps Taken", steps],
            ["Computation Time (ms)", round(sim_time_ms, 1)],
            ["Avg CPU (%)", avg_cpu],
            ["Peak RAM (GB)", peak_ram],
            ["System Power (W)", power_str],
            ["Path Length (m)", round(L, 2)],
            ["Efficiency (%)", round(efficiency, 1)]
        ]
        for label, value in metrics:
            ws.cell(row, 1, label).font = Font(bold=True)
            ws.cell(row, 2, value)
            row += 1
        row += 2

    # Summary Table
    row += 1
    ws.cell(row, 1, f"SUMMARY OF {N_RUNS} RUNS").font = Font(bold=True, size=14)
    row += 2
    headers = ["Run", "Success", "Steps", "Sim Time (ms)", "Avg CPU %", "Peak RAM (GB)",
               "System Power (W)", "Path Length", "Efficiency %", "Seed"]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row, col, h)
        cell.font = Font(bold=True)
        cell.border = thin_border
    row += 1

    success_count = sum(1 for d in summary_data if d["Success"])
    for d in summary_data:
        values = [d["Run"], "Yes" if d["Success"] else "No", d["Steps"], d["Sim_Time_ms"],
                  d["Avg_CPU_%"], d["Peak_RAM_GB"], d["System_Power_W"],
                  d["Path_Length"], d["Efficiency_%"], d["Seed"]]
        for col, val in enumerate(values, 1):
            ws.cell(row, col, val).border = thin_border
        row += 1

    ws.cell(row + 2, 1, f"Success Rate: {success_count}/{N_RUNS} ({success_count/N_RUNS*100:.1f}%)").font = Font(bold=True)

    wb.save("results/dynamic_metrics_50_seeded_runs.xlsx")

    print("\n" + "="*90)
    print(f"ALL {N_RUNS} RUNS COMPLETED")
    print("Excel saved → results/dynamic_metrics_50_seeded_runs.xlsx")
    print("GIFs saved in results/ folder (successful runs)")