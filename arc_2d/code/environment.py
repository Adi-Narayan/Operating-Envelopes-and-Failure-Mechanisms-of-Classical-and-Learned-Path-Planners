"""
UAV Target Tracking Environment
Implements the environment described in the ARC paper
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle

# Try to import shapely for UAVEnvironment (optional for ARCEnvironment)
try:
    from shapely.geometry import Point, Polygon, LineString
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from config import ENV_CONFIG, REWARD_CONFIG


class UAVEnvironment:
    def __init__(self, env_def, config=None):
        """
        Initialize UAV environment
        
        Args:
            env_def: Dictionary defining the environment (from ENVIRONMENTS in config)
            config: Optional config override (uses ENV_CONFIG by default)
        """
        self.config = config if config else ENV_CONFIG
        self.env_def = env_def
        
        # Environment dimensions
        self.width = self.config['width']
        self.height = self.config['height']
        
        # UAV parameters
        self.uav_speed = self.config['uav_speed']
        self.dt = self.config['sampling_interval']
        self.num_sensors = self.config['num_sensors']
        self.sensor_range = self.config['sensor_range']
        self.action_range = np.pi / 20  # From paper
        
        # Thresholds
        self.d_min_obstacle = self.config['d_min_obstacle']
        self.d_min_target = self.config['d_min_target']
        
        # Build obstacles
        self.obstacles = self._build_obstacles(env_def['obstacles'])
        
        # Target configuration
        self.target_motion = env_def['target_motion']
        self.target_start = np.array(env_def['target_start'], dtype=np.float32)
        if self.target_motion:
            self.target_velocity = np.array(env_def['target_velocity'], dtype=np.float32)
        else:
            self.target_velocity = np.array([0.0, 0.0], dtype=np.float32)
        
        # UAV start range
        self.uav_start_range = env_def['uav_start_range']
        
        self.dynamic_obstacles = env_def.get('dynamic_obstacles', False)
        self.obstacle_speed_scale = env_def.get('obstacle_speed_scale', 15.0)
        self.obstacle_motion_model = env_def.get('obstacle_motion_model', 'legacy')
        self.obstacle_velocity_seed = env_def.get('obstacle_velocity_seed', None)
        self.obstacle_dt = float(env_def.get('obstacle_dt', self.dt))
        self.obstacle_substeps = int(env_def.get('obstacle_substeps', 1))
        self.obstacle_space_scale = float(env_def.get('obstacle_space_scale', 1.0))
        self.obstacle_apf_speed_scale = env_def.get('obstacle_apf_speed_scale', None)
        import copy
        self.base_obstacle_defs = copy.deepcopy(env_def['obstacles'])
        self.active_obstacle_defs = copy.deepcopy(env_def['obstacles'])
        self.obs_velocities = []

        # State variables
        self.uav_pos = None
        self.uav_heading = None
        self.target_pos = None
        self.step_count = 0
        
    def _build_obstacles(self, obstacle_defs):
        """Build obstacle geometries from definitions"""
        obstacles = []
        for obs_def in obstacle_defs:
            if obs_def['type'] == 'polygon':
                obstacles.append(Polygon(obs_def['vertices']))
            elif obs_def['type'] == 'circle':
                center = Point(obs_def['center'])
                obstacles.append(center.buffer(obs_def['radius']))
        return obstacles
    
    def reset(self, uav_start_pos=None):
        """
        Reset environment to initial state
        
        Args:
            uav_start_pos: Optional fixed start position, otherwise random
            
        Returns:
            observation: Initial observation
        """
        # Random UAV start position if not provided
        max_retries = 10
        collision_free = False
        retry = 0
        
        while not collision_free and retry < max_retries:
            if uav_start_pos is None:
                x_range, y_range = self.uav_start_range
                self.uav_pos = np.array([
                    np.random.uniform(x_range[0], x_range[1]),
                    np.random.uniform(y_range[0], y_range[1])
                ], dtype=np.float32)
            else:
                self.uav_pos = np.array(uav_start_pos, dtype=np.float32)
            
            # Check if starting position is collision-free
            try:
                sensor_distances = []
                for i in range(self.num_sensors):
                    angle = (i - (self.num_sensors - 1) / 2) * np.pi / (self.num_sensors - 1)
                    dist = self._get_sensor_distance(angle)
                    sensor_distances.append(dist)
                
                if np.min(sensor_distances) > self.d_min_obstacle * 2:  # At least 2x minimum distance
                    collision_free = True
                else:
                    retry += 1
                    if uav_start_pos is not None:
                        break  # Don't retry if position explicitly provided
            except:
                retry += 1
        
        # Random initial heading
        self.uav_heading = np.random.uniform(-np.pi, np.pi)
        
        # Reset target position
        self.target_pos = self.target_start.copy()
        
        import copy
        if self.dynamic_obstacles:
            self._init_dynamic_obstacle_velocities()
        self.active_obstacle_defs = copy.deepcopy(self.base_obstacle_defs)
        self.obstacles = self._build_obstacles(self.active_obstacle_defs)

        # Reset step counter
        self.step_count = 0
        
        return self._get_observation()

    def _init_dynamic_obstacle_velocities(self):
        """Initialize dynamic obstacle velocities using selected motion model."""
        self.obs_velocities = []

        if self.obstacle_motion_model == 'apf':
            rng = np.random.default_rng(self.obstacle_velocity_seed)
            speed_scale = self.obstacle_apf_speed_scale if self.obstacle_apf_speed_scale is not None else self.obstacle_speed_scale
            for _ in self.base_obstacle_defs:
                angle = rng.uniform(0.0, 2.0 * np.pi)
                speed = rng.uniform(0.05 * speed_scale, speed_scale)
                if self.obstacle_apf_speed_scale is not None:
                    speed *= self.obstacle_space_scale
                self.obs_velocities.append(np.array([speed * np.cos(angle), speed * np.sin(angle)], dtype=float))
            return

        for _ in self.base_obstacle_defs:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.1 * self.obstacle_speed_scale, self.obstacle_speed_scale)
            self.obs_velocities.append(np.array([speed * np.cos(angle), speed * np.sin(angle)], dtype=float))

    def _obs_centroid(self, obs_def):
        if obs_def['type'] == 'circle':
            return np.array(obs_def['center'], dtype=float)
        verts = np.array(obs_def['vertices'], dtype=float)
        return np.mean(verts, axis=0)

    def _obs_radius(self, obs_def):
        if obs_def['type'] == 'circle':
            return float(obs_def['radius'])
        centroid = self._obs_centroid(obs_def)
        verts = np.array(obs_def['vertices'], dtype=float)
        return float(np.max(np.linalg.norm(verts - centroid, axis=1)))

    def _translate_obstacle(self, obs_def, dx, dy):
        if obs_def['type'] == 'circle':
            cx, cy = obs_def['center']
            obs_def['center'] = (cx + dx, cy + dy)
            return
        verts = np.array(obs_def['vertices'], dtype=float)
        obs_def['vertices'] = [(float(v[0] + dx), float(v[1] + dy)) for v in verts]

    def _update_dynamic_obstacles_apf(self):
        """APF-like obstacle update: centroid bounce + pairwise elastic normal exchange."""
        xmin, xmax = 0.0, float(self.width)
        ymin, ymax = 0.0, float(self.height)

        substeps = max(1, self.obstacle_substeps)
        dt_obs = self.obstacle_dt if self.obstacle_dt > 0 else self.dt

        for _ in range(substeps):
            for i, obs_def in enumerate(self.active_obstacle_defs):
                vel = self.obs_velocities[i]
                c = self._obs_centroid(obs_def)
                r = self._obs_radius(obs_def)

                nx = c[0] + vel[0] * dt_obs
                ny = c[1] + vel[1] * dt_obs

                if nx - r < xmin:
                    nx = xmin + r
                    vel[0] = abs(vel[0])
                if nx + r > xmax:
                    nx = xmax - r
                    vel[0] = -abs(vel[0])
                if ny - r < ymin:
                    ny = ymin + r
                    vel[1] = abs(vel[1])
                if ny + r > ymax:
                    ny = ymax - r
                    vel[1] = -abs(vel[1])

                self._translate_obstacle(obs_def, nx - c[0], ny - c[1])

            for i in range(len(self.active_obstacle_defs)):
                for j in range(i + 1, len(self.active_obstacle_defs)):
                    o1 = self.active_obstacle_defs[i]
                    o2 = self.active_obstacle_defs[j]

                    c1 = self._obs_centroid(o1)
                    c2 = self._obs_centroid(o2)
                    r1 = self._obs_radius(o1)
                    r2 = self._obs_radius(o2)

                    dx = c2[0] - c1[0]
                    dy = c2[1] - c1[1]
                    dist = np.hypot(dx, dy)
                    gap = r1 + r2

                    if dist < gap and dist > 1e-9:
                        nx = dx / dist
                        ny = dy / dist
                        overlap = gap - dist

                        self._translate_obstacle(o1, -nx * overlap * 0.5, -ny * overlap * 0.5)
                        self._translate_obstacle(o2, nx * overlap * 0.5, ny * overlap * 0.5)

                        v1 = self.obs_velocities[i]
                        v2 = self.obs_velocities[j]
                        v1n = v1[0] * nx + v1[1] * ny
                        v2n = v2[0] * nx + v2[1] * ny

                        v1[0] += (v2n - v1n) * nx
                        v1[1] += (v2n - v1n) * ny
                        v2[0] += (v1n - v2n) * nx
                        v2[1] += (v1n - v2n) * ny
    
    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: Normalized action in [-1, 1]
            
        Returns:
            observation: New observation
            reward: Reward received
            done: Whether episode is done
            info: Additional information
        """
        # Denormalize action to actual heading change
        delta_heading = action * self.action_range
        
        # Update UAV heading
        prev_heading = self.uav_heading
        self.uav_heading += delta_heading
        self.uav_heading = self._normalize_angle(self.uav_heading)
        
        # Update UAV position
        self.uav_pos[0] += self.uav_speed * np.cos(self.uav_heading) * self.dt
        self.uav_pos[1] += self.uav_speed * np.sin(self.uav_heading) * self.dt
        
        # Update target position (if moving)
        self.target_pos += self.target_velocity * self.dt
        
        # Wrap target position to stay in bounds
        self.target_pos[0] = np.clip(self.target_pos[0], 0, self.width)
        self.target_pos[1] = np.clip(self.target_pos[1], 0, self.height)
        
        if self.dynamic_obstacles:
            self._update_dynamic_obstacles()

        # Check boundaries (UAV)
        out_of_bounds = (self.uav_pos[0] < 0 or self.uav_pos[0] > self.width or
                        self.uav_pos[1] < 0 or self.uav_pos[1] > self.height)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward, info = self._calculate_reward(delta_heading, observation)
        
        # Check terminal conditions
        done = False
        if info['collision'] or out_of_bounds:
            done = True
            reward += REWARD_CONFIG['k_terminal'] * -1
            info['success'] = False
        elif info['target_reached']:
            done = True
            reward += REWARD_CONFIG['k_terminal']
            info['success'] = True
        
        self.step_count += 1
        if self.step_count >= self.config['max_steps']:
            done = True
        
        info['steps'] = self.step_count
        
        return observation, reward, done, info
    
    def _update_dynamic_obstacles(self):
        """Update positions of dynamic obstacles and bounce off walls."""
        if self.obstacle_motion_model == 'apf':
            self._update_dynamic_obstacles_apf()
            self.obstacles = self._build_obstacles(self.active_obstacle_defs)
            return

        for i, (obs_def, vel) in enumerate(zip(self.active_obstacle_defs, self.obs_velocities)):
            if obs_def['type'] == 'polygon':
                verts = np.array(obs_def['vertices'])
                min_x, max_x = np.min(verts[:, 0]), np.max(verts[:, 0])
                min_y, max_y = np.min(verts[:, 1]), np.max(verts[:, 1])
                
                if min_x + vel[0] * self.dt < 0 or max_x + vel[0] * self.dt > self.width:
                    vel[0] *= -1
                if min_y + vel[1] * self.dt < 0 or max_y + vel[1] * self.dt > self.height:
                    vel[1] *= -1
                
                for j in range(len(verts)):
                    obs_def['vertices'][j] = (verts[j][0] + vel[0] * self.dt, verts[j][1] + vel[1] * self.dt)
                    
            elif obs_def['type'] == 'circle':
                cx, cy = obs_def['center']
                r = obs_def['radius']
                
                if cx - r + vel[0] * self.dt < 0 or cx + r + vel[0] * self.dt > self.width:
                    vel[0] *= -1
                if cy - r + vel[1] * self.dt < 0 or cy + r + vel[1] * self.dt > self.height:
                    vel[1] *= -1
                
                obs_def['center'] = (cx + vel[0] * self.dt, cy + vel[1] * self.dt)

        # Handle simple circle approximation collision bouncing
        for i in range(len(self.active_obstacle_defs)):
            for j in range(i + 1, len(self.active_obstacle_defs)):
                o1 = self.active_obstacle_defs[i]
                o2 = self.active_obstacle_defs[j]
                
                def get_c_r(o):
                    if o['type'] == 'circle':
                        return np.array(o['center']), o['radius']
                    else:
                        verts = np.array(o['vertices'])
                        cent = np.mean(verts, axis=0)
                        r = np.max(np.linalg.norm(verts - cent, axis=1))
                        return cent, r
                
                c1, r1 = get_c_r(o1)
                c2, r2 = get_c_r(o2)
                dist = np.linalg.norm(c1 - c2)
                if dist < r1 + r2 and dist > 1e-9:
                    n = (c1 - c2) / dist
                    overlap = (r1 + r2) - dist
                    
                    # Shift positions
                    if o1['type'] == 'circle':
                        o1['center'] = (o1['center'][0] + n[0] * overlap * 0.5, o1['center'][1] + n[1] * overlap * 0.5)
                    else:
                        for k in range(len(o1['vertices'])):
                            o1['vertices'][k] = (o1['vertices'][k][0] + n[0] * overlap * 0.5, o1['vertices'][k][1] + n[1] * overlap * 0.5)
                            
                    if o2['type'] == 'circle':
                        o2['center'] = (o2['center'][0] - n[0] * overlap * 0.5, o2['center'][1] - n[1] * overlap * 0.5)
                    else:
                        for k in range(len(o2['vertices'])):
                            o2['vertices'][k] = (o2['vertices'][k][0] - n[0] * overlap * 0.5, o2['vertices'][k][1] - n[1] * overlap * 0.5)

                    v1 = np.array(self.obs_velocities[i])
                    v2 = np.array(self.obs_velocities[j])
                    v1n = np.dot(v1, n)
                    v2n = np.dot(v2, n)
                    self.obs_velocities[i][0] += (v2n - v1n) * n[0]
                    self.obs_velocities[i][1] += (v2n - v1n) * n[1]
                    self.obs_velocities[j][0] += (v1n - v2n) * n[0]
                    self.obs_velocities[j][1] += (v1n - v2n) * n[1]

        self.obstacles = self._build_obstacles(self.active_obstacle_defs)

    def _get_observation(self):
        """
        Get current observation
        
        Returns:
            observation: [psi, chi, d0, d1, ..., dn]
        """
        # UAV heading (psi)
        psi = self.uav_heading
        
        # Relative azimuth to target (chi)
        target_vec = self.target_pos - self.uav_pos
        target_angle = np.arctan2(target_vec[1], target_vec[0])
        chi = self._normalize_angle(target_angle)
        
        # Distance to target (d0)
        d0 = np.linalg.norm(target_vec)
        
        # Sensor distances
        sensor_distances = []
        for i in range(self.num_sensors):
            angle = psi + (i - (self.num_sensors - 1) / 2) * np.pi / (self.num_sensors - 1)
            distance = self._get_sensor_distance(angle)
            sensor_distances.append(distance)
        
        # Construct observation
        obs = np.array([psi, chi, d0] + sensor_distances, dtype=np.float32)
        
        # Normalize observation
        obs_normalized = self._normalize_observation(obs)
        
        return obs_normalized
    
    def _normalize_observation(self, obs):
        """Normalize observation to [0, 1] range"""
        obs_norm = obs.copy()
        # Angles to [0, 1]
        obs_norm[0] = (obs[0] + np.pi) / (2 * np.pi)  # psi
        obs_norm[1] = (obs[1] + np.pi) / (2 * np.pi)  # chi
        # Distances to [0, 1]
        obs_norm[2] = np.clip(obs[2] / 500.0, 0, 1)  # d0
        for i in range(3, len(obs)):
            obs_norm[i] = obs[i] / self.sensor_range  # sensor distances
        return obs_norm
    
    def _get_sensor_distance(self, angle):
        """Get distance reading from sensor at given angle"""
        # Create ray from UAV position
        ray_length = self.sensor_range
        ray_end = self.uav_pos + ray_length * np.array([np.cos(angle), np.sin(angle)])
        ray = LineString([self.uav_pos, ray_end])
        
        min_distance = self.sensor_range
        
        # Check intersection with each obstacle
        try:
            for obstacle in self.obstacles:
                if ray.intersects(obstacle):
                    intersection = ray.intersection(obstacle)
                    if intersection.is_empty:
                        continue
                    
                    # Get closest intersection point
                    if intersection.geom_type == 'Point':
                        dist = np.linalg.norm(np.array([intersection.x, intersection.y]) - self.uav_pos)
                        min_distance = min(min_distance, dist)
                    elif intersection.geom_type == 'MultiPoint':
                        for pt in intersection.geoms:
                            dist = np.linalg.norm(np.array([pt.x, pt.y]) - self.uav_pos)
                            min_distance = min(min_distance, dist)
                    elif intersection.geom_type == 'LineString':
                        coords = list(intersection.coords)
                        for coord in coords:
                            dist = np.linalg.norm(np.array(coord) - self.uav_pos)
                            min_distance = min(min_distance, dist)
        except Exception:
            # If intersection fails, return sensor_range
            pass
        
        # Ensure minimum non-zero distance to avoid division by zero
        return max(min_distance, 0.1)
    
    def _calculate_reward(self, delta_heading, observation):
        """Calculate reward based on current state"""
        info = {
            'collision': False,
            'target_reached': False,
            'los_reward': 0,
            'obstacle_reward': 0,
            'action_reward': 0,
        }
        
        # Denormalize relevant observations
        psi = observation[0] * 2 * np.pi - np.pi
        chi = observation[1] * 2 * np.pi - np.pi
        d0 = observation[2] * 500.0
        sensor_distances = observation[3:] * self.sensor_range
        
        # LOS reward (Equation 19)
        los_angle = abs(chi - psi)
        if los_angle > np.pi:
            los_angle = 2 * np.pi - los_angle
        r_los = REWARD_CONFIG['lambda_los'] * (np.pi / 2 - los_angle)
        info['los_reward'] = r_los
        
        # Obstacle reward (Equation 20)
        r_obstacle = 0
        x_sum = 0
        y_sum = 0
        for i, d_i in enumerate(sensor_distances):
            # Safety check: prevent division by zero
            d_i = max(float(d_i), 0.1)  # Minimum distance of 0.1 meters
            angle_offset = (i / (self.num_sensors - 1)) * np.pi
            inv_d = 1.0 / d_i - 1.0 / self.sensor_range
            # Clamp to prevent extreme values
            inv_d = np.clip(inv_d, -10.0, 10.0)
            x_sum += inv_d * np.cos(angle_offset)
            y_sum += inv_d * np.sin(angle_offset)
        r_obstacle = REWARD_CONFIG['sigma_obstacle'] * (x_sum**2 + y_sum**2)
        info['obstacle_reward'] = r_obstacle
        
        # Action penalty (Equation 22)
        r_action = REWARD_CONFIG['alpha_action'] * abs(delta_heading)
        info['action_reward'] = r_action
        
        # Check collision
        if np.any(sensor_distances < self.d_min_obstacle):
            info['collision'] = True
        
        # Check target reached
        if d0 < self.d_min_target:
            info['target_reached'] = True
        
        # Total reward (Equation 23)
        reward = r_los + r_obstacle + r_action
        
        return reward, info
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def render(self, ax=None, show_sensors=True):
        """Render the environment"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Clear axis
        ax.clear()
        
        # Set limits
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        
        # Draw obstacles
        for obstacle in self.obstacles:
            if obstacle.geom_type == 'Polygon':
                coords = list(obstacle.exterior.coords)
                patch = MplPolygon(coords, facecolor='gray', edgecolor='black', alpha=0.5)
                ax.add_patch(patch)
        
        # Draw target
        ax.plot(self.target_pos[0], self.target_pos[1], 'g*', markersize=20, label='Target')
        
        # Draw UAV
        ax.plot(self.uav_pos[0], self.uav_pos[1], 'ro', markersize=10, label='UAV')
        
        # Draw heading
        heading_length = 15
        heading_end = self.uav_pos + heading_length * np.array([np.cos(self.uav_heading), np.sin(self.uav_heading)])
        ax.arrow(self.uav_pos[0], self.uav_pos[1], 
                heading_end[0] - self.uav_pos[0], heading_end[1] - self.uav_pos[1],
                head_width=5, head_length=5, fc='red', ec='red')
        
        # Draw sensors
        if show_sensors:
            for i in range(self.num_sensors):
                angle = self.uav_heading + (i - (self.num_sensors - 1) / 2) * np.pi / (self.num_sensors - 1)
                distance = self._get_sensor_distance(angle)
                sensor_end = self.uav_pos + distance * np.array([np.cos(angle), np.sin(angle)])
                ax.plot([self.uav_pos[0], sensor_end[0]], [self.uav_pos[1], sensor_end[1]], 
                       'b-', alpha=0.3, linewidth=0.5)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        return ax


# ============================================================
# ARC ENVIRONMENT — From Jupyter notebook
# ============================================================

def clamp(x, lo, hi):
    """Clamp value between bounds"""
    return max(lo, min(hi, x))


def angle_wrap(a):
    """Wrap angle to [-pi, pi]"""
    return np.arctan2(np.sin(a), np.cos(a))


def dist_point_to_aabb(p, center, half_w, half_h):
    """Distance from point to axis-aligned bounding box"""
    cx, cy = center
    qx = clamp(p[0], cx - half_w, cx + half_w)
    qy = clamp(p[1], cy - half_h, cy + half_h)
    closest = np.array([qx, qy], dtype=float)
    v = p - closest
    rho = np.linalg.norm(v)
    grad_rho = v / rho if rho > 1e-12 else np.array([0.0, 0.0])
    return max(rho, 0.0), grad_rho, closest


def dist_point_to_circle(p, center, radius):
    """Distance from point to circle"""
    v = p - center
    d = np.linalg.norm(v)
    grad_rho = v / d if d > 1e-12 else np.array([1.0, 0.0])
    rho = max(d - radius, 0.0)
    closest = center + grad_rho * radius
    return rho, grad_rho, closest


def dist_point_to_triangle(p, vertices):
    """Distance from point to triangle"""
    def closest_on_segment(a, b, pt):
        ab = b - a
        t = np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-12)
        return a + clamp(t, 0.0, 1.0) * ab

    def point_in_triangle(pt, v0, v1, v2):
        def sign(pa, pb, pc):
            return (pa[0] - pc[0]) * (pb[1] - pc[1]) - (pb[0] - pc[0]) * (pa[1] - pc[1])
        d1, d2, d3 = sign(pt, v0, v1), sign(pt, v1, v2), sign(pt, v2, v0)
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
    """Get distance from point to obstacle"""
    t = obs['type']
    if t == 'circle':
        cx, cy, r = obs['params']
        return dist_point_to_circle(p, np.array([cx, cy], float), r)
    elif t in ('square', 'rectangle'):
        cx, cy, hw, hh = obs['params']
        return dist_point_to_aabb(p, np.array([cx, cy], float), hw, hh)
    elif t == 'triangle':
        return dist_point_to_triangle(p, obs['params'])


import copy

def obs_centroid(obs):
    if obs['type'] == 'triangle':
        return np.array(obs['params'], dtype=float).mean(axis=0)
    return np.array(obs['params'][:2], dtype=float)


def obs_radius(obs):
    t = obs['type']
    if t == 'circle':
        return obs['params'][2]
    elif t in ('square', 'rectangle'):
        _, _, hw, hh = obs['params']
        return np.hypot(hw, hh)
    elif t == 'triangle':
        c = obs_centroid(obs)
        return max(np.linalg.norm(np.array(v) - c) for v in obs['params'])


class DynObstacle:
    def __init__(self, obs_dict, vx: float, vy: float):
        self.base   = copy.deepcopy(obs_dict)
        self.label  = obs_dict.get("label", "Unknown")
        self.otype  = obs_dict["type"]
        self.vel    = np.array([vx, vy], dtype=float)
        c           = obs_centroid(obs_dict)
        self.cx     = float(c[0])
        self.cy     = float(c[1])
        self._build_obs()

    def _build_obs(self):
        t  = self.otype
        bp = self.base["params"]
        if t == 'circle':
            _, _, r = bp
            self.obs = {**self.base, "params": (self.cx, self.cy, r)}
        elif t in ('square', 'rectangle'):
            _, _, hw, hh = bp
            self.obs = {**self.base, "params": (self.cx, self.cy, hw, hh)}
        elif t == 'triangle':
            bc  = obs_centroid(self.base)
            off = np.array([self.cx - bc[0], self.cy - bc[1]])
            new_verts = [(v[0]+off[0], v[1]+off[1]) for v in bp]
            self.obs  = {**self.base, "params": new_verts}

    def step(self, dt: float, bounds):
        xmin, xmax, ymin, ymax = bounds
        r = obs_radius(self.obs)

        self.cx += self.vel[0] * dt
        self.cy += self.vel[1] * dt

        if self.cx - r < xmin:
            self.cx = xmin + r; self.vel[0] = abs(self.vel[0])
        if self.cx + r > xmax:
            self.cx = xmax - r; self.vel[0] = -abs(self.vel[0])
        if self.cy - r < ymin:
            self.cy = ymin + r; self.vel[1] = abs(self.vel[1])
        if self.cy + r > ymax:
            self.cy = ymax - r; self.vel[1] = -abs(self.vel[1])

        self._build_obs()

    def handle_obs_collision(self, other: "DynObstacle"):
        dx = other.cx - self.cx
        dy = other.cy - self.cy
        dist = np.hypot(dx, dy)
        gap  = obs_radius(self.obs) + obs_radius(other.obs)
        if dist < gap and dist > 1e-9:
            nx, ny  = dx/dist, dy/dist
            overlap = gap - dist
            self.cx  -= nx * overlap * 0.5
            self.cy  -= ny * overlap * 0.5
            other.cx += nx * overlap * 0.5
            other.cy += ny * overlap * 0.5
            v1n  = self.vel[0]*nx  + self.vel[1]*ny
            v2n  = other.vel[0]*nx + other.vel[1]*ny
            self.vel[0]  += (v2n - v1n) * nx
            self.vel[1]  += (v2n - v1n) * ny
            other.vel[0] += (v1n - v2n) * nx
            other.vel[1] += (v1n - v2n) * ny
            self._build_obs(); other._build_obs()


def in_collision(q, obstacles, margin=0.0):
    """Check if point is in collision with obstacles"""
    for obs in obstacles:
        rho, _, _ = get_distance(q, obs)
        if rho <= margin:
            return True
    return False


def has_los(p, goal, obstacles, n_samples=8):
    """Check if there is line-of-sight from p to goal"""
    for t in np.linspace(0.01, 0.99, n_samples):
        mid = p + t * (goal - p)
        if in_collision(mid, obstacles, margin=0.05):
            return False
    return True


def U_att_val(q, goal, k_a=2.0):
    """Attractive potential value"""
    return 0.5 * k_a * float(np.linalg.norm(q - goal) ** 2)


def U_rep_val(q, goal, obstacles, k_r=0.03, goal_pow=2):
    """Repulsive potential value"""
    rho_goal = np.linalg.norm(q - goal) + 1e-12
    total = 0.0
    for obs in obstacles:
        rho, _, _ = get_distance(q, obs)
        rho = max(rho, 1e-3)
        total += k_r * (1.0 / rho) * (rho_goal ** goal_pow)
    return total


class ARCEnvironment:
    """
    Continuous 2-D UAV navigation environment from ARC paper.
    
    Observation vector (obs_dim = 10):
        [dx_norm, dy_norm,           unit vector to goal
         dist_to_goal_norm,          scalar distance (normalised)
         vx_norm, vy_norm,           velocity (normalised)
         min_obs_dist_norm,          nearest obstacle clearance
         U_att_norm,                 attractive potential
         U_rep_norm,                 repulsive potential
         heading_err_norm,           heading error to goal
         ang_vel_norm]               angular velocity

    Action vector (act_dim = 2) in [-1, 1]:
        [delta_heading_norm,  speed_norm]

    Reward includes components for progress, alignment, clearance, boundary,
    action smoothness, and LOS bonus.
    """

    OBS_DIM = 10
    ACT_DIM = 2

    MAX_STEPS = 800
    STEP_SIZE = 0.12
    MAX_DELTA_H = np.pi / 4
    COLLISION_M = 0.05

    # Reward parameters
    R_STEP = -0.02
    K_PROGRESS = 4.0
    K_ALIGN = 0.3
    K_CLEARANCE = 0.8
    D_SAFE = 0.80
    K_ACTION = 0.01
    K_BOUNDARY = 0.20
    BOUND_SAFE = 0.50
    K_LOS = 0.15
    R_GOAL = 250.0
    R_COLLISION = -250.0

    def __init__(self, obstacles=None, start=None, goal=None,
                 bounds=None, seed=None, jitter_start=False,
                 goal_radius=0.30, dynamic_obstacles=False, obstacle_speed_scale=0.05):
        """
        Initialize ARC environment
        
        Args:
            obstacles: List of obstacle definitions with 'type' and 'params'
            start: Starting position [x, y]
            goal: Goal position [x, y]
            bounds: Environment bounds (x_min, x_max, y_min, y_max)
            seed: Random seed
            jitter_start: Add random jitter to start position
            goal_radius: Radius for goal achievement
            dynamic_obstacles: Whether obstacles are continuously moving
            obstacle_speed_scale: Magnitude parameter for dynamic speeds
        """
        # Default environment from config
        if obstacles is None or start is None or goal is None or bounds is None:
            from config import ARC_CONFIG
            obstacles = ARC_CONFIG['OBSTACLES']
            start = ARC_CONFIG['START']
            goal = ARC_CONFIG['GOAL']
            bounds = ARC_CONFIG['ENV_BOUNDS']
            goal_radius = ARC_CONFIG.get('GOAL_RADIUS', 0.30)
            dynamic_obstacles = ARC_CONFIG.get('DYNAMIC_OBSTACLES', False)
            obstacle_speed_scale = ARC_CONFIG.get('OBSTACLE_SPEED_SCALE', 0.05)

        self._base_obstacles = copy.deepcopy(obstacles)
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacle_speed_scale = obstacle_speed_scale
        self._base_start = start.copy() if isinstance(start, np.ndarray) else np.array(start, dtype=float)
        self.goal = goal.copy() if isinstance(goal, np.ndarray) else np.array(goal, dtype=float)
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        self.jitter_start = jitter_start
        self.goal_radius = goal_radius
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        # Rebuild dynamic obstacles if enabled
        if self.dynamic_obstacles:
            self.dyn_obstacles = []
            for obs in self._base_obstacles:
                angle  = self.rng.uniform(0, 2*np.pi)
                speed  = self.rng.uniform(0.05 * self.obstacle_speed_scale, self.obstacle_speed_scale)
                vx, vy = speed * np.cos(angle), speed * np.sin(angle)
                self.dyn_obstacles.append(DynObstacle(obs, vx, vy))
            self.obstacles = [d.obs for d in self.dyn_obstacles]
        else:
            self.obstacles = copy.deepcopy(self._base_obstacles)
            self.dyn_obstacles = []
            
        # Apply jitter if enabled
        if self.jitter_start:
            jitter = self.rng.uniform(-0.3, 0.3, size=2)
            candidate = self._base_start + jitter
            xmin, xmax, ymin, ymax = self.bounds
            candidate = np.clip(candidate, [xmin + 0.15, ymin + 0.15],
                               [xmax - 0.15, ymax - 0.15])
            # Ensure we didn't land inside an obstacle
            if not in_collision(candidate, self.obstacles, margin=0.15):
                self.start = candidate
            else:
                self.start = self._base_start.copy()
        else:
            self.start = self._base_start.copy()

        self.pos = self.start.copy()
        self.vel = np.zeros(2)
        self.heading = np.arctan2(*(self.goal - self.start)[::-1])
        self._prev_heading = self.heading
        self.step_n = 0
        self._prev_dist = np.linalg.norm(self.pos - self.goal)
        self._prev_Upot = (U_att_val(self.pos, self.goal) +
                          U_rep_val(self.pos, self.goal, self.obstacles))
        self.path = [self.pos.copy()]
        return self._obs()

    def _obs(self):
        """Get current observation"""
        to_goal = self.goal - self.pos
        dist_goal = np.linalg.norm(to_goal) + 1e-12
        dx, dy = to_goal / dist_goal

        min_d = min(get_distance(self.pos, obs)[0] for obs in self.obstacles)
        min_d = np.clip(min_d, 0.0, 5.0) / 5.0

        U_a = U_att_val(self.pos, self.goal)
        U_r = U_rep_val(self.pos, self.goal, self.obstacles)

        goal_angle = np.arctan2(to_goal[1], to_goal[0])
        heading_err = angle_wrap(goal_angle - self.heading) / np.pi
        ang_vel = angle_wrap(self.heading - self._prev_heading) / (np.pi / 4)
        ang_vel = np.clip(ang_vel, -1.0, 1.0)

        return np.array([
            dx, dy,
            np.clip(dist_goal / 15.0, 0.0, 1.0),
            np.clip(self.vel[0] / 0.2, -1.0, 1.0),
            np.clip(self.vel[1] / 0.2, -1.0, 1.0),
            min_d,
            np.clip(U_a / 50.0, 0.0, 1.0),
            np.clip(U_r / 10.0, 0.0, 5.0),
            float(heading_err),
            float(ang_vel),
        ], dtype=np.float32)

    def step(self, action):
        """Execute one step in the environment"""
        # Update dynamic obstacles
        if self.dynamic_obstacles:
            dt = 0.1
            for d in self.dyn_obstacles:
                d.step(dt, self.bounds)
            for i in range(len(self.dyn_obstacles)):
                for j in range(i+1, len(self.dyn_obstacles)):
                    self.dyn_obstacles[i].handle_obs_collision(self.dyn_obstacles[j])
            self.obstacles = [d.obs for d in self.dyn_obstacles]
            
        delta_h = float(action[0]) * self.MAX_DELTA_H
        speed = (float(action[1]) + 1.0) / 2.0
        speed = 0.4 + 0.6 * speed

        self._prev_heading = self.heading
        self.heading = angle_wrap(self.heading + delta_h)
        move = (np.array([np.cos(self.heading), np.sin(self.heading)])
                * self.STEP_SIZE * speed)

        prev_pos = self.pos.copy()
        new_pos = self.pos + move

        xmin, xmax, ymin, ymax = self.bounds
        new_pos[0] = clamp(new_pos[0], xmin + 0.01, xmax - 0.01)
        new_pos[1] = clamp(new_pos[1], ymin + 0.01, ymax - 0.01)

        hit = False
        for t in np.linspace(0, 1, 5):
            mid = prev_pos + t * (new_pos - prev_pos)
            if in_collision(mid, self.obstacles, margin=self.COLLISION_M):
                hit = True
                break

        self.vel = (new_pos - self.pos) if not hit else np.zeros(2)
        if not hit:
            self.pos = new_pos
        self.step_n += 1
        self.path.append(self.pos.copy())

        # Calculate reward
        dist_now = float(np.linalg.norm(self.pos - self.goal))
        progress = float(self._prev_dist - dist_now)

        to_goal = self.goal - self.pos
        goal_ang = float(np.arctan2(to_goal[1], to_goal[0]))
        heading_err = float(abs(angle_wrap(goal_ang - self.heading)))
        align = float(np.cos(heading_err))

        clearance = min(get_distance(self.pos, obs)[0] for obs in self.obstacles)
        clearance_clamped = max(clearance, 0.05)
        if clearance_clamped < self.D_SAFE:
            clearance_pen = -self.K_CLEARANCE * (1.0 / clearance_clamped - 1.0 / self.D_SAFE)
            clearance_pen = float(np.clip(clearance_pen, -5.0, 0.0))
        else:
            clearance_pen = 0.0

        margin_x = min(self.pos[0] - xmin, xmax - self.pos[0])
        margin_y = min(self.pos[1] - ymin, ymax - self.pos[1])
        wall_d = float(min(margin_x, margin_y))
        boundary_pen = -self.K_BOUNDARY * max(0.0, (self.BOUND_SAFE - wall_d) / self.BOUND_SAFE)

        action_pen = -self.K_ACTION * abs(delta_h)

        los_bonus = self.K_LOS * (1.0 if has_los(self.pos, self.goal, self.obstacles) else 0.0)

        reward = (
            self.R_STEP +
            self.K_PROGRESS * progress +
            self.K_ALIGN * align +
            clearance_pen +
            boundary_pen +
            action_pen +
            los_bonus
        )

        self._prev_dist = dist_now

        done = False
        if dist_now <= self.goal_radius:
            reward += self.R_GOAL
            done = True
        elif hit:
            reward += self.R_COLLISION
            done = True
        elif self.step_n >= self.MAX_STEPS:
            done = True

        info = {
            "success": dist_now <= self.goal_radius,
            "collision": hit,
        }

        return self._obs(), float(reward), done, info

    def path_array(self):
        """Get path as numpy array"""
        return np.array(self.path)

    def render(self, ax=None, show_path=True, title=None):
        """
        Render the environment with path visualization
        
        Args:
            ax: Matplotlib axis (creates new figure if None)
            show_path: Whether to show the UAV path
            title: Title for the plot
            
        Returns:
            ax: Matplotlib axis with rendered environment
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = None
        
        xmin, xmax, ymin, ymax = self.bounds
        
        # Draw bounds
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.set_ylim(ymin - 0.5, ymax + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw boundary
        boundary = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(boundary)
        
        # Draw obstacles
        for i, obs in enumerate(self.obstacles):
            if obs['type'] == 'rectangle':
                cx, cy, hw, hh = obs['params']
                rect = plt.Rectangle((cx - hw, cy - hh), 2*hw, 2*hh,
                                    fill=True, facecolor='red', edgecolor='darkred',
                                    alpha=0.6, linewidth=1.5)
                ax.add_patch(rect)
            elif obs['type'] == 'circle':
                cx, cy, r = obs['params']
                circle = plt.Circle((cx, cy), r, fill=True, facecolor='orange',
                                   edgecolor='orange', alpha=0.6, linewidth=1.5)
                ax.add_patch(circle)
            elif obs['type'] == 'triangle':
                verts = obs['params']
                triangle = MplPolygon(verts, fill=True, facecolor='brown',
                                     edgecolor='maroon', alpha=0.6, linewidth=1.5)
                ax.add_patch(triangle)
            elif obs['type'] == 'square':
                cx, cy, hw, hh = obs['params']
                rect = plt.Rectangle((cx - hw, cy - hh), 2*hw, 2*hh,
                                    fill=True, facecolor='purple', edgecolor='purple',
                                    alpha=0.6, linewidth=1.5)
                ax.add_patch(rect)
        
        # Draw path if available
        if show_path and len(self.path) > 1:
            path_array = np.array(self.path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b-', alpha=0.5,
                   linewidth=1, label='UAV Path')
        
        # Draw start position
        ax.plot(self.start[0], self.start[1], 'go', markersize=12,
               label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Draw goal
        goal_circle = plt.Circle(self.goal, self.goal_radius, fill=False,
                                edgecolor='green', linewidth=2, linestyle='--',
                                label='Goal')
        ax.add_patch(goal_circle)
        ax.plot(self.goal[0], self.goal[1], 'g*', markersize=20,
               label='Goal Center', markeredgecolor='darkgreen', markeredgewidth=1)
        
        # Draw current position
        ax.plot(self.pos[0], self.pos[1], 'ro', markersize=10,
               label='Current Position', markeredgecolor='darkred', markeredgewidth=2)
        
        # Draw heading
        heading_length = 0.5
        heading_end = self.pos + heading_length * np.array([np.cos(self.heading),
                                                           np.sin(self.heading)])
        ax.arrow(self.pos[0], self.pos[1],
                heading_end[0] - self.pos[0], heading_end[1] - self.pos[1],
                head_width=0.2, head_length=0.15, fc='red', ec='red', alpha=0.8)
        
        ax.set_xlabel('X (units)', fontsize=12)
        ax.set_ylabel('Y (units)', fontsize=12)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            dist_to_goal = np.linalg.norm(self.pos - self.goal)
            ax.set_title(f'ARC Environment (Step {self.step_n}, Distance to Goal: {dist_to_goal:.2f})',
                        fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        
        return ax