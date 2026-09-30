"""
Moving-obstacle training for the learned baselines.

make_dynamic(env) turns an arc_2d UAVEnvironment (layout 'corridor_gauntlet') into a
training environment whose obstacles move with the SAME motion model, speeds and
agent-to-obstacle time scale as the paired evaluation suite (harness.ObstacleTimeline,
one 3.0 m agent step = ARC_DT = 1/6 s of obstacle motion at 18 m/s). Every episode draws
a fresh obstacle-velocity seed from [TRAIN_SEED_BASE, TRAIN_SEED_BASE + 1e9), which
never overlaps the evaluation seeds 0..49. A step also ends as a collision if the
exact clearance to any obstacle drops below 5 m (the evaluation criterion), which
catches obstacles that hit the agent from outside the forward ray fan.
"""
import types

import numpy as np

import harness as H

TRAIN_SEED_BASE = 1_000_000


def make_dynamic(env, collision_radius=H.COLLISION_RADIUS):
    base = [dict(o) for o in H.OBSTACLES]
    base_geoms = H._to_shapely(base)
    orig_reset, orig_step = env.reset, env.step
    k_terminal = 100.0
    env.dynamic_obstacles = True

    def _init_dynamic_obstacle_velocities(self):
        seed = TRAIN_SEED_BASE + int(np.random.randint(0, 1_000_000_000))
        self._tl = H.ObstacleTimeline(seed, base)
        self._t = 0.0
        self._cur_obs = self._tl.at(0.0)

    def _update_dynamic_obstacles(self):
        self._t += H.ARC_DT
        self._cur_obs = self._tl.at(self._t)
        self.obstacles = H._to_shapely(self._cur_obs)

    def reset(self, uav_start_pos=None):
        self.obstacles = base_geoms          # start-position check against the t = 0 layout
        obs = orig_reset(uav_start_pos)
        return obs

    def step(self, action):
        obs, reward, done, info = orig_step(action)
        if not done:
            c = H.clearance(self.uav_pos, self._cur_obs)
            if c < collision_radius:
                done = True
                reward -= k_terminal
                info["collision"], info["success"] = True, False
        return obs, reward, done, info

    env._init_dynamic_obstacle_velocities = types.MethodType(_init_dynamic_obstacle_velocities, env)
    env._update_dynamic_obstacles = types.MethodType(_update_dynamic_obstacles, env)
    env.reset = types.MethodType(reset, env)
    env.step = types.MethodType(step, env)
    return env
