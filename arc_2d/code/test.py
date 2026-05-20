"""
Testing script with visualization and animation
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle

from config import (ENV_CONFIG, NETWORK_CONFIG, ENVIRONMENTS, VIS_CONFIG, MODEL_CONFIG)
from environment import UAVEnvironment
from networks import ARCAgent
from replay_buffer import ObservationHistory


class ARCTester:
    """Tester for trained ARC agent"""
    
    def __init__(self, model_path, env_name, device=None):
        """
        Initialize tester
        
        Args:
            model_path: Path to trained model
            env_name: Environment name from config
            device: PyTorch device
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_name = env_name
        
        # Create environment
        self.env_def = ENVIRONMENTS[env_name]
        self.env = UAVEnvironment(self.env_def, ENV_CONFIG)
        
        # Find a safe collision-free fixed start position for testing
        self.test_start_pos = self._find_safe_test_position()
        
        # Observation and action dimensions
        self.obs_dim = 3 + ENV_CONFIG['num_sensors']
        self.action_dim = 1
        
        # Create and load agent
        from config import TRAIN_CONFIG
        agent_config = {**NETWORK_CONFIG, **TRAIN_CONFIG}
        self.agent = ARCAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            config=agent_config,
            device=self.device
        )
        self.agent.load(model_path)
        self.agent.policy_net.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Testing on: {self.env_def['name']}")
        print(f"Fixed test start position: [{self.test_start_pos[0]:.1f}, {self.test_start_pos[1]:.1f}]")
    
    def _find_safe_test_position(self):
        """Find a collision-free fixed start position for testing"""
        # Try multiple attempts to find a safe position
        for attempt in range(20):
            x_range, y_range = self.env_def['uav_start_range']
            test_pos = np.array([
                np.random.uniform(x_range[0], x_range[1]),
                np.random.uniform(y_range[0], y_range[1])
            ], dtype=np.float32)
            
            # Check if this position is collision-free
            try:
                obs = self.env.reset(test_pos)
                if obs is not None:
                    return test_pos
            except:
                continue
        
        # Fallback: use random position from range
        x_range, y_range = self.env_def['uav_start_range']
        return np.array([
            np.random.uniform(x_range[0], x_range[1]),
            np.random.uniform(y_range[0], y_range[1])
        ], dtype=np.float32)
    
    def test_episode(self, start_pos=None, max_steps=None):
        """
        Run a single test episode
        
        Args:
            start_pos: Fixed start position (uses test start position by default)
            max_steps: Maximum steps (uses config if None)
            
        Returns:
            Dictionary with trajectory and statistics
        """
        if max_steps is None:
            max_steps = ENV_CONFIG['max_steps']
        
        # Use fixed test start position if not specified
        if start_pos is None:
            start_pos = self.test_start_pos
        
        # Reset environment with fixed start position
        obs = self.env.reset(start_pos)
        
        # Initialize observation history
        obs_history = ObservationHistory(
            history_length=NETWORK_CONFIG['history_length'],
            obs_dim=self.obs_dim
        )
        obs_history.reset(obs)
        
        # Storage for trajectory
        trajectory = {
            'uav_positions': [self.env.uav_pos.copy()],
            'target_positions': [self.env.target_pos.copy()],
            'actions': [],
            'rewards': [],
            'observations': [obs.copy()],
            'obstacles': []
        }
        import copy
        if hasattr(self.env, 'active_obstacle_defs'):
            trajectory['obstacles'].append(copy.deepcopy(self.env.active_obstacle_defs))
        
        # Run episode
        total_reward = 0
        step = 0
        done = False
        hidden = None
        
        while not done and step < max_steps:
            # Select action (no exploration noise)
            history_tensor = obs_history.get_tensor(self.device)
            action, hidden = self.agent.select_action(history_tensor, hidden, noise_scale=0.0)
            
            # Execute action
            next_obs, reward, done, info = self.env.step(action[0])
            
            # Update observation history
            obs_history.append(next_obs)
            
            # Store data
            trajectory['uav_positions'].append(self.env.uav_pos.copy())
            trajectory['target_positions'].append(self.env.target_pos.copy())
            trajectory['actions'].append(action[0])
            trajectory['rewards'].append(reward)
            trajectory['observations'].append(next_obs.copy())
            if hasattr(self.env, 'active_obstacle_defs'):
                trajectory['obstacles'].append(copy.deepcopy(self.env.active_obstacle_defs))
            
            total_reward += reward
            step += 1
        
        # Convert to numpy arrays
        trajectory['uav_positions'] = np.array(trajectory['uav_positions'])
        trajectory['target_positions'] = np.array(trajectory['target_positions'])
        trajectory['actions'] = np.array(trajectory['actions'])
        trajectory['rewards'] = np.array(trajectory['rewards'])
        
        # Statistics
        stats = {
            'total_reward': total_reward,
            'steps': step,
            'success': info.get('success', False),
            'collision': info.get('collision', False),
            'distance_to_target': np.linalg.norm(self.env.uav_pos - self.env.target_pos),
        }
        
        return trajectory, stats
    
    def visualize_trajectory(self, trajectory, save_path=None):
        """
        Visualize the trajectory with enhanced aesthetics
        
        Args:
            trajectory: Trajectory dictionary from test_episode
            save_path: Path to save figure (shows if None)
        """
        fig, ax = plt.subplots(figsize=VIS_CONFIG['figure_size'], dpi=VIS_CONFIG['dpi'])
        
        # Set dark background for modern look
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        # Draw obstacles
        for obstacle in self.env.obstacles:
            if obstacle.geom_type == 'Polygon':
                coords = list(obstacle.exterior.coords)
                patch = MplPolygon(coords, facecolor='#34495e', edgecolor='#2c3e50', 
                                  alpha=0.7, linewidth=2)
                ax.add_patch(patch)
            elif obstacle.geom_type == 'MultiPolygon':
                for geom in obstacle.geoms:
                    coords = list(geom.exterior.coords)
                    patch = MplPolygon(coords, facecolor='#34495e', edgecolor='#2c3e50', 
                                      alpha=0.7, linewidth=2)
                    ax.add_patch(patch)
        
        # Draw target trajectory (if moving)
        target_positions = trajectory['target_positions']
        if self.env.target_motion:
            ax.plot(target_positions[:, 0], target_positions[:, 1], 
                   '--', color='#3498db', linewidth=2.5, alpha=0.6, 
                   label='Target Path', zorder=3)
        
        # Draw target position
        target_end = target_positions[-1]
        ax.plot(target_end[0], target_end[1], '*', 
               color='#27ae60', markersize=25, markeredgecolor='#fff', 
               markeredgewidth=2, label='Target', zorder=5)
        
        # Draw UAV trajectory
        uav_positions = trajectory['uav_positions']
        num_points = len(uav_positions)
        
        # Color gradient from red to green
        for i in range(num_points - 1):
            progress = i / num_points
            if progress < 0.5:
                r, g, b = 1.0, 2.0 * progress, 0
            else:
                r, g, b = 2.0 * (1.0 - progress), 1.0, 0
            
            ax.plot(uav_positions[i:i+2, 0], uav_positions[i:i+2, 1],
                   color=(r, g, b), linewidth=2, alpha=0.8, zorder=2)
        
        # Start and end markers
        ax.plot(uav_positions[0, 0], uav_positions[0, 1], 'o', 
               color='#e74c3c', markersize=10, markeredgecolor='#fff', 
               markeredgewidth=2, label='Start', zorder=4)
        ax.plot(uav_positions[-1, 0], uav_positions[-1, 1], 's', 
               color='#f39c12', markersize=10, markeredgecolor='#fff', 
               markeredgewidth=2, label='End', zorder=4)
        
        # Set axis properties
        ax.set_xlim(-10, ENV_CONFIG['width'] + 10)
        ax.set_ylim(-10, ENV_CONFIG['height'] + 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'{self.env_def["name"]} - Trajectory', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_animation(self, trajectory, save_path=None):
        """Create animation of the trajectory"""
        fig, ax = plt.subplots(figsize=VIS_CONFIG['figure_size'], dpi=VIS_CONFIG['dpi'])
        
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        # Draw static or dynamic obstacles
        obs_patches = []
        if 'obstacles' in trajectory and len(trajectory['obstacles']) > 0:
            for obs_def in trajectory['obstacles'][0]:
                if obs_def['type'] == 'polygon':
                    patch = MplPolygon(obs_def['vertices'], facecolor='#34495e', edgecolor='#2c3e50', alpha=0.7, linewidth=2)
                    ax.add_patch(patch)
                    obs_patches.append(patch)
                elif obs_def['type'] == 'circle':
                    patch = MplCircle(obs_def['center'], obs_def['radius'], facecolor='#34495e', edgecolor='#2c3e50', alpha=0.7, linewidth=2)
                    ax.add_patch(patch)
                    obs_patches.append(patch)
        else:
            for obstacle in self.env.obstacles:
                if obstacle.geom_type == 'Polygon':
                    coords = list(obstacle.exterior.coords)
                    patch = MplPolygon(coords, facecolor='#34495e', edgecolor='#2c3e50', alpha=0.7, linewidth=2)
                    ax.add_patch(patch)
                    obs_patches.append(patch)
        
        ax.set_xlim(-10, ENV_CONFIG['width'] + 10)
        ax.set_ylim(-10, ENV_CONFIG['height'] + 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        uav_positions = trajectory['uav_positions']
        target_positions = trajectory['target_positions']
        
        # Animation elements
        uav_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.6, label='UAV Path')
        uav_point, = ax.plot([], [], 'ro', markersize=10, label='UAV')
        target_point, = ax.plot([], [], 'g*', markersize=20, label='Target')
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(loc='upper left')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{self.env_def["name"]} - Animation')
        
        def animate(frame):
            if frame < len(uav_positions):
                uav_line.set_data(uav_positions[:frame+1, 0], uav_positions[:frame+1, 1])
                uav_point.set_data([uav_positions[frame, 0]], [uav_positions[frame, 1]])
                target_point.set_data([target_positions[frame, 0]], [target_positions[frame, 1]])
                
                distance = np.linalg.norm(uav_positions[frame] - target_positions[frame])
                info_text.set_text(f'Step: {frame}/{len(uav_positions)}\nDistance: {distance:.1f}m')
                
                if 'obstacles' in trajectory and len(trajectory['obstacles']) > frame:
                    frame_defs = trajectory['obstacles'][frame]
                    for patch, obs_def in zip(obs_patches, frame_defs):
                        if obs_def['type'] == 'polygon':
                            patch.set_xy(obs_def['vertices'])
                        elif obs_def['type'] == 'circle':
                            patch.set_center(obs_def['center'])
                            patch.set_radius(obs_def['radius'])
            
            return [uav_line, uav_point, target_point, info_text] + obs_patches
        
        anim = animation.FuncAnimation(fig, animate, frames=len(uav_positions),
                                      interval=100, blit=True, repeat=True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            anim.save(save_path, writer='pillow', fps=VIS_CONFIG['fps'])
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


def test_all_environments(model_dir, model_name):
    """Test on all environments"""
    model_path = os.path.join(model_dir, model_name)
    
    for env_name in ['env1', 'env2', 'env3', 'env4', 'env5', 'env6']:
        print(f"\n{'='*70}")
        print(f"Testing: {env_name}")
        print(f"{'='*70}")
        
        tester = ARCTester(model_path, env_name)
        trajectory, stats = tester.test_episode()
        
        print(f"\nResults:")
        print(f"  Steps: {stats['steps']}")
        print(f"  Reward: {stats['total_reward']:.2f}")
        print(f"  Success: {stats['success']}")
        print(f"  Collision: {stats['collision']}")
        print(f"  Distance to Target: {stats['distance_to_target']:.2f}m")
        
        # Visualize
        save_dir = VIS_CONFIG.get('save_dir', 'visualization')
        os.makedirs(save_dir, exist_ok=True)
        
        traj_path = os.path.join(save_dir, f'{env_name}_trajectory.png')
        tester.visualize_trajectory(trajectory, traj_path)
        
        anim_path = os.path.join(save_dir, f'{env_name}_animation.gif')
        tester.create_animation(trajectory, anim_path)
