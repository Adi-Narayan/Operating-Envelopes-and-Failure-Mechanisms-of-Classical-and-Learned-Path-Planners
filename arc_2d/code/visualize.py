"""
Visualization utilities for ARC environment - Notebook style
Matches the exact visualization format from ARC.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from environment import ARCEnvironment
from config import ARC_CONFIG


def plot_training_curves(trainer_rewards, trainer_successes=None, save_path=None):
    """
    Plot training curves in notebook style
    Shows episode rewards and success rate
    
    Args:
        trainer_rewards: List of rewards per episode
        trainer_successes: List of success indicators per episode
        save_path: Optional path to save figure
    """
    episodes = np.arange(1, len(trainer_rewards) + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Episode Rewards
    ax1.plot(episodes, trainer_rewards, 'b-', alpha=0.7, linewidth=1.5, label='Episode Reward')
    
    # Add rolling average
    window = min(100, len(trainer_rewards) // 4)
    if window > 1:
        rolling_avg = np.convolve(trainer_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(np.arange(window, len(trainer_rewards) + 1), rolling_avg, 
                'r--', alpha=0.8, linewidth=2, label=f'Rolling Avg ({window} ep)')
    
    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Training Progress - Episode Rewards', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    if trainer_successes is not None:
        success_rate = []
        for i in range(len(trainer_successes)):
            window_sr = min(100, i + 1)
            sr = np.mean(trainer_successes[max(0, i - window_sr + 1):i + 1]) * 100
            success_rate.append(sr)
        
        ax2.plot(episodes, success_rate, 'g-', alpha=0.7, linewidth=1.5, label='Success Rate')
        ax2.fill_between(episodes, 0, success_rate, alpha=0.2, color='green')
        ax2.set_ylim([0, 105])
    else:
        # If no success data, show average reward trend
        success_rate_alt = []
        window_alt = min(50, len(trainer_rewards) // 5)
        for i in range(len(trainer_rewards)):
            sr = np.mean(trainer_rewards[max(0, i - window_alt + 1):i + 1])
            success_rate_alt.append(sr)
        ax2.plot(episodes, success_rate_alt, 'g-', alpha=0.7, linewidth=1.5, label='Average Reward Trend')
        ax2.legend(loc='best', fontsize=11)
    
    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold') if trainer_successes else ax2.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Success Rate (Last 100 Episodes)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_eval_paths(eval_results, n_show=10, save_path=None):
    """
    Plot evaluation episode paths in notebook style
    Shows multiple trajectories from evaluation runs
    
    Args:
        eval_results: List of evaluation result dicts with 'path' or 'positions' keys
        n_show: Number of paths to show at once
        save_path: Optional path to save figure
    """
    n_eps = min(n_show, len(eval_results))
    n_cols = min(5, n_eps)
    n_rows = (n_eps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_eps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    xmin, xmax, ymin, ymax = ARC_CONFIG['ENV_BOUNDS']
    
    for idx in range(n_eps):
        ax = axes[idx]
        result = eval_results[idx]
        
        # Get path
        if 'path' in result:
            path = result['path']
        elif 'positions' in result:
            path = result['positions']
        else:
            path = None
        
        # Draw environment
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.set_ylim(ymin - 0.5, ymax + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Draw boundaries
        boundary = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(boundary)
        
        # Draw goal
        goal = ARC_CONFIG['GOAL']
        goal_circle = Circle(goal, ARC_CONFIG['GOAL_RADIUS'], fill=False,
                            edgecolor='green', linewidth=2, linestyle='--')
        ax.add_patch(goal_circle)
        ax.plot(goal[0], goal[1], 'g*', markersize=20, label='Goal')
        
        # Draw start
        start = ARC_CONFIG['START']
        ax.plot(start[0], start[1], 'bo', markersize=10, label='Start')
        
        # Draw path
        if path is not None:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.6, linewidth=1.5, label='Path')
            if len(path) > 0:
                ax.plot(path[-1, 0], path[-1, 1], 'ro', markersize=8, label='End')
        
        # Title with result info
        success = result.get('success', False)
        collision = result.get('collision', False)
        status = "✓ Success" if success else ("✗ Collision" if collision else "Running")
        ax.set_title(f"Episode {idx+1} - {status}", fontsize=11, fontweight='bold')
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_eps, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Evaluation Paths (showing {n_eps}/{len(eval_results)} episodes)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_comparison(apf_metrics, arc_metrics, save_path=None):
    """
    Plot ARC vs APF benchmark comparison in notebook style
    
    Args:
        apf_metrics: Dict with APF performance metrics
        arc_metrics: Dict with ARC performance metrics
        save_path: Optional path to save figure
    """
    metrics = list(apf_metrics.keys())
    apf_vals = [apf_metrics[m] for m in metrics]
    arc_vals = [arc_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, apf_vals, width, label='APF', color='skyblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, arc_vals, width, label='ARC', color='lightcoral', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Benchmark Comparison: APF vs ARC', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_episode(env, episode_num=0, save_path=None):
    """
    Plot a single episode of the environment with the UAV path
    
    Args:
        env: Environment after running an episode
        episode_num: Episode number (for title)
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render(ax=ax, show_path=True, title=f"Episode {episode_num} - UAV Path and Environment")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig, ax


def plot_episode_summary(env, episode_reward, episode_info, episode_num=0, save_path=None):
    """
    Plot episode with summary statistics
    
    Args:
        env: Environment after running an episode
        episode_reward: Total reward for episode
        episode_info: Info dict from step
        episode_num: Episode number
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Environment visualization
    env.render(ax=ax1, show_path=True, title=f"Episode {episode_num} - Trajectory")
    
    # Right: Episode statistics
    ax2.axis('off')
    stats_text = f"""
    Episode {episode_num} Summary
    ═══════════════════════════════
    
    Total Reward:     {episode_reward:.2f}
    Path Length:      {len(env.path)} steps
    Success:          {episode_info.get('success', False)}
    Collision:        {episode_info.get('collision', False)}
    
    Start Position:   {env.start}
    End Position:     {env.pos}
    Goal Position:    {env.goal}
    
    Distance to Goal: {np.linalg.norm(env.pos - env.goal):.3f}
    Goal Radius:      {env.goal_radius}
    """
    
    ax2.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("ARC Visualization Module")
    print("Available functions:")
    print("  - plot_training_curves(rewards, successes)")
    print("  - plot_eval_paths(results, n_show=10)")
    print("  - plot_comparison(apf_metrics, arc_metrics)")
    print("  - plot_episode(env, episode_num)")
    print("  - plot_episode_summary(env, reward, info, episode_num)")


def plot_episode(env, episode_num=0, save_path=None):
    """
    Plot a single episode of the environment with the UAV path
    
    Args:
        env: Environment after running an episode
        episode_num: Episode number (for title)
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render(ax=ax, show_path=True, title=f"Episode {episode_num} - UAV Path and Environment")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig, ax


def plot_training_progress(rewards, successes=None, collisions=None, save_path=None):
    """
    Plot training progress over episodes
    
    Args:
        rewards: List of episode rewards
        successes: List of success indicators (0/1)
        collisions: List of collision indicators (0/1)
        save_path: Optional path to save the figure
    """
    episodes = np.arange(len(rewards))
    
    if successes is not None and collisions is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot rewards
        axes[0].plot(episodes, rewards, color='blue', alpha=0.7, linewidth=1.5, label='Episode Reward')
        axes[0].axhline(y=np.mean(rewards), color='blue', linestyle='--', alpha=0.5, label='Mean Reward')
        axes[0].set_ylabel('Reward', fontsize=11, fontweight='bold')
        axes[0].set_title('Training Progress', fontsize=13, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot success rate (over last 100 episodes)
        window = min(100, len(successes))
        success_rate = []
        for i in range(len(successes)):
            start = max(0, i - window + 1)
            sr = np.mean(successes[start:i+1]) * 100
            success_rate.append(sr)
        
        axes[1].plot(episodes, success_rate, color='green', alpha=0.7, linewidth=1.5, label='Success Rate')
        axes[1].set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
        axes[1].set_ylim([0, 105])
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # Plot collision rate (over last 100 episodes)
        collision_rate = []
        for i in range(len(collisions)):
            start = max(0, i - window + 1)
            cr = np.mean(collisions[start:i+1]) * 100
            collision_rate.append(cr)
        
        axes[2].plot(episodes, collision_rate, color='red', alpha=0.7, linewidth=1.5, label='Collision Rate')
        axes[2].set_ylabel('Collision Rate (%)', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Episode', fontsize=11, fontweight='bold')
        axes[2].set_ylim([0, 105])
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(episodes, rewards, color='blue', alpha=0.7, linewidth=1.5, label='Episode Reward')
        ax.axhline(y=np.mean(rewards), color='blue', linestyle='--', alpha=0.5, label='Mean Reward')
        ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=11, fontweight='bold')
        ax.set_title('Training Progress - Episode Rewards', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_episode_comparison(states_list, rewards_list, titles=None, save_path=None):
    """
    Plot multiple episodes side-by-side for comparison
    
    Args:
        states_list: List of (positions, obstacles) tuples
        rewards_list: List of episode rewards
        titles: List of titles for each subplot
        save_path: Optional path to save the figure
    """
    n_episodes = len(states_list)
    n_cols = min(3, n_episodes)
    n_rows = (n_episodes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_episodes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (ax, reward) in enumerate(zip(axes, rewards_list)):
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw boundaries
        boundary = Rectangle((0, 0), 10, 10, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(boundary)
        
        # Draw goal
        goal = ARC_CONFIG['GOAL']
        goal_circle = Circle(goal, ARC_CONFIG['GOAL_RADIUS'], fill=False,
                            edgecolor='green', linewidth=2, linestyle='--')
        ax.add_patch(goal_circle)
        ax.plot(goal[0], goal[1], 'g*', markersize=20)
        
        # Set title
        if titles and idx < len(titles):
            title = f"{titles[idx]} (Reward: {reward:.2f})"
        else:
            title = f"Episode {idx+1} (Reward: {reward:.2f})"
        ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(rewards_list), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def visualize_episode(env, num_steps=None, save_gif=False, save_dir='visualization'):
    """
    Run an episode and visualize it
    
    Args:
        env: ARCEnvironment instance
        num_steps: Number of steps to run (None = full episode)
        save_gif: Whether to save frames for GIF creation
        save_dir: Directory to save frames
    """
    import os
    
    obs = env.reset()
    done = False
    step = 0
    episode_reward = 0.0
    
    if save_gif:
        os.makedirs(save_dir, exist_ok=True)
    
    frames = []
    
    while not done and (num_steps is None or step < num_steps):
        # Take random action
        action = np.random.uniform(-1, 1, 2)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        # Render every 10 steps
        if step % 10 == 0:
            fig, ax = plt.subplots(figsize=(10, 10))
            env.render(ax=ax, show_path=True,
                      title=f"Step {step}: Reward Sum = {episode_reward:.2f}")
            
            if save_gif:
                frame_path = os.path.join(save_dir, f'frame_{step:04d}.png')
                plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                frames.append(frame_path)
            
            plt.show()
        
        step += 1
    
    print(f"\nEpisode finished!")
    print(f"Total steps: {step}")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Success: {info['success']}")
    print(f"Collision: {info['collision']}")
    
    if save_gif and frames:
        print(f"\nSaved {len(frames)} frames to {save_dir}")
        print(f"Hint: Use imageio or PIL to create GIF from frames")
    
    return step, episode_reward, info


if __name__ == "__main__":
    # Example: Visualize a single episode
    print("Creating ARC environment...")
    env = ARCEnvironment()
    
    print("\nRunning episode with visualization...")
    print("(This will display 1 plot every 10 steps)")
    steps, total_reward, info = visualize_episode(env, num_steps=100)
    
    # Create a summary plot
    print("\nCreating trajectory plot...")
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render(ax=ax, show_path=True, title="Complete Episode Trajectory")
    plt.tight_layout()
    plt.show()
    
    print("Visualization complete!")
