"""
Visualize training results and create comparison plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import VIS_CONFIG


def load_training_data(trainer):
    """
    Extract training data from trainer object
    
    Args:
        trainer: ARCTrainer instance
        
    Returns:
        Dictionary with training data
    """
    return {
        'episode_rewards': np.array(trainer.episode_rewards),
        'episode_steps': np.array(trainer.episode_steps),
        'success_rate': np.array(trainer.success_rate),
        'collision_rate': np.array(trainer.collision_rate),
    }


def smooth_curve(data, window=20):
    """Apply moving average smoothing"""
    if len(data) < window:
        return data
    
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    # Pad to maintain original length
    pad_left = (len(data) - len(smoothed)) // 2
    pad_right = len(data) - len(smoothed) - pad_left
    return np.pad(smoothed, (pad_left, pad_right), mode='edge')


def plot_training_results(data, save_path=None):
    """
    Create comprehensive training results plot
    
    Args:
        data: Dictionary with training data
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    episodes = np.arange(len(data['episode_rewards']))
    
    # 1. Episode Rewards
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, data['episode_rewards'], alpha=0.3, color='blue', label='Raw')
    ax1.plot(episodes, smooth_curve(data['episode_rewards']), color='blue', linewidth=2, label='Smoothed')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Episode Steps
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, data['episode_steps'], alpha=0.3, color='green', label='Raw')
    ax2.plot(episodes, smooth_curve(data['episode_steps']), color='green', linewidth=2, label='Smoothed')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Steps', fontsize=12)
    ax2.set_title('Episode Steps (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Success Rate
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(episodes, data['success_rate'] * 100, alpha=0.3, color='purple', label='Raw')
    ax3.plot(episodes, smooth_curve(data['success_rate']) * 100, color='purple', linewidth=2, label='Smoothed')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Success Rate (%)', fontsize=12)
    ax3.set_title('Success Rate', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Collision Rate
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(episodes, data['collision_rate'] * 100, alpha=0.3, color='red', label='Raw')
    ax4.plot(episodes, smooth_curve(data['collision_rate']) * 100, color='red', linewidth=2, label='Smoothed')
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Collision Rate (%)', fontsize=12)
    ax4.set_title('Collision Rate (Lower is Better)', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Average Reward (windowed)
    ax5 = fig.add_subplot(gs[2, 0])
    window = 20
    if len(data['episode_rewards']) >= window:
        avg_rewards = [np.mean(data['episode_rewards'][max(0, i-window):i+1]) 
                      for i in range(len(data['episode_rewards']))]
        ax5.plot(episodes, avg_rewards, color='darkblue', linewidth=2)
        ax5.fill_between(episodes, avg_rewards, alpha=0.3)
    ax5.set_xlabel('Episode', fontsize=12)
    ax5.set_ylabel(f'Average Reward (last {window} episodes)', fontsize=12)
    ax5.set_title('Rolling Average Reward', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Combined Success/Collision
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(episodes, smooth_curve(data['success_rate']) * 100, 
            color='green', linewidth=2, label='Success Rate')
    ax6.plot(episodes, smooth_curve(data['collision_rate']) * 100, 
            color='red', linewidth=2, label='Collision Rate')
    ax6.set_xlabel('Episode', fontsize=12)
    ax6.set_ylabel('Rate (%)', fontsize=12)
    ax6.set_title('Success vs Collision Rate', fontsize=14, fontweight='bold')
    ax6.set_ylim(0, 105)
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.suptitle('ARC Training Results', fontsize=18, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training results: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison_results(env7_results, env8_results, save_path=None):
    """
    Plot comparison between environment 7 and 8 test results
    
    Args:
        env7_results: Test results dictionary for env7
        env8_results: Test results dictionary for env8
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['success_rate', 'collision_rate', 'avg_reward', 'avg_steps']
    titles = ['Success Rate', 'Collision Rate', 'Average Reward', 'Average Steps']
    colors = ['green', 'red', 'blue', 'purple']
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[idx // 2, idx % 2]
        
        env7_val = env7_results[metric]
        env8_val = env8_results[metric]
        
        # For rates, convert to percentage
        if 'rate' in metric:
            env7_val *= 100
            env8_val *= 100
            ylabel = 'Rate (%)'
        elif 'reward' in metric:
            ylabel = 'Reward'
        else:
            ylabel = 'Steps'
        
        bars = ax.bar(['Environment 7\n(Sparse Obstacles)', 'Environment 8\n(Dense Obstacles)'],
                     [env7_val, env8_val], color=[color, color], alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('ARC Performance Comparison: Sparse vs Dense Obstacles', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison results: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_summary_report(trainer, test_results_env7, test_results_env8, save_path):
    """
    Create a comprehensive summary report image
    
    Args:
        trainer: ARCTrainer instance
        test_results_env7: Test results for environment 7
        test_results_env8: Test results for environment 8
        save_path: Path to save report
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    training_data = load_training_data(trainer)
    episodes = np.arange(len(training_data['episode_rewards']))
    
    # Training curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1_twin = ax1.twinx()
    
    l1 = ax1.plot(episodes, smooth_curve(training_data['episode_rewards']), 
                 color='blue', linewidth=2, label='Avg Reward')
    l2 = ax1_twin.plot(episodes, smooth_curve(training_data['success_rate']) * 100, 
                      color='green', linewidth=2, label='Success Rate (%)')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12, color='blue')
    ax1_twin.set_ylabel('Success Rate (%)', fontsize=12, color='green')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Test comparison
    ax2 = fig.add_subplot(gs[0, 2])
    metrics = ['success_rate', 'collision_rate']
    env7_vals = [test_results_env7[m] * 100 for m in metrics]
    env8_vals = [test_results_env8[m] * 100 for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, env7_vals, width, label='Env7 (Sparse)', color='lightgreen', edgecolor='black')
    bars2 = ax2.bar(x + width/2, env8_vals, width, label='Env8 (Dense)', color='lightcoral', edgecolor='black')
    
    ax2.set_ylabel('Rate (%)', fontsize=12)
    ax2.set_title('Test Performance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Success', 'Collision'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Final statistics table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    final_episodes = 20
    final_avg_reward = np.mean(training_data['episode_rewards'][-final_episodes:])
    final_success = np.mean(training_data['success_rate'][-final_episodes:]) * 100
    final_collision = np.mean(training_data['collision_rate'][-final_episodes:]) * 100
    final_steps = np.mean(training_data['episode_steps'][-final_episodes:])
    
    table_data = [
        ['Metric', 'Training (last 20 ep)', 'Test Env7', 'Test Env8', 'Improvement'],
        ['Success Rate', f'{final_success:.1f}%', 
         f'{test_results_env7["success_rate"]*100:.1f}%',
         f'{test_results_env8["success_rate"]*100:.1f}%',
         f'{test_results_env7["success_rate"]*100 - 70:.1f}%'],
        ['Collision Rate', f'{final_collision:.1f}%',
         f'{test_results_env7["collision_rate"]*100:.1f}%',
         f'{test_results_env8["collision_rate"]*100:.1f}%',
         f'{13.6 - test_results_env8["collision_rate"]*100:.1f}%'],
        ['Avg Reward', f'{final_avg_reward:.1f}',
         f'{test_results_env7["avg_reward"]:.1f}',
         f'{test_results_env8["avg_reward"]:.1f}', '-'],
        ['Avg Steps', f'{final_steps:.1f}',
         f'{test_results_env7["avg_steps"]:.1f}',
         f'{test_results_env8["avg_steps"]:.1f}', '-'],
    ]
    
    table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, 5):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.suptitle('ARC Algorithm - Training Summary Report', fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary report: {save_path}")
    plt.close()


def visualize_training_results(trainer, test_results_env7=None, test_results_env8=None):
    """
    Create all training visualizations
    
    Args:
        trainer: ARCTrainer instance
        test_results_env7: Optional test results for env7
        test_results_env8: Optional test results for env8
    """
    vis_dir = VIS_CONFIG['save_dir']
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract training data
    training_data = load_training_data(trainer)
    
    # Create training results plot
    training_path = os.path.join(vis_dir, 'training_results.png')
    plot_training_results(training_data, save_path=training_path)
    
    # Create comparison plot if test results provided
    if test_results_env7 and test_results_env8:
        comparison_path = os.path.join(vis_dir, 'comparison_results.png')
        plot_comparison_results(test_results_env7, test_results_env8, save_path=comparison_path)
        
        # Create summary report
        summary_path = os.path.join(vis_dir, 'summary_report.png')
        create_summary_report(trainer, test_results_env7, test_results_env8, save_path=summary_path)


if __name__ == "__main__":
    print("This module should be imported and used with a trained ARCTrainer instance")
    print("Example usage:")
    print("  from train import ARCTrainer")
    print("  from visualize_training import visualize_training_results")
    print("  trainer = ARCTrainer('env7')")
    print("  # ... after training ...")
    print("  visualize_training_results(trainer)")