"""
Training script for ARC environment with simple policy
Demonstrates training with the ARC environment from the notebook
"""

import numpy as np
import torch
from environment import ARCEnvironment
from config import ARC_CONFIG
from tqdm import trange
import time


class SimpleARCTrainer:
    """Simple trainer for ARC environment using random/learned policy"""
    
    def __init__(self, seed=42, device=None):
        """Initialize trainer"""
        self.device = device if device else torch.device('cpu')
        self.rng = np.random.default_rng(seed)
        self.env = ARCEnvironment(seed=seed)
        print(f"Using device: {self.device}")
        print(f"Environment: Bounds {ARC_CONFIG['ENV_BOUNDS']}")
        
    def train_random_episodes(self, num_episodes=100, max_steps_per_episode=None):
        """Run training with random actions (baseline)"""
        
        if max_steps_per_episode is None:
            max_steps_per_episode = self.env.MAX_STEPS
        
        print(f"\n{'='*70}")
        print(f"Training ARC Environment with Random Policy")
        print(f"Episodes: {num_episodes}")
        print(f"Max steps per episode: {max_steps_per_episode}")
        print(f"{'='*70}\n")
        
        episode_rewards = []
        episode_successes = []
        episode_collisions = []
        episode_lengths = []
        episode_paths = []
        
        start_time = time.time()
        
        for episode in trange(num_episodes, desc="Training"):
            obs = self.env.reset()
            episode_reward = 0.0
            success = False
            collision = False
            
            for step in range(max_steps_per_episode):
                # Random action
                action = self.rng.uniform(-1, 1, size=self.env.ACT_DIM).astype(np.float32)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                success = info.get('success', False)
                collision = info.get('collision', False)
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_successes.append(success)
            episode_collisions.append(collision)
            episode_lengths.append(step + 1)
            
            # Store path for visualization
            episode_paths.append(self.env.path_array().copy())
            
            if (episode + 1) % max(1, num_episodes // 10) == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                success_rate = np.mean(episode_successes[-100:]) * 100
                collision_rate = np.mean(episode_collisions[-100:]) * 100
                print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:7.2f} | "
                      f"Success: {success_rate:5.1f}% | Collision: {collision_rate:5.1f}%")
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("Training Summary")
        print(f"{'='*70}")
        print(f"Total episodes: {num_episodes}")
        print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/num_episodes:.3f}s per episode)")
        print(f"Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.4f}")
        print(f"Final success rate (last 100): {np.mean(episode_successes[-100:]) * 100:.1f}%")
        print(f"Final collision rate (last 100): {np.mean(episode_collisions[-100:]) * 100:.1f}%")
        print(f"Average steps per episode: {np.mean(episode_lengths):.1f}")
        print(f"{'='*70}\n")
        
        return {
            'rewards': episode_rewards,
            'successes': episode_successes,
            'collisions': episode_collisions,
            'lengths': episode_lengths,
            'paths': episode_paths,
        }


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ARC environment agent')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Maximum steps per episode (default: env MAX_STEPS)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    trainer = SimpleARCTrainer(seed=args.seed)
    results = trainer.train_random_episodes(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps
    )
    
    return results


if __name__ == "__main__":
    main()
