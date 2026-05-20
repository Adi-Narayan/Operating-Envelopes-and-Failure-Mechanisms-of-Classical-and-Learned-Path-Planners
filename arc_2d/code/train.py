"""
Training script for ARC algorithm
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import (ENV_CONFIG, NETWORK_CONFIG, TRAIN_CONFIG, ENVIRONMENTS, 
                   MODEL_CONFIG, TB_CONFIG)
from environment import UAVEnvironment
from networks import ARCAgent
from replay_buffer import ReplayBuffer, ObservationHistory
import time

class ARCTrainer:
    """Trainer for ARC algorithm"""
    
    def __init__(self, env_name='env7', device=None):
        """
        Initialize trainer
        
        Args:
            env_name: Name of environment from config
            device: PyTorch device (auto-detected if None)
        """
        self.training_times = []
        self.env_name = env_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Create environment
        self.env_def = ENVIRONMENTS[env_name]
        self.env = UAVEnvironment(self.env_def, ENV_CONFIG)
        
        # Observation and action dimensions
        self.obs_dim = 3 + ENV_CONFIG['num_sensors']  # psi, chi, d0, + sensor distances
        self.action_dim = 1
        
        # Create agent (merge network and training configs)
        agent_config = {**NETWORK_CONFIG, **TRAIN_CONFIG}
        self.agent = ARCAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            config=agent_config,
            device=self.device
        )
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=TRAIN_CONFIG['memory_capacity'],
            obs_dim=self.obs_dim,
            history_length=NETWORK_CONFIG['history_length'],
            device=self.device
        )
        
        # Create directories
        os.makedirs(MODEL_CONFIG['save_dir'], exist_ok=True)
        os.makedirs(TB_CONFIG['log_dir'], exist_ok=True)
        
        # TensorBoard writer
        log_dir = os.path.join(TB_CONFIG['log_dir'], f'ARC_{env_name}')
        self.writer = SummaryWriter(log_dir)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = []
        self.collision_rate = []
        
        # Best model tracking
        self.best_avg_reward = -float('inf')
        
    def train(self, num_episodes=None):
        """
        Train the agent
        
        Args:
            num_episodes: Number of episodes to train (uses config if None)
        """
        start_time = time.time()
        if num_episodes is None:
            num_episodes = TRAIN_CONFIG['num_episodes']
        
        print(f"\nStarting training for {num_episodes} episodes...")
        print(f"Environment: {self.env_def['name']}")
        print(f"Number of parallel UAVs: {ENV_CONFIG['num_uavs']}")
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            episode_stats = self._run_episode(episode)
            
            # Log statistics
            self._log_episode(episode, episode_stats)
            
            # Save model periodically
            if (episode + 1) % MODEL_CONFIG['save_interval'] == 0:
                self._save_checkpoint(episode + 1)
            
            # Save best model
            if episode >= 20:  # Start tracking after some episodes
                recent_avg_reward = np.mean(self.episode_rewards[-20:])
                if recent_avg_reward > self.best_avg_reward:
                    self.best_avg_reward = recent_avg_reward
                    self._save_best_model()

            elapsed = time.time() - start_time

            if (episode + 1) % 50 == 0:
                self.training_times.append((episode + 1, elapsed))
                print(f"Episode {episode+1} | Training time: {elapsed/60:.2f} minutes")
        
        print("\nTraining completed!")
        self.writer.close()
        
        # Final save
        self._save_checkpoint(num_episodes)
    
    def _run_episode(self, episode):
        """Run one training episode with multiple UAVs"""
        episode_stats = {
            'rewards': [],
            'steps': [],
            'successes': 0,
            'collisions': 0,
            'losses': [],
        }
        
        # Run multiple UAVs in parallel (exploring starts)
        for uav_idx in range(ENV_CONFIG['num_uavs']):
            # Reset environment
            obs = self.env.reset()
            
            # Initialize observation history
            obs_history = ObservationHistory(
                history_length=NETWORK_CONFIG['history_length'],
                obs_dim=self.obs_dim
            )
            obs_history.reset(obs)
            
            # Episode variables
            episode_reward = 0
            step = 0
            done = False
            hidden = None
            
            while not done and step < ENV_CONFIG['max_steps']:
                # Select action
                history_tensor = obs_history.get_tensor(self.device)
                action, hidden = self.agent.select_action(
                    history_tensor, 
                    hidden,
                    noise_scale=TRAIN_CONFIG['exploration_noise']
                )
                
                # Execute action
                next_obs, reward, done, info = self.env.step(action[0])
                
                # Update observation history
                next_obs_history = ObservationHistory(
                    history_length=NETWORK_CONFIG['history_length'],
                    obs_dim=self.obs_dim
                )
                # Copy current history and add new observation
                for h in list(obs_history.history)[1:]:
                    next_obs_history.history.append(h)
                next_obs_history.history.append(next_obs)
                
                # Store transition in replay buffer
                self.replay_buffer.push(
                    history=obs_history.get(),
                    action=action,
                    reward=reward,
                    next_history=next_obs_history.get(),
                    done=done
                )
                
                # Update for next step
                obs_history = next_obs_history
                episode_reward += reward
                step += 1
                
                # Train if enough samples
                if len(self.replay_buffer) >= TRAIN_CONFIG['batch_size']:
                    batch = self.replay_buffer.sample(TRAIN_CONFIG['batch_size'])
                    losses = self.agent.update(
                        batch,
                        gamma=TRAIN_CONFIG['gamma'],
                        tau=TRAIN_CONFIG['tau']
                    )
                    episode_stats['losses'].append(losses)
            
            # Record episode statistics
            episode_stats['rewards'].append(episode_reward)
            episode_stats['steps'].append(step)
            if 'success' in info and info['success']:
                episode_stats['successes'] += 1
            if 'collision' in info and info['collision']:
                episode_stats['collisions'] += 1
        
        return episode_stats
    
    def _log_episode(self, episode, stats):
        """Log episode statistics to TensorBoard and internal tracking"""
        # Average across UAVs
        avg_reward = np.mean(stats['rewards'])
        avg_steps = np.mean(stats['steps'])
        success_rate = stats['successes'] / ENV_CONFIG['num_uavs']
        collision_rate = stats['collisions'] / ENV_CONFIG['num_uavs']
        
        # Store for plotting
        self.episode_rewards.append(avg_reward)
        self.episode_steps.append(avg_steps)
        self.success_rate.append(success_rate)
        self.collision_rate.append(collision_rate)
        
        # Log to TensorBoard
        self.writer.add_scalar('Training/Episode_Reward', avg_reward, episode)
        self.writer.add_scalar('Training/Episode_Steps', avg_steps, episode)
        self.writer.add_scalar('Training/Success_Rate', success_rate, episode)
        self.writer.add_scalar('Training/Collision_Rate', collision_rate, episode)
        
        # Log losses if available
        if stats['losses']:
            avg_critic_loss = np.mean([l['critic_loss'] for l in stats['losses']])
            avg_actor_loss = np.mean([l['actor_loss'] for l in stats['losses']])
            avg_q = np.mean([l['avg_q'] for l in stats['losses']])
            
            self.writer.add_scalar('Training/Critic_Loss', avg_critic_loss, episode)
            self.writer.add_scalar('Training/Actor_Loss', avg_actor_loss, episode)
            self.writer.add_scalar('Training/Avg_Q_Value', avg_q, episode)
        
        # Log buffer size
        self.writer.add_scalar('Training/Buffer_Size', len(self.replay_buffer), episode)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"\nEpisode {episode + 1}:")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Steps: {avg_steps:.1f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Collision Rate: {collision_rate:.2%}")
    
    def _save_checkpoint(self, episode):
        """Save model checkpoint"""
        filepath = os.path.join(
            MODEL_CONFIG['save_dir'],
            f"{MODEL_CONFIG['checkpoint_prefix']}_ep{episode}.pth"
        )
        self.agent.save(filepath)
        print(f"\nCheckpoint saved: {filepath}")
    
    def _save_best_model(self):
        """Save best performing model"""
        filepath = os.path.join(
            MODEL_CONFIG['save_dir'],
            MODEL_CONFIG['best_model_name']
        )
        self.agent.save(filepath)
        print(f"\nBest model saved: {filepath} (Avg reward: {self.best_avg_reward:.2f})")


def train_arc(env_name='env7', num_episodes=None):
    """
    Convenience function to train ARC
    
    Args:
        env_name: Environment name from config
        num_episodes: Number of episodes (uses config default if None)
    """
    trainer = ARCTrainer(env_name=env_name)
    trainer.train(num_episodes=num_episodes)
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ARC agent')
    parser.add_argument('--env', type=str, default='env7',
                       help='Environment name from config')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes')
    
    args = parser.parse_args()
    
    train_arc(env_name=args.env, num_episodes=args.episodes)