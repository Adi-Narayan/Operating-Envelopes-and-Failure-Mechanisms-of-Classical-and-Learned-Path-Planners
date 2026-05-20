"""
Replay Buffer for storing and sampling experiences
"""

import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for ARC algorithm"""
    
    def __init__(self, capacity, obs_dim, history_length, device):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            obs_dim: Dimension of observation
            history_length: Length of observation history
            device: PyTorch device
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.device = device
        
        # Storage
        self.buffer = deque(maxlen=capacity)
        
    def push(self, history, action, reward, next_history, done):
        """
        Add experience to buffer
        
        Args:
            history: (history_length, obs_dim) numpy array
            action: Scalar or (action_dim,) numpy array
            reward: Scalar
            next_history: (history_length, obs_dim) numpy array
            done: Boolean
        """
        experience = {
            'history': history,
            'action': action,
            'reward': reward,
            'next_history': next_history,
            'done': done
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary with batched tensors
        """
        experiences = random.sample(self.buffer, batch_size)
        
        # Stack into batches
        histories = np.stack([e['history'] for e in experiences])
        actions = np.stack([e['action'] for e in experiences])
        rewards = np.array([e['reward'] for e in experiences])
        next_histories = np.stack([e['next_history'] for e in experiences])
        dones = np.array([e['done'] for e in experiences])
        
        # Convert to tensors
        batch = {
            'history': torch.FloatTensor(histories).to(self.device),
            'action': torch.FloatTensor(actions).to(self.device),
            'reward': torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            'next_history': torch.FloatTensor(next_histories).to(self.device),
            'done': torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        }
        
        # Ensure action is 2D
        if len(batch['action'].shape) == 1:
            batch['action'] = batch['action'].unsqueeze(1)
        
        return batch
    
    def __len__(self):
        return len(self.buffer)


class ObservationHistory:
    """Manages observation history for a single episode"""
    
    def __init__(self, history_length, obs_dim):
        """
        Args:
            history_length: Number of observations to keep
            obs_dim: Dimension of each observation
        """
        self.history_length = history_length
        self.obs_dim = obs_dim
        self.history = deque(maxlen=history_length)
        
    def reset(self, initial_obs):
        """
        Reset history with initial observation
        
        Args:
            initial_obs: (obs_dim,) numpy array
        """
        self.history.clear()
        # Fill with initial observation
        for _ in range(self.history_length):
            self.history.append(initial_obs.copy())
    
    def append(self, obs):
        """Add new observation to history"""
        self.history.append(obs.copy())
    
    def get(self):
        """Get current history as numpy array"""
        return np.array(list(self.history))
    
    def get_tensor(self, device):
        """Get current history as PyTorch tensor with batch dimension"""
        history_array = self.get()
        return torch.FloatTensor(history_array).unsqueeze(0).to(device)