"""
Actor-Recurrent-Critic (ARC) Neural Networks
Implements the network architecture from the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMFeatureExtractor(nn.Module):
    """LSTM network for extracting features from observation history"""
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            hidden: Optional previous hidden state
            
        Returns:
            output: (batch_size, hidden_size)
            hidden: New hidden state
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last output
        output = lstm_out[:, -1, :]
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        )


class Actor(nn.Module):
    """Actor network - outputs action given state"""
    
    def __init__(self, lstm_output_size, hidden_size, action_dim=1):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch_size, lstm_output_size)
            
        Returns:
            action: (batch_size, action_dim) in range [-1, 1]
        """
        x = F.relu(self.fc1(lstm_output))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        
        return action


class Critic(nn.Module):
    """Critic network - outputs Q-value given state and action"""
    
    def __init__(self, lstm_output_size, action_dim, hidden_size):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(lstm_output_size + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output, action):
        """
        Args:
            lstm_output: (batch_size, lstm_output_size)
            action: (batch_size, action_dim)
            
        Returns:
            q_value: (batch_size, 1)
        """
        x = torch.cat([lstm_output, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value


class ARCNetwork(nn.Module):
    """Complete ARC network combining LSTM, Actor, and Critic"""
    
    def __init__(self, obs_dim, action_dim, lstm_hidden_size, lstm_num_layers, 
                 actor_hidden_size, critic_hidden_size):
        super(ARCNetwork, self).__init__()
        
        # Shared LSTM feature extractor
        self.lstm = LSTMFeatureExtractor(
            input_size=obs_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers
        )
        
        # Actor network
        self.actor = Actor(
            lstm_output_size=lstm_hidden_size,
            hidden_size=actor_hidden_size,
            action_dim=action_dim
        )
        
        # Critic network
        self.critic = Critic(
            lstm_output_size=lstm_hidden_size,
            action_dim=action_dim,
            hidden_size=critic_hidden_size
        )
    
    def forward_actor(self, history, hidden=None):
        """Forward pass through LSTM and Actor"""
        lstm_output, new_hidden = self.lstm(history, hidden)
        action = self.actor(lstm_output)
        return action, new_hidden
    
    def forward_critic(self, history, action, hidden=None):
        """Forward pass through LSTM and Critic"""
        lstm_output, new_hidden = self.lstm(history, hidden)
        q_value = self.critic(lstm_output, action)
        return q_value, new_hidden
    
    def get_lstm_output(self, history, hidden=None):
        """Get LSTM output only"""
        return self.lstm(history, hidden)


class ARCAgent:
    """ARC Agent managing policy and value networks"""
    
    def __init__(self, obs_dim, action_dim, config, device):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Create policy network (actor + LSTM)
        self.policy_net = ARCNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lstm_hidden_size=config['lstm_hidden_size'],
            lstm_num_layers=config['lstm_num_layers'],
            actor_hidden_size=config['actor_hidden_size'],
            critic_hidden_size=config['critic_hidden_size']
        ).to(device)
        
        # Create target policy network
        self.target_policy_net = ARCNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lstm_hidden_size=config['lstm_hidden_size'],
            lstm_num_layers=config['lstm_num_layers'],
            actor_hidden_size=config['actor_hidden_size'],
            critic_hidden_size=config['critic_hidden_size']
        ).to(device)
        
        # Copy parameters to target network
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.RMSprop(
            list(self.policy_net.lstm.parameters()) + list(self.policy_net.actor.parameters()),
            lr=config['learning_rate']
        )
        
        self.critic_optimizer = torch.optim.RMSprop(
            list(self.policy_net.lstm.parameters()) + list(self.policy_net.critic.parameters()),
            lr=config['learning_rate']
        )
    
    def select_action(self, history, hidden=None, noise_scale=0.0):
        """
        Select action given observation history
        
        Args:
            history: (1, seq_len, obs_dim) or (seq_len, obs_dim)
            hidden: Optional LSTM hidden state
            noise_scale: Exploration noise scale
            
        Returns:
            action: Numpy array
            new_hidden: New hidden state
        """
        if len(history.shape) == 2:
            history = history.unsqueeze(0)
        
        with torch.no_grad():
            action, new_hidden = self.policy_net.forward_actor(history, hidden)
            
            if noise_scale > 0:
                noise = torch.randn_like(action) * noise_scale
                action = action + noise
                action = torch.clamp(action, -1, 1)
        
        return action.cpu().numpy()[0], new_hidden
    
    def update(self, batch, gamma, tau):
        """
        Update networks using a batch of experiences
        
        Args:
            batch: Dictionary with keys 'history', 'action', 'reward', 'next_history', 'done'
            gamma: Discount factor
            tau: Soft update parameter
            
        Returns:
            Dictionary with loss values
        """
        history = batch['history']
        action = batch['action']
        reward = batch['reward']
        next_history = batch['next_history']
        done = batch['done']
        
        # Compute target Q-value
        with torch.no_grad():
            next_action, _ = self.target_policy_net.forward_actor(next_history)
            next_q, _ = self.target_policy_net.forward_critic(next_history, next_action)
            target_q = reward + (1 - done) * gamma * next_q
        
        # Compute current Q-value
        current_q, _ = self.policy_net.forward_critic(history, action)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor loss (policy gradient)
        lstm_output, _ = self.policy_net.get_lstm_output(history)
        action_pred = self.policy_net.actor(lstm_output)
        actor_loss = -self.policy_net.critic(lstm_output.detach(), action_pred).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target network
        self._soft_update(tau)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'avg_q': current_q.mean().item()
        }
    
    def _soft_update(self, tau):
        """Soft update target network parameters"""
        for target_param, param in zip(self.target_policy_net.parameters(), 
                                       self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_policy_net': self.target_policy_net.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_policy_net.load_state_dict(checkpoint['target_policy_net'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])