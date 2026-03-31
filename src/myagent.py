import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    """Actor Network (Policy)"""
    def __init__(self, input_size=36, hidden_size=128, output_size=1296):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits


class Critic(nn.Module):
    """Critic Network (Value Function)"""
    def __init__(self, input_size=36, hidden_size=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class ActorCriticAgent:
    """
    Actor-Critic Agent for Checkers
    Combines Actor and Critic networks with action selection
    """
    def __init__(self, learning_rate_actor=0.001, learning_rate_critic=0.001, gamma=0.99):
        self.actor = Actor()
        self.critic = Critic()
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        
        self.gamma = gamma
        
    def select_action(self, state, action_mask, training=True):
        """
        Select an action using the actor network with action masking
        
        Args:
            state: Board state, numpy array shape (6, 6)
            action_mask: Binary mask, numpy array shape (1296,)
            training: If True, sample. If False, greedy.
        
        Returns:
            action: Selected action (integer)
            log_prob: Log probability of action
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0)  # (1, 36)
        action_mask_tensor = torch.FloatTensor(action_mask)  # (1296,)
        
        # Get logits
        with torch.no_grad() if not training else torch.enable_grad():
            action_logits = self.actor(state_tensor).squeeze(0)  # (1296,)
        
        # Apply mask
        masked_logits = action_logits.clone()
        masked_logits[action_mask_tensor == 0] = -1e8
        
        # Get probabilities
        action_probs = F.softmax(masked_logits, dim=0)
        
        if training:
            # Sample from distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        else:
            # Greedy
            action = torch.argmax(action_probs)
            log_prob = torch.log(action_probs[action])
        
        return action.item(), log_prob
    
    def get_value(self, state):
        """
        Get state value from critic
        
        Args:
            state: Board state, numpy array shape (6, 6)
        
        Returns:
            value: State value
        """
        state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0)
        value = self.critic(state_tensor)
        return value.item()
    
    def update(self, states, actions, advantages, returns):
        """
        Update actor and critic networks
        To be implemented next
        """
        pass
    
    def save(self, filepath):
        """Save agent"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load agent"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])