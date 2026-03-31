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
        """Select an action using the actor network with action masking"""
        state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0)
        action_mask_tensor = torch.FloatTensor(action_mask)
        
        with torch.no_grad() if not training else torch.enable_grad():
            action_logits = self.actor(state_tensor).squeeze(0)
        
        # Apply mask
        masked_logits = action_logits.clone()
        masked_logits[action_mask_tensor == 0] = -1e8
        
        # Check for NaN
        if torch.isnan(masked_logits).any():
            print("WARNING: NaN detected in logits, using random valid action")
            valid_actions = torch.where(action_mask_tensor == 1)[0]
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
            return action, torch.tensor(0.0)
        
        # Get probabilities with numerical stability
        action_probs = F.softmax(masked_logits, dim=0)
        
        # Check for NaN in probabilities
        if torch.isnan(action_probs).any() or action_probs.sum() == 0:
            print("WARNING: NaN in probabilities, using random valid action")
            valid_actions = torch.where(action_mask_tensor == 1)[0]
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
            return action, torch.tensor(0.0)
        
        if training:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        else:
            action = torch.argmax(action_probs)
            log_prob = torch.log(action_probs[action] + 1e-8)
        
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
        """Update actor and critic networks"""
        # Convert lists to tensors
        states_tensor = torch.FloatTensor(np.array([s.flatten() for s in states]))
        actions_tensor = torch.LongTensor(actions)
        advantages_tensor = torch.FloatTensor(advantages).unsqueeze(1)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1)
        
        # ============ UPDATE ACTOR ============
        action_logits = self.actor(states_tensor)
        action_probs = F.softmax(action_logits, dim=1)
        log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)) + 1e-8)  # Add small epsilon
        
        actor_loss = -(log_probs * advantages_tensor.detach()).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # Clip gradients
        self.actor_optimizer.step()
        
        # ============ UPDATE CRITIC ============
        predicted_values = self.critic(states_tensor)
        critic_loss = F.mse_loss(predicted_values, returns_tensor)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  # Clip gradients
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
        
    def save(self, filepath, episode=None, metrics=None):
        """
        Save agent with full training state
        
        Args:
            filepath: Path to save file
            episode: Current episode number (optional)
            metrics: Dictionary of training metrics (optional)
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'gamma': self.gamma,
        }
        
        # Add optional metadata
        if episode is not None:
            checkpoint['episode'] = episode
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, filepath)
        print(f"✓ Saved checkpoint to {filepath}")

    def load(self, filepath):
        """
        Load agent and training state
        
        Args:
            filepath: Path to checkpoint file
        
        Returns:
            episode: Episode number (if saved)
            metrics: Training metrics (if saved)
        """
        checkpoint = torch.load(filepath)
        
        # Load networks
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Load optimizers
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Load hyperparameters
        self.gamma = checkpoint.get('gamma', 0.99)
        
        # Return metadata
        episode = checkpoint.get('episode', None)
        metrics = checkpoint.get('metrics', None)
        
        print(f"✓ Loaded checkpoint from {filepath}")
        if episode is not None:
            print(f"  Resuming from episode {episode}")
        
        return episode, metrics