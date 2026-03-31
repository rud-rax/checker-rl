import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor Network (Policy)
    Input: Board state (36 features)
    Output: Action logits (1296 actions)
    """
    def __init__(self, input_size=36, hidden_size=128, output_size=1296):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Args:
            state: Flattened board state, shape (batch_size, 36) or (36,)
        Returns:
            action_logits: Raw scores for each action, shape (batch_size, 1296) or (1296,)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits


class Critic(nn.Module):
    """
    Critic Network (Value Function)
    Input: Board state (36 features)
    Output: State value (single number)
    """
    def __init__(self, input_size=36, hidden_size=128):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state):
        """
        Args:
            state: Flattened board state, shape (batch_size, 36) or (36,)
        Returns:
            value: State value, shape (batch_size, 1) or (1,)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# Example usage:
if __name__ == "__main__":
    # Create networks
    actor = Actor()
    critic = Critic()
    
    # Test with dummy input
    dummy_state = torch.randn(1, 36)  # Batch size 1, 36 features
    
    # Forward pass
    action_logits = actor(dummy_state)
    state_value = critic(dummy_state)
    
    print(f"Actor output shape: {action_logits.shape}")  # Should be (1, 1296)
    print(f"Critic output shape: {state_value.shape}")    # Should be (1, 1)
    
    print(f"\nActor parameters: {sum(p.numel() for p in actor.parameters())}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters())}")