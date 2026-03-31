import pickle
from mycheckersenv import env
from myagent import ActorCriticAgent
import numpy as np
import matplotlib.pyplot as plt


def compute_advantages_and_returns(rewards, values, next_value, gamma=0.99, done=False):
    """Compute advantages and returns for an episode"""
    advantages = []
    returns = []

    # Compute returns (discounted cumulative rewards)
    G = next_value if not done else 0
    returns_list = []

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns_list.insert(0, G)

    returns = returns_list

    # Compute advantages (TD error)
    for i in range(len(rewards)):
        if i == len(rewards) - 1:
            next_val = next_value if not done else 0
        else:
            next_val = values[i + 1]

        advantage = rewards[i] + gamma * next_val - values[i]
        advantages.append(advantage)

    return advantages, returns

def play_episode(env, agent, training=True):
    """Play one full episode of self-play"""
    states = []
    actions = []
    values = []
    log_probs = []
    player_indices = []
    
    env.reset()
    
    for agent_name in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        done = termination or truncation
        
        if done:
            action = None
            # Capture final rewards BEFORE they're cleared
            if not hasattr(env, '_final_rewards'):
                env._final_rewards = env.rewards.copy()
        else:
            state = obs['observation']
            action_mask = obs['action_mask']
            
            value = agent.get_value(state)
            action, log_prob = agent.select_action(state, action_mask, training=training)
            
            states.append(state)
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)
            
            # Track which player made this move
            player_idx = 0 if agent_name == "player_0" else 1
            player_indices.append(player_idx)
        
        env.step(action)
    
    # Get final rewards (saved when game ended)
    final_rewards = env._final_rewards
    
    # Clean up temporary attribute
    delattr(env, '_final_rewards')
    
    # Assign rewards to each state-action pair
    rewards = []
    for player_idx in player_indices:
        player_name = f"player_{player_idx}"
        rewards.append(final_rewards[player_name])
    
    # Compute advantages and returns
    next_value = 0
    advantages, returns = compute_advantages_and_returns(
        rewards, values, next_value, gamma=agent.gamma, done=True
    )
    
    episode_data = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'values': values,
        'log_probs': log_probs,
        'advantages': advantages,
        'returns': returns,
        'total_reward': final_rewards["player_0"],
        'episode_length': len(states)
    }
    
    return episode_data


def plot_training_curves(rewards, lengths, actor_losses, critic_losses):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Rewards
    axes[0, 0].scatter(rewards)
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")

    # Episode lengths
    axes[0, 1].plot(lengths)
    axes[0, 1].set_title("Episode Lengths")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")

    # Actor loss
    axes[1, 0].plot(actor_losses)
    axes[1, 0].set_title("Actor Loss")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Loss")

    # Critic loss
    axes[1, 1].plot(critic_losses)
    axes[1, 1].set_title("Critic Loss")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("✓ Saved training curves to training_curves.png")


def save_training_state(
    episode,
    rewards,
    lengths,
    actor_losses,
    critic_losses,
    filepath="training_state.pkl",
):
    """Save training metrics"""
    state = {
        "episode": episode,
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
    }
    with open(filepath, "wb") as f:
        pickle.dump(state, f)
    print(f"✓ Saved training state to {filepath}")


def load_training_state(filepath="training_state.pkl"):
    """Load training metrics"""
    try:
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        print(f"✓ Loaded training state from {filepath}")
        return state
    except FileNotFoundError:
        print(f"No previous training state found at {filepath}")
        return None


def train(num_episodes=5000, save_interval=500, log_interval=100, resume_from=None):
    """
    Main training loop with resume capability

    Args:
        num_episodes: Total episodes to train
        save_interval: Save every N episodes
        log_interval: Log every N episodes
        resume_from: Path to checkpoint to resume from (optional)
    """
    # Create environment
    checkers_env = env()

    # Create agent
    agent = ActorCriticAgent(
        learning_rate_actor=0.0001,  # Was 0.001, now 10x smaller
        learning_rate_critic=0.0001,  # Was 0.001, now 10x smaller
        gamma=0.99
    )

    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []
    start_episode = 0

    # Resume from checkpoint if provided
    if resume_from is not None:
        print(f"Resuming training from {resume_from}")

        # Load agent
        episode, _ = agent.load(resume_from)
        start_episode = episode if episode is not None else 0

        # Load training state
        training_state = load_training_state()
        if training_state is not None:
            episode_rewards = training_state["episode_rewards"]
            episode_lengths = training_state["episode_lengths"]
            actor_losses = training_state["actor_losses"]
            critic_losses = training_state["critic_losses"]
            print(f"Loaded {len(episode_rewards)} episodes of training history")

    print("=" * 60)
    print(f"Starting Self-Play Training (Episodes {start_episode} to {num_episodes})")
    print("=" * 60)

    for episode in range(start_episode, num_episodes):
        # Play one episode
        episode_data = play_episode(checkers_env, agent, training=True)

        # Update agent
        actor_loss, critic_loss = agent.update(
            states=episode_data["states"],
            actions=episode_data["actions"],
            advantages=episode_data["advantages"],
            returns=episode_data["returns"],
        )

        # Store metrics
        episode_rewards.append(episode_data["total_reward"])
        episode_lengths.append(episode_data["episode_length"])
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            avg_actor_loss = np.mean(actor_losses[-log_interval:])
            avg_critic_loss = np.mean(critic_losses[-log_interval:])

            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Actor Loss: {avg_actor_loss:.4f}")
            print(f"  Critic Loss: {avg_critic_loss:.4f}")
            print("-" * 60)

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            # Save agent weights
            agent.save(
                f"checkpoints/agent_episode_{episode + 1}.pth",
                episode=episode + 1,
                metrics={"avg_reward": np.mean(episode_rewards[-100:])},
            )

            # Save training state
            save_training_state(
                episode + 1,
                episode_rewards,
                episode_lengths,
                actor_losses,
                critic_losses,
            )

    # Final save
    agent.save("checkpoints/agent_final.pth", episode=num_episodes)
    save_training_state(
        num_episodes, episode_rewards, episode_lengths, actor_losses, critic_losses
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, actor_losses, critic_losses)

    return agent, episode_rewards, episode_lengths


# if __name__ == "__main__":
#     import os
#     os.makedirs('checkpoints', exist_ok=True)

#     # To start fresh training:
#     agent, rewards, lengths = train(num_episodes=5000)

#     # To resume from checkpoint:
#     # agent, rewards, lengths = train(
#     #     num_episodes=10000,
#     #     resume_from="checkpoints/agent_episode_5000.pth"
#     # )


if __name__ == "__main__":
    # Create checkpoints directory
    import os

    os.makedirs("checkpoints", exist_ok=True)

    # Train agent
    agent, rewards, lengths = train(
        num_episodes=5000, save_interval=500, log_interval=100
    )
