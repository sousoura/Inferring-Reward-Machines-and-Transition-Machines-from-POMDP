import matplotlib.pyplot as plt
import numpy as np
from typing import List


def save_rewards(rewards: List[float], filename: str = "rewards.txt"):
    """Save rewards to text file"""
    with open(filename, 'w') as f:
        for i, reward in enumerate(rewards):
            f.write(f"{i},{reward}\n")
    print(f"Rewards saved to {filename}")


def plot_learning_curve(rewards: List[float], window_size: int = 100,
                        save_path: str = "learning_curve.png"):
    """Plot and save learning curve"""
    episodes = range(len(rewards))

    # Calculate moving average
    moving_avg = []
    for i in range(len(rewards)):
        if i < window_size:
            moving_avg.append(np.mean(rewards[:i + 1]))
        else:
            moving_avg.append(np.mean(rewards[i - window_size + 1:i + 1]))

    plt.figure(figsize=(10, 6))

    # Plot raw rewards
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Rewards')

    # Plot moving average
    plt.plot(episodes, moving_avg, color='red', linewidth=2,
             label=f'Moving Average (window={window_size})')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Learning curve saved to {save_path}")


def plot_q_values_heatmap(agent, rm_state: str, save_path: str = "q_values_heatmap.png"):
    """Plot heatmap of Q-values for a specific RM state"""
    # Extract unique TM states and observations
    tm_states = sorted(agent.env.tm.get_all_states())
    observations = sorted(agent.env.observations)
    actions = sorted(agent.env.actions)

    # Create heatmap for each action
    fig, axes = plt.subplots(1, len(actions), figsize=(5 * len(actions), 6))
    if len(actions) == 1:
        axes = [axes]

    for action_idx, action in enumerate(actions):
        q_matrix = np.zeros((len(tm_states), len(observations)))

        for i, tm_state in enumerate(tm_states):
            for j, obs in enumerate(observations):
                q_matrix[i, j] = agent.Q[rm_state][tm_state][obs][action]

        im = axes[action_idx].imshow(q_matrix, cmap='viridis', aspect='auto')
        axes[action_idx].set_title(f'Action {action}')
        axes[action_idx].set_xlabel('Observation')
        axes[action_idx].set_ylabel('TM State')

        # Set tick labels
        axes[action_idx].set_xticks(range(len(observations)))
        axes[action_idx].set_xticklabels(observations, rotation=45)
        axes[action_idx].set_yticks(range(len(tm_states)))
        axes[action_idx].set_yticklabels(tm_states)

        plt.colorbar(im, ax=axes[action_idx])

    plt.suptitle(f'Q-values for RM State: {rm_state}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Q-values heatmap saved to {save_path}")