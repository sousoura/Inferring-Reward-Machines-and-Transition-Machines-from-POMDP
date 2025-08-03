import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from grid_state import GridState


def plot_grid_value_function(q_values: Dict[GridState, Dict[int, float]],
                             grid_size: int,
                             trans_state: str,
                             reward_state: str,
                             save_path="grid_value_function.png"):
    """Plot value function heatmap for a specific automaton state"""

    # Create value grid
    value_grid = np.zeros((grid_size, grid_size))

    for state, action_values in q_values.items():
        if (isinstance(state, GridState) and
                state.trans_state == trans_state and
                state.reward_state == reward_state):
            # Value is max Q-value for this state
            if action_values:
                value_grid[state.row - 1, state.col - 1] = max(action_values.values())

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(value_grid, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='State Value')
    plt.title(f'Value Function\n(Trans: {trans_state}, Reward: {reward_state})')
    plt.xlabel('Column')
    plt.ylabel('Row')

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add ticks
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Value function heatmap saved to {save_path}")


def plot_policy_arrows(q_values: Dict[GridState, Dict[int, float]],
                       grid_size: int,
                       trans_state: str,
                       reward_state: str,
                       save_path="grid_policy.png"):
    """Plot policy as arrows on grid"""

    # Action to arrow mapping
    action_arrows = {
        0: '↑',
        1: '→',
        2: '↓',
        3: '←'
    }

    plt.figure(figsize=(12, 10))

    # Create policy grid
    for state, action_values in q_values.items():
        if (isinstance(state, GridState) and
                state.trans_state == trans_state and
                state.reward_state == reward_state and
                action_values):
            # Get best action
            best_action = max(action_values.items(), key=lambda x: x[1])[0]

            # Plot arrow
            plt.text(state.col - 0.5, grid_size - state.row + 0.5,
                     action_arrows[best_action],
                     ha='center', va='center', fontsize=20)

    # Set up grid
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    plt.xticks(range(grid_size + 1))
    plt.yticks(range(grid_size + 1))
    plt.grid(True)

    plt.title(f'Learned Policy\n(Trans: {trans_state}, Reward: {reward_state})')
    plt.xlabel('Column')
    plt.ylabel('Row (inverted)')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Policy arrows saved to {save_path}")