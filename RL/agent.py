import numpy as np
from typing import Dict, Optional, Set, Tuple
from collections import defaultdict
from state import BaseState
from grid_state import GridState
from dbmm import DBMM


class Agent:
    """Q-learning agent"""

    def __init__(self, action_space_size: int, learning_rate=0.1,
                 discount_factor=0.9, epsilon=0.1):
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-table: Q[State][action]
        self.Q = defaultdict(lambda: defaultdict(float))

    def choose_action(self, state: BaseState) -> int:
        """Choose action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)

        # Get Q-values for current state
        q_values = [self.Q[state][a] for a in range(self.action_space_size)]

        # Choose action with highest Q-value
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best_actions)

    def update(self, state: BaseState, action: int, reward: float,
               next_state: BaseState, done: bool):
        """Update Q-value"""
        current_q = self.Q[state][action]

        if done:
            target = reward
        else:
            # Max Q-value for next state
            next_q_values = [self.Q[next_state][a]
                             for a in range(self.action_space_size)]
            target = reward + self.gamma * max(next_q_values)

        # Q-learning update
        self.Q[state][action] = current_q + self.lr * (target - current_q)

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Decay exploration rate"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_q_values(self, state: BaseState) -> Dict[int, float]:
        """Get Q-values for a state"""
        return dict(self.Q[state])


class QRM_Agent:
    """Q-learning for Reward Machines (QRM) agent"""

    def __init__(self, rm: DBMM, action_space_size: int,
                 learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.rm = rm
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Get all RM states
        self.rm_states = self.rm.get_all_states()

        # Q-table for each RM state: Q[rm_state][env_state][action]
        self.Q = {}
        for rm_state in self.rm_states:
            self.Q[rm_state] = defaultdict(lambda: defaultdict(float))

    def reset(self):
        """Reset the agent and RM to initial state"""
        self.rm.reset()

    def choose_action(self, state: GridState) -> int:
        """Choose action using epsilon-greedy strategy based on current RM state"""
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)

        obs = get_obs(state)

        # Get Q-values for current RM state and environment state
        q_values = [self.Q[self.rm.current_state][obs][a]
                    for a in range(self.action_space_size)]

        # Choose action with highest Q-value
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return np.random.choice(best_actions)

    def update(self, state: GridState, action: int, reward: float,
               next_state: GridState, done: bool, label: str):
        """Update Q-values for all RM states using off-policy learning"""

        # For each RM state, perform Q-learning update
        for rm_state in self.rm_states:
            # Get next RM state after transition
            next_rm_state = self.rm.get_assume_next_state(rm_state, label)

            if next_rm_state is None:
                # No valid transition from this RM state
                continue

            obs = get_obs(state)
            next_obs = next_state.row + next_state.col * 26
            tm_state = state.trans_state.replace("_", "")
            rm_reward = self.rm.get_assume_output(rm_state, obs, action, tm_state)

            if rm_reward is None:
                continue

            current_q = self.Q[rm_state][obs][action]

            if done:
                target = rm_reward
            else:
                # Max Q-value for next state in the next RM state
                next_q_values = [self.Q[next_rm_state][next_obs][a]
                                 for a in range(self.action_space_size)]
                target = rm_reward + self.gamma * max(next_q_values)

            # Update Q-value
            self.Q[rm_state][obs][action] = current_q + self.lr * (target - current_q)

        # Restore RM to actual current state and update it
        self.rm.transition(label)

    def update_only(self, state: GridState, action: int, reward: float,
               next_state: GridState, done: bool, label: str):
        """Update Q-value"""
        obs = get_obs(state)
        current_q = self.Q[self.rm.get_current_state()][obs][action]
        next_rm_state = self.rm.get_assume_next_state(self.rm.get_current_state(), label)

        if done:
            target = reward
        else:
            # Max Q-value for next state
            next_q_values = [self.Q[next_rm_state][obs][a]
                             for a in range(self.action_space_size)]
            target = reward + self.gamma * max(next_q_values)

        # Q-learning update
        self.Q[self.rm.get_current_state()][obs][action] = current_q + self.lr * (target - current_q)

        # Restore RM to actual current state and update it
        self.rm.transition(label)

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Decay exploration rate"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_q_values(self, state: GridState, rm_state: Optional[str] = None) -> Dict[int, float]:
        """Get Q-values for a state (optionally for a specific RM state)"""
        if rm_state is None:
            rm_state = self.rm.current_state

        obs = get_obs(state)
        return dict(self.Q[rm_state][obs])

    def get_current_rm_state(self) -> str:
        """Get current RM state"""
        return self.rm.current_state


def get_obs(state):
    return str(state.row) + "|" + str(state.col)