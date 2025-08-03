import json
import numpy as np
from typing import Tuple, Dict, Set, Optional, List
from grid_state import GridState
from state import StepResult


class GridEnvironment:
    """Grid environment with transition and reward automata"""

    def __init__(self, env_file: str):
        """Initialize environment from JSON file"""
        with open(env_file, 'r') as f:
            data = json.load(f)

        # Environment size
        self.size = data['environment']['size']

        # Label mapping
        self.labels = data['labels']['mapping']
        self.label_count = data['labels']['count']

        # Transition automaton
        self.trans_states = list(data['transition_automaton']['transitions'].keys())
        self.trans_transitions = data['transition_automaton']['transitions']
        self.impassable = data['transition_automaton']['impassable_positions']

        # Reward automaton
        self.reward_states = list(data['reward_automaton']['transitions'].keys())
        self.reward_transitions = data['reward_automaton']['transitions']
        self.rewards = data['reward_automaton']['rewards']
        self.terminal_state = data['reward_automaton']['terminal_state']

        # Action space
        self.actions = [0, 1, 2, 3]  # up, right, down, left
        self.action_deltas = {
            0: (0, 1),  # up
            1: (1, 0),   # right
            2: (0, -1),   # down
            3: (-1, 0)   # left
        }

        # Current state components
        self.agent_pos = None
        self.trans_state = None
        self.reward_state = None

    def reset(self, start_pos: Optional[Tuple[int, int]] = None) -> GridState:
        """Reset environment to initial state"""
        # Set starting position
        if start_pos is None:
            # Random even position (no label)
            while True:
                row = np.random.randint(1, self.size + 1)
                col = np.random.randint(1, self.size + 1)
                if row % 2 == 0 or col % 2 == 0:  # Even position
                    self.agent_pos = (row, col)
                    break
        else:
            self.agent_pos = start_pos

        # Reset automata to initial states
        self.trans_state = self.trans_states[0]
        self.reward_state = self.reward_states[0]

        return self._get_current_state()

    def step(self, action: int) -> StepResult:
        """Execute action and return result"""
        # Check if already in terminal state
        if self.reward_state == self.terminal_state:
            return StepResult(
                state=self._get_current_state(),
                reward=0.0,
                done=True,
                info={'terminal': True, 'label': None}
            )

        # Calculate new position
        delta_row, delta_col = self.action_deltas[action]
        new_pos = (self.agent_pos[0] + delta_row, self.agent_pos[1] + delta_col)

        # Check if new position is valid
        if not self._is_valid_position(new_pos):
            # Invalid move, stay in place
            return StepResult(
                state=self._get_current_state(),
                reward=0.0,
                done=False,
                info={'invalid_move': True, 'label': None}
            )

        # Move to new position
        self.agent_pos = new_pos

        # Check for label at new position
        label = self._get_label(new_pos)
        reward = 0.0

        if label is not None:
            # Get reward before transition
            reward = self._get_reward(label)
            # Transition automata
            self._transition_automata(label)

        # Check if reached terminal state
        done = (self.reward_state == self.terminal_state)

        return StepResult(
            state=self._get_current_state(),
            reward=reward,
            done=done,
            info={'label': label}
        )

    def get_action_space_size(self) -> int:
        """Get size of action space"""
        return len(self.actions)

    def get_all_trans_states(self) -> List[str]:
        """Get all transition states"""
        return self.trans_states

    def get_all_reward_states(self) -> List[str]:
        """Get all reward states"""
        return self.reward_states

    def get_grid_size(self) -> int:
        """Get grid size"""
        return self.size

    def render(self):
        """Simple text rendering of the environment"""
        print(f"\nTransition State: {self.trans_state}, Reward State: {self.reward_state}")
        print("Grid (A=agent, X=impassable, L=label):")

        for row in range(1, self.size + 1):
            line = ""
            for col in range(1, self.size + 1):
                if (row, col) == self.agent_pos:
                    line += "A "
                elif f"({row}, {col})" in self.impassable.get(self.trans_state, []):
                    line += "X "
                elif row % 2 == 1 and col % 2 == 1:
                    line += "L "
                else:
                    line += ". "
            print(line)

    def _get_current_state(self) -> GridState:
        """Get current state"""
        return GridState(
            row=self.agent_pos[0],
            col=self.agent_pos[1],
            trans_state=self.trans_state,
            reward_state=self.reward_state
        )

    def get_current_position(self):
        return self.agent_pos

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid and not impassable"""
        row, col = pos

        # Check bounds
        if row < 1 or row > self.size or col < 1 or col > self.size:
            return False

        # Check if position is impassable in current transition state
        pos_str = f"({row}, {col})"
        if self.trans_state in self.impassable and pos_str in self.impassable[self.trans_state]:
            return False

        return True

    def _get_label(self, pos: Tuple[int, int]) -> Optional[int]:
        """Get label at position (if any)"""
        row, col = pos
        # Labels only exist at odd positions
        if row % 2 == 1 and col % 2 == 1:
            pos_str = f"({row}, {col})"
            return self.labels.get(pos_str)
        return None

    def get_label_now(self):
        return self._get_label(self.get_current_position())

    def _transition_automata(self, label: int):
        """Transition both automata based on label"""
        # Transition automaton
        if self.trans_state in self.trans_transitions:
            transitions = self.trans_transitions[self.trans_state]
            if str(label) in transitions:
                self.trans_state = transitions[str(label)]

        # Reward automaton
        if self.reward_state in self.reward_transitions:
            transitions = self.reward_transitions[self.reward_state]
            if str(label) in transitions:
                self.reward_state = transitions[str(label)]

    def _get_reward(self, label: int) -> float:
        """Get reward for triggering a label"""
        if self.reward_state in self.rewards:
            state_rewards = self.rewards[self.reward_state]
            if str(label) in state_rewards:
                return state_rewards[str(label)]
        return 0.0

    def get_current_trans_state(self):
        return self.trans_state
