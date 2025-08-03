from typing import Dict, Set, Optional, Tuple
import numpy as np
from state import State, StepResult
from dbmm import DBMM


class Environment:
    """MDP Environment with TM and RM"""

    def __init__(self, tm: DBMM, rm: DBMM,
                 terminal_states: Set[Tuple[str, str, int, int]],
                 observation_to_label: Dict[int, str]):
        self.tm = tm
        self.rm = rm
        self.terminal_states = terminal_states
        self.observation_to_label = observation_to_label

        # Extract observations and actions
        self.observations = self._extract_observations()
        self.actions = self._extract_actions()

        self.current_observation: Optional[int] = None

    def _extract_observations(self) -> list:
        """Extract all possible observations"""
        observations = set()

        # From TM fingerprints
        for state_data in self.tm.states.values():
            for key in state_data.get('fingerprint', {}).keys():
                obs, _ = map(str, key.split(','))
                observations.add(obs)

        # From TM recording
        for key in self.tm.recording.keys():
            obs, _ = map(str, key.split(','))
            observations.add(obs)

        return sorted(list(observations))

    def _extract_actions(self) -> list:
        """Extract all possible actions"""
        actions = set()

        # From TM fingerprints
        for state_data in self.tm.states.values():
            for key in state_data.get('fingerprint', {}).keys():
                _, action = map(str, key.split(','))
                actions.add(action)

        # From TM recording
        for key in self.tm.recording.keys():
            _, action = map(str, key.split(','))
            actions.add(action)

        return sorted(list(actions))

    def reset(self, start_observation: Optional[int] = None) -> State:
        """Reset environment to initial state"""
        self.tm.reset()
        self.rm.reset()

        # Set initial observation
        if start_observation is None:
            self.current_observation = np.random.choice(self.observations)
        else:
            self.current_observation = start_observation

        # Process initial label if exists
        self._process_label_transition()

        return self._get_current_state()

    def outside_reset(self, state):
        self.current_observation = state.row + state.col * 26

        """Reset environment to initial state"""
        self.tm.reset()
        self.rm.reset()

        # Process initial label if exists
        self._process_label_transition()

        return self._get_current_state()

    def step(self, action: int) -> StepResult:
        """Execute action and return result"""
        # Get reward
        reward = self.rm.output(self.current_observation, action, self.tm.get_current_state())
        if reward is None:
            reward = 0.0

        # Get next observation
        next_observation = self.tm.output(self.current_observation, action)

        if next_observation is None:
            # Episode ends
            return StepResult(
                state=self._get_current_state(),
                reward=reward,
                done=True,
                info={'label': None}
            )

        # Update observation
        self.current_observation = next_observation

        # Process label transition
        label = self._process_label_transition()

        # Check if terminal
        done = self._is_terminal(action)

        return StepResult(
            state=self._get_current_state(),
            reward=reward,
            done=done,
            info={'label': label}
        )

    def _get_current_state(self) -> State:
        """Get current state"""
        return State(
            tm_state=self.tm.get_current_state(),
            rm_state=self.rm.get_current_state(),
            observation=self.current_observation
        )

    def get_tm_rm(self):
        return self.tm, self.rm

    def get_automata_state(self):
        """Get automata state"""
        return self.tm.current_state, self.rm.current_state

    def _process_label_transition(self) -> Optional[str]:
        """Process label-based transitions"""
        if self.current_observation in self.observation_to_label:
            label = self.observation_to_label[self.current_observation]
            if label != "None":
                self.tm.transition(str(label))
                self.rm.transition(str(label))
                return label
        return None

    def _is_terminal(self, action: int) -> bool:
        """Check if current state-action is terminal"""
        return (self.tm.get_current_state(),
                self.rm.get_current_state(),
                self.current_observation,
                action) in self.terminal_states

    def get_action_space_size(self) -> int:
        """Get size of action space"""
        return len(self.actions)

    def get_all_tm_states(self) -> Set[str]:
        """Get all TM states"""
        return self.tm.get_all_states()

    def get_all_rm_states(self) -> Set[str]:
        """Get all RM states"""
        return self.rm.get_all_states()
