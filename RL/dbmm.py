import json
from typing import Dict, Optional, Set


class DBMM:
    """Dual-Behavior Mealy Machine implementation"""

    def __init__(self, json_path: str, is_rm: bool = False, recording_path: Optional[str] = None):
        """Initialize DBMM from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.initial_state = data['initial_state']
        self.states = data['states']
        self.state_count = data['state_count']
        self.is_rm = is_rm
        self.current_state = self.initial_state

        # Load recording if provided
        self.recording = {}
        if recording_path:
            self._load_recording(recording_path)

    def _load_recording(self, recording_path: str):
        """Load recording data from file"""
        with open(recording_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' -> ')
                if len(parts) == 2:
                    key = parts[0]
                    value = parts[1]
                    if self.is_rm:
                        self.recording[key] = str(value)
                    else:
                        self.recording[key] = str(value)

    def reset(self):
        """Reset to initial state"""
        self.current_state = self.initial_state

    def get_current_state(self) -> str:
        """Get current state"""
        return self.current_state

    def get_all_states(self) -> Set[str]:
        """Get all states"""
        return set(self.states.keys())

    def transition(self, label: str) -> Optional[str]:
        """Transition based on label (beta input)"""
        if self.current_state in self.states:
            transitions = self.states[self.current_state].get('transitions', {})
            if label in transitions:
                self.current_state = transitions[label]
                return self.current_state
        return None

    def get_assume_next_state(self, state, label: str) -> Optional[str]:
        if state in self.states:
            transitions = self.states[state].get('transitions', {})
            if label in transitions:
                return transitions[label]
        if label is None or label == 'None':
            return state
        return None

    def get_assume_output(self, assume_state, obs: str, action: int, tm_state: Optional[str] = None):
        """Get output for observation-action pair (alpha input)"""
        if self.is_rm:
            # For RM, construct key with TM state
            if tm_state is None:
                return None
            key = f"{obs}-{tm_state},{action}"
        else:
            # For TM, key is just obs,action
            key = f"{obs},{action}"

        # Check recording first
        if key in self.recording:
            return self.recording[key]

        # Check fingerprint
        if assume_state in self.states:
            fingerprint = self.states[assume_state].get('fingerprint', {})
            if key in fingerprint:
                if self.is_rm:
                    return float(fingerprint[key])
                else:
                    return str(fingerprint[key])

        return None

    def output(self, obs: int, action: int, tm_state: Optional[str] = None) -> Optional:
        return self.get_assume_output(self.current_state, obs, action, tm_state)
