import json
from typing import Set, Tuple, Dict, Optional
from dbmm import DBMM


def extract_terminal_states_and_labels(
        trajectories_path: str,
        tm_path: str,
        rm_path: str,
        tm_recording_path: Optional[str] = None,
        rm_recording_path: Optional[str] = None
) -> Tuple[Set[Tuple[str, str, int, int]], Dict[int, str]]:
    """Extract terminal states and observation-label mapping from trajectories"""

    with open(trajectories_path, 'r') as f:
        trajectories = [json.loads(line) for line in f]

    # Initialize machines
    tm = DBMM(tm_path, is_rm=False, recording_path=tm_recording_path)
    rm = DBMM(rm_path, is_rm=True, recording_path=rm_recording_path)

    terminal_states = set()
    observation_to_label = {}

    for trajectory in trajectories:
        if len(trajectory) < 2:
            continue

        # Reset machines
        tm.reset()
        rm.reset()

        # Process trajectory
        for i in range(len(trajectory)):
            step = trajectory[i]
            label, obs = step[0][0], step[0][1]
            action = step[1]

            # Store observation to label mapping
            observation_to_label[obs] = label

            # Get current states
            tm_state = tm.get_current_state()
            rm_state = rm.get_current_state()

            # If this is the second-to-last step, it's terminal
            if i == len(trajectory) - 2:
                terminal_states.add((tm_state, rm_state, obs, action))

            # Transition on label for next iteration
            if label != "None":
                tm.transition(str(label))
                rm.transition(str(label))

    return terminal_states, observation_to_label
