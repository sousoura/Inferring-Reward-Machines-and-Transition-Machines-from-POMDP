import os
from dbmm import DBMM
from environment import Environment
from agent import Agent
from trainer import Trainer
from terminal_states_extractor import extract_terminal_states_and_labels
from visualization import save_rewards, plot_learning_curve


def main():
    # File paths
    tm_path = "mealy_tm.json"
    rm_path = "mealy_rm.json"
    trajectories_path = "trajectories.json"

    # Optional recording paths
    tm_recording_path = "tm_recording.txt" if os.path.exists("tm_recording.txt") else None
    rm_recording_path = "rm_recording.txt" if os.path.exists("rm_recording.txt") else None

    print("=== Starting DBMM-based RL Framework ===")

    # Extract terminal states and labels
    print("\n1. Extracting terminal states and labels...")
    terminal_states, observation_to_label = extract_terminal_states_and_labels(
        trajectories_path, tm_path, rm_path,
        tm_recording_path, rm_recording_path
    )
    print(f"Found {len(terminal_states)} terminal states")
    print(f"Found {len(observation_to_label)} observation-label mappings")

    # Initialize machines
    print("\n2. Initializing environment and agent...")
    tm = DBMM(tm_path, is_rm=False, recording_path=tm_recording_path)
    rm = DBMM(rm_path, is_rm=True, recording_path=rm_recording_path)

    # Create environment
    env = Environment(tm, rm, terminal_states, observation_to_label)
    print(f"Environment created with {len(env.observations)} observations "
          f"and {len(env.actions)} actions")

    # Create agent
    agent = Agent(action_space_size=env.get_action_space_size())

    # Create trainer
    trainer = Trainer(agent, env)

    # Train
    print("\n3. Training agent...")
    rewards = trainer.train(num_episodes=5000)

    # Save results
    print("\n4. Saving results...")
    save_rewards(rewards, "rewards.txt")
    plot_learning_curve(rewards, window_size=100, save_path="learning_curve.png")

    print("\n=== Training Complete ===")
    print(f"Final average reward (last 100 episodes): {sum(rewards[-100:]) / 100:.3f}")


if __name__ == "__main__":
    main()
