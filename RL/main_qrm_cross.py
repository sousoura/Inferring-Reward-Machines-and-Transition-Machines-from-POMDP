import os
import numpy as np
from dbmm import DBMM
from environment import Environment
from grid_environment import GridEnvironment
from agent import Agent, QRM_Agent
from trainer import Trainer, Mask_trainer, Mask_QRM_trainer
from terminal_states_extractor import extract_terminal_states_and_labels
from visualization import save_rewards, plot_learning_curve
from grid_visualization import plot_grid_value_function, plot_policy_arrows


def get_machine_env():
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

    return env, tm, rm


def get_real_env():
    # Environment file path
    env_file = "env_para.json"

    # Check if file exists
    if not os.path.exists(env_file):
        print(f"Error: {env_file} not found!")
        return

    print("=== Starting Grid Environment RL ===")

    # Create environment
    print("\n1. Creating grid environment...")
    env = GridEnvironment(env_file)
    print(f"Environment created:")
    print(f"  - Grid size: {env.get_grid_size()}x{env.get_grid_size()}")
    print(f"  - Transition states: {len(env.get_all_trans_states())}")
    print(f"  - Reward states: {len(env.get_all_reward_states())}")

    return env


def main():
    m_env, tm, rm = get_machine_env()

    env = get_real_env()

    # Create agent
    print("\n2. Creating agent...")
    qrm_agent = QRM_Agent(
        rm,
        action_space_size=env.get_action_space_size(),
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3
    )

    # Create trainer
    trainer = Mask_QRM_trainer(qrm_agent, env)

    # Train
    print("\n3. Training agent...")
    rewards = trainer.train(num_episodes=1500)

    # Save results
    print("\n4. Saving results...")
    save_rewards(rewards, "mask_rewards.txt")
    plot_learning_curve(rewards, window_size=100, save_path="mask_learning_curve.png")

    # Grid-specific visualizations
    print("\n=== Training Complete ===")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.3f}")


if __name__ == "__main__":
    main()
