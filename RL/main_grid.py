import os
import numpy as np
from grid_environment import GridEnvironment
from agent import Agent
from trainer import Trainer
from demo_runner import DemoRunner
from visualization import save_rewards, plot_learning_curve
from grid_visualization import plot_grid_value_function, plot_policy_arrows


def main():
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

    # Create agent
    print("\n2. Creating agent...")
    agent = Agent(
        action_space_size=env.get_action_space_size(),
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3
    )

    # Create trainer
    trainer = Trainer(agent, env)

    # Train
    print("\n3. Training agent...")
    rewards = trainer.train(num_episodes=1500)

    # Save results
    print("\n4. Saving results...")
    save_rewards(rewards, "grid_rewards.txt")
    plot_learning_curve(rewards, window_size=100, save_path="grid_learning_curve.png")

    # print("\n=== Training Complete ===")
    # print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.3f}")
    #
    # # Run demonstration
    # print("\n5. Running demonstration...")
    # demo_runner = DemoRunner(agent, env)
    # demo_runner.run_demo(max_steps=30, verbose=True)


if __name__ == "__main__":
    main()
