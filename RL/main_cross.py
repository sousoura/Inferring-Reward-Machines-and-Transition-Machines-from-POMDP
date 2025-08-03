import os
import numpy as np
import argparse
from dbmm import DBMM
from environment import Environment
from grid_environment import GridEnvironment
from agent import Agent
from trainer import Trainer, Mask_trainer
from terminal_states_extractor import extract_terminal_states_and_labels
from visualization import save_rewards, plot_learning_curve
from grid_visualization import plot_grid_value_function, plot_policy_arrows

# --- 自动定位项目根目录 ---
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))


def get_machine_env():
    """从根目录加载推断出的状态机并创建环境"""
    tm_path = os.path.join(PROJECT_ROOT, "mealy_tm.json")
    rm_path = os.path.join(PROJECT_ROOT, "mealy_rm.json")
    trajectories_path = os.path.join(PROJECT_ROOT, "trajectories.json")

    tm_recording_path_full = os.path.join(PROJECT_ROOT, "tm_recording.txt")
    rm_recording_path_full = os.path.join(PROJECT_ROOT, "rm_recording.txt")

    tm_recording_path = tm_recording_path_full if os.path.exists(tm_recording_path_full) else None
    rm_recording_path = rm_recording_path_full if os.path.exists(rm_recording_path_full) else None

    print("=== Starting DBMM-based RL Framework ===")

    print("\n1. Extracting terminal states and labels...")
    terminal_states, observation_to_label = extract_terminal_states_and_labels(
        trajectories_path, tm_path, rm_path,
        tm_recording_path, rm_recording_path
    )
    print(f"Found {len(terminal_states)} terminal states")
    print(f"Found {len(observation_to_label)} observation-label mappings")

    print("\n2. Initializing environment and agent...")
    tm = DBMM(tm_path, is_rm=False, recording_path=tm_recording_path)
    rm = DBMM(rm_path, is_rm=True, recording_path=rm_recording_path)

    env = Environment(tm, rm, terminal_states, observation_to_label)
    return env, terminal_states, observation_to_label


def get_real_env():
    """从根目录加载真实环境参数并创建环境"""
    env_file = os.path.join(PROJECT_ROOT, "env_para.json")
    if not os.path.exists(env_file):
        print(f"Error: {env_file} not found!")
        return None

    print("=== Starting Grid Environment RL ===")
    print("\n1. Creating grid environment...")
    env = GridEnvironment(env_file)
    print(f"Environment created:")
    print(f"  - Grid size: {env.get_grid_size()}x{env.get_grid_size()}")
    print(f"  - Transition states: {len(env.get_all_trans_states())}")
    print(f"  - Reward states: {len(env.get_all_reward_states())}")
    return env


def main(args):
    """
    主函数，根据传入的参数运行完整的RL流程。
    Args:
        args (Namespace): 包含所有命令行参数的对象。
    """
    m_env, terminal_states, observation_to_label = get_machine_env()
    env = get_real_env()
    if env is None:
        print("Failed to create real environment. Exiting.")
        return

    print("\n2. Creating agent with specified hyperparameters...")
    agent = Agent(
        action_space_size=env.get_action_space_size(),
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=args.epsilon
    )
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Discount Factor: {args.gamma}")
    print(f"  - Epsilon: {args.epsilon}")

    trainer = Mask_trainer(agent, env, m_env)

    print(f"\n3. Training agent for {args.episodes} episodes...")
    rewards = trainer.train(num_episodes=args.episodes)

    print("\n4. Saving results...")
    rewards_path = os.path.join(PROJECT_ROOT, "tm_rm_rewards.txt")
    curve_path = os.path.join(PROJECT_ROOT, "tm_rm_learning_curve.png")

    save_rewards(rewards, rewards_path)
    plot_learning_curve(rewards, window_size=100, save_path=curve_path)
    print(f"Rewards saved to: {rewards_path}")
    print(f"Learning curve saved to: {curve_path}")

    print("\n=== Training Complete ===")
    if len(rewards) >= 100:
        print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Reinforcement Learning with specified hyperparameters.")

    # 添加超参数作为命令行参数
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for the agent. Default: 0.1')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor for future rewards. Default: 0.95')
    parser.add_argument('--epsilon', type=float, default=0.3,
                        help='Epsilon for epsilon-greedy exploration. Default: 0.3')
    parser.add_argument('--episodes', type=int, default=1500,
                        help='Number of episodes to train. Default: 1500')

    cli_args = parser.parse_args()
    main(cli_args)
