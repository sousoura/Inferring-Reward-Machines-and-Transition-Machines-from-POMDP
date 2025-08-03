import argparse
import os
import random
from agent import RandomAgent
from env import BigEnv
from trajectory_writer import TrajectoryWriter


def main(args):
    """
    主函数，用于从复杂环境中生成轨迹数据。

    Args:
        args (Namespace): 包含所有命令行参数的对象。
    """
    # --- 1. 环境和Agent设置 ---
    env = BigEnv()
    agent = RandomAgent(env.action_space)

    # --- 2. 设置输出路径 ---
    # 构建输出文件的完整路径，使其位于项目根目录
    project_root = os.path.join(os.path.dirname(__file__), "..")
    output_path = os.path.normpath(os.path.join(project_root, args.output))

    # 初始化轨迹记录器
    trajectory_writer = TrajectoryWriter(output_path, env.env_size + 1)
    print(f"Trajectories will be saved to: {output_path}")

    # --- 3. 轨迹生成循环 ---
    for traj_num in range(1, args.num_trajectories + 1):
        obs = env.reset()
        done = False
        total_reward = 0
        info = {'label': env.get_label(env.agent_pos)}
        reward = 0
        next_obs = None
        action = None

        trajectory_writer.start_trajectory()

        while not done:
            label = info.get('label', 'None')
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            trajectory_writer.record_step(label, obs["agent_pos"], action, reward)
            obs = next_obs

        # 记录最后一步
        label = info.get('label', 'None')
        trajectory_writer.record_step(label, next_obs["agent_pos"], action, reward)

        # 结束当前轨迹（暂存到内存）
        trajectory_writer.end_trajectory()

        if traj_num % 100 == 0 or traj_num == args.num_trajectories:
            print(f"Generated trajectory {traj_num}/{args.num_trajectories}, Total Reward: {total_reward:.2f}")

    # --- 4. 收尾工作 ---
    # 将所有收集到的轨迹一次性写入文件
    trajectory_writer.write_to_file()
    env.close()
    print("\nGeneration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trajectories from the complex environment.")

    parser.add_argument('--num-trajectories', type=int, default=1000,
                        help='Number of trajectories to generate. Default: 10000.')

    parser.add_argument('--seed', type=int, default=41,
                        help='Random seed for reproducibility. Default: 41.')

    parser.add_argument('--output', type=str, default='trajectories.json',
                        help='Output filename. Will be saved in the project root. Default: trajectories.json.')

    cli_args = parser.parse_args()

    # 设置随机种子
    random.seed(cli_args.seed)
    print(f"Using random seed: {cli_args.seed}")

    main(cli_args)
