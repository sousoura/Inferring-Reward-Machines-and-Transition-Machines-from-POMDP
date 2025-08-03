import random
import argparse
import os

from env import SimpleEnv
from agent import RandomAgent
from trajectory_writer import TrajectoryWriter


def main(args):
    """
    主函数，用于生成轨迹数据。

    Args:
        args (Namespace): 包含所有命令行参数的对象。
            - env_size (int): 环境大小。
            - num_trajectories (int): 要生成的轨迹数量。
            - output (str): 输出文件的名称。
    """
    # --- 1. 环境和Agent设置 ---

    # 根据参数创建环境
    env = SimpleEnv(size=args.env_size, fsm_type=args.env_size)

    # 创建Agent
    agent = RandomAgent(env.action_space, env.observation_space)

    # --- 2. 设置输出路径 ---

    # 构建相对于当前脚本的输出路径，指向项目根目录
    # os.path.dirname(__file__) 获取当前文件所在目录 (experiment1_env)
    # ".." 代表上一级目录 (Inferring TM and RM)
    project_root = os.path.join(os.path.dirname(__file__), "..")
    output_path = os.path.join(project_root, args.output)

    # 确保路径是规范的（例如，将 "a/b/.." 转换为 "a"）
    output_path = os.path.normpath(output_path)

    # 初始化轨迹记录器
    trajectory_writer = TrajectoryWriter(output_path, map_width=args.env_size)
    print(f"Trajectories will be saved to: {output_path}")

    # --- 3. 轨迹生成循环 ---

    average_length = 0

    for traj_num in range(1, args.num_trajectories + 1):
        obs = env.reset()
        done = False
        total_reward = 0
        info = {'label': env.get_label()}

        trajectory_writer.start_trajectory()

        step_count = 0

        while not done:
            label = info.get('label', 'None')
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)

            total_reward += reward
            trajectory_writer.record_step(label, obs, action, reward)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs

            step_count += 1
            if step_count >= 1000:  # 防止无限循环
                break

        # 记录最后一步
        label = info.get('label', 'None')
        trajectory_writer.record_step(label, obs, action, reward)

        average_length += step_count

        trajectory_writer.end_trajectory()

        # 打印进度
        if traj_num % 50 == 0 or traj_num == args.num_trajectories:
            print(f"Generated trajectory {traj_num}/{args.num_trajectories}")

    # --- 4. 收尾工作 ---

    trajectory_writer.write_to_file()

    print(f"\nGeneration complete.")
    print(f"Average trajectory length: {average_length / args.num_trajectories:.2f}")

    env.close()


if __name__ == "__main__":
    # --- 参数解析器 ---
    parser = argparse.ArgumentParser(description="Generate trajectories from the environment.")

    parser.add_argument('--env-size', type=int, default=3,
                        help='Size of the grid environment (e.g., 3, 4, or 5). Default: 3.')

    parser.add_argument('--num-trajectories', type=int, default=250,
                        help='Number of trajectories to generate. Default: 250.')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility. Default: 42.')

    parser.add_argument('--output', type=str, default='trajectories.json',
                        help='Output filename. Will be saved in the project root. Default: trajectories.json.')

    # 解析参数
    cli_args = parser.parse_args()

    # 设置随机种子
    random.seed(cli_args.seed)
    print(f"Using random seed: {cli_args.seed}")

    # 运行主程序
    main(cli_args)
