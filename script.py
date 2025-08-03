import argparse
import os
import subprocess
import sys


def run_command(cmd_list):
    """
    打印并执行一个命令行指令。
    如果指令失败（返回非零退出码），则脚本会中止。
    """
    # 将所有参数转换为字符串，以防万一
    cmd_list_str = [str(item) for item in cmd_list]
    command_str = ' '.join(cmd_list_str)
    print(f"\n>>> Executing: {command_str}")
    try:
        # 使用 subprocess.run 执行命令
        subprocess.run(cmd_list_str, check=True, text=True)
    except FileNotFoundError:
        print(f"--- ERROR: Command not found. Is Python installed and in your PATH? ---")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: Command failed with exit code {e.returncode} ---")
        # 如果子脚本失败，则主脚本也应该停止
        sys.exit(1)


def handle_exp1(args):
    """处理实验一的完整流程"""
    print("=" * 15, "Running Experiment 1", "=" * 15)

    # --- 步骤 1: 生成轨迹 ---
    cmd_traj_gen = [
        'python',
        os.path.join('experiment1_env', 'main.py'),
        '--env-size', args.env_size,
        '--num-trajectories', args.num_trajectories,
        '--seed', args.seed
    ]
    run_command(cmd_traj_gen)

    # --- 步骤 2: 从轨迹推断自动机 ---
    cmd_algo = [
        'python',
        os.path.join('Algorithms', 'script.py'),
        '--input', 'trajectories.json'
    ]
    # 添加消融实验参数
    if args.no_supplementation:
        cmd_algo.append('--no-supplementation')
    if args.no_alpha_preprocess:
        cmd_algo.append('--no-alpha-preprocess')
    if args.no_beta_preprocess:
        cmd_algo.append('--no-beta-preprocess')
    run_command(cmd_algo)

    print("\n" + "=" * 15, "Experiment 1 Finished", "=" * 15)


def handle_exp2(args):
    """处理实验二的完整流程"""
    print("=" * 15, "Running Experiment 2", "=" * 15)

    # --- 步骤 1: 生成环境参数 ---
    cmd_env_gen = [
        'python',
        os.path.join('experiment2_env', 'env_para_generation.py'),
        '--env-size', args.env_size,
        '--labels-count', args.labels_count,
        '--transition-states-count', args.transition_states_count,
        '--reward-states-count', args.reward_states_count,
        '--seed', args.seed
    ]
    run_command(cmd_env_gen)

    # --- 步骤 2: 生成轨迹 ---
    cmd_traj_gen = [
        'python',
        os.path.join('experiment2_env', 'main.py'),
        '--num-trajectories', args.num_trajectories,
        '--seed', args.seed
    ]
    run_command(cmd_traj_gen)

    # --- 步骤 3: 从轨迹推断自动机 ---
    cmd_algo = [
        'python',
        os.path.join('Algorithms', 'script.py'),
        '--input', 'trajectories.json'
    ]
    # 添加消融实验参数
    if args.no_supplementation:
        cmd_algo.append('--no-supplementation')
    if args.no_alpha_preprocess:
        cmd_algo.append('--no-alpha-preprocess')
    if args.no_beta_preprocess:
        cmd_algo.append('--no-beta-preprocess')
    run_command(cmd_algo)

    # --- 步骤 4: 运行强化学习 ---
    cmd_rl = [
        'python',
        os.path.join('RL', 'main_cross.py'),
        '--lr', args.lr,
        '--gamma', args.gamma,
        '--epsilon', args.epsilon,
        '--episodes', args.episodes
    ]
    run_command(cmd_rl)

    print("\n" + "=" * 15, "Experiment 2 Finished", "=" * 15)


def handle_clean(args):
    """清理所有生成的实验文件"""
    print("=" * 15, "Cleaning generated files", "=" * 15)
    files_to_delete = [
        'trajectories.json',
        'env_para.json',
        'mealy_tm.json',
        'mealy_rm.json',
        'tm_recording.txt',
        'rm_recording.txt',
        'grid_rewards.txt',
        'grid_learning_curve.png'
    ]
    for filename in files_to_delete:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Removed: {filename}")
            except OSError as e:
                print(f"Error removing {filename}: {e}")
        else:
            print(f"Skipped (not found): {filename}")
    print("\n" + "=" * 15, "Cleaning Finished", "=" * 15)


def main():
    # --- 主解析器 ---
    parser = argparse.ArgumentParser(description="Master script for running TM and RM inference experiments.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- “实验一”子命令解析器 ---
    parser_exp1 = subparsers.add_parser('exp1', help='Run the complete workflow for Experiment 1.')
    parser_exp1.add_argument('--env-size', type=int, default=3,
                             help='Size of the simple environment (e.g., 3, 4, or 5).')
    parser_exp1.add_argument('--num-trajectories', type=int, default=250, help='Number of trajectories to generate.')
    parser_exp1.add_argument('--seed', type=int, default=42, help='Random seed for trajectory generation.')
    parser_exp1.add_argument('--no-supplementation', action='store_true',
                             help='Disable state supplementation in algorithm.')
    parser_exp1.add_argument('--no-alpha-preprocess', action='store_true',
                             help='Disable alpha preprocessing in algorithm.')
    parser_exp1.add_argument('--no-beta-preprocess', action='store_true',
                             help='Disable beta preprocessing in algorithm.')
    parser_exp1.set_defaults(func=handle_exp1)

    # --- “实验二”子命令解析器 ---
    parser_exp2 = subparsers.add_parser('exp2', help='Run the complete workflow for Experiment 2.')
    # 全局参数
    parser_exp2.add_argument('--seed', type=int, default=42, help='Global random seed for all steps.')
    # 环境生成参数
    exp2_env = parser_exp2.add_argument_group('Environment Generation Parameters')
    exp2_env.add_argument('--env-size', type=int, default=25, help='Size of the grid environment.')
    exp2_env.add_argument('--labels-count', type=int, default=5, help='Number of distinct labels.')
    exp2_env.add_argument('--transition-states-count', type=int, default=7,
                          help='Number of states in the transition automaton.')
    exp2_env.add_argument('--reward-states-count', type=int, default=3,
                          help='Number of states in the reward automaton.')
    # 轨迹生成参数
    exp2_traj = parser_exp2.add_argument_group('Trajectory Generation Parameters')
    exp2_traj.add_argument('--num-trajectories', type=int, default=10000, help='Number of trajectories to generate.')
    # 算法推断参数
    exp2_algo = parser_exp2.add_argument_group('Algorithm Inference Parameters')
    exp2_algo.add_argument('--no-supplementation', action='store_true', help='Disable state supplementation.')
    exp2_algo.add_argument('--no-alpha-preprocess', action='store_true', help='Disable alpha preprocessing.')
    exp2_algo.add_argument('--no-beta-preprocess', action='store_true', help='Disable beta preprocessing.')
    # 强化学习参数
    exp2_rl = parser_exp2.add_argument_group('Reinforcement Learning Parameters')
    exp2_rl.add_argument('--lr', type=float, default=0.1, help='Learning rate for the RL agent.')
    exp2_rl.add_argument('--gamma', type=float, default=0.95, help='Discount factor for the RL agent.')
    exp2_rl.add_argument('--epsilon', type=float, default=0.3, help='Epsilon for exploration in RL.')
    exp2_rl.add_argument('--episodes', type=int, default=1500, help='Number of training episodes for RL.')
    parser_exp2.set_defaults(func=handle_exp2)

    # --- “清理”子命令解析器 ---
    parser_clean = subparsers.add_parser('clean', help='Clean up all generated files from experiments.')
    parser_clean.set_defaults(func=handle_clean)

    # 解析参数并调用对应的处理函数
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
