import argparse
import os
import time
from parse_trajectories import parse_trajectories
from state_based_solve_rm_ilp import solve_reward_machine_ILP_simplified
from build_rm import print_simplified_reward_machine, output_simplified_reward_machine


def main(args):
    """
    Main function to run the ILP-based RM inference.
    """
    # --- Path Setup ---
    # Automatically determine the project root and construct full paths
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
    traj_file = os.path.join(project_root, args.input)
    output_file = os.path.join(project_root, args.output)

    # 1. Parse trajectories
    print(f"Parsing trajectories from {traj_file} ...")
    if not os.path.exists(traj_file):
        print(f"Error: Trajectory file not found at {traj_file}")
        return

    trajectories = parse_trajectories(traj_file)
    if not trajectories:
        print("Didn't find trajectories data or file is empty.")
        return

    # 2. Call ILP solver
    K = args.k
    print(f"Solving ILP for Reward Machine with K = {K}")
    start_time = time.time()
    print("Start time:", time.strftime("%H:%M:%S", time.localtime(start_time)))

    sol, reward_machine = solve_reward_machine_ILP_simplified(trajectories, K)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if reward_machine is None:
        print(f"ILP could not find a solution for K = {K}. Try a larger K value.")
        print(f"Total time elapsed: {elapsed_time:.6f} seconds.")
        return

    print(f"ILP successfully solved Reward Machine with K = {K}")
    print_simplified_reward_machine(reward_machine)
    print(f"Running time: {elapsed_time:.6f} seconds.")

    # 3. Build and output mealy_rm.json
    output_simplified_reward_machine(reward_machine, output_file, elapsed_time)
    print(f"Inferred Reward Machine saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer a Reward Machine from trajectories using an ILP-based approach.")
    parser.add_argument('--k', type=int, default=3,
                        help='The number of states (K) for the Reward Machine. Default: 3')
    parser.add_argument('--input', type=str, default='trajectories.json',
                        help='Input trajectory file name (located in project root). Default: trajectories.json')
    parser.add_argument('--output', type=str, default='ilp_mealy_rm.json',
                        help='Output file name for the inferred machine (saved in project root). Default: mealy_rm.json')

    cli_args = parser.parse_args()
    main(cli_args)
