import os
import subprocess
import sys
import time
import argparse
from multiprocessing import Process


def run_script(script_path, *args):
    """运行Python脚本并传递参数"""
    # 确保所有路径参数都是字符串
    cmd = [sys.executable, script_path] + [str(arg) for arg in args]
    print(f"Running: {' '.join(cmd)}")
    # check=True 会在子进程返回非零退出码时抛出异常
    subprocess.run(cmd, check=True)


def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="Infer TM and RM from trajectories with ablation options.")
    parser.add_argument('--input', type=str, default='trajectories.json',
                        help='Name of the input trajectory file located in the root directory.')
    parser.add_argument('--no-supplementation', action='store_true',
                        help='If set, state supplementation (step 4) will be skipped.')
    parser.add_argument('--no-alpha-preprocess', action='store_true',
                        help='If set, alpha_preprocess will be disabled.')
    parser.add_argument('--no-beta-preprocess', action='store_true',
                        help='If set, beta_preprocess will be disabled.')
    args = parser.parse_args()

    start_time = time.time()

    # --- 路径和参数设置 ---
    # 脚本将从其所在目录（Algorithms）运行，因此根目录是'../'
    project_root = ".."

    # 主要输入和输出文件 (位于根目录)
    input_trajectories = os.path.join(project_root, args.input)
    output_mealy_tm = os.path.join(project_root, "mealy_tm.json")
    output_mealy_rm = os.path.join(project_root, "mealy_rm.json")
    tm_recording = os.path.join(project_root, "tm_recording.txt")  # <--- 已修改
    rm_recording = os.path.join(project_root, "rm_recording.txt")  # <--- 已修改

    # 中间文件 (位于当前目录, 即 Algorithms/)
    tm_formal_trajectories = "TM_formal_trajectories.txt"
    reduced_tm_trajectories = "reduced_TM_trajectories.txt"
    supplemented_trajectories = "supplemented_trajectories.json"
    rm_formal_trajectories = "RM_formal_trajectories.txt"
    reduced_rm_trajectories = "reduced_RM_trajectories.txt"

    # 脚本路径
    path_prefix = "preprocessing_and_observation_supplement"
    get_formal_tm_script = os.path.join(path_prefix, "get_formal_TM_trajactories.py")
    get_final_tm_script = os.path.join(path_prefix, "get_final_TM_trajectory.py")
    find_machine_script = os.path.join("DB-RPNI", "find_mealy_machine.py")
    state_supplement_script = os.path.join(path_prefix, "state_supplementation.py")
    get_formal_rm_script = os.path.join(path_prefix, "get_formal_RM_trajactories.py")
    get_final_rm_script = os.path.join(path_prefix, "get_final_RM_trajectory.py")
    draw_machine_script = "draw_machine.py"

    # 消融实验参数
    alpha_param = "no_alpha_preprocess" if args.no_alpha_preprocess else "alpha_preprocess"
    beta_param = "no_beta_preprocess" if args.no_beta_preprocess else "beta_preprocess"
    use_supplementation = not args.no_supplementation

    # 确保工作目录是脚本所在目录 (Algorithms/)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # --- 流程开始 ---

    # 1. 为学习TM生成自动机学习数据
    print("Step 1: Generating automata learning data for learning TM...")
    run_script(get_formal_tm_script,
               input_trajectories,
               tm_formal_trajectories)

    # 2. 规约TM数据
    print("Step 2: Reducing the TM data...")
    run_script(get_final_tm_script,
               tm_formal_trajectories,
               reduced_tm_trajectories,
               tm_recording,  # <--- 路径已更新
               alpha_param,
               beta_param)

    # 3. 从数据中学习TM
    print("Step 3: Finding the TM from the data...")
    run_script(find_machine_script,
               reduced_tm_trajectories,
               output_mealy_tm)

    # 根据参数决定是否执行步骤4和修改步骤5的输入
    if use_supplementation:
        # 4. 状态补充
        print("Step 4: State supplementation...")
        run_script(state_supplement_script,
                   input_trajectories,
                   output_mealy_tm,
                   supplemented_trajectories)
        print("State supplementation applied.")

        # 5. 为学习RM生成自动机学习数据 (使用补充后的轨迹)
        print("Step 5: Generating automata learning data for learning RM (using supplemented trajectories)...")
        run_script(get_formal_rm_script,
                   supplemented_trajectories,
                   rm_formal_trajectories)
    else:
        print("Step 4: State supplementation... SKIPPED")
        # 5. 为学习RM生成自动机学习数据 (使用原始轨迹)
        print("Step 5: Generating automata learning data for learning RM (using original trajectories)...")
        run_script(get_formal_rm_script,
                   input_trajectories,
                   rm_formal_trajectories)

    # 6. 规约RM数据
    print("Step 6: Reducing the RM data...")
    run_script(get_final_rm_script,
               rm_formal_trajectories,
               reduced_rm_trajectories,
               rm_recording,  # <--- 路径已更新
               alpha_param,
               beta_param)

    # 7. 从数据中学习RM
    print("Step 7: Finding the RM from the data...")
    run_script(find_machine_script,
               reduced_rm_trajectories,
               output_mealy_rm)

    end_time = time.time()
    print(f"\nRuntime: {end_time - start_time:.2f} seconds")

    # 8. 绘制状态机（并行处理）
    print("Step 8: Draw Machines...")
    p1 = Process(target=run_script, args=(draw_machine_script, output_mealy_tm))
    p2 = Process(target=run_script, args=(draw_machine_script, output_mealy_rm))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("\nFinished!")


if __name__ == "__main__":
    main()
