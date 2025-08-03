import os
import subprocess
import sys
import time
from multiprocessing import Process


def run_script(script_path, *args):
    """运行Python脚本并传递参数"""
    cmd = [sys.executable, script_path] + list(args)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_draw_machine(json_file):
    """运行 draw_machine.py 的辅助函数"""
    run_script("LearningAlgorithm/draw_machine.py", json_file)


def main():
    # 确保工作目录是脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 8. 调用draw_machine.py
    print("Draw Machine...")
    p1 = Process(target=run_draw_machine, args=("mealy_tm.json",))
    p1.start()
    p1.join()

    print("Finished!")


if __name__ == "__main__":
    main()