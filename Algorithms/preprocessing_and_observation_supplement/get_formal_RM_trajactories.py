#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import sys

def transform_trajectory(trajectory):
    """
    将单个轨迹转换为所需的 RM_formal_trajectories.txt 格式。

    参数:
        trajectory (list): 一个轨迹，包含多个三元组 [[观察标记, 观察编号], 动作, 奖励]

    返回:
        tuple: (输入符号列表, 输出符号列表)
    """
    transformed_inputs = []
    transformed_outputs = []

    # 忽略最后一个三元组
    for i in range(len(trajectory) - 1):
        current_triplet = trajectory[i]
        next_triplet = trajectory[i + 1]

        # 解析当前三元组
        current_state_label, current_obs_id = current_triplet[0]
        current_action = current_triplet[1]
        current_reward = current_triplet[2]

        # 解析下一个三元组的观察标记
        next_state_label, next_obs_id = next_triplet[0]

        # 构建输入符号和输出符号
        input_symbol = f"{current_obs_id},{current_action} {next_state_label}"
        output_symbol = f"{current_reward}, Default"

        transformed_inputs.append(input_symbol)
        transformed_outputs.append(output_symbol)

    return transformed_inputs, transformed_outputs


def main():
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
        output_filename = sys.argv[2]
    else:
        raise Exception("No input filename")

    unique_inputs = set()
    transformed_lines = []

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                try:
                    trajectory = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"JSON Parse mistake is on the {line_num} line: {e}")
                    continue

                if not isinstance(trajectory, list):
                    print(f"The {line_num} line's data format is not list, skip.")
                    continue

                if len(trajectory) < 2:
                    print(f"The {line_num} line's trajectory's length is not enough, skip.")
                    continue

                inputs, outputs = transform_trajectory(trajectory)

                # 更新唯一输入符号集合
                unique_inputs.update(inputs)

                # 构建输出格式
                input_str = " ".join(inputs)
                output_str = "[" + ", ".join(outputs) + "]"
                transformed_line = f"{input_str} -> {output_str}"
                transformed_lines.append(transformed_line)

        # 排序唯一输入符号，确保一致性
        sorted_unique_inputs = sorted(unique_inputs)
        unique_inputs_line = " ".join(sorted_unique_inputs)

        # 写入输出文件
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            # 写入第一行：所有唯一的输入符号
            outfile.write(unique_inputs_line + "\n")

            # 写入转换后的轨迹
            for transformed_line in transformed_lines:
                outfile.write(transformed_line + "\n")

        print(f"Generated: {output_filename}")

    except FileNotFoundError:
        print(f"Can't find {input_filename}.")
    except Exception as e:
        print(f"Mistake happened: {e}")


if __name__ == "__main__":
    main()
