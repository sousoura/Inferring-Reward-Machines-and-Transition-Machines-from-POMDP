#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from collections import defaultdict
import sys


def parse_tm_line(line: str):
    """
    将形如:
      806,2 None 805,0 SettlementPoint -> [805, Default, 755, Default]
    的一行解析成:
      [("806,2", "805"), ("None", "Default"), ("805,0", "755"), ("SettlementPoint", "Default")]

    返回值: 一个 list，内部是 (input_symbol, output_symbol) 的元组。
    """
    line = line.strip()
    if not line:
        return []
    # 拆分 ->，左边是输入串，右边是输出串(带方括号)
    if "->" not in line:
        print(f"Warning: No '->' was found in: {line}")
        return []
    left_part, right_part = line.split("->")
    left_part = left_part.strip()  # "806,2 None 805,0 SettlementPoint"
    right_part = right_part.strip()  # "[805, Default, 755, Default]"

    # 去除右边的方括号
    if right_part.startswith("[") and right_part.endswith("]"):
        right_part = right_part[1:-1].strip()  # "805, Default, 755, Default"
    else:
        print(f"Warning: output should be in []: {right_part}")

    # 左边空格分割 -> 输入符号 list
    input_symbols = left_part.split()  # ["806,2", "None", "805,0", "SettlementPoint"]

    # 右边用逗号+空格分割 -> 输出符号 list
    # 注意要先按 ", " 分割
    output_symbols = right_part.split(", ")
    # 现在 output_symbols = ["805", "Default", "755", "Default"]

    if len(input_symbols) != len(output_symbols):
        print(f"Warning: Mismatch between the number of inputs and outputs: {line}")

    # 成对配对
    pairs = []
    for inp, outp in zip(input_symbols, output_symbols):
        pairs.append((inp, outp))
    return pairs


def stringify_tm_line(pairs):
    """
    将 (input_symbol, output_symbol) 的列表恢复成:
      inp1 inp2 ... -> [out1, out2, ...]
    """
    if not pairs:
        return ""
    inputs = []
    outputs = []
    for (inp, outp) in pairs:
        inputs.append(inp)
        outputs.append(outp)
    # 左边用空格连接
    left_str = " ".join(inputs)
    # 右边用逗号+空格连接，并加上方括号
    right_str = "[" + ", ".join(outputs) + "]"
    return f"{left_str} -> {right_str}"


def step1_remove_None_default(all_trajectories):
    """
    对每条轨迹(除第一行外)，移除 input="None"且output="Default" 的对。
    all_trajectories 是一个 list，其中每个元素对应一行的 list-of-pairs。
    """
    new_trajectories = []
    for pairs in all_trajectories:
        filtered = []
        for (inp, out) in pairs:
            # 若 inp="None" 并且 out="Default" 则跳过
            if (inp == "None") and (out == "Default"):
                continue
            filtered.append((inp, out))
        new_trajectories.append(filtered)
    return new_trajectories


def step2_remove_global_consistent(all_trajectories, record_filename):
    """
    检查所有“输出 != Default”的输入符号，如果它在所有出现过的地方输出都是同一个值，
    则将该 (input_symbol -> output_symbol) 称为“全局唯一映射”。
    - 从所有轨迹里删除这些对
    - 记录到 record_filename (避免重复)

    返回: new_trajectories
    """
    # 1) 收集 "输出 != Default" 的配对
    #    mapping[input_symbol] = set_of_output_symbols
    mapping = defaultdict(set)

    for pairs in all_trajectories:
        for (inp, out) in pairs:
            if out != "Default":
                mapping[inp].add(out)

    # 2) 找出“全局唯一映射”的输入符号
    #    条件: 对于某 input_sym，set_of_output_symbols 的大小=1
    global_consistent = {}
    for inp, out_set in mapping.items():
        if len(out_set) == 1:
            the_out = next(iter(out_set))
            global_consistent[inp] = the_out

    # 3) 检查这些输入符号在所有出现的位置是否都映射到同一个输出
    #    由于上一步已经保证每个输入符号只有一个输出符号，因此只需进一步确保
    #    它们在所有轨迹中出现时都映射到同一个输出
    #    实际上，这在上一步已经完成

    # 4) 从所有轨迹里删除这些对
    new_trajectories = []
    for pairs in all_trajectories:
        filtered = []
        for (inp, out) in pairs:
            # 若属于全局唯一映射之一，则不要了
            if inp in global_consistent and out == global_consistent[inp]:
                continue
            filtered.append((inp, out))
        new_trajectories.append(filtered)

    # 5) 把这些全局映射写入 record_filename
    #    格式: input_symbol -> output_symbol
    #    要避免重复行，所以用一个 set 记录
    recorded_pairs = set()
    if os.path.exists(record_filename):
        # 先读入已有的记录，避免重复
        with open(record_filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "->" in line:
                    recorded_pairs.add(line)

    # 新加入
    for inp, out in global_consistent.items():
        new_line = f"{inp} -> {out}"
        recorded_pairs.add(new_line)

    # 写回 record_filename
    with open(record_filename, "w", encoding="utf-8") as f:
        for line in sorted(recorded_pairs):
            f.write(line + "\n")

    return new_trajectories


def collect_remaining_input_symbols(all_trajectories):
    """
    收集所有简化后的轨迹中出现的输入符号，去除重复，返回一个 set。
    """
    input_symbols = set()
    for pairs in all_trajectories:
        for (inp, _) in pairs:
            input_symbols.add(inp)
    return input_symbols


def main():
    if len(sys.argv) > 5:
        in_filename = sys.argv[1]
        out_filename = sys.argv[2]
        record_file = sys.argv[3]
        alpha_mode = sys.argv[4]
        beta_mode = sys.argv[5]
    else:
        raise Exception("No input filename")

    # in_filename = "TM_formal_trajectories.txt"


    # 如果不想保留之前的记录，可以先清空 record_file
    # 若希望保留并追加新的记录，注释掉以下两行
    if os.path.exists(record_file):
        os.remove(record_file)

    # 1) 读入 TM_formal_trajectories.txt
    with open(in_filename, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        print(f"{in_filename} is empty.")
        return

    # 第一行是所有可能的输入符号
    # 现在我们需要更新它，所以暂时不使用
    # first_line = lines[0]

    # 其余行是实际的轨迹
    traj_lines = lines[1:]

    # 解析每行变成 list of (input_symbol, output_symbol)
    all_trajectories = []
    for line in traj_lines:
        pairs = parse_tm_line(line)
        if pairs:
            all_trajectories.append(pairs)

    if beta_mode == "beta_preprocess":
        # 2) 第一步: 移除 input="None" & output="Default"
        print(beta_mode)
        all_trajectories = step1_remove_None_default(all_trajectories)
        print("done")
    else:
        print("no beta_preprocess")

    if alpha_mode == "alpha_preprocess":
        # 3) 第二步: 找到“输出 != Default”且全局唯一映射的输入符号，并记录+删除
        print(alpha_mode)
        all_trajectories = step2_remove_global_consistent(all_trajectories, record_file)
        print("done")
    else:
        print("no alpha_preprocess")

    # 4) 收集所有剩余的输入符号
    remaining_inputs = collect_remaining_input_symbols(all_trajectories)
    # 按字典序排序，确保一致性
    sorted_remaining_inputs = sorted(list(remaining_inputs))
    # 生成第一行字符串，用空格分隔
    new_first_line = " ".join(sorted_remaining_inputs)

    # 5) 输出到 reduced_TM_trajectories.txt
    with open(out_filename, "w", encoding="utf-8") as f:
        # 写入更新后的第一行
        f.write(new_first_line + "\n")

        # 然后写处理后的轨迹
        for pairs in all_trajectories:
            if not pairs:
                # 若整行都删空了，就空行
                f.write("\n")
            else:
                line_str = stringify_tm_line(pairs)
                f.write(line_str + "\n")

    print(f"The learning data: {out_filename}")
    print(f"The recoding file: {record_file}")


if __name__ == "__main__":
    main()
