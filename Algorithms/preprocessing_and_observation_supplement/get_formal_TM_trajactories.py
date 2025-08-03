import json
import sys


def transform_trajectory(trajectory_line):
    """
    将单行 trajectory (列表) 转换为：
      (输入串, 输出串, 该行里出现过的所有输入符号集合)
    """
    # trajectory_line 形如:
    # [
    #   [["None", 806], 2, -0.1],
    #   [["None", 805], 0, -0.1],
    #   [["SettlementPoint", 755], 0, -0.1],
    #   ...
    # ]

    # 存储每一步的 (mark, obsID, action)
    steps = []
    for item in trajectory_line:
        mark, obs_id = item[0]  # item[0] = ["SomeMark", some_id]
        action = item[1]
        steps.append((mark, obs_id, action))

    # 生成输入、输出符号
    input_symbols = []
    output_symbols = []

    # 假设和题主的例子一致：“最后一步”不产生新的对，而是并入倒数第二步
    # 所以真正要遍历到 steps.length - 1 次
    for i in range(len(steps) - 1):
        curr_mark, curr_obs, curr_action = steps[i]
        next_mark, next_obs, _ = steps[i + 1]

        # 1) 输入符号：  "curr_obs,curr_action next_mark"
        inp = f"{curr_obs},{curr_action} {next_mark}"
        input_symbols.append(inp)

        # 2) 输出符号： "[next_obs, Default]"
        out = f"{next_obs}, Default"
        output_symbols.append(out)

    # 上面只是实现了“每一步都和它的下一步配对”，
    # 若想把“最后一步”单独输出，也可以在此处额外处理:
    #   (mark, obs_id, action) = steps[-1]
    #   inp = f"{obs_id},{action} ???"
    #   out = f"[ ???, Default]"
    #   具体看需求

    # 把所有输入符号连起来
    input_str = " ".join(input_symbols)
    # 把所有输出符号放进一个列表格式
    # 例如 [805, Default, 755, Default] ...
    # 因为题主示例是连在一个列表里
    output_list = []
    for out in output_symbols:
        # out 现在形如  "805, Default"
        # 分割一下做成 ["805", "Default"] 再展开
        items = out.split(", ")
        output_list.extend(items)

    # 形如 -> [805, Default, 755, Default]
    output_str = "[" + ", ".join(output_list) + "]"

    # 该行最终输出:  806,2 None 805,0 SettlementPoint -> [805, Default, 755, Default]
    final_line = f"{input_str} -> {output_str}"

    # 另外，为了方便后面收集所有“输入符号”，把 input_symbols 返回回去
    return final_line, set(input_symbols)


def main():
    if len(sys.argv) > 2:
        input_filename = sys.argv[1]
        output_filename = sys.argv[2]
    else:
        raise Exception("No input filename")

    # 读取 trajectories.json（这里假设里面是一行一行的 JSON，每行都是一个大列表）
    # 如果你的文件结构不同，需要按实际情况处理
    all_lines = []
    with open(input_filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 假设每行都是一个完整的 JSON 列表
            traj = json.loads(line)
            all_lines.append(traj)

    # 收集所有可能的输入符号
    all_input_symbols = set()

    # 存储转换后的结果
    transformed_lines = []

    for trajectory_line in all_lines:
        final_line, input_symbol_set = transform_trajectory(trajectory_line)
        transformed_lines.append(final_line)
        all_input_symbols.update(input_symbol_set)

    # 将所有输入符号排序(以防止每次运行顺序不一致),再写进第一行
    all_input_symbols = sorted(list(all_input_symbols))
    all_input_symbols_line = " ".join(all_input_symbols)

    # 写出到 TM_formal_trajectories.txt
    with open(output_filename, "w", encoding="utf-8") as out_f:
        # 第一行: 所有(不重复)的输入符号
        out_f.write(all_input_symbols_line + "\n")
        # 后面每行: 转换后的 mealy machine 行
        for line_str in transformed_lines:
            out_f.write(line_str + "\n")


if __name__ == "__main__":
    main()
