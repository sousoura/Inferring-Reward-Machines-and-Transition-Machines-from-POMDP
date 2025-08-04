import json


def build_mealy_machine_json(reward_machine, output_file, use_state_labeling=False):
    """
    将 ILP 求解器输出的 reward_machine 转换为 Mealy Machine 格式，
    最终写入 output_file（如 "mealy_rm.json"）。

    输入 reward_machine 的格式（使用1-索引）：
      {
         "states": [1, 2, ..., K],
         "initial_state": 1,
         "transitions": {
            (i, s, a, s_next): { "next_state": j, "reward": r, "steps": [...] }  # 对于转移标记
            或
            (i, s_next, None, None): { "next_state": j, "reward": r, "steps": [...] }  # 对于状态标记
         }
      }

    输入参数：
      - reward_machine: 从ILP求解器获得的RM结构
      - output_file: 输出文件路径
      - use_state_labeling: 是否使用状态标记方式(L: S → 2^P)而非转移标记(L: (S × A × S) → 2^P)

    输出格式：
      {
         "initial_state": "q1",
         "states": {
           "q1": {
              "fingerprint": { "s,a": "r", ... } 或 { "s_next": "r", ... },
              "transitions": { "s,a": "qj", ... } 或 { "s_next": "qj", ... }
           },
           "q2": { ... },
           ...
         },
         "state_count": K
      }
    """
    # 根据 reward_machine 中 states 的个数确定 K
    if "states" in reward_machine:
        K = len(reward_machine["states"])
        min_state = min(reward_machine["states"])  # 应该是1
    else:
        # 根据 transitions 中出现的状态推断 K
        states_set = set()
        for key in reward_machine["transitions"]:
            i = key[0]  # 第一个元素总是i
            states_set.add(i)
            states_set.add(reward_machine["transitions"][key]["next_state"])
        K = len(states_set)
        min_state = min(states_set)  # 应该是1

    mealy_rm = {
        "initial_state": f"q{min_state}",  # 初始状态应该是q1
        "states": {},
        "state_count": K
    }

    # 初始化每个状态的 fingerprint 和 transitions 字典
    # 对于1-索引的状态，我们需要从 min_state 开始遍历
    for i in range(min_state, min_state + K):
        state_name = f"q{i}"
        mealy_rm["states"][state_name] = {"fingerprint": {}, "transitions": {}}

    # 检查是否需要自动检测标记方式
    if not use_state_labeling and any(key[2] is None and key[3] is None for key in reward_machine["transitions"]):
        use_state_labeling = True
        print("自动检测到状态标记方式 (L: S → 2^P)")

    # 遍历 reward_machine["transitions"] 中的每一条转移规则
    for key, info in reward_machine["transitions"].items():
        i = key[0]  # 当前状态i
        next_state = info["next_state"]  # 下一个状态j
        reward_val = info["reward"]  # 奖励r

        if use_state_labeling:
            # 对于状态标记方式 (L: S → 2^P)
            s_next = key[1]  # 第二个元素是s_next
            # 直接使用s_next作为fingerprint和transitions的key
            fingerprint_key = f"{s_next}"
        else:
            # 对于转移标记方式 (L: (S × A × S) → 2^P)
            s, a, s_next = key[1], key[2], key[3]
            # 使用"s,a"作为fingerprint和transitions的key
            fingerprint_key = f"{s},{a}"

        mealy_rm["states"][f"q{i}"]["fingerprint"][fingerprint_key] = str(reward_val)
        mealy_rm["states"][f"q{i}"]["transitions"][fingerprint_key] = f"q{next_state}"

    with open(output_file, 'w') as f:
        json.dump(mealy_rm, f, indent=2)
    print(f"Mealy RM saved to {output_file}")


def print_simplified_reward_machine(reward_machine):
    """
    打印简化版本的 Reward Machine（使用 L: S → 2^P 标记方式和 j ≥ i 约束）

    输入:
      - reward_machine: 从 solve_reward_machine_ILP_simplified 返回的 reward_machine 对象
    """
    # 打印基本信息
    print("Simplified Reward Machine:")
    print(f"States: {reward_machine['states']}")
    print(f"Initial State: u{reward_machine['initial_state']}")
    print("\nTransitions:")

    # 按当前状态组织转移
    transitions_by_state = {}
    for key, info in reward_machine["transitions"].items():
        i, s_next, _, _ = key
        j = info["next_state"]
        r = info["reward"]

        if i not in transitions_by_state:
            transitions_by_state[i] = []

        transitions_by_state[i].append((s_next, r, j))

    # 打印每个状态的转移
    for i in sorted(transitions_by_state.keys()):
        print(f"\nState u{i}:")
        for s_next, r, j in sorted(transitions_by_state[i]):
            print(f"  Input: s' = {s_next} → Reward: {r}, Next State: u{j}")

    # 打印没有转移的状态
    no_transition_states = set(reward_machine["states"]) - set(transitions_by_state.keys())
    if no_transition_states:
        print("\nStates with no outgoing transitions:")
        for i in sorted(no_transition_states):
            print(f"  State u{i}")


def output_simplified_reward_machine(reward_machine, file_path=None, elapsed_time=None):
    """
    打印简化版本的 Reward Machine（使用 L: S → 2^P 标记方式和 j ≥ i 约束）到控制台或文件

    输入:
      - reward_machine: 从 solve_reward_machine_ILP_simplified 返回的 reward_machine 对象
      - file_path: 可选，如果提供，输出将写入此文件路径而非打印到控制台
    """
    # 准备输出函数
    if file_path:
        file = open(file_path, 'w', encoding='utf-8')

        def write_line(text):
            file.write(text + '\n')
    else:
        def write_line(text):
            print(text)

    # 输出基本信息
    write_line("Simplified Reward Machine:")
    write_line(f"States: {reward_machine['states']}")
    write_line(f"Initial State: u{reward_machine['initial_state']}")
    write_line("\nTransitions:")

    # 按当前状态组织转移
    transitions_by_state = {}
    for key, info in reward_machine["transitions"].items():
        i, s_next, _, _ = key
        j = info["next_state"]
        r = info["reward"]

        if i not in transitions_by_state:
            transitions_by_state[i] = []

        transitions_by_state[i].append((s_next, r, j))

    # 输出每个状态的转移
    for i in sorted(transitions_by_state.keys()):
        write_line(f"\nState u{i}:")
        for s_next, r, j in sorted(transitions_by_state[i]):
            write_line(f"  Input: s' = {s_next} → Reward: {r}, Next State: u{j}")

    # 输出没有转移的状态
    no_transition_states = set(reward_machine["states"]) - set(transitions_by_state.keys())
    if no_transition_states:
        write_line("\nStates with no outgoing transitions:")
        for i in sorted(no_transition_states):
            write_line(f"  State u{i}")

    if elapsed_time:
        write_line(f"代码运行时间: {elapsed_time:.6f} 秒")
        for i in sorted(no_transition_states):
            write_line(f"  State u{i}")

    # 如果打开了文件，关闭它
    if file_path:
        file.close()
        print(f"结果已写入文件: {file_path}")


if __name__ == "__main__":
    # 测试示例1 - 使用转移标记 (L: (S × A × S) → 2^P)
    sample_rm_transition = {
        "states": [1, 2],
        "initial_state": 1,
        "transitions": {
            (1, "755", 3, "756"): {
                "next_state": 2,
                "reward": 1,
                "steps": [("755", 3, 1, "756")]
            },
            (2, "756", 3, "757"): {
                "next_state": 2,
                "reward": 0,
                "steps": [("756", 3, 0, "757")]
            }
        }
    }

    # 测试示例2 - 使用状态标记 (L: S → 2^P)
    sample_rm_state = {
        "states": [1, 2],
        "initial_state": 1,
        "transitions": {
            (1, "756", None, None): {
                "next_state": 2,
                "reward": 1,
                "steps": [("755", 3, 1, "756")]
            },
            (2, "757", None, None): {
                "next_state": 2,
                "reward": 0,
                "steps": [("756", 3, 0, "757")]
            }
        }
    }

    build_mealy_machine_json(sample_rm_transition, "mealy_rm_test_transition.json")
    build_mealy_machine_json(sample_rm_state, "mealy_rm_test_state.json", use_state_labeling=True)
    # 测试自动检测功能
    build_mealy_machine_json(sample_rm_state, "mealy_rm_test_auto.json")
