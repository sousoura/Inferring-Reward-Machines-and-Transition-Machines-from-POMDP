import json
import numpy as np
import random
import time
import networkx as nx
import sys
import argparse
import os


def print_progress(message):
    """打印带有时间戳的进度信息"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # 确保实时显示


def generate_environment_parameters(
        env_size=7,
        labels_count=4,
        transition_states_count=3,
        reward_states_count=4,
        output_file="env_para.json"
):
    """生成强化学习环境参数"""

    # print_progress("开始生成环境参数...")

    # 1. 环境基本参数
    # print_progress("Step 1/6: 初始化环境基本参数")
    env_params = {
        "environment": {
            "size": env_size
        },
        "labels": {
            "count": labels_count,
            "mapping": {}
        },
        "transition_automaton": {
            "states_count": transition_states_count,
            "transitions": {},
            "impassable_positions": {}
        },
        "reward_automaton": {
            "states_count": reward_states_count,
            "transitions": {},
            "rewards": {},
            "terminal_state": f"q'_{reward_states_count - 1}"  # 最后一个状态作为终止状态
        }
    }

    # 2. 生成标签映射 - 仅在(单数,单数)位置生成标签
    # print_progress("Step 2/6: 生成标签映射")
    label_positions = []
    for i in range(1, env_size + 1, 2):
        for j in range(1, env_size + 1, 2):
            label_positions.append((i, j))

    # print_progress(f"  - 找到 {len(label_positions)} 个可标记位置")

    # 为每个位置随机分配标签
    for pos in label_positions:
        label = random.randint(0, labels_count - 1)
        env_params["labels"]["mapping"][str(pos)] = label

    # print_progress(f"  - 已完成所有位置的标签分配")

    # 3. 生成转移自动机
    # print_progress("Step 3/6: 生成转移自动机")

    # 创建状态名称
    transition_states = [f"q_{i}" for i in range(transition_states_count)]

    # 初始化转移图 - 使用MultiDiGraph允许多标签边
    G_transition = nx.MultiDiGraph()
    for state in transition_states:
        G_transition.add_node(state)

    # print_progress(f"  - 生成状态间的转移...")

    # 确保自动机是强连通的 (每个状态可以到达其他任何状态)
    # 首先创建一个环确保连通性
    for i in range(transition_states_count):
        next_state = (i + 1) % transition_states_count
        label = random.randint(0, labels_count - 1)
        G_transition.add_edge(transition_states[i], transition_states[next_state], label=str(label))

    # 添加额外的随机转移 - 不再检查是否已有边
    for i in range(transition_states_count):
        for j in range(transition_states_count):
            if i != j:  # 允许添加到同一目标的多条边
                if random.random() < 0.6:  # 60%的概率添加额外转移
                    label = random.randint(0, labels_count - 1)
                    G_transition.add_edge(transition_states[i], transition_states[j], label=str(label))

    # 添加自环
    for i in range(transition_states_count):
        if random.random() < 0.3:  # 30%的概率添加自环
            label = random.randint(0, labels_count - 1)
            G_transition.add_edge(transition_states[i], transition_states[i], label=str(label))

    # print_progress(f"  - 验证转移自动机的连通性...")

    # 验证是否强连通 - 创建一个简单图用于检查连通性
    simple_G = nx.DiGraph()
    for u, v in G_transition.edges():
        simple_G.add_edge(u, v)

    if not nx.is_strongly_connected(simple_G):
        # print_progress(f"  - 警告: 生成的自动机不是强连通的，正在添加必要的转移...")
        # 找出不连通的部分并添加连接
        for component in nx.strongly_connected_components(simple_G):
            if len(component) < len(transition_states):
                outside_states = set(transition_states) - set(component)
                for state in component:
                    for outside in outside_states:
                        # 不检查是否有边，直接添加新边
                        label = random.randint(0, labels_count - 1)
                        G_transition.add_edge(state, outside, label=str(label))

    # 构建转移字典 - 允许同一状态下不同标签转移到同一目标
    transitions = {}
    used_labels = {}

    for state in transition_states:
        transitions[state] = {}
        used_labels[state] = set()

    # 处理所有边，考虑多重边的情况
    for u, v, key, data in G_transition.edges(data=True, keys=True):
        label = data.get('label')
        if label is None:
            continue

        # 检查该状态是否已经使用了这个标签
        if label in used_labels[u]:
            # 如果该标签已经有转移，随机选择一个新标签
            available_labels = [str(i) for i in range(labels_count) if str(i) not in used_labels[u]]
            if available_labels:
                label = random.choice(available_labels)
                transitions[u][label] = v
                used_labels[u].add(label)
        else:
            transitions[u][label] = v
            used_labels[u].add(label)

    # 确保每个标签至少被某个状态处理
    for label in range(labels_count):
        label_str = str(label)
        handled = False
        for state, used in used_labels.items():
            if label_str in used:
                handled = True
                break

        if not handled:
            # 随机选择一个状态添加这个标签的转移
            state = random.choice(transition_states)
            target = random.choice(transition_states)

            # 如果该状态已用完所有标签，跳过
            if len(used_labels[state]) >= labels_count:
                continue

            transitions[state][label_str] = target
            used_labels[state].add(label_str)

    env_params["transition_automaton"]["transitions"] = transitions

    # 4. 设置不可通行位置
    # print_progress("Step 4/6: 设置不可通行位置")

    # 为每个状态随机选择部分标签位置作为不可通行
    impassable_positions = {}
    for state in transition_states:
        # 随机选择约30%的位置为不可通行
        num_impassable = max(1, int(len(label_positions) * 0.3))
        selected_positions = random.sample(label_positions, num_impassable)
        impassable_positions[state] = selected_positions

    # 确保任何位置至少在一种状态下是可通行的
    for pos in label_positions:
        blocked_in_all_states = True
        for state, blocked_positions in impassable_positions.items():
            if pos not in blocked_positions:
                blocked_in_all_states = False
                break

        if blocked_in_all_states:
            # 随机选择一个状态使该位置可通行
            state = random.choice(transition_states)
            impassable_positions[state].remove(pos)

    # 将不可通行位置转换为字符串格式
    for state in impassable_positions:
        env_params["transition_automaton"]["impassable_positions"][state] = [str(pos) for pos in
                                                                             impassable_positions[state]]

    # 5. 生成奖励自动机 - 使用优化方法避免环检测
    # print_progress("Step 5/6: 生成奖励自动机 (优化版)")

    # 创建状态名称
    reward_states = [f"q'_{i}" for i in range(reward_states_count)]
    terminal_state = reward_states[-1]  # 最后一个状态作为终止状态

    # print_progress(f"  - 构建初始无环图...")

    # 使用MultiDiGraph允许同一对状态间有多条边
    G_reward = nx.MultiDiGraph()
    for state in reward_states:
        G_reward.add_node(state)

    # 为每个非终止状态添加向前的转移
    for i in range(reward_states_count - 1):  # 不包括终止状态
        state = reward_states[i]

        # 每个状态随机连接到后面的一些状态
        for j in range(i + 1, reward_states_count):
            # 随机决定添加多少条边到这个目标
            num_edges = random.randint(0, min(3, labels_count // 2))  # 最多添加3条或一半标签数量
            for _ in range(num_edges):
                if random.random() < 0.6:  # 60%的概率添加转移
                    label = random.randint(0, labels_count - 1)
                    # 生成随机的浮点数奖励值，正负混合
                    target_state = reward_states[j]
                    reward = 0
                    # 如果目标是终止状态，确保奖励为正
                    if target_state == terminal_state:
                        reward = round(random.uniform(1.0, 15.0), 2)
                    else:
                        # 否则，奖励可以是正或负
                        reward = round(random.uniform(-5.0, 15.0), 2)
                    G_reward.add_edge(state, reward_states[j], label=str(label), reward=reward)

    # 确保从每个非终止状态都有至少一条到其他状态的路径
    for i in range(reward_states_count - 1):
        state = reward_states[i]
        if len(G_reward.out_edges(state)) == 0:
            # 如果没有出边，添加一条到下一个状态或终止状态的边
            target_idx = random.randint(i + 1, reward_states_count - 1)
            label = random.randint(0, labels_count - 1)
            # 使用随机浮点数奖励
            target_state = reward_states[j]
            reward = 0
            # 如果目标是终止状态，确保奖励为正
            if target_state == terminal_state:
                reward = round(random.uniform(1.0, 15.0), 2)
            else:
                # 否则，奖励可以是正或负
                reward = round(random.uniform(-5.0, 15.0), 2)
            G_reward.add_edge(state, reward_states[target_idx], label=str(label), reward=reward)

    # 确保每个状态都能到达终止状态
    # print_progress(f"  - 确保所有状态可达终止状态...")

    # 检查是否所有状态都能到达终止状态 - 需要创建简单图
    simple_G_reward = nx.DiGraph()
    for u, v in G_reward.edges():
        simple_G_reward.add_edge(u, v)

    for i in range(reward_states_count - 1):
        state = reward_states[i]
        if not nx.has_path(simple_G_reward, state, terminal_state):
            # 添加一条直接到终止状态的边
            label = random.randint(0, labels_count - 1)
            # 到终止状态的边通常给予正奖励
            reward = round(random.uniform(1.0, 20.0), 2)
            G_reward.add_edge(state, terminal_state, label=str(label), reward=reward)

    # print_progress(f"  - 添加回边并确保负奖励回环...")

    # 现在我们添加一些回边（从编号大的状态到编号小的状态）
    # 但确保这些回边有足够大的负奖励
    added_back_edges = 0
    for i in range(1, reward_states_count):
        state = reward_states[i]

        # 只对非终止状态添加回边
        if state == terminal_state:
            continue

        # 为每个早期状态添加回边的概率
        num_back_edges = random.randint(0, i)  # 可以添加到多个早期状态
        for _ in range(num_back_edges):
            if random.random() < 0.5:  # 50%概率添加回边
                # 随机选择一个较早的状态作为目标
                target_idx = random.randint(0, i - 1)
                target = reward_states[target_idx]

                # 确定最大可能的正奖励
                max_possible_reward = 0
                # 找出从target到当前state可能的最大正奖励路径
                for path in nx.all_simple_paths(simple_G_reward, target, state):
                    path_reward = 0
                    for idx in range(len(path) - 1):
                        # 正确使用get_edge_data获取所有边的数据
                        edge_dict = G_reward.get_edge_data(path[idx], path[idx + 1])
                        if edge_dict:
                            # 找出所有连接这两个节点的边中最大的奖励
                            max_edge_reward = max(
                                [data.get('reward', float('-inf')) for data in edge_dict.values()]
                            )
                            if max_edge_reward != float('-inf'):
                                path_reward += max_edge_reward
                    max_possible_reward = max(max_possible_reward, path_reward)

                # 设置足够大的负奖励来抵消可能的正奖励
                # 使用浮点数并增加一些随机性
                safety_margin = random.uniform(1.0, 5.0)
                back_edge_reward = -1.0 * (max_possible_reward + safety_margin)
                # 四舍五入到2位小数
                back_edge_reward = round(back_edge_reward, 2)

                # 添加回边 - 可以添加多条不同标签的回边
                label = random.randint(0, labels_count - 1)
                G_reward.add_edge(state, target, label=str(label), reward=back_edge_reward)
                added_back_edges += 1

    # print_progress(f"  - 添加了 {added_back_edges} 条带负奖励的回边")

    # 构建奖励自动机的转移和奖励字典，确保所有转移都有奖励
    transitions = {}
    rewards = {}
    used_labels = {}

    # 初始化所有状态的转移和奖励字典
    for state in reward_states:
        transitions[state] = {}
        rewards[state] = {}
        used_labels[state] = set()

    # 处理所有边 - 同时记录转移和奖励
    for u, v, key, data in G_reward.edges(data=True, keys=True):
        label = data.get('label')
        reward = data.get('reward')

        if label is None or reward is None:
            continue

        # 检查是否已使用该标签
        if label in used_labels[u]:
            # 如果该标签已经有转移，随机选择一个新标签
            available_labels = [str(i) for i in range(labels_count) if str(i) not in used_labels[u]]
            if available_labels:
                label = random.choice(available_labels)
                transitions[u][label] = v
                rewards[u][label] = reward
                used_labels[u].add(label)
        else:
            transitions[u][label] = v
            rewards[u][label] = reward
            used_labels[u].add(label)

    # 添加标签奖励 - 确保所有在转移中出现的标签都有对应的奖励
    for state in reward_states:
        # 终止状态不需要奖励定义
        if state == terminal_state:
            continue

        # 为每个标签添加对应的奖励
        for label_str in range(labels_count):
            label_str = str(label_str)

            # 如果标签在转移中出现但没有奖励，添加奖励值
            if label_str in transitions[state] and label_str not in rewards[state]:
                # 生成随机奖励值
                reward_value = round(random.uniform(-5.0, 15.0), 2)
                rewards[state][label_str] = reward_value

    # 验证所有转移都有对应的奖励
    for state in reward_states:
        if state == terminal_state:  # 跳过终止状态
            continue
        for label in transitions[state]:
            if label not in rewards[state]:
                # print_progress(f"  - 警告: 状态 {state} 的标签 {label} 没有奖励定义，正在添加默认奖励")
                rewards[state][label] = round(random.uniform(-2.0, 5.0), 2)

    # 存储到环境参数
    env_params["reward_automaton"]["transitions"] = transitions
    env_params["reward_automaton"]["rewards"] = rewards

    # 6. 保存参数到文件
    # print_progress("Step 6/6: 保存参数到文件")

    with open(output_file, "w") as f:
        json.dump(env_params, f, indent=2)

    # print_progress(f"环境参数已成功保存到 {output_file}")
    return env_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate environment parameters for RL.")

    parser.add_argument('--env-size', type=int, default=25, help='Size of the grid environment.')
    parser.add_argument('--labels-count', type=int, default=5, help='Number of distinct labels.')
    parser.add_argument('--transition-states-count', type=int, default=7,
                        help='Number of states in the transition automaton.')
    parser.add_argument('--reward-states-count', type=int, default=3, help='Number of states in the reward automaton.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--output', type=str, default='env_para.json',
                        help='Output filename for the environment parameters.')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Using random seed: {args.seed}")

    # 构建输出文件的完整路径，使其位于项目根目录
    project_root = os.path.join(os.path.dirname(__file__), "..")
    output_path = os.path.normpath(os.path.join(project_root, args.output))

    # 调用生成函数
    generate_environment_parameters(
        env_size=args.env_size,
        labels_count=args.labels_count,
        transition_states_count=args.transition_states_count,
        reward_states_count=args.reward_states_count,
        output_file=output_path
    )
