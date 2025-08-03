#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
find_mealy_machine.py
=====================
从指定的轨迹数据文件中学习并构建一个 Dual-Behavior Mealy Machine (DBMM)，
然后使用类 RPNI 的状态合并算法（Blue Fringe 方法）进行最小化，
最终将学习到的自动机以 JSON 格式输出。

用法:
  python find_mealy_machine.py <input_file> <output_file>
"""

import sys
import json
import collections
import itertools
import argparse
from typing import List, Set, Tuple, Dict, Optional, Any
from merge_algorithm import Graph, construct_quotient_graph_with_fingerprints


# --- 数据结构定义 ---
class Node:
    _ids = itertools.count(0)  # 用于生成唯一ID

    def __init__(self):
        self.id = next(Node._ids)
        self.fingerprint = {}  # 第一类符号的输入->输出映射
        self.transitions = {}  # 第二类符号的转移
        self.color = None  # 用于RPNI算法（'red'或'blue'或None）
        self.parent = None  # 用于构建前缀树

    def __str__(self):
        color_str = f", color={self.color}" if self.color else ""
        return f"Node({self.id}){color_str}: fp={self.fingerprint}, trans={list(self.transitions.keys())}"

    def __repr__(self):
        return self.__str__()


# --- 文件解析 ---
def parse_input_file(file_path: str) -> Tuple[Set[str], List[List[Tuple[str, str]]]]:
    # print(f"开始解析输入文件: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"输入文件 {file_path} 未找到。")
    except Exception as e:
        raise IOError(f"读取文件 {file_path} 时出错: {e}")
    if not lines: raise ValueError("输入文件为空或只包含空行。")
    all_symbols = set(lines[0].split(" "))
    if not all_symbols or all_symbols == {''}: raise ValueError("第一行未能读取到任何有效符号。")
    traces = []
    for idx, line in enumerate(lines[1:], start=2):
        if "->" not in line: raise ValueError(f"第 {idx} 行格式错误：缺少 '->'。")
        parts = line.split("->")
        if len(parts) != 2: raise ValueError(f"第 {idx} 行格式错误：必须包含一个 '->'。")
        inputs_str, outputs_str = parts[0].strip(), parts[1].strip()
        if not (outputs_str.startswith("[") and outputs_str.endswith("]")): raise ValueError(
            f"第 {idx} 行格式错误：输出需用 '[]' 包裹。")
        outputs_str_inner = outputs_str[1:-1].strip()
        input_symbols = inputs_str.split(" ") if inputs_str else []
        output_symbols = [x.strip() for x in outputs_str_inner.split(", ")] if outputs_str_inner else []
        if len(input_symbols) != len(output_symbols): raise ValueError(
            f"第 {idx} 行错误：输入输出数量不匹配 ({len(input_symbols)} vs {len(output_symbols)})!")
        if not input_symbols: continue
        traces.append(list(zip(input_symbols, output_symbols)))
    # print(f"成功解析 {len(traces)} 条轨迹，共 {sum(len(t) for t in traces)} 个输入/输出对")
    if not traces: raise ValueError("输入文件中未找到有效的轨迹数据。")
    return all_symbols, traces


# --- PTT 构建 ---
def build_prefix_tree(traces: List[List[Tuple[str, str]]]) -> Node:
    """构建前缀树转换器(PTT)"""
    print("Generating PTT...")
    root = Node()
    root.color = 'red'  # 根节点初始为红色

    alpha_symbols = set()  # 记录第一类符号
    beta_symbols = set()  # 记录第二类符号

    for trace_idx, trace in enumerate(traces, 1):
        current = root
        for input_symbol, output_symbol in trace:
            if output_symbol == "Default":  # 第二类符号，导致状态转移
                beta_symbols.add(input_symbol)
                if input_symbol not in current.transitions:
                    new_state = Node()
                    new_state.parent = current
                    current.transitions[input_symbol] = new_state
                current = current.transitions[input_symbol]
            else:  # 第一类符号，更新当前状态的指纹
                alpha_symbols.add(input_symbol)
                # 检查指纹一致性
                if input_symbol in current.fingerprint and current.fingerprint[input_symbol] != output_symbol:
                    raise ValueError(
                        f"Conflict：state{current.id}s input '{input_symbol}' has different output：'{current.fingerprint[input_symbol]}' with '{output_symbol}'")
                current.fingerprint[input_symbol] = output_symbol

    all_states = collect_all_states(root)
    print(f"The PTT has: {len(all_states)} states")
    print(f"α: {len(alpha_symbols)}, Instances: {list(alpha_symbols)[:5]}")
    print(f"β: {len(beta_symbols)}, Instances: {list(beta_symbols)[:5]}")

    # 打印PTT结构
    # print("\n初始前缀树结构:")
    for state in all_states:
        fp_sample = {k: state.fingerprint[k] for k in list(state.fingerprint.keys())[:3]} if state.fingerprint else {}
        if len(state.fingerprint) > 3:
            fp_sample["..."] = f"(there are another {len(state.fingerprint) - 3})"
        trans_sample = list(state.transitions.keys())[:3]
        if len(state.transitions) > 3:
            trans_sample.append(f"...(there are another {len(state.transitions) - 3})")

        # print(f"  状态{state.id}({state.color}): 指纹={fp_sample}, 转移={trans_sample}")

    reassign_ids_bfs(root)

    return root


def reassign_ids_bfs(root: Node) -> None:
    """按BFS顺序重新分配节点ID"""
    # 使用BFS收集所有节点
    visited = set()
    queue = collections.deque([root])
    bfs_order = []

    while queue:
        node = queue.popleft()
        if node.id in visited:
            continue

        visited.add(node.id)
        bfs_order.append(node)

        # 按顺序添加后继节点
        for symbol in sorted(node.transitions.keys()):
            next_node = node.transitions[symbol]
            if next_node.id not in visited:
                queue.append(next_node)

    # 重置ID计数器
    Node._ids = itertools.count(0)

    # 按BFS顺序分配新ID
    for node in bfs_order:
        node.id = next(Node._ids)


# --- 辅助函数 ---
def collect_all_states(root: Node) -> List[Node]:
    """收集自动机中的所有状态"""
    states = []
    visited = set()
    queue = collections.deque([root])

    while queue:
        state = queue.popleft()
        if state.id in visited:
            continue

        visited.add(state.id)
        states.append(state)

        for target in state.transitions.values():
            queue.append(target)

    return states


def collect_all_symbols(states: List[Node]) -> Set[str]:
    """收集所有使用的符号"""
    alpha_symbols = set()  # 第一类符号
    beta_symbols = set()  # 第二类符号

    for state in states:
        alpha_symbols.update(state.fingerprint.keys())
        beta_symbols.update(state.transitions.keys())

    return alpha_symbols.union(beta_symbols)


def update_blue_states(red_states: Set[Node], blue_states: Set[Node], all_states: List[Node]) -> None:
    """更新蓝色状态集（红色状态的未处理直接后继）"""
    # 清除已经变成红色的蓝色状态
    blue_states.difference_update(red_states)

    # 添加红色状态的新直接后继
    new_blues = set()
    for red in red_states:
        for target in red.transitions.values():
            if target not in red_states and target not in blue_states:
                target.color = 'blue'
                new_blues.add(target)

    blue_states.update(new_blues)


def create_graph_for_merging(all_states: List[Node], root: Node) -> Tuple[Graph, Dict[int, Node]]:
    """将我们的自动机转换为merge_algorithm使用的Graph格式"""
    G = Graph()
    state_map = {state.id: state for state in all_states}  # ID到节点的映射

    # 收集所有符号
    all_symbols = collect_all_symbols(all_states)

    # 设置图属性
    G.initial_vertex = root.id
    G.symbols = all_symbols
    G.vertices = {state.id for state in all_states}

    # 添加所有状态和指纹
    for state in all_states:
        G.add_vertex(state.id, state.fingerprint.copy())

    # 添加所有转移
    for state in all_states:
        for symbol, target in state.transitions.items():
            G.add_transition(state.id, symbol, target.id)

    return G, state_map


def update_automaton_from_merge(root: Node, G_prime: Graph, equiv_classes: Dict, state_map: Dict[int, Node]) -> Node:
    """根据合并结果更新原自动机结构"""
    # 创建新的根节点
    new_root = Node()
    new_root.id = G_prime.initial_vertex

    # 创建等价类到新节点的映射
    new_nodes = {}
    id_to_equiv = {}  # 原始ID到等价类代表的映射

    # 找出每个原始节点所属的等价类
    for repr_id, members in equiv_classes.items():
        for member_id in members:
            id_to_equiv[member_id] = repr_id

    # 创建新节点
    for vertex_id in G_prime.vertices:
        if vertex_id not in new_nodes:
            new_node = Node()
            new_node.id = vertex_id

            # 合并指纹
            for orig_id in equiv_classes.get(vertex_id, [vertex_id]):
                if orig_id in state_map:
                    orig_node = state_map[orig_id]
                    new_node.fingerprint.update(orig_node.fingerprint)

            # 设置颜色（根据包含的原始节点的颜色）
            contained_nodes = [state_map[orig_id] for orig_id in equiv_classes.get(vertex_id, [vertex_id]) if
                               orig_id in state_map]

            if any(node.color == 'red' for node in contained_nodes):
                new_node.color = 'red'
            elif any(node.color == 'blue' for node in contained_nodes):
                new_node.color = 'blue'

            new_nodes[vertex_id] = new_node

    # 添加转移
    for vertex_id in G_prime.vertices:
        node = new_nodes[vertex_id]
        if vertex_id in G_prime.transitions:
            for symbol, target_id in G_prime.transitions[vertex_id].items():
                node.transitions[symbol] = new_nodes[target_id]

    return new_nodes[G_prime.initial_vertex]


# --- RPNI 算法 ---
def blue_fringe_rpni(root: Node) -> Node:
    """使用Blue Fringe启发式实现RPNI状态合并算法"""
    # print("\n开始执行Blue Fringe RPNI算法...")

    # 初始化红蓝状态集
    all_states = collect_all_states(root)
    red_states = {state for state in all_states if state.color == 'red'}
    blue_states = set()

    # 初始更新蓝色状态集
    update_blue_states(red_states, blue_states, all_states)

    # 输出初始状态
    # print(f"初始状态: {len(all_states)} 个总状态, {len(red_states)} 个红色状态, {len(blue_states)} 个蓝色状态")
    # print(f"红色状态: {[s.id for s in red_states]}")
    # print(f"蓝色状态: {[s.id for s in blue_states]}")

    # 统计信息
    iteration_count = 0
    merge_attempts = 0
    successful_merges = 0

    # 主循环
    while blue_states:
        iteration_count += 1
        print(f"\nIteration #{iteration_count}:")

        # 选择一个蓝色状态处理
        blue = min(blue_states, key=lambda x: x.id)  # 选择ID最小的蓝色状态
        # print(f"  处理蓝色状态: {blue.id}")
        merged = False

        # 尝试与所有红色状态合并
        for red in sorted(red_states, key=lambda x: x.id):
            merge_attempts += 1
            # print(f"  尝试合并: 红色状态 {red.id} 与蓝色状态 {blue.id} (第{merge_attempts}次尝试)")

            # 创建用于合并检查的图
            G, state_map = create_graph_for_merging(all_states, root)

            # 尝试合并
            G_prime, is_compatible, equiv_classes = construct_quotient_graph_with_fingerprints(G, red.id, blue.id)

            if is_compatible:
                successful_merges += 1
                print(" ", red.id, "and", blue.id, "merged successfully!")

                # 合并成功，更新自动机
                root = update_automaton_from_merge(root, G_prime, equiv_classes, state_map)
                merged = True

                # 更新状态集
                all_states = collect_all_states(root)
                red_states = {state for state in all_states if state.color == 'red'}
                blue_states = {state for state in all_states if state.color == 'blue'}

                # print(
                #     f"  合并后: {len(all_states)} 个总状态, {len(red_states)} 个红色状态, {len(blue_states)} 个蓝色状态")
                break
            else:
                # print(f"  ✗ 合并失败，指纹不兼容")
                pass

        if not merged:
            # 如果无法合并，将蓝色状态提升为红色
            # print(f"  蓝色状态 {blue.id} 无法与任何红色状态合并，将其提升为红色")
            blue.color = 'red'
            red_states.add(blue)
            blue_states.remove(blue)

        # 更新蓝色状态集
        update_blue_states(red_states, blue_states, all_states)
        print(f"  Red States: {[s.id for s in sorted(red_states, key=lambda x: x.id)]}")
        print(f"  Blue States: {[s.id for s in sorted(blue_states, key=lambda x: x.id)]}")

    # 输出最终统计
    print("\nDB-RPNI completed!")
    print(f"Total number of iterations: {iteration_count}")
    print(f"Number of merger attempts: {merge_attempts}")
    print(f"Number of successful mergers: {successful_merges}")
    print(f"Number of final states: {len(all_states)}")

    return root


# --- 输出 ---
def assign_state_ids_and_dump_json(root: Node, output_file: str) -> None:
    """分配可读的状态ID并输出JSON"""
    print("\nGenerating Mealy Machine...")
    states = {}
    id_mapping = {}  # 内部ID到q0,q1等的映射

    # BFS遍历分配新ID
    all_states = collect_all_states(root)

    # 确保根节点是q0
    id_mapping[root.id] = "q0"

    # 分配其余状态的ID
    next_id = 1
    for state in all_states:
        if state.id not in id_mapping:
            id_mapping[state.id] = f"q{next_id}"
            next_id += 1

    # 构建状态字典
    for state in all_states:
        state_id = id_mapping[state.id]
        states[state_id] = {
            "fingerprint": state.fingerprint,
            "transitions": {symbol: id_mapping[target.id] for symbol, target in state.transitions.items()}
        }

    # 构建DBMM字典
    dbmm = {
        "initial_state": id_mapping[root.id],
        "states": states,
        "state_count": len(states)
    }

    # 输出最终自动机信息
    print(f"The Final Dual Behavior Mealy Machine:")
    print(f"  The Number of States: {dbmm['state_count']}")
    print(f"  The Initial State: {dbmm['initial_state']}")

    print("\n  Detail of States:")
    for state_id, state_info in states.items():
        fp_count = len(state_info["fingerprint"])
        trans_count = len(state_info["transitions"])
        print(f"    {state_id}: {fp_count} Fingerprint, {trans_count} Transitions")

        # 打印部分指纹样例
        if fp_count > 0:
            fp_sample = list(state_info["fingerprint"].items())[:3]
            fp_str = ", ".join([f"'{k}' → '{v}'" for k, v in fp_sample])
            if fp_count > 3:
                fp_str += f", ... (there are another {fp_count - 3} fingerprints)"
            print(f"      Instance: {fp_str}")

        # 打印部分转移样例
        if trans_count > 0:
            trans_sample = list(state_info["transitions"].items())[:3]
            trans_str = ", ".join([f"'{k}' → {v}" for k, v in trans_sample])
            if trans_count > 3:
                trans_str += f", ... (there are another {trans_count - 3} transitions)"
            print(f"      Instance: {trans_str}")

    # 输出到JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dbmm, f, indent=2, ensure_ascii=False)

    print(f"\nThe machine has been saved to: {output_file}")


# --- 主函数 ---
def main(args):
    """主程序入口。"""
    input_file = args.input_file
    output_file = args.output_file

    try:
        print("\n===== Learning DBMM =====")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")

        _, traces = parse_input_file(input_file)
        root = build_prefix_tree(traces)
        minimized_root = blue_fringe_rpni(root)
        assign_state_ids_and_dump_json(minimized_root, output_file)

        print("\n===== Finished =====")

    except FileNotFoundError as e:
        print(f"\n错误：{str(e)}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, TypeError, KeyError, IOError, AssertionError) as e:
        print(f"\n错误: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n发生未知错误: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Learn a Dual-Behavior Mealy Machine using DB-RPNI.")
    parser.add_argument("input_file", help="Path to the input file with trace data.")
    parser.add_argument("output_file", help="Path to the output JSON file for the learned machine.")

    args = parser.parse_args()

    # Reset Node ID counter
    Node._ids = itertools.count(0)

    # Execute main logic directly
    main(args)
