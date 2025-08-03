#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
draw_mealy_machine.py
=====================
从 mealy_tm.json 或 mealy_rm.json 文件中读取 Mealy 机的定义，
并使用 NetworkX 和 Matplotlib 将其可视化为图形表示。

用法:
    python draw_mealy_machine.py tm
    python draw_mealy_machine.py rm
"""

import json
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_mealy_machine(json_file):
    """
    读取 mealy_tm.json 或 mealy_rm.json 文件并返回 Mealy 机的定义。

    参数:
        json_file (str): JSON 文件路径。

    返回:
        dict: Mealy 机的定义。
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_graph(mealy_machine):
    """
    根据 Mealy 机的定义构建 NetworkX 有向图，仅包含第二类输入导致的转移。

    参数:
        mealy_machine (dict): Mealy 机的定义。

    返回:
        NetworkX DiGraph: 构建的有向图。
    """
    G = nx.DiGraph()

    # 添加所有状态为节点
    for state in mealy_machine['states']:
        G.add_node(state)

    # 添加 transitions（第二类输入导致的转移）
    for state_id, state_info in mealy_machine['states'].items():
        for input_sym, target_state in state_info.get('transitions', {}).items():
            label = f"{input_sym}"
            G.add_edge(state_id, target_state, label=label, color='black')

    return G

def draw_graph(G, initial_state):
    """
    使用 NetworkX 和 Matplotlib 绘制有向图。

    参数:
        G (NetworkX DiGraph): 要绘制的有向图。
        initial_state (str): 初始状态，用于特殊标记。

    返回:
        matplotlib.figure.Figure: 绘制的图形对象。
        dict: 节点的位置字典。
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # 使用 spring layout 布局

    # 提取边标签和颜色
    edge_labels = nx.get_edge_attributes(G, 'label')
    edge_colors = [edata['color'] for _, _, edata in G.edges(data=True)]

    # 绘制节点
    node_colors = []
    for node in G.nodes():
        if node == initial_state:
            node_colors.append('lightgreen')  # 初始状态颜色
        else:
            node_colors.append('lightblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, edgecolors='black', ax=ax)

    # 绘制边
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color=edge_colors, ax=ax)

    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # 绘制边标签，调整 label_pos 至 0.75
    try:
        # 尝试使用 label_pos 参数（NetworkX 2.5+ 支持）
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', label_pos=0.75, ax=ax)
    except TypeError:
        # 如果 NetworkX 版本不支持 label_pos，使用手动计算位置
        for (u, v), label in edge_labels.items():
            x = pos[u][0] + 0.75 * (pos[v][0] - pos[u][0])
            y = pos[u][1] + 0.75 * (pos[v][1] - pos[u][1])
            ax.text(x, y, label, fontsize=9, color='red')

    ax.axis('off')
    plt.tight_layout()

    return fig, pos

def display_fingerprints(mealy_machine, treeview):
    """
    在 Tkinter 的 Treeview 中显示每个状态的指纹（第一类输入）。

    参数:
        mealy_machine (dict): Mealy 机的定义。
        treeview (tkinter.ttk.Treeview): 用于显示指纹的 Treeview 组件。
    """
    for state_id, state_info in mealy_machine['states'].items():
        fingerprints = state_info.get('fingerprint', {})
        if fingerprints:
            for input_sym, output_sym in fingerprints.items():
                treeview.insert('', 'end', values=(state_id, input_sym, output_sym))
        else:
            treeview.insert('', 'end', values=(state_id, '-', '-'))

def save_graph_as_image(fig, filename='mealy_machine.png'):
    """
    将绘制的图形保存为图片文件。

    参数:
        fig (matplotlib.figure.Figure): 绘制的图形对象。
        filename (str): 输出文件名。
    """
    # 打开文件保存对话框
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
    )
    if file_path:
        fig.savefig(file_path)
        messagebox.showinfo("保存成功", f"图形已保存为 {file_path}")

def create_gui(mealy_machine):
    """
    创建并运行 Tkinter GUI，用于显示 Mealy 机图形和指纹列表。

    参数:
        mealy_machine (dict): Mealy 机的定义。
    """
    root = tk.Tk()
    root.title("Mealy Machine Visualization")
    root.geometry("1600x800")

    # 创建主框架
    main_frame = ttk.Frame(root)
    main_frame.pack(fill='both', expand=True, padx=10, pady=10)

    # 分割为左右两部分：左边为图形，右边为指纹列表
    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side='left', fill='both', expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side='right', fill='y')

    # 绘制图形
    G = build_graph(mealy_machine)
    initial_state = mealy_machine['initial_state']
    fig, pos = draw_graph(G, initial_state)

    # 在左边嵌入 Matplotlib 图形
    canvas = FigureCanvasTkAgg(fig, master=left_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # 添加保存按钮
    save_button = ttk.Button(left_frame, text="保存为图片", command=lambda: save_graph_as_image(fig))
    save_button.pack(pady=10)

    # 在右边创建指纹列表
    fingerprint_label = ttk.Label(right_frame, text="Fingerprints", font=('Arial', 14, 'bold'))
    fingerprint_label.pack(pady=5)

    columns = ('State', 'Input Symbol', 'Output Symbol')
    tree = ttk.Treeview(right_frame, columns=columns, show='headings')
    tree.heading('State', text='State')
    tree.heading('Input Symbol', text='Input Symbol')
    tree.heading('Output Symbol', text='Output Symbol')

    # 设置列宽
    tree.column('State', width=100, anchor='center')
    tree.column('Input Symbol', width=150, anchor='center')
    tree.column('Output Symbol', width=150, anchor='center')

    # 添加滚动条
    scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    tree.pack(fill='both', expand=True)

    # 填充指纹数据
    display_fingerprints(mealy_machine, tree)

    # 运行 GUI
    root.mainloop()

def main():
    # 修改命令行参数处理
    if len(sys.argv) > 1:
        input_json = sys.argv[1]
    else:
        sys.exit(1)

    try:
        mealy_machine = load_mealy_machine(input_json)
        create_gui(mealy_machine)
    except FileNotFoundError:
        print(f"[ERROR] 找不到文件: {input_json}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERROR] 解析 JSON 文件失败: {input_json}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
