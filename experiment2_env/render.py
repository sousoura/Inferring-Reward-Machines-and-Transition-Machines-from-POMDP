import cv2
import numpy as np
import json
import networkx as nx
from collections import defaultdict


class Renderer:
    """基于OpenCV的强化学习环境可视化器"""

    def __init__(self, config_file="env_para.json", cell_size=60, margin=20):
        """
        初始化渲染器

        参数:
            config_file: 环境配置文件路径
            cell_size: 网格单元格大小(像素)
            margin: 边距大小(像素)
        """
        # 加载环境配置
        self.load_config(config_file)

        # 渲染参数
        self.cell_size = cell_size
        self.margin = margin
        self.grid_line_width = 1
        self.agent_radius = int(cell_size * 0.3)
        self.automaton_node_radius = 30

        # 计算网格绘制区域大小
        self.grid_width = (self.env_size + 1) * cell_size + 2 * margin
        self.grid_height = (self.env_size + 1) * cell_size + 2 * margin

        # 计算自动机绘制区域大小
        self.automaton_width = 400
        self.automaton_height = 300

        # 计算总窗口大小
        self.window_width = self.grid_width + self.automaton_width
        self.window_height = max(self.grid_height, self.automaton_height * 2 + 10)

        # 创建颜色映射
        self.colors = self._create_color_map()

        # 生成自动机布局
        self._generate_automaton_layouts()

        # 创建窗口
        self.window_name = "RL Environment Visualization"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

    def load_config(self, config_file):
        """从JSON文件加载环境配置"""
        with open(config_file, 'r') as f:
            config = json.load(f)

        # 环境基本参数
        self.env_size = config["environment"]["size"]

        # 标签系统
        self.labels_count = config["labels"]["count"]
        self.label_mapping = {}
        for pos_str, label in config["labels"]["mapping"].items():
            # 将字符串格式的坐标转换为元组
            pos = eval(pos_str)
            self.label_mapping[pos] = label

        # 转移自动机
        self.transition_states = list(config["transition_automaton"]["transitions"].keys())
        self.transition_transitions = config["transition_automaton"]["transitions"]
        self.impassable_positions = {}
        for state, positions in config["transition_automaton"]["impassable_positions"].items():
            self.impassable_positions[state] = [eval(pos) for pos in positions]

        # 奖励自动机
        self.reward_states = list(config["reward_automaton"]["transitions"].keys()) + [
            config["reward_automaton"]["terminal_state"]]
        self.reward_transitions = config["reward_automaton"]["transitions"]
        self.rewards = config["reward_automaton"]["rewards"]
        self.terminal_state = config["reward_automaton"]["terminal_state"]

    def _create_color_map(self):
        """创建颜色映射"""
        colors = {
            'background': (240, 240, 240),
            'grid_line': (180, 180, 180),
            'agent': (0, 0, 255),
            'impassable': (200, 200, 200),
            'label_text': (0, 0, 0),
            'node_normal': (220, 220, 220),
            'node_current': (100, 200, 100),
            'node_terminal': (200, 100, 100),
            'edge': (100, 100, 100),
            'text': (0, 0, 0)
        }

        # 为标签创建不同的颜色
        label_colors = {}
        for i in range(self.labels_count):
            # 使用HSV色彩空间生成均匀分布的颜色
            hue = int(180 * (i / max(1, self.labels_count - 1)))
            color = cv2.cvtColor(np.uint8([[[hue, 180, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
            # OpenCV使用BGR顺序
            label_colors[i] = (int(color[2]), int(color[1]), int(color[0]))

        colors['labels'] = label_colors
        return colors

    def _generate_automaton_layouts(self):
        """生成自动机的图形布局"""
        # 为转移自动机创建图
        G_transition = nx.DiGraph()
        for state in self.transition_states:
            G_transition.add_node(state)

        # 添加边
        for state, transitions in self.transition_transitions.items():
            for label, next_state in transitions.items():
                G_transition.add_edge(state, next_state, label=label)

        # 使用spring布局生成节点位置
        self.transition_pos = nx.spring_layout(G_transition, seed=42)
        self.transition_edges = defaultdict(list)

        # 整理边信息，用于绘制
        for u, v, data in G_transition.edges(data=True):
            self.transition_edges[(u, v)].append(data['label'])

        # 为奖励自动机创建图
        G_reward = nx.DiGraph()
        for state in self.reward_states:
            G_reward.add_node(state)

        # 添加边
        for state, transitions in self.reward_transitions.items():
            for label, next_state in transitions.items():
                G_reward.add_edge(state, next_state, label=label)

        # 使用spring布局生成节点位置
        self.reward_pos = nx.spring_layout(G_reward, seed=42)
        self.reward_edges = defaultdict(list)

        # 整理边信息，用于绘制
        for u, v, data in G_reward.edges(data=True):
            self.reward_edges[(u, v)].append(data['label'])

    def render(self, agent_pos, transition_state, reward_state):
        """
        渲染环境状态

        参数:
            agent_pos: 智能体当前位置，格式为(x, y)
            transition_state: 当前转移自动机状态
            reward_state: 当前奖励自动机状态
        """
        # 创建画布
        canvas = np.ones((self.window_height, self.window_width, 3), dtype=np.uint8) * 255

        # 绘制环境网格
        self._draw_grid(canvas, agent_pos, transition_state)

        # 绘制转移自动机
        automaton_offset_x = self.grid_width
        automaton_offset_y = 0
        self._draw_automaton(canvas, self.transition_states, self.transition_pos,
                             self.transition_edges, transition_state, None,
                             automaton_offset_x, automaton_offset_y, "Transition Automaton")

        # 绘制奖励自动机
        automaton_offset_y = self.automaton_height + 10
        self._draw_automaton(canvas, self.reward_states, self.reward_pos,
                             self.reward_edges, reward_state, self.terminal_state,
                             automaton_offset_x, automaton_offset_y, "Reward Automaton")

        # 显示画布
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1)

        # 如果按下ESC键，返回True表示退出
        return key == 27

    def _grid_to_pixel(self, grid_x, grid_y):
        """将网格坐标转换为像素坐标"""
        # 完全重写的坐标转换函数
        # 保持X轴不变，但Y轴需要反转（因为屏幕坐标Y轴向下）
        # 这里使用正确的偏移确保所有格子都显示在视图中
        pixel_x = int(self.margin + grid_x * self.cell_size)
        # 注意这里我们从下到上映射Y坐标
        pixel_y = int(self.margin + (self.env_size - grid_y) * self.cell_size)
        return (pixel_x, pixel_y)

    def _draw_grid(self, canvas, agent_pos, transition_state):
        """绘制环境网格和智能体"""
        # 绘制背景
        cv2.rectangle(canvas, (0, 0), (self.grid_width, self.grid_height),
                      self.colors['background'], -1)

        # 获取当前不可通行位置
        impassable = self.impassable_positions.get(transition_state, [])

        # 绘制网格线和单元格
        for x in range(self.env_size + 1):
            for y in range(self.env_size + 1):
                # 计算单元格的左上角和右下角
                left_top = self._grid_to_pixel(x, y)
                right_bottom = self._grid_to_pixel(x + 1, y - 1)

                # 绘制网格线
                cv2.rectangle(canvas, left_top, right_bottom,
                              self.colors['grid_line'], self.grid_line_width)

                # 单元格中心点用于标签和填充
                cell_center_x = (left_top[0] + right_bottom[0]) // 2
                cell_center_y = (left_top[1] + right_bottom[1]) // 2

                # 如果位置有标签，填充颜色并显示标签
                if (x, y) in self.label_mapping:
                    label = self.label_mapping[(x, y)]
                    color = self.colors['labels'][label]

                    # 填充单元格
                    cv2.rectangle(canvas,
                                  (left_top[0] + self.grid_line_width, left_top[1] + self.grid_line_width),
                                  (right_bottom[0] - self.grid_line_width, right_bottom[1] - self.grid_line_width),
                                  color, -1)

                    # 绘制标签文本
                    cv2.putText(canvas, f"L{label}", (cell_center_x - 10, cell_center_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['label_text'], 2)

                # 如果位置不可通行，添加阴影和X标记
                if (x, y) in impassable:
                    # 填充灰色背景
                    cv2.rectangle(canvas,
                                  (left_top[0] + self.grid_line_width, left_top[1] + self.grid_line_width),
                                  (right_bottom[0] - self.grid_line_width, right_bottom[1] - self.grid_line_width),
                                  self.colors['impassable'], -1)

                    # 画X表示不可通行
                    cv2.line(canvas, left_top, right_bottom, (150, 150, 150), 2)
                    cv2.line(canvas, (left_top[0], right_bottom[1]), (right_bottom[0], left_top[1]), (150, 150, 150), 2)

        # 绘制坐标轴标签
        for i in range(self.env_size + 1):
            # X轴标签
            x_label_pos = self._grid_to_pixel(i, -0.5)
            cv2.putText(canvas, str(i), (x_label_pos[0], self.grid_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Y轴标签
            y_label_pos = self._grid_to_pixel(-0.5, i)
            cv2.putText(canvas, str(i), (5, y_label_pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 绘制智能体
        agent_center = self._grid_to_pixel(agent_pos[0] + 0.5, agent_pos[1] - 0.5)
        cv2.circle(canvas, agent_center, self.agent_radius, (0, 0, 255), -1)

        # 显示智能体位置和标签信息
        label = self.label_mapping.get(agent_pos, None)
        label_display = f"L{label}" if label is not None else "None"

        info_text = f"Position: {agent_pos}"
        cv2.putText(canvas, info_text, (10, self.grid_height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        info_text = f"Label: {label_display}"
        cv2.putText(canvas, info_text, (10, self.grid_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        info_text = f"Transition State: {transition_state}"
        cv2.putText(canvas, info_text, (10, self.grid_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def _draw_automaton(self, canvas, states, positions, edges, current_state,
                        terminal_state, offset_x, offset_y, title):
        """绘制自动机图"""
        # 绘制标题
        cv2.putText(canvas, title, (offset_x + 10, offset_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)

        # 计算最大和最小坐标，用于缩放
        min_x = min(pos[0] for pos in positions.values())
        max_x = max(pos[0] for pos in positions.values())
        min_y = min(pos[1] for pos in positions.values())
        max_y = max(pos[1] for pos in positions.values())

        # 缩放因子
        scale_x = (self.automaton_width - 80) / (max_x - min_x if max_x > min_x else 1)
        scale_y = (self.automaton_height - 80) / (max_y - min_y if max_y > min_y else 1)

        # 缩放位置并加入偏移
        scaled_positions = {}
        for state, pos in positions.items():
            x = (pos[0] - min_x) * scale_x + offset_x + 40
            y = (pos[1] - min_y) * scale_y + offset_y + 60
            scaled_positions[state] = (int(x), int(y))

        # 首先绘制边
        for (u, v), labels in edges.items():
            # 获取起点和终点坐标
            start = scaled_positions[u]
            end = scaled_positions[v]

            # 计算方向向量
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx * dx + dy * dy)

            if length > 0:
                # 单位向量
                ux = dx / length
                uy = dy / length

                # 调整起点和终点，避免与节点重叠
                start_adjusted = (int(start[0] + ux * self.automaton_node_radius),
                                  int(start[1] + uy * self.automaton_node_radius))
                end_adjusted = (int(end[0] - ux * self.automaton_node_radius),
                                int(end[1] - uy * self.automaton_node_radius))

                # 绘制边
                cv2.arrowedLine(canvas, start_adjusted, end_adjusted,
                                self.colors['edge'], 1, tipLength=0.2)

                # 绘制标签
                label_text = ",".join(labels)
                label_pos = (int(start[0] + dx * 0.5), int(start[1] + dy * 0.5))
                # 为标签添加背景，使其更易读
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(canvas,
                              (label_pos[0] - 5, label_pos[1] - text_size[1] - 5),
                              (label_pos[0] + text_size[0] + 5, label_pos[1] + 5),
                              (255, 255, 255), -1)
                cv2.putText(canvas, label_text, label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)

        # 然后绘制节点
        for state in states:
            if state in scaled_positions:
                position = scaled_positions[state]

                # 决定节点颜色
                if state == current_state:
                    color = self.colors['node_current']
                elif state == terminal_state:
                    color = self.colors['node_terminal']
                else:
                    color = self.colors['node_normal']

                # 绘制节点
                cv2.circle(canvas, position, self.automaton_node_radius, color, -1)
                cv2.circle(canvas, position, self.automaton_node_radius, (0, 0, 0), 1)

                # 绘制状态名称
                cv2.putText(canvas, state, (position[0] - 15, position[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

    def close(self):
        """关闭窗口"""
        cv2.destroyAllWindows()