import random
import gym
import numpy as np
import json
import os # <-- 导入 os 模块
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from render import Renderer


class BigEnv(gym.Env):
    """
    BigEnv: 一个2D离散空间的强化学习环境，具有基于自动机的转移和奖励逻辑
    """

    def __init__(self, config_file="env_para.json"):
        """
        初始化环境

        Args:
            config_file: 包含环境参数的JSON文件名 (应位于项目根目录)
        """
        super(BigEnv, self).__init__()

        # --- 路径处理 ---
        # 构建配置文件的完整路径，假定它在项目的根目录
        project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
        full_config_path = os.path.join(project_root, config_file)

        if not os.path.exists(full_config_path):
            raise FileNotFoundError(f"环境配置文件未找到: {full_config_path}")

        # 加载环境配置
        self.load_config(full_config_path)

        # 设置动作空间 (上、下、左、右)
        self.action_space = spaces.Discrete(4)

        # 构建状态到索引的映射
        self.transition_state_to_idx = {state: i for i, state in enumerate(self.transition_transitions.keys())}
        self.reward_state_to_idx = {state: i for i, state in enumerate(self.reward_transitions.keys())}
        if self.terminal_state not in self.reward_state_to_idx:
            self.reward_state_to_idx[self.terminal_state] = len(self.reward_state_to_idx)

        # 反向映射
        self.idx_to_transition_state = {i: state for state, i in self.transition_state_to_idx.items()}
        self.idx_to_reward_state = {i: state for state, i in self.reward_state_to_idx.items()}

        # 设置观察空间
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.MultiDiscrete([self.env_size + 1, self.env_size + 1]),  # 位置坐标
            "transition_state": spaces.Discrete(len(self.transition_state_to_idx)),  # 转移自动机状态
            "reward_state": spaces.Discrete(len(self.reward_state_to_idx))  # 奖励自动机状态
        })

        # 初始化Agent位置、自动机状态等
        self.agent_pos = None
        self.transition_state = None
        self.reward_state = None

        # 用于渲染的变量
        self.fig = None
        self.ax = None

        # 用于绘图的颜色映射
        self.label_colors = {}
        for i in range(self.labels_count):
            self.label_colors[i] = plt.cm.jet(i / max(1, self.labels_count - 1))

        # 初始化环境状态
        self.reset()

        self.renderer = Renderer()

    def load_config(self, config_path):
        """
        从JSON文件加载环境配置
        """
        print(f"从 {config_path} 加载环境配置...")
        with open(config_path, 'r') as f:
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
        self.transition_states_count = config["transition_automaton"]["states_count"]
        self.transition_transitions = {}
        for state, transitions in config["transition_automaton"]["transitions"].items():
            self.transition_transitions[state] = transitions

        self.impassable_positions = {}
        for state, positions in config["transition_automaton"]["impassable_positions"].items():
            self.impassable_positions[state] = [eval(pos) for pos in positions]

        # 奖励自动机
        self.reward_states_count = config["reward_automaton"]["states_count"]
        self.reward_transitions = {}
        for state, transitions in config["reward_automaton"]["transitions"].items():
            self.reward_transitions[state] = transitions

        self.rewards = {}
        for state, rewards in config["reward_automaton"]["rewards"].items():
            self.rewards[state] = rewards

        self.terminal_state = config["reward_automaton"]["terminal_state"]

    def reset(self):
        """
        重置环境状态

        Returns:
            observation: 初始观察
        """
        # 随机初始化Agent的位置
        self.agent_pos = (random.randint(0, self.env_size // 2) * 2, random.randint(0, self.env_size // 2) * 2)
        # 初始化自动机状态
        self.transition_state = list(self.transition_transitions.keys())[0]
        self.reward_state = list(self.reward_transitions.keys())[0]

        return self._get_observation()

    def step(self, action):
        """
        执行一个动作并返回下一个状态等信息
        """
        curr_x, curr_y = self.agent_pos

        if action == 0:  # 上
            new_pos = (curr_x, min(curr_y + 1, self.env_size))
        elif action == 1:  # 右
            new_pos = (min(curr_x + 1, self.env_size), curr_y)
        elif action == 2:  # 下
            new_pos = (curr_x, max(curr_y - 1, 0))
        elif action == 3:  # 左
            new_pos = (max(curr_x - 1, 0), curr_y)
        else:
            raise ValueError(f"无效的动作: {action}")

        if new_pos in self.impassable_positions.get(self.transition_state, []):
            new_pos = self.agent_pos

        self.agent_pos = new_pos
        label = self.get_label(self.agent_pos)

        reward = 0
        if label != "None":
            reward = float(self.rewards.get(self.reward_state, {}).get(str(label), 0))

        self._update_automaton_states(label)
        done = (self.reward_state == self.terminal_state)
        observation = self._get_observation()
        info = {"label": label}

        return observation, reward, done, info

    def _update_automaton_states(self, label):
        """
        根据当前标签更新自动机状态
        """
        if label != "None":
            label_str = str(label)
            if label_str in self.transition_transitions.get(self.transition_state, {}):
                self.transition_state = self.transition_transitions[self.transition_state][label_str]

            if label_str in self.reward_transitions.get(self.reward_state, {}):
                self.reward_state = self.reward_transitions[self.reward_state][label_str]

    def _get_observation(self):
        """
        获取当前环境的观察
        """
        return {
            "agent_pos": np.array(self.agent_pos),
            "transition_state": self.transition_state_to_idx[self.transition_state],
            "reward_state": self.reward_state_to_idx[self.reward_state]
        }

    def get_label(self, position):
        """
        获取指定位置的标签
        """
        return self.label_mapping.get(position, "None")

    def render(self, mode='human'):
        if mode != 'human':
            return
        return self.renderer.render(
            self.agent_pos,
            self.transition_state,
            self.reward_state
        )

    def close(self):
        """
        关闭环境
        """
        if self.renderer:
            self.renderer.close()
