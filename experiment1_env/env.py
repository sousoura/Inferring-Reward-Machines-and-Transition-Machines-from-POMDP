import json
import os
import gym
from gym import spaces
import numpy as np

from agent import QLearningAgent
from render import MapRenderer


class SimpleEnv(gym.Env):
    """
    支持3×3和5×5的网格环境，并可根据外部配置文件定义不同的任务流（状态机）。
    """

    def __init__(self, size=3, fsm_type=3, config_path="experiment1_env/environment_config.txt"):
        """
        :param size: 地图大小，可选 3 或 5
        :param fsm_type: 任务流类型，可选 3、5、7
        :param config_path: 存放地图与任务流配置的文件路径
        """
        super(SimpleEnv, self).__init__()

        # 读取配置文件
        self.config = self._load_config(config_path)

        # 检查 size 合法性
        if str(size) not in self.config["maps"]:
            raise ValueError(f"Didn't find the definition of size={size}")
        self.size = size

        # 初始化地图标签
        self.labels = self._parse_labels(str(size))

        # 从 flows 中取对应 size 下的 fsm_type
        if str(size) not in self.config["flows"]:
            raise ValueError(f"Didn't find size={size}'s TA definition")

        if str(fsm_type) not in self.config["flows"][str(size)]:
            raise ValueError(f"Didn't find fsm_type={fsm_type}'s TA definition")

        self.fsm_conf = self.config["flows"][str(size)][str(fsm_type)]

        # 记录一下自动机的当前状态
        self.fsm_state = self.fsm_conf["start_state"]

        # gym 动作空间：0=上, 1=下, 2=左, 3=右
        self.action_space = spaces.Discrete(4)

        # FSM 状态数量
        num_states = len(self.fsm_conf["states"])
        # 观测空间：[x, y, fsm_state_idx]
        low = np.array([0, 0, 0], dtype=np.float32)
        high = np.array([self.size - 1, self.size - 1, num_states - 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 初始化智能体在地图上的位置
        self.agent_pos = None

        # 渲染器
        self.renderer = None

        # 重置
        self.reset()

    def _load_config(self, config_path):
        """
        读取并解析 JSON 配置文件
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Didn't find config file: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config

    def _parse_labels(self, size_str):
        """
        根据配置文件里 maps 中的 'labels'，构造位置->label 的字典
        """
        label_dict = {}
        label_data = self.config["maps"][size_str]["labels"]
        # label_data 是形如 { "(0, 0)": "A", "(0, 2)": "B", ... }
        for pos_str, lbl in label_data.items():
            # pos_str 类似 "(0, 0)"
            # 去掉括号，然后逗号分隔
            # 或者使用更安全的方式：eval() 或正则。这里演示简单做法：
            cleaned = pos_str.strip("() ")
            row_str, col_str = cleaned.split(",")
            row = int(row_str)
            col = int(col_str)
            label_dict[(row, col)] = lbl
        return label_dict

    def reset(self):
        """
        重置环境：重置智能体位置以及自动机状态
        """
        # 随机位置
        self.agent_pos = [
            np.random.randint(0, self.size),
            np.random.randint(0, self.size)
        ]
        # 重置自动机状态
        self.fsm_state = self.fsm_conf["start_state"]

        return self._get_obs()

    def _get_obs(self):
        """
        观测由 (agent_row, agent_col, fsm_state_idx) 组成
        """
        state_idx = self.fsm_conf["states"].index(self.fsm_state)
        return np.array([self.agent_pos[0], self.agent_pos[1], state_idx], dtype=np.float32)

    def get_label(self, position=None):
        """
        根据坐标获取地图上的 label（如果没有则返回 "None"）
        """
        if position is None:
            position = tuple(self.agent_pos)
        return self.labels.get(tuple(position), "None")

    def step(self, action):
        """
        执行动作:
        1. 检查是否阻塞
        2. 如果不阻塞则移动
        3. 检查标签 -> 尝试 FSM 转移
        4. 判断是否到达终止条件
        5. 计算奖励
        """
        # 1. 检查是否阻塞
        blocked_actions = self.fsm_conf["blocked_actions"][self.fsm_state]
        if action in blocked_actions:
            # 阻塞时，可自行定义处理方式：这里示例为原地不动并给惩罚
            reward = -1.0
            done = False
            info = {"blocked": True, "label": "None"}
            return self._get_obs(), reward, done, info

        # 2. 如果不阻塞，则更新位置
        new_pos = self.agent_pos.copy()
        if action == 0:  # 上
            new_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # 下
            new_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # 左
            new_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # 右
            new_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        self.agent_pos = new_pos

        # 3. 查看当前标签，并根据 (当前状态, label) 决定下一个状态
        label = self.get_label()
        cur_state = self.fsm_state
        next_state = self._get_next_state(cur_state, label)
        self.fsm_state = next_state

        # 4. 若已在 terminal_state，再次碰到 end_label 则结束
        done = False
        reward = 0.0
        if self.fsm_state == self.fsm_conf["terminal_state"]:
            done = True
            reward = 1.0  # 示例：踩到任何 label 额外给 0.5

        info = {
            "label": label,
            "old_state": cur_state,
            "new_state": self.fsm_state
        }

        return self._get_obs(), reward, done, info

    def _get_next_state(self, current_state, label):
        """
        根据 transitions 配置，查找 (current_state, label) 是否有定义
        """
        transitions = self.fsm_conf["transitions"]  # 比如 {"S0,A": "S1", "S1,B": "S2"}
        key = f"{current_state},{label}"
        return transitions.get(key, current_state)  # 若没定义则留在当前状态

    def render(self, mode='human'):
        """
        若需要可接入渲染器，这里做一个简写
        """
        if self.renderer is None:
            self.renderer = MapRenderer(self)
        self.renderer.render(self.agent_pos)

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None


def main():
    """
    演示：加载配置文件 environment_config.txt 后，
         构造一个 3×3 环境、FSM类型=5 的示例，然后随机选动作进行体验
    """
    env = SimpleEnv(size=3, fsm_type=5, config_path="experiment1_env/environment_config.txt")

    obs = env.reset()
    print("初始观测:", obs)

    done = False
    step_count = 0
    while not done and step_count < 20:  # 防止无限循环
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step={step_count}, Action={action}, Obs={obs}, Reward={reward}, Done={done}, Info={info}")
        step_count += 1
        # env.render()

    env.close()


if __name__ == "__main__":
    main()
