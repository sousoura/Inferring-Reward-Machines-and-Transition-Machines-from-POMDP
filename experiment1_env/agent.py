# agent.py
import numpy as np
import random

class QLearningAgent:
    def __init__(self, action_space, observation_space):
        """
        初始化Q-Learning Agent
        :param action_space: 动作空间
        :param observation_space: 观察空间
        """
        self.action_space = action_space
        self.observation_space = observation_space
        
        # Q-Learning参数
        self.alpha = 0.1      # 学习率
        self.gamma = 0.9      # 折扣因子
        self.epsilon = 0.1    # 探索率
        
        # 初始化Q表
        self.q_table = {}
    
    def choose_action(self, observation):
        """
        选择动作
        :param observation: 当前观察
        :return: 选择的动作
        """
        state = self._obs_to_state(observation)
        
        # 探索：以epsilon的概率随机选择动作
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        # 利用：选择Q值最大的动作
        if state in self.q_table:
            return np.argmax(self.q_table[state])
        else:
            # 如果状态未见过，初始化并随机选择
            self.q_table[state] = np.zeros(self.action_space.n)
            return self.action_space.sample()
    
    def learn(self, obs, action, reward, next_obs, done):
        """
        Q-Learning学习
        :param obs: 当前观察
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_obs: 下一个观察
        :param done: 是否结束
        """
        state = self._obs_to_state(obs)
        next_state = self._obs_to_state(next_obs)
        
        # 确保状态在Q表中
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space.n)
        
        # Q-Learning更新公式
        q_predict = self.q_table[state][action]
        
        if done:
            q_target = reward  # 终止状态
        else:
            q_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # 更新Q值
        self.q_table[state][action] += self.alpha * (q_target - q_predict)
    
    def _obs_to_state(self, obs):
        """
        将观察转换为状态表示
        :param obs: 观察数组
        :return: 状态元组
        """
        return tuple(map(int, obs))  # 将浮点数转换为整数以便哈希


class RandomAgent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def choose_action(self, observation):
        return self.action_space.sample()

    def learn(self, obs, action, reward, next_obs, done):
        pass
