# recorder.py
import json
import matplotlib.pyplot as plt
import os


class Recorder:
    def __init__(self, filename='reward.json'):
        self.filename = filename
        self.rewards = []

        # 如果文件存在，读取现有的数据
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                try:
                    self.rewards = json.load(f)
                except json.JSONDecodeError:
                    self.rewards = []
        else:
            # 如果文件不存在，创建一个空文件
            with open(self.filename, 'w') as f:
                json.dump(self.rewards, f)

        # 初始化绘图
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.rewards, label='Cumulative Reward')
        self.ax.set_xlabel('Trajectory')
        self.ax.set_ylabel('Cumulative Reward')
        self.ax.set_title('Agent Learning Progress')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def record_reward(self, reward):
        """
        记录一个轨迹的累积奖励，并更新文件和绘图。
        :param reward: 当前轨迹的累积奖励
        """
        self.rewards.append(reward)
        # 写入文件
        with open(self.filename, 'w') as f:
            json.dump(self.rewards, f)
        # 更新绘图
        self.line.set_xdata(range(len(self.rewards)))
        self.line.set_ydata(self.rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """
        关闭绘图窗口。
        """
        plt.ioff()
        plt.show()
