import json
import os

class TrajectoryWriter:
    def __init__(self, filename, map_width=3):
        """
        初始化轨迹记录器。
        :param filename: 输出文件的完整路径。
        :param map_width: 地图宽度，用于位置编码。
        """
        self.filename = filename
        self.map_width = map_width
        self.all_trajectories = []  # 在内存中存储所有轨迹
        self.current_trajectory = []

    def encode_position(self, x, y):
        """
        将 (x, y) 位置编码为一个唯一的数字。
        """
        return int(x * self.map_width + y)

    def start_trajectory(self):
        """
        准备记录一条新的轨迹。
        """
        self.current_trajectory = []

    def record_step(self, label, obs, action, reward):
        """
        记录轨迹中的一个时间步。
        """
        x, y = obs[:2]
        x = int(x)
        y = int(y)

        position_number = self.encode_position(x, y)
        timestep_data = [[label, position_number], int(action), float(reward)]
        self.current_trajectory.append(timestep_data)

    def end_trajectory(self):
        """
        结束当前轨迹并将其暂存到内存列表中。
        """
        if self.current_trajectory:
            self.all_trajectories.append(self.current_trajectory)
        # 为下一次调用安全地重置
        self.current_trajectory = []

    def write_to_file(self):
        """
        将内存中所有轨迹一次性写入文件，覆盖任何现有内容。
        """
        print(f"Writing {len(self.all_trajectories)} trajectories to {self.filename}...")
        # 使用 'w' (write) 模式来打开文件，这将自动清空并覆盖文件
        with open(self.filename, 'w') as f:
            for trajectory in self.all_trajectories:
                json_line = json.dumps(trajectory)
                f.write(json_line + '\n')
        print("Write complete.")
