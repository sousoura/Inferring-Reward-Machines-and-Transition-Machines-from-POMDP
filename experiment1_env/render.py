# render.py
import numpy as np
import cv2

class MapRenderer:
    def __init__(self, env):
        """
        初始化渲染器
        :param env: SimpleEnv环境实例
        """
        self.env = env
        self.cell_size = 100  # 每个格子的像素大小
        self.grid_size = env.size
        self.window_size = self.grid_size * self.cell_size
        
        # 颜色定义 (BGR格式)
        self.colors = {
            'background': (255, 255, 255),  # 白色背景
            'grid': (200, 200, 200),        # 灰色网格线
            'agent': (0, 0, 255),           # 红色智能体
            'A': (255, 0, 0),               # 蓝色标签A
            'B': (0, 255, 0),               # 绿色标签B
            'C': (0, 255, 255),             # 黄色标签C
            'D': (255, 0, 255),             # 紫色标签D
            'E': (255, 128, 0)              # 浅蓝色标签E
        }
        
    def render(self, agent_pos):
        """
        渲染环境
        :param agent_pos: 智能体位置 [x, y]
        """
        # 创建白色背景图像
        image = np.ones((self.window_size, self.window_size, 3), dtype=np.uint8) * 255
        
        # 绘制网格线
        for i in range(1, self.grid_size):
            # 水平线
            cv2.line(image, 
                     (0, i * self.cell_size), 
                     (self.window_size, i * self.cell_size), 
                     self.colors['grid'], 1)
            # 垂直线
            cv2.line(image, 
                     (i * self.cell_size, 0), 
                     (i * self.cell_size, self.window_size), 
                     self.colors['grid'], 1)
        
        # 绘制标签
        for pos, label in self.env.labels.items():
            center_x = int((pos[1] + 0.5) * self.cell_size)
            center_y = int((pos[0] + 0.5) * self.cell_size)
            
            # 绘制标签背景圆圈
            cv2.circle(image, (center_x, center_y), 
                      int(self.cell_size * 0.3), self.colors[label], -1)
            
            # 绘制标签文本
            cv2.putText(image, label, 
                       (center_x - 10, center_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 绘制智能体
        agent_center_x = int((agent_pos[1] + 0.5) * self.cell_size)
        agent_center_y = int((agent_pos[0] + 0.5) * self.cell_size)
        cv2.circle(image, (agent_center_x, agent_center_y), 
                  int(self.cell_size * 0.2), self.colors['agent'], -1)

        # 显示图像
        cv2.imshow("SimpleEnv", image)
        cv2.waitKey(1)  # 等待100毫秒
    
    def close(self):
        """关闭所有窗口"""
        cv2.destroyAllWindows()
