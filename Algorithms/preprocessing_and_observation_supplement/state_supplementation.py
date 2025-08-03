import json
import ast
import sys

# 定义Mealy机类
class MealyMachine:
    def __init__(self):
        self.states = {}
        self.initial_state = None
        self.state_count = 0

    def add_state(self, state_name, fingerprint, transitions):
        self.states[state_name] = {
            'fingerprint': fingerprint,
            'transitions': transitions
        }

    def get_next_state(self, current_state, input_symbol):
        if current_state not in self.states:
            return current_state, None
        
        state_info = self.states[current_state]
        
        # 从input_symbol中提取state,label,action
        parts = input_symbol.split(',')
        if len(parts) != 3:
            print("input_symbol is not valid")
            return current_state, None
            
        state, label, action = parts
        
        # 首先检查是否是转移输入（使用label）
        if label in state_info['transitions']:
            next_state = state_info['transitions'][label]
            return next_state, None
            
        # 然后检查是否是输出输入（使用完整的state,action）
        state_action = f"{state},{action}"
        if state_action in state_info['fingerprint']:
            output = state_info['fingerprint'][state_action]
            return current_state, output
            
        return current_state, None

def load_mealy_machine(json_file):
    mealy_machine = MealyMachine()

    # 从JSON文件中读取Mealy机的结构
    with open(json_file, 'r') as f:
        data = json.load(f)

    mealy_machine.initial_state = data['initial_state']
    mealy_machine.state_count = data['state_count']

    # 添加所有状态
    for state_name, state_data in data['states'].items():
        mealy_machine.add_state(
            state_name,
            state_data['fingerprint'],
            state_data['transitions']
        )

    return mealy_machine

def process_trajectories(mealy_machine, trajectories_file, output_file):
    supplemented_trajectories = []

    with open(trajectories_file, 'r') as f:
        line_number = 0
        for line in f:
            line_number += 1
            line = line.strip()
            if not line:
                continue

            # 解析当前行的轨迹
            try:
                trajectory = ast.literal_eval(line)
            except Exception as e:
                print(f"在解析第{line_number}行时出错：{e}")
                continue

            current_automaton_state = mealy_machine.initial_state
            supplemented_trajectory = []

            for step in trajectory:

                # 提取当前步的信息
                label_state, action, reward = step
                label, state = label_state
                # 创建输入符号
                input_symbol = f"{state},{label},{action}"

                # 获取当前自动机状态
                current_automaton_state, output_symbol = mealy_machine.get_next_state(current_automaton_state, input_symbol)

                # 将自动机状态添加到当前步
                supplemented_step = [[label, f"{state}-{current_automaton_state}"], action, reward]
                supplemented_trajectory.append(supplemented_step)

            supplemented_trajectories.append(supplemented_trajectory)

    # 将补充后的轨迹数据写入输出文件，每个轨迹一行
    with open(output_file, 'w') as f:
        for traj in supplemented_trajectories:
            json_str = json.dumps(traj)
            f.write(json_str + '\n')

    print(f"The result has been written to：{output_file}")

if __name__ == "__main__":
    print("processing")

    if len(sys.argv) > 1:
        trajectories_file = sys.argv[1]
        mealy_json_file = sys.argv[2]
        output_file = sys.argv[3]

    else:
        raise Exception("No input filename")

    # 文件路径
    # mealy_json_file = "mealy_tm.json"
    # trajectories_file = "trajectories.json"
    # output_file =

    # 加载并构建完整的Mealy机
    mealy_machine = load_mealy_machine(mealy_json_file)

    # 处理轨迹并输出结果
    process_trajectories(mealy_machine, trajectories_file, output_file)

    print("finished")
