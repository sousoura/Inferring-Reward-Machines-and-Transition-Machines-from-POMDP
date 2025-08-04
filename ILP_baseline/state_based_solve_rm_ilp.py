import pulp


def solve_reward_machine_ILP_simplified(traces, K):
    """
    使用简化约束的ILP求解Reward Machine

    输入:
      - traces: List of trajectories. 每条轨迹是一个 [(s, a, r, s_next), ...] 的列表.
      - K: 假设的 Reward Machine 状态个数，即 U = {u1, u2, ..., u_K}.

    简化约束:
      - 使用 L: S → 2^P 而非 L: (S × A × S) → 2^P
      - 强制使用 j ≥ i 约束（上三角矩阵）

    输出:
      - solution: 每条轨迹每步对应的 RM 映射（即 (u_i -> u_j) 的分配情况）。
      - reward_machine: 从 ILP 解中提取的 Reward Machine 结构.
    """
    # 建立 ILP 模型
    prob = pulp.LpProblem("SimplifiedRewardMachineILP", pulp.LpMinimize)

    # 定义决策变量 O[(m, n, i, j)]
    # m: 轨迹索引, n: 轨迹步索引, i,j: RM 状态索引 (从1开始)
    O = {}
    for m, traj in enumerate(traces, 1):  # m 从 1 开始
        for n in range(1, len(traj) + 1):  # n 从 1 开始
            for i in range(1, K + 1):  # i 从 1 开始
                # 只考虑 j ≥ i 的情况（上三角矩阵）
                for j in range(i, K + 1):
                    var_name = f"O_{m}_{n}_{i}_{j}"
                    O[(m, n, i, j)] = pulp.LpVariable(var_name, cat='Binary')

    # 收集所有唯一的目标状态 s'
    S_next = set()
    for traj in traces:
        for s, a, r, s_next in traj:
            S_next.add(s_next)

    # 对每个状态收集所有观察到的奖励
    rewards_by_state = {}
    for s_next in S_next:
        rewards_by_state[s_next] = set()
        for traj in traces:
            for s, a, r, s_next_t in traj:
                if s_next_t == s_next:
                    rewards_by_state[s_next].add(r)

    # 1. 约束：每一步必须唯一地分配到 (u_i, u_j) - 公式(6)
    for m, traj in enumerate(traces, 1):
        for n in range(1, len(traj) + 1):
            prob += pulp.lpSum(O.get((m, n, i, j), 0) for i in range(1, K + 1)
                               for j in range(i, K + 1)) == 1, f"uniqueness_{m}_{n}"

    # 2. 初始状态约束 - 公式(7)
    for m, traj in enumerate(traces, 1):
        # 对于每条轨迹的第一步，必须从u1开始
        prob += pulp.lpSum(O.get((m, 1, 1, j), 0) for j in range(1, K + 1)) == 1, f"initial_state_{m}"

    # 3. 连续性约束 - 公式(8)
    for m, traj in enumerate(traces, 1):
        for n in range(1, len(traj)):  # n 到 len(traj)-1
            for j in range(1, K + 1):
                left_sum = pulp.lpSum(O.get((m, n, i, j), 0) for i in range(1, j + 1))
                right_sum = pulp.lpSum(O.get((m, n + 1, j, j_prime), 0) for j_prime in range(j, K + 1))
                prob += left_sum == right_sum, f"continuity_{m}_{n}_{j}"

    # 4. 定义指示变量 I(s',i,j)
    I_s_ij = {}
    for s_next in S_next:
        for i in range(1, K + 1):
            for j in range(i, K + 1):
                var_name = f"I_{s_next}_{i}_{j}"
                I_s_ij[(s_next, i, j)] = pulp.LpVariable(var_name, cat='Binary')

                # 设置指示变量定义
                sum_term = pulp.LpAffineExpression()
                for m, traj in enumerate(traces, 1):
                    for n in range(1, len(traj) + 1):
                        idx = n - 1  # 转回0索引以访问轨迹
                        if n <= len(traj) and traj[idx][3] == s_next:
                            sum_term += O.get((m, n, i, j), 0)

                # 大M方法实现指示符号
                M = len(traces) * 10  # 足够大的常数
                prob += sum_term <= M * I_s_ij[(s_next, i, j)], f"indicator1_{s_next}_{i}_{j}"
                prob += sum_term >= I_s_ij[(s_next, i, j)], f"indicator2_{s_next}_{i}_{j}"

    # 5. 定义指示变量 I(s',r,i,j)
    I_sr_ij = {}
    for s_next in S_next:
        for r in rewards_by_state[s_next]:
            for i in range(1, K + 1):
                for j in range(i, K + 1):
                    var_name = f"I_{s_next}_{r}_{i}_{j}"
                    I_sr_ij[(s_next, r, i, j)] = pulp.LpVariable(var_name, cat='Binary')

                    # 设置指示变量定义
                    sum_term = pulp.LpAffineExpression()
                    for m, traj in enumerate(traces, 1):
                        for n in range(1, len(traj) + 1):
                            idx = n - 1  # 转回0索引以访问轨迹
                            if n <= len(traj) and traj[idx][2] == r and traj[idx][3] == s_next:
                                sum_term += O.get((m, n, i, j), 0)

                    # 大M方法实现指示符号
                    M = len(traces) * 10  # 足够大的常数
                    prob += sum_term <= M * I_sr_ij[(s_next, r, i, j)], f"reward_indicator1_{s_next}_{r}_{i}_{j}"
                    prob += sum_term >= I_sr_ij[(s_next, r, i, j)], f"reward_indicator2_{s_next}_{r}_{i}_{j}"

    # 6. δᵤ决定性约束
    for s_next in S_next:
        for i in range(1, K + 1):
            prob += pulp.lpSum(I_s_ij.get((s_next, i, j), 0) for j in range(i, K + 1)) <= 1, \
                    f"determinism_{s_next}_{i}"

    # 7. δᵣ决定性约束
    for s_next in S_next:
        for i in range(1, K + 1):
            for j in range(i, K + 1):
                prob += pulp.lpSum(I_sr_ij.get((s_next, r, i, j), 0)
                                   for r in rewards_by_state[s_next]) <= 1, \
                        f"reward_determinism_{s_next}_{i}_{j}"

    # 8. 目标函数：最小化非自环转移
    objective = pulp.lpSum([I_s_ij.get((s_next, i, j), 0)
                            for s_next in S_next
                            for i in range(1, K + 1)
                            for j in range(i, K + 1)
                            if i != j])

    prob += objective, "MinimizeNonSelfTransitions"

    # 求解 ILP
    solver = pulp.PULP_CBC_CMD(msg=True)
    result = prob.solve(solver)
    print("Solver Status:", pulp.LpStatus[result])

    if pulp.LpStatus[result] != "Optimal":
        print("未找到最优解！")
        return None, None

    # 提取 ILP 求解结果
    solution = {}  # solution[m-1] 为第 m 条轨迹的步映射列表
    for m, traj in enumerate(traces, 1):
        solution[m - 1] = []  # 输出结果用0索引
        for n in range(1, len(traj) + 1):
            idx = n - 1  # 转回0索引以访问轨迹
            assigned = None
            for i in range(1, K + 1):
                for j in range(i, K + 1):
                    if (m, n, i, j) in O and pulp.value(O[(m, n, i, j)]) > 0.5:
                        assigned = (i, j)
                        break
                if assigned is not None:
                    break
            solution[m - 1].append({"assignment": assigned, "trace_step": traj[idx]})

    # 构造 Reward Machine 结构
    reward_machine = {
        "states": list(range(1, K + 1)),  # RM 状态 u1, u2, …, uK
        "initial_state": 1,
        "transitions": {}
    }

    # 提取 Reward Machine
    for s_next in S_next:
        for i in range(1, K + 1):
            for j in range(i, K + 1):
                if (s_next, i, j) in I_s_ij and pulp.value(I_s_ij[(s_next, i, j)]) > 0.5:
                    # 找到对应的奖励
                    reward = None
                    for r in rewards_by_state[s_next]:
                        if (s_next, r, i, j) in I_sr_ij and pulp.value(I_sr_ij[(s_next, r, i, j)]) > 0.5:
                            reward = r
                            break

                    # 记录此条转移的实例
                    instances = []
                    for traj in traces:
                        for s_t, a_t, r_t, s_next_t in traj:
                            if s_next_t == s_next and r_t == reward:
                                instances.append((s_t, a_t, r_t, s_next_t))

                    # 对于状态标记，使用None作为a和s的占位符
                    reward_machine["transitions"][(i, None, None, s_next)] = {
                        "next_state": j,
                        "reward": reward,
                        "steps": instances
                    }

    return solution, reward_machine
