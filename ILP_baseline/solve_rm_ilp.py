# solve_rm_ilp.py

import pulp


def solve_reward_machine_ILP(traces, K):
    """
    输入:
      - traces: List of trajectories. 每条轨迹是一个 [(s, a, r, s_next), ...] 的列表.
      - K: 假设的 Reward Machine 状态个数，即 U = {u1, u2, ..., u_K}.

    输出:
      - solution: 每条轨迹每步对应的 RM 映射（即 (u_i -> u_j) 的分配情况）。
      - reward_machine: 从 ILP 解中提取的 Reward Machine 结构.
    """
    # 建立 ILP 模型
    prob = pulp.LpProblem("RewardMachineILP", pulp.LpMinimize)

    # 重要：这里我们遵循论文使用1-索引，而不是0-索引
    # 定义决策变量 O[(m, n, i, j)]
    # m: 轨迹索引, n: 轨迹步索引, i,j: RM 状态索引 (从1开始)
    O = {}
    for m, traj in enumerate(traces, 1):  # m 从 1 开始
        for n in range(1, len(traj) + 1):  # n 从 1 开始
            for i in range(1, K + 1):  # i 从 1 开始
                for j in range(1, K + 1):  # j 从 1 开始
                    var_name = f"O_{m}_{n}_{i}_{j}"
                    O[(m, n, i, j)] = pulp.LpVariable(var_name, cat='Binary')

    # 收集所有唯一的 (s, a, s') 转移
    SAS_o = set()
    for traj in traces:
        for s, a, r, s_next in traj:
            SAS_o.add((s, a, s_next))

    # 对每个 (s, a, s') 收集所有观察到的奖励
    rewards = {}
    for sas in SAS_o:
        rewards[sas] = set()
        for traj in traces:
            for s, a, r, s_next in traj:
                if (s, a, s_next) == sas:
                    rewards[sas].add(r)

    # 1. 约束：每一步必须唯一地分配到 (u_i, u_j) - 公式(6)
    for m, traj in enumerate(traces, 1):
        for n in range(1, len(traj) + 1):
            prob += pulp.lpSum(
                O[(m, n, i, j)] for i in range(1, K + 1) for j in range(1, K + 1)) == 1, f"uniqueness_{m}_{n}"

    # 2. 初始状态约束 - 公式(7)
    for m, traj in enumerate(traces, 1):
        # 对于每条轨迹的第一步，必须从u1开始
        prob += pulp.lpSum(O[(m, 1, 1, j)] for j in range(1, K + 1)) == 1, f"initial_state_{m}"

    # 3. 连续性约束 - 公式(8)
    for m, traj in enumerate(traces, 1):
        for n in range(1, len(traj)):  # n 到 len(traj)-1
            for j in range(1, K + 1):
                prob += (pulp.lpSum(O[(m, n, i, j)] for i in range(1, K + 1)) ==
                         pulp.lpSum(O[(m, n + 1, j, j_prime)] for j_prime in range(1, K + 1))), \
                        f"continuity_{m}_{n}_{j}"

    # 定义指示变量 I(s,a,s',i,j) - 公式(7)
    I_sas_ij = {}
    for s, a, s_next in SAS_o:
        for i in range(1, K + 1):
            for j in range(1, K + 1):
                var_name = f"I_{s}_{a}_{s_next}_{i}_{j}"
                I_sas_ij[(s, a, s_next, i, j)] = pulp.LpVariable(var_name, cat='Binary')

                # 设置指示变量定义 - 公式(7)
                # I(s,a,s',i,j) = [∑(m,n,(s,a,s')=(s_m_n,a_m_n,s_m_n+1)) O(m,n,i,j) ≥ 1]
                sum_term = pulp.LpAffineExpression()
                for m, traj in enumerate(traces, 1):
                    for n in range(1, len(traj) + 1):
                        idx = n - 1  # 转回0索引以访问轨迹
                        if n <= len(traj) and (traj[idx][0], traj[idx][1], traj[idx][3]) == (s, a, s_next):
                            sum_term += O[(m, n, i, j)]

                # 大M方法实现指示符号: sum_term ≥ 1 => I = 1, 否则 I = 0
                M = len(traces) * 10  # 足够大的常数
                prob += sum_term <= M * I_sas_ij[(s, a, s_next, i, j)], f"indicator1_{s}_{a}_{s_next}_{i}_{j}"
                prob += sum_term >= I_sas_ij[(s, a, s_next, i, j)], f"indicator2_{s}_{a}_{s_next}_{i}_{j}"

    # 定义指示变量 I(s,a,r,s',i,j) - 公式(8)
    I_sasr_ij = {}
    for s, a, s_next in SAS_o:
        for r in rewards[(s, a, s_next)]:
            for i in range(1, K + 1):
                for j in range(1, K + 1):
                    var_name = f"I_{s}_{a}_{r}_{s_next}_{i}_{j}"
                    I_sasr_ij[(s, a, r, s_next, i, j)] = pulp.LpVariable(var_name, cat='Binary')

                    # 设置指示变量定义 - 公式(8)
                    sum_term = pulp.LpAffineExpression()
                    for m, traj in enumerate(traces, 1):
                        for n in range(1, len(traj) + 1):
                            idx = n - 1  # 转回0索引以访问轨迹
                            if n <= len(traj) and (traj[idx][0], traj[idx][1], traj[idx][2], traj[idx][3]) == (
                            s, a, r, s_next):
                                sum_term += O[(m, n, i, j)]

                    # 大M方法实现指示符号
                    M = len(traces) * 10  # 足够大的常数
                    prob += sum_term <= M * I_sasr_ij[
                        (s, a, r, s_next, i, j)], f"reward_indicator1_{s}_{a}_{r}_{s_next}_{i}_{j}"
                    prob += sum_term >= I_sasr_ij[
                        (s, a, r, s_next, i, j)], f"reward_indicator2_{s}_{a}_{r}_{s_next}_{i}_{j}"

    # 4. δᵤ决定性约束 - 公式(5)
    for s, a, s_next in SAS_o:
        for i in range(1, K + 1):
            prob += pulp.lpSum(I_sas_ij[(s, a, s_next, i, j)] for j in range(1, K + 1)) <= 1, \
                    f"determinism_{s}_{a}_{s_next}_{i}"

    # 5. δᵣ决定性约束 - 公式(6)
    for s, a, s_next in SAS_o:
        for i in range(1, K + 1):
            for j in range(1, K + 1):
                prob += pulp.lpSum(I_sasr_ij[(s, a, r, s_next, i, j)] for r in rewards[(s, a, s_next)]) <= 1, \
                        f"reward_determinism_{s}_{a}_{s_next}_{i}_{j}"

    # 6. 目标函数：最小化非自环转移 - 公式(1)
    objective = pulp.lpSum(I_sas_ij[(s, a, s_next, i, j)]
                           for s, a, s_next in SAS_o
                           for i in range(1, K + 1)
                           for j in range(1, K + 1) if i != j)
    prob += objective, "MinimizeNonSelfTransitions"

    # 求解 ILP
    solver = pulp.PULP_CBC_CMD(msg=True)
    result = prob.solve(solver)
    print("Solver Status:", pulp.LpStatus[result])

    if pulp.LpStatus[result] != "Optimal":
        print("未找到最优解！")
        return None

    # 提取 ILP 求解结果
    solution = {}  # solution[m-1] 为第 m 条轨迹的步映射列表
    for m, traj in enumerate(traces, 1):
        solution[m - 1] = []  # 输出结果用0索引
        for n in range(1, len(traj) + 1):
            idx = n - 1  # 转回0索引以访问轨迹
            assigned = None
            for i in range(1, K + 1):
                for j in range(1, K + 1):
                    if pulp.value(O[(m, n, i, j)]) > 0.5:
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

    for s, a, s_next in SAS_o:
        for i in range(1, K + 1):
            for j in range(1, K + 1):
                if pulp.value(I_sas_ij.get((s, a, s_next, i, j), 0)) > 0.5:
                    # 找到对应的奖励
                    reward = None
                    for r in rewards[(s, a, s_next)]:
                        if pulp.value(I_sasr_ij.get((s, a, r, s_next, i, j), 0)) > 0.5:
                            reward = r
                            break

                    # 记录此条转移的实例
                    instances = []
                    for traj in traces:
                        for s_t, a_t, r_t, s_next_t in traj:
                            if (s_t, a_t, s_next_t) == (s, a, s_next) and r_t == reward:
                                instances.append((s_t, a_t, r_t, s_next_t))

                    reward_machine["transitions"][(i, s, a, s_next)] = {
                        "next_state": j,
                        "reward": reward,
                        "steps": instances
                    }

    return solution, reward_machine
