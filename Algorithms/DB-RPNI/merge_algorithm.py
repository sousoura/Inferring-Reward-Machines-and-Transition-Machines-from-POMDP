class Graph:
    def __init__(self, vertices=None, initial_vertex=None, symbols=None):
        self.vertices = vertices or set()
        self.initial_vertex = initial_vertex
        self.symbols = symbols or set()
        self.transitions = {}
        self.fingerprints = {}  # 节点指纹

        for vertex in self.vertices:
            self.add_vertex(vertex, {})

    def add_vertex(self, v, fingerprint=None):
        self.vertices.add(v)
        self.fingerprints[v] = fingerprint or {}  # 默认为空指纹

    def add_transition(self, v, symbol, target):
        if v not in self.transitions:
            self.transitions[v] = {}
        self.transitions[v][symbol] = target

    def get_successor(self, v, symbol):
        if v in self.transitions and symbol in self.transitions[v]:
            return self.transitions[v][symbol]
        return None

    def get_fingerprint(self, v):
        return self.fingerprints.get(v, {})

    def __str__(self):
        result = []
        print("alpha_inputs:", sorted(self.symbols))
        for v in sorted(self.vertices):
            line = f"{v} (fingerprint: {self.get_fingerprint(v)})"
            for symbol in sorted(self.symbols):
                succ = self.get_successor(v, symbol)
                line += f" {succ if succ is not None else 'n'}"
            result.append(line)
        return "\n".join(result)


def are_fingerprints_compatible(fp1, fp2):
    """检查两个指纹是否兼容"""
    for key in set(fp1.keys()) & set(fp2.keys()):
        if fp1[key] != fp2[key]:
            return False
    return True


def merge_fingerprints(fingerprints):
    """合并多个兼容的指纹"""
    result = {}
    for fp in fingerprints:
        for key, value in fp.items():
            if key in result and result[key] != value:
                raise ValueError("试图合并不兼容的指纹")
            result[key] = value
    return result


def construct_quotient_graph_with_fingerprints(G, v1, v2):
    fp1 = G.get_fingerprint(v1)
    fp2 = G.get_fingerprint(v2)

    if not are_fingerprints_compatible(fp1, fp2):
        return None, False, None

    parent = {v: v for v in G.vertices}  # 初始化并查集
    rank = {v: 0 for v in G.vertices}  # 用于并查集的路径压缩
    merged_fingerprints = {v: G.get_fingerprint(v).copy() for v in G.vertices}  # 每个等价类的合并指纹

    def find(x):
        """查找并查集中x的代表元素(带路径压缩)"""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        """合并x和y所在的等价类，如果兼容则返回True"""
        root_x = find(x)
        root_y = find(y)

        if root_x == root_y:
            return True  # 已经在同一等价类中

        # 检查指纹兼容性
        if not are_fingerprints_compatible(merged_fingerprints[root_x], merged_fingerprints[root_y]):
            # print(f"  ✗ 指纹兼容性检查失败: {root_x} 和 {root_y} 不兼容")
            return False

        # 合并指纹
        merged_fp = merge_fingerprints([merged_fingerprints[root_x], merged_fingerprints[root_y]])

        if rank[root_x] < rank[root_y]:
            v_father, v_son = root_y, root_x
        else:
            v_father, v_son = root_x, root_y
            if rank[v_son] == rank[v_father]:
                rank[root_x] += 1

        twinning(v_father, v_son)

        parent[v_son] = v_father
        merged_fingerprints[v_father] = merged_fp

        return True

    def add_pair_into_worklist(v_1, v_2, worklist, processed_pairs):
        pair = (v_1, v_2) if v_1 < v_2 else (v_2, v_1)
        if pair not in processed_pairs:
            worklist.append(pair)
            processed_pairs.add(pair)

    def twinning(v_father, v_son):
        for symbol in set(all_successors.get(v_son, {}).keys()) - set(all_successors.get(v_father, {}).keys()):
            all_successors[v_father][symbol] = all_successors[v_son][symbol]

        return True

    all_successors = {}
    for v in G.vertices:
        all_successors[v] = {}
        for symbol in G.symbols:
            succ = G.get_successor(v, symbol)
            if succ is not None:
                all_successors[v][symbol] = succ

    worklist = [(v1, v2)]  # 初始工作列表
    processed_pairs = set([(v1, v2)])  # 已处理的节点对，避免重复处理

    iteration = 0
    congruence_checks = 0
    congruence_merges = 0

    if not union(v1, v2):
        return None, False, None

    while worklist:
        iteration += 1
        u, v = worklist.pop(0)

        common_symbols = set(all_successors.get(u, {}).keys()) & set(all_successors.get(v, {}).keys())

        for symbol in common_symbols:
            # 获取原始后继
            u_succ_orig = all_successors[u][symbol]
            v_succ_orig = all_successors[v][symbol]

            # 获取当前等价类代表元
            u_succ = find(u_succ_orig)
            v_succ = find(v_succ_orig)

            congruence_checks += 1

            if u_succ == v_succ:
                continue  # 相同代表元，无需合并

            if union(u_succ, v_succ):
                congruence_merges += 1

                add_pair_into_worklist(u_succ, v_succ, worklist, processed_pairs)
            else:
                return None, False, None

    # 第3步: 构造商图
    G_prime = Graph()
    G_prime.symbols = G.symbols.copy()

    # 收集等价类
    equiv_classes = {}
    for v in G.vertices:
        repr_v = find(v)
        if repr_v not in equiv_classes:
            equiv_classes[repr_v] = []
        equiv_classes[repr_v].append(v)

    # 添加节点和其指纹
    for repr_v in equiv_classes:
        G_prime.add_vertex(repr_v, merged_fingerprints[repr_v])

    # 设置初始节点
    G_prime.initial_vertex = find(G.initial_vertex)

    # 添加转移
    transition_count = 0
    for repr_v, members in equiv_classes.items():
        for symbol in G.symbols:
            # 只需检查任一成员的转移，因为同一等价类中的成员在同一符号下的后继属于同一等价类
            for v in members:
                succ = G.get_successor(v, symbol)
                if succ is not None:
                    G_prime.add_transition(repr_v, symbol, find(succ))
                    transition_count += 1
                    break

    # print(f"  商图构建完成, 共 {len(G_prime.vertices)} 个节点, {transition_count} 个转移")
    # print(f"  合并成功!")

    return G_prime, True, equiv_classes


if __name__ == "__main__":
    # 测试代码
    G = Graph(vertices={n for n in range(27)}, initial_vertex=1, symbols={'m', 'p'})

    # 添加节点和指纹
    G.add_vertex(0, fingerprint={"color": "red", "size": "large"})
    G.add_vertex(13, fingerprint={"color": "yet", "shape": "circle"})

    # 添加转移
    G.add_transition(0, 'm', 1)
    G.add_transition(0, 'p', 3)
    G.add_transition(1, 'm', 2)
    G.add_transition(1, 'p', 10)
    G.add_transition(3, 'm', 4)
    G.add_transition(3, 'p', 12)
    G.add_transition(2, 'm', 8)
    G.add_transition(2, 'p', 7)
    G.add_transition(10, 'm', 26)
    G.add_transition(10, 'p', 11)
    G.add_transition(4, 'm', 5)
    G.add_transition(4, 'p', 16)
    G.add_transition(12, 'm', 22)
    G.add_transition(12, 'p', 13)
    G.add_transition(8, 'm', 18)
    G.add_transition(8, 'p', 9)
    G.add_transition(7, 'm', 21)
    G.add_transition(7, 'p', 19)
    G.add_transition(11, 'm', 14)
    G.add_transition(5, 'p', 6)
    G.add_transition(16, 'p', 17)
    G.add_transition(22, 'm', 23)
    G.add_transition(13, 'm', 15)
    G.add_transition(19, 'm', 20)
    G.add_transition(23, 'm', 24)
    G.add_transition(24, 'm', 25)

    print("Original Graph G:")
    print(G)

    # 构造商图
    G_prime, is_compatible, equiv_classes = construct_quotient_graph_with_fingerprints(G, 0, 3)

    if is_compatible:
        print("\nQuotient Graph G':")
        print(G_prime)

        print("\nEquivalent Class:")
        for repr_v, members in equiv_classes.items():
            print(f"{repr_v}: {members}")
    else:
        print("\nNodes are not compatible to construct a quotient graph")