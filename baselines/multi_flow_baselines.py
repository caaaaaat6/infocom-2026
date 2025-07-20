# baselines/multi_flow_baselines.py
import pulp
from typing import Dict, List


def solve_multi_flow_ilp_minimize_congestion(path_pools: Dict[tuple, List[dict]], G, flows: List[tuple]):
    """
    使用 ILP 求解多流分配问题，以最小化最大链路拥塞。
    - path_pools: 每个流的候选路径池。
    - G: 原始网络图。
    - flows: (s,d) 对的列表。
    """
    # 1. 创建问题实例，目标是最小化
    prob = pulp.LpProblem("Multi_Flow_Minimize_Congestion", pulp.LpMinimize)

    # 2. 创建决策变量
    # x_i_j = 1 如果流 i (由(s,d)标识) 选择了它的第 j 条路径
    x = {}
    flow_map = {flow: i for i, flow in enumerate(flows)}  # 给每个流一个整数索引

    for flow_id, paths in path_pools.items():
        if not paths: continue
        for j, _ in enumerate(paths):
            x[(flow_id, j)] = pulp.LpVariable(f"x_{flow_map[flow_id]}_{j}", cat=pulp.LpBinary)

    # y = 最大拥塞 (一个连续变量，我们要最小化它)
    y = pulp.LpVariable("max_congestion", lowBound=0, cat=pulp.LpContinuous)

    # 3. 设置目标函数: 最小化 y
    prob += y

    # 4. 添加约束
    # 约束1: 每个流 *必须* 选择一条路径 (假设所有流都必须被路由)
    for flow_id in flows:
        if flow_id in path_pools and path_pools[flow_id]:
            prob += pulp.lpSum([x[(flow_id, j)] for j, _ in enumerate(path_pools[flow_id])]) == 1
        else:
            # 如果一个流没有可行的候选路径，打印一个警告
            # 这种情况意味着网络无法满足所有请求
            print(f"警告: 流 {flow_id} 没有可行的候选路径，无法被路由。")

    # 约束2: 每条物理链路的拥塞程度必须小于等于 y
    for u, v in G.edges():
        congestion_on_link = pulp.lpSum([x[(flow_id, j)]
                                         for flow_id, paths in path_pools.items()
                                         if paths
                                         for j, path_info in enumerate(paths)
                                         if (u, v) in path_info['edge_list'] or (v, u) in path_info['edge_list']])
        prob += congestion_on_link <= y

    # 5. 求解问题
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 6. 返回结果
    max_congestion = -1
    if pulp.LpStatus[prob.status] == "Optimal":
        max_congestion = pulp.value(prob.objective)

    return max_congestion