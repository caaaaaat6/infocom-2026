# baselines/multi_flow_baselines.py

from typing import Dict, List, Tuple

import numpy as np
import pulp
import networkx as nx


def solve_multi_flow_ilp_minimize_congestion(path_pools: Dict[tuple, List[dict]], G: nx.Graph,
                                             flows: List[tuple]) -> float:
    """
    使用 ILP (整数线性规划) 求解多流分配问题，以最小化最大链路拥塞。

    这个函数是你 "Proposed (ILP)" 方案的核心。

    参数:
    - path_pools: 一个字典，键是流的(s,d)元组，值是该流的候选路径列表。
                  每个路径是一个包含 'edges' 列表的字典。
    - G: 原始网络图，用于获取所有链路的列表。
    - flows: (s,d) 对的列表，代表所有需要被路由的流。

    返回:
    - float: 计算出的网络中的最小的最大拥塞度。如果无解，返回 -1。
    """
    # 1. 创建问题实例，目标是最小化
    prob = pulp.LpProblem("Minimize_Max_Congestion", pulp.LpMinimize)

    # 2. 创建决策变量
    # x_flow_path = 1 如果流 flow 选择了它的第 path_idx 条候选路径
    x = {}
    for flow_id, paths in path_pools.items():
        if not paths: continue  # 如果一个流没有候选路径，则跳过
        for j, _ in enumerate(paths):
            # 为了让变量名在 PuLP 中唯一且可读
            s, d = flow_id
            x[(flow_id, j)] = pulp.LpVariable(f"x_s{s}d{d}_p{j}", cat=pulp.LpBinary)

    # y = 最大拥塞 (一个连续变量，我们要最小化它)
    y = pulp.LpVariable("y_max_congestion", lowBound=0, cat=pulp.LpContinuous)

    # 3. 设置目标函数: 最小化 y
    prob += y, "Objective_Minimize_Max_Congestion"

    # 4. 添加约束
    # 约束 1: 每个有可行路径的流，必须恰好选择一条路径
    for flow_id in flows:
        if flow_id in path_pools and path_pools[flow_id]:
            prob += pulp.lpSum(
                [x.get((flow_id, j), 0) for j, _ in enumerate(path_pools[flow_id])]) == 1, f"Flow_{flow_id}_must_route"
        # else:
        #     # 如果一个流在路径池中不存在，意味着它没有找到任何可行路径
        #     # 这种情况意味着无法满足所有流，ILP可能会无解
        #     # 在一个更复杂的模型中，可以允许流被拒绝
        #     print(f"警告: 流 {flow_id} 没有可行的候选路径，可能导致ILP无解。")

    # 约束 2: 每条物理链路的拥塞程度必须小于等于 y
    for u, v in G.edges():
        # 累加所有使用了这条边的路径变量
        congestion_on_link = pulp.lpSum([
            x.get((flow_id, j), 0)
            for flow_id, paths in path_pools.items()
            if paths
            for j, path_info in enumerate(paths)
            if (u, v) in path_info.get('edges', []) or (v, u) in path_info.get('edges', [])
        ])
        prob += congestion_on_link <= y, f"Congestion_on_link_{u}_{v}"

    # 5. 求解问题
    # 使用 PuLP 捆绑的 CBC (COIN-OR Branch and Cut) 解算器
    # msg=False 可以关闭冗长的求解过程输出
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
    except pulp.PulpError as e:
        print(f"PuLP 求解器出错: {e}")
        return -1

    # 6. 返回结果
    max_congestion = -1
    if pulp.LpStatus[prob.status] == "Optimal":
        max_congestion = pulp.value(prob.objective)
    elif pulp.LpStatus[prob.status] == "Infeasible":
        print("警告: ILP 问题无解。这可能意味着在给定约束下，无法为所有流找到路径。")
    else:
        print(f"警告: ILP 未找到最优解，状态为: {pulp.LpStatus[prob.status]}")

    return max_congestion


def calculate_max_congestion(G: nx.Graph, chosen_paths: Dict[tuple, dict]) -> int:
    """
    一个通用的辅助函数，用于计算给定路由方案的最大拥塞。
    这是所有多流策略的“裁判”。

    参数:
    - G: 原始网络图。
    - chosen_paths: 一个字典 { (s,d): path_info }, path_info 包含 'edges'。

    返回:
    - int: 网络中的最大链路拥塞度。
    """
    if not chosen_paths:
        return np.nan

    # 初始化每条边的负载为 0
    link_loads = {edge: 0 for edge in G.edges()}

    for flow, path_info in chosen_paths.items():
        if path_info and 'edges' in path_info:
            for u, v in path_info['edges']:
                # 在无向图中，(u,v) 和 (v,u) 是同一条边
                if (u, v) in link_loads:
                    link_loads[(u, v)] += 1
                elif (v, u) in link_loads:
                    link_loads[(v, u)] += 1
                else:
                    # 这种情况理论上不应该发生，如果 G 和 path_info 都正确的话
                    print(f"警告: 路径中包含图中不存在的边 {(u, v)}")

    return max(link_loads.values()) if link_loads else 0