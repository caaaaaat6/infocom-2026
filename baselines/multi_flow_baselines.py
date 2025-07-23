# baselines/multi_flow_baselines.py

import random
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import pulp

# 导入核心模块
from baselines.utils import check_path_feasibility  # 假设你创建了这个辅助函数
from core.encoding_schemes import EncodingScheme


def calculate_max_congestion(G: nx.Graph, chosen_paths: Dict[tuple, dict]) -> float:
    """
    一个通用的辅助函数，用于计算给定路由方案的最大拥塞。
    """
    if not chosen_paths:
        return np.nan  # 如果没有路由任何流，拥塞度是无效的

    link_loads = {tuple(sorted(edge)): 0 for edge in G.edges()}

    for flow, path_info in chosen_paths.items():
        if path_info and 'edge_list' in path_info:
            for u, v in path_info['edge_list']:
                edge = tuple(sorted((u, v)))
                if edge in link_loads:
                    link_loads[edge] += 1

    return max(link_loads.values()) if link_loads else 0


def solve_multi_flow_ilp_minimize_congestion(path_pools: Dict[tuple, List[dict]], G: nx.Graph, flows: List[tuple]) -> \
Tuple[Dict, float]:
    """
    使用 ILP 求解多流分配问题，以最小化最大链路拥塞。
    返回: (选择的路径字典, 最小的最大拥塞度)
    """
    prob = pulp.LpProblem("Minimize_Max_Congestion", pulp.LpMinimize)

    # 决策变量
    x = {}  # x_flow_path
    for flow_id, paths in path_pools.items():
        if not paths: continue
        for j, _ in enumerate(paths):
            s, d = flow_id
            x[(flow_id, j)] = pulp.LpVariable(f"x_s{s}d{d}_p{j}", cat=pulp.LpBinary)
    y = pulp.LpVariable("y_max_congestion", lowBound=0, cat=pulp.LpContinuous)

    # 目标函数
    prob += y, "Objective_Minimize_Max_Congestion"

    # 约束
    for flow_id in flows:
        if flow_id in path_pools and path_pools[flow_id]:
            prob += pulp.lpSum(
                [x.get((flow_id, j), 0) for j, _ in enumerate(path_pools[flow_id])]) == 1, f"Flow_{flow_id}_must_route"

    for u, v in G.edges():
        edge = tuple(sorted((u, v)))
        congestion_on_link = pulp.lpSum([
            x.get((flow_id, j), 0)
            for flow_id, paths in path_pools.items()
            if paths
            for j, path_info in enumerate(paths)
            if tuple(sorted((u, v))) in [tuple(sorted(e)) for e in path_info.get('edge_list', [])]
        ])
        prob += congestion_on_link <= y, f"Congestion_on_link_{edge[0]}_{edge[1]}"

    # 求解
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 结果提取
    chosen_paths = {}
    max_congestion = np.nan

    if pulp.LpStatus[prob.status] == "Optimal":
        max_congestion = pulp.value(prob.objective)
        for flow_id, paths in path_pools.items():
            if not paths: continue
            for j, path_info in enumerate(paths):
                var = x.get((flow_id, j))
                if var and pulp.value(var) == 1:
                    chosen_paths[flow_id] = path_info
                    break

    return chosen_paths, np.nan if max_congestion == 0.0 else max_congestion


def solve_multi_flow_lp_randomized_rounding(path_pools: Dict[tuple, List[dict]], G: nx.Graph, flows: List[tuple],
                                            seed: int) -> Tuple[Dict, float]:
    """
    使用 LP 松弛 + 随机化取整，来近似求解多流分配问题。
    返回: (选择的路径字典, 该方案的最大拥塞度)
    """
    # 1. 创建 LP 问题实例
    prob = pulp.LpProblem("Minimize_Max_Congestion_LP", pulp.LpMinimize)
    rng = np.random.default_rng(seed)

    # 2. 创建决策变量 (现在是 0 到 1 之间的连续变量)
    x = {}
    for flow_id, paths in path_pools.items():
        if not paths: continue
        for j, _ in enumerate(paths):
            s, d = flow_id
            # --- 核心修改: 变量是连续的 ---
            x[(flow_id, j)] = pulp.LpVariable(f"x_s{s}d{d}_p{j}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
    y = pulp.LpVariable("y_max_congestion", lowBound=0, cat=pulp.LpContinuous)

    # 3. 目标函数和约束 (与 ILP 完全相同！)
    prob += y, "Objective_Minimize_Max_Congestion"
    for flow_id in flows:
        if flow_id in path_pools and path_pools[flow_id]:
            prob += pulp.lpSum(
                [x.get((flow_id, j), 0) for j, _ in enumerate(path_pools[flow_id])]) == 1, f"Flow_{flow_id}_must_route"

    for u, v in G.edges():
        edge = tuple(sorted((u, v)))
        congestion_on_link = pulp.lpSum([
            x.get((flow_id, j), 0)
            for flow_id, paths in path_pools.items()
            if paths
            for j, path_info in enumerate(paths)
            if tuple(sorted((u, v))) in [tuple(sorted(e)) for e in path_info.get('edge_list', [])]
        ])
        prob += congestion_on_link <= y, f"Congestion_on_link_{edge[0]}_{edge[1]}"

    # 4. 求解 LP 问题 (这会非常快)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 5. LP 求解失败处理
    if pulp.LpStatus[prob.status] != "Optimal":
        print(f"警告: LP 未找到最优解，状态为: {pulp.LpStatus[prob.status]}")
        return {}, np.nan

    # --- 6. 随机化取整 (Randomized Rounding) ---
    chosen_paths = {}
    # LP 的解 x_ij 代表了选择路径 j 的“概率”
    lp_solution = {var.name: var.varValue for var in prob.variables() if var.name.startswith('x_')}

    for flow_id in flows:
        if flow_id not in path_pools or not path_pools[flow_id]:
            continue

        candidate_paths = path_pools[flow_id]
        probabilities = []
        for j, _ in enumerate(candidate_paths):
            s, d = flow_id
            var_name = f"x_s{s}d{d}_p{j}"
            probabilities.append(lp_solution.get(var_name, 0))

        # 确保概率总和为 1 (由于浮点数精度问题，需要归一化)
        prob_sum = sum(probabilities)
        if prob_sum > 1e-6:  # 避免除零
            normalized_probs = [p / prob_sum for p in probabilities]

            # --- 核心: 按概率随机选择一条路径 ---
            chosen_path_idx = rng.choice(len(candidate_paths), p=normalized_probs)
            chosen_paths[flow_id] = candidate_paths[chosen_path_idx]

    # 7. 计算最终整数解的拥塞度
    final_max_congestion = calculate_max_congestion(G, chosen_paths)

    return chosen_paths, final_max_congestion


def run_greedy_assignment(G: nx.Graph, path_pools: Dict[tuple, List[dict]], seed: int) -> Dict:
    """
    Greedy-Assignment 基线策略。
    """
    greedy_paths = {}
    rng = np.random.default_rng(seed)
    available_flows = list(path_pools.keys())
    shuffled_flows = rng.permutation(available_flows)

    # --- 核心修改：将 NumPy 数组转换回元组 ---
    for flow_array in shuffled_flows:
        # 将 array([s, d]) 转换回元组 (s, d)
        flow = tuple(flow_array)

        pool = path_pools[flow]
        if pool:
            best_path_for_this_flow = min(pool, key=lambda p: p.get('cost', float('inf')))
            greedy_paths[flow] = best_path_for_this_flow

    return greedy_paths


def run_shortest_path_first(G: nx.Graph, flows: List[tuple], scheme: EncodingScheme, r_theta: float,
                            delta: float) -> Dict:
    """
    Shortest-Path-First (SPF) 基线策略，带可行性检查。
    """
    spf_paths = {}

    for s, d in flows:
        try:
            path_nodes = nx.dijkstra_path(G, s, d, weight='cost')

            # 对这条最短路进行QEC可行性检查
            if check_path_feasibility(path_nodes, G, scheme, r_theta):
                path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
                spf_paths[(s, d)] = {'edge_list': path_edges, 'nodes': path_nodes}
        except nx.NetworkXNoPath:
            pass

    return spf_paths