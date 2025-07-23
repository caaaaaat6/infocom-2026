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