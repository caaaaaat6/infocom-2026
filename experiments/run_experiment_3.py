# experiments/run_experiment_3.py
import json
import os
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import repeat

# 使用绝对路径导入所有需要的模块
import config
from core.network_generator import create_random_network
from core.graph_transformer import transform_graph
from core.algorithm import find_min_cost_feasible_path
from core.encoding_schemes import SINGLE_SCHEME_85_1_7  # 实验三统一使用 d=7 码
from baselines.multi_flow_baselines import solve_multi_flow_ilp_minimize_congestion
from config import get_timestamped_filename


def generate_path_pool_for_flow(args):
    """
    [并行任务] 为单个流生成候选路径池。
    在这个简化版本中，我们只为每个流寻找一条最优路径。
    """
    flow, G, schemes, r_theta, delta = args
    s, d = flow

    # 我们需要回溯路径来获取边的列表
    g_prime = transform_graph(G, schemes)

    # 假设 find_min_cost_feasible_path 经过修改，可以返回路径的边列表
    # (cost, error, path_edges) = find_min_cost_feasible_path(...)
    # 这里我们先用一个简化版本
    cost, error = find_min_cost_feasible_path(g_prime, schemes, s, d, r_theta, delta)

    if cost is not None:
        # TODO: 必须实现路径回溯来得到真实的 edge_list
        # 作为一个临时的占位符，我们用 dijkstra 来获取一个路径的边
        try:
            path_nodes = nx.dijkstra_path(G, s, d, weight='cost')
            path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
            path_info = {'cost': cost, 'error': error, 'edge_list': path_edges}
            return (s, d), [path_info]
        except nx.NetworkXNoPath:
            return (s, d), []
    else:
        return (s, d), []


def calculate_max_congestion(G, flows, chosen_paths: dict):
    """一个辅助函数，用于计算给定路由方案的最大拥塞。"""
    if not chosen_paths:
        return 0
    link_loads = {edge: 0 for edge in G.edges()}
    for flow, path_info in chosen_paths.items():
        if path_info:
            for u, v in path_info['edge_list']:
                # 处理无向图的边
                if (u, v) in link_loads:
                    link_loads[(u, v)] += 1
                elif (v, u) in link_loads:
                    link_loads[(v, u)] += 1
    return max(link_loads.values()) if link_loads else 0


def main():
    print("--- 正在运行实验三：网络级多流性能评估 (拥塞对比) ---")

    scenarios = ["Proposed (ILP)", "Greedy-Sequential", "Shortest-Path-First"]
    all_results = {name: {'max_congestion': []} for name in scenarios}

    # 外层循环：遍历并发流的数量
    num_flows_list = config.PARAMS["NUM_FLOWS_LIST"]
    for num_flows in tqdm(num_flows_list, desc="Varying Number of Flows"):

        run_congestions = {name: [] for name in scenarios}

        # 内层循环：进行多次统计运行
        for i in range(config.PARAMS["NUM_RUNS"]):
            G = create_random_network(config.PARAMS["DEFAULT_NUM_NODES"], seed=i)
            nodes = list(G.nodes())
            if len(nodes) < num_flows * 2: continue

            # 生成 num_flows 个不重复的 SD 对
            flows = []
            while len(flows) < num_flows:
                s, d = random.sample(nodes, 2)
                if (s, d) not in flows and (d, s) not in flows:
                    flows.append((s, d))

            # --- 1. 并行生成路径池 (所有算法共用) ---
            path_pool_tasks = list(zip(flows,
                                       repeat(G),
                                       repeat(SINGLE_SCHEME_85_1_7),
                                       repeat(config.DEFAULT_R_THETA),  # 使用一个固定的 r_theta
                                       repeat(config.DEFAULT_DELTA)))  # 使用一个固定的 delta

            path_pools = {}
            with Pool(processes=cpu_count() - 1) as pool:
                for flow_id, paths in pool.imap_unordered(generate_path_pool_for_flow, path_pool_tasks):
                    if paths:
                        path_pools[flow_id] = paths

            # --- 2. 运行 Proposed (ILP) ---
            max_congestion_ilp = solve_multi_flow_ilp_minimize_congestion(path_pools, G, flows)
            if max_congestion_ilp != -1:
                run_congestions["Proposed (ILP)"].append(max_congestion_ilp)

            # --- 3. 运行 Greedy-Sequential 基线 ---
            # 贪心策略：逐个为流选择其候选池中成本最低的路径
            greedy_paths = {}
            G_residual_edges = set(G.edges())
            sorted_flows = sorted(flows, key=lambda f: random.random())  # 随机顺序
            for flow in sorted_flows:
                if flow in path_pools:
                    # (简化) 只考虑池中第一条路径
                    candidate_path = path_pools[flow][0]
                    # 检查资源是否可用
                    if all((u, v) in G_residual_edges or (v, u) in G_residual_edges for u, v in
                           candidate_path['edge_list']):
                        greedy_paths[flow] = candidate_path
                        # 更新残余资源
                        for u, v in candidate_path['edge_list']:
                            if (u, v) in G_residual_edges: G_residual_edges.remove((u, v))
                            if (v, u) in G_residual_edges: G_residual_edges.remove((v, u))
            max_congestion_greedy = calculate_max_congestion(G, flows, greedy_paths)
            run_congestions["Greedy-Sequential"].append(max_congestion_greedy)

            # --- 4. 运行 Shortest-Path-First 基线 ---
            spf_paths = {}
            for s, d in flows:
                try:
                    path_nodes = nx.shortest_path(G, s, d)
                    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
                    spf_paths[(s, d)] = {'edge_list': path_edges}
                except nx.NetworkXNoPath:
                    pass
            max_congestion_spf = calculate_max_congestion(G, flows, spf_paths)
            run_congestions["Shortest-Path-First"].append(max_congestion_spf)

        # 汇总当前 num_flows 的结果
        for name in scenarios:
            avg_congestion = np.mean(run_congestions[name]) if run_congestions[name] else np.nan
            all_results[name]['max_congestion'].append(avg_congestion)

    # --- 5. 保存结果 ---
    output_data = {"parameters": config.PARAMS, "results": all_results}
    results_filename = get_timestamped_filename("experiment_3_results")
    os.makedirs(config.PARAMS["RESULTS_DIR"], exist_ok=True)
    filepath = os.path.join(config.PARAMS["RESULTS_DIR"], results_filename)

    print(f"\n实验完成，正在将结果保存到 {filepath} ...")
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=4)
    print("保存成功。")


if __name__ == "__main__":
    main()