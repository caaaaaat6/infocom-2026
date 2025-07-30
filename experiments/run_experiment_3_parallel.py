# experiments/run_experiment_3_parallel.py
import json
import os
import random
import traceback

import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import repeat
from typing import Dict, List

# --- 使用绝对路径导入 ---
import experiments.config as config
from core.network_generator import create_random_network
from core.graph_transformer import transform_graph
from core.path_pool_algorithm_SPFA_based import find_min_cost_feasible_path
from core.encoding_schemes import SINGLE_SCHEME_145_1_9
from baselines.multi_flow_baselines import (
    solve_multi_flow_lp_randomized_rounding,
    calculate_max_congestion,
    run_greedy_assignment,
    run_shortest_path_first
)
from experiments.config import get_timestamped_filename


# --- 并行任务单元 ---
def generate_path_pool_for_flow(args: tuple):
    """[并行工作函数] 为单个流生成候选路径池。"""
    flow, G, schemes, r_theta, delta, pool_size = args
    s, d = flow
    try:
        path_pool = find_min_cost_feasible_path(
            G_prime=transform_graph(G, schemes),
            schemes=schemes, source=s, dest=d,
            error_threshold=r_theta, delta_eta=delta, pool_size=pool_size
        )
        for path in path_pool:
            path['edge_list'] = path.pop('edges', [])
        return (s, d), path_pool
    except Exception as e:
        traceback.print_exc()
        print(f"为流 {s}->{d} 生成路径池时出错: {e}")
        return (s, d), []


# --- 宏观模拟单元 ---
def run_single_macro_simulation(task_args: dict):
    """
    [并行工作函数] 执行一次完整的、针对特定 `num_flows` 的宏观模拟。
    """
    num_flows = task_args['num_flows']
    run_index = task_args['run_index']
    p_super_switch = task_args['p_super_switch']
    avg_degree = task_args['avg_degree']

    # 生成网络和流
    G, super_switches = create_random_network(num_nodes=config.PARAMS["DEFAULT_NUM_NODES"], p_super_switch=p_super_switch, avg_degree=avg_degree, seed=run_index)
    flows = []
    if len(super_switches) > 1:
        rng = np.random.default_rng(run_index)
        node_indices = list(range(len(super_switches)))
        while len(flows) < num_flows:
            s_idx, d_idx = rng.choice(node_indices, size=2, replace=False)
            s, d = super_switches[s_idx], super_switches[d_idx]
            if (s, d) not in flows and (d, s) not in flows:
                flows.append((s, d))

    # --- 阶段一：路径池生成 ---
    path_pool_tasks = list(zip(
        flows,
        repeat(G),
        repeat(SINGLE_SCHEME_145_1_9),
        repeat(config.PARAMS['DEFAULT_R_THETA']),
        repeat(config.PARAMS['DEFAULT_DELTA']),
        repeat(config.PARAMS.get("PATH_POOL_SIZE", 3))
    ))
    path_pools: Dict[tuple, List[dict]] = dict(map(generate_path_pool_for_flow, path_pool_tasks))
    path_pools = {k: v for k, v in path_pools.items() if v}  # 移除没有路径的流

    # --- 阶段二：应用不同的分配策略 ---
    metrics_results = {}

    # 策略一: Proposed (LP)
    chosen_paths_lp, max_congestion_lp = solve_multi_flow_lp_randomized_rounding(path_pools, G, flows, seed=run_index)
    metrics_results["Proposed (LP)"] = {
        "max_congestion": max_congestion_lp,
        "acceptance_ratio": len(chosen_paths_lp) / num_flows
    }

    # 策略二: Greedy-Assignment
    greedy_paths = run_greedy_assignment(G, path_pools, seed=run_index)
    metrics_results["Greedy-Assignment"] = {
        "max_congestion": calculate_max_congestion(G, greedy_paths),
        "acceptance_ratio": len(greedy_paths) / num_flows
    }

    # 策略三: Shortest-Path-First
    spf_paths = run_shortest_path_first(G, flows, SINGLE_SCHEME_145_1_9[0], config.PARAMS['DEFAULT_R_THETA'], config.PARAMS['DEFAULT_DELTA'])
    metrics_results["Shortest-Path-First"] = {
        "max_congestion": calculate_max_congestion(G, spf_paths),
        "acceptance_ratio": len(spf_paths) / num_flows
    }

    return {'num_flows': num_flows, 'metrics': metrics_results}


# --- 主函数 ---
def main():
    print("--- 正在运行实验三：网络级多流性能评估 (拥塞对比) ---")
    # ... (打印参数) ...
    print('实验参数:')
    print(json.dumps(config.PARAMS, indent=4))

    scenarios = ["Proposed (LP)", "Greedy-Assignment", "Shortest-Path-First"]
    all_results = {name: {'max_congestion': [], 'acceptance_ratio': []} for name in scenarios}

    num_flows_list = config.PARAMS["NUM_FLOWS_LIST"]
    num_runs = config.PARAMS["NUM_RUNS"]
    p_super_switch = config.PARAMS["DEFAULT_P_SUPER_SWITCH"]
    avg_degree = config.PARAMS["DEFAULT_AVG_DEGREE"]

    master_tasks = [{'num_flows': nf, 'run_index': i, 'p_super_switch': p_super_switch, 'avg_degree': avg_degree} for nf in num_flows_list for i in range(num_runs)]

    # --- 并行执行所有宏观模拟 ---
    results_list = []
    with Pool(processes=cpu_count() - 1) as pool:
        with tqdm(total=len(master_tasks), desc="总实验进度 (Experiment 3)") as pbar:
            for result in pool.imap_unordered(run_single_macro_simulation, master_tasks):
                results_list.append(result)
                pbar.update(1)

    # --- 串行 debug 用，不使用 Pool，直接用一个简单的 for 循环 ---
    # with tqdm(total=len(master_tasks), desc="总模拟进度 (Debug Mode)") as pbar:
    #     for task in master_tasks:
    #         result = run_single_macro_simulation(task)  # 直接调用任务函数
    #         results_list.append(result)
    #         pbar.update(1)

    # --- 汇总结果 ---
    print("\n所有模拟运行完成，正在汇总结果...")
    aggregated_results = {nf: {name: {'congestions': [], 'ratios': []} for name in scenarios} for nf in num_flows_list}
    for item in results_list:
        nf, metrics = item['num_flows'], item['metrics']
        for name, values in metrics.items():
            aggregated_results[nf][name]['congestions'].append(values['max_congestion'])
            aggregated_results[nf][name]['ratios'].append(values['acceptance_ratio'])

    for nf in num_flows_list:
        for name in scenarios:
            valid_c = [c for c in aggregated_results[nf][name]['congestions'] if not np.isnan(c)]
            all_results[name]['max_congestion'].append(np.mean(valid_c) if valid_c else np.nan)

            valid_r = [r for r in aggregated_results[nf][name]['ratios'] if not np.isnan(r)]
            all_results[name]['acceptance_ratio'].append(np.mean(valid_r) if valid_r else np.nan)

    # --- 保存结果 ---
    output_data = {"parameters": config.PARAMS, "results": all_results}
    # ... (保存 JSON 文件的逻辑) ...
    results_filename = get_timestamped_filename("experiment_3_results")
    os.makedirs(config.PARAMS["RESULTS_DIR"], exist_ok=True)
    filepath = os.path.join(config.PARAMS["RESULTS_DIR"], results_filename)

    print(f"\n实验完成，正在将结果保存到 {filepath} ...")
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=4)
    print("保存成功。")


if __name__ == "__main__":
    main()