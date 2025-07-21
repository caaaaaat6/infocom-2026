# experiments/run_experiment_3.py
import json
import os
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import repeat
from typing import Dict, List

# 使用绝对路径导入
import config
from core.graph_transformer import transform_graph
from core.network_generator import create_random_network
# 假设你已经把路径池算法放到了 algorithm.py 中
from core.path_pool_algorithm import find_min_cost_feasible_path
# 假设实验三统一使用 d=7 的码
from core.encoding_schemes import SINGLE_SCHEME_145_1_9
# 导入 ILP 求解器和 Greedy 基线
from baselines.multi_flow_baselines import solve_multi_flow_ilp_minimize_congestion, calculate_max_congestion
from config import get_timestamped_filename


# --- 并行任务单元 ---
def generate_path_pool_for_flow(args):
    """
    [并行任务] 为单个流生成候选路径池。
    这个函数会被多个子进程并行调用。
    """
    # 解包传入的参数
    flow, G, schemes, r_theta, delta, pool_size = args
    s, d = flow

    # 调用你的路径池生成算法
    # 注意：这里需要传入 G_original, 即 G
    g_prime = transform_graph(G, schemes)
    path_pool = find_min_cost_feasible_path(
        g_prime, schemes, s, d, r_theta, delta, pool_size=pool_size
    )

    # 返回流的标识符和它对应的路径池
    return (s, d), path_pool


def main():
    print("--- 正在运行实验三：网络级多流性能评估 (拥塞对比) ---")

    scenarios = ["Proposed (ILP)", "Greedy-Sequential", "Shortest-Path-First"]
    all_results = {name: {'max_congestion': []} for name in scenarios}

    # 从配置文件读取实验参数
    num_flows_list = config.PARAMS["NUM_FLOWS_LIST"]
    num_runs = config.PARAMS["NUM_RUNS"]
    path_pool_size = config.PARAMS.get("PATH_POOL_SIZE", 3)  # 在config中定义路径池大小 M

    # --- 1. 计算总的迭代次数 (外层*中层) ---
    total_iterations = len(num_flows_list) * num_runs

    # --- 2. 创建一个单一的、手动的 tqdm 进度条 ---
    with tqdm(total=total_iterations, desc="总实验进度 (Experiment 3)") as pbar:
        # 外层循环：遍历并发流的数量
        for num_flows in num_flows_list:

            run_congestions = {name: [] for name in scenarios}

            # 内层循环：进行多次统计运行
            for i in range(num_runs):
                # --- 动态更新后缀信息 ---
                pbar.set_postfix(m=num_flows, run=f'{i + 1}/{num_runs}')

                G, super_switches = create_random_network(config.PARAMS["DEFAULT_NUM_NODES"], seed=i)
                nodes = list(G.nodes())

                # 生成 num_flows 个不重复的 SD 对
                flows = []
                if len(nodes) > 1:
                    while len(flows) < num_flows:
                        s, d = random.sample(super_switches, 2)
                        if (s, d) not in flows and (d, s) not in flows:
                            flows.append((s, d))

                # --- 阶段一：并行生成路径池 (所有算法共用) ---
                path_pool_tasks = list(zip(
                    flows,
                    repeat(G),
                    repeat(SINGLE_SCHEME_145_1_9),
                    repeat(config.PARAMS['DEFAULT_R_THETA']),
                    repeat(config.PARAMS['DEFAULT_DELTA']),
                    repeat(path_pool_size)
                ))

                path_pools: Dict[tuple, List[dict]] = {}
                # 使用 with 语句确保进程池被正确关闭
                with Pool(processes=cpu_count() - 1) as pool:
                    # 注意：这里我们不再用 tqdm 包裹 pool.imap_unordered
                    # 因为我们只想看到总进度
                    results_iterator = pool.imap_unordered(generate_path_pool_for_flow, path_pool_tasks)

                    for flow_id, paths in pool.imap_unordered(generate_path_pool_for_flow, path_pool_tasks):
                        if paths:  # 只添加找到了路径的流
                            path_pools[flow_id] = paths

                # --- 阶段二：串行求解和评估 ---

                # --- 运行 Proposed (ILP) ---
                max_congestion_ilp = solve_multi_flow_ilp_minimize_congestion(path_pools, G, flows)
                if max_congestion_ilp != -1:
                    run_congestions["Proposed (ILP)"].append(max_congestion_ilp)

                # --- 运行 Greedy-Sequential 基线 ---
                greedy_paths = {}
                # 随机打乱处理顺序
                sorted_flows = sorted(list(path_pools.keys()), key=lambda f: random.random())
                for flow in sorted_flows:
                    # 贪心地选择池中成本最低的路径
                    best_path = min(path_pools[flow], key=lambda p: p['cost'])
                    # 在这个简化的贪心模型中，我们不考虑资源冲突，只看分配后的结果
                    greedy_paths[flow] = best_path
                max_congestion_greedy = calculate_max_congestion(G, greedy_paths)
                run_congestions["Greedy-Sequential"].append(max_congestion_greedy)

                # --- 运行 Shortest-Path-First 基线 ---
                spf_paths = {}
                for s, d in flows:
                    try:
                        path_nodes = nx.shortest_path(G, s, d, weight='cost')  # 按成本最短
                        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
                        spf_paths[(s, d)] = {'edge_list': path_edges}
                    except nx.NetworkXNoPath:
                        pass
                max_congestion_spf = calculate_max_congestion(G, spf_paths)
                run_congestions["Shortest-Path-First"].append(max_congestion_spf)

                # --- 3. 在中层循环的末尾，手动更新总进度条 ---
                pbar.update(1)

            # 汇总当前 num_flows 的结果
            for name in scenarios:
                valid_congestions = [c for c in run_congestions[name] if not np.isnan(c) and c != 0]
                avg_congestion = np.mean(valid_congestions) if valid_congestions else np.nan
                all_results[name]['max_congestion'].append(avg_congestion)

    # --- 保存结果 ---
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