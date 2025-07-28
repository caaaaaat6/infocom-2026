# experiments/run_experiment_2.py
import json
import os
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# 使用绝对路径导入所有需要的模块
import experiments.config as config
from core.network_generator import create_random_network
from core.graph_transformer import transform_graph
from core.algorithm_SPFA_based import find_min_cost_feasible_path
from core.encoding_schemes import (
    SINGLE_SCHEME_13_1_3,
    SINGLE_SCHEME_85_1_7,
    SINGLE_SCHEME_145_1_9,
    MULTI_SCHEME_PORTFOLIO_FOR_EXP2 as MULTI_SCHEME_PORTFOLIO
)
from experiments.config import get_timestamped_filename


# --- 1. 将单次模拟运行的逻辑封装成一个顶级函数 ---
def run_single_simulation(run_args):
    """
    执行一次独立的、包含所有对比场景的模拟。
    run_args 是一个元组: (run_index, r_theta, scenarios, delta)
    返回一个字典，包含这次运行的所有结果。
    """
    run_index, r_theta, scenarios, delta = run_args

    try:
        # 每次运行使用不同的随机种子，确保独立性
        G, super_switches = create_random_network(config.PARAMS["DEFAULT_NUM_NODES"],
                                  avg_degree=config.PARAMS["DEFAULT_AVG_DEGREE"],
                                  p_super_switch=config.PARAMS["DEFAULT_P_SUPER_SWITCH"],
                                  seed=run_index)

        if len(super_switches) < 2:
            return {name: None for name in scenarios}

        source, dest = random.sample(super_switches, 2)

        # 在同一次运行中，为所有场景找到路径
        results_this_run = {}
        for name, schemes in scenarios.items():
            G_prime = transform_graph(G, schemes)
            cost, _ = find_min_cost_feasible_path(
                G_prime, schemes, source, dest, r_theta, delta
            )
            results_this_run[name] = cost

        return results_this_run
    except Exception as e:
        print(f"一次模拟在 r_theta={r_theta} 时发生错误: {e}")
        return {name: None for name in scenarios}


def main():
    print("--- 正在运行实验二：多编码方案选择优势 (并行版) ---")
    print('Params:')
    print(json.dumps(config.PARAMS, indent=4))

    # 定义实验场景
    scenarios = {
        "Proposed (Multi-Scheme d=3,7,9)": MULTI_SCHEME_PORTFOLIO,
        "Single-Scheme (d=3)": SINGLE_SCHEME_13_1_3,
        "Single-Scheme (d=7)": SINGLE_SCHEME_85_1_7,
        "Single-Scheme (d=9)": SINGLE_SCHEME_145_1_9
    }

    # --- 2. 创建所有任务的参数列表 ---
    tasks = []
    for r_theta in config.PARAMS["ERROR_THRESHOLDS"]:
        delta = config.PARAMS["EPSILON"] * r_theta / config.PARAMS["DEFAULT_NUM_NODES"]
        for i in range(config.PARAMS["NUM_RUNS"]):
            # 每个任务是一个元组，包含了运行一次模拟所需的所有参数
            tasks.append((i, r_theta, scenarios, delta))

    print(f"总共需要执行 {len(tasks)} 次独立的模拟运行。")

    # --- 3. 创建进程池并一次性分发所有任务 ---
    num_processes = cpu_count() // 2
    print(f"使用 {num_processes} 个CPU核心进行并行计算...")

    results_list = []
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks), desc="总模拟进度 (Experiment 2)") as pbar:
            # imap_unordered 会在每个任务完成时立即返回结果
            for result in pool.imap_unordered(run_single_simulation, tasks):
                results_list.append(result)
                pbar.update(1)

    # --- 4. 在所有任务完成后，对结果进行后处理和汇总 ---
    print("\n所有模拟运行完成，正在汇总结果...")

    # 初始化结果存储结构
    all_results = {name: {'costs': [], 'accept_ratios': []} for name in scenarios}

    aggregated_by_r = {r: [] for r in config.PARAMS["ERROR_THRESHOLDS"]}
    for task, result in zip(tasks, results_list):
        r_theta = task[1]
        aggregated_by_r[r_theta].append(result)

    for r_theta in config.PARAMS["ERROR_THRESHOLDS"]:
        run_costs = {name: [] for name in scenarios}
        run_successes = {name: 0 for name in scenarios}

        list_of_results_for_r = aggregated_by_r[r_theta]

        for single_run_result in list_of_results_for_r:
            for name in scenarios:
                cost = single_run_result[name]
                if cost is not None:
                    run_costs[name].append(cost)
                    run_successes[name] += 1

        for name in scenarios:
            avg_cost = np.mean(run_costs[name]) if run_costs[name] else np.nan
            accept_ratio = run_successes[name] / config.PARAMS["NUM_RUNS"]
            all_results[name]['costs'].append(avg_cost)
            all_results[name]['accept_ratios'].append(accept_ratio)

    # --- 5. 保存结果 ---
    # 修正：直接保存所有参数
    output_data = {
        "parameters": config.PARAMS,
        "results": all_results
    }

    results_filename = get_timestamped_filename("experiment_2_results")
    os.makedirs(config.PARAMS["RESULTS_DIR"], exist_ok=True)
    filepath = os.path.join(config.PARAMS["RESULTS_DIR"], results_filename)

    print(f"\n实验完成，正在将结果保存到 {filepath} ...")
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=4)
    print("保存成功。")


if __name__ == "__main__":
    main()