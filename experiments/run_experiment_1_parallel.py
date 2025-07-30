# experiments/run_experiment_1_parallel.py
import json
import os
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback

# 使用绝对路径导入所有需要的模块
import experiments.config as config
from core.network_generator import create_random_network
from core.graph_transformer import transform_graph, transform_graph_to_directed, transform_graph_for_decode_always, \
    transform_graph_for_forward_always
# from core.algorithm import find_min_cost_feasible_path
from core.algorithm_SPFA_based import find_min_cost_feasible_path
from core.encoding_schemes import SCHEME_145_1_9  # 假设实验一使用 d=5 码
from baselines.greedy_baselines import find_path_decode_always, find_path_forward_always
from experiments.config import get_timestamped_filename


# --- 1. 将单次模拟运行的逻辑封装成一个顶级函数 ---
def run_single_simulation(run_args):
    """
    执行一次独立的、包含所有对比场景的模拟。
    run_args 是一个元组: (run_index, r_theta, schemes_dict, delta)
    返回一个字典，包含 r_theta 和 本次运行的所有结果。
    """
    run_index, r_theta, schemes_dict, delta = run_args
    random.seed(run_index)

    # 为了严格的可复现性
    rng  = np.random.default_rng(run_index)

    try:
        G, super_switches = create_random_network(config.PARAMS["DEFAULT_NUM_NODES"],
                                  avg_degree=config.PARAMS["DEFAULT_AVG_DEGREE"],
                                  p_super_switch=config.PARAMS["DEFAULT_P_SUPER_SWITCH"],
                                  seed=run_index)

        if len(super_switches) < 2:
            return {'r_theta': r_theta,
                    'costs': {"Proposed": None, "Decode-Always": None, "Forward-Always": None}}

        source, dest = rng.choice(super_switches,size=2, replace=False)

        results_this_run = {}

        # Proposed
        g_prime = transform_graph(G, [schemes_dict["Proposed"]])
        cost_p, _ = find_min_cost_feasible_path(g_prime, [schemes_dict["Proposed"]], source, dest, r_theta, delta)
        results_this_run["Proposed"] = cost_p

        # Decode-Always
        g_da = transform_graph_for_decode_always(G, [schemes_dict["Decode-Always"]], source)
        cost_da, _ = find_min_cost_feasible_path(g_da, [schemes_dict["Decode-Always"]], source, dest,
                                                       r_theta, delta)
        results_this_run["Decode-Always"] = cost_da

        # Forward-Always
        g_fa = transform_graph_for_forward_always(G, [schemes_dict["Forward-Always"]], dest)
        cost_fa, _ = find_min_cost_feasible_path(g_fa, [schemes_dict["Forward-Always"]], source, dest, r_theta, delta)
        results_this_run["Forward-Always"] = cost_fa

        return {'r_theta': r_theta, 'costs': results_this_run}
    except Exception as e:
        print(f"一次模拟在 r_theta={r_theta} 时发生错误: {e}")
        traceback.print_exc()
        return {'r_theta': r_theta,
                'costs': {"Proposed": None, "Decode-Always": None, "Forward-Always": None}}


def main():
    print("--- 正在运行实验一：核心算法有效性验证 (并行版) ---")
    print('实验参数:')
    print(json.dumps(config.PARAMS, indent=4))

    # 定义实验场景和它们使用的编码方案
    scenarios = {
        "Proposed": SCHEME_145_1_9,
        "Decode-Always": SCHEME_145_1_9,
        "Forward-Always": SCHEME_145_1_9
    }

    # --- 2. 创建所有任务的参数列表 ---
    tasks = []
    for r_theta in config.PARAMS["ERROR_THRESHOLDS"]:
        delta = config.PARAMS["EPSILON"] * r_theta / config.PARAMS["DEFAULT_NUM_NODES"]
        for i in range(config.PARAMS["NUM_RUNS"]):
            tasks.append((i, r_theta, scenarios, delta))

    print(f"总共需要执行 {len(tasks)} 次独立的模拟运行。")

    # --- 3. 创建进程池并一次性分发所有任务 ---
    num_processes = cpu_count() - 1
    print(f"使用 {num_processes} 个CPU核心进行并行计算...")

    results_list = []
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks), desc="总模拟进度 (Experiment 1)") as pbar:
            for result in pool.imap_unordered(run_single_simulation, tasks):
                results_list.append(result)
                pbar.update(1)

    # --- 3.5 串行 debug 用，不使用 Pool，直接用一个简单的 for 循环 ---
    # with tqdm(total=len(tasks), desc="总模拟进度 (Debug Mode)") as pbar:
    #     for task in tasks:
    #         result = run_single_simulation(task)  # 直接调用任务函数
    #         results_list.append(result)
    #         pbar.update(1)


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
            cost_dict = single_run_result["costs"]
            for name in scenarios:
                cost = cost_dict[name]
                if cost is not None:
                    run_costs[name].append(cost)
                    run_successes[name] += 1

        for name in scenarios:
            avg_cost = np.mean(run_costs[name]) if run_costs[name] else np.nan
            accept_ratio = run_successes[name] / config.PARAMS["NUM_RUNS"]
            all_results[name]['costs'].append(avg_cost)
            all_results[name]['accept_ratios'].append(accept_ratio)

    # --- 5. 保存结果 ---
    output_data = {
        "parameters": config.PARAMS,
        "results": all_results
    }

    results_filename = get_timestamped_filename("experiment_1_results")
    os.makedirs(config.PARAMS["RESULTS_DIR"], exist_ok=True)
    filepath = os.path.join(config.PARAMS["RESULTS_DIR"], results_filename)

    print(f"\n实验完成，正在将结果保存到 {filepath} ...")
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=4)
    print("保存成功。")


if __name__ == "__main__":
    main()