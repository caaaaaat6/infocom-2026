# experiments/run_experiment_4.py
import json
import os
import random
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import repeat

# 使用绝对路径导入
import experiments.config as config
from core.network_generator import create_random_network
from core.graph_transformer import transform_graph
from core.algorithm_SPFA_based import find_min_cost_feasible_path
from core.encoding_schemes import MULTI_SCHEME_PORTFOLIO  # 导入包含多种编码的列表
from experiments.config import get_timestamped_filename


# --- 并行任务单元 ---
def run_single_scalability_test(args: tuple):
    """
    [并行工作函数] 执行一次独立的模拟，并返回算法的运行时间。
    """
    run_index, num_nodes, schemes_to_use = args
    rng = np.random.default_rng(seed=run_index)

    try:
        # --- 准备工作 ---
        G, super_switches = create_random_network(
            num_nodes,
            seed=run_index,
            avg_degree=config.PARAMS["DEFAULT_AVG_DEGREE"],
            p_super_switch=config.PARAMS["DEFAULT_P_SUPER_SWITCH"]
        )

        if len(super_switches) < 2:
            return None  # 无法进行测试

        source, dest = rng.choice(super_switches, 2)

        # --- 计时开始 ---
        start_time = time.perf_counter()

        # --- 核心算法调用 ---
        find_min_cost_feasible_path(
            G_prime=transform_graph(G, schemes_to_use),
            schemes=schemes_to_use,
            source=source,
            dest=dest,
            error_threshold=config.PARAMS['DEFAULT_R_THETA'],
            delta=config.PARAMS['DEFAULT_DELTA'],
        )

        # --- 计时结束 ---
        end_time = time.perf_counter()

        return {'num_nodes': num_nodes, 'run_time': end_time - start_time}

    except Exception as e:
        print(f"一次模拟在 num_nodes={num_nodes} 时出错: {e}")
        return None


def run_exp4a_vs_network_size():
    """实验 4a: 测试运行时间随网络规模的变化。"""
    print("\n--- 正在运行实验 4a: 运行时间 vs. 网络规模 ---")

    # 自变量: 网络节点数
    node_sizes = config.PARAMS.get("NODE_SIZES_FOR_SCALABILITY", [20, 80, 160, 240, 320, 400])
    num_runs = config.PARAMS["NUM_RUNS"]

    # 固定参数: 使用一个包含3个编码方案的组合
    schemes_to_use = MULTI_SCHEME_PORTFOLIO[:3]

    tasks = []
    for n in node_sizes:
        for i in range(num_runs):
            tasks.append((i, n, schemes_to_use))

    print(f"总共需要执行 {len(tasks)} 次模拟...")

    results_list = []
    with Pool(processes=cpu_count() - 1) as pool:
        with tqdm(total=len(tasks), desc="Exp 4a Progress (vs. Network Size)") as pbar:
            for result in pool.imap_unordered(run_single_scalability_test, tasks):
                if result is not None:
                    results_list.append(result)
                pbar.update(1)

    # --- 串行 debug 用 ---
    # with tqdm(total=len(tasks), desc="总模拟进度 (Debug Mode)") as pbar:
    #     for task in tasks:
    #         result = run_single_scalability_test(task)  # 直接调用任务函数
    #         if result is not None:
    #             results_list.append(result)
    #         pbar.update(1)

    # 汇总结果
    avg_times = []
    # 这里我们需要将 results_list 重新与 tasks 关联，或者修改返回
    # 一个更简单的汇总方法
    aggregated_results = {n: [] for n in node_sizes}
    for item in results_list:
        num_nodes = item['num_nodes']
        run_time = item['run_time']
        aggregated_results[num_nodes].append(run_time)


    for n in node_sizes:
        avg_times.append(np.mean(aggregated_results[n]) if aggregated_results[n] else np.nan)

    return {"node_sizes": node_sizes, "avg_times": avg_times}


def run_exp4b_vs_num_schemes():
    """实验 4b: 测试运行时间随编码方案数量的变化。"""
    print("\n--- 正在运行实验 4b: 运行时间 vs. 编码方案数量 ---")

    # 自变量: 编码方案数量 k
    scheme_counts = config.PARAMS.get("SCHEME_COUNTS_FOR_SCALABILITY", [1, 2, 3, 4, 5])
    num_runs = config.PARAMS["NUM_RUNS"]

    # 固定参数: 网络规模
    num_nodes = config.PARAMS.get("DEFAULT_NUM_NODES_FOR_SCALABILITY", 50)

    # 1. 创建所有任务的参数列表
    tasks = []
    for k in scheme_counts:
        # 从总的方案组合中，取出前 k 个
        schemes_to_use = MULTI_SCHEME_PORTFOLIO[:k]
        for i in range(num_runs):
            tasks.append((i, num_nodes, schemes_to_use))

    print(f"总共需要执行 {len(tasks)} 次模拟...")

    # 2. 并行执行所有任务
    results_list = []
    with Pool(processes=cpu_count() - 1) as pool:
        with tqdm(total=len(tasks), desc="Exp 4b Progress (vs. #Schemes)") as pbar:
            # imap_unordered 能在结果就绪时立即返回，效率高
            for result in pool.imap_unordered(run_single_scalability_test, tasks):
                if result is not None:
                    results_list.append(result)
                pbar.update(1)

    # --- 串行 debug 用 ---
    # with tqdm(total=len(tasks), desc="Exp 4b Progress (Debug Mode)") as pbar:
    #     for task in tasks:
    #         result = run_single_scalability_test(task)  # 直接调用任务函数
    #         if result is not None:
    #             results_list.append(result)
    #         pbar.update(1)

    # 3. 汇总结果
    # aggregated_results: { k -> [time1, time2, ...] }
    aggregated_results = {k: [] for k in scheme_counts}
    for key_param, run_time in results_list:
        # key_param 在这里就是 k (编码方案数量)
        if key_param in aggregated_results:
            aggregated_results[key_param].append(run_time)

    avg_times = []
    for k in scheme_counts:
        times = aggregated_results[k]
        avg_times.append(np.mean(times) if times else np.nan)

    # 4. 返回格式化的结果字典
    return {"scheme_counts": scheme_counts, "avg_times": avg_times}


def main():
    """主函数，按顺序运行所有可扩展性实验并保存结果。"""

    results_4a = run_exp4a_vs_network_size()
    results_4b = run_exp4b_vs_num_schemes() # 暂时注释，先完成4a

    # --- 保存结果 ---
    output_data = {
        "parameters": config.PARAMS,
        "results": {
            "experiment_4a": results_4a,
            "experiment_4b": results_4b,
        }
    }

    results_filename = get_timestamped_filename("experiment_4_scalability")
    os.makedirs(config.PARAMS["RESULTS_DIR"], exist_ok=True)
    filepath = os.path.join(config.PARAMS["RESULTS_DIR"], results_filename)

    print(f"\n可扩展性实验完成，正在将结果保存到 {filepath} ...")
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=4, default=str)
    print("保存成功。")


if __name__ == "__main__":
    main()
