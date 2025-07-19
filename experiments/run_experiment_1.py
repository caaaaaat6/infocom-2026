# experiments/run_experiment_1.py
import json
import os
import random
import numpy as np
from tqdm import tqdm  # tqdm 是一个漂亮的进度条库，对于长时间运行的模拟很有用

# 导入配置和项目模块
import config
from core.network_generator import create_random_network
from core.graph_transformer import transform_graph, transform_graph_to_directed
from core.algorithm import find_min_cost_feasible_path
from core.encoding_schemes import EncodingScheme, SCHEME_B
from baselines.greedy_baselines import find_path_decode_always, find_path_forward_always

# 为实验一，我们只使用一个单编码方案
SINGLE_SCHEME = SCHEME_B


def main():
    print("--- 正在运行实验一：核心算法有效性验证 ---")

    # 定义要运行的算法和它们的函数
    algorithms = {
        "Proposed": find_min_cost_feasible_path,
        "Decode-Always": find_path_decode_always,
        "Forward-Always": find_path_forward_always
    }

    # --- 1. 计算总的迭代次数 ---
    num_thresholds = len(config.ERROR_THRESHOLDS)
    num_runs_per_threshold = config.NUM_RUNS
    total_iterations = num_thresholds * num_runs_per_threshold

    # 最终结果的存储结构
    # { "Proposed": {"costs": [...], "accept_ratios": [...]}, "Decode-Always": ... }
    all_results = {name: {'costs': [], 'accept_ratios': []} for name in algorithms}

    # --- 2. 创建一个单一的、手动的 tqdm 进度条 ---
    with tqdm(total=total_iterations, desc="总实验进度 (Total Progress)") as pbar:

        # 外层循环：遍历自变量 (错误率阈值)
        # for r_theta in tqdm(config.ERROR_THRESHOLDS, desc="错误率阈值进度", position=0):
        for r_theta in config.ERROR_THRESHOLDS:

            # 用于存储单次运行结果的临时字典
            run_costs = {name: [] for name in algorithms}
            run_successes = {name: 0 for name in algorithms}
            delta = config.EPSILON * r_theta / config.DEFAULT_NUM_NODES

            # 内层循环：进行多次运行以获得统计平均值
            # for _ in tqdm(range(config.NUM_RUNS), desc='每个阈值上运行进度', leave=False, position=1):
            for _ in range(config.NUM_RUNS):
                G, super_switches = create_random_network(config.DEFAULT_NUM_NODES, avg_degree=config.DEFAULT_AVG_DEGREE, p_super_switch=config.DEFAULT_P_SUPER_SWITCH)
                source, dest = random.sample(list(super_switches), 2)

                # --- 运行 Proposed 算法 ---
                G_prime = transform_graph(G, [SINGLE_SCHEME])
                cost, _ = algorithms["Proposed"](G_prime, [SINGLE_SCHEME], source, dest, r_theta,
                                                 delta)
                if cost is not None:
                    run_costs["Proposed"].append(cost)
                    run_successes["Proposed"] += 1

                # --- 运行基线算法 ---
                G_directed = transform_graph_to_directed(G)
                cost_da, _ = algorithms["Decode-Always"](G_directed, SINGLE_SCHEME, source, dest, r_theta, delta)
                if cost_da is not None:
                    run_costs["Decode-Always"].append(cost_da)
                    run_successes["Decode-Always"] += 1

                cost_fa, _ = algorithms["Forward-Always"](G_directed, SINGLE_SCHEME, source, dest, r_theta)
                if cost_fa is not None:
                    run_costs["Forward-Always"].append(cost_fa)
                    run_successes["Forward-Always"] += 1

                # --- 4. 手动更新进度条 ---
                # 内层循环每完成一次，总进度就前进 1
                pbar.update(1)

            # 计算这个 r_theta 点上的平均统计值
            for name in algorithms:
                # 如果一次成功运行都没有，成本记为 NaN (Not a Number)
                avg_cost = np.mean(run_costs[name]) if run_costs[name] else np.nan
                accept_ratio = run_successes[name] / config.NUM_RUNS

                all_results[name]['costs'].append(avg_cost)
                all_results[name]['accept_ratios'].append(accept_ratio)

    # --- 将最终结果保存到JSON文件 ---
    os.makedirs(config.RESULTS_DIR, exist_ok=True)  # 确保结果文件夹存在
    filepath = os.path.join(config.RESULTS_DIR, config.EXP1_RESULTS_FILE)

    print(f"\n实验完成，正在将结果保存到 {filepath} ...")
    with open(filepath, 'w') as f:
        # indent=4 让json文件格式化，易于阅读
        json.dump(all_results, f, indent=4)
    print("保存成功。")


if __name__ == "__main__":
    main()