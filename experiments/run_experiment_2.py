# experiments/run_experiment_2.py
import json
import os
import random
import numpy as np
from tqdm import tqdm

# 使用绝对路径导入所有需要的模块
import experiments.config as config
from core.network_generator import create_random_network
from core.graph_transformer import transform_graph
from core.algorithm import find_min_cost_feasible_path
# --- 1. 导入新的编码方案 ---
from core.encoding_schemes import (
    SINGLE_SCHEME_41_1_3 as SINGLE_SCHEME_D5,  # d=5，应该是 n=41 还是 n=25？这里根据你的定义来
    SINGLE_SCHEME_85_1_7 as SINGLE_SCHEME_D7,
    SINGLE_SCHEME_145_1_9 as SINGLE_SCHEME_D9,
    MULTI_SCHEME_PORTFOLIO
)
from experiments.config import EXP2_RESULTS_FILE


def main():
    print("--- 正在运行实验二：多编码方案选择优势 (表面码对比) ---")

    # --- 2. 更新实验场景定义 ---
    scenarios = {
        "Proposed (Multi-Scheme d=5,7,9)": MULTI_SCHEME_PORTFOLIO,
        "Single-Scheme (d=5)": SINGLE_SCHEME_D5,
        "Single-Scheme (d=7)": SINGLE_SCHEME_D7,
        "Single-Scheme (d=9)": SINGLE_SCHEME_D9
    }

    all_results = {name: {'costs': [], 'accept_ratios': []} for name in scenarios}

    total_iterations = len(config.ERROR_THRESHOLDS) * config.NUM_RUNS
    with tqdm(total=total_iterations, desc="总实验进度 (Experiment 2)") as pbar:

        for r_theta in config.ERROR_THRESHOLDS:

            run_costs = {name: [] for name in scenarios}
            run_successes = {name: 0 for name in scenarios}
            delta = config.EPSILON * r_theta / config.DEFAULT_NUM_NODES

            for i in range(config.NUM_RUNS):
                pbar.set_postfix(r_theta=f'{r_theta:.3f}', run=f'{i + 1}/{config.NUM_RUNS}')

                G = create_random_network(config.DEFAULT_NUM_NODES, avg_degree=config.DEFAULT_AVG_DEGREE,
                                          p_super_switch=config.DEFAULT_P_SUPER_SWITCH)

                super_switches = [n for n, data in G.nodes(data=True) if data['type'] == 'super_switch']
                if len(super_switches) < 2:
                    pbar.update(1)
                    continue
                source, dest = random.sample(super_switches, 2)

                for name, schemes in scenarios.items():
                    G_prime = transform_graph(G, schemes)

                    # 假设你的算法现在不需要传入 G_original 了
                    cost, _ = find_min_cost_feasible_path(
                        G_prime, schemes, source, dest, r_theta, delta
                    )

                    if cost is not None:
                        run_costs[name].append(cost)
                        run_successes[name] += 1

                pbar.update(1)

            for name in scenarios:
                avg_cost = np.mean(run_costs[name]) if run_costs[name] else np.nan
                accept_ratio = run_successes[name] / config.NUM_RUNS
                all_results[name]['costs'].append(avg_cost)
                all_results[name]['accept_ratios'].append(accept_ratio)

    # --- 保存结果 ---
    results_filename = EXP2_RESULTS_FILE
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(config.RESULTS_DIR, results_filename)

    print(f"\n实验完成，正在将结果保存到 {filepath} ...")
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=4)
    print("保存成功。")


if __name__ == "__main__":
    # 为了能独立运行，需要正确处理路径
    # 注意: 最佳实践是通过项目根目录的 main.py 启动
    main()