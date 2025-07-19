# experiments/run_experiment_1.py
import json
import os
# ... (所有导入) ...
from multiprocessing import Pool, cpu_count
from functools import partial
import random

from tqdm import tqdm

from baselines.greedy_baselines import find_path_forward_always, find_path_decode_always
from core.algorithm import find_min_cost_feasible_path
from core.encoding_schemes import *
from core.graph_transformer import transform_graph
from core.network_generator import create_random_network
from experiments import config
from experiments.config import PARAMS


# --- 1. 将单次模拟运行的逻辑封装成一个顶级函数 ---
# 这个函数必须定义在 main() 函数之外，以便被子进程正确地 pickle 和调用
def run_single_simulation(run_index, r_theta, schemes, delta):
    """
    执行一次独立的模拟。
    返回一个字典，包含这次运行的结果。
    """
    try:
        # 每次运行使用不同的随机种子，确保独立性
        G, super_switches = create_random_network(config.PARAMS["DEFAULT_NUM_NODES"],
                                  avg_degree=config.PARAMS["DEFAULT_AVG_DEGREE"],
                                  p_super_switch=config.PARAMS["DEFAULT_P_SUPER_SWITCH"],
                                  seed=run_index)  # 使用运行索引作为种子

        if len(super_switches) < 2:
            return {"Proposed": None, "Decode-Always": None, "Forward-Always": None}

        source, dest = random.sample(super_switches, 2)

        # --- 运行所有算法 ---
        results_this_run = {}

        # Proposed
        g_prime = transform_graph(G, [schemes["Proposed"]])
        cost_p, _ = find_min_cost_feasible_path(g_prime, [schemes["Proposed"]], source, dest, r_theta, delta)
        results_this_run["Proposed"] = cost_p

        # Decode-Always
        cost_da, _ = find_path_decode_always(G, schemes["Decode-Always"], source, dest, r_theta, delta)
        results_this_run["Decode-Always"] = cost_da

        # Forward-Always
        cost_fa, _ = find_path_forward_always(G, schemes["Forward-Always"], source, dest, r_theta)
        results_this_run["Forward-Always"] = cost_fa

        return results_this_run
    except Exception as e:
        # 捕获子进程中的异常，避免整个程序崩溃
        print(f"一次模拟在 r_theta={r_theta} 时发生错误: {e}")
        return {"Proposed": None, "Decode-Always": None, "Forward-Always": None}


def main():
    print("--- 正在运行实验一：核心算法有效性验证 (并行版) ---")

    # 编码方案的定义
    schemes_for_algos = {
        "Proposed": SCHEME_145_1_9,
        "Decode-Always": SCHEME_145_1_9,
        "Forward-Always": SCHEME_145_1_9
    }
    algorithms = list(schemes_for_algos.keys())

    all_results = {name: {'costs': [], 'accept_ratios': []} for name in algorithms}

    # --- 2. 使用 multiprocessing.Pool ---
    # 使用所有可用的CPU核心，或者 (cpu_count() - 1) 来保留一个核心
    num_processes = cpu_count() - 1
    print(f"使用 {num_processes} 个CPU核心进行并行计算...")

    with Pool(processes=num_processes) as pool:
        # 外层循环依然是串行的
        for r_theta in tqdm(config.PARAMS["ERROR_THRESHOLDS"], desc="总进度 (Threshold Sweep)"):

            delta = config.PARAMS["EPSILON"] * r_theta / config.PARAMS["DEFAULT_NUM_NODES"]

            # --- 3. 准备任务列表并分发 ---
            # functools.partial 允许我们"预填"一些参数给 run_single_simulation
            # 这样 Pool 的 map 函数就能只传入变化的参数 (run_index)
            task_func = partial(run_single_simulation, r_theta=r_theta, schemes=schemes_for_algos, delta=delta)

            # pool.map 会自动将 range(NUM_RUNS) 中的每个值作为第一个参数传给 task_func
            # 并且会阻塞，直到所有任务完成
            # tqdm 包裹 pool.map 可以显示并行任务的进度！
            list_of_results = list(tqdm(pool.imap_unordered(task_func, range(config.PARAMS["NUM_RUNS"])),
                                        total=config.PARAMS["NUM_RUNS"],
                                        desc=f"并行模拟 for r={r_theta:.3f}",
                                        leave=False))

            # --- 4. 收集并汇总结果 ---
            run_costs = {name: [] for name in algorithms}
            run_successes = {name: 0 for name in algorithms}

            for single_run_result in list_of_results:
                for name in algorithms:
                    cost = single_run_result[name]
                    if cost is not None:
                        run_costs[name].append(cost)
                        run_successes[name] += 1

            # 计算平均值
            for name in algorithms:
                avg_cost = np.mean(run_costs[name]) if run_costs[name] else np.nan
                accept_ratio = run_successes[name] / config.PARAMS["NUM_RUNS"]
                all_results[name]['costs'].append(avg_cost)
                all_results[name]['accept_ratios'].append(accept_ratio)

        # --- 1. 创建用于保存的总数据结构 ---
        output_data = {
            "parameters": PARAMS,
            "results": all_results
        }

        # --- 2. 保存结果 ---
        results_filename = config.get_timestamped_filename(config.EXP1_RESULTS_FILE)
        os.makedirs(PARAMS["RESULTS_DIR"], exist_ok=True)
        filepath = os.path.join(PARAMS["RESULTS_DIR"], results_filename)

    print(f"\n实验完成，正在将结果保存到 {filepath} ...")
    with open(filepath, 'w') as f:
        # indent=4 让json文件格式化，易于阅读
        json.dump(output_data, f, indent=4)
    print("保存成功。")


if __name__ == "__main__":
    main()