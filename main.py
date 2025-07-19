# main.py
import random
import time
from typing import List

from core.algorithm import find_min_cost_feasible_path
from core.encoding_schemes import DEFAULT_SCHEMES
from core.graph_transformer import transform_graph
from core.network_generator import create_random_network


def get_source_and_dest_from_super_switches(super_switches: List[int], seed):
    random.seed(seed)
    source_and_dest_list = random.sample(super_switches, 2)
    source = source_and_dest_list[0]
    dest = source_and_dest_list[1]
    return source, dest


def run_single_flow_experiment():
    """运行单流、多编码方案的实验。"""
    print("--- 运行单流、多编码方案实验 ---")

    # 1. 实验设置
    G, super_switches = create_random_network(num_nodes=500, p_super_switch=0.4, avg_degree=5, seed=42)
    schemes = DEFAULT_SCHEMES
    source, dest = get_source_and_dest_from_super_switches(super_switches, seed=42)
    error_threshold = 0.05  # 端到端错误率阈值
    delta = 0.000001  # 精度离散化步长

    # 2. 转换图
    print("正在转换原始图 G...")
    G_prime = transform_graph(G, schemes)
    print(f"原始图: {len(G.nodes())} 个节点, {len(G.edges())} 条边。")
    print(f"扩展图: {len(G_prime.nodes())} 个节点, {len(G_prime.edges())} 条边。")

    # 3. 运行算法
    print(f"\n正在寻找从 {source} 到 {dest} 的路径，错误率阈值为 {error_threshold}...")
    # --- 计时开始 ---
    start_time = time.perf_counter()
    cost, error_rate = find_min_cost_feasible_path(G_prime, schemes, source, dest, error_threshold, delta)
    # --- 计时结束 ---
    end_time = time.perf_counter()
    duration = end_time - start_time

    # 4. 显示结果
    if cost is not None:
        print("\n--- 找到可行路径! ---")
        print(f"  最低成本: {cost:.2f}")
        print(f"  最终错误率: {error_rate:.4f} (<= {error_threshold})")
    else:
        print("\n--- 未找到可行路径 ---")

    # 5. 可视化 (可选)
    # plot_transformed_graph(G_prime, G, schemes)

    print(f"\n算法运行时间: {duration:.4f} 秒") # <--- 打印运行时间

def run_multi_flow_experiment():
    """多流实验的占位函数。"""
    print("\n--- 多流实验 (未实现) ---")
    print("这部分将包括:")
    print("1. 对每个流，运行单流算法来计算一个候选路径池。")
    print("2. 建立并求解整数线性规划(ILP)或其松弛版本(LP)，以选择最佳的路径组合。")


if __name__ == "__main__":
    run_single_flow_experiment()
    # run_multi_flow_experiment()