import time

import pytest

from new.baselines.greedy_baselines import find_path_decode_always
from new.core.encoding_schemes import DEFAULT_SCHEMES, SCHEME_B
from new.core.graph_transformer import transform_graph, transform_graph_to_directed
from new.core.network_generator import create_random_network
from new.main import get_source_and_dest_from_super_switches

@pytest.mark.slow
def test_decode_always():
    """运行单流、多编码方案的实验。"""
    print("--- 运行单流、多编码方案实验 ---")

    # 1. 实验设置
    G, super_switches = create_random_network(num_nodes=500, p_super_switch=0.5, avg_degree=5, seed=42)
    schemes = SCHEME_B
    source, dest = get_source_and_dest_from_super_switches(super_switches, seed=42)
    error_threshold = 0.03  # 端到端错误率阈值
    delta = 0.000001  # 精度离散化步长

    # 2. 转换图
    print("正在转换原始图 G...")
    G_prime = transform_graph_to_directed(G)
    print(f"原始图: {len(G.nodes())} 个节点, {len(G.edges())} 条边。")
    print(f"扩展图: {len(G_prime.nodes())} 个节点, {len(G_prime.edges())} 条边。")

    # 3. 运行算法
    print(f"\n正在寻找从 {source} 到 {dest} 的路径，错误率阈值为 {error_threshold}...")
    # --- 计时开始 ---
    start_time = time.perf_counter()
    cost, error_rate = find_path_decode_always(G_prime, schemes, source, dest, error_threshold, delta)
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

    print(f"\n算法运行时间: {duration:.4f} 秒")  # <--- 打印运行时间