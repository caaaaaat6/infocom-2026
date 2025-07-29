# baselines/utils.py

from typing import List
import networkx as nx
import numpy as np

# 使用绝对导入
from core.encoding_schemes import EncodingScheme
# 从 core.algorithm 导入 p_comp 计算，而不是在 baselines 里重复定义
from core.path_pool_algorithm import calculate_composite_prob_from_paper


def check_path_feasibility(path_nodes: List,
                           G: nx.Graph,
                           scheme: EncodingScheme,
                           r_theta: float) -> bool:
    """
    检查一条给定的物理路径，在 "Forward-Always" 策略下，是否满足QEC约束。
    这个版本严格遵循“动态退相干检查”模型 (模型E)。

    参数:
    - path_nodes: 节点的有序列表，例如 ['S', 'A', 'B', 'D']。
    - G: 原始网络图，用于获取边的属性。
    - scheme: 用于最终解码的编码方案。
    - r_theta: 端到端的错误率阈值。

    返回:
    - bool: 如果路径在QEC约束下可行，则返回 True，否则返回 False。
    """
    if len(path_nodes) < 2:
        return False

    # --- 1. 计算路径的累积属性：物理错误率和传播时间 ---
    p_physical_list = []
    total_time = 0
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        if not G.has_edge(u, v):
            print(f"警告: 检查可行性时发现图中不存在的边 {(u, v)}")
            return False
        p_physical_list.append(G.edges[u, v]['p_physical'])
        total_time += G.edges[u, v]['time']

    p_comp_total = calculate_composite_prob_from_paper(p_physical_list)
    final_logical_error = scheme.calculate_logical_error_rate(p_comp_total)

    # --- 2. 检查端到端错误率约束 ---
    if final_logical_error > r_theta:
        return False

    # --- 3. 实现“动态退相干检查” ---

    # 加上在终点解码所需的时间
    total_time_with_decode = total_time

    # 根据刚刚计算出的 final_logical_error，来计算这条路径的“临界时间”
    if final_logical_error < 1e-15:
        t_critical = float('inf')
    else:
        t_critical = scheme.t_logical / final_logical_error

    # 检查总的传播+解码时间是否超过了这个临界时间
    if total_time_with_decode >= t_critical:
        return False

    # --- 如果所有检查都通过 ---
    return True