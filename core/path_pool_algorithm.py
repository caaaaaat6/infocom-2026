# path_pool_algorithm.py
from typing import List, Dict, Tuple, Any

import networkx as nx
import numpy as np

from .encoding_schemes import EncodingScheme

# --- 标签结构 ---
# (成本, 离散化精度整数, 自上次解码经过的时间, 物理错误累加器, 路径节点列表, 当前段的逻辑寿命)
Label = Tuple[float, int, float, List[float], List[Any], Any]


def calculate_composite_prob_from_paper(P_v: List[float]) -> float:
    """计算复合信道的物理传输错误率。"""
    if not P_v:
        return 0.0
    return 1.0 - np.prod([1.0 - p for p in P_v])


def find_min_cost_feasible_path(G_prime: nx.DiGraph,
                                schemes: List[EncodingScheme],
                                source: int,
                                dest: int,
                                error_threshold: float,
                                delta: float,
                                pool_size: int = 1):
    """
    一个可以生成候选路径池的单流算法 (简化版 M-支配逻辑)。

    核心思想:
    在每个节点 v 的每个离散化精度状态 (acc_idx) 下，最多保留 pool_size (M) 条
    成本最低的路径。
    """
    # 准备工作
    dest_nodes = [n for n in G_prime.nodes()
                  if G_prime.nodes[n].get('original_node') == dest
                  and G_prime.nodes[n]['type'] == 'decode']
    if not dest_nodes:  # 如果目的节点是普通交换机
        raise Exception('dest_nodes must be super_switch')
    source = f'{source}_fo'

    # 标签存储结构: {(node, acc_idx) -> [Label_1, Label_2, ...]}
    # 列表中的标签始终按成本升序排列
    labels: Dict[Tuple[Any, int], List[Label]] = {}

    initial_accuracy = 1.0  # 初始精度为100%
    initial_acc_idx = int(np.ceil(initial_accuracy / delta))
    # 初始标签: 成本=0, 精度=满, 距上次解码时间=0, q=[], 前驱=None
    # 初始逻辑寿命设为无穷大，因为源点出发的段没有寿命限制
    initial_label: Label = (0.0, initial_acc_idx, 0.0, [], [source], None)
    labels[(source, initial_acc_idx)] = [initial_label]

    # --- 主循环 (伪代码第4-23行) ---
    # Bellman-Ford 的主循环，迭代 |V'|-1 次
    for _ in range(len(G_prime.nodes()) - 1):
        # 遍历图中的每一条边
        for u, v, edge_data in G_prime.edges(data=True):
            # 遍历源节点 u 上所有的标签组
            keys_for_u = [key for key in labels if key[0] == u]

            for key_u in keys_for_u:
                for l_u in labels[key_u]:

                    cost_u, acc_idx_u, time_u, P_u, path_u, pred_u = l_u

                    # 准备生成新标签 l'
                    new_label: Label = None

                    # 累积当前复合信道段的物理错误率
                    P_v = P_u + [edge_data['p_physical']]

                    v_node_data = G_prime.nodes[v]  # 目标节点 v 的属性

                    # 计算到达v时，自上次解码以来经过的传播时间
                    time_v = time_u + edge_data['time']
                    new_path = path_u + [v]

                    # --- 情况1：v 是一个解码节点 ---
                    if v_node_data['type'] == 'decode':
                        scheme_idx = v_node_data['scheme_idx']
                        scheme = schemes[scheme_idx]

                        # 计算复合信道的物理错误率:
                        p_comp = calculate_composite_prob_from_paper(P_v)

                        # 计算该复合信道的逻辑错误率，使用公式（3）
                        p_logical = scheme.calculate_logical_error_rate(p_comp)

                        # 计算错误率 r，使用公式（4）
                        eta_u = acc_idx_u * delta
                        r_old = 1 - eta_u
                        r = r_old + (1 - r_old) * p_logical
                        t_cycle = scheme.t_cycle
                        logical_decoherence_time = t_cycle / p_logical

                        # 检查路径是否仍然可行 (错误率是否低于阈值)
                        if r <= error_threshold and time_v < logical_decoherence_time:
                            new_cost = cost_u + edge_data['cost'] + v_node_data['op_cost']
                            eta_v_real = 1.0 - r
                            acc_idx_v = int(np.ceil(eta_v_real / delta))
                            # 解码后，物理错误累加器被清空
                            new_label = (new_cost, acc_idx_v, 0.0, [], new_path, u)

                    # --- 转发 (普通交换机或超级交换机的转发节点) (伪代码第15行) ---
                    else:  # 'switch' or 'forward_only'
                        # 精度直接由复合信道的物理错误率决定
                        new_cost = cost_u + edge_data['cost']
                        new_label = (new_cost, acc_idx_u, time_v, P_v, new_path, u)

                    # --- 简化的 M-支配逻辑 ---
                    if new_label:
                        new_cost, new_acc_idx, _, _, _, _ = new_label
                        label_group_key = (v, new_acc_idx)

                        if label_group_key not in labels:
                            labels[label_group_key] = []

                        label_group = labels[label_group_key]

                        if len(label_group) < pool_size:
                            label_group.append(new_label)
                            label_group.sort(key=lambda l: l[0])
                        else:
                            highest_cost_in_group = label_group[-1][0]
                            if new_cost < highest_cost_in_group:
                                label_group[-1] = new_label
                                label_group.sort(key=lambda l: l[0])

    # --- 路径重构 ---
    final_labels = []
    # 收集所有到达目的节点的标签
    # (注意：dest_nodes 是解码节点，所以它们的 acc_idx 是更新过的)
    for (node, acc_idx), label_list in labels.items():
        if node in dest_nodes:
            final_labels.extend(label_list)

    if not final_labels:
        return []

    # 按成本排序并取出前 M 个作为路径池
    sorted_final_labels = sorted(final_labels, key=lambda l: l[0])

    path_pool = []
    # 确保我们不添加重复的路径
    seen_paths = set()

    for label in sorted_final_labels:
        if len(path_pool) >= pool_size:
            break

        final_cost = label[0]
        final_accuracy = label[1] * delta
        final_error_rate = 1.0 - final_accuracy
        final_path_nodes_expanded = label[4]

        # 将路径节点名从扩展图转换回原始图
        original_path_nodes = []
        for node in final_path_nodes_expanded:
            original_node = node
            if isinstance(node, str) and '_' in node:
                original_node = G_prime.nodes[node]['original_node']

            if not original_path_nodes or original_path_nodes[-1] != original_node:
                original_path_nodes.append(original_node)

        # 使用元组来检查路径是否重复
        path_tuple = tuple(original_path_nodes)
        if path_tuple in seen_paths:
            continue

        seen_paths.add(path_tuple)

        original_path_edges = list(zip(original_path_nodes[:-1], original_path_nodes[1:]))

        path_info = {
            "cost": final_cost,
            "error_rate": final_error_rate,
            "nodes": original_path_nodes,
            "edges": original_path_edges  # 注意这里是 'edges'
        }
        path_pool.append(path_info)

    return path_pool
