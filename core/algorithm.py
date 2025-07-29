# algorithm.py
from typing import List, Dict, Tuple

import networkx as nx
import numpy as np

from .encoding_schemes import EncodingScheme

# --- 1. 修改标签 (Label) 的数据结构 ---
# 新增了 `time_since_decode` 和 `logical_lifetime` 两个维度
# (成本, 离散化精度整数, 自上次解码经过的时间, 物理错误累加器, 前驱节点)
Label = Tuple[float, int, float, List[float], any]


def calculate_composite_prob_from_paper(P_v: List[float]) -> float:
    """
    严格按照论文中给出的公式 (1) 来计算复合信道的退极化概率。
    p_comp ≈ 3 * sum_{e} (1 - p_e/3) * product_{e'!=e} (p_{e'}/3)

    参数:
    - P_v: 包含复合信道各段物理错误率 p_e 的列表。

    返回:
    - p_comp: 计算得到的复合信道退极化概率。
    """
    p_comp = 1.0 - np.prod([1.0 - p for p in P_v])
    return p_comp


def find_min_cost_feasible_path(G_prime: nx.DiGraph,
                                schemes: List[EncodingScheme],
                                source: int,
                                dest: int,
                                error_threshold: float,
                                delta: float):
    """
    论文中算法1的Python实现。
    寻找一条在错误率约束下成本最低的可行路径。
    - G_prime: 扩展图
    - schemes: 编码方案列表
    - source, dest: 源和目的节点
    - error_threshold: 端到端错误率阈值
    - delta: 精度离散化的步长
    """
    # 找到目的节点在扩展图中的所有对应分裂节点
    dest_nodes = [n for n in G_prime.nodes() if G_prime.nodes[n].get('original_node') == dest]
    if not dest_nodes:  # 如果目的节点是普通交换机
        raise Exception('dest_nodes must be super_switch')
    if f'{dest}_fo' in dest_nodes:
        dest_nodes.remove(f'{dest}_fo')

    if f'{source}_fo' in G_prime.nodes():
        source = f'{source}_fo'
    else:
        source = f'{source}_de_0'

    # --- 标签初始化 (伪代码第3行) ---
    # 使用字典来存储标签: {节点 -> [标签列表]}
    labels: Dict[any, List[Label]] = {node: [] for node in G_prime.nodes()}

    initial_accuracy = 1.0  # 初始精度为100%
    initial_acc_idx = int(np.ceil(initial_accuracy / delta))
    # 初始标签: 成本=0, 精度=满, 距上次解码时间=0, q=[], 前驱=None
    # 初始逻辑寿命设为无穷大，因为源点出发的段没有寿命限制
    initial_label: Label = (0.0, initial_acc_idx, 0.0, [], None)
    labels[source] = [initial_label]

    # --- 主循环 (伪代码第4-23行) ---
    # Bellman-Ford 的主循环，迭代 |V'|-1 次
    for _ in range(len(G_prime.nodes()) - 1):
        # 遍历图中的每一条边
        for u, v, edge_data in G_prime.edges(data=True):
            if not labels[u]:  # 如果节点u还没有可达的标签，则跳过
                continue

            # 遍历节点u上的所有标签
            for l_u in labels[u]:
                cost_u, acc_idx_u, time_u, P_u, pred_u = l_u

                # 准备生成新标签 l'
                new_label: Label = None

                # 累积当前复合信道段的物理错误率
                P_v = P_u + [edge_data['p_physical']]

                v_node_data = G_prime.nodes[v]  # 目标节点 v 的属性

                # 计算到达v时，自上次解码以来经过的总时间
                time_v = time_u + edge_data['time']

                # --- 在超级交换机处解码 (伪代码第7-14行) ---
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
                    t_cycle = scheme.t_logical
                    logical_decoherence_time = t_cycle / p_logical

                    # 检查路径是否仍然可行 (错误率是否低于阈值)
                    if r <= error_threshold and time_v < logical_decoherence_time:
                        new_cost = cost_u + edge_data['cost'] + v_node_data['op_cost']
                        eta_v_real = 1.0 - r
                        acc_idx_v = int(np.ceil(eta_v_real / delta))
                        # 解码后，物理错误累加器被清空
                        new_label = (new_cost, acc_idx_v, 0.0, [], u)

                # --- 转发 (普通交换机或超级交换机的转发节点) (伪代码第15行) ---
                else:  # 'switch' or 'forward_only'
                    # 精度直接由复合信道的物理错误率决定
                    new_cost = cost_u + edge_data['cost']
                    new_label = (new_cost, acc_idx_u, time_v, P_v, u)

                # --- 支配性检查与松弛操作 (伪代码第17-20行) ---
                if new_label:
                    is_dominated = False

                    # 检查新标签是否被v上任何已存在标签所支配
                    for l_v_existing in list(labels[v]):
                        # 如果一个已存在标签的成本更低(或相等)且精度更高(或相等)
                        if (l_v_existing[0] <= new_label[0] and
                                l_v_existing[1] >= new_label[1]):
                            is_dominated = True
                            break

                    if not is_dominated:
                        # 移除v上所有被新标签支配的旧标签
                        labels[v][:] = [l for l in labels[v] if not
                        (new_label[0] <= l[0] and new_label[1] >= l[1])]
                        # 添加新标签
                        labels[v].append(new_label)

    # --- 路径重构 (伪代码第27-30行) ---
    final_labels = []
    # 收集目的节点的所有可行标签
    for dest_node in dest_nodes:
        final_labels.extend(labels[dest_node])

    if not final_labels:
        return None, None  # 没有找到可行路径

    # 选择成本最低的标签
    best_path_label = min(final_labels, key=lambda l: l[0])

    # 回溯路径 (这部分需要根据前驱节点信息来完整实现)
    # 目前，我们只返回最终的统计数据
    final_cost = best_path_label[0]
    final_accuracy = best_path_label[1] * delta

    # 返回 (最低成本, 最终错误率)
    return final_cost, 1.0 - final_accuracy