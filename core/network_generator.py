# network_generator.py
import random

import networkx as nx
import numpy as np

SUPER_SWITCH = 'super_switch'
SWITCH = 'switch'


def calculate_depolarizing_prob(length_km, attenuation_db_per_km=0.15):
    """根据光纤长度和衰减系数计算光子损耗概率。"""
    transmissivity = 10**(-attenuation_db_per_km * length_km / 10.0)
    return 3 / 4 * (1.0 - transmissivity)


def create_random_network(num_nodes, p_super_switch=0.3, avg_degree=3, seed=42):
    """
    创建一个随机的量子网络拓扑图。
    - num_nodes: 节点总数
    - p_super_switch: 一个节点是超级交换机的概率
    - avg_degree: 节点的平均度数
    """
    # 使用 Barabasi-Albert 模型生成一个类似无标度的网络
    G = nx.barabasi_albert_graph(num_nodes, int(avg_degree / 2), seed=seed)

    super_switches = []

    # 分配节点类型和边的属性
    for node in G.nodes():
        if random.random() < p_super_switch:
            G.nodes[node]['type'] = SUPER_SWITCH  # 超级交换机
            super_switches.append(node)
        else:
            G.nodes[node]['type'] = SWITCH  # 普通交换机

    for u, v in G.edges():
        G.edges[u, v]['cost'] = random.uniform(1, 5)  # 链路成本 (例如，延迟)
        # --- 通过模拟链路长度来生成物理错误率 ---
        # 1. 使用指数分布生成一个链路长度 L (单位: km)
        # scale = 1/lambda。这里设置平均链路长度为 3 km。
        # 指数分布会生成很多小于3的值和少量远大于3的值。
        avg_length = 2.0
        link_length = np.random.exponential(scale=avg_length)

        # 确保长度在合理范围内，避免极端值
        link_length = max(0.1, min(link_length, 20.0))  # 假设最短0.1km，最长20km

        # 传播时间 (秒)，光速在光纤中约 3e5 km/s -> 5µs/km
        prop_time = link_length * 3.33e-6

        # 2. 根据物理公式计算 p_physical
        # 这将自动产生一个偏向于低错误率的分布
        p_physical = calculate_depolarizing_prob(link_length)

        G.edges[u, v]['p_physical'] = p_physical
        G.edges[u, v]['length_km'] = link_length
        G.edges[u, v]['time'] = prop_time

    return G, super_switches


