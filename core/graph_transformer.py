# graph_transformer.py
from typing import List

import networkx as nx

from .encoding_schemes import EncodingScheme


def transform_graph(G: nx.Graph, schemes: List[EncodingScheme]) -> nx.DiGraph:
    """
    将原始图 G 转换为用于多编码问题的扩展图 G" (在代码中我们称之为 G_prime)。
    - G: 原始网络图
    - schemes: 可用的编码方案列表
    返回: 扩展后的有向图
    """
    G_prime = nx.DiGraph()  # 必须使用有向图以支持 Bellman-Ford
    k = len(schemes)

    # 辅助字典，用于映射原始节点到其在 G_prime 中的分裂节点
    node_map = {}

    for node in G.nodes():
        if G.nodes[node]['type'] == 'switch':
            # 普通交换机保持不变
            G_prime.add_node(node, type='switch')
            node_map[node] = [node]
        else:  # 'super_switch'
            # 将超级交换机分裂成 k+1 个节点
            fo_node = f"{node}_fo"  # 转发节点 (forward-only)
            de_nodes = [f"{node}_de_{i}" for i in range(k)]  # k 个解码节点 (decode)

            G_prime.add_node(fo_node, type='forward_only', original_node=node)
            for i, scheme in enumerate(schemes):
                G_prime.add_node(de_nodes[i], type='decode', original_node=node,
                                 scheme_idx=i, op_cost=scheme.cost)

            node_map[node] = [fo_node] + de_nodes

    # 重构边
    for u, v in G.edges():
        u_nodes_prime = node_map[u]
        v_nodes_prime = node_map[v]

        edge_cost = G.edges[u, v]['cost']
        edge_p_phys = G.edges[u, v]['p_physical']
        edge_time = G.edges[u, v]['time']

        # 在 u 的分裂节点和 v 的分裂节点之间创建全连接边
        for u_p in u_nodes_prime:
            for v_p in v_nodes_prime:
                G_prime.add_edge(u_p, v_p, cost=edge_cost, p_physical=edge_p_phys, time=edge_time)
                G_prime.add_edge(v_p, u_p, cost=edge_cost, p_physical=edge_p_phys, time=edge_time)

    return G_prime


def transform_graph_to_directed(G: nx.Graph) -> nx.DiGraph:
    """
    将原始图 G 转换为用于多编码问题的扩展图 G" (在代码中我们称之为 G_prime)。
    - G: 原始网络图
    - schemes: 可用的编码方案列表
    返回: 扩展后的有向图
    """
    G_bidirectional = G.to_directed()

    return G_bidirectional

