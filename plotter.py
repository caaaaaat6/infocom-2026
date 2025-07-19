# plotter.py
import matplotlib.pyplot as plt
import networkx as nx


def plot_network(G: nx.Graph, source: int, dest: int):
    """绘制原始的量子网络拓扑图。"""
    # ... (这个函数保持不变，为了方便对比) ...
    pos = nx.spring_layout(G, seed=42)
    node_colors = []
    for node in G.nodes():
        if node == source:
            node_colors.append('green')
        elif node == dest:
            node_colors.append('red')
        elif G.nodes[node]['type'] == 'super_switch':
            node_colors.append('skyblue')
        else:
            node_colors.append('lightgray')

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'p_physical')
    formatted_labels = {k: f"{v:.3f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_labels, font_color='firebrick')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("原始量子网络拓扑图 (G)")
    plt.show()


def plot_transformed_graph(G_prime: nx.DiGraph, G_original: nx.Graph, schemes):
    """
    绘制扩展后的辅助图 (G' 或 G")。
    - G_prime: 扩展图
    - G_original: 原始图，用于获取节点位置，保持布局一致性
    - schemes: 编码方案列表，用于解码节点的标签
    """
    plt.figure(figsize=(18, 12))

    # 1. 计算布局
    # 首先为原始图计算一个布局，然后根据原始节点的位置来摆放分裂后的节点
    pos_original = nx.spring_layout(G_original, seed=42, k=0.8)  # k增大了节点间距

    pos = {}
    for node in G_prime.nodes():
        node_data = G_prime.nodes[node]
        original_node = node_data.get('original_node', node)  # 获取原始节点
        base_pos = pos_original[original_node]

        # 根据节点类型微调位置，让分裂的节点围绕在原始位置附近
        if node_data['type'] == 'forward_only':
            pos[node] = (base_pos[0] - 0.05, base_pos[1] + 0.05)
        elif node_data['type'] == 'decode':
            scheme_idx = node_data['scheme_idx']
            # 让不同的解码节点错开位置
            offset = 0.1 * (scheme_idx - (len(schemes) - 1) / 2)
            pos[node] = (base_pos[0] + offset, base_pos[1] - 0.05)
        else:  # 普通交换机
            pos[node] = base_pos

    # 2. 设置节点颜色和形状
    node_colors = []
    node_shapes = []  # 'o' for circle, 's' for square
    for node in G_prime.nodes():
        node_type = G_prime.nodes[node]['type']
        if node_type == 'switch':
            node_colors.append('lightgray')
            node_shapes.append('o')  # 圆形代表普通交换机
        elif node_type == 'forward_only':
            node_colors.append('orange')
            node_shapes.append('s')  # 方形代表转发节点
        elif node_type == 'decode':
            node_colors.append('skyblue')
            node_shapes.append('d')  # 菱形代表解码节点

    # 3. 绘制不同类型的节点
    # 为了能使用不同的形状，我们需要分别绘制每种形状的节点

    switches = [n for n, s in zip(G_prime.nodes(), node_shapes) if s == 'o']
    fos = [n for n, s in zip(G_prime.nodes(), node_shapes) if s == 's']
    des = [n for n, s in zip(G_prime.nodes(), node_shapes) if s == 'd']

    nx.draw_networkx_nodes(G_prime, pos, nodelist=switches, node_shape='o', node_color='lightgray', node_size=1000)
    nx.draw_networkx_nodes(G_prime, pos, nodelist=fos, node_shape='s', node_color='orange', node_size=1000)
    nx.draw_networkx_nodes(G_prime, pos, nodelist=des, node_shape='d', node_color='skyblue', node_size=1000)

    # 4. 绘制边和标签
    nx.draw_networkx_edges(G_prime, pos, arrowstyle='->', arrowsize=15, connectionstyle='arc3,rad=0.1')

    # 创建更详细的节点标签
    labels = {}
    for node in G_prime.nodes():
        node_type = G_prime.nodes[node]['type']
        if node_type == 'decode':
            scheme_idx = G_prime.nodes[node]['scheme_idx']
            scheme_name_short = schemes[scheme_idx].name.split('_')[0]
            labels[node] = f"{G_prime.nodes[node]['original_node']}\nDE ({scheme_name_short})"
        elif node_type == 'forward_only':
            labels[node] = f"{G_prime.nodes[node]['original_node']}\nFO"
        else:
            labels[node] = str(node)

    nx.draw_networkx_labels(G_prime, pos, labels=labels, font_size=8)

    # 5. 设置图表属性
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("扩展后的辅助图 (G\")", fontsize=16)
    plt.axis('off')  # 关闭坐标轴
    plt.show()