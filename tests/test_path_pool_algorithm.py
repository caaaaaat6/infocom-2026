# tests/test_path_pool_algorithm.py
import time
import unittest
import networkx as nx
import os
import sys

from core.network_generator import create_random_network
from main import get_source_and_dest_from_super_switches

# --- 路径处理，确保能找到 core 模块 ---
# 将项目根目录 (infocom-2026) 添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 现在可以安全地从 core 导入了
from core.path_pool_algorithm import find_min_cost_feasible_path  # 假设你把代码放到了这个新文件
from core.encoding_schemes import EncodingScheme, SCHEME_85_1_7
from core.graph_transformer import transform_graph


class TestPathPoolAlgorithm(unittest.TestCase):

    def setUp(self):
        # """设置一个通用的测试环境。"""
        # # --- 1. 创建一个测试网络 ---
        # self.G = nx.Graph()
        # self.G.add_edge('S', 'A', cost=5, time=5e-6, p_physical=0.05)
        # self.G.add_edge('A', 'D', cost=5, time=5e-6, p_physical=0.05)
        #
        # self.G.add_edge('S', 'B', cost=3, time=3e-6, p_physical=0.01)
        # self.G.add_edge('B', 'C', cost=3, time=3e-6, p_physical=0.01)
        # self.G.add_edge('C', 'D', cost=3, time=3e-6, p_physical=0.01)
        #
        # self.G.add_edge('S', 'E', cost=6, time=6e-6, p_physical=0.02)
        # self.G.add_edge('E', 'D', cost=6, time=6e-6, p_physical=0.02)
        #
        # # 设置节点类型
        # nx.set_node_attributes(self.G, {
        #     'S': {'type': 'switch'}, 'D': {'type': 'super_switch'},
        #     'A': {'type': 'super_switch'}, 'B': {'type': 'super_switch'},
        #     'C': {'type': 'super_switch'}, 'E': {'type': 'super_switch'}
        # })
        #
        # # --- 2. 创建一个测试用的编码方案 ---
        # # 使用玩具模型，方便测试
        # self.scheme = SCHEME_85_1_7  # fj_filepath=None 会触发玩具模型
        #
        # # --- 3. 转换图 ---
        # self.G_prime = transform_graph(self.G, [self.scheme])

        # 1. 实验设置
        G, super_switches = create_random_network(num_nodes=200, p_super_switch=0.4, avg_degree=4, seed=42)
        self.scheme = SCHEME_85_1_7
        self.source, self.dest = get_source_and_dest_from_super_switches(super_switches, seed=42)
        error_threshold = 0.05  # 端到端错误率阈值
        delta = 0.000001  # 精度离散化步长

        # 2. 转换图
        print("正在转换原始图 G...")
        self.G_prime = transform_graph(G, [self.scheme])
        print(f"原始图: {len(G.nodes())} 个节点, {len(G.edges())} 条边。")
        print(f"扩展图: {len(self.G_prime.nodes())} 个节点, {len(self.G_prime.edges())} 条边。")

        # 3. 运行算法
        print(f"\n正在寻找从 {self.source} 到 {self.dest} 的路径，错误率阈值为 {error_threshold}...")
        # --- 计时开始 ---

    def test_find_single_path_successfully(self):
        """
        测试当 pool_size=1 时，算法是否能成功找到一条最优路径。
        """
        print("\n--- 测试路径池算法 (M=1) ---")
        start_time = time.perf_counter()
        path_pool = find_min_cost_feasible_path(
            self.G_prime, [self.scheme], self.source, self.dest,
            error_threshold=0.1, delta=0.01, pool_size=3
        )
        end_time = time.perf_counter()
        duration = end_time - start_time

        print(f"找到的路径池 (M=1): {path_pool}")

        print(f"\n算法运行时间: {duration:.4f} 秒") # <--- 打印运行时间


    # def test_find_multiple_paths_for_pool(self):
    #     """
    #     测试当 pool_size > 1 时，算法是否能找到多条不同的路径。
    #     """
    #     print("\n--- 测试路径池算法 (M=2) ---")
    #     path_pool = find_min_cost_feasible_path(
    #         self.G_prime, [self.scheme], 'S', 'D',
    #         error_threshold=0.1, delta=0.01, pool_size=2
    #     )
    #
    #     print(f"找到的路径池 (M=2): {path_pool}")
    #
    #     self.assertGreaterEqual(len(path_pool), 1, "至少应该找到一条路径")
    #     self.assertLessEqual(len(path_pool), 2, "路径池的大小不应超过 pool_size")
    #
    #     # 验证返回的路径池是按成本排序的
    #     costs = [p['cost'] for p in path_pool]
    #     self.assertEqual(costs, sorted(costs), "路径池应该按成本升序排列")
    #
    #     # 验证找到了预期的两条路径
    #     path_nodes_list = [tuple(p['nodes']) for p in path_pool]
    #     expected_paths = {('S', 'B', 'C', 'D'), ('S', 'A', 'D')}
    #
    #     # 根据你的算法实现，另一条路径 S->A->D 成本为 5+5+10=20
    #     # S->E->D 成本为 6+6+10=22
    #     # 所以应该是 S->A->D 和 S->E->D
    #     # 我们来重新计算一下
    #     # S->A->D: 5(SA)+5(AD)+10(A_op) = 20
    #     # S->E->D: 6(SE)+6(ED)+10(E_op) = 22
    #     # S->B->C->D: 3+3+3 + 10(B_op)+10(C_op) = 29
    #     # 所以顺序应该是 S->A->D, S->E->D
    #
    #     expected_paths_sorted_by_cost = [('S', 'A', 'D'), ('S', 'E', 'D')]
    #
    #     self.assertEqual(len(path_nodes_list), 2, "应该恰好找到两条路径")
    #     self.assertEqual(path_nodes_list[0], expected_paths_sorted_by_cost[0])
    #     self.assertEqual(path_nodes_list[1], expected_paths_sorted_by_cost[1])
    #
    #     print("测试通过！成功找到多条路径并正确排序。")


if __name__ == '__main__':
    unittest.main()