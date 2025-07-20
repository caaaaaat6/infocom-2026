# tests/test_multi_flow_solver.py

import unittest
import networkx as nx

# 使用绝对导入，假设你的项目根目录是 infocom-2026
# 如果独立运行，需要处理sys.path
from baselines.multi_flow_baselines import solve_multi_flow_ilp_minimize_congestion
from core.network_generator import create_random_network  # 我们这里手动创建，但为了导入先写上


class TestMultiFlowSolver(unittest.TestCase):

    def setUp(self):
        """
        在每个测试用例运行前，设置好测试环境。
        """
        # --- 1. 创建一个我们能手动分析的钻石网络 ---
        self.G = nx.Graph()
        self.G.add_edges_from([
            ('S', 'A'), ('S', 'B'),
            ('A', 'D'), ('B', 'D'),
            ('A', 'B')
        ])

        # --- 2. 定义两个流 ---
        self.flows = [('S', 'D'), ('A', 'B')]

        # --- 3. 手动为每个流创建候选路径池 ---
        self.path_pools = {}

        # 路径池 for Flow 1: S -> D
        path_S_D_1 = {'path': ['S', 'A', 'D'], 'edge_list': [('S', 'A'), ('A', 'D')]}
        path_S_D_2 = {'path': ['S', 'B', 'D'], 'edge_list': [('S', 'B'), ('B', 'D')]}
        # 增加一条次优路径，让问题更丰富
        path_S_D_3 = {'path': ['S', 'A', 'B', 'D'], 'edge_list': [('S', 'A'), ('A', 'B'), ('B', 'D')]}
        self.path_pools[('S', 'D')] = [path_S_D_1, path_S_D_2, path_S_D_3]

        # 路径池 for Flow 2: A -> B
        path_A_B_1 = {'path': ['A', 'B'], 'edge_list': [('A', 'B')]}
        self.path_pools[('A', 'B')] = [path_A_B_1]

    def test_diamond_network(self):
        """
        测试在经典的钻石网络场景下，ILP求解器是否能找到最优的负载均衡方案。
        """
        print("\n--- 正在测试多流拥塞最小化 ILP 求解器 ---")

        # 调用我们要测试的函数
        max_congestion = solve_multi_flow_ilp_minimize_congestion(
            path_pools=self.path_pools,
            G=self.G,
            flows=self.flows
        )

        print(f"ILP 求解器计算出的最小最大拥塞为: {max_congestion}")

        # --- 4. 断言结果是否符合我们的预期 ---
        # 我们手动推算出的最优解是 1
        expected_max_congestion = 1.0

        self.assertIsNotNone(max_congestion, "求解器未能找到最优解 (返回 None)")
        self.assertGreater(max_congestion, -1, "求解器返回了一个表示失败的值")
        self.assertEqual(max_congestion, expected_max_congestion,
                         f"结果不符合预期！应该是 {expected_max_congestion}，但得到了 {max_congestion}")

        print("测试通过！结果符合预期。")


# --- 如何运行这个测试文件 ---
if __name__ == '__main__':
    # 为了能独立运行，需要正确处理Python的模块查找路径
    import sys
    import os

    # 将项目根目录 (infocom-2026) 添加到 sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # 运行测试
    unittest.main()