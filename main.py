# main.py
import argparse
import random
import time
import traceback
from typing import List

from core.algorithm import find_min_cost_feasible_path
from core.encoding_schemes import DEFAULT_SCHEMES
from core.graph_transformer import transform_graph
from core.network_generator import create_random_network
from experiments import run_experiment_1, run_experiment_2, run_experiment_1_parallel, plot_results, \
    run_experiment_2_parallel, run_experiment_3_parallel


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


def main(run_experiment_3=None):
    # 1. 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Run experiments for the Quantum Routing project."
    )

    # 2. 添加一个参数来选择要运行的实验
    # `choices` 参数限制了用户只能输入我们定义好的选项
    parser.add_argument(
        'experiment_name',
        type=str,
        choices=['exp1', 'exp2', 'exp3', 'exp1p', 'exp2p', 'exp3p', 'plot', 'all'],
        help="The name of the experiment to run ('exp1', 'exp2', 'exp3', 'exp1p', 'exp2p', 'exp3p', 'plot', 'all')."
    )

    # 3. 解析命令行传入的参数
    args = parser.parse_args()

    # 4. 根据参数，调用对应的函数
    print("==============================================")
    print("=      Quantum Routing Experiment Runner     =")
    print("==============================================")

    if args.experiment_name == 'exp1' or args.experiment_name == 'all':
        print("\n>>> Starting Experiment 1: Core Algorithm Validation...")
        run_experiment_1.main()

    if args.experiment_name == 'exp1p' or args.experiment_name == 'all':
        print("\n>>> Starting Experiment 1 In Parallel: Core Algorithm Validation...")
        run_experiment_1_parallel.main()

    if args.experiment_name == 'exp2' or args.experiment_name == 'all':
        print("\n>>> Starting Experiment 2: Multi-Scheme Advantage...")
        run_experiment_2.main() # 假设你已经创建了这个文件和函数

    if args.experiment_name == 'exp2p' or args.experiment_name == 'all':
        print("\n>>> Starting Experiment 2 In Parallel: Multi-Scheme Advantage...")
        run_experiment_2_parallel.main() # 假设你已经创建了这个文件和函数

    if args.experiment_name == 'exp3' or args.experiment_name == 'all':
        print("\n>>> Starting Experiment 3: Multi-Flow Performance...")
        run_experiment_3.main() # 假设你已经创建了这个文件和函数

    if args.experiment_name == 'exp3p' or args.experiment_name == 'all':
        print("\n>>> Starting Experiment 3 In Parallel: Multi-Scheme Advantage...")
        run_experiment_3_parallel.main()  # 假设你已经创建了这个文件和函数

    if args.experiment_name == 'plot':
        print("\n>>> Generating result plots...")
        plot_results.main() # 假设你的绘图逻辑都在这个脚本里

    print("\n--- All selected tasks finished. ---")


if __name__ == "__main__":
    # 增加一个 try-except 块，可以在没有提供参数时给出提示
    try:
        # run_single_flow_experiment()
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        print("\n错误: 请提供一个实验名称来运行。")
        print("用法: python main.py [exp1|exp2|exp3|plot|all]")
        print("例如: python main.py exp1")
