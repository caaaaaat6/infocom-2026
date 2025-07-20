# experiments/plot_results.py

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# 导入配置文件，我们将从中获取样式和路径信息
import config


def find_latest_results_file(base_name: str) -> str:
    """
    在结果目录中查找具有特定基础名称的最新结果文件。
    """
    # 从 config 中获取结果目录的路径
    results_dir = config.PARAMS.get("RESULTS_DIR", "results")
    search_pattern = os.path.join(results_dir, f"{base_name}_*.json")

    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        return None

    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


# --- 新增：专门为实验二设计的绘图函数 ---
def plot_experiment_2():
    """
    加载最新的实验二结果，并生成两张对比图：
    1. 平均路径成本 vs. 错误率阈值
    2. 路径接受率 vs. 错误率阈值
    """
    print("--- 正在绘制实验二的结果图表 (多编码方案对比) ---")

    base_filename = "experiment_2_results_2025-07-20_14-04-06.json"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    filepath = os.path.join(project_root, 'results', base_filename)

    timestamp = config.get_timestamp(filename=base_filename)

    if not filepath:
        print(f"错误: 在 '{config.PARAMS['RESULTS_DIR']}/' 目录下找不到任何 '{base_filename}_*.json' 文件。")
        print("请先成功运行 run_experiment_2.py。")
        return

    print(f"正在从最新文件加载数据: {filepath}")
    with open(filepath, 'r') as f:
        full_data = json.load(f)

    # 从加载的数据中分别提取参数和结果
    params = full_data["parameters"]
    results = full_data["results"]

    # 从参数字典中获取X轴数据
    x_values = params["ERROR_THRESHOLDS"]

    # --- 图一：平均路径成本对比 ---
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用一个美观的绘图风格
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    for i, (name, data) in enumerate(results.items()):
        ax1.plot(x_values, data['costs'],
                 marker=config.PARAMS['MARKERS'][i % len(config.PARAMS['MARKERS'])],
                 linestyle=config.PARAMS['LINESTYLES'][i % len(config.PARAMS['LINESTYLES'])],
                 color=config.PARAMS['COLORS'][i % len(config.PARAMS['COLORS'])],
                 label=name)

    ax1.set_xlabel("End-to-End Error Rate Threshold ($r_\\theta$)", fontsize=12)
    ax1.set_ylabel("Average Path Cost (Log Scale)", fontsize=12)
    ax1.set_title("Experiment 2: Cost Comparison of Encoding Strategies", fontsize=14, weight='bold')
    ax1.legend(title="Routing Strategy")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_yscale('log')  # 成本差异可能很大，对数坐标轴效果好

    # 自动保存图像
    cost_fig_path = config.get_experiment_2_cost_pdf_name(timestamp=timestamp)
    plt.savefig(cost_fig_path, bbox_inches='tight')
    print(f"成本对比图已保存到: {cost_fig_path}")
    plt.show()

    # --- 图二：路径接受率对比 ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for i, (name, data) in enumerate(results.items()):
        ax2.plot(x_values, data['accept_ratios'],
                 marker=config.PARAMS['MARKERS'][i % len(config.PARAMS['MARKERS'])],
                 linestyle=config.PARAMS['LINESTYLES'][i % len(config.PARAMS['LINESTYLES'])],
                 color=config.PARAMS['COLORS'][i % len(config.PARAMS['COLORS'])],
                 label=name)

    ax2.set_xlabel("End-to-End Error Rate Threshold ($r_\\theta$)", fontsize=12)
    ax2.set_ylabel("Path Acceptance Ratio", fontsize=12)
    ax2.set_title("Experiment 2: Feasibility Comparison of Encoding Strategies", fontsize=14, weight='bold')
    ax2.legend(title="Routing Strategy")
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_ylim(0, 1.05)  # 接受率在 0 到 1 之间

    # 自动保存图像
    acceptance_fig_path = config.get_experiment_2_acceptance_pdf_name(timestamp=timestamp)
    plt.savefig(acceptance_fig_path, bbox_inches='tight')
    print(f"接受率对比图已保存到: {acceptance_fig_path}")
    plt.show()


def main():
    # plot_experiment_1()
    plot_experiment_2()


if __name__ == "__main__":
    # 为了能独立运行，需要正确处理路径
    import sys

    # 将项目根目录 (infocom-2026/) 添加到查找路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    main()