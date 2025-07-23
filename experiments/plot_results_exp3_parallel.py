# experiments/plot_results.py

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# 导入配置文件，我们将从中获取样式和路径信息
import config


def find_latest_results_file(base_name: str) -> str:
    """在结果目录中智能地查找最新的结果文件。"""
    # ... (这个函数保持不变) ...
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(project_root, config.PARAMS["RESULTS_DIR"])
    search_pattern = os.path.join(results_dir, f"{base_name}_*.json")
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


# --- (plot_experiment_1 和 plot_experiment_2 函数) ---
# ...


# --- 新增：专门为实验三设计的绘图函数 ---
def plot_experiment_3(base_filename):
    """
    加载最新的实验三结果，并生成两张对比图：
    1. 最大链路拥塞 vs. 并发流数量
    2. 流接受率 vs. 并发流数量
    """
    print("--- 正在绘制实验三的结果图表 (多流性能对比) ---")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    filepath = os.path.join(project_root, 'results', base_filename)

    timestamp = config.get_timestamp(filename=base_filename)

    if not filepath:
        results_dir = config.PARAMS.get("RESULTS_DIR", "results")
        print(f"错误: 在 '{results_dir}/' 目录下找不到任何 '{base_filename}_*.json' 文件。")
        return

    print(f"正在从最新文件加载数据: {filepath}")
    with open(filepath, 'r') as f:
        full_data = json.load(f)

    params = full_data["parameters"]
    results = full_data["results"]

    # 从参数字典中获取X轴数据
    x_values = params["NUM_FLOWS_LIST"]

    # --- 专业绘图风格设置 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    font_options = {'family': 'serif', 'size': 12}
    plt.rc('font', **font_options)

    # --- 图一：最大链路拥塞对比 ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    for i, (name, data) in enumerate(results.items()):
        ax1.plot(x_values, data['max_congestion'],
                 marker=config.PARAMS['MARKERS'][i % len(config.PARAMS['MARKERS'])],
                 linestyle=config.PARAMS['LINESTYLES'][i % len(config.PARAMS['LINESTYLES'])],
                 color=config.PARAMS['COLORS'][i % len(config.PARAMS['COLORS'])],
                 label=name)

    ax1.set_xlabel("Number of Concurrent Flows ($m$)", fontsize=14)
    ax1.set_ylabel("Average Maximum Link Congestion", fontsize=14)
    ax1.set_title("Experiment 3: Load Balancing Comparison", fontsize=16, weight='bold')
    ax1.legend(title="Routing Strategy", fontsize=11)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    # 设置Y轴从0开始
    ax1.set_ylim(bottom=0)

    # 自动保存图像
    congestion_fig_path = config.get_experiment_3_congestion_pdf_name(timestamp=timestamp)
    plt.savefig(congestion_fig_path, bbox_inches='tight')
    print(f"拥塞对比图已保存到: {congestion_fig_path}")
    plt.show()

    # --- 图二：流接受率对比 ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for i, (name, data) in enumerate(results.items()):
        # 将接受率转换为百分比
        accept_ratios_percent = [r * 100 for r in data['acceptance_ratio']]
        ax2.plot(x_values, accept_ratios_percent,
                 marker=config.PARAMS['MARKERS'][i % len(config.PARAMS['MARKERS'])],
                 linestyle=config.PARAMS['LINESTYLES'][i % len(config.PARAMS['LINESTYLES'])],
                 color=config.PARAMS['COLORS'][i % len(config.PARAMS['COLORS'])],
                 label=name)

    ax2.set_xlabel("Number of Concurrent Flows ($m$)", fontsize=14)
    ax2.set_ylabel("Flow Acceptance Ratio (%)", fontsize=14)
    ax2.set_title("Experiment 3: Throughput Comparison", fontsize=16, weight='bold')
    ax2.legend(title="Routing Strategy", fontsize=11)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_ylim(-5, 105)  # Y轴范围设为-5%到105%，留出边距
    # 将Y轴格式化为百分比
    ax2.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))

    # 自动保存图像
    acceptance_fig_path = config.get_experiment_3_acceptance_pdf_name(timestamp=timestamp)
    plt.savefig(acceptance_fig_path, bbox_inches='tight')
    print(f"接受率对比图已保存到: {acceptance_fig_path}")
    plt.show()


def main():
    # plot_experiment_1()
    # plot_experiment_2()
    plot_experiment_3(base_filename='experiment_3_results_2025-07-23_00-26-32.json')


if __name__ == "__main__":
    # 允许独立运行
    main()