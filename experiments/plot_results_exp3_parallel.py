# experiments/plot_results.py

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# 导入配置文件，我们将从中获取样式和路径信息
import config


def find_results_file(base_name: str) -> (str, str):
    """在结果目录中智能地查找最新的结果文件。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    filepath = os.path.join(project_root, 'results', base_name)

    timestamp = config.get_timestamp(filename=base_name)

    return filepath, timestamp



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


def plot_experiment_3_congetsion_per_throughput(data_filepath: str, timestamp: str):
    """
    加载实验三的结果，并生成三张对比图，包括归一化的“单位吞吐量拥塞”。
    """
    print("--- 正在绘制实验三的结果图表 (多流性能对比) ---")

    print(f"正在从文件加载数据: {data_filepath}")
    with open(data_filepath, 'r') as f:
        full_data = json.load(f)

    params = full_data["parameters"]
    results = full_data["results"]

    x_values = params["NUM_FLOWS_LIST"]
    num_runs = params["NUM_RUNS"]

    # --- 数据后处理：计算吞吐量和归一化指标 ---
    for name, data in results.items():
        # 计算每个数据点的平均接受流数量 (吞吐量)
        # 注意：acceptance_ratio 是 (accepted_flows / total_flows)
        # 所以 accepted_flows = acceptance_ratio * total_flows
        data['throughput'] = [ratio * num_flows for ratio, num_flows in zip(data['acceptance_ratio'], x_values)]

        # --- 核心计算：单位吞吐量拥塞 ---
        # Congestion per Throughput = Average Max Congestion / Number of Accepted Flows
        # 我们需要处理吞吐量为0的情况，避免除零错误
        data['congestion_per_throughput'] = [
            congestion / throughput if throughput > 0 else np.nan
            for congestion, throughput in zip(data['max_congestion'], data['throughput'])
        ]

    # --- 专业绘图风格设置 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    font_options = {'family': 'serif', 'size': 12}
    plt.rc('font', **font_options)

    # --- 图一：流接受率对比 (有效性) ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(x_values, data['acceptance_ratio'],
                 marker=config.PARAMS['MARKERS'][i], label=name)
    ax1.set_xlabel("Number of Concurrent Flow Requests ($m$)", fontsize=14)
    ax1.set_ylabel("Flow Acceptance Ratio", fontsize=14)
    ax1.set_title("Experiment 3: Effectiveness of Routing Strategies", fontsize=16, weight='bold')
    ax1.legend(title="Strategy", fontsize=11)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_ylim(-0.05, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    fig1.tight_layout()
    plt.savefig(config.get_experiment_3_acceptance_pdf_name(timestamp=timestamp))
    plt.show()

    # --- 图二：平均最大拥塞对比 (原始效率) ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i, (name, data) in enumerate(results.items()):
        ax2.plot(x_values, data['max_congestion'],
                 marker=config.PARAMS['MARKERS'][i], label=name)
    ax2.set_xlabel("Number of Concurrent Flow Requests ($m$)", fontsize=14)
    ax2.set_ylabel("Average Maximum Link Congestion", fontsize=14)
    ax2.set_title("Experiment 3: Raw Load Balancing Comparison", fontsize=16, weight='bold')
    ax2.legend(title="Strategy", fontsize=11)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_ylim(bottom=0)
    fig2.tight_layout()
    plt.savefig(config.get_experiment_3_congestion_pdf_name(timestamp=timestamp))
    plt.show()

    # --- 图三：单位吞吐量拥塞 (性价比) ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for i, (name, data) in enumerate(results.items()):
        ax3.plot(x_values, data['congestion_per_throughput'],
                 marker=config.PARAMS['MARKERS'][i], label=name)
    ax3.set_xlabel("Number of Concurrent Flow Requests ($m$)", fontsize=14)
    ax3.set_ylabel("Congestion per Unit of Throughput", fontsize=14)
    ax3.set_title("Experiment 3: Normalized Efficiency of Routing Strategies", fontsize=16, weight='bold')
    ax3.legend(title="Strategy", fontsize=11)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax3.set_yscale('log')
    ax3.set_ylim(bottom=0.1)
    fig3.tight_layout()
    plt.savefig(config.get_experiment_3_congestion_per_throughput_pdf_name(timestamp=timestamp))
    plt.show()


def main():
    filename, timestamp = find_results_file("final_time_experiment_3_results_2025-07-30_12-36-45.json")
    # plot_experiment_3(base_filename='experiment_3_results_2025-07-23_00-26-32.json')
    plot_experiment_3_congetsion_per_throughput(data_filepath=filename, timestamp=timestamp)


if __name__ == "__main__":
    # 允许独立运行
    main()