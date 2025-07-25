# experiments/plot_results.py

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 使用绝对导入
import experiments.config as config

def find_results_file(base_name: str) -> (str, str):
    """在结果目录中智能地查找最新的结果文件。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    filepath = os.path.join(project_root, 'results', base_name)

    timestamp = config.get_timestamp(filename=base_name)

    return filepath, timestamp


def find_latest_results_file(base_name: str) -> str:
    """在结果目录中智能地查找最新的结果文件。"""
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / config.PARAMS.get("RESULTS_DIR", "results")
    search_pattern = str(results_dir / f"{base_name}_*.json")

    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


def plot_single_graph(ax, x_values, results_data, y_key, y_label, title, is_percent=False, use_log_scale=False):
    """
    一个通用的辅助函数，用于绘制单张图表。
    """
    for i, (name, data) in enumerate(results_data.items()):
        y_values = data.get(y_key, [])
        if is_percent:
            y_values = [y * 100 for y in y_values]

        ax.plot(x_values, y_values,
                marker=config.PARAMS['MARKERS'][i % len(config.PARAMS['MARKERS'])],
                linestyle=config.PARAMS['LINESTYLES'][i % len(config.PARAMS['LINESTYLES'])],
                color=config.PARAMS['COLORS'][i % len(config.PARAMS['COLORS'])],
                label=name)

    ax.set_xlabel(params.get("x_label", "X-axis"), fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.legend(title="Strategy", fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if is_percent:
        ax.set_ylim(-5, 105)
        ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    else:
        ax.set_ylim(bottom=0)

    if use_log_scale:
        ax.set_ylim(bottom=0.1)
        ax.set_yscale('log')
        ax.set_ylabel(f"{y_label} (Log Scale)", fontsize=14)


def plot_experiment_results(experiment_name: str, base_filename: str, plot_configs: list):
    """
    一个通用的主绘图函数，可以为任何实验的结果数据绘图。
    """
    print(f"--- 正在为 {experiment_name} 绘制结果图表 ---")

    filepath, _ = find_results_file(base_filename)
    if not filepath:
        results_dir = config.PARAMS.get("RESULTS_DIR", "results")
        print(f"错误: 在 '{results_dir}/' 目录下找不到 '{base_filename}_*.json' 文件。")
        return

    print(f"正在从最新文件加载数据: {filepath}")
    with open(filepath, 'r') as f:
        full_data = json.load(f)

    global params  # 声明 params 为全局变量，以便 plot_single_graph 可以访问
    params = full_data["parameters"]
    results = full_data["results"]

    # --- 专业绘图风格设置 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    font_options = {'family': 'serif', 'size': 12}
    plt.rc('font', **font_options)

    # 遍历该实验需要绘制的所有图表配置
    for plot_config in plot_configs:
        fig, ax = plt.subplots(figsize=(8, 6))

        # 从参数中获取X轴数据
        x_values = params[plot_config["x_key"]]
        params["x_label"] = plot_config["x_label"]  # 临时存储xlabel

        plot_single_graph(ax, x_values, results, **plot_config["y_plot"])

        fig.tight_layout()

        # 自动保存图像
        save_path = os.path.join(os.path.dirname(filepath), f"{base_filename}_{plot_config['save_suffix']}.pdf")
        plt.savefig(save_path)
        print(f"图表已保存到: {save_path}")
        plt.show()


def plot_exp3_normalized_efficiency(data_filepath: str):
    """
    加载实验三的结果，并只绘制“单位吞-吐量拥塞”这张性价比图。
    """
    print("--- 正在绘制实验三的归一化效率图表 (性价比对比) ---")

    print(f"正在从文件加载数据: {data_filepath}")
    with open(data_filepath, 'r') as f:
        full_data = json.load(f)

    params = full_data["parameters"]
    results = full_data["results"]

    x_values = params["NUM_FLOWS_LIST"]

    # --- 1. 数据后处理：计算吞吐量和归一化指标 ---
    for name, data in results.items():
        # 计算每个数据点的平均接受流数量 (绝对吞吐量)
        data['throughput'] = [
            ratio * num_flows
            for ratio, num_flows in zip(data['accept_ratios'], x_values)
        ]

        # --- 核心计算：单位吞吐量拥塞 ---
        # Congestion per Throughput = Average Max Congestion / Number of Accepted Flows
        data['congestion_per_throughput'] = [
            congestion / throughput if throughput > 1e-6 else np.nan  # 避免除零
            for congestion, throughput in zip(data['max_congestion'], data['throughput'])
        ]

    # --- 2. 专业绘图风格设置 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    font_options = {'family': 'serif', 'size': 12}
    plt.rc('font', **font_options)

    # --- 3. 绘制图表 ---
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, data) in enumerate(results.items()):
        # 使用我们刚刚计算出的新指标来绘图
        ax.plot(x_values, data['congestion_per_throughput'],
                marker=config.PARAMS['MARKERS'][i % len(config.PARAMS['MARKERS'])],
                linestyle=config.PARAMS['LINESTYLES'][i % len(config.PARAMS['LINESTYLES'])],
                color=config.PARAMS['COLORS'][i % len(config.PARAMS['COLORS'])],
                label=name)

    ax.set_xlabel("Number of Concurrent Flow Requests", fontsize=14)
    ax.set_ylabel("Congestion per Unit of Throughput", fontsize=14)
    ax.set_title("", fontsize=16, weight='bold')
    ax.legend(title="Strategy", fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0.1)
    ax.set_yscale('log')

    fig.tight_layout()

    # 自动保存图像
    efficiency_fig_path = os.path.join(os.path.dirname(data_filepath), "exp3_normalized_efficiency.pdf")
    plt.savefig(efficiency_fig_path)
    print(f"归一化效率图已保存到: {efficiency_fig_path}")
    plt.show()


def main():
    # --- 实验一的绘图配置 ---
    exp1_plots = [
        {
            "x_key": "ERROR_THRESHOLDS",
            "x_label": "End-to-End Error Rate Threshold ($r_\\theta$)",
            "y_plot": {
                "y_key": "accept_ratios",
                "y_label": "Path Acceptance Ratio",
                "title": "",
                "is_percent": True
            },
            "save_suffix": "acceptance"
        },
        {
            "x_key": "ERROR_THRESHOLDS",
            "x_label": "End-to-End Error Rate Threshold ($r_\\theta$)",
            "y_plot": {
                "y_key": "costs",
                "y_label": "Average Path Cost",
                "title": ""
            },
            "save_suffix": "cost"
        }
    ]
    plot_experiment_results("Experiment 1", "final_experiment_1_results_2025-07-22_07-24-43_mac.json", exp1_plots)

    # --- 实验二的绘图配置 ---
    exp2_plots = [
        {
            "x_key": "ERROR_THRESHOLDS",
            "x_label": "End-to-End Error Rate Threshold ($r_\\theta$)",
            "y_plot": {
                "y_key": "accept_ratios",
                "y_label": "Path Acceptance Ratio",
                "title": "",
                "is_percent": True
            },
            "save_suffix": "acceptance"
        },
        {
            "x_key": "ERROR_THRESHOLDS",
            "x_label": "End-to-End Error Rate Threshold ($r_\\theta$)",
            "y_plot": {
                "y_key": "costs",
                "y_label": "Average Path Cost",
                "title": "",
                "use_log_scale": False  # 实验二成本差异大，使用对数坐标
            },
            "save_suffix": "cost_log"
        }
    ]
    plot_experiment_results("Experiment 2", "final_experiment_2_results_2025-07-21_16-04-16.json", exp2_plots)

    # --- 实验三的绘图配置 ---
    exp3_plots = [
        {
            "x_key": "NUM_FLOWS_LIST",
            "x_label": "Number of Concurrent Flow Requests",
            "y_plot": {
                "y_key": "accept_ratios",
                "y_label": "Flow Acceptance Ratio",
                "title": "",
                "is_percent": True
            },
            "save_suffix": "acceptance"
        },
        {
            "x_key": "NUM_FLOWS_LIST",
            "x_label": "Number of Concurrent Flow Requests",
            "y_plot": {
                "y_key": "max_congestion",
                "y_label": "Average Maximum Link Congestion",
                "title": ""
            },
            "save_suffix": "congestion"
        },
        # (可选) 归一化效率图


    ]
    plot_experiment_results("Experiment 3", "final_experiment_3_results_2025-07-24_07-58-04_mac.json", exp3_plots)


if __name__ == "__main__":
    main()
    plot_exp3_normalized_efficiency(find_results_file('final_experiment_3_results_2025-07-24_07-58-04_mac.json')[0])