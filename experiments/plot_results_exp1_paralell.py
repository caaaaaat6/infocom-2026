# experiments/plot_results.py

import json
import os
import glob
import matplotlib.pyplot as plt

# 使用绝对导入，假设你从根目录的 main.py 启动
# 如果独立运行，需要在文件末尾处理 sys.path
import config


def find_latest_results_file(base_name: str) -> str:
    """
    在结果目录中智能地查找最新的结果文件。
    """
    # 动态构建项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(project_root, config.PARAMS["RESULTS_DIR"])

    search_pattern = os.path.join(results_dir, f"{base_name}_*.json")

    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        return None

    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def plot_single_experiment_results(experiment_name: str, base_filename: str):
    """
    一个通用的绘图函数，可以为任何实验的结果数据绘图。
    - experiment_name: 实验的名称，用于图表标题 (e.g., "Experiment 1")
    - base_filename: 结果文件的基础名称 (e.g., "experiment_1_results")
    """
    print(f"--- 正在为 {experiment_name} 绘制结果图表 ---")

    base_filename = "experiment_1_results_2025-07-21_03-33-27.json"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    filepath = os.path.join(project_root, 'results', base_filename)

    timestamp = config.get_timestamp(filename=base_filename)

    if not filepath:
        results_dir = config.PARAMS.get("RESULTS_DIR", "results")
        print(f"错误: 在 '{results_dir}/' 目录下找不到任何 '{base_filename}_*.json' 文件。")
        print(f"请先成功运行相应的 run_{base_filename}.py。")
        return

    print(f"正在从最新文件加载数据: {filepath}")
    with open(filepath, 'r') as f:
        full_data = json.load(f)

    params = full_data["parameters"]
    results = full_data["results"]
    x_values = params["ERROR_THRESHOLDS"]

    # --- 专业绘图风格设置 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    font_options = {'family': 'serif', 'size': 12}
    plt.rc('font', **font_options)

    # --- 图一：路径接受率对比 ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(x_values, data['accept_ratios'],
                 marker=config.PARAMS['MARKERS'][i % len(config.PARAMS['MARKERS'])],
                 linestyle=config.PARAMS['LINESTYLES'][i % len(config.PARAMS['LINESTYLES'])],
                 color=config.PARAMS['COLORS'][i % len(config.PARAMS['COLORS'])],
                 label=name)

    ax1.set_xlabel("End-to-End Error Rate Threshold ($r_\\theta$)", fontsize=14)
    ax1.set_ylabel("Path Acceptance Ratio", fontsize=14)
    ax1.set_title(f"{experiment_name}: Feasibility Comparison", fontsize=16, weight='bold')
    ax1.legend(title="Routing Strategy", fontsize=11)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_ylim(-0.05, 1.05)
    # 将Y轴格式化为百分比
    ax1.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    fig1.tight_layout()

    # acceptance_fig_path = os.path.join(os.path.dirname(filepath), f"{base_filename}_acceptance.pdf")
    acceptance_fig_path = config.get_experiment_1_acceptance_pdf_name(timestamp=timestamp)
    plt.savefig(acceptance_fig_path)
    print(f"接受率对比图已保存到: {acceptance_fig_path}")
    plt.show()

    # --- 图二：平均路径成本对比 ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i, (name, data) in enumerate(results.items()):
        ax2.plot(x_values, data['costs'],
                 marker=config.PARAMS['MARKERS'][i % len(config.PARAMS['MARKERS'])],
                 linestyle=config.PARAMS['LINESTYLES'][i % len(config.PARAMS['LINESTYLES'])],
                 color=config.PARAMS['COLORS'][i % len(config.PARAMS['COLORS'])],
                 label=name)

    ax2.set_xlabel("End-to-End Error Rate Threshold ($r_\\theta$)", fontsize=14)
    ax2.set_ylabel("Average Path Cost", fontsize=14)
    ax2.set_title(f"{experiment_name}: Cost Comparison", fontsize=16, weight='bold')
    ax2.legend(title="Routing Strategy", fontsize=11)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig2.tight_layout()

    # cost_fig_path = os.path.join(os.path.dirname(filepath), f"{base_filename}_cost.pdf")
    cost_fig_path = config.get_experiment_1_cost_pdf_name(timestamp=timestamp)
    plt.savefig(cost_fig_path)
    print(f"成本对比图已保存到: {cost_fig_path}")
    plt.show()


def main():
    plot_single_experiment_results(
        experiment_name="Experiment 1",
        base_filename="experiment_1_results"
    )
    # 当实验二跑完后，你只需要取消下面的注释即可
    # plot_single_experiment_results(
    #     experiment_name="Experiment 2",
    #     base_filename="experiment_2_results"
    # )


if __name__ == "__main__":
    main()