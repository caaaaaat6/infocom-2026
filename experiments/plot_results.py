# experiments/plot_results.py
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import experiments.config as config  # 从同级目录的config导入配置


def plot_experiment_1():
    """读取实验一的结果并绘图。"""
    print("--- 正在绘制实验一的结果图表 ---")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 将项目根目录和结果文件夹、文件名拼接起来
    filepath = os.path.join(project_root, config.RESULTS_DIR, 'experiment_1_results_2025-07-19_16-20-24.json')

    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"错误: 结果文件 {filepath} 未找到。请先运行 run_experiment_1.py。")
        return

    # 获取X轴数据
    x_values = config.ERROR_THRESHOLDS

    # --- 图一：平均路径成本 vs. 错误率阈值 ---
    plt.figure(figsize=(8, 6))
    for i, (name, data) in enumerate(results.items()):
        plt.plot(x_values, data['costs'],
                 marker=config.MARKERS[i],
                 linestyle=config.LINESTYLES[i],
                 color=config.COLORS[i],
                 label=name)

    plt.xlabel("End-to-End Error Rate Threshold ($r_\\theta$)")
    plt.ylabel("Average Path Cost")
    plt.title("Experiment 1: Path Cost vs. Error Rate Threshold")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, config.RESULTS_DIR, "exp1_cost.pdf"))  # 保存为PDF，用于论文
    plt.show()

    # --- 图二：路径接受率 vs. 错误率阈值 ---
    plt.figure(figsize=(8, 6))
    for i, (name, data) in enumerate(results.items()):
        plt.plot(x_values, data['accept_ratios'],
                 marker=config.MARKERS[i],
                 linestyle=config.LINESTYLES[i],
                 color=config.COLORS[i],
                 label=name)

    plt.xlabel("End-to-End Error Rate Threshold ($r_\\theta$)")
    plt.ylabel("Path Acceptance Ratio")
    plt.title("Experiment 1: Acceptance Ratio vs. Error Rate Threshold")
    plt.ylim(0, 1.05)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))  # 将Y轴格式化为百分比
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, config.RESULTS_DIR, "exp1_acceptance.pdf"))
    plt.show()


def main():
    plot_experiment_1()
    # 未来可以调用 plot_experiment_2(), plot_experiment_3()


if __name__ == "__main__":
    main()