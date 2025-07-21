# experiments/config.py
import datetime
import os

import numpy as np

# --- 通用模拟参数 ---
NUM_RUNS = 50 # 每个数据点的统计运行次数
DEFAULT_NUM_NODES = 500
DEFAULT_AVG_DEGREE = 4
DEFAULT_P_SUPER_SWITCH = 0.4

# --- 物理参数 ---
FIBER_ATTENUATION_DB_PER_KM = 0.15
AVG_LINK_LENGTH_KM = 2.0
ERROR_RATE_LOW = 0.001
ERROR_RATE_HIGH = 0.15

# --- 算法参数 ---
EPSILON = 0.01

# --- 实验一: 自变量 ---

ERROR_THRESHOLDS = np.linspace(0.01, 0.25, 10)
DISCRETIZATION_DELTA = [EPSILON * error_threshold / DEFAULT_NUM_NODES for error_threshold in ERROR_THRESHOLDS]

# --- 实验二: 自变量 (占位) ---
# ...

# --- 实验三: 自变量 (占位) ---
POOL_SIZE = 3
DEFAULT_R_THETA = 0.05
DEFAULT_DELTA = 0.01
NUM_FLOWS_LIST = list(range(5, 51, 5))
NUM_FLOWS_LIST = list(range(1, 11, 2))

# --- 绘图参数 ---
MARKERS = ['o', 's', 'd', '^', 'v', 'p']
LINESTYLES = ['-', '--', ':', '-.']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# --- 结果文件路径 ---
RESULTS_DIR = "results"

# --- 在 config.py 中直接生成带时间戳的文件名 (简单直接) ---
# 获取当前时间并格式化成 "YYYY-MM-DD_HH-MM-SS" 的形式
# 示例: "2025-07-20_15-30-55"
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 将基础文件名和时间戳拼接起来
EXP1_RESULTS_FILE = f"experiment_1_results"
EXP2_RESULTS_FILE = f"experiment_2_results"
EXP3_RESULTS_FILE = f"experiment_2_results"

# 将参数组织在一个字典中，方便整体引用和保存
PARAMS = {
    # --- 通用模拟参数 ---
    'NUM_RUNS': NUM_RUNS,  # 每个数据点的统计运行次数
    'DEFAULT_NUM_NODES': DEFAULT_NUM_NODES,
    'DEFAULT_AVG_DEGREE': DEFAULT_AVG_DEGREE,
    'DEFAULT_P_SUPER_SWITCH': DEFAULT_P_SUPER_SWITCH,

    # --- 物理参数 ---
    'FIBER_ATTENUATION_DB_PER_KM': FIBER_ATTENUATION_DB_PER_KM,
    'AVG_LINK_LENGTH_KM': AVG_LINK_LENGTH_KM,
    'ERROR_RATE_LOW': ERROR_RATE_LOW,
    'ERROR_RATE_HIGH': ERROR_RATE_HIGH,

    # --- 算法参数 ---
    'EPSILON': EPSILON,

    # --- 实验一: 自变量 ---
    'ERROR_THRESHOLDS': list(ERROR_THRESHOLDS),
    'DISCRETIZATION_DELTA': DISCRETIZATION_DELTA,

    # --- 实验二: 自变量 (占位) ---
    # ...

    # --- 实验三: 自变量 (占位) ---
    'POOL_SIZE': POOL_SIZE,
    'DEFAULT_R_THETA': DEFAULT_R_THETA,
    'DEFAULT_DELTA': DEFAULT_DELTA,
    'NUM_FLOWS_LIST': NUM_FLOWS_LIST,

    # --- 绘图参数 ---
    'MARKERS': MARKERS,
    'LINESTYLES': LINESTYLES,
    'COLORS': COLORS,

    # --- 结果文件路径 ---
    'RESULTS_DIR': RESULTS_DIR,
}


def get_timestamped_filename(base_name: str, extension: str = "json") -> str:
    """
    为一个基础文件名添加当前时间戳，确保文件名合法。

    这个函数会生成一个类似 "base_name_YYYY-MM-DD_HH-MM-SS.extension" 的字符串。

    参数:
    - base_name (str): 文件的基础名称，例如 "experiment_1_results"。
    - extension (str): 文件的扩展名，默认为 "json"。

    返回:
    - str: 带有时间戳的完整文件名。
    """
    # 1. 获取当前时间
    now = datetime.datetime.now()

    # 2. 将时间格式化为清晰、安全（不含非法字符如冒号）的字符串
    # 格式: 年-月-日_时-分-秒
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # 3. 将基础名称、时间戳和扩展名拼接起来
    # f-string 是现代Python中拼接字符串的最佳方式
    return f"{base_name}_{timestamp}.{extension}"


def get_timestamp(filename: str):
    timestamp = filename.split('results')[1]
    return timestamp


def get_experiment_acceptance_pdf_name(experiment_num: int, timestamp):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    acceptance_fig_path = os.path.join(project_root, PARAMS['RESULTS_DIR'], f"exp{experiment_num}_acceptance_comparison_{timestamp}.pdf")
    return acceptance_fig_path


def get_experiment_cost_pdf_name(experiment_num: int, timestamp):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cost_fig_path = os.path.join(project_root, PARAMS['RESULTS_DIR'], f"exp{experiment_num}_cost_comparison_{timestamp}.pdf")
    return cost_fig_path


def get_experiment_1_acceptance_pdf_name(timestamp):
    return get_experiment_acceptance_pdf_name(experiment_num=1, timestamp=timestamp)


def get_experiment_2_acceptance_pdf_name(timestamp):
    return get_experiment_acceptance_pdf_name(experiment_num=2, timestamp=timestamp)


def get_experiment_3_acceptance_pdf_name(timestamp):
    return get_experiment_acceptance_pdf_name(experiment_num=3, timestamp=timestamp)


def get_experiment_1_cost_pdf_name(timestamp):
    return get_experiment_cost_pdf_name(experiment_num=1, timestamp=timestamp)


def get_experiment_2_cost_pdf_name(timestamp):
    return get_experiment_cost_pdf_name(experiment_num=2, timestamp=timestamp)


def get_experiment_3_cost_pdf_name(timestamp):
    return get_experiment_cost_pdf_name(experiment_num=3, timestamp=timestamp)
