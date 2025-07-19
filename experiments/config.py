# experiments/config.py
import datetime

import numpy as np

# --- 通用模拟参数 ---
NUM_RUNS = 50  # 每个数据点的统计运行次数
DEFAULT_NUM_NODES = 50
DEFAULT_AVG_DEGREE = 4
DEFAULT_P_SUPER_SWITCH = 0.5

# --- 物理参数 ---
FIBER_ATTENUATION_DB_PER_KM = 0.15
AVG_LINK_LENGTH_KM = 2.0
ERROR_RATE_LOW = 0.001
ERROR_RATE_HIGH = 0.15

# --- 算法参数 ---
EPSILON = 0.01

# --- 实验一: 自变量 ---
# np.linspace(start, stop, num_points)
# --- 通用模拟参数 ---
# NUM_RUNS = 100  # 每个数据点的统计运行次数
# DEFAULT_NUM_NODES = 100
# DEFAULT_AVG_DEGREE = 4
# DEFAULT_P_SUPER_SWITCH = 0.4
#
# # --- 物理参数 ---
# FIBER_ATTENUATION_DB_PER_KM = 0.15
# AVG_LINK_LENGTH_KM = 2.0
# ERROR_RATE_LOW = 0.001
# ERROR_RATE_HIGH = 0.15
#
# # --- 算法参数 ---
# EPSILON = 0.01

ERROR_THRESHOLDS = np.linspace(0.01, 0.25, 10)
DISCRETIZATION_DELTA = [EPSILON * error_threshold / DEFAULT_NUM_NODES for error_threshold in ERROR_THRESHOLDS]

# --- 实验二: 自变量 (占位) ---
# ...

# --- 实验三: 自变量 (占位) ---
NUM_FLOWS_LIST = list(range(5, 51, 5))

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
EXP1_RESULTS_FILE = f"experiment_1_results_{TIMESTAMP}.json"
EXP2_RESULTS_FILE = f"experiment_2_results_{TIMESTAMP}.json"