# encoding_schemes.py
import csv
import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


def load_fj_coeffs_from_csv(filepath: str, distance: int) -> Dict[int, float]:
    """
    从一个包含多列数据的 CSV 文件中，为指定码距(distance)的编码方案加载 f_j 系数。

    CSV文件格式要求:
    - 第一行是表头，例如: 'j,f_j [d=9],f_j [d=7],f_j [d=5],f_j [d=3]'
    - 第一列是错误重量 j。
    - 后续列是不同码距 d 对应的 f_j 值。

    参数:
    - filepath: CSV 文件的路径。
    - distance: 想要提取数据的码距 d (例如, 3, 5, 7, 9)。

    返回:
    - 一个字典，其中 key 是整数 j，value 是浮点数 f_j。
    """
    f_j_coeffs = {}

    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            # 使用 csv.reader 来处理文件，能更好地处理逗号和引号
            reader = csv.reader(f)

            # 1. 读取并解析表头
            header = next(reader)

            # 2. 找到对应码距 d 的列索引
            # 我们要寻找像 'f_j [d=5]' 这样的列名
            target_column_name = f'f_j [d={distance}]'
            try:
                target_idx = header.index(target_column_name)
            except ValueError:
                print(f"错误: 在文件 '{filepath}' 的表头中找不到列 '{target_column_name}'。")
                raise

            # 3. 逐行读取数据
            for row in reader:
                # 忽略空行
                if not row:
                    continue

                try:
                    # 提取 j 和 对应的 f_j
                    j = int(row[0])

                    # 检查目标列是否有数据
                    if target_idx < len(row) and row[target_idx]:
                        f_j = float(row[target_idx])
                        f_j_coeffs[j] = f_j
                    # 如果目标列为空（比如 d=3 在 j>13 时），我们就停止读取
                    else:
                        break

                except (ValueError, IndexError):
                    # 如果行格式不正确，打印警告并跳过
                    print(f"警告: 无法解析行 '{','.join(row)}'，已跳过。")

    except FileNotFoundError:
        print(f"错误: 找不到 f_j 系数文件 '{filepath}'。")
        raise

    return f_j_coeffs


def load_fj_coeffs_from_csv(filepath: str, distance: int) -> Dict[int, float]:
    """
    从一个包含多列数据的 CSV 文件中，为指定码距(distance)的编码方案加载 f_j 系数。

    CSV文件格式要求:
    - 第一行是表头，例如: 'j,f_j [d=9],f_j [d=7],f_j [d=5],f_j [d=3]'
    - 第一列是错误重量 j。
    - 后续列是不同码距 d 对应的 f_j 值。

    参数:
    - filepath: CSV 文件的路径。
    - distance: 想要提取数据的码距 d (例如, 3, 5, 7, 9)。

    返回:
    - 一个字典，其中 key 是整数 j，value 是浮点数 f_j。
    """
    f_j_coeffs = {}

    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            # 使用 csv.reader 来处理文件，能更好地处理逗号和引号
            reader = csv.reader(f)

            # 1. 读取并解析表头
            header = next(reader)

            # 2. 找到对应码距 d 的列索引
            # 我们要寻找像 'f_j [d=5]' 这样的列名
            target_column_name = f'f_j [d={distance}]'
            try:
                target_idx = header.index(target_column_name)
            except ValueError:
                print(f"错误: 在文件 '{filepath}' 的表头中找不到列 '{target_column_name}'。")
                raise

            # 3. 逐行读取数据
            for row in reader:
                # 忽略空行
                if not row:
                    continue

                try:
                    # 提取 j 和 对应的 f_j
                    j = int(row[0])

                    # 检查目标列是否有数据
                    if target_idx < len(row) and row[target_idx]:
                        f_j = float(row[target_idx])
                        f_j_coeffs[j] = f_j
                    # 如果目标列为空（比如 d=3 在 j>13 时），我们就停止读取
                    else:
                        break

                except (ValueError, IndexError):
                    # 如果行格式不正确，打印警告并跳过
                    print(f"警告: 无法解析行 '{','.join(row)}'，已跳过。")

    except FileNotFoundError:
        print(f"错误: 找不到 f_j 系数文件 '{filepath}'。")
        raise

    return f_j_coeffs


@dataclass
class EncodingScheme:
    """代表一个量子纠错码方案。"""

    name: str  # 编码方案的名称
    n: int  # 物理比特数
    k: int  # 逻辑比特数
    d: int  # 码距
    cost: float  # 执行一次解码操作的成本
    fj_filepath: str = None
    # f_j 系数: 给定 j 个物理错误时，发生逻辑错误的条件概率
    f_j_coeffs: Dict[int, float] = field(default_factory=dict, init=False)
    t_cycle: float = 0.0

    def __init__(self, name, n, k, d, cost, fj_filepath, t_cycle):
        self.name = name
        self.n = n
        self.k = k
        self.d = d
        self.cost = cost
        self.fj_filepath = fj_filepath
        self.f_j_coeffs = load_fj_coeffs_from_csv(fj_filepath, d)
        self.t_cycle = t_cycle
        # self.__post_init__()

    def __post_init__(self):
        # --- 重要提示 ---
        # TODO
        # 这只是一个 f_j 的“玩具模型”。对于真实的论文，你必须用
        # 实际数值模拟得到的结果来替换它。
        # 这个模型假设:
        # - 在 t = floor((d-1)/2) 个错误以内，可以完美纠正
        # - 超过 t 个错误后，逻辑错误率线性增长
        t = (self.d - 1) // 2
        self.f_j_coeffs = np.zeros(self.n + 1)
        for j in range(t + 1, self.n + 1):
            # 确保概率不超过1
            self.f_j_coeffs[j] = min(1.0, (j - t) / (self.n - t))

    def calculate_logical_error_rate(self, p_physical: float) -> float:
        """
        使用二项分布公式计算逻辑错误率。
        输入: p_physical - 物理信道的错误率
        返回: 逻辑错误率
        """
        if p_physical == 0:
            return 0.0

        p_logical = 0.0
        for j in range(1, self.n + 1):
            # 发生 j 个错误的二项分布概率
            binom_prob = (math.comb(self.n, j) *
                          (p_physical ** j) *
                          ((1 - p_physical) ** (self.n - j)))
            # 累加逻辑错误率
            p_logical += self.f_j_coeffs[j] * binom_prob

        return p_logical


# --- 用于实验的示例编码方案 ---
# 一个低成本、低保护度的编码方案
SCHEME_A = EncodingScheme(name="7比特斯坦码", n=7, k=1, d=3, cost=10, fj_filepath='data/Surface_L.csv', t_cycle=10e-6)

# 一个高成本、高保护度的编码方案
SCHEME_B = EncodingScheme(name="145比特表面码", n=145, k=1, d=9, cost=50, fj_filepath='data/Surface_L.csv', t_cycle=20e-6)

# 超级交换机上可用的编码方案组合
DEFAULT_SCHEMES = [SCHEME_A, SCHEME_B]
# DEFAULT_SCHEMES = [SCHEME_A]
DEFAULT_SCHEMES = [SCHEME_B]

# --- 实验二所需的编码方案 ---
# 假设你已经为它们准备好了 f_j 系数文件
SCHEME_41_1_3 = EncodingScheme(name="Surface_41_1_5", n=41, k=1, d=5, cost=8,
                              t_cycle=8e-6,
                              fj_filepath="data/Surface_L.csv.txt")

SCHEME_85_1_7 = EncodingScheme(name="Surface_85_1_7", n=85, k=1, d=7, cost=10,
                              t_cycle=10e-6,
                              fj_filepath="data/Surface_L.csv.txt")

SCHEME_145_1_9 = EncodingScheme(name="Surface_145_1_9", n=145, k=1, d=9, cost=50,
                               t_cycle=50e-6,
                               fj_filepath="data/Surface_L.csv.txt")

# --- 定义编码组合 ---
# 单一方案列表，用于基线
SINGLE_SCHEME_41_1_3 = [SCHEME_41_1_3]
SINGLE_SCHEME_85_1_7 = [SCHEME_85_1_7]
SINGLE_SCHEME_145_1_9 = [SCHEME_145_1_9]

# 多方案组合，用于你的 Proposed 算法
MULTI_SCHEME_PORTFOLIO = [SCHEME_41_1_3, SCHEME_85_1_7, SCHEME_145_1_9]