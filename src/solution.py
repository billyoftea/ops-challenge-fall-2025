import pandas as pd
import numpy as np
from numba import njit
from joblib import Parallel, delayed


@njit
def rolling_rank_numba(values, window):
    """
    使用 Numba JIT 加速的滚动排名计算
    
    对每个位置 i，计算窗口内当前值的百分位排名
    """
    n = len(values)
    result = np.empty(n, dtype=np.float32)
    
    for i in range(n):
        # 确定窗口起始位置
        start = max(0, i - window + 1)
        window_data = values[start:i+1]
        current_val = values[i]
        
        # 计算小于等于当前值的数量
        rank = np.sum(window_data <= current_val)
        result[i] = rank / len(window_data)
    
    return result


def process_group(group_data, window):
    """处理单个 symbol 的函数，用于并行计算"""
    return rolling_rank_numba(group_data, window)


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    """
    高性能滚动排名计算
    
    优化策略：
    1. 使用 Numba JIT 编译加速核心计算
    2. 使用 joblib 多进程并行处理不同的 symbol
    3. 避免 pandas apply() 的性能瓶颈
    """
    # 读取数据
    df = pd.read_parquet(input_path)
    
    # 为每个 symbol 分配一个组 ID，保持原始顺序
    df['_group_id'] = df.groupby('symbol', sort=False).ngroup()
    
    # 提取每个 symbol 的数据（包含组 ID 和索引信息）
    group_data_list = []
    for group_id, (symbol, group) in enumerate(df.groupby('symbol', sort=False)):
        group_data_list.append((group_id, group.index.values, group['Close'].values))
    
    # 并行处理所有 symbol（使用所有可用 CPU 核心）
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(process_group)(close_data, window) 
        for _, _, close_data in group_data_list
    )
    
    # 创建结果数组，按原始索引顺序填充
    final_result = np.empty(len(df), dtype=np.float32)
    for (group_id, indices, _), group_result in zip(group_data_list, results):
        final_result[indices] = group_result
    
    return final_result[:, None]  # must be [N, 1]