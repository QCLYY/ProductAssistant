"""稀疏向量 L2 归一化工具

L2 归一化将向量"拉伸"到单位长度（L2 范数 = 1），
使 Milvus 的 IP（内积）度量等价于余弦相似度。
"""

from typing import Dict
import numpy as np


def normalize_sparse_vector(sparse_dict: Dict[int, float]) -> Dict[int, float]:
    """
    对稀疏向量做 L2 归一化（只处理非零维度，不影响零维度）。

    当 L2 范数 < 1e-9 时视为零向量，原样返回。

    Args:
        sparse_dict: {token_id: weight} 原始稀疏向量

    Returns:
        L2 归一化后的稀疏向量（范数 = 1.0）
    """
    if not sparse_dict:
        return {}

    values = np.array(list(sparse_dict.values()), dtype=np.float64)
    l2_norm = np.linalg.norm(values)

    if l2_norm < 1e-9:
        return sparse_dict

    normalized_values = values / l2_norm
    return dict(zip(sparse_dict.keys(), normalized_values))
