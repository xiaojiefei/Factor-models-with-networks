"""
网络矩阵工具函数

包含行标准化、加权网络生成等实用函数
"""

import numpy as np
from typing import List


def row_normalize(W: np.ndarray) -> np.ndarray:
    """
    行标准化: 使矩阵每一行的元素之和为1 (对角线为0)

    为什么需要行标准化?
    ------------------
    1. 数值稳定性: 防止 A = I - ΛW 出现奇异(行列式接近0)
    2. 参数可解释: Λ = diag(ρ) 中 ρ∈[0,1] 有明确含义(反应强度比例)
    3. 可比性: 不同资产度数不同,标准化后可在同一尺度比较

    示例:
        原始矩阵:           行标准化后:
        [[0, 1, 1],        [[0,   0.5, 0.5],   (第0行和=2,每个元素÷2)
         [1, 0, 0],    ->   [1,   0,   0  ],   (第1行和=1,保持不变)
         [0, 1, 0]]         [0,   1,   0  ]]   (第2行和=1,保持不变)

    注意:
        - 对角线元素设为0(无自环)
        - 孤立节点(行全为0)保持全0

    Args:
        W: K×K 的邻接矩阵 (可以是有权或无权)

    Returns:
        W_norm: 行标准化后的矩阵 (每行和为1或0)
    """
    W_norm = W.copy()
    np.fill_diagonal(W_norm, 0)  # 确保对角线为0(无自环)
    row_sums = W_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除以0(孤立节点)
    W_norm = W_norm / row_sums
    return W_norm


def check_row_normalized(W: np.ndarray, tol: float = 1e-10) -> bool:
    """
    检查矩阵是否已经行标准化

    Args:
        W: 待检查的矩阵
        tol: 容差

    Returns:
        是否已经行标准化
    """
    row_sums = W.sum(axis=1)
    # 对角线应该为0
    if not np.allclose(np.diag(W), 0, atol=tol):
        return False
    # 每行和应该为0(孤立节点)或1
    for s in row_sums:
        if not (np.isclose(s, 0, atol=tol) or np.isclose(s, 1, atol=tol)):
            return False
    return True


def generate_weighted_network_from_correlation(returns: np.ndarray) -> np.ndarray:
    """
    从收益率序列生成加权网络 (用相关系数作为权重)

    Args:
        returns: T×K 收益率矩阵

    Returns:
        W: 行标准化的加权网络矩阵
    """
    # 计算相关系数矩阵
    corr = np.corrcoef(returns.T)
    # 取绝对值(只关心相关强度,不关心正负)
    W_raw = np.abs(corr)
    # 行标准化
    W = row_normalize(W_raw)
    return W


def generate_weighted_network_from_granger(data: np.ndarray, maxlag: int = 1) -> np.ndarray:
    """
    从Granger因果检验生成加权网络 (用F统计量作为权重)

    注意: 这只是示例框架,实际需要实现Granger因果检验

    Args:
        data: T×K 时间序列数据
        maxlag: 最大滞后阶数

    Returns:
        W: 行标准化的加权网络矩阵
    """
    K = data.shape[1]
    W_raw = np.zeros((K, K))

    # 这里应该实现Granger因果检验
    # 对于每对(i,j),检验i是否是j的Granger原因
    # 用F统计量作为权重

    # 示例: 随机生成(实际应用中需要替换为真正的检验)
    for i in range(K):
        for j in range(K):
            if i != j:
                # 这里应该是真实的Granger因果F统计量
                W_raw[i, j] = np.random.rand()

    # 行标准化
    W = row_normalize(W_raw)
    return W


def combine_networks(W_list: List[np.ndarray], delta: np.ndarray) -> np.ndarray:
    """
    根据权重delta组合多个网络

    W* = sum(delta_j * W_j)

    Args:
        W_list: d个K×K网络矩阵的列表
        delta: d个权重,满足sum(delta)=1

    Returns:
        W_star: 复合网络
    """
    assert np.isclose(np.sum(delta), 1.0), "delta之和必须等于1"
    assert len(W_list) == len(delta), "网络数量和权重数量必须一致"

    W_star = sum(d * W for d, W in zip(delta, W_list))
    return W_star


def compute_network_density(W: np.ndarray) -> float:
    """
    计算网络密度 (非零元素比例)

    Args:
        W: K×K网络矩阵

    Returns:
        density: 网络密度 ∈ [0, 1]
    """
    K = W.shape[0]
    # 排除对角线后的最大可能边数
    max_edges = K * (K - 1)
    # 计算实际边数(非零元素)
    actual_edges = np.sum(W > 0)
    density = actual_edges / max_edges
    return density


def compute_node_out_degree(W: np.ndarray) -> np.ndarray:
    """
    计算每个节点的出度 (对外连接数)

    Args:
        W: K×K网络矩阵

    Returns:
        out_degree: K×1 出度向量
    """
    return np.sum(W > 0, axis=1)


def compute_node_in_degree(W: np.ndarray) -> np.ndarray:
    """
    计算每个节点的入度 (被连接数)

    Args:
        W: K×K网络矩阵

    Returns:
        in_degree: K×1 入度向量
    """
    return np.sum(W > 0, axis=0)


# =============================================================================
# 示例用法
# =============================================================================

if __name__ == "__main__":
    # 测试行标准化
    print("=" * 60)
    print("测试行标准化")
    print("=" * 60)

    W = np.array([[0, 1, 1],
                  [1, 0, 0],
                  [0, 1, 0]], dtype=float)

    print("原始矩阵:")
    print(W)
    print("每行和:", W.sum(axis=1))

    W_norm = row_normalize(W)
    print("\n行标准化后:")
    print(W_norm)
    print("每行和(应该都是1):", W_norm.sum(axis=1))
    print("对角线(应该都是0):", np.diag(W_norm))

    # 测试加权矩阵
    print("\n" + "=" * 60)
    print("测试加权网络")
    print("=" * 60)

    # 生成随机收益率
    np.random.seed(42)
    returns = np.random.randn(100, 5)

    W_corr = generate_weighted_network_from_correlation(returns)
    print("\n基于相关性的加权网络:")
    print(W_corr)
    print("每行和:", W_corr.sum(axis=1))
    print("网络密度:", compute_network_density(W_corr))
    print("各节点出度:", compute_node_out_degree(W_corr))

    # 测试复合网络
    print("\n" + "=" * 60)
    print("测试复合网络")
    print("=" * 60)

    W_list = [W_norm, W_corr[:3, :3]]  # 注意:这里只是为了演示
    delta = np.array([0.6, 0.4])

    W_star = combine_networks(W_list, delta)
    print("\n复合网络 W* = 0.6*W1 + 0.4*W2:")
    print(W_star)
    print("检查是否行标准化:", check_row_normalized(W_star))
