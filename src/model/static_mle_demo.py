"""
静态多层网络模型估计演示 (Concentrated MLE)

模拟数据 + 完整估计流程
- 资产数量: 13
- 网络层数: 3
- 估计参数: delta (3个) + Lambda (13个对角元素)

结构方程: A * R_t = beta * F_t + eta_t
其中 A = I - Lambda * (sum_j delta_j * W_j)

================
关键概念说明:
================

1. 网络矩阵可以是有权或无权:
   - 无权矩阵(0/1): 论文做法,因果检验结果二元
   - 加权矩阵: 可用相关系数/F统计量等作为权重
   - 关键要求: 必须行标准化!

2. 行标准化:
   - 定义: 使矩阵每行元素之和为1(对角线为0)
   - 目的: 数值稳定性 + 参数可解释性
   - 示例: [[0,1,1],[1,0,0],[0,1,0]] -> [[0,0.5,0.5],[1,0,0],[0,1,0]]
"""

import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import logm
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# 设置字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决坐标轴负号显示为方框的问题
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)


# =============================================================================
# 第一部分: 网络工具函数
# =============================================================================

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
    """检查矩阵是否已经行标准化"""
    row_sums = W.sum(axis=1)
    if not np.allclose(np.diag(W), 0, atol=tol):
        return False
    for s in row_sums:
        if not (np.isclose(s, 0, atol=tol) or np.isclose(s, 1, atol=tol)):
            return False
    return True


def compute_network_density(W: np.ndarray) -> float:
    """计算网络密度 (非零元素比例)"""
    K = W.shape[0]
    max_edges = K * (K - 1)
    actual_edges = np.sum(W > 0)
    return actual_edges / max_edges


def generate_weighted_network_from_correlation(returns: np.ndarray) -> np.ndarray:
    """
    从收益率序列生成加权网络 (用相关系数作为权重)

    Args:
        returns: T×K 收益率矩阵

    Returns:
        W: 行标准化的加权网络矩阵
    """
    corr = np.corrcoef(returns.T)
    W_raw = np.abs(corr)
    W = row_normalize(W_raw)
    return W


# =============================================================================
# 第二部分: 模拟数据生成
# =============================================================================

def generate_networks(K: int = 13, d: int = 3, weighted: bool = False) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    生成d个行标准化的邻接矩阵 (模拟3层网络)

    关于无权 vs 加权矩阵:
    --------------------
    论文使用无权矩阵(0/1)是因为因果检验结果是二元的(拒绝/不拒绝原假设)。
    但数学上完全可以处理加权矩阵,只要满足行标准化即可。

    Args:
        K: 资产数量
        d: 网络层数
        weighted: 是否生成加权网络(默认False,生成无权网络)

    Returns:
        W_list: d个K×K的行标准化网络矩阵列表
        true_delta: 真实的网络权重
    """
    W_list = []

    if weighted:
        # 加权网络示例: 用随机权重
        for _ in range(d):
            W_raw = np.random.rand(K, K)
            np.fill_diagonal(W_raw, 0)
            W = row_normalize(W_raw)
            W_list.append(W)
    else:
        # 网络1: 稀疏网络 (模拟Granger因果, 边较少)
        W1 = np.zeros((K, K))
        edges1 = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,5),
                  (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (6,12)]
        for i, j in edges1:
            W1[i, j] = 1.0
        W1 = row_normalize(W1)
        W_list.append(W1)

        # 网络2: 中等密度 (模拟分位数因果)
        W2 = np.random.rand(K, K)
        np.fill_diagonal(W2, 0)
        threshold = np.percentile(W2, 70)
        W2 = (W2 > threshold).astype(float)
        W2 = row_normalize(W2)
        W_list.append(W2)

        # 网络3: 较密网络 (模拟非参数分位数因果)
        W3 = np.random.rand(K, K)
        np.fill_diagonal(W3, 0)
        threshold = np.percentile(W3, 50)
        W3 = (W3 > threshold).astype(float)
        W3 = row_normalize(W3)
        W_list.append(W3)

    # 真实的delta
    true_delta = np.array([0.3, 0.3, 0.4])

    return W_list, true_delta


def generate_data(T: int = 252, K: int = 13, n_factors: int = 4,
                  true_delta: np.ndarray = None,
                  W_list: List[np.ndarray] = None) -> Dict:
    """
    生成模拟数据

    结构方程: A * R_t = beta * F_t + eta_t

    数据生成过程:
    1. 生成因子 F_t ~ N(0, 0.02^2)
    2. 设定真实的 beta (因子暴露)
    3. 设定真实的 Lambda = diag(rho) (异质反应系数)
    4. 构建复合网络 W* = sum(delta_j * W_j)
    5. 构建 A = I - Lambda * W*
    6. 生成结构残差 eta_t ~ N(0, Sigma_eta)
    7. 根据 R_t = A^{-1} * (beta * F_t + eta_t) 生成收益率
    """
    # 生成因子
    F = np.random.randn(T, n_factors) * 0.02

    # 真实的因子暴露 beta
    beta = np.random.randn(K, n_factors) * 0.5
    beta[0] = [1.0, 0.3, 0.2, 0.1]

    # 真实的异质反应系数
    true_rho = np.random.uniform(0.05, 0.25, K)  # 设小一点保证稳定性
    Lambda_true = np.diag(true_rho)

    # 构建复合网络
    W_star = sum(dj * W for dj, W in zip(true_delta, W_list))

    # 构建 A
    A_true = np.eye(K) - Lambda_true @ W_star

    # 生成结构残差
    sigma_eta = np.random.uniform(0.01, 0.03, K)
    Sigma_eta = np.diag(sigma_eta**2)
    eta = np.random.multivariate_normal(np.zeros(K), Sigma_eta, size=T)

    # 生成收益率
    factor_component = F @ beta.T
    total_shock = factor_component + eta

    R = np.zeros((T, K))
    for t in range(T):
        R[t] = np.linalg.solve(A_true, total_shock[t])

    return {
        'R': R, 'F': F, 'beta_true': beta, 'Lambda_true': Lambda_true,
        'W_star_true': W_star, 'A_true': A_true, 'eta': eta,
        'sigma_eta': sigma_eta, 'true_delta': true_delta
    }


# =============================================================================
# 第三部分: 核心估计函数
# =============================================================================

def estimate_beta_sigma(A: np.ndarray, R: np.ndarray, F: np.ndarray,
                        include_alpha: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 1: 给定A, 估计alpha、beta和Sigma_eta

    完整结构方程: A * R = alpha + beta * F + eta
    这是关于变换后的收益率 A*R 对[1, F]做OLS

    其中:
        alpha: K×1 预期收益率(截距项)
        beta: K×n_factors 因子暴露
        eta: T×K 结构残差

    OLS公式: [alpha, beta]_hat = ([1,F]'[1,F])^{-1} [1,F]' (A*R)'

    Args:
        A: K×K 结构矩阵
        R: T×K 收益率矩阵
        F: T×n_factors 因子矩阵
        include_alpha: 是否估计截距项(预期收益率),默认True

    Returns:
        alpha_hat: K×1 预期收益率估计
        beta_hat: K×n_factors 因子暴露估计
        eta_hat: T×K 残差
        Sigma_eta: K×K 残差协方差(对角阵)
    """
    T, K = R.shape
    n_factors = F.shape[1]

    # 变换收益率: R_tilde = A * R
    R_tilde = (A @ R.T).T

    if include_alpha:
        # 在F中添加截距项(一列1)
        F_aug = np.column_stack([np.ones(T), F])  # T × (1+n_factors)

        # OLS估计: B_hat = [alpha_hat, beta_hat]
        B_hat = np.linalg.solve(F_aug.T @ F_aug, F_aug.T @ R_tilde).T  # K × (1+n_factors)

        alpha_hat = B_hat[:, 0]
        beta_hat = B_hat[:, 1:]

        # 计算残差
        eta_hat = R_tilde - F_aug @ B_hat.T
    else:
        # 无截距项(向后兼容)
        alpha_hat = np.zeros(K)
        beta_hat = np.linalg.solve(F.T @ F, F.T @ R_tilde).T
        eta_hat = R_tilde - F @ beta_hat.T

    # 估计协方差(对角阵)
    sigma_sq = np.var(eta_hat, axis=0, ddof=0)
    Sigma_eta = np.diag(sigma_sq)

    return alpha_hat, beta_hat, eta_hat, Sigma_eta


def concentrated_loglik(params: np.ndarray, R: np.ndarray, F: np.ndarray,
                        W_list: List[np.ndarray]) -> float:
    """
    计算集中对数似然

    集中对数似然公式:
    ln L_c = -T/2 * ln|S_eta| + T * ln|A| + const

    各项含义:
    - -T/2 * ln|S_eta|: 残差方差越小越好(拟合优度)
    - T * ln|A|: Jacobian项,补偿线性变换带来的扭曲

    参数:
        params = [delta_1, ..., delta_d, rho_1, ..., rho_K]

    Returns:
        -loglik: 负的对数似然(因为scipy minimize求最小值)
    """
    d = len(W_list)
    K = R.shape[1]
    T = R.shape[0]

    delta = params[:d]
    rho = params[d:]

    # 约束检查
    if not np.isclose(np.sum(delta), 1.0, atol=1e-6):
        return 1e10
    if np.any(delta < 0) or np.any(rho < 0):
        return 1e10

    # 构建复合网络
    W_star = sum(dj * W for dj, W in zip(delta, W_list))

    # 构建 A
    Lambda = np.diag(rho)
    A = np.eye(K) - Lambda @ W_star

    # 检查A是否可逆
    try:
        det_A = np.linalg.det(A)
        if np.abs(det_A) < 1e-12:
            return 1e10
    except:
        return 1e10

    # Step 1: 估计alpha、beta和Sigma_eta(alpha在似然计算中不需要)
    _, _, _, Sigma_eta = estimate_beta_sigma(A, R, F)

    # Step 2: 计算集中对数似然
    diag_Sigma = np.diag(Sigma_eta)
    if np.any(diag_Sigma <= 0):
        return 1e10

    logdet_Sigma = np.sum(np.log(diag_Sigma))
    logdet_A = np.log(np.abs(det_A))

    loglik = -0.5 * T * logdet_Sigma + T * logdet_A

    return -loglik


def estimate_model(R: np.ndarray, F: np.ndarray, W_list: List[np.ndarray],
                   method: str = 'SLSQP') -> Dict:
    """
    主估计函数: 使用集中MLE估计delta和Lambda

    优化问题:
        max  -T/2 * ln|S_eta(A)| + T * ln|A|
        s.t. sum(delta) = 1, delta >= 0, rho >= 0

    其中 A = I - Lambda * (sum_j delta_j * W_j)
    """
    d = len(W_list)
    K = R.shape[1]

    print(f"开始估计: {K}个资产, {d}层网络")
    print(f"总参数: {d}个delta + {K}个rho = {d + K}个")

    # 初始值
    delta_init = np.ones(d) / d
    rho_init = np.ones(K) * 0.15
    x0 = np.concatenate([delta_init, rho_init])

    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:d]) - 1.0}
    ]

    # 边界
    bounds = [(0, 1) for _ in range(d)] + [(0,None) for _ in range(K)]

    # 优化
    print("优化中...")
    result = minimize(
        concentrated_loglik,
        x0,
        args=(R, F, W_list),
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
    )

    if not result.success:
        print(f"警告: 优化未收敛: {result.message}")

    # 解析结果
    delta_hat = result.x[:d]
    rho_hat = result.x[d:]

    # 构建最终估计的矩阵
    W_star_hat = sum(d * W for d, W in zip(delta_hat, W_list))
    Lambda_hat = np.diag(rho_hat)
    A_hat = np.eye(K) - Lambda_hat @ W_star_hat

    # 估计alpha、beta和Sigma
    alpha_hat, beta_hat, eta_hat, Sigma_eta_hat = estimate_beta_sigma(A_hat, R, F)

    return {
        'delta_hat': delta_hat,
        'rho_hat': rho_hat,
        'Lambda_hat': Lambda_hat,
        'W_star_hat': W_star_hat,
        'A_hat': A_hat,
        'alpha_hat': alpha_hat,
        'beta_hat': beta_hat,
        'Sigma_eta_hat': Sigma_eta_hat,
        'eta_hat': eta_hat,
        'loglik': -result.fun,
        'success': result.success,
        'niter': result.nit
    }


# =============================================================================
# 第四部分: 结果分析和可视化
# =============================================================================

def print_results(results: Dict, true_params: Dict, W_list: List[np.ndarray]):
    """打印估计结果并与真实值比较"""

    print("\n" + "="*60)
    print("估计结果总结")
    print("="*60)

    K = len(results['rho_hat'])
    d = len(results['delta_hat'])

    # 网络权重 delta
    print(f"\n【网络权重 δ (共{d}层)】")
    print(f"{'网络层':<10} {'估计值':<12} {'真实值':<12} {'误差':<12}")
    print("-" * 50)
    for j in range(d):
        true_val = true_params['true_delta'][j]
        est_val = results['delta_hat'][j]
        print(f"网络{j+1:<6} {est_val:<12.4f} {true_val:<12.4f} {abs(est_val-true_val):<12.4f}")

    # 预期收益率 alpha
    print(f"\n【预期收益率 α (共{K}个资产, 显示前5个)】")
    print(f"{'资产':<8} {'估计值':<12} {'真实值':<12} {'误差':<12}")
    print("-" * 50)
    for i in range(min(5, K)):
        # 模拟数据的真实alpha为0(未显式设置)
        true_val = 0.0
        est_val = results['alpha_hat'][i]
        print(f"资产{i+1:<4} {est_val:<12.6f} {true_val:<12.6f} {abs(est_val-true_val):<12.6f}")

    # 异质反应系数 rho
    print(f"\n【异质反应系数 ρ (共{K}个资产, 显示前5个)】")
    print(f"{'资产':<8} {'估计值':<12} {'真实值':<12} {'误差':<12}")
    print("-" * 50)
    for i in range(min(5, K)):
        true_val = true_params['Lambda_true'][i, i]
        est_val = results['rho_hat'][i]
        print(f"资产{i+1:<4} {est_val:<12.4f} {true_val:<12.4f} {abs(est_val-true_val):<12.4f}")

    # 复合网络性质
    print(f"\n【复合网络 W* = Σδ_j·W_j 的性质】")
    print(f"网络密度: {np.mean(results['W_star_hat'] > 0):.4f}")
    print(f"权重总和: {np.sum(results['delta_hat']):.4f}")

    # 检查行标准化
    is_normalized = check_row_normalized(results['W_star_hat'])
    print(f"复合网络是否行标准化: {is_normalized}")

    # 拟合优度
    print(f"\n【拟合优度】")
    print(f"对数似然值: {results['loglik']:.2f}")
    print(f"优化迭代次数: {results['niter']}")
    print(f"优化成功: {results['success']}")


def plot_results(results: Dict, true_params: Dict, W_list: List[np.ndarray]):
    """可视化估计结果"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    d = len(results['delta_hat'])
    K = len(results['rho_hat'])

    # 1. delta 估计 vs 真实值
    ax = axes[0, 0]
    x = np.arange(d)
    width = 0.35
    ax.bar(x - width/2, results['delta_hat'], width, label='估计值', alpha=0.8)
    ax.bar(x + width/2, true_params['true_delta'], width, label='真实值', alpha=0.8)
    ax.set_xlabel('网络层')
    ax.set_ylabel('权重 δ')
    ax.set_title('网络权重估计')
    ax.set_xticks(x)
    ax.set_xticklabels([f'网络{j+1}' for j in range(d)])
    ax.legend()

    # 2. rho 估计 vs 真实值
    ax = axes[0, 1]
    idx = np.arange(K)
    ax.scatter(idx, results['rho_hat'], label='估计值', alpha=0.7, s=50)
    ax.scatter(idx, np.diag(true_params['Lambda_true']), label='真实值', alpha=0.7, s=50)
    ax.plot(idx, results['rho_hat'], 'b-', alpha=0.3)
    ax.plot(idx, np.diag(true_params['Lambda_true']), 'r--', alpha=0.3)
    ax.set_xlabel('资产编号')
    ax.set_ylabel('反应系数 ρ')
    ax.set_title('异质反应系数估计')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 复合网络热力图
    ax = axes[0, 2]
    im = ax.imshow(results['W_star_hat'], cmap='Blues', aspect='auto')
    ax.set_title('估计的复合网络 W*')
    ax.set_xlabel('资产j')
    ax.set_ylabel('资产i')
    plt.colorbar(im, ax=ax)

    # 4. 残差分布
    ax = axes[1, 0]
    ax.hist(results['eta_hat'].flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('残差')
    ax.set_ylabel('频数')
    ax.set_title('残差分布')
    ax.axvline(x=0, color='r', linestyle='--', label='零线')
    ax.legend()

    # 5. 网络层对比
    ax = axes[1, 1]
    for j, W in enumerate(W_list):
        ax.plot(np.sum(W > 0, axis=1), label=f'网络{j+1} (δ={results["delta_hat"][j]:.2f})', marker='o')
    ax.set_xlabel('资产编号')
    ax.set_ylabel('出度 (连接数)')
    ax.set_title('各网络层的连接度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. A矩阵的特征值
    ax = axes[1, 2]
    eigvals = np.linalg.eigvals(results['A_hat'])
    sorted_eigvals = sorted(np.abs(eigvals), reverse=True)
    ax.scatter(range(len(eigvals)), sorted_eigvals)
    ax.axhline(y=1, color='r', linestyle='--', label='|λ|=1 (稳定性边界)')
    ax.set_xlabel('特征值序号')
    ax.set_ylabel('|特征值|')
    ax.set_title('A矩阵的特征值')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加文本信息
    n_large_eig = np.sum(np.array(sorted_eigvals) > 1)
    ax.text(0.5, 0.95, f'大于1的特征值: {n_large_eig}个',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('estimation_results.png', dpi=150, bbox_inches='tight')
    print("\n结果图已保存至: estimation_results.png")
    plt.show()


# =============================================================================
# 第五部分: 主程序
# =============================================================================

def main():
    """主程序: 生成数据 + 估计模型 + 展示结果"""

    print("="*60)
    print("静态多层网络模型估计演示")
    print("="*60)
    print("\n关键概念:")
    print("- 网络矩阵: 可以是有权或无权,但必须行标准化")
    print("- 行标准化: 每行元素之和为1,对角线为0")
    print("- delta: 全样本一个值,表示该网络层的重要性权重")

    # 参数设置
    T = 252
    K = 13
    d = 3
    n_factors = 4

    # 1. 生成网络结构
    print(f"\n1. 生成{d}层网络结构 (K={K})...")
    W_list, true_delta = generate_networks(K, d, weighted=False)
    print(f"   网络1密度: {compute_network_density(W_list[0]):.3f}")
    print(f"   网络2密度: {compute_network_density(W_list[1]):.3f}")
    print(f"   网络3密度: {compute_network_density(W_list[2]):.3f}")

    # 检查行标准化
    for j, W in enumerate(W_list):
        is_norm = check_row_normalized(W)
        print(f"   网络{j+1}行标准化检查: {is_norm}")

    # 2. 生成模拟数据
    print(f"\n2. 生成模拟数据 (T={T}期)...")
    data = generate_data(T, K, n_factors, true_delta, W_list)
    print(f"   收益率均值: {np.mean(data['R']):.6f}")
    print(f"   收益率标准差: {np.std(data['R']):.6f}")
    print(f"   真实网络权重: {true_delta}")

    # 3. 估计模型
    print(f"\n3. 开始估计...")
    results = estimate_model(data['R'], data['F'], W_list)

    # 4. 打印结果
    print_results(results, data, W_list)

    # 5. 可视化
    print(f"\n4. 生成可视化...")
    plot_results(results, data, W_list)

    # 6. 模型诊断
    print(f"\n5. 模型诊断...")
    print(f"   A矩阵行列式: {np.linalg.det(results['A_hat']):.6f}")
    print(f"   A矩阵条件数: {np.linalg.cond(results['A_hat']):.2f}")
    print(f"   最小特征值: {np.min(np.abs(np.linalg.eigvals(results['A_hat']))):.4f}")
    print(f"   残差标准差均值: {np.mean(np.std(results['eta_hat'], axis=0)):.6f}")

    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)

    return results, data, W_list


# =============================================================================
# 第六部分: 滚动窗口 MLE — 真实 DY 网络数据
# =============================================================================
# 说明:
#   R 代码 TVP-VAR-DY.R 生成全部时间点的 DY 切片网络, 存为堆叠 CSV:
#     - dy_all_ret.csv / dy_all_vol.csv: (K*T_net) × K, 每 K 行 = 一个时点
#     - dy_dates_ret.csv / dy_dates_vol.csv: t_index + date
#   本模块加载这些数据, 在每个时点执行集中 MLE, 追踪:
#     - delta(t): 网络层权重的时间演化
#     - centrality(t): 复合网络特征向量中心性的时间演化
# =============================================================================


def load_stacked_networks(filepath: str, K: int = 31) -> np.ndarray:
    """
    加载 R 输出的堆叠邻接矩阵 CSV, 重塑为 3D 数组

    文件格式: (K * T_net) 行 × K 列, 无表头
    每 K 行为一个时间点的 K×K 行标准化邻接矩阵

    Args:
        filepath: dy_all_ret.csv 或 dy_all_vol.csv 的路径
        K: 行业数量 (默认 31)

    Returns:
        W_3d: (T_net, K, K) 的 3D 数组, W_3d[t] 为第 t 个时点的邻接矩阵
    """
    raw = np.loadtxt(filepath, delimiter=',')
    T_net = raw.shape[0] // K
    assert raw.shape[0] == K * T_net, \
        f"行数 {raw.shape[0]} 不是 K={K} 的整数倍"
    assert raw.shape[1] == K, \
        f"列数 {raw.shape[1]} != K={K}"
    W_3d = raw.reshape(T_net, K, K)
    print(f"  加载网络: {filepath}")
    print(f"  形状: {T_net} 个时间点 × {K}×{K} 邻接矩阵")
    return W_3d


def load_network_dates(filepath: str) -> pd.DatetimeIndex:
    """加载日期映射文件"""
    df = pd.read_csv(filepath)
    dates = pd.to_datetime(df['date'])
    print(f"  日期范围: {dates.iloc[0].date()} ~ {dates.iloc[-1].date()} ({len(dates)} 天)")
    return dates


def load_returns(filepath: str) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    加载行业日收益率

    Returns:
        R: (T, K) 收益率矩阵
        dates: DatetimeIndex
        names: 行业名称列表
    """
    df = pd.read_csv(filepath)
    dates = pd.to_datetime(df['date'])
    names = [c for c in df.columns if c != 'date']
    R = df[names].values
    print(f"  收益率: {R.shape[0]} 天 × {R.shape[1]} 行业")
    return R, dates, names


def align_data(ret_dates: pd.DatetimeIndex,
               net_dates_ret: pd.DatetimeIndex,
               net_dates_vol: pd.DatetimeIndex) -> Tuple[pd.DatetimeIndex, Dict]:
    """
    对齐三个数据源的日期 (取交集)

    Returns:
        common_dates: 公共日期 (排序后)
        idx_maps: 字典, 包含每个数据源中公共日期对应的行索引
            idx_maps['ret'][i] = 收益率数据中第 i 个公共日期的行号
            idx_maps['net_ret'][i] = 收益率网络中的时间索引
            idx_maps['net_vol'][i] = 波动率网络中的时间索引
    """
    common = set(ret_dates) & set(net_dates_ret) & set(net_dates_vol)
    common_dates = pd.DatetimeIndex(sorted(common))

    # 构建索引映射
    ret_date2idx = {d: i for i, d in enumerate(ret_dates)}
    net_ret_date2idx = {d: i for i, d in enumerate(net_dates_ret)}
    net_vol_date2idx = {d: i for i, d in enumerate(net_dates_vol)}

    idx_maps = {
        'ret': np.array([ret_date2idx[d] for d in common_dates]),
        'net_ret': np.array([net_ret_date2idx[d] for d in common_dates]),
        'net_vol': np.array([net_vol_date2idx[d] for d in common_dates]),
    }

    print(f"\n  日期对齐:")
    print(f"    收益率: {len(ret_dates)} 天")
    print(f"    收益率网络: {len(net_dates_ret)} 天")
    print(f"    波动率网络: {len(net_dates_vol)} 天")
    print(f"    公共日期: {len(common_dates)} 天")
    print(f"    范围: {common_dates[0].date()} ~ {common_dates[-1].date()}")

    return common_dates, idx_maps


def construct_factors(R: np.ndarray, method: str = "equal_weight") -> np.ndarray:
    """
    从收益率矩阵构造共同因子 F_t

    Args:
        R: (T, K) 收益率矩阵
        method: "equal_weight" (等权市场收益) 或 "pca" (PCA第一主成分)

    Returns:
        F: (T, 1) 因子矩阵
    """
    if method == "pca":
        # PCA 第一主成分
        R_demean = R - R.mean(axis=0)
        _, _, Vt = np.linalg.svd(R_demean, full_matrices=False)
        F = (R_demean @ Vt[0].T).reshape(-1, 1)
        # 标准化
        F = F / F.std()
        print(f"  因子构造: PCA第一主成分 (解释方差比待检查)")
    else:
        # 等权市场收益率
        F = R.mean(axis=1, keepdims=True)
        print(f"  因子构造: 等权市场收益率 (均值={F.mean():.6f})")

    return F


def estimate_single_window(R_win: np.ndarray, F_win: np.ndarray,
                           W_list: List[np.ndarray],
                           multi_start: bool = True) -> Dict:
    """
    单窗口集中 MLE 估计 (复用 concentrated_loglik 和 estimate_beta_sigma)

    与 estimate_model 的区别:
    - 无 print 输出 (滚动循环中调用数百次)
    - 可选多起点优化提升鲁棒性
    - d=2 (固定两层网络: 收益率 + 波动率)

    Args:
        R_win: (w, K) 窗口内收益率
        F_win: (w, n_f) 窗口内因子
        W_list: [W_ret, W_vol] 两个 K×K 邻接矩阵
        multi_start: 是否使用多起点

    Returns:
        dict: delta_hat, rho_hat, loglik, success
    """
    d = len(W_list)  # = 2
    K = R_win.shape[1]

    # 初始值列表
    if multi_start:
        init_deltas = [
            np.array([0.5, 0.5]),
            np.array([0.3, 0.7]),
            np.array([0.7, 0.3]),
            np.array([0.1, 0.9]),
        ]
    else:
        init_deltas = [np.ones(d) / d]

    # 约束和边界
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:d]) - 1.0}]
    bounds = [(0, 1)] * d + [(0, 2)] * K  # rho 上界 2 (安全余量)

    best_result = None
    best_loglik = -np.inf

    for delta_init in init_deltas:
        rho_init = np.ones(K) * 0.1
        x0 = np.concatenate([delta_init, rho_init])

        try:
            result = minimize(
                concentrated_loglik,
                x0,
                args=(R_win, F_win, W_list),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
            )
            loglik = -result.fun
            if loglik > best_loglik and result.success:
                best_loglik = loglik
                best_result = result
        except Exception:
            continue

    # 如果没有成功的, 取最后一次结果 (即使未收敛)
    if best_result is None:
        # 回退: 用默认初始值, 接受非收敛结果
        x0 = np.concatenate([np.array([0.5, 0.5]), np.ones(K) * 0.1])
        try:
            best_result = minimize(
                concentrated_loglik, x0,
                args=(R_win, F_win, W_list),
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
            )
        except Exception:
            return {
                'delta_hat': np.array([np.nan, np.nan]),
                'rho_hat': np.full(K, np.nan),
                'loglik': np.nan,
                'success': False,
            }

    delta_hat = best_result.x[:d]
    rho_hat = best_result.x[d:]

    return {
        'delta_hat': delta_hat,
        'rho_hat': rho_hat,
        'loglik': -best_result.fun,
        'success': best_result.success,
    }


def compute_eigenvector_centrality(W: np.ndarray) -> np.ndarray:
    """
    计算网络的特征向量中心性

    W[i,j] 表示 j 对 i 的影响, 因此用 W.T 的主特征向量衡量
    各节点作为"影响源"的重要性

    Args:
        W: K×K 行标准化邻接矩阵

    Returns:
        centrality: K 维向量 (归一化, 和为1)
    """
    eigvals, eigvecs = np.linalg.eig(W.T)
    # 取最大特征值对应的特征向量
    idx = np.argmax(np.abs(eigvals))
    v = np.abs(eigvecs[:, idx]).astype(float)
    # 归一化
    v_sum = v.sum()
    if v_sum > 0:
        v = v / v_sum
    return v


def rolling_estimation(R: np.ndarray, F: np.ndarray,
                       W_ret_3d: np.ndarray, W_vol_3d: np.ndarray,
                       common_dates: pd.DatetimeIndex,
                       idx_maps: Dict,
                       window: int = 252,
                       step: int = 22,
                       verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    滚动窗口 MLE 估计

    对于每个评估日期 t:
    1. 取 [t-window+1, t] 的收益率和因子
    2. 取 t 时刻的 DY 收益率和波动率邻接矩阵
    3. 运行集中 MLE → delta(t), rho(t)
    4. 计算复合网络 W*(t) 和特征向量中心性

    Args:
        R: 完整收益率矩阵 (原始, 未对齐)
        F: 完整因子矩阵 (原始, 未对齐)
        W_ret_3d: (T_net_ret, K, K) 收益率网络
        W_vol_3d: (T_net_vol, K, K) 波动率网络
        common_dates: 对齐后的公共日期
        idx_maps: 索引映射
        window: 滚动窗口大小 (天)
        step: 步长 (天)
        verbose: 是否打印进度

    Returns:
        results_df: 包含 date, delta_ret, delta_vol, rho_0..rho_K, loglik, success
        centrality_df: 包含 date, 各行业中心性值
    """
    K = R.shape[1]
    n_common = len(common_dates)

    # 评估点: 从 window-1 开始, 每 step 个公共日
    eval_indices = list(range(window - 1, n_common, step))
    n_eval = len(eval_indices)

    if verbose:
        print(f"\n{'='*60}")
        print(f"滚动窗口 MLE 估计")
        print(f"{'='*60}")
        print(f"  窗口大小: {window} 天")
        print(f"  步长: {step} 天")
        print(f"  公共日期: {n_common} 天")
        print(f"  评估点数: {n_eval}")
        print(f"  日期范围: {common_dates[eval_indices[0]].date()} ~ "
              f"{common_dates[eval_indices[-1]].date()}")

    # 存储结果
    results_list = []
    centrality_list = []

    for count, ci in enumerate(eval_indices):
        date_t = common_dates[ci]

        # 收益率窗口: 在 ret 索引中取 [ci-window+1 : ci+1]
        ret_indices = idx_maps['ret'][ci - window + 1: ci + 1]
        R_win = R[ret_indices]
        F_win = F[ret_indices]

        # 当前时点的网络
        net_ret_idx = idx_maps['net_ret'][ci]
        net_vol_idx = idx_maps['net_vol'][ci]
        W_ret_t = W_ret_3d[net_ret_idx]
        W_vol_t = W_vol_3d[net_vol_idx]

        # MLE 估计
        res = estimate_single_window(R_win, F_win, [W_ret_t, W_vol_t])

        # 复合网络和中心性
        if res['success'] and not np.any(np.isnan(res['delta_hat'])):
            W_star = res['delta_hat'][0] * W_ret_t + res['delta_hat'][1] * W_vol_t
            cent = compute_eigenvector_centrality(W_star)
        else:
            cent = np.full(K, np.nan)

        # 记录
        row = {
            'date': date_t,
            'delta_ret': res['delta_hat'][0],
            'delta_vol': res['delta_hat'][1],
            'loglik': res['loglik'],
            'success': res['success'],
        }
        for i in range(K):
            row[f'rho_{i}'] = res['rho_hat'][i]
        results_list.append(row)

        cent_row = {'date': date_t}
        for i in range(K):
            cent_row[f'industry_{i}'] = cent[i]
        centrality_list.append(cent_row)

        # 进度
        if verbose and ((count + 1) % 20 == 0 or count == 0 or count == n_eval - 1):
            status = "✓" if res['success'] else "✗"
            print(f"  [{count+1:3d}/{n_eval}] {date_t.date()} | "
                  f"δ_ret={res['delta_hat'][0]:.3f} δ_vol={res['delta_hat'][1]:.3f} | "
                  f"loglik={res['loglik']:.1f} | {status}")

    results_df = pd.DataFrame(results_list)
    centrality_df = pd.DataFrame(centrality_list)

    # 统计
    n_success = results_df['success'].sum()
    if verbose:
        print(f"\n  收敛率: {n_success}/{n_eval} ({100*n_success/n_eval:.1f}%)")
        print(f"  δ_ret 均值: {results_df['delta_ret'].mean():.3f}")
        print(f"  δ_vol 均值: {results_df['delta_vol'].mean():.3f}")

    return results_df, centrality_df


# =============================================================================
# 第七部分: 可视化 — 滚动估计结果
# =============================================================================

def plot_delta_evolution(results_df: pd.DataFrame,
                         save_path: str = None):
    """绘制 δ(t) 网络权重的时间演化"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    dates = pd.to_datetime(results_df['date'])
    mask = results_df['success'].values

    # δ 时间序列
    ax = axes[0]
    ax.plot(dates[mask], results_df.loc[mask, 'delta_ret'],
            label='δ_ret (收益率网络)', color='steelblue', linewidth=1.2)
    ax.plot(dates[mask], results_df.loc[mask, 'delta_vol'],
            label='δ_vol (波动率网络)', color='coral', linewidth=1.2)
    ax.set_ylabel('网络权重 δ')
    ax.set_title('多层网络权重的时间演化')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # 标注重要事件
    events = {
        '2008-09-15': '雷曼破产',
        '2015-06-15': '中国股灾',
        '2020-03-11': 'COVID-19',
    }
    for date_str, label in events.items():
        try:
            event_date = pd.Timestamp(date_str)
            if dates.min() <= event_date <= dates.max():
                ax.axvline(event_date, color='gray', linestyle='--', alpha=0.5)
                ax.text(event_date, 0.95, label, rotation=90,
                        va='top', ha='right', fontsize=8, alpha=0.7)
        except Exception:
            pass

    # 对数似然
    ax = axes[1]
    ax.plot(dates[mask], results_df.loc[mask, 'loglik'],
            color='darkgreen', linewidth=0.8)
    ax.set_ylabel('集中对数似然')
    ax.set_xlabel('日期')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    plt.show()


def plot_centrality_heatmap(centrality_df: pd.DataFrame,
                            industry_names: List[str],
                            save_path: str = None):
    """绘制中心性时间演化热力图"""
    dates = pd.to_datetime(centrality_df['date'])
    K = len(industry_names)

    # 提取中心性矩阵
    cent_cols = [f'industry_{i}' for i in range(K)]
    cent_matrix = centrality_df[cent_cols].values.T  # (K, T_eval)

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(cent_matrix, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')

    # Y 轴: 行业名称
    ax.set_yticks(range(K))
    ax.set_yticklabels(industry_names, fontsize=7)

    # X 轴: 日期 (稀疏标注)
    n_ticks = min(10, len(dates))
    tick_pos = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([dates.iloc[i].strftime('%Y-%m') for i in tick_pos],
                       rotation=45, ha='right', fontsize=8)

    ax.set_title('复合网络特征向量中心性的时间演化')
    ax.set_xlabel('日期')
    ax.set_ylabel('行业')
    plt.colorbar(im, ax=ax, label='中心性', shrink=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    plt.show()


def plot_top_central_industries(centrality_df: pd.DataFrame,
                                industry_names: List[str],
                                top_n: int = 5,
                                save_path: str = None):
    """绘制中心性最高的行业时间序列"""
    dates = pd.to_datetime(centrality_df['date'])
    K = len(industry_names)
    cent_cols = [f'industry_{i}' for i in range(K)]
    cent_matrix = centrality_df[cent_cols].values  # (T_eval, K)

    # 找出平均中心性最高的行业
    mean_cent = np.nanmean(cent_matrix, axis=0)
    top_idx = np.argsort(mean_cent)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx in top_idx:
        ax.plot(dates, cent_matrix[:, idx],
                label=industry_names[idx], linewidth=1.2, alpha=0.8)

    ax.set_xlabel('日期')
    ax.set_ylabel('特征向量中心性')
    ax.set_title(f'中心性最高的 {top_n} 个行业')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    plt.show()


# =============================================================================
# 第八部分: 滚动估计主入口
# =============================================================================

def main_rolling(data_dir: str = "D:/桌面数据/工作论文/带耦合多层网络/src/data/processed",
                 results_dir: str = "D:/桌面数据/工作论文/带耦合多层网络/results",
                 window: int = 252,
                 step: int = 22,
                 factor_method: str = "equal_weight"):
    """
    滚动窗口 MLE 估计主函数

    前置条件: 已运行 TVP-VAR-DY.R 生成:
        - dy_all_ret.csv, dy_all_vol.csv
        - dy_dates_ret.csv, dy_dates_vol.csv

    Args:
        data_dir: 数据目录
        results_dir: 结果输出目录
        window: 滚动窗口大小 (默认 252 = 1年)
        step: 步长 (默认 22 ≈ 月频)
        factor_method: 因子构造方法 ("equal_weight" 或 "pca")
    """
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("滚动窗口 MLE 估计 — 多层网络因子模型")
    print("="*60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    W_ret_3d = load_stacked_networks(
        os.path.join(data_dir, "dy_all_ret.csv"), K=31)
    W_vol_3d = load_stacked_networks(
        os.path.join(data_dir, "dy_all_vol.csv"), K=31)
    net_dates_ret = load_network_dates(
        os.path.join(data_dir, "dy_dates_ret.csv"))
    net_dates_vol = load_network_dates(
        os.path.join(data_dir, "dy_dates_vol.csv"))
    R_full, ret_dates, industry_names = load_returns(
        os.path.join(data_dir, "industry_returns.csv"))

    # 2. 日期对齐
    print("\n[2/5] 日期对齐...")
    common_dates, idx_maps = align_data(ret_dates, net_dates_ret, net_dates_vol)

    # 3. 构造因子
    print("\n[3/5] 构造共同因子...")
    F_full = construct_factors(R_full, method=factor_method)

    # 4. 验证 (单点测试)
    print("\n[4/5] 单点验证...")
    mid = len(common_dates) // 2
    test_ret_idx = idx_maps['ret'][mid - window + 1: mid + 1]
    R_test = R_full[test_ret_idx]
    F_test = F_full[test_ret_idx]
    W_ret_test = W_ret_3d[idx_maps['net_ret'][mid]]
    W_vol_test = W_vol_3d[idx_maps['net_vol'][mid]]

    # 检查网络合法性
    print(f"  网络行和检查 (应≈1): ret={W_ret_test.sum(axis=1).mean():.4f}, "
          f"vol={W_vol_test.sum(axis=1).mean():.4f}")
    print(f"  网络对角线检查 (应=0): ret={np.diag(W_ret_test).sum():.6f}, "
          f"vol={np.diag(W_vol_test).sum():.6f}")

    test_res = estimate_single_window(R_test, F_test, [W_ret_test, W_vol_test])
    print(f"  单点估计: δ_ret={test_res['delta_hat'][0]:.3f}, "
          f"δ_vol={test_res['delta_hat'][1]:.3f}, "
          f"收敛={test_res['success']}")

    if not test_res['success']:
        print("  ⚠ 单点验证未收敛, 继续执行但请注意结果质量")

    # 5. 滚动估计
    print("\n[5/5] 开始滚动估计...")
    results_df, centrality_df = rolling_estimation(
        R_full, F_full, W_ret_3d, W_vol_3d,
        common_dates, idx_maps,
        window=window, step=step)

    # 保存结果
    results_path = os.path.join(results_dir, "rolling_delta.csv")
    centrality_path = os.path.join(results_dir, "rolling_centrality.csv")
    results_df.to_csv(results_path, index=False)
    centrality_df.to_csv(centrality_path, index=False)
    print(f"\n结果已保存:")
    print(f"  {results_path}")
    print(f"  {centrality_path}")

    # 可视化
    print("\n生成可视化...")
    plot_delta_evolution(
        results_df,
        save_path=os.path.join(results_dir, "delta_evolution.png"))
    plot_centrality_heatmap(
        centrality_df, industry_names,
        save_path=os.path.join(results_dir, "centrality_heatmap.png"))
    plot_top_central_industries(
        centrality_df, industry_names, top_n=5,
        save_path=os.path.join(results_dir, "top_central_industries.png"))

    print("\n" + "="*60)
    print("滚动估计完成!")
    print("="*60)

    return results_df, centrality_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "rolling":
        # python static_mle_demo.py rolling
        main_rolling()
    else:
        # 默认: 静态模拟演示
        results, data, W_list = main()
