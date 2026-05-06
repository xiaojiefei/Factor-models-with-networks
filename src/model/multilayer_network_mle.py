"""
多层网络因子模型 — 滚动窗口集中 MLE 估计

结构方程: A * R_t = alpha + beta * F_t + eta_t
其中 A = I - Lambda * (sum_j delta_j * W_j)

数据管线:
  1. prepare_data.py 生成周度 industry_returns.csv, industry_volatility.csv
  2. TVP-VAR-DY.R 生成 dy_all_ret.csv, dy_all_vol.csv (堆叠邻接矩阵)
  3. 本脚本加载网络 + 收益率, 滚动估计 delta(t) 和中心性(t)

参考文献:
  Bonaccolto et al. (2019) — Estimation and model-based combination
  of causality networks among large US banks and insurance companies

Usage:
  python multilayer_network_mle.py
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Plot settings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# Part 1: Network Utility Functions
# =============================================================================

def row_normalize(W: np.ndarray) -> np.ndarray:
    """
    Row-normalize adjacency matrix: each row sums to 1, diagonal = 0.

    Why row-normalize?
    1. Numerical stability: prevents A = I - LambdaW from being singular
    2. Interpretability: rho in [0,1] has clear meaning (reaction intensity)
    3. Comparability: assets with different degrees are on the same scale

    Args:
        W: K x K adjacency matrix (weighted or unweighted)

    Returns:
        W_norm: row-normalized matrix (each row sums to 1 or 0)
    """
    W_norm = W.copy()
    np.fill_diagonal(W_norm, 0)
    row_sums = W_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero (isolated nodes)
    W_norm = W_norm / row_sums
    return W_norm


def check_row_normalized(W: np.ndarray, tol: float = 1e-10) -> bool:
    """Check whether a matrix is row-normalized."""
    row_sums = W.sum(axis=1)
    if not np.allclose(np.diag(W), 0, atol=tol):
        return False
    for s in row_sums:
        if not (np.isclose(s, 0, atol=tol) or np.isclose(s, 1, atol=tol)):
            return False
    return True


def compute_network_density(W: np.ndarray) -> float:
    """Compute network density (fraction of non-zero off-diagonal entries)."""
    K = W.shape[0]
    max_edges = K * (K - 1)
    actual_edges = np.sum(W > 0)
    return actual_edges / max_edges


# =============================================================================
# Part 2: Core Estimation Functions (Concentrated MLE)
# =============================================================================

def estimate_beta_sigma(A: np.ndarray, R: np.ndarray, F: np.ndarray,
                        include_alpha: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 1 of concentrated MLE: given A, estimate alpha, beta, Sigma_eta via OLS.

    Structural equation: A * R = alpha + beta * F + eta
    This is OLS of transformed returns (A*R) on [1, F].

    Args:
        A: K x K structural matrix
        R: T x K return matrix
        F: T x n_factors factor matrix
        include_alpha: whether to estimate intercept (expected returns)

    Returns:
        alpha_hat: K x 1 expected return estimates
        beta_hat: K x n_factors factor loading estimates
        eta_hat: T x K residuals
        Sigma_eta: K x K residual covariance (diagonal)
    """
    T, K = R.shape

    # Transform returns: R_tilde = A * R
    R_tilde = (A @ R.T).T

    if include_alpha:
        F_aug = np.column_stack([np.ones(T), F])  # T x (1+n_factors)
        B_hat = np.linalg.solve(F_aug.T @ F_aug, F_aug.T @ R_tilde).T  # K x (1+n_factors)
        alpha_hat = B_hat[:, 0]
        beta_hat = B_hat[:, 1:]
        eta_hat = R_tilde - F_aug @ B_hat.T
    else:
        alpha_hat = np.zeros(K)
        beta_hat = np.linalg.solve(F.T @ F, F.T @ R_tilde).T
        eta_hat = R_tilde - F @ beta_hat.T

    # Diagonal covariance
    sigma_sq = np.var(eta_hat, axis=0, ddof=0)
    Sigma_eta = np.diag(sigma_sq)

    return alpha_hat, beta_hat, eta_hat, Sigma_eta


def concentrated_loglik(params: np.ndarray, R: np.ndarray, F: np.ndarray,
                        W_list: List[np.ndarray]) -> float:
    """
    Compute negative concentrated log-likelihood.

    ln L_c = -T/2 * ln|Sigma_eta| + T * ln|A| + const

    where:
    - -T/2 * ln|Sigma_eta|: goodness-of-fit (smaller residual variance = better)
    - T * ln|A|: Jacobian term (compensates for linear transformation distortion)

    Args:
        params: [delta_1, ..., delta_d, rho_1, ..., rho_K]
        R: T x K return matrix
        F: T x n_factors factor matrix
        W_list: list of d K x K adjacency matrices

    Returns:
        -loglik: negative log-likelihood (scipy minimizes)
    """
    d = len(W_list)
    K = R.shape[1]
    T = R.shape[0]

    delta = params[:d]
    rho = params[d:]

    # Constraint checks
    if not np.isclose(np.sum(delta), 1.0, atol=1e-6):
        return 1e10
    if np.any(delta < 0) or np.any(rho < 0):
        return 1e10

    # Composite network
    W_star = sum(dj * W for dj, W in zip(delta, W_list))

    # Structural matrix A
    Lambda = np.diag(rho)
    A = np.eye(K) - Lambda @ W_star

    # Check invertibility
    try:
        det_A = np.linalg.det(A)
        if np.abs(det_A) < 1e-12:
            return 1e10
    except Exception:
        return 1e10

    # Step 1: estimate beta and Sigma_eta
    _, _, _, Sigma_eta = estimate_beta_sigma(A, R, F)

    # Step 2: concentrated log-likelihood
    diag_Sigma = np.diag(Sigma_eta)
    if np.any(diag_Sigma <= 0):
        return 1e10

    logdet_Sigma = np.sum(np.log(diag_Sigma))
    logdet_A = np.log(np.abs(det_A))

    loglik = -0.5 * T * logdet_Sigma + T * logdet_A

    return -loglik


# =============================================================================
# Part 3: Rolling Window MLE — Data Loading and Estimation
# =============================================================================

def load_stacked_networks(filepath: str, K: int = 31) -> np.ndarray:
    """
    Load R-exported stacked adjacency matrix CSV, reshape to 3D array.

    File format: (K * T_net) rows x K columns, no header.
    Each block of K rows is one time point's K x K row-normalized adjacency matrix.

    Args:
        filepath: path to dy_all_ret.csv or dy_all_vol.csv
        K: number of industries (default 31)

    Returns:
        W_3d: (T_net, K, K) 3D array, W_3d[t] is the adjacency matrix at time t
    """
    raw = np.loadtxt(filepath, delimiter=',')
    T_net = raw.shape[0] // K
    assert raw.shape[0] == K * T_net, \
        f"Row count {raw.shape[0]} is not a multiple of K={K}"
    assert raw.shape[1] == K, \
        f"Column count {raw.shape[1]} != K={K}"
    W_3d = raw.reshape(T_net, K, K)
    print(f"  Loaded network: {filepath}")
    print(f"  Shape: {T_net} time points x {K}x{K} adjacency matrices")
    return W_3d


def load_network_dates(filepath: str) -> pd.DatetimeIndex:
    """Load date mapping file."""
    df = pd.read_csv(filepath)
    dates = pd.to_datetime(df['date'])
    print(f"  Date range: {dates.iloc[0].date()} ~ {dates.iloc[-1].date()} ({len(dates)} periods)")
    return dates


def load_returns(filepath: str) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    Load industry returns (weekly).

    Returns:
        R: (T, K) return matrix
        dates: DatetimeIndex
        names: industry name list
    """
    df = pd.read_csv(filepath)
    dates = pd.to_datetime(df['date'])
    names = [c for c in df.columns if c != 'date']
    R = df[names].values
    print(f"  Returns: {R.shape[0]} periods x {R.shape[1]} industries")
    return R, dates, names


def align_data(ret_dates: pd.DatetimeIndex,
               net_dates_ret: pd.DatetimeIndex,
               net_dates_vol: pd.DatetimeIndex) -> Tuple[pd.DatetimeIndex, Dict]:
    """
    Align three data sources by taking date intersection.

    Returns:
        common_dates: sorted common dates
        idx_maps: dict with index mappings for each data source
    """
    common = set(ret_dates) & set(net_dates_ret) & set(net_dates_vol)
    common_dates = pd.DatetimeIndex(sorted(common))

    ret_date2idx = {d: i for i, d in enumerate(ret_dates)}
    net_ret_date2idx = {d: i for i, d in enumerate(net_dates_ret)}
    net_vol_date2idx = {d: i for i, d in enumerate(net_dates_vol)}

    idx_maps = {
        'ret': np.array([ret_date2idx[d] for d in common_dates]),
        'net_ret': np.array([net_ret_date2idx[d] for d in common_dates]),
        'net_vol': np.array([net_vol_date2idx[d] for d in common_dates]),
    }

    print(f"\n  Date alignment:")
    print(f"    Returns: {len(ret_dates)} periods")
    print(f"    Return networks: {len(net_dates_ret)} periods")
    print(f"    Volatility networks: {len(net_dates_vol)} periods")
    print(f"    Common dates: {len(common_dates)} periods")
    print(f"    Range: {common_dates[0].date()} ~ {common_dates[-1].date()}")

    return common_dates, idx_maps


def construct_factors(R: np.ndarray, method: str = "equal_weight") -> np.ndarray:
    """
    Construct common factor F_t from return matrix.

    Args:
        R: (T, K) return matrix
        method: "equal_weight" (equal-weighted market return) or "pca" (first PC)

    Returns:
        F: (T, 1) factor matrix
    """
    if method == "pca":
        R_demean = R - R.mean(axis=0)
        _, _, Vt = np.linalg.svd(R_demean, full_matrices=False)
        F = (R_demean @ Vt[0].T).reshape(-1, 1)
        F = F / F.std()
        print(f"  Factor: PCA first principal component")
    else:
        F = R.mean(axis=1, keepdims=True)
        print(f"  Factor: equal-weighted market return (mean={F.mean():.6f})")

    return F


def estimate_single_window(R_win: np.ndarray, F_win: np.ndarray,
                           W_list: List[np.ndarray],
                           multi_start: bool = True) -> Dict:
    """
    Single-window concentrated MLE estimation with multi-start optimization.

    Args:
        R_win: (w, K) windowed returns
        F_win: (w, n_f) windowed factors
        W_list: [W_ret, W_vol] two K x K adjacency matrices
        multi_start: whether to use multiple starting points

    Returns:
        dict: delta_hat, rho_hat, loglik, success
    """
    d = len(W_list)
    K = R_win.shape[1]

    # Initial value list
    if multi_start:
        init_deltas = [
            np.array([0.5, 0.5]),
            np.array([0.3, 0.7]),
            np.array([0.7, 0.3]),
            np.array([0.1, 0.9]),
        ]
    else:
        init_deltas = [np.ones(d) / d]

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:d]) - 1.0}]
    bounds = [(0, 1)] * d + [(0, 2)] * K  # rho upper bound 2 (safety margin)

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

    # Fallback if no successful result
    if best_result is None:
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
    Compute eigenvector centrality of a network.

    W[i,j] represents j's influence on i, so we use W.T's leading
    eigenvector to measure each node's importance as an influence source.

    Args:
        W: K x K row-normalized adjacency matrix

    Returns:
        centrality: K-dim vector (normalized, sums to 1)
    """
    eigvals, eigvecs = np.linalg.eig(W.T)
    idx = np.argmax(np.abs(eigvals))
    v = np.abs(eigvecs[:, idx]).astype(float)
    v_sum = v.sum()
    if v_sum > 0:
        v = v / v_sum
    return v


def rolling_estimation(R: np.ndarray, F: np.ndarray,
                       W_ret_3d: np.ndarray, W_vol_3d: np.ndarray,
                       common_dates: pd.DatetimeIndex,
                       idx_maps: Dict,
                       window: int = 52,
                       step: int = 4,
                       verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling window MLE estimation.

    For each evaluation date t:
    1. Take [t-window+1, t] returns and factors
    2. Take DY return and volatility adjacency matrices at time t
    3. Run concentrated MLE -> delta(t), rho(t)
    4. Compute composite network W*(t) and eigenvector centrality

    Args:
        R: full return matrix (original, not aligned)
        F: full factor matrix (original, not aligned)
        W_ret_3d: (T_net_ret, K, K) return networks
        W_vol_3d: (T_net_vol, K, K) volatility networks
        common_dates: aligned common dates
        idx_maps: index mappings
        window: rolling window size (weeks, default 52 = 1 year)
        step: step size (weeks, default 4 ~ monthly)
        verbose: whether to print progress

    Returns:
        results_df: date, delta_ret, delta_vol, rho_0..rho_K, loglik, success
        centrality_df: date, industry centrality values
    """
    K = R.shape[1]
    n_common = len(common_dates)

    eval_indices = list(range(window - 1, n_common, step))
    n_eval = len(eval_indices)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Rolling Window MLE Estimation")
        print(f"{'='*60}")
        print(f"  Window: {window} weeks")
        print(f"  Step: {step} weeks")
        print(f"  Common dates: {n_common} weeks")
        print(f"  Evaluation points: {n_eval}")
        print(f"  Date range: {common_dates[eval_indices[0]].date()} ~ "
              f"{common_dates[eval_indices[-1]].date()}")

    results_list = []
    centrality_list = []

    for count, ci in enumerate(eval_indices):
        date_t = common_dates[ci]

        # Return window
        ret_indices = idx_maps['ret'][ci - window + 1: ci + 1]
        R_win = R[ret_indices]
        F_win = F[ret_indices]

        # Current time point networks
        net_ret_idx = idx_maps['net_ret'][ci]
        net_vol_idx = idx_maps['net_vol'][ci]
        W_ret_t = W_ret_3d[net_ret_idx]
        W_vol_t = W_vol_3d[net_vol_idx]

        # MLE estimation
        res = estimate_single_window(R_win, F_win, [W_ret_t, W_vol_t])

        # Composite network and centrality
        if res['success'] and not np.any(np.isnan(res['delta_hat'])):
            W_star = res['delta_hat'][0] * W_ret_t + res['delta_hat'][1] * W_vol_t
            cent = compute_eigenvector_centrality(W_star)
        else:
            cent = np.full(K, np.nan)

        # Record results
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

        # Progress
        if verbose and ((count + 1) % 20 == 0 or count == 0 or count == n_eval - 1):
            status = "OK" if res['success'] else "FAIL"
            print(f"  [{count+1:3d}/{n_eval}] {date_t.date()} | "
                  f"d_ret={res['delta_hat'][0]:.3f} d_vol={res['delta_hat'][1]:.3f} | "
                  f"loglik={res['loglik']:.1f} | {status}")

    results_df = pd.DataFrame(results_list)
    centrality_df = pd.DataFrame(centrality_list)

    n_success = results_df['success'].sum()
    if verbose:
        print(f"\n  Convergence rate: {n_success}/{n_eval} ({100*n_success/n_eval:.1f}%)")
        print(f"  delta_ret mean: {results_df['delta_ret'].mean():.3f}")
        print(f"  delta_vol mean: {results_df['delta_vol'].mean():.3f}")

    return results_df, centrality_df


# =============================================================================
# Part 4: Visualization
# =============================================================================

def plot_delta_evolution(results_df: pd.DataFrame,
                         save_path: str = None):
    """Plot delta(t) network weight time evolution."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    dates = pd.to_datetime(results_df['date'])
    mask = results_df['success'].values

    ax = axes[0]
    ax.plot(dates[mask], results_df.loc[mask, 'delta_ret'],
            label='delta_ret (return network)', color='steelblue', linewidth=1.2)
    ax.plot(dates[mask], results_df.loc[mask, 'delta_vol'],
            label='delta_vol (volatility network)', color='coral', linewidth=1.2)
    ax.set_ylabel('Network weight delta')
    ax.set_title('Time Evolution of Multilayer Network Weights')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Mark important events
    events = {
        '2008-09-15': 'Lehman',
        '2015-06-15': 'China Crash',
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

    ax = axes[1]
    ax.plot(dates[mask], results_df.loc[mask, 'loglik'],
            color='darkgreen', linewidth=0.8)
    ax.set_ylabel('Concentrated Log-likelihood')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved: {save_path}")
    plt.show()


def plot_centrality_heatmap(centrality_df: pd.DataFrame,
                            industry_names: List[str],
                            save_path: str = None):
    """Plot centrality time evolution heatmap."""
    dates = pd.to_datetime(centrality_df['date'])
    K = len(industry_names)

    cent_cols = [f'industry_{i}' for i in range(K)]
    cent_matrix = centrality_df[cent_cols].values.T  # (K, T_eval)

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(cent_matrix, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')

    ax.set_yticks(range(K))
    ax.set_yticklabels(industry_names, fontsize=7)

    n_ticks = min(10, len(dates))
    tick_pos = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([dates.iloc[i].strftime('%Y-%m') for i in tick_pos],
                       rotation=45, ha='right', fontsize=8)

    ax.set_title('Time Evolution of Composite Network Eigenvector Centrality')
    ax.set_xlabel('Date')
    ax.set_ylabel('Industry')
    plt.colorbar(im, ax=ax, label='Centrality', shrink=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved: {save_path}")
    plt.show()


def plot_top_central_industries(centrality_df: pd.DataFrame,
                                industry_names: List[str],
                                top_n: int = 5,
                                save_path: str = None):
    """Plot time series of top central industries."""
    dates = pd.to_datetime(centrality_df['date'])
    K = len(industry_names)
    cent_cols = [f'industry_{i}' for i in range(K)]
    cent_matrix = centrality_df[cent_cols].values  # (T_eval, K)

    mean_cent = np.nanmean(cent_matrix, axis=0)
    top_idx = np.argsort(mean_cent)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx in top_idx:
        ax.plot(dates, cent_matrix[:, idx],
                label=industry_names[idx], linewidth=1.2, alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Eigenvector Centrality')
    ax.set_title(f'Top {top_n} Most Central Industries')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved: {save_path}")
    plt.show()


# =============================================================================
# Part 5: Main Entry Point
# =============================================================================

def main_rolling(data_dir: str = "D:/桌面数据/工作论文/带耦合多层网络/src/data/processed",
                 results_dir: str = "D:/桌面数据/工作论文/带耦合多层网络/results",
                 window: int = 52,
                 step: int = 4,
                 factor_method: str = "equal_weight"):
    """
    Rolling window MLE estimation main function.

    Prerequisites: TVP-VAR-DY.R must have been run to generate:
        - dy_all_ret.csv, dy_all_vol.csv
        - dy_dates_ret.csv, dy_dates_vol.csv

    Args:
        data_dir: data directory
        results_dir: results output directory
        window: rolling window size (default 52 = 1 year of weekly data)
        step: step size (default 4 ~ monthly)
        factor_method: factor construction method ("equal_weight" or "pca")
    """
    os.makedirs(results_dir, exist_ok=True)

    # Check that R output files exist
    required_files = [
        "dy_all_ret.csv", "dy_all_vol.csv",
        "dy_dates_ret.csv", "dy_dates_vol.csv",
        "industry_returns.csv",
    ]
    for fname in required_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Required file not found: {fpath}\n"
                f"Please run TVP-VAR-DY.R first to generate network files."
            )

    print("=" * 60)
    print("Rolling Window MLE — Multilayer Network Factor Model")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")

    # Determine K from industry_names.txt
    names_path = os.path.join(data_dir, "industry_names.txt")
    if os.path.exists(names_path):
        with open(names_path, 'r', encoding='utf-8') as f:
            K = sum(1 for line in f if line.strip())
        print(f"  K = {K} industries (from industry_names.txt)")
    else:
        K = 31
        print(f"  K = {K} industries (default)")

    W_ret_3d = load_stacked_networks(
        os.path.join(data_dir, "dy_all_ret.csv"), K=K)
    W_vol_3d = load_stacked_networks(
        os.path.join(data_dir, "dy_all_vol.csv"), K=K)
    net_dates_ret = load_network_dates(
        os.path.join(data_dir, "dy_dates_ret.csv"))
    net_dates_vol = load_network_dates(
        os.path.join(data_dir, "dy_dates_vol.csv"))
    R_full, ret_dates, industry_names = load_returns(
        os.path.join(data_dir, "industry_returns.csv"))

    # 2. Date alignment
    print("\n[2/5] Aligning dates...")
    common_dates, idx_maps = align_data(ret_dates, net_dates_ret, net_dates_vol)

    # 3. Construct factors
    print("\n[3/5] Constructing common factor...")
    F_full = construct_factors(R_full, method=factor_method)

    # 4. Validation (single-point test)
    print("\n[4/5] Single-point validation...")
    mid = len(common_dates) // 2
    test_ret_idx = idx_maps['ret'][mid - window + 1: mid + 1]
    R_test = R_full[test_ret_idx]
    F_test = F_full[test_ret_idx]
    W_ret_test = W_ret_3d[idx_maps['net_ret'][mid]]
    W_vol_test = W_vol_3d[idx_maps['net_vol'][mid]]

    # Check network validity
    print(f"  Row sum check (should ~1): ret={W_ret_test.sum(axis=1).mean():.4f}, "
          f"vol={W_vol_test.sum(axis=1).mean():.4f}")
    print(f"  Diagonal check (should =0): ret={np.diag(W_ret_test).sum():.6f}, "
          f"vol={np.diag(W_vol_test).sum():.6f}")

    test_res = estimate_single_window(R_test, F_test, [W_ret_test, W_vol_test])
    print(f"  Single-point estimate: delta_ret={test_res['delta_hat'][0]:.3f}, "
          f"delta_vol={test_res['delta_hat'][1]:.3f}, "
          f"converged={test_res['success']}")

    if not test_res['success']:
        print("  WARNING: Single-point validation did not converge, proceeding anyway")

    # 5. Rolling estimation
    print("\n[5/5] Starting rolling estimation...")
    results_df, centrality_df = rolling_estimation(
        R_full, F_full, W_ret_3d, W_vol_3d,
        common_dates, idx_maps,
        window=window, step=step)

    # Save results
    results_path = os.path.join(results_dir, "rolling_delta.csv")
    centrality_path = os.path.join(results_dir, "rolling_centrality.csv")
    results_df.to_csv(results_path, index=False)
    centrality_df.to_csv(centrality_path, index=False)
    print(f"\nResults saved:")
    print(f"  {results_path}")
    print(f"  {centrality_path}")

    # Visualize
    print("\nGenerating plots...")
    plot_delta_evolution(
        results_df,
        save_path=os.path.join(results_dir, "delta_evolution.png"))
    plot_centrality_heatmap(
        centrality_df, industry_names,
        save_path=os.path.join(results_dir, "centrality_heatmap.png"))
    plot_top_central_industries(
        centrality_df, industry_names, top_n=5,
        save_path=os.path.join(results_dir, "top_central_industries.png"))

    print("\n" + "=" * 60)
    print("Rolling estimation complete!")
    print("=" * 60)

    return results_df, centrality_df


if __name__ == "__main__":
    main_rolling()
