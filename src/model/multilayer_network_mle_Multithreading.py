"""
多层网络因子模型 — 滚动窗口集中 MLE 估计

结构方程: A * R_t = alpha + beta * F_t + eta_t
其中 A = I - Lambda * (sum_j delta_j * W_j)

数据管线:
  管线A (日度10行业): lm_har.py → RollingVAR-DY.R → 本脚本
  管线B (周度31行业): prepare_data.py → TVP-VAR-DY.R → 本脚本

参考文献:
  Bonaccolto et al. (2019) — Estimation and model-based combination
  of causality networks among large US banks and insurance companies

Usage:
  python multilayer_network_mle_Multithreading.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Plot settings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# Configuration
# =============================================================================

_BASE = "D:/桌面数据/工作论文/带耦合多层网络"


@dataclass
class NetworkConfig:
    data_dir: str
    results_dir: str
    K: int
    window: int
    step: int
    factor_method: str
    layer_names: List[str]
    layer_labels: List[str]
    network_files: List[str]
    date_files: List[str]
    returns_file: str
    industry_names: List[str]
    lm_codes: List[str] = field(default_factory=list)
    dirichlet_alpha: float = 1.0
    rho_init_values: List[float] = field(default_factory=lambda: [-0.5, -0.1, 0.1, 0.3, 0.5])
    maxiter: int = 1000
    ftol: float = 1e-8
    max_windows: Optional[int] = None
    n_jobs: Optional[int] = None

    @classmethod
    def daily_10(cls) -> "NetworkConfig":
        return cls(
            data_dir=f"{_BASE}/data/raw/lm_results",
            results_dir=f"{_BASE}/results",
            K=10,
            window=252,
            step=1,
            factor_method="equal_weight",
            layer_names=["csv", "jsv_pos", "jsv_neg"],
            layer_labels=["Continuous Vol", "Positive Jump", "Negative Jump"],
            network_files=[
                "dy_tvp_all_csv.csv", "dy_tvp_all_jsv_pos.csv", "dy_tvp_all_jsv_neg.csv",
            ],
            date_files=[
                "dy_tvp_dates_csv.csv", "dy_tvp_dates_jsv_pos.csv", "dy_tvp_dates_jsv_neg.csv",
            ],
            returns_file="",
            lm_codes=[f"{c:06d}" for c in range(32, 42)],
            industry_names=[
                "能源", "原材料", "工业", "可选消费", "主要消费",
                "医药卫生", "金融地产", "信息技术", "通信服务", "公用事业",
            ],
            dirichlet_alpha=1.1,
            n_jobs=None,
        )

    @classmethod
    def weekly_31(cls) -> "NetworkConfig":
        return cls(
            data_dir=f"{_BASE}/src/data/processed",
            results_dir=f"{_BASE}/results",
            K=31,
            window=52,
            step=4,
            factor_method="equal_weight",
            layer_names=["ret", "vol"],
            layer_labels=["Return Network", "Volatility Network"],
            network_files=["dy_all_ret.csv", "dy_all_vol.csv"],
            date_files=["dy_dates_ret.csv", "dy_dates_vol.csv"],
            returns_file="industry_returns.csv",
            lm_codes=[],
            industry_names=[],
            dirichlet_alpha=1.2,
            n_jobs=None,
        )


# =============================================================================
# Part 1: Network Utility Functions
# =============================================================================

def row_normalize(W: np.ndarray) -> np.ndarray:
    """Row-normalize adjacency matrix: each row sums to 1, diagonal = 0."""
    W_norm = W.copy()
    np.fill_diagonal(W_norm, 0)
    row_sums = W_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
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


def max_row_normalize(W_3d: np.ndarray) -> np.ndarray:
    """
    Max row normalization for time-varying networks (Eq.25 of Bonaccolto 2019).

    For each column j, divide by the max column-sum across all time points:
      W_{i,j,t} = W^U_{i,j,t} / max_t( sum_i W^U_{i,j,t} )

    This preserves relative density changes over time, unlike standard
    row normalization which forces each row to sum to 1 at every t.
    """
    T_net, K, _ = W_3d.shape
    W_out = W_3d.copy()

    for t in range(T_net):
        np.fill_diagonal(W_out[t], 0)

    col_sums = W_out.sum(axis=1)  # (T_net, K) — column sum for each t
    max_col_sums = col_sums.max(axis=0)  # (K,) — max across time for each column
    max_col_sums[max_col_sums == 0] = 1.0

    for t in range(T_net):
        W_out[t] = W_out[t] / max_col_sums[np.newaxis, :]

    print(f"  Max-normalized: row_sum range [{W_out.sum(axis=2).min():.3f}, "
          f"{W_out.sum(axis=2).max():.3f}]")
    return W_out


# =============================================================================
# Part 2: Core Estimation Functions (Concentrated MLE)
# =============================================================================

def estimate_beta_sigma(A: np.ndarray, R: np.ndarray, F: np.ndarray,
                        include_alpha: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 1 of concentrated MLE: given A, estimate alpha, beta, Sigma_eta via OLS.

    Structural equation: A * R = alpha + beta * F + eta
    """
    T, K = R.shape
    if A.ndim == 2:
        R_tilde = (A @ R.T).T
    elif A.ndim == 3:
        if A.shape[0] != T:
            raise ValueError(f"A has {A.shape[0]} time points, but R has {T} observations")
        R_tilde = np.einsum('tij,tj->ti', A, R)
    else:
        raise ValueError(f"A must be 2D or 3D, got shape {A.shape}")

    if include_alpha:
        F_aug = np.column_stack([np.ones(T), F])
        B_hat = np.linalg.solve(F_aug.T @ F_aug, F_aug.T @ R_tilde).T
        alpha_hat = B_hat[:, 0]
        beta_hat = B_hat[:, 1:]
        eta_hat = R_tilde - F_aug @ B_hat.T
    else:
        alpha_hat = np.zeros(K)
        beta_hat = np.linalg.solve(F.T @ F, F.T @ R_tilde).T
        eta_hat = R_tilde - F @ beta_hat.T

    sigma_sq = np.var(eta_hat, axis=0, ddof=0)
    Sigma_eta = np.diag(sigma_sq)

    return alpha_hat, beta_hat, eta_hat, Sigma_eta


def _compose_network_sequence(delta: np.ndarray, W_list: List[np.ndarray]) -> np.ndarray:
    W_arrays = [W if W.ndim == 3 else W[np.newaxis, :, :] for W in W_list]
    T_net = W_arrays[0].shape[0]
    if any(W.shape[0] != T_net for W in W_arrays):
        raise ValueError("All network layers must have the same number of time points")
    if len(delta) != len(W_arrays):
        raise ValueError(f"delta has {len(delta)} elements, but there are {len(W_arrays)} layers")
    return sum(dj * W for dj, W in zip(delta, W_arrays))


def _paper_rho_bounds(W: np.ndarray) -> Tuple[float, float]:
    eigvals = np.linalg.eigvals(W)
    eigvals = np.real_if_close(eigvals, tol=1000).real
    positive = eigvals[eigvals > 1e-10]
    negative = eigvals[eigvals < -1e-10]

    upper = 1.0 / positive.max() if positive.size > 0 else np.inf
    lower = 1.0 / negative.min() if negative.size > 0 else -np.inf
    return lower, upper


def _is_stable_network_effect(B: np.ndarray, tol: float = 1e-8) -> bool:
    spectral_radius = np.max(np.abs(np.linalg.eigvals(B)))
    return spectral_radius < 1.0 - tol


def concentrated_loglik(params: np.ndarray, R: np.ndarray, F: np.ndarray,
                        W_list: List[np.ndarray],
                        scalar_rho: bool = False,
                        dirichlet_alpha: float = 0.0) -> float:
    """
    Negative concentrated log-likelihood.

    ln L_c = -T/2 * ln|Sigma_eta| + sum_t ln|A_t| + const

    Args:
        scalar_rho: if False, params = [delta_1..d, rho_1..K] (Bonaccolto one-step MLE);
                    if True, params = [delta_1..d, rho] (legacy scalar-rho path)
        dirichlet_alpha: symmetric Dirichlet prior strength (0 = no prior)
    """
    d = len(W_list)
    K = R.shape[1]
    T = R.shape[0]

    delta = params[:d]

    if not np.isclose(np.sum(delta), 1.0, atol=1e-6):
        return 1e10
    if np.any(delta < -1e-10):
        return 1e10

    W_star_seq = _compose_network_sequence(delta, W_list)

    if scalar_rho:
        rho_scalar = params[d]
        bounds = [_paper_rho_bounds(W_star) for W_star in W_star_seq]
        lower = max(bound[0] for bound in bounds)
        upper = min(bound[1] for bound in bounds)
        if not (lower < rho_scalar < upper):
            return 1e10
        A = np.eye(K) - rho_scalar * W_star_seq
    else:
        rho = params[d:]
        Lambda = np.diag(rho)
        if any(not _is_stable_network_effect(Lambda @ W_star) for W_star in W_star_seq):
            return 1e10
        A = np.array([np.eye(K) - Lambda @ W_star for W_star in W_star_seq])

    try:
        signs, logabsdets = np.linalg.slogdet(A)
        if np.any(signs <= 0) or np.any(logabsdets < -20):
            return 1e10
        logabsdet_sum = np.sum(logabsdets)
    except Exception:
        return 1e10

    _, _, _, Sigma_eta = estimate_beta_sigma(A, R, F)

    diag_Sigma = np.diag(Sigma_eta)
    if np.any(diag_Sigma <= 0):
        return 1e10

    logdet_Sigma = np.sum(np.log(diag_Sigma))

    loglik = -0.5 * T * logdet_Sigma + logabsdet_sum

    if dirichlet_alpha > 0:
        delta_clipped = np.clip(delta, 1e-10, None)
        loglik += (dirichlet_alpha - 1.0) * np.sum(np.log(delta_clipped))

    return -loglik


# =============================================================================
# Part 3: Rolling Window MLE — Data Loading and Estimation
# =============================================================================

def load_stacked_networks(filepath: str, K: int = 31) -> np.ndarray:
    """
    Load R-exported stacked adjacency matrix CSV, reshape to 3D array.

    File format: (K * T_net) rows x K columns, no header.
    """
    raw = np.loadtxt(filepath, delimiter=',')
    T_net = raw.shape[0] // K
    assert raw.shape[0] == K * T_net, \
        f"Row count {raw.shape[0]} is not a multiple of K={K}"
    assert raw.shape[1] == K, \
        f"Column count {raw.shape[1]} != K={K}"
    W_3d = raw.reshape(T_net, K, K)
    print(f"  Loaded network: {os.path.basename(filepath)}")
    print(f"  Shape: {T_net} time points x {K}x{K} adjacency matrices")
    return W_3d


def load_network_dates(filepath: str) -> pd.DatetimeIndex:
    """Load date mapping file."""
    df = pd.read_csv(filepath)
    dates = pd.to_datetime(df['date'])
    print(f"  Date range: {dates.iloc[0].date()} ~ {dates.iloc[-1].date()} ({len(dates)} periods)")
    return dates


def load_returns(filepath: str) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """Load industry returns from CSV file."""
    df = pd.read_csv(filepath)
    dates = pd.to_datetime(df['date'])
    names = [c for c in df.columns if c != 'date']
    R = df[names].values
    print(f"  Returns: {R.shape[0]} periods x {R.shape[1]} industries")
    return R, dates, names


def load_daily_returns_from_lm(data_dir: str,
                               codes: List[str]) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    Load daily returns by merging r_now column from LM HAR result files.

    Each {code}_lm_har.csv has col 0 = date (YYYYMMDD int) and col 'r_now'.
    """
    frames = []
    for code in codes:
        fpath = os.path.join(data_dir, f"{code}_lm_har.csv")
        df = pd.read_csv(fpath)
        date_col = df.columns[0]
        df['date'] = pd.to_datetime(df[date_col].astype(int).astype(str), format='%Y%m%d')
        frames.append(df[['date', 'r_now']].rename(columns={'r_now': code}))

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on='date', how='inner')

    merged = merged.sort_values('date').drop_duplicates(subset='date').reset_index(drop=True)
    dates = pd.DatetimeIndex(merged['date'])
    names = [c for c in merged.columns if c != 'date']
    R = merged[names].values

    print(f"  Returns (LM r_now): {R.shape[0]} days x {R.shape[1]} industries")
    return R, dates, names


def align_data(ret_dates: pd.DatetimeIndex,
               net_dates_list: List[pd.DatetimeIndex],
               layer_names: List[str]) -> Tuple[pd.DatetimeIndex, Dict]:
    """
    Align returns and multiple network date sources by set intersection.

    Returns:
        common_dates: sorted common dates
        idx_maps: {'ret': [...], 'net_csv': [...], 'net_jsv_pos': [...], ...}
    """
    common = set(ret_dates)
    for nd in net_dates_list:
        common = common & set(nd)
    common_dates = pd.DatetimeIndex(sorted(common))

    ret_date2idx = {d: i for i, d in enumerate(ret_dates)}
    idx_maps = {
        'ret': np.array([ret_date2idx[d] for d in common_dates]),
    }
    for name, nd in zip(layer_names, net_dates_list):
        d2i = {d: i for i, d in enumerate(nd)}
        idx_maps[f'net_{name}'] = np.array([d2i[d] for d in common_dates])

    print(f"\n  Date alignment:")
    print(f"    Returns: {len(ret_dates)} periods")
    for name, nd in zip(layer_names, net_dates_list):
        print(f"    Network [{name}]: {len(nd)} periods")
    print(f"    Common dates: {len(common_dates)} periods")
    print(f"    Range: {common_dates[0].date()} ~ {common_dates[-1].date()}")

    return common_dates, idx_maps


def construct_factors(R: np.ndarray, method: str = "equal_weight") -> np.ndarray:
    """Construct common factor F_t from return matrix."""
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


def _generate_init_deltas(d: int) -> List[np.ndarray]:
    """Generate multi-start initial delta vectors on the (d-1)-simplex."""
    if d == 2:
        return [
            np.array([0.5, 0.5]),
            np.array([0.3, 0.7]),
            np.array([0.7, 0.3]),
            np.array([0.1, 0.9]),
        ]
    elif d == 3:
        return [
            np.array([1/3, 1/3, 1/3]),
            np.array([0.6, 0.2, 0.2]),
            np.array([0.2, 0.6, 0.2]),
            np.array([0.2, 0.2, 0.6]),
            np.array([0.45, 0.45, 0.1]),
            np.array([0.45, 0.1, 0.45]),
            np.array([0.1, 0.45, 0.45]),
        ]
    else:
        return [np.ones(d) / d]


def estimate_single_window(R_win: np.ndarray, F_win: np.ndarray,
                           W_list: List[np.ndarray],
                           multi_start: bool = True,
                           dirichlet_alpha: float = 0.0,
                           rho_init_values: Optional[List[float]] = None,
                           maxiter: int = 1000,
                           ftol: float = 1e-8) -> Dict:
    """
    One-step concentrated MLE following Bonaccolto et al. (2019).

    Optimize delta and heterogeneous rho_i jointly:
        A_t = I - diag(rho) * sum_j delta_j W_{j,t}
    """
    d = len(W_list)
    K = R_win.shape[1]
    T = R_win.shape[0]
    W_seq_list = [W if W.ndim == 3 else np.repeat(W[np.newaxis, :, :], T, axis=0)
                  for W in W_list]
    if any(W.shape[0] != T for W in W_seq_list):
        raise ValueError("Each network layer must have one matrix per return observation")

    if multi_start:
        init_deltas = _generate_init_deltas(d)
    else:
        init_deltas = [np.ones(d) / d]
    rho_starts = rho_init_values or [-0.5, -0.1, 0.1, 0.3, 0.5]

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:d]) - 1.0}]
    bounds = [(0, 1)] * d + [(-10.0, 10.0)] * K

    best_result = None
    best_loglik = -np.inf

    for delta_init in init_deltas:
        for rho_init_val in rho_starts:
            x0 = np.concatenate([delta_init, np.full(K, rho_init_val)])
            try:
                result = minimize(
                    concentrated_loglik, x0,
                    args=(R_win, F_win, W_seq_list, False, dirichlet_alpha),
                    method='SLSQP', bounds=bounds, constraints=constraints,
                    options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
                )
                loglik = -result.fun
                if result.success and np.isfinite(loglik) and loglik > best_loglik:
                    best_loglik = loglik
                    best_result = result
            except Exception:
                continue

    if best_result is None:
        return {
            'delta_hat': np.full(d, np.nan),
            'lambda_hat': np.full(K, np.nan),
            'loglik': np.nan,
            'success': False,
        }

    return {
        'delta_hat': best_result.x[:d],
        'lambda_hat': best_result.x[d:],
        'loglik': best_loglik,
        'success': best_result.success,
    }


def compute_eigenvector_centrality(W: np.ndarray) -> np.ndarray:
    """Compute eigenvector centrality of a network."""
    eigvals, eigvecs = np.linalg.eig(W.T)
    idx = np.argmax(np.abs(eigvals))
    v = np.abs(eigvecs[:, idx]).astype(float)
    v_sum = v.sum()
    if v_sum > 0:
        v = v / v_sum
    return v


def _estimate_window_job(args: Tuple[int, int],
                         R: np.ndarray, F: np.ndarray,
                         W_layers_3d: List[np.ndarray],
                         layer_names: List[str],
                         idx_maps: Dict[str, np.ndarray],
                         common_dates: pd.DatetimeIndex,
                         window: int,
                         dirichlet_alpha: float,
                         rho_init_values: List[float],
                         maxiter: int,
                         ftol: float) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    count, ci = args
    K = R.shape[1]
    date_t = common_dates[ci]

    ret_indices = idx_maps['ret'][ci - window + 1: ci + 1]
    R_win = R[ret_indices]
    F_win = F[ret_indices]

    W_list_win = []
    for name, W_3d in zip(layer_names, W_layers_3d):
        net_indices = idx_maps[f'net_{name}'][ci - window + 1: ci + 1]
        W_list_win.append(W_3d[net_indices])

    res = estimate_single_window(R_win, F_win, W_list_win,
                                 dirichlet_alpha=dirichlet_alpha,
                                 rho_init_values=rho_init_values,
                                 maxiter=maxiter,
                                 ftol=ftol)

    W_list_t = [W_win[-1] for W_win in W_list_win]
    if res['success'] and not np.any(np.isnan(res['delta_hat'])):
        W_star = sum(dj * Wt for dj, Wt in zip(res['delta_hat'], W_list_t))
        Lambda = np.diag(res['lambda_hat'])
        W_effective = Lambda @ W_star
        cent = compute_eigenvector_centrality(W_effective)
    else:
        cent = np.full(K, np.nan)

    result_row = {
        'date': date_t,
        'loglik': res['loglik'],
        'success': res['success'],
    }
    for j, name in enumerate(layer_names):
        result_row[f'delta_{name}'] = res['delta_hat'][j]
    for i in range(K):
        result_row[f'lambda_{i}'] = res['lambda_hat'][i]

    cent_row = {'date': date_t}
    for i in range(K):
        cent_row[f'industry_{i}'] = cent[i]

    return count, result_row, cent_row


def rolling_estimation(R: np.ndarray, F: np.ndarray,
                       W_layers_3d: List[np.ndarray],
                       layer_names: List[str],
                       common_dates: pd.DatetimeIndex,
                       idx_maps: Dict,
                       window: int = 252,
                       step: int = 22,
                       dirichlet_alpha: float = 0.0,
                       verbose: bool = True,
                       n_jobs: Optional[int] = None,
                       rho_init_values: Optional[List[float]] = None,
                       maxiter: int = 1000,
                       ftol: float = 1e-8,
                       max_windows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parallel rolling window one-step concentrated MLE estimation.

    Each rolling window is independent, so windows are distributed across processes.
    """
    d = len(W_layers_3d)
    n_common = len(common_dates)

    eval_indices = list(range(window - 1, n_common, step))
    if max_windows is not None:
        eval_indices = eval_indices[:max_windows]
    n_eval = len(eval_indices)
    n_jobs_eff = n_jobs or max((os.cpu_count() or 2) - 1, 1)
    rho_starts = rho_init_values or [-0.5, -0.1, 0.1, 0.3, 0.5]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Rolling Window One-Step Concentrated MLE — Multithreading")
        print(f"{'='*60}")
        print(f"  Window: {window} periods")
        print(f"  Step: {step} periods")
        print(f"  Layers: {d} ({', '.join(layer_names)})")
        print(f"  Dirichlet alpha: {dirichlet_alpha}")
        print(f"  Rho starts: {rho_starts}")
        print(f"  SLSQP maxiter: {maxiter}, ftol: {ftol}")
        print(f"  Common dates: {n_common} periods")
        print(f"  Evaluation points: {n_eval}")
        print(f"  Workers: {n_jobs_eff}")
        print(f"  Date range: {common_dates[eval_indices[0]].date()} ~ "
              f"{common_dates[eval_indices[-1]].date()}")

    tasks = [(count, ci) for count, ci in enumerate(eval_indices)]

    result_slots: List[Optional[Dict[str, Any]]] = [None] * n_eval
    centrality_slots: List[Optional[Dict[str, Any]]] = [None] * n_eval

    job_results = Parallel(
        n_jobs=n_jobs_eff,
        prefer="processes",
    )(
        delayed(_estimate_window_job)(
            task, R, F, W_layers_3d, layer_names, idx_maps,
            common_dates, window, dirichlet_alpha, rho_starts,
            maxiter, ftol
        )
        for task in tasks
    )

    completed = 0
    for count, result_row, cent_row in job_results:
        result_slots[count] = result_row
        centrality_slots[count] = cent_row
        completed += 1

        if verbose and (completed % 20 == 0 or completed == 1 or completed == n_eval):
            status = "OK" if result_row['success'] else "FAIL"
            delta_str = " ".join(
                f"d_{name}={result_row[f'delta_{name}']:.3f}" for name in layer_names)
            lam_values = [result_row[f'lambda_{i}'] for i in range(R.shape[1])]
            lam_mean = np.mean(lam_values)
            print(f"  [{completed:3d}/{n_eval}] {pd.Timestamp(result_row['date']).date()} | "
                  f"{delta_str} rho_avg={lam_mean:.3f} | "
                  f"loglik={result_row['loglik']:.1f} | {status}")

    results_df = pd.DataFrame(result_slots)
    centrality_df = pd.DataFrame(centrality_slots)

    n_success = results_df['success'].sum()
    if verbose:
        print(f"\n  One-step convergence rate: {n_success}/{n_eval} ({100*n_success/n_eval:.1f}%)")
        for name in layer_names:
            col = f'delta_{name}'
            print(f"  {col} mean: {results_df[col].mean():.3f}")

    return results_df, centrality_df


# =============================================================================
# Part 4: Visualization
# =============================================================================

_LAYER_COLORS = ['steelblue', 'coral', 'forestgreen', 'purple', 'orange']


def plot_delta_evolution(results_df: pd.DataFrame,
                         layer_names: List[str],
                         layer_labels: List[str],
                         save_path: str = None):
    """Plot delta(t) network weight time evolution."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    dates = pd.to_datetime(results_df['date'])
    mask = results_df['success'].values

    ax = axes[0]
    for j, (name, label) in enumerate(zip(layer_names, layer_labels)):
        col = f'delta_{name}'
        color = _LAYER_COLORS[j % len(_LAYER_COLORS)]
        ax.plot(dates[mask], results_df.loc[mask, col],
                label=f'{col} ({label})', color=color, linewidth=1.2)
    ax.set_ylabel('Network weight delta')
    ax.set_title('Time Evolution of Multilayer Network Weights')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

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
    cent_matrix = centrality_df[cent_cols].values.T

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
    cent_matrix = centrality_df[cent_cols].values

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

def main_rolling(config: Optional[NetworkConfig] = None):
    """
    Rolling window MLE estimation main function.

    Args:
        config: NetworkConfig instance (defaults to daily_10())
    """
    if config is None:
        config = NetworkConfig.daily_10()

    os.makedirs(config.results_dir, exist_ok=True)

    required = config.network_files + config.date_files
    if config.returns_file:
        required.append(config.returns_file)
    for fname in required:
        fpath = os.path.join(config.data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Required file not found: {fpath}\n"
                f"Please run RollingVAR-DY.R (or TVP-VAR-DY.R) first."
            )

    print("=" * 60)
    print("Rolling Window MLE — Multilayer Network Factor Model (Multithreading)")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    K = config.K
    d = len(config.layer_names)
    print(f"  K = {K} industries, d = {d} layers ({', '.join(config.layer_names)})")

    W_layers_3d = []
    net_dates_list = []
    for nf, df_name in zip(config.network_files, config.date_files):
        W_3d = load_stacked_networks(os.path.join(config.data_dir, nf), K=K)
        W_3d = max_row_normalize(W_3d)
        nd = load_network_dates(os.path.join(config.data_dir, df_name))
        W_layers_3d.append(W_3d)
        net_dates_list.append(nd)

    if config.lm_codes:
        R_full, ret_dates, ret_names = load_daily_returns_from_lm(
            config.data_dir, config.lm_codes)
    else:
        R_full, ret_dates, ret_names = load_returns(
            os.path.join(config.data_dir, config.returns_file))

    industry_names = config.industry_names if config.industry_names else ret_names

    # 2. Date alignment
    print("\n[2/5] Aligning dates...")
    common_dates, idx_maps = align_data(ret_dates, net_dates_list, config.layer_names)

    # 3. Construct factors
    print("\n[3/5] Constructing common factor...")
    F_full = construct_factors(R_full, method=config.factor_method)

    # 4. Validation (single-point test)
    print("\n[4/5] Single-point validation...")
    mid = len(common_dates) // 2
    test_ret_idx = idx_maps['ret'][mid - config.window + 1: mid + 1]
    R_test = R_full[test_ret_idx]
    F_test = F_full[test_ret_idx]

    W_test_list = []
    for name, W_3d in zip(config.layer_names, W_layers_3d):
        net_indices = idx_maps[f'net_{name}'][mid - config.window + 1: mid + 1]
        W_win = W_3d[net_indices]
        W_t = W_win[-1]
        print(f"  [{name}] row_sum={W_t.sum(axis=1).mean():.4f}, "
              f"diag_sum={np.diag(W_t).sum():.6f}")
        W_test_list.append(W_win)

    test_res = estimate_single_window(R_test, F_test, W_test_list,
                                      dirichlet_alpha=config.dirichlet_alpha,
                                      rho_init_values=config.rho_init_values,
                                      maxiter=config.maxiter,
                                      ftol=config.ftol)
    delta_str = ", ".join(
        f"d_{name}={test_res['delta_hat'][j]:.3f}"
        for j, name in enumerate(config.layer_names))
    lam_str = f", rho_avg={np.mean(test_res['lambda_hat']):.3f}"
    status = "OK" if test_res['success'] else "FAIL"
    print(f"  Single-point estimate: {delta_str}{lam_str}, status={status}")

    if not test_res['success']:
        print("  WARNING: Single-point validation did not converge, proceeding anyway")

    # 5. Rolling estimation
    print("\n[5/5] Starting rolling estimation...")
    results_df, centrality_df = rolling_estimation(
        R_full, F_full, W_layers_3d, config.layer_names,
        common_dates, idx_maps,
        window=config.window, step=config.step,
        dirichlet_alpha=config.dirichlet_alpha,
        n_jobs=config.n_jobs,
        rho_init_values=config.rho_init_values,
        maxiter=config.maxiter,
        ftol=config.ftol,
        max_windows=config.max_windows)

    # Save results
    results_path = os.path.join(config.results_dir, "rolling_delta_multithreading.csv")
    centrality_path = os.path.join(config.results_dir, "rolling_centrality_multithreading.csv")
    results_df.to_csv(results_path, index=False)
    centrality_df.to_csv(centrality_path, index=False)
    print(f"\nResults saved:")
    print(f"  {results_path}")
    print(f"  {centrality_path}")

    # Visualize
    print("\nGenerating plots...")
    plot_delta_evolution(
        results_df, config.layer_names, config.layer_labels,
        save_path=os.path.join(config.results_dir, "delta_evolution_multithreading.png"))
    plot_centrality_heatmap(
        centrality_df, industry_names,
        save_path=os.path.join(config.results_dir, "centrality_heatmap_multithreading.png"))
    plot_top_central_industries(
        centrality_df, industry_names, top_n=5,
        save_path=os.path.join(config.results_dir, "top_central_industries_multithreading.png"))

    print("\n" + "=" * 60)
    print("Rolling estimation complete!")
    print("=" * 60)

    return results_df, centrality_df


if __name__ == "__main__":
    main_rolling()
