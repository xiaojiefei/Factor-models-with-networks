"""
多层网络因子模型 — 窗口内固定网络的滚动集中 MLE 估计

结构方程: A * R_t = alpha + beta * F_t + eta_t
其中 A = I - Lambda * (sum_j delta_j * W_j)

本脚本是 Bonaccolto et al. (2019) 固定网络模型的 rolling static-window extension:
每个窗口内先构造固定 W_j，再使用 paper-aligned concentrated MLE。

数据管线:
  管线A (日度10行业): lm_har.py → RollingVAR-DY.R/TVP-VAR-DY.R → 本脚本
  管线B (周度31行业): prepare_data.py → TVP-VAR-DY.R → 本脚本

参考文献:
  Bonaccolto et al. (2019) — Estimation and model-based combination
  of causality networks among large US banks and insurance companies

Usage:
  python multilayer_network_mle_static_window.py
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

_BASE = "C:/Users/Administrator/Desktop/带耦合多层网络"


@dataclass
class NetworkConfig:
    data_dir: str
    results_dir: str
    K: int
    window: int
    step: int
    network_window: int
    factor_method: str
    layer_names: List[str]
    layer_labels: List[str]
    network_files: List[str]
    date_files: List[str]
    returns_file: str
    industry_names: List[str]
    lm_codes: List[str] = field(default_factory=list)
    rho_init_values: List[float] = field(default_factory=lambda: [-0.5, -0.1, 0.1, 0.3, 0.5])
    maxiter: int = 2000
    ftol: float = 1e-8
    max_windows: Optional[int] = None
    n_jobs: Optional[int] = None

    @classmethod
    def daily_10(cls) -> "NetworkConfig":
        return cls(
            data_dir=f"{_BASE}/data/raw/lm_results",
            results_dir=f"{_BASE}/results",
            K=10,
            window=90,
            step=5,
            network_window=1,
            factor_method="pca",
            layer_names=["csv", "jsv_pos", "jsv_neg"],
            layer_labels=["Continuous Vol", "Positive Jump", "Negative Jump"],
            network_files=[
                "granger_loo_all_csv.csv", "granger_loo_all_jsv_pos.csv", "granger_loo_all_jsv_neg.csv",
            ],
            date_files=[
                "granger_loo_dates_csv.csv", "granger_loo_dates_jsv_pos.csv", "granger_loo_dates_jsv_neg.csv",
            ],
            returns_file="",
            lm_codes=[f"{c:06d}" for c in range(32, 42)],
            industry_names=[
                "能源", "原材料", "工业", "可选消费", "主要消费",
                "医药卫生", "金融地产", "信息技术", "通信服务", "公用事业",
            ],
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
            network_window=52,
            factor_method="equal_weight",
            layer_names=["ret", "vol"],
            layer_labels=["Return Network", "Volatility Network"],
            network_files=["dy_all_ret.csv", "dy_all_vol.csv"],
            date_files=["dy_dates_ret.csv", "dy_dates_vol.csv"],
            returns_file="industry_returns.csv",
            lm_codes=[],
            industry_names=[],
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


def window_mean_row_normalize(W_window: np.ndarray) -> np.ndarray:
    """Average the window's raw network matrices, then row-normalize the fixed network."""
    if len(W_window) == 0:
        raise ValueError("Network window is empty")
    W_fixed = np.nanmean(W_window, axis=0)
    if not np.all(np.isfinite(W_fixed)):
        raise ValueError("Network window contains no finite fixed matrix")
    return row_normalize(W_fixed)


def _window_slice(end_idx: int, window: int) -> slice:
    start_idx = end_idx - window + 1
    if start_idx < 0:
        raise ValueError(f"Window ending at {end_idx} is shorter than {window} periods")
    return slice(start_idx, end_idx + 1)


# =============================================================================
# Part 2: Core Estimation Functions (Concentrated MLE)
# =============================================================================

def _delta_to_softmax(delta: np.ndarray) -> np.ndarray:
    """Convert d simplex weights to d-1 unconstrained softmax parameters."""
    delta_clipped = np.clip(delta, 1e-10, None)
    return np.log(delta_clipped[1:] / delta_clipped[0])


def concentrated_loglik(b, W, Y, X):
    """
    Negative concentrated log-likelihood.
    Direct translation of MATLAB LogLikFullHet_combW.m (Panzica Roberto).

    b: parameter vector
       F==1: b = [rho_1, ..., rho_N]
       F>1:  b = [b_0(unused), b_1, ..., b_{F-1}, rho_1, ..., rho_N]
             delta_0 = 1/(1+d), delta_j = exp(b_j)/(1+d), d = sum(exp(b[1:F]))
    W: list of F weight matrices, each NxN
    Y: TxN matrix of returns
    X: TxK matrix of explanatory variables
    """
    F = len(W)
    N = Y.shape[1]
    T = Y.shape[0]

    # d=sum(exp(b(2:F)));
    d = np.sum(np.exp(b[1:F]))

    # construct A
    Et = np.zeros_like(Y, dtype=float)
    if F == 1:
        # A=eye(size(Y,2))-diag(b)*W;
        A = np.eye(N) - np.diag(b) @ W[0]
    elif F > 1:
        # W_star = (1/(1+d))*W{1} + (exp(b(2))/(1+d))*W{2} + ...
        W_star = (1.0 / (1.0 + d)) * W[0]
        for hh in range(1, F):
            W_star = W_star + (np.exp(b[hh]) / (1.0 + d)) * W[hh]
        # A=eye(N)-diag(b(F+1:end))*W_star;
        A = np.eye(N) - np.diag(b[F:]) @ W_star

    # Y1=Y-repmat(mean(Y),size(Y,1),1);
    Y1 = Y - np.mean(Y, axis=0, keepdims=True)
    # X1=X-repmat(mean(X),size(X,1),1);
    X1 = X - np.mean(X, axis=0, keepdims=True)

    # for j=1:size(Y1,1); Et(j,:)=(A*(Y1(j,:)'))'; end
    for j in range(T):
        Et[j, :] = A @ Y1[j, :]

    # XtX=(X1'*X1); XtY=(X1'*Et);
    XtX = X1.T @ X1
    XtY = X1.T @ Et
    # B=(XtX\XtY)';
    B = np.linalg.solve(XtX, XtY).T
    # Et2=Et-X1*B';
    Et2 = Et - X1 @ B.T

    # O=diag(diag(cov(Et2)));
    O = np.diag(np.diag(np.cov(Et2, rowvar=False)))

    # l=0;
    l = 0.0
    diag_O = np.diag(O)
    # for j=1:size(Y1,1);
    #     l=l+log(det(A))-0.5*log(det(O))-0.5*(((Et2(j,:))/O)*Et2(j,:)');
    # end
    for j in range(T):
        l = l + np.log(np.linalg.det(A)) - 0.5 * np.log(np.linalg.det(O)) \
            - 0.5 * np.sum(Et2[j, :] ** 2 / diag_O)

    # L=-l;
    L = -l
    return L


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
                           rho_init_values: Optional[List[float]] = None,
                           maxiter: int = 1000,
                           ftol: float = 1e-8) -> Dict:
    """
    One-step concentrated MLE matching MATLAB LogLikFullHet_combW.m.

    Uses softmax parametrization for delta (unconstrained optimization).
    Parameter vector matches MATLAB convention:
      F==1: b = [rho_1, ..., rho_N]
      F>1:  b = [b_0(unused), b_1, ..., b_{F-1}, rho_1, ..., rho_N]
    """
    d = len(W_list)
    K = R_win.shape[1]
    if any(W.ndim != 2 for W in W_list):
        raise ValueError("Static-window estimation expects each network layer to be a 2D matrix")

    if multi_start:
        init_deltas = _generate_init_deltas(d)
    else:
        init_deltas = [np.ones(d) / d]
    rho_starts = rho_init_values or [-0.5, -0.1, 0.1, 0.3, 0.5]

    best_result = None
    best_loglik = -np.inf

    for delta_init in init_deltas:
        for rho_init_val in rho_starts:
            if d == 1:
                x0 = np.full(K, rho_init_val)
            else:
                free_delta = _delta_to_softmax(delta_init)
                x0 = np.concatenate([[0.0], free_delta, np.full(K, rho_init_val)])
            try:
                result = minimize(
                    concentrated_loglik, x0,
                    args=(W_list, R_win, F_win),
                    method='L-BFGS-B',
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

    b_opt = best_result.x
    if d == 1:
        delta_hat = np.array([1.0])
        rho_hat = b_opt
    else:
        dd = np.sum(np.exp(b_opt[1:d]))
        delta_hat = np.empty(d)
        delta_hat[0] = 1.0 / (1.0 + dd)
        for j in range(1, d):
            delta_hat[j] = np.exp(b_opt[j]) / (1.0 + dd)
        rho_hat = b_opt[d:]

    return {
        'delta_hat': delta_hat,
        'lambda_hat': rho_hat,
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
                         return_window: int,
                         network_window: int,
                         rho_init_values: List[float],
                         maxiter: int,
                         ftol: float) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    count, ci = args
    K = R.shape[1]
    date_t = common_dates[ci]

    ret_slice = _window_slice(ci, return_window)
    ret_indices = idx_maps['ret'][ret_slice]
    R_win = R[ret_indices]
    F_win = F[ret_indices]

    net_slice = _window_slice(ci, network_window)
    W_list_fixed = []
    for name, W_3d in zip(layer_names, W_layers_3d):
        net_indices = idx_maps[f'net_{name}'][net_slice]
        W_list_fixed.append(window_mean_row_normalize(W_3d[net_indices]))

    res = estimate_single_window(R_win, F_win, W_list_fixed,
                                 rho_init_values=rho_init_values,
                                 maxiter=maxiter,
                                 ftol=ftol)

    if res['success'] and not np.any(np.isnan(res['delta_hat'])):
        W_star = sum(dj * W for dj, W in zip(res['delta_hat'], W_list_fixed))
        Lambda = np.diag(res['lambda_hat'])
        W_effective = Lambda @ W_star
        composite_cent = compute_eigenvector_centrality(W_star)
        effective_cent = compute_eigenvector_centrality(W_effective)
    else:
        composite_cent = np.full(K, np.nan)
        effective_cent = np.full(K, np.nan)

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
        cent_row[f'composite_industry_{i}'] = composite_cent[i]
        cent_row[f'effective_industry_{i}'] = effective_cent[i]

    return count, result_row, cent_row


def rolling_estimation(R: np.ndarray, F: np.ndarray,
                       W_layers_3d: List[np.ndarray],
                       layer_names: List[str],
                       common_dates: pd.DatetimeIndex,
                       idx_maps: Dict,
                       window: int = 252,
                       step: int = 5,
                       network_window: Optional[int] = None,
                       verbose: bool = True,
                       n_jobs: Optional[int] = None,
                       rho_init_values: Optional[List[float]] = None,
                       maxiter: int = 1000,
                       ftol: float = 1e-8,
                       max_windows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parallel rolling window one-step MLE with fixed networks inside each window.

    Each window first averages every layer's raw daily matrices into one K x K
    matrix, then row-normalizes that fixed matrix before MLE.
    """
    d = len(W_layers_3d)
    n_common = len(common_dates)
    network_window_eff = network_window or window
    min_start = max(window, network_window_eff) - 1

    eval_indices = list(range(min_start, n_common, step))
    if max_windows is not None:
        eval_indices = eval_indices[:max_windows]
    n_eval = len(eval_indices)
    if n_eval == 0:
        raise ValueError(
            f"Not enough common dates ({n_common}) for return window {window} "
            f"and network window {network_window_eff}"
        )
    n_jobs_eff = n_jobs or max((os.cpu_count() or 2) - 1, 1)
    rho_starts = rho_init_values or [-0.5, -0.1, 0.1, 0.3, 0.5]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Rolling Window One-Step Concentrated MLE — Static Window Networks")
        print(f"{'='*60}")
        print(f"  Return window: {window} periods")
        print(f"  Network window: {network_window_eff} network periods")
        print(f"  Step: {step} common-date periods")
        print(f"  Layers: {d} ({', '.join(layer_names)})")
        print(f"  Network inside window: fixed mean matrix per layer")
        print(f"  Rho starts: {rho_starts}")
        print(f"  L-BFGS-B maxiter: {maxiter}, ftol: {ftol}")
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
            common_dates, window, network_window_eff,
            rho_starts,
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
                            save_path: str = None,
                            prefix: str = "composite"):
    """Plot centrality time evolution heatmap."""
    dates = pd.to_datetime(centrality_df['date'])
    K = len(industry_names)

    cent_cols = [f'{prefix}_industry_{i}' for i in range(K)]
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

    ax.set_title(f'Time Evolution of {prefix.title()} Network Eigenvector Centrality')
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
                                save_path: str = None,
                                prefix: str = "composite"):
    """Plot time series of top central industries."""
    dates = pd.to_datetime(centrality_df['date'])
    K = len(industry_names)
    cent_cols = [f'{prefix}_industry_{i}' for i in range(K)]
    cent_matrix = centrality_df[cent_cols].values

    mean_cent = np.nanmean(cent_matrix, axis=0)
    top_idx = np.argsort(mean_cent)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx in top_idx:
        ax.plot(dates, cent_matrix[:, idx],
                label=industry_names[idx], linewidth=1.2, alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Eigenvector Centrality')
    ax.set_title(f'Top {top_n} Most Central Industries ({prefix.title()} Network)')
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
    print("Rolling Window MLE — Static Window Multilayer Network Factor Model")
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
    mid = max(config.window - 1, config.network_window - 1, len(common_dates) // 2)
    if mid >= len(common_dates):
        raise ValueError(
            f"Not enough common dates ({len(common_dates)}) for return window "
            f"{config.window} and network window {config.network_window}"
        )
    ret_slice = _window_slice(mid, config.window)
    net_slice = _window_slice(mid, config.network_window)
    test_ret_idx = idx_maps['ret'][ret_slice]
    R_test = R_full[test_ret_idx]
    F_test = F_full[test_ret_idx]

    W_test_list = []
    for name, W_3d in zip(config.layer_names, W_layers_3d):
        net_indices = idx_maps[f'net_{name}'][net_slice]
        W_fixed = window_mean_row_normalize(W_3d[net_indices])
        print(f"  [{name}] fixed_window row_sum={W_fixed.sum(axis=1).mean():.4f}, "
              f"diag_sum={np.diag(W_fixed).sum():.6f}")
        W_test_list.append(W_fixed)

    test_res = estimate_single_window(R_test, F_test, W_test_list,
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
        network_window=config.network_window,
        n_jobs=config.n_jobs,
        rho_init_values=config.rho_init_values,
        maxiter=config.maxiter,
        ftol=config.ftol,
        max_windows=config.max_windows)

    # Save results
    results_path = os.path.join(config.results_dir, "rolling_delta_static_window.csv")
    centrality_path = os.path.join(config.results_dir, "rolling_centrality_static_window.csv")
    results_df.to_csv(results_path, index=False)
    centrality_df.to_csv(centrality_path, index=False)
    print(f"\nResults saved:")
    print(f"  {results_path}")
    print(f"  {centrality_path}")

    # Visualize
    print("\nGenerating plots...")
    plot_delta_evolution(
        results_df, config.layer_names, config.layer_labels,
        save_path=os.path.join(config.results_dir, "delta_evolution_static_window.png"))
    plot_centrality_heatmap(
        centrality_df, industry_names,
        save_path=os.path.join(config.results_dir, "centrality_heatmap_static_window.png"))
    plot_top_central_industries(
        centrality_df, industry_names, top_n=5,
        save_path=os.path.join(config.results_dir, "top_central_industries_static_window.png"))

    print("\n" + "=" * 60)
    print("Rolling estimation complete!")
    print("=" * 60)

    return results_df, centrality_df


if __name__ == "__main__":
    main_rolling()
