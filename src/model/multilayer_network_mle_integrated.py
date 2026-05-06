"""
多层网络因子模型 — 集成Granger因果网络计算的滚动集中MLE估计

结构方程: A * R_t = alpha + beta * F_t + eta_t
其中 A = I - Lambda * (sum_j delta_j * W_j)

本脚本在每个MLE窗口内现场计算留一法Granger因果网络，确保收益率窗口与网络计算窗口严格对齐。

数据管线:
  lm_har.py → 本脚本（现场计算Granger网络 + MLE估计）

参考文献:
  Bonaccolto et al. (2019) — Estimation and model-based combination
  of causality networks among large US banks and insurance companies

Usage:
  python multilayer_network_mle_integrated.py
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
from scipy import stats
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

_CANDIDATES = [
    "C:/Users/Administrator/Desktop/带耦合多层网络",
    "D:/桌面数据/工作论文/带耦合多层网络",
]
_BASE = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[0])


@dataclass
class NetworkConfig:
    """Configuration for integrated Granger-MLE estimation."""
    # Data paths
    data_dir: str
    results_dir: str

    # Industry settings
    K: int
    industry_names: List[str]
    lm_codes: List[str]

    # Rolling window parameters - MLE收益率窗口
    # 说明：这是MLE估计中使用的收益率数据窗口长度（交易日）
    # 每个时间点t，使用 [t-window+1, t] 的收益率数据进行估计
    window: int = 252

    # 滚动步长（交易日）
    # 说明：每隔多少个交易日进行一次估计
    # step=1：每天估计（最密集，计算量大）
    # step=5：每周估计（平衡）
    # step=22：每月估计（稀疏，计算快）
    step: int = 5

    # Granger network parameters
    # Granger因果网络计算窗口（交易日）
    # 说明：用于计算Granger因果网络的风险指标数据窗口长度
    # 可以独立于MLE窗口设置，例如：
    #   - MLE用252天（约1年收益率）
    #   - Granger用500天（更长数据获得更稳定的网络）
    # 必须 >= window，且数据量要足够（建议 >= 2*K*lag）
    granger_window: int = 500

    # VAR滞后阶数
    # 说明：Granger因果检验中VAR模型的滞后阶数
    # 常用值：5（周度效应）、10、22（月度效应）
    # 过小：遗漏重要滞后关系；过大：自由度损失
    granger_lag: int = 5

    # Granger因果检验显著性水平
    # 说明：双边卡方检验的显著性水平 alpha
    # alpha越小，网络越稀疏（更多0）；alpha越大，网络越稠密
    # 推荐值：0.10-0.20（解决稀疏性问题）
    # 原设定0.05导致92%零元素，建议至少0.10
    granger_alpha: float = 0.15

    # Network layers
    layer_names: List[str] = field(default_factory=lambda: ["csv", "jsv_pos", "jsv_neg"])
    layer_labels: List[str] = field(default_factory=lambda: ["Continuous Vol", "Positive Jump", "Negative Jump"])
    layer_columns: Dict[str, str] = field(default_factory=lambda: {
        "csv": "CSVt_d",
        "jsv_pos": "JSVt_zheng_d",
        "jsv_neg": "JSVt_fu_d",
    })

    # Factor construction
    factor_method: str = "pca"  # "pca", "equal_weight", or "ff3"
    ff3_path: Optional[str] = None
    ff3_market_id: str = "P9709"  # 全部A股

    # MLE optimization
    rho_init_values: List[float] = field(default_factory=lambda: [-0.5, -0.1, 0.1, 0.3, 0.5])
    maxiter: int = 2000
    ftol: float = 1e-8
    max_windows: Optional[int] = None
    n_jobs: Optional[int] = None

    @classmethod
    def daily_10(cls) -> "NetworkConfig":
        """
        默认配置：10个上证行业指数日度数据。

        窗口设置建议：
        - 短期分析：window=252, granger_window=252（同步）
        - 长期稳定网络：window=252, granger_window=500（分离，推荐）
        """
        return cls(
            data_dir=f"{_BASE}/data/raw/lm_results",
            results_dir=f"{_BASE}/results",
            K=10,
            window=252,           # MLE收益率窗口：252天（约1年）
            step=1,               # 每天估计一次（密集估计）
            granger_window=500,   # Granger窗口：500天（更长数据，更稳定网络）
            granger_lag=5,        # VAR滞后5阶
            granger_alpha=0.15,   # 15%显著性水平（解决稀疏性）
            industry_names=[
                "能源", "原材料", "工业", "可选消费", "主要消费",
                "医药卫生", "金融地产", "信息技术", "通信服务", "公用事业",
            ],
            lm_codes=[f"{c:06d}" for c in range(32, 42)],
            factor_method="ff3",
            ff3_path=f"{_BASE}/data/raw/FF3/三因子模型指标(日)193352534(仅供上海对外经贸大学使用)/STK_MKT_THRFACDAY.csv",
            n_jobs=None,          # 自动使用最大CPU核心数
        )


# =============================================================================
# Part 1: Granger Causality Network Functions
# =============================================================================

def _lag_matrix(series: np.ndarray, lag: int) -> np.ndarray:
    """Create lagged matrix for VAR."""
    rows = []
    for t in range(lag, len(series)):
        rows.append(series[t - lag:t][::-1])
    return np.asarray(rows, dtype=float)


def _residual_variance(y: np.ndarray, x: np.ndarray) -> float:
    """Compute residual variance from OLS regression with robust fallback."""
    # Clean data: remove any NaN/Inf values
    valid_mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if not np.all(valid_mask):
        y = y[valid_mask]
        x = x[valid_mask, :]

    # Check for insufficient data after cleaning
    if len(y) < x.shape[1] + 5 or x.shape[0] < 2:
        return 1e6  # Return large variance as penalty

    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        return float(np.mean(np.square(resid)))
    except (np.linalg.LinAlgError, ValueError):
        # Fallback 1: stronger ridge-regularized solve
        try:
            xtx = x.T @ x
            reg_xtx = xtx + 1e-4 * np.eye(xtx.shape[0])
            beta = np.linalg.solve(reg_xtx, x.T @ y)
            resid = y - x @ beta
            return float(np.mean(np.square(resid)))
        except (np.linalg.LinAlgError, ValueError):
            # Fallback 2: return variance of y (unexplained variance)
            return float(np.var(y))


def conditional_granger_stat(y_i: np.ndarray, y_j: np.ndarray, y_k: np.ndarray, lag: int) -> float:
    """
    Compute conditional Granger causality statistic for j -> i given k.

    H0: y_j does not Granger-cause y_i conditional on y_k
    """
    target = y_i[lag:]
    own_lag = _lag_matrix(y_i, lag)
    cause_lag = _lag_matrix(y_j, lag)
    cond_lag = _lag_matrix(y_k, lag)
    intercept = np.ones((len(target), 1))

    # Unrestricted model: y_i = f(y_i_lag, y_j_lag, y_k_lag)
    unrestricted = np.column_stack([intercept, own_lag, cause_lag, cond_lag])
    # Restricted model: y_i = f(y_i_lag, y_k_lag)  [exclude y_j]
    restricted = np.column_stack([intercept, own_lag, cond_lag])

    sigma_unrestricted = _residual_variance(target, unrestricted)
    sigma_restricted = _residual_variance(target, restricted)

    if not np.isfinite(sigma_unrestricted) or not np.isfinite(sigma_restricted):
        return 0.0
    if sigma_unrestricted <= 0 or sigma_restricted <= sigma_unrestricted:
        return 0.0

    # LM statistic: T * log(sigma_R / sigma_U)
    statistic = len(target) * np.log(sigma_restricted / sigma_unrestricted)
    if not np.isfinite(statistic) or statistic < 0:
        return 0.0
    return float(statistic)


def loo_conditional_granger_networks(data_window: np.ndarray, lag: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute leave-one-out conditional Granger causality networks.

    For each removed industry k, test j -> i conditional on k, then aggregate.

    Args:
        data_window: T x K array of risk measures (e.g., CSV or JSV)
        lag: VAR lag order
        alpha: significance level for chi-square test

    Returns:
        aggregate_adjacency: K x K adjacency matrix (1 if j -> i significant)
        lgc_removed: K vector of LGC counts per removed industry
    """
    K = data_window.shape[1]
    # adjacency_by_removed[k, i, j] = 1 if j -> i significant when removing k
    adjacency_by_removed = np.zeros((K, K, K), dtype=float)
    critical_value = stats.chi2.ppf(1.0 - alpha, lag)

    for removed_k in range(K):
        for target_i in range(K):
            if target_i == removed_k:
                continue
            for source_j in range(K):
                if source_j in (removed_k, target_i):
                    continue

                statistic = conditional_granger_stat(
                    data_window[:, target_i],
                    data_window[:, source_j],
                    data_window[:, removed_k],
                    lag,
                )
                if statistic > critical_value:
                    adjacency_by_removed[removed_k, target_i, source_j] = 1.0

    # LGC count for each removed industry
    lgc_removed = adjacency_by_removed.sum(axis=(1, 2))

    # Aggregate: edge exists if significant for all but one removal
    aggregate_adjacency = (adjacency_by_removed.sum(axis=0) >= K - 2).astype(float)
    np.fill_diagonal(aggregate_adjacency, 0.0)

    return aggregate_adjacency, lgc_removed


# =============================================================================
# Part 2: Network Utility Functions
# =============================================================================

def row_normalize(W: np.ndarray) -> np.ndarray:
    """Row-normalize adjacency matrix: each row sums to 1, diagonal = 0."""
    W_norm = W.copy()
    np.fill_diagonal(W_norm, 0)
    row_sums = W_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_norm = W_norm / row_sums
    return W_norm


def compute_network_density(W: np.ndarray) -> float:
    """Compute network density (fraction of non-zero off-diagonal entries)."""
    K = W.shape[0]
    max_edges = K * (K - 1)
    actual_edges = np.sum(W > 0)
    return actual_edges / max_edges


def _window_slice(end_idx: int, window: int) -> slice:
    """Create slice for window ending at end_idx."""
    start_idx = end_idx - window + 1
    if start_idx < 0:
        raise ValueError(f"Window ending at {end_idx} is shorter than {window} periods")
    return slice(start_idx, end_idx + 1)


# =============================================================================
# Part 3: Core Estimation Functions (Concentrated MLE)
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


def compute_eigenvector_centrality(W: np.ndarray) -> np.ndarray:
    """Compute eigenvector centrality of a network."""
    eigvals, eigvecs = np.linalg.eig(W.T)
    idx = np.argmax(np.abs(eigvals))
    v = np.abs(eigvecs[:, idx]).astype(float)
    v_sum = v.sum()
    if v_sum > 0:
        v = v / v_sum
    return v


# =============================================================================
# Part 4: Rolling Window Estimation with Integrated Granger
# =============================================================================

def load_data_with_risk_measures(config: NetworkConfig) -> Tuple[np.ndarray, pd.DatetimeIndex, Dict[str, np.ndarray]]:
    """
    Load daily returns and risk measures (CSV, JSV_POS, JSV_NEG) from LM HAR files.

    Returns:
        R: T x K returns matrix
        dates: DatetimeIndex
        layer_data: dict mapping layer_name -> T x K risk measure matrix
    """
    frames = []
    risk_frames = {name: [] for name in config.layer_names}

    for code in config.lm_codes:
        fpath = os.path.join(config.data_dir, f"{code}_lm_har.csv")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"LM HAR file not found: {fpath}")

        df = pd.read_csv(fpath)
        date_col = df.columns[0]
        df['date'] = pd.to_datetime(df[date_col].astype(int).astype(str), format='%Y%m%d')

        # Returns
        frames.append(df[['date', 'r_now']].rename(columns={'r_now': code}))

        # Risk measures for each layer
        for layer_name, col_name in config.layer_columns.items():
            if col_name not in df.columns:
                raise ValueError(f"Column {col_name} not found in {fpath}")
            risk_frames[layer_name].append(df[['date', col_name]].rename(columns={col_name: code}))

    # Merge returns
    merged_returns = frames[0]
    for f in frames[1:]:
        merged_returns = merged_returns.merge(f, on='date', how='inner')
    merged_returns = merged_returns.sort_values('date').drop_duplicates(subset='date').reset_index(drop=True)

    dates = pd.DatetimeIndex(merged_returns['date'])
    R_df = merged_returns[config.lm_codes].replace([np.inf, -np.inf], np.nan)

    # Merge each layer's risk measures first (to check for NaN/Inf in risk data)
    layer_data_raw = {}
    for layer_name in config.layer_names:
        merged_risk = risk_frames[layer_name][0]
        for f in risk_frames[layer_name][1:]:
            merged_risk = merged_risk.merge(f, on='date', how='inner')
        merged_risk = merged_risk.sort_values('date').drop_duplicates(subset='date').reset_index(drop=True)
        layer_data_raw[layer_name] = merged_risk

    # Find common dates across returns and all risk layers
    common_dates = set(dates)
    for layer_name in config.layer_names:
        common_dates = common_dates.intersection(set(layer_data_raw[layer_name]['date']))
    common_dates = sorted(common_dates)

    # Filter all data to common dates
    date_mask = dates.isin(common_dates)
    R_df = R_df[date_mask]
    dates = dates[date_mask]

    # Clean data: drop rows with any NaN/Inf in returns OR risk measures
    valid_mask = ~R_df.isna().any(axis=1).values
    for layer_name in config.layer_names:
        risk_df = layer_data_raw[layer_name]
        risk_df = risk_df[risk_df['date'].isin(dates)][config.lm_codes].replace([np.inf, -np.inf], np.nan)
        valid_mask = valid_mask & ~risk_df.isna().any(axis=1).values

    # Apply valid_mask to all data
    R = R_df[valid_mask].values
    dates = dates[valid_mask]

    # Build final layer_data with cleaned dates
    layer_data = {}
    for layer_name in config.layer_names:
        merged_risk = layer_data_raw[layer_name]
        merged_risk = merged_risk[merged_risk['date'].isin(dates)].sort_values('date').reset_index(drop=True)
        layer_data[layer_name] = merged_risk[config.lm_codes].values

    print(f"Loaded data: {len(dates)} periods x {config.K} industries")
    print(f"Date range: {dates[0].date()} ~ {dates[-1].date()}")
    for name in config.layer_names:
        print(f"  Layer [{name}]: {config.layer_columns[name]}")

    return R, dates, layer_data


def load_ff3_factors(ff3_path: str, market_id: str = "P9709") -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Load CSMAR Fama-French 3 factors (RiskPremium1, SMB1, HML1)."""
    df = pd.read_csv(ff3_path)
    df = df[df['MarkettypeID'] == market_id].copy()
    df['TradingDate'] = pd.to_datetime(df['TradingDate'])
    df = df.sort_values('TradingDate').reset_index(drop=True)
    for col in ['RiskPremium1', 'SMB1', 'HML1']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['RiskPremium1', 'SMB1', 'HML1'])
    dates = pd.DatetimeIndex(df['TradingDate'])
    data = df[['RiskPremium1', 'SMB1', 'HML1']].values
    print(f"  FF3 loaded: {len(dates)} days, {dates[0].date()} ~ {dates[-1].date()}")
    return dates, data


def construct_factors(R: np.ndarray, method: str = "pca",
                      ff3_data: Optional[np.ndarray] = None) -> np.ndarray:
    """Construct common factor F_t from return matrix or external data."""
    if method == "ff3":
        if ff3_data is None:
            raise ValueError("ff3_data must be provided when method='ff3'")
        print(f"  Factor: CSMAR Fama-French 3 factors (MKT, SMB, HML), shape={ff3_data.shape}")
        return ff3_data
    elif method == "pca":
        R_demean = R - R.mean(axis=0)
        _, _, Vt = np.linalg.svd(R_demean, full_matrices=False)
        F = (R_demean @ Vt[0].T).reshape(-1, 1)
        F = F / F.std()
        print(f"  Factor: PCA first principal component")
    else:
        F = R.mean(axis=1, keepdims=True)
        print(f"  Factor: equal-weighted market return (mean={F.mean():.6f})")
    return F


def _estimate_window_job_integrated(
    args: Tuple[int, int],
    R: np.ndarray,
    F: np.ndarray,
    layer_data: Dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    config: NetworkConfig
) -> Tuple[int, Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    """
    Parallel worker for single window: compute Granger networks + MLE.

    Returns:
        count: window index
        result_row: MLE estimation results
        cent_row: centrality measures
        density_row: network densities for each layer
    """
    count, end_idx = args

    # 分离MLE窗口和Granger窗口
    # MLE窗口：用于收益率数据（通常较短，如252天）
    mle_window = config.window
    mle_start_idx = end_idx - mle_window + 1

    # Granger窗口：用于计算因果网络（可以更长，如500天，更稳定）
    granger_window = config.granger_window
    granger_start_idx = end_idx - granger_window + 1

    # 检查数据足够性
    if granger_start_idx < 0:
        raise ValueError(
            f"Window ending at {end_idx} needs {granger_window} periods "
            f"for Granger network, but only {end_idx + 1} available"
        )

    date_t = str(dates[end_idx].date())

    # Slice returns for MLE window（MLE用较短窗口）
    R_win = R[mle_start_idx:end_idx + 1]
    F_win = F[mle_start_idx:end_idx + 1]

    # Compute Granger networks for each layer（Granger用较长窗口）
    W_list = []
    density_row = {}
    for layer_name in config.layer_names:
        # 使用更长的granger_window计算网络，提高稳定性
        risk_window = layer_data[layer_name][granger_start_idx:end_idx + 1]
        W, _ = loo_conditional_granger_networks(
            risk_window,
            config.granger_lag,
            config.granger_alpha
        )
        W_norm = row_normalize(W)
        W_list.append(W_norm)
        density_row[f'density_{layer_name}'] = compute_network_density(W)

    # Run MLE
    res = estimate_single_window(
        R_win, F_win, W_list,
        rho_init_values=config.rho_init_values,
        maxiter=config.maxiter,
        ftol=config.ftol
    )

    # Compute centrality
    K = R.shape[1]
    if res['success'] and not np.any(np.isnan(res['delta_hat'])):
        W_star = sum(dj * W for dj, W in zip(res['delta_hat'], W_list))
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
    for j, name in enumerate(config.layer_names):
        result_row[f'delta_{name}'] = res['delta_hat'][j]
    for i in range(K):
        result_row[f'lambda_{i}'] = res['lambda_hat'][i]

    cent_row = {'date': date_t}
    for i in range(K):
        cent_row[f'composite_industry_{i}'] = composite_cent[i]
        cent_row[f'effective_industry_{i}'] = effective_cent[i]

    return count, result_row, cent_row, density_row


def rolling_estimation_integrated(
    R: np.ndarray,
    F: np.ndarray,
    layer_data: Dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    config: NetworkConfig,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parallel rolling window MLE with integrated Granger network computation.

    Each window:
    1. Slices returns and risk measures for the window period
    2. Computes 3-layer Granger causality networks on-the-fly
    3. Runs one-step concentrated MLE
    """
    d = len(config.layer_names)
    n_periods = len(dates)
    window = config.window
    step = config.step

    # Evaluation points: windows with enough data for BOTH MLE and Granger network
    # First valid window needs max(window, granger_window) periods
    first_valid_idx = max(window, config.granger_window) - 1
    eval_indices = list(range(first_valid_idx, n_periods, step))
    if config.max_windows is not None:
        eval_indices = eval_indices[:config.max_windows]
    n_eval = len(eval_indices)

    if n_eval == 0:
        raise ValueError(f"Not enough periods ({n_periods}) for window {window}")

    n_jobs_eff = config.n_jobs or max((os.cpu_count() or 2) - 1, 1)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Rolling Window MLE with Integrated Granger Networks")
        print(f"{'='*60}")
        print(f"  MLE return window: {window} periods")
        print(f"  Granger network window: {config.granger_window} periods ({config.granger_window/252:.1f} years)")
        print(f"  Step:                  {step} periods")
        print(f"  Layers:                {d} ({', '.join(config.layer_names)})")
        print(f"  Granger lag:            {config.granger_lag}, alpha: {config.granger_alpha}")
        print(f"  Rho starts:             {config.rho_init_values}")
        print(f"  L-BFGS-B maxiter:       {config.maxiter}, ftol: {config.ftol}")
        print(f"  Total periods:          {n_periods}")
        print(f"  Evaluation points:      {n_eval}")
        print(f"  Workers:                {n_jobs_eff}")
        print(f"  Date range:             {dates[eval_indices[0]].date()} ~ {dates[eval_indices[-1]].date()}")

    tasks = [(i, end_idx) for i, end_idx in enumerate(eval_indices)]

    result_slots: List[Optional[Dict[str, Any]]] = [None] * n_eval
    centrality_slots: List[Optional[Dict[str, Any]]] = [None] * n_eval
    density_slots: List[Optional[Dict[str, float]]] = [None] * n_eval

    job_results = Parallel(
        n_jobs=n_jobs_eff,
        prefer="processes",
    )(
        delayed(_estimate_window_job_integrated)(
            task, R, F, layer_data, dates, config
        )
        for task in tasks
    )

    completed = 0
    for count, result_row, cent_row, density_row in job_results:
        result_slots[count] = result_row
        centrality_slots[count] = cent_row
        density_slots[count] = density_row
        completed += 1

        if verbose and (completed % 20 == 0 or completed == 1 or completed == n_eval):
            status = "OK" if result_row['success'] else "FAIL"
            delta_str = " ".join(
                f"d_{name}={result_row[f'delta_{name}']:.3f}" for name in config.layer_names)
            lam_mean = np.mean([result_row[f'lambda_{i}'] for i in range(R.shape[1])])
            densities = " ".join(
                f"{name}={density_row[f'density_{name}']:.2f}" for name in config.layer_names)
            print(f"  [{completed:3d}/{n_eval}] {pd.Timestamp(result_row['date']).date()} | "
                  f"{delta_str} rho_avg={lam_mean:.3f} | "
                  f"density: {densities} | {status}")

    results_df = pd.DataFrame(result_slots)
    centrality_df = pd.DataFrame(centrality_slots)
    density_df = pd.DataFrame(density_slots)

    n_success = results_df['success'].sum()
    if verbose:
        print(f"\n  Convergence rate: {n_success}/{n_eval} ({100*n_success/n_eval:.1f}%)")
        for name in config.layer_names:
            col = f'delta_{name}'
            print(f"  {col} mean: {results_df[col].mean():.3f}")
            print(f"  density_{name} mean: {density_df[f'density_{name}'].mean():.3f}")

    return results_df, centrality_df, density_df


# =============================================================================
# Part 5: Visualization
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


def plot_network_density_evolution(density_df: pd.DataFrame,
                                   layer_names: List[str],
                                   layer_labels: List[str],
                                   save_path: str = None):
    """Plot network density time evolution."""
    fig, ax = plt.subplots(figsize=(14, 5))

    dates = pd.to_datetime(density_df.index)

    for j, (name, label) in enumerate(zip(layer_names, layer_labels)):
        col = f'density_{name}'
        color = _LAYER_COLORS[j % len(_LAYER_COLORS)]
        ax.plot(dates, density_df[col], label=label, color=color, linewidth=1.0)

    ax.set_ylabel('Network Density')
    ax.set_xlabel('Date')
    ax.set_title('Time Evolution of Granger Network Densities')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

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
# Part 6: Main Entry Point
# =============================================================================

def main_integrated(config: Optional[NetworkConfig] = None):
    """
    Main function for integrated Granger-MLE rolling estimation.

    Args:
        config: NetworkConfig instance (defaults to daily_10())
    """
    if config is None:
        config = NetworkConfig.daily_10()

    # =======================================================================
    # 参数调整区域 - 根据研究需求修改以下参数
    # =======================================================================
    #
    # 【核心概念：MLE窗口 vs Granger窗口 可以分离】
    #
    # 在本集成版本中，MLE收益率窗口 和 Granger因果网络计算窗口 是独立的：
    #   - window (MLE窗口): 用于提取收益率数据 R_t，进行因子模型估计
    #   - granger_window (网络窗口): 用于计算因果网络 W_j，基于更长历史
    #
    # 这种分离设计的优势：
    #   1. MLE可以用较短窗口（如252天）捕捉近期动态
    #   2. Granger网络可以用更长窗口（如500天）获得更稳定的网络结构
    #   3. 解决稀疏性问题：更长数据 → 更多跳跃事件 → 更稠密的网络
    #
    # -----------------------------------------------------------------------
    # 参数详细说明：
    # -----------------------------------------------------------------------
    #
    # window (int): MLE收益率估计窗口长度（交易日）
    #   - 含义：每个时间点，使用过去多少天的收益率数据进行MLE估计
    #   - 默认: 252（约1年）
    #   - 建议范围: 126-500（半年到两年）
    #   - 影响：
    #       * 越小：对近期变化越敏感，但估计方差大
    #       * 越大：估计更稳定，但可能错过结构性变化
    #
    # granger_window (int): Granger因果网络计算窗口长度（交易日）
    #   - 含义：计算Granger因果网络时，使用过去多少天的风险指标数据
    #   - 默认: 500（约2年）
    #   - 建议范围: >= window，建议 2*window 或更大
    #   - 关键约束: 必须 >= window，且需满足 granger_window > 2*K*granger_lag
    #   - 影响：
    #       * 越大：网络越稳定，稀疏性问题越轻（推荐）
    #       * 越小：网络时变性更强，但可能过于稀疏
    #   - 注意：这是解决网络稀疏性（全零矩阵）的关键参数！
    #
    # step (int): 滚动估计步长（交易日）
    #   - 含义：每隔多少个交易日进行一次完整的（MLE+Granger）估计
    #   - 默认: 1（每天估计，最密集）
    #   - 建议值:
    #       * 1：日度估计（数据最全，计算最慢）
    #       * 5：周度估计（平衡选择，推荐）
    #       * 22：月度估计（计算最快，丢失高频信息）
    #   - 影响：步长只影响估计频率，不影响每个窗口的计算
    #
    # granger_lag (int): VAR模型滞后阶数
    #   - 含义：Granger因果检验中，VAR模型包含多少阶滞后
    #   - 默认: 5
    #   - 建议: 5（捕捉周度效应）或 10-22（捕捉月度效应）
    #   - 过小：遗漏重要的跨期传导关系
    #   - 过大：损失自由度，尤其当granger_window不够大时
    #
    # granger_alpha (float): Granger因果检验显著性水平
    #   - 含义：双边卡方检验的显著性水平（Type I错误概率）
    #   - 默认: 0.15（15%）
    #   - 建议范围: 0.10-0.20
    #   - 关键理解：
    #       * 越小（如0.01）：网络越稀疏，更多0，MLE不可识别风险高
    #       * 越大（如0.20）：网络越稠密，边更多，但可能引入噪声
    #   - 原设定0.05导致91%零元素（JSV层52%全零行），强烈建议>=0.10
    #   - 这是解决网络稀疏性的第二关键参数！
    #
    # factor_method (str): 共同因子构造方法
    #   - 含义：如何从行业收益率提取共同因子 F_t
    #   - "pca": 第一主成分（默认，吸收最大方差方向）
    #   - "equal_weight": 简单等权平均（所有行业同等权重）
    #   - 选择建议：
    #       * PCA：当行业间相关性高时，能更好捕捉共同波动
    #       * Equal Weight：当存在极端行业时更稳健
    #   - 若δ估计出现角点解（如d_jsv_pos=1），尝试切换此方法
    #
    # rho_init_values (List[float]): ρ参数初始值搜索点
    #   - 含义：MLE优化时，从哪些初始值开始搜索ρ（网络暴露参数）
    #   - 默认: [-0.5, -0.1, 0.1, 0.3, 0.5]
    #   - 调整建议：
    #       * 若估计发散（rho极大如100+或极小如-50）：收窄到[-0.3, 0, 0.3]
    #       * 若收敛但rho集中在边界：增加中间点如[-0.2, 0, 0.2]
    #
    # maxiter (int): L-BFGS-B优化器最大迭代次数
    #   - 默认: 2000
    #   - 若出现收敛失败（success=False），增大到3000-5000
    #
    # n_jobs (int): 并行计算进程数
    #   - None: 自动使用 (CPU核心数 - 1)，推荐
    #   - 具体数字: 如4或8，根据你的CPU核心数调整
    #   - 注意：每个进程会占用大量内存，内存不足时减小此值
    #
    # max_windows (Optional[int]): 最大估计窗口数（用于测试）
    #   - None: 估计所有可用窗口
    #   - 如100: 只跑前100个窗口，快速测试代码是否正确
    #
    # =======================================================================

    # 示例配置：MLE用252天，Granger网络用500天（窗口分离）
    # 这是推荐配置：短窗口MLE捕捉近期动态，长窗口Granger获得稳定网络
    config = NetworkConfig(
        data_dir=config.data_dir,
        results_dir=config.results_dir,
        K=config.K,
        industry_names=config.industry_names,
        lm_codes=config.lm_codes,
        window=500,              # MLE收益率窗口：252天（约1年）
        granger_window=500,      # Granger网络窗口：500天（约2年，更稳定）
        step=1,                  # 滚动步长：每5天估计一次
        granger_lag=5,           # VAR滞后：5阶
        granger_alpha=0.05,      # 显著性水平：15%（解决稀疏性，更稠密网络）
        layer_names=config.layer_names,
        layer_labels=config.layer_labels,
        layer_columns=config.layer_columns,
        factor_method="ff3",     # 因子方法：CSMAR Fama-French 3因子（外生）
        ff3_path=os.path.join(config.data_dir, "..", "FF3",
                              "三因子模型指标(日)193352534(仅供上海对外经贸大学使用)",
                              "STK_MKT_THRFACDAY.csv"),
        rho_init_values=[-0.3, 0.0, 0.3],  # ρ初始值：收窄范围防发散
        maxiter=3000,            # 最大迭代：3000次
        ftol=1e-8,
        max_windows=None,        # 全部窗口（设为100可快速测试）
        n_jobs=None,             # 并行进程：None=自动使用最大CPU
    )

    # =======================================================================
    # 参数调整区域结束
    # =======================================================================

    os.makedirs(config.results_dir, exist_ok=True)

    # Check data files exist
    for code in config.lm_codes:
        fpath = os.path.join(config.data_dir, f"{code}_lm_har.csv")
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Required file not found: {fpath}\n"
                f"Please run lm_har.py first."
            )

    print("=" * 60)
    print("Integrated Rolling Window MLE with On-the-Fly Granger Networks")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading returns and risk measures...")
    R, dates, layer_data = load_data_with_risk_measures(config)

    # 2. Construct factors
    print("\n[2/4] Constructing common factor...")
    if config.factor_method == "ff3":
        if config.ff3_path is None:
            raise ValueError("ff3_path must be set when factor_method='ff3'")
        ff3_dates, ff3_raw = load_ff3_factors(config.ff3_path, config.ff3_market_id)
        common = dates.intersection(ff3_dates)
        print(f"  Date alignment: returns={len(dates)}, ff3={len(ff3_dates)}, common={len(common)}")
        ret_mask = dates.isin(common)
        ff3_mask = ff3_dates.isin(common)
        R = R[ret_mask]
        dates = dates[ret_mask]
        for name in config.layer_names:
            layer_data[name] = layer_data[name][ret_mask]
        ff3_aligned = ff3_raw[ff3_mask]
        F = construct_factors(R, method="ff3", ff3_data=ff3_aligned)
    else:
        F = construct_factors(R, method=config.factor_method)

    # 3. Single-point validation
    print("\n[3/4] Single-point validation...")
    mid = config.window - 1 + (len(dates) - config.window) // 2
    test_slice = _window_slice(mid, config.window)
    R_test = R[test_slice]
    F_test = F[test_slice]

    W_test_list = []
    for name in config.layer_names:
        risk_window = layer_data[name][test_slice]
        W, _ = loo_conditional_granger_networks(risk_window, config.granger_lag, config.granger_alpha)
        W_norm = row_normalize(W)
        print(f"  [{name}] density={compute_network_density(W):.3f}, row_sum_mean={W_norm.sum(axis=1).mean():.4f}")
        W_test_list.append(W_norm)

    test_res = estimate_single_window(
        R_test, F_test, W_test_list,
        rho_init_values=config.rho_init_values,
        maxiter=config.maxiter,
        ftol=config.ftol
    )
    delta_str = ", ".join(f"d_{name}={test_res['delta_hat'][j]:.3f}" for j, name in enumerate(config.layer_names))
    lam_str = f", rho_avg={np.mean(test_res['lambda_hat']):.3f}"
    status = "OK" if test_res['success'] else "FAIL"
    print(f"  Single-point: {delta_str}{lam_str}, status={status}")

    # 4. Rolling estimation
    print("\n[4/4] Starting rolling estimation...")
    results_df, centrality_df, density_df = rolling_estimation_integrated(
        R, F, layer_data, dates, config
    )

    # Save results
    results_path = os.path.join(config.results_dir, "rolling_delta_integrated.csv")
    centrality_path = os.path.join(config.results_dir, "rolling_centrality_integrated.csv")
    density_path = os.path.join(config.results_dir, "rolling_density_integrated.csv")

    results_df.to_csv(results_path, index=False)
    centrality_df.to_csv(centrality_path, index=False)
    density_df.to_csv(density_path, index=False)

    print(f"\nResults saved:")
    print(f"  {results_path}")
    print(f"  {centrality_path}")
    print(f"  {density_path}")

    # Visualize
    print("\nGenerating plots...")
    plot_delta_evolution(
        results_df, config.layer_names, config.layer_labels,
        save_path=os.path.join(config.results_dir, "delta_evolution_integrated.png"))
    plot_network_density_evolution(
        density_df, config.layer_names, config.layer_labels,
        save_path=os.path.join(config.results_dir, "density_evolution_integrated.png"))
    plot_centrality_heatmap(
        centrality_df, config.industry_names,
        save_path=os.path.join(config.results_dir, "centrality_heatmap_integrated.png"))
    plot_top_central_industries(
        centrality_df, config.industry_names, top_n=5,
        save_path=os.path.join(config.results_dir, "top_central_industries_integrated.png"))

    print("\n" + "=" * 60)
    print("Rolling estimation complete!")
    print("=" * 60)

    return results_df, centrality_df, density_df


if __name__ == "__main__":
    main_integrated()
