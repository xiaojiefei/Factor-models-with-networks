"""
Rolling leave-one-out conditional Granger causality networks.

This script constructs three directed networks from LM-HAR daily risk measures:
CSVt_d, JSVt_zheng_d, and JSVt_fu_d. For each rolling window and each excluded
industry k, it applies Hué et al. (2019)'s conditional Granger likelihood-ratio
test to every remaining ordered pair j -> i.

Output format matches the stacked adjacency matrices used by the MLE scripts:
(K * T_net) rows x K columns, no header.

Usage:
    python rolling_loo_granger.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed


PROJECT_DIR = "D:/桌面数据/工作论文/带耦合多层网络"
LM_DIR = os.path.join(PROJECT_DIR, "data", "raw", "lm_results")


@dataclass(frozen=True)
class GrangerConfig:
    data_dir: str = LM_DIR
    output_dir: str = LM_DIR
    codes: List[str] = field(default_factory=lambda: [f"{c:06d}" for c in range(32, 42)])
    columns: Dict[str, str] = field(default_factory=lambda: {
        "csv": "CSVt_d",
        "jsv_pos": "JSVt_zheng_d",
        "jsv_neg": "JSVt_fu_d",
    })
    window: int = 252
    step: int = 5
    lag: int = 5
    alpha: float = 0.05
    n_jobs: Optional[int] = None


def load_lm_layers_shared_dates(config: GrangerConfig) -> Tuple[pd.DatetimeIndex, Dict[str, np.ndarray]]:
    frames = []
    value_columns = []
    for code in config.codes:
        path = os.path.join(config.data_dir, f"{code}_lm_har.csv")
        df = pd.read_csv(path)
        date_col = df.columns[0]
        required_cols = list(config.columns.values())
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols!r} not found in {path}")

        part = df[[date_col, *required_cols]].copy()
        part[date_col] = pd.to_datetime(part[date_col].astype(int).astype(str), format="%Y%m%d")
        rename_map = {date_col: "date"}
        for label, col_name in config.columns.items():
            out_col = f"{label}_{code}"
            rename_map[col_name] = out_col
            value_columns.append(out_col)
        frames.append(part.rename(columns=rename_map))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="inner")
    merged = merged.sort_values("date").replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=value_columns).reset_index(drop=True)

    dates = pd.DatetimeIndex(merged["date"])
    layer_data = {}
    for label in config.columns:
        layer_cols = [f"{label}_{code}" for code in config.codes]
        layer_data[label] = merged[layer_cols].to_numpy(dtype=float)
    return dates, layer_data


def _lag_matrix(series: np.ndarray, lag: int) -> np.ndarray:
    rows = []
    for t in range(lag, len(series)):
        rows.append(series[t - lag:t][::-1])
    return np.asarray(rows, dtype=float)


def _residual_variance(y: np.ndarray, x: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(np.mean(np.square(resid)))


def conditional_granger_stat(y_i: np.ndarray, y_j: np.ndarray, y_k: np.ndarray, lag: int) -> float:
    target = y_i[lag:]
    own_lag = _lag_matrix(y_i, lag)
    cause_lag = _lag_matrix(y_j, lag)
    cond_lag = _lag_matrix(y_k, lag)
    intercept = np.ones((len(target), 1))

    unrestricted = np.column_stack([intercept, own_lag, cause_lag, cond_lag])
    restricted = np.column_stack([intercept, own_lag, cond_lag])

    sigma_unrestricted = _residual_variance(target, unrestricted)
    sigma_restricted = _residual_variance(target, restricted)
    if not np.isfinite(sigma_unrestricted) or not np.isfinite(sigma_restricted):
        return 0.0
    if sigma_unrestricted <= 0 or sigma_restricted <= sigma_unrestricted:
        return 0.0

    statistic = len(target) * np.log(sigma_restricted / sigma_unrestricted)
    if not np.isfinite(statistic) or statistic < 0:
        return 0.0
    return float(statistic)


def conditional_granger_pvalue(y_i: np.ndarray, y_j: np.ndarray, y_k: np.ndarray, lag: int) -> float:
    statistic = conditional_granger_stat(y_i, y_j, y_k, lag)
    return float(stats.chi2.sf(statistic, lag))


def loo_conditional_granger_networks(data_window: np.ndarray, lag: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute LOO conditional Granger networks for a single window."""
    k_industries = data_window.shape[1]
    adjacency_by_removed = np.zeros((k_industries, k_industries, k_industries), dtype=float)
    critical_value = stats.chi2.ppf(1.0 - alpha, lag)

    for removed_k in range(k_industries):
        for target_i in range(k_industries):
            if target_i == removed_k:
                continue
            for source_j in range(k_industries):
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

    lgc_removed = adjacency_by_removed.sum(axis=(1, 2))
    aggregate_adjacency = (adjacency_by_removed.sum(axis=0) == k_industries - 2).astype(float)
    np.fill_diagonal(aggregate_adjacency, 0.0)
    return aggregate_adjacency, lgc_removed


def _estimate_window_job(args: Tuple[int, int],
                         dates: pd.DatetimeIndex,
                         data: np.ndarray,
                         window: int,
                         lag: int,
                         alpha: float) -> Tuple[int, np.ndarray, np.ndarray, pd.Timestamp]:
    """Parallel worker for single window Granger estimation.

    Returns:
        (count, adjacency_matrix, lgc_vector, date)
    """
    count, end_idx = args
    start_idx = end_idx - window + 1
    data_window = data[start_idx:end_idx + 1]

    adjacency, lgc_removed = loo_conditional_granger_networks(data_window, lag, alpha)
    date = dates[end_idx]

    return count, adjacency, lgc_removed, date


def export_layer_networks(dates: pd.DatetimeIndex,
                          data: np.ndarray,
                          eval_indices: List[int],
                          label: str,
                          config: GrangerConfig,
                          verbose: bool = True) -> None:
    """Parallel rolling window Granger causality network estimation."""
    k_industries = data.shape[1]
    n_eval = len(eval_indices)
    stacked = np.zeros((k_industries * n_eval, k_industries), dtype=float)
    lgc_rows = np.zeros((n_eval, k_industries), dtype=float)
    net_dates = [None] * n_eval

    n_jobs = config.n_jobs or max((os.cpu_count() or 2) - 1, 1)

    if verbose:
        print(f"\nLayer {label}: {n_eval} rolling windows")
        print(f"  Parallel workers: {n_jobs}")

    tasks = [(count, end_idx) for count, end_idx in enumerate(eval_indices)]

    job_results = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
    )(
        delayed(_estimate_window_job)(
            task, dates, data, config.window, config.lag, config.alpha
        )
        for task in tasks
    )

    completed = 0
    for count, adjacency, lgc_removed, date in job_results:
        stacked[count * k_industries:(count + 1) * k_industries, :] = adjacency
        lgc_rows[count, :] = lgc_removed
        net_dates[count] = date
        completed += 1

        if verbose and (completed % 50 == 0 or completed == 1 or completed == n_eval):
            density = adjacency.sum() / (k_industries * (k_industries - 1))
            print(f"  [{completed:4d}/{n_eval}] {date.date()} density={density:.3f}")

    stacked_path = os.path.join(config.output_dir, f"granger_loo_all_{label}.csv")
    dates_path = os.path.join(config.output_dir, f"granger_loo_dates_{label}.csv")
    lgc_path = os.path.join(config.output_dir, f"granger_loo_lgc_removed_{label}.csv")

    np.savetxt(stacked_path, stacked, delimiter=",", fmt="%.0f")
    pd.DataFrame({"t_index": np.arange(1, len(net_dates) + 1), "date": net_dates}).to_csv(dates_path, index=False)
    lgc_df = pd.DataFrame(lgc_rows, columns=config.codes)
    lgc_df.insert(0, "date", net_dates)
    lgc_df.insert(0, "t_index", np.arange(1, len(net_dates) + 1))
    lgc_df.to_csv(lgc_path, index=False)

    if verbose:
        print(f"  Saved: {stacked_path}")
        print(f"  Saved: {dates_path}")
        print(f"  Saved: {lgc_path}")


def main() -> None:
    config = GrangerConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    n_jobs = config.n_jobs or max((os.cpu_count() or 2) - 1, 1)

    print("=" * 70)
    print("Rolling LOO conditional Granger causality networks (Parallel)")
    print("=" * 70)
    print(f"window={config.window}, step={config.step}, lag={config.lag}, alpha={config.alpha}")
    print(f"Parallel workers: {n_jobs}")

    dates, layer_data = load_lm_layers_shared_dates(config)
    eval_indices = list(range(config.window - 1, len(dates), config.step))
    print(f"Shared finite dates: {dates[0].date()} ~ {dates[-1].date()} ({len(dates)} days)")
    print(f"Rolling endpoints: {len(eval_indices)}")

    for label, data in layer_data.items():
        col_name = config.columns[label]
        print(f"\nLoaded {label} ({col_name}): {data.shape[0]} x {data.shape[1]}")
        export_layer_networks(dates, data, eval_indices, label, config)

    print("\n" + "=" * 70)
    print("All layers complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
