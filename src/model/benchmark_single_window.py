"""
Benchmark one rolling-window MLE estimation.

Usage:
  D:/Python/python.exe src/model/benchmark_single_window.py
  D:/Python/python.exe src/model/benchmark_single_window.py --window-index 100 --repeat 3
"""

import argparse
import os
import time
from typing import List

import numpy as np

from multilayer_network_mle_Multithreading import (
    NetworkConfig,
    align_data,
    construct_factors,
    estimate_single_window,
    load_daily_returns_from_lm,
    load_network_dates,
    load_returns,
    load_stacked_networks,
    max_row_normalize,
)


def _parse_rho_starts(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _check_required_files(config: NetworkConfig) -> None:
    required = config.network_files + config.date_files
    if config.returns_file:
        required.append(config.returns_file)

    missing = [
        os.path.join(config.data_dir, fname)
        for fname in required
        if not os.path.exists(os.path.join(config.data_dir, fname))
    ]
    if missing:
        missing_text = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing required files:\n{missing_text}")


def _load_inputs(config: NetworkConfig):
    _check_required_files(config)

    W_layers_3d = []
    net_dates_list = []
    for network_file, date_file in zip(config.network_files, config.date_files):
        W_3d = load_stacked_networks(os.path.join(config.data_dir, network_file), K=config.K)
        W_layers_3d.append(max_row_normalize(W_3d))
        net_dates_list.append(load_network_dates(os.path.join(config.data_dir, date_file)))

    if config.lm_codes:
        R_full, ret_dates, _ = load_daily_returns_from_lm(config.data_dir, config.lm_codes)
    else:
        R_full, ret_dates, _ = load_returns(os.path.join(config.data_dir, config.returns_file))

    common_dates, idx_maps = align_data(ret_dates, net_dates_list, config.layer_names)
    F_full = construct_factors(R_full, method=config.factor_method)
    return R_full, F_full, W_layers_3d, common_dates, idx_maps


def _slice_window(config: NetworkConfig,
                  R_full: np.ndarray,
                  F_full: np.ndarray,
                  W_layers_3d: List[np.ndarray],
                  common_dates,
                  idx_maps,
                  window_index: int):
    eval_indices = list(range(config.window - 1, len(common_dates), config.step))
    if not 0 <= window_index < len(eval_indices):
        raise ValueError(
            f"window_index must be in [0, {len(eval_indices) - 1}], got {window_index}"
        )

    ci = eval_indices[window_index]
    ret_indices = idx_maps["ret"][ci - config.window + 1:ci + 1]
    R_win = R_full[ret_indices]
    F_win = F_full[ret_indices]

    W_list_win = []
    for name, W_3d in zip(config.layer_names, W_layers_3d):
        net_indices = idx_maps[f"net_{name}"][ci - config.window + 1:ci + 1]
        W_list_win.append(W_3d[net_indices])

    return common_dates[ci], len(eval_indices), R_win, F_win, W_list_win


def benchmark_single_window(args: argparse.Namespace) -> None:
    config = NetworkConfig.weekly_31() if args.weekly else NetworkConfig.daily_10()
    if args.dirichlet_alpha is not None:
        config.dirichlet_alpha = args.dirichlet_alpha

    print("=" * 72)
    print("Single Rolling-Window MLE Benchmark")
    print("=" * 72)
    print(f"Dataset: {'weekly_31' if args.weekly else 'daily_10'}")
    print(f"Window size: {config.window}")
    print(f"Step: {config.step}")
    print(f"Layers: {', '.join(config.layer_names)}")
    print(f"Dirichlet alpha: {config.dirichlet_alpha}")
    print(f"Rho starts: {args.rho_starts}")
    print(f"SLSQP maxiter: {args.maxiter}, ftol: {args.ftol}")
    print(f"Multi-start: {not args.no_multistart}")

    load_start = time.perf_counter()
    R_full, F_full, W_layers_3d, common_dates, idx_maps = _load_inputs(config)
    load_seconds = time.perf_counter() - load_start

    date_t, n_eval, R_win, F_win, W_list_win = _slice_window(
        config, R_full, F_full, W_layers_3d, common_dates, idx_maps, args.window_index
    )

    print("\nWindow selected:")
    print(f"  window_index: {args.window_index}/{n_eval - 1}")
    print(f"  end date: {date_t.date()}")
    print(f"  R_win shape: {R_win.shape}")
    print(f"  F_win shape: {F_win.shape}")
    print(f"  W layer shapes: {[W.shape for W in W_list_win]}")
    print(f"  Data loading/preparation time: {load_seconds:.2f} seconds")

    elapsed = []
    last_result = None
    for run_id in range(1, args.repeat + 1):
        start = time.perf_counter()
        last_result = estimate_single_window(
            R_win,
            F_win,
            W_list_win,
            multi_start=not args.no_multistart,
            dirichlet_alpha=config.dirichlet_alpha,
            rho_init_values=args.rho_starts,
            maxiter=args.maxiter,
            ftol=args.ftol,
        )
        seconds = time.perf_counter() - start
        elapsed.append(seconds)
        status = "OK" if last_result["success"] else "FAIL"
        print(f"\nRun {run_id}/{args.repeat}: {seconds:.2f} seconds | {status}")

    delta = last_result["delta_hat"]
    rho = last_result["lambda_hat"]
    print("\nEstimate summary:")
    for name, value in zip(config.layer_names, delta):
        print(f"  delta_{name}: {value:.6f}")
    print(f"  rho_avg: {np.nanmean(rho):.6f}")
    print(f"  rho_min: {np.nanmin(rho):.6f}")
    print(f"  rho_max: {np.nanmax(rho):.6f}")
    print(f"  loglik: {last_result['loglik']:.6f}")

    print("\nTiming summary:")
    print(f"  mean: {np.mean(elapsed):.2f} seconds")
    print(f"  min: {np.min(elapsed):.2f} seconds")
    print(f"  max: {np.max(elapsed):.2f} seconds")
    print(f"  estimated full rolling time without parallelism: {np.mean(elapsed) * n_eval / 3600:.2f} hours")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one rolling-window MLE estimation.")
    parser.add_argument("--weekly", action="store_true", help="Use weekly_31 config instead of daily_10.")
    parser.add_argument("--window-index", type=int, default=0, help="Evaluation window ordinal to benchmark.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeated runs for the same window.")
    parser.add_argument("--no-multistart", action="store_true", help="Use only one initial value for a faster diagnostic.")
    parser.add_argument("--rho-starts", type=_parse_rho_starts, default=[-0.5, -0.1, 0.1, 0.3, 0.5], help="Comma-separated rho initial values, e.g. -0.1,0.1,0.3.")
    parser.add_argument("--maxiter", type=int, default=1000, help="SLSQP maxiter for each start.")
    parser.add_argument("--ftol", type=float, default=1e-8, help="SLSQP ftol for each start.")
    parser.add_argument("--dirichlet-alpha", type=float, default=None, help="Override config.dirichlet_alpha.")
    return parser.parse_args()


if __name__ == "__main__":
    benchmark_single_window(parse_args())
