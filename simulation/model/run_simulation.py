"""
Simulation test: sequential MLE with real-time output.

Runs the integrated MLE on synthetic 5-industry data (returns, volatility, CAViaR).
Prints delta and rho estimates for each window as they complete.

True parameters:
  delta = [0.5, 0.3, 0.2]  (vol, tail, ret)
  rho   = [0.3, 0.25, 0.35, 0.2, 0.15]
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "model"))

from multilayer_network_mle_integrated import (
    NetworkConfig,
    loo_conditional_granger_networks,
    row_normalize,
    compute_network_density,
    estimate_single_window,
    construct_factors,
)


def load_simulation_data(config):
    """Load simulation CSV files."""
    frames = []
    risk_frames = {name: [] for name in config.layer_names}

    for code in config.lm_codes:
        fpath = os.path.join(config.data_dir, f"{code}_lm_har.csv")
        df = pd.read_csv(fpath)
        df['date'] = pd.to_datetime(df['date'].astype(int).astype(str), format='%Y%m%d')
        frames.append(df[['date', 'r_now']].rename(columns={'r_now': code}))
        for layer_name, col_name in config.layer_columns.items():
            risk_frames[layer_name].append(df[['date', col_name]].rename(columns={col_name: code}))

    merged_returns = frames[0]
    for f in frames[1:]:
        merged_returns = merged_returns.merge(f, on='date', how='inner')
    merged_returns = merged_returns.sort_values('date').reset_index(drop=True)

    dates = pd.DatetimeIndex(merged_returns['date'])
    R = merged_returns[config.lm_codes].values

    layer_data = {}
    for layer_name in config.layer_names:
        merged_risk = risk_frames[layer_name][0]
        for f in risk_frames[layer_name][1:]:
            merged_risk = merged_risk.merge(f, on='date', how='inner')
        merged_risk = merged_risk.sort_values('date').reset_index(drop=True)
        layer_data[layer_name] = merged_risk[config.lm_codes].values

    return R, dates, layer_data


def main():
    sim_data_dir = os.path.join(PROJECT_ROOT, "simulation", "data")
    sim_results_dir = os.path.join(PROJECT_ROOT, "simulation", "results")
    os.makedirs(sim_results_dir, exist_ok=True)

    with open(os.path.join(sim_data_dir, "true_parameters.json")) as f:
        true_params = json.load(f)

    config = NetworkConfig(
        data_dir=sim_data_dir,
        results_dir=sim_results_dir,
        K=5,
        industry_names=["Energy", "Materials", "Finance", "Technology", "Healthcare"],
        lm_codes=["IND01", "IND02", "IND03", "IND04", "IND05"],
        window=252,
        granger_window=252,
        step=50,              # larger step for faster test
        granger_lag=5,
        granger_alpha=0.10,
        layer_names=["volatility", "caviar_var", "returns"],
        layer_labels=["Volatility", "Tail Risk (CAViaR)", "Returns"],
        layer_columns={
            "volatility": "volatility",
            "caviar_var": "caviar_var",
            "returns": "r_now",
        },
        factor_method="pca",
        rho_init_values=[-0.3, 0.0, 0.3, 0.5],
        maxiter=3000,
        ftol=1e-8,
        max_windows=None,
        n_jobs=1,
    )

    print("=" * 70)
    print("SIMULATION TEST: Multilayer Network MLE (Sequential, Real-time)")
    print("=" * 70)
    print(f"\nTrue parameters:")
    print(f"  delta = {true_params['delta']}  (vol=0.5, tail=0.3, ret=0.2)")
    print(f"  rho   = {true_params['rho']}")

    # Load data
    print("\n[1/3] Loading data...")
    R, dates, layer_data = load_simulation_data(config)
    print(f"  {len(dates)} periods x {config.K} industries")

    # Construct factor
    print("\n[2/3] Constructing PCA factor...")
    F = construct_factors(R, method=config.factor_method)

    # Rolling estimation - SEQUENTIAL with real-time output
    print("\n[3/3] Rolling MLE estimation (real-time output)")
    print("=" * 70)

    window = config.window
    granger_window = config.granger_window
    step = config.step
    first_valid = max(window, granger_window) - 1
    eval_indices = list(range(first_valid, len(dates), step))
    n_eval = len(eval_indices)

    print(f"  Window={window}, Granger_window={granger_window}, Step={step}")
    print(f"  Evaluation points: {n_eval}")
    print(f"  Date range: {dates[eval_indices[0]].date()} ~ {dates[eval_indices[-1]].date()}")
    print()

    header = (f"{'#':>3} | {'Date':>10} | "
              f"{'d_vol':>6} {'d_tail':>6} {'d_ret':>6} | "
              f"{'rho_0':>6} {'rho_1':>6} {'rho_2':>6} {'rho_3':>6} {'rho_4':>6} | "
              f"{'loglik':>10} | {'density':>15} | {'status':>4}")
    print(header)
    print("-" * len(header))

    results = []
    t_start = time.time()

    for count, end_idx in enumerate(eval_indices):
        mle_start = end_idx - window + 1
        granger_start = end_idx - granger_window + 1

        date_t = dates[end_idx].date()
        R_win = R[mle_start:end_idx + 1]
        F_win = F[mle_start:end_idx + 1]

        # Compute Granger networks
        W_list = []
        densities = []
        for layer_name in config.layer_names:
            risk_win = layer_data[layer_name][granger_start:end_idx + 1]
            W, _ = loo_conditional_granger_networks(risk_win, config.granger_lag, config.granger_alpha)
            W_norm = row_normalize(W)
            W_list.append(W_norm)
            densities.append(compute_network_density(W))

        # MLE
        res = estimate_single_window(
            R_win, F_win, W_list,
            rho_init_values=config.rho_init_values,
            maxiter=config.maxiter,
            ftol=config.ftol
        )

        d = res['delta_hat']
        rho = res['lambda_hat']
        status = "OK" if res['success'] else "FAIL"
        density_str = f"{densities[0]:.2f}/{densities[1]:.2f}/{densities[2]:.2f}"

        print(f"{count+1:3d} | {date_t} | "
              f"{d[0]:6.3f} {d[1]:6.3f} {d[2]:6.3f} | "
              f"{rho[0]:6.3f} {rho[1]:6.3f} {rho[2]:6.3f} {rho[3]:6.3f} {rho[4]:6.3f} | "
              f"{res['loglik']:10.1f} | {density_str:>15} | {status:>4}",
              flush=True)

        results.append({
            'date': str(date_t),
            'delta_vol': d[0], 'delta_tail': d[1], 'delta_ret': d[2],
            'rho_0': rho[0], 'rho_1': rho[1], 'rho_2': rho[2],
            'rho_3': rho[3], 'rho_4': rho[4],
            'loglik': res['loglik'], 'success': res['success'],
            'density_vol': densities[0], 'density_tail': densities[1],
            'density_ret': densities[2],
        })

    elapsed = time.time() - t_start

    # Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(sim_results_dir, "sim_results.csv"), index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Time elapsed: {elapsed:.1f}s ({elapsed/n_eval:.1f}s per window)")
    print(f"  Convergence: {results_df['success'].sum()}/{n_eval}")

    print(f"\n  Delta estimates vs true:")
    print(f"    {'Layer':<12} {'True':>6} {'Mean':>6} {'Std':>6} {'Bias':>6}")
    for j, name in enumerate(['vol', 'tail', 'ret']):
        col = f'delta_{name}'
        est = results_df[col]
        true_val = true_params['delta'][j]
        print(f"    {name:<12} {true_val:6.3f} {est.mean():6.3f} {est.std():6.3f} {est.mean()-true_val:+6.3f}")

    print(f"\n  Rho estimates vs true:")
    print(f"    {'Industry':<12} {'True':>6} {'Mean':>6} {'Std':>6} {'Bias':>6}")
    for i in range(config.K):
        col = f'rho_{i}'
        est = results_df[col]
        true_val = true_params['rho'][i]
        print(f"    {config.industry_names[i]:<12} {true_val:6.3f} {est.mean():6.3f} {est.std():6.3f} {est.mean()-true_val:+6.3f}")

    print(f"\n  Network density (mean):")
    for name in ['vol', 'tail', 'ret']:
        print(f"    {name}: {results_df[f'density_{name}'].mean():.3f}")

    delta_sum = results_df[['delta_vol', 'delta_tail', 'delta_ret']].sum(axis=1)
    print(f"\n  Delta sum: min={delta_sum.min():.6f}, max={delta_sum.max():.6f}")
    print(f"\n  Results saved to: simulation/results/sim_results.csv")


if __name__ == "__main__":
    main()
