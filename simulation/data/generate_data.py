"""
Generate synthetic data for 5 industries: returns, volatility, tail risk (CAViaR).

Three network layers:
  - Layer 1: Volatility spillover network (from realized volatility)
  - Layer 2: Tail risk network (from CAViaR VaR)
  - Layer 3: Return spillover network (from returns)

Known true network structure for verification.

Output per industry: date, r_now, volatility, caviar_var
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

np.random.seed(42)

N = 5
T = 1500
INDUSTRY_CODES = ["IND01", "IND02", "IND03", "IND04", "IND05"]
INDUSTRY_NAMES = ["Energy", "Materials", "Finance", "Technology", "Healthcare"]

# --- True network structures ---

# Layer 1 (Volatility): dense
W_vol_raw = np.array([
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0],
], dtype=float)

# Layer 2 (Tail risk / CAViaR): moderate
W_tail_raw = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
], dtype=float)

# Layer 3 (Returns): sparser
W_ret_raw = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
], dtype=float)


def row_normalize(W):
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


W_vol = row_normalize(W_vol_raw)
W_tail = row_normalize(W_tail_raw)
W_ret = row_normalize(W_ret_raw)

# True parameters
TRUE_DELTA = np.array([0.5, 0.3, 0.2])  # Vol dominant
TRUE_RHO = np.array([0.3, 0.25, 0.35, 0.2, 0.15])

# Combined network
W_star = TRUE_DELTA[0] * W_vol + TRUE_DELTA[1] * W_tail + TRUE_DELTA[2] * W_ret
Lambda = np.diag(TRUE_RHO)
A = np.eye(N) - Lambda @ W_star
A_inv = np.linalg.inv(A)

# Factor model
TRUE_BETA = np.array([1.0, 0.8, 1.2, 0.9, 0.7])
TRUE_SIGMA_ETA = np.array([0.01, 0.012, 0.015, 0.011, 0.009])

# --- Generate time series ---
dates = pd.bdate_range(start="2014-01-02", periods=T)

# Market factor
factor = np.zeros(T)
factor_vol = np.zeros(T)
factor_vol[0] = 0.01
for t in range(1, T):
    factor_vol[t] = 0.001 + 0.85 * factor_vol[t-1] + 0.1 * factor[t-1]**2
    factor[t] = 0.02 * factor[t-1] + np.sqrt(max(factor_vol[t], 1e-8)) * np.random.randn()

# Structural residuals
eta = np.zeros((T, N))
for i in range(N):
    eta[:, i] = TRUE_SIGMA_ETA[i] * np.random.standard_t(df=5, size=T)

# Returns: R = A^{-1} * (beta * F + eta)
R = np.zeros((T, N))
for t in range(T):
    structural_rhs = TRUE_BETA * factor[t] + eta[t]
    R[t] = A_inv @ structural_rhs

# --- Realized Volatility (GARCH-like with spillovers) ---
vol = np.zeros((T, N))
vol[0] = 0.01 * np.ones(N)
for t in range(1, T):
    for i in range(N):
        own = 0.002 + 0.85 * vol[t-1, i] + 0.10 * R[t-1, i]**2
        spillover = 0.03 * np.sum(W_vol_raw[i] * vol[t-1])
        vol[t, i] = max(own + spillover, 1e-6)

# --- CAViaR Tail Risk (Engle & Manganelli 2004 style) ---
# VaR_t = beta0 + beta1 * VaR_{t-1} + beta2 * |R_{t-1}| + spillover
caviar = np.zeros((T, N))
caviar[0] = -0.02 * np.ones(N)  # initial VaR (negative = loss)
for t in range(1, T):
    for i in range(N):
        own = -0.002 + 0.90 * caviar[t-1, i] - 0.05 * np.abs(R[t-1, i])
        # Tail risk spillover from neighbors
        spillover = 0.02 * np.sum(W_tail_raw[i] * caviar[t-1])
        caviar[t, i] = min(own + spillover, -0.001)  # VaR always negative

# --- Save to CSV ---
output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

for i, code in enumerate(INDUSTRY_CODES):
    df = pd.DataFrame({
        "date": dates.strftime("%Y%m%d").astype(int),
        "r_now": R[:, i],
        "volatility": vol[:, i],
        "caviar_var": caviar[:, i],
    })
    df.to_csv(output_dir / f"{code}_lm_har.csv", index=False)
    print(f"Saved {code}_lm_har.csv: {len(df)} rows, "
          f"r_mean={R[:, i].mean():.5f}, vol_mean={vol[:, i].mean():.5f}, "
          f"caviar_mean={caviar[:, i].mean():.4f}")

# Save true parameters
params = {
    "delta": TRUE_DELTA.tolist(),
    "rho": TRUE_RHO.tolist(),
    "beta": TRUE_BETA.tolist(),
    "sigma_eta": TRUE_SIGMA_ETA.tolist(),
    "layer_names": ["volatility", "caviar_var", "returns"],
    "layer_columns": {"volatility": "volatility", "caviar_var": "caviar_var", "returns": "r_now"},
    "W_vol_density": float(W_vol_raw.sum() / (N*(N-1))),
    "W_tail_density": float(W_tail_raw.sum() / (N*(N-1))),
    "W_ret_density": float(W_ret_raw.sum() / (N*(N-1))),
}
with open(output_dir / "true_parameters.json", "w") as f:
    json.dump(params, f, indent=2)

print(f"\nTrue parameters:")
print(f"  delta = {TRUE_DELTA}  (vol=0.5, tail=0.3, ret=0.2)")
print(f"  rho   = {TRUE_RHO}")
print(f"  Vol network density  = {params['W_vol_density']:.2f}")
print(f"  Tail network density = {params['W_tail_density']:.2f}")
print(f"  Ret network density  = {params['W_ret_density']:.2f}")
