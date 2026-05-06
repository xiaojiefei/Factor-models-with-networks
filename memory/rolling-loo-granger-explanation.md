# `src/model/rolling_loo_granger.py` explanation

## Purpose

[src/model/rolling_loo_granger.py](../src/model/rolling_loo_granger.py) constructs rolling leave-one-out conditional Granger causality networks for the three LM-HAR risk measures:

1. `CSVt_d` -> continuous volatility layer, exported as `csv`.
2. `JSVt_zheng_d` -> positive jump volatility layer, exported as `jsv_pos`.
3. `JSVt_fu_d` -> negative jump volatility layer, exported as `jsv_neg`.

The statistical test follows Hué et al. (2019), Eq. (14):

$$
U_{j\to i\mid k}=T\log\left(\frac{\hat\sigma^2_{i,2}}{\hat\sigma^2_{i,1}}\right),
$$

where:

- $\hat\sigma^2_{i,1}$ is the fitted residual variance from the unrestricted model that includes lags of target $i$, source $j$, and conditioning node $k$.
- $\hat\sigma^2_{i,2}$ is the fitted residual variance from the restricted model that includes lags of target $i$ and conditioning node $k$, but excludes source $j$.
- The null hypothesis is no Granger causality from $j$ to $i$ conditional on $k$.
- The rejection rule is $U_{j\to i\mid k}>\chi^2_{1-\alpha}(M)$, where $M$ is the lag order.

The final network aggregation follows the leave-one-out construction described on page 12 of the ECMODE manuscript: for every excluded node $k$, build a matrix $W^{-k}$ with the $k$-th row, $k$-th column, and diagonal set to zero; then keep an edge only if it appears in all admissible leave-one-out matrices:

$$
W_{ij}=I\left(\sum_{k=1}^{n} W_{ij}^{-k}=n-2\right).
$$

This means the exported network is a strict direct-causality network: edge $j\to i$ is retained only when it remains significant after conditioning on every third-party node $k\ne i,j$.

## Inputs

The script reads one CSV per industry from:

```text
data/raw/lm_results/{code}_lm_har.csv
```

By default, the industry codes are:

```text
000032, 000033, ..., 000041
```

For each code file, it uses:

- the first column as date, expected in `YYYYMMDD` integer/string format;
- one risk-measure column selected from `GrangerConfig.columns`:
  - `CSVt_d`,
  - `JSVt_zheng_d`,
  - `JSVt_fu_d`.

The files are inner-joined by date, sorted, converted to numeric arrays, and rows with missing or infinite values are dropped.

## Outputs

For each layer label `csv`, `jsv_pos`, and `jsv_neg`, the script writes three files under `data/raw/lm_results/`.

### 1. Stacked adjacency matrix

```text
granger_loo_all_{label}.csv
```

Shape:

```text
(K * T_net) rows x K columns
```

where:

- `K` is the number of industries;
- `T_net` is the number of rolling network dates.

Each block of `K` rows is one $K\times K$ directed adjacency matrix. Element `[target_i, source_j] = 1` means source industry $j$ Granger-causes target industry $i$ after the leave-one-out filtering rule.

### 2. Network dates

```text
granger_loo_dates_{label}.csv
```

Columns:

- `t_index`: network time index starting from 1;
- `date`: rolling-window evaluation date.

### 3. Leave-one-out LGC counts

```text
granger_loo_lgc_removed_{label}.csv
```

Columns:

- `t_index`,
- `date`,
- one column per industry code.

For each date and excluded industry $k$, this records Hué et al.'s $LGC(-k)$: the number of significant conditional Granger links among the remaining industries when node $k$ is excluded and used as the conditioning node.

## Code walkthrough

### Lines 0-10: module docstring

States that the script builds rolling leave-one-out conditional Granger networks from three LM-HAR risk measures. It also states the output format: stacked adjacency matrices compatible with the existing MLE pipeline.

### Line 12

```python
from __future__ import annotations
```

Postpones evaluation of type annotations. This keeps annotations lightweight and avoids some forward-reference issues.

### Lines 14-16

```python
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
```

Imports standard-library tools:

- `os` for path joining and directory creation;
- `dataclass` and `field` for immutable configuration;
- `Dict`, `List`, and `Tuple` for type hints.

### Lines 18-20

```python
import numpy as np
import pandas as pd
from scipy import stats
```

Imports third-party libraries:

- `numpy` for matrix operations and OLS;
- `pandas` for reading and merging CSV files;
- `scipy.stats` for chi-square critical values and survival probabilities.

### Lines 23-24

```python
PROJECT_DIR = "D:/桌面数据/工作论文/带耦合多层网络"
LM_DIR = os.path.join(PROJECT_DIR, "data", "raw", "lm_results")
```

Defines the project root and the LM-HAR results directory.

### Lines 27-40: `GrangerConfig`

```python
@dataclass(frozen=True)
class GrangerConfig:
```

Defines an immutable configuration object. The key fields are:

- `data_dir`: input directory;
- `output_dir`: output directory;
- `codes`: industry codes, default `000032` to `000041`;
- `columns`: mapping from output layer labels to LM-HAR columns;
- `window`: rolling window length, default 252 trading days;
- `step`: rolling step, default 5 trading days;
- `lag`: Granger lag order $M$, default 1;
- `alpha`: significance level, default 0.05.

### Lines 43-63: `load_lm_panel`

```python
def load_lm_panel(config: GrangerConfig, col_name: str) -> Tuple[pd.DatetimeIndex, np.ndarray]:
```

Loads one risk-measure panel across all industry codes.

- Line 44 initializes an empty list for each industry's data frame.
- Lines 45-47 loop over industry codes and read `{code}_lm_har.csv`.
- Line 48 treats the first column as the date column.
- Lines 49-50 fail early if the requested risk column is missing.
- Lines 51-52 keep only date and the requested risk column, then rename them to `date` and the industry code.
- Line 53 converts `YYYYMMDD` dates to `datetime`.
- Line 54 appends the industry frame.
- Lines 56-58 merge all industry series by inner join on date.
- Line 59 sorts by date, replaces infinities with missing values, and drops incomplete rows.
- Lines 61-62 return the date index and a numeric matrix with columns ordered by `config.codes`.

### Lines 66-70: `_lag_matrix`

```python
def _lag_matrix(series: np.ndarray, lag: int) -> np.ndarray:
```

Builds lagged regressors for one time series.

For each time $t$ from `lag` to the end, it collects:

```text
series[t-lag], ..., series[t-1]
```

and reverses the slice so that the most recent lag appears first. With `lag=1`, this is simply $y_{t-1}$.

### Lines 73-76: `_residual_variance`

```python
def _residual_variance(y: np.ndarray, x: np.ndarray) -> float:
```

Runs OLS by least squares:

```python
beta, *_ = np.linalg.lstsq(x, y, rcond=None)
```

Then computes fitted residuals and returns their sample mean squared residual:

$$
\hat\sigma^2 = \frac{1}{T}\sum_t \hat u_t^2.
$$

This matches Hué et al.'s Eq. (14) likelihood-ratio form, not the finite-sample F-test variance formula.

### Lines 79-99: `conditional_granger_stat`

```python
def conditional_granger_stat(y_i, y_j, y_k, lag):
```

Computes Hué et al.'s conditional Granger statistic $U_{j\to i\mid k}$.

- Line 80 defines the target vector $y_{i,t}$ after discarding initial lag rows.
- Lines 81-83 construct lag matrices for target $i$, source $j$, and conditioning node $k$.
- Line 84 adds an intercept.
- Line 86 defines the unrestricted model corresponding to Hué Eq. (15): target lags + source lags + conditioning lags.
- Line 87 defines the restricted model corresponding to Hué Eq. (16): target lags + conditioning lags only.
- Lines 89-90 compute the unrestricted and restricted residual variances.
- Lines 91-94 return zero if the statistic cannot be computed or if the restricted variance is not larger than the unrestricted variance.
- Line 96 computes:

$$
U_{j\to i\mid k}=T\log\left(\frac{\hat\sigma^2_R}{\hat\sigma^2_U}\right).
$$

- Lines 97-99 guard against invalid numerical values and return the statistic.

### Lines 102-104: `conditional_granger_pvalue`

Computes the p-value from the chi-square survival function:

```python
stats.chi2.sf(statistic, lag)
```

The degrees of freedom are `lag`, because the restricted model removes `lag` coefficients on the source variable.

### Lines 107-132: `loo_conditional_granger_networks`

```python
def loo_conditional_granger_networks(data_window, lag, alpha):
```

Builds leave-one-out conditional Granger networks for one rolling window.

- Line 108 gets the number of industries `K`.
- Line 109 creates `adjacency_by_removed`, a three-dimensional array indexed as `[removed_k, target_i, source_j]`.
- Line 110 computes the chi-square critical value $\chi^2_{1-\alpha}(M)$.
- Lines 112-128 implement Hué Eq. (24): for each removed node $k$, test every ordered pair $j\to i$ among the remaining nodes.
- Lines 114-118 ensure that the removed node and self-loops are excluded.
- Lines 120-125 compute $U_{j\to i\mid k}$.
- Lines 126-127 set $W_{ij}^{-k}=1$ if the statistic exceeds the chi-square critical value.
- Line 129 computes $LGC(-k)$ by summing all significant links for each excluded node.
- Line 130 applies the ECMODE manuscript page-12 aggregation rule:

$$
W_{ij}=I\left(\sum_k W_{ij}^{-k}=K-2\right).
$$

Only $K-2$ conditioning nodes are admissible for a fixed pair $(i,j)$, because $k$ cannot equal $i$ or $j$.

- Line 131 removes diagonal self-loops.
- Line 132 returns the final aggregated adjacency matrix and the vector of $LGC(-k)$ counts.

### Lines 135-168: `export_layer_networks`

```python
def export_layer_networks(dates, data, label, config):
```

Runs rolling-window network construction for one risk layer and writes outputs.

- Line 136 defines rolling-window end indices starting at `window - 1` and moving by `step`.
- Line 137 gets the number of industries.
- Line 138 allocates the stacked adjacency matrix.
- Line 139 allocates one row of $LGC(-k)$ counts per rolling date.
- Line 140 stores network dates.
- Lines 142-153 loop over windows, build a network, store the network block and LGC counts, and print progress.
- Lines 155-157 define output file paths.
- Line 159 writes the stacked adjacency matrix without headers.
- Line 160 writes the date mapping.
- Lines 161-164 write the $LGC(-k)$ table.
- Lines 166-168 print saved paths.

### Lines 171-183: `main`

```python
def main() -> None:
```

Creates the default config, ensures the output directory exists, prints configuration information, and runs the export workflow for all three risk layers.

### Lines 186-187

```python
if __name__ == "__main__":
    main()
```

Runs the script only when called directly, for example:

```bash
python src/model/rolling_loo_granger.py
```

## Important interpretation notes

1. The implemented conditional test is Hué et al.'s linear conditional Granger LR test from Eq. (14)-(16), using a chi-square threshold.
2. It does not implement the Diks-Wolski nonlinear data-sharpening DP test. The ECMODE manuscript mentions that nonlinear extension after presenting the LOO construction, but the current script follows Hué's Eq. (14) conditional Granger statistic.
3. The final exported adjacency matrix is binary and strict: an edge must survive all admissible leave-one-out conditioning nodes.
4. The separate `granger_loo_lgc_removed_{label}.csv` files preserve Hué et al.'s $LGC(-k)$ information for each excluded industry.
