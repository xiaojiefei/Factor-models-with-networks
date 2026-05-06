# `multilayer_network_mle_Multithreading.py` 代码说明与论文匹配度

> 对应代码文件：`src/model/multilayer_network_mle_Multithreading.py`
> 说明对象：Bonaccolto et al. (2019) 多层网络组合集中 MLE 的当前 Python 并行实现
> 当前默认运行配置：10 个上证行业指数、日度收益率、三层 TVP-DY 网络、rolling one-step concentrated MLE

---

## 0. 这个脚本整体在做什么？

文件：`src/model/multilayer_network_mle_Multithreading.py`

它做的是：

$$
A_t R_t = \alpha + \beta F_t + \eta_t
$$

其中：

$$
A_t = I - \operatorname{diag}(\rho)\left(\sum_{j=1}^d \delta_j W_{j,t}\right)
$$

也就是：

1. 读入 10 个行业的日度收益率 $R_t$；
2. 读入 3 层动态网络：
   - continuous volatility network：`csv`
   - positive jump network：`jsv_pos`
   - negative jump network：`jsv_neg`
3. 对每个 rolling window 做 one-step concentrated MLE；
4. 同时估计：
   - 三个网络层组合权重 $\delta_1,\delta_2,\delta_3$
   - 十个行业异质性网络暴露参数 $\rho_i$
5. 输出：
   - 每个窗口的 $\delta(t)$、$\rho_i(t)$、log-likelihood；
   - 每个窗口的中心性；
   - 三张图。

当前已有输出：

- `results/rolling_delta_multithreading.csv`
- `results/rolling_centrality_multithreading.csv`
- `results/delta_evolution_multithreading.png`
- `results/centrality_heatmap_multithreading.png`
- `results/top_central_industries_multithreading.png`

---

# 1. 文件头与依赖导入

位置：`src/model/multilayer_network_mle_Multithreading.py:1-37`

```python
"""
多层网络因子模型 — 滚动窗口集中 MLE 估计
...
"""
```

这部分说明脚本目标：

$$
A R_t = \alpha + \beta F_t + \eta_t
$$

其中：

$$
A = I - \Lambda \sum_j \delta_j W_j
$$

和原文关系：

- Bonaccolto et al. (2019) 的核心形式是：

$$
R_t = \alpha + \beta f_t + \epsilon_t
$$

并且网络结构通过：

$$
(I - \rho W)
$$

或多网络组合进入结构方程。

你的代码写成：

$$
A_tR_t = \alpha + \beta F_t + \eta_t
$$

本质上就是把网络效应移到左边，做结构残差的集中似然估计。

---

## 1.1 环境变量：限制每个进程内部线程数

位置：`src/model/multilayer_network_mle_Multithreading.py:19-23`

```python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
```

作用：

- joblib 会开启多个 Python 子进程。
- 每个子进程里面又会调用 `numpy/scipy` 的线性代数。
- 如果不限制 BLAS 线程，可能出现：
  - 一个 Python 进程开很多 BLAS 线程；
  - 多个 joblib 进程叠加；
  - CPU 反而过度竞争。

所以这里把每个子进程内部的数值线程设成 1。

匹配度：

- 这不是论文内容，而是工程实现。
- 对估计结果没有理论影响，主要影响运行速度和 CPU 利用率。

---

## 1.2 joblib、numpy、pandas、scipy 等导入

位置：`src/model/multilayer_network_mle_Multithreading.py:24-32`

```python
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')
```

作用：

- `joblib.Parallel`：并行估计 rolling windows。
- `numpy`：矩阵运算。
- `pandas`：读写 CSV、日期对齐。
- `scipy.optimize.minimize`：SLSQP 约束优化。
- `matplotlib`：画图。
- `dataclass`：保存配置。

匹配度：

- `minimize` 对应论文中 “numerical constrained maximum likelihood estimation” 的实现。
- `joblib`、`matplotlib` 是工程部分，不属于论文模型。

---

## 1.3 中文字体设置

位置：`src/model/multilayer_network_mle_Multithreading.py:34-36`

```python
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

作用：

- 图里能显示中文行业名称。
- 负号正常显示。

匹配度：纯画图工程设置。

---

# 2. 配置类 `NetworkConfig`

位置：`src/model/multilayer_network_mle_Multithreading.py:43-113`

---

## 2.1 基础路径

位置：`src/model/multilayer_network_mle_Multithreading.py:43`

```python
_BASE = "D:/桌面数据/工作论文/带耦合多层网络"
```

作用：

- 项目根路径。
- 后面所有输入输出路径都基于它拼接。

---

## 2.2 `NetworkConfig` 字段解释

位置：`src/model/multilayer_network_mle_Multithreading.py:46-66`

```python
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
    rho_init_values: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    maxiter: int = 1000
    ftol: float = 1e-8
    max_windows: Optional[int] = None
    n_jobs: Optional[int] = None
```

这些字段对应：

| 字段                | 含义                             |
| ------------------- | -------------------------------- |
| `data_dir`        | 输入数据所在目录                 |
| `results_dir`     | 输出结果目录                     |
| `K`               | 行业数量                         |
| `window`          | rolling window 长度              |
| `step`            | rolling step                     |
| `factor_method`   | 因子构造方法                     |
| `layer_names`     | 网络层名称                       |
| `layer_labels`    | 图上显示的网络层标签             |
| `network_files`   | 网络矩阵文件                     |
| `date_files`      | 网络日期文件                     |
| `returns_file`    | 收益率文件                       |
| `industry_names`  | 行业名称                         |
| `lm_codes`        | 10 个行业代码                    |
| `dirichlet_alpha` | 对$\delta$ 的 Dirichlet 弱正则 |
| `rho_init_values` | $\rho_i$ 初始值                |
| `maxiter`         | SLSQP 最大迭代次数               |
| `ftol`            | SLSQP 收敛容忍度                 |
| `max_windows`     | 限制估计窗口数量，用于调试       |
| `n_jobs`          | 并行进程数                       |

匹配度：

- `K` 对应论文中的 $N$ 或 $K$，即资产/机构/行业数量。
- `layer_names` 对应多层网络 $W_1,\dots,W_d$。
- `dirichlet_alpha` 不是原文标准 MLE，而是你加的轻微正则，用于减少角点解。
- `rho_init_values`、`maxiter`、`ftol` 是数值优化设置，不是论文理论部分。

---

## 2.3 日度 10 行业配置 `daily_10()`

位置：`src/model/multilayer_network_mle_Multithreading.py:68-93`

```python
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
        ...
    )
```

这是当前默认配置，因为主函数默认调用：

```python
config = NetworkConfig.daily_10()
```

当前输入数据是：

### 收益率输入

来自：

- `data/raw/lm_results/000032_lm_har.csv`
- ...
- `data/raw/lm_results/000041_lm_har.csv`

每个文件当前维度：

```text
4167 行 × 22 列
```

代码使用的是每个文件的 `r_now` 列作为行业收益率。

### 网络输入

当前配置用的是 TVP 文件：

- `data/raw/lm_results/dy_tvp_all_csv.csv`
- `data/raw/lm_results/dy_tvp_all_jsv_pos.csv`
- `data/raw/lm_results/dy_tvp_all_jsv_neg.csv`

当前维度：

```text
dy_tvp_all_csv.csv      : 41650 × 10 = 4165 个 10×10 网络
dy_tvp_all_jsv_pos.csv  : 41660 × 10 = 4166 个 10×10 网络
dy_tvp_all_jsv_neg.csv  : 41660 × 10 = 4166 个 10×10 网络
```

对应日期文件：

```text
dy_tvp_dates_csv.csv      : 4165 × 2
dy_tvp_dates_jsv_pos.csv  : 4166 × 2
dy_tvp_dates_jsv_neg.csv  : 4166 × 2
```

注意：当前三层 TVP 网络日期长度仍然不完全一致，`csv` 层少 1 期。代码通过日期交集对齐，所以能运行，但会损失不共同的日期。

### 输出

结果输出到：

- `results/rolling_delta_multithreading.csv`
- `results/rolling_centrality_multithreading.csv`

当前结果维度：

```text
rolling_delta_multithreading.csv      : 3914 × 16
rolling_centrality_multithreading.csv : 3914 × 11
```

为什么是 3914 行？

大致逻辑是：

$$
4165 \text{ common dates} - 252 \text{ window} + 1 = 3914
$$

因为 `step=1`，所以每天滚动一次。

匹配度：

- `window=252` 表示一年交易日滚动窗口，这是合理的经验设定，不是原文固定要求。
- `step=1` 表示每日估计一个窗口参数。
- `layer_names=["csv","jsv_pos","jsv_neg"]` 对应你自己的三层 DY 网络，不是 Bonaccolto 原文的银行网络、保险网络、common assets 网络，但思想相同：多层 causality networks 的模型组合。

---

## 2.4 周度 31 行业配置 `weekly_31()`

位置：`src/model/multilayer_network_mle_Multithreading.py:95-113`

这部分是备用配置：

```python
K=31
window=52
step=4
layer_names=["ret", "vol"]
network_files=["dy_all_ret.csv", "dy_all_vol.csv"]
```

含义：

- 用 31 个行业周度数据；
- 两层网络：收益网络、波动网络；
- 52 周窗口，4 周滚动一步。

当前默认没有用它。

匹配度：

- 多层网络模型思想匹配。
- 数据频率和网络构造是你项目自己的扩展。

---

# 3. 网络工具函数

位置：`src/model/multilayer_network_mle_Multithreading.py:120-174`

---

## 3.1 `row_normalize(W)`

位置：`src/model/multilayer_network_mle_Multithreading.py:120-127`

```python
def row_normalize(W: np.ndarray) -> np.ndarray:
    """Row-normalize adjacency matrix: each row sums to 1, diagonal = 0."""
    W_norm = W.copy()
    np.fill_diagonal(W_norm, 0)
    row_sums = W_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_norm = W_norm / row_sums
    return W_norm
```

作用：

- 对单个 $K \times K$ 网络做行标准化；
- 对角线设为 0；
- 每一行除以该行总和。

但当前主流程没有使用它。

匹配度：

- 一般网络模型常用 row-normalization。
- 但 Bonaccolto 原文 Eq.(25) 对时变网络强调的是 max normalization，而不是每期强制行和为 1。
- 所以当前不用它是对的。

---

## 3.2 `check_row_normalized(W)`

位置：`src/model/multilayer_network_mle_Multithreading.py:130-138`

作用：

- 检查矩阵是否对角线为 0；
- 每行和是否为 0 或 1。

当前主流程没有使用。

匹配度：辅助函数，不影响论文复现。

---

## 3.3 `compute_network_density(W)`

位置：`src/model/multilayer_network_mle_Multithreading.py:141-146`

作用：

$$
\text{density} = \frac{\#\{W_{ij}>0\}}{K(K-1)}
$$

当前主流程没有使用。

匹配度：

- 网络描述统计有用；
- 不属于核心 MLE。

---

## `max_row_normalize(W_3d)`

位置：`src/model/multilayer_network_mle_Multithreading.py:149-174`

这是重要函数。

输入：

$$
W_{3d}: T_{\text{net}} \times K \times K
$$

例如当前：

```text
csv      : 4165 × 10 × 10
jsv_pos  : 4166 × 10 × 10
jsv_neg  : 4166 × 10 × 10
```

代码逻辑：

```python
for t in range(T_net):
    np.fill_diagonal(W_out[t], 0)
```

每期网络对角线清零。

```python
col_sums = W_out.sum(axis=1)
max_col_sums = col_sums.max(axis=0)
```

这里要注意：

- `W_out.sum(axis=1)` 是按行维度求和，得到每个“列节点”的总流入。
- 变量名写的是 `col_sums`，确实是 column sums。

然后：

```python
W_out[t] = W_out[t] / max_col_sums[np.newaxis, :]
```

每个时间点的网络矩阵，按每个列节点的历史最大列和缩放。

匹配度：

- 这非常接近 Bonaccolto et al. (2019) 对 time-varying network 的 max normalization 思想：

$$
w_{ij,t} = \frac{w^U_{ij,t}}{\max_t \sum_i w^U_{ij,t}}
$$

- 目的不是每期强行归一化，而是保留网络强度随时间变化。

注意点：

- 函数名叫 `max_row_normalize`，但实际是按 **column max sum** 做标准化。
- 从注释看，这是有意匹配 Eq.(25) 的列标准化，不是 bug。
- 但函数名可能误导。

---

# 4. 核心集中 MLE 函数

这是整个脚本最重要部分。

位置：`src/model/multilayer_network_mle_Multithreading.py:181-307`

---

## 4.1 `estimate_beta_sigma(A, R, F)`

位置：`src/model/multilayer_network_mle_Multithreading.py:181-212`

这个函数对应 concentrated likelihood 里的“给定网络参数，OLS 估计因子暴露和残差方差”。

输入：

| 变量              |            当前维度 | 含义                        |
| ----------------- | ------------------: | --------------------------- |
| `A`             | `252 × 10 × 10` | 每期结构矩阵$A_t$         |
| `R`             |       `252 × 10` | rolling window 内行业收益率 |
| `F`             |        `252 × 1` | rolling window 内共同因子   |
| `include_alpha` |            `True` | 是否估计截距                |

代码：

```python
T, K = R.shape
```

当前默认：

```text
T = 252
K = 10
```

---

### 4.1.1 构造左边变量 $A_t R_t$

位置：`src/model/multilayer_network_mle_Multithreading.py:189-196`

```python
if A.ndim == 2:
    R_tilde = (A @ R.T).T
elif A.ndim == 3:
    R_tilde = np.einsum('tij,tj->ti', A, R)
```

如果 $A$ 是固定矩阵：

$$
\tilde R_t = A R_t
$$

如果 $A_t$ 是时变矩阵：

$$
\tilde R_t = A_t R_t
$$

当前走的是 3D 路径：

```text
A: 252 × 10 × 10
R: 252 × 10
R_tilde: 252 × 10
```

匹配度：

- Bonaccolto 原文主要是固定网络或给定窗口内网络；
- 你这里结合 Billio §4，允许窗口内每期 $W_t$ 不同，因此 $A_t$ 时变。
- 这属于合理扩展，和“窗口内 fully time-varying likelihood”一致。

---

### 4.1.2 OLS 估计 $\alpha,\beta$

位置：`src/model/multilayer_network_mle_Multithreading.py:198-207`

```python
F_aug = np.column_stack([np.ones(T), F])
B_hat = np.linalg.solve(F_aug.T @ F_aug, F_aug.T @ R_tilde).T
alpha_hat = B_hat[:, 0]
beta_hat = B_hat[:, 1:]
eta_hat = R_tilde - F_aug @ B_hat.T
```

数学上：

$$
A_tR_t = \alpha + \beta F_t + \eta_t
$$

给定 $A_t$，直接 OLS：

$$
\hat B = (X'X)^{-1}X'\tilde R
$$

其中：

$$
X = [1, F]
$$

当前维度：

```text
F_aug: 252 × 2
B_hat before transpose: 2 × 10
B_hat after transpose: 10 × 2
alpha_hat: 10
beta_hat: 10 × 1
eta_hat: 252 × 10
```

匹配度：

- 这正是 concentrated MLE 的核心。
- 给定网络参数 $(\delta,\rho)$，先用 OLS 集中掉 $\alpha,\beta,\Sigma_\eta$。
- 和你之前理解的“每一组候选 $A$ 都重新 OLS”一致。

---

### 4.1.3 估计 $\Sigma_\eta$

位置：`src/model/multilayer_network_mle_Multithreading.py:209-210`

```python
sigma_sq = np.var(eta_hat, axis=0, ddof=0)
Sigma_eta = np.diag(sigma_sq)
```

含义：

$$
\Sigma_\eta = \operatorname{diag}(\sigma^2_{\eta,1},\dots,\sigma^2_{\eta,K})
$$

这里只估计对角协方差，默认结构残差之间条件独立。

匹配度：

- Bonaccolto 原文里通常需要残差协方差结构。你这里用 diagonal covariance 是一个简化。
- 优点：估计稳定、参数少。
- 代价：如果结构残差之间仍有同期相关，则 likelihood 可能不完全匹配原文最一般设定。

---

## 4.2 `_compose_network_sequence(delta, W_list)`

位置：`src/model/multilayer_network_mle_Multithreading.py:215-222`

输入：

```text
delta: 3
W_list: 3 个元素，每个是 252 × 10 × 10
```

输出：

```text
W_star_seq: 252 × 10 × 10
```

数学：

$$
W_t^* = \sum_{j=1}^d \delta_j W_{j,t}
$$

当前：

$$
W_t^*
=
\delta_{\text{csv}}W_{\text{csv},t}
+
\delta_{\text{jsv+}}W_{\text{jsv+},t}
+
\delta_{\text{jsv-}}W_{\text{jsv-},t}
$$

匹配度：

- 这是 Bonaccolto et al. (2019) “model-based combination of causality networks”的核心思想。
- 多个候选网络不是事后平均，而是在 likelihood 中估计组合权重 $\delta_j$。

---

## 4.3 `_paper_rho_bounds(W)`

位置：`src/model/multilayer_network_mle_Multithreading.py:225-233`

这个函数用于 scalar rho 情况：

$$
A = I - \rho W
$$

为了保证 $A$ 可逆，需要 $\rho$ 落在特征值约束区间内。

当前主路径没有用它，因为当前是异质性 $\rho_i$，即：

$$
\Lambda = \operatorname{diag}(\rho_i)
$$

匹配度：

- scalar $\rho$ 的特征值约束和原文稳定性条件匹配。
- 当前异质性 $\rho_i$ 情况不能直接用这个简单区间，所以代码主路径另用了 spectral radius 检查。

---

## 4.4 `_is_stable_network_effect(B)`

位置：`src/model/multilayer_network_mle_Multithreading.py:236-238`

```python
spectral_radius = np.max(np.abs(np.linalg.eigvals(B)))
return spectral_radius < 1.0 - tol
```

检查：

$$
\rho_{\max}(\Lambda W_t^*) < 1
$$

如果成立，则：

$$
I - \Lambda W_t^*
$$

可逆，并且网络乘数：

$$
(I-\Lambda W_t^*)^{-1}
$$

稳定。

匹配度：

- 这是对 Bonaccolto / spatial autoregressive model 稳定性条件的工程实现。
- 对异质性 $\rho_i$ 而言，比 scalar 特征值区间更通用。

---

## 4.5 `concentrated_loglik(...)`

位置：`src/model/multilayer_network_mle_Multithreading.py:241-307`

这是最核心函数：给定一组候选参数 $(\delta,\rho)$，返回负的 concentrated log-likelihood。

### 输入维度

当前每个窗口：

```text
params: 13
  前 3 个 = delta_csv, delta_jsv_pos, delta_jsv_neg
  后 10 个 = rho_0,...,rho_9

R: 252 × 10
F: 252 × 1
W_list: 3 个 252 × 10 × 10 网络序列
```

---

### 4.5.1 取出 delta

位置：`src/model/multilayer_network_mle_Multithreading.py:255-264`

```python
d = len(W_list)
K = R.shape[1]
T = R.shape[0]
delta = params[:d]
```

当前：

```text
d = 3
K = 10
T = 252
delta = params[:3]
```

然后检查：

```python
if not np.isclose(np.sum(delta), 1.0, atol=1e-6):
    return 1e10
if np.any(delta < -1e-10):
    return 1e10
```

也就是要求：

$$
\sum_j \delta_j = 1,\quad \delta_j \ge 0
$$

匹配度：

- 这和 Bonaccolto 多网络组合权重约束匹配。
- 也是你现在看到角点解的直接来源之一：simplex 约束容易让 MLE 到边界。

---

### 4.5.2 合成网络

位置：`src/model/multilayer_network_mle_Multithreading.py:266`

```python
W_star_seq = _compose_network_sequence(delta, W_list)
```

数学：

$$
W_t^* = \sum_j \delta_j W_{j,t}
$$

当前输出：

```text
W_star_seq: 252 × 10 × 10
```

匹配度：高度匹配原文多网络组合。

---

### 4.5.3 构造 $A_t$

位置：`src/model/multilayer_network_mle_Multithreading.py:268-283`

当前主路径是 `scalar_rho=False`，所以走：

```python
rho = params[d:]
if np.any(rho < 0):
    return 1e10
Lambda = np.diag(rho)
if any(not _is_stable_network_effect(Lambda @ W_star) for W_star in W_star_seq):
    return 1e10
A = np.array([np.eye(K) - Lambda @ W_star for W_star in W_star_seq])
```

数学：

$$
\Lambda = \operatorname{diag}(\rho_1,\dots,\rho_K)
$$

$$
A_t = I - \Lambda W_t^*
$$

当前维度：

```text
rho: 10
Lambda: 10 × 10
A: 252 × 10 × 10
```

匹配度：

- $\Lambda$ 异质性暴露参数和 Bonaccolto 的 heterogeneous exposure 思路匹配。
- $A_t$ 每期变化，是你结合 Billio §4 的扩展。
- 稳定性逐期检查也合理。

注意：

- 
- 如果理论上允许负网络暴露，即某些行业是 absorber / hedge，那么这个约束会排除负 $\rho_i$。
- Bonaccolto 原文是否允许负值要看具体设定；很多 spatial/network model 允许参数在稳定区间内取负。因此这是一个需要你确认的建模选择。

---

### 4.5.4 log determinant 项

位置：`src/model/multilayer_network_mle_Multithreading.py:285-291`

```python
signs, logabsdets = np.linalg.slogdet(A)
if np.any(signs <= 0) or np.any(logabsdets < -20):
    return 1e10
logabsdet_sum = np.sum(logabsdets)
```

数学：

$$
\sum_{t=1}^T \log |A_t|
$$

这是从变量变换 $R_t \mapsto A_tR_t$ 的 Jacobian 来的。

匹配度：

- 这是 concentrated likelihood 的关键项。
- 与 Bonaccolto / spatial likelihood 中的 $\log|I-\rho W|$ 匹配。
- 因为这里 $W_t$ 时变，所以写成 $\sum_t \log|A_t|$，这和 Billio §4 的时变网络 likelihood 更接近。

注意：

```python
if np.any(signs <= 0)
```

这要求 determinant 符号为正。严格说 likelihood 里是 $\log |A_t|$，只要绝对值非零即可。但如果理论希望 $A_t$ 保持正定/正 orientation，这个检查更保守。

---

### 4.5.5 给定 $A_t$，OLS 集中掉 $\alpha,\beta,\Sigma_\eta$

位置：`src/model/multilayer_network_mle_Multithreading.py:293-299`

```python
_, _, _, Sigma_eta = estimate_beta_sigma(A, R, F)
diag_Sigma = np.diag(Sigma_eta)
if np.any(diag_Sigma <= 0):
    return 1e10
logdet_Sigma = np.sum(np.log(diag_Sigma))
```

数学：

$$
\hat\Sigma_\eta(\delta,\rho)
$$

然后：

$$
\log|\hat\Sigma_\eta|
$$

因为当前 $\Sigma_\eta$ 是对角矩阵，所以：

$$
\log|\Sigma_\eta| = \sum_i \log \sigma_i^2
$$

匹配度：

- concentrated MLE 逻辑匹配。
- 对角协方差是假设简化。

---

### 4.5.6 concentrated log-likelihood

位置：`src/model/multilayer_network_mle_Multithreading.py:301`

```python
loglik = -0.5 * T * logdet_Sigma + logabsdet_sum
```

数学：

$$
\ell_c(\delta,\rho)
=
-\frac{T}{2}\log|\hat\Sigma_\eta|
+
\sum_{t=1}^T \log |A_t|
+
C
$$

其中常数项省略。

匹配度：

- 这是对原文集中似然的核心复现。
- 当前每期 $A_t$ 不同，所以 Jacobian 是逐期求和。

---

### 4.5.7 Dirichlet 正则

位置：`src/model/multilayer_network_mle_Multithreading.py:303-305`

```python
if dirichlet_alpha > 0:
    delta_clipped = np.clip(delta, 1e-10, None)
    loglik += (dirichlet_alpha - 1.0) * np.sum(np.log(delta_clipped))
```

数学上相当于加一个 symmetric Dirichlet prior：

$$
(\alpha_D - 1)\sum_j \log \delta_j
$$

当 `dirichlet_alpha > 1` 时，它惩罚角点。

当前 daily config：

```python
dirichlet_alpha=1.1
```

所以只是轻微惩罚角点。

匹配度：

- 这不是 Bonaccolto 原文纯 MLE。
- 这是你为了避免角点解加的 regularized MLE / penalized likelihood。
- 如果要严格复现原文，应该设 `dirichlet_alpha=1.0` 或关闭正则。
- 如果想稳定估计，可以保留，但论文中要说明是 penalized likelihood。

---

### 4.5.8 返回负似然

位置：`src/model/multilayer_network_mle_Multithreading.py:307`

```python
return -loglik
```

因为 `scipy.optimize.minimize` 是最小化，所以返回负 log-likelihood。

---

# 5. 数据读取与对齐

位置：`src/model/multilayer_network_mle_Multithreading.py:314-423`

---

## 5.1 `load_stacked_networks(filepath, K)`

位置：`src/model/multilayer_network_mle_Multithreading.py:314-329`

输入文件格式：

```text
K*T_net 行 × K 列
```

例如：

```text
dy_tvp_all_csv.csv: 41650 × 10
```

代码：

```python
raw = np.loadtxt(filepath, delimiter=',')
T_net = raw.shape[0] // K
W_3d = raw.reshape(T_net, K, K)
```

当前：

```text
41650 × 10 → 4165 × 10 × 10
```

匹配度：

- 这是 R 脚本输出格式到 Python 网络序列的转换。
- 不属于论文理论，但保证了 $W_{j,t}$ 的输入。

---

## 5.2 `load_network_dates(filepath)`

位置：`src/model/multilayer_network_mle_Multithreading.py:332-337`

读取：

```text
t_index,date
```

输出：

```python
pd.DatetimeIndex
```

用于把每个网络矩阵和日期对应起来。

---

## 5.3 `load_returns(filepath)`

位置：`src/model/multilayer_network_mle_Multithreading.py:340-347`

这是周度 31 行业备用路径。

当前 daily config 不使用它。

---

## 5.4 `load_daily_returns_from_lm(data_dir, codes)`

位置：`src/model/multilayer_network_mle_Multithreading.py:350-375`

当前主流程使用这个函数。

输入：

```text
000032_lm_har.csv
...
000041_lm_har.csv
```

每个文件包含：

```text
Unnamed: 0, CSVt_d, CSVt_w, ..., r_now, ...
```

代码：

```python
date_col = df.columns[0]
df['date'] = pd.to_datetime(df[date_col].astype(int).astype(str), format='%Y%m%d')
frames.append(df[['date', 'r_now']].rename(columns={'r_now': code}))
```

含义：

- 第一列是日期，例如 `20090302`；
- `r_now` 是当前收益率；
- 把每个行业的 `r_now` 合并成一张收益率矩阵。

输出：

```text
R: 4167 × 10
dates: 4167
names: ["000032", ..., "000041"]
```

匹配度：

- $R_t$ 对应论文里的资产/机构收益率向量。
- 你这里把对象从银行保险公司换成 10 个上证行业指数，是应用场景变化。

---

## 5.5 `align_data(...)`

位置：`src/model/multilayer_network_mle_Multithreading.py:378-408`

作用：

把收益率日期和三层网络日期取交集：

```python
common = set(ret_dates)
for nd in net_dates_list:
    common = common & set(nd)
common_dates = pd.DatetimeIndex(sorted(common))
```

然后建立索引映射：

```python
idx_maps['ret']
idx_maps['net_csv']
idx_maps['net_jsv_pos']
idx_maps['net_jsv_neg']
```

当前因为三层网络长度不完全一致：

```text
returns: 4167
csv network: 4165
jsv_pos network: 4166
jsv_neg network: 4166
common dates: 约 4165
```

匹配度：

- 原文通常假定数据已经对齐；
- 这里是工程上保证 $R_t$ 和 $W_{j,t}$ 同期匹配。
- 很重要：如果日期错位，MLE 的经济含义会错。

---

## 5.6 `construct_factors(R, method)`

位置：`src/model/multilayer_network_mle_Multithreading.py:411-423`

当前配置：

```python
factor_method="equal_weight"
```

所以走：

```python
F = R.mean(axis=1, keepdims=True)
```

数学：

$$
F_t = \frac{1}{K}\sum_{i=1}^K R_{i,t}
$$

输出：

```text
F: 4167 × 1
```

如果设为 `pca`，则用第一主成分。

匹配度：

- 原文中的 $F_t$ 是 common factor / market factor。
- 你当前用行业平均收益作为共同因子，是一种可解释代理。
- 但这不是原文唯一设定；如果你想让 common factor 更像统计公共因子，可以切到 PCA。

---

# 6. 单窗口估计

位置：`src/model/multilayer_network_mle_Multithreading.py:426-512`

---

## 6.1 `_generate_init_deltas(d)`

位置：`src/model/multilayer_network_mle_Multithreading.py:426-447`

当前 $d=3$，返回 7 个初始点：

```python
[1/3, 1/3, 1/3]
[0.6, 0.2, 0.2]
[0.2, 0.6, 0.2]
[0.2, 0.2, 0.6]
[0.45, 0.45, 0.1]
[0.45, 0.1, 0.45]
[0.1, 0.45, 0.45]
```

作用：

- 避免 SLSQP 被单一初始值卡住；
- 多起点估计，选 log-likelihood 最大的结果。

这也解释了你看到的 `1/3,1/3,1/3`：

- 如果优化器从第一个初始点没有动；
- 或者不同初始点 log-likelihood 很接近，而第一个成功结果被选中；
- 就会出现很多均分解。

匹配度：

- 原文不会规定具体初始值；
- 这是数值实现细节。

---

## 6.2 `estimate_single_window(...)`

位置：`src/model/multilayer_network_mle_Multithreading.py:449-512`

这是每一个 rolling window 的估计函数。

输入：

```text
R_win: 252 × 10
F_win: 252 × 1
W_list: 3 个 252 × 10 × 10
```

输出：

```python
{
  "delta_hat": 长度 3,
  "lambda_hat": 长度 10,
  "loglik": 标量,
  "success": True/False
}
```

注意：这里输出字段叫 `lambda_hat`，但经济含义是 $\rho_i$。这只是旧命名遗留。

---

### 6.2.1 维度检查

位置：`src/model/multilayer_network_mle_Multithreading.py:462-468`

```python
d = len(W_list)
K = R_win.shape[1]
T = R_win.shape[0]
```

当前：

```text
d=3
K=10
T=252
```

然后保证每层网络都有 252 个时间点。

---

### 6.2.2 设置 multi-start

位置：`src/model/multilayer_network_mle_Multithreading.py:470-474`

```python
init_deltas = _generate_init_deltas(d)
rho_starts = rho_init_values or [0.1, 0.3, 0.5]
```

当前：

```text
7 个 delta 初始点 × 3 个 rho 初始点 = 21 次优化 / 每个窗口
```

一共有 3914 个窗口，所以总优化次数约：

$$
3914 \times 21 = 82194
$$

这就是为什么需要多进程。

---

### 6.2.3 约束与边界

位置：`src/model/multilayer_network_mle_Multithreading.py:476-477`

```python
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:d]) - 1.0}]
bounds = [(0, 1)] * d + [(0, 2.0)] * K
```

对应：

$$
\sum_{j=1}^3 \delta_j = 1
$$

$$
0 \le \delta_j \le 1
$$

$$
0 \le \rho_i \le 2
$$

匹配度：

- $\delta$ 的 simplex 约束匹配原文。
- $\rho_i \in [0,2]$ 是工程边界，不是原文精确边界。
- 真正理论约束由 spectral radius 检查保证。

注意：

- 如果允许负 $\rho_i$，这里会限制掉。
- 这会影响“网络吸收器/负反馈”类型解释。

---

### 6.2.4 SLSQP 优化

位置：`src/model/multilayer_network_mle_Multithreading.py:482-497`

```python
for delta_init in init_deltas:
    for rho_init_val in rho_starts:
        x0 = np.concatenate([delta_init, np.full(K, rho_init_val)])
        result = minimize(...)
```

每次优化的参数：

```text
x0: 13 维
```

其中：

```text
前 3 维：delta 初始值
后 10 维：rho_i 初始值，全设为 0.1 或 0.3 或 0.5
```

使用：

```python
method='SLSQP'
```

因为需要：

- 等式约束；
- box bounds；
- 非线性目标函数。

选择标准：

```python
if result.success and loglik > best_loglik:
    best_result = result
```

也就是 21 次优化里选 log-likelihood 最大的成功结果。

匹配度：

- 原文说 constrained MLE，代码用 SLSQP 是合理实现。
- 但原文不会指定优化器。

---

### 6.2.5 失败返回

位置：`src/model/multilayer_network_mle_Multithreading.py:499-505`

如果 21 次优化全部失败，返回 NaN。

---

### 6.2.6 成功返回

位置：`src/model/multilayer_network_mle_Multithreading.py:507-512`

返回：

```python
delta_hat = best_result.x[:d]
lambda_hat = best_result.x[d:]
```

当前输出列对应：

```text
delta_csv
delta_jsv_pos
delta_jsv_neg
lambda_0 ... lambda_9
```

其中 `lambda_i` 实际是 $\rho_i$。

---

# 7. 中心性计算

位置：`src/model/multilayer_network_mle_Multithreading.py:515-523`

```python
eigvals, eigvecs = np.linalg.eig(W.T)
idx = np.argmax(np.abs(eigvals))
v = np.abs(eigvecs[:, idx]).astype(float)
v = v / v.sum()
```

作用：

- 对有效网络 $W_{\text{effective}}$ 计算 eigenvector centrality。
- 输入不是原始网络，而是：

$$
W_{\text{effective},t}
=
\Lambda \left(\sum_j \delta_j W_{j,t}\right)
$$

匹配度：

- Bonaccolto 原文关注网络组合和系统性影响，不一定必须用 eigenvector centrality。
- 这是你的结果分析扩展。
- 经济含义是：在估计出的网络暴露加权后，哪个行业位于传播结构中心。

---

# 8. 单个 rolling window 的 job 函数

位置：`src/model/multilayer_network_mle_Multithreading.py:526-579`

这个函数是 joblib 并行时每个子进程执行的任务。

---

## 8.1 输入

```python
args: (count, ci)
R: 全样本收益率
F: 全样本因子
W_layers_3d: 三层全样本网络
idx_maps: 日期索引映射
window: 252
```

其中：

- `count` 是第几个 rolling 输出；
- `ci` 是 common_dates 中的当前窗口终点位置。

---

## 8.2 切出 rolling window 数据

位置：`src/model/multilayer_network_mle_Multithreading.py:541-548`

```python
ret_indices = idx_maps['ret'][ci - window + 1: ci + 1]
R_win = R[ret_indices]
F_win = F[ret_indices]
```

得到：

```text
R_win: 252 × 10
F_win: 252 × 1
```

然后对每层网络：

```python
net_indices = idx_maps[f'net_{name}'][ci - window + 1: ci + 1]
W_list_win.append(W_3d[net_indices])
```

得到：

```text
W_list_win:
  csv     : 252 × 10 × 10
  jsv_pos : 252 × 10 × 10
  jsv_neg : 252 × 10 × 10
```

匹配度：

- 这是时变窗口内 likelihood 的数据构造。
- 每个 $t$ 使用当期 $W_{j,t}$，不是只用窗口最后一期网络。

---

## 8.3 调用单窗口 MLE

位置：`src/model/multilayer_network_mle_Multithreading.py:550-554`

```python
res = estimate_single_window(...)
```

输出：

```text
delta_hat: 3
lambda_hat/rho_hat: 10
loglik: 1
success: bool
```

---

## 8.4 用窗口最后一期网络算中心性

位置：`src/model/multilayer_network_mle_Multithreading.py:556-563`

```python
W_list_t = [W_win[-1] for W_win in W_list_win]
W_star = sum(dj * Wt for dj, Wt in zip(res['delta_hat'], W_list_t))
Lambda = np.diag(res['lambda_hat'])
W_effective = Lambda @ W_star
cent = compute_eigenvector_centrality(W_effective)
```

这里注意：

- MLE 用的是窗口内 252 期网络；
- 但是中心性只用窗口终点那一期网络。

数学：

$$
W^*_{ci} = \sum_j \hat\delta_j W_{j,ci}
$$

$$
W^{eff}_{ci} = \operatorname{diag}(\hat\rho) W^*_{ci}
$$

然后计算中心性。

匹配度：

- 估计部分和原文更相关；
- 中心性是结果解释层面的扩展。

---

## 8.5 组织结果行

位置：`src/model/multilayer_network_mle_Multithreading.py:565-579`

输出两行字典：

### `result_row`

```text
date
loglik
success
delta_csv
delta_jsv_pos
delta_jsv_neg
lambda_0 ... lambda_9
```

写入：

- `results/rolling_delta_multithreading.csv`

### `cent_row`

```text
date
industry_0 ... industry_9
```

写入：

- `results/rolling_centrality_multithreading.csv`

---

# 9. 并行 rolling estimation

位置：`src/model/multilayer_network_mle_Multithreading.py:582-670`

---

## 9.1 生成窗口终点

位置：`src/model/multilayer_network_mle_Multithreading.py:601-607`

```python
eval_indices = list(range(window - 1, n_common, step))
```

当前：

```text
window = 252
step = 1
n_common ≈ 4165
eval_indices = 251, 252, ..., 4164
n_eval = 3914
```

所以第一个估计窗口是：

```text
common_dates[0:252]
```

窗口终点：

```text
common_dates[251]
```

当前结果第一行日期是：

```text
2010-03-10
```

---

## 9.2 自动设置并行进程数

位置：`src/model/multilayer_network_mle_Multithreading.py:608`

```python
n_jobs_eff = n_jobs or max((os.cpu_count() or 2) - 1, 1)
```

如果你电脑 16 核，则默认用 15 个进程。

匹配度：工程加速，不影响论文模型。

---

## 9.3 生成任务列表

位置：`src/model/multilayer_network_mle_Multithreading.py:627`

```python
tasks = [(count, ci) for count, ci in enumerate(eval_indices)]
```

每个任务就是一个 rolling window。

---

## 9.4 joblib 并行执行

位置：`src/model/multilayer_network_mle_Multithreading.py:632-642`

```python
job_results = Parallel(
    n_jobs=n_jobs_eff,
    prefer="processes",
)(
    delayed(_estimate_window_job)(...)
    for task in tasks
)
```

含义：

- 每个窗口相互独立；
- 用多个 Python 进程同时估计；
- 每个进程执行 `_estimate_window_job()`；
- 最后收集结果。

匹配度：

- 纯工程加速。
- 不改变估计模型。
- 理论上多进程结果应与单进程一致，除非数值库非确定性或浮点细微差异。

---

## 9.5 按原始顺序放回结果

位置：`src/model/multilayer_network_mle_Multithreading.py:644-659`

因为并行任务返回顺序可能不代表日期顺序，所以代码用：

```python
result_slots[count] = result_row
centrality_slots[count] = cent_row
```

保证输出顺序仍按 rolling window 顺序排列。

然后每 20 个窗口打印一次：

```text
[ 20/3914] 2010-04-... | d_csv=... d_jsv_pos=... d_jsv_neg=... rho_avg=... | loglik=... | OK
```

---

## 9.6 汇总为 DataFrame

位置：`src/model/multilayer_network_mle_Multithreading.py:660-670`

```python
results_df = pd.DataFrame(result_slots)
centrality_df = pd.DataFrame(centrality_slots)
```

输出两个 DataFrame。

---

# 10. 画图函数

位置：`src/model/multilayer_network_mle_Multithreading.py:677-794`

---

## 10.1 `plot_delta_evolution`

位置：`src/model/multilayer_network_mle_Multithreading.py:680-728`

输入：

```text
results_df: rolling_delta_multithreading.csv 对应 DataFrame
```

画两部分：

1. 上图：三个 $\delta_j(t)$ 时间序列；
2. 下图：log-likelihood 时间序列。

输出：

- `results/delta_evolution_multithreading.png`

匹配度：

- 用于观察网络层权重随时间变化。
- 这是你项目的结果分析部分，不是原文估计必需。

---

## 10.2 `plot_centrality_heatmap`

位置：`src/model/multilayer_network_mle_Multithreading.py:731-763`

输入：

```text
centrality_df: date + industry_0 ... industry_9
```

输出：

- `results/centrality_heatmap_multithreading.png`

展示每个行业中心性随时间变化。

---

## 10.3 `plot_top_central_industries`

位置：`src/model/multilayer_network_mle_Multithreading.py:766-794`

计算每个行业平均中心性，选前 5 个画时间序列。

输出：

- `results/top_central_industries_multithreading.png`

---

# 11. 主入口 `main_rolling`

位置：`src/model/multilayer_network_mle_Multithreading.py:801-929`

这是完整运行流程。

---

## 11.1 默认配置

位置：`src/model/multilayer_network_mle_Multithreading.py:808-809`

```python
if config is None:
    config = NetworkConfig.daily_10()
```

所以直接运行脚本时，默认走日度 10 行业。

---

## 11.2 检查输入文件是否存在

位置：`src/model/multilayer_network_mle_Multithreading.py:811-822`

检查：

```text
dy_tvp_all_csv.csv
dy_tvp_all_jsv_pos.csv
dy_tvp_all_jsv_neg.csv
dy_tvp_dates_csv.csv
dy_tvp_dates_jsv_pos.csv
dy_tvp_dates_jsv_neg.csv
```

如果缺失，就提示先运行 R 脚本。

注意：

- 当前配置检查的是 `dy_tvp_*` 文件；
- 但项目记忆里下一步说要重跑 `RollingVAR-DY.R` 生成 `dy_all_*` 文件；
- 这里存在一个配置选择问题：你现在脚本默认跑的是 TVP 网络，不是 RollingVAR-DY 生成的 `dy_all_*` 网络。

---

## 11.3 读取三层网络

位置：`src/model/multilayer_network_mle_Multithreading.py:834-841`

对每个网络文件：

```python
W_3d = load_stacked_networks(...)
W_3d = max_row_normalize(W_3d)
nd = load_network_dates(...)
W_layers_3d.append(W_3d)
net_dates_list.append(nd)
```

当前结果：

```text
csv      : 4165 × 10 × 10
jsv_pos  : 4166 × 10 × 10
jsv_neg  : 4166 × 10 × 10
```

每层都会先做 max normalization。

匹配度：

- 加载 $W_{j,t}$：匹配多层网络输入。
- max normalization：匹配 Bonaccolto Eq.(25) 思想。
- TVP-DY 网络本身是你对原文网络构造方法的替换。

---

## 11.4 读取收益率

位置：`src/model/multilayer_network_mle_Multithreading.py:843-848`

因为 daily config 有 `lm_codes`，所以走：

```python
load_daily_returns_from_lm(...)
```

输出：

```text
R_full: 4167 × 10
ret_dates: 4167
ret_names: 10
```

---

## 11.5 日期对齐

位置：`src/model/multilayer_network_mle_Multithreading.py:852-854`

```python
common_dates, idx_maps = align_data(...)
```

将收益率和三层网络按日期交集对齐。

这一步非常关键，否则 $R_t$ 可能对应错 $W_t$。

---

## 11.6 构造共同因子

位置：`src/model/multilayer_network_mle_Multithreading.py:856-858`

```python
F_full = construct_factors(R_full, method=config.factor_method)
```

当前：

```python
factor_method="equal_weight"
```

所以：

$$
F_t = \text{10行业平均收益率}
$$

---

## 11.7 单点验证

位置：`src/model/multilayer_network_mle_Multithreading.py:860-889`

取中间日期附近一个窗口：

```python
mid = len(common_dates) // 2
```

切出：

```text
R_test: 252 × 10
F_test: 252 × 1
W_test_list: 3 个 252 × 10 × 10
```

先跑一次：

```python
test_res = estimate_single_window(...)
```

作用：

- 在全量 rolling 之前检查模型是否能跑通；
- 打印一个估计结果。

匹配度：

- 工程验证，不是论文模型。

---

## 11.8 启动 rolling estimation

位置：`src/model/multilayer_network_mle_Multithreading.py:891-902`

```python
results_df, centrality_df = rolling_estimation(...)
```

这一步产生核心结果。

当前输出：

```text
3914 个窗口
每个窗口估计 13 个核心参数：
  3 个 delta
  10 个 rho/lambda
```

---

## 11.9 保存结果

位置：`src/model/multilayer_network_mle_Multithreading.py:904-911`

保存：

```python
rolling_delta_multithreading.csv
rolling_centrality_multithreading.csv
```

当前 `rolling_delta_multithreading.csv` 列：

```text
date
loglik
success
delta_csv
delta_jsv_pos
delta_jsv_neg
lambda_0
...
lambda_9
```

---

## 11.10 画图

位置：`src/model/multilayer_network_mle_Multithreading.py:913-923`

保存三张图：

```text
delta_evolution_multithreading.png
centrality_heatmap_multithreading.png
top_central_industries_multithreading.png
```

---

## 11.11 脚本入口

位置：`src/model/multilayer_network_mle_Multithreading.py:932-933`

```python
if __name__ == "__main__":
    main_rolling()
```

直接运行：

```bash
python src/model/multilayer_network_mle_Multithreading.py
```

就会执行完整流程。

---

# 12. 当前数据输入输出总览

## 当前默认输入

### 1. 收益率 $R_t$

路径：

- `data/raw/lm_results/000032_lm_har.csv`
- ...
- `data/raw/lm_results/000041_lm_har.csv`

使用列：

```text
r_now
```

维度：

```text
4167 × 10
```

经济含义：

$$
R_t = (R_{1t},\dots,R_{10t})'
$$

即 10 个上证行业指数收益率。

---

### 2. 三层网络 $W_{j,t}$

路径：

- `data/raw/lm_results/dy_tvp_all_csv.csv`
- `data/raw/lm_results/dy_tvp_all_jsv_pos.csv`
- `data/raw/lm_results/dy_tvp_all_jsv_neg.csv`

维度：

```text
csv      : 4165 × 10 × 10
jsv_pos  : 4166 × 10 × 10
jsv_neg  : 4166 × 10 × 10
```

经过日期交集后共同使用。

---

### 3. 共同因子 $F_t$

当前由代码内部构造：

$$
F_t = \frac{1}{10}\sum_i R_{i,t}
$$

维度：

```text
4167 × 1
```

---

## 当前输出

### 1. 网络权重和 rho

路径：

- `results/rolling_delta_multithreading.csv`

维度：

```text
3914 × 16
```

列：

```text
date
loglik
success
delta_csv
delta_jsv_pos
delta_jsv_neg
lambda_0 ... lambda_9
```

其中：

- `delta_*` 是三层网络组合权重；
- `lambda_i` 实际是 $\rho_i$。

---

### 2. 中心性

路径：

- `results/rolling_centrality_multithreading.csv`

维度：

```text
3914 × 11
```

列：

```text
date
industry_0 ... industry_9
```

---

### 3. 图片

路径：

- `results/delta_evolution_multithreading.png`
- `results/centrality_heatmap_multithreading.png`
- `results/top_central_industries_multithreading.png`

---

# 13. 和 Bonaccolto et al. (2019) 的匹配度总结

## 高度匹配的部分

### 1. 多网络组合

代码：

$$
W_t^* = \sum_j \delta_j W_{j,t}
$$

对应：

- Bonaccolto 的 model-based combination of causality networks。

匹配度：高。

---

### 2. simplex 约束

代码：

$$
\delta_j \ge 0,\quad \sum_j \delta_j=1
$$

匹配度：高。

---

### 3. 网络结构矩阵

代码：

$$
A_t = I - \operatorname{diag}(\rho)W_t^*
$$

匹配原文：

$$
I - \Lambda W
$$

匹配度：高。

---

### 4. 集中 MLE

代码逻辑：

给定 $(\delta,\rho)$：

1. 构造 $A_t$；
2. 计算 $A_tR_t$；
3. OLS 得到 $\alpha,\beta,\Sigma_\eta$；
4. 代回：

$$
\ell_c
=
-\frac{T}{2}\log|\Sigma_\eta|
+
\sum_t \log|A_t|
$$

匹配度：高。

---

### 5. max normalization

代码：

$$
w_{ij,t}
=
\frac{w^U_{ij,t}}
{\max_t\sum_i w^U_{ij,t}}
$$

匹配 Bonaccolto Eq.(25) 思想。

匹配度：较高。

---

## 属于合理扩展的部分

### 1. 窗口内 fully time-varying $W_t$

Bonaccolto 原文更多是给定网络组合估计，代码结合了 Billio §4 的逐期时变网络 likelihood：

$$
\sum_t \log|A_t|
$$

匹配度：不是完全原文，但和你当前研究设定一致。

---

### 2. DY / TVP-DY 网络作为 $W_j$

Bonaccolto 原文是银行和保险公司因果网络组合。

你这里是：

- continuous volatility DY network；
- positive jump DY network；
- negative jump DY network。

这是应用层替换。

匹配度：方法思想匹配，数据来源不同。

---

### 3. eigenvector centrality

这是后续网络分析，不是原文 MLE 必要部分。

匹配度：扩展分析。

---

## 和原文存在偏离或需要说明的部分

### 1. `dirichlet_alpha=1.1`

代码加了 Dirichlet penalty：

$$
(\alpha_D-1)\sum_j \log\delta_j
$$

这不是纯 MLE。

如果论文写“严格复现 Bonaccolto MLE”，应设：

```python
dirichlet_alpha=1.0
```

如果保留，就应表述为：

> penalized concentrated likelihood with a weak symmetric Dirichlet regularization on network weights.

---

### 2. $\rho_i$ 被限制为非负

代码：

$$
0 \le \rho_i \le 2
$$

这会排除负网络效应。

如果理论允许负 $\rho_i$，代码需要放宽为比如：

```python
(-2.0, 2.0)
```

但同时稳定性检查仍需保留。

---

### 3. $\Sigma_\eta$ 设为对角矩阵

代码：

$$
\Sigma_\eta = \operatorname{diag}(\sigma_1^2,\dots,\sigma_K^2)
$$

如果原文使用 full covariance，这里是简化。

影响：

- 优点：估计更稳定；
- 缺点：可能低估结构残差相关性。

---

### 4. 当前默认用 `equal_weight` 因子

项目记忆里曾说切到 PCA，但当前文件实际是：

```python
factor_method="equal_weight"
```

位置：`src/model/multilayer_network_mle_Multithreading.py:76`

如果你想使用 PCA，需要改成：

```python
factor_method="pca"
```

---

### 5. 当前默认读的是 `dy_tvp_*`，不是 `dy_all_*`

当前配置：

```python
network_files=[
    "dy_tvp_all_csv.csv",
    "dy_tvp_all_jsv_pos.csv",
    "dy_tvp_all_jsv_neg.csv",
]
```

位置：`src/model/multilayer_network_mle_Multithreading.py:79-84`

但项目记忆里下一步说要重跑 `RollingVAR-DY.R` 生成 `dy_all_*`。

所以你现在要明确：

- 如果要用 TVP-VAR-DY 网络，当前配置没问题；
- 如果要用 RollingVAR-DY 网络，需要改成：
  - `dy_all_csv.csv`
  - `dy_all_jsv_pos.csv`
  - `dy_all_jsv_neg.csv`
  - 对应 `dy_dates_*`

---

# 14. 为什么会出现 1/3 或近似 0/1？

结合代码，原因主要在这里：

1. 初始点包含：

$$
(1/3,1/3,1/3)
$$

位置：`src/model/multilayer_network_mle_Multithreading.py:437`

如果似然面平坦，SLSQP 可能不动。

2. simplex 约束：

$$
\delta_j \ge 0,\quad \sum_j\delta_j=1
$$

位置：`src/model/multilayer_network_mle_Multithreading.py:476-477`

如果某层稍占优势，MLE 容易推到边界。

3. 同时估计 10 个 $\rho_i$，会削弱 $\delta_j$ 识别。

因为模型真正进入的是：

$$
\operatorname{diag}(\rho)\sum_j\delta_j W_j
$$

$\rho_i$ 和 $\delta_j$ 有替代关系。

---

# 15. 当前脚本的判断

这个脚本在“方法主线”上是对的：

$$
\text{多层网络组合}
\rightarrow
A_t = I - \Lambda W_t^*
\rightarrow
A_tR_t = \alpha+\beta F_t+\eta_t
\rightarrow
集中似然
\rightarrow
rolling one-step MLE
$$

但有几个需要你决定的建模口径：

1. **到底用 TVP-DY 还是 RollingVAR-DY 网络？**当前代码默认用 TVP 文件，但项目记忆里下一步是重跑 RollingVAR-DY。
2. **共同因子用 equal-weight 还是 PCA？**当前代码用 equal-weight，不是 PCA。
3. **是否保留 Dirichlet 正则？**保留能减少角点，但不再是纯原文 MLE。
4. **$\rho_i$ 是否允许负值？**当前只允许非负，会影响网络吸收效应解释。
5. **是否需要诊断 delta 的弱识别？**
   目前大量 `1/3` 和近角点，建议加 multi-start loglik 诊断。

---

# 16. 建议的下一步

建议下一步把这个脚本改成“诊断版”，额外输出每个窗口的：

- multi-start log-likelihood；
- 被选中的初始点；
- SLSQP 迭代次数 `nit`；
- 是否停在 `(1/3,1/3,1/3)`；
- 是否接近 simplex 角点；
- 最优和次优 log-likelihood 差距。

这样可以判断当前 $\delta$ 的异常分布究竟是：

1. 真实由数据强烈支持某一层网络；
2. 还是由于似然面太平坦，$\delta$ 弱识别；
3. 或者 SLSQP 数值优化停在初始点。
