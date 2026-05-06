# Plan: 全时间点切片网络 + 滚动窗口 MLE

## Context

用户需要在**每个时间点**都有 DY 切片网络，以便在每个 t 运行多层空间依赖线性因子模型，追踪：
- 网络权重 δ_j(t) 的时间演化
- 复合网络中心性的时间演化

当前 R 代码只保存 5 个切片，需要改为保存全部。滚动 MLE 代码整合到现有 `static_mle_demo.py` 中。

---

## 修改 1: R 代码 (`src/model/TVP-VAR-DY.R`)

### 改动要点
- **删除** `extract_and_save_slice()` 函数（原第 58-80 行）
- **重写** `run_dy()` 函数：导出全部时变邻接矩阵
- **存储格式**：堆叠 CSV（K*T_net 行 × K 列），每 K=31 行为一个时间点的邻接矩阵
- **日期映射**：单独保存 `dy_dates_ret.csv` / `dy_dates_vol.csv`

### 输出文件

| 文件 | 形状 | 说明 |
|------|------|------|
| `dy_all_ret.csv` | ~(31×6167) × 31 | 收益率网络堆叠矩阵 |
| `dy_all_vol.csv` | ~(31×6146) × 31 | 波动率网络堆叠矩阵 |
| `dy_dates_ret.csv` | ~6167 × 2 | t_index + date |
| `dy_dates_vol.csv` | ~6146 × 2 | t_index + date |

### 新 `run_dy()` 逻辑
```
1. load_data() → xts数据
2. ConnectednessApproach() → dca 对象
3. 遍历 t=1..T_net:
   - adj = dca$CT[,,t], diag=0, row_normalize
   - 写入堆叠矩阵的第 [(t-1)*K+1 : t*K] 行
4. write.table → dy_all_*.csv
5. dates mapping → dy_dates_*.csv
```

---

## 修改 2: Python 代码 (`src/model/static_mle_demo.py`)

### 新增内容（追加到文件末尾，保留原有全部代码）

#### 新增函数列表

| 函数 | 作用 |
|------|------|
| `load_stacked_networks(filepath, K=31)` | 加载堆叠CSV → 3D数组 (T_net, K, K) |
| `load_network_dates(filepath)` | 加载日期映射 → DatetimeIndex |
| `load_returns(filepath)` | 加载行业收益率 → (ndarray, DatetimeIndex) |
| `align_data(ret_dates, net_dates_ret, net_dates_vol)` | 日期对齐 → 公共日期 + 索引映射 |
| `construct_factors(R, method)` | 构建共同因子（等权市场/PCA） |
| `estimate_single_window(R_win, F_win, W_list)` | 单窗口 MLE 估计（复用核心函数） |
| `compute_eigenvector_centrality(W)` | 特征向量中心性 |
| `rolling_estimation(...)` | 滚动窗口主循环 |
| `plot_delta_evolution(results_df)` | 绘制 δ(t) 时间序列 |
| `plot_centrality_heatmap(centrality_df)` | 中心性热力图 |
| `main_rolling()` | 滚动估计主入口 |

#### 核心滚动逻辑 (`rolling_estimation`)
```
参数: window=252, step=22 (可配置)
对于每个评估日期 t (从第 window 天开始, 每 step 天一次):
  1. R_win = R[t-251 : t+1]        # 252天窗口
  2. F_win = F[t-251 : t+1]        # 对应因子
  3. W_ret_t = W_ret_3d[net_idx]   # 该日的DY收益率网络
  4. W_vol_t = W_vol_3d[net_idx]   # 该日的DY波动率网络
  5. 调用 estimate_single_window → δ₁(t), δ₂(t), ρ(t)
  6. W*(t) = δ₁·W_ret + δ₂·W_vol
  7. centrality(t) = eigenvector_centrality(W*)
```

#### 关键复用
- `row_normalize()` → 直接复用（第47行）
- `estimate_beta_sigma()` → 直接复用（第237行）
- `concentrated_loglik()` → 直接复用（第295行）
- `estimate_model()` → 改造为 `estimate_single_window()`（去掉print，加多起点）

### 输出

| 文件 | 内容 |
|------|------|
| `results/rolling_delta.csv` | date, delta_ret, delta_vol, rho_0..rho_30, loglik, success |
| `results/rolling_centrality.csv` | date, 31个行业中心性值 |
| `results/delta_evolution.png` | δ(t) 折线图 |
| `results/centrality_heatmap.png` | 中心性热力图 |

---

## 数据流

```
R:  returns.csv  ─→ TVP-VAR ─→ CT[31,31,T] ─→ dy_all_ret.csv + dy_dates_ret.csv
    volatility.csv ─→ TVP-VAR ─→ CT[31,31,T] ─→ dy_all_vol.csv + dy_dates_vol.csv

Python (static_mle_demo.py → main_rolling):
    dy_all_ret.csv  ─→ reshape ─→ W_ret[T,31,31]
    dy_all_vol.csv  ─→ reshape ─→ W_vol[T,31,31]
    returns.csv     ─→ R[6178,31]
    
    align dates → ~6146 common dates
    F = equal_weight(R)
    
    for t in eval_points (~270):
      MLE(R_win, F_win, [W_ret_t, W_vol_t]) → δ(t), ρ(t)
      W*(t) → centrality(t)
    
    → rolling_delta.csv + rolling_centrality.csv + plots
```

---

## 验证

1. **网络加载验证**：随机抽查行和≈1、对角线=0
2. **日期对齐验证**：打印各数据源日期数量和公共日期数量
3. **单窗口试验**：先在中间时间点跑一次 MLE，检查 δ 和 ρ 合理性
4. **收敛率**：全部跑完后统计优化成功率，预期 >80%
5. **经济直觉**：
   - 2008 GFC / 2015 股灾 / 2020 COVID 期间 δ_vol 应上升
   - 金融行业（银行、非银金融）在危机期应更中心

---

## 关键文件

| 文件 | 操作 |
|------|------|
| `src/model/TVP-VAR-DY.R` | 重写 run_dy()，删除 extract_and_save_slice() |
| `src/model/static_mle_demo.py` | 追加滚动估计相关函数 + main_rolling() |
| `src/data/processed/dy_all_*.csv` | R 运行后生成 |
| `src/data/processed/dy_dates_*.csv` | R 运行后生成 |
| `results/` | Python 运行后生成（新建目录） |
