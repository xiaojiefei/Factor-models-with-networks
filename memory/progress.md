# 详细进度日志

> 每次会话结束时在末尾追加新条目，格式统一。

---

## 2026-04-14 — 会话 #7：日度→周度数据转换（三文件联动）

### 本次完成
- **`prepare_data.py` — 新增周度重采样**：
  - 新增 `resample_to_weekly()` 函数：使用 `'W-FRI'` 锚定周五，取每周最后一个交易日收盘价
  - `main()` 流程：日度价格 → 周度重采样 → 周收益率 → GARCH周波动率
  - GARCH年化因子：`np.sqrt(252)` → `np.sqrt(52)`（52周/年）
  - EWMA回退span：`22` → `4`（4周≈1个月）
  - `data_info.json` 新增 `"frequency": "weekly"` 字段
  - 注释/打印信息：`"日收益率"` → `"周收益率"`
- **`TVP-VAR-DY.R` — 周度参数适配**：
  - `nfore = 10` → `nfore = 4`（FEVD预测步长：日度10天≈2周 → 周度4周≈1个月）
  - `nlag = 1` 保持不变（周度VAR(1)合理）
  - `kappa1 = 0.99, kappa2 = 0.96` 保持不变
  - 添加周度数据注释
- **`multilayer_network_mle.py` — 滚动窗口周度适配**：
  - `window = 252` → `window = 52`（1年=52周）
  - `step = 22` → `step = 4`（~1个月=4周）
  - 所有 `days` → `weeks`/`periods`，`daily` → 通用描述
  - 模块文档更新为周度数据管线描述
- **运行 `prepare_data.py` 验证**：
  - 日度价格 6180×31 → 周度价格 1292×31 → 周收益率 1290×31
  - 31个行业GARCH(1,1)全部收敛（无EWMA回退）
  - 收益率与波动率维度完全一致（1290×31）

### 关键产出文件
```
src/data/
├── prepare_data.py                    # 修改：周度重采样+GARCH（248行）
└── processed/
    ├── industry_returns.csv           # 更新：周收益率（1290×31）
    ├── industry_volatility.csv        # 更新：GARCH周波动率（1290×31）
    ├── industry_prices.csv            # 保持：日度收盘价（原始备份）
    └── data_info.json                 # 更新：含frequency="weekly"

src/model/
├── TVP-VAR-DY.R                       # 修改：nfore=4（周度适配）
└── multilayer_network_mle.py          # 修改：window=52, step=4
```

### 使用方法
```bash
# Step 0: 数据预处理（已完成）
cd D:/桌面数据/工作论文/带耦合多层网络/src/data
D:/Python/python.exe prepare_data.py

# Step 1: R中运行（~5-10分钟，周度数据大幅加速）
setwd("D:/桌面数据/工作论文/带耦合多层网络/src/model")
source("TVP-VAR-DY.R")

# Step 2: Python运行
cd D:/桌面数据/工作论文/带耦合多层网络/src/model
D:/Python/python.exe multilayer_network_mle.py
```

### 下次继续
- [ ] 运行R代码生成周度DY网络
- [ ] 运行 `multilayer_network_mle.py` 滚动MLE
- [ ] 分析结果：δ(t)时间演化、中心性变化
- [ ] 确认新文件正常后删除 `static_mle_demo.py`
- [ ] 实现其他因果网络估计方法（GR, QB, Qo, QN）

### 当前阻塞
- 无

---

## 2026-04-14 — 会话 #6：GARCH波动率+代码重构+运行数据管线

### 本次完成
- **`prepare_data.py` — GARCH(1,1)波动率替换滚动窗口**：
  - 添加 `arch` 库导入，新增 `calculate_volatility_garch()` 函数
  - GARCH(1,1)条件波动率：输出维度与收益率完全一致（6178×31，之前滚动窗口丢失21行）
  - EWMA回退机制：GARCH不收敛时自动切换
  - 添加维度一致性断言（shape + index）
  - 修复Unicode打印问题（`✓` → `[OK]`，GBK编码兼容）
  - `data_info.json` 新增 `volatility_method` 和 `n_observations_volatility` 字段
- **重构 `static_mle_demo.py` → `multilayer_network_mle.py`**（1239行→774行）：
  - 删除模拟相关代码（~465行）：`generate_networks`, `generate_data`, `estimate_model`, `print_results`, `plot_results`, `main()`
  - 保留核心管线：网络工具 + MLE核心 + 滚动估计 + 可视化 + main_rolling
  - 入口简化：直接调用 `main_rolling()`，无需 `sys.argv` 分发
  - 新增文件存在性检查（R输出文件）
  - K值从 `industry_names.txt` 动态读取，不再硬编码31
- **运行 `prepare_data.py`**：
  - 31个行业GARCH(1,1)全部收敛（无EWMA回退）
  - 波动率统计合理：均值23-32%（年化），最大78-83%
  - 输出文件已更新：`industry_volatility.csv`（6178×31）
- **Python环境发现**：
  - 系统默认 `python` 指向 Python 3.14（`C:\Python314`），`arch` 不可用
  - 需使用 `D:/Python/python.exe`（Python 3.10 Anaconda，`arch` 7.0.0 已安装）

### 关键产出文件
```
src/data/
├── prepare_data.py                    # 修改：GARCH(1,1)波动率（224行）
└── processed/
    ├── industry_volatility.csv        # 更新：GARCH波动率（6178×31）
    ├── industry_returns.csv           # 重新生成（6178×31）
    ├── industry_prices.csv            # 重新生成
    └── data_info.json                 # 更新：含volatility_method

src/model/
├── multilayer_network_mle.py          # 新建：滚动MLE（774行）
└── static_mle_demo.py                # 旧文件（待删除）
```

### 使用方法
```bash
# Step 0: 数据预处理（已完成）
cd D:/桌面数据/工作论文/带耦合多层网络/src/data
D:/Python/python.exe prepare_data.py

# Step 1: R中运行（20-40分钟）
setwd("D:/桌面数据/工作论文/带耦合多层网络/src/model")
source("TVP-VAR-DY.R")

# Step 2: Python运行
cd D:/桌面数据/工作论文/带耦合多层网络/src/model
D:/Python/python.exe multilayer_network_mle.py
```

### 下次继续
- [ ] 运行R代码生成全时间点DY网络
- [ ] 运行 `multilayer_network_mle.py` 滚动MLE
- [ ] 分析结果：δ(t)时间演化、中心性变化
- [ ] 确认新文件正常后删除 `static_mle_demo.py`
- [ ] 实现其他因果网络估计方法（GR, QB, Qo, QN）

### 当前阻塞
- 无

---

## 2026-04-14 — 会话 #5：全时间点切片网络+滚动窗口MLE

### 本次完成
- **重写 TVP-VAR-DY.R**（121行）：
  - 删除 `extract_and_save_slice()` 函数
  - 重写 `run_dy()` 导出全部时间点的邻接矩阵
  - 存储格式：堆叠CSV（K*T_net行×K列），每31行=一个时点
  - 新增日期映射文件 `dy_dates_ret.csv` / `dy_dates_vol.csv`
- **整合滚动MLE到 static_mle_demo.py**（640行→1239行）：
  - 第六部分：数据加载（load_stacked_networks, load_returns, align_data）
  - 因子构造（等权市场收益率 / PCA第一主成分）
  - estimate_single_window()：多起点优化，复用核心concentrated_loglik
  - rolling_estimation()：滚动窗口主循环（window=252, step=22）
  - compute_eigenvector_centrality()：复合网络中心性
  - 第七部分：可视化（δ演化图/中心性热力图/Top行业折线图）
  - 第八部分：main_rolling() 主入口
- **Bug修复**：
  - 修复变量名遮蔽（`sum(d * W for d, W in ...)` → `sum(dj * W for dj, W in ...)`）
  - import os 移到文件顶部
- **文档**：
  - 创建 `plan/rolling-mle-plan.md` 工作流计划文档（从C盘迁移到项目目录）

### 关键产出文件
```
plan/
└── rolling-mle-plan.md           # 滚动MLE实施计划

src/model/
├── TVP-VAR-DY.R                  # 重写：全时间点导出（121行）
└── static_mle_demo.py            # 扩展：静态+滚动MLE（1239行）
```

### 使用方法
```bash
# Step 1: R中运行（20-40分钟）
setwd("D:/桌面数据/工作论文/带耦合多层网络/src/model")
source("TVP-VAR-DY.R")

# Step 2: Python运行（5-15分钟）
python src/model/static_mle_demo.py rolling
```

### 下次继续
- [ ] 运行R代码生成全时间点DY网络
- [ ] 运行Python滚动MLE
- [ ] 分析结果：δ(t)时间演化、中心性变化
- [ ] 实现其他因果网络方法（GR, QB, Qo, QN）

### 当前阻塞
- 无

---

## 2026-04-14 — 会话 #4：整理Wind数据+编写DY代码

### 本次完成
- 整理Wind行业数据：
  - 处理31个行业Excel文件（1999-2025年）
  - 生成`industry_returns.csv`（6178日×31行业）
  - 生成`industry_volatility.csv`（滚动22日年化波动率）
- 编写DY溢出网络R代码（`TVP-VAR-DY.R`精简版）：
  - 使用`ConnectednessApproach`包实现TVP-VAR
  - 提取时变切片网络：`dca$CT[,,t]`
  - 行标准化邻接矩阵
  - 保存为CSV供Python使用
- 修复R代码问题：
  - 修复字符串重复语法（`rep("=", 60)`替代`"="*60`）
  - 添加`prior = "BayesPrior"`参数
  - 注释掉`window.size`参数（用户修改）

### 关键产出文件
```
src/data/processed/
├── industry_returns.csv          # 日收益率
├── industry_volatility.csv       # 滚动波动率
└── dy_slice_*_t*.csv            # DY切片网络（运行后生成）

src/model/
├── TVP-VAR-DY.R                  # DY模型代码（精简版，~130行）
├── README_DY.md                  # 使用说明
└── static_mle_demo.py            # 静态MLE（之前完成）
```

### 使用方法
```r
setwd("D:/桌面数据/工作论文/带耦合多层网络/src/model")
source("TVP-VAR-DY.R")
# 自动运行: result_ret <- run_dy("returns")
# 自动运行: result_vol <- run_dy("volatility")
```

### 下次继续
- [ ] 运行R代码生成DY切片网络
- [ ] 将切片网络接入多层网络因子模型
- [ ] 估计网络权重$\delta_j$和反应系数$\Lambda$

### 当前阻塞
- 无

---

## 2026-04-13 — 会话 #2：模型理解与代码实现

### 本次完成
- 深入理解了 $(I - \Lambda W)^{-1}$ 的含义（网络乘数效应）
- 详细推导了集中MLE的完整流程（三步：估计beta/Sigma → 构建似然 → 约束优化）
- 解释了 Jacobian 项 $T\ln|A|$ 的来源（变量变换公式）
- 澄清了多元正态分布的使用原因（先一般后特殊）
- 解释了 $\delta_j$ 是静态参数（全样本一个值）
- 讨论了时变参数的可行性和实现方案
- **完成了静态模型估计代码**：`src/model/static_mle_demo.py`
  - 13个资产，3层网络
  - 完整的模拟数据生成
  - 集中MLE估计实现
  - 结果可视化（6个子图）
- **创建了网络工具函数**：`src/model/network_utils.py`
  - 行标准化函数
  - 加权网络生成
  - 网络密度计算
- **撰写了概念文档**：
  - `memory/model-combination-method.md`（模型详解）
  - `memory/network-matrix-faq.md`（网络矩阵FAQ）

### 关键代码成果
```
src/model/
├── static_mle_demo.py    # 主估计代码（550+行）
├── network_utils.py       # 网络工具函数
└── __init__.py
```

### 下次继续
- [ ] 接入真实数据（Kenneth French 48行业组合）
- [ ] 实现4种因果网络估计方法（GR, QB, Qo, QN）
- [ ] 计算网络统计量（密度、中心性等）
- [ ] 与论文Table 3的结果对照验证

### 当前阻塞
- 无

---

## 2026-04-13 — 会话 #1：项目初始化

### 本次完成
- 搭建了完整的持久记忆系统（CLAUDE.md / memory/ / hooks）
- 项目目录下已有论文 PDF：Bonaccolto et al. (2019)

### 下次继续
- [x] 精读论文，提取核心方法论
- [x] 决定使用的编程语言和工具链
- [x] 搭建项目代码骨架

### 当前阻塞
- 无

---

## 2026-04-14 — 会话 #3：代码完善与文档补充

### 本次完成
- 修正 `static_mle_demo.py`：加入截距项 alpha（预期收益率）估计
  - 修改 `estimate_beta_sigma` 函数，返回4个值（alpha, beta, eta, Sigma）
  - 更新 `estimate_model` 和 `concentrated_loglik` 的调用
  - 更新 `print_results` 显示 alpha 估计结果
- 技术问答澄清：
  - 解释 F_t 为什么要乘以 0.02（因子标准差校准）
  - 解释 W_star 计算（复合网络加权求和）
  - 解释 bounds 参数含义（优化边界约束）
  - 解释 Rho 边界 [0, 0.5] 的由来（数值稳定性考虑）
  - 介绍 PCA 第一主成分作为共同因子的方法
- 完善 `memory/model-combination-method.md`：
  - 修正 F_t 和 beta 的维度描述（$n_f = 4$）
  - 新增 1.2.1 节详细解释 $\Lambda = \text{diag}(\rho)$ 参数

### 关键理解深化
| 问题 | 解答 |
|------|------|
| 预期收益率 alpha | 代码中已加入，对应论文 $E[R_t]$ |
| 复合网络 W* | $W^* = \sum \delta_j W_j$，各层网络的加权组合 |
| Lambda 矩阵 | 对角矩阵，每行 W* 乘以对应资产的 $\rho_i$ |
| Rho 边界 | 原设 [0,0.5] 是我自己加的，现已改为 [0,None] 符合论文 |

### 下次继续
- [ ] 接入 Kenneth French 48行业组合真实数据
- [ ] 实现4种因果网络估计方法（GR, QB, Qo, QN）
- [ ] 计算网络统计量（密度、同配性、特征向量中心性）
- [ ] 估计复合网络权重 δ_j 并与论文 Table 3 对照

### 当前阻塞
- 无

---

<!-- 
格式模板（每次追加）：

## YYYY-MM-DD — 会话 #N：[主题]

### 本次完成
- 具体做了什么

### 下次继续
- [ ] 任务1
- [ ] 任务2

### 当前阻塞
- 如有，写明原因
-->
