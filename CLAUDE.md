# 项目记忆：带耦合多层网络 — 论文复现

> ⚡ 本文件由 Claude Code 每次启动时自动加载。
> 每次会话结束前必须更新本文件及 memory/ 下的相关文件。

---

## 📌 项目概览

| 字段 | 内容 |
|------|------|
| **复现论文** | Bonaccolto et al. (2019) — *Estimation and model-based combination of causality networks among large US banks and insurance companies* |
| **PDF 路径** | `文献/Bonaccolto 等 - 2019 - Estimation and model-based combination...pdf` |
| **研究方向** | 金融网络、因果网络、银行与保险公司系统性风险 |
| **项目状态** | 🟡 周度数据管线完成，待运行R生成网络 → Python滚动MLE |
| **最后更新** | 2026-04-14（会话#7：日度→周度数据转换）|

---

## 🎯 当前任务

- **当前阶段**：三文件已改为周度数据，待运行R生成网络 → Python滚动MLE
- **已完成代码**：
  - `src/model/multilayer_network_mle.py` — 滚动窗口MLE（周度：window=52, step=4）
  - `src/model/network_utils.py` — 网络工具函数
  - `src/model/TVP-VAR-DY.R` — DY溢出网络（周度适配：nfore=4）
  - `src/data/prepare_data.py` — Wind数据预处理（周度重采样+GARCH波动率）
- **数据准备**：
  - 31个行业周度数据（2000-2025，1290周观测）
  - 周收益率和GARCH(1,1)周波动率已导出，维度完全一致（1290×31）
- **下一步行动**：
  1. 在R中运行 `source("TVP-VAR-DY.R")` 生成全时间点网络（~5-10分钟）
  2. 运行 `python multilayer_network_mle.py` 执行滚动MLE
  3. 查看 δ(t) 和中心性的时间演化

---

## ✅ 已完成

- [x] 创建项目目录结构
- [x] 搭建持久记忆系统（CLAUDE.md + memory/ + hooks）
- [x] 精读论文，提取核心方法论到 `memory/paper-notes.md`
- [x] 详细理解 $(I - \Lambda W)^{-1}$ 网络乘数效应
- [x] 完整推导集中MLE估计流程
- [x] 澄清静态参数 $\delta_j$ 与时变参数的区分
- [x] 完成静态模型估计代码（模拟数据验证）
- [x] 撰写模型详解文档 `memory/model-combination-method.md`
- [x] **2026-04-14**：修正代码加入截距项（alpha/预期收益率）
- [x] **2026-04-14**：完善文档补充$\Lambda$参数详细解释，修正维度描述
- [x] **2026-04-14**：整理Wind行业数据（31行业，6178日观测）
- [x] **2026-04-14**：编写DY溢出网络R代码（精简版，提取切片网络）
- [x] **2026-04-14**：重写R代码导出全时间点切片网络（堆叠CSV格式）
- [x] **2026-04-14**：整合滚动窗口MLE到static_mle_demo.py（数据加载/日期对齐/滚动估计/中心性/可视化）
- [x] **2026-04-14**：修复变量名遮蔽bug（d→dj）
- [x] **2026-04-14**：创建工作流计划文档 `plan/rolling-mle-plan.md`
- [x] **2026-04-14**：`prepare_data.py` 波动率从滚动窗口改为GARCH(1,1)，维度一致（6178×31）
- [x] **2026-04-14**：重构 `static_mle_demo.py` → `multilayer_network_mle.py`（1239行→774行，删除模拟代码）
- [x] **2026-04-14**：运行 `prepare_data.py` 生成GARCH波动率数据（31行业全部收敛）
- [x] **2026-04-14**：三文件联动改为周度数据（prepare_data.py + TVP-VAR-DY.R + multilayer_network_mle.py）
- [x] **2026-04-14**：运行 `prepare_data.py` 生成周度数据（1290×31，GARCH全部收敛）

---

## 🔲 待完成（按优先级）

- [ ] 运行R代码生成全时间点DY网络（dy_all_ret.csv, dy_all_vol.csv）— 周度数据预计5-10分钟
- [ ] 运行Python滚动MLE（`python multilayer_network_mle.py`）— window=52, step=4
- [ ] 分析δ(t)时间演化和中心性变化
- [ ] 删除旧文件 `static_mle_demo.py`（确认新文件正常后）
- [ ] 实现其他因果网络估计方法（GR, QB, Qo, QN）作为对比

---

## 📁 项目文件结构

```
带耦合多层网络/
├── CLAUDE.md                    ← 本文件（主记忆，自动加载）
├── memory/
│   ├── progress.md              ← 详细进度日志（每步记录）
│   ├── paper-notes.md           ← 论文方法论笔记
│   ├── model-combination-method.md  ← 模型详解（含MLE推导）
│   ├── network-matrix-faq.md    ← 网络矩阵常见问题
│   └── session-log.md           ← 历次会话摘要
├── plan/
│   └── rolling-mle-plan.md      ← 滚动MLE实施计划
├── src/
│   ├── model/
│   │   ├── multilayer_network_mle.py ← 滚动MLE（774行，从static_mle_demo.py重构）
│   │   ├── static_mle_demo.py   ← 旧文件（待删除，已被multilayer_network_mle.py替代）
│   │   ├── TVP-VAR-DY.R         ← DY溢出网络（全时间点导出，121行）
│   │   ├── network_utils.py     ← 网络工具函数
│   │   └── __init__.py
│   ├── data/
│   │   ├── prepare_data.py      ← Wind数据预处理（已完成）
│   │   ├── wind下载/            ← 原始Wind行业Excel（31个）
│   │   └── processed/           ← 处理后数据
│   │       ├── industry_returns.csv      ← 周收益率（1290×31）
│   │       ├── industry_volatility.csv   ← GARCH(1,1)周波动率（1290×31）
│   │       ├── industry_prices.csv       ← 收盘价
│   │       ├── industry_names.txt        ← 行业名称
│   │       ├── data_info.json            ← 数据信息
│   │       ├── dy_all_ret.csv            ← DY收益率网络堆叠（R运行后生成）
│   │       ├── dy_all_vol.csv            ← DY波动率网络堆叠（R运行后生成）
│   │       ├── dy_dates_ret.csv          ← 收益率网络日期映射（R运行后生成）
│   │       └── dy_dates_vol.csv          ← 波动率网络日期映射（R运行后生成）
│   ├── causality/               ← 因果检验方法（待实现）
│   └── network/                 ← 网络分析（待实现）
├── results/                     ← 滚动MLE结果（Python运行后生成）
│   ├── rolling_delta.csv        ← δ(t)时间序列
│   ├── rolling_centrality.csv   ← 中心性时间序列
│   ├── delta_evolution.png      ← δ演化图
│   ├── centrality_heatmap.png   ← 中心性热力图
│   └── top_central_industries.png ← Top行业中心性
├── 文献/                        ← 参考文献 PDF
└── .claude/                     ← Hook 配置
```

---

## 🧠 核心方法论总结

### 1. 模型设定
结构方程：$A \cdot R_t = \beta F_t + \eta_t$，其中 $A = I - \Lambda \sum_{j=1}^d \delta_j W_j$

### 2. 估计方法：集中MLE
- **Step 1**：给定 $A$，OLS估计 $\beta$ 和 $\Sigma_\eta$
- **Step 2**：代回似然，得到仅关于 $(\delta, \Lambda)$ 的集中似然
- **Step 3**：约束优化（$\delta \geq 0$, $\sum \delta_j = 1$）

### 3. 关键理解
| 概念 | 解释 |
|------|------|
| $(I - \Lambda W)^{-1}$ | 网络乘数效应，捕捉冲击的无限阶传播 |
| $\delta_j$ | 静态参数，全样本一个值（时变需要特殊处理） |
| 行标准化 | 必须满足（数值稳定性 + 参数可解释性） |
| 有权/无权矩阵 | 都可以，关键是行标准化 |

### 4. 已完成代码文件
- `src/model/multilayer_network_mle.py` — 滚动窗口MLE（周度：window=52, step=4）
  - Part 1：网络工具函数（row_normalize等）
  - Part 2：核心MLE（estimate_beta_sigma, concentrated_loglik）
  - Part 3：滚动MLE数据加载/日期对齐/估计
  - Part 4：可视化（δ演化/中心性热力图）
  - Part 5：`main_rolling()` 主入口
- `src/model/TVP-VAR-DY.R` — 全时间点DY网络导出（周度适配：nfore=4）
- `src/model/network_utils.py` — 网络工具函数
- `src/data/prepare_data.py` — Wind数据预处理（周度重采样+GARCH波动率，248行）

---

## 🤖 Claude 行为指令

每次会话开始时：
1. 读取本文件，了解项目当前状态
2. 读取 `memory/progress.md` 获取详细进度
3. 读取 `memory/paper-notes.md` 获取论文理解
4. 接续上次工作，不需要用户重新解释背景

每次会话结束前（用户说"结束"/"再见"/"保存"等）：
1. 更新本文件的"当前任务"和"已完成"部分
2. 在 `memory/progress.md` 追加本次进度
3. 在 `memory/session-log.md` 追加本次会话摘要
4. 如有重要技术决策，记录到 `memory/decisions.md`

*最后更新：2026-04-14（会话#7：日度→周度数据转换）*
