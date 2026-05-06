# 会话历史日志

> 每次会话的简短摘要，帮助快速了解项目演进过程。

---

## 会话 #1 — 2026-04-13

**主要工作**：项目初始化，搭建持久记忆系统  
**产出文件**：CLAUDE.md、memory/（4个文件）、.claude/hooks/（2个脚本）、.claude/settings.json  
**下次优先**：精读论文 PDF，填写 paper-notes.md

---

## 会话 #2 — 2026-04-13

**主要工作**：模型理解与代码实现  
**产出文件**：`src/model/static_mle_demo.py`（550+行）、`src/model/network_utils.py`、`memory/model-combination-method.md`、`memory/network-matrix-faq.md`  
**关键成果**：完成静态MLE估计代码（模拟数据验证），深入理解$(I-\Lambda W)^{-1}$网络乘数效应  
**下次优先**：接入真实数据，实现4种因果网络估计方法

---

## 会话 #3 — 2026-04-14

**主要工作**：代码完善与文档补充  
**产出更新**：
- `src/model/static_mle_demo.py`：加入alpha截距项估计
- `memory/model-combination-method.md`：修正维度描述，新增$\Lambda$参数详解
**关键澄清**：
- 预期收益率alpha已加入代码（对应论文$E[R_t]$）
- $F_t$维度应为$n_f \times 1$（$n_f=4$），beta为$K \times n_f$
- Rho边界0.5是我自己加的数值稳定性约束，论文只要求$\rho_i > 0$
**下次优先**：接入Kenneth French 48行业组合数据，实现因果网络估计方法

---

## 会话 #4 — 2026-04-14

**主要工作**：整理Wind行业数据，编写DY溢出网络R代码
**产出文件**：
- `src/data/prepare_data.py` — Wind数据预处理
- `src/data/processed/` — 31行业收益率/波动率数据
- `src/model/TVP-VAR-DY.R` — DY模型精简版代码
- `src/model/README_DY.md` — 使用说明
**关键成果**：
- 整理31个行业6178日观测数据
- 实现TVP-VAR-DY切片网络提取
- 邻接矩阵已行标准化，可直接用于因子模型
**代码修复**：
- R字符串重复语法修复
- 添加BayesPrior参数
**下次优先**：运行R代码生成切片网络，接入多层因子模型

---

## 会话 #5 — 2026-04-14

**主要工作**：全时间点切片网络导出 + 滚动窗口MLE整合
**产出文件**：
- `src/model/TVP-VAR-DY.R` — 重写为全时间点导出版（121行）
- `src/model/static_mle_demo.py` — 追加滚动MLE模块（640→1239行）
- `plan/rolling-mle-plan.md` — 工作流计划文档
**关键成果**：
- R代码改为保存全部时变邻接矩阵（堆叠CSV格式）
- Python滚动MLE：window=252, step=22, 多起点优化, 中心性计算+可视化
- 修复变量名遮蔽bug（d→dj）
- 完整数据流：R生成网络CSV → Python加载+日期对齐+滚动估计
**下次优先**：运行R代码→运行Python滚动MLE→分析δ(t)和中心性演化

---

<!-- 新会话追加格式：

## 会话 #N — YYYY-MM-DD

**主要工作**：xxx
**产出文件**：xxx
**下次优先**：xxx

-->

## 会话 #6 — 2026-04-14

**主要工作**：数据管线改进（GARCH波动率 + 代码重构 + 运行验证）
**产出文件**：
- `src/data/prepare_data.py` — GARCH(1,1)波动率替换滚动窗口（224行）
- `src/model/multilayer_network_mle.py` — 从static_mle_demo.py重构（774行）
- `src/data/processed/industry_volatility.csv` — GARCH波动率（6178×31）
**关键成果**：
- 收益率与波动率维度完全一致（6178×31），消除21行差异
- 31个行业GARCH(1,1)全部收敛
- 代码从1239行精简至774行，删除模拟代码
- 发现Python环境问题：需用 `D:/Python/python.exe`（3.10）而非默认3.14
**下次优先**：运行R代码生成DY网络 → 运行Python滚动MLE → 分析结果

---

## 会话 #7 — 2026-04-14

**主要工作**：三文件联动修改，日度数据→周度数据（加速TVP-VAR估计）
**产出文件**：
- `src/data/prepare_data.py` — 新增周度重采样 + 年化因子√252→√52（248行）
- `src/model/TVP-VAR-DY.R` — nfore 10→4（周度适配）
- `src/model/multilayer_network_mle.py` — window 252→52, step 22→4
- `src/data/processed/` — 周度数据（1290×31）
**关键成果**：
- 数据量从6178日→1290周（减少80%），TVP-VAR预计从20-40分钟缩短到5-10分钟
- 三文件输入输出保持一致，文件名不变，仅内容从日度变为周度
- 31个行业GARCH(1,1)在周度数据上全部收敛
- `data_info.json` 新增 `frequency: "weekly"` 字段
**下次优先**：运行R代码生成周度DY网络 → 运行Python滚动MLE → 分析结果
