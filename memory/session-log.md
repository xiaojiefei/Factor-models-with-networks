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

---

## 会话 #8 — 2026-04-15

**主要工作**：模型概念深化（δ_j含义、传染机制、邻接矩阵一般性）  
**关键澄清**：
- δ_j是网络层混合权重（凸组合系数），反映哪种网络视角更能解释同期关联
- 金融传染 = 冲击的传导（不是相关性），通过收益率溢出(W1)和波动率溢出(W2)两个渠道
- 邻接矩阵不一定要与风险相关，任何满足A1-A4假设的关系矩阵均可（投入产出表、地理邻近、供应链等）
- ρ_i索引资产（哪个行业更易受传染），δ_j索引网络层（哪种方法更重要）
**下次优先**：运行R+Python管线 / 项目书修改

---

## 会话 #9 — 2026-04-16

**主要工作**：项目申报书全面诊断与修改
**文件**：`项目/附件2：上海对外经贸大学研究生科研创新培育项目申报书.docx`
**关键成果**：
- 全面诊断申报书7个部分，输出修改优先级（P0/P1/P2）
- 研究方案大幅扩充：目标#3补全、模块2多层耦合细节扩充、模块3从1步→2步
- 立项依据：开头段重写（多冲击视角）、新增Gap总结（4条不足）
- 新增难点#3（驱动因素分离+内生性控制）
- 用户已完成修改并更新Word文档
**申报书修改后状态**：
- 主要目标：3条完整 ✅
- 难点：3条对应3模块 ✅
- 框架思路：模块1(3步)+模块2(3步)+模块3(2步) ✅
- 文献评述+Gap总结 ✅
- 立项依据开头已改为多冲击视角 ✅
- 待完善：预期研究成果(空)、研究阶段时间表(空)、参考文献目录(空)
**下次优先**：填写预期研究成果+研究阶段+参考文献目录；继续代码管线

---

## 会话 #10 — 2026-04-17

**主要工作**：申报书立项依据+文献评述精修
**文件**：`项目/附件2：上海对外经贸大学研究生科研创新培育项目申报书.docx`
**关键成果**：
- 文献评述3段修改：
  - 第一段（研究视角）：突出风险关联vs韧性关联的差异与互补关系
  - 第二段（冲击来源）：修正"联合冲击"误导为"多种冲击下韧性差异性"
  - 第四段（驱动因素）：区分韧性水平驱动（现有）vs韧性关联强度驱动（本项目）
- 全文校对发现并修复：
  - 引用错误："Tobias and Brunnermeier" → "Adrian和Brunnermeier"（2处）
  - 缺失括号："Wei and Zhou, 2024a)"
  - 引用格式不统一（英文et al. vs中文等）
  - 错别字："拓结构" → "拓扑结构"
  - 文献评述第四段末缺总结句
  - 文献综述引言与章节标题不匹配
  - Brunnermeier(2024)在参考文献中但正文未引用
- 开头段精简：
  - 去掉"金融端/非金融端"分类框架
  - 加入逻辑桥梁：行业关联 → 为冲击传导提供渠道 → 风险研究 → 韧性gap
  - 嵌入Brunnermeier(2024)支撑韧性视角
  - 修复"金融"重复
- **用户反馈**：要求所有建议必须先确保逻辑通顺（已记录到feedback memory）
**下次优先**：补全参考文献目录（30+篇）+预期研究成果+研究阶段时间表；继续代码管线

---

## 会话 #11 — 2026-04-18

**主要工作**：精读 Billio et al. (2023)，补充推导细节到 `memory/billio2023-derivation.md`  
**产出更新**：
- `memory/billio2023-derivation.md`：新增3个小节
  - §1.2.1：Eq.(2) 的来源——从联立方程 (★) 出发，3×3 矩阵展开
  - §1.2.3：为什么要减去均值 E[R_t]（冲击 vs 正常水平）
  - §2.3.1：Eq.(4) 的来源——令 a_ij = ρ·W_ij，870 个参数→1 个
**关键理解**：
- Eq.(2) 不是从 Eq.(1) 推导出来的，而是一个全新的建模假设（联立方程）
- (I-ρW) 结构是"反馈系统移项"的数学必然，同样出现在 Leontief、PageRank 等
- 最终估计参数：K 个 ρ_i（数值优化）+ α, β̄, Ω_η（OLS 解析解）
**下次优先**：继续论文精读或回到代码管线（运行R+Python）

---

## 会话 #12 — 2026-04-20

**主要工作**：研究创新方向头脑风暴（基于 Bonaccolto 2019 框架）
**产出文件**：
- `memory/research-ideas.md`（新建）：11个创新方向 + OVB分析 + DY溢出合理性论证
**关键成果**：
- 推荐论文结构：A(混合网络)+B(δ驱动因素)+C(韧性视角) 为主体，I(嵌套检验)+J(内生性) 为必做，D(体制切换)+E(预测组合) 为扩展
- 非对称效应的根本困难：$(I-\rho W)^{-1}$ 闭式解因 max/min 非线性丢失
- 为什么不能去掉 ρ：行标准化 W 的 $\lambda_{max}=1$ → $\det(I-W)=0$
**下次优先**：运行代码管线 / §4精读 / 项目书

---

## 会话 #13-14 — 2026-04-20/21

**主要工作**：Billio (2023) §4 Model Estimation 逐段精讲，重写衍生笔记§6
**产出更新**：
- `memory/billio2023-derivation.md`：§6 从~110行扩充为~230行
  - 新增：参数总览、识别条件、阶条件、假设4.1-4.3详解
  - 新增：特征值前置知识（定义、推导、数值验证）
  - 重写：似然函数逐项解释、集中技巧详解、完整伪代码、Bonaccolto对照表
**关键概念澄清**：
- 均值参数（$\alpha^*, \beta^*$）vs 协方差参数（$\Sigma_{\varepsilon^*}$）
- 特征值从零讲起：$Wv=\lambda v$ → $\det(I-\rho W)=\prod(1-\rho\lambda_i)$ → 行标准化 $\lambda_{max}=1$
- $\frac{1}{\lambda_{min}} < \rho < \frac{1}{\lambda_{max}}$ 的数值验证（4个特征值逐一列约束）
**下次优先**：运行R+Python管线 / 项目书修改

---

## 会话 #14 — 2026-04-21

**主要工作**：Billio (2023) §6 I/O表解读 + 外生/内生暴露 + β*非线性 + 同期性问题
**产出文件**：
- `memory/billio2023-io-as-adjacency.md`（新建）：Billio (2023) §6完整解读（I/O表作为邻接矩阵的三层逻辑、制造业冲击例子、实证结果、与Bonaccolto对比）
- `memory/billio2023-derivation.md`（3处新增）：
  - §2.3.3：外生暴露(β̄ᵢ)与内生暴露(ρ)的区分
  - §3.1.1：β*=(I-ρW)⁻¹β̄为什么是非线性函数（Neumann级数展开）
  - §5.4：同期性问题对比（Billio用外生I/O表 vs Bonaccolto用频率错配技巧）
**关键概念澄清**：
- I/O表天然就是有向有权图（节点=行业，边=供货，权重=I/O系数）
- β̄ᵢ是外生暴露（无网络也存在），ρ是内生暴露（同行效应，自我强化）
- β*包含ρ的所有幂次（Neumann级数），所以是非线性函数
- Bonaccolto频率错配：日度lag-1因果检验 → 周度同期因子模型
**下次优先**：运行R+Python管线 / 项目书修改

---

## 会话 #15 — 2026-04-21

**主要工作**：Billio (2023) §2.3 原文段落精讲，补充网络效应的三个深层后果
**产出更新**：
- `memory/billio2023-derivation.md`（1处新增）：
  - §3.5 网络效应的三个深层后果：
    - §3.5.1：行业特有因子通过$(I-\rho W)^{-1}$"伪装"成公共因子（错误归因）
    - §3.5.2：结构独立的特质冲击通过网络变得相关（解释Table 9残差相关从0.10降到0.03）
    - §3.5.3：LeSage-Pace直接/间接效应分解不适用（因子$F_t$是公共的，非个体特有变量）
**下次优先**：运行R+Python管线 / 项目书修改

---

## 会话 #16 — 2026-04-21

**主要工作**：Billio (2023) §2.4 风险分解与分散化逐步推导
**产出更新**：
- `memory/billio2023-derivation.md`：§4 全面重写扩充（~50行→~170行）
  - §4.0：起点回顾 + 符号说明（论文$A$ vs 笔记$\Omega$）
  - §4.1：Eq.(16) 逐步推导（$\text{Var}(BX) = B\Sigma B'$ 规则）
  - §4.2：与传统模型对比 + 2×2手算验证（非对角协方差出现）
  - §4.3：组合风险 Eq.(19)
  - §4.4：简化假设推导——Sherman-Morrison求逆完全图$\Omega$、证明$\Omega\omega = \frac{1}{K(1-\rho)}\mathbf{1}$、组合特质风险$\frac{\bar{\sigma}^2}{K(1-\rho)^2}$
  - §4.5：放大因子表（$\rho=0.5$时需4倍资产）
  - §4.6：极限行为 Eq.(21)
  - §4.7：三个核心信息总结表
- 校正Eq.(16)符号：$\sigma_m^2$替换$\Sigma_F$，$\Omega_\eta$替换$\Sigma_\eta$，引用论文原文形式
**下次优先**：运行R+Python管线 / 项目书修改

---

## 会话 #17 — 2026-04-22

**主要工作**：Billio (2023) §5.1 系统性风险与 ρ 的关系 + §6.6-6.7 数据结构补充
**产出更新**：
- `memory/billio2023-derivation.md`（4处新增/修改）：
  - §5.1.1：负 ρ 削弱系统性风险（Neumann 级数交替相消 + amplifier/absorber/insulator 分类）
  - §5.1.2 重写：双渠道结构（量变=β*放大 vs 质变=特质风险不可分散）+ 论文 Markowitz/Cochrane 定义
  - §6.6：η_t 身份澄清（结构残差 vs 约化残差对比表）
  - §6.7.1：数据结构具象化（K=31, T=52 维度展开 + Python 代码对应）
**用户反馈**：系统性风险核心应聚焦特质风险渠道（质变），而非 β* 放大（量变）
**下次优先**：运行R+Python管线 / 项目书修改

---

## 会话 #18-19 — 2026-04-24

**主要工作**：LM跳跃检验批量运行 + 滚动窗口VAR DY溢出网络（3层）
**产出文件**：
- `src/model/lm_har.py` — LM+HAR.py的模块化重写（457行，9个函数，修复5个bug，向量化6个循环）
- `src/model/run_lm_batch.py` — 批量运行10个行业LM检验
- `data/raw/lm_results/*.csv` × 10 — LM检验结果（4167日×21列）
- `src/model/RollingVAR-DY.R` — 滚动窗口VAR DY溢出（3层网络：连续/正跳/负跳）
- `data/raw/lm_results/dy_all_jsv_pos.csv` — 正跳DY网络（39710行×10列）
- `data/raw/lm_results/dy_all_jsv_neg.csv` — 负跳DY网络（39710行×10列）
**关键成果**：
- 重写LM+HAR.py：修复变量别名bug（J=T/I=J/k被覆盖）、向量化所有循环、加入类型提示
- 10个行业日度跳跃检验全部完成（000032-000041，~10秒/行业）
- 3层多层网络设计：CSVt_d(连续)、JSVt_zheng_d(正跳)、JSVt_fu_d(负跳) → $W_1, W_2, W_3$
- 修复CSVt_d中Inf问题（20171130全bar跳跃→meanvol除零），jsv_pos/jsv_neg已成功
- 用户手动调整VAR参数：nlag=4→1, nfore=10→5
**下次优先**：重跑CSVt_d的DY → 适配MLE代码读取新3层网络 → 运行滚动MLE

---

## 会话 #20 — 2026-04-24

**主要工作**：诊断δ角点解问题 + 实现两步法MLE + Max标准化 + 行业名修正
**产出文件**：
- `src/model/multilayer_network_mle.py` — 两步法MLE（Step1标量ρ→δ̂, Step2固定δ̂→λ_i）+ max_row_normalize + PCA因子 + 上证行业正确名称
- `src/model/RollingVAR-DY.R` — 导出原始矩阵（去掉row_normalize，为max normalization做准备）
- `src/model/TVP-VAR-DY.R` — 适配10行业日度跳跃数据 + 导出原始矩阵
**关键成果**：
- 诊断出δ打到0/1边界的根因：10个自由λ_i让δ不可识别（13参数，似然面平坦）
- 两步法解决：Step1用标量ρ锁定δ（4参数→可识别），Step2固定δ估计异质性λ_i（10参数）
- 实现论文Eq.25的max row normalization（保留网络密度时变特征）
- 修正行业名称：000032-000041是上证行业指数（能源/原材料/工业/...），非申万行业
**下次优先**：重跑R代码生成未标准化网络 → 运行两步法MLE → 分析结果

---

## 会话 #21 — 2026-04-25

**主要工作**：严格按 Billio §4 将 MLE 改为窗口内时变网络 likelihood 与约束

**产出文件**：

- `src/model/multilayer_network_mle.py` — rolling window 内传入完整 `W_τ` 序列，使用逐期 `A_τ` 和 `sum log|A_τ|`
- `CLAUDE.md`、`memory/progress.md`、`memory/session-log.md`、`memory/decisions.md` — 更新项目记忆

**关键成果**：

- scalar ρ 约束从 endpoint `W*` 改为窗口内所有 `W_τ*` 的交集特征值约束
- heterogeneous λ_i 约束改为对窗口内所有 `W_τ*` 检查 spectral radius 稳定性
- `estimate_beta_sigma()` 支持 3D `A_seq`，逐期计算结构变换后的残差
- 已通过 `python -m py_compile "src/model/multilayer_network_mle.py"`

**下次优先**：重跑 RollingVAR-DY.R 生成未标准化网络 → 用 `D:/Python/python.exe` 运行 MLE → 分析 δ(t)、ρ(t)、λ_i(t)

---

## 会话 #22 — 2026-04-25

**主要工作**：严格改为 Bonaccolto §2 one-step constrained concentrated MLE，并新增多进程滚动估计版本

**产出文件**：

- `src/model/multilayer_network_mle.py` — 两步法MLE替换为one-step联合估计 $\delta_j$ 与异质性 $\rho_i$
- `src/model/multilayer_network_mle_Multithreading.py` — 新增多进程rolling window估计脚本，输出 `_multithreading` 后缀结果文件
- `CLAUDE.md`、`memory/progress.md`、`memory/session-log.md`、`memory/decisions.md` — 更新项目记忆

**关键成果**：

- 严格按Bonaccolto文献逻辑：不再使用Step1标量ρ识别δ、Step2固定δ估λ_i的工程化两步法
- one-step优化变量为 $[\delta_1,\ldots,\delta_d,\rho_1,\ldots,\rho_K]$，同时保留窗口内时变 $A_t$ 与 $\sum_t\log|A_t|$
- 多进程版使用 `ProcessPoolExecutor`、`batch_size` 与BLAS线程限制，缓解Windows spawn开销和CPU利用率低的问题
- 澄清 `window=252`、`step=1`、堆叠DY网络CSV、Dirichlet alpha、以及收益率 `r_now` 的加载位置
- 发现当前DY网络文件长度不一致：`dy_all_csv.csv` 为3970期，`dy_all_jsv_pos/neg.csv` 为3971期，建议重跑R生成一致的原始矩阵

**下次优先**：重跑 RollingVAR-DY.R → 运行多进程one-step MLE → 检查 `rolling_delta_multithreading.csv` 中 δ(t)、异质性ρ_i(t) 与 `d_jsv_pos≈1` 是否持续

---

## 会话 #23 — 2026-04-25

**主要工作**：将多进程MLE脚本改为贴近旧代码的 joblib Parallel 写法，并澄清 concentrated MLE 的“OLS + 优化A参数”逻辑

**产出文件**：

- `src/model/multilayer_network_mle_Multithreading.py` — 从 ProcessPoolExecutor/batch 版本改为 `joblib.Parallel(..., prefer="processes")` + `delayed` 写法
- `CLAUDE.md`、`memory/progress.md`、`memory/session-log.md` — 更新项目记忆

**关键成果**：

- 删除自定义 `NetworkConfig.batch_size`、手动 `task_batches` 和 `_estimate_window_batch()`，更贴近用户旧预测代码中的并行方式
- 每个 rolling window 现在对应一个 `delayed(_estimate_window_job)(...)`，由 joblib 自己调度任务
- 已通过 `python -m py_compile "src/model/multilayer_network_mle_Multithreading.py"`
- 澄清 concentrated MLE：每一组候选 $(\delta,\rho)$ 下都重新构造 $A_t$ 并对 $A_tR_t$ 做OLS，再将 $\Sigma_\eta$ 代回似然；真正数值优化的是3个网络权重 + 10个异质性 $\rho_i$

**下次优先**：运行joblib版本MLE脚本观察CPU利用率 → 如正常，再重跑R生成一致DY网络并运行完整one-step MLE

---

## 会话 #24 — 2026-04-28

**主要工作**：诊断 $\delta$ 弱识别问题，并新增窗口内固定W的 Bonaccolto-style static-window MLE脚本

**产出文件**：

- `src/model/multilayer_network_mle_static_window.py` — 新增252日窗口、5日步长、窗口内固定W的rolling MLE版本
- `CLAUDE.md`、`memory/progress.md`、`memory/session-log.md`、`memory/decisions.md` — 更新项目记忆

**关键成果**：

- 解释 $\delta$ 出现1/3停滞或0/1角点的主要原因：网络层高度相关、$\rho_i$ 与 $\delta_j$ 共同进入 $\operatorname{diag}(\rho)\sum_j\delta_jW_j$、SLSQP在平坦似然面上易停在初值或边界
- 新增 static-window benchmark：每个窗口用252个交易日，步长为5，每个窗口每层网络固定为窗口内原始 $W_{j,t}$ 的均值后行标准化
- 修正固定A似然的Jacobian缩放：static版本使用 $T\log|A|$，dynamic版本继续使用 $\sum_t\log|A_t|$
- 明确 static-window 版本不使用全样本 max normalization；max normalization 仅保留给 fully time-varying 版本用于保留网络密度时变信息
- 澄清当前代码中 $\rho\ge0$ 由 likelihood penalty 和 optimizer bounds 同时强制；若要允许负 $\rho_i$，需要同时改这两处并保留 spectral radius 稳定性约束

**下次优先**：运行 `src/model/multilayer_network_mle_static_window.py` → 检查 `rolling_delta_static_window.csv` → 与 `_multithreading` 动态版本结果对比，判断识别问题来源

---

## 会话 #25 — 2026-04-28

**主要工作**：核对 static-window MLE 与Bonaccolto/Billio原文的一致性，并解释rho边界、优化器、PCA因子对单窗口估计的影响

**产出更新**：

- `CLAUDE.md`：更新项目状态、当前任务、已完成项、待办项、方法总结和最后更新时间
- `memory/progress.md`：追加本会话详细进度
- `memory/session-log.md`：追加本会话摘要
- `memory/decisions.md`：记录rho边界与factor specification的技术决策

**关键结论**：

- `src/model/multilayer_network_mle_static_window.py` 的固定网络集中似然在 `dirichlet_alpha=1.0` 时与Bonaccolto固定网络MLE形式一致：$-T/2\log|\Sigma_\eta| + T\log|A|$。
- Bonaccolto原文没有要求异质性 $\rho_i\ge0$；负 $\rho_i$ 应允许，有限对称rho边界只能作为数值优化稳定化设定。
- 无界/过宽rho可产生 `rho_avg=1631.414` 这类经济上不可解释的结果，说明存在弱识别或数值病态。
- rho边界从 `[-10,10]` 改为 `[-5,5]` 后δ和rho符号显著变化，说明结果对边界敏感，不是单纯迭代次数不足。
- 改用PCA因子后单窗口结果变为更均衡的 `d_csv=0.229, d_jsv_pos=0.333, d_jsv_neg=0.438, rho_avg=0.565`，说明共同因子设定会影响网络权重识别。
- `multilayer_network_mle_Multithreading.py` 是Billio式时变网络扩展，不应等同于Bonaccolto主文固定网络经验设定；其中的spectral-radius检查、determinant sign限制、`dirichlet_alpha>1`等需要作为扩展/数值设定单独说明。

**下次优先**：为static-window版本补充rho边界敏感性诊断输出 → 跑 `[-1,1]`、`[-2,2]`、`[-5,5]`、`[-10,10]` 与 equal_weight/PCA 对照 → 多窗口验证δ是否仍角点或1/3停滞

---

## 2026-05-05 — 会话 #26

**讨论主题**：网络稀疏性诊断、多线程优化、参数对应关系

**关键发现**：

1. **网络稀疏性问题**：Granger因果网络存在大量全零矩阵
   - CSV层：91.9%零元素，49.0%全零行
   - JSV_POS层：91.3%零元素，44.7%全零行  
   - JSV_NEG层：92.4%零元素，52.1%全零行
   - 前5个时间点中4个完全全零，导致MLE不可识别

2. **MLE运行结果**：用户运行565窗口估计，收敛率100%，但rho仍有极端值
   - d_csv/d_jsv_pos/d_jsv_neg均值相对均衡（0.32-0.35）
   - 但rho_avg出现-14.029、16.916等异常值

3. **脚本对应关系**：Granger与MLE的step参数不完全对应
   - Granger实际生成~7天间隔的网络
   - MLE step=5在共同日期上采样，不完全一一对应

**修改文件**：
- `src/model/rolling_loo_granger.py`：添加多线程支持（joblib.Parallel）

**下一步行动**：
- 修复网络稀疏性（放宽alpha或增加窗口）
- 添加rho边界限制
- 统一step配置或明确文档说明


---

## 会话 #27 — 2026-05-05

**主要工作**：对比待投论文(ECMODE-D-26-00256)§4.1估计方法与当前代码

**关键发现**：
1. **论文窗口设置**：250个交易日窗口，步长5个交易日（§4.1明确说明）
2. **方法差异**：
   - 论文：GARCHSK（方差+偏度+峰度）+ 非线性Granger + data sharpening
   - 当前：LM跳跃检验 + HAR-RV + 线性条件Granger
3. **风险维度**：论文五维（价格/波动/非对称/极端/尾部）vs 当前三维（csv/jsv_pos/jsv_neg）

**产出文件**：
- `src/model/multilayer_network_mle_integrated.py` — 新增整合版本，分离MLE窗口与网络生成窗口

**对照表**：

| 维度 | 论文 | 当前代码 |
|------|------|----------|
| 窗口 | 250日, step=5 | 252日, step=1(可调) |
| 波动率模型 | GARCHSK | LM+HAR-RV |
| 因果检验 | 非线性Granger | 线性Granger |
| 网络估计 | Bonaccolto δ(t) | ✓ 相同 |

**下次优先**：
1. 统一窗口配置为(250,5)
2. 放宽Granger检验alpha至0.10-0.20
3. 添加rho边界[-5,5]
4. 评估迁移到GARCHSK+非线性Granger的可行性

---

## 会话 #28 — 2026-05-05

**主要工作**：集成脚本`multilayer_network_mle_integrated.py`修复与运行

**关键修复**：
1. **窗口起始点修复**：`first_valid_idx = max(window, granger_window) - 1`，确保Granger网络有足够数据
2. **数据清理修复**：添加`replace([np.inf, -np.inf], np.nan)`和`dropna()`，与`rolling_loo_granger.py`保持一致
3. **索引修复**：`DatetimeIndex`不能用`reset_index()`，改用布尔掩码索引
4. **数值稳定性**：`_residual_variance`添加try-except回退到ridge正则化

**架构变更**：
- **现场计算Granger网络**替代预计算DY网络，避免数据对齐问题
- MLE窗口(`window=252`)与Granger窗口(`granger_window=252`)分离配置
- 支持独立调整：短窗口MLE捕捉近期动态，长窗口Granger获得稳定网络

**当前配置**（main_integrated中）：
```python
window=252              # MLE收益率窗口
granger_window=252      # Granger网络窗口（与论文一致）
step=5                  # 每5天估计一次
granger_alpha=0.10      # 10%显著性水平（解决稀疏性）
factor_method="pca"     # PCA第一主成分
rho_init_values=[-0.3, 0.0, 0.3]  # 收窄初始值范围
```

**运行状态**：
- 评估点：734个窗口（4167期数据，step=5）
- 工人进程：55个（自动检测）
- 当前状态：运行中（用户反馈正在运行）

**重要说明**：
- **不再使用DY网络**：当前采用现场计算的Granger因果网络，预计算的`dy_all_*.csv`文件已不再需要
- `RollingVAR-DY.R`和`TVP-VAR-DY.R`保留但当前未使用
- `rolling_loo_granger.py`被集成脚本替代

**下次优先**：
1. 监控运行结果，检查收敛率和delta分布
2. 若rho仍有极端值，添加硬边界[-5,5]
3. 若delta出现角点，尝试切换factor_method为equal_weight
4. 评估是否需要迁移到GARCHSK+非线性Granger（长期）

---

## 会话 #28续 — 2026-05-05

**主要工作**：修复 `_residual_variance` SVD不收敛错误

**问题诊断**：
- 运行时出现 `LinAlgError: SVD did not converge`
- 即使try-except fallback使用 `pinv` 也失败
- 根本原因是输入数据仍包含NaN/Inf，且ridge惩罚(1e-6)不够强

**修复内容**：
1. **前置数据清理**：添加 `valid_mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)`，在进入lstsq前清理数据
2. **数据量检查**：清理后若样本不足，返回大惩罚值1e6
3. **双层fallback**：
   - Fallback 1：更强的ridge惩罚(1e-4) + `solve` 替代 `pinv`
   - Fallback 2：直接返回 `np.var(y)` 作为 unexplained variance

**修改文件**：
- `src/model/multilayer_network_mle_integrated.py` — `_residual_variance()` 函数（约第152行）

**下次优先**：
重新运行集成脚本，验证SVD错误已修复


---

## 会话 #28续2 — 2026-05-05

**主要工作**：修复 `load_data_with_risk_measures` 日期对齐与 NaN/Inf 清理问题

**问题诊断**：
- 原代码只清理收益率中的 NaN/Inf，`layer_data` 仅通过 `isin(dates)` 对齐日期
- 风险指标中的 NaN/Inf 未被清理，导致传入 Granger 计算时出现 SVD 错误

**修复内容**：
1. **先合并所有层风险指标**（不做过滤）到 `layer_data_raw`
2. **找共同日期**：收益率和所有风险层的日期交集
3. **统一过滤**：所有数据过滤到共同日期
4. **联合清理**：`valid_mask` 同时检查收益率**和**风险指标的 NaN/Inf
5. **统一应用**：`R`、`dates`、`layer_data` 使用同样的 `valid_mask`

**修改文件**：
- `src/model/multilayer_network_mle_integrated.py` — `load_data_with_risk_measures()` 函数（约第520-560行）

**下次优先**：
重新运行集成脚本，验证 SVD 错误已修复


---

## 会话 #28续3 — 2026-05-05

**主要工作**：澄清 `step` 参数的工作机制

**用户疑问**：代码中是否存在两个 step 参数需要配合？

**解答**：
- 代码中**只有一个** `config.step` 参数
- 它控制**滚动估计的频率**（每隔多少天进行一次完整的 MLE+Granger 估计）
- 关键代码：`eval_indices = list(range(first_valid_idx, n_periods, step))`

**工作机制**（以 step=5 为例）：
- 评估点：`[251, 256, 261, 266, ...]` 每隔 5 天
- 每个评估点 `t`：MLE 和 Granger 都使用 `[t-window+1, t]` 的数据
- 每个窗口独立计算 Granger 网络，然后立即进行 MLE 估计

**与旧代码的区别**：
- 旧代码：`rolling_loo_granger.py`（预计算网络，step≈7）→ `multilayer_network_mle_*.py`（读取网络，step=5）
- 新代码：集成脚本中 step 统一控制，现场计算 Granger 网络，无需跨文件配合

**下次优先**：
重新运行集成脚本，验证修复后的代码正常工作

---

## 会话 #29 — 2026-05-06

**主要工作**：创建模拟测试环境验证MLE代码正确性，发现PCA因子内生性和delta弱识别问题，绘制delta演化图

**产出文件**：
- `simulation/data/generate_data.py` — 5行业合成数据（收益率/波动率/CAViaR，T=1500）
- `simulation/model/run_simulation.py` — 顺序MLE实时输出脚本
- `simulation/model/oracle_test.py` — Oracle测试（真实网络直接输入MLE）
- `simulation/results/sim_results.csv` — 25窗口MLE结果
- `simulation/results/delta_evolution_simulation.png` — Delta走势图

**关键发现**：
1. **MLE代码无bug**：100%收敛，delta_sum=1.0，数值正确
2. **PCA因子内生性**：PCA从R=A^{-1}(βF+η)提取被网络效应污染→rho系统偏误；Bonaccolto原文用外生FF4因子
3. **Delta弱识别**：网络层相关性(corr=0.62)导致delta趋向1/3而非真实值；但组合W*可恢复(Frobenius误差0.113)
4. **Granger密度7-8%**：与Bonaccolto Table 1一致，属正常范围
5. **Rho需边界**：后期窗口出现±15极端值

**下次优先**：替换PCA为外生因子（中国FF因子）→ 添加rho边界[-5,5] → 运行真实数据完整MLE

---

## 会话 #30 — 2026-05-06

**主要工作**：集成CSMAR Fama-French 3因子（外生）替代PCA（内生），验证rho估计改善

**产出更新**：
- `src/model/multilayer_network_mle_integrated.py` — 5处修改：
  - `_BASE` 自动检测路径（支持两台电脑）
  - `NetworkConfig` 新增 `ff3_path`, `ff3_market_id`
  - 新增 `load_ff3_factors()` 函数
  - `construct_factors()` 新增 `method="ff3"` 分支
  - `main_integrated()` FF3日期对齐逻辑

**关键验证结果**（同一窗口PCA vs FF3）：
- rho范围：PCA [-53.1, 15.2] → FF3 [-0.64, 5.1]
- loglik：PCA 11387.7 → FF3 11675.6（+288）
- **PCA内生性问题已确认并修复**

**当前状态**：
- 已解决：PCA因子内生性（→FF3外生因子）
- 未解决：网络稀疏性（jsv_pos密度3.3%）、rho需边界约束[-5,5]、delta弱识别

**下次优先**：添加rho边界[-5,5] → 解决网络稀疏性 → 运行完整滚动MLE

---

## 会话 #31 — 2026-05-06

**主要工作**：创建 `multilayer_network_mle_integrated_Quantile.py`，用非参数分位数因果检验（Jeong et al. 2012）替代LOO条件Granger因果构造K×K网络

**产出文件**：
- `src/model/multilayer_network_mle_integrated_Quantile.py`（新建，~780行）— 分位数因果版MLE集成脚本

**关键修改**：
1. 删除LOO Granger 4个函数（~120行），替换为 `from np_quantile_causality import qn_causality_network`
2. NetworkConfig：删除 `granger_lag`/`granger_alpha`，新增 `quantile_tau=0.05`/`quantile_alpha=0.05`/`quantile_type="mean"`，重命名 `granger_window` → `network_window`
3. Rolling worker网络调用从 `loo_conditional_granger_networks()` → `qn_causality_network()`
4. 所有函数名和输出文件名从 `*_integrated` → `*_quantile`
5. 修复 `__file__` NameError：`try/except NameError` 模式支持Jupyter环境

**关键发现**：
- `src/model/np_quantile_causality.py` 已有完整的Jeong et al. (2012)测试Python实现，无需从R重新移植
- 分位数因果网络预期密度50-70%（vs LOO Granger的3-8%），可有效解决网络稀疏性问题
- 计算量更大：每窗口~45,000次LP求解（K=10），但joblib并行可缓解

**下次优先**：运行分位数因果版MLE验证网络密度 → 对比两种网络方法 → 添加rho边界[-5,5] → 完整滚动MLE

