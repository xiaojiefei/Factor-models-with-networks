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
| **项目状态** | 🟡 已创建分位数因果版MLE脚本（替代LOO Granger），预期网络密度从3-8%提升至50-70%；当前重点：运行验证、添加rho边界[-5,5]、完整滚动MLE |
| **项目申报书** | 📝 立项依据+文献评述已精修，预期成果/阶段/参考文献待填 |
| **最后更新** | 2026-05-06（会话#31：创建分位数因果版MLE集成脚本，替代LOO Granger网络构造）|

---

## 🎯 当前任务

- **当前阶段**：已创建分位数因果版MLE集成脚本 `multilayer_network_mle_integrated_Quantile.py`，用Jeong et al. (2012)非参数分位数因果检验替代LOO条件Granger来构造K×K网络。预期网络密度从3-8%提升至50-70%，解决网络稀疏性问题。FF3外生因子已集成。当前重点：(1)运行分位数因果版验证网络密度；(2)添加rho数值边界[-5,5]；(3)运行完整滚动MLE。
- **已完成代码**：
  - `src/model/multilayer_network_mle_integrated.py` — **LOO Granger版主脚本**，集成Granger网络+MLE+FF3因子
  - `src/model/multilayer_network_mle_integrated_Quantile.py` — **分位数因果版主脚本（新）**，替代LOO Granger，使用 `np_quantile_causality.py`
  - `src/model/np_quantile_causality.py` — 非参数分位数因果检验Python实现（Jeong et al. 2012完整移植）
  - `src/model/multilayer_network_mle.py` — 早期版本（基于预计算DY网络）
  - `src/model/multilayer_network_mle_Multithreading.py` — 早期多进程版本（基于预计算DY网络）
  - `src/model/multilayer_network_mle_static_window.py` — 早期static-window版本（基于预计算DY网络）
  - `src/model/lm_har.py` — LM跳跃检验模块化重写（457行，9个函数）
  - `src/model/run_lm_batch.py` — LM批量运行脚本（10个行业）
  - `src/model/rolling_loo_granger.py` — 滚动LOO条件Granger因果网络（预计算版本，当前被集成脚本替代）
  - `src/model/RollingVAR-DY.R` — 滚动窗口VAR DY溢出（R版本，当前被Python Granger替代）
  - `src/model/TVP-VAR-DY.R` — TVP-VAR DY溢出网络（R版本，当前未使用）
  - `src/model/network_utils.py` — 网络工具函数
  - `src/data/prepare_data.py` — Wind数据预处理
- **数据准备**：
  - 10个行业日度5分钟高频数据 → LM跳跃检验 → `data/raw/lm_results/`（4167日×21列）
  - **FF3因子**：CSMAR `STK_MKT_THRFACDAY.csv`，P9709（全部A股），8508交易日(1991-2026)，与收益率数据4166天完全对齐
  - 因子模型收益率来自 `{code}_lm_har.csv` 的 `r_now` 列
  - 行业代码000032-000041 = 上证行业指数
- **下一步行动**：
  1. 运行 `multilayer_network_mle_integrated_Quantile.py` 单窗口验证分位数因果网络密度
  2. 添加rho数值边界[-5,5]防止个别窗口发散（两个版本均需）
  3. 运行完整滚动MLE（FF3因子，4166期，step=5）
  4. 对比分位数因果与LOO Granger的delta/rho估计差异
  5. Delta弱识别处理：报告composite W*而非单独delta

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
- [x] **2026-04-16**：项目申报书全面诊断，研究方案大幅扩充（目标/难点/3模块完善）
- [x] **2026-04-16**：立项依据修改（多冲击视角+Gap总结）
- [x] **2026-04-17**：文献评述精修（风险vs韧性关联互补、多冲击差异性、韧性关联强度驱动区分）
- [x] **2026-04-17**：全文校对（引用格式修正Adrian/Brunnermeier、错别字拓扑、缺失括号等）
- [x] **2026-04-17**：开头段精简（去掉金融端/非金融端分类、加入Brunnermeier 2024、补逻辑桥梁）
- [x] **2026-04-17**：参考文献目录初步建立（7篇，正文引用30+篇待补）
- [x] **2026-04-18**：Billio (2023) 精读补充推导（Eq.2来源、减均值原因、Eq.4来源）
- [x] **2026-04-20**：研究创新方向头脑风暴（11个方向 + OVB分析，保存至 `memory/research-ideas.md`）
- [x] **2026-04-20-21**：§4 Model Estimation 逐段精讲（§6全面重写，含识别条件/假设4.1-4.3/似然函数/集中技巧/特征值补充）
- [x] **2026-04-21**：Billio (2023) §6 I/O表解读，创建 `memory/billio2023-io-as-adjacency.md`
- [x] **2026-04-21**：补充 §2.3.3 外生暴露(β̄ᵢ) vs 内生暴露(ρ) 详解
- [x] **2026-04-21**：补充 §3.1.1 β*=(I-ρW)⁻¹β̄ 非线性函数解释（Neumann级数展开）
- [x] **2026-04-21**：补充 §5.4 同期性问题（Billio用I/O表 vs Bonaccolto用频率错配技巧）
- [x] **2026-04-21**：补充 §3.5 网络效应三个深层后果（行业因子伪装公共因子、独立冲击变相关、LeSage-Pace分解不适用）
- [x] **2026-04-21**：§4 风险分解与分散化逐步推导扩充（Eq.16推导、2×2手算验证、Sherman-Morrison求逆完全图、组合特质风险公式、极限行为）
- [x] **2026-04-21**：校正Eq.(16)符号与论文原文一致（$A=\Omega=(I-\rho W)^{-1}$, $\sigma_m^2$, $\Omega_\eta$）
- [x] **2026-04-22**：§5.1.1 负ρ削弱系统性风险（Neumann级数交替相消 + amplifier/absorber/insulator分类）
- [x] **2026-04-22**：§5.1.2 重写为双渠道结构（量变=β*放大 vs 质变=特质风险不可分散），补充Markowitz/Cochrane定义
- [x] **2026-04-22**：§6.6 补充η_t身份澄清（结构残差vs约化残差对比表）
- [x] **2026-04-22**：§6.7.1 新增数据结构具象化（K=31, T=52维度展开 + Python代码对应）
- [x] **2026-04-24**：重写 `LM+HAR.py` → `lm_har.py`（模块化，修复5个bug，向量化6个循环）
- [x] **2026-04-24**：编写 `run_lm_batch.py` 批量运行10个行业LM跳跃检验（000032-000041）
- [x] **2026-04-24**：运行批量LM检验，生成 `data/raw/lm_results/`（10个文件，4167日×21列）
- [x] **2026-04-24**：编写 `RollingVAR-DY.R` 滚动窗口VAR DY溢出网络（3层：连续/正跳/负跳）
- [x] **2026-04-24**：修复CSVt_d中Inf值问题（20171130全bar跳跃→meanvol=Inf）
- [x] **2026-04-24**：运行DY溢出，生成3层10×10网络（jsv_pos/jsv_neg已完成，csv待重跑）
- [x] **2026-04-24**：诊断δ角点解问题（10个自由λ_i让δ不可识别，13参数→似然面平坦）
- [x] **2026-04-24**：实现两步法MLE（Step1标量ρ→δ̂, Step2固定δ̂→异质性λ_i）
- [x] **2026-04-24**：修正行业名称（000032-000041 = 上证行业指数，非申万行业）
- [x] **2026-04-24**：切换因子方法 equal_weight → PCA第一主成分
- [x] **2026-04-24**：实现Max Row Normalization（论文Eq.25，Python端max_row_normalize）
- [x] **2026-04-24**：R代码改为导出原始矩阵（RollingVAR-DY.R + TVP-VAR-DY.R 去掉行标准化）
- [x] **2026-04-24**：TVP-VAR-DY.R 适配10行业日度跳跃数据（tvp_前缀输出）
- [x] **2026-04-25**：严格按Billio §4修改 `multilayer_network_mle.py`：窗口内逐期使用时变 $A_t$，scalar ρ 对窗口内所有 $W_t^*$ 取交集特征值约束，异质 λ_i 对窗口内所有 $W_t^*$ 做稳定性检查，通过 `python -m py_compile` 验证
- [x] **2026-04-25**：严格按Bonaccolto §2将MLE从两步法改为one-step constrained concentrated MLE，联合估计 $\delta_j$ 与异质性 $\rho_i$，删除核心输出中的 `rho_scalar`/`step1_success`/`step2_success`/`weak_identification`
- [x] **2026-04-25**：新增 `src/model/multilayer_network_mle_Multithreading.py` 多进程版本，先用ProcessPoolExecutor并行滚动窗口，后改为贴近旧预测代码的 `joblib.Parallel(..., prefer="processes")` 写法并去掉自定义 `batch_size`
- [x] **2026-04-25**：澄清10行业日度因子模型收益率来源：`load_daily_returns_from_lm()` 从 `{code}_lm_har.csv` 读取 `r_now` 列；DY网络CSV为堆叠矩阵并按日期交集切片
- [x] **2026-04-28**：新增 `src/model/multilayer_network_mle_static_window.py`，作为Bonaccolto-style窗口内固定网络benchmark：252日窗口、5日步长、窗口内原始W取均值后行标准化、固定A似然使用 $T\log|A|$
- [x] **2026-04-28**：完成static-window似然与Bonaccolto原文PDF核对：`dirichlet_alpha=1.0` 时为纯集中MLE，固定A使用 $T\log|A|$，允许负异质性 $\rho_i$，有限rho边界只能作为数值优化设定
- [x] **2026-04-28**：完成单窗口敏感性初步诊断：无界/宽rho可导致极大 $\rho$，rho边界变化会改变 $\delta$ 与 $\rho$ 符号，PCA因子可明显缓解角点并给出更均衡的网络权重
- [x] **2026-05-06**：创建模拟测试环境（simulation/），5行业合成数据（收益率/波动率/CAViaR），验证MLE代码无bug，发现PCA内生性和delta弱识别
- [x] **2026-05-06**：集成CSMAR FF3外生因子（RiskPremium1/SMB1/HML1），替代PCA因子，rho从[-53,15]→[-0.64,5.1]，loglik+288
- [x] **2026-05-06**：创建 `multilayer_network_mle_integrated_Quantile.py`，用非参数分位数因果（Jeong et al. 2012）替代LOO Granger网络构造
- [x] **2026-05-06**：发现并复用已有的 `np_quantile_causality.py`（354行，完整Jeong et al. 2012 Python移植）
- [x] **2026-05-06**：修复 `__file__` NameError（Jupyter/交互环境兼容，`try/except NameError` 模式）

---

## 🔲 待完成（按优先级）

- [ ] 运行 `multilayer_network_mle_integrated_Quantile.py` 验证分位数因果网络密度（预期50-70%）
- [ ] 添加rho数值边界限制：`[-5,5]`（防止估计发散，两个版本均需）
- [ ] 运行完整滚动MLE（FF3因子，4166期，step=5）
- [ ] 对比分位数因果与LOO Granger的delta/rho估计差异
- [ ] Delta弱识别处理：报告composite W*而非单独delta
- [ ] 分析 $\delta(t)$ 时间演化、异质性 $\rho_i(t)$ 和中心性变化
- [ ] 项目申报书：填写预期研究成果、研究阶段时间表、参考文献目录
- [ ] 删除旧文件 `static_mle_demo.py`（确认新文件正常后）
- [ ] 实现其他因果网络估计方法（GR, QB, Qo, QN）作为对比
- [ ] **长期**：评估是否迁移到GARCHSK+非线性Granger以匹配论文方法

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
│   │   ├── multilayer_network_mle_integrated.py ← LOO Granger版集成MLE（FF3因子）
│   │   ├── multilayer_network_mle_integrated_Quantile.py ← **分位数因果版集成MLE（新）**
│   │   ├── np_quantile_causality.py ← 非参数分位数因果检验（Jeong et al. 2012）
│   │   ├── multilayer_network_mle.py ← 早期版本（基于预计算DY网络）
│   │   ├── multilayer_network_mle_Multithreading.py ← 早期多进程版本
│   │   ├── multilayer_network_mle_static_window.py ← 早期static-window版本
│   │   ├── lm_har.py            ← LM跳跃检验模块化（457行，9个函数）
│   │   ├── run_lm_batch.py      ← LM批量运行（10个行业）
│   │   ├── RollingVAR-DY.R      ← 滚动窗口VAR DY溢出（3层，10行业日度）
│   │   ├── TVP-VAR-DY.R         ← TVP-VAR DY溢出网络（31行业周度）
│   │   ├── LM+HAR.py            ← 旧LM代码（保留参考，不可import）
│   │   ├── static_mle_demo.py   ← 旧文件（待删除）
│   │   ├── network_utils.py     ← 网络工具函数
│   │   └── __init__.py
│   ├── data/
│   │   ├── prepare_data.py      ← Wind数据预处理（已完成）
│   │   ├── wind下载/            ← 原始Wind行业Excel（31个）
│   │   └── processed/           ← 处理后数据（31行业周度）
├── data/
│   └── raw/
│       ├── min5_data/           ← 5分钟高频数据（10个行业）
│       ├── daily_data/          ← 日度数据（10个行业）
│       ├── FF3/                 ← CSMAR Fama-French 3因子（外生）
│       │   └── STK_MKT_THRFACDAY.csv  ← P9709全部A股(8508日,1991-2026)
│       └── lm_results/          ← LM跳跃检验结果+DY网络
│           ├── {code}_lm_har.csv × 10  ← LM检验结果（4167日×21列）
│           ├── dy_all_csv.csv          ← 连续波动率DY网络（待重跑）
│           ├── dy_all_jsv_pos.csv      ← 正跳波动率DY网络（39710行×10列）
│           ├── dy_all_jsv_neg.csv      ← 负跳波动率DY网络（39710行×10列）
│           └── dy_dates_*.csv          ← 日期映射
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
| $\delta_j$ | 网络层组合权重，满足 $\delta_j\ge0$ 且 $\sum_j\delta_j=1$ |
| $\rho_i$ | Bonaccolto异质性网络暴露参数；代码输出中的 `lambda_hat`/`lambda_i` 为兼容旧列名，经济含义等同于 $\rho_i$ |
| Max标准化(Eq.25) | 时变W用max normalization保留网络密度变化，非标准行标准化 |
| Static-window W | 窗口内固定W版本不再用全样本max normalization；每个252日窗口内先对原始 $W_{j,t}$ 取均值，再对 $\bar W_j$ 行标准化 |
| 时变网络MLE | 窗口内每期使用 $A_t=I-\operatorname{diag}(\rho)W_t^*$；对所有 $W_t^*$ 同时施加Billio §4稳定性约束 |
| One-step MLE | 严格按Bonaccolto §2，在一个约束优化中联合估计 $(\delta, \rho)$，而不是先估δ再固定δ估ρ |
| Dirichlet alpha | δ的可选正则项；`alpha=1.0`为纯MLE，`alpha>1`轻微抑制角点解 |
| rho数值边界 | Bonaccolto不要求异质性 $\rho_i\ge0$；有限对称边界（如 `[-5,5]`）只能作为数值优化稳定化设定，必须做敏感性分析 |
| 因子设定 | **PCA因子有内生性问题**：从R=A^{-1}(βF+η)提取PCA被网络效应A^{-1}污染，导致rho系统偏误。Bonaccolto原文用外生Fama-French 4因子。当前使用CSMAR FF3外生因子（P9709全部A股），rho从[-53,15]→[-0.64,5.1] |
| Delta弱识别 | 网络层相关性(corr≈0.62)导致delta趋向1/3；但composite W*可恢复(Frobenius误差0.113)。结构性限制，非代码bug |
| 分位数因果 | Jeong et al. (2012)非参数检验，测试x_{t-1}是否在分位数tau处Granger-cause y_t。统计量渐近N(0,1)。预期网络密度50-70%（vs LOO Granger的3-8%），解决稀疏性 |

### 4. 已完成代码文件

- `src/model/multilayer_network_mle_integrated.py` — **LOO Granger版**：集成Granger网络+MLE+FF3因子，支持 `factor_method="ff3"/"pca"/"equal_weight"`
- `src/model/multilayer_network_mle_integrated_Quantile.py` — **分位数因果版（新）**：用 `qn_causality_network()` 替代LOO Granger，参数为 `quantile_tau/quantile_alpha/quantile_type`
- `src/model/np_quantile_causality.py` — 非参数分位数因果检验完整Python实现（Jeong et al. 2012），含 `qn_causality_network()` K×K邻接矩阵构建
- `src/model/multilayer_network_mle.py` — 早期版本（基于预计算DY网络）
- `src/model/network_utils.py` — 网络工具函数
- `src/data/prepare_data.py` — Wind数据预处理（周度重采样+GARCH波动率，248行）

### 5. 关键发现与问题

**网络稀疏性问题（2026-05-05发现）**：
- Granger因果网络存在严重的稀疏性：JSV_POS/NEG层约92%零元素，45-52%全零行
- 前5个时间点中4个为完全全零矩阵，导致MLE中delta参数不可识别
- 原因：跳跃波动率本身稀疏（93%+零值）+ Granger检验显著性水平5%过严
- 解决方案：放宽alpha至0.10-0.20，或改用分位数因果方法（QN密度可达50-70%）

**565窗口MLE运行结果**：
- 收敛率100%，delta均值相对均衡（0.32-0.35）
- 但rho仍有极端值（-14.029, 16.916），需添加数值边界限制（如[-5,5]）

**脚本参数对应**：
- Granger生成网络实际间隔约7天，MLE step=5不完全对应，需统一配置

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

最后更新：2026-05-06（会话#31：创建分位数因果版MLE集成脚本，替代LOO Granger网络构造，修复__file__ NameError）
