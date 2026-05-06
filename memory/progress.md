# 详细进度日志

> 每次会话结束时在末尾追加新条目，格式统一。

---

## 2026-04-16 — 会话 #9：项目申报书全面诊断与修改

### 本次完成
- **项目申报书全面诊断**（`项目/附件2：...申报书.docx`）：
  - 读取Word文档所有表格，逐部分分析完成度和问题
  - 按P0/P1/P2三级优先级输出修改建议
- **研究方案（五）大幅修改**：
  - 主要目标#3：补全（驱动因素+传导机制）
  - 难点#3：新增（驱动因素分离+内生性控制）
  - 模块1：补充地缘政治冲击变量（GPR指数）、增加具体事件（2008危机、COVID-19等）
  - 模块2第三步：多层耦合分析从1句话扩充为4个子步骤（多层构建→耦合模型→参数估计→中心性），无公式版
  - 模块3：从1步扩充到2步（NAR驱动因素分析 + 异质性与政策建议）
  - 框架思路描述修改：匹配标题"多种外部冲击"视角
- **立项依据（三）修改**：
  - 开头段重写：从"气候政策不确定性"聚焦改为"多种外部冲击叠加"视角
  - 新增"4.文献评述"段落：归纳4条现有研究不足
- **输出但用户尚未填入**：
  - 预期研究成果建议（2篇论文+1次会议+1个工具包）
  - 研究阶段时间表建议（6阶段，2026.05-2027.12）
  - 数据说明+可行性分析建议文本
  - 参考文献目录框架

### 下次继续
- [ ] 填写预期研究成果+研究阶段+参考文献目录
- [ ] 运行R代码生成周度DY网络
- [ ] 运行 `multilayer_network_mle.py` 滚动MLE
- [ ] 分析结果：δ(t)时间演化、中心性变化

### 当前阻塞
- 无

---

## 2026-04-17 — 会话 #10：申报书立项依据+文献评述精修

### 本次完成
- **文献评述3段修改**：
  - 第一段（研究视角）：删除"韧性不是风险的简单反面"，改为阐述风险关联vs韧性关联的差异与互补关系
  - 第二段（冲击来源）：删除"交叉叠加、同时作用"（暗示联合冲击），改为强调行业面对多种冲击时韧性差异
  - 第四段（驱动因素）：区分"韧性水平的驱动因素"（现有研究）vs"韧性关联强度的驱动因素"（本项目），用"然而"转折自然对比
- **全文校对**（发现9个问题）：
  - 引用姓名错误："Tobias and Brunnermeier" → "Adrian和Brunnermeier"（2处）
  - 括号缺失："Wei and Zhou, 2024a)" → "(Wei和Zhou，2024a)"
  - 引用格式不统一："Tang et al." → "Tang等"（统一用中文"等"）
  - 错别字："拓结构" → "拓扑结构"
  - 文献评述第四段末总结句缺失（"本课题旨在弥补上述不足..."）
  - 文献综述引言方向名与章节标题不匹配（信息关联 vs 关联效应）
  - Brunnermeier(2024)在参考文献列表但正文未引用
  - Yang等(2025)两次引用对应不同主题需区分
  - 语句不流畅："的反应的" 双"的"堆叠
- **开头段精简**：
  - 删除"金融端/非金融端"分类框架，用自然语言带过冲击类型
  - 加入逻辑桥梁："行业间关联网络为冲击的跨行业传导与放大提供了渠道"
  - 嵌入Brunnermeier(2024)引用："韧性已成为宏观金融研究的核心概念"
  - 修复"金融市场"与"金融尾部风险"中"金融"重复
- **反馈记忆保存**：用户要求所有建议必须先确保逻辑通顺（→ `feedback_logic_first.md`）

### 申报书当前完成状态
| 部分 | 状态 |
|------|------|
| 三、立项依据（开头+意义） | ✅ 精修完成 |
| 三、文献综述（2节） | ✅ 内容完成（格式小问题待修） |
| 三、文献评述（4段） | ✅ 精修完成 |
| 三、参考文献目录 | ⚠️ 仅7篇（需30+篇） |
| 五、研究方案 | ✅ 已完成 |
| 六、预期研究成果 | ❌ 空白 |
| 研究阶段时间表 | ❌ 空白 |

### 下次继续
- [ ] 补全参考文献目录（正文引用30+篇）
- [ ] 填写预期研究成果
- [ ] 填写研究阶段时间表
- [ ] 运行R代码生成周度DY网络
- [ ] 运行Python滚动MLE

### 当前阻塞
- 无

---

## 2026-04-18 — 会话 #11：Billio (2023) 精读补充推导

### 本次完成
- **`memory/billio2023-derivation.md` 补充3个小节**：
  - §1.2.1 Eq.(2) 的来源：从联立方程 (★) 出发推导，用 3×3 矩阵展开展示 A 矩阵结构（对角线=1，非对角=-a_ij）
  - §1.2.3 为什么减去均值 E[R_t]：网络传导的是"意外波动"而非正常收益水平，实操中等价于加截距项 ᾱ
  - §2.3.1 Eq.(4) 的来源：核心假设 a_ij = ρ·W_ij，将 K(K-1)=870 个自由参数压缩为 1 个 ρ
- **概念澄清（对话式Q&A）**：
  - Eq.(2) 是建模假设，不是从 Eq.(1) 的数学推导
  - (I-ρW) 结构是"反馈系统移项"的数学必然（类比 Leontief、PageRank）
  - 论文最终估计的参数：ρ_i（数值优化）+ ᾱ, β̄, Ω_η（OLS 解析解）+ λ（Fama-MacBeth 风险溢价）

### 下次继续
- [ ] 继续论文精读（如有更多疑问）
- [ ] 补全参考文献目录（30+篇）
- [ ] 运行R代码生成周度DY网络
- [ ] 运行Python滚动MLE

### 当前阻塞
- 无

---

### 本次完成
- **概念澄清（纯Q&A，无代码修改）**：
  - δ_j的经济含义：网络层混合权重，凸组合系数，反映哪种网络视角更能解释同期关联
  - 金融传染的本质：冲击的传导（非相关性），通过收益率溢出(W1)和波动率溢出(W2)两渠道
  - 邻接矩阵的一般性：不限于风险相关，任何满足A1-A4假设的关系矩阵均可（投入产出表、地理邻近等）
  - ρ_i vs δ_j区分：ρ_i索引资产（易感性），δ_j索引网络层（方法重要性）

### 下次继续
- [ ] 运行R+Python管线
- [ ] 项目书修改

### 当前阻塞
- 无

---

## 2026-04-20 — 会话 #12：研究创新方向头脑风暴

### 本次完成
- **`memory/research-ideas.md` 创建**：
  - 基于 Bonaccolto (2019) 框架的 11 个创新方向
  - 方向一（⭐⭐⭐⭐⭐）：混合金融+实体网络（加入投入产出表作为 $W_3$）
  - 方向二（⭐⭐⭐⭐⭐）：$\delta_j(t)$ 驱动因素回归（VIX、GPR、MPU等宏观变量）
  - 方向三（⭐⭐⭐⭐）：韧性视角重构（从风险传染翻转为韧性分析）
  - 方向四（⭐⭐⭐）：非对称网络效应（外生体制切换方案最可行）
  - 方向五（⭐⭐⭐）：样本外预测+组合构建
  - 方向六（⭐⭐⭐⭐）：贝叶斯估计替代MLE
  - 方向七（⭐⭐⭐⭐）：状态空间模型替代滚动窗口
  - 方向八（⭐⭐⭐⭐）：尾部相依网络（CoVaR / Copula）
  - 方向九（⭐⭐⭐⭐⭐，必做）：嵌套模型检验（LR test + AIC/BIC）
  - 方向十（⭐⭐⭐⭐⭐，必做）：网络内生性处理（$W_{t-1}$ 滞后网络）
  - 方向十一（⭐⭐⭐）：多重网络度量（参与系数、层间度相关、社区检测）
  - OVB（遗漏变量偏误）分析：遗漏的 $W_3$ 去了哪里
  - DY 溢出网络作为 W 的合理性论证（3个经济机制）
- **概念澄清**：
  - 非对称效应的根本困难：$(I-\rho W)^{-1}$ 闭式解丢失
  - 为什么不能去掉 $\rho$：$W$ 行标准化 → $\lambda_{max}=1$ → $\det(I-W)=0$
  - DY 溢出 vs 因果检验的区别：稠密 vs 稀疏

### 下次继续
- [ ] 运行R+Python管线
- [ ] 项目书修改

### 当前阻塞
- 无

---

## 2026-04-21 — 会话 #15：网络效应三个深层后果

### 本次完成
- **`memory/billio2023-derivation.md` 新增 §3.5**：
  - §3.5.1：行业特有因子通过 $(I-\rho W)^{-1}$ "伪装"成公共因子——油价只直接影响能源，但约化形式下所有行业都有非零 beta，忽略网络会错误归因
  - §3.5.2：结构独立的特质冲击通过网络变得相关——$\varepsilon^* = (I-\rho W)^{-1}\eta$ 把独立冲击搅混，解释了 Table 9 残差相关从 0.10 降到 0.03
  - §3.5.3：LeSage-Pace 直接/间接效应分解不适用——因子 $F_t$ 是公共变量（所有资产共享），不满足空间计量要求的"个体特有解释变量"条件

### 下次继续
- [ ] 运行R代码生成周度DY网络
- [ ] 运行Python滚动MLE
- [ ] 项目书修改

### 当前阻塞
- 无

---

## 2026-04-21 — 会话 #16：§2.4 风险分解与分散化逐步推导

### 本次完成
- **`memory/billio2023-derivation.md` §4 全面重写扩充**（~50行→~170行）：
  - §4.0：起点回顾（约化形式 Eq.8）+ 符号说明（论文 $A$ = 笔记 $\Omega = (I-\rho W)^{-1}$，论文 $\sigma_m^2$ = 笔记 $\Sigma_F$，论文 $\Omega_\eta$ = 笔记 $\Sigma_\eta$）

---

## 2026-04-22 — 会话 #17：§5.1 系统性风险与 ρ 的关系 + §6.6-6.7 数据结构补充

### 本次完成
- **`memory/billio2023-derivation.md` §5.1 新增3个小节**：
  - §5.1.1：为什么 $\rho_i < 0$ 会削弱系统性风险（结构方程直觉、Neumann 级数交替相消、制造业数值例子、避风港经济含义、amplifier/absorber/insulator 三分类）
  - §5.1.2 重写：$\rho$ 与系统性风险的直接联系——特质风险渠道
    - 第一版以 $\beta^*$ 为中心（用户纠正：不对，应该是特质风险）
    - 第二版改为双渠道结构：渠道一（因子暴露放大=量变）+ 渠道二（特质风险不可分散=质变⭐）
    - 补充论文 §2.4 对 systematic risk 的定义（Markowitz 1952, Sharpe 1964, Cochrane 2011）
    - Cochrane 的呼吁 → 本论文回答：网络连接性是 systematic risk 的 determinant
- **`memory/billio2023-derivation.md` §6.6 补充**：
  - $\eta_t$ 身份澄清：结构方程残差（$\Omega_\eta$ 对角）vs 约化形式残差 $\varepsilon_t^*$（非对角），含对比表
- **`memory/billio2023-derivation.md` §6.7.1 新增**：
  - 数据结构具象化（K=31, T=52, n_f=1）：$Z_t$、$\hat{\eta}_t$、$\hat{\bar{\alpha}}$、$\hat{\bar{\beta}}$ 的维度表
  - 52×31 残差矩阵展开、沿列求 $\hat{\sigma}_i^2$、31×31 对角 $\hat{\Omega}$ 矩阵展开
  - 4 行 Python 代码与数学公式的一一对应
- **概念澄清（对话式Q&A）**：
  - 代码中 $E[R_t]$ 如何计算：OLS 截距项吸收，等价于先 demean
  - $\eta_t$ 是结构方程残差（过滤掉网络效应后的纯特质冲击）
  - 系统性风险的本质改变是特质风险从可分散变成不可分散（质变），不是 $\beta^*$ 放大（量变）

### 用户反馈
- §5.1.2 第一版以 $\beta^*$ 为中心，用户纠正："不是应该针对特质风险吗？特质风险才是不能被分散的"→ 已修正为特质风险渠道为核心

### 下次继续
- [ ] 运行R代码生成周度DY网络
- [ ] 运行Python滚动MLE
- [ ] 项目书修改（预期成果/阶段/参考文献）

### 当前阻塞
- 无

---

## 2026-04-24 — 会话 #18-19：LM跳跃检验批量运行 + 3层DY溢出网络

### 本次完成

- **重写 `LM+HAR.py` → `lm_har.py`**（457行，9个函数）：
  - 修复5个关键bug、向量化6个循环、统一日期解析

- **编写 `run_lm_batch.py` + 批量运行**：
  - 10个行业全部成功（000032-000041），输出至 `data/raw/lm_results/`

- **编写 `RollingVAR-DY.R` + 运行DY溢出**：
  - 3层网络：CSVt_d / JSVt_zheng_d / JSVt_fu_d，参数nlag=1, nfore=5, window.size=200
  - jsv_pos + jsv_neg 成功（39710×10），csv 待重跑（Inf已修复）

### 下次继续

- [ ] 重跑CSVt_d的DY溢出
- [ ] 适配MLE代码读取新3层网络
- [ ] 运行滚动MLE

### 当前阻塞
- 无

---

## 2026-04-24 — 会话 #20：MLE诊断修复 + Max标准化 + 两步法

### 本次完成

- **诊断δ角点解问题**：
  - 原因：10个自由λ_i补偿任意δ → 似然面在δ方向平坦 → SLSQP打到0/1边界
  - 验证：3层网络相关性0.37-0.90，高相关时δ不可区分
  - 参数过多：3δ+10λ=13参数

- **实现两步法MLE** (`estimate_single_window`)：
  - Step 1：标量ρ + multi-start → 可识别δ̂（4优化变量）
  - Step 2：固定δ̂，L-BFGS-B估计异质性λ_i（10参数）
  - 新增 `_step2_loglik` 目标函数

- **修正行业名称**（000032-000041 = 上证行业指数）：
  - 旧（错误）：交通运输/传媒/公用事业/农林牧渔/医药生物/商贸零售/国防军工/基础化工/家用电器/建筑材料
  - 新（正确）：能源/原材料/工业/可选消费/主要消费/医药卫生/金融地产/信息技术/通信服务/公用事业

- **切换因子方法**：equal_weight → PCA第一主成分

- **实现Max Row Normalization**（论文Eq.25）：
  - Python: `max_row_normalize()` — 每列除以该列跨所有时间点的最大列和
  - R代码: RollingVAR-DY.R + TVP-VAR-DY.R 改为导出原始矩阵（仅diag=0，不做行标准化）

- **修改TVP-VAR-DY.R**适配跳跃数据（10行业日度，tvp_前缀输出）

- **concentrated_loglik改进**：支持scalar_rho/vector_rho + Dirichlet先验 + slogdet

### 修改文件
```text
src/model/multilayer_network_mle.py  — 两步法MLE + max标准化 + 行业名修正 + PCA因子
src/model/RollingVAR-DY.R           — 导出原始矩阵（去掉row_normalize）
src/model/TVP-VAR-DY.R              — 适配跳跃数据 + 导出原始矩阵
```

### 下次继续
- [ ] 重跑R代码（RollingVAR-DY.R）生成未标准化的3层网络
- [ ] 运行Python MLE（两步法）
- [ ] 分析δ(t)时间演化和λ_i异质性

### 当前阻塞
- 需要重跑R代码才能使用max normalization（当前dy_all_*.csv是旧的行标准化版本）

---

## 2026-04-25 — 会话 #21：Billio §4 时变网络MLE逻辑修正

### 本次完成

- **严格按 Billio §4 修正时变邻接矩阵逻辑**（`src/model/multilayer_network_mle.py`）：
  - 将 rolling window 内的网络输入从单一 endpoint `W_t` 改为窗口内完整序列 `W_\tau`。
  - `estimate_beta_sigma()` 支持 3D `A_seq`，逐期计算 $A_\tau R_\tau$。
  - `concentrated_loglik()` 使用 $\sum_\tau \log|A_\tau|$，不再用 endpoint 版本的 $T\log|A|$。
  - scalar $\rho$ 约束改为对窗口内所有 $W_\tau^*=\sum_j\delta_j W_{j,\tau}$ 取交集边界：下界取 max、上界取 min。
  - Step 2 异质 $\lambda_i$ 对窗口内所有 $W_\tau^*$ 检查 spectral radius 稳定性。
  - `rolling_estimation()` 和 single-point validation 均传入窗口内完整网络序列。

- **验证**：
  - 已运行 `python -m py_compile "src/model/multilayer_network_mle.py"`，语法检查通过。

### 修改文件
```text
src/model/multilayer_network_mle.py  — endpoint network approximation → fully time-varying window likelihood
CLAUDE.md                           — 更新项目状态、当前任务、方法总结
memory/progress.md                  — 追加本会话进度
memory/session-log.md               — 追加本会话摘要
memory/decisions.md                 — 记录采用 fully time-varying likelihood 的技术决策
```

### 下次继续
- [ ] 重跑R代码（RollingVAR-DY.R）生成未标准化的3层网络
- [ ] 用 `D:/Python/python.exe` 运行 `src/model/multilayer_network_mle.py`
- [ ] 检查 rolling_delta.csv 中 δ(t)、ρ(t)、λ_i(t)、step2_success、weak_identification
- [ ] 分析δ时间演化、λ_i异质性和中心性变化

### 当前阻塞
- 仍需重跑R代码生成未标准化 DY 网络；旧 `dy_all_*.csv` 是行标准化版本，不适合当前 max normalization 管线。

---

## 2026-04-25 — 会话 #22：Bonaccolto one-step MLE + 多进程滚动估计

### 本次完成

- **严格按 Bonaccolto et al. (2019) §2 修正估计流程**（`src/model/multilayer_network_mle.py`）：
  - 将上一版工程化两步法（Step1标量ρ识别δ，Step2固定δ估异质λ_i）改为文献中的 one-step constrained concentrated MLE。
  - 优化变量改为 $[\delta_1,\ldots,\delta_d,\rho_1,\ldots,\rho_K]$，在同一SLSQP问题中联合估计。
  - 保留约束 $\delta_j\in[0,1]$、$\sum_j\delta_j=1$，异质性 $\rho_i$ 使用非负边界并由窗口内稳定性检查控制可行域。
  - 输出中继续保留 `lambda_hat`/`lambda_i` 列名以兼容旧结果表，但其经济含义已改为 Bonaccolto 的异质性 $\rho_i$。
  - 不再将 `rho_scalar`、`step1_success`、`step2_success`、`weak_identification` 作为核心模型输出。

- **保留并统一 Billio §4 时变网络 likelihood**：
  - rolling window 内继续使用完整网络序列 $W_{j,\tau}$，而不是窗口末端 endpoint 网络。
  - 每期构造 $W_\tau^*=\sum_j\delta_j W_{j,\tau}$ 与 $A_\tau=I-\operatorname{diag}(\rho)W_\tau^*$。
  - 似然项使用 $\sum_\tau\log|A_\tau|$，并对窗口内所有 $W_\tau^*$ 同时做稳定性检查。

- **新增多进程滚动估计脚本**（`src/model/multilayer_network_mle_Multithreading.py`）：
  - 从单进程版复制并改造，使用 `ProcessPoolExecutor` 并行估计不同rolling windows。
  - 添加 `n_jobs` 与 `batch_size` 配置，默认 `n_jobs=None` 使用CPU核心数-1，`batch_size=4` 降低Windows spawn调度开销。
  - 添加 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 默认设为1，避免多进程与BLAS内部线程过度竞争。
  - 多进程版输出单独文件：`rolling_delta_multithreading.csv`、`rolling_centrality_multithreading.csv`、`delta_evolution_multithreading.png` 等，避免覆盖单进程结果。

- **数据与参数解释澄清**：
  - `window=252` 表示每次MLE使用252个交易日（约一年）；`step=1` 表示窗口每天向前滚动一次，因此3970个共同日期会产生3719个估计窗口。
  - DY网络CSV是按时间堆叠的矩阵：10行业时每10行对应一个 $10\times10$ 邻接矩阵，Python读取后reshape为 `(T, K, K)` 并按日期交集选取窗口切片。
  - `data/raw/lm_results/dy_all_csv.csv` 当前为3970期，而 `dy_all_jsv_pos.csv` / `dy_all_jsv_neg.csv` 为3971期；代码能按日期交集运行，但说明csv层可能是旧结果，仍建议重跑R。
  - `Dirichlet alpha` 是δ的可选正则项；`alpha=1.0` 等价纯MLE，`alpha>1` 轻微抑制角点解。
  - 10行业日度因子模型收益率来自 `load_daily_returns_from_lm()` 读取 `{code}_lm_har.csv` 的 `r_now` 列。

- **验证**：
  - `src/model/multilayer_network_mle.py` 已通过 `python -m py_compile`。
  - `src/model/multilayer_network_mle_Multithreading.py` 已通过 `python -m py_compile`。

### 修改文件
```text
src/model/multilayer_network_mle.py                 — 两步法 → Bonaccolto one-step MLE
src/model/multilayer_network_mle_Multithreading.py  — 新增多进程rolling window版本
CLAUDE.md                                           — 更新项目状态、当前任务、方法总结
memory/progress.md                                  — 追加本会话进度
memory/session-log.md                               — 追加本会话摘要
memory/decisions.md                                 — 记录one-step MLE技术决策
```

### 下次继续

- [ ] 重跑 `src/model/RollingVAR-DY.R`，生成未标准化且日期一致的3层DY网络。
- [ ] 用 `D:/Python/python.exe` 运行 `src/model/multilayer_network_mle_Multithreading.py`，必要时调整 `n_jobs` 和 `batch_size`。
- [ ] 检查 `rolling_delta_multithreading.csv` 中 $\delta(t)$ 与异质性 $\rho_i(t)$，尤其确认 `d_jsv_pos≈1` 是否持续存在。
- [ ] 若要严格纯Bonaccolto MLE，将 `dirichlet_alpha` 设为 `1.0` 后重跑对比。

### 当前阻塞

- 需要重跑R代码生成新的原始DY网络；当前3层网络日期长度不完全一致，且csv层可能仍是旧版本。

---

## 2026-04-25 — 会话 #23：joblib并行写法调整 + 集中MLE逻辑澄清

### 本次完成

- **多进程滚动估计脚本改为贴近旧预测代码写法**（`src/model/multilayer_network_mle_Multithreading.py`）：
  - 用户提供了一个旧项目中可正常调用CPU的 `joblib.Parallel(..., prefer="processes")` 并行预测脚本作为参考。
  - 将多进程实现从 `ProcessPoolExecutor` 改为 `joblib.Parallel` + `delayed`。
  - 进一步按用户要求“贴近原来的代码”，删除自定义 `NetworkConfig.batch_size`、手动 `task_batches` 和 `_estimate_window_batch()`。
  - 当前写法为每个 rolling window 对应一个 `delayed(_estimate_window_job)(...)`，由 joblib 自己管理任务分派粒度。
  - 保留 `_multithreading` 后缀输出文件，避免覆盖单进程估计结果。

- **澄清集中MLE估计逻辑**：
  - 用户确认“是否可以理解为先OLS，然后同时估计A中的参数，即三个网络权重+10个rho”。
  - 已澄清：不是先固定一次OLS后再估A，而是在每一组候选 $(\delta,\rho)$ 下构造 $A_t$，对 $A_tR_t$ 做OLS得到 $\alpha,\beta,\Sigma_\eta$，再代回集中似然比较该候选参数。
  - 当前核心数值优化参数确实是3个网络组合权重 $\delta_j$ + 10个行业异质性 $\rho_i$，共13个网络参数。

- **验证**：
  - 已运行 `python -m py_compile "src/model/multilayer_network_mle_Multithreading.py"`，语法检查通过。

### 修改文件
```text
src/model/multilayer_network_mle_Multithreading.py  — ProcessPoolExecutor/batch版本 → joblib Parallel贴近旧代码版本
CLAUDE.md                                           — 更新多进程脚本状态与最后更新时间
memory/progress.md                                  — 追加本会话进度
memory/session-log.md                               — 追加本会话摘要
```

### 下次继续

- [ ] 运行 `src/model/multilayer_network_mle_Multithreading.py`，观察joblib版本是否能像旧预测代码一样提高CPU利用率。
- [ ] 若CPU仍低，优先检查单个rolling window耗时、joblib进程数、Windows任务管理器中Python子进程数量。
- [ ] 重跑 `src/model/RollingVAR-DY.R`，生成未标准化且日期一致的3层DY网络。
- [ ] 用结果文件分析 $\delta(t)$、异质性 $\rho_i(t)$ 与 `d_jsv_pos≈1` 是否持续。

### 当前阻塞

- 尚未实际运行joblib版本MLE脚本验证CPU利用率；DY网络仍建议重跑以消除csv层与jump层日期长度不一致问题。

---

## 2026-04-28 — 会话 #24：static-window固定W版本 + 识别问题诊断

### 本次完成

- **诊断 $\delta$ 的识别问题**：
  - 解释了为什么 one-step MLE 中3个网络权重 $\delta_j$ 与10个异质性 $\rho_i$ 容易弱识别。
  - 核心原因是模型主要通过 $\operatorname{diag}(\rho)\sum_j\delta_jW_j$ 识别网络效应；当3层DY网络高度相关时，$\rho_i$ 可以补偿不同 $\delta$ 组合。
  - 说明 $\delta$ 停在1/3可能来自初始化/平坦似然面，打到0/1角点可能来自SLSQP在弱识别 simplex 上选择边界解。

- **新增 Bonaccolto-style 窗口内固定W脚本**（`src/model/multilayer_network_mle_static_window.py`）：
  - 新建文件，不覆盖现有 `multilayer_network_mle_Multithreading.py` 动态版本。
  - 设置 `window=252`、`step=5`：每次使用252个交易日估计，窗口每次向前滑动5个交易日。
  - 每个rolling window只估计一个固定网络矩阵：对窗口内原始 $W_{j,t}$ 取时间均值，再对 $\bar W_j$ 行标准化。
  - 固定A似然使用 $T\log|A|$，区别于动态版本的 $\sum_t\log|A_t|$。
  - 输出文件使用 `_static_window` 后缀，避免覆盖动态版本结果。

- **修正窗口内固定W版本的标准化逻辑**：
  - 根据用户提醒，静态窗口版本不再使用全样本 max normalization。
  - 新逻辑为 `window_mean_row_normalize()`：先在252日窗口内对原始网络取均值，再对固定矩阵行标准化。
  - 保留 dynamic/time-varying 版本中的 max normalization 作为密度时变保留方案；static-window版本作为更接近Bonaccolto原始固定网络设定的对照。

- **澄清约束与代码实现**：
  - 标量 $\rho$ 的特征值约束应为 $1/\lambda_{\min}(W)<\rho<1/\lambda_{\max}(W)$；异质性 $\rho_i$ 版本不能直接套用该标量边界，当前用 $\rho(\operatorname{diag}(\rho)W)<1$ 的 spectral radius 检查。
  - 当前代码强制 $\rho\ge0$ 体现在 likelihood 中的 `if np.any(rho < 0): return 1e10` 和优化器 bounds `[(0, 2.0)] * K`。
  - 解释了 `bounds = [(0, 1)] * d + [(-2.0, 2.0)] * K` 的含义：前d个参数是 $\delta_j\in[0,1]$，后K个参数是 $\rho_i\in[-2,2]$。

### 修改文件
```text
src/model/multilayer_network_mle_static_window.py  — 新增窗口内固定W的Bonaccolto-style rolling MLE脚本
CLAUDE.md                                           — 更新项目状态、当前任务、已完成列表、文件结构和方法总结
memory/progress.md                                  — 追加本会话进度
memory/session-log.md                               — 追加本会话摘要
memory/decisions.md                                 — 记录static-window固定W与行标准化技术决策
```

### 下次继续

- [ ] 运行 `src/model/multilayer_network_mle_static_window.py`，生成 `rolling_delta_static_window.csv` 与 `rolling_centrality_static_window.csv`。
- [ ] 检查 static-window 版本中 $\delta(t)$ 是否仍出现1/3停滞或0/1角点。
- [ ] 与 `rolling_delta_multithreading.csv` 对比，判断识别问题是否由时变W、网络层相关性、异质性 $\rho_i$ 补偿或优化器行为造成。
- [ ] 如需进一步允许负网络暴露，修改 `rho < 0` penalty 与 optimizer bounds，并保留 spectral radius 稳定性检查。

### 当前阻塞

- static-window版本尚未运行验证；需要实际跑结果后才能判断是否缓解 $\delta$ 弱识别。

---

## 2026-04-28 — 会话 #25：PDF对照审查 + rho边界/PCA敏感性诊断

### 本次完成

- **核对 `src/model/multilayer_network_mle_static_window.py` 的似然函数与Bonaccolto原文**：
  - 确认 static-window 版本在每个窗口内使用固定 $A=I-\operatorname{diag}(\rho)\sum_j\delta_j\bar W_j$，Jacobian项为 $T\log|A|$，与Bonaccolto固定网络集中MLE形式一致。
  - 确认 `dirichlet_alpha=1.0` 时Dirichlet项为0，因此是纯 concentrated MLE；`alpha>1` 只能解释为数值正则/先验，不应称为原文约束。
  - 确认Bonaccolto对异质性 $\rho_i$ 没有非负约束；允许负 $\rho_i$ 与Billio中网络可能削弱系统性暴露的经济解释兼容。
  - 明确文献要求核心是 $A$ 非奇异/可逆；spectral-radius `<1`、determinant必须为正、`logabsdet<-20`阈值等都属于比原文更强的数值/稳定性限制，若保留必须单独标注。

- **检查 time-varying `multilayer_network_mle_Multithreading.py` 与两篇论文关系**：
  - 该脚本更接近Billio式时变网络扩展：窗口内逐期构造 $A_t=I-\operatorname{diag}(\rho)W_t^*$，似然使用 $\sum_t\log|A_t|$。
  - 它不是Bonaccolto主文的固定网络经验设定；若用于论文结果，需要明确写成动态扩展/稳健性版本。
  - 当前动态脚本中仍存在较强的隐藏限制：`dirichlet_alpha>1`、heterogeneous rho的有限边界、spectral-radius稳定性检查、以及对负determinant的排除逻辑；这些不应混同为Bonaccolto原文约束。

- **解释单窗口估计结果中的rho异常与边界敏感性**：
  - 无界或过宽rho时出现 `rho_avg=1631.414`，说明优化器利用了数学上可逆但经济上不可解释的大 $\rho_i$ 区域，属于弱识别/数值病态信号。
  - 将rho边界设为 `[-10,10]` 后估计为 `d_csv=0.000, d_jsv_pos=0.176, d_jsv_neg=0.824, rho_avg=1.520`，量级更正常，但仍偏向角点。
  - 将边界改为 `[-5,5]` 后变为 `d_csv=0.000, d_jsv_pos=0.295, d_jsv_neg=0.705, rho_avg=-1.535`，说明结果对rho边界高度敏感，不是单纯“迭代次数不够”。
  - 结论：有限对称rho边界可以作为数值优化边界使用，但必须显式命名并做敏感性表，不能写成理论约束。

- **解释优化器与迭代次数的作用**：
  - SLSQP适合处理simplex约束，但在平坦、多峰、弱识别似然面上容易停在初始值附近或simplex边界。
  - 增加 `maxiter` 只能解决“尚未收敛”的问题；如果不同rho边界、不同初值给出相近likelihood但不同参数，则根因是识别弱/多解，而非单纯迭代不足。
  - 后续应同时记录 `success`、`message`、`nit`、`loglik`、`rho_min/max`、`max_abs_rho`，用以区分优化失败与弱识别。

- **解释 PCA 因子导致δ更均衡的原因**：
  - 用户改为PCA后单窗口估计变为 `d_csv=0.229, d_jsv_pos=0.333, d_jsv_neg=0.438, rho_avg=0.565`。
  - 解释为PCA第一主成分更好吸收行业共同波动，减少网络项替代遗漏共同因子的压力，因此δ不再被迫推到某一跳跃网络角点。
  - 这说明factor specification是当前识别诊断的关键维度，应将 equal_weight 与 PCA 作为稳健性对照，并报告二者对δ和rho的影响。

### 修改文件
```text
CLAUDE.md             — 更新项目状态、当前任务、已完成项、待办项和方法总结
memory/progress.md    — 追加本会话进度
memory/session-log.md — 追加本会话摘要
memory/decisions.md   — 记录rho边界与factor specification的技术决策
```

### 下次继续

- [ ] 在 static-window 脚本中显式加入/记录 `rho_bounds` 数值边界，默认允许负值，并输出边界敏感性诊断表。
- [ ] 对 `[-1,1]`、`[-2,2]`、`[-5,5]`、`[-10,10]` 做单窗口对比，记录 `delta`、`rho_avg/min/max`、`max_abs_rho`、`loglik`、`nit`、`message`。
- [ ] 对 equal_weight 与 PCA 因子设定分别跑相同窗口，确认PCA缓解角点是否稳定。
- [ ] 扩展到多个rolling windows后，再与 `_multithreading` 动态版本做系统对比。

### 当前阻塞

- 目前只有单窗口试验结果，尚不足以判断整体时间序列中δ是否稳定；需要批量窗口结果和rho/factor敏感性表。

---

## 2026-05-05 — 会话 #26：网络稀疏性诊断与多线程优化

### 本次完成

- **复现用户运行的MLE结果分析**：用户运行了 `multilayer_network_mle_static_window.py`（window=252, step=1, network_window=1），获得565个窗口的估计结果，收敛率100%，但rho仍出现极端值（如-14.029, 16.916等）。

- **解释Network window与fixed mean matrix的区别**：
  - Network window：原始网络数据的时间跨度（取多少天的网络平均）
  - Fixed mean matrix：对Network window内的网络先取平均，再行标准化的处理方法
  - 当前配置 network_window=1 意味着每个窗口只用当天的网络快照

- **检查Granger网络稀疏性问题**：
  - CSV层：91.9%零元素，49.0%全零行
  - JSV_POS层：91.3%零元素，44.7%全零行  
  - JSV_NEG层：92.4%零元素，52.1%全零行
  - 发现JSV_NEG前5个时间点中4个为**完全全零矩阵**（密度=0）

- **分析网络稀疏性对MLE识别的影响**：
  - 当W为全零矩阵时，行标准化后W_fixed=0
  - W* = sum(delta_j * W_j) = 0，与delta无关
  - A = I，Jacobian项log|A|=0消失
  - delta参数不可识别，rho估计可能发散

- **将rolling_loo_granger.py改写为多线程版本**：
  - 添加`n_jobs`配置参数，默认自动检测CPU核心数
  - 使用`joblib.Parallel`实现并行窗口估计
  - 预计5-8倍加速（从分钟级到秒级）

- **验证两个脚本step参数的不一致**：
  - Granger生成网络：约每7天一个（实际step≈7）
  - MLE step=5：在共同日期上每5个点采样
  - 不完全对应，需要统一配置或明确文档说明

### 本次发现的关键问题

| 问题 | 影响 | 优先级 |
|------|------|--------|
| 跳跃网络（JSV_POS/NEG）大量全零矩阵 | MLE delta不可识别 | 🔴 高 |
| Granger因果检验显著性水平5%过严 | 网络边稀疏 | 🔴 高 |
| Window=252天不足以累积足够跳跃 | 零行比例>44% | 🟡 中 |
| Rho无边界限制 | 估计值发散 | 🟡 中 |
| Step参数不一致 | 日期对应混乱 | 🟢 低 |

### 下次继续

- [ ] 修复网络稀疏性：放宽Granger检验alpha=0.05→0.10或增加窗口长度
- [ ] 在static-window脚本中添加rho_bounds（如[-5,5]）防止估计发散
- [ ] 统一Granger和MLE脚本的step配置（或改为step=1完全对应）
- [ ] 重新生成网络并验证密度提升效果
- [ ] 运行完整的565窗口MLE并分析delta时间序列稳定性

### 当前阻塞

- 网络稀疏性（全零矩阵）导致MLE识别弱，需要先生成更稠密的网络才能进行可靠的参数估计

---

## 2026-05-05 — 会话 #27：对比待投论文窗口设置与估计方法

### 本次完成

- **分析待投论文(ECMODE-D-26-00256)§4.1估计方法**：
  - 窗口设置：**250个交易日**，**步长5个交易日**（每周滚动一次）
  - 当前代码默认：`window=252`, `step=1`（日度滚动），可配置为`step=5`

- **识别关键方法差异**（与当前代码对比）：

| 维度 | 论文方法 | 当前代码 | 差异影响 |
|------|----------|----------|----------|
| **风险建模** | GARCHSK(León et al., 2005)：条件方差+偏度+峰度 | LM跳跃检验+HAR-RV：连续+正跳+负跳波动率 | 论文捕捉高阶矩(偏度/峰度)，当前聚焦跳跃行为 |
| **因果检验** | 多元非线性Granger(Diks and Wolski, 2016)+data sharpening | 线性条件Granger(Hué et al., 2019) | 论文方法可能捕获非线性依赖，当前方法更稳健但可能遗漏非线性 |
| **风险维度** | 五维：价格、波动、非对称、极端、尾部 | 三维：csv(连续)、jsv_pos(正跳)、jsv_neg(负跳) | 当前跳跃波动率与论文"极端/尾部风险"有重叠但不完全对应 |
| **网络估计** | Bonaccolto et al. (2019)时变权重δ_d(t) | 相同 | ✓ 一致 |
| **MLE估计** | 独立滚动窗口（可分离MLE窗口与网络窗口） | 已实现`window`与`granger_window`分离 | ✓ `multilayer_network_mle_integrated.py`已支持 |

- **创建整合版本脚本**：`src/model/multilayer_network_mle_integrated.py`
  - 分离MLE收益率窗口(`window`)与网络生成窗口(`granger_window`)
  - 默认配置：`window=252`, `granger_window=500`, `step=5`
  - 支持论文式统一窗口(250,5)或分离配置

### 核心发现

1. **窗口参数对齐**：论文(250,5)与当前代码(252,1)接近，统一为(250,5)即可匹配
2. **方法差异显著**：GARCHSK+非线性Granger vs LM+线性Granger，这是本质差异
3. **网络稀疏性仍是瓶颈**：无论用哪种因果方法，JSV层稀疏性问题需通过放宽alpha(0.05→0.10-0.20)解决

### 下次继续

- [ ] 统一窗口配置为论文设置：`window=250`, `step=5`
- [ ] 放宽Granger检验显著性水平：`alpha=0.05→0.10-0.20`
- [ ] 添加rho数值边界：`[-5,5]`防止估计发散
- [ ] 评估长期迁移到GARCHSK+非线性Granger的可行性

### 当前阻塞

- 网络稀疏性（JSV层92%零元素）需先通过放宽alpha解决，才能进行可靠MLE估计

---

## 2026-05-05 — 会话 #28：集成脚本修复与运行

### 本次完成

- **集成脚本`multilayer_network_mle_integrated.py`关键修复**：
  1. **窗口起始点修复**：`first_valid_idx = max(window, granger_window) - 1`，确保第一个窗口有足够数据供MLE和Granger网络使用
  2. **数据清理修复**：添加`replace([np.inf, -np.inf], np.nan)`和`dropna()`处理，与`rolling_loo_granger.py`保持一致，解决SVD不收敛错误
  3. **索引修复**：`DatetimeIndex`不能用`reset_index()`，改用布尔掩码索引
  4. **数值稳定性**：`_residual_variance`添加try-except回退到ridge正则化（1e-6惩罚）

- **架构重大变更**：
  - **现场计算Granger网络**替代预计算DY网络，避免数据对齐问题
  - MLE窗口与Granger窗口分离配置（`window=252`, `granger_window=252`）
  - 短窗口MLE捕捉近期动态，长窗口Granger获得稳定网络

- **当前运行配置**（main_integrated中硬编码）：
  ```python
  window=252              # MLE收益率窗口（约1年）
  granger_window=252      # Granger网络窗口（与MLE同步，符合论文设定）
  step=5                  # 每5天估计一次
  granger_alpha=0.10      # 10%显著性水平（解决JSV层稀疏性）
  factor_method="pca"     # PCA第一主成分
  rho_init_values=[-0.3, 0.0, 0.3]  # 收窄初始值范围防发散
  ```

- **运行状态确认**：
  - 评估点：734个窗口（4167期数据，step=5）
  - 工人进程：55个（自动检测CPU核心数）
  - 状态：用户反馈正在运行中

### 重要说明

- **不再使用DY网络**：当前采用现场计算的Granger因果网络，预计算的`dy_all_*.csv`文件已不再需要
- `RollingVAR-DY.R`和`TVP-VAR-DY.R`保留但当前未使用
- `rolling_loo_granger.py`的功能被集成脚本替代
- 当前代码可直接从`{code}_lm_har.csv`读取数据并现场计算网络

### 下次继续

- [ ] 监控运行结果，检查734个窗口的收敛率和delta分布
- [ ] 若rho仍有极端值（如>10或<-10），添加硬边界[-5,5]
- [ ] 若delta出现角点（如接近0或1），尝试切换factor_method为equal_weight
- [ ] 长期：评估迁移到GARCHSK+非线性Granger的可行性

### 当前阻塞

- 无（脚本正在运行，等待结果）

---

## 2026-05-06 — 会话 #29：模拟测试环境构建 + 代码正确性验证 + Delta演化图

### 本次完成

- **创建模拟测试环境**（`simulation/` 文件夹）：
  - `simulation/data/generate_data.py`：5行业合成数据（T=1500），含收益率/波动率/CAViaR尾风险
  - `simulation/model/run_simulation.py`：顺序MLE，实时打印delta和rho
  - `simulation/model/oracle_test.py`：Oracle测试（直接用真实网络）
  - 真实参数：delta=[0.5, 0.3, 0.2], rho=[0.3, 0.25, 0.35, 0.2, 0.15]

- **模拟结果**（25个窗口）：
  - 100%收敛率，delta均值：vol=0.360, tail=0.302, ret=0.338
  - delta弱识别：趋向1/3，网络层相关性(corr=0.62)导致
  - rho后期出现极端值（±15），需边界约束

- **关键发现**：
  1. PCA因子内生性：从R=A^{-1}(βF+η)提取PCA被A^{-1}污染→rho系统偏误
  2. Bonaccolto原文用外生Fama-French 4因子
  3. Delta弱识别来自网络层相关性，非代码bug；组合W*可恢复
  4. Granger密度7-8%与Bonaccolto Table 1一致

- **Delta演化图**：`simulation/results/delta_evolution_simulation.png`

### 下次继续

- [ ] 替换PCA因子为外生因子（中国FF因子或沪深300）
- [ ] 添加rho数值边界[-5,5]
- [ ] 在模拟环境中测试外生因子是否修复rho偏误

### 当前阻塞

- 需要获取中国市场外生因子数据

---

## 2026-05-06 — 会话 #30：CSMAR FF3外生因子集成 + PCA vs FF3对比验证

### 本次完成

- **集成CSMAR Fama-French 3因子到MLE代码**（`src/model/multilayer_network_mle_integrated.py`）：
  1. `_BASE` 自动检测两台电脑路径（D盘/C盘）
  2. `NetworkConfig` 新增 `ff3_path` 和 `ff3_market_id` 字段
  3. 新增 `load_ff3_factors()` 函数：读取CSMAR CSV，筛选P9709（全部A股），解析日期，返回(dates, T×3矩阵)
  4. `construct_factors()` 新增 `method="ff3"` 分支，支持外部传入ff3_data
  5. `main_integrated()` 添加FF3日期对齐逻辑：收益率dates与FF3 dates取交集，统一裁剪R/dates/layer_data
  6. `daily_10()` 默认配置改为 `factor_method="ff3"`

- **FF3数据验证**：
  - 数据来源：CSMAR `STK_MKT_THRFACDAY.csv`，P9709（全部A股）
  - 覆盖范围：1991-07-01 ~ 2026-04-30，共8508个交易日
  - 与收益率数据（2009-03-02 ~ 2026-04-23）交集：4166天，**零数据损失**
  - 单位：百分比收益率（decimal），与r_now一致，无需缩放

- **PCA vs FF3单窗口对比**（同一252天窗口）：

  | 指标 | PCA | FF3 |
  |------|-----|-----|
  | rho范围 | [-53.1, 15.2] | **[-0.64, 5.1]** |
  | rho均值 | -3.60 | **0.62** |
  | loglik | 11387.7 | **11675.6** (+288) |
  | delta | csv=0.23, neg=0.76 | csv=0.99 |

  FF3显著改善rho估计，消除PCA内生性导致的rho发散

### 关键结论

1. **PCA因子内生性问题已解决**：FF3作为外生因子不受A^{-1}污染，rho不再发散
2. **FF3提供更高log-likelihood**：+288点，说明FF3能更好解释收益率变异
3. **Delta角点仍存在**：但原因是jsv_pos密度=3.3%（网络稀疏性），不是因子问题
4. **rho仍需边界约束**：个别窗口rho=5.1，建议添加[-5,5]

### 修改文件

```text
src/model/multilayer_network_mle_integrated.py — FF3因子集成（5处修改）
```

### 下次继续

- [ ] 添加rho数值边界[-5,5]防止个别窗口发散
- [ ] 运行完整滚动MLE（FF3因子，4166期，step=5）
- [ ] 解决网络稀疏性：放宽alpha至0.15-0.20，或换用DY溢出/分位数因果
- [ ] Delta弱识别处理：报告composite W*而非单独delta

### 当前阻塞

- 网络稀疏性（jsv_pos密度3.3%）导致delta角点，需要更稠密的网络方法

---

## 2026-05-06 — 会话 #31：非参数分位数因果网络集成（替代LOO Granger）

### 本次完成

- **创建 `src/model/multilayer_network_mle_integrated_Quantile.py`**（~780行）：
  - 从 `multilayer_network_mle_integrated.py` 复制并修改，用非参数分位数因果检验（Jeong et al. 2012）替代LOO条件Granger因果来构造K×K网络
  - 直接调用已有的 `src/model/np_quantile_causality.py` 中的 `qn_causality_network()` 函数

- **具体修改内容**：
  1. **Docstring更新**：标题改为"集成非参数分位数因果网络计算的滚动集中MLE估计"，添加Jeong et al. (2012)和Balcilar et al. (2016)参考文献
  2. **Import section**（L40-43）：添加 `sys.path` 动态插入，使用 `try/except NameError` 处理 `__file__` 在Jupyter环境中未定义的问题，然后 `from np_quantile_causality import qn_causality_network`
  3. **NetworkConfig**：
     - 删除 `granger_lag`、`granger_alpha` 字段
     - 重命名 `granger_window` → `network_window`（默认500）
     - 新增 `quantile_tau: float = 0.05`（左尾分位数，风险聚焦）
     - 新增 `quantile_alpha: float = 0.05`（单侧z检验显著性水平）
     - 新增 `quantile_type: str = "mean"`（"mean"或"variance"因果）
     - `daily_10()` 工厂方法同步更新
  4. **删除LOO Granger函数**（~120行）：
     - 删除 `_lag_matrix()`、`_residual_variance()`、`conditional_granger_stat()`、`loo_conditional_granger_networks()`
  5. **Rolling worker修改**：
     - 函数名从 `_estimate_window_job_integrated` → `_estimate_window_job_quantile`
     - 网络构造调用从 `loo_conditional_granger_networks(risk_window, ...)` → `qn_causality_network(risk_window, tau=config.quantile_tau, alpha=config.quantile_alpha, test_type=config.quantile_type)`
     - `granger_window` → `network_window`
  6. **主函数和输出**：
     - `rolling_estimation_integrated()` → `rolling_estimation_quantile()`
     - `main_integrated()` → `main_quantile()`
     - 所有输出文件名从 `*_integrated.*` → `*_quantile.*`
     - verbose输出标签更新（显示tau、alpha、type而非lag、granger_alpha）
     - 单窗口验证使用 `qn_causality_network()`

- **修复 `__file__` NameError**：
  - 用户在Jupyter/交互环境运行时报 `NameError: name '__file__' is not defined`
  - 解决方案：`try/except NameError` 模式，fallback到 `os.path.join(os.getcwd(), "src", "model")`
  - 已用 `python -m py_compile` 和 `import` 测试验证

- **发现已有实现**：`src/model/np_quantile_causality.py`（354行）已包含完整的Jeong et al. (2012)测试Python移植，包括：
  - `_dpill()` — Silverman带宽
  - `_weighted_quantile_regression()` — LP线性规划加权分位数回归
  - `_lprq2()` — 局部线性分位数回归
  - `np_quantile_causality()` — 成对因果检验
  - `qn_causality_network()` — K×K有向邻接矩阵构建

### 修改文件
```text
src/model/multilayer_network_mle_integrated_Quantile.py  — 新建（~780行，分位数因果版MLE集成脚本）
```

### 方法对比

| 维度 | LOO Granger（旧） | 分位数因果（新） |
|------|-------------------|-----------------|
| 检验类型 | 线性条件Granger | 非参数分位数因果 |
| 统计量 | F检验 | N(0,1)渐近z检验 |
| 优势 | 计算快 | 捕捉尾部/非线性依赖 |
| 网络密度 | ~3-8%（稀疏） | ~50-70%（稠密） |
| 复杂度 | O(K³)每层 | O(K²×T) per LP solve |
| 参数 | lag, alpha | tau, alpha, type |

### 计算复杂度注意

分位数检验对每对(i,j)运行一次 `linprog`（每个观测一次LP），K=10时需90对 × ~500 LP ≈ 45,000次LP求解/窗口。比LOO Granger OLS慢。已有 `joblib.Parallel` 并行缓解。

### 下次继续

- [ ] 运行 `multilayer_network_mle_integrated_Quantile.py`，验证分位数因果网络密度是否达到50-70%
- [ ] 对比分位数因果与LOO Granger的delta/rho估计差异
- [ ] 添加rho数值边界[-5,5]（适用于两个版本）
- [ ] 运行完整滚动MLE（FF3因子，step=5）
- [ ] 调整 `quantile_tau`（0.05 vs 0.10 vs 0.50）看网络密度和MLE结果变化

### 当前阻塞

- 分位数因果计算量大，首次完整运行可能较慢；可先用 `max_windows=1` 单窗口测试

