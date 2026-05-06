# 技术决策记录

> 每次做出重要技术选择时记录，避免下次重新讨论。

---

## 决策模板

```markdown
## YYYY-MM-DD：[决策主题]
- **选择**：XXX
- **备选方案**：YYY
- **理由**：为什么选 XXX
- **影响**：影响哪些文件/模块
```

---

## 2026-04-13：项目记忆系统架构

- **选择**：CLAUDE.md（主记忆）+ memory/（详细记录）+ hooks（自动化）
- **备选方案**：纯 TodoWrite、纯聊天记录
- **理由**：CLAUDE.md 由 Claude Code 自动加载，零配置，最可靠；hooks 提供自动化辅助
- **影响**：`.claude/settings.json`、`.claude/hooks/`、`memory/`

---

## 2026-04-25：MLE采用窗口内 fully time-varying network likelihood

- **选择**：`multilayer_network_mle.py` 在 rolling window 内逐期使用 $A_t=I-\rho W_t^*$ 或 $A_t=I-\Lambda W_t^*$，似然项使用 $\sum_t\log|A_t|$，约束对窗口内所有 $W_t^*$ 同时成立。
- **备选方案**：继续使用窗口终点 endpoint `W_t` 代表整个窗口，只在 endpoint 上检查特征值/稳定性约束。
- **理由**：Billio §4 对时变网络的 Assumption 4.1/4.2 要求同一个 $\rho$ 或 $\Lambda$ 对所有相关 $W_t$ 满足可逆/稳定约束；endpoint approximation 不等同于严格时变网络MLE。
- **影响**：`estimate_beta_sigma()` 支持 3D `A_seq`；`concentrated_loglik()` 对窗口内全序列计算约束和 log determinant；`rolling_estimation()` 传入窗口内完整网络序列。

---

## 2026-04-25：MLE严格采用 Bonaccolto one-step constrained concentrated MLE

- **选择**：`multilayer_network_mle.py` 使用 Bonaccolto et al. (2019) §2 的 one-step constrained concentrated MLE，在同一个优化问题中联合估计 $\delta_j$ 与异质性 $\rho_i$。
- **备选方案**：继续使用工程化两步法：Step1 用标量 $\rho$ 估计 $\delta$，Step2 固定 $\delta$ 后估计异质性 $\lambda_i$。
- **理由**：用户要求严格按照文献；Bonaccolto 的 Model-based network combination 没有两步法，核心是对 $(\delta,\rho)$ 的联合集中似然优化。两步法只能作为识别诊断或工程 workaround，不应作为主结果。
- **影响**：核心输出不再包含 `rho_scalar`、`step1_success`、`step2_success`、`weak_identification`；结果列中保留 `lambda_i`/`lambda_hat` 只是为了兼容旧表名，其经济含义等同于 Bonaccolto 的异质性 $\rho_i$。

---

## 2026-04-25：新增多进程滚动估计脚本作为执行加速实现

- **选择**：新建 `src/model/multilayer_network_mle_Multithreading.py`，用 `joblib.Parallel(..., prefer="processes")` 并行估计独立rolling windows，写法贴近用户旧预测代码。
- **备选方案**：只运行单进程版本，或使用Python threading。
- **理由**：one-step MLE在 `window=252, step=1` 下需要3719个窗口，每个窗口多起点SLSQP，单进程很慢；用户已有旧代码验证 `joblib.Parallel(..., prefer="processes")` 能在本机较好调用CPU。
- **影响**：多进程版输出独立的 `_multithreading` 结果文件，不覆盖单进程结果；保留BLAS线程限制以避免多进程下内部线性代数线程过度竞争。

---

## 2026-04-28：新增 static-window 固定网络MLE作为Bonaccolto对照版本

- **选择**：新建 `src/model/multilayer_network_mle_static_window.py`，在每个252日rolling window内将每层原始网络 $W_{j,t}$ 先取时间均值，再对 $\bar W_j$ 行标准化；窗口内使用固定 $A=I-\operatorname{diag}(\rho)\sum_j\delta_j\bar W_j$，滚动步长设为5日。
- **备选方案**：继续只使用 fully time-varying 版本；或在窗口内只取末日endpoint网络；或继续对静态窗口版本使用全样本max normalization。
- **理由**：Bonaccolto原文是样本内固定网络设定，static-window版本更适合作为文献基准和识别诊断；窗口均值比endpoint使用更多窗口内信息；固定W后不需要max normalization保留时变密度，行标准化更符合固定邻接矩阵的可比性。
- **影响**：static-window版本输出 `_static_window` 后缀结果文件；固定A的似然Jacobian项使用 $T\log|A|$，与动态版本的 $\sum_t\log|A_t|$ 区分；下一步需对比 static-window 与 dynamic 结果判断 $\delta$ 角点/1/3停滞是否来自时变W或弱识别。

---

## 2026-04-28：rho边界作为数值优化设定而非文献理论约束

- **选择**：异质性 $\rho_i$ 默认允许负值；如使用有限对称边界（如 `[-5,5]`、`[-10,10]`），必须标注为 numerical optimizer bounds，并配套做边界敏感性分析。
- **备选方案**：完全无界估计 $\rho_i$；或将 $\rho_i\ge0$ / 固定区间边界写成理论约束。
- **理由**：Bonaccolto原文没有要求异质性 $\rho_i\ge0$，似然核心要求是 $A$ 非奇异；但单窗口实验显示无界/过宽rho可得到 `rho_avg=1631.414` 等经济上不可解释结果，说明需要数值稳定化。
- **影响**：报告结果时必须区分理论约束（$\delta_j\ge0, \sum_j\delta_j=1, A$可逆）与数值边界（rho bounds）；后续应输出 `rho_min/max`、`max_abs_rho`、`loglik`、`nit`、`message` 并比较多个边界。

---

## 2026-04-28：factor specification作为δ识别诊断维度

- **选择**：将 equal_weight 与 PCA 因子设定作为MLE稳健性对照，重点观察两者对 $\delta$ 角点、1/3停滞和 $\rho_i$ 量级的影响。
- **备选方案**：固定使用 equal_weight 因子；或直接将PCA结果作为唯一主结果。
- **理由**：单窗口结果显示改用PCA后 $\delta$ 从跳跃网络偏角点转为更均衡（`d_csv=0.229, d_jsv_pos=0.333, d_jsv_neg=0.438, rho_avg=0.565`），说明共同因子遗漏会让网络项吸收系统性共动，从而影响网络权重识别。
- **影响**：后续结果表应同时报告因子设定；若PCA稳定缓解角点，需在论文中解释为共同因子控制加强后的网络组合权重，而不是简单归因于优化器改进。

---

## 待决策

- [ ] 编程语言和依赖管理工具（建议 Python + uv）
- [ ] 数据存储格式（CSV / Parquet / 数据库）
- [ ] 是否使用 Hydra 管理实验配置
