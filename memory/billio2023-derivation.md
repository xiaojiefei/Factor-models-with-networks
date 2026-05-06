# Billio et al. (2023) 精读笔记：从 §2.2 开始逐步推导

> 适用读者：计量经济学初学者
> 论文：Billio, Caporin, Panzica, Pelizzon (2023) — *The impact of network connectivity on factor exposures, asset pricing, and portfolio diversification*
> 与本项目关系：本文模型是 Bonaccolto et al. (2019) 多层耦合网络的**单层特例**，数学框架完全一致

---

## 0. 符号约定

在开始之前，统一符号：

| 符号                | 维度           | 含义                                                     |
| ------------------- | -------------- | -------------------------------------------------------- |
| $K$               | 标量           | 资产（行业）数量                                         |
| $M$               | 标量           | 共同因子数量                                             |
| $T$               | 标量           | 时间序列长度                                             |
| $R_t$             | $K \times 1$ | 第$t$ 期各资产的收益率向量                             |
| $F_t$             | $M \times 1$ | 第$t$ 期的共同因子收益率                               |
| $W$               | $K \times K$ | 网络邻接矩阵（行标准化）                                 |
| $\rho$            | 标量           | 网络效应强度参数                                         |
| $\bar{\beta}$     | $K \times M$ | **结构性**因子载荷矩阵（去掉网络效应后的真实暴露） |
| $\beta^*$         | $K \times M$ | **约化形式**因子载荷（OLS 能直接估计到的）         |
| $\eta_t$          | $K \times 1$ | **结构性**特质冲击（各资产之间独立！）             |
| $\varepsilon_t^*$ | $K \times 1$ | **约化形式**残差（各资产之间不独立）               |

**上标带 bar ($\bar{\beta}$)** = 结构参数（真实的、被网络遮蔽的）
**上标带 star ($\beta^*$)** = 约化形式参数（可直接观测/估计的）

---

## 1. §2.2 起点：为什么需要结构模型？

### 1.1 传统因子模型（Eq.1）

标准的线性因子模型：

$$
R_t = \alpha + \beta F_t + \varepsilon_t \tag{1}
$$

其中 $\varepsilon_t$ 假设为 i.i.d.（独立同分布），协方差矩阵 $\Sigma_\varepsilon$ 是**对角矩阵**（即不同资产的残差互不相关）。

**问题来了**：如果资产之间存在供应链、资金流等真实的经济关联，那么一个资产的特质冲击会传导到另一个资产。此时 $\varepsilon_t$ 的各分量**不可能独立**。

传统模型忽略了这种关联 → **模型误设（misspecification）**。

### 1.2 引入结构模型（Eq.2）

Billio 等人的核心想法：把资产之间的同期关联**显式写进模型**。

#### 1.2.1 Eq.(2) 是怎么来的？——从联立方程出发

**注意：Eq.(2) 不是从 Eq.(1) 数学推导出来的，而是一个全新的建模假设。**

传统模型假设每个资产独立地对因子做反应。但现实中，钢铁涨了 → 汽车成本变了 → 汽车也动了。所以我们写一个更真实的模型——**每个资产的收益同时取决于其他资产的收益**：

$$
R_{i,t} - \mathbb{E}[R_{i,t}] = \underbrace{\sum_{j \neq i} a_{ij}(R_{j,t} - \mathbb{E}[R_{j,t}])}_{\text{受其他资产影响}} + \bar{\beta}_i F_t + \eta_{i,t} \tag{$\star$}
$$

这是一个**联立方程（simultaneous equations）**：每个资产的收益出现在自己方程的左边，也出现在别人方程的右边。

**移项**，把右边的求和移到左边：

$$
(R_{i,t} - \mathbb{E}[R_{i,t}]) - \sum_{j \neq i} a_{ij}(R_{j,t} - \mathbb{E}[R_{j,t}]) = \bar{\beta}_i F_t + \eta_{i,t}
$$

用 3 个资产的例子展开（$\tilde{R}_i = R_{i,t} - \mathbb{E}[R_{i,t}]$ 是超额收益的简写）：

$$
\begin{cases} 1 \cdot \tilde{R}_1 - a_{12}\tilde{R}_2 - a_{13}\tilde{R}_3 = \bar{\beta}_1 F_t + \eta_1 \\ -a_{21}\tilde{R}_1 + 1 \cdot \tilde{R}_2 - a_{23}\tilde{R}_3 = \bar{\beta}_2 F_t + \eta_2 \\ -a_{31}\tilde{R}_1 - a_{32}\tilde{R}_2 + 1 \cdot \tilde{R}_3 = \bar{\beta}_3 F_t + \eta_3 \end{cases}
$$

写成矩阵形式：

$$
\underbrace{\begin{pmatrix} 1 & -a_{12} & -a_{13} \\ -a_{21} & 1 & -a_{23} \\ -a_{31} & -a_{32} & 1 \end{pmatrix}}_{A} \begin{pmatrix} \tilde{R}_1 \\ \tilde{R}_2 \\ \tilde{R}_3 \end{pmatrix} = \bar{\beta} F_t + \eta_t
$$

**这就是 Eq.(2)**。$A$ 矩阵的结构为：

- **对角线 = 1**（自己对自己的系数）
- **非对角线 = $-a_{ij}$**（$j$ 对 $i$ 的影响权重，带负号是因为移项）

|             | 说明                                                 |
| ----------- | ---------------------------------------------------- |
| Eq.(1)      | 传统模型：每个资产独立对因子反应                     |
| $(\star)$ | 新假设：资产$i$ 的收益还受其他资产收益的同期影响   |
| Eq.(2)      | $(\star)$ 移项后的矩阵写法，$A$ 编码了"谁影响谁" |

后面 §2.2 再进一步，用网络矩阵 $W$ 把 $A$ 结构化为 $A = I - \rho W$，把 $K^2$ 个自由参数降到 1 个。

#### 1.2.2 Eq.(2) 的符号与含义

$$
A \cdot (R_t - \mathbb{E}[R_t]) = \bar{\beta} F_t + \eta_t \tag{2}
$$

逐项解释：

| 项                        | 含义                                                                                                                 |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| $R_t - \mathbb{E}[R_t]$ | 收益率的**超额部分**（去掉均值，只留波动）                                                                     |
| $A$                     | $K \times K$ 矩阵，捕捉资产间的**同期互动关系**                                                              |
| $\bar{\beta}$           | **结构性**因子载荷 — 去掉网络效应后，资产对因子的**真实暴露**                                           |
| $\eta_t$                | **结构性**特质冲击 — 在结构层面是**独立的**，$\text{Cov}(\eta_{i,t}, \eta_{j,t}) = 0$（$i \neq j$） |

#### 1.2.3 为什么要减去均值 $\mathbb{E}[R_t]$？

假设钢铁行业平均每周涨 0.2%。某周它涨了 1.5%。

|                                   | 值             | 含义                              |
| --------------------------------- | -------------- | --------------------------------- |
| $R_{i,t}$                       | 1.5%           | 总收益                            |
| $\mathbb{E}[R_{i,t}]$           | 0.2%           | "正常"水平（风险补偿等长期均值）  |
| $R_{i,t} - \mathbb{E}[R_{i,t}]$ | **1.3%** | **意外的部分**（冲击/惊喜） |

**网络传导的是意外波动，不是正常水平。** 钢铁每周涨 0.2% 是常态，不会让汽车行业"受到冲击"。但钢铁突然多涨了 1.3%——这个**意外**才会通过供应链传导到汽车。

**如果不减均值会怎样？**

- 等式左边均值 $\neq 0$，右边 $\eta_{i,t}$ 均值 $= 0$
- 均值对不上，$\bar{\beta}$ 被迫把均值调整也吃进去，不再是纯粹的因子暴露

**实际操作中怎么处理？** 在估计时（Eq.35），$\mathbb{E}[R_t]$ 被替换为截距 $\bar{\alpha}$：$e_t = R_t - \bar{\alpha} - \Lambda W R_t - \bar{\beta} F_t$。所以"减均值"在实操中等价于**加一个截距项**。

> **一句话**：减均值 = 只关注"意外波动"部分，因为网络传导的是**冲击**，不是正常收益水平。

**直觉**：

- 方程 (2) 说的是：**扣除掉资产之间互相影响后的收益率**（$A \cdot (R_t - \mathbb{E}[R_t])$），才等于对因子的真实暴露 + 真正独立的噪声。
- $A$ 矩阵的作用就是"过滤掉"网络效应。

### 1.3 关键区分：结构 vs 约化

为什么区分 $\bar{\beta}$（结构）和 $\beta^*$（约化）？

从 Eq.(2) 解出 $R_t$：

$$
R_t = \mathbb{E}[R_t] + A^{-1} \bar{\beta} F_t + A^{-1} \eta_t \tag{3}
$$

对比传统模型 $R_t = \alpha + \beta F_t + \varepsilon_t$，可以识别出：

$$
\beta^* = A^{-1} \bar{\beta}, \quad \varepsilon_t^* = A^{-1} \eta_t
$$

**含义**：

- 你用 OLS 估计传统模型得到的 $\beta$（即 $\beta^*$），其实是 $A^{-1} \bar{\beta}$
- 它**混合了**真实的因子暴露 $\bar{\beta}$ **和**网络效应 $A^{-1}$
- 残差 $\varepsilon_t^*$ 也**不再独立**，因为 $A^{-1}$ 把独立的 $\eta_t$ "搅混"了

---

## 2. §2.2 续：用网络矩阵 $W$ 结构化 $A$

### 2.1 $A$ 矩阵不能随便设

$A$ 是一个 $K \times K$ 矩阵，有 $K^2$ 个参数。如果 $K = 30$（30个行业），就有 900 个参数，根本估不出来（参数太多，数据不够）。

**解决办法**：用一个**已知的**网络邻接矩阵 $W$ 来给 $A$ 施加结构。

### 2.2 邻接矩阵 $W$ 是什么？

$W$ 是一个 $K \times K$ 的矩阵，描述资产之间的关联强度：

$$
W_{ij} = \text{资产 } j \text{ 对资产 } i \text{ 的影响权重}
$$

**举个 3×3 的例子**：假设有钢铁、汽车、银行三个行业

$$
W = \begin{pmatrix} 0 & 0.6 & 0.4 \\ 0.7 & 0 & 0.3 \\ 0.5 & 0.5 & 0 \end{pmatrix}
$$

- 第 1 行 $(0, 0.6, 0.4)$：钢铁受汽车影响权重 0.6，受银行影响权重 0.4
- 对角线为 0：自己不影响自己
- **行标准化**：每行之和 = 1（$0 + 0.6 + 0.4 = 1$）

#### 2.2.1 对角线为 0，$R_i$ 不会消失吗？

不会。回到方程：

$$
\underbrace{\tilde{R}_{i}}_{\text{左边：资产 i 自己}} = \rho \sum_{j=1}^{K} W_{ij} \underbrace{\tilde{R}_{j}}_{\text{右边：邻居们}} + \bar{\beta}_i F_t + \eta_{i,t}
$$

$R_i$ 在**等号左边**（系数为 1），$W$ 只管"**别人对我的影响**"。$W_{ii} = 0$ 的意思是：钢铁不通过网络影响自己——这很合理，"自己影响自己"不是网络传染，那是自回归。

用钢铁展开：$\tilde{R}_{\text{steel}} = \rho [0 \cdot \tilde{R}_{\text{steel}} + 0.6 \cdot \tilde{R}_{\text{auto}} + 0.4 \cdot \tilde{R}_{\text{bank}}] + \bar{\beta}_{\text{steel}} F_t + \eta_{\text{steel}}$

移项后看 $(I - \rho W)$ 的对角线更清楚：

$$
(I - \rho W) = \begin{pmatrix} \mathbf{1} & -0.6\rho & -0.4\rho \\ -0.7\rho & \mathbf{1} & -0.3\rho \\ -0.5\rho & -0.5\rho & \mathbf{1} \end{pmatrix}
$$

**对角线是 1**（来自单位矩阵 $I$），每个资产自己的收益系数 = 1，没有消失。

**网络从哪来？** 本文用美国投入产出表（I/O table）。你的项目用 DY 溢出网络、Granger 因果等。

### 2.3 最简模型：标量 $\rho$（Eq.4-7）

最简单的设定：所有资产对网络冲击的反应强度**相同**，用一个标量 $\rho$ 控制。

#### 2.3.1 Eq.(4) 是怎么来的？——从 $(\star)$ 到 SAR

回顾 §1.2.1 的联立方程 $(\star)$：

$$
R_{i,t} - \mathbb{E}[R_{i,t}] = \sum_{j \neq i} a_{ij}(R_{j,t} - \mathbb{E}[R_{j,t}]) + \bar{\beta}_i F_t + \eta_{i,t} \tag{$\star$}
$$

问题：$a_{ij}$ 有 $K(K-1)$ 个（$K=30$ 时有 870 个），估不出来。

**核心假设**：让影响系数 = 统一强度 × 已知权重：

$$
\boxed{a_{ij} = \rho \cdot W_{ij}}
$$

- $W_{ij}$：**已知的**，从外部数据（投入产出表等）得到
- $\rho$：**未知的**，一个标量，控制网络影响的整体强度

代入 $(\star)$：

$$
R_{i,t} - \mathbb{E}[R_{i,t}] = \rho \sum_{j \neq i} W_{ij}(R_{j,t} - \mathbb{E}[R_{j,t}]) + \bar{\beta}_i F_t + \eta_{i,t}
$$

因为 $W_{ii} = 0$（自己不影响自己），$\sum_{j \neq i}$ 可以写成 $\sum_{j=1}^{K}$。

右边的求和 $\sum_j W_{ij} \tilde{R}_j$ 就是矩阵乘法 $(W\tilde{R})_i$ 的第 $i$ 个分量，所以矩阵形式为：

$$
R_t - \mathbb{E}[R_t] = \rho W (R_t - \mathbb{E}[R_t]) + \bar{\beta} F_t + \eta_t \tag{4}
$$

**推导链总结**：

$$
\text{Eq.}(1) \xrightarrow{\text{发现残差不独立}} (\star) \xrightarrow{\text{令 } a_{ij} = \rho W_{ij}} \text{Eq.}(4) \xrightarrow{\text{移项}} \text{Eq.}(6)
$$

| 步骤                 | 自由参数数量               | 说明                   |
| -------------------- | -------------------------- | ---------------------- |
| $(\star)$ 联立方程 | $K(K-1) = 870$           | 太多，估不出来         |
| Eq.(4) 用$W$ 约束  | **1 个**（$\rho$） | $W$ 是已知的外部数据 |

#### 2.3.2 Eq.(4) 逐项理解

$$
R_t - \mathbb{E}[R_t] = \rho W (R_t - \mathbb{E}[R_t]) + \bar{\beta} F_t + \eta_t \tag{4}
$$

- 左边：资产 $i$ 的超额收益
- 右边第一项 $\rho W (R_t - \mathbb{E}[R_t])$：资产 $i$ 的收益**受到邻居收益的影响**
  - $W$ 把邻居的收益加权平均
  - $\rho$ 控制影响有多大
- 右边第二项 $\bar{\beta} F_t$：因子的直接影响
- 右边第三项 $\eta_t$：独立的特质冲击

**对资产 $i$ 展开**（写出第 $i$ 个方程）：

$$
R_{i,t} - \mathbb{E}[R_{i,t}] = \rho \sum_{j=1}^{K} W_{ij} (R_{j,t} - \mathbb{E}[R_{j,t}]) + \bar{\beta}_i F_t + \eta_{i,t}
$$

**用我们的 3×3 例子**（钢铁）：

$$
R_{\text{steel},t} - \mathbb{E}[R_{\text{steel},t}] = \rho \cdot [0.6 \cdot (R_{\text{auto},t} - \mathbb{E}[R_{\text{auto},t}]) + 0.4 \cdot (R_{\text{bank},t} - \mathbb{E}[R_{\text{bank},t}}])] + \bar{\beta}_{\text{steel}} F_t + \eta_{\text{steel},t}
$$

也就是说：钢铁的收益 = $\rho$ × (60%汽车收益 + 40%银行收益) + 因子影响 + 噪声。

**整理成矩阵形式**（移项）：

$$
(R_t - \mathbb{E}[R_t]) - \rho W (R_t - \mathbb{E}[R_t]) = \bar{\beta} F_t + \eta_t
$$

$$
(I - \rho W)(R_t - \mathbb{E}[R_t]) = \bar{\beta} F_t + \eta_t \tag{6}
$$

对比 Eq.(2)，可以识别出：

$$
\boxed{A = I - \rho W} \tag{7}
$$

**成果**：$A$ 矩阵从 $K^2$ 个自由参数，降到了**只有 1 个参数** $\rho$！（$W$ 是已知的）

#### 2.3.3 外生暴露 vs 内生暴露——$\bar{\beta}_i$ 和 $\rho$ 的经济含义

> 原文："the coefficients in the vector $\bar{\beta}_i$ represent the exposure to the common factors, or **exogenous exposure**, while the coefficient $\rho$ tracks the **endogenous risk exposure**, which is influenced by the network structure and is thus called **network exposure**."

回到单资产方程 Eq.(5)：

$$
R_{i,t} - E[R_{i,t}] = \underbrace{\rho \sum_j w_{ij}(R_{j,t} - E[R_{j,t}])}_{\text{内生暴露（endogenous）}} + \underbrace{\bar{\beta}_i F_t}_{\text{外生暴露（exogenous）}} + \eta_{i,t}
$$

**$\bar{\beta}_i F_t$ — 外生暴露（exogenous exposure）**

$F_t$ 是 FF4 因子（市场、规模、价值、动量），来自**系统外部**——宏观经济、央行政策、投资者情绪等。"外生"的意思是：**不管这些行业之间有没有供应链关系，市场涨 1%，每个行业都会按自己的 $\bar{\beta}_i$ 做出反应。** 这个反应与行业间的网络连接完全无关。

例：美联储加息（外部冲击）→ 市场因子下跌 → 钢铁行业按 $\bar{\beta}_{steel}$ 下跌。在这个过程中，钢铁跌多少**不取决于汽车行业有没有跌**。

**$\rho \sum_j w_{ij} \tilde{R}_j$ — 内生暴露（endogenous risk exposure）**

"内生"的意思是：**这部分风险不是来自外部因子，而是来自系统内部——其他资产的收益率变动。** 影响源（$R_j$）本身也是模型的被解释变量——$R_j$ 出现在行业 $j$ 方程的等号左边（因变量），同时出现在行业 $i$ 方程的等号右边（解释变量）。大家互相影响，风险在系统内部**自我产生、自我放大**。

例：汽车行业因芯片短缺产出下降（汽车的特质冲击 $\eta_{auto}$）→ 通过供应链传导到钢铁（$w_{steel, auto} > 0$）→ 钢铁行业也受影响。这个传导完全发生在行业系统**内部**，与外部宏观因子无关。

**为什么叫 "network exposure"？** 因为 $\rho$ 的效果**完全依赖于网络结构 $W$**。如果 $W = 0$（没有网络），$\rho \times 0 = 0$，内生暴露消失。$\rho$ 只是一个标量"强度开关"，它**通过 $W$ 起作用**。

| 类型     | 参数              | 来源                                  | 是否取决于网络  |
| -------- | ----------------- | ------------------------------------- | --------------- |
| 外生暴露 | $\bar{\beta}_i$ | 外部因子$F_t$（宏观经济）           | 否              |
| 内生暴露 | $\rho$          | 其他资产的收益$R_{j,t}$（系统内部） | 是（通过$W$） |

**一句话总结**：$\bar{\beta}_i$ 量化"外面的世界怎么影响我"，$\rho$ 量化"我的邻居怎么影响我"。前者与网络无关（外生），后者完全由网络决定（内生）。

---

## 3. §2.3 约化形式与网络乘数效应（Eq.8-14）

### 3.1 约化形式（Eq.8）

从 Eq.(6) 解出 $R_t$：

$$
R_t = \mathbb{E}[R_t] + (I - \rho W)^{-1} \bar{\beta} F_t + (I - \rho W)^{-1} \eta_t \tag{8}
$$

这就是**约化形式**：所有东西都用可观测变量表达。

对比传统模型 $R_t = \alpha + \beta F_t + \varepsilon_t$：

$$
\beta^* = (I - \rho W)^{-1} \bar{\beta}, \quad \varepsilon_t^* = (I - \rho W)^{-1} \eta_t
$$

**关键点**：传统 OLS 估计的 $\beta$，实际上是 $(I - \rho W)^{-1} \bar{\beta}$，被网络放大了！

#### 3.1.1 为什么论文说 $\beta^*$ 是"非线性函数"？

> 原文："the reduced-form parameters $\beta^*$ are **nonlinear functions** of the interconnections between assets (the matrix $A$) and of the structural exposure to common structural factors (the matrix $\bar{\beta}$)"

乍一看，$\beta^* = A^{-1}\bar{\beta}$ 就是矩阵乘向量，似乎是线性的。为什么说"非线性"？

**关键区分**：论文说的"非线性"不是关于 $\bar{\beta}$ 的，而是**关于参数 $\rho$（即网络连接 $A$）的**。

代入 $A = I - \rho W$：

$$
\beta^* = (I - \rho W)^{-1} \bar{\beta}
$$

用 Neumann 级数展开（见下节 §3.2）：

$$
\beta^* = \bar{\beta} + \rho W\bar{\beta} + \rho^2 W^2\bar{\beta} + \rho^3 W^3\bar{\beta} + \cdots
$$

$\beta^*$ 包含 $\rho$ 的**所有幂次**（$\rho, \rho^2, \rho^3, \ldots$），所以它关于 $\rho$ 不是线性函数，而是无穷级数——**这就是非线性的来源**。

**数值直觉**：假设某行业的结构 beta $\bar{\beta}_i = 0.8$

| $\rho$ | $\rho^2$ | $\rho^3$ | $\beta^*_i$（近似） |
| -------- | ---------- | ---------- | --------------------- |
| 0.3      | 0.09       | 0.027      | ≈ 0.8 + 小量         |
| 0.6      | 0.36       | 0.216      | ≈ 0.8 + 大量         |

$\rho$ 从 0.3 翻倍到 0.6，但 $\rho^2$ 翻了4倍（0.09→0.36），$\rho^3$ 翻了8倍（0.027→0.216）。高阶传播项被**超线性放大**，所以 $\beta^*$ 的增长快于 $\rho$ 的增长——非线性。

**一句话总结**：矩阵求逆把线性的 $A = I - \rho W$ 变成了非线性的 $A^{-1} = I + \rho W + \rho^2 W^2 + \cdots$。$\beta^*$ 关于 $\bar{\beta}$ 是线性的，但关于网络参数 $\rho$ 是非线性的。这正是"网络乘数效应"的数学本质。

### 3.2 网络乘数展开（Eq.9）——本文最重要的公式之一

矩阵 $(I - \rho W)^{-1}$ 可以用**Neumann 级数**展开：

$$
\boxed{(I - \rho W)^{-1} = I + \rho W + \rho^2 W^2 + \rho^3 W^3 + \cdots} \tag{9}
$$

**这个展开成立的条件**：$|\rho| < 1$（确保级数收敛）

**为什么这个展开如此重要？** 因为它给出了冲击传播的**逐阶分解**：

| 阶数 | 项             | 含义                       | 衰减速度   |
| ---- | -------------- | -------------------------- | ---------- |
| 0 阶 | $I$          | 自身的直接效应             | 1          |
| 1 阶 | $\rho W$     | **直接邻居**的效应   | $\rho$   |
| 2 阶 | $\rho^2 W^2$ | **邻居的邻居**的效应 | $\rho^2$ |
| 3 阶 | $\rho^3 W^3$ | **三阶邻居**的效应   | $\rho^3$ |
| ...  | ...            | ...                        | ...        |

**为什么会收敛？** 因为 $|\rho| < 1$，所以 $\rho^j \to 0$（当 $j \to \infty$）。高阶邻居的效应越来越小。

#### 手算验证（2×2 例子）

设 $K = 2$，$\rho = 0.3$：

$$
W = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
$$

（两个资产互相影响，权重为 1）

计算 $I - \rho W$：

$$
I - 0.3W = \begin{pmatrix} 1 & -0.3 \\ -0.3 & 1 \end{pmatrix}
$$

**直接求逆**：

$$
\det = 1 \times 1 - (-0.3)(-0.3) = 1 - 0.09 = 0.91
$$

$$
(I - 0.3W)^{-1} = \frac{1}{0.91}\begin{pmatrix} 1 & 0.3 \\ 0.3 & 1 \end{pmatrix} = \begin{pmatrix} 1.099 & 0.330 \\ 0.330 & 1.099 \end{pmatrix}
$$

**用级数验证**：

$$
I + 0.3W + 0.09W^2 + 0.027W^3 + \cdots
$$

注意 $W^2 = I$（因为 $W$ 是置换矩阵），所以：

$$
= I(1 + 0.09 + 0.0081 + \cdots) + W(0.3 + 0.027 + \cdots)
$$

$$
= I \cdot \frac{1}{1 - 0.09} + W \cdot \frac{0.3}{1 - 0.09}
$$

$$
= \frac{1}{0.91} I + \frac{0.3}{0.91} W = \begin{pmatrix} 1.099 & 0.330 \\ 0.330 & 1.099 \end{pmatrix} \quad \checkmark
$$

完全一致！级数展开确实等于矩阵求逆。

### 3.3 四项分解（Eq.10）

把 Eq.(9) 代入 Eq.(8)：

$$
R_t = \mathbb{E}[R_t] + \underbrace{\bar{\beta} F_t}_{\text{(a) 结构性因子暴露}} + \underbrace{\sum_{j=1}^{\infty} \rho^j W^j \bar{\beta} F_t}_{\text{(b) 网络引致的因子暴露}} + \underbrace{\eta_t}_{\text{(c) 结构性特质冲击}} + \underbrace{\sum_{j=1}^{\infty} \rho^j W^j \eta_t}_{\text{(d) 网络传播的特质冲击}} \tag{10}
$$

**这四项的经济含义**：

**(a) 结构性因子暴露 $\bar{\beta} F_t$**

资产对共同因子的**直接、真实**反应。比如市场涨 1%，钢铁行业本身（不考虑供应链传导）应该涨多少。

**(b) 网络引致的因子暴露 $\sum \rho^j W^j \bar{\beta} F_t$**

通过网络**间接**获得的因子暴露。市场涨了 → 汽车行业涨了 → 通过供应链传导到钢铁行业 → 钢铁额外涨了一点。

**(c) 结构性特质冲击 $\eta_t$**

资产自身的独立冲击。比如钢铁行业发生矿难，这是钢铁独有的冲击。

**(d) 网络传播的特质冲击 $\sum \rho^j W^j \eta_t$**

邻居的特质冲击通过网络传到自己身上。汽车行业的工厂爆炸（汽车的特质冲击），通过供应链影响到了钢铁。

### 3.4 核心洞察：间接暴露（Eq.12-14 的例子）

论文给了一个精彩的例子。假设：

- 资产 $i$ **只连接到**资产 $i+1$（$W$ 的第 $i$ 行只有 $W_{i,i+1} = 1$，其余为 0）
- 有 2 个因子
- 资产 $i$ 只暴露于因子 1（$\bar{\beta}_{i,1} \neq 0, \bar{\beta}_{i,2} = 0$）
- 资产 $i+1$ 同时暴露于因子 1 和 2（$\bar{\beta}_{i+1,1} \neq 0, \bar{\beta}_{i+1,2} \neq 0$）

那么资产 $i$ 的收益率中，因子部分为：

$$
\underbrace{\bar{\beta}_{i,1} F_{1,t}}_{\text{直接暴露于因子1}} + \underbrace{\rho \bar{\beta}_{i+1,1} F_{1,t}}_{\text{间接暴露于因子1（通过邻居）}} + \underbrace{\rho \bar{\beta}_{i+1,2} F_{2,t}}_{\text{间接暴露于因子2（通过邻居）}} + \text{高阶项} \tag{14}
$$

**结论**：

- 虽然资产 $i$ 本身**不暴露于因子 2**
- 但因为它的邻居 $i+1$ 暴露于因子 2
- 资产 $i$ 通过网络**间接暴露**于因子 2，暴露程度为 $\rho \bar{\beta}_{i+1,2}$

**后果**：如果你用传统 OLS 估计因子模型，你会发现资产 $i$ 对因子 2 有显著的 $\beta$。但这不是"真实"暴露，而是网络传导造成的**虚假暴露**。

#### 3.4.1 易混淆点：$\rho$ 的指数是跳数，不是资产编号

Eq.(14) 中 $\rho \bar{\beta}_{i+1,2}$ 的下标 $i+1$ 是**邻居的名字**（资产编号），**不是** $\rho$ 的次方。

$$
\rho \bar{\beta}_{\underbrace{i+1}_{\text{资产编号（邻居叫谁）}},\underbrace{2}_{\text{因子编号}}}
$$

$\rho$ 的指数取决于**网络中跳了几步**（距离），而非资产编号：

| 跳数 | $\rho$ 的次方   | 含义           | 例子                                                  |
| ---- | ----------------- | -------------- | ----------------------------------------------------- |
| 0 跳 | $\rho^0 = 1$    | 自身直接效应   | 资产$i$ 对因子 1 的暴露 $\bar{\beta}_{i,1}$       |
| 1 跳 | $\rho^1 = \rho$ | 直接邻居的影响 | 资产$i+1$ 是 1 跳邻居，系数 $\rho$                |
| 2 跳 | $\rho^2$        | 邻居的邻居     | 若$i \to i+1 \to i+2$，则 $i+2$ 的系数 $\rho^2$ |

所以即使邻居碰巧叫"资产 $i+1$"，$\rho$ 的指数仍然是 1（因为只跳了 1 步），不是 $i+1$。

### 3.5 网络效应的三个深层后果（§2.3 原文补充）

> 原文："Such a result can be further generalized by focusing, for instance, on sector-specific risk factors. Those factors, in the presence of a network exposure, despite being sector specific, will have a systematic impact on all connected assets..."

这段话是 §3.3-3.4 的推论，讲了网络乘数 $(I-\rho W)^{-1}$ 带来的三个深层后果。

#### 3.5.1 后果一：行业特有因子会"伪装"成公共因子

假设油价只**直接影响**能源行业，其他行业的结构性暴露为零：

$$
\bar{\beta} = \begin{pmatrix} 0 \\ 0 \\ 0.8 \\ 0 \end{pmatrix} \quad \text{（只有能源行业 } \bar{\beta}_3 \neq 0 \text{）}
$$

但约化形式 $\beta^* = (I-\rho W)^{-1}\bar{\beta}$ 之后：

$$
\beta^* = \begin{pmatrix} 0.15 \\ 0.22 \\ 0.95 \\ 0.10 \end{pmatrix} \quad \text{（所有行业都有非零暴露！）}
$$

如果你**忽略网络**，直接用 OLS 回归收益率对油价因子，你会发现每个行业都对油价有显著的 beta。你会得出结论："油价是所有行业的公共因子。"

**但这个结论是错的。** 油价在结构上只影响能源行业（$\bar{\beta}_3 = 0.8$），其他行业的 $\beta^*$ 完全来自网络传导——能源涨了 → 通过供应链传到钢铁 → 钢铁跟着动了。

> 原文："we might label as **common** a factor that in reality is structurally related just to a **subset** of the investment universe and that impacts other assets only through network connections."

**一句话**：$(I-\rho W)^{-1}$ 把局部因子"放大"成了看似公共的因子。不建模网络就会**错误归因**。

#### 3.5.2 后果二：独立的特质冲击通过网络变得相关

同样的逻辑，但换成残差 $\eta_t$。

结构模型假设每个行业的特质冲击**互不相关**：

$$
\eta_t \sim N(0, \Omega_\eta), \quad \Omega_\eta = \text{对角阵}
$$

但约化形式：$\varepsilon^*_t = (I-\rho W)^{-1}\eta_t$

即使 $\eta_1$ 和 $\eta_2$ 之间 $\text{Cov} = 0$，经过 $(I-\rho W)^{-1}$ 的"搅混"，$\varepsilon^*_1$ 和 $\varepsilon^*_2$ 之间 $\text{Cov} \neq 0$。

**例子**：银行 A 出了一个纯粹自己的问题（坏账，$\eta_A < 0$）。通过网络传导，银行 B 的**观测到的**残差也变负（$\varepsilon^*_B < 0$）。如果你不知道网络的存在，你会以为它们共享了某个隐藏的公共因子——但其实只是传染。

> 原文："if we assume they are uncorrelated, the existence of network connections implies that the structural shocks of one asset impact on the returns of all the assets connected to it."

这正是 §4.1（Table 9）里实证结果的理论基础：传统模型的平均残差相关 = 0.10，网络模型降到 0.03——因为网络模型正确"吸收"了这种传导效应。

#### 3.5.3 后果三：为什么空间计量的"直接/间接效应分解"在这里不适用

> 原文："We note that the latter decompositions are appropriate in a framework where, for given dependent-variable measures across subjects, we have a number of covariates, each of which is available with **variable-specific observations**. In our case, we do have **common factors** (not asset-specific variables) and therefore these decompositions cannot be applied."

在空间计量经济学（LeSage and Pace, 2009）中，有一个经典的"直接效应 vs 间接效应"分解方法。它要求**每个个体有自己的解释变量值**（asset-specific covariates）。

| 分解条件      | 空间计量经典模型                                             | 本文的因子模型                                                  |
| ------------- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| 解释变量      | $X_i$——每个个体有**自己的** $X$ 值               | $F_t$——所有资产共享**相同的**因子值                   |
| 可变维度      | 载荷（$\beta_i$）**和**变量值（$X_i$）都因个体而异 | 只有载荷（$\bar{\beta}_i$）因资产而异，$F_t$ 对所有资产相同 |
| 直接/间接分解 | ✅ 可以清楚定义                                              | ❌ 公式不适用                                                   |

**直觉**：空间计量的分解公式需要区分"$X_i$ 变了一单位对 $Y_i$ 的影响（直接）"和"$X_j$ 变了一单位对 $Y_i$ 的影响（间接）"。但在因子模型中，$F_t$ 是**公共的**——不存在"只让银行 A 的市场因子变一单位而其他银行不变"这种操作，因为所有银行面对的是**同一个**市场因子。所以直接/间接的分解失去了定义基础。

**这段话的目的**：作者在说"我们的模型跟空间计量很像（都有 $(I-\rho W)^{-1}$ 的乘数结构），但数学上不能直接搬用空间计量的标准分解公式。"这是一个**方法论的边界声明**，告诉读者不要误用空间计量的现成工具。

---

## 4. §2.4 风险分解与分散化——逐步推导

### 4.0 起点回顾

从 §3.1 的约化形式 Eq.(8)：

$$
R_t - \mathbb{E}[R_t] = \underbrace{(I - \rho W)^{-1}}_{\Omega} \bar{\beta} F_t + \underbrace{(I - \rho W)^{-1}}_{\Omega} \eta_t
$$

简记 $\Omega = (I - \rho W)^{-1}$，令 $\tilde{R}_t = R_t - \mathbb{E}[R_t]$：

$$
\tilde{R}_t = \Omega\bar{\beta} F_t + \Omega\eta_t
$$

$\tilde{R}_t$ 由两个**独立**的随机项组成（$F_t$ 和 $\eta_t$ 独立，因为 $F_t$ 是外部因子，$\eta_t$ 是结构性特质冲击）。

> **符号说明**：论文 Eq.(16) 中的 $A$ 指 $(I-\rho W)^{-1}$（约化形式的网络乘数矩阵），本笔记用 $\Omega$ 表示同一物体，避免与结构矩阵 $A = I - \rho W$（Eq.7）混淆。论文用 $\sigma_m^2$ 表示单一市场因子的方差，本笔记推广为一般因子协方差矩阵 $\Sigma_F$。论文用 $\Omega_\eta$ 表示结构性残差的对角协方差矩阵，本笔记等价地用 $\Sigma_\eta$。

### 4.1 方差分解 Eq.(16) 的逐步推导

**目标**：求 $\text{Var}(\tilde{R}_t)$。

**用到的线性代数规则**：如果 $Y = BX$（$B$ 为常数矩阵），且 $\text{Var}(X) = \Sigma$，则 $\text{Var}(Y) = B \Sigma B'$。

**推导**：因为 $F_t$ 和 $\eta_t$ 独立，随机变量之和的方差 = 各自方差之和：

$$
\text{Var}(\tilde{R}_t) = \text{Var}(\Omega\bar{\beta} F_t) + \text{Var}(\Omega\eta_t)
$$

逐项计算：

**第一项**（因子风险）——$\text{Var}(\Omega\bar{\beta} F_t)$：

把 $\Omega\bar{\beta}$ 看成常数矩阵 $B$，$F_t$ 看成随机向量 $X$：

$$
\text{Var}(\Omega\bar{\beta} F_t) = (\Omega\bar{\beta}) \cdot \underbrace{\text{Var}(F_t)}_{\Sigma_F} \cdot (\Omega\bar{\beta})' = \Omega\bar{\beta}\Sigma_F\bar{\beta}'\Omega'
$$

**如果只有一个市场因子**（$M=1$），则 $\Sigma_F = \sigma_m^2$（标量），$\bar{\beta}$ 退化为 $K \times 1$ 向量，此项简化为 $\Omega\bar{\beta}\bar{\beta}'\Omega'\sigma_m^2$——这就是论文 Eq.(16) 中的写法。

**第二项**（特质风险）——$\text{Var}(\Omega\eta_t)$：

$$
\text{Var}(\Omega\eta_t) = \Omega \cdot \underbrace{\text{Var}(\eta_t)}_{\Omega_\eta} \cdot \Omega' = \Omega\;\Omega_\eta\;\Omega'
$$

**合在一起**：

$$
\boxed{\mathbb{V}[R_t] = \Omega\bar{\beta}\bar{\beta}'\Omega'\sigma_m^2 + \Omega\;\Omega_\eta\;\Omega'} \tag{16}
$$

> 论文原文形式：$\mathbb{V}[R_t] = A\bar{\beta}\bar{\beta}'A'\sigma_m^2 + A\Omega_\eta A'$，其中 $A = (I-\rho W)^{-1}$。
> 一般化形式（多因子）：$\mathbb{V}[R_t] = \Omega\bar{\beta}\Sigma_F\bar{\beta}'\Omega' + \Omega\;\Omega_\eta\;\Omega'$。

### 4.2 与传统模型对比——为什么"网络让特质风险不再独立"

对比传统模型的风险分解 $\text{Var}[R_t] = \beta \Sigma_F \beta' + \Sigma_\varepsilon$：

|            | 传统模型                       | 网络增强模型                                             |
| ---------- | ------------------------------ | -------------------------------------------------------- |
| 系统性风险 | $\beta \Sigma_F \beta'$      | $\Omega \bar{\beta} \bar{\beta}' \Omega' \sigma_m^2$   |
| 特质风险   | $\Sigma_\varepsilon$（对角） | $\Omega \;\Omega_\eta\; \Omega'$（**非对角！**） |

**为什么特质风险变成非对角？**

结构性冲击 $\eta_t$ 的协方差 $\Omega_\eta$ 是对角阵（各行业的"真实"特质冲击互不相关）。但约化形式的残差 $\varepsilon^*_t = \Omega\eta_t$，它的协方差是 $\Omega\;\Omega_\eta\;\Omega'$。

即使 $\Omega_\eta = \text{diag}(\sigma_1^2, \ldots, \sigma_K^2)$，乘上 $\Omega$（有大量非零非对角元素）后，$\Omega\;\Omega_\eta\;\Omega'$ 的非对角元素**不为零**。

#### 2×2 手算验证

设 $K=2$，$\rho=0.3$，$W = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$：

$$
\Omega = (I - 0.3W)^{-1} = \frac{1}{0.91}\begin{pmatrix} 1 & 0.3 \\ 0.3 & 1 \end{pmatrix}
$$

$$
\Sigma_\eta = \begin{pmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{pmatrix}
$$

$$
\Omega\Sigma_\eta\Omega' = \frac{1}{0.91^2}\begin{pmatrix} 1 & 0.3 \\ 0.3 & 1 \end{pmatrix}\begin{pmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{pmatrix}\begin{pmatrix} 1 & 0.3 \\ 0.3 & 1 \end{pmatrix}
$$

中间矩阵乘法（先算前两个矩阵的乘积）：

$$
\begin{pmatrix} 1 & 0.3 \\ 0.3 & 1 \end{pmatrix}\begin{pmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{pmatrix} = \begin{pmatrix} \sigma_1^2 & 0.3\sigma_2^2 \\ 0.3\sigma_1^2 & \sigma_2^2 \end{pmatrix}
$$

再乘右边的 $\Omega'$：

$$
\begin{pmatrix} \sigma_1^2 & 0.3\sigma_2^2 \\ 0.3\sigma_1^2 & \sigma_2^2 \end{pmatrix}\begin{pmatrix} 1 & 0.3 \\ 0.3 & 1 \end{pmatrix} = \begin{pmatrix} \sigma_1^2 + 0.09\sigma_2^2 & 0.3(\sigma_1^2 + \sigma_2^2) \\ 0.3(\sigma_1^2 + \sigma_2^2) & 0.09\sigma_1^2 + \sigma_2^2 \end{pmatrix}
$$

**看非对角元素**：$\frac{0.3(\sigma_1^2 + \sigma_2^2)}{0.91^2} \neq 0$！

即使两个资产的结构性冲击完全独立（$\Sigma_\eta$ 对角），约化形式的残差之间出现了正相关——这就是网络传染在协方差中留下的痕迹。

### 4.3 组合风险——投资组合权重 $\omega$ 代入

投资组合收益率 $R_p = \omega'R_t = \omega_1 R_1 + \omega_2 R_2 + \cdots + \omega_K R_K$（$\omega$ 是 $K \times 1$ 权重向量，$\sum_i \omega_i = 1$）。

$R_p$ 是标量，所以 $\text{Var}(R_p)$ 也是标量。

#### 4.3.1 为什么 $\omega'\Sigma\omega$ 是标量？——维度追踪

线性代数规则：如果 $Y = c'X$（常数向量 $c$ 转置乘随机向量 $X$），则 $\text{Var}(Y) = c'\text{Var}(X)c$。

维度：

$$
\underbrace{\omega'}_{1 \times K} \underbrace{\Sigma}_{K \times K} \underbrace{\omega}_{K \times 1} = \underbrace{\text{标量}}_{1 \times 1}
$$

分两步看：先算 $\omega'\Sigma$：$(1 \times K)(K \times K) = 1 \times K$（行向量）；再乘 $\omega$：$(1 \times K)(K \times 1) = 1 \times 1$（标量）。

#### 4.3.2 用 $K=2$ 手算验证

设 $\omega = \begin{pmatrix} 0.6 \\ 0.4 \end{pmatrix}$，$\Sigma = \begin{pmatrix} \sigma_{11} & \sigma_{12} \\ \sigma_{12} & \sigma_{22} \end{pmatrix}$：

$$
\omega'\Sigma\omega = (0.6,\; 0.4)\begin{pmatrix} \sigma_{11} & \sigma_{12} \\ \sigma_{12} & \sigma_{22} \end{pmatrix}\begin{pmatrix} 0.6 \\ 0.4 \end{pmatrix}
$$

**第一步** $\omega'\Sigma = (0.6\sigma_{11}+0.4\sigma_{12},\; 0.6\sigma_{12}+0.4\sigma_{22})$

**第二步** 再乘 $\omega$：$= 0.36\sigma_{11} + 0.48\sigma_{12} + 0.16\sigma_{22}$

这就是教科书里的组合方差公式 $\sum_i\sum_j \omega_i\omega_j\sigma_{ij}$ 的矩阵简写。

#### 4.3.3 代入 Eq.(16) 得组合方差 Eq.(17)

$\Sigma = \text{Var}(\tilde{R}_t) = \Omega\bar{\beta}\bar{\beta}'\Omega'\sigma_m^2 + \Omega\Omega_\eta\Omega'$，代入 $\omega'\Sigma\omega$，矩阵乘法对加法可分配（$\omega'(A+B)\omega = \omega'A\omega + \omega'B\omega$）：

$$
\text{Var}(R_p) = \omega'\text{Var}(\tilde{R}_t)\omega = \underbrace{\omega'\Omega\bar{\beta}\bar{\beta}'\Omega'\omega\;\sigma_m^2}_{\text{组合系统性风险（有网络）}} + \underbrace{\omega'\Omega\;\Omega_\eta\;\Omega'\omega}_{\text{组合特质风险（有网络）}} \tag{17}
$$

第一项的维度追踪：

$$
\underbrace{\omega'}_{1 \times K}\underbrace{\Omega}_{K \times K}\underbrace{\bar{\beta}}_{K \times 1}\underbrace{\bar{\beta}'}_{1 \times K}\underbrace{\Omega'}_{K \times K}\underbrace{\omega}_{K \times 1} = 1 \times 1 \quad \text{（标量）}
$$

从左往右连乘：$(1 \times K)(K \times K) \to 1 \times K$，$(1 \times K)(K \times 1) \to 1 \times 1$（标量），$(1 \times 1)(1 \times K) \to 1 \times K$，$(1 \times K)(K \times K) \to 1 \times K$，$(1 \times K)(K \times 1) \to 1 \times 1$。✓

#### 4.3.4 Eq.(18)——"加零"分解技巧

Eq.(17) 给出了有网络时的组合方差，但我们想知道：**网络到底多贡献了多少风险？**

为此需要一个"没有网络"的基准。如果没有网络（$\rho = 0$，即 $\Omega = I$），Eq.(17) 退化为：

$$
\text{Var}(R_p)\big|_{\text{无网络}} = \omega'\bar{\beta}\Sigma_F\bar{\beta}'\omega + \omega'\Omega_\eta\omega
$$

**技巧**：在 Eq.(17) 右边**加上再减去**这两个无网络基准项（$+a - a = 0$，数值上什么都没变）：

$$
\text{Var}(R_p) = \omega'\Omega\bar{\beta}\Sigma_F\bar{\beta}'\Omega'\omega + \omega'\Omega\;\Omega_\eta\;\Omega'\omega \underbrace{\pm \omega'\bar{\beta}\Sigma_F\bar{\beta}'\omega}_{\text{加减系统性基准}} \underbrace{\pm \omega'\Omega_\eta\omega}_{\text{加减特质性基准}} \tag{18}
$$

> **为什么这么做？** 类比：你考了 85 分，如果没有辅导你能考 70 分。把成绩写成 $85 = 85 + 70 - 70$，然后重组为 $85 = 70 + (85 - 70) = 70 + 15$。这样就能看出"基础实力 = 70，辅导贡献 = 15"。Eq.(18) 对风险做了完全相同的事。

#### 4.3.5 Eq.(19)——四项风险分解

把 Eq.(18) 的六项重新分组（把"有网络"和"无网络"配对做差）：

$$
\text{Var}(R_p) = \underbrace{\omega'\bar{\beta}\Sigma_F\bar{\beta}'\omega}_{I} + \underbrace{(\omega'\Omega\bar{\beta}\Sigma_F\bar{\beta}'\Omega'\omega - \omega'\bar{\beta}\Sigma_F\bar{\beta}'\omega)}_{II} + \underbrace{\omega'\Omega_\eta\omega}_{III} + \underbrace{(\omega'\Omega\;\Omega_\eta\;\Omega'\omega - \omega'\Omega_\eta\omega)}_{IV} \tag{19}
$$

四项的经济含义：

| 项            | 公式                                             | 经济含义                                                                       |
| ------------- | ------------------------------------------------ | ------------------------------------------------------------------------------ |
| **I**   | $\omega'\bar{\beta}\Sigma_F\bar{\beta}'\omega$ | **结构性系统风险**：没有网络也存在的系统性风险（外生）                   |
| **II**  | 有网络系统性$-$ 无网络系统性                   | **网络对系统性风险的贡献**：网络让系统性风险增加了多少（内生系统性效应） |
| **III** | $\omega'\Omega_\eta\omega$                     | **结构性特质风险**：没有网络时的特质风险（外生）                         |
| **IV**  | 有网络特质$-$ 无网络特质                       | **网络对特质风险的贡献**：网络让特质风险放大了多少（内生放大效应）       |

> 验证：$I + II + III + IV = (\text{无网络系统性}) + (\text{有网络系统性} - \text{无网络系统性}) + (\text{无网络特质}) + (\text{有网络特质} - \text{无网络特质}) = \text{有网络系统性} + \text{有网络特质} = \text{Eq.(17)}$。✓

**传统金融学的"分散化定理"**说：特质风险可以通过增加资产数量分散掉（$\frac{\bar{\sigma}^2}{K} \to 0$）。但在有网络时，特质风险部分变成了 $\omega'\Omega\;\Omega_\eta\;\Omega'\omega$——里面多了 $\Omega$ 的放大效应。分散化是否仍然有效？

### 4.4 简化假设下的推导——严格按论文脚注11的路径

为得到可解释的解析结果，做三个简化假设（论文原文）：

1. **等权组合**：$\omega = \frac{1}{K}\mathbf{1}$（$\mathbf{1} = (1,\ldots,1)'$，$K$ 维全1向量）
2. **完全连接网络**：$W_{ij} = \frac{1}{K-1}$（$i \neq j$），$W_{ii} = 0$（行标准化后）
3. **等方差**：$\Omega_\eta = \bar{\sigma}^2 I$（论文进一步设 $\bar{\sigma}^2 = 1$）

**目标**：推导组合特质风险 $\omega'\mathcal{A}\,\Omega_\eta\,\mathcal{A}'\omega$ 的显式表达，其中 $\mathcal{A} = (I - \rho W)^{-1}$。

论文的推导思路（对应脚注11）：逐元素求 $\mathcal{A}$ → 逐元素求 $\mathcal{A}^2$ → 对所有元素求和 → 化简。

#### 4.4.1 第一步：完全图 $W$ 的结构

$$
W = \frac{1}{K-1}(\mathbf{1}\mathbf{1}' - I)
$$

验证：非对角元素 $= \frac{1}{K-1}$，对角元素 $= 0$，行和 $= 1$。✓

#### 4.4.2 第二步：$\mathcal{A} = (I - \rho W)^{-1}$ 的元素（论文脚注11）

完全图 $W$ 结构特殊：对角线全0，非对角线全 $\frac{1}{K-1}$。因此 $I - \rho W$ 也有相同的"常对角+常非对角"结构，其逆矩阵 $\mathcal{A}$ 同样如此。

**记号**：设 $\mathcal{A}$ 的对角元素为 $\alpha_d$，非对角元素为 $\alpha_o$。

论文脚注11给出（此处 $K$ = 资产数）：

$$
\alpha_d = \frac{(K-2)\rho - (K-1)}{D}, \qquad \alpha_o = \frac{-\rho}{D}
$$

其中

$$
D \;\equiv\; \rho^2 + (K-2)\rho - (K-1)
$$

**关键因式分解**：

$$
D = \rho^2 + (K-2)\rho - (K-1) = (\rho - 1)(\rho + K - 1) = -(K-1+\rho)(1-\rho)
$$

验证：$(\rho - 1)(\rho + K - 1) = \rho^2 + (K-1)\rho - \rho - (K-1) = \rho^2 + (K-2)\rho - (K-1)$。✓

**数值验证**（$K=3, \rho=0.5$）：

$$
D = 0.25 + 0.5 - 2 = -1.25 = -(2.5)(0.5) \;\checkmark
$$

$$
\alpha_d = \frac{(1)(0.5) - 2}{-1.25} = \frac{-1.5}{-1.25} = 1.2, \qquad \alpha_o = \frac{-0.5}{-1.25} = 0.4
$$

直接矩阵求逆验证：$(I - 0.5W)^{-1}$ 的对角 $= 1.2$，非对角 $= 0.4$。✓

> **注意**：论文脚注11的原始公式使用 $K$ 表示**邻居数**（= 资产数 $-1$），而非资产总数。上面已统一为 $K$ = 资产总数。

#### 4.4.3 第三步：$\mathcal{A}^2$ 的元素（论文脚注11）

$\mathcal{A}$ 是对称矩阵且具有"常对角 $\alpha_d$ + 常非对角 $\alpha_o$"结构，因此 $\mathcal{A}^2 = \mathcal{A}\mathcal{A}'$ 也有相同结构。

逐元素计算（$(\mathcal{A}^2)_{ij} = \sum_k \mathcal{A}_{ik}\mathcal{A}_{kj}$）：

**对角元素**（$i = j$）：一个 $\alpha_d^2$ 项 + $(K-1)$ 个 $\alpha_o^2$ 项

$$
(\mathcal{A}^2)_{ii} = \alpha_d^2 + (K-1)\alpha_o^2
$$

**非对角元素**（$i \neq j$）：$k=i$ 贡献 $\alpha_d \alpha_o$，$k=j$ 贡献 $\alpha_o \alpha_d$，其余 $(K-2)$ 个贡献 $\alpha_o^2$

$$
(\mathcal{A}^2)_{ij} = 2\alpha_d\alpha_o + (K-2)\alpha_o^2
$$

论文脚注11给出的等价形式（代入 $\alpha_d, \alpha_o$ 后整理）：

$$
(\mathcal{A}^2)_{ii} = \frac{K\rho^2 + [(K-2)\rho - (K-1)]^2}{D^2}
$$

$$
(\mathcal{A}^2)_{ij} = \frac{(K-1)\rho^2 - 2\rho[(K-2)\rho - (K-1)]}{D^2} \quad (i \neq j)
$$

**数值验证**（$K=3, \rho=0.5$）：

$$
(\mathcal{A}^2)_{ii} = 1.2^2 + 2 \times 0.4^2 = 1.44 + 0.32 = 1.76
$$

$$
(\mathcal{A}^2)_{ij} = 2 \times 1.2 \times 0.4 + 1 \times 0.4^2 = 0.96 + 0.16 = 1.12
$$

与脚注公式：$\frac{3 \times 0.25 + (-1.5)^2}{(-1.25)^2} = \frac{2.75}{1.5625} = 1.76$，$\frac{2 \times 0.25 + 2 \times 0.5 \times 1.5}{1.5625} = \frac{1.75}{1.5625} = 1.12$。✓

#### 4.4.4 第四步：对所有元素求和——完美平方消去

等权组合 $\omega = \frac{1}{K}\mathbf{1}$ 下：

$$
\omega'\mathcal{A}^2\omega = \frac{1}{K^2}\,\mathbf{1}'\mathcal{A}^2\mathbf{1}
$$

$\mathbf{1}'\mathcal{A}^2\mathbf{1}$ 就是 $\mathcal{A}^2$ 所有元素之和。$\mathcal{A}^2$ 有 $K$ 个对角元素和 $K(K-1)$ 个非对角元素：

$$
\mathbf{1}'\mathcal{A}^2\mathbf{1} = K\left[\alpha_d^2 + (K-1)\alpha_o^2\right] + K(K-1)\left[2\alpha_d\alpha_o + (K-2)\alpha_o^2\right]
$$

展开整理（合并 $\alpha_o^2$ 项）：

$$
= K\Big[\alpha_d^2 + 2(K-1)\alpha_d\alpha_o + (K-1)\alpha_o^2\big(1 + (K-2)\big)\Big]
$$

$$
= K\Big[\alpha_d^2 + 2(K-1)\alpha_d\alpha_o + (K-1)^2\alpha_o^2\Big]
$$

$$
= K\Big[\alpha_d + (K-1)\alpha_o\Big]^2
$$

**关键恒等式**——计算 $\alpha_d + (K-1)\alpha_o$：

$$
\alpha_d + (K-1)\alpha_o = \frac{(K-2)\rho - (K-1)}{D} + \frac{-(K-1)\rho}{D} = \frac{(K-2)\rho - (K-1) - (K-1)\rho}{D}
$$

$$
= \frac{-\rho - (K-1)}{D} = \frac{-(K-1+\rho)}{D}
$$

代入 $D = -(K-1+\rho)(1-\rho)$：

$$
\alpha_d + (K-1)\alpha_o = \frac{-(K-1+\rho)}{-(K-1+\rho)(1-\rho)} = \frac{1}{1-\rho}
$$

$(K-1+\rho)$ 完美消去！结果与 $K$ 无关！

因此：

$$
\mathbf{1}'\mathcal{A}^2\mathbf{1} = K \cdot \frac{1}{(1-\rho)^2} = \frac{K}{(1-\rho)^2}
$$

最终：

$$
\omega'\mathcal{A}^2\omega = \frac{1}{K^2} \cdot \frac{K}{(1-\rho)^2} = \frac{1}{K(1-\rho)^2}
$$

$$
\boxed{\text{组合特质风险} = \bar{\sigma}^2 \cdot \omega'\mathcal{A}^2\omega = \frac{\bar{\sigma}^2}{K(1-\rho)^2}} \tag{20}
$$

#### 4.4.5 第五步：数值验证

用 Python 直接计算 $\omega'(I-\rho W)^{-2}\omega$，与公式 $\frac{1}{K(1-\rho)^2}$ 对比：

| $K$ | $\rho$ | 直接矩阵计算 | 公式$\frac{1}{K(1-\rho)^2}$ | 一致？ |
| ----- | -------- | ------------ | ----------------------------- | ------ |
| 3     | 0.3      | 0.680272     | 0.680272                      | ✓     |
| 3     | 0.5      | 1.333333     | 1.333333                      | ✓     |
| 5     | 0.5      | 0.800000     | 0.800000                      | ✓     |
| 10    | 0.5      | 0.400000     | 0.400000                      | ✓     |
| 30    | 0.5      | 0.133333     | 0.133333                      | ✓     |
| 30    | 0.7      | 0.370370     | 0.370370                      | ✓     |

> **关于论文 Eq.(20) 印刷形式的说明**：论文中 Eq.(20) 的分式为 $\frac{K+\rho^2}{(K+\rho)^2(\rho-1)^2}$。经数值验证，这个分式与正确结果 $\frac{1}{K(1-\rho)^2}$ 不一致。原因是论文脚注11使用的 $K$ 实际上代表**邻居数**（= 资产数 $-1$），但 Eq.(20) 文中注明"$K$ is the number of assets"。这种 $K$ 定义不一致导致了印刷形式的偏差。论文的**结论**（分散化效果被网络削弱、极限趋于零）完全正确。

### 4.5 与无网络情况的对比

| 情况                 | 组合特质风险                           | $K=30, \rho=0.5$ 时的数值                |
| -------------------- | -------------------------------------- | ------------------------------------------ |
| 无网络（$\rho=0$） | $\frac{\bar{\sigma}^2}{K}$           | $0.033\bar{\sigma}^2$                    |
| 有网络（$\rho>0$） | $\frac{\bar{\sigma}^2}{K(1-\rho)^2}$ | $0.133\bar{\sigma}^2$（**4倍！**） |

**放大因子** $= \frac{1}{(1-\rho)^2}$——这个因子完全来自网络乘数 $\Omega = (I-\rho W)^{-1}$：

| $\rho$ | 放大倍数$\frac{1}{(1-\rho)^2}$ | 含义                                      |
| -------- | -------------------------------- | ----------------------------------------- |
| 0        | 1.0                              | 无网络效应                                |
| 0.2      | 1.56                             | 弱网络，风险增加56%                       |
| 0.5      | 4.0                              | 中等网络，需要4倍资产数才能达到同样分散化 |
| 0.7      | 11.1                             | 强网络，分散化严重受损                    |
| 0.9      | 100                              | 极强网络，几乎无法分散                    |

### 4.6 极限行为——分散化最终仍然有效（Eq.21）

$$
\lim_{K \to \infty} \frac{\bar{\sigma}^2}{K(1-\rho)^2} = 0 \tag{21}
$$

**好消息**：无论 $\rho$ 多大（只要 $\rho < 1$），当资产数量 $K \to \infty$ 时，组合特质风险趋于零。分散化**最终有效**。

**坏消息**：收敛速度慢了 $\frac{1}{(1-\rho)^2}$ 倍。要达到传统模型中 $K$ 个资产的分散化效果，网络模型需要 $\frac{K}{(1-\rho)^2}$ 个资产。例如 $\rho=0.5$ 时，你需要 4 倍资产才能达到同样低的特质风险。

### 4.7 推导总结——本节的三个核心信息

| 核心信息                       | 数学表达                                                                             | 直觉                                 |
| ------------------------------ | ------------------------------------------------------------------------------------ | ------------------------------------ |
| 网络把独立的特质冲击变成相关的 | $\Omega\;\Omega_\eta\;\Omega'$ 是非对角阵                                          | $\Omega$ 把各行业的独立冲击"搅混"  |
| 网络让分散化更难               | 特质风险从$\frac{\bar{\sigma}^2}{K}$ 膨胀到 $\frac{\bar{\sigma}^2}{K(1-\rho)^2}$ | 冲击通过网络传染，抵消了分散化的好处 |
| 分散化最终仍然有效             | $K \to \infty$ 时特质风险 → 0                                                     | 但需要更多资产才能达到同样效果       |

**推导链总结**：

$$
\text{Eq.(8)} \xrightarrow{\text{Var}(AX)=A\Sigma A'} \text{Eq.(16)} \xrightarrow{\omega'\cdot\omega} \text{Eq.(17)} \xrightarrow{\text{加零}} \text{Eq.(18)} \xrightarrow{\text{重组}} \text{Eq.(19) 四项分解} \xrightarrow{\text{简化}} \text{Eq.(20)} \xrightarrow{K\to\infty} \text{Eq.(21)}
$$

---

## 5. §3 模型推广

### 5.1 异质网络反应（§3.1, Eq.22）

标量 $\rho$ 的限制：所有资产对网络冲击的反应一样强。

**推广**：每个资产有自己的 $\rho_i$：

$$
A = I - \Lambda W, \quad \Lambda = \text{diag}(\rho_1, \rho_2, \ldots, \rho_K) \tag{22}
$$

**展开来看**：$A$ 的第 $i$ 行为：

$$
A_{i,\cdot} = e_i' - \rho_i W_{i,\cdot}
$$

即：

$$
(I - \Lambda W) \cdot (R_t - \mathbb{E}[R_t]) = \bar{\beta} F_t + \eta_t
$$

对资产 $i$：

$$
R_{i,t} - \mathbb{E}[R_{i,t}] - \rho_i \sum_j W_{ij}(R_{j,t} - \mathbb{E}[R_{j,t}]) = \bar{\beta}_i F_t + \eta_{i,t}
$$

- $\rho_i > 0$：资产 $i$ 的收益被邻居**正向拉动**（传染效应）
- $\rho_i < 0$：资产 $i$ 的收益被邻居**反向推动**（避险效应/风险吸收）
- $\rho_i = 0$：资产 $i$ 不受网络影响

#### 5.1.1 为什么 $\rho_i < 0$ 会削弱系统性风险？

**结构方程直觉**：

$$
R_{i,t} - \mathbb{E}[R_i] = \rho_i \sum_j W_{ij}(R_{j,t} - \mathbb{E}[R_j]) + \bar{\beta}_i F_t + \eta_{i,t}
$$

当 $\rho_i > 0$ 时，邻居的正向冲击会"拉高"资产 $i$，邻居的负向冲击会"拖低"资产 $i$——**同向传染**，放大系统性波动。

当 $\rho_i < 0$ 时，邻居的正向冲击反而"拉低"资产 $i$，邻居的负向冲击反而"推高"资产 $i$——**逆向对冲**，吸收系统性波动。

**Neumann 级数展开**：

约化形式 $\beta^* = (I - \rho W)^{-1}\bar{\beta}$ 可展开为：

$$
\beta^* = \bar{\beta} + \rho W\bar{\beta} + \rho^2 W^2\bar{\beta} + \rho^3 W^3\bar{\beta} + \cdots
$$

- $\rho > 0$：所有项**同号叠加**，$\beta^* > \bar{\beta}$（放大因子暴露）
- $\rho < 0$：**奇数阶项为负**（$\rho W\bar{\beta}$ 减去，$\rho^3 W^3\bar{\beta}$ 减去），偶数阶项为正（$\rho^2 W^2\bar{\beta}$ 加上）——交替相消，最终 $|\beta^*| < |\bar{\beta}|$

**数值例子**（制造业，假设 $\rho_i = -0.12$，$\bar{\beta}_i = 1.0$）：

$$
\beta_i^* \approx 1.0 + (-0.12)(0.8) + (-0.12)^2(0.6) + \cdots \approx 1.0 - 0.096 + 0.009 \approx 0.91
$$

结构性暴露 $\bar{\beta}_i = 1.0$，但约化形式暴露 $\beta_i^* \approx 0.91$——**网络"吸收"了部分市场风险**。

**经济含义——避风港（flight-to-safety）**：

- 当市场下跌时，资金从高风险行业流入防御性行业（公用事业、医疗）
- 这些行业与邻居的收益率呈反向关系 → $\rho_i < 0$
- 它们是天然的系统性风险吸收器
- 在 Table 6 中，论文确实报告了部分行业的 $\rho_i$ 为负且不显著，这些行业正是不受网络传染的"孤岛"

> **总结**：$\rho > 0$ = 传染器（amplifier），$\rho < 0$ = 吸收器（absorber），$\rho = 0$ = 绝缘体（insulator）。

#### 5.1.2 $\rho$ 与系统性风险的直接联系——特质风险渠道

**论文对"系统性风险"的定义**（§2.4 开篇）：

> "The term 'systematic risk' [...] refers to the risk to which an investor in a **well-diversified portfolio** is exposed, which stems from the dependence of the returns on common factors."
> — Markowitz (1952), Sharpe (1964), Lintner (1965), Ross (1976)

传统定义下，系统性风险 = 充分分散化后仍然存在的风险 = 公共因子风险。特质风险在充分分散化后趋于零，不算系统性风险。

但 Cochrane (2011) 指出："there is a need for a better understanding of the **determinants** of systematic risk"。本论文的回答是：**网络连接性（$\rho$）正是系统性风险的一个决定因素**——它通过使特质冲击变得不可分散，扩大了系统性风险的边界。

网络对系统性风险的贡献有两个渠道，但**特质风险渠道才是本质性的"质变"**。

**渠道一（因子暴露放大）——量变**：

$\beta^* = (I - \Lambda W)^{-1}\bar{\beta}$，$\rho > 0$ 放大 $\beta^*$，$\rho < 0$ 缩小 $\beta^*$。但因子风险 $\beta^* F_t$ 在传统模型中**本来就不可分散**，$\rho$ 只改变其大小，没有改变其性质。

**渠道二（特质风险变不可分散）——质变** ⭐：

传统模型中，特质冲击 $\eta_t$ 独立 → 组合特质风险 $\frac{\bar{\sigma}^2}{K} \to 0$（$K \to \infty$ 可分散）。

但网络模型中，约化形式的残差：

$$
\varepsilon_t^* = (I - \rho W)^{-1}\eta_t
$$

$(I-\rho W)^{-1}$ 将独立的 $\eta_t$（$\Omega_\eta$ 对角）搅混成相关的 $\varepsilon_t^*$（$\Omega_{\varepsilon^*}$ 非对角 $\neq 0$，见 §3.5.2）。组合特质风险变为：

$$
\frac{\bar{\sigma}^2}{K} \longrightarrow \frac{\bar{\sigma}^2}{K(1-\rho)^2}
$$

|          | 传统模型              | 网络模型（$\rho > 0$）                                         | 网络模型（$\rho < 0$）                        |
| :------- | :-------------------- | :--------------------------------------------------------------- | :---------------------------------------------- |
| 因子风险 | 不可分散              | 不可分散（$\beta^*$ 放大，量变）                               | 不可分散（$\beta^*$ 缩小，量变）              |
| 特质风险 | **可分散** → 0 | **难以分散**，需 $\frac{1}{(1-\rho)^2}$ 倍资产（⭐质变） | **更易分散**，$(1-\rho)^2 > 1$ 加速收敛 |

**这才是网络创造系统性风险的本质**：$\rho > 0$ 让本来可以分散掉的特质风险变得难以分散——个体冲击通过网络变成集体冲击。$\rho < 0$ 反过来：$(1-\rho)^2 > 1$（如 $\rho = -0.3$ 时 $(1-(-0.3))^2 = 1.69$），分母变大，特质风险更容易分散，因此**削弱系统性风险**。

> **总结**：$\beta^*$ 放大是"量变"（不可分散的东西变更大），特质风险从可分散变成难以分散是"质变"——后者才是网络效应对系统性风险的核心贡献。

**参数量**：从 1 个 $\rho$ 增加到 $K$ 个 $\rho_i$。

### 5.2 时变网络（§3.3, Eq.24）

允许网络结构随时间变化：

$$
A_t = I - \Lambda W_t \tag{24}
$$

**后果**：

- 约化形式 beta 变成时变的：$\beta_t^* = (I - \Lambda W_t)^{-1} \bar{\beta}$
- 即使结构性 $\bar{\beta}$ 不变，由于 $W_t$ 在变，$\beta_t^*$ 也在变
- 本文实证中 $W_t$ 来自年度更新的 I/O 表

### 5.3 与 Bonaccolto et al. (2019) 的关系——关键联系

|            | Billio (2023)      | Bonaccolto (2019)                                                |
| ---------- | ------------------ | ---------------------------------------------------------------- |
| $A$ 矩阵 | $I - \Lambda W$  | $I - \Lambda \sum_{j=1}^d \delta_j W_j$                        |
| 网络数量   | 1 个$W$          | $d$ 个 $W_j$，加权组合                                       |
| 新增参数   | 无                 | $\delta_j$（各层权重，$\delta_j \geq 0, \sum \delta_j = 1$） |
| 解读       | 一种网络的效应强度 | **哪种网络最重要？**                                       |

Bonaccolto 的 $A = I - \Lambda W^*$，其中**复合网络** $W^* = \sum_j \delta_j W_j$。

本质上，Billio 是 Bonaccolto 的**特例**（$d = 1$, $\delta_1 = 1$）。

### 5.4 同期性问题：W 从哪来？两篇论文的不同策略

模型 $A = I - \Lambda W$ 描述的是**同期（contemporaneous）**关系：资产 $i$ 在时刻 $t$ 的收益同时受到其他资产在**同一时刻 $t$** 收益的影响。这意味着 $W$ 必须代表某种**同期联系**。

两篇论文用了完全不同的策略获取 $W$：

#### 5.4.1 Billio (2023)：外生经济数据（I/O 表）

- $W$ 来自美国经济分析局（BEA）的投入产出表——记录"行业 $i$ 给行业 $j$ 供了多少货"
- I/O 表是**外生数据**，不是从收益率统计出来的 → **天然没有内生性问题**
- 额外使用**上一年**的 I/O 表预测当年回报（引入一年滞后）→ 进一步避免内生性
- 投入产出关系本身就是同期的（行业 $i$ 今天给行业 $j$ 供货 → 行业 $j$ 今天的产出就受影响）

#### 5.4.2 Bonaccolto (2019)：统计因果检验 + 频率差

Bonaccolto 的 $W$ 来自 Granger 因果检验和分位数因果检验。

**Granger 因果本质上是滞后关系**：测试的是"$i$ 昨天的收益率能否预测 $j$ 今天的收益率"（lag=1）。这不是同期关系。

**那怎么把滞后的 Granger 因果塞进同期模型 $A = I - \Lambda W$ 里？**

答案是**频率差**：

| 步骤                                        | 频率           | 数据            |
| ------------------------------------------- | -------------- | --------------- |
| Step 1：估计因果网络$W$                   | **日度** | 日收益率，lag=1 |
| Step 2：估计因子模型$A = I - \Lambda W^*$ | **周度** | 周收益率        |

日度的 lag-1 因果关系，在周度视角下变成了同期关系。直觉：

- 周一：银行 A 的收益率变动
- 周二：通过 lag-1 因果传导到银行 B
- 周三：再传导到银行 C
- ...
- 把一周打包来看：A、B、C 在**同一周**内都动了 → 周度频率下是**同期关联**

所以 Bonaccolto 的摘要说 "the standard Granger causality **to detect for the presence of contemporaneous links**"——Granger 因果用来检测同期联系。看似矛盾，但在日→周的频率转换下逻辑成立。

**为什么不直接用日度数据估因子模型？** 原文（p.11）："We do not consider daily returns as they would require the introduction of heteroskedastic dynamics in the variance of the residuals." 日度收益率有很强的 GARCH 效应，如果用日度估因子模型，残差的方差结构太复杂。周度数据规避了这个问题。

#### 5.4.3 对比总结

| 维度             | Billio (2023)                  | Bonaccolto (2019)              |
| ---------------- | ------------------------------ | ------------------------------ |
| $W$ 来源       | 投入产出表（外生经济数据）     | Granger 因果检验（统计估计）   |
| $W$ 的估计频率 | 年度（一年一张表）             | 全样本（整个样本期一个$W$）  |
| 因子模型频率     | 月度                           | 周度                           |
| 同期性解决方案   | 天然同期（供需关系就是同期的） | 频率差（日度 lag → 周度同期） |
| 内生性风险       | 低（$W$ 外生 + 滞后一年）    | 较高（$W$ 从收益率估出来）   |
| 时变性           | 有（每年一张$W_t$）          | 无（全样本一个$W$）          |

**对你的项目的启示**：

- DY 溢出网络用日度/周度数据估计 → 在周度/月度因子模型中使用，频率差逻辑与 Bonaccolto 类似
- 如果要加入投入产出表作为额外的网络层（$W_3$），它天然满足外生性条件，是很好的补充
- 多层网络组合 $W^* = \delta_1 W_{DY,ret} + \delta_2 W_{DY,vol} + \delta_3 W_{IO}$ 可以同时利用统计因果和经济关联

---

## 6. §4 集中极大似然估计（Concentrated MLE）— 逐段精讲

> 本节对应论文 §4 Model estimation（p.206-208）的每一段原文，逐句解释。

### 6.0 总览：我们要估什么？

论文开篇列出了所有待估参数：

| 参数                                         | 维度           | 含义                                      |
| -------------------------------------------- | -------------- | ----------------------------------------- |
| $\bar{\alpha}$                             | $K \times 1$ | 每个行业的截距（代替$\mathbb{E}[R_t]$） |
| $\bar{\beta}$                              | $K \times M$ | 每个行业对每个因子的**结构性暴露**  |
| $\Lambda = \text{diag}(\rho_1,...,\rho_K)$ | $K$ 个值     | 每个行业对网络冲击的**易感性**      |
| $\Omega_\eta$                              | $K$ 个值     | 每个行业的结构性残差方差（对角阵）        |

参数总数 $\Theta = (\bar{\alpha}, \bar{\beta}, \Lambda, \Omega_\eta)$：

$$
K + KM + K + K = K(3 + M) \text{ 个参数}
$$

如果 $K = 15, M = 4$：$15 \times 7 = 105$ 个参数。

论文说估计分两步：(1) 先估模型参数 $\Theta$；(2) 再用 Fama-MacBeth 两步法估风险溢价 $\Lambda_{premium}$。

### 6.1 起点：联立方程模型（Eq.27）

$$
A R_t = \bar{\alpha} + \bar{\beta} F_t + \eta_t \tag{27}
$$

**"simultaneous equations model"**（联立方程模型）：所有行业的收益率在同一时点 $t$ **同时、互相影响**。

用 $K=3$ 展开，$A = I - \Lambda W$：

$$
\begin{pmatrix} 1 & -\rho_1 W_{12} & -\rho_1 W_{13} \\ -\rho_2 W_{21} & 1 & -\rho_2 W_{23} \\ -\rho_3 W_{31} & -\rho_3 W_{32} & 1 \end{pmatrix} \begin{pmatrix} R_1 \\ R_2 \\ R_3 \end{pmatrix} = \begin{pmatrix} \bar{\alpha}_1 \\ \bar{\alpha}_2 \\ \bar{\alpha}_3 \end{pmatrix} + \bar{\beta} F_t + \begin{pmatrix} \eta_1 \\ \eta_2 \\ \eta_3 \end{pmatrix}
$$

### 6.2 识别条件（Eq.28）

**"识别"（identification）= 数据是否能唯一确定参数。**

类比：$x + y = 10$ 一个方程两个未知数 → 无穷多解 → 不可识别。$x+y=10, x-y=2$ → 唯一解 → 可识别。

论文的**阶条件**（order condition）：结构模型的参数数量 $\leq$ 约化形式模型的参数数量。

约化形式（两边左乘 $A^{-1}$）：

$$
R_t = A^{-1}\bar{\alpha} + A^{-1}\bar{\beta}F_t + A^{-1}\eta_t = \alpha^* + \beta^* F_t + \varepsilon^*_t \tag{28}
$$

约化形式能提供的信息量：

| 参数类型   | 数量                  | 来源                             |
| ---------- | --------------------- | -------------------------------- |
| 均值参数   | $K(M+1)$            | $K$ 个截距 + $KM$ 个因子载荷 |
| 协方差参数 | $\frac{1}{2}K(K+1)$ | 残差协方差矩阵（对称阵）         |

#### 网络如何减少参数

不用网络假设：$A$ 有 $K^2$ 个自由参数（$K=15$ 时 225 个）。
用 $A = I - \Lambda W, \quad \Lambda = \text{diag}(\rho_1, \rho_2, \ldots, \rho_K)$：$\Lambda$ 只有 $K = 15$ 个参数。**从 225 个减到 15 个！**

但这还不够满足阶条件。论文的解决办法：

**假设 $\Omega_\eta$ 是对角阵**（不同行业的结构性残差独立）。

- 协方差参数从 $\frac{K(K+1)}{2} = 120$ 个减到 $K = 15$ 个
- 提供了 $\frac{K(K+1)}{2} - K = 105$ 个额外约束
- 经济含义：如果正确建模了网络效应 $\Lambda W$，剩下的残差就**应该**独立

### 6.3 假设 4.1：ρ 的取值范围（Eq.29-31）

#### 前置知识：特征值速览

**什么是特征值/特征向量？** 给一个矩阵 $W$（$K \times K$），如果存在非零向量 $v$ 和标量 $\lambda$ 使得：

$$
W v = \lambda v
$$

则 $\lambda$ 叫**特征值**（eigenvalue），$v$ 叫**特征向量**（eigenvector）。

直觉：矩阵乘以向量通常改变方向又改变长度。但特征向量是特殊方向——矩阵乘上去后**方向不变**，只**伸缩 $\lambda$ 倍**。一个 $K \times K$ 矩阵有 $K$ 个特征值。

**行标准化矩阵的特征值**：$W$ 行标准化（每行之和=1）意味着 $W \cdot \mathbf{1} = \mathbf{1}$（$\mathbf{1}=(1,...,1)'$）。对比定义 $Wv = \lambda v$，取 $v=\mathbf{1}$，得 $\lambda=1$。所以 **$\lambda_{max}=1$ 是行标准化矩阵的必然结果**。

**关键的线性代数事实**：如果 $Wv = \lambda_i v$，那么

$$
(I - \rho W)v = v - \rho Wv = v - \rho\lambda_i v = (1-\rho\lambda_i)v
$$

所以 $I - \rho W$ 的特征值 = $1 - \rho\lambda_i$。矩阵行列式 = 所有特征值之积：

$$
\det(I - \rho W) = \prod_{i=1}^K (1 - \rho\lambda_i)
$$

**$\det = 0$ 当且仅当某个 $(1-\rho\lambda_i)=0$，即 $\rho = 1/\lambda_i$。** 所以 $\rho$ 不能等于任何特征值的倒数，否则矩阵不可逆。

#### ρ 的约束推导

$A = I - \rho W$ 必须可逆（$\det(A) \neq 0$），否则约化形式不存在。

为了让所有 $(1-\rho\lambda_i) > 0$（不只不为零，还要为正，保证稳定性），需要：

$$
\frac{1}{\lambda_{min}} < \rho < \frac{1}{\lambda_{max}}
$$

其中 $\lambda_{min}, \lambda_{max}$ 是 $W$ 的最小和最大特征值。

**数值验证**：假设 $W$ 有 4 个特征值 $\lambda_1=-0.5, \lambda_2=0.2, \lambda_3=0.6, \lambda_4=1.0$，对每个列出约束：

| 特征值$\lambda_i$ | 约束$1 - \rho\lambda_i > 0$ | 解出$\rho$                            |
| ------------------- | ----------------------------- | --------------------------------------- |
| $-0.5$            | $1+0.5\rho > 0$             | $\rho > -2$（即 $> 1/\lambda_1$）   |
| $0.2$             | $1-0.2\rho > 0$             | $\rho < 5$（即 $< 1/\lambda_2$）    |
| $0.6$             | $1-0.6\rho > 0$             | $\rho < 1.67$（即 $< 1/\lambda_3$） |
| $1.0$             | $1-\rho > 0$                | $\rho < 1$（即 $< 1/\lambda_4$）    |

同时满足所有约束 → 上界取最紧 $\min(5, 1.67, 1) = 1 = 1/\lambda_{max}$ ✓，下界取最紧 $-2 = 1/\lambda_{min}$ ✓。

结论：两端特征值 $\lambda_{min}$ 和 $\lambda_{max}$ 决定了 $\rho$ 的合法区间，中间特征值的约束自动被包含。

**代入行标准化的情况**：

| 特征值            | 值                      | $\rho$ 的约束          |
| ----------------- | ----------------------- | ------------------------ |
| $\lambda_{max}$ | $= 1$（行标准化必然） | $\rho < 1/1 = 1$       |
| $\lambda_{min}$ | 通常是负数，比如$-2$  | $\rho > 1/(-2) = -0.5$ |

经济上 $\rho_i > 0$（传染效应为正），所以实际约束通常是 $\rho \in (0, 1)$。

#### 公式的隐含前提：$\lambda_{min}$ 必须为负

如果所有特征值都是正的（如 $\lambda = 0.2, 0.5, 0.6, 1.0$），会怎样？

所有约束都是上界（$\rho < 1/\lambda_i$），没有下界。此时公式 $\frac{1}{\lambda_{min}} < \rho < \frac{1}{\lambda_{max}}$ 给出 $5 < \rho < 1$（空集），显然不对——正确答案应该是 $\rho < 1$。

**原因**：公式**隐含假设 $\lambda_{min} < 0$**。负特征值产生下界 $\rho > 1/\lambda_{min}$（负数），正特征值产生上界 $\rho < 1/\lambda_{max}$。没有负特征值就没有来自特征值的下界。

**为什么论文可以直接用？** 行标准化的网络矩阵（$W_{ij} \geq 0$，对角线 $W_{ii}=0$）**必然存在负特征值**。原因：$\text{tr}(W) = \sum_i W_{ii} = 0$（对角线全零），而特征值之和 = 迹（trace），即 $\sum_i \lambda_i = 0$。既然 $\lambda_{max}=1 > 0$，要让总和为零，**必然有负特征值来抵消**。

#### 时变网络的约束（假设4.1）

如果 $W_t$ 随时间变，每个时点有不同特征值。$\rho$ 必须让**每个时点**的 $I - \rho W_t$ 都可逆 → 取最紧的约束：

$$
\bar{\lambda}^{-1}_{min} < \rho < \bar{\lambda}^{-1}_{max} \tag{29}
$$

$$
\bar{\lambda}_{max} = \min_t\{\lambda_{t,max}\} \tag{30}
$$

$$
\bar{\lambda}_{min} = \max_t\{\lambda_{t,min}\} \tag{31}
$$

类比：车要通过很多门洞，必须比**最窄的门洞**还窄。$\bar{\lambda}_{max}$ 就是最窄的门洞。

### 6.4 假设 4.2-4.3：异质 ρ 的约束

**假设 4.2**：对异质 $\rho_i$ 的情况，要求 $I - \Lambda W_t$ 对每个时点 $t$ 都可逆。实际操作中在数值优化时检查 $\det(I - \Lambda W_t)$。

**假设 4.3**：如果行业 $j$ 在**所有时点**都没有任何网络连接（$W_t$ 的第 $j$ 行全为零），则 $\rho_j$ 必须限制为零——因为 $\rho_j \times 0 = 0$，无论 $\rho_j$ 多少结果都一样，参数不可识别。

类比：孤岛居民无法估计其"社交传染系数"。使用 DY 溢出网络（全连通）时此假设自动满足。

### 6.5 为什么用集中似然

论文说：在 $A$ 的结构假设和 $\Omega_\eta$ 对角两个限制下，可以用**全信息极大似然**（FIML）。但 $K$ 稍大时参数太多（$K=15, M=4$ 有 105 个），直接优化困难。

解决方案：空间计量经济学的成熟技巧——**集中似然**（concentrated likelihood）。引用 Elhorst (2003) 和 LeSage & Pace (2009)。

核心思想：把一部分参数用**解析解**（OLS）代替，使需要数值优化的参数大幅减少。

### 6.6 对数似然函数（Eq.33-35）

假设 $\eta_t \sim \mathcal{N}(0, \Omega_\eta)$（正态分布，仅为数学方便——脚注14说即使数据非正态，也可以用拟极大似然 QMLE）。

**$\eta_t$ 的身份澄清**：$\eta_t$ 是**结构方程**的残差，不是约化形式的残差：

$$
\underbrace{(I - \rho W)}_A (R_t - \mathbb{E}[R_t]) = \bar{\beta} F_t + \eta_t \quad \leftarrow \text{结构方程}
$$

约化形式 $R_t - \mathbb{E}[R_t] = \beta^* F_t + \varepsilon_t^*$ 的残差是 $\varepsilon_t^* = (I-\rho W)^{-1}\eta_t$。

|            | $\eta_t$（结构残差）                 | $\varepsilon_t^*$（约化残差）             |
| :--------- | :------------------------------------- | :------------------------------------------ |
| 所属方程   | 结构方程（左边乘了$A$）              | 约化形式（解出$R_t$）                     |
| 协方差矩阵 | $\Omega_\eta$ **对角**（假设） | $\Omega_{\varepsilon^*}$ **非对角** |
| 含义       | 过滤掉网络效应后的"纯"特质冲击         | 包含网络传染的观测残差                      |

$\Omega_\eta$ 可以假设为对角，正是因为结构方程左边的 $A = (I-\rho W)$ 已经把行业间的网络溢出"剥离"了——剩下的 $\eta_t$ 是每个行业自己的独立冲击。MLE 估计的正是这个 $\Omega_\eta$（对角矩阵，只有 $K$ 个参数）。

#### 从多元正态密度函数推导 Eq.(33)-(34)

**第一步**：$K$ 维正态分布 $\eta \sim \mathcal{N}(0, \Omega)$ 在 $\eta = e$ 处的密度函数：

$$
f(e) = \frac{1}{(2\pi)^{K/2} \cdot |\Omega|^{1/2}} \cdot \exp\left(-\frac{1}{2} e' \Omega^{-1} e\right)
$$

| 部分                 | 含义                                                                            |
| -------------------- | ------------------------------------------------------------------------------- |
| $(2\pi)^{K/2}$     | 归一化常数（让密度积分=1），和参数无关                                          |
| $\|\Omega\|^{1/2}$ | 协方差矩阵行列式的平方根，衡量"散布体积"                                        |
| $e'\Omega^{-1}e$   | **马氏距离**（Mahalanobis distance）的平方，衡量 $e$ 离均值的标准化距离 |

**第二步**：取对数，得单个时点的对数似然：

$$
\log f(e) = \underbrace{-\frac{K}{2}\log(2\pi)}_{\text{常数，不含参数}} - \frac{1}{2}\log|\Omega| - \frac{1}{2} e'\Omega^{-1}e
$$

第一项不含任何待估参数，对"找最大值"无影响，用 $\propto$（正比于）省掉：

$$
l_t(\Theta) \propto -\frac{1}{2}\log|\Omega| - \frac{1}{2} e'_t \Omega^{-1} e_t \tag{34}
$$

**第三步**：$T$ 个时点独立，联合概率 = 各时点之积，取对数后积变和：

$$
f(e_1,...,e_T) = \prod_{t=1}^T f(e_t) \implies L(\Theta) = \log\prod_t f(e_t) = \sum_{t=1}^T \log f(e_t) = \sum_{t=1}^T l_t(\Theta) \tag{33}
$$

**推导链**：$\eta_t \sim \mathcal{N}(0,\Omega)$ → 密度函数 → 取对数 → 去常数 → Eq.(34) → 时间独立连乘变连加 → Eq.(33)。

#### Eq.(34) 各项的含义

**第一项** $-\frac{1}{2}\log|\Omega|$：

- $|\Omega|$ = 行列式。$\Omega$ 对角 → $|\Omega| = \prod_i \sigma_i^2$
- 衡量残差的总体"散布程度"。方差越大 → 似然越小
- 直觉：模型解释力差 → 残差大 → 似然低

**第二项** $-\frac{1}{2}e'_t \Omega^{-1} e_t$：

- $\Omega$ 对角 → $\Omega^{-1} = \text{diag}(1/\sigma_1^2, ..., 1/\sigma_K^2)$
- 展开：$e'_t \Omega^{-1} e_t = \sum_{i=1}^K \frac{e_{i,t}^2}{\sigma_i^2}$（标准化残差平方和）
- 残差越小 → 似然越大

$$
e_t = R_t - \bar{\alpha} - \Lambda W R_t - \bar{\beta} F_t \tag{35}
$$

残差 = 实际收益率 - 截距 - 网络效应 - 因子效应。展开第 $i$ 个行业：

$$
e_{i,t} = R_{i,t} - \bar{\alpha}_i - \rho_i \sum_{j=1}^K W_{ij} R_{j,t} - \sum_{m=1}^M \bar{\beta}_{i,m} F_{m,t}
$$

注意 Eq.(35) 等价于结构模型 $AR_t = \bar{\alpha} + \bar{\beta}F_t + \eta_t$ 的移项形式（$A = I - \Lambda W$，$AR_t = R_t - \Lambda W R_t$）。

### 6.7 集中似然的核心技巧（Eq.36-37）— ⭐⭐⭐最重要

**关键假设**：如果 $\Lambda$（即所有 $\rho_i$）**已知**。

定义"滤波后的收益率"：

$$
Z_t \equiv R_t - \Lambda W_t R_t = (I - \Lambda W_t) R_t \tag{36}
$$

$Z_t$ 的含义：从收益率中**减去网络效应**后的残余。$\Lambda$ 已知 + $W_t$ 已知 + $R_t$ 已知 → $Z_t$ 可以计算。

代入模型：原模型 $(I-\Lambda W)R_t = \bar{\alpha} + \bar{\beta}F_t + \eta_t$，左边 = $Z_t$：

$$
Z_t = \bar{\alpha} + \bar{\beta} F_t + \eta_t
$$

**这就是标准的多元线性回归！** OLS 直接给出解析解：

定义 $X_t = [1, F_t']$（常数项 + 因子），$X = [X'_1, ..., X'_T]'$（堆叠所有时点），$Z = [Z'_1, ..., Z'_T]'$：

$$
\begin{pmatrix} \hat{\bar{\alpha}}(\Lambda) \\ \hat{\bar{\beta}}(\Lambda) \end{pmatrix} = (X'X)^{-1} X'Z \tag{37}
$$

**注意标记 $\hat{\bar{\alpha}}(\Lambda)$**：圆括号表示这些估计值是 $\Lambda$ 的函数。不同的 $\Lambda$ → 不同的 $Z_t$ → 不同的 OLS 结果。

方差估计（对残差取对角方差）：

$$
\hat{\eta}_t(\Lambda) = Z_t - \hat{\bar{\alpha}}(\Lambda) - \hat{\bar{\beta}}(\Lambda) F_t
$$

$$
\hat{\sigma}_i^2(\Lambda) = \frac{1}{T}\sum_{t=1}^T \hat{\eta}_{i,t}^2, \quad \hat{\Omega}(\Lambda) = \text{diag}(\hat{\sigma}_1^2, ..., \hat{\sigma}_K^2)
$$

#### 6.7.1 数据结构具象化（以 K=31, T=52, n_f=1 为例）

**$Z_t$**：单个时点 $t$ 的变换收益率，长度 31 的向量

$$
z_{i,t} = r_{i,t} - \rho_i \sum_j W_{ij} r_{j,t} \quad \Rightarrow \quad Z_t = \begin{pmatrix} z_{1,t} \\ z_{2,t} \\ \vdots \\ z_{31,t} \end{pmatrix}_{31 \times 1}
$$

**$\hat{\eta}_t$**：对单个时点 $t$，每个元素就是"变换收益率 - 截距 - 因子项"：

$$
\hat{\eta}_{i,t} = z_{i,t} - \hat{\bar{\alpha}}_i - \hat{\bar{\beta}}_i \cdot F_t
$$

| 符号                   | 维度  | 含义                                          |
| :--------------------- | :---- | :-------------------------------------------- |
| $Z_t$                | 31×1 | 第$t$ 周 31 个行业的变换收益率              |
| $\hat{\bar{\alpha}}$ | 31×1 | 31 个截距（OLS 估出，全样本固定）             |
| $\hat{\bar{\beta}}$  | 31×1 | 31 个因子载荷（$n_f=1$ 时每个行业一个标量） |
| $F_t$                | 标量  | 第$t$ 周的市场因子值                        |
| $\hat{\eta}_t$       | 31×1 | 第$t$ 周 31 个行业的结构残差                |

把所有 $T=52$ 周堆起来，$\hat{\eta}$ 是一个 $T \times K$ 矩阵：

$$
\underbrace{\hat{\eta}}_{52 \times 31} = \begin{pmatrix}
\hat{\eta}_{1,1} & \hat{\eta}_{2,1} & \cdots & \hat{\eta}_{31,1} \\
\hat{\eta}_{1,2} & \hat{\eta}_{2,2} & \cdots & \hat{\eta}_{31,2} \\
\vdots & \vdots & \ddots & \vdots \\
\hat{\eta}_{1,52} & \hat{\eta}_{2,52} & \cdots & \hat{\eta}_{31,52}
\end{pmatrix}
$$

每一行 = 一周，每一列 = 一个行业。

**$\hat{\sigma}_i^2$**：沿着**列方向**（时间维度）求平方的均值——取第 $i$ 列的 52 个数，各自平方再平均：

$$
\hat{\sigma}_1^2 = \frac{\hat{\eta}_{1,1}^2 + \hat{\eta}_{1,2}^2 + \cdots + \hat{\eta}_{1,52}^2}{52}
$$

对 31 个行业各算一个，得到 31 个方差，摆上对角线：

$$
\hat{\Omega} = \begin{pmatrix}
\hat{\sigma}_1^2 & 0 & \cdots & 0 \\
0 & \hat{\sigma}_2^2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \hat{\sigma}_{31}^2
\end{pmatrix}_{31 \times 31}
$$

非对角全是 0——因为结构假设 $\eta_t$ 各行业独立（网络效应已被 $A$ 剥离，见 §6.6）。

**对应代码**（`multilayer_network_mle.py` 第107-122行）：

```python
R_tilde = (A @ R.T).T              # Z_t: (52, 31)
eta_hat = R_tilde - F_aug @ B_hat.T  # η̂:  (52, 31)
sigma_sq = (eta_hat**2).mean(axis=0)  # σ̂²: (31,)  ← 沿列求平方均值
Sigma_eta = np.diag(sigma_sq)         # Ω̂:  (31,31) 对角矩阵
```

### 6.8 代回似然 → 只剩 Λ

**疑问**：Eq.(34) 里只看到 $\Omega$ 和 $e_t$，$\bar{\alpha}$ 和 $\bar{\beta}$ 在哪？

**答**：它们**藏在 $e_t$ 里面**。因为 $e_t = R_t - \bar{\alpha} - \Lambda W R_t - \bar{\beta} F_t$（Eq.35），所以 $e_t$ 是 $(\bar{\alpha}, \bar{\beta}, \Lambda)$ 三者的函数。Eq.(34) 实际上依赖所有四组参数：

$$
l_t(\underbrace{\bar{\alpha}, \bar{\beta}}_{\text{通过 } e_t \text{ 进入}},\; \underbrace{\Lambda}_{\text{通过 } e_t \text{ 进入}},\; \underbrace{\Omega}_{\text{直接出现}})
$$

把 OLS 解 $\hat{\bar{\alpha}}(\Lambda), \hat{\bar{\beta}}(\Lambda)$ 代入 $e_t$，残差变成只依赖 $\Lambda$ 的量：

$$
\hat{e}_t(\Lambda) = R_t - \hat{\bar{\alpha}}(\Lambda) - \Lambda W R_t - \hat{\bar{\beta}}(\Lambda) F_t
$$

再把 $\hat{\Omega}(\Lambda)$ 也代入，整个 $l_t$ 只剩 $\Lambda$：

$$
l_t = -\frac{1}{2}\log|\hat{\Omega}(\Lambda)| - \frac{1}{2} \hat{e}'_t(\Lambda) \; \hat{\Omega}^{-1}(\Lambda) \; \hat{e}_t(\Lambda)
$$

总似然即为集中似然：

$$
L_{\text{conc}}(\Lambda) = \sum_{t=1}^T l_t = L\big(\hat{\bar{\alpha}}(\Lambda),\; \hat{\bar{\beta}}(\Lambda),\; \Lambda,\; \hat{\Omega}(\Lambda)\big)
$$

整个似然函数现在**只依赖于 $\Lambda$**！

**参数维度对比**：

|                    | 原始似然$L(\Theta)$ | 集中似然$L_{conc}(\Lambda)$ |
| ------------------ | --------------------- | ----------------------------- |
| 需要数值优化的参数 | $K(3+M)$            | **$K$**               |
| $K=15, M=4$      | 105                   | **15**                  |
| $K=31, M=1$      | 124                   | **31**                  |

脚注15说：集中 $\bar{\alpha}, \bar{\beta}$ 后剩 $2K$ 个参数；再集中 $\Omega$ 后只剩 $K$ 个。标准误可以从完整似然的海森矩阵数值计算。

### 6.9 时变网络（Eq.38）

如果 $W_t$ 随时间变化，唯一的区别是 $Z_t$ 的定义用 $W_t$：

$$
Z_t = R_t - \Lambda W_t R_t \tag{38}
$$

其余步骤完全相同。**这就是为什么 DY 时变网络可以直接嵌入此框架。**

### 6.10 正态性假设的说明（脚注14）

论文承认金融数据不是正态的。但：

- 正态假设只是为了数学方便（似然函数有闭式表达）
- 即使数据非正态，用正态似然做估计 = **拟极大似然**（Quasi-MLE / QMLE）
- QMLE 在温和条件下仍然是一致估计量（大样本收敛到真值）

### 6.11 完整算法伪代码

```
输入：R (T×K 收益率), F (T×M 因子), W_t (K×K 网络矩阵序列)

function concentrated_loglik(ρ₁, ρ₂, ..., ρ_K):
  
    # Step 1: 构造 Λ 对角矩阵
    Λ = diag(ρ₁, ρ₂, ..., ρ_K)
  
    # Step 2: 减去网络效应，构造 Z_t
    for t = 1, ..., T:
        Z_t = R_t - Λ W_t R_t         # "滤波后的收益率"
  
    # Step 3: 标准 OLS
    X = [1, F]                          # 设计矩阵 (T × (M+1))
    [α̂, β̂] = (X'X)⁻¹ X'Z              # OLS 解析解
  
    # Step 4: 计算残差
    for t = 1, ..., T:
        η̂_t = Z_t - α̂ - β̂ F_t
  
    # Step 5: 方差估计
    for i = 1, ..., K:
        σ̂²_i = (1/T) Σ_t η̂²_{i,t}     # 第 i 个行业的残差方差
    Ω̂ = diag(σ̂²₁, σ̂²₂, ..., σ̂²_K)
  
    # Step 6: 计算对数似然值
    loglik = -T/2 × Σ_i log(σ̂²_i) - 1/2 × Σ_t Σ_i η̂²_{i,t}/σ̂²_i
  
    return loglik

# 主程序：数值优化
ρ̂ = argmax_{ρ₁,...,ρ_K} concentrated_loglik(ρ₁, ..., ρ_K)
    subject to: I - diag(ρ) × W_t 对所有 t 可逆

# 最优解找到后：
Λ̂ = diag(ρ̂₁, ..., ρ̂_K)
α̂_final = α̂(Λ̂)       # 用最优 Λ̂ 算最终的 OLS 解
β̂_final = β̂(Λ̂)
Ω̂_final = Ω̂(Λ̂)
```

### 6.12 与 Bonaccolto (2019) 的完全类比

| 步骤           | Billio (2023)                         | Bonaccolto (2019)                               | 你的代码                        |
| -------------- | ------------------------------------- | ----------------------------------------------- | ------------------------------- |
| 网络           | $\Lambda W_t$                       | $\Lambda (\sum \delta_j W_j)_t$               | `Lambda @ W_star`             |
| 待搜索参数     | $K$ 个 $\rho_i$                   | $K$ 个 $\rho_i$ + $(d-1)$ 个 $\delta_j$ | `scipy.optimize.minimize`     |
| 构造$Z_t$    | $R_t - \Lambda W R_t$               | $R_t - \Lambda (\sum \delta_j W_j) R_t$       | `Z = R - Lambda @ W_star @ R` |
| 给定搜索参数后 | OLS 估计$\bar{\alpha}, \bar{\beta}$ | OLS 估计$\bar{\alpha}, \bar{\beta}$           | `np.linalg.lstsq`             |
| 方差估计       | $\hat{\Omega}_\eta$ 对角            | $\hat{\Sigma}_\eta$ 对角                      | `np.var(residuals)`           |
| 代回似然       | 集中似然关于$\Lambda$               | 集中似然关于$(\delta, \Lambda)$               | `concentrated_loglik()`       |
| 约束           | $A_t$ 可逆                          |  $$ 可逆 + $\delta \geq 0, \sum\delta=1$      | bounds + constraints            |

**数学框架完全相同！** Bonaccolto 只是多了一组 $\delta$ 参数（网络层权重）。

---

## 7. §4.1 Fama-MacBeth 风险溢价估计

### 7.1 传统两步法

**第一步**（时间序列回归）：对每个资产 $i$，用全部时间序列估计 $\beta_i$：

$$
R_{i,t} = \alpha_i + \beta_i F_t + \varepsilon_{i,t}, \quad t = 1, ..., T
$$

**第二步**（截面回归）：用估计的 $\hat{\beta}$ 解释资产的平均超额收益：

$$
\bar{R}^e = \hat{\beta} \Lambda + \nu \tag{42}
$$

$\Lambda$ 就是**风险溢价**（risk premium）。

### 7.2 网络模型下的修正

在时变网络的情况下，论文的方法是**先过滤掉网络效应**（Eq.48）：

$$
Z_t = (I - \hat{\Lambda} W_t)(R_t - r_f) \tag{48}
$$

然后用 $Z_t$ 的时间均值 $\bar{Z}^e$ 和结构性 $\hat{\bar{\beta}}$ 做截面回归：

$$
\bar{Z}^e = \hat{\bar{\beta}} \Lambda + \nu \tag{49}
$$

**好处**：用结构性 beta（不含网络效应）估计风险溢价，消除了网络导致的 beta 膨胀。

---

## 8. §5 模拟分析（Monte Carlo Simulation）——逐句详解

> **为什么要做模拟？** 模拟（Monte Carlo）的作用是：我们**自己编造数据**，这些数据的"真实答案"（真实参数值）是我们自己设定的，然后用估计方法去估计参数，看估计出来的值和真实值差多远。如果差得很小且样本越大差距越小，说明估计方法是靠谱的（一致性）。如果忽略网络会导致什么后果，模拟也能直接展示出来。

---

### 8.1 模拟设计一：标量 $\rho$，固定 $W$（§5.1）

#### 8.1.1 数据生成过程（DGP）——逐行解释

论文的 DGP 就是 Eq.(51)：

$$
(I - \rho W)(R_t - E[R_t]) = \bar{\beta} F_t + \eta_t
$$

**翻译成人话**：我们用这个公式"制造"出模拟的收益率数据 $R_t$。下面逐个参数解释为什么这样设定。

---

**参数1：资产数 $K = 100$**

选100个资产。为什么？因为要测试"大截面"下的表现（实际金融数据通常有几十到几百只股票/行业）。100够大，能看出截面效应。

---

**参数2：$\rho \in \{0, 0.25, 0.5, 0.75\}$**

网络效应强度取4个值。

- $\rho = 0$：没有网络效应，模型退化为传统因子模型——这是**对照组**
- $\rho = 0.25$：弱网络效应
- $\rho = 0.5$：中等网络效应
- $\rho = 0.75$：强网络效应

为什么不取 $\rho = 1$？因为行标准化的 $W$ 最大特征值 $\lambda_{\max} = 1$，$\rho$ 必须严格小于 $1/\lambda_{\max} = 1$（否则 $(I - \rho W)$ 不可逆）。

---

**参数3：因子载荷 $\beta_i \sim \text{Uniform}(0.8, 1.2)$**

$$
\beta_i \sim U(0.8, 1.2), \quad i = 1, 2, \ldots, K
$$

**为什么用均匀分布？** 均匀分布 $U(a, b)$ 的意思是：在 $[a, b]$ 区间内每个值被抽到的概率完全相等。这里用它是为了：

- 让每个资产的 $\beta$ 值不同（异质性），但又不会差太多
- 均值 $= (0.8 + 1.2)/2 = 1.0$，和市场 $\beta = 1$ 一致
- 方差很小 $= (1.2 - 0.8)^2/12 = 0.0133$，说明 $\beta$ 集中在1附近

**为什么不用正态分布？** 后面的稳健性检验也试了正态分布 $\beta_i \sim N(1, \sigma^2)$，结论不变。用均匀分布是因为它不会产生极端值（比如负的 $\beta$），让基准设定更简单。

---

**参数4：因子收益率 $F_t \sim N(\mu_F, \sigma_F^2)$，$\mu_F = 0$，$\sigma_F = 15\%$（年化）**

$$
F_t \sim N(0, \sigma_F^2)
$$

**为什么用正态分布？** 这是金融学中最标准的假设。因子收益率可正可负，围绕0对称分布。正态分布是最简单的、有理论支撑的选择（中心极限定理）。

**为什么 $\mu_F = 0$？** 因子均值设为0是因为模型中已经把期望收益 $E[R_t]$ 单独拿出来了（减去了均值），所以因子 $F_t$ 驱动的是**偏离期望的波动**，不含均值。

**$\sigma_F = 15\%$ 是什么意思？** 15% 是年化标准差。实际中标普500指数的年化波动率大约在15-20%，所以15%是一个合理的校准值。由于模拟的是月度数据，月度标准差 $= 15\%/\sqrt{12} \approx 4.33\%$。

---

**参数5：期望收益率 $E[R_t]$**

$$
E[R_t] = r_f + (I - \rho W)^{-1} \bar{\beta} \Lambda
$$

其中 $r_f = 1\%$（年化无风险利率），$\Lambda = 5\%$（年化因子风险溢价）。

**为什么是 $(I - \rho W)^{-1} \bar{\beta} \Lambda$ 而不是 $\bar{\beta} \Lambda$？** 因为网络效应放大了因子暴露。约化形式的 $\beta^* = (I - \rho W)^{-1} \bar{\beta}$，所以期望超额收益 $= \beta^* \Lambda = (I - \rho W)^{-1} \bar{\beta} \Lambda$。这确保了生成的数据和定价方程完全一致。

---

**参数6：网络矩阵 $W$ 的生成——Bernoulli随机图**

$$
w_{i,j} \sim \text{Bernoulli}(p_B), \quad p_B = 0.3
$$

然后行标准化。

**Bernoulli分布是什么？** 最简单的0/1分布。$\text{Bernoulli}(p)$ 的意思是：以概率 $p$ 取值1（有连接），以概率 $1-p$ 取值0（无连接）。

**具体操作**：对每一对 $(i, j)$（$i \neq j$），独立地抛一次"偏心硬币"：

- 正面（概率0.3）：$w_{i,j} = 1$（行业 $i$ 和 $j$ 有连接）
- 反面（概率0.7）：$w_{i,j} = 0$（无连接）

然后把对角线设为0（自己不连接自己），最后每行除以该行之和（行标准化）。

**为什么用 $p_B = 0.3$？** 这意味着平均每个行业和约30%的其他行业有连接——对应一个中等密度的稀疏网络。真实的行业网络通常不是所有行业都互相连接的（那是完全图），30%是一个合理的中间值。

**这种随机图叫什么？** Erdős-Rényi随机图模型 $G(n, p)$，是最经典的网络生成方法。

---

**参数7：特质冲击 $\eta_t \sim N(0, \Omega)$，$\Omega$ 为对角阵**

$$
\eta_t \sim N(\mathbf{0}, \Omega), \quad \Omega = \text{diag}(\omega_{1,1}, \omega_{2,2}, \ldots, \omega_{K,K})
$$

其中对角元素的**标准差**（不是方差）从均匀分布抽取：

$$
\omega_{i,i}^{1/2} \sim U(10\%, 25\%)
$$

**翻译**：每个资产有自己的特质冲击波动率，这个波动率在10%到25%（年化）之间均匀抽取。

**为什么用正态分布？** 特质冲击用正态分布是因为：

- 它是金融计量中的标准假设
- MLE方法假设正态分布（或者用QMLE框架在非正态下也一致）
- 正态冲击保证了约化形式的残差也是正态的（线性变换保持正态性）

**为什么 $\Omega$ 必须是对角阵？** 这是模型的**关键识别假设**（§4 中讨论过）：结构性特质冲击在各资产间是独立的（无相关性）。如果 $\Omega$ 不是对角阵，模型参数就无法识别——网络效应和特质冲击的相关性混在一起，分不开。

**10%-25% 是什么含义？** 年化波动率10%-25%。转成月度：$\sigma_{月} = \sigma_{年}/\sqrt{12}$。10%年化 → 2.9%月度，25%年化 → 7.2%月度。这个范围对应中等到中高波动率的行业。

---

**参数8：重复次数和样本量**

- 重复 $500$ 次模拟（500个 replications）
- 每次模拟的样本量 $T \in \{200, 500, 1000\}$（月度数据：约16年、42年、83年）

**为什么要重复500次？** 每次模拟产生一组参数估计。重复500次后，可以画出估计量的分布，计算均值和标准差，判断估计量是否有偏、是否收敛。

**为什么用3种样本量？** 为了观察估计量随样本增大的渐近行为。如果估计量是一致的（consistent），那么 $T$ 越大，估计值和真实值的差距（distortion）应该越小。

---

#### 8.1.2 模拟的具体步骤——从生成数据到估计参数

整个模拟过程如下（一次模拟的流程）：

**Step 1**：一次性生成不随模拟次数变化的参数

- 生成 $W$：$100 \times 100$ 的 Bernoulli 随机图，然后行标准化
- 生成 $\bar{\beta}$：从 $U(0.8, 1.2)$ 抽取100个值
- 生成 $\Omega$：从 $U(10\%, 25\%)$ 抽取100个标准差，平方得到方差，构成对角阵

**Step 2**：计算期望收益率

$$
E[R_t] = r_f + (I - \rho W)^{-1} \bar{\beta} \Lambda = 0.01/12 + (I - \rho W)^{-1} \bar{\beta} \times 0.05/12
$$

（除以12是因为从年化转为月度）

**Step 3**：生成 $T$ 个时间点的数据
对每个 $t = 1, 2, \ldots, T$：

1. 生成因子：$F_t \sim N(0, (0.15/\sqrt{12})^2)$
2. 生成冲击：$\eta_t \sim N(\mathbf{0}, \Omega / 12)$（注意方差也要月度化）
3. 计算去均值收益率：$(R_t - E[R_t]) = (I - \rho W)^{-1}(\bar{\beta} F_t + \eta_t)$
4. 加上均值：$R_t = E[R_t] + (I - \rho W)^{-1}(\bar{\beta} F_t + \eta_t)$

**Step 4**：用集中MLE估计模型参数

- 输入：$R_t$（我们"制造"的收益率）、$F_t$（因子，已知）、$W$（网络，已知）
- 输出：$\hat{\rho}$、$\hat{\bar{\beta}}$、$\hat{\Omega}$

**Step 5**：记录估计误差

- $\hat{\rho} - \rho$（网络效应参数的偏差）
- $\hat{\bar{\beta}}_i - \bar{\beta}_i$（每个资产的因子载荷偏差）

**Step 6**：用误设模型（忽略网络的普通因子模型）也估计一次

$$
R_t = \gamma_0 + \gamma_1 F_t + \varepsilon_t \quad \text{（OLS 回归）}
$$

记录 $\hat{\gamma}_1 - \bar{\beta}$（看 OLS beta 偏离结构 beta 多远）和残差相关性。

---

#### 8.1.3 结果解读

**Table 1**（正确模型下 $\hat{\rho}$ 和 $\hat{\bar{\beta}}$ 的偏差）：

| 真实$\rho$ | $T$ | $\hat{\rho} - \rho$ 均值 | $\hat{\rho} - \rho$ 标准差 |
| ------------ | ----- | -------------------------- | ---------------------------- |
| 0            | 200   | 0.067                      | 0.084                        |
| 0            | 1000  | 0.015                      | 0.024                        |
| 0.5          | 200   | 0.050                      | 0.058                        |
| 0.5          | 1000  | 0.012                      | 0.019                        |
| 0.75         | 200   | 0.028                      | 0.032                        |
| 0.75         | 1000  | 0.007                      | 0.011                        |

**解读**：

- 所有情况下偏差均值和标准差都随 $T$ 增大而减小 → **估计量是一致的**
- $\rho = 0$ 时有轻微正偏（0.067 at $T=200$），这是小样本偏差，$T=1000$ 时几乎消失
- $\hat{\rho}$ 有**向上偏差**（overestimation），这在空间计量中是已知现象

**Table 2**（误设模型下 $\hat{\gamma}_1 - \bar{\beta}$ 的偏差）：

| 真实$\rho$ | $T$ | $\hat{\gamma}_1 - \bar{\beta}$ 均值 | 含义                    |
| ------------ | ----- | ------------------------------------- | ----------------------- |
| 0            | any   | 0.000                                 | 无网络时无偏差（对照）  |
| 0.25         | any   | 0.337                                 | $\beta$ 被高估约34%   |
| 0.5          | any   | 1.013                                 | $\beta$ 被高估约1倍！ |
| 0.75         | any   | 3.033                                 | $\beta$ 被高估约3倍！ |

**关键发现**：如果真实世界有网络效应（$\rho > 0$），但你用传统因子模型（忽略网络），你估计出来的 $\beta$（即 $\hat{\gamma}_1$）会系统性地大于真实的结构 $\bar{\beta}$。高估幅度随 $\rho$ 增大急剧增加。

**为什么？** 因为传统OLS估计的其实是约化形式 $\beta^* = (I - \rho W)^{-1} \bar{\beta}$，不是结构 $\bar{\beta}$。网络乘数 $(I - \rho W)^{-1}$ 把 $\bar{\beta}$ 放大了。$\rho = 0.5$ 时放大约2倍，$\rho = 0.75$ 时放大约4倍。

**Table 2 还报告了残差相关性**：

| 真实$\rho$ | 误设模型残差平均相关 | 正确模型残差平均相关 |
| ------------ | -------------------- | -------------------- |
| 0            | 0.000                | −0.001              |
| 0.25         | 0.009                | −0.002              |
| 0.5          | 0.034                | −0.002              |
| 0.75         | 0.149                | −0.002              |

**关键发现**：忽略网络的模型残差存在正相关（$\rho$ 越大相关越强），而正确指定网络模型的残差几乎无相关。这直接验证了 §3.5.2 的理论推导（网络把独立冲击变成相关的）。

---

**Table 3**（风险溢价估计）：

| 模型 | $\rho$ | $\hat{\Lambda}$ 均值 | 真实值 |
| ---- | -------- | ---------------------- | ------ |
| 传统 | any      | ≈ 0.418               | 0.4167 |
| 网络 | any      | ≈ 0.419               | 0.4167 |

**为什么两个模型的风险溢价估计差不多？** 这就是 Proposition 4.1 的核心结果：当 $W$ 是时不变的，即使你用错误模型（忽略网络），OLS估计的 $\Lambda$（风险溢价）仍然是一致的。因为 $\hat{\gamma}_1$ 收敛于 $\beta^*$，而定价方程 $E[R] - r_f = \beta^* \Lambda$ 中用的也是 $\beta^*$，所以 $\Lambda$ 的估计不受影响。

---

#### 8.1.4 稳健性检验

论文还做了以下变体，都不改变主要结论：

| 变体           | 修改内容                                              |
| -------------- | ----------------------------------------------------- |
| $\beta$ 分布 | $\beta_i \sim N(1, \sigma^2)$（正态，而非均匀）     |
| 因子波动率     | $\sigma_F = 25\%$（从15%加大到25%）                 |
| 网络密度       | $p_B = 0.15$（稀疏）或 $p_B = 0.45$（密集）       |
| 风险溢价       | $\Lambda = 3\%$ 或 $\Lambda = 10\%$               |
| 特质波动率     | $\omega_{i,i}^{1/2} \sim U(20\%, 50\%)$（加大冲击） |

---

### 8.2 模拟设计二：异质 $\rho_i$（§5.2）

#### 8.2.1 有什么不同？

设计一中所有资产共享一个 $\rho$。设计二让每个资产有自己的 $\rho_i$：

$$
\rho_i \sim N(0.5, 0.01), \quad i = 1, 2, \ldots, K
$$

**为什么用正态分布？** 正态分布 $N(\mu, \sigma^2)$ 的含义是：值集中在均值 $\mu$ 附近，标准差 $\sigma$ 控制分散程度。这里 $\mu = 0.5$、$\sigma = 0.1$（$\sigma^2 = 0.01$），99%的值落在 $0.5 \pm 2.58 \times 0.1 = [0.242, 0.758]$ 之间。

**为什么要用异质 $\rho_i$？** 真实世界中，不同行业对网络冲击的反应强度不同（零售业可能对供应链冲击更敏感，而公用事业可能不太敏感）。异质 $\rho_i$ 更贴近现实。

**其他变化**：

- $K = 20$（从100减少到20），因为异质 $\rho_i$ 需要估计20个参数（而非1个），计算量大增
- 其余参数不变

#### 8.2.2 结果

- $\hat{\rho}_i$ 的截面平均偏差随 $T$ 增大而减小 → **异质参数也能一致估计**（Fig. 5）
- 误设模型下 $\beta$ 的高估和残差相关和设计一一致
- 残差相关甚至更严重（异质 $\rho_i$ 使得网络效应更复杂）

**风险溢价（Table 4）**：略有高估（$\hat{\Lambda} \approx 0.44$ vs 真值 0.4167），因为异质 $\rho_i$ 放大了变量误差问题（error-in-variable），但随 $T$ 增大而改善。

#### 8.2.3 负 $\rho_i$ 的风险吸收效应

论文还做了一个变体：让一半资产的 $\rho_i > 0$（正常传染），另一半 $\rho_i < 0$（风险吸收）。

**关键发现**（Fig. 6）：当组合中加入 $\rho_i < 0$ 的资产后，网络效应对特质风险的贡献（第IV项）可以变成**负数**——即网络反而帮助分散化！这对应"避风港效应"（flight-to-safety）：当冲击传导到某些反周期行业时，这些行业的反向网络效应**吸收**了部分风险。

---

### 8.3 模拟设计三：时变 $W_t$ + 异质 $\rho_i$（§5.3）

#### 8.3.1 有什么不同？

DGP 变成 Eq.(53)：

$$
(I - \Lambda W_t)(R_t - E[R_t | W_t]) = \bar{\beta} F_t + \eta_t
$$

最大的变化是 $W$ 不再固定——网络随时间变化。

**$W_t$ 是怎样变化的？**

1. 在 $t = 1$ 时，和设计一一样生成初始 $W_1$（Bernoulli 随机图，$p_B = 0.3$）
2. 每隔 $m = 20$ 个观测（约20个月），$W_t$ 可以发生变化
3. 变化方式：每条边 $w_{i,j}$（只取0或1）由一个**马尔可夫链**驱动

**马尔可夫链是什么？** 一个描述"状态转移"的模型。边 $w_{i,j}$ 有两个状态（0 = 断开，1 = 连接）。转移概率：

$$
P(\text{下一期} = 0 \mid \text{当前} = 0) = p_{00} = 0.9
$$

$$
P(\text{下一期} = 1 \mid \text{当前} = 1) = p_{11} = 0.9
$$

**翻译**：如果当前两个行业有连接（$w_{i,j} = 1$），下一期90%的概率还是有连接，10%的概率断开。如果当前没有连接，下一期90%的概率还是没有。

**为什么这样设计？** 为了保证网络的**持久性**（persistence）。真实世界的行业关联不会每个月都大变——供应链关系、贸易伙伴都是缓慢变化的。$p_{00} = p_{11} = 0.9$ 保证了网络结构高度稳定，只偶尔有小幅变化。

**标准化方式不同**：设计一用行标准化（每行和=1），设计三改用**最大行标准化**（max row normalization）：

$$
W_{i,j,t} = \frac{W^U_{i,j,t}}{\max_t \sum_{i=1}^{N} W^U_{i,j,t}}
$$

**为什么换标准化？** 论文 §3.3 讨论过：普通行标准化下，如果网络变密（更多连接），每条边的权重自动变小，会"稀释"网络密度变化的影响。最大行标准化用所有时间点的最大行和作为分母，保留了网络密度随时间变化的信息。

#### 8.3.2 期望收益率变成时变的

和设计一、二不同，现在 $E[R_t | W_t]$ 依赖于 $W_t$：

$$
E[R_t | W_t] = r_f + (I - \Lambda W_t)^{-1} \bar{\beta} \Lambda_{RP}
$$

每次 $W_t$ 变化，期望收益率也跟着变。这使得问题更复杂、更贴近现实。

#### 8.3.3 结果

**Table 5**（$\beta$ 估计偏差）：

| 模型                 | $T$ | 偏差均值 | 偏差标准差 |
| -------------------- | ----- | -------- | ---------- |
| 误设（传统因子模型） | 200   | 0.153    | 0.008      |
| 误设（传统因子模型） | 1000  | 0.194    | 0.002      |
| 正确（网络模型）     | 200   | 0.082    | 0.014      |
| 正确（网络模型）     | 1000  | 0.040    | 0.006      |

**关键发现**：

- 误设模型下，$\beta$ 偏差**不随 $T$ 收敛**（偏差从0.153增到0.194）！这是因为当 $W_t$ 时变时，传统因子模型估计的是某种"平均 $\beta^*$"，不是结构 $\bar{\beta}$，也不是任何一期的 $\beta^*_t$
- 正确模型下，偏差从0.082降到0.040 → **一致收敛**

这是时变网络最重要的后果：**只有正确指定的网络模型才能在 $W_t$ 变化时保持一致性。**

**风险溢价（Table 6）**：

- 两个模型的 $\hat{\Lambda}$ 均值都接近真值
- 但正确模型的**标准差更小**！（更高效的估计量）
- GLS 和 OLS 差别不大

**为什么风险溢价差别不大？** 论文解释：虽然单个时点的 $\beta^*_t$ 是时变的，但传统模型对 $\beta$ 的估计实际上是在对所有 $W_t$ 做隐式平均。由于 $W_t$ 的变化是温和的（马尔可夫链有高度持续性），这种平均值恰好接近结构 $\bar{\beta}$。所以风险溢价的点估计差别不大。但**标准误更小**是网络模型的核心优势——更精确的统计推断。

---

### 8.4 模拟分析总结

| 设计                                  | 关键参数                                       | 主要发现                                    |
| ------------------------------------- | ---------------------------------------------- | ------------------------------------------- |
| 设计一：标量$\rho$，固定 $W$      | $K=100$, $\rho \in \{0, 0.25, 0.5, 0.75\}$ | MLE一致、忽略网络→$\beta$高估、残差相关  |
| 设计二：异质$\rho_i$，固定 $W$    | $K=20$, $\rho_i \sim N(0.5, 0.01)$         | 异质参数也能一致估计、负$\rho_i$吸收风险  |
| 设计三：异质$\rho_i$ + 时变 $W_t$ | $K=20$, 马尔可夫链驱动 $W_t$               | 传统模型$\beta$不一致、网络模型标准误更小 |

**对你的项目的启示**：

- 你的 Bonaccolto (2019) 复现也需要模拟验证。可以参考这个设计，把单层 $W$ 改为多层 $\sum \delta_j W_j$
- DY 溢出网络天然是时变的（设计三的场景），所以你的网络模型能提供比传统因子模型更精确的推断
- 模拟中 $K=20$ 而非100，是因为异质 $\rho_i$ 增加了参数量——你的31个行业和 $K=20$ 量级类似

---

### 8.5 §6 实证分析核心发现

**数据**：15 个美国宏观行业，月度收益率（2005-2018），I/O 表作为网络。

**Table 8（$\rho_i$ 估计值）**：

- 6 个行业显著：农业(0.37)、批发(0.41)、零售(0.53)、信息(0.47)、娱乐(0.22)、政府(0.33)
- 制造业 $\rho = -0.12$：负值意味着网络效应**减弱**了风险（风险吸收）
- 金融 $\rho = 0.04$：不显著

**$\rho_i$ 的显著性如何判定？** MLE 估计完参数后，用对数似然的 **Hessian 矩阵**（二阶导数矩阵）计算标准误：

$$
\text{SE}(\hat{\rho}_i) = \sqrt{\left[-\frac{\partial^2 L}{\partial \rho_i^2}\right]^{-1}}
$$

然后做 t 检验：$t = \hat{\rho}_i / \text{SE}(\hat{\rho}_i)$。若 $|t| < 1.96$，则在5%水平下不显著。金融业 $\hat{\rho} = 0.04$ 不显著，意味着其标准误相对于0.04来说太大（例如 $\text{SE} = 0.05$ 则 $t = 0.8$，远小于1.96），统计上无法拒绝 $H_0: \rho_i = 0$。论文脚注15也提到："Standard errors can be recovered from the full-model likelihood by making numerical evaluations of the Hessian."

**Table 9（残差相关性）**：

- 传统模型：平均残差相关 0.10
- 网络模型：平均残差相关 **0.03**（大幅降低）

**Table 10（风险溢价 $\Lambda$）**：

- 两个模型的 $\Lambda$ 值相近
- 但网络模型的 p 值大幅改善：MKT 从 0.31 → **0.004**

**这里的 p 值从哪来？不是 $\rho$ 的 p 值。** 这是 Fama-MacBeth 第二步**截面回归**中风险溢价 $\Lambda$ 的 p 值。具体来说：

$$
\bar{R}^e = \hat{\beta}^* \Lambda + \nu
$$

- 因变量 $\bar{R}^e$：每个资产的平均超额收益（$K$ 个数）
- 自变量 $\hat{\beta}^*$：从第一步估计出的因子载荷（$K$ 个数）
- 待估参数 $\Lambda$：因子风险溢价（标量）
- 标准误经 Shanken (1992) 修正（处理 $\hat{\beta}^*$ 本身的估计误差）

这就是一个截面 OLS，$\Lambda$ 的 p 值就是回归标准输出。

**为什么网络模型的 p 值更小（0.004 vs 0.310）？**

| 模型         | 用的$\hat{\beta}$                          | 截面回归残差$\nu$ 的性质     | $\Lambda$ 的标准误 |
| ------------ | -------------------------------------------- | ------------------------------ | -------------------- |
| 传统因子模型 | $\hat{\gamma}_1$（约化形式，混着网络效应） | 残差有截面相关（被网络"污染"） | 大 → p 值大         |
| 网络模型     | $\hat{\beta}^*$（正确分离了网络）          | 残差更干净（网络效应已处理）   | 小 → p 值小         |

$\Lambda$ 的**点估计**两个模型差不多（Proposition 4.1），但传统模型的截面回归残差里残留着网络引起的相关性，导致标准误被放大。网络模型把这层相关性剥掉了，残差更独立，标准误更小，同样的系数值对应更小的 p 值。

---

## 9. 全文总结与项目启示

### 9.1 Billio (2023) 的核心贡献

1. 把网络关联写进因子模型，区分了**结构性** vs **约化形式** beta
2. 证明忽略网络会导致 beta 高估、残差相关、分散化效果误判
3. 集中MLE估计方法高效可行
4. 风险溢价估计的精度大幅提升

### 9.2 对你的项目的启示

| 知识点                            | 在你的项目中的应用                                              |
| --------------------------------- | --------------------------------------------------------------- |
| $A = I - \Lambda W$             | 你用$A = I - \Lambda \sum \delta_j W_j$，多了 $\delta$ 权重 |
| 集中似然估计                      | 你的 `multilayer_network_mle.py` 已经实现了这个框架           |
| $(I - \Lambda W)^{-1}$ 网络乘数 | 你用它计算中心性、分析冲击传播                                  |
| 异质$\rho_i$                    | 你的$\Lambda = \text{diag}(\rho_i)$ 允许每个行业不同反应      |
| 时变网络                          | 你用滚动窗口 DY 网络实现$W_t$                                 |

---

*最后更新：2026-04-22（会话#17：§4.4 按论文脚注11重写 + §5 模拟分析逐句详解）*
