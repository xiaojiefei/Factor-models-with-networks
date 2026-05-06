# 模型组合方法详解：带多层空间依赖的线性因子模型

> 论文：Bonaccolto et al. (2019), *Journal of Empirical Finance*  
> 核心贡献：将多个因果网络通过线性因子模型进行数据驱动的最优组合

---

## 一、模型设定

### 1.1 结构方程 (Equation 4)

$$A \cdot R_t = E[R_t] + \beta \cdot F_t + \eta_t$$

其中变换矩阵 $A$ 定义为：

$$A = I - \Lambda \cdot \left(\sum_{j=1}^{d} \delta_j \cdot W_j\right)$$

### 1.2 参数说明

| 符号 | 维度 | 含义 |
|------|------|------|
| $R_t$ | $K \times 1$ | $K$ 个资产的收益率向量 |
| $E[R_t]$ | $K \times 1$ | 资产期望收益 |
| $F_t$ | $n_f \times 1$ | 公共因子向量，$n_f = 4$（Fama-French 4因子：市场、规模、价值、动量）|
| $\beta$ | $K \times n_f$ | 因子暴露矩阵（结构性 beta），$n_f = 4$|
| $\eta_t$ | $K \times 1$ | 结构残差，假设 $\text{Cov}(\eta) = \Sigma_\eta$ 为对角阵 |
| $W_j$ | $K \times K$ | 第 $j$ 层网络的**行标准化**邻接矩阵 |
| $\delta_j$ | 标量 | 第 $j$ 层网络的权重，满足 $\delta_j \geq 0$, $\sum \delta_j = 1$ |
| $\Lambda$ | $K \times K$ 对角阵 | 异质反应系数矩阵 $\text{diag}(\rho_1, ..., \rho_K)$ |
| $I$ | $K \times K$ | 单位矩阵 |

### 1.2.1 各变量维度详解（以 K=31 行业、d=2 层网络为例）

#### $R_t$：$K \times 1$（31×1）— 一个时刻所有行业的收益率快照

```
R_t = [ 交通运输今天的收益率 ]   ← 第1行
      [ 传媒今天的收益率     ]   ← 第2行
      [ 公用事业今天的收益率  ]   ← 第3行
      [       ...            ]
      [ 食品饮料今天的收益率  ]   ← 第31行
```

下标 $t$ 表示"某一天"，整个向量是那**一天**31个行业的截面数据。

#### $F_t$：$n_f \times 1$ — 一天的公共因子

若用 PCA 第一主成分（$n_f=1$），$F_t$ 是一个标量。若用 Fama-French 4因子（$n_f=4$）：

```
F_t = [ 市场超额收益 ]    ← 第1个因子
      [ 规模因子SMB   ]    ← 第2个因子
      [ 价值因子HML   ]    ← 第3个因子
      [ 动量因子MOM   ]    ← 第4个因子
```

我们项目中用等权市场收益率或 PCA，$n_f = 1$。

#### $\beta$：$K \times n_f$（31×1 或 31×4）— 每个行业对每个因子的敏感度

以 $n_f=1$ 为例（31×1）：

```
β = [ 交通运输对市场因子的暴露 ]   例如 1.05
    [ 传媒对市场因子的暴露     ]   例如 1.20
    [ 公用事业对市场因子的暴露  ]   例如 0.65
    [         ...              ]
    [ 食品饮料对市场因子的暴露  ]   例如 0.80
```

**乘法验证**：$\beta \cdot F_t$ = $(31 \times 1) \cdot (1 \times 1)$ = $31 \times 1$ ✓；若 $n_f=4$：$(31 \times 4) \cdot (4 \times 1) = 31 \times 1$ ✓

#### $\eta_t$：$K \times 1$（31×1）— 因子和网络都解释不了的部分

**核心假设**：$\text{Cov}(\eta_t) = \Sigma_\eta$ 是**对角阵**，即各行业的特质冲击**互不相关**。行业间所有相关性都已被因子 $F_t$ 和网络 $W^*$ 解释掉。

#### $W_j$：$K \times K$（31×31）— 第 $j$ 层网络邻接矩阵

```
W₁ (收益率DY网络):

       交通  传媒  公用  ...  食品
交通 [  0   0.05 0.12  ... 0.03 ]  ← 交通运输受其他行业影响的权重
传媒 [ 0.08  0   0.06  ... 0.04 ]  ← 传媒受其他行业影响的权重
...
食品 [ 0.02 0.07 0.03  ...  0   ]
```

**关键性质**：对角线 = 0（自己不影响自己），每行之和 = 1（行标准化），$(i,j)$ 元素 = 行业 $j$ 对行业 $i$ 的影响强度。

#### $\delta_j$：标量 — 每层网络的混合权重

我们有 $d=2$ 层：$\delta_1$ = 收益率网络权重，$\delta_2$ = 波动率网络权重。约束 $\delta_1 + \delta_2 = 1$，$\delta_j \geq 0$。$\delta$ 告诉你**哪种网络视角更重要**。

#### $W^* = \sum \delta_j W_j$：$K \times K$（31×31）— 两层网络的加权混合

$$W^* = \delta_1 \times W_1 + \delta_2 \times W_2$$

维度：$(31 \times 31) = \text{标量} \times (31 \times 31) + \text{标量} \times (31 \times 31)$。$W^*$ 仍然行标准化（因为 $\sum \delta_j = 1$ 且每个 $W_j$ 已行标准化）。

#### $A = I - \Lambda W^*$：$K \times K$（31×31）— 变换矩阵

维度：$(31 \times 31) = (31 \times 31) - (31 \times 31) \cdot (31 \times 31)$。对角线都是 1（因为 $W^*$ 对角线=0），非对角线是负的小数。

#### 等式整体维度验证

$$\underbrace{A}_{31 \times 31} \cdot \underbrace{R_t}_{31 \times 1} = \underbrace{E[R_t]}_{31 \times 1} + \underbrace{\beta}_{31 \times n_f} \cdot \underbrace{F_t}_{n_f \times 1} + \underbrace{\eta_t}_{31 \times 1}$$

左边：$(31 \times 31) \cdot (31 \times 1) = 31 \times 1$ ✓ ；右边：$(31 \times 1) + (31 \times 1) + (31 \times 1) = 31 \times 1$ ✓

---

### 1.2.2 直觉理解：$A \cdot R_t$ 是"去传染后的收益"

#### 拆解

$$A \cdot R_t = (I - \Lambda W^*) \cdot R_t = R_t - \Lambda W^* R_t$$

移项后：

$$R_t = \underbrace{E[R_t] + \beta F_t + \eta_t}_{\text{自身基本面}} + \underbrace{\Lambda W^* R_t}_{\text{来自邻居的传染}}$$

#### $W^* R_t$：邻居的加权平均收益

用 3 个行业的例子（银行、保险、证券）：

$$W^* R_t = \begin{bmatrix} 0 & 0.6 & 0.4 \\ 0.7 & 0 & 0.3 \\ 0.5 & 0.5 & 0 \end{bmatrix} \begin{bmatrix} -2\% \\ +1\% \\ +3\% \end{bmatrix} = \begin{bmatrix} +1.8\% \\ -0.5\% \\ -0.5\% \end{bmatrix}$$

银行的"邻居加权收益" = 60%×保险收益 + 40%×证券收益 = +1.8%。$W^*$ 的第 $i$ 行是行业 $i$ 看其他行业的"权重视角"。

#### $\Lambda W^* R_t$：实际接收的传染量

$\rho_i$ 控制行业 $i$ 实际接收多少邻居传染。以 $\rho = (0.3, 0.5, 0.2)$ 为例：

$$\Lambda W^* R_t = \begin{bmatrix} 0.3 \\ 0.5 \\ 0.2 \end{bmatrix} \odot \begin{bmatrix} +1.8\% \\ -0.5\% \\ -0.5\% \end{bmatrix} = \begin{bmatrix} +0.54\% \\ -0.25\% \\ -0.10\% \end{bmatrix}$$

- 银行 $\rho_1=0.3$：邻居给了+1.8%，银行只"接收"30% → +0.54%
- 保险 $\rho_2=0.5$：邻居给了-0.5%，保险"接收"50% → -0.25%

#### $A \cdot R_t = R_t - \Lambda W^* R_t$：剥离传染

| 行业 | 原始收益 $R_t$ | 网络传染 $\Lambda W^* R_t$ | 去传染后 $A R_t$ | 解读 |
|------|---------------|--------------------------|-----------------|------|
| 银行 | -2.00% | +0.54%（邻居拉高） | -2.54% | 去掉拉高后，自身更差 |
| 保险 | +1.00% | -0.25%（邻居拖累） | +1.25% | 去掉拖累后，自身更好 |
| 证券 | +3.00% | -0.10%（邻居拖累） | +3.10% | 去掉拖累后，自身更好 |

银行今天跌了2%，但其中有0.54%是保险和证券"传染"给它的正面效应。去掉传染后，银行自身基本面其实跌了2.54%。

> **一句话总结**：$A \cdot R_t$ = 剥离网络传染后的"纯净"收益。它把每个行业从邻居那里"接收"到的影响减掉，只留下能被因子 $\beta F_t$ 和特质冲击 $\eta_t$ 解释的部分。这也是 $\eta_t$ 可以假设为对角协方差的原因——行业间的相关性已通过 $A$ 被"剥离"。

---

### 1.2.3 关键参数详解：$\Lambda = \text{diag}(\rho_1, \ldots, \rho_K)$

$\Lambda$ 是 **K×K 对角矩阵**，表示**异质反应系数**（Heterogeneous Response Coefficients）。

#### 数学形式

$$\Lambda = \begin{bmatrix} \rho_1 & 0 & \cdots & 0 \\ 0 & \rho_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \rho_K \end{bmatrix}$$

#### 为什么用对角矩阵？

**核心假设**：每个资产对网络影响的**反应强度不同**，但不同资产之间不直接交叉影响。

| 特性 | 解释 |
|------|------|
| **对角性** | 只影响自身，不直接交叉影响其他资产（简化模型） |
| **异质性** | 允许 $\rho_i \neq \rho_j$，不同资产敏感度可以不同 |

#### 计算效果

$\Lambda \cdot W^*$ 的乘法会将 $W^*$ 的**第 i 行全部乘以 $\rho_i$**：

$$\Lambda \cdot W^* = \begin{bmatrix} \rho_1 w_{11} & \rho_1 w_{12} & \cdots \\ \rho_2 w_{21} & \rho_2 w_{22} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

#### 经济学含义

| 符号 | 含义 | 直观解释 |
|------|------|----------|
| $\rho_i$ | 资产 i 的网络反应系数 | 资产 i 受到网络冲击时的"敏感度" |
| $1 - \rho_i$ | 资产 i 的独立部分 | 不受网络影响的比例 |

**具体例子**：
- 若 $\rho_i = 0.3$：30% 的冲击通过网络传播，70% 是独立的
- 若 $\rho_i = 0.8$：80% 通过网络传播，该资产高度互联

**异质 vs 同质**：
- **本文（异质）**：$\Lambda = \text{diag}(\rho_1, \ldots, \rho_K)$，每个资产有自己的反应系数
- **简化（同质）**：$\Lambda = \rho \cdot I$，所有资产反应系数相同

#### 代码对应

```python
# rho: K×1 向量，每个元素是一个资产的反应系数
rho = np.array([0.2, 0.4, 0.15, ...])  # K 个元素

# Lambda: K×K 对角矩阵
Lambda = np.diag(rho)

# 计算 A = I - Lambda @ W_star
A = np.eye(K) - Lambda @ W_star
```

### 1.3 复合网络 (Equation 6)

$$W^* = \sum_{j=1}^{d} \delta_j \cdot W_j$$

$W^*$ 是各层网络的凸组合，权重 $\delta_j$ 由模型估计得到。

---

## 二、经济学直觉

### 2.1 单层网络简化

若只有一层网络（$d=1$, $\delta_1=1$），模型退化为 Billio et al. (2017)：

$$R_t = E[R_t] + (I - \Lambda W)^{-1}\beta F_t + (I - \Lambda W)^{-1}\eta_t$$

关键理解：**$(I - \Lambda W)^{-1}$ 捕捉了冲击在网络中的传播效应**。

#### 从零开始理解：$(I - \Lambda W)^{-1}$ 究竟是什么？

**【第一步】先讲故事，不要管公式**

假设有两个行业：银行业和保险业。它们之间有风险传染关系：
- 如果银行出事（比如破产），会影响保险公司
- 如果保险公司出事，也会反作用于银行

现在，银行业受到一个冲击（比如次贷危机），会发生什么？

```
第0步：银行直接受到冲击
        影响大小 = 1

第1步：银行的问题传染给保险公司
        银行 ──传染──→ 保险
        影响大小 = 0.3（假设传染系数是0.3）

第2步：保险公司出问题，又反馈给银行
        保险 ──传染──→ 银行
        影响大小 = 0.3 × 0.3 = 0.09

第3步：银行再次受到影响，又传染给保险
        影响大小 = 0.3 × 0.3 × 0.3 = 0.027
        
... 这个过程无限循环下去
```

**【第二步】把故事变成数字**

假设传染系数 $\rho = 0.3$，那么银行业最终受到的总影响是：

$$\text{总影响} = 1 + 0.09 + 0.0027 + 0.000081 + \cdots = 1.0989$$

这是无穷等比数列求和：$1 + 0.3^2 + 0.3^4 + 0.3^6 + \cdots = \frac{1}{1 - 0.09} \approx 1.0989$

**关键发现**：
- 如果只看第0步，银行受到的影响是 1
- 但考虑网络传染后，银行最终受到的影响是 1.0989
- **网络让冲击放大了约10%**

**【第三步】$(I - \Lambda W)^{-1}$ 就是帮你算这个总影响的**

在上面的例子中：
- $I$ 是单位矩阵（代表"第0步：直接冲击"）
- $\Lambda W$ 代表网络传染（第1步）
- $(\Lambda W)^2$ 代表二次传染（第2步）
- $(\Lambda W)^3$ 代表三次传染（第3步）
- ...

$(I - \Lambda W)^{-1}$ 用公式把这些全部加起来：

$$(I - \Lambda W)^{-1} = I + \Lambda W + (\Lambda W)^2 + (\Lambda W)^3 + \cdots$$

**【第四步】具体例子**

假设2个资产，网络矩阵和反应系数：

$$W = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad \Lambda = \begin{bmatrix} 0.3 & 0 \\ 0 & 0.3 \end{bmatrix}$$

计算 $(I - \Lambda W)^{-1}$：

$$I - \Lambda W = \begin{bmatrix} 1 & -0.3 \\ -0.3 & 1 \end{bmatrix}$$

$$(I - \Lambda W)^{-1} = \begin{bmatrix} 1.0989 & 0.3297 \\ 0.3297 & 1.0989 \end{bmatrix}$$

**解读**：
- $(1,1)$ 位置 = 1.0989：资产1受到1单位冲击后，**最终**自己受到1.0989的影响（含反馈）
- $(1,2)$ 位置 = 0.3297：资产1受到1单位冲击后，资产2最终受到0.3297的影响

**【第五步】为什么要写成 $A \cdot R_t = \cdots$ 的形式？**

原始模型可以写成两种形式：

**形式A（简化式）**：$R_t = (I - \Lambda W)^{-1} \beta F_t + (I - \Lambda W)^{-1} \eta_t$

问题：
- beta 前面乘了 $(I - \Lambda W)^{-1}$，不再是"纯净的因子暴露"
- 残差 $\eta_t$ 变成了 $(I - \Lambda W)^{-1} \eta_t$，资产之间相关了
- 用OLS估计会有偏误

**形式B（结构式）**：$(I - \Lambda W) \cdot R_t = \beta F_t + \eta_t$

优势：
- beta 是**纯净的因子暴露**（不含网络效应）
- 残差 $\eta_t$ 是**独立的**（对角协方差）
- 网络效应被显式地放在左边
- **可以用OLS估计！**

**【总结】一句话理解**：

> $(I - \Lambda W)^{-1}$ 回答了这个问题：如果一个资产受到冲击，考虑网络传染后，最终所有资产会受到多大的影响？

论文用 $A = I - \Lambda W^*$ 把网络效应"剥离"出来，这样就可以先估计纯净的因子暴露，再用MLE估计网络参数。

### 2.2 多层网络的视角组合

本文的核心创新：

- 将不同方法估计的因果网络视为不同的"观测视角"
  - GR 网络：均值层面的线性关系
  - QB/Qo/QN 网络：分位数层面的风险传染

- 用数据驱动的方式学习最优组合权重 $\delta_j$

- 复合网络 $W^*$ 综合了所有视角的信息

---

## 三、估计方法：集中最大似然 (Concentrated MLE) 详细推导

### 3.1 问题背景：为什么要用MLE？

我们有结构方程：
$$A \cdot R_t = E[R_t] + \beta F_t + \eta_t$$

**待估参数**：
- $\beta$：$K \times f$ 的因子暴露矩阵（可能较多，比如48个资产 × 4个因子 = 192个参数）
- $\Sigma_\eta$：$K \times K$ 的对角协方差矩阵（48个参数）
- $\delta = (\delta_1, ..., \delta_d)$：$d$ 个网络层权重（比如4种网络方法，就是4个参数）
- $\Lambda = \text{diag}(\rho_1, ..., \rho_K)$：$K$ 个异质反应系数（48个参数）

**总参数数量**：约 192 + 48 + 4 + 48 = 292 个！

如果直接对这292个参数做优化，会非常困难（维度太高，容易陷入局部最优）。

**MLE的思想**：假设残差 $\eta_t$ 服从正态分布，通过最大化似然函数来估计参数。

---

### 3.2 完整的似然函数推导

#### 【第一步】写出 $\eta_t$ 的分布

从结构方程解出残差：
$$\eta_t = A \cdot R_t - E[R_t] - \beta F_t$$

假设 $\eta_t \sim N(0, \Sigma_\eta)$，其中 $\Sigma_\eta$ 是对角阵：
$$\Sigma_\eta = \text{diag}(\sigma_1^2, \sigma_2^2, ..., \sigma_K^2)$$

#### 【第一步.5】关键问题：为什么用多元正态分布，而不是K个独立的一元正态？

**【问题背景】**

我们有 $K$ 个资产，每个时刻 $t$ 有一个残差向量：
$$\eta_t = (\eta_{1t}, \eta_{2t}, ..., \eta_{Kt})'$$

**【方案1：用K个独立的一元正态】**

如果对每个资产单独建模：
$$L = \prod_{t=1}^T \prod_{i=1}^K \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{\eta_{i,t}^2}{2\sigma_i^2}\right)$$

**问题**：这样假设**从一开始**就强制 $\eta_{i,t}$ 和 $\eta_{j,t}$ 独立，但结构残差是否独立是**假设**，不是**推导结果**。

**【方案2：用多元正态分布（论文做法）】**

$$L = \prod_{t=1}^T (2\pi)^{-K/2} |\Sigma_\eta|^{-1/2} \exp\left(-\frac{1}{2} \eta_t' \Sigma_\eta^{-1} \eta_t\right)$$

这里 $\Sigma_\eta$ 是**完整的** $K \times K$ 协方差矩阵。

**优势**：
1. **一般性**：不预设独立性，协方差矩阵可以非对角
2. **可检验**：估计后可以检验非对角元素是否显著
3. **识别需要**：模型识别要求 $\Sigma_\eta$ 为对角阵，这是对参数的**约束**，不是对分布的**预设**

**【当协方差为对角时】**

论文假设 $\Sigma_\eta = \text{diag}(\sigma_1^2, ..., \sigma_K^2)$，此时：
$$|\Sigma_\eta| = \prod_{i=1}^K \sigma_i^2, \quad \Sigma_\eta^{-1} = \text{diag}\left(\frac{1}{\sigma_1^2}, ..., \frac{1}{\sigma_K^2}\right)$$

代入多元正态，**正好等于** K个独立一元正态的似然！

**【总结】**

> 用多元正态是**先一般后特殊**的标准做法：
> - 先写一般形式（完整协方差矩阵）
> - 再施加约束（对角协方差）
> - 这样既保持数学严谨性，又符合模型假设

---

#### 【第二步】构造似然函数

对于多元正态分布，似然函数为：
$$L(\theta) = \prod_{t=1}^T (2\pi)^{-K/2} |\Sigma_\eta|^{-1/2} \exp\left(-\frac{1}{2} \eta_t' \Sigma_\eta^{-1} \eta_t\right)$$

取对数得到**对数似然**：
$$\ln L(\theta) = -\frac{TK}{2} \ln(2\pi) - \frac{T}{2} \ln|\Sigma_\eta| - \frac{1}{2} \sum_{t=1}^T \eta_t' \Sigma_\eta^{-1} \eta_t$$

#### 【第三步】关键的变量替换

这里有一个问题：$\eta_t$ 依赖于 $A$，而 $A$ 又依赖于 $(\delta, \Lambda)$。同时，$\eta_t$ 是**结构残差**。

但我们观测到的是 $R_t$，不是 $A \cdot R_t$。需要用到**变量变换公式**（Jacobian）：

$$R_t = A^{-1} (E[R_t] + \beta F_t + \eta_t)$$

从 $\eta_t$ 到 $R_t$ 的变换，Jacobian 行列式为 $|A^{-1}| = |A|^{-1}$。

因此，关于 $R_t$ 的对数似然要加上 Jacobian 项：
$$\ln L_R(\theta) = \ln L(\theta) + T \ln|A|$$

完整表达式：
$$\ln L_R(\theta) = -\frac{TK}{2} \ln(2\pi) - \frac{T}{2} \ln|\Sigma_\eta| - \frac{1}{2} \sum_{t=1}^T \eta_t' \Sigma_\eta^{-1} \eta_t + T \ln|A|$$

---

### 3.3 "集中"（Concentrated）的思想

#### 【核心问题】

直接优化 $\ln L_R(\theta)$ 需要同时优化292个参数，太复杂！

#### 【关键观察】

给定 $A$（即给定 $\delta$ 和 $\Lambda$），关于 $\beta$ 和 $\Sigma_\eta$ 的最优解**有解析表达式**！

这意味着：
1. 对于任意给定的 $(\delta, \Lambda)$，我们可以**解析地**算出最优的 $\hat{\beta}$ 和 $\hat{\Sigma}_\eta$
2. 把这些最优解代回似然函数，就得到一个**只关于 $(\delta, \Lambda)$ 的函数**
3. 优化维度从292降到 4 + 48 = 52！

这就是"集中"（Concentrated）的含义：把可以解析求解的参数"集中"掉，降低优化维度。

---

### 3.4 详细的三步推导

#### 【Step 1】给定 $A$，求 $\hat{\beta}$ 和 $\hat{\Sigma}_\eta$

将 $\eta_t = A R_t - \beta F_t$（假设 $E[R_t]$ 已中心化）代入对数似然：

$$\ln L = -\frac{T}{2} \ln|\Sigma_\eta| - \frac{1}{2} \sum_{t=1}^T (A R_t - \beta F_t)' \Sigma_\eta^{-1} (A R_t - \beta F_t) + T \ln|A|$$

对 $\beta$ 求导并令为0：
$$\frac{\partial \ln L}{\partial \beta} = \sum_{t=1}^T \Sigma_\eta^{-1} (A R_t - \beta F_t) F_t' = 0$$

解得：
$$\hat{\beta}(A) = \left(\sum_{t=1}^T A R_t F_t'\right) \left(\sum_{t=1}^T F_t F_t'\right)^{-1}$$

矩阵形式：
$$\hat{\beta}(A) = (A R)' F (F' F)^{-1}$$

其中：
- $R$ 是 $T \times K$ 的收益率矩阵
- $F$ 是 $T \times n_f$ 的因子矩阵（$n_f = 4$）
- $\hat{\beta}(A)$ 是 $K \times f$ 矩阵

**计算残差**：
$$\hat{\eta}_t = A R_t - \hat{\beta}(A) F_t$$

**估计协方差**：
$$\hat{\Sigma}_\eta(A) = \frac{1}{T} \sum_{t=1}^T \hat{\eta}_t \hat{\eta}_t' = \frac{1}{T} \hat{\eta}' \hat{\eta}$$

由于 $\Sigma_\eta$ 是对角阵，我们实际上只关心对角线元素：
$$\hat{\sigma}_i^2 = \frac{1}{T} \sum_{t=1}^T \hat{\eta}_{i,t}^2$$

---

#### 【Step 1.5】关键问题：Jacobian 项 $T \ln|A|$ 的由来

这一步是最抽象的，我们用**从简单到复杂**的方式理解。

##### 【最简单的例子：一元情况】

假设你假设 $\epsilon \sim N(0, 1)$，但你观测到的是 $Y = 2\epsilon$。

**问题**：$Y$ 服从什么分布？

直观上，$Y$ 把 $\epsilon$ 拉伸了2倍，所以 $Y \sim N(0, 4)$（方差变成 $2^2$）。

严格来说，变量替换公式是：
$$f_Y(y) = f_\epsilon(\epsilon(y)) \cdot |\frac{d\epsilon}{dy}|$$

这里 $\epsilon = y/2$，所以 $|\frac{d\epsilon}{dy}| = 1/2$。

$f_Y(y) = f_\epsilon(y/2) \cdot \frac{1}{2}$

**$|\frac{d\epsilon}{dy}| = 1/2$ 就是 Jacobian！**

##### 【回到本文：多元情况】

我们有结构方程：
$$A \cdot R_t = \beta F_t + \eta_t$$

**问题的本质**：
- 我们**假设** $\eta_t \sim N(0, \Sigma_\eta)$（残差服从正态）
- 我们**观测**到的是 $R_t$（收益率数据）
- 需要**从 $\eta_t$ 的分布推导 $R_t$ 的分布**

从结构方程解出：
$$R_t = A^{-1} \beta F_t + A^{-1} \eta_t$$

这是线性变换：$\eta_t \rightarrow R_t = A^{-1} \eta_t + \text{常数}$

**多元变量替换公式**：
$$f_R(R_t) = f_\eta(\eta(R_t)) \cdot |\frac{\partial \eta}{\partial R}|$$

这里 $\eta = A R_t - \beta F_t$，所以 $|\frac{\partial \eta}{\partial R}| = |A|$。

**Jacobian 就是 $|A|$（A 的行列式）**。

##### 【对数似然的完整形式】

$$\ln f_R(R_t) = \underbrace{\ln f_\eta(\eta_t)}_{\text{正态密度}} + \underbrace{\ln|A|}_{\text{Jacobian 项}}$$

对 $T$ 个观测求和：
$$\sum_{t=1}^T \ln f_R(R_t) = \sum_{t=1}^T \ln f_\eta(\eta_t) + T \ln|A|$$

**这就是对数似然中 $T \ln|A|$ 的来源！**

##### 【Jacobian 的经济含义】为什么必须有这一项？

**具体例子**：假设 $A = I - 0.3W$，计算得 $|A| = 0.91 < 1$。

- $|A| < 1$：变换"压缩"了概率空间的体积
- 如果不加 $\ln|A|$ 这一项：优化会偏向于让 $|A|$ 很小（压缩很厉害）
- 结果：估计会**严重偏误**

**Jacobian 项的作用**：**补偿**线性变换带来的"扭曲"，保证估计无偏。

---

#### 【Step 2】构建集中对数似然

将 $\hat{\beta}(A)$ 和 $\hat{\Sigma}_\eta(A)$ 代回原对数似然（**包含 Jacobian 项**）：

$$\ln L_c(\delta, \Lambda) = -\frac{T}{2} \ln|\hat{\Sigma}_\eta(A(\delta, \Lambda))| - \frac{1}{2} \sum_{t=1}^T \hat{\eta}_t' \hat{\Sigma}_\eta^{-1} \hat{\eta}_t + T \ln|A(\delta, \Lambda)|$$

**简化第二项**：
$$\sum_{t=1}^T \hat{\eta}_t' \hat{\Sigma}_\eta^{-1} \hat{\eta}_t = \text{tr}\left(\hat{\Sigma}_\eta^{-1} \sum_{t=1}^T \hat{\eta}_t \hat{\eta}_t'\right) = \text{tr}\left(\hat{\Sigma}_\eta^{-1} \cdot T \hat{\Sigma}_\eta\right) = T \cdot \text{tr}(I_K) = TK$$

这是一个常数！

因此集中对数似然简化为：

$$\boxed{\ln L_c(\delta, \Lambda) = -\frac{T}{2} \ln|\hat{\Sigma}_\eta(A(\delta, \Lambda))| + T \ln|A(\delta, \Lambda)| + \text{const}}$$

等价于最大化：
$$\ln L_c(\delta, \Lambda) = -\frac{T}{2} \ln|\hat{\Sigma}_\eta(A(\delta, \Lambda))| + T \ln|A(\delta, \Lambda)|$$

**各项含义**：
- $-\frac{T}{2} \ln|\hat{\Sigma}_\eta|$：残差方差越小越好（拟合优度）
- $T \ln|A|$：网络效应的调整项（Jacobian）

---

#### 【Step 3】约束优化

现在优化问题变为：

$$\max_{\delta, \Lambda} \left\{ -\frac{T}{2} \ln|\hat{\Sigma}_\eta(A(\delta, \Lambda))| + T \ln|A(\delta, \Lambda)| \right\}$$

约束条件：
$$\text{s.t. } \delta_j \geq 0, \quad \sum_{j=1}^d \delta_j = 1, \quad \Lambda \text{ 对角且 } \rho_i \geq 0$$

**为什么需要约束？**
1. $\delta_j \geq 0$：网络权重不能为负（识别假设A3）
2. $\sum \delta_j = 1$：权重之和为1（凸组合，识别假设A3）
3. $\rho_i \geq 0$：反应系数为正（经济意义：网络传染应该是正向的）

**数值求解**：
```python
from scipy.optimize import minimize

def concentrated_loglik(params, R, F, W_list):
    """
    计算集中对数似然
    params = [delta_1, ..., delta_d, rho_1, ..., rho_K]
    """
    d = len(W_list)
    K = R.shape[1]
    
    delta = params[:d]
    rho = params[d:]
    
    # 构建 A 矩阵
    W_star = sum(delta[j] * W_list[j] for j in range(d))
    Lambda = np.diag(rho)
    A = np.eye(K) - Lambda @ W_star
    
    # 给定 A，估计 beta 和 Sigma_eta
    beta_hat = estimate_beta(A, R, F)  # OLS
    eta = A @ R.T - beta_hat @ F.T  # 残差
    Sigma_eta = np.diag(np.var(eta, axis=1, ddof=0))  # 对角协方差
    
    # 计算集中对数似然
    logdet_Sigma = np.sum(np.log(np.diag(Sigma_eta)))  # 对角阵的对数行列式
    logdet_A = np.linalg.slogdet(A)[1]  # |A| 的对数
    
    loglik = -0.5 * T * logdet_Sigma + T * logdet_A
    return -loglik  # minimize 需要最小化，所以取负

# 优化
result = minimize(concentrated_loglik, x0=initial_params, 
                  method='SLSQP', constraints=constraints)
```

---

### 3.5 为什么集中MLE有效？直观理解

#### 【类比】多元回归中的分步估计

假设你要估计 $Y = X\beta + Z\gamma + \epsilon$。

方法1：同时估计 $\beta$ 和 $\gamma$（维度高）
方法2：
1. 对任意给定的 $\gamma$，求最优 $\hat{\beta}(\gamma)$（OLS有解析解）
2. 代入得到只关于 $\gamma$ 的残差平方和
3. 优化 $\gamma$

方法2就是"集中"的思想——利用解析解降低维度。

#### 【本文的情况】

- 给定网络参数 $(\delta, \Lambda)$，最优因子暴露 $\beta$ 有解析解（OLS）
- 最优协方差 $\Sigma_\eta$ 也有解析解（样本方差）
- 所以可以把它们"集中"掉，只优化网络参数

#### 【计算优势】

| 方法 | 优化维度 | 问题 |
|------|----------|------|
| 直接MLE | 292维 | 高维优化，慢且容易陷入局部最优 |
| 集中MLE | 52维 | 低维优化，快且稳定 |

---

### 3.6 估计流程总结

```
输入：收益率 R (T×K)，因子 F (T×f)，网络矩阵 W_1,...,W_d (K×K)

Step 1: 初始化 delta 和 Lambda
        delta = [1/d, 1/d, ..., 1/d]  (等权重)
        Lambda = diag([0.1, 0.1, ..., 0.1])  (小值)

Step 2: 迭代优化
        for each iteration:
            # 2.1 构建复合网络和 A 矩阵
            W_star = sum(delta[j] * W_j)
            A = I - Lambda @ W_star
            
            # 2.2 给定 A，解析求解 beta 和 Sigma_eta (OLS)
            beta_hat = OLS(A @ R, F)
            eta = A @ R - beta_hat @ F
            Sigma_eta = diag(var(eta))
            
            # 2.3 计算集中对数似然
            loglik = -0.5*T*log(|Sigma_eta|) + T*log(|A|)
            
            # 2.4 更新 delta 和 Lambda (优化器)
            
Step 3: 输出
        delta_hat: 各网络方法的最优权重
        Lambda_hat: 各资产的网络反应系数
        beta_hat: 纯净的因子暴露
        W_star = sum(delta_hat[j] * W_j): 最优复合网络
```

---

### 3.7 关键公式速查

| 步骤 | 公式 | 含义 |
|------|------|------|
| 给定 $A$ 求 $\beta$ | $\hat{\beta}(A) = (A R)' F (F' F)^{-1}$ | OLS估计 |
| 计算残差 | $\hat{\eta} = A R - \hat{\beta} F$ | 结构残差 |
| 估计方差 | $\hat{\Sigma}_\eta = \frac{1}{T} \hat{\eta}' \hat{\eta}$ | 样本协方差（对角）|
| 集中似然 | $\ln L_c = -\frac{T}{2} \ln|\hat{\Sigma}_\eta| + T \ln|A|$ | 目标函数 |

---

**求解器推荐**：`scipy.optimize.minimize` with `method='SLSQP'` 或 `'trust-constr'`

---

## 四、识别假设 (Assumptions A1-A4)

| 假设 | 数学表达 | 经济含义 |
|------|----------|----------|
| A1 | $W_j \neq 0$ | 每层网络必须包含实际信息 |
| A2 | $W_i \neq W_j, \forall i \neq j$ | 不同层网络不能完全相同 |
| A3 | $\delta_j \geq 0, \sum \delta_j = 1$ | 凸组合约束 |
| A4 | $W^*$ 每行至少一个非零元素 | 每个资产至少与一个其他资产相连 |

**关键识别条件**：若两层网络完全相同（违反 A2），则无法区分其权重。

---

## 五、与标准因子模型对比

| 维度 | 标准线性因子模型 | 本文网络增强模型 |
|------|------------------|------------------|
| **基本方程** | $R_t = \alpha + \beta F_t + \epsilon_t$ | $A \cdot R_t = E[R_t] + \beta F_t + \eta_t$ |
| **残差结构** | $\epsilon_t$ 可相关 | $\eta_t$ 假设不相关（结构性）|
| **网络嵌入** | 无 | 通过 $A = I - \Lambda W^*$ 嵌入 |
| **beta 类型** | 简化式 beta | 结构性 beta（经网络调整）|
| **解释重点** | 因子解释收益 | 因子 + 网络共同解释 |

**核心洞见**：网络结构 $W^*$ 捕捉了**因子无法解释的部分相关性**——这正是金融传染研究的核心。

---

## 六、复现实现要点

### 6.1 数据预处理

- 日度收益率 → 周度收益率（因子模型使用周度）
- 所有网络矩阵必须**行标准化**
- 因子数据需与收益率数据日期对齐

### 6.2 数值稳定性

| 问题 | 解决方案 |
|------|----------|
| $A$ 不可逆 | 优化中加入行列式约束或正则化 |
| 参数边界 | 使用对数变换处理 $\rho_i > 0$ 约束 |
| 局部最优 | 多初始值尝试（等权重、随机权重）|

### 6.3 优化初始值建议

```python
# delta 初始值：等权重
delta_init = np.ones(d) / d

# Lambda 初始值：从单层网络估计
rho_init = estimate_single_layer_rho(W_1)
```

### 6.4 输出解读

- **$\delta_j$**：第 $j$ 种因果检验方法的相对重要性
  - 论文发现：QN（非参数分位数）通常权重最大
  - 说明：捕捉非线性极端风险传染对理解行业间联动至关重要

- **$\rho_i$**：资产 $i$ 对网络影响的敏感度
  - 值越大：该资产受网络传染效应越强

- **$W^*$**：最优复合网络，可用于后续网络分析（中心性、密度等）

---

## 七、关键公式汇总

| 编号 | 公式 | 名称 |
|------|------|------|
| Eq. 1 | $A(R_t - E[R_t]) = \bar{\beta} \cdot F_t + \eta_t$ | 结构方程（原始形式）|
| Eq. 2 | $R_t = E[R_t] + A^{-1}\bar{\beta} \cdot F_t + A^{-1}\eta_t$ | 简化式方程 |
| Eq. 3 | $A = I - \sum_{j=1}^{d} \rho_j \cdot W_j$ | 单参数多层形式 |
| **Eq. 4** | $A = I - \Lambda \cdot (\sum_{j=1}^{d} \delta_j \cdot W_j)$ | **核心模型（异质反应）**|
| Eq. 6 | $W^* = \sum_{j=1}^{d} \delta_j \cdot W_j$ | 复合网络 |
| Eq. 7 | $w_{i,j} = 1\{\text{拒绝 } H_0\}$ | 网络边定义 |

---

## 八、参考文献

- Billio et al. (2012): Granger 因果网络估计
- Billio et al. (2017): 带网络依赖的线性因子模型（本文基础）
- Anselin (1988): 空间计量经济学方法

---

*文档整理时间：2026-04-13*  
*对应论文章节：Section 2.3 - Model-based combination of causality networks*
