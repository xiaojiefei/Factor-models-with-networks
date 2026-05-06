# 受限因变量、Tobit 模型与 Heckman 两步法：从直觉到推导

> 本笔记根据 `文献/高级计量经济学Ⅱ-6.受限因变量、Tobit模型、Heckman两步法 - 知乎.pdf` 用 PyMuPDF 提取的 5 页文本整理，并补全 PDF 中因公式抽取缺失而不完整的数学推导。重点不是背公式，而是理解：为什么 OLS 会错、Tobit 似然函数怎么来、逆 Mills 比率为什么出现、Heckman 两步法到底在修正什么。


> LaTeX 渲染提示：本文档使用标准 Markdown 数学公式写法，行内公式用 `$...$`，块级公式用 `$$...$$`。如果 VS Code 预览不能显示公式，请安装或启用 Markdown+Math、Markdown Preview Enhanced 等数学公式扩展，或用 Obsidian/Typora 打开。

## 0. PDF 原文结构对应

PyMuPDF 提取结果显示，这份 PDF 主要分成三部分：

1. **受限因变量**：区分删失（censored）和截断（truncated），并说明受限因变量下直接 OLS 会不一致。
2. **Tobit Model: MLE 方法**：把样本分成被删失的观测和未删失的观测，构造 Tobit 似然函数，并说明它可以分解为 Probit 部分和截断样本密度部分。
3. **Heckman's Sample Selection Model**：讨论样本选择偏差，说明 Heckman 两步法：第一步估计选择方程，第二步把 Inverse Mills Ratio 加入主回归。

由于 PDF 是网页导出的版本，很多公式在文本抽取中丢失了，所以本文档下面会按原文逻辑补全完整推导。

---

## 1. 为什么需要受限因变量模型？

普通线性回归写成：

$$
y_i = x_i'\beta + u_i, \quad E(u_i\mid x_i)=0.
$$

OLS 的核心前提是：我们观测到的 $y_i$ 就是真实连续变量，而且误差项在给定 $x_i$ 后均值为 0。

但很多经济变量并不是这样观测到的。例如：

| 场景 | 真实潜变量 | 实际观测变量 | 问题 |
|---|---|---|---|
| 家庭消费某商品 | 想买多少 | 不买时记录为 0 | 左删失 / censoring |
| 工资 | 市场潜在工资 | 只观察就业者工资 | 样本选择 / selection |
| 捐款金额 | 愿意捐多少 | 低于 0 的不可能，记为 0 | 下限约束 |
| 考试分数 | 潜在能力 | 分数限制在 0 到 100 | 双侧删失 |
| 企业投资 | 潜在投资倾向 | 投资额不能为负 | 角点解 |

这类问题的共同点是：

> 背后有一个连续的潜变量 $y_i^*$，但我们观察到的 $y_i$ 是经过限制、截断、删失或选择之后的结果。

所以，如果直接用 OLS 回归观测到的 $y_i$，通常会产生偏误或不一致。

---

## 2. 三个概念先分清：删失、截断、样本选择

### 2.1 删失 Censoring

删失是指：样本个体还在数据里，但因变量在某些区间被统一记录为边界值。

最典型的左删失：

$$
y_i = \begin{cases}
y_i^*, & y_i^*>0, \\
0, & y_i^*\le 0.
\end{cases}
$$

这里 $y_i^*$ 是真实潜变量，$y_i$ 是观测变量。

例子：

- 家庭对某商品的消费金额；
- 企业投资额；
- 捐款金额；
- 工作小时数。

如果某人潜在消费倾向是负的，现实中不可能消费负金额，所以数据里只看到 0。

关键点：

> 删失数据中，$y_i=0$ 的个体仍然在样本里，只是它们的真实 $y_i^*$ 没被完整观测。

---

### 2.2 截断 Truncation

截断是指：某些个体根本不进入样本。

例如只调查收入大于 5000 的人：

$$
\text{sample observed only if } y_i^*>5000.
$$

那么收入低于 5000 的人完全不在数据中。

删失和截断的区别：

| 类型 | 个体是否在样本中 | 因变量如何记录 |
|---|---|---|
| 删失 censoring | 在 | 边界值，如 0 |
| 截断 truncation | 不在 | 完全看不到 |

这一区别非常重要，因为它们对应的 likelihood 不一样。

---

### 2.3 样本选择 Sample Selection

样本选择是指：我们只在满足某个选择条件时观察到结果变量。

例如工资方程：

$$
w_i^* = x_i'\beta + u_i,
$$

但工资 $w_i$ 只有在个体就业时才被观察到。

就业选择方程：

$$
d_i^* = z_i'\gamma + v_i,
$$

$$
d_i = 1(d_i^*>0).
$$

如果 $d_i=1$，观察到工资；如果 $d_i=0$，工资缺失。

关键问题是：就业不是随机的。如果影响就业的不可观测因素 $v_i$ 和影响工资的不可观测因素 $u_i$ 相关，则：

$$
E(u_i\mid d_i=1,x_i)\ne 0.
$$

这会导致只对就业者样本做 OLS 得到有偏估计。

---

## 3. Tobit 模型：从潜变量开始

### 3.1 模型设定

Tobit 模型通常从一个潜变量模型开始：

$$
y_i^* = x_i'\beta + u_i,
$$

其中：

$$
u_i \mid x_i \sim N(0,\sigma^2).
$$

但我们观察到的是：

$$
y_i = \max(0,y_i^*).
$$

也就是：

$$
y_i = \begin{cases}
y_i^*, & y_i^*>0, \\
0, & y_i^*\le 0.
\end{cases}
$$

这就是标准左删失 Tobit 模型。

---

## 4. 为什么 Tobit 不能直接用 OLS？

如果用 OLS 回归 $y_i$ 对 $x_i$，实际上是在估计：

$$
E(y_i\mid x_i).
$$

但 Tobit 下：

$$
y_i=\max(0,y_i^*),
$$

所以：

$$
E(y_i\mid x_i) \ne x_i'\beta.
$$

原因是 $y_i$ 不是线性变量，而是潜变量经过非线性截断后的结果。

更具体地，令：

$$
z_i = \frac{x_i'\beta}{\sigma}.
$$

因为：

$$
y_i = y_i^* \cdot 1(y_i^*>0),
$$

所以：

$$
E(y_i\mid x_i)=P(y_i^*>0\mid x_i)E(y_i^*\mid y_i^*>0,x_i).
$$

其中：

$$
P(y_i^*>0\mid x_i)=\Phi(z_i).
$$

而正值条件下的潜变量均值为：

$$
E(y_i^*\mid y_i^*>0,x_i)=x_i'\beta+\sigma\frac{\phi(z_i)}{\Phi(z_i)}.
$$

因此：

$$
E(y_i\mid x_i)=\Phi(z_i)x_i'\beta+\sigma\phi(z_i).
$$

这个式子是非线性的，所以直接 OLS 不会估计出结构参数 $\beta$。

---

## 5. Tobit 似然函数推导

Tobit 的关键是：样本中有两类观测。

### 5.1 第一类：$y_i=0$

当 $y_i=0$ 时，说明：

$$
y_i^*\le 0.
$$

也就是：

$$
x_i'\beta+u_i\le 0.
$$

等价于：

$$
u_i\le -x_i'\beta.
$$

标准化：

$$
\frac{u_i}{\sigma}\le -\frac{x_i'\beta}{\sigma}.
$$

所以概率为：

$$
P(y_i=0\mid x_i)=\Phi\left(-\frac{x_i'\beta}{\sigma}\right).
$$

---

### 5.2 第二类：$y_i>0$

当 $y_i>0$ 时，观测到完整的：

$$
y_i=y_i^*=x_i'\beta+u_i.
$$

因此：

$$
y_i\mid x_i \sim N(x_i'\beta,\sigma^2).
$$

密度为：

$$
f(y_i\mid x_i)=\frac{1}{\sigma}\phi\left(\frac{y_i-x_i'\beta}{\sigma}\right).
$$

---

### 5.3 合并 likelihood

令 $d_i=1(y_i>0)$。

单个观测的 likelihood 为：

$$
L_i =
\left[\frac{1}{\sigma}\phi\left(\frac{y_i-x_i'\beta}{\sigma}\right)\right]^{d_i}
\left[\Phi\left(-\frac{x_i'\beta}{\sigma}\right)\right]^{1-d_i}.
$$

取对数：

$$
\ell_i = d_i\left[-\log\sigma+\log\phi\left(\frac{y_i-x_i'\beta}{\sigma}\right)\right]
+(1-d_i)\log\Phi\left(-\frac{x_i'\beta}{\sigma}\right).
$$

总 log-likelihood：

$$
\ell(\beta,\sigma)=\sum_{i=1}^N \ell_i.
$$

Tobit 的 MLE 就是最大化这个 log-likelihood。

---

## 6. Tobit 系数如何解释？

Tobit 的 $\beta$ 不是简单的：$x$ 增加 1，观测 $y$ 增加 $\beta$。

因为 $x$ 同时影响两件事：

1. 是否从 0 变成正值；
2. 在正值样本中，正值的大小。

### 6.1 对潜变量的边际效应

潜变量：

$$
y_i^*=x_i'\beta+u_i.
$$

所以：

$$
\frac{\partial E(y_i^*\mid x_i)}{\partial x_{ik}}=\beta_k.
$$

这是结构意义上的边际效应。

---

### 6.2 对观测变量的边际效应

观测变量期望：

$$
E(y_i\mid x_i)=\Phi(z_i)x_i'\beta+\sigma\phi(z_i),
$$

其中 $z_i=x_i'\beta/\sigma$。

对 $x_{ik}$ 求导，结果为：

$$
\frac{\partial E(y_i\mid x_i)}{\partial x_{ik}}=\Phi(z_i)\beta_k.
$$

直觉：

> 观测变量的边际效应 = 潜变量边际效应 × 成为非零观测的概率。

如果很多个体都卡在 0，那么 $\Phi(z_i)$ 小，观测变量对 $x$ 的反应也会被压低。

---

### 6.3 对正值条件均值的边际效应

正值条件下：

$$
E(y_i\mid y_i>0,x_i)=x_i'\beta+\sigma\lambda(z_i),
$$

其中：

$$
\lambda(z_i)=\frac{\phi(z_i)}{\Phi(z_i)}.
$$

这个 $\lambda(z)$ 就是逆 Mills 比率。

它的边际效应不是简单的 $\beta_k$，因为 $\lambda(z_i)$ 也随 $x_i$ 变化。

---

## 7. Heckman 两步法解决什么问题？

Tobit 假设“是否被观察”和“结果大小”由同一个潜变量机制决定。

但很多场景不是这样。例如工资：

- 是否就业由就业选择决定；
- 工资水平由工资方程决定。

这两个过程不同。

Heckman 模型把它分成两个方程。

### 7.1 结果方程

潜在工资：

$$
y_i^* = x_i'\beta + u_i.
$$

但只有被选择进入样本时才能观察到 $y_i$。

---

### 7.2 选择方程

选择机制：

$$
s_i^* = z_i'\gamma + v_i,
$$

$$
s_i = 1(s_i^*>0).
$$

只有当 $s_i=1$ 时，我们观察到：

$$
y_i=y_i^*.
$$

---

### 7.3 关键假设：误差项联合正态

Heckman 模型假设：

$$
\begin{pmatrix}
u_i \\
v_i
\end{pmatrix}
\sim N\left(
\begin{pmatrix}
0 \\
0
\end{pmatrix},
\begin{pmatrix}
\sigma_u^2 & \rho\sigma_u \\
\rho\sigma_u & 1
\end{pmatrix}
\right).
$$

这里把 $v_i$ 的方差标准化为 1，是因为 Probit 选择方程只能识别尺度。

如果：

$$
\rho=0,
$$

说明选择误差和结果误差不相关，样本选择不是问题。

如果：

$$
\rho\ne 0,
$$

说明进入样本的人在不可观测因素上并不是随机的，直接 OLS 会有 selection bias。

---

## 8. Heckman 偏误从哪里来？

我们只在 $s_i=1$ 的样本中观察到 $y_i$。

因此我们实际估计的是：

$$
E(y_i\mid x_i,s_i=1).
$$

代入结果方程：

$$
E(y_i\mid x_i,s_i=1)=x_i'\beta+E(u_i\mid s_i=1).
$$

而：

$$
s_i=1 \Longleftrightarrow s_i^*>0 \Longleftrightarrow z_i'\gamma+v_i>0.
$$

所以：

$$
s_i=1 \Longleftrightarrow v_i>-z_i'\gamma.
$$

于是：

$$
E(u_i\mid s_i=1)=E(u_i\mid v_i>-z_i'\gamma).
$$

如果 $u_i$ 和 $v_i$ 相关，则这个条件期望不是 0。

这就是样本选择偏误。

---

## 9. 逆 Mills 比率的推导

由于 $(u_i,v_i)$ 联合正态，有条件期望：

$$
E(u_i\mid v_i)=\rho\sigma_u v_i.
$$

因此：

$$
E(u_i\mid v_i>-z_i'\gamma)=\rho\sigma_u E(v_i\mid v_i>-z_i'\gamma).
$$

因为 $v_i\sim N(0,1)$，对标准正态有：

$$
E(v_i\mid v_i>a)=\frac{\phi(a)}{1-\Phi(a)}.
$$

令：

$$
a=-z_i'\gamma.
$$

则：

$$
E(v_i\mid v_i>-z_i'\gamma)=\frac{\phi(-z_i'\gamma)}{1-\Phi(-z_i'\gamma)}.
$$

利用标准正态对称性：

$$
\phi(-c)=\phi(c), \quad 1-\Phi(-c)=\Phi(c).
$$

所以：

$$
E(v_i\mid v_i>-z_i'\gamma)=\frac{\phi(z_i'\gamma)}{\Phi(z_i'\gamma)}.
$$

定义逆 Mills 比率：

$$
\lambda_i=\lambda(z_i'\gamma)=\frac{\phi(z_i'\gamma)}{\Phi(z_i'\gamma)}.
$$

于是：

$$
E(u_i\mid s_i=1)=\rho\sigma_u\lambda_i.
$$

因此：

$$
E(y_i\mid x_i,s_i=1)=x_i'\beta+\rho\sigma_u\lambda_i.
$$

这就是 Heckman 修正项的来源。

---

## 10. Heckman 两步法

### 10.1 第一步：估计选择方程 Probit

先用全样本估计：

$$
s_i = 1(z_i'\gamma+v_i>0).
$$

也就是 Probit：

$$
P(s_i=1\mid z_i)=\Phi(z_i'\gamma).
$$

得到 $\hat{\gamma}$。

然后计算：

$$
\hat{\lambda}_i=\frac{\phi(z_i'\hat{\gamma})}{\Phi(z_i'\hat{\gamma})}.
$$

---

### 10.2 第二步：在被选择样本中加入 IMR 做 OLS

对 $s_i=1$ 的样本估计：

$$
y_i=x_i'\beta+\theta\hat{\lambda}_i+\varepsilon_i,
$$

其中：

$$
\theta=\rho\sigma_u.
$$

如果 $\theta$ 显著不为 0，说明存在样本选择偏误。

如果 $\theta$ 不显著，可以认为选择偏误不明显。

---

## 11. Heckman 两步法的直觉

直接对就业者工资做 OLS 的问题是：

> 就业者不是随机抽出来的，他们因为某些不可观测特质更容易就业，而这些特质也可能影响工资。

Heckman 第一步用 Probit 估计每个人进入样本的概率。

第二步把由选择过程引起的“非随机性”浓缩成一个变量 $\hat{\lambda}_i$，放回工资方程里。

所以 Heckman 两步法本质上是在结果方程中加入一个遗漏变量修正项。

---

## 12. Tobit 和 Heckman 的区别

| 维度 | Tobit | Heckman |
|---|---|---|
| 核心问题 | 因变量被删失 | 因变量只在选择样本中被观察 |
| 是否有单独选择方程 | 没有 | 有 |
| 结果为 0 的个体 | 仍在样本中 | 结果变量可能缺失 |
| 模型机制 | 一个潜变量同时决定是否为正和正值大小 | 选择机制和结果机制分开 |
| 典型例子 | 消费额、投资额、捐款额 | 工资只观察就业者 |
| 估计方法 | MLE | 两步法或 FIML |

一句话区分：

> Tobit 适合“观察到了 0”；Heckman 适合“根本没观察到结果变量”。

---

## 13. 为什么 Heckman 需要排除变量？

理论上，Heckman 可以只靠 Probit 的非线性识别。

但实践中最好让选择方程 $z_i$ 中包含至少一个不进入结果方程 $x_i$ 的变量。

这个变量叫 exclusion restriction。

例如工资模型：

- 选择方程：是否就业；
- 结果方程：工资水平。

可以进入选择方程但不直接进入工资方程的变量：

- 家中幼儿数量；
- 配偶收入；
- 家庭照料责任；
- 离就业中心距离。

它们影响是否就业，但在控制教育、经验等变量后，不应直接影响工资。

如果没有排除变量，$\hat{\lambda}_i$ 可能和 $x_i$ 高度共线，第二步估计不稳定。

---

## 14. 手算理解：为什么逆 Mills 比率越大，选择偏误越强？

逆 Mills 比率：

$$
\lambda(c)=\frac{\phi(c)}{\Phi(c)}.
$$

当 $c=z_i'\gamma$ 很小时，说明该个体被选择的概率很低。

如果一个本来不太可能被选择的人最终被选择了，说明他可能有很强的正向不可观测选择因素 $v_i$。

如果 $v_i$ 和结果误差 $u_i$ 正相关，那么他的结果变量也会系统性偏高。

所以：

- 选择概率低但被选中 → $\lambda_i$ 大；
- 选择概率高且被选中 → $\lambda_i$ 小；
- $\lambda_i$ 捕捉“被选中样本的非随机性”。

---

## 15. 常见误区

### 误区 1：Tobit 和 Heckman 都是处理 0，所以一样

不一样。

Tobit 的 0 是真实观测到的边界值；Heckman 的缺失是因为没进入样本。

如果你把 Heckman 问题误用 Tobit，会把“没有观察工资”当成“工资等于 0”，这是错误的。

---

### 误区 2：Heckman 第二步就是随便加一个控制变量

不是。

$\hat{\lambda}_i$ 是从联合正态假设和选择条件期望严格推导出来的：

$$
E(u_i\mid s_i=1)=\rho\sigma_u\lambda_i.
$$

它不是经验上随便加的变量，而是选择偏误的控制函数。

---

### 误区 3：Tobit 的 $\beta$ 就是观测 $y$ 的边际效应

不是。

Tobit 的 $\beta$ 是潜变量 $y^*$ 的边际效应。

观测变量的边际效应是：

$$
\Phi(z_i)\beta_k.
$$

---

### 误区 4：Heckman 只要跑两步就一定正确

不一定。

Heckman 依赖几个关键条件：

1. 选择方程设定正确；
2. 联合正态假设合理；
3. 最好有有效排除变量；
4. 选择变量和结果变量的经济机制要讲得通。

---

## 16. 一个完整例子：女性工资的样本选择

假设你研究教育对女性工资的影响。

结果方程：

$$
\log(wage_i)=\beta_0+\beta_1 educ_i+\beta_2 exper_i+u_i.
$$

但工资只有就业女性才有。

选择方程：

$$
work_i^*=\gamma_0+\gamma_1 educ_i+\gamma_2 age_i+\gamma_3 children_i+\gamma_4 spouse\_income_i+v_i.
$$

$$
work_i=1(work_i^*>0).
$$

第一步：用全样本估计 Probit：

$$
P(work_i=1)=\Phi(z_i'\gamma).
$$

计算：

$$
\hat\lambda_i=\frac{\phi(z_i'\hat\gamma)}{\Phi(z_i'\hat\gamma)}.
$$

第二步：只对就业女性估计：

$$
\log(wage_i)=\beta_0+\beta_1 educ_i+\beta_2 exper_i+\theta\hat\lambda_i+\varepsilon_i.
$$

如果 $\theta>0$ 且显著，说明选择进入就业样本的人在不可观测工资能力上更高，直接 OLS 高估或低估教育回报的方向取决于 $\lambda$ 与教育等变量的相关结构。

---

## 17. Tobit、Truncated Regression、Heckman 的 likelihood 对比

### 17.1 Tobit likelihood

左删失 Tobit：

$$
L_i=
\begin{cases}
\Phi\left(-\frac{x_i'\beta}{\sigma}\right), & y_i=0, \\
\frac{1}{\sigma}\phi\left(\frac{y_i-x_i'\beta}{\sigma}\right), & y_i>0.
\end{cases}
$$

---

### 17.2 Truncated regression likelihood

如果只观察 $y_i>0$ 的样本，那么密度要写成条件密度：

$$
f(y_i\mid y_i>0,x_i)=\frac{f(y_i\mid x_i)}{P(y_i>0\mid x_i)}.
$$

因此：

$$
f(y_i\mid y_i>0,x_i)=
\frac{\frac{1}{\sigma}\phi\left(\frac{y_i-x_i'\beta}{\sigma}\right)}
{\Phi\left(\frac{x_i'\beta}{\sigma}\right)}.
$$

这和 Tobit 不一样，因为 Tobit 里 $y=0$ 的人仍然在样本里；截断回归里他们消失了。

---

### 17.3 Heckman full likelihood

Heckman 也可以用 full-information maximum likelihood 估计。

它同时写出：

1. 被选择且观察到结果的联合密度；
2. 未被选择的概率。

但两步法更直观，也更容易手工理解。

---

## 18. 与普通 OLS 的关系

### 18.1 什么时候 OLS 可以用？

如果没有删失、截断、选择问题，并且：

$$
E(u_i\mid x_i)=0,
$$

OLS 是合适的。

---

### 18.2 什么时候 Tobit 更合适？

当因变量有大量边界值，而且这些边界值来自潜变量被压到边界，例如：

$$
y_i=\max(0,y_i^*),
$$

Tobit 更合适。

---

### 18.3 什么时候 Heckman 更合适？

当结果变量只在非随机选择样本中被观察到，例如：

- 只有就业者有工资；
- 只有上市公司披露某指标；
- 只有申请贷款者才有贷款利率；
- 只有出口企业才有出口额。

如果选择是否发生与结果误差相关，Heckman 更合适。

---

## 19. 从代码角度理解 Heckman 两步法

伪代码如下：

```python
# Step 1: Probit selection equation on full sample
selection_model = Probit(s, Z).fit()
gamma_hat = selection_model.params
index = Z @ gamma_hat
lambda_hat = normal_pdf(index) / normal_cdf(index)

# Step 2: OLS outcome equation on selected sample only
selected = (s == 1)
X_aug = np.column_stack([X[selected], lambda_hat[selected]])
outcome_model = OLS(y[selected], X_aug).fit()
```

第二步中 `lambda_hat` 的系数就是：

$$
\theta=\rho\sigma_u.
$$

它是否显著，是判断选择偏误的重要依据。

---

## 20. 学习路线建议

如果你现在完全没看懂，建议按以下顺序理解：

1. **先分清观测机制**：
   - 是 $y=0$ 但人在样本里？Tobit。
   - 是 $y$ 缺失，因为人没进入样本？Heckman。

2. **再理解潜变量**：
   - Tobit：$y^*$ 被压到边界；
   - Heckman：选择方程决定是否能看到 $y^*$。

3. **再看 likelihood**：
   - Tobit：零值概率 + 正值密度；
   - Heckman：选择条件下误差均值不为 0。

4. **最后理解逆 Mills 比率**：
   - 它不是凭空来的；
   - 它是正态变量在被截断条件下的条件期望。

---

## 21. 一句话总结

Tobit 模型解决的是：

> 因变量被边界限制，所以观测值不是线性潜变量。

Heckman 两步法解决的是：

> 结果变量只在非随机选择样本中被观察到，所以误差项条件均值不再为 0。

逆 Mills 比率解决的是：

> 把“被选择进入样本”造成的误差项条件均值，显式写成一个可估计的修正项。

---

## 22. 最核心公式速查

### Tobit

潜变量：

$$
y_i^*=x_i'\beta+u_i, \quad u_i\sim N(0,\sigma^2).
$$

观测变量：

$$
y_i=\max(0,y_i^*).
$$

Likelihood：

$$
L_i =
\left[\frac{1}{\sigma}\phi\left(\frac{y_i-x_i'\beta}{\sigma}\right)\right]^{1(y_i>0)}
\left[\Phi\left(-\frac{x_i'\beta}{\sigma}\right)\right]^{1(y_i=0)}.
$$

观测变量均值：

$$
E(y_i\mid x_i)=\Phi(z_i)x_i'\beta+\sigma\phi(z_i),
\quad z_i=\frac{x_i'\beta}{\sigma}.
$$

观测变量边际效应：

$$
\frac{\partial E(y_i\mid x_i)}{\partial x_{ik}}=\Phi(z_i)\beta_k.
$$

---

### Heckman

结果方程：

$$
y_i^*=x_i'\beta+u_i.
$$

选择方程：

$$
s_i^*=z_i'\gamma+v_i,
\quad s_i=1(s_i^*>0).
$$

误差相关：

$$
\operatorname{corr}(u_i,v_i)=\rho.
$$

选择后的条件均值：

$$
E(y_i\mid x_i,s_i=1)=x_i'\beta+\rho\sigma_u\lambda_i.
$$

逆 Mills 比率：

$$
\lambda_i=\frac{\phi(z_i'\gamma)}{\Phi(z_i'\gamma)}.
$$

第二步回归：

$$
y_i=x_i'\beta+\theta\hat\lambda_i+\varepsilon_i,
\quad \theta=\rho\sigma_u.
$$
