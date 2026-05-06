# Billio et al. (2023) §6：投入产出表作为邻接矩阵的完整解读

> **论文**：Billio, Caporin, Panzica & Pelizzon (2023)  
> **标题**：The impact of network connectivity on factor exposures, asset pricing, and portfolio diversification  
> **期刊**：International Review of Economics and Finance, 84, 196–223

---

## 1. 数据来源

作者使用美国经济分析局（BEA）编制的 **投入产出需求表**（Input–Output requirements tables）。
具体用的是 **"行业×行业 完全需求表"**（industry-by-industry total requirements table）。

- 时间范围：2004–2017，每年一张
- 行业分类：NAICS 两位码，共 15 个宏观行业
- 含义：向行业 j 每交付 1 美元的最终产品，行业 i 需要直接和间接提供多少美元的产出

---

## 2. 为什么 I/O 表可以当邻接矩阵 — 三层逻辑

### 2.1 经济含义层：I/O 表本身就是"谁连谁"

I/O 表的每个元素 $a_{ij}$ 表示"行业 i 的产出中有多少流向行业 j 作为投入"。这天然就是一个**有向有权图**：

- **节点** = 行业
- **有向边** i→j = 行业 i 给行业 j 供货
- **边权重** = 供货量（I/O 系数）

> 原文："the I/O matrix can be useful in understanding to which extent the output of an industry contributes to the input of other industries."

### 2.2 冲击传播层：供需链 = 冲击传播路径

供需关系不仅是静态的连接，更是冲击传播的渠道：

- 行业 A 出了问题 → 无法给行业 B 供货 → 行业 B 也受影响 → 行业 B 无法给行业 C 供货 → ...
- I/O 系数越大 = 边越粗 = 传播越强

> 原文："a shock to a single firm (or sector) could have a much larger impact on the macroeconomy if it reduces the output of not only this firm (or sector), but also of others that are connected to it through a network of input output linkages"

理论支撑：
- **Acemoglu et al. (2012)**：行业间投入产出联系使微观冲击可能引发宏观波动
- **Herskovic (2018)**：I/O 网络的集中度、稀疏度、生产率影响随机折现因子（SDF）
- **Ozdagli & Weber (2017)**：I/O 结构是货币政策冲击传播的重要渠道

### 2.3 资产定价层：冲击传播 → 利润变化 → 股价变动

冲击沿 I/O 网络传播 → 行业产出变化 → 利润变化 → 股票回报变化

传统因子模型（如 FF4）无法捕捉这种"内生系统性风险"，因为：
- 冲击可能源自微观层面（不是宏观因子）
- 传播路径取决于网络结构（不是因子暴露）
- 公共因子（如市场指数）无法代理这种局部传播

> 原文："If the shock origin is not related to global macroeconomic conditions, the production contraction might not be directly observable and might not be properly proxied by common factors, such as the market index."

---

## 3. 具体操作：I/O 表 → 邻接矩阵

1. **去掉对角线**：$w_{ii} = 0$（行业不对自己供货，不允许自环）
2. **行标准化**：按 Eq.(25) 标准化，使每行之和为 1
3. 得到 14 张年度邻接矩阵 $W_t$（2004–2017）

使用方式：月度回报数据 + 上一年的 I/O 网络（引入一年滞后，避免内生性）

---

## 4. 制造业冲击的例子（论文原文）

制造业遭受外生冲击 → 产出下降  
→ 农业、建筑业因无法获得足够的工业品投入而收缩  
→ 网络结构决定收缩强度（边越粗 = 收缩越剧烈）  
→ 工业股、农业股、建筑股价格均下跌  
→ 如果冲击不是宏观层面的，市场指数等公共因子无法捕捉

---

## 5. 实证结果摘要

- 15 个行业中，零售贸易（ρ=0.53）和信息业（ρ=0.47）空间效应最强且显著
- 空间模型残差平均相关：0.03（vs 四因子模型的 0.10）
- 引入 I/O 网络后，风险溢价的标准误大幅下降，统计显著性提高
  - 市场因子：p=0.004（vs 四因子模型 p=0.310）
  - HML 因子：p=0.015（vs 四因子模型 p=0.444）

---

## 6. 与 Bonaccolto et al. (2019) 的关系

| 维度 | Billio et al. (2023) | Bonaccolto et al. (2019) |
|------|---------------------|--------------------------|
| 邻接矩阵来源 | I/O 投入产出表 | 因果检验（GR, QB, Qo, QN） |
| 网络数量 | 单层（但时变） | 多层（4种方法 × 多分位数） |
| 组合方式 | 无（单一 W） | δ 加权组合多层网络 |
| 核心模型 | $A = I - \Lambda W_t$ | $A = I - \Lambda \sum \delta_j W_j$ |
| 本质相同点 | 都用 SAR 模型 $(I - \Lambda W)$ 的网络乘数效应 |

---

*最后更新：2026-04-21*
