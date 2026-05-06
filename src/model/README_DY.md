# TVP-VAR-DY 溢出网络分析使用说明

## 文件说明

- `TVP-VAR-DY.R`: DY溢出网络分析主代码
- `prepare_data.py`: Python数据预处理脚本
- `../data/processed/`: 数据输出目录

## 数据准备

### 1. 数据预处理（已完成）

Wind行业数据已处理为以下文件：

```
src/data/processed/
├── industry_prices.csv          # 行业收盘价
├── industry_returns.csv         # 行业日收益率
├── industry_volatility.csv      # 行业滚动波动率(22日)
├── industry_names.txt           # 行业名称列表
└── data_info.json               # 数据信息
```

### 2. 行业列表(31个)

交通运输、传媒、公用事业、农林牧渔、医药生物、商贸零售、国防军工、
基础化工、家用电器、建筑材料、建筑装饰、房地产、有色金属、机械设备、
汽车、煤炭、环保、电力设备、电子、石油石化、社会服务、纺织服饰、
综合、美容护理、计算机、轻工制造、通信、钢铁、银行、非银金融、食品饮料

## DY模型运行步骤

### 1. 在R中设置工作目录

```r
setwd("D:/桌面数据/工作论文/带耦合多层网络/src/model")
```

### 2. 加载代码

```r
source("TVP-VAR-DY.R")
```

### 3. 运行完整分析

```r
results <- main()
```

这将自动运行：
- 收益率溢出网络分析
- 波动率溢出网络分析
- 提取时变网络切片(5个时间点)
- 保存所有邻接矩阵

## 输出文件

运行后会生成以下文件在 `src/data/processed/` 目录：

### 平均邻接矩阵（全样本平均）

| 文件名 | 说明 |
|--------|------|
| `dy_network_to_ret_norm.csv` | 收益率To网络（行标准化） |
| `dy_network_from_ret_norm.csv` | 收益率From网络（行标准化） |
| `dy_network_to_vol_norm.csv` | 波动率To网络（行标准化） |
| `dy_network_from_vol_norm.csv` | 波动率From网络（行标准化） |

### 时变邻接矩阵（特定时间点）

| 文件名格式 | 说明 |
|------------|------|
| `dy_network_to_ret_t{时间点}_{日期}_norm.csv` | 收益率To网络切片 |
| `dy_network_from_ret_t{时间点}_{日期}_norm.csv` | 收益率From网络切片 |
| `dy_network_to_vol_t{时间点}_{日期}_norm.csv` | 波动率To网络切片 |
| `dy_network_from_vol_t{时间点}_{日期}_norm.csv` | 波动率From网络切片 |

### 溢出指数

| 文件名 | 说明 |
|--------|------|
| `dy_spillover_indices_ret.csv` | 各行业收益率溢出指数 |
| `dy_spillover_indices_vol.csv` | 各行业波动率溢出指数 |

## 网络类型说明

### To 网络（发出溢出）

- `W_to[i,j]` = 行业j对行业i的溢出贡献
- 列是**发出者**（To others）
- 表示该行业对其他行业的影响力

### From 网络（接收溢出）

- `W_from[i,j]` = 行业i从行业j接收的溢出
- 列是**来源**（From others）
- 表示该行业受其他行业影响的程度

### 行标准化

所有邻接矩阵都经过行标准化，每行元素之和为1：

```r
W_norm[i,j] = W[i,j] / sum(W[i,])
```

## 自定义分析

### 提取特定时间点的邻接矩阵

```r
# 加载数据并运行DY分析
data <- load_industry_data("returns")
dca_result <- ConnectednessApproach(data, ...)

# 提取第100个时间点的邻接矩阵
slice <- extract_time_varying_adjacency(
  dca_result,
  t = 100,              # 时间点
  threshold = 0.5,      # 阈值(%)
  network_type = "to",  # "to" 或 "from"
  data_type = "returns"
)
```

### 修改分析参数

在 `main()` 函数中修改以下参数：

```r
params <- list(
  nlag = 2,           # VAR滞后阶数
  nfore = 10,         # 预测期数
  window.size = 252,  # 滚动窗口大小(日)
  kappa1 = 0.99,      # TVP衰减参数1
  kappa2 = 0.96,      # TVP衰减参数2
  threshold = 0.5     # 邻接矩阵阈值(%)
)
```

## 在Python中使用邻接矩阵

```python
import numpy as np
import pandas as pd

# 读取收益率To网络
W_ret = np.loadtxt('src/data/processed/dy_network_to_ret_norm.csv', delimiter=',')

# 读取波动率To网络
W_vol = np.loadtxt('src/data/processed/dy_network_to_vol_norm.csv', delimiter=',')

print(f"收益率网络形状: {W_ret.shape}")
print(f"波动率网络形状: {W_vol.shape}")

# 检查行标准化
print(f"行和: {W_ret.sum(axis=1)}")
```

## 注意事项

1. **运行时间**: TVP-VAR估计可能需要10-30分钟，取决于电脑性能
2. **内存要求**: 建议至少8GB内存
3. **R包依赖**: 首次运行需要安装 `ConnectednessApproach` 包
4. **中文路径**: 如果出现编码问题，确保R的locale设置为UTF-8

## 参考文献

- Diebold, F. X., & Yilmaz, K. (2009). Measuring financial asset return and volatility spillovers. Economics Letters.
- Diebold, F. X., & Yilmaz, K. (2012). Better to give than to receive: Predictive directional measurement of volatility spillovers. International Journal of Forecasting.
- Diebold, F. X., & Yilmaz, K. (2014). On the network topology of variance decompositions: Measuring the connectedness of financial firms. Journal of Econometrics.
- Gabauer, D. (2020). The Connectedness Approach: A Review and New Developments. Working Paper.
