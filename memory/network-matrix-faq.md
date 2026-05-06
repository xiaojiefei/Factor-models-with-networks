# 网络矩阵常见问题解答

## 问题1：只能使用无权矩阵吗？

**答案：不是。**

### 无权矩阵 vs 加权矩阵

**无权矩阵（论文做法）**：
```python
W = [[0, 1, 0],    # 1 = 存在连接, 0 = 无连接
     [1, 0, 1],
     [0, 1, 0]]
```
- 论文使用无权矩阵是因为因果检验结果是二元的
- 原假设 H₀：i 对 j 无因果影响
- 结果：拒绝 H₀ → w_{i,j} = 1；接受 → w_{i,j} = 0

**加权矩阵（扩展）**：
```python
W = [[0, 0.8, 0],    # 0.8 = 连接强度
     [0.3, 0, 0.7],
     [0, 0.5, 0]]
```
- 数学上完全可以处理
- 可以表示因果关系的**强度**
- 例如：Granger因果的F统计量、相关系数等

### 如何在代码中使用加权矩阵？

修改 `generate_networks` 函数：

```python
def generate_weighted_networks(K: int = 13):
    """生成加权网络示例"""
    W_list = []
    
    # 方法1：用相关系数作为权重
    returns = np.random.randn(252, K)  # 历史收益率
    corr_matrix = np.corrcoef(returns.T)
    np.fill_diagonal(corr_matrix, 0)
    W1 = row_normalize(np.abs(corr_matrix))  # 取绝对值后标准化
    W_list.append(W1)
    
    # 方法2：用Granger因果的F统计量
    # F_stats = compute_granger_f_stats(data)  # 你的计算函数
    # W2 = row_normalize(F_stats)
    # W_list.append(W2)
    
    return W_list
```

**关键要求**：无论有权无权，都必须**行标准化**！

---

## 问题2：行标准化是什么意思？

### 定义

行标准化 = 让矩阵每一行的元素之和为1（对角线为0）

### 计算方法

```python
def row_normalize(W):
    """行标准化"""
    W_norm = W.copy()
    np.fill_diagonal(W_norm, 0)  # 对角线设为0（无自环）
    row_sums = W_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # 避免除以0
    W_norm = W_norm / row_sums
    return W_norm
```

### 直观示例

**原始矩阵**（无权）：
```
      资产0  资产1  资产2   行和
资产0   0     1     1     2
资产1   1     0     0     1
资产2   0     1     0     1
```

**行标准化后**：
```
      资产0   资产1   资产2   行和
资产0   0    0.5    0.5     1
资产1   1     0      0      1
资产2   0     1      0      1
```

### 为什么要行标准化？

| 原因 | 解释 |
|------|------|
| **1. 数值稳定性** | 防止 $A = I - \Lambda W$ 出现奇异（行列式接近0） |
| **2. 参数可解释** | $\Lambda = \text{diag}(\rho)$ 中，$\rho \in [0,1]$ 有明确含义 |
| **3. 可比性** | 不同资产度数不同，标准化后可比较 |

### 参数 $\rho$ 的经济含义

假设 $\rho_i = 0.3$：
- 表示资产 $i$ 受到网络影响的**强度比例**
- $1 - \rho_i = 0.7$ 表示资产 $i$ 的**独立部分**
- 如果行和不为1，这个解释就不成立

---

## 快速验证

运行以下代码验证行标准化是否正确：

```python
import numpy as np

# 测试行标准化
W = np.array([[0, 1, 1],
              [1, 0, 0],
              [0, 1, 0]], dtype=float)

W_norm = row_normalize(W)

print("标准化后的矩阵:")
print(W_norm)
print("\n每行的和(应该都是1):")
print(W_norm.sum(axis=1))
print("\n对角线(应该都是0):")
print(np.diag(W_norm))
```

**输出**：
```
标准化后的矩阵:
[[0.  0.5 0.5]
 [1.  0.  0. ]
 [0.  1.  0. ]]

每行的和(应该都是1):
[1. 1. 1.]

对角线(应该都是0):
[0. 0. 0.]
```

---

## 总结

1. **矩阵可以有权或无权**，关键是行标准化
2. **行标准化是必须的**，否则模型估计不稳定
3. **加权矩阵**在实证中可能更有意义（捕捉因果强度）
