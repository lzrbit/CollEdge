# 有向协作机制下的联盟博弈公式推导

## 1. 问题背景

原论文《Decentralized Dynamic Cooperation of Personalized Models for Federated Continual Learning》中的 Benefit calculation 基于**对称聚合**假设。引入有向协作机制后，聚合变为**非对称**的，因此需要重新审视相关公式。

---

## 2. 原始论文公式回顾

### 2.1 Benefit Calculation (Section 3.2)

原论文定义客户端 $i$ 在联盟 $\mathcal{C}$ 中的收益为聚合梯度与自身梯度的对齐程度：

$$
\phi_i(\mathcal{C}) = \frac{\langle g_{\mathcal{C}}, g_i \rangle}{\|g_{\mathcal{C}}\|}
$$

其中聚合梯度为：
$$
g_{\mathcal{C}} = \sum_{j \in \mathcal{C}} g_j
$$

展开后得到（Appendix A）：

$$
\phi_i(\mathcal{C}) = \frac{\sum_{j \in \mathcal{C} \setminus \{i\}} \|g_j\| \cdot \cos(g_i, g_j)}{\sqrt{\sum_{j,k \in \mathcal{C} \setminus \{i\}} \|g_j\| \|g_k\| \cos(g_j, g_k)}}
$$

### 2.2 关键假设

原公式基于以下**隐含假设**：

1. **对称聚合**：联盟内所有成员获得相同的聚合模型 $\theta_{\mathcal{C}}$
2. **对称相似度**：$\cos(g_i, g_j) = \cos(g_j, g_i)$（余弦相似度天然对称）
3. **均等权重**：每个成员对聚合模型的贡献仅由梯度范数决定

---

## 3. 有向协作下的影响分析

### 3.1 有向协作机制回顾

在有向协作机制中，客户端 $i$ 获得的个性化聚合模型为：

$$
\theta_i^{agg} = w_{self} \cdot \theta_i + (1 - w_{self}) \cdot \sum_{j \in \mathcal{C}, j \neq i} w_{ij} \cdot \theta_j
$$

其中权重 $w_{ij}$ 基于有向协作分数 $D[i][j]$ 计算：

$$
w_{ij} = \frac{\exp(D[i][j] / \tau) \cdot s_j}{\sum_{k \in \mathcal{C}, k \neq i} \exp(D[i][k] / \tau) \cdot s_k}
$$

### 3.2 核心变化

| 原始机制 | 有向协作机制 |
|---------|------------|
| 所有成员获得相同模型 $\theta_{\mathcal{C}}$ | 每个成员获得个性化模型 $\theta_i^{agg}$ |
| 权重仅取决于样本数/梯度范数 | 权重还取决于有向协作分数 $D[i][j]$ |
| $w_{ij} = w_{ji}$（对称） | $w_{ij} \neq w_{ji}$（非对称） |

---

## 4. 公式推导分析

### 4.1 是否需要修改联盟形成阶段的 Benefit 计算？

**结论：可以不修改，但建议引入有向性改进。**

#### 4.1.1 不修改的理由

当前实现采用**两阶段分离**的设计：

1. **联盟形成阶段**：基于对称相似度决定联盟结构
2. **聚合阶段**：基于有向协作分数进行个性化聚合

这种分离设计的优点：
- 联盟形成保持稳定（博弈论性质不变）
- 有向协作在聚合阶段发挥作用
- 实现复杂度较低

#### 4.1.2 建议修改的理由

如果希望联盟形成也考虑有向性，则需要重新定义 Benefit 函数。因为：

- 客户端 $i$ 加入联盟 $\mathcal{C}$ 后，获得的不再是统一的 $g_{\mathcal{C}}$
- 而是个性化的 $g_i^{agg} = \sum_{j \in \mathcal{C}} w_{ij} \cdot g_j$
- Benefit 应该反映这种个性化聚合的效果

---

## 5. 有向 Benefit 公式推导（改进版）

### 5.1 重新定义 Benefit

对于有向协作机制，定义客户端 $i$ 在联盟 $\mathcal{C}$ 中的收益为：

$$
\phi_i^{dir}(\mathcal{C}) = \frac{\langle g_i^{agg}, g_i \rangle}{\|g_i^{agg}\|}
$$

其中个性化聚合梯度为：

$$
g_i^{agg} = \sum_{j \in \mathcal{C}} w_{ij} \cdot g_j
$$

权重满足 $\sum_{j} w_{ij} = 1$，且 $w_{ii}$ 可以取特殊值（自身权重）。

### 5.2 展开推导

令 $\mathcal{C}_{-i} = \mathcal{C} \setminus \{i\}$，将聚合梯度分解：

$$
g_i^{agg} = w_{ii} \cdot g_i + \sum_{j \in \mathcal{C}_{-i}} w_{ij} \cdot g_j
$$

#### 分子计算：

$$
\langle g_i^{agg}, g_i \rangle = w_{ii} \|g_i\|^2 + \sum_{j \in \mathcal{C}_{-i}} w_{ij} \langle g_j, g_i \rangle
$$

$$
= w_{ii} \|g_i\|^2 + \sum_{j \in \mathcal{C}_{-i}} w_{ij} \|g_j\| \|g_i\| \cos(g_j, g_i)
$$

#### 分母计算：

$$
\|g_i^{agg}\|^2 = \langle g_i^{agg}, g_i^{agg} \rangle
$$

$$
= w_{ii}^2 \|g_i\|^2 + 2 w_{ii} \sum_{j \in \mathcal{C}_{-i}} w_{ij} \langle g_i, g_j \rangle + \sum_{j,k \in \mathcal{C}_{-i}} w_{ij} w_{ik} \langle g_j, g_k \rangle
$$

### 5.3 简化形式

定义：
- $\ell_j = \|g_j\|$：客户端 $j$ 的梯度范数
- $s_{jk} = \cos(g_j, g_k)$：梯度余弦相似度（对称）
- $D_{ij} = D[i][j]$：有向协作分数（非对称）

则有向 Benefit 为：

$$
\phi_i^{dir}(\mathcal{C}) = \frac{w_{ii} \ell_i + \sum_{j \in \mathcal{C}_{-i}} w_{ij} \ell_j s_{ji}}{\sqrt{w_{ii}^2 \ell_i^2 + 2 w_{ii} \sum_{j \in \mathcal{C}_{-i}} w_{ij} \ell_i \ell_j s_{ij} + \sum_{j,k \in \mathcal{C}_{-i}} w_{ij} w_{ik} \ell_j \ell_k s_{jk}}}
$$

### 5.4 与原公式的对比

当 $w_{ij} = \frac{\ell_j}{\sum_{k \in \mathcal{C}} \ell_k}$（原始按梯度范数加权）且 $w_{ii} = 0$ 时，有向公式退化为原公式：

$$
\phi_i^{dir}(\mathcal{C}) \rightarrow \phi_i(\mathcal{C}) = \frac{\sum_{j \in \mathcal{C}_{-i}} \ell_j s_{ji}}{\sqrt{\sum_{j,k \in \mathcal{C}_{-i}} \ell_j \ell_k s_{jk}}}
$$

---

## 6. Appendix A 公式的有向版本

### 6.1 原始 Appendix A 推导

原论文 Appendix A 给出了展开形式：

$$
\phi_i(\mathcal{C}) = \frac{\sum_{j \neq i} \ell_j s_{ij}}{\sqrt{\sum_{j \neq i} \ell_j^2 + \sum_{j \neq i} \sum_{k \neq i, k \neq j} \ell_j \ell_k s_{jk}}}
$$

### 6.2 有向版本的 Appendix A

对于有向协作机制，设 $w_{ii} = w_{self}$（自身权重参数），其他权重 $w_{ij}$ 由有向分数决定。

**定理（有向 Benefit 展开式）**：

$$
\phi_i^{dir}(\mathcal{C}) = \frac{w_{self} \ell_i + \sum_{j \neq i} w_{ij} \ell_j s_{ji}}{N_i}
$$

其中归一化因子：

$$
N_i = \sqrt{w_{self}^2 \ell_i^2 + 2 w_{self} \ell_i \sum_{j \neq i} w_{ij} \ell_j s_{ij} + \sum_{j \neq i} \sum_{k \neq i} w_{ij} w_{ik} \ell_j \ell_k s_{jk}}
$$

### 6.3 有向权重计算

权重 $w_{ij}$ 基于有向协作分数计算：

$$
w_{ij} = (1 - w_{self}) \cdot \frac{\exp(D_{ij} / \tau) \cdot \ell_j}{\sum_{k \neq i} \exp(D_{ik} / \tau) \cdot \ell_k}
$$

注意这里 $D_{ij} \neq D_{ji}$，体现了有向性。

---

## 7. 联盟博弈性质分析

### 7.1 原始博弈性质

原论文证明了 coalitional affinity game 满足：
- **非负性**：$\phi_i(\mathcal{C}) \geq 0$（当梯度对齐时）
- **单调性**：添加相似客户端不会降低收益
- **超可加性**：合适的联盟合并可以增加总收益

### 7.2 有向博弈性质变化

引入有向协作后：

| 性质 | 原始博弈 | 有向博弈 | 分析 |
|------|---------|---------|------|
| 对称性 | $\phi_i$ 对 $j$ 和 $j$ 对 $i$ 关系对称 | 不再对称 | $D_{ij} \neq D_{ji}$ 导致 |
| 非负性 | 保持 | 可能变化 | 当 $D_{ij} < 0$ 时可能为负 |
| 单调性 | 保持 | 依赖于 $D$ 矩阵 | 添加低 $D$ 值客户端可能降低收益 |
| 稳定性 | 存在稳定划分 | 需要重新证明 | 非对称博弈的稳定性更复杂 |

### 7.3 稳定性条件

对于有向博弈，稳定划分的条件变为：

对于划分 $\Pi = \{\mathcal{C}_1, ..., \mathcal{C}_K\}$，不存在联盟 $\mathcal{S}$ 使得：
$$
\forall i \in \mathcal{S}: \phi_i^{dir}(\mathcal{S}) > \phi_i^{dir}(\mathcal{C}(i))
$$

由于 $\phi_i^{dir}$ 对不同客户端是非对称的，稳定划分可能更难达到或不存在。

---

## 8. 实现建议

### 8.1 方案一：保持分离（当前实现）

```
联盟形成：使用原始对称 Benefit 公式
聚合阶段：使用有向协作权重
```

**优点**：
- 保持联盟博弈的理论性质
- 实现简单，稳定性有保证
- 有向协作在聚合阶段发挥作用

**缺点**：
- 联盟形成未考虑有向性
- 可能形成对某些客户端不利的联盟

### 8.2 方案二：统一有向（需要修改）

```
联盟形成：使用有向 Benefit 公式 φ_i^{dir}
聚合阶段：使用有向协作权重
```

**修改代码**：

```python
def _compute_coalition_reward_directed(self, union: tuple, c_id: int) -> float:
    """Compute directed reward for client in coalition."""
    others = list(set(union) - {c_id})
    
    if not others:
        return 0.0
    
    # Get directed collaboration scores
    D = self.directed_matrix if hasattr(self, 'directed_matrix') else None
    
    # Self weight
    w_self = self.config.directed_self_weight
    tau = self.config.directed_temperature
    
    l2_i = self.clients[c_id].l2_norm or 1e-8
    
    # Compute directed weights w_ij
    raw_scores = []
    l2_others = []
    for j in others:
        d_ij = D[c_id][j] if D is not None else self.similarity_matrix[c_id, j]
        raw_scores.append(d_ij / tau)
        l2_others.append(self.clients[j].l2_norm or 1e-8)
    
    # Softmax weighting
    raw_scores = np.array(raw_scores) - np.max(raw_scores)
    exp_scores = np.exp(raw_scores)
    l2_others = np.array(l2_others)
    weights = (1 - w_self) * (exp_scores * l2_others) / (exp_scores * l2_others).sum()
    
    # Numerator: w_self * l2_i + sum(w_ij * l2_j * s_ji)
    numerator = w_self * l2_i
    for idx, j in enumerate(others):
        s_ji = self.similarity_matrix[j, c_id]  # Note: j->i similarity
        numerator += weights[idx] * l2_others[idx] * s_ji
    
    # Denominator (simplified): ||g_i^{agg}||
    # ... (complex computation)
    
    return numerator / denominator
```

**优点**：
- 联盟形成充分考虑有向性
- 理论上更合理

**缺点**：
- 实现复杂
- 稳定性需要重新验证
- 计算开销更大

### 8.3 推荐方案

**推荐使用方案一**（当前实现），理由如下：

1. **实验验证有效**：当前实现已展示准确率提升（+0.69% 最终准确率，+5.67% 平均任务准确率）

2. **理论简洁**：联盟博弈性质保持不变，有向协作作为聚合阶段的增强

3. **计算高效**：不需要重复计算有向 Benefit

4. **渐进改进**：如果需要，可以在未来版本中引入完全有向的联盟博弈

---

## 9. 总结

### 9.1 公式是否需要重新推导？

| 组件 | 是否需要修改 | 原因 |
|------|------------|------|
| Benefit calculation | **可选** | 当前分离设计有效；完全有向需要重新推导 |
| Appendix A 展开式 | **可选** | 已给出有向版本推导供参考 |
| 联盟稳定性证明 | **需要** | 非对称博弈的稳定性分析不同 |
| 聚合公式 | **已修改** | 有向聚合已实现 |

### 9.2 核心结论

1. **当前实现合理**：两阶段分离设计（对称联盟形成 + 有向聚合）是有效的工程折中

2. **有向 Benefit 公式**：已给出完整推导，可在需要时使用

3. **理论扩展**：完全有向的联盟博弈是未来的研究方向，需要证明新的稳定性条件

### 9.3 建议

- 短期：保持当前实现，通过实验验证有效性
- 中期：实现有向 Benefit 计算，进行消融实验对比
- 长期：研究非对称联盟博弈的理论性质
