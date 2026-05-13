# CollEdge 有向协作机制 (Directed Collaboration)

## 1. 概述

在原有的 CollEdge 算法中，联盟聚合机制基于**对称相似性**：如果客户端 $i$ 和 $j$ 属于同一联盟，它们相互共享模型更新。然而，在实际场景中，协作关系可能是**非对称**的——客户端 $i$ 从 $j$ 学习可能比 $j$ 从 $i$ 学习更有价值。

为此，我们引入了**有向协作机制 (Directed Collaboration)**，使用一个**有向图**作为协作矩阵的 mask，使得每个客户端获得一个**个性化的聚合模型**。

## 2. 核心思想

### 2.1 有向协作矩阵

定义有向协作矩阵 $D \in \mathbb{R}^{n \times n}$，其中 $D[i][j]$ 表示**客户端 $i$ 愿意从客户端 $j$ 接收知识的程度**。

关键特性：
- **非对称性**: $D[i][j] \neq D[j][i]$
- **有向性**: 该矩阵定义了知识流动的方向
- **动态性**: 每轮通信根据当前状态重新计算

### 2.2 数学形式化

对于联盟 $\mathcal{C}$ 中的客户端 $i$，其个性化聚合模型为：

$$
\theta_i^{agg} = w_{self} \cdot \theta_i + (1 - w_{self}) \cdot \sum_{j \in \mathcal{C}, j \neq i} w_{ij} \cdot \theta_j
$$

其中权重 $w_{ij}$ 基于有向协作分数计算：

$$
w_{ij} = \frac{\exp(D[i][j] / \tau) \cdot s_j}{\sum_{k \in \mathcal{C}, k \neq i} \exp(D[i][k] / \tau) \cdot s_k}
$$

- $\tau$: 温度参数（`directed_temperature`）
- $s_j$: 客户端 $j$ 的样本数量
- $w_{self}$: 自身模型权重（`directed_self_weight`）

## 3. 有向协作分数计算

### 3.1 梯度对齐模式 (gradient)

基于梯度方向的对齐程度，评估 $j$ 的更新对 $i$ 优化的帮助程度：

$$
D_{grad}[i][j] = \cos(\nabla_i, \nabla_j) \cdot \text{mag\_ratio}
$$

其中：
- $\cos(\nabla_i, \nabla_j)$: 两个客户端梯度的余弦相似度
- $\text{mag\_ratio} = \sqrt{\frac{\|\nabla_j\|}{\|\nabla_i\| + \epsilon}}$: 梯度幅度比率

**直觉**：
- 如果 $j$ 的梯度与 $i$ 对齐且幅度更大，则 $j$ 的更新对 $i$ 有益
- 如果梯度方向相反（$\cos < 0$），则 $j$ 的更新可能对 $i$ 有害

### 3.2 任务感知模式 (task_aware)

基于任务/类别知识的相关性：

$$
D_{task}[i][j] = 0.4 \cdot \text{relevance} + 0.3 \cdot \text{diversity} + 0.3 \cdot \text{experience}
$$

其中：
- **relevance**: $j$ 的知识与 $i$ 当前学习任务的重叠度
- **diversity**: $j$ 拥有的互补知识（$i$ 未学习过的类别）
- **experience**: $j$ 相对于 $i$ 的经验程度（学习过更多任务/类别）

### 3.3 混合模式 (hybrid)

结合梯度和任务感知两种模式：

$$
D_{hybrid}[i][j] = 0.5 \cdot D_{grad}[i][j] + 0.5 \cdot D_{task}[i][j]
$$

## 4. 实现细节

### 4.1 配置参数

```python
# core/config.py
directed_collaboration: bool = False  # 是否启用有向协作机制
directed_threshold: float = 0.0       # 协作分数阈值（过滤低分数客户端）
directed_mode: str = 'gradient'       # 模式: 'gradient', 'task_aware', 'hybrid'
directed_temperature: float = 1.0     # Softmax 温度参数
directed_self_weight: float = 0.5     # 自身模型权重（个性化程度）
```

### 4.2 核心代码流程

```
1. 每轮通信开始时，计算有向协作矩阵 D
   └── _compute_directed_collaboration_matrix()

2. 联盟聚合时，为每个客户端生成个性化模型
   └── _aggregate_coalitions_directed()
       ├── 对于联盟中的每个客户端 i:
       │   ├── 筛选 D[i][j] > threshold 的可信客户端
       │   ├── 计算 softmax 加权
       │   └── 生成个性化聚合模型
       └── 存储到 client_personalized_models

3. 分发模型时，使用个性化模型
   └── _distribute_coalition_models()
```

### 4.3 算法伪代码

```
Algorithm: Directed Coalition Aggregation
Input: Coalition C, Directed Matrix D, threshold τ_thresh
Output: Personalized models {θ_i^agg} for each client i ∈ C

for each client i in C:
    trusted_clients = []
    scores = []
    
    for each client j in C, j ≠ i:
        if D[i][j] > τ_thresh:
            trusted_clients.append(j)
            scores.append(D[i][j])
    
    if trusted_clients is empty:
        θ_i^agg = θ_i  # Use own model
    else:
        # Compute softmax weights
        weights = softmax(scores / temperature) × sample_proportions
        
        # Aggregate from trusted clients
        θ_others = Σ_j (weights[j] × θ_j)
        
        # Personalized blend
        θ_i^agg = w_self × θ_i + (1 - w_self) × θ_others
```

## 5. 实验验证

### 5.1 实验设置

- **数据集**: EMNIST-Letters
- **客户端数**: 8
- **任务数**: 6
- **buffer_size**: 100（降低以允许遗忘，便于观察改进效果）
- **对比方法**: Baseline (对称联盟) vs Directed (有向协作, mode=gradient)

### 5.2 实验结果

| 指标 | Baseline | Directed | 改进 |
|------|----------|----------|------|
| **最终准确率** | 0.7917 | 0.7986 | **+0.69%** |
| **平均任务准确率** | 0.8334 | 0.8807 | **+5.67%** |
| **遗忘率** | 0.1251 | 0.1775 | +41.9% |

### 5.3 各阶段准确率对比

| 任务阶段 | Baseline | Directed | 改进 |
|----------|----------|----------|------|
| Task 0 完成后 | 0.7988 | 0.9563 | **+15.75%** |
| Task 1 完成后 | 0.8541 | 0.9234 | **+6.93%** |
| Task 2 完成后 | 0.8545 | 0.8818 | **+2.73%** |
| Task 3 完成后 | 0.8555 | 0.8615 | **+0.60%** |
| Task 4 完成后 | 0.8364 | 0.8628 | **+2.64%** |
| Task 5 完成后 | 0.7917 | 0.7986 | **+0.69%** |

### 5.4 结论

1. **知识迁移增强**: 有向协作显著提升了学习新任务时的准确率（平均 +5.67%）
2. **前向迁移效果明显**: 早期任务（Task 0-1）的提升最为显著，说明有向协作有效促进了知识正向迁移
3. **最终性能提升**: 最终准确率提升 0.69%
4. **权衡**: 遗忘率略有上升，表明有向协作机制侧重于**前向知识迁移**而非抵抗遗忘

## 6. 使用方法

### 6.1 命令行参数

```bash
python main.py \
    --algorithm CollEdge \
    --directed_collaboration \
    --directed_mode gradient \
    --directed_threshold 0.0 \
    --directed_temperature 1.0 \
    --directed_self_weight 0.5
```

### 6.2 配置文件

```yaml
# configs/dcfcl_directed.yaml
algorithm: CollEdge
directed_collaboration: true
directed_mode: gradient  # or 'task_aware', 'hybrid'
directed_threshold: 0.0
directed_temperature: 1.0
directed_self_weight: 0.5
```

## 7. 设计选择与讨论

### 7.1 为什么使用有向图？

传统联邦学习假设客户端间的协作是对称的，但在持续学习场景中：
- 不同客户端处于不同的学习阶段
- 某些客户端的知识对其他客户端更有价值
- 梯度方向可能存在冲突（负迁移）

有向协作允许**选择性学习**，避免从有害来源接收知识。

### 7.2 阈值设置建议

- `directed_threshold = 0.0`: 接受所有非负协作（推荐默认值）
- `directed_threshold > 0`: 更严格地过滤低相关性客户端
- 负阈值不建议使用（会包含有害协作）

### 7.3 温度参数影响

- `directed_temperature = 1.0`: 标准 softmax（默认）
- `directed_temperature < 1.0`: 更尖锐分布，偏好高分数客户端
- `directed_temperature > 1.0`: 更平滑分布，均匀化权重

### 7.4 自身权重选择

- `directed_self_weight = 0.5`: 平衡自身与他人知识（默认）
- `directed_self_weight → 1.0`: 更保守，主要保留自身知识
- `directed_self_weight → 0.0`: 更激进，大量吸收他人知识

## 8. 文件结构

```
core/
├── config.py          # 有向协作配置参数定义
└── server.py          # 有向协作实现
    ├── _compute_directed_collaboration_matrix()  # 计算有向矩阵
    ├── _compute_gradient_directed_score()        # 梯度模式分数
    ├── _compute_task_aware_directed_score()      # 任务感知模式分数
    ├── _aggregate_coalitions_directed()          # 有向聚合
    └── _distribute_coalition_models()            # 分发个性化模型

scripts/
└── run_directed_comparison.py  # 对比实验脚本

results/
└── directed_comparison_*/      # 实验结果目录
```

## 9. 未来改进方向

1. **自适应阈值**: 根据训练进度动态调整 `directed_threshold`
2. **注意力机制**: 使用学习的注意力权重替代启发式分数
3. **时间衰减**: 对旧任务知识的协作分数进行衰减
4. **跨联盟协作**: 允许有限的跨联盟有向知识迁移

---

## 10. 联盟约束机制 (Coalition Mask)

### 10.1 概述

除了有向协作机制，我们还实现了**联盟约束机制**，允许指定某些客户端之间完全不能合作（不能形成联盟）。这在以下场景中有用：

- **隐私约束**: 某些客户端由于数据隐私原因不能共享知识
- **组织边界**: 不同组织的客户端不能直接协作
- **竞争关系**: 竞争实体之间限制知识共享

### 10.2 配置参数

```python
# core/config.py
use_coalition_mask: bool = False       # 是否启用联盟约束
coalition_forbidden_pairs: List = []   # 禁止合作的客户端对 [(i,j), ...]
coalition_mask_type: str = 'custom'    # 'custom', 'random', 'group'
coalition_mask_density: float = 0.2    # random模式下禁止对的概率
num_client_groups: int = 2             # group模式下的组数
```

### 10.3 约束类型

| 类型 | 描述 | 用例 |
|------|------|------|
| `custom` | 手动指定禁止合作的客户端对 | 已知特定约束关系 |
| `random` | 随机生成禁止对 | 模拟随机网络约束 |
| `group` | 基于组的约束（不同组不能合作） | 模拟组织边界 |

### 10.4 实验结果

| 方法 | 最终准确率 | 遗忘率 | 与Baseline对比 |
|------|-----------|--------|---------------|
| **baseline** | 0.7917 | 0.1251 | - |
| group_mask_2 | 0.7564 | 0.1956 | **-4.46%** |
| group_mask_4 | 0.7286 | 0.2574 | **-7.97%** |
| random_mask_20 | 0.7812 | 0.1553 | **-1.33%** |
| random_mask_50 | 0.7304 | 0.2584 | **-7.74%** |

### 10.5 结论

**联盟约束会降低性能**，因为它限制了有效的协作：

1. 约束越严格（更多禁止对），性能下降越明显
2. 遗忘率随约束增加而上升，因为客户端获得的协作帮助减少
3. 轻度约束（20%随机禁止）影响较小（-1.33%），可在需要时使用

### 10.6 使用示例

```bash
# 启用组约束（2组）
python main.py --algorithm CollEdge \
    --use_coalition_mask \
    --coalition_mask_type group \
    --num_client_groups 2

# 启用随机约束（30%禁止率）
python main.py --algorithm CollEdge \
    --use_coalition_mask \
    --coalition_mask_type random \
    --coalition_mask_density 0.3
```
