# CollEdge：核心贡献详解

## 概述

CollEdge 在原始 DCFCL 方法的三路损失（CE + KD + Proto-Aug）基础上，引入了三项核心改进：

1. **回放缓冲区机制（DER++）**：基于"暗经验"的持续学习抗遗忘模块
2. **有向协作机制 + 联盟掩码**：非对称知识流动与协作约束控制
3. **"智能涌现"现象**：客户端习得从未在本地出现过的类别知识

---

## 一、回放缓冲区机制（DER++）

### 1.1 背景：联邦持续学习中的灾难性遗忘

在联邦持续学习（Federated Continual Learning）场景中，客户端按任务序列逐步学习新类别。由于神经网络在训练新任务时容易覆盖旧知识，产生**灾难性遗忘**（Catastrophic Forgetting）现象。

DCFCL 原有的知识蒸馏（KD）和原型增强（Proto-Aug）损失已能部分缓解遗忘，但在任务间差异较大时效果有限。CollEdge 在此基础上引入 **DER++**（Dark Experience Replay++），通过维护历史样本的"暗经验"（过去模型输出的 logits）来约束当前模型行为，显著增强抗遗忘能力。

> 参考论文：Buzzega et al., *"Dark Experience for General Continual Learning: A Strong, Simple Baseline"*, NeurIPS 2020

### 1.2 Replay Buffer 实现

**代码位置**：`core/replay_buffer.py`

每个客户端维护独立的 `ReplayBuffer` 实例，存储历史训练样本的三元组：

```
(examples, labels, logits)
  ↑样本输入  ↑真实标签  ↑存入时刻的模型输出 logits（暗经验）
```

**核心设计：蓄水池采样（Reservoir Sampling）**

蓄水池采样保证缓冲区中的样本均匀代表历史数据流，无需预先知道总样本量：

```python
def _reservoir_index(self) -> int:
    """Return index to place new sample, or -1 if rejected."""
    if self.num_seen < self.buffer_size:
        return self.num_seen        # 缓冲区未满，直接插入
    rand = np.random.randint(0, self.num_seen + 1)
    return rand if rand < self.buffer_size else -1  # 以 B/N 的概率接受
```

**关键特性**：
- **跨轮次、跨任务持久保存**：缓冲区在整个联邦学习过程中累积，而非每轮重置
- **懒初始化存储**：首次调用 `add_data()` 时才根据实际输入维度分配存储空间
- **固定容量**：每个客户端最多保存 `buffer_size`（默认 500）条历史样本
- **隐私安全**：原始数据不离开客户端，仅在本地用于回放训练

### 1.3 DER++ 双路损失

**代码位置**：`FL_model/colledge.py`，`CollEdgeClient.train()`

CollEdge 的完整训练损失为：

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{kd}} \mathcal{L}_{\text{KD}} + \lambda_{\text{proto}} \mathcal{L}_{\text{Proto}} + \alpha \mathcal{L}_{\text{DER}} + \beta \mathcal{L}_{\text{ER}}
$$

其中 DER++ 贡献了后两项：

| 损失项 | 公式 | 作用 |
|--------|------|------|
| $\mathcal{L}_{\text{DER}}$ | $\text{MSE}(f_\theta(x_{\text{buf}}),\ \hat{z}_{\text{buf}})$ | 对回放样本，当前 logits 匹配存储时的历史 logits（维持暗经验一致性） |
| $\mathcal{L}_{\text{ER}}$ | $\text{CE}(f_\theta(x_{\text{buf}}),\ y_{\text{buf}})$ | 对回放样本，使用真实标签的交叉熵（保持分类能力） |

- $x_{\text{buf}}, y_{\text{buf}}$：从缓冲区随机采样的历史样本和标签
- $\hat{z}_{\text{buf}}$：存入缓冲区时保存的 logits（"暗经验"，冻结的历史知识快照）
- $\alpha$（`der_alpha`）、$\beta$（`der_beta`）：两路损失的权重系数

**存入逻辑**：每个 local epoch 训练完毕后，将当前 batch 数据及当前模型 logits 存入缓冲区：

```python
if self.config.use_der:
    with torch.no_grad():
        _, _, store_logits = self.model(x)
    self.replay_buffer.add_data(x.detach(), y.detach(), store_logits.detach())
```

### 1.4 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_der` | `True` | DER++ 模块主开关（CLI: `--use_der` / `--no_use_der`） |
| `buffer_size` | `500` | 每个客户端缓冲区最大样本数 |
| `der_alpha` | `0.5` | DER 损失权重（logits MSE） |
| `der_beta` | `0.5` | ER 损失权重（标签 CE） |

### 1.5 实验效果

以 EMNIST-Letters 数据集（seed=42）为例：

| 方法 | 平均任务准确率 | 遗忘率 |
|------|---------------|--------|
| DCFCL（无 DER++） | 52.62% | 56.82% |
| **CollEdge（含 DER++）** | **84.78%** | **1.54%** |

DER++ 将遗忘率从 **56.82% 降至 1.54%**，同时平均准确率提升 **32.16 个百分点**。

---

## 二、有向协作机制 + 联盟掩码（Coalition Mask）

### 2.1 背景：对称协作的局限性

原始 DCFCL 中，联盟内的模型聚合使用基于样本数量的加权平均——这是一种**对称协作**方式：若 A 和 B 在同一联盟，则 A 从 B 获得的知识量等同于 B 从 A 获得的知识量。

然而现实中，知识转移是**有方向的**：某个客户端可能从某些伙伴处获益更多，而对其他伙伴的知识无需借鉴甚至有害（如任务分布差异大时）。

CollEdge 通过**有向协作矩阵**精确建模这种非对称关系。

### 2.2 有向协作矩阵

**代码位置**：`core/server.py`，`_compute_directed_collaboration_matrix()`

有向协作矩阵 $D \in \mathbb{R}^{n \times n}$ 满足：

- $D[i][j]$ 表示**客户端 $i$ 愿意从客户端 $j$ 接收知识的程度**
- $D[i][j] \neq D[j][i]$（非对称）
- $D[i][i] = 1.0$（完全信任自身）

#### 三种评分模式

**（1）Gradient 模式**（`directed_mode='gradient'`）

基于梯度对齐度衡量 $j$ 对 $i$ 的帮助程度：

$$
D[i][j] = \cos(\nabla_i, \nabla_j) \times \sqrt{\frac{\|\nabla_j\|}{\|\nabla_i\| + \epsilon}}
$$

- $\cos(\nabla_i, \nabla_j)$：梯度方向余弦相似度（负值表示更新方向对立，有害）
- $\sqrt{\|\nabla_j\| / \|\nabla_i\|}$：梯度幅值比，鼓励向更新量更大（信息量更丰富）的客户端学习，比值被裁剪至 $[0.5, 2.0]$

**（2）Task-Aware 模式**（`directed_mode='task_aware'`）

基于任务知识关联度衡量 $j$ 对 $i$ 的价值：

$$
D[i][j] = 0.4 \cdot r_{ij} + 0.3 \cdot d_{ij} + 0.3 \cdot \max(0, e_{ij})
$$

| 分量 | 含义 |
|------|------|
| $r_{ij}$（Relevance） | $j$ 已学类别与 $i$ 当前任务类别的重叠比例 |
| $d_{ij}$（Diversity） | $j$ 拥有 $i$ 尚未学到的类别比例（互补知识） |
| $e_{ij}$（Experience） | $j$ 比 $i$ 学过更多类别的程度，用 $\tanh(\|C_j\|/\|C_i\| - 1)$ 衡量 |

**（3）Hybrid 模式**（`directed_mode='hybrid'`）

线性组合两种模式：

$$
D[i][j] = 0.5 \cdot D_{\text{grad}}[i][j] + 0.5 \cdot D_{\text{task}}[i][j]
$$

### 2.3 个性化模型聚合

**代码位置**：`core/server.py`，`_aggregate_coalitions_directed()`

每个客户端 $i$ 获得专属的个性化聚合模型，而非联盟内所有成员共享一个模型：

$$
\theta_i^{\text{new}} = w_{\text{self}} \cdot \theta_i + (1 - w_{\text{self}}) \cdot \sum_{j \in \mathcal{T}_i} w_{ij} \cdot \theta_j
$$

其中：
- $w_{\text{self}}$（`directed_self_weight`）：自身模型的保留权重
- $\mathcal{T}_i$：置信度超过阈值 $\tau$（`directed_threshold`）的可信联盟成员集合
- $w_{ij}$：基于 $D[i][j]$ 通过 softmax + 样本数联合归一化得到的权重

### 2.4 联盟掩码（Coalition Mask）

**代码位置**：`core/server.py`，`_init_coalition_mask()` / `_create_partition_table()`

联盟掩码是一个布尔矩阵 $M \in \{0,1\}^{n \times n}$，$M[i][j] = 1$ 表示客户端 $i$ 和 $j$ **禁止**加入同一联盟。

在联盟划分表初始化阶段，所有包含被禁对的划分方案将被过滤：

```
全量划分方案 → is_valid_partition() 过滤 → 有效划分方案
```

#### 三种掩码类型

| 类型 | 配置参数 | 说明 |
|------|----------|------|
| `custom` | `coalition_forbidden_pairs` | 手动指定禁止协作的客户端对 |
| `random` | `coalition_mask_density`（默认 0.2） | 以给定密度随机生成禁止对 |
| `group` | `num_client_groups`（默认 2） | 跨组客户端不得合作，同组内自由联盟 |

### 2.5 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `directed_collaboration` | `False` | 启用有向协作机制 |
| `directed_mode` | `'gradient'` | 评分模式：`gradient` / `task_aware` / `hybrid` |
| `directed_self_weight` | `0.5` | 聚合时自身模型的权重 |
| `directed_temperature` | `1.0` | Softmax 温度参数（控制权重集中程度） |
| `directed_threshold` | `0.0` | 置信度阈值，低于此值的客户端将被排除 |
| `use_coalition_mask` | `False` | 启用联盟掩码 |
| `coalition_mask_type` | `'custom'` | 掩码类型：`custom` / `random` / `group` |
| `num_client_groups` | `2` | group 类型掩码的组数 |

---

## 三、"智能涌现"现象（Emergent Knowledge）

### 3.1 现象定义

**"智能涌现"**（Emergence）是指：在联邦持续学习过程中，某客户端**能够正确预测其本地训练数据中从未出现过的类别**。

这些知识不可能来自客户端自身的训练数据，而是通过联邦协作在模型参数层面被隐式传递的——多个客户端各自学习部分类别，通过联盟聚合后，系统整体涌现出超越个体的集体智能。

> 参考论文：*"A collective AI via lifelong learning and sharing at the edge"*

**对比示意**：

```
Local 训练（无涌现）：
  Client A 看过类别 {1, 2, 3} → 只能预测 {1, 2, 3}

CollEdge 联邦训练（有涌现）：  
  Client A 看过类别 {1, 2, 3}
  Client B 看过类别 {4, 5, 6}
  通过联盟聚合 → Client A 也能预测 {4, 5, 6} ✓（涌现）
```

### 3.2 评估指标

**代码位置**：`core/server.py`，`evaluate_emergence()`（第 1270 行）

涌现评估完成后返回以下指标：

| 指标 | 说明 |
|------|------|
| `global_emergence_rate` | 全局未见类别准确率（核心涌现指标） |
| `seen_accuracy` | 已见类别准确率（参照对比） |
| `per_client_emergence` | 每个客户端的未见/已见类别准确率分解 |
| `knowledge_transfer_matrix` | $n \times n$ 知识传递矩阵，$T[i][j]$ 表示客户端 $i$ 通过联邦学习习得的来自 $j$ 独有类别的数量 |
| `emergence_by_class` | 每个类别在各客户端的涌现统计 |
| `emergence_samples` | 具体涌现样本（raw 输入、真实标签、预测标签、置信度） |

### 3.3 评估方法

涌现评估在所有任务训练完成后调用，流程如下：

1. **构建全局类别视角**：汇总所有客户端在每个任务上见过的类别集合 $\{C_t^{(k)}\}$
2. **识别未见类别**：对每个客户端 $k$，找出全局存在但 $k$ 从未本地训练的类别 $U^{(k)} = \mathcal{Y} \setminus C_{\text{local}}^{(k)}$
3. **测试未见类别**：用客户端 $k$ 的模型推理包含 $U^{(k)}$ 类别的测试样本，计算准确率
4. **构建知识传递矩阵**：统计每个客户端对其他客户端独有类别的预测情况

$$
T[i][j] = \left| \left\{ c \in C_{\text{local}}^{(j)} \setminus C_{\text{local}}^{(i)} \mid \text{Client } i \text{ 能正确预测类别 } c \right\} \right|
$$

### 3.4 对照实验设计

**代码位置**：`scripts/run_emergence_evaluation.py`

通过对比 **CollEdge（联邦）** vs **Local（无联邦）** 两种条件，量化联邦协作带来的涌现增益：

- Local 条件下：无联盟聚合，客户端各自独立训练，理论涌现率为 0
- CollEdge 条件下：联盟聚合传递知识，涌现率反映协作的有效性

实验中通过解析训练日志中的 `Global Emergence Rate` 字段进行跨条件比较。

### 3.5 启用方式

在配置文件或命令行中开启涌现评估：

```yaml
# configs/colledge_complete_emnist.yaml
evaluate_emergence: true
```

```bash
python main.py --evaluate_emergence
```

评估结果将自动保存至实验目录的 `emergence_analysis/` 子目录，包含：
- `emergence_metadata.json`：完整的量化指标（JSON 格式）
- `emergence_samples.pkl`：所有涌现样本的原始数据（Pickle 格式）

---

## 四、三项贡献的协同效应

三个模块可通过配置参数独立开关，支持消融实验验证各自贡献：

| 变体 | `use_der` | `directed_collaboration` | `use_coalition_mask` | 说明 |
|------|-----------|--------------------------|----------------------|------|
| DCFCL Baseline | ✗ | ✗ | ✗ | 原始 DCFCL |
| CollEdge Base | ✓ | ✗ | ✗ | 仅 DER++ |
| +Directed | ✓ | ✓ | ✗ | DER++ + 有向协作 |
| +Mask | ✓ | ✗ | ✓ | DER++ + 掩码 |
| **Full** | **✓** | **✓** | **✓** | **完整 CollEdge** |

运行完整消融实验：

```bash
python scripts/run_module_ablation.py --dataset emnist
python scripts/run_module_ablation.py --dataset cifar100 --quick
```

三项贡献相互增强：
- **DER++** 保障各任务知识的本地留存，为联邦协作提供高质量的知识基础
- **有向协作 + 掩码** 精确控制知识流向，避免有害知识传播，放大有益迁移
- **涌现现象** 是两者协同作用的系统级结果，体现联邦持续学习的集体智能

---

## 五、代码导航

| 模块 | 文件 | 核心类/方法 |
|------|------|------------|
| 配置中心 | `core/config.py` | `Config` dataclass |
| 回放缓冲区 | `core/replay_buffer.py` | `ReplayBuffer` |
| DER++ 训练 | `FL_model/colledge.py` | `CollEdgeClient.train()` |
| 有向协作矩阵 | `core/server.py` | `_compute_directed_collaboration_matrix()` |
| Gradient 评分 | `core/server.py` | `_compute_gradient_directed_score()` |
| Task-Aware 评分 | `core/server.py` | `_compute_task_aware_directed_score()` |
| 个性化聚合 | `core/server.py` | `_aggregate_coalitions_directed()` |
| 联盟掩码初始化 | `core/server.py` | `_init_coalition_mask()` |
| 掩码过滤 | `core/server.py` | `_create_partition_table()` |
| 涌现评估 | `core/server.py` | `evaluate_emergence()` |
| 涌现数据保存 | `core/server.py` | `save_emergence_data()` |
| 消融实验脚本 | `scripts/run_module_ablation.py` | — |
| 涌现实验脚本 | `scripts/run_emergence_evaluation.py` | — |
| 完整方法配置 | `configs/colledge_complete_emnist.yaml` | — |
