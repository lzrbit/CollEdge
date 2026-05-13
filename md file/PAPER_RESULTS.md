# 论文实验结果汇总 (PAPER_RESULTS)

> **自动生成于** 2026-05-12 19:14:15  
> **结果根目录** `results/paper_experiments/`  
> **生成脚本** `scripts/paper_experiments/collect_results.py`  
> **实验脚本** `scripts/paper_experiments/`

---

## 实验配置

| 项目 | EMNIST-Letters | CIFAR-100 |
|------|:--------------:|:---------:|
| 客户端数 | 8 | 10 |
| 任务数 | 6 | 10 |
| 通信轮次 | 60 | 100 |
| 本地 epoch | 100 | 50 |
| 模型 | CNN | ResNet-18 |
| 学习率 | 1e-4 | 1e-3 |
| 随机种子 | 42 | 42 |
| 数据划分 | EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571 | CIFAR100_split_cn10_tn10_cet10_cs1_s2571 |

**指标说明**：
- **Avg Task Acc ↑**：每个任务阶段训练完成后整体加权准确率的均值（对应论文 Average Accuracy 指标）
- **Avg Forget ↓**：各任务阶段相对最佳历史准确率的平均遗忘率

---

## 一、主要对比实验

所有 10 个算法在两个数据集上的完整对比。
DCFCL 和 CollEdge (Full) 来自消融实验目录（A / E 变体）。

| 算法 | EMNIST Avg Acc ↑ | EMNIST Avg Forget ↓ | CIFAR100 Avg Acc ↑ | CIFAR100 Avg Forget ↓ | Status |
|------|:----------------:|:-------------------:|:------------------:|:---------------------:|:------:|
| Local | 39.94% | 97.87% | 19.73% | 67.24% | ✓/✓ |
| FedAvg | 50.35% | 57.49% | 10.21% | 12.10% | ✓/✓ |
| FedProx | 53.84% | 54.12% | 10.79% | 12.98% | ✓/✓ |
| FedLwF | 50.33% | 57.33% | 10.56% | 12.46% | ✓/✓ |
| SCAFFOLD | 42.48% | 60.74% | 8.26% | 7.66% | ✓/✓ |
| PerAvg | 50.52% | 66.41% | 21.50% | 11.46% | ✓/✓ |
| pFedMe | 9.01% | 5.79% | 2.05% | 3.37% | ✓/✓ |
| ClusterFL | 45.70% | 65.46% | 9.47% | 15.49% | ✓/✓ |
| ── | ── | ── | ── | ── | ── |
| DCFCL (original) | 46.75% | 65.32% | 12.12% | 24.01% | ✓/✓ |
| **CollEdge (ours)** | **90.39%** | **5.48%** | **24.98%** | **41.28%** | ✓/✓ |

---

## 二、消融实验（CollEdge 模块贡献）

验证 DER++、有向协作（Directed）、联盟掩码（Mask）三个模块的独立贡献。

| 变体 | 模块配置 | EMNIST Avg Acc ↑ | EMNIST Avg Forget ↓ | CIFAR100 Avg Acc ↑ | CIFAR100 Avg Forget ↓ | Status |
|------|----------|:----------------:|:-------------------:|:------------------:|:---------------------:|:------:|
| DCFCL (baseline) | `DER✗  Dir✗  Mask✗` | 46.75% | 65.32% | 12.12% | 24.01% | ✓/✓ |
| CollEdge w/ DER++ only | `DER✓  Dir✗  Mask✗` | 85.52% | 2.08% | 25.29% | 8.39% | ✓/✓ |
| CollEdge w/ DER++ + Directed | `DER✓  Dir✓  Mask✗` | 91.53% | 3.23% | 26.56% | 44.32% | ✓/✓ |
| CollEdge w/ DER++ + Mask | `DER✓  Dir✗  Mask✓` | 86.02% | 3.84% | 22.74% | 14.55% | ✓/✓ |
| **CollEdge Full** (ours) | `DER✓  Dir✓  Mask✓` | 90.39% | 5.48% | 24.98% | 41.28% | ✓/✓ |
| CollEdge w/ Directed only | `DER✗  Dir✓  Mask✗` | 50.82% | 77.09% | 16.80% | 55.31% | ✓/✓ |
| CollEdge w/ Mask only | `DER✗  Dir✗  Mask✓` | 42.24% | 72.29% | 10.98% | 20.28% | ✓/✓ |

---

## 三、实验进度

**30 / 30** 个实验已完成。

### Baselines

| 算法 | EMNIST | CIFAR100 |
|------|:------:|:--------:|
| Local | ✓ | ✓ |
| FedAvg | ✓ | ✓ |
| FedProx | ✓ | ✓ |
| FedLwF | ✓ | ✓ |
| SCAFFOLD | ✓ | ✓ |
| PerAvg | ✓ | ✓ |
| pFedMe | ✓ | ✓ |
| ClusterFL | ✓ | ✓ |

### 消融变体

| 变体 | EMNIST | CIFAR100 |
|------|:------:|:--------:|
| DCFCL (baseline) | ✓ | ✓ |
| CollEdge w/ DER++ only | ✓ | ✓ |
| CollEdge w/ DER++ + Directed | ✓ | ✓ |
| CollEdge w/ DER++ + Mask | ✓ | ✓ |
| **CollEdge Full** (ours) | ✓ | ✓ |
| CollEdge w/ Directed only | ✓ | ✓ |
| CollEdge w/ Mask only | ✓ | ✓ |

---

## 四、实验结果目录索引

```
results/paper_experiments/
├── baselines/
│   ├── EMNIST-Letters/    # 8 baselines
│   └── CIFAR100/          # 8 baselines
└── ablation/
    ├── EMNIST-Letters/    # 7 CollEdge 变体
    │   ├── A_DCFCL_baseline/
    │   ├── B_DER_only/
    │   ├── C_DER_Directed/
    │   ├── D_DER_Mask/
    │   ├── E_DER_Dir_Mask_Full/
    │   ├── F_noDER_Directed/
    │   └── G_noDER_Mask/
    └── CIFAR100/          # 同上
```

> 每个实验子目录包含：
> - `training.log` — 完整训练日志（含最终 Avg Task Accuracy、Avg Forgetting 统计）
> - `results.json` — 结构化指标数据（all_accuracies、per_task_acc 等）
