# CIFAR-100 实验计划

> **状态**: 待执行（EMNIST 实验完成后开展）  
> **预计总时长**: ~84 小时（单 GPU，4+3 runs × ~12 h/run）  
> **前置条件**: 先用 EMNIST 的最优超参作为参考，根据情况决定是否复用或重新调参

---

## 背景与问题诊断

当前 CIFAR-100 的 ablation 结果显示 Full CollEdge (E) **不是最优**：

| 配置 | Avg Task Acc | Final Acc | Forgetting |
|------|:---:|:---:|:---:|
| A. DCFCL Baseline | 12.12% | 9.99% | 25.45% |
| B. DER only | **25.29%** | **38.53%** | 4.38% |
| C. DER+Directed | 26.56% | 23.63% | 44.29% |
| D. DER+Mask | 22.74% | 26.64% | 14.74% |
| **E. Full (DER+Dir+Mask)** | 24.98% | 19.39% | **43.58%** |
| F. Dir only | 16.80% | 8.04% | 62.56% |
| G. Mask only | 10.98% | 10.12% | 25.71% |

**根本原因**：`directed_self_weight=0.5` 让客户端在困难的 CIFAR-100 任务上过度接受来自 peer 的参数，破坏了 DER++ 缓冲区建立的局部稳定性，导致 forgetting 高达 43.58%。

---

## Stage 3：CIFAR-100 超参扫描

**脚本**: `scripts/paper_experiments/06_full_hp_tune_cifar100.sh`（已精简为 4 runs）

### 扫描网格

| Tag | `self_weight` | `threshold` | `temperature` | 预计耗时 |
|-----|:---:|:---:|:---:|:---:|
| sw0.7_thr0.00_temp1.0 | 0.7 | 0.0 | 1.0 | ~12 h |
| sw0.8_thr0.00_temp1.0 | 0.8 | 0.0 | 1.0 | ~12 h |
| sw0.8_thr0.05_temp1.0 | 0.8 | 0.05 | 1.0 | ~12 h |
| sw0.9_thr0.05_temp1.0 | 0.9 | 0.05 | 1.0 | ~12 h |

**目标**: Full CollEdge Avg Task Acc ≥ 26.56%（超过当前最优 C 的 26.56%）

### 执行命令

```bash
cd /home/lzr/Documents/DCFCLv2/CollEdge
conda activate dcfcl
nohup bash scripts/paper_experiments/06_full_hp_tune_cifar100.sh \
      > cifar_hp_tune.log 2>&1 &
echo "PID=$!"
tail -f cifar_hp_tune.log
```

### 查看最优结果

```bash
python scripts/paper_experiments/collect_full_results.py
# 或仅看 CIFAR-100 HP tune 部分
python scripts/paper_experiments/pick_best_hp.py CIFAR100
```

---

## Stage 4：CIFAR-100 有向聚合策略对比

**脚本**: `scripts/paper_experiments/08_directed_modes_cifar100.sh`

**前置**: 必须先完成 Stage 3，用 `pick_best_hp.py` 找到最优超参后，通过环境变量传入。

### 三种策略

| 策略 | `directed_mode` | 描述 |
|------|:---:|---|
| Gradient-based | `gradient` | 余弦相似度对齐参数更新方向 |
| Task-aware | `task_aware` | 基于任务相关性打分 |
| Hybrid | `hybrid` | 上述两者的 0.5/0.5 加权平均 |

### 执行命令

```bash
# 1) 从 Stage 3 结果中选出最优超参
eval $(python scripts/paper_experiments/pick_best_hp.py CIFAR100)
echo "Best HP: SW=$COLLEDGE_SW  THR=$COLLEDGE_THR  TEMP=$COLLEDGE_TEMP"

# 2) 用最优超参运行三种策略
nohup bash scripts/paper_experiments/08_directed_modes_cifar100.sh \
      > cifar_modes.log 2>&1 &
tail -f cifar_modes.log
```

---

## 实验完成后：汇总与论文更新

所有实验（EMNIST + CIFAR）完成后，运行一键分析脚本：

```bash
cd /home/lzr/Documents/DCFCLv2/CollEdge
python scripts/paper_experiments/analyze_and_update.py
```

该脚本将：
1. 找到每个数据集的最优 HP 配置
2. 更新 `paper/figure_data/fig3_aggregation_effect.json`（加入三种策略曲线）
3. 更新 `paper/figure_data/table4_ablation.json`（Full CollEdge 行）
4. 新建 `paper/figure_data/table_directed_modes.json`（三策略对比表）
5. 重新生成 `paper/figures/fig3_aggregation_effect.pdf`
6. 更新 `Zirui-Li-IEEE-JSTSP-Special-Issue/main.tex` 中的相关表格和图说

---

## 时间线参考

```
EMNIST 完成（约今天）
  └─→ 启动 CIFAR Stage 3（HP sweep, ~48 h）
         └─→ 启动 CIFAR Stage 4（mode compare, ~36 h）
                └─→ 运行 analyze_and_update.py
                       └─→ pdflatex main.tex
```

---

## 优先度说明

| 项目 | 影响 | 优先级 |
|---|---|:---:|
| CIFAR Stage 3 (HP tune) | Table 4 ablation 关键数据 | **高** |
| CIFAR Stage 4 (mode cmp) | Table directed_modes + Fig 3 | 中 |
| 论文更新 | 需等所有实验完成 | 等待 |
