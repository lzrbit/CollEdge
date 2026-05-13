#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_results.py
==================
扫描 results/paper_experiments/ 下全部实验结果，生成 PAPER_RESULTS.md。

用法（在 repo 根目录执行）：
    python scripts/paper_experiments/collect_results.py

输出：
    PAPER_RESULTS.md  （自动覆盖写入）
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = REPO / "results" / "paper_experiments"
OUTPUT_FILE  = REPO / "PAPER_RESULTS.md"

# ---------------------------------------------------------------------------
# 显示顺序 & 名称映射
# ---------------------------------------------------------------------------
BASELINE_ORDER = [
    "Local", "FedAvg", "FedProx", "FedLwF",
    "SCAFFOLD", "PerAvg", "pFedMe", "ClusterFL",
]

ABLATION_ORDER = [
    "A_DCFCL_baseline",
    "B_DER_only",
    "C_DER_Directed",
    "D_DER_Mask",
    "E_DER_Dir_Mask_Full",
    "F_noDER_Directed",
    "G_noDER_Mask",
]

ABLATION_DISPLAY = {
    "A_DCFCL_baseline":   "DCFCL (baseline)",
    "B_DER_only":         "CollEdge w/ DER++ only",
    "C_DER_Directed":     "CollEdge w/ DER++ + Directed",
    "D_DER_Mask":         "CollEdge w/ DER++ + Mask",
    "E_DER_Dir_Mask_Full":"**CollEdge Full** (ours)",
    "F_noDER_Directed":   "CollEdge w/ Directed only",
    "G_noDER_Mask":       "CollEdge w/ Mask only",
}

ABLATION_FLAGS = {
    "A_DCFCL_baseline":   "DER✗  Dir✗  Mask✗",
    "B_DER_only":         "DER✓  Dir✗  Mask✗",
    "C_DER_Directed":     "DER✓  Dir✓  Mask✗",
    "D_DER_Mask":         "DER✓  Dir✗  Mask✓",
    "E_DER_Dir_Mask_Full":"DER✓  Dir✓  Mask✓",
    "F_noDER_Directed":   "DER✗  Dir✓  Mask✗",
    "G_noDER_Mask":       "DER✗  Dir✗  Mask✓",
}

DATASETS = ["EMNIST-Letters", "CIFAR100"]


# ---------------------------------------------------------------------------
# 指标解析
# ---------------------------------------------------------------------------

def parse_log_metrics(log_path: Path) -> dict:
    """从 training.log 解析 Avg Task Accuracy 和 Avg Forgetting。"""
    metrics = {}
    if not log_path.exists():
        return metrics
    try:
        content = log_path.read_text(errors="replace")
        # 取最后一次出现（多 task 训练最终汇总行）
        m = re.findall(r"Avg Task Accuracy.*?:\s*([0-9.]+)", content)
        if m:
            metrics["avg_task_acc"] = float(m[-1])
        m = re.findall(r"Avg Forgetting:\s*([0-9.]+)", content)
        if m:
            metrics["avg_forgetting"] = float(m[-1])
        # Phase-end overall accuracies list（用于验证）
        m = re.findall(r"Phase-end overall accuracies:\s*(\[[^\]]*\])", content)
        if m:
            try:
                metrics["phase_end_accs"] = json.loads(m[-1])
            except Exception:
                pass
    except Exception:
        pass
    return metrics


def parse_results_json(results_path: Path) -> dict:
    """从 results.json 解析基本指标（fallback）。"""
    metrics = {}
    if not results_path.exists():
        return metrics
    try:
        data = json.loads(results_path.read_text())
        metrics["final_accuracy"] = data.get("final_accuracy")
        metrics["forgetting_rate"] = data.get("forgetting_rate")
        all_acc  = data.get("all_accuracies", [])
        all_fgt  = data.get("all_forgetting", [])
        if all_acc:
            metrics["_all_acc"] = all_acc
        if all_fgt:
            metrics["avg_forgetting_json"] = sum(all_fgt) / len(all_fgt)
    except Exception:
        pass
    return metrics


def get_metrics(exp_dir: Path) -> dict | None:
    """合并 log + json 指标，优先使用 log 中精确计算值。"""
    log_m  = parse_log_metrics(exp_dir / "training.log")
    json_m = parse_results_json(exp_dir / "results.json")

    if not log_m and not json_m:
        return None

    avg_task_acc = log_m.get("avg_task_acc")
    avg_forgetting = log_m.get("avg_forgetting")

    # fallback: 直接用 final_accuracy / forgetting_rate
    if avg_task_acc is None:
        avg_task_acc = json_m.get("final_accuracy")
    if avg_forgetting is None:
        avg_forgetting = json_m.get("forgetting_rate") or json_m.get("avg_forgetting_json")

    return {
        "avg_task_acc":  avg_task_acc,
        "avg_forgetting": avg_forgetting,
        "exp_dir": str(exp_dir),
    }


# ---------------------------------------------------------------------------
# 结果查找
# ---------------------------------------------------------------------------

def find_latest_exp(parent: Path) -> Path | None:
    """在 parent 下找最新的实验子目录（含 results.json）。"""
    if not parent.exists():
        return None
    candidates = sorted(
        [d for d in parent.iterdir() if d.is_dir() and (d / "results.json").exists()]
    )
    return candidates[-1] if candidates else None


def collect_baselines(dataset: str) -> dict:
    """读取所有 baseline 算法结果。"""
    base_dir = RESULTS_ROOT / "baselines" / dataset
    results = {}
    for algo in BASELINE_ORDER:
        if not base_dir.exists():
            results[algo] = None
            continue
        # 找以 algo_ 开头的目录（最新一个）
        matching = sorted(
            [d for d in base_dir.iterdir()
             if d.is_dir() and d.name.startswith(algo + "_") and (d / "results.json").exists()]
        )
        if matching:
            results[algo] = get_metrics(matching[-1])
        else:
            results[algo] = None
    return results


def collect_ablation(dataset: str) -> dict:
    """读取所有消融变体结果。"""
    abl_dir = RESULTS_ROOT / "ablation" / dataset
    results = {}
    for variant in ABLATION_ORDER:
        variant_dir = abl_dir / variant
        exp_dir = find_latest_exp(variant_dir)
        results[variant] = get_metrics(exp_dir) if exp_dir else None
    return results


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------

def fmt_acc(v, scale=100, decimals=2):
    if v is None:
        return "—"
    return f"{v * scale:.{decimals}f}%"


def fmt_fgt(v, scale=100, decimals=2):
    if v is None:
        return "—"
    return f"{v * scale:.{decimals}f}%"


def completion_icon(m):
    return "✓" if m is not None and m.get("avg_task_acc") is not None else "✗"


# ---------------------------------------------------------------------------
# Markdown 生成
# ---------------------------------------------------------------------------

def generate_markdown(
    bl_emnist, bl_cifar100,
    abl_emnist, abl_cifar100,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    # ── 标题 ────────────────────────────────────────────────────
    lines += [
        "# 论文实验结果汇总 (PAPER_RESULTS)",
        "",
        f"> **自动生成于** {now}  ",
        f"> **结果根目录** `results/paper_experiments/`  ",
        f"> **生成脚本** `scripts/paper_experiments/collect_results.py`  ",
        f"> **实验脚本** `scripts/paper_experiments/`",
        "",
        "---",
    ]

    # ── 实验配置说明 ────────────────────────────────────────────
    lines += [
        "",
        "## 实验配置",
        "",
        "| 项目 | EMNIST-Letters | CIFAR-100 |",
        "|------|:--------------:|:---------:|",
        "| 客户端数 | 8 | 10 |",
        "| 任务数 | 6 | 10 |",
        "| 通信轮次 | 60 | 100 |",
        "| 本地 epoch | 100 | 50 |",
        "| 模型 | CNN | ResNet-18 |",
        "| 学习率 | 1e-4 | 1e-3 |",
        "| 随机种子 | 42 | 42 |",
        "| 数据划分 | EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571 | CIFAR100_split_cn10_tn10_cet10_cs1_s2571 |",
        "",
        "**指标说明**：",
        "- **Avg Task Acc ↑**：每个任务阶段训练完成后整体加权准确率的均值（对应论文 Average Accuracy 指标）",
        "- **Avg Forget ↓**：各任务阶段相对最佳历史准确率的平均遗忘率",
        "",
        "---",
    ]

    # ── 主对比实验 ───────────────────────────────────────────────
    lines += [
        "",
        "## 一、主要对比实验",
        "",
        "所有 10 个算法在两个数据集上的完整对比。",
        "DCFCL 和 CollEdge (Full) 来自消融实验目录（A / E 变体）。",
        "",
        "| 算法 | EMNIST Avg Acc ↑ | EMNIST Avg Forget ↓ | CIFAR100 Avg Acc ↑ | CIFAR100 Avg Forget ↓ | Status |",
        "|------|:----------------:|:-------------------:|:------------------:|:---------------------:|:------:|",
    ]

    # 8 baselines
    for algo in BASELINE_ORDER:
        e = bl_emnist.get(algo)
        c = bl_cifar100.get(algo)
        icon = f"{completion_icon(e)}/{completion_icon(c)}"
        lines.append(
            f"| {algo} "
            f"| {fmt_acc(e.get('avg_task_acc') if e else None)} "
            f"| {fmt_fgt(e.get('avg_forgetting') if e else None)} "
            f"| {fmt_acc(c.get('avg_task_acc') if c else None)} "
            f"| {fmt_fgt(c.get('avg_forgetting') if c else None)} "
            f"| {icon} |"
        )

    # separator row
    lines.append("| ── | ── | ── | ── | ── | ── |")

    # DCFCL (from ablation A)
    e = abl_emnist.get("A_DCFCL_baseline")
    c = abl_cifar100.get("A_DCFCL_baseline")
    icon = f"{completion_icon(e)}/{completion_icon(c)}"
    lines.append(
        f"| DCFCL (original) "
        f"| {fmt_acc(e.get('avg_task_acc') if e else None)} "
        f"| {fmt_fgt(e.get('avg_forgetting') if e else None)} "
        f"| {fmt_acc(c.get('avg_task_acc') if c else None)} "
        f"| {fmt_fgt(c.get('avg_forgetting') if c else None)} "
        f"| {icon} |"
    )

    # CollEdge Full (from ablation E)
    e = abl_emnist.get("E_DER_Dir_Mask_Full")
    c = abl_cifar100.get("E_DER_Dir_Mask_Full")
    icon = f"{completion_icon(e)}/{completion_icon(c)}"
    lines.append(
        f"| **CollEdge (ours)** "
        f"| **{fmt_acc(e.get('avg_task_acc') if e else None)}** "
        f"| **{fmt_fgt(e.get('avg_forgetting') if e else None)}** "
        f"| **{fmt_acc(c.get('avg_task_acc') if c else None)}** "
        f"| **{fmt_fgt(c.get('avg_forgetting') if c else None)}** "
        f"| {icon} |"
    )

    lines.append("")

    # ── 消融实验 ─────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 二、消融实验（CollEdge 模块贡献）",
        "",
        "验证 DER++、有向协作（Directed）、联盟掩码（Mask）三个模块的独立贡献。",
        "",
        "| 变体 | 模块配置 | EMNIST Avg Acc ↑ | EMNIST Avg Forget ↓ | CIFAR100 Avg Acc ↑ | CIFAR100 Avg Forget ↓ | Status |",
        "|------|----------|:----------------:|:-------------------:|:------------------:|:---------------------:|:------:|",
    ]

    for variant in ABLATION_ORDER:
        display = ABLATION_DISPLAY[variant]
        flags   = ABLATION_FLAGS[variant]
        e = abl_emnist.get(variant)
        c = abl_cifar100.get(variant)
        icon = f"{completion_icon(e)}/{completion_icon(c)}"
        lines.append(
            f"| {display} "
            f"| `{flags}` "
            f"| {fmt_acc(e.get('avg_task_acc') if e else None)} "
            f"| {fmt_fgt(e.get('avg_forgetting') if e else None)} "
            f"| {fmt_acc(c.get('avg_task_acc') if c else None)} "
            f"| {fmt_fgt(c.get('avg_forgetting') if c else None)} "
            f"| {icon} |"
        )

    lines.append("")

    # ── 完成情况汇总 ──────────────────────────────────────────────
    total = (len(BASELINE_ORDER) * 2) + (len(ABLATION_ORDER) * 2)
    done = 0
    for algo in BASELINE_ORDER:
        if bl_emnist.get(algo) and bl_emnist[algo].get("avg_task_acc") is not None:
            done += 1
        if bl_cifar100.get(algo) and bl_cifar100[algo].get("avg_task_acc") is not None:
            done += 1
    for v in ABLATION_ORDER:
        if abl_emnist.get(v) and abl_emnist[v].get("avg_task_acc") is not None:
            done += 1
        if abl_cifar100.get(v) and abl_cifar100[v].get("avg_task_acc") is not None:
            done += 1

    lines += [
        "---",
        "",
        "## 三、实验进度",
        "",
        f"**{done} / {total}** 个实验已完成。",
        "",
        "### Baselines",
        "",
        "| 算法 | EMNIST | CIFAR100 |",
        "|------|:------:|:--------:|",
    ]
    for algo in BASELINE_ORDER:
        e_icon = completion_icon(bl_emnist.get(algo))
        c_icon = completion_icon(bl_cifar100.get(algo))
        lines.append(f"| {algo} | {e_icon} | {c_icon} |")

    lines += [
        "",
        "### 消融变体",
        "",
        "| 变体 | EMNIST | CIFAR100 |",
        "|------|:------:|:--------:|",
    ]
    for v in ABLATION_ORDER:
        e_icon = completion_icon(abl_emnist.get(v))
        c_icon = completion_icon(abl_cifar100.get(v))
        lines.append(f"| {ABLATION_DISPLAY[v]} | {e_icon} | {c_icon} |")

    lines += [
        "",
        "---",
        "",
        "## 四、实验结果目录索引",
        "",
        "```",
        "results/paper_experiments/",
        "├── baselines/",
        "│   ├── EMNIST-Letters/    # 8 baselines",
        "│   └── CIFAR100/          # 8 baselines",
        "└── ablation/",
        "    ├── EMNIST-Letters/    # 7 CollEdge 变体",
        "    │   ├── A_DCFCL_baseline/",
        "    │   ├── B_DER_only/",
        "    │   ├── C_DER_Directed/",
        "    │   ├── D_DER_Mask/",
        "    │   ├── E_DER_Dir_Mask_Full/",
        "    │   ├── F_noDER_Directed/",
        "    │   └── G_noDER_Mask/",
        "    └── CIFAR100/          # 同上",
        "```",
        "",
        "> 每个实验子目录包含：",
        "> - `training.log` — 完整训练日志（含最终 Avg Task Accuracy、Avg Forgetting 统计）",
        "> - `results.json` — 结构化指标数据（all_accuracies、per_task_acc 等）",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    print(f"[collect_results] Scanning: {RESULTS_ROOT}")

    bl_emnist   = collect_baselines("EMNIST-Letters")
    bl_cifar100 = collect_baselines("CIFAR100")
    abl_emnist  = collect_ablation("EMNIST-Letters")
    abl_cifar100 = collect_ablation("CIFAR100")

    # 统计
    def count(d):
        return sum(1 for v in d.values() if v and v.get("avg_task_acc") is not None)

    print(f"  Baselines  EMNIST   : {count(bl_emnist)}/{len(BASELINE_ORDER)}")
    print(f"  Baselines  CIFAR100 : {count(bl_cifar100)}/{len(BASELINE_ORDER)}")
    print(f"  Ablation   EMNIST   : {count(abl_emnist)}/{len(ABLATION_ORDER)}")
    print(f"  Ablation   CIFAR100 : {count(abl_cifar100)}/{len(ABLATION_ORDER)}")

    md = generate_markdown(bl_emnist, bl_cifar100, abl_emnist, abl_cifar100)
    OUTPUT_FILE.write_text(md, encoding="utf-8")
    print(f"\n[collect_results] Written → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
