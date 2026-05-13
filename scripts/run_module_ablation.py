#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CollEdge 模块消融脚本

消融目标：验证改进后 CollEdge 完整版方法中各模块的贡献。

模块定义
--------
A. DER++        （use_der）          ：暗经验回放，抑制遗忘
B. 有向协作      （directed_collaboration）：非对称联盟聚合
C. 联盟 Mask    （use_coalition_mask）：限制部分客户端联盟组合

消融矩阵（7 组实验）
-------------------
| 实验名称             | DER++ | 有向协作 | mask |
|---------------------|-------|---------|------|
| DCFCL (baseline)    |  ✗   |   ✗    |  ✗  |
| CollEdge-base      |  ✓   |   ✗    |  ✗  |
| CollEdge+Directed  |  ✓   |   ✓    |  ✗  |
| CollEdge+Mask      |  ✓   |   ✗    |  ✓  |
| CollEdge+D+M (Full)|  ✓   |   ✓    |  ✓  |
| (extra) no-DER+Dir  |  ✗   |   ✓    |  ✗  |
| (extra) no-DER+Mask |  ✗   |   ✗    |  ✓  |

用法
----
  # 快速验证（少轮次）
  python scripts/run_module_ablation.py --quick

  # 完整消融（EMNIST，使用 YAML 基础配置）
  python scripts/run_module_ablation.py

  # 指定数据集（CIFAR100）
  python scripts/run_module_ablation.py --dataset cifar100

  # 仅运行核心 5 组（跳过 extra 两组）
  python scripts/run_module_ablation.py --core_only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

PYTHON = sys.executable
MAIN = str(REPO_DIR / "main.py")

# ---------------------------------------------------------------------------
# 基础配置（EMNIST-Letters）
# ---------------------------------------------------------------------------
EMNIST_BASE = dict(
    dataset="EMNIST-Letters",
    data_split_file="split_files/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl",
    num_users=8,
    num_tasks=6,
    num_rounds=60,
    local_epochs=100,
    batch_size=64,
    model="cnn",
    lr=1e-4,
    weight_decay=1e-5,
    seed=42,
    # DCFCL 公共参数
    sw=0.1,
    lambda_kd=0.2,
    lambda_proto_aug=2.0,
    global_weight=0.9,
    ema_global=0.9,
    dcfcl_broadcast=0,
    # DER++ 参数（use_der 会由实验本身控制）
    buffer_size=500,
    der_alpha=0.5,
    der_beta=0.5,
    # 有向协作参数（directed_collaboration 由实验控制）
    directed_mode="gradient",
    directed_temperature=1.0,
    directed_self_weight=0.5,
    # Mask 参数（use_coalition_mask 由实验控制）
    coalition_mask_type="group",
    num_client_groups=2,
)

CIFAR100_BASE = dict(
    dataset="CIFAR100",
    data_split_file="split_files/CIFAR100_split_cn10_tn10_cet10_cs1_s2571.pkl",
    num_users=10,
    num_tasks=10,
    num_rounds=100,
    local_epochs=50,
    batch_size=64,
    model="resnet18",
    lr=1e-3,
    weight_decay=1e-3,
    seed=42,
    sw=0.1,
    lambda_kd=0.2,
    lambda_proto_aug=2.0,
    global_weight=0.9,
    ema_global=0.9,
    dcfcl_broadcast=0,
    buffer_size=500,
    der_alpha=0.5,
    der_beta=0.5,
    directed_mode="gradient",
    directed_temperature=1.0,
    directed_self_weight=0.5,
    coalition_mask_type="group",
    num_client_groups=2,
)

# ---------------------------------------------------------------------------
# 消融矩阵（模块开关）
# 每个 entry 键对应 main.py 的 flag；store_true 类型以 bool 方式处理。
# ---------------------------------------------------------------------------
ABLATION_VARIANTS = {
    # ----- 核心 5 组 -----
    "DCFCL_baseline": dict(
        algorithm="DCFCL",   # 原始 DCFCL（无 DER, 无有向, 无 mask）
    ),
    "CollEdge_base": dict(
        algorithm="CollEdge",
        use_der=True,
        directed_collaboration=False,
        use_coalition_mask=False,
    ),
    "CollEdge_Directed": dict(
        algorithm="CollEdge",
        use_der=True,
        directed_collaboration=True,
        use_coalition_mask=False,
    ),
    "CollEdge_Mask": dict(
        algorithm="CollEdge",
        use_der=True,
        directed_collaboration=False,
        use_coalition_mask=True,
    ),
    "CollEdge_Full": dict(
        algorithm="CollEdge",
        use_der=True,
        directed_collaboration=True,
        use_coalition_mask=True,
    ),
    # ----- 额外 2 组：验证 DER 与其他模块的独立性 -----
    "DCFCL_NoDER_Directed": dict(
        algorithm="CollEdge",
        use_der=False,
        directed_collaboration=True,
        use_coalition_mask=False,
    ),
    "DCFCL_NoDER_Mask": dict(
        algorithm="CollEdge",
        use_der=False,
        directed_collaboration=False,
        use_coalition_mask=True,
    ),
}

CORE_VARIANTS = [
    "DCFCL_baseline",
    "CollEdge_base",
    "CollEdge_Directed",
    "CollEdge_Mask",
    "CollEdge_Full",
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def build_cmd(base_config: dict, variant_cfg: dict, result_dir: str) -> list:
    """从基础配置和变体配置组装 CLI 命令。"""
    merged = {**base_config, **variant_cfg}
    cmd = [PYTHON, MAIN]

    bool_flags_true = []   # store_true flags 需要单独处理
    bool_flags_false = []  # store_false flags（如 --no_use_der）

    for k, v in merged.items():
        if k == "algorithm":
            cmd += ["--algorithm", str(v)]
        elif isinstance(v, bool):
            if k == "use_der":
                if v:
                    cmd.append("--use_der")
                else:
                    cmd.append("--no_use_der")
            elif k == "directed_collaboration":
                if v:
                    cmd.append("--directed_collaboration")
                # False = 不加 flag（默认 off）
            elif k == "use_coalition_mask":
                if v:
                    cmd.append("--use_coalition_mask")
        else:
            cmd += [f"--{k}", str(v)]

    cmd += ["--result_dir", result_dir]
    return cmd


def run_experiment(name: str, cmd: list, timeout: int = 7200) -> dict | None:
    """运行单次实验并解析结果。"""
    print(f"\n{'='*60}")
    print(f"[RUN]  {name}")
    # 只显示关键参数
    key_args = ["--algorithm", "--use_der", "--no_use_der",
                "--directed_collaboration", "--use_coalition_mask"]
    shown = []
    i = 0
    while i < len(cmd):
        if cmd[i] in key_args:
            shown.append(cmd[i])
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                shown.append(cmd[i + 1])
                i += 2
            else:
                i += 1
        else:
            i += 1
    print(f"       {' '.join(shown)}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_DIR),
        )
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name}")
        return None

    if result.returncode != 0:
        print(f"[FAIL]  {name}  (exit={result.returncode}, {elapsed:.0f}s)")
        if result.stderr:
            print(result.stderr[-800:])
        return None

    print(f"[DONE]  {name}  ({elapsed:.0f}s)")

    # 解析 results.json
    metrics = _parse_results(cmd, elapsed)
    if metrics:
        metrics["name"] = name
        metrics["elapsed_s"] = elapsed
    return metrics


def _parse_results(cmd: list, elapsed: float) -> dict | None:
    """查找最新结果目录并解析 results.json。"""
    # 从 cmd 提取 algorithm 和 dataset
    algorithm = _flag_val(cmd, "--algorithm", "")
    dataset = _flag_val(cmd, "--dataset", "")
    result_dir_base = _flag_val(cmd, "--result_dir", "./results")

    prefix = f"{algorithm}_{dataset}_"
    try:
        candidates = sorted(
            [d for d in os.listdir(result_dir_base) if d.startswith(prefix)],
            reverse=True,
        )
    except FileNotFoundError:
        return None

    if not candidates:
        return None

    json_path = os.path.join(result_dir_base, candidates[0], "results.json")
    if not os.path.exists(json_path):
        return None

    with open(json_path) as f:
        data = json.load(f)

    num_tasks = int(_flag_val(cmd, "--num_tasks", "6"))
    all_acc = data.get("all_accuracies", [])
    rpt = len(all_acc) // num_tasks if num_tasks > 0 else 0
    if rpt > 0:
        phase_end = [all_acc[rpt * (t + 1) - 1] for t in range(num_tasks)]
        avg_task = sum(phase_end) / len(phase_end)
    else:
        phase_end = []
        avg_task = data.get("final_accuracy", 0)

    return {
        "final_accuracy": data.get("final_accuracy", 0),
        "forgetting_rate": data.get("forgetting_rate", 0),
        "avg_task_accuracy": avg_task,
        "phase_end_accs": phase_end,
        "result_dir": candidates[0],
    }


def _flag_val(cmd: list, flag: str, default: str) -> str:
    """从命令列表中提取某 flag 的值。"""
    try:
        idx = cmd.index(flag)
        return cmd[idx + 1]
    except (ValueError, IndexError):
        return default


def print_summary(all_results: dict):
    """打印消融结果汇总表。"""
    print("\n" + "=" * 80)
    print("CollEdge 模块消融汇总")
    print("=" * 80)

    header = f"{'实验名称':<30} {'Final Acc':>10} {'Avg Task Acc':>13} {'Forgetting':>11}"
    print(header)
    print("-" * 70)

    for name, r in all_results.items():
        if r is None:
            print(f"  {name:<28} {'FAILED':>10}")
            continue
        fa = r.get("final_accuracy", 0)
        at = r.get("avg_task_accuracy", 0)
        fg = r.get("forgetting_rate", 0)
        print(f"  {name:<28} {fa*100:>9.2f}%  {at*100:>12.2f}%  {fg*100:>10.2f}%")

    print("=" * 80)

    # 模块贡献分析
    keys = list(all_results.keys())
    base_key = "CollEdge_base"
    full_key = "CollEdge_Full"

    if base_key in all_results and all_results[base_key]:
        base_acc = all_results[base_key]["final_accuracy"]
        print(f"\n变体相对于 {base_key} 的增益（Final Acc）:")
        for k, r in all_results.items():
            if k == base_key or r is None:
                continue
            delta = r["final_accuracy"] - base_acc
            sign = "+" if delta >= 0 else ""
            print(f"  {k:<28}  {sign}{delta*100:.2f}%")


def save_results(all_results: dict, output_dir: str):
    """保存消融结果到 JSON 文件。"""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"module_ablation_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存至: {out_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CollEdge 模块消融实验")
    parser.add_argument("--dataset", choices=["emnist", "cifar100"], default="emnist",
                        help="数据集 (emnist|cifar100)")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：减少轮次（num_rounds=12, local_epochs=20）用于验证")
    parser.add_argument("--core_only", action="store_true",
                        help="仅运行核心 5 组消融，跳过 extra 2 组")
    parser.add_argument("--result_dir", type=str, default="./results",
                        help="结果保存目录")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="单次实验超时（秒）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 选择基础配置
    base_cfg = EMNIST_BASE.copy() if args.dataset == "emnist" else CIFAR100_BASE.copy()

    # 快速模式
    if args.quick:
        base_cfg.update(num_rounds=12, local_epochs=20)
        print("[快速模式] num_rounds=12, local_epochs=20")

    # 选择变体
    variants = CORE_VARIANTS if args.core_only else list(ABLATION_VARIANTS.keys())
    print(f"\n消融变体数量: {len(variants)}")
    print(f"{'核心 5 组' if args.core_only else '全部 7 组'}: {variants}")

    result_dir = os.path.abspath(args.result_dir)
    all_results = {}

    for name in variants:
        variant_cfg = ABLATION_VARIANTS[name]
        cmd = build_cmd(base_cfg, variant_cfg, result_dir)
        metrics = run_experiment(name, cmd, timeout=args.timeout)
        all_results[name] = metrics

    # 汇总输出
    print_summary(all_results)

    # 保存 JSON
    ablation_dir = os.path.join(result_dir, "module_ablation_results")
    save_results(all_results, ablation_dir)


if __name__ == "__main__":
    main()
