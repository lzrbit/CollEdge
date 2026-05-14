#!/usr/bin/env python3
"""Summarise hyper-parameter sweep / directed-mode results.

Usage:
    python scripts/paper_experiments/collect_full_results.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    per_task = d.get("per_task_acc") or []
    avg_task = (
        sum(sum(row) / len(row) for row in per_task) / len(per_task)
        if per_task
        else float("nan")
    )
    return {
        "final": d.get("final_accuracy", float("nan")),
        "forget": d.get("forgetting_rate", float("nan")),
        "avg_task": avg_task,
    }


def _scan(root: Path) -> list[tuple[str, dict]]:
    rows = []
    for run_dir in sorted(root.glob("*/")):
        cfg_name = run_dir.name
        files = glob.glob(str(run_dir / "**/results.json"), recursive=True)
        if not files:
            rows.append((cfg_name, None))
            continue
        files.sort()
        rows.append((cfg_name, _load(files[-1])))
    return rows


def _print_section(title: str, root: Path) -> None:
    print(f"\n=== {title} ===")
    print(f"  source: {root.relative_to(REPO)}")
    if not root.exists():
        print("  (no results yet)")
        return
    rows = _scan(root)
    if not rows:
        print("  (empty)")
        return
    print(f"  {'config':<40s} {'final':>8s} {'forget':>8s} {'avg_task':>9s}")
    for name, m in rows:
        if m is None:
            print(f"  {name:<40s} {'MISSING':>8s}")
        else:
            print(
                f"  {name:<40s} {m['final']:>8.4f} {m['forget']:>8.4f} "
                f"{m['avg_task']:>9.4f}"
            )
    # Highlight best by avg_task
    valid = [(n, m) for n, m in rows if m is not None]
    if valid:
        best = max(valid, key=lambda x: x[1]["avg_task"])
        print(f"  >> best avg_task: {best[0]}  ({best[1]['avg_task']:.4f})")


def main() -> None:
    base = REPO / "results" / "paper_experiments"
    for ds in ("EMNIST-Letters", "CIFAR100"):
        _print_section(f"Ablation [{ds}]", base / "ablation" / ds)
    for ds in ("EMNIST-Letters", "CIFAR100"):
        _print_section(f"HP-tune [{ds}]", base / "hp_tune" / ds)
    for ds in ("EMNIST-Letters", "CIFAR100"):
        _print_section(f"Directed modes [{ds}]", base / "directed_modes" / ds)


if __name__ == "__main__":
    main()
