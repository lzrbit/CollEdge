#!/usr/bin/env python3
"""Auto-pick the best (highest avg_task accuracy) hyper-parameter config
from a hp_tune directory and print it as shell 'export' statements so the
calling pipeline can `eval` the output.

Usage:
    eval $(python scripts/paper_experiments/pick_best_hp.py EMNIST-Letters)
    eval $(python scripts/paper_experiments/pick_best_hp.py CIFAR100)
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def parse_tag(name: str) -> tuple[float, float, float]:
    """Parse sw / thr / temp from directory name like 'full_sw0.7_thr0.05_temp1.0'."""
    sw = float(re.search(r"sw([\d.]+)", name).group(1))
    thr = float(re.search(r"thr([\d.]+)", name).group(1))
    temp = float(re.search(r"temp([\d.]+)", name).group(1))
    return sw, thr, temp


def avg_task_from(path: str) -> float:
    d = json.load(open(path))
    per_task = d.get("per_task_acc") or []
    if not per_task:
        return float("nan")
    return sum(sum(row) / len(row) for row in per_task) / len(per_task)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: pick_best_hp.py <EMNIST-Letters|CIFAR100>", file=sys.stderr)
        sys.exit(1)

    dataset = sys.argv[1]
    sweep_dir = REPO / "results" / "paper_experiments" / "hp_tune" / dataset

    if not sweep_dir.exists():
        print(f"# sweep_dir not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    best_score = float("-inf")
    best_sw, best_thr, best_temp = 0.7, 0.05, 1.0  # fallback defaults

    for run_dir in sorted(sweep_dir.glob("full_*")):
        files = glob.glob(str(run_dir / "**/results.json"), recursive=True)
        if not files:
            continue
        score = avg_task_from(sorted(files)[-1])
        if score > best_score:
            best_score = score
            try:
                best_sw, best_thr, best_temp = parse_tag(run_dir.name)
            except Exception:
                pass

    # Output as shell exports so callers can `eval $(python pick_best_hp.py ...)`
    print(f"export COLLEDGE_SW={best_sw}")
    print(f"export COLLEDGE_THR={best_thr}")
    print(f"export COLLEDGE_TEMP={best_temp}")
    # Also emit human-readable comment
    print(f"# best avg_task={best_score:.4f}  from dataset={dataset}")


if __name__ == "__main__":
    main()
