#!/usr/bin/env bash
# ==============================================================
# run_all.sh — 主控脚本，依次执行全部 4 组实验
#
# 执行顺序（建议按此顺序，先短后长）：
#   01  Baselines EMNIST-Letters   ( 8 runs × 60 rounds )
#   02  Baselines CIFAR-100        ( 8 runs × 100 rounds )
#   03  Ablation EMNIST-Letters    ( 7 runs × 60 rounds  )
#   04  Ablation CIFAR-100         ( 7 runs × 100 rounds )
#
# Usage:
#   # 完整执行
#   bash scripts/paper_experiments/run_all.sh
#
#   # 仅执行某一组（单独调用各脚本）
#   bash scripts/paper_experiments/01_baselines_emnist.sh
# ==============================================================

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

_log "========================================================"
_log "  PAPER EXPERIMENTS — Full Run"
_log "  Scripts: $SCRIPTS_DIR"
_log "========================================================"

run_script() {
    local s="$1"
    _log ""
    _log "-------- Running $s --------"
    if bash "$SCRIPTS_DIR/$s"; then
        _log "-------- $s COMPLETED --------"
    else
        _log "-------- $s FAILED (continuing) --------"
    fi
}

run_script "01_baselines_emnist.sh"
run_script "02_baselines_cifar100.sh"
run_script "03_ablation_emnist.sh"
run_script "04_ablation_cifar100.sh"

_log ""
_log "========================================================"
_log "  ALL DONE — generate summary:"
_log "  python scripts/paper_experiments/collect_results.py"
_log "========================================================"
