#!/usr/bin/env bash
# ==============================================================
# run_pipeline_cifar.sh
# Purpose : CIFAR-only sub-pipeline. EMNIST stages are already done,
#           so this only runs CIFAR Stage 3 (HP sweep) and Stage 4
#           (directed-mode comparison), carrying the best HP forward.
#
# Stage order:
#   [3] CIFAR-100 HP sweep  (06_full_hp_tune_cifar100.sh, 4 runs, ~48 h)
#   [4] CIFAR-100 mode cmp  (08_directed_modes_cifar100.sh,3 runs, ~36 h)
#
# Estimated total wall-clock: ~84 h (single GPU).
#
# Usage:
#   cd CollEdge
#   nohup bash scripts/paper_experiments/run_pipeline_cifar.sh \
#         > pipeline_cifar.log 2>&1 &
#   tail -f pipeline_cifar.log
# ==============================================================

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
SCRIPTS="scripts/paper_experiments"

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
_log "============================="
_log "  CollEdge CIFAR-only pipeline"
_log "============================="

# ──────────────────────────────────────────────────────────────
# STAGE 3 : CIFAR-100 hyper-parameter sweep (~48 h, 4 runs)
# ──────────────────────────────────────────────────────────────
_log "STAGE 3/4 — CIFAR-100 HP sweep (est. 48 h)"
unset COLLEDGE_SW COLLEDGE_THR COLLEDGE_TEMP 2>/dev/null || true
bash "$SCRIPTS/06_full_hp_tune_cifar100.sh"
_log "STAGE 3 DONE"

# ──────────────────────────────────────────────────────────────
# STAGE 4 : CIFAR-100 directed-mode comparison (~36 h, 3 runs)
# ──────────────────────────────────────────────────────────────
_log "STAGE 4/4 — Picking best CIFAR-100 HP …"
eval "$(python "$SCRIPTS/pick_best_hp.py" CIFAR100)"
_log "  → COLLEDGE_SW=$COLLEDGE_SW  THR=$COLLEDGE_THR  TEMP=$COLLEDGE_TEMP"
export COLLEDGE_SW COLLEDGE_THR COLLEDGE_TEMP
_log "STAGE 4/4 — CIFAR-100 mode comparison (est. 36 h)"
bash "$SCRIPTS/08_directed_modes_cifar100.sh"
_log "STAGE 4 DONE"

_log "============================="
_log "  CIFAR PIPELINE COMPLETE"
_log "============================="
