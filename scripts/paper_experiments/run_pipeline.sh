#!/usr/bin/env bash
# ==============================================================
# run_pipeline.sh
# Purpose : Master pipeline that runs all experiment stages in
#           order and automatically carries the best hyper-
#           parameter forward between stages.
#
# Stage order:
#   [1] EMNIST HP sweep     (05_full_hp_tune_emnist.sh,  10 runs, ~3.7 h)
#   [2] EMNIST mode compare (07_directed_modes_emnist.sh, 3 runs, ~1.1 h)
#   [3] CIFAR-100 HP sweep  (06_full_hp_tune_cifar100.sh, 4 runs, ~48 h)
#   [4] CIFAR-100 mode cmp  (08_directed_modes_cifar100.sh,3 runs, ~36 h)
#
# Estimated total wall-clock: ~89 h (single GPU)
#   EMNIST stages: ~5 h     (finish same day)
#   CIFAR stages : ~84 h    (run overnight / multi-day)
#
# Usage:
#   cd CollEdge
#   nohup bash scripts/paper_experiments/run_pipeline.sh \
#         > pipeline.log 2>&1 &
#   tail -f pipeline.log          # watch progress
#   grep ">>>\|<<<\|STAGE\|DONE" pipeline.log   # summary only
# ==============================================================

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
SCRIPTS="scripts/paper_experiments"

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
_log "============================="
_log "  CollEdge experiment pipeline"
_log "============================="

# ──────────────────────────────────────────────────────────────
# STAGE 1 : EMNIST hyper-parameter sweep (~3.7 h, 10 runs)
# ──────────────────────────────────────────────────────────────
_log "STAGE 1/4 — EMNIST HP sweep (est. 3.7 h)"
bash "$SCRIPTS/05_full_hp_tune_emnist.sh"
_log "STAGE 1 DONE"

# ──────────────────────────────────────────────────────────────
# STAGE 2 : EMNIST directed-mode comparison (~1.1 h, 3 runs)
# Auto-picks best HP from stage 1.
# ──────────────────────────────────────────────────────────────
_log "STAGE 2/4 — Picking best EMNIST HP …"
eval "$(python "$SCRIPTS/pick_best_hp.py" EMNIST-Letters)"
_log "  → COLLEDGE_SW=$COLLEDGE_SW  THR=$COLLEDGE_THR  TEMP=$COLLEDGE_TEMP"
export COLLEDGE_SW COLLEDGE_THR COLLEDGE_TEMP
_log "STAGE 2/4 — EMNIST mode comparison (est. 1.1 h)"
bash "$SCRIPTS/07_directed_modes_emnist.sh"
_log "STAGE 2 DONE"

# ──────────────────────────────────────────────────────────────
# STAGE 3 : CIFAR-100 hyper-parameter sweep (~48 h, 4 runs)
# ──────────────────────────────────────────────────────────────
_log "STAGE 3/4 — CIFAR-100 HP sweep (est. 48 h)"
unset COLLEDGE_SW COLLEDGE_THR COLLEDGE_TEMP  # let script use its own defaults
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
_log "  ALL STAGES COMPLETE"
_log "============================="
