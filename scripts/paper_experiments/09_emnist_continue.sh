#!/usr/bin/env bash
# ==============================================================
# 09_emnist_continue.sh
# Purpose : Resume EMNIST experiments from where the pipeline
#           was interrupted. Already-finished runs are skipped.
#
# Already done (results exist):
#   sw0.5_thr0.00_temp1.0  ✓
#   sw0.6_thr0.00_temp1.0  ✓
#   sw0.7_thr0.00_temp1.0  (currently running, skipped here)
#
# Remaining Stage A (self_weight sweep):
#   sw0.8_thr0.00_temp1.0
#
# Stage B (threshold sweep, using sw=0.7 as anchor):
#   sw0.7_thr0.05_temp1.0
#   sw0.7_thr0.10_temp1.0
#
# Stage C (temperature sweep, using sw=0.7, thr=0.05):
#   sw0.7_thr0.05_temp0.5
#   sw0.7_thr0.05_temp2.0
#
# Stage B corners:
#   sw0.6_thr0.05_temp1.0
#   sw0.8_thr0.05_temp1.0
#
# Stage 2 — directed-mode comparison (3 runs, uses best HP):
#   gradient / task_aware / hybrid
#
# Total remaining: 8 + 3 = 11 runs  (~45 min on a single GPU)
#
# Usage:
#   cd CollEdge
#   nohup bash scripts/paper_experiments/09_emnist_continue.sh \
#         >> pipeline_emnist.log 2>&1 &
#   tail -f pipeline_emnist.log
# ==============================================================

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
SCRIPTS="scripts/paper_experiments"

# ─── Activate dcfcl conda environment ────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/lzr/miniconda3)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dcfcl
echo "[env] python = $(which python)"

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─── Helper: skip if results already exist ────────────────────────────────────
run_if_missing() {
    # $1=tag  rest=python args
    local tag=$1; shift
    local out="results/paper_experiments/hp_tune/EMNIST-Letters/full_${tag}"
    if ls "${out}"/*/results.json 2>/dev/null | grep -q .; then
        _log "SKIP $tag (results already exist)"
        return 0
    fi
    mkdir -p "$out"
    _log ">>> $tag"
    python main.py "$@" --result_dir "$out" \
    && _log "<<< $tag DONE" || _log "<<< $tag FAILED"
}

COMMON="--config configs/dcfcl_emnist.yaml --seed 42 --algorithm CollEdge
    --use_der --buffer_size 500 --der_alpha 0.5 --der_beta 0.5
    --directed_collaboration --directed_mode gradient
    --use_coalition_mask --coalition_mask_type group --num_client_groups 2"

_log "=== START 09_emnist_continue.sh (remaining HP + mode runs) ==="

# Wait for the currently-running sw0.7 to finish (if still active)
SW7_DIR="results/paper_experiments/hp_tune/EMNIST-Letters/full_sw0.7_thr0.00_temp1.0"
_log "Checking if sw0.7_thr0.00_temp1.0 is still running..."
while ps aux | grep -q "[p]ython.*sw0.7_thr0.00_temp1.0"; do
    _log "  sw0.7 still running — waiting 30s..."
    sleep 30
done
_log "sw0.7_thr0.00_temp1.0 done (or was already done)."

# ─── Stage A: remaining self_weight value ─────────────────────────────────────
run_if_missing "sw0.8_thr0.00_temp1.0" \
    $COMMON \
    --directed_self_weight 0.8 --directed_threshold 0.0 --directed_temperature 1.0

# ─── Stage B: threshold sweep (anchor on sw=0.7) ──────────────────────────────
run_if_missing "sw0.7_thr0.05_temp1.0" \
    $COMMON \
    --directed_self_weight 0.7 --directed_threshold 0.05 --directed_temperature 1.0

run_if_missing "sw0.7_thr0.10_temp1.0" \
    $COMMON \
    --directed_self_weight 0.7 --directed_threshold 0.10 --directed_temperature 1.0

# ─── Stage C: temperature sweep ───────────────────────────────────────────────
run_if_missing "sw0.7_thr0.05_temp0.5" \
    $COMMON \
    --directed_self_weight 0.7 --directed_threshold 0.05 --directed_temperature 0.5

run_if_missing "sw0.7_thr0.05_temp2.0" \
    $COMMON \
    --directed_self_weight 0.7 --directed_threshold 0.05 --directed_temperature 2.0

# ─── Stage B corners ──────────────────────────────────────────────────────────
run_if_missing "sw0.6_thr0.05_temp1.0" \
    $COMMON \
    --directed_self_weight 0.6 --directed_threshold 0.05 --directed_temperature 1.0

run_if_missing "sw0.8_thr0.05_temp1.0" \
    $COMMON \
    --directed_self_weight 0.8 --directed_threshold 0.05 --directed_temperature 1.0

_log "STAGE A+B+C (HP sweep) DONE"

# ─── Auto-pick best HP ────────────────────────────────────────────────────────
_log "Picking best EMNIST HP..."
eval "$(python "$SCRIPTS/pick_best_hp.py" EMNIST-Letters)"
_log "  → SW=$COLLEDGE_SW  THR=$COLLEDGE_THR  TEMP=$COLLEDGE_TEMP"
export COLLEDGE_SW COLLEDGE_THR COLLEDGE_TEMP

# ─── Stage 2: directed-mode comparison ────────────────────────────────────────
_log "STAGE 2 — Directed mode comparison (3 runs, ~12 min)"

MODES_DIR="results/paper_experiments/directed_modes/EMNIST-Letters"
run_mode() {
    local mode=$1
    local out="${MODES_DIR}/mode_${mode}"
    if ls "${out}"/*/results.json 2>/dev/null | grep -q .; then
        _log "SKIP mode_${mode} (results already exist)"
        return 0
    fi
    mkdir -p "$out"
    _log ">>> mode=$mode (sw=$COLLEDGE_SW thr=$COLLEDGE_THR temp=$COLLEDGE_TEMP)"
    python main.py \
        --config configs/dcfcl_emnist.yaml --seed 42 \
        --algorithm CollEdge \
        --use_der --buffer_size 500 --der_alpha 0.5 --der_beta 0.5 \
        --directed_collaboration \
        --directed_mode $mode \
        --directed_self_weight $COLLEDGE_SW \
        --directed_threshold $COLLEDGE_THR \
        --directed_temperature $COLLEDGE_TEMP \
        --use_coalition_mask \
        --coalition_mask_type group \
        --num_client_groups 2 \
        --result_dir "$out" \
    && _log "<<< mode=$mode DONE" || _log "<<< mode=$mode FAILED"
}

run_mode gradient
run_mode task_aware
run_mode hybrid

_log "STAGE 2 — mode comparison DONE"
_log "=== ALL EMNIST EXPERIMENTS COMPLETE ==="
_log "Run analysis: python scripts/paper_experiments/collect_full_results.py"
