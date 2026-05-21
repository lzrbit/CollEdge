#!/usr/bin/env bash
# ==============================================================
# 08_directed_modes_cifar100.sh
# Purpose : CIFAR-100 counterpart of 07_directed_modes_emnist.sh.
#           Compares gradient / task_aware / hybrid directed-aggregation
#           strategies on the FULL CollEdge configuration.
#
#           This version runs the first two modes IN PARALLEL on
#           the same GPU (one ResNet-18/CIFAR-100 job fits in ~6 GB,
#           so two fit comfortably in 16 GB) to roughly halve the
#           wall-clock cost of Stage 4.
#
# Total runs: 3 (gradient ‖ task_aware, then hybrid)
#
# Results : results/paper_experiments/directed_modes/CIFAR100/
# Usage   : bash scripts/paper_experiments/08_directed_modes_cifar100.sh
# ==============================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

OUT_BASE="results/paper_experiments/directed_modes/CIFAR100"
CFG="configs/dcfcl_cifar100.yaml"
SEED=42

SW=${COLLEDGE_SW:-0.8}
THR=${COLLEDGE_THR:-0.05}
TEMP=${COLLEDGE_TEMP:-1.0}

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
_log "=== START 08_directed_modes_cifar100.sh (sw=$SW thr=$THR temp=$TEMP) ==="

run_mode() {
    local mode=$1
    local out="$OUT_BASE/mode_${mode}"
    mkdir -p "$out"
    _log ">>> mode=$mode"
    python main.py \
        --config "$CFG" --seed $SEED \
        --algorithm CollEdge \
        --use_der \
        --buffer_size 500 --der_alpha 0.5 --der_beta 0.5 \
        --directed_collaboration \
        --directed_mode $mode \
        --directed_self_weight $SW \
        --directed_threshold $THR \
        --directed_temperature $TEMP \
        --use_coalition_mask \
        --coalition_mask_type group \
        --num_client_groups 2 \
        --result_dir "$out" \
    && _log "<<< mode=$mode DONE" || _log "<<< mode=$mode FAILED"
}

# Note: when the pipeline transitions from Stage 3 to Stage 4 it may
# overlap with a still-running parallel 06b probe. To avoid a 3-process
# GPU memory blow-up, run the first mode (gradient) solo first; by the
# time it finishes, any earlier overlap will have cleared, and we can
# safely run the remaining two modes in parallel.
_log "Running gradient solo first (clears any prior parallel probe)…"
run_mode gradient

_log "Launching task_aware and hybrid in parallel…"
run_mode task_aware &
PID_T=$!
run_mode hybrid &
PID_H=$!
wait $PID_T; wait $PID_H
_log "Parallel pair complete."

_log "=== END 08_directed_modes_cifar100.sh ==="
