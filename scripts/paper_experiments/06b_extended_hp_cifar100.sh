#!/usr/bin/env bash
# ==============================================================
# 06b_extended_hp_cifar100.sh
# Purpose : Extended CIFAR-100 HP sweep, triggered ONLY when the
#           primary sweep (06_full_hp_tune_cifar100.sh) does not
#           clear the ablation target.
#
#           Empirical data from the primary sweep (Runs 1–2):
#             sw=0.7 → 24.77%/forget=66.86%
#             sw=0.8 → 23.97%/forget=72.49%
#           ⇒ Higher sw is the WRONG direction with the group mask
#             on (the per-group peer pool is only 4, so cranking up
#             sw isolates the client further and destroys cross-task
#             knowledge in the DER buffer).
#
#           This script therefore probes the OPPOSITE direction
#           (lower sw / more peer trust). Mode is kept at gradient
#           — Stage 4 (08_directed_modes_cifar100.sh) is the right
#           place to vary the scoring mode, and pick_best_hp.py
#           doesn't propagate mode anyway.
#
#   sw=0.3 thr=0.00 temp=1.0   — trust peers more
#   sw=0.5 thr=0.10 temp=1.0   — original sw + strict filter
#   sw=0.4 thr=0.05 temp=0.5   — slightly peer-biased + filter + sharp softmax
#
# Total runs: 3 (~3 h on RTX 5080).
#
# Results : results/paper_experiments/hp_tune/CIFAR100/full_*
# Usage   : bash scripts/paper_experiments/06b_extended_hp_cifar100.sh
# ==============================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

OUT_BASE="results/paper_experiments/hp_tune/CIFAR100"
CFG="configs/dcfcl_cifar100.yaml"
SEED=42

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
_log "=== START 06b_extended_hp_cifar100.sh ==="

run_full() {
    local sw=$1 thr=$2 temp=$3 tag=$4
    local out="$OUT_BASE/full_${tag}"
    # Skip if a previous run already produced results.json for this tag
    # (allows the orchestrator to re-launch 06b idempotently after a
    # parallel pre-launch).
    if find "$out" -name results.json 2>/dev/null | grep -q .; then
        _log "≡≡≡ $tag already complete, skipping"
        return 0
    fi
    mkdir -p "$out"
    _log ">>> $tag (sw=$sw thr=$thr temp=$temp)"
    python main.py \
        --config "$CFG" --seed $SEED \
        --algorithm CollEdge \
        --use_der \
        --buffer_size 500 --der_alpha 0.5 --der_beta 0.5 \
        --directed_collaboration \
        --directed_mode gradient \
        --directed_self_weight $sw \
        --directed_threshold $thr \
        --directed_temperature $temp \
        --use_coalition_mask \
        --coalition_mask_type group \
        --num_client_groups 2 \
        --result_dir "$out" \
    && _log "<<< $tag DONE" || _log "<<< $tag FAILED"
}

# Run the three configs in parallel pairs (2 + 1) to halve wall-clock.
run_full 0.3 0.00 1.0 "sw0.3_thr0.00_temp1.0" &
PID_1=$!
run_full 0.5 0.10 1.0 "sw0.5_thr0.10_temp1.0" &
PID_2=$!
wait $PID_1; wait $PID_2

run_full 0.4 0.05 0.5 "sw0.4_thr0.05_temp0.5"

_log "=== END 06b_extended_hp_cifar100.sh ==="
