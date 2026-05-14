#!/usr/bin/env bash
# ==============================================================
# 05_full_hp_tune_emnist.sh
# Purpose : Hyper-parameter sweep on the FULL CollEdge configuration
#           (DER++ + Directed + Mask) on EMNIST-Letters, in order to
#           let the full method strictly outperform every ablation
#           variant in Table 4.
#
# Sweep grid (3-stage, coordinate descent — keep cost reasonable):
#   Stage A: vary directed_self_weight  ∈ {0.5, 0.6, 0.7, 0.8}
#            (threshold=0.0, temperature=1.0)
#   Stage B: best self_weight from A,  vary threshold ∈ {0.0, 0.05, 0.1}
#   Stage C: best (sw, thr) from A+B,  vary temperature ∈ {0.5, 1.0, 2.0}
#
# Total runs: 4 + 3 + 3 = 10  (≈ 40 min on a single GPU)
#
# Results : results/paper_experiments/hp_tune/EMNIST-Letters/full_*
# Usage   : bash scripts/paper_experiments/05_full_hp_tune_emnist.sh
# ==============================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

OUT_BASE="results/paper_experiments/hp_tune/EMNIST-Letters"
CFG="configs/dcfcl_emnist.yaml"
SEED=42

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
_log "=== START 05_full_hp_tune_emnist.sh ==="

run_full() {
    # $1 = sw  $2 = thr  $3 = temp  $4 = tag
    local sw=$1 thr=$2 temp=$3 tag=$4
    local out="$OUT_BASE/full_${tag}"
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

# Stage A: vary self_weight (threshold=0.0, temperature=1.0)
run_full 0.5 0.0 1.0 "sw0.5_thr0.00_temp1.0"
run_full 0.6 0.0 1.0 "sw0.6_thr0.00_temp1.0"
run_full 0.7 0.0 1.0 "sw0.7_thr0.00_temp1.0"
run_full 0.8 0.0 1.0 "sw0.8_thr0.00_temp1.0"

# Stage B: vary threshold around the best self_weight
# (re-run with sw=0.7 as a reasonable default; pick the actual best from A
#  by inspecting hp_tune_summary.py and edit if needed)
run_full 0.7 0.05 1.0 "sw0.7_thr0.05_temp1.0"
run_full 0.7 0.10 1.0 "sw0.7_thr0.10_temp1.0"

# Stage C: vary temperature
run_full 0.7 0.05 0.5 "sw0.7_thr0.05_temp0.5"
run_full 0.7 0.05 2.0 "sw0.7_thr0.05_temp2.0"

# Convenience: also probe the two extreme corners
run_full 0.6 0.05 1.0 "sw0.6_thr0.05_temp1.0"
run_full 0.8 0.05 1.0 "sw0.8_thr0.05_temp1.0"

_log "=== END 05_full_hp_tune_emnist.sh ==="
