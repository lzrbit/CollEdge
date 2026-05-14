#!/usr/bin/env bash
# ==============================================================
# 06_full_hp_tune_cifar100.sh
# Purpose : Hyper-parameter sweep on the FULL CollEdge configuration
#           (DER++ + Directed + Mask) on CIFAR-100. The current Full
#           configuration over-trusts peers (self_weight=0.5) and the
#           directed update destabilises the DER++ buffer
#           (final acc 19.4% vs. DER-only 38.5%). The sweep below
#           biases the aggregation more strongly toward the local
#           model and filters weakly-aligned peers.
#
# Sweep grid (3-stage):
#   Stage A: self_weight ∈ {0.5, 0.7, 0.8, 0.9}   (thr=0.0, T=1.0)
#   Stage B: thr ∈ {0.0, 0.05, 0.1}                (sw=best from A)
#   Stage C: T ∈ {0.5, 1.0, 2.0}                   (sw, thr = best A+B)
#
# Total runs: 4 + 3 + 3 = 10  (~5–6 h on a single GPU)
#
# Results : results/paper_experiments/hp_tune/CIFAR100/full_*
# Usage   : bash scripts/paper_experiments/06_full_hp_tune_cifar100.sh
# ==============================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

OUT_BASE="results/paper_experiments/hp_tune/CIFAR100"
CFG="configs/dcfcl_cifar100.yaml"
SEED=42

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
_log "=== START 06_full_hp_tune_cifar100.sh ==="

run_full() {
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

# ──────────────────────────────────────────────────────────────
# CIFAR-100 single run ≈ 12 h on a single GPU, so we restrict
# to 4 highest-value configs.  The principal axis of improvement
# is directed_self_weight (controls how much a client trusts its
# own model vs peers).  Current sw=0.5 destabilises DER++ on the
# harder CIFAR-100 tasks; we sweep up to 0.9.
# ──────────────────────────────────────────────────────────────
run_full 0.7 0.0  1.0 "sw0.7_thr0.00_temp1.0"
run_full 0.8 0.0  1.0 "sw0.8_thr0.00_temp1.0"
run_full 0.8 0.05 1.0 "sw0.8_thr0.05_temp1.0"
run_full 0.9 0.05 1.0 "sw0.9_thr0.05_temp1.0"

_log "=== END 06_full_hp_tune_cifar100.sh ==="
