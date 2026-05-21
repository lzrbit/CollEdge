#!/usr/bin/env bash
# ==============================================================
# 09_ablation_retune_CF_cifar.sh
# Purpose : Re-run CIFAR-100 ablation rows C (DER+Directed) and
#           F (Directed only) using the *Full*-tuned HP, so that
#           every Dir-containing row in the ablation table uses
#           the same directed-aggregation hyper-parameters as the
#           Full configuration.
#
#           A/B/D/G are independent of these HPs and need not be
#           re-run.
#
#           Both runs are launched in PARALLEL on the same GPU
#           (≤12 GB total) to halve wall-clock cost.
#
# Env-var inputs (export before running, or eval pick_best_hp.py):
#   COLLEDGE_SW    — directed_self_weight (default: 0.8)
#   COLLEDGE_THR   — directed_threshold   (default: 0.05)
#   COLLEDGE_TEMP  — directed_temperature (default: 1.0)
#
# Outputs:
#   results/paper_experiments/ablation/CIFAR100/C_DER_Directed_retuned/
#   results/paper_experiments/ablation/CIFAR100/F_noDER_Directed_retuned/
#
# Total runs: 2 in parallel (~1 h on RTX 5080).
# ==============================================================

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

OUT_BASE="results/paper_experiments/ablation/CIFAR100"
CFG="configs/dcfcl_cifar100.yaml"
SEED=42

SW=${COLLEDGE_SW:-0.8}
THR=${COLLEDGE_THR:-0.05}
TEMP=${COLLEDGE_TEMP:-1.0}

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
_log "=== START 09_ablation_retune_CF_cifar.sh (sw=$SW thr=$THR temp=$TEMP) ==="

# ──────────────────────────────────────────────────────────────
# C_DER_Directed_retuned : DER++ + Directed (no Mask) with Full HP
# ──────────────────────────────────────────────────────────────
OUT_C="$OUT_BASE/C_DER_Directed_retuned"
mkdir -p "$OUT_C"
_log ">>> C_DER_Directed_retuned (async)"
python main.py \
    --config "$CFG" --seed $SEED \
    --algorithm CollEdge \
    --use_der \
    --buffer_size 500 --der_alpha 0.5 --der_beta 0.5 \
    --directed_collaboration \
    --directed_mode gradient \
    --directed_self_weight $SW \
    --directed_threshold $THR \
    --directed_temperature $TEMP \
    --result_dir "$OUT_C" \
&& _log "<<< C_DER_Directed_retuned DONE" || _log "<<< C_DER_Directed_retuned FAILED" &
PID_C=$!

# ──────────────────────────────────────────────────────────────
# F_noDER_Directed_retuned : Directed only (no DER, no Mask) with Full HP
# ──────────────────────────────────────────────────────────────
OUT_F="$OUT_BASE/F_noDER_Directed_retuned"
mkdir -p "$OUT_F"
_log ">>> F_noDER_Directed_retuned (async)"
python main.py \
    --config "$CFG" --seed $SEED \
    --algorithm CollEdge \
    --no_use_der \
    --directed_collaboration \
    --directed_mode gradient \
    --directed_self_weight $SW \
    --directed_threshold $THR \
    --directed_temperature $TEMP \
    --result_dir "$OUT_F" \
&& _log "<<< F_noDER_Directed_retuned DONE" || _log "<<< F_noDER_Directed_retuned FAILED" &
PID_F=$!

wait $PID_C; wait $PID_F
_log "=== END 09_ablation_retune_CF_cifar.sh ==="
