#!/usr/bin/env bash
# ==============================================================
# 10_perclient_emnist.sh
# Purpose : Re-run a focused subset of EMNIST algorithms after
#           server.py was extended with per-client per-task
#           accuracy logging (`per_client_diag_acc` /
#           `per_client_final_acc` in results.json) so that the
#           DCFCL-style "client x task" bubble matrix can be
#           plotted.
# Algos   : FedAvg, FedProx, DCFCL, CollEdge (gradient best HP)
# Out     : results/paper_experiments/perclient/EMNIST-Letters/<alg>/
# ==============================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

OUT_BASE="results/paper_experiments/perclient/EMNIST-Letters"
CFG="configs/dcfcl_emnist.yaml"
SEED=42
mkdir -p "$OUT_BASE"

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
_log "=== START 10_perclient_emnist.sh ==="

# ---------- FedAvg ----------
_log ">>> FedAvg"
python main.py --config "$CFG" --seed $SEED \
    --algorithm FedAvg \
    --result_dir "$OUT_BASE/FedAvg" \
&& _log "<<< FedAvg DONE" || _log "<<< FedAvg FAILED"

# ---------- FedProx ----------
_log ">>> FedProx"
python main.py --config "$CFG" --seed $SEED \
    --algorithm FedProx \
    --result_dir "$OUT_BASE/FedProx" \
&& _log "<<< FedProx DONE" || _log "<<< FedProx FAILED"

# ---------- DCFCL baseline ----------
_log ">>> DCFCL"
python main.py --config "$CFG" --seed $SEED \
    --algorithm DCFCL \
    --result_dir "$OUT_BASE/DCFCL" \
&& _log "<<< DCFCL DONE" || _log "<<< DCFCL FAILED"

# ---------- CollEdge (gradient mode, HP-tuned best) ----------
_log ">>> CollEdge"
python main.py --config "$CFG" --seed $SEED \
    --algorithm CollEdge \
    --use_der --buffer_size 500 --der_alpha 0.5 --der_beta 0.5 \
    --directed_collaboration --directed_mode gradient \
    --directed_self_weight 0.8 --directed_threshold 0.0 --directed_temperature 1.0 \
    --use_coalition_mask --coalition_mask_type group --num_client_groups 2 \
    --result_dir "$OUT_BASE/CollEdge" \
&& _log "<<< CollEdge DONE" || _log "<<< CollEdge FAILED"

_log "=== END 10_perclient_emnist.sh ==="
