#!/usr/bin/env bash
# ==============================================================
# 07_directed_modes_emnist.sh
# Purpose : Compare the three directed-collaboration strategies of
#           CollEdge on EMNIST-Letters:
#             - gradient   : cosine alignment of parameter differentials
#             - task_aware : task-relevance score
#             - hybrid     : 0.5 * gradient + 0.5 * task_aware
#
#           Each strategy is run inside the FULL CollEdge configuration
#           (DER++ + Directed + Mask) so that the only varying factor
#           is `directed_mode`. The numbers feed Tables 1, 4 and Fig. 3.
#
# Total runs: 3
#
# Results : results/paper_experiments/directed_modes/EMNIST-Letters/
# Usage   : bash scripts/paper_experiments/07_directed_modes_emnist.sh
# ==============================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

OUT_BASE="results/paper_experiments/directed_modes/EMNIST-Letters"
CFG="configs/dcfcl_emnist.yaml"
SEED=42

# Default tuned hyper-params (override after hp-tune is finished)
SW=${COLLEDGE_SW:-0.7}
THR=${COLLEDGE_THR:-0.05}
TEMP=${COLLEDGE_TEMP:-1.0}

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
_log "=== START 07_directed_modes_emnist.sh (sw=$SW thr=$THR temp=$TEMP) ==="

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

run_mode gradient
run_mode task_aware
run_mode hybrid

_log "=== END 07_directed_modes_emnist.sh ==="
