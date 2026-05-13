#!/usr/bin/env bash
# ==============================================================
# 03_ablation_emnist.sh
# Purpose : CollEdge 全模块消融实验（EMNIST-Letters）
#
# 消融矩阵（7 组实验）：
# ┌─────────────────────────────┬───────┬──────────┬──────┐
# │ 实验                        │ DER++ │ 有向协作 │ Mask │
# ├─────────────────────────────┼───────┼──────────┼──────┤
# │ A_DCFCL_baseline            │  ✗   │    ✗    │  ✗  │
# │ B_DER_only                  │  ✓   │    ✗    │  ✗  │
# │ C_DER_Directed              │  ✓   │    ✓    │  ✗  │
# │ D_DER_Mask                  │  ✓   │    ✗    │  ✓  │
# │ E_DER_Dir_Mask_Full (提案)   │  ✓   │    ✓    │  ✓  │
# │ F_noDER_Directed            │  ✗   │    ✓    │  ✗  │
# │ G_noDER_Mask                │  ✗   │    ✗    │  ✓  │
# └─────────────────────────────┴───────┴──────────┴──────┘
#
# 说明：
#   - 以 configs/dcfcl_emnist.yaml 为基础配置
#   - DER++ 公共参数: buffer_size=500, der_alpha=0.5, der_beta=0.5
#   - 有向协作公共参数: directed_mode=gradient, directed_self_weight=0.5
#   - Mask 公共参数: coalition_mask_type=group, num_client_groups=2
#
# Results : results/paper_experiments/ablation/EMNIST-Letters/
# Usage   : bash scripts/paper_experiments/03_ablation_emnist.sh
# ==============================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

ABL_BASE="results/paper_experiments/ablation/EMNIST-Letters"
CFG="configs/dcfcl_emnist.yaml"

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

_log "=== START 03_ablation_emnist.sh ==="

# ──────────────────────────────────────────────────────────────
# [A/7] DCFCL baseline（原始 DCFCL，无 DER++/有向/Mask）
# ──────────────────────────────────────────────────────────────
mkdir -p "$ABL_BASE/A_DCFCL_baseline"
_log ">>> [A/7] DCFCL_baseline"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm DCFCL \
    --result_dir "$ABL_BASE/A_DCFCL_baseline" \
&& _log "<<< [A/7] DCFCL_baseline DONE" || _log "<<< [A/7] DCFCL_baseline FAILED"

# ──────────────────────────────────────────────────────────────
# [B/7] CollEdge-base（仅 DER++，无有向协作，无 Mask）
# ──────────────────────────────────────────────────────────────
mkdir -p "$ABL_BASE/B_DER_only"
_log ">>> [B/7] DER_only"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm CollEdge \
    --use_der \
    --buffer_size 500 --der_alpha 0.5 --der_beta 0.5 \
    --result_dir "$ABL_BASE/B_DER_only" \
&& _log "<<< [B/7] DER_only DONE" || _log "<<< [B/7] DER_only FAILED"

# ──────────────────────────────────────────────────────────────
# [C/7] CollEdge + DER++ + 有向协作（无 Mask）
# ──────────────────────────────────────────────────────────────
mkdir -p "$ABL_BASE/C_DER_Directed"
_log ">>> [C/7] DER_Directed"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm CollEdge \
    --use_der \
    --buffer_size 500 --der_alpha 0.5 --der_beta 0.5 \
    --directed_collaboration \
    --directed_mode gradient \
    --directed_temperature 1.0 \
    --directed_self_weight 0.5 \
    --directed_threshold 0.0 \
    --result_dir "$ABL_BASE/C_DER_Directed" \
&& _log "<<< [C/7] DER_Directed DONE" || _log "<<< [C/7] DER_Directed FAILED"

# ──────────────────────────────────────────────────────────────
# [D/7] CollEdge + DER++ + Mask（无有向协作）
# ──────────────────────────────────────────────────────────────
mkdir -p "$ABL_BASE/D_DER_Mask"
_log ">>> [D/7] DER_Mask"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm CollEdge \
    --use_der \
    --buffer_size 500 --der_alpha 0.5 --der_beta 0.5 \
    --use_coalition_mask \
    --coalition_mask_type group \
    --num_client_groups 2 \
    --result_dir "$ABL_BASE/D_DER_Mask" \
&& _log "<<< [D/7] DER_Mask DONE" || _log "<<< [D/7] DER_Mask FAILED"

# ──────────────────────────────────────────────────────────────
# [E/7] CollEdge Full（DER++ + 有向协作 + Mask，完整提案方法）
# ──────────────────────────────────────────────────────────────
mkdir -p "$ABL_BASE/E_DER_Dir_Mask_Full"
_log ">>> [E/7] DER_Dir_Mask_Full"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm CollEdge \
    --use_der \
    --buffer_size 500 --der_alpha 0.5 --der_beta 0.5 \
    --directed_collaboration \
    --directed_mode gradient \
    --directed_temperature 1.0 \
    --directed_self_weight 0.5 \
    --directed_threshold 0.0 \
    --use_coalition_mask \
    --coalition_mask_type group \
    --num_client_groups 2 \
    --result_dir "$ABL_BASE/E_DER_Dir_Mask_Full" \
&& _log "<<< [E/7] DER_Dir_Mask_Full DONE" || _log "<<< [E/7] DER_Dir_Mask_Full FAILED"

# ──────────────────────────────────────────────────────────────
# [F/7] CollEdge + 有向协作（无 DER++，无 Mask）
# ──────────────────────────────────────────────────────────────
mkdir -p "$ABL_BASE/F_noDER_Directed"
_log ">>> [F/7] noDER_Directed"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm CollEdge \
    --no_use_der \
    --directed_collaboration \
    --directed_mode gradient \
    --directed_temperature 1.0 \
    --directed_self_weight 0.5 \
    --directed_threshold 0.0 \
    --result_dir "$ABL_BASE/F_noDER_Directed" \
&& _log "<<< [F/7] noDER_Directed DONE" || _log "<<< [F/7] noDER_Directed FAILED"

# ──────────────────────────────────────────────────────────────
# [G/7] CollEdge + Mask（无 DER++，无有向协作）
# ──────────────────────────────────────────────────────────────
mkdir -p "$ABL_BASE/G_noDER_Mask"
_log ">>> [G/7] noDER_Mask"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm CollEdge \
    --no_use_der \
    --use_coalition_mask \
    --coalition_mask_type group \
    --num_client_groups 2 \
    --result_dir "$ABL_BASE/G_noDER_Mask" \
&& _log "<<< [G/7] noDER_Mask DONE" || _log "<<< [G/7] noDER_Mask FAILED"

_log "=== END 03_ablation_emnist.sh ==="
