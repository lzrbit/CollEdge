#!/usr/bin/env bash
# ==============================================================
# 02_baselines_cifar100.sh
# Purpose : 在 CIFAR-100 上运行所有 baseline 算法
# Algorithms: Local / FedAvg / FedProx / FedLwF / SCAFFOLD
#             PerAvg / pFedMe / ClusterFL  (共 8 个)
# Results : results/paper_experiments/baselines/CIFAR100/
# Usage   : bash scripts/paper_experiments/02_baselines_cifar100.sh
# ==============================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

RESULT_BASE="results/paper_experiments/baselines/CIFAR100"
CFG="configs/dcfcl_cifar100.yaml"
mkdir -p "$RESULT_BASE"

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

_log "=== START 02_baselines_cifar100.sh ==="

# ---------- [1/8] Local ----------
_log ">>> [1/8] Local"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm Local \
    --result_dir "$RESULT_BASE" \
&& _log "<<< [1/8] Local DONE" || _log "<<< [1/8] Local FAILED"

# ---------- [2/8] FedAvg ----------
_log ">>> [2/8] FedAvg"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm FedAvg \
    --result_dir "$RESULT_BASE" \
&& _log "<<< [2/8] FedAvg DONE" || _log "<<< [2/8] FedAvg FAILED"

# ---------- [3/8] FedProx ----------
_log ">>> [3/8] FedProx"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm FedProx \
    --result_dir "$RESULT_BASE" \
&& _log "<<< [3/8] FedProx DONE" || _log "<<< [3/8] FedProx FAILED"

# ---------- [4/8] FedLwF ----------
_log ">>> [4/8] FedLwF"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm FedLwF \
    --result_dir "$RESULT_BASE" \
&& _log "<<< [4/8] FedLwF DONE" || _log "<<< [4/8] FedLwF FAILED"

# ---------- [5/8] SCAFFOLD ----------
_log ">>> [5/8] SCAFFOLD"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm SCAFFOLD \
    --result_dir "$RESULT_BASE" \
&& _log "<<< [5/8] SCAFFOLD DONE" || _log "<<< [5/8] SCAFFOLD FAILED"

# ---------- [6/8] PerAvg ----------
_log ">>> [6/8] PerAvg"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm PerAvg \
    --result_dir "$RESULT_BASE" \
&& _log "<<< [6/8] PerAvg DONE" || _log "<<< [6/8] PerAvg FAILED"

# ---------- [7/8] pFedMe ----------
_log ">>> [7/8] pFedMe"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm pFedMe \
    --result_dir "$RESULT_BASE" \
&& _log "<<< [7/8] pFedMe DONE" || _log "<<< [7/8] pFedMe FAILED"

# ---------- [8/8] ClusterFL ----------
_log ">>> [8/8] ClusterFL"
python main.py \
    --config "$CFG" --seed 42 \
    --algorithm ClusterFL \
    --result_dir "$RESULT_BASE" \
&& _log "<<< [8/8] ClusterFL DONE" || _log "<<< [8/8] ClusterFL FAILED"

_log "=== END 02_baselines_cifar100.sh ==="
