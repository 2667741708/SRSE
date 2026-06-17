#!/usr/bin/env bash
set -euo pipefail

# CIFAR100H SRSE main-table launcher.
# CIFAR100H uses CIFAR-100 images with hierarchical candidate-label generation.

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
OUT="${OUT:-${ROOT}/out_ultimate/reproduce_srse}"
GPU_ID="${GPU_ID:-0}"
MAIN_SCRIPT="${MAIN_SCRIPT:-${ROOT}/reproducibility/code/main/main.py}"
SEEDS="${SEEDS:-1 2 3}"
read -r -a SEED_ARGS <<< "${SEEDS}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

usage() {
  cat <<'USAGE'
Usage:
  bash reproduce_srse_cifar100h_experiments.sh <target>

Targets:
  q05_eta02       CIFAR100H q=0.5 eta=0.2, seeds 1/2/3.
  main_table      Run all CIFAR100H main-table SRSE conditions.
  all             Alias for main_table.
  help            Print this help.

Runtime overrides:
  PROJECT_ROOT, PY, DATA_ROOT, OUT, GPU_ID, MAIN_SCRIPT, SEEDS.
USAGE
}

run_py() {
  echo
  printf '[run]'
  printf ' %q' "$PY" "$@"
  echo
  "$PY" "$@"
}

common_cifar_args=(
  --train_root "${DATA_ROOT}"
  --lpi 10
  --network R18
  --epochs 500
  --batch_size 256
  --lr 0.1
  --wd 0.001
  --momentum 0.9
  --lr_scheduler cosine
  --mixup_alpha 1.0
  --lsr 0.0
  --consistency_weight 1.0
  --ema_alpha 0.999
  --k_val 15
  --delta 0.25
  --history_len 15
  --consensus_power 2.0
  --sim_mode_1 topology_daes
  --sim_mode_2 topology_daes
  --knn_heads 1
  --topology_rel_gamma 2.0
  --topology_rel_eps 1e-12
  --daes_entropy_coeff 0.5
  --max_w_model 0.5
  --model_warmup_epochs 10
  --out "${OUT}"
  --cuda_dev "${GPU_ID}"
)

run_condition() {
  local pr="$1"
  local nr="$2"
  local exp="$3"
  run_py "${MAIN_SCRIPT}" \
    --dataset CIFAR100H \
    "${common_cifar_args[@]}" \
    --pr "${pr}" \
    --nr "${nr}" \
    --seeds "${SEED_ARGS[@]}" \
    --exp_name "main_table/${exp}"
}

run_main_table() {
  run_condition 0.5 0.2 c100h_pr05_nr02_srse_e500_seed123
}

target="${1:-help}"
case "${target}" in
  q05_eta02) run_condition 0.5 0.2 c100h_pr05_nr02_srse_e500_seed123 ;;
  main_table|all) run_main_table ;;
  help|-h|--help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
