#!/usr/bin/env bash
set -euo pipefail

# CIFAR-10 SRSE main-table launcher.
# Runs the public SRSE method only; baseline-comparison scripts are local-only.

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
OUT="${OUT:-${ROOT}/out_ultimate/reproduce_srse}"
GPU_ID="${GPU_ID:-0}"
MAIN_SCRIPT="${MAIN_SCRIPT:-${ROOT}/reproducibility/code/main/main.py}"
SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CIFAR_DOWNLOAD="${CIFAR_DOWNLOAD:-0}"
read -r -a SEED_ARGS <<< "${SEEDS}"

download_args=()
case "${CIFAR_DOWNLOAD}" in
  1|true|TRUE|yes|YES) download_args+=(--download) ;;
esac

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

usage() {
  cat <<'USAGE'
Usage:
  bash reproduce_srse_cifar10_experiments.sh <target>

Targets:
  q01_eta01       CIFAR-10 q=0.1 eta=0.1, seeds 1/2/3.
  q01_eta02       CIFAR-10 q=0.1 eta=0.2, seeds 1/2/3.
  q01_eta03       CIFAR-10 q=0.1 eta=0.3, seeds 1/2/3.
  q03_eta01       CIFAR-10 q=0.3 eta=0.1, seeds 1/2/3.
  q03_eta02       CIFAR-10 q=0.3 eta=0.2, seeds 1/2/3.
  q03_eta03       CIFAR-10 q=0.3 eta=0.3, seeds 1/2/3.
  q05_eta01       CIFAR-10 q=0.5 eta=0.1, seeds 1/2/3.
  q05_eta02       CIFAR-10 q=0.5 eta=0.2, seeds 1/2/3.
  q05_eta03       CIFAR-10 q=0.5 eta=0.3, seeds 1/2/3.
  main_table      Run all CIFAR-10 main-table SRSE conditions.
  all             Alias for main_table.
  help            Print this help.

Runtime overrides:
  PROJECT_ROOT, PY, DATA_ROOT, OUT, GPU_ID, MAIN_SCRIPT, SEEDS, EPOCHS,
  BATCH_SIZE, NUM_WORKERS, CIFAR_DOWNLOAD.
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
  "${download_args[@]}"
  --lpi 10
  --network R18
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr 0.1
  --wd 0.001
  --momentum 0.9
  --lr_scheduler cosine
  --mixup_alpha 1.0
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
    --dataset CIFAR10 \
    "${common_cifar_args[@]}" \
    --pr "${pr}" \
    --nr "${nr}" \
    --seeds "${SEED_ARGS[@]}" \
    --exp_name "main_table/${exp}"
}

run_main_table() {
  run_condition 0.1 0.1 c10_pr01_nr01_srse_e500_seed123
  run_condition 0.1 0.2 c10_pr01_nr02_srse_e500_seed123
  run_condition 0.1 0.3 c10_pr01_nr03_srse_e500_seed123
  run_condition 0.3 0.1 c10_pr03_nr01_srse_e500_seed123
  run_condition 0.3 0.2 c10_pr03_nr02_srse_e500_seed123
  run_condition 0.3 0.3 c10_pr03_nr03_srse_e500_seed123
  run_condition 0.5 0.1 c10_pr05_nr01_srse_e500_seed123
  run_condition 0.5 0.2 c10_pr05_nr02_srse_e500_seed123
  run_condition 0.5 0.3 c10_pr05_nr03_srse_e500_seed123
}

target="${1:-help}"
case "${target}" in
  q01_eta01) run_condition 0.1 0.1 c10_pr01_nr01_srse_e500_seed123 ;;
  q01_eta02) run_condition 0.1 0.2 c10_pr01_nr02_srse_e500_seed123 ;;
  q01_eta03) run_condition 0.1 0.3 c10_pr01_nr03_srse_e500_seed123 ;;
  q03_eta01) run_condition 0.3 0.1 c10_pr03_nr01_srse_e500_seed123 ;;
  q03_eta02) run_condition 0.3 0.2 c10_pr03_nr02_srse_e500_seed123 ;;
  q03_eta03) run_condition 0.3 0.3 c10_pr03_nr03_srse_e500_seed123 ;;
  q05_eta01) run_condition 0.5 0.1 c10_pr05_nr01_srse_e500_seed123 ;;
  q05_eta02) run_condition 0.5 0.2 c10_pr05_nr02_srse_e500_seed123 ;;
  q05_eta03) run_condition 0.5 0.3 c10_pr05_nr03_srse_e500_seed123 ;;
  main_table|all) run_main_table ;;
  help|-h|--help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
