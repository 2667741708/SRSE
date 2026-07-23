#!/usr/bin/env bash
set -euo pipefail

# CIFAR-100 SRSE main-table launcher.
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
  bash reproduce_srse_cifar100_experiments.sh <target>

Targets:
  q001_eta00      CIFAR-100 q=0.01 eta=0.0, seeds 1/2/3.
  q001_eta01      CIFAR-100 q=0.01 eta=0.1, seeds 1/2/3.
  q001_eta02      CIFAR-100 q=0.01 eta=0.2, seeds 1/2/3.
  q001_eta03      CIFAR-100 q=0.01 eta=0.3, seeds 1/2/3.
  q003_eta01      CIFAR-100 q=0.03 eta=0.1, seeds 1/2/3.
  q003_eta02      CIFAR-100 q=0.03 eta=0.2, seeds 1/2/3.
  q003_eta03      CIFAR-100 q=0.03 eta=0.3, seeds 1/2/3.
  q005_eta00      CIFAR-100 q=0.05 eta=0.0, seeds 1/2/3.
  q005_eta01      CIFAR-100 q=0.05 eta=0.1, seeds 1/2/3.
  q005_eta02      CIFAR-100 q=0.05 eta=0.2, seeds 1/2/3.
  q005_eta03      CIFAR-100 q=0.05 eta=0.3, seeds 1/2/3.
  q005_eta04      CIFAR-100 q=0.05 eta=0.4, seeds 1/2/3.
  q005_eta05      CIFAR-100 q=0.05 eta=0.5, seeds 1/2/3.
  q01_eta00       CIFAR-100 q=0.1 eta=0.0, seeds 1/2/3.
  main_table      Run all CIFAR-100 main-table SRSE conditions.
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
    --dataset CIFAR100 \
    "${common_cifar_args[@]}" \
    --pr "${pr}" \
    --nr "${nr}" \
    --seeds "${SEED_ARGS[@]}" \
    --exp_name "main_table/${exp}"
}

run_main_table() {
  run_condition 0.01 0.0 c100_pr001_nr00_srse_e500_seed123
  run_condition 0.01 0.1 c100_pr001_nr01_srse_e500_seed123
  run_condition 0.01 0.2 c100_pr001_nr02_srse_e500_seed123
  run_condition 0.01 0.3 c100_pr001_nr03_srse_e500_seed123
  run_condition 0.03 0.1 c100_pr003_nr01_srse_e500_seed123
  run_condition 0.03 0.2 c100_pr003_nr02_srse_e500_seed123
  run_condition 0.03 0.3 c100_pr003_nr03_srse_e500_seed123
  run_condition 0.05 0.0 c100_pr005_nr00_srse_e500_seed123
  run_condition 0.05 0.1 c100_pr005_nr01_srse_e500_seed123
  run_condition 0.05 0.2 c100_pr005_nr02_srse_e500_seed123
  run_condition 0.05 0.3 c100_pr005_nr03_srse_e500_seed123
  run_condition 0.05 0.4 c100_pr005_nr04_srse_e500_seed123
  run_condition 0.05 0.5 c100_pr005_nr05_srse_e500_seed123
  run_condition 0.1 0.0 c100_pr01_nr00_srse_e500_seed123
}

target="${1:-help}"
case "${target}" in
  q001_eta00) run_condition 0.01 0.0 c100_pr001_nr00_srse_e500_seed123 ;;
  q001_eta01) run_condition 0.01 0.1 c100_pr001_nr01_srse_e500_seed123 ;;
  q001_eta02) run_condition 0.01 0.2 c100_pr001_nr02_srse_e500_seed123 ;;
  q001_eta03) run_condition 0.01 0.3 c100_pr001_nr03_srse_e500_seed123 ;;
  q003_eta01) run_condition 0.03 0.1 c100_pr003_nr01_srse_e500_seed123 ;;
  q003_eta02) run_condition 0.03 0.2 c100_pr003_nr02_srse_e500_seed123 ;;
  q003_eta03) run_condition 0.03 0.3 c100_pr003_nr03_srse_e500_seed123 ;;
  q005_eta00) run_condition 0.05 0.0 c100_pr005_nr00_srse_e500_seed123 ;;
  q005_eta01) run_condition 0.05 0.1 c100_pr005_nr01_srse_e500_seed123 ;;
  q005_eta02) run_condition 0.05 0.2 c100_pr005_nr02_srse_e500_seed123 ;;
  q005_eta03|main_c100_eta03) run_condition 0.05 0.3 c100_pr005_nr03_srse_e500_seed123 ;;
  q005_eta04) run_condition 0.05 0.4 c100_pr005_nr04_srse_e500_seed123 ;;
  q005_eta05) run_condition 0.05 0.5 c100_pr005_nr05_srse_e500_seed123 ;;
  q01_eta00) run_condition 0.1 0.0 c100_pr01_nr00_srse_e500_seed123 ;;
  main_table|all) run_main_table ;;
  help|-h|--help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
