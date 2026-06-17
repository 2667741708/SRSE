#!/usr/bin/env bash
set -euo pipefail

# Fast executable checks for public SRSE entry points.
# These runs are intentionally short and are not paper-result reproductions.

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
CROWD_ROOT="${CROWD_ROOT:-${ROOT}}"
OUT="${OUT:-${ROOT}/out_ultimate/smoke_srse_entrypoints}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-1}"
CIFAR_BATCH_SIZE="${CIFAR_BATCH_SIZE:-128}"
CROWD_BATCH_SIZE="${CROWD_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CIFAR_DOWNLOAD="${CIFAR_DOWNLOAD:-0}"

MAIN_SCRIPT="${MAIN_SCRIPT:-${ROOT}/reproducibility/code/main/main.py}"
ABLATION_SCRIPT="${ABLATION_SCRIPT:-${ROOT}/reproducibility/code/component_ablation/srse_ablation.py}"
PSS_SCRIPT="${PSS_SCRIPT:-${ROOT}/reproducibility/code/persistent_state/srse_persistent.py}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

download_args=()
case "${CIFAR_DOWNLOAD}" in
  1|true|TRUE|yes|YES) download_args+=(--download) ;;
esac

usage() {
  cat <<'USAGE'
Usage:
  bash smoke_test_srse_entrypoints.sh <target>

Targets:
  cifar10        One-epoch CIFAR-10 SRSE main-entry smoke.
  cifar100       One-epoch CIFAR-100 SRSE main-entry smoke.
  cifar100h      One-epoch CIFAR100H SRSE main-entry smoke.
  cifar          Run cifar10, cifar100, and cifar100h.
  benthic        One-epoch Benthic crowd-dataset smoke.
  plankton       One-epoch Plankton crowd-dataset smoke.
  treeversity    One-epoch Treeversity crowd-dataset smoke.
  crowd          Run benthic, plankton, and treeversity.
  topology       One-epoch Topology-DAES ablation-entry smoke.
  component      One-epoch component-ablation flag smoke.
  pss            One-epoch persistent-supervision-state entry smoke.
  no_reg         One-epoch no-CR/no-MixUp fairness-row smoke.
  ablations      Run topology, component, pss, and no_reg.
  all            Run cifar, crowd, and ablations.
  help           Print this help.

Runtime overrides:
  PROJECT_ROOT, PY, DATA_ROOT, CROWD_ROOT, OUT, GPU_ID, SEED, EPOCHS,
  CIFAR_BATCH_SIZE, CROWD_BATCH_SIZE, NUM_WORKERS, CIFAR_DOWNLOAD.

Notes:
  Smoke runs only prove that public data loaders, command-line arguments, and
  training loops execute. Use reproduce_srse_paper_experiments.sh for full
  500-epoch paper reproductions.
USAGE
}

run_py() {
  echo
  printf '[smoke]'
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
  --batch_size "${CIFAR_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr 0.1
  --wd 0.001
  --momentum 0.9
  --lr_scheduler cosine
  --mixup_alpha 1.0
  --lsr 0.0
  --ema_alpha 0.999
  --k_val 15
  --delta 0.25
  --history_len 15
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
  --seeds "${SEED}"
)

run_cifar_main() {
  local dataset="$1"
  local pr="$2"
  local nr="$3"
  local exp="$4"
  run_py "${MAIN_SCRIPT}" \
    --dataset "${dataset}" \
    "${common_cifar_args[@]}" \
    --pr "${pr}" \
    --nr "${nr}" \
    --exp_name "${exp}"
}

run_cifar() {
  run_cifar_main CIFAR10 0.1 0.1 smoke/cifar10_q01_eta01
  run_cifar_main CIFAR100 0.05 0.3 smoke/cifar100_q005_eta03
  run_cifar_main CIFAR100H 0.5 0.2 smoke/cifar100h_q05_eta02
}

resolve_crowd_root() {
  local dataset="$1"
  local root="${CROWD_ROOT}/${dataset}"
  if [[ "${dataset}" == "Treeversity" && ! -f "${root}/annotations.json" && -f "${CROWD_ROOT}/Treeversity#6/annotations.json" ]]; then
    root="${CROWD_ROOT}/Treeversity#6"
  fi
  printf '%s' "${root}"
}

run_crowd_main() {
  local dataset="$1"
  local lpi="$2"
  local protocol="$3"
  local root
  root="$(resolve_crowd_root "${dataset}")"
  run_py "${MAIN_SCRIPT}" \
    --dataset "${dataset}" \
    --train_root "${root}" \
    --lpi "${lpi}" \
    --slice 2 \
    --split_protocol "${protocol}" \
    --pr 0.05 \
    --nr 0.5 \
    --network R50 \
    --epochs "${EPOCHS}" \
    --batch_size "${CROWD_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --lr 0.05 \
    --wd 0.0005 \
    --momentum 0.9 \
    --lr_scheduler step \
    --lr_decay_epochs 60 80 \
    --lr_decay_rate 0.2 \
    --mixup_alpha 1.0 \
    --lsr 0.0 \
    --ema_alpha 0.999 \
    --k_val 5 \
    --delta 1.0 \
    --history_len 15 \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --knn_heads 1 \
    --topology_rel_gamma 2.0 \
    --topology_rel_eps 1e-12 \
    --daes_entropy_coeff 0.5 \
    --max_w_model 0.1 \
    --model_warmup_epochs 10 \
    --cuda_dev "${GPU_ID}" \
    --seeds "${SEED}" \
    --out "${OUT}" \
    --exp_name "smoke/${dataset}_lpi${lpi}_${protocol}"
}

run_crowd() {
  run_crowd_main Benthic 3 pals_3fold
  run_crowd_main Plankton 10 standard
  run_crowd_main Treeversity 10 pals_3fold
}

run_topology() {
  run_py "${ABLATION_SCRIPT}" \
    --dataset CIFAR100 \
    "${common_cifar_args[@]}" \
    --pr 0.05 \
    --nr 0.3 \
    --exp_name smoke/topology_daes_full
}

run_component() {
  run_py "${ABLATION_SCRIPT}" \
    --dataset CIFAR100 \
    "${common_cifar_args[@]}" \
    --pr 0.05 \
    --nr 0.3 \
    --ablate_no_candidate_prior \
    --ablate_no_reliable_mixup \
    --exp_name smoke/component_no_candidate_prior_no_mixup
}

run_pss() {
  run_py "${PSS_SCRIPT}" \
    --dataset CIFAR100 \
    "${common_cifar_args[@]}" \
    --pr 0.05 \
    --nr 0.3 \
    --source_update_mode pico_soft_target \
    --source_update_scope all \
    --source_update_alpha 0.1 \
    --exp_name smoke/pss_pico_soft_target
}

run_no_reg() {
  run_py "${ABLATION_SCRIPT}" \
    --dataset CIFAR10 \
    "${common_cifar_args[@]}" \
    --pr 0.1 \
    --nr 0.3 \
    --ablate_no_cr \
    --ablate_no_reliable_mixup \
    --exp_name smoke/no_reg_cifar10_q01_eta03
}

run_ablations() {
  run_topology
  run_component
  run_pss
  run_no_reg
}

target="${1:-help}"
case "${target}" in
  cifar10) run_cifar_main CIFAR10 0.1 0.1 smoke/cifar10_q01_eta01 ;;
  cifar100) run_cifar_main CIFAR100 0.05 0.3 smoke/cifar100_q005_eta03 ;;
  cifar100h) run_cifar_main CIFAR100H 0.5 0.2 smoke/cifar100h_q05_eta02 ;;
  cifar) run_cifar ;;
  benthic) run_crowd_main Benthic 3 pals_3fold ;;
  plankton) run_crowd_main Plankton 10 standard ;;
  treeversity) run_crowd_main Treeversity 10 pals_3fold ;;
  crowd) run_crowd ;;
  topology) run_topology ;;
  component) run_component ;;
  pss) run_pss ;;
  no_reg) run_no_reg ;;
  ablations) run_ablations ;;
  all)
    run_cifar
    run_crowd
    run_ablations
    ;;
  help|-h|--help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
