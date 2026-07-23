#!/usr/bin/env bash
set -euo pipefail

# Run a short SRSE validation against reference logs while preserving the full experiment
# horizon in --epochs. The training process stops after MAX_RUN_EPOCHS and the
# first N test-accuracy records are compared against an existing completed log.

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
OUT="${OUT:-${ROOT}/out_ultimate/epoch_test_acc_validation}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${ROOT}/reference_logs}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-1}"
MAX_RUN_EPOCHS="${MAX_RUN_EPOCHS:-2}"
TOLERANCE="${TOLERANCE:-0.01}"
MAX_W_MODEL="${MAX_W_MODEL:-0.5}"
FEATURE_EXTRACT_VIEW="${FEATURE_EXTRACT_VIEW:-weak_strong_fusion}"
TARGET="${1:-main_c100_eta03}"

MAIN_SCRIPT="${MAIN_SCRIPT:-${ROOT}/reproducibility/code/main/main.py}"
ABLATION_SCRIPT="${ABLATION_SCRIPT:-${ROOT}/reproducibility/code/component_ablation/srse_ablation.py}"
PSS_SCRIPT="${PSS_SCRIPT:-${ROOT}/reproducibility/code/persistent_state/srse_persistent.py}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-${ROOT}/reproducibility/code/utils/epoch_test_acc_compare.py}"

export PYTHONPATH="${ROOT}/reproducibility/code:${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

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
  --max_run_epochs "${MAX_RUN_EPOCHS}"
  --batch_size 256
  --lr 0.1
  --wd 0.001
  --momentum 0.9
  --lr_scheduler cosine
  --mixup_alpha 1.0
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
  --max_w_model "${MAX_W_MODEL}"
  --model_warmup_epochs 10
  --feature_extract_view "${FEATURE_EXTRACT_VIEW}"
  --out "${OUT}"
  --cuda_dev "${GPU_ID}"
)

reference_log="${REFERENCE_LOG:-}"
candidate_log=""

case "${TARGET}" in
  main_c100_eta03)
    exp_name="epoch_check/main_c100_eta03"
    default_reference="${REFERENCE_ROOT}/main_c100_eta03/seed_${SEED}/run.log"
    reference_log="${reference_log:-${default_reference}}"
    run_py "${MAIN_SCRIPT}" \
      --dataset CIFAR100 \
      "${common_cifar_args[@]}" \
      --pr 0.05 \
      --nr 0.3 \
      --seeds "${SEED}" \
      --exp_name "${exp_name}"
    candidate_log="${OUT}/${exp_name}/seed_${SEED}/epoch_metrics.csv"
    ;;
  component_a0_c100_eta03)
    exp_name="epoch_check/component_a0_c100_eta03"
    default_reference="${REFERENCE_ROOT}/component_a0_c100_eta03/seed_${SEED}/run.log"
    reference_log="${reference_log:-${default_reference}}"
    run_py "${ABLATION_SCRIPT}" \
      --dataset CIFAR100 \
      "${common_cifar_args[@]}" \
      --pr 0.05 \
      --nr 0.3 \
      --seeds "${SEED}" \
      --exp_name "${exp_name}"
    candidate_log="${OUT}/${exp_name}/seed_${SEED}/run.log"
    ;;
  pss_upllrs_c100_eta03)
    exp_name="epoch_check/pss_upllrs_c100_eta03"
    default_reference="${REFERENCE_ROOT}/pss_upllrs_c100_eta03/seed_${SEED}/run.log"
    reference_log="${reference_log:-${default_reference}}"
    run_py "${PSS_SCRIPT}" \
      --dataset CIFAR100 \
      "${common_cifar_args[@]}" \
      --pr 0.05 \
      --nr 0.3 \
      --seeds "${SEED}" \
      --source_update_evidence p2 \
      --source_update_mode none \
      --source_update_scope none \
      --persistent_promotion_mode hard \
      --promotion_scope nse_estimated_noise_highconf \
      --promotion_threshold 0.95 \
      --promotion_source p2 \
      --promotion_label_space non_candidate \
      --exp_name "${exp_name}"
    candidate_log="${OUT}/${exp_name}/seed_${SEED}/run.log"
    ;;
  compare_only)
    if [[ -z "${REFERENCE_LOG:-}" || -z "${CANDIDATE_LOG:-}" ]]; then
      echo "compare_only requires REFERENCE_LOG and CANDIDATE_LOG" >&2
      exit 2
    fi
    reference_log="${REFERENCE_LOG}"
    candidate_log="${CANDIDATE_LOG}"
    ;;
  *)
    cat >&2 <<'USAGE'
Usage:
  bash verify_epoch_test_acc_against_reference_logs.sh <target>

Targets:
  main_c100_eta03
  component_a0_c100_eta03
  pss_upllrs_c100_eta03
  compare_only

Environment:
  PROJECT_ROOT, PY, DATA_ROOT, OUT, REFERENCE_ROOT, GPU_ID, SEED,
  MAX_RUN_EPOCHS, TOLERANCE, MAX_W_MODEL, FEATURE_EXTRACT_VIEW,
  REFERENCE_LOG, CANDIDATE_LOG.
USAGE
    exit 2
    ;;
esac

echo
echo "[compare] reference=${reference_log}"
echo "[compare] candidate=${candidate_log}"
"${PY}" "${COMPARE_SCRIPT}" compare \
  --reference "${reference_log}" \
  --candidate "${candidate_log}" \
  --epochs "${MAX_RUN_EPOCHS}" \
  --tolerance "${TOLERANCE}"
