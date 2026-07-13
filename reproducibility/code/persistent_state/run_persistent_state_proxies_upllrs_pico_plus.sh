#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CODE_ROOT="${CODE_ROOT:-${ROOT}/reproducibility/code}"
GPU_ID="${1:-1}"
PY="${PY:-python}"
SCRIPT="${SCRIPT:-${ROOT}/reproducibility/code/persistent_state/srse_persistent.py}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
OUT="${OUT:-${ROOT}/out_ultimate/srse_persistent_state_v2}"
LAUNCH_LOG_DIR="${OUT}/_launcher_logs"
mkdir -p "${LAUNCH_LOG_DIR}"
LAUNCH_NAME="${LAUNCH_NAME:-$(basename "$0" .sh)}"
LAUNCH_LOG="${LAUNCH_LOG_DIR}/${LAUNCH_NAME}.log"

wait_for_gpu() {
  local threshold="${GPU_WAIT_MEM_MB:-1500}"
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" | awk '{print $1}')"
    if [ "${used}" -le "${threshold}" ]; then
      echo "[launcher] gpu ${GPU_ID} memory ${used}MiB <= ${threshold}MiB; starting" | tee -a "${LAUNCH_LOG}"
      break
    fi
    echo "[launcher] waiting for gpu ${GPU_ID}: memory ${used}MiB > ${threshold}MiB" | tee -a "${LAUNCH_LOG}"
    sleep 300
  done
}

cd "${ROOT}"
export PYTHONPATH="${CODE_ROOT}:${ROOT}:${PYTHONPATH:-}"

BASE_ARGS=(
  --dataset CIFAR100
  --train_root "${DATA_ROOT}"
  --network R18
  --epochs 500
  --batch_size 256
  --lr 0.1
  --wd 0.001
  --momentum 0.9
  --lr_scheduler cosine
  --mixup_alpha 1.0
  --lsr 0.0
  --k_val 15
  --delta 0.25
  --history_len 15
  --sim_mode_1 topology_daes
  --sim_mode_2 topology_daes
  --knn_heads 1
  --topology_rel_gamma 2.0
  --daes_entropy_coeff 0.5
  --max_w_model 0.5
  --model_warmup_epochs 10
  --out "${OUT}"
  --seeds 1 2 3
  --cuda_dev "${GPU_ID}"
  --pr 0.05
  --nr 0.3
  --source_update_evidence p2
)

is_done() {
  local exp="$1"
  local log="${OUT}/${exp}/master_log.txt"
  [ -f "${log}" ] && grep -q -- "--- Run 3 Finished" "${log}"
}

run_exp() {
  local exp="$1"
  shift
  if is_done "${exp}"; then
    echo "[launcher] skip completed ${exp}" | tee -a "${LAUNCH_LOG}"
    return 0
  fi
  echo "[launcher] start ${exp}" | tee -a "${LAUNCH_LOG}"
  echo "[launcher] script=${SCRIPT}" | tee -a "${LAUNCH_LOG}"
  echo "[launcher] args: ${BASE_ARGS[*]} --exp_name ${exp} $*" | tee -a "${LAUNCH_LOG}"
  "${PY}" "${SCRIPT}" "${BASE_ARGS[@]}" --exp_name "${exp}" "$@" 2>&1 | tee -a "${LAUNCH_LOG}"
}

wait_for_gpu

run_exp "UPLLRS_ReliablePromotion" \
  --source_update_mode none \
  --source_update_scope none \
  --persistent_promotion_mode hard \
  --promotion_scope nse_estimated_noise_highconf \
  --promotion_threshold 0.95 \
  --promotion_source p2 \
  --promotion_label_space non_candidate

run_exp "PiCOPlus_SoftTargetCarryover" \
  --source_update_mode pico_soft_target \
  --source_update_scope all \
  --source_update_alpha 0.1

echo "[launcher] completed v2 gpu${GPU_ID} suite" | tee -a "${LAUNCH_LOG}"
