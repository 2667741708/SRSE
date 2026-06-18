#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
PY="${PY:-/home/c201/miniconda3/envs/torch_cuda128_whm/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/c201/公共/whm/PALS-SOFT/自适应LSR草稿_v3ref_only_20260409}"
HUB_ROOT="${HUB_ROOT:-/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/experiments/nse_reproducibility}"
SCRIPT="${SCRIPT:-${HUB_ROOT}/NSE_persistent.py}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
OUT="${OUT:-/home/c201/公共/whm/PALS-SOFT/双视图单视图实验结果/results/nse_persistent_state_v2}"
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

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

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

run_exp "FREDIS_RefineDisamb" \
  --source_update_mode fredis_move \
  --source_update_scope all \
  --fredis_top_non_candidate_only \
  --fredis_refine_threshold 0.05 \
  --fredis_refine_min_conf 0.85 \
  --fredis_disamb_threshold 0.85 \
  --fredis_disamb_max_conf 0.05 \
  --fredis_min_disamb_over_refine 2.0

run_exp "IRNet_ScoreCorrection" \
  --source_update_mode irnet_correct \
  --source_update_scope all \
  --irnet_tau_boundary 0.0 \
  --irnet_min_non_candidate_conf 0.85

run_exp "PALS_SARI_LabelAugment" \
  --source_update_mode pals_augment \
  --source_update_scope all_highconf \
  --source_update_schedule linear \
  --source_update_threshold_start 0.95 \
  --source_update_threshold_end 0.85

echo "[launcher] completed v2 gpu${GPU_ID} suite" | tee -a "${LAUNCH_LOG}"
