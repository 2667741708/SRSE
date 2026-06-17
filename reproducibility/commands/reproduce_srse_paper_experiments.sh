#!/usr/bin/env bash
set -euo pipefail

# Unified reproduction launcher for the current SRSE manuscript package.
# This file uses package-local paths and overrideable runtime variables.
#
# Example:
#   PROJECT_ROOT=/path/to/unpacked/repo PY=/path/to/python GPU_ID=0 \
#     bash reproducibility/commands/reproduce_srse_paper_experiments.sh main_c100_eta03

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
CROWD_ROOT="${CROWD_ROOT:-${ROOT}}"
OUT="${OUT:-${ROOT}/out_ultimate/reproduce_srse}"
GPU_ID="${GPU_ID:-0}"

MAIN_SCRIPT="${MAIN_SCRIPT:-${ROOT}/reproducibility/code/main/main.py}"
ABLATION_SCRIPT="${ABLATION_SCRIPT:-${ROOT}/reproducibility/code/component_ablation/srse_ablation.py}"
PSS_SCRIPT="${PSS_SCRIPT:-${ROOT}/reproducibility/code/persistent_state/srse_persistent.py}"
AGG_CROWD_SCRIPT="${AGG_CROWD_SCRIPT:-${ROOT}/reproducibility/code/main/aggregate_srse_main_crowd_results.py}"
CIFAR10_SCRIPT="${CIFAR10_SCRIPT:-${ROOT}/reproducibility/commands/reproduce_srse_cifar10_experiments.sh}"
CIFAR100_SCRIPT="${CIFAR100_SCRIPT:-${ROOT}/reproducibility/commands/reproduce_srse_cifar100_experiments.sh}"
CIFAR100H_SCRIPT="${CIFAR100H_SCRIPT:-${ROOT}/reproducibility/commands/reproduce_srse_cifar100h_experiments.sh}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

usage() {
  cat <<'USAGE'
Usage:
  bash reproduce_srse_paper_experiments.sh <target>

Targets:
  cifar10_table          CIFAR-10 SRSE main-table q/eta grid, seeds 1/2/3.
  cifar100_table         CIFAR-100 SRSE main-table q/eta grid, seeds 1/2/3.
  cifar100h_table        CIFAR100H SRSE main-table condition, seeds 1/2/3.
  cifar_main_tables      Run CIFAR-10, CIFAR-100, and CIFAR100H main tables.
  main_c100_eta03        Main CIFAR-100 q=0.05 eta=0.3 SRSE run, seeds 1/2/3.
  topology_micro_eta03   One-factor Topology-DAES micro-ablation at eta=0.3, seed 1.
  topology_micro_eta04   One-factor Topology-DAES micro-ablation at eta=0.4, seed 1.
  component_table        Legacy/full component-ablation rows A0-A11, seeds 1/2/3.
  pss_table              Persistent supervision-state proxy rows, seeds 1/2/3.
  no_reg_table           CIFAR no-MixUp/no-CR fairness table, seeds 1/2/3.
  crowd_srse_table       SRSE crowdsourced-dataset rows, seeds 1/2/3.
  all                    Run all targets above sequentially.
  help                   Print this help.

Runtime overrides:
  PROJECT_ROOT, PY, DATA_ROOT, CROWD_ROOT, OUT, GPU_ID, MAIN_SCRIPT,
  ABLATION_SCRIPT, PSS_SCRIPT, AGG_CROWD_SCRIPT, CIFAR10_SCRIPT,
  CIFAR100_SCRIPT, CIFAR100H_SCRIPT, SEEDS.
USAGE
}

run_py() {
  echo
  printf '[run]'
  printf ' %q' "$PY" "$@"
  echo
  "$PY" "$@"
}

run_sh() {
  local script="$1"
  shift
  echo
  printf '[run] bash'
  printf ' %q' "${script}" "$@"
  echo
  bash "${script}" "$@"
}

run_cifar10_table() {
  run_sh "${CIFAR10_SCRIPT}" main_table
}

run_cifar100_table() {
  run_sh "${CIFAR100_SCRIPT}" main_table
}

run_cifar100h_table() {
  run_sh "${CIFAR100H_SCRIPT}" main_table
}

run_cifar_main_tables() {
  run_cifar10_table
  run_cifar100_table
  run_cifar100h_table
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
)

run_main_c100_eta03() {
  run_py "${MAIN_SCRIPT}" \
    --dataset CIFAR100 \
    "${common_cifar_args[@]}" \
    --pr 0.05 \
    --nr 0.3 \
    --seeds 1 2 3 \
    --exp_name main_table/c100_pr005_nr03_srse_e500_seed123
}

run_topology_micro_row() {
  local eta="$1"
  local row="$2"
  shift 2
  run_py "${ABLATION_SCRIPT}" \
    --dataset CIFAR100 \
    "${common_cifar_args[@]}" \
    --pr 0.05 \
    --nr "${eta}" \
    --seeds 1 \
    --exp_name "topology_micro_ablation/q005_eta${eta}/${row}" \
    "$@"
}

run_topology_micro() {
  local eta="$1"
  run_topology_micro_row "${eta}" none_full_topology_daes
  run_topology_micro_row "${eta}" no_masked_entropy_reliability \
    --topology_rel_gamma 0.0
  run_topology_micro_row "${eta}" no_adaptive_temperature \
    --daes_entropy_coeff 0.0
  run_topology_micro_row "${eta}" no_similarity_power \
    --daes_sim_power 1.0
  run_topology_micro_row "${eta}" no_two_pass_propagation \
    --sim_mode_2 none
  run_topology_micro_row "${eta}" no_source_gate \
    --ablate_no_candidate_prior
  run_topology_micro_row "${eta}" no_class_balanced_quota \
    --ablate_global_topk_quota
}

run_component_row() {
  local exp="$1"
  shift
  run_py "${ABLATION_SCRIPT}" \
    --dataset CIFAR100 \
    "${common_cifar_args[@]}" \
    --pr 0.05 \
    --nr 0.3 \
    --seeds 1 2 3 \
    --exp_name "component_table/${exp}" \
    "$@"
}

run_component_table() {
  run_component_row A0_full
  run_component_row A1_topology_daes_to_exp_both --sim_mode_1 exp --sim_mode_2 exp
  run_component_row A2_stage2_exp --sim_mode_1 topology_daes --sim_mode_2 exp
  run_component_row A3_stage1_exp --sim_mode_1 exp --sim_mode_2 topology_daes
  run_component_row A4_uniform_ri --ablate_uniform_ri
  run_component_row A5_no_candidate_prior_projection --ablate_no_candidate_prior
  run_component_row A6_knn_only --max_w_model 0.0
  run_component_row A7_model_only --max_w_model 1.0
  run_component_row A8_detect_salvage_only --disable_salvage_training
  run_component_row A9_no_reliable_mixup --ablate_no_reliable_mixup
  run_component_row A10_no_cr_strong_view --ablate_no_cr
  run_component_row A11_no_cr_no_mixup --ablate_no_cr --ablate_no_reliable_mixup
}

run_pss_row() {
  local exp="$1"
  shift
  run_py "${PSS_SCRIPT}" \
    --dataset CIFAR100 \
    --train_root "${DATA_ROOT}" \
    --network R18 \
    --epochs 500 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 0.001 \
    --momentum 0.9 \
    --lr_scheduler cosine \
    --mixup_alpha 1.0 \
    --lsr 0.0 \
    --k_val 15 \
    --delta 0.25 \
    --history_len 15 \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --knn_heads 1 \
    --topology_rel_gamma 2.0 \
    --daes_entropy_coeff 0.5 \
    --max_w_model 0.5 \
    --model_warmup_epochs 20 \
    --out "${OUT}/pss_table" \
    --seeds 1 2 3 \
    --cuda_dev "${GPU_ID}" \
    --pr 0.05 \
    --nr 0.3 \
    --source_update_evidence p2 \
    --exp_name "${exp}" \
    "$@"
}

run_pss_table() {
  run_pss_row FREDIS_RefineDisamb \
    --source_update_mode fredis_move \
    --source_update_scope all \
    --fredis_top_non_candidate_only \
    --fredis_refine_threshold 0.05 \
    --fredis_refine_min_conf 0.85 \
    --fredis_disamb_threshold 0.85 \
    --fredis_disamb_max_conf 0.05 \
    --fredis_min_disamb_over_refine 2.0
  run_pss_row IRNet_ScoreCorrection \
    --source_update_mode irnet_correct \
    --source_update_scope all \
    --irnet_tau_boundary 0.0 \
    --irnet_min_non_candidate_conf 0.85
  run_pss_row PALS_SARI_LabelAugment \
    --source_update_mode pals_augment \
    --source_update_scope all_highconf \
    --source_update_schedule linear \
    --source_update_threshold_start 0.95 \
    --source_update_threshold_end 0.85
  run_pss_row UPLLRS_ReliablePromotion \
    --source_update_mode none \
    --source_update_scope none \
    --persistent_promotion_mode hard \
    --promotion_scope srse_estimated_noise_highconf \
    --promotion_threshold 0.95 \
    --promotion_source p2 \
    --promotion_label_space non_candidate
  run_pss_row PiCOPlus_SoftTargetCarryover \
    --source_update_mode pico_soft_target \
    --source_update_scope all \
    --source_update_alpha 0.1
}

run_no_reg() {
  local dataset="$1"
  local pr="$2"
  local nr="$3"
  local exp="$4"
  run_py "${ABLATION_SCRIPT}" \
    --dataset "${dataset}" \
    "${common_cifar_args[@]}" \
    --pr "${pr}" \
    --nr "${nr}" \
    --seeds 1 2 3 \
    --exp_name "no_cr_no_mixup_table/${exp}" \
    --ablate_no_cr \
    --ablate_no_reliable_mixup
}

run_no_reg_table() {
  run_no_reg CIFAR10 0.1 0.1 CIFAR10_pr01_nr01
  run_no_reg CIFAR10 0.1 0.3 CIFAR10_pr01_nr03
  run_no_reg CIFAR10 0.5 0.1 CIFAR10_pr05_nr01
  run_no_reg CIFAR10 0.5 0.3 CIFAR10_pr05_nr03
  run_no_reg CIFAR100 0.03 0.1 CIFAR100_pr003_nr01
  run_no_reg CIFAR100 0.03 0.3 CIFAR100_pr003_nr03
  run_no_reg CIFAR100 0.05 0.3 CIFAR100_pr005_nr03
  run_no_reg CIFAR100 0.05 0.5 CIFAR100_pr005_nr05
}

run_crowd_srse() {
  local dataset="$1"
  local lpi="$2"
  local gpu="$3"
  local protocol="$4"
  local exp="${dataset}_lpi${lpi}_${protocol}_slice2_maxw01_warm10_e100_seed123"
  local crowd_train_root="${CROWD_ROOT}/${dataset}"
  if [[ "${dataset}" == "Treeversity" && ! -f "${crowd_train_root}/annotations.json" && -f "${CROWD_ROOT}/Treeversity#6/annotations.json" ]]; then
    crowd_train_root="${CROWD_ROOT}/Treeversity#6"
  fi
  run_py "${MAIN_SCRIPT}" \
    --dataset "${dataset}" \
    --train_root "${crowd_train_root}" \
    --lpi "${lpi}" \
    --slice 2 \
    --split_protocol "${protocol}" \
    --pr 0.05 \
    --nr 0.5 \
    --network R50 \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 4 \
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
    --seeds 1 2 3 \
    --cuda_dev "${gpu}" \
    --out "${OUT}/crowd_srse_table" \
    --exp_name "${exp}"
}

run_crowd_srse_table() {
  run_crowd_srse Benthic 3 "${GPU_ID}" pals_3fold
  run_crowd_srse Benthic 10 "${GPU_ID}" pals_3fold
  run_crowd_srse Plankton 3 "${GPU_ID}" standard
  run_crowd_srse Treeversity 3 "${GPU_ID}" pals_3fold
  if [[ -f "${AGG_CROWD_SCRIPT}" ]]; then
    run_py "${AGG_CROWD_SCRIPT}" \
      --root "${OUT}/crowd_srse_table" \
      --write-summary "${OUT}/crowd_srse_table/summary" \
      --expected-epochs 100
  fi
}

target="${1:-help}"
case "${target}" in
  cifar10_table) run_cifar10_table ;;
  cifar100_table) run_cifar100_table ;;
  cifar100h_table) run_cifar100h_table ;;
  cifar_main_tables) run_cifar_main_tables ;;
  main_c100_eta03) run_main_c100_eta03 ;;
  topology_micro_eta03) run_topology_micro 0.3 ;;
  topology_micro_eta04) run_topology_micro 0.4 ;;
  component_table) run_component_table ;;
  pss_table) run_pss_table ;;
  no_reg_table) run_no_reg_table ;;
  crowd_srse_table) run_crowd_srse_table ;;
  all)
    run_cifar_main_tables
    run_topology_micro 0.3
    run_topology_micro 0.4
    run_component_table
    run_pss_table
    run_no_reg_table
    run_crowd_srse_table
    ;;
  help|-h|--help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
