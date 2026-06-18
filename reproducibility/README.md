# SRSE Paper Reproducibility Package

This directory contains the public SRSE reproducibility package. It collects the
training entries, dataset notes, and command launchers needed to rerun the SRSE
experiments from a clean checkout.

## Scope

This package records three evidence tracks:

1. Main SRSE results and standard comparison tables.
2. Component ablations A0--A11.
3. Persistent supervision state proxy experiments.

Generated outputs, private provenance manifests, baseline launchers, and
machine-local wrappers are not part of this public package.

## Trusted Code Snapshots

| Track | Local snapshot | Source role |
|---|---|---|
| Main SRSE results | [main.py](code/main/main.py#L1) | Main-result training entry |
| Component ablations | [srse_ablation.py](code/component_ablation/srse_ablation.py#L1) | Public component and Topology-DAES ablation entry |
| Persistent-state v2 proxy snapshot | [srse_persistent.py](code/persistent_state/srse_persistent.py#L1) | Persistent-state proxy training entry |
| Persistent-state helper tests | [test_persistent_state_helpers.py](code/persistent_state/test_persistent_state_helpers.py#L1) | Local helper tests for update operators |

Shared import packages live under `code/data/` and `code/utils/`. The active
launchers add `reproducibility/code` to `PYTHONPATH`, so the training snapshots
continue to use the original import names, such as `from data.dataset ...` and
`from utils.topology ...`.

Use the unified launcher
[reproduce_srse_paper_experiments.sh](commands/reproduce_srse_paper_experiments.sh).
It is the package-local entry point for the current paper-facing main-result,
component-ablation, persistent-state-proxy, no-MixUp/no-CR, and crowdsourced
dataset reproduction commands.

Generated result logs, checkpoints, feature caches, machine-local provenance
manifests, and timestamped launcher outputs are intentionally not shipped in
this GitHub package.

## Reproduction Commands

Use these package-local command templates from the repository root:

- [reproduce_srse_paper_experiments.sh](commands/reproduce_srse_paper_experiments.sh)
- [reproduce_srse_cifar10_experiments.sh](commands/reproduce_srse_cifar10_experiments.sh)
- [reproduce_srse_cifar100_experiments.sh](commands/reproduce_srse_cifar100_experiments.sh)
- [reproduce_srse_cifar100h_experiments.sh](commands/reproduce_srse_cifar100h_experiments.sh)
- [smoke_test_srse_entrypoints.sh](commands/smoke_test_srse_entrypoints.sh)
- [verify_epoch_test_acc_against_reference_logs.sh](commands/verify_epoch_test_acc_against_reference_logs.sh)

These commands document the intended reproduction entry points. Set
`PROJECT_ROOT`, `PY`, `DATA_ROOT`, `CROWD_ROOT`, and `GPU_ID` for the target
machine before running full experiments.

For quick epoch-level validation against an existing completed log, keep the
original experiment horizon in `--epochs` and use `--max_run_epochs` to stop the
execution early. For example, the following command preserves the CIFAR-100
e500 schedule but only executes the first two epochs before comparing the first
two `test_acc` values:

```bash
MAX_RUN_EPOCHS=2 \
MAX_W_MODEL=0.5 \
FEATURE_EXTRACT_VIEW=weak_strong_fusion \
REFERENCE_ROOT=/path/to/reference_logs \
PY=/path/to/python \
DATA_ROOT=/path/to/datasets \
bash reproducibility/commands/verify_epoch_test_acc_against_reference_logs.sh main_c100_eta03
```

The current SRSE code uses `--model_warmup_epochs 10` for the progressive
model-view fusion schedule. This is the compatibility setting for the revised
warmup formula; older logs may record `20` under the previous formulation.
The model-view fusion cap defaults to `--max_w_model 0.5`.

SRSE reliable-set selection is class-balanced by default: predictions must be
supported by the source prior, then each predicted class receives a `--delta`
quota and samples are ranked by classwise discrepancy.

The persistent-state proxy suite uses the stable entry names
`srse_persistent.py`,
`run_persistent_state_proxies_fredis_irnet_pals_sari.sh`,
`run_persistent_state_proxies_upllrs_pico_plus.sh`, and
`run_persistent_state_proxies_modelpred.sh`. Timestamped development variants
are not part of the GitHub reproducibility surface.

## Local Validation

Before full training, run the launcher help and smoke checks from the repository
root:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh help
bash reproducibility/commands/smoke_test_srse_entrypoints.sh help
```

Full result reproduction requires the datasets described in
[DATASETS_README.md](DATASETS_README.md), the Python environment from the root
`environment.yml` or `requirements.txt`, and the experiment commands listed
above.
