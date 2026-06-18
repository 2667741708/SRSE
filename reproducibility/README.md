# SRSE Paper Reproducibility Package

This directory is the paper-facing reproducibility package for the current SRSE
manuscript. It separates the scripts and results that support the reported
paper claims from older exploratory or deprecated code.

## Scope

This package records three evidence tracks:

1. Main SRSE results and standard comparison tables.
2. Component ablations A0--A8.
3. Persistent supervision state proxy experiments.

The package does not claim that every historical script in the repository is
active. Legacy scripts are documented in [archive_policy.md](archive_policy.md).

## Trusted Code Snapshots

| Track | Local snapshot | Source role |
|---|---|---|
| Main SRSE results | [main.py](code/main/main.py#L1) | Main-result training entry |
| Component ablations | [NSE_ABLATION.py](code/component_ablation/NSE_ABLATION.py#L1) | Current `main.pdf` A0--A8 entry |
| Persistent-state v2 proxy snapshot | [NSE_persistent.py](code/persistent_state/NSE_persistent.py#L1) | Persistent-state proxy training entry |
| Persistent-state helper tests | [test_persistent_state_helpers.py](code/persistent_state/test_persistent_state_helpers.py#L1) | Local helper tests for update operators |

Shared import packages live under `code/data/` and `code/utils/`. The active
launchers add `reproducibility/code` to `PYTHONPATH`, so the training snapshots
continue to use the original import names, such as `from data.dataset ...` and
`from utils.topology ...`.

For a cleaner handoff surface with descriptive experiment script names, use the
unified launcher
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

The persistent-state proxy suite uses the stable entry names
`NSE_persistent.py`,
`run_persistent_state_proxies_fredis_irnet_pals_sari.sh`,
`run_persistent_state_proxies_upllrs_pico_plus.sh`, and
`run_persistent_state_proxies_modelpred.sh`. Timestamped development variants
are not part of the GitHub reproducibility surface.

## Reference Check

IRNet is cited as the formal TPAMI 2026 publication, not as the older arXiv
preprint.

## Verification Status

Last checked on 2026-05-23:

- `py -3 -m py_compile` passed for every copied Python snapshot in
  [code/](code/main/main.py#L1).
- `latexmk -pdf -interaction=nonstopmode main.tex` passed from the repository
  root and regenerated `main.pdf`.
- `main.log` had no undefined citation or undefined reference warnings.

Additional check on 2026-05-24:

- c201 single-seed validation completed for 7 representative rows: FREDIS,
  IRNet, PALS/SARI, UPLLRS, PiCO+, component A0, and the main CIFAR-100
  `q=0.05, eta=0.3` row.

Additional validation launched on 2026-05-25:

- c201 noise-rate slice validation for CIFAR-10 `q=0.5, eta=0.1/0.2/0.3`
  and CIFAR-100 `q=0.05, eta=0.1/0.4/0.5`, using the same main NSE script
  with seed `1`.
