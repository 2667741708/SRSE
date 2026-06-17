# Reproducibility Code

This directory contains the code and command entry points for reproducing the
SRSE manuscript experiments.

Use the package-level launcher:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh help
```

Dataset-specific CIFAR main-table launchers are also available:

- `commands/reproduce_srse_cifar10_experiments.sh`
- `commands/reproduce_srse_cifar100_experiments.sh`
- `commands/reproduce_srse_cifar100h_experiments.sh`

Main code snapshots:

- `code/main/main.py`: SRSE main training entry.
- `code/component_ablation/srse_ablation.py`: component and Topology-DAES
  ablation entry.
- `code/persistent_state/srse_persistent.py`: persistent supervision-state
  proxy entry.

Datasets, generated outputs, checkpoints, raw logs, baseline-comparison
experiment scripts, and machine-local launcher wrappers are intentionally
excluded from this public package.
