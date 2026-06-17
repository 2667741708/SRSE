# Reproducibility Code

This directory contains the code and command entry points for reproducing the
SRSE manuscript experiments.

Use the package-level launcher:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh help
```

Main code snapshots:

- `code/main/main.py`: SRSE main training entry.
- `code/component_ablation/srse_ablation.py`: component and Topology-DAES
  ablation entry.
- `code/persistent_state/srse_persistent.py`: persistent supervision-state
  proxy entry.
- `code/baselines/`: baseline-control utilities retained for comparison
  experiments.

Datasets, generated outputs, checkpoints, raw logs, and machine-local launcher
wrappers are intentionally excluded from this public package.
