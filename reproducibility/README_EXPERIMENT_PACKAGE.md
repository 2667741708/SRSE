# SRSE Experiment Package

This package is a clean code-and-command release for reproducing the current
SRSE manuscript experiments. It intentionally excludes datasets, generated
feature caches, training outputs, raw logs, checkpoints, and machine-local
launch wrappers.

## Contents

- `reproducibility/code/`: training, ablation, and persistent-state proxy
  source code.
- `reproducibility/commands/reproduce_srse_paper_experiments.sh`:
  unified reproduction launcher with package-local defaults.
- `data/*.py`: dataset loaders and candidate-label construction helpers.
- `data/read.md` and `reproducibility/DATASETS_README.md`: dataset download
  and expected layout notes.
- `utils/*.py`: shared model, augmentation, loss, topology, and training
  utilities.

## Quick Start

```bash
export PROJECT_ROOT=/path/to/unpacked/package
export PY=/path/to/python
export DATA_ROOT=/path/to/cifar/root
export CROWD_ROOT=/path/to/dcic/root
export GPU_ID=0

bash reproducibility/commands/reproduce_srse_paper_experiments.sh help
bash reproducibility/commands/reproduce_srse_paper_experiments.sh main_c100_eta03
```

The unified launcher supports separate targets for the main CIFAR-100 run,
CIFAR-10/CIFAR-100/CIFAR100H main-table grids, Topology-DAES micro-ablation
rows, component ablations, persistent supervision-state proxies,
no-MixUp/no-CR fairness checks, and SRSE crowdsourced-dataset rows. The CIFAR
main-table grids are also available through dataset-specific scripts under
`reproducibility/commands/`.

## Excluded By Design

- Dataset archives and extracted dataset folders.
- Baseline-comparison experiment scripts, which are kept local-only.
- `out_ultimate`, checkpoint directories, `.pt/.pth/.ckpt` files.
- Raw `run.log`, `master_log.txt`, and launcher log files.
- Machine-local launch wrappers with hard-coded absolute paths.
- Python bytecode, `__pycache__`, and temporary cache directories.
