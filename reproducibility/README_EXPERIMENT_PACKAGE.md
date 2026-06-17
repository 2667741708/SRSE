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
git clone https://github.com/2667741708/SRSE.git
cd SRSE

conda env create -f environment.yml
conda activate srse

bash reproducibility/commands/reproduce_srse_paper_experiments.sh help

# Optional command preview: prints the Python command without starting training.
PY=echo bash reproducibility/commands/reproduce_srse_paper_experiments.sh main_c100_eta03

export PROJECT_ROOT=$PWD
export DATA_ROOT=$PWD/data
export CROWD_ROOT=$PWD
export GPU_ID=0
export OUT=$PWD/out_ultimate/reproduce_srse

# Run this after placing CIFAR/DCIC data under DATA_ROOT/CROWD_ROOT.
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
