# SRSE

[中文说明](README_CN.md)

Source-Restored Supervision Extraction (SRSE) is a reproducibility package for noisy partial-label learning experiments. The repository contains the public code snapshot, dataset placement notes, environment specifications, and command launcher used for the current SRSE manuscript experiments.

This release is intentionally code-focused. It excludes datasets, generated feature caches, raw training logs, checkpoints, baseline-comparison experiment scripts, and machine-local launch wrappers.

## What Is Included

```text
data/                         Dataset loader source and download notes.
utils/                        Shared models, losses, schedulers, and topology utilities.
reproducibility/
  code/main/main.py           Main SRSE training entry point.
  code/component_ablation/    Component and Topology-DAES ablation entry points.
  code/persistent_state/      Persistent supervision-state proxy experiments.
  commands/                   Unified paper-experiment launcher.
```

The current public snapshot does not require an external ANN/KNN backend. The
paper-facing training entries use the in-script PyTorch chunked KNN
implementation, and unused legacy neighbor-selection helpers have been removed
from the release code.

## Environment

Recommended Linux setup:

```bash
conda create -n srse python=3.10 -y
conda activate srse

# Pick the CUDA build that matches your driver. CUDA 12.1 is a practical default
# for recent NVIDIA systems; CPU-only runs are possible but too slow for full tables.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

An equivalent Conda environment is provided:

```bash
conda env create -f environment.yml
conda activate srse
```

Full 500-epoch CIFAR runs were produced on NVIDIA GPUs. Small numerical differences can occur across GPU models, PyTorch/CUDA versions, and cuDNN determinism settings.

## Quick Start

Clone the repository and create the environment:

```bash
git clone https://github.com/2667741708/SRSE.git
cd SRSE

conda env create -f environment.yml
conda activate srse
```

Run a launcher sanity check that does not require datasets or a GPU:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh help

# Optional command preview: prints the Python command without starting training.
PY=echo bash reproducibility/commands/reproduce_srse_paper_experiments.sh main_c100_eta03
```

After placing datasets as described below, run a one-epoch executable smoke
check before starting a full table:

```bash
export PROJECT_ROOT=$PWD
export DATA_ROOT=$PWD/data
export CROWD_ROOT=$PWD
export GPU_ID=0

bash reproducibility/commands/smoke_test_srse_entrypoints.sh cifar
```

Then run the selected full experiment:

```bash
export PROJECT_ROOT=$PWD
export DATA_ROOT=$PWD/data
export CROWD_ROOT=$PWD
export GPU_ID=0
export OUT=$PWD/out_ultimate/reproduce_srse

bash reproducibility/commands/reproduce_srse_paper_experiments.sh main_c100_eta03
```

The public launchers default to offline CIFAR loading. Set
`CIFAR_DOWNLOAD=1` only when you want TorchVision to download CIFAR files into
`DATA_ROOT`.

## Datasets

Dataset binaries are excluded. Download and place them according to:

- `data/read.md`
- `reproducibility/DATASETS_README.md`

Main sources:

- CIFAR-10/100 official page: https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-10 Python archive: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- CIFAR-100 Python archive: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
- DCIC Zenodo record: https://zenodo.org/records/7180818
- DCIC DOI: https://doi.org/10.5281/zenodo.7152309
- DCIC source code: https://github.com/Emprime/dcic

Expected local layout:

```text
data/
  cifar-10-batches-py/
  cifar-100-python/
Benthic/
  annotations.json
  fold1/ ... fold5/
Plankton/
  annotations.json
  fold1/ ... fold5/
Treeversity#6/
  annotations.json
  fold1/ ... fold5/
```

`CIFAR100H` is generated from the CIFAR-100 hierarchy and does not require a separate image archive.

### CIFAR100H Candidate-Label Protocol

`CIFAR100H` uses the same CIFAR-100 images and fine labels as `CIFAR100`; the only difference is the candidate-label construction. The loader activates the hierarchical protocol when `--dataset CIFAR100H` is used.

The implementation builds the standard CIFAR-100 hierarchy with 20 superclasses and 5 fine classes per superclass. For a training sample with true fine label `y`, only labels from the same superclass as `y` can enter the candidate set. Labels from other superclasses have zero sampling probability.

For each sample, the transition row is:

- `P(y is included) = 1 - nr`, where `nr` is the noisy-label rate passed by `--nr`.
- `P(c is included) = pr` for each sibling fine class `c != y` in the same superclass, where `pr` is the partial-label rate passed by `--pr`.
- `P(c is included) = 0` for every fine class outside the true label's superclass.

The loader samples a binary candidate vector from this row and resamples if the vector is empty. Therefore, with `nr = 0`, the true label is always included; with `nr > 0`, CIFAR100H becomes a noisy candidate-label protocol where the true label may be absent. Compared with uniform CIFAR-100 partial labels, CIFAR100H restricts distractor labels to semantically related sibling classes instead of sampling from all 99 non-ground-truth classes.

## Pretrained Weights

CIFAR experiments use ResNet-18 from scratch. Crowdsourced datasets use TorchVision ImageNet-1K pretrained backbones:

- ResNet-18 `IMAGENET1K_V1`: https://download.pytorch.org/models/resnet18-f37072fd.pth
- ResNet-50 `IMAGENET1K_V1`: https://download.pytorch.org/models/resnet50-0676ba61.pth

TorchVision downloads these files automatically into the default torch hub cache (`~/.cache/torch/hub/checkpoints`). For offline runs, download the files above into that cache directory before launching training.

## Reproduction Commands

Use the unified launcher from the repository root:

```bash
export PROJECT_ROOT=$PWD
export PY=python
export DATA_ROOT=$PWD/data
export CROWD_ROOT=$PWD
export GPU_ID=0
export OUT=$PWD/out_ultimate/reproduce_srse

bash reproducibility/commands/reproduce_srse_paper_experiments.sh help
```

Main CIFAR-100 result:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh main_c100_eta03
```

Topology-DAES micro-ablation:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh topology_micro_eta03
bash reproducibility/commands/reproduce_srse_paper_experiments.sh topology_micro_eta04
```

Other manuscript tables:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh cifar10_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh cifar100_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh cifar100h_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh component_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh pss_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh no_reg_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh crowd_srse_table
```

The CIFAR main-table launchers are also available as separate dataset-level
scripts:

```bash
bash reproducibility/commands/reproduce_srse_cifar10_experiments.sh main_table
bash reproducibility/commands/reproduce_srse_cifar100_experiments.sh main_table
bash reproducibility/commands/reproduce_srse_cifar100h_experiments.sh main_table
```

The launcher prints each full Python command before execution and stores outputs under `OUT`.
For crowdsourced datasets, `crowd_srse_table` runs Benthic, Plankton, and
Treeversity at both `lpi=3` and `lpi=10`.

For quick executable checks, use:

```bash
bash reproducibility/commands/smoke_test_srse_entrypoints.sh all
```

This smoke script uses `EPOCHS=1` by default and verifies public CIFAR,
CIFAR100H, crowdsourced-dataset, component-ablation, Topology-DAES, PSS, and
no-CR/no-MixUp entry points. It is not a result reproduction; use the paper
launcher above for the full 500-epoch runs.

## Expected Result Scale

With the same seeds and hyperparameters, the main CIFAR-100 setting `q=0.05, eta=0.3` should reproduce the manuscript result around the high-79% final-accuracy range. The `eta=0.4` Topology-DAES full row is expected around 78% final accuracy. Treat small deviations as normal unless they materially exceed the reported seed-to-seed standard deviation.

## Package Hygiene

The repository excludes:

- Dataset archives and extracted dataset folders.
- Baseline-comparison experiment scripts, which are kept local-only.
- Generated outputs such as `out_ultimate`, checkpoints, `.pt/.pth/.ckpt`, `.npz/.npy/.pkl`, and raw logs.
- Python bytecode and cache directories.
- Machine-local launch wrappers with hard-coded absolute paths.

## Citation

The manuscript citation will be added after the paper metadata is finalized. Until then, please cite this GitHub repository and the relevant manuscript version used for comparison.
