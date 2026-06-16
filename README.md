# SRSE

Source-Restored Supervision Extraction (SRSE) for noisy partial-label learning.
This repository contains the clean reproduction code, dataset download notes,
environment specification, and launcher commands used for the current SRSE
manuscript experiments.

The release intentionally does not include datasets, generated caches, raw
training logs, checkpoints, or machine-local launch wrappers.

## Environment

Recommended Linux environment:

```bash
conda create -n srse python=3.10 -y
conda activate srse

# Pick the CUDA build that matches your driver. CUDA 12.1 is a safe default on
# recent NVIDIA systems; CPU-only runs are possible but too slow for full tables.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

An equivalent Conda environment is provided in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate srse
```

Full 500-epoch CIFAR runs were produced on NVIDIA GPUs. Small numerical
differences can occur across GPU models, PyTorch/CUDA versions, and cuDNN
determinism settings.

## Datasets

Dataset binaries are excluded. Download and place them according to:

- `data/read.md`
- `reproducibility/DATASETS_README.md`

Main URLs:

- CIFAR-10/100 official page: https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-10 Python archive: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- CIFAR-100 Python archive: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
- DCIC Zenodo record: https://zenodo.org/records/7180818
- DCIC DOI: https://doi.org/10.5281/zenodo.7152309
- DCIC source code: https://github.com/Emprime/dcic
- Optional CUB-200-2011: https://www.vision.caltech.edu/datasets/cub_200_2011/

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

`CIFAR100H` is generated from the CIFAR-100 hierarchy and does not require a
separate image archive.

## Pretrained Weights

CIFAR experiments use ResNet-18 from scratch. Crowdsourced datasets and optional
CUB runs use TorchVision ImageNet-1K pretrained backbones:

- ResNet-18 `IMAGENET1K_V1`:
  https://download.pytorch.org/models/resnet18-f37072fd.pth
- ResNet-50 `IMAGENET1K_V1`:
  https://download.pytorch.org/models/resnet50-0676ba61.pth

TorchVision downloads these automatically into the default torch hub cache
(`~/.cache/torch/hub/checkpoints`). For offline runs, download the files above
into that cache directory before launching training.

## Reproduction Commands

Use the unified launcher:

```bash
export PROJECT_ROOT=$PWD
export PY=python
export DATA_ROOT=$PWD/data
export CROWD_ROOT=$PWD
export GPU_ID=0
export OUT=$PWD/out_ultimate/reproduce_srse

bash reproducibility/commands/reproduce_srse_paper_experiments_20260616.sh help
```

Main CIFAR-100 result:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments_20260616.sh main_c100_eta03
```

Topology-DAES micro-ablation:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments_20260616.sh topology_micro_eta03
bash reproducibility/commands/reproduce_srse_paper_experiments_20260616.sh topology_micro_eta04
```

Other manuscript tables:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments_20260616.sh component_table
bash reproducibility/commands/reproduce_srse_paper_experiments_20260616.sh pss_table
bash reproducibility/commands/reproduce_srse_paper_experiments_20260616.sh no_reg_table
bash reproducibility/commands/reproduce_srse_paper_experiments_20260616.sh crowd_srse_table
```

The launcher prints each full Python command before execution and stores outputs
under `OUT`.

## Expected Scale of Results

With the same seeds and hyperparameters, the main CIFAR-100 setting
`q=0.05, eta=0.3` should reproduce the manuscript result around the high-79%
final-accuracy range. The `eta=0.4` Topology-DAES full row is expected around
78% final accuracy. Treat small deviations as normal unless they exceed the
reported seed-to-seed standard deviation materially.

## Package Hygiene

The repository excludes:

- Dataset archives and extracted dataset folders.
- `out_ultimate`, checkpoints, `.pt/.pth/.ckpt`, and raw logs.
- Python bytecode and cache directories.
- Machine-local launch wrappers with hard-coded absolute paths.
