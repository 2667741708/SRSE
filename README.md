# SRSE

[中文说明](README_CN.md)

Official reproduction code for **Source-Restricted Supervision Extraction
(SRSE)** for noisy partial-label learning.

## Environment

The experiments require Linux and an NVIDIA GPU. Create the provided Conda
environment:

```bash
git clone https://github.com/2667741708/SRSE.git
cd SRSE
conda env create -f environment.yml
conda activate srse
```

## Datasets

Download CIFAR-10, CIFAR-100, Benthic, Plankton, and Treeversity and follow the
directory layout in
[`reproducibility/DATASETS_README.md`](reproducibility/DATASETS_README.md).
`CIFAR100H` is generated from the CIFAR-100 hierarchy and requires no separate
image archive.

The launchers use these runtime roots:

```bash
export DATA_ROOT=$PWD/data
export CROWD_ROOT=$PWD
```

## Pretrained Weights

CIFAR-10, CIFAR-100, and CIFAR100H use ResNet-18 trained from scratch and do
not require ImageNet-pretrained ResNet-18 weights. The reported Benthic,
Plankton, and Treeversity experiments use ResNet-50 with TorchVision
`IMAGENET1K_V1` weights:

- [ResNet-50 IMAGENET1K_V1](https://download.pytorch.org/models/resnet50-0676ba61.pth)

TorchVision downloads the file automatically to
`~/.cache/torch/hub/checkpoints/`. For offline runs, place
`resnet50-0676ba61.pth` in that directory before training.

## Reproduction

Set the runtime paths from the repository root:

```bash
export PROJECT_ROOT=$PWD
export PY=python
export DATA_ROOT=$PWD/data
export CROWD_ROOT=$PWD
export GPU_ID=0
export OUT=$PWD/outputs
```

List all targets:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh help
```

Run all reported experiments:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh all
```

Individual targets are available for the main and ablation results:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh cifar10_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh cifar100_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh cifar100h_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh main_c100_eta03
bash reproducibility/commands/reproduce_srse_paper_experiments.sh topology_micro_eta03
bash reproducibility/commands/reproduce_srse_paper_experiments.sh topology_micro_eta04
bash reproducibility/commands/reproduce_srse_paper_experiments.sh component_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh pss_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh no_reg_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh crowd_srse_table
```

The launcher prints each Python command and writes results under `OUT`.
`CIFAR_DOWNLOAD=1` enables automatic CIFAR download; the default is offline
loading from `DATA_ROOT`.

## Quick Check

After placing the datasets, verify the public entry points with one-epoch
runs:

```bash
bash reproducibility/commands/smoke_test_srse_entrypoints.sh all
```

This check verifies execution only. Use the reproduction launcher above for
the full experiments.
