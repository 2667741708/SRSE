# Source-Restricted Supervision Extraction

This anonymous repository contains the SRSE implementation and the launchers
used for the experiments in the accompanying TNNLS manuscript. It supports
CIFAR-10, CIFAR-100, hierarchical CIFAR-100 (CIFAR100H), Benthic, Plankton,
and Treeversity.

## Repository Layout

```text
reproducibility/
  code/
    main/main.py                 Main SRSE training entry point
    component_ablation/         Component-ablation entry point and checks
    persistent_state/           Carry-over supervision proxy experiments
    data/                        CIFAR and crowdsourced dataset loaders
    utils/                       Models, losses, propagation, and evaluation
  commands/                     Paper-table launchers and smoke tests
environment.yml                 Complete Conda environment
requirements.txt                Extra packages for an existing PyTorch setup
```

Generated datasets, pretrained weights, checkpoints, and raw logs are not
included.

## Environment

The recommended setup is Linux, Python 3.10, and an NVIDIA GPU:

```bash
conda env create -f environment.yml
conda activate srse
```

Alternatively, install a PyTorch/TorchVision build compatible with the local
CUDA driver and then install the remaining packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Full 500-epoch runs require a CUDA-capable GPU. Numerical results can vary
slightly with the GPU, CUDA, cuDNN, and PyTorch versions.

## Data

Place the datasets under the repository root as follows:

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

CIFAR-10 and CIFAR-100 are available from the
[official CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html). The
crowdsourced datasets are available from the
[DCIC Zenodo record](https://zenodo.org/records/7180818). CIFAR100H uses the
CIFAR-100 images and constructs hierarchical candidate labels at runtime; it
does not require a separate image archive.

Set `CIFAR_DOWNLOAD=1` to allow TorchVision to download CIFAR files. The
default is offline loading.

Crowdsourced experiments use TorchVision ImageNet-1K pretrained ResNet
weights. In an offline environment, place the required ResNet-18 and
ResNet-50 weights in `~/.cache/torch/hub/checkpoints/` before running them.

## Reproduction

Set the runtime paths from the repository root:

```bash
export PROJECT_ROOT=$PWD
export DATA_ROOT=$PWD/data
export CROWD_ROOT=$PWD
export OUT=$PWD/outputs
export GPU_ID=0
export PY=python
```

List all supported targets:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh help
```

Run the main result tables:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh cifar10_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh cifar100_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh cifar100h_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh crowd_srse_table
```

The crowdsourced target runs Benthic, Plankton, and Treeversity with both
reported labels-per-image settings. The launchers use seeds 1, 2, and 3 by
default and write each run under `OUT`.

Run the ablation and supervision-state experiments:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh topology_micro_eta03
bash reproducibility/commands/reproduce_srse_paper_experiments.sh topology_micro_eta04
bash reproducibility/commands/reproduce_srse_paper_experiments.sh component_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh pss_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh no_reg_table
```

The dataset-specific CIFAR launchers are also available:

```bash
bash reproducibility/commands/reproduce_srse_cifar10_experiments.sh main_table
bash reproducibility/commands/reproduce_srse_cifar100_experiments.sh main_table
bash reproducibility/commands/reproduce_srse_cifar100h_experiments.sh main_table
```

## Verification

Check the command paths without loading data:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh help
PY=echo bash reproducibility/commands/reproduce_srse_paper_experiments.sh main_c100_eta03
```

After placing the datasets, run one-epoch smoke tests before full training:

```bash
bash reproducibility/commands/smoke_test_srse_entrypoints.sh cifar
bash reproducibility/commands/smoke_test_srse_entrypoints.sh crowd
```

Use `all` instead of `cifar` or `crowd` to check every public entry point.
These smoke tests validate execution only; they do not reproduce final paper
accuracy.

## Outputs

Each training run stores its configuration, logs, final metrics, and optional
checkpoints under `OUT`. The main CIFAR-100 setting `q=0.05, eta=0.3` should be
in the high-79% final-accuracy range with the reported configuration. Compare
three-seed mean and standard deviation rather than requiring bitwise-identical
results across hardware stacks.
