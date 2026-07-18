# SRSE

[English README](README.md)

**Source-Restricted Supervision Extraction (SRSE)** 的含噪部分标签学习实验复现代码。

## 环境

完整实验需要 Linux 和 NVIDIA GPU。使用仓库提供的 Conda 环境：

```bash
git clone https://github.com/2667741708/SRSE.git
cd SRSE
conda env create -f environment.yml
conda activate srse
```

## 数据集

下载 CIFAR-10、CIFAR-100、Benthic、Plankton 和 Treeversity，并按照
[`reproducibility/DATASETS_README.md`](reproducibility/DATASETS_README.md)
放置数据。`CIFAR100H` 由 CIFAR-100 类层级生成，不需要额外的图像压缩包。

复现脚本使用以下数据根目录：

```bash
export DATA_ROOT=$PWD/data
export CROWD_ROOT=$PWD
```

## 预训练权重

CIFAR-10、CIFAR-100 和 CIFAR100H 使用从头训练的 ResNet-18，不需要
ResNet-18 的 ImageNet 预训练权重。论文报告的 Benthic、Plankton 和
Treeversity 实验使用 TorchVision `IMAGENET1K_V1` 预训练的 ResNet-50：

- [ResNet-50 IMAGENET1K_V1](https://download.pytorch.org/models/resnet50-0676ba61.pth)

TorchVision 会自动将权重下载到 `~/.cache/torch/hub/checkpoints/`。离线运行时，
请预先将 `resnet50-0676ba61.pth` 放入该目录。

## 复现实验

在仓库根目录设置运行路径：

```bash
export PROJECT_ROOT=$PWD
export PY=python
export DATA_ROOT=$PWD/data
export CROWD_ROOT=$PWD
export GPU_ID=0
export OUT=$PWD/outputs
```

查看全部实验目标：

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh help
```

运行论文报告的全部实验：

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh all
```

也可以分别运行主结果和消融实验：

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

launcher 会打印实际执行的 Python 命令，并将结果写入 `OUT`。
设置 `CIFAR_DOWNLOAD=1` 可自动下载 CIFAR；默认从 `DATA_ROOT` 离线读取。

## 快速检查

数据放置完成后，可用单 epoch 运行检查公开入口：

```bash
bash reproducibility/commands/smoke_test_srse_entrypoints.sh all
```

该命令只检查程序能否执行。完整结果应使用上面的复现实验 launcher。
