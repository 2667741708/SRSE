# SRSE

[English README](README.md)

Source-Restored Supervision Extraction (SRSE) 是一个面向 noisy partial-label learning（含噪部分标签学习）的公开复现实验包。本仓库包含当前 SRSE 论文实验使用的公开代码快照、数据放置说明、环境配置和统一复现实验入口。

这个 release 只发布 SRSE 代码与复现脚本，不包含数据集、生成的特征缓存、原始训练日志、checkpoint、baseline 对比实验脚本或带有机器本地绝对路径的启动脚本。

## 仓库内容

```text
data/                         数据集加载器源码与下载说明。
utils/                        共享模型、损失函数、学习率调度和 topology 工具。
reproducibility/
  code/main/main.py           SRSE 主训练入口。
  code/component_ablation/    组件消融与 Topology-DAES 消融入口。
  code/persistent_state/      持久化监督状态 proxy 实验。
  commands/                   统一论文实验 launcher。
```

当前公开快照不需要额外的 ANN/KNN 后端库。论文主线训练入口使用脚本内部的
PyTorch chunked KNN 实现，未被调用的旧邻居选择 helper 函数已经从 release
代码中移除。

## 环境配置

推荐 Linux 环境：

```bash
conda create -n srse python=3.10 -y
conda activate srse

# 根据本机驱动选择匹配的 CUDA 版本。CUDA 12.1 对较新的 NVIDIA 环境是一个实用默认值；
# CPU-only 可以跑通小规模检查，但不适合完整表格实验。
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

也可以直接使用 Conda 环境文件：

```bash
conda env create -f environment.yml
conda activate srse
```

完整 500-epoch CIFAR 实验是在 NVIDIA GPU 上生成的。不同 GPU 型号、PyTorch/CUDA 版本和 cuDNN determinism 设置可能带来小幅数值差异。

## 数据集

仓库不包含数据集二进制文件。请按照下面两个文件下载并放置数据：

- `data/read.md`
- `reproducibility/DATASETS_README.md`

主要数据源：

- CIFAR-10/100 官方页面：https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-10 Python archive：https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- CIFAR-100 Python archive：https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
- DCIC Zenodo record：https://zenodo.org/records/7180818
- DCIC DOI：https://doi.org/10.5281/zenodo.7152309
- DCIC source code：https://github.com/Emprime/dcic

期望的本地目录布局：

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

`CIFAR100H` 由 CIFAR-100 类层级生成，不需要额外的图像压缩包。

### CIFAR100H 候选标签构造协议

`CIFAR100H` 使用与 `CIFAR100` 相同的 CIFAR-100 图像和 fine labels；区别只在候选标签集合的构造方式。运行时使用 `--dataset CIFAR100H` 会启用 hierarchical protocol。

实现中采用 CIFAR-100 标准层级：20 个 superclasses，每个 superclass 包含 5 个 fine classes。对于真实 fine label 为 `y` 的训练样本，只有与 `y` 属于同一个 superclass 的 fine labels 才允许进入候选集合；其他 superclasses 的标签采样概率为 0。

对每个样本，转移行定义为：

- `P(y 被包含) = 1 - nr`，其中 `nr` 是通过 `--nr` 传入的 noisy-label rate。
- 对同一 superclass 内每个 sibling fine class `c != y`，`P(c 被包含) = pr`，其中 `pr` 是通过 `--pr` 传入的 partial-label rate。
- 对真实标签所在 superclass 之外的所有 fine classes，`P(c 被包含) = 0`。

loader 会按上述概率采样一个二值候选向量，如果候选集合为空则重新采样。因此，当 `nr = 0` 时，真实标签一定在候选集合内；当 `nr > 0` 时，CIFAR100H 是 noisy candidate-label protocol，真实标签可能缺失。相比 uniform CIFAR-100 partial labels，CIFAR100H 把干扰标签限制在语义相关的 sibling classes 内，而不是从全部 99 个非真实类别中均匀采样。

## 预训练权重

CIFAR 实验使用从头训练的 ResNet-18。众包数据集使用 TorchVision ImageNet-1K 预训练 backbone：

- ResNet-18 `IMAGENET1K_V1`：https://download.pytorch.org/models/resnet18-f37072fd.pth
- ResNet-50 `IMAGENET1K_V1`：https://download.pytorch.org/models/resnet50-0676ba61.pth

TorchVision 会自动下载到默认 torch hub 缓存目录（`~/.cache/torch/hub/checkpoints`）。离线运行时，请先把上述文件下载到该缓存目录。

## 复现实验命令

在仓库根目录使用统一 launcher：

```bash
export PROJECT_ROOT=$PWD
export PY=python
export DATA_ROOT=$PWD/data
export CROWD_ROOT=$PWD
export GPU_ID=0
export OUT=$PWD/out_ultimate/reproduce_srse

bash reproducibility/commands/reproduce_srse_paper_experiments.sh help
```

主 CIFAR-100 结果：

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh main_c100_eta03
```

Topology-DAES 微消融：

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh topology_micro_eta03
bash reproducibility/commands/reproduce_srse_paper_experiments.sh topology_micro_eta04
```

其他论文表格：

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh component_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh pss_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh no_reg_table
bash reproducibility/commands/reproduce_srse_paper_experiments.sh crowd_srse_table
```

launcher 会在执行前打印完整 Python 命令，并把输出写入 `OUT`。

## 预期结果尺度

在相同 seeds 和超参数下，主 CIFAR-100 设置 `q=0.05, eta=0.3` 应复现到论文中的 high-79% final accuracy 区间。`eta=0.4` 的 Topology-DAES full row 预期约为 78% final accuracy。只要偏差没有显著超过报告的 seed-to-seed standard deviation，小幅波动属于正常现象。

## 发布包边界

仓库排除了：

- 数据集压缩包和解压后的数据集目录。
- baseline 对比实验脚本，这部分仅在本地保留。
- `out_ultimate`、checkpoint、`.pt/.pth/.ckpt`、`.npz/.npy/.pkl` 和原始日志等生成产物。
- Python 字节码和缓存目录。
- 带硬编码绝对路径的机器本地启动脚本。

## 引用

论文元数据定稿后会补充正式 citation。在此之前，请引用本 GitHub 仓库以及用于比较的对应论文版本。
