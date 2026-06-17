# Dataset Download Notes

This release package intentionally excludes dataset binaries, extracted image
folders, generated feature caches, checkpoints, and training outputs. Keep the
dataset loader source files under `reproducibility/code/data/*.py`, and place
the actual data under the runtime roots described below.

## Required runtime layout

```text
data/
  cifar-10-batches-py/
  cifar-100-python/
Benthic/
  annotations.json
  fold1/
  fold2/
  fold3/
  fold4/
  fold5/
Plankton/
  annotations.json
  fold1/
  fold2/
  fold3/
  fold4/
  fold5/
Treeversity#6/
  annotations.json
  fold1/
  fold2/
  fold3/
  fold4/
  fold5/
```

Some loaders accept `Treeversity` and internally map it to `Treeversity#6`.
If your extracted DCIC folder is named `Treeversity`, either keep that folder
and use the compatible loader path or create a symlink/copy named
`Treeversity#6`.

## CIFAR-10 / CIFAR-100

- Official dataset page:
  https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-10 Python archive:
  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- CIFAR-100 Python archive:
  https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

`CIFAR100H` in our scripts is the hierarchical CIFAR-100 candidate-label
protocol generated from CIFAR-100 class hierarchy information. It does not
require a separate image archive beyond CIFAR-100.

## Crowdsourced DCIC datasets

- DCIC official Zenodo record:
  https://zenodo.org/records/7180818
- DCIC DOI landing page:
  https://doi.org/10.5281/zenodo.7152309
- DCIC source code:
  https://github.com/Emprime/dcic

Use the Benthic, Plankton, and Treeversity subsets from the DCIC release.
The loaders expect each dataset root to contain `annotations.json` and fold
subdirectories. The paper commands use `slice=2` and the same fold convention
as the local loaders.
