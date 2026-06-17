# Dataset Download Notes

The release package keeps only loader source files in this `data/` directory.
Dataset binaries, extracted image folders, generated feature caches,
checkpoints, and training outputs are intentionally excluded.

## CIFAR-10 / CIFAR-100

- Official dataset page:
  https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-10 Python archive:
  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- CIFAR-100 Python archive:
  https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

Expected local folders after extraction:

```text
data/cifar-10-batches-py/
data/cifar-100-python/
```

`CIFAR100H` is generated from the CIFAR-100 hierarchy and does not require a
separate image archive.

## Crowdsourced DCIC datasets

- DCIC official Zenodo record:
  https://zenodo.org/records/7180818
- DCIC DOI landing page:
  https://doi.org/10.5281/zenodo.7152309
- DCIC source code:
  https://github.com/Emprime/dcic

Use the Benthic, Plankton, and Treeversity subsets. Expected runtime folders:

```text
Benthic/annotations.json
Plankton/annotations.json
Treeversity#6/annotations.json
```

Each folder should also contain the fold image subdirectories used by the DCIC
annotations. Some loaders accept `Treeversity` and map it to `Treeversity#6`;
keep both names aligned if your extraction uses only one of them.
