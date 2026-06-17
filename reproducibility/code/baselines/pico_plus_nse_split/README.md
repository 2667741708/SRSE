# PiCO+ With SRSE-Style Clean/Noisy Split

This directory defines a controlled baseline-edit experiment for PiCO+.

The experiment starts from the official PiCO/PiCO+ repository:

```text
https://github.com/hbzju/PiCO
```

Only the clean/noisy sample selection rule is replaced. The official PiCO+
classification losses, contrastive losses, confidence EMA, prototype updates,
MixUp branch, queue logic, and candidate/noisy pseudo-label handling are kept.

## Baseline Setting

Paper-facing setting:

```text
CIFAR-100, q=0.05, eta=0.3, epochs=500, seeds=1 2 3
```

PALS/SARI reports CIFAR experiments with SGD momentum 0.9, weight decay 0.001,
cosine learning-rate schedule, and 500 epochs. The original PiCO+ README uses
800 epochs for CIFAR examples, but the PALS protocol uses 500 epochs for CIFAR
baselines. We therefore use 500 epochs for this comparison.

## Split Replacement

Official PiCO+:

```text
distance-to-prototype -> sorted near-prototype samples -> reliable/clean
```

This controlled edit:

```text
native candidate source + stored epoch features + topology KNN propagation
-> per-class reliable selection -> reliable/clean
```

The original candidate set is not changed. The maintained PiCO+ confidence
state remains a training target, not a source prior for supervision extraction.

## Files

- `pico_plus_nse_split.patch`: applies the split replacement to the official
  PiCO repository.
- `pico_plus_nse_split.patch`: apply this patch to the official repository,
  then run the CIFAR-100 `q=0.05, eta=0.3` experiment with the same seed,
  optimizer, and 500-epoch protocol used by the SRSE launcher.
