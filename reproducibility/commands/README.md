# SRSE Reproduction Commands

Keep experiment launch commands in this directory instead of Python file
headers. Python files under `reproducibility/code/` should describe their entry
point role only; runnable command lines belong here.

## Canonical Launcher

Use the unified launcher from the repository root:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments.sh help

# Optional command preview: prints Python commands without starting training.
PY=echo bash reproducibility/commands/reproduce_srse_paper_experiments.sh main_c100_eta03
```

The launcher supports the current paper-facing targets:

- `cifar10_table`
- `cifar100_table`
- `cifar100h_table`
- `cifar_main_tables`
- `main_c100_eta03`
- `topology_micro_eta03`
- `topology_micro_eta04`
- `component_table`
- `pss_table`
- `no_reg_table`
- `crowd_srse_table`
- `all`

Runtime variables such as `PROJECT_ROOT`, `PY`, `DATA_ROOT`, `CROWD_ROOT`,
`OUT`, and `GPU_ID` can be overridden without editing Python source files.

## Dataset-Specific CIFAR Launchers

The CIFAR main-table conditions are also split into dataset-specific scripts:

- `reproduce_srse_cifar10_experiments.sh`: CIFAR-10 SRSE main table over
  `q in {0.1, 0.3, 0.5}` and `eta in {0.1, 0.2, 0.3}`.
- `reproduce_srse_cifar100_experiments.sh`: CIFAR-100 SRSE main table over
  the public manuscript grid:
  `q=0.01, eta=0.0/0.1/0.2/0.3`;
  `q=0.03, eta=0.1/0.2/0.3`;
  `q=0.05, eta=0.0/0.1/0.2/0.3/0.4/0.5`;
  and `q=0.1, eta=0.0`.
- `reproduce_srse_cifar100h_experiments.sh`: CIFAR100H SRSE main condition
  `q=0.5, eta=0.2`.

Each script accepts `main_table`, `all`, `help`, and per-condition targets
such as `q005_eta03`. The scripts use the same runtime overrides as the
unified launcher, plus `SEEDS` for changing the seed list.

## Maintenance Rule

When adding or changing an experiment command, update the launcher or add a
separate command file in this directory. Do not add long command blocks to the
top of `main.py`, ablation scripts, persistent-state scripts, or local-only
comparison scripts.
