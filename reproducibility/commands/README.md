# SRSE Reproduction Commands

Keep experiment launch commands in this directory instead of Python file
headers. Python files under `reproducibility/code/` should describe their entry
point role only; runnable command lines belong here.

## Canonical Launcher

Use the unified launcher from the repository root:

```bash
bash reproducibility/commands/reproduce_srse_paper_experiments_20260616.sh help
```

The launcher supports the current paper-facing targets:

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

## Maintenance Rule

When adding or changing an experiment command, update the launcher or add a
separate command file in this directory. Do not add long command blocks to the
top of `main.py`, ablation scripts, persistent-state scripts, or baseline
training scripts.
