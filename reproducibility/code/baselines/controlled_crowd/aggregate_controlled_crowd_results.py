#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import re
import statistics
from pathlib import Path


EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+).*?method=(?P<method>[a-z_]+).*?"
    r"Test Acc:\s+(?P<acc>[0-9.]+)\s+Best Acc:\s+(?P<best>[0-9.]+)\s+Best Epoch:\s+(?P<best_epoch>\d+)"
)


def parse_log(path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "epoch": int(m.group("epoch")),
                "epochs": int(m.group("epochs")),
                "method": m.group("method"),
                "acc": float(m.group("acc")),
                "best_acc": float(m.group("best")),
                "best_epoch": int(m.group("best_epoch")),
            }
        )
    return rows


def infer_meta(log_path):
    parts = log_path.parts
    # .../<root>/<method>/<dataset>/lpi3_slice2_seed1/results.log
    if len(parts) < 5:
        return None
    method = parts[-4]
    dataset = parts[-3]
    m = re.match(r"lpi(?P<lpi>\d+)_slice(?P<slice>\d+)_seed(?P<seed>\d+)", parts[-2])
    if not m:
        return None
    return {
        "method": method,
        "dataset": dataset,
        "lpi": int(m.group("lpi")),
        "slice": int(m.group("slice")),
        "seed": int(m.group("seed")),
    }


def collect(root, expected_epochs=None):
    rows = []
    for log_path in Path(root).rglob("results.log"):
        meta = infer_meta(log_path)
        if not meta:
            continue
        evals = parse_log(log_path)
        if not evals:
            rows.append({**meta, "complete": False, "final_epoch": 0, "final_acc": "", "best_epoch": "", "best_acc": "", "log": str(log_path)})
            continue
        last = evals[-1]
        target_epochs = expected_epochs if expected_epochs is not None else last["epochs"]
        complete = last["epoch"] >= target_epochs
        rows.append(
            {
                **meta,
                "complete": complete,
                "final_epoch": last["epoch"],
                "final_acc": last["acc"],
                "best_epoch": last["best_epoch"],
                "best_acc": last["best_acc"],
                "log": str(log_path),
            }
        )
    rows.sort(key=lambda r: (r["method"], r["dataset"], r["lpi"], r["seed"]))
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["dataset"], row["lpi"], row["slice"]), []).append(row)
    summary = []
    for (method, dataset, lpi, slice_id), items in sorted(groups.items()):
        complete = [r for r in items if r["complete"]]
        finals = [float(r["final_acc"]) for r in complete]
        bests = [float(r["best_acc"]) for r in complete]
        summary.append(
            {
                "method": method,
                "dataset": dataset,
                "lpi": lpi,
                "slice": slice_id,
                "complete_n": len(complete),
                "complete_seeds": " ".join(str(r["seed"]) for r in complete),
                "final_mean": statistics.mean(finals) if finals else "",
                "final_std": statistics.pstdev(finals) if len(finals) > 1 else (0.0 if finals else ""),
                "best_mean": statistics.mean(bests) if bests else "",
                "best_std": statistics.pstdev(bests) if len(bests) > 1 else (0.0 if bests else ""),
            }
        )
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path, summary):
    lines = ["# Controlled Crowd Baseline Summary", ""]
    lines.append("| Method | Dataset | LPI | Slice | Seeds | Final Acc mean +- std | Best Acc mean +- std |")
    lines.append("|---|---|---:|---:|---|---:|---:|")
    for row in summary:
        if row["complete_n"]:
            lines.append(
                f"| {row['method']} | {row['dataset']} | {row['lpi']} | {row['slice']} | {row['complete_seeds']} | "
                f"{row['final_mean']:.2f} +- {row['final_std']:.2f} | "
                f"{row['best_mean']:.2f} +- {row['best_std']:.2f} |"
            )
        else:
            lines.append(
                f"| {row['method']} | {row['dataset']} | {row['lpi']} | {row['slice']} | none | pending | pending |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--is-complete", type=Path)
    parser.add_argument("--expected-epochs", type=int, default=None)
    args = parser.parse_args()

    if args.is_complete:
        evals = parse_log(args.is_complete)
        if not evals:
            raise SystemExit(1)
        expected_epochs = args.expected_epochs if args.expected_epochs is not None else evals[-1]["epochs"]
        raise SystemExit(0 if evals[-1]["epoch"] >= expected_epochs else 1)

    rows = collect(args.root, expected_epochs=args.expected_epochs)
    summary = summarize(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "controlled_crowd_detail.csv", rows)
    write_csv(args.out / "controlled_crowd_summary.csv", summary)
    write_md(args.out / "controlled_crowd_summary.md", summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
