#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path

ACC_RE = re.compile(r"top1 Accuracy: ([0-9.]+)/([0-9.]+) \(([0-9.]+)%\)")
EPOCH_RE = re.compile(r"Train Epoch: (\d+) ")


def parse_log(path: Path):
    epoch = 0
    evals = []
    text = path.read_text(errors="ignore")
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epoch = int(m.group(1))
        m = ACC_RE.search(line)
        if m:
            evals.append(
                {
                    "epoch": epoch,
                    "correct": float(m.group(1)),
                    "total": float(m.group(2)),
                    "acc": float(m.group(3)),
                }
            )
    return evals


def complete(path: Path, expected_epochs: int = 100) -> bool:
    evals = parse_log(path)
    return bool(evals) and evals[-1]["epoch"] >= expected_epochs and len(evals) >= expected_epochs


def infer_meta(path: Path):
    name = None
    for part in path.parts:
        if part.startswith("metricsR50_"):
            name = part.removeprefix("metricsR50_")
            break
    if not name:
        return None
    m = re.match(
        r"(?P<dataset>[A-Za-z]+)_lpi(?P<lpi>\d+)_slice(?P<slice>\d+)_seed(?P<seed>\d+)_officialparams_pals_",
        name,
    )
    if not m:
        return None
    return {
        "dataset": m.group("dataset"),
        "lpi": int(m.group("lpi")),
        "slice": int(m.group("slice")),
        "seed": int(m.group("seed")),
    }


def collect(root: Path):
    rows = []
    for log in root.rglob("results.log"):
        meta = infer_meta(log)
        if not meta:
            continue
        evals = parse_log(log)
        if not evals:
            continue
        best = max(evals, key=lambda x: x["acc"])
        last = evals[-1]
        rows.append(
            {
                **meta,
                "complete": complete(log),
                "num_eval": len(evals),
                "final_epoch": last["epoch"],
                "final_acc": last["acc"],
                "best_epoch": best["epoch"],
                "best_acc": best["acc"],
                "log": str(log),
            }
        )
    rows.sort(key=lambda r: (r["dataset"], r["lpi"], r["seed"]))
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["dataset"], row["lpi"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (dataset, lpi), items in sorted(groups.items()):
        complete_items = [r for r in items if r["complete"]]
        final_vals = [r["final_acc"] for r in complete_items]
        best_vals = [r["best_acc"] for r in complete_items]
        summary.append(
            {
                "dataset": dataset,
                "lpi": lpi,
                "complete_seeds": " ".join(str(r["seed"]) for r in complete_items),
                "complete_n": len(complete_items),
                "final_mean": statistics.mean(final_vals) if final_vals else "",
                "final_std": statistics.pstdev(final_vals) if len(final_vals) > 1 else (0.0 if final_vals else ""),
                "best_mean": statistics.mean(best_vals) if best_vals else "",
                "best_std": statistics.pstdev(best_vals) if len(best_vals) > 1 else (0.0 if best_vals else ""),
            }
        )
    return summary


def write_outputs(rows, summary, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = out_dir / "pals_officialparams_fold2_lpi3_detail.csv"
    summary_csv = out_dir / "pals_officialparams_fold2_lpi3_summary.csv"
    with detail_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["dataset"])
        writer.writeheader()
        writer.writerows(summary)
    md = out_dir / "pals_officialparams_fold2_lpi3_summary.md"
    lines = ["# PALS Official-Params Fold2 LPI=3 Summary", ""]
    lines.append("| Dataset | LPI | Seeds | Final Acc mean +- std | Best Acc mean +- std |")
    lines.append("|---|---:|---|---:|---:|")
    for row in summary:
        if row["complete_n"]:
            lines.append(
                f"| {row['dataset']} | {row['lpi']} | {row['complete_seeds']} | "
                f"{row['final_mean']:.2f} +- {row['final_std']:.2f} | "
                f"{row['best_mean']:.2f} +- {row['best_std']:.2f} |"
            )
        else:
            lines.append(f"| {row['dataset']} | {row['lpi']} | none | pending | pending |")
    md.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--is-complete", type=Path)
    args = parser.parse_args()

    if args.is_complete:
        raise SystemExit(0 if complete(args.is_complete) else 1)
    rows = collect(args.root)
    summary = summarize(rows)
    if args.write_summary:
        write_outputs(rows, summary, args.write_summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
