#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


def read_metrics(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def infer_dataset(exp_dir: Path):
    name = exp_dir.name.lower()
    if name.startswith("benthic_"):
        return "Benthic"
    if name.startswith("plankton_"):
        return "Plankton"
    if name.startswith("treeversity_"):
        return "Treeversity"
    return None


def collect(root: Path, expected_epochs: int):
    detail = []
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "summary" and p.name != "launch_logs"):
        dataset = infer_dataset(exp_dir)
        if dataset is None:
            continue
        for seed_dir in sorted(exp_dir.glob("seed_*")):
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except Exception:
                continue
            rows = read_metrics(seed_dir / "epoch_metrics.csv")
            if not rows:
                detail.append({
                    "dataset": dataset,
                    "seed": seed,
                    "complete": False,
                    "final_epoch": 0,
                    "final_acc": "",
                    "best_acc": "",
                    "best_epoch": "",
                    "metrics": str(seed_dir / "epoch_metrics.csv"),
                })
                continue
            parsed = []
            for row in rows:
                parsed.append({
                    "epoch": int(float(row["epoch"])),
                    "test_acc": float(row["test_acc"]),
                    "best_test_acc": float(row["best_test_acc"]),
                })
            last = parsed[-1]
            best = max(parsed, key=lambda x: x["test_acc"])
            detail.append({
                "dataset": dataset,
                "seed": seed,
                "complete": last["epoch"] >= expected_epochs,
                "final_epoch": last["epoch"],
                "final_acc": last["test_acc"],
                "best_acc": best["test_acc"],
                "best_epoch": best["epoch"],
                "metrics": str(seed_dir / "epoch_metrics.csv"),
            })
    return sorted(detail, key=lambda r: (r["dataset"], r["seed"]))


def summarize(detail):
    groups = {}
    for row in detail:
        groups.setdefault(row["dataset"], []).append(row)
    summary = []
    for dataset, rows in sorted(groups.items()):
        complete = [r for r in rows if r["complete"]]
        finals = [float(r["final_acc"]) for r in complete]
        bests = [float(r["best_acc"]) for r in complete]
        summary.append({
            "dataset": dataset,
            "complete_seeds": " ".join(str(r["seed"]) for r in complete),
            "complete_n": len(complete),
            "final_mean": statistics.mean(finals) if finals else "",
            "final_std": statistics.pstdev(finals) if len(finals) > 1 else (0.0 if finals else ""),
            "best_mean": statistics.mean(bests) if bests else "",
            "best_std": statistics.pstdev(bests) if len(bests) > 1 else (0.0 if bests else ""),
        })
    return summary


def write_outputs(detail, summary, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"srse_main_crowd_{tag}_detail.csv"
    summary_path = out_dir / f"srse_main_crowd_{tag}_summary.csv"
    json_path = out_dir / f"srse_main_crowd_{tag}_summary.json"
    md_path = out_dir / f"srse_main_crowd_{tag}_summary.md"

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0].keys()) if detail else ["dataset"])
        writer.writeheader()
        writer.writerows(detail)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["dataset"])
        writer.writeheader()
        writer.writerows(summary)
    json_path.write_text(json.dumps({"detail": detail, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")

    title_tag = tag.upper() if tag.lower().startswith("lpi") else tag
    lines = [f"# SRSE main.py crowd {title_tag} summary", ""]
    lines.append("| Dataset | Seeds | Final Acc mean +- std | Best Acc mean +- std |")
    lines.append("|---|---|---:|---:|")
    for row in summary:
        if row["complete_n"]:
            lines.append(
                f"| {row['dataset']} | {row['complete_seeds']} | "
                f"{row['final_mean']:.2f} +- {row['final_std']:.2f} | "
                f"{row['best_mean']:.2f} +- {row['best_std']:.2f} |"
            )
        else:
            lines.append(f"| {row['dataset']} | none | pending | pending |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument("--expected-epochs", type=int, default=100)
    parser.add_argument("--tag", default="lpi3")
    args = parser.parse_args()

    detail = collect(args.root, args.expected_epochs)
    summary = summarize(detail)
    if args.write_summary:
        write_outputs(detail, summary, args.write_summary, args.tag)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
