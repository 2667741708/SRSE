#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


SUMMARY_RE = re.compile(r"Epoch\s+0*(\d+)\s+Summary:\s+Acc=([0-9]+(?:\.[0-9]+)?)%")
TEST_ACC_RE = re.compile(r"Epoch\s+0*(\d+)\s*/\s*\d+.*?Test Acc:\s*([0-9]+(?:\.[0-9]+)?)")
PICO_RE = re.compile(r"Epoch\s+0*(\d+)\s*:\s*Acc\s+([0-9]+(?:\.[0-9]+)?)")
PALS_EPOCH_RE = re.compile(r"Train Epoch:\s*0*(\d+)\b")
PALS_ACC_RE = re.compile(r"top1 Accuracy:\s*[0-9.]+/[0-9.]+\s*\(([0-9]+(?:\.[0-9]+)?)%\)")


def read_epoch_metrics_csv(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("epoch") or not row.get("test_acc"):
                continue
            rows.append((int(float(row["epoch"])), float(row["test_acc"])))
    return rows


def read_log(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    current_epoch: int | None = None
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        pals_epoch = PALS_EPOCH_RE.search(line)
        if pals_epoch:
            current_epoch = int(pals_epoch.group(1))

        for pattern in (SUMMARY_RE, TEST_ACC_RE, PICO_RE):
            match = pattern.search(line)
            if match:
                rows.append((int(match.group(1)), float(match.group(2))))
                break
        else:
            pals_acc = PALS_ACC_RE.search(line)
            if pals_acc and current_epoch is not None:
                rows.append((current_epoch, float(pals_acc.group(1))))
    return rows


def read_records(path: Path) -> list[tuple[int, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.name == "epoch_metrics.csv":
        records = read_epoch_metrics_csv(path)
    else:
        records = read_log(path)
    if not records:
        raise ValueError(f"no test accuracy records found in {path}")
    return records


def first_n(path: Path, n: int) -> list[tuple[int, float]]:
    return read_records(path)[:n]


def print_records(path: Path, n: int) -> int:
    for idx, (epoch, acc) in enumerate(first_n(path, n), start=1):
        print(f"{idx},epoch={epoch},test_acc={acc:.6f}")
    return 0


def compare(reference: Path, candidate: Path, n: int, tolerance: float) -> int:
    ref = first_n(reference, n)
    cand = first_n(candidate, n)
    if len(ref) < n:
        raise ValueError(f"reference has only {len(ref)} test accuracy records, expected {n}: {reference}")
    if len(cand) < n:
        raise ValueError(f"candidate has only {len(cand)} test accuracy records, expected {n}: {candidate}")

    failed = False
    print("idx,reference_epoch,reference_acc,candidate_epoch,candidate_acc,abs_diff,status")
    for idx, ((ref_epoch, ref_acc), (cand_epoch, cand_acc)) in enumerate(zip(ref, cand), start=1):
        diff = abs(ref_acc - cand_acc)
        ok = diff <= tolerance
        failed = failed or not ok
        status = "PASS" if ok else "FAIL"
        print(f"{idx},{ref_epoch},{ref_acc:.6f},{cand_epoch},{cand_acc:.6f},{diff:.6f},{status}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract or compare per-epoch test accuracy records.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract_p = sub.add_parser("extract")
    extract_p.add_argument("path", type=Path)
    extract_p.add_argument("--epochs", type=int, default=2)

    compare_p = sub.add_parser("compare")
    compare_p.add_argument("--reference", type=Path, required=True)
    compare_p.add_argument("--candidate", type=Path, required=True)
    compare_p.add_argument("--epochs", type=int, default=2)
    compare_p.add_argument("--tolerance", type=float, default=0.01)

    args = parser.parse_args(argv)
    if args.cmd == "extract":
        return print_records(args.path, args.epochs)
    if args.cmd == "compare":
        return compare(args.reference, args.candidate, args.epochs, args.tolerance)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
