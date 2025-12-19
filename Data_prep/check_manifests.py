#!/usr/bin/env python3
"""
check_manifests.py

Quick sanity checks for prepared dataset manifests:
- Count tiles in train/val/extra
- Count unique src_id per split
- Detect overlapping src_id (data leakage risk)
- Optionally recompute class pixel counts from masks (slow; use --class-counts)
"""

import argparse
from pathlib import Path
import csv
from collections import Counter, defaultdict

def read_csv_rows(path: Path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def summarize_split(name: str, rows):
    n = len(rows)
    src_ids = [r["src_id"] for r in rows]
    uniq = set(src_ids)
    print(f"\n[{name}]")
    print(f"  tiles: {n}")
    print(f"  unique src_id: {len(uniq)}")
    # show top-5 most frequent src_ids
    cnt = Counter(src_ids)
    top5 = cnt.most_common(5)
    print(f"  top-5 src_id by tile count: {top5}")
    return uniq

def find_leakage(src_train, src_val):
    overlap = src_train.intersection(src_val)
    if overlap:
        print("\n[LEAKAGE] src_id present in BOTH train and val:")
        for i, sid in enumerate(sorted(overlap)):
            print(" ", sid)
            if i >= 19:
                print("  ... (truncated)")
                break
    else:
        print("\n[LEAKAGE] None detected ?")

def sample_rows(rows, k=5, fields=("image_path","mask_path","src_id","y","x")):
    print(f"\nSample {k} rows:")
    for i, r in enumerate(rows[:k]):
        line = ", ".join(f"{f}={r.get(f,'')}" for f in fields if f in r)
        print(" ", line)

def recompute_class_counts(rows):
    """Slow: open each mask png and count IDs (0..3)."""
    import numpy as np
    from PIL import Image
    counts = [0,0,0,0]
    for i, r in enumerate(rows):
        mp = r.get("mask_path")
        if not mp:
            continue
        arr = np.array(Image.open(mp), dtype=np.uint8)
        for c in range(4):
            counts[c] += int((arr == c).sum())
        if (i+1) % 1000 == 0:
            print(f"  processed {i+1} masks...", end="\r")
    print()
    total = sum(counts)
    pct = [ (c/total*100 if total>0 else 0.0) for c in counts ]
    print("\n[Class pixel counts (recomputed)]")
    print(f"  background: {counts[0]} ({pct[0]:.2f}%)")
    print(f"  stroma:     {counts[1]} ({pct[1]:.2f}%)")
    print(f"  benign:     {counts[2]} ({pct[2]:.2f}%)")
    print(f"  tumor:      {counts[3]} ({pct[3]:.2f}%)")
    return counts

def main():
    ap = argparse.ArgumentParser(description="Check prepared dataset manifests.")
    ap.add_argument("--manifests-dir", required=True, type=Path,
                    help="Path to processed/manifests/ (contains train.csv, val.csv, extra.csv)")
    ap.add_argument("--samples", type=int, default=5, help="Print N sample rows per split")
    ap.add_argument("--class-counts", action="store_true",
                    help="Recompute class pixel counts from masks (slow)")
    args = ap.parse_args()

    train_csv = args.manifests_dir / "train.csv"
    val_csv   = args.manifests_dir / "val.csv"
    extra_csv = args.manifests_dir / "extra.csv"

    rows_train = read_csv_rows(train_csv) if train_csv.exists() else []
    rows_val   = read_csv_rows(val_csv)   if val_csv.exists()   else []
    rows_extra = read_csv_rows(extra_csv) if extra_csv.exists() else []

    # Summaries
    src_train = summarize_split("train", rows_train) if rows_train else set()
    if rows_train and args.samples: sample_rows(rows_train, k=args.samples)

    src_val = summarize_split("val", rows_val) if rows_val else set()
    if rows_val and args.samples: sample_rows(rows_val, k=args.samples)

    if rows_extra:
        # extra has no masks
        print(f"\n[extra]")
        print(f"  tiles: {len(rows_extra)}")
        src_extra = set(r["src_id"] for r in rows_extra)
        print(f"  unique src_id: {len(src_extra)}")
        if args.samples: sample_rows(rows_extra, k=args.samples, fields=("image_path","src_id","y","x"))

    # Leakage check
    if rows_train and rows_val:
        find_leakage(src_train, src_val)

    # Optional: class counts
    if args.class_counts and rows_train:
        recompute_class_counts(rows_train)

    print("\nDone.")

if __name__ == "__main__":
    main()
