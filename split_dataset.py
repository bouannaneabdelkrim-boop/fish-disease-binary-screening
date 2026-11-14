#!/usr/bin/env python3
"""
split_dataset.py -- stratified 60/20/20 split for the NCMCS 2026 paper (V4).

Input : data/Diseased fish (1,500 images, the six disease classes merged)
        data/Healthy fish  (250 images)
Output: OUT/{train,val,test}/{Diseased,Healthy}/  (files are copied, the
        source folder name is kept as a filename prefix) and
        OUT/split_files.csv  (file list of the three sets, to release with the code).

The split is stratified on the binary label (seed 42), so every set keeps the
6:1 ratio: Diseased 900/300/300 and Healthy 150/50/50 (Table 1 of the paper).
The warning "found 2 class folders, expected 7" is expected with merged folders.

Usage:
  python split_dataset.py --src data --out split
"""
import argparse, csv, os, random, shutil, sys
from collections import Counter

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.60, 0.20, 0.20])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"output folder {a.out} is not empty")

    classes = sorted(d for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)))
    if len(classes) != 7:
        print(f"warning: found {len(classes)} class folders, expected 7: {classes}")
    rng = random.Random(a.seed)
    rows = []
    for fine in classes:
        files = sorted(f for f in os.listdir(os.path.join(a.src, fine))
                       if os.path.splitext(f)[1].lower() in EXTS)
        rng.shuffle(files)
        n = len(files)
        n_val, n_test = round(a.ratios[1] * n), round(a.ratios[2] * n)
        binary = "Healthy" if "healthy" in fine.lower() else "Diseased"
        for k, f in enumerate(files):
            split = "test" if k < n_test else ("val" if k < n_test + n_val else "train")
            dst_dir = os.path.join(a.out, split, binary)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f"{fine.replace(' ', '_')}__{f}")
            shutil.copy2(os.path.join(a.src, fine, f), dst)
            rows.append({"file": os.path.join(fine, f), "class": fine, "binary": binary, "split": split})

    with open(os.path.join(a.out, "split_files.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "class", "binary", "split"])
        w.writeheader()
        w.writerows(rows)

    c = Counter((r["binary"], r["split"]) for r in rows)
    print(f"{'':10s} {'train':>6s} {'val':>5s} {'test':>5s} {'total':>6s}")
    for b in ("Diseased", "Healthy"):
        tr, va, te = c[(b, "train")], c[(b, "val")], c[(b, "test")]
        print(f"{b:10s} {tr:6d} {va:5d} {te:5d} {tr + va + te:6d}")
    ntr = c[("Diseased", "train")] + c[("Healthy", "train")]
    for b in ("Diseased", "Healthy"):
        print(f"class weight {b}: {ntr}/(2*{c[(b, 'train')]}) = {ntr / (2 * c[(b, 'train')]):.4f}")


if __name__ == "__main__":
    main()
