"""
Plot training curves for the paper.

Usage
  python plot_curves.py                      # reads results/history.csv (from train_with_history.py)
  python plot_curves.py --log train_log.txt  # parses a saved console log of your ORIGINAL train.py run
                                             # (lines like: Epoch [3/15] (12.3s) | Train Loss: ... | Val Acc: ...%)

Output: figures/training_curves.pdf  (two panels: loss and accuracy; best epoch marked)
"""
import argparse
import os
import re
import csv
import matplotlib.pyplot as plt

LINE = re.compile(
    r"Epoch \[(\d+)/\d+\].*?Train Loss: ([\d.]+) \| Train Acc: ([\d.]+)% \| "
    r"Val Loss: ([\d.]+) \| Val Acc: ([\d.]+)%"
)


def from_csv(path):
    rows = list(csv.DictReader(open(path)))
    return ([int(r["epoch"]) for r in rows],
            [float(r["train_loss"]) for r in rows],
            [float(r["val_loss"]) for r in rows],
            [100 * float(r["train_acc"]) for r in rows],
            [100 * float(r["val_acc"]) for r in rows])


def from_log(path):
    ep, tl, vl, ta, va = [], [], [], [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = LINE.search(line)
        if m:
            ep.append(int(m.group(1)))
            tl.append(float(m.group(2)))
            ta.append(float(m.group(3)))
            vl.append(float(m.group(4)))
            va.append(float(m.group(5)))
    if not ep:
        raise SystemExit("No epoch lines found in the log.")
    return ep, tl, vl, ta, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/history.csv")
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default="figures/training_curves.pdf")
    a = ap.parse_args()

    ep, tl, vl, ta, va = from_log(a.log) if a.log else from_csv(a.csv)
    best = min(range(len(vl)), key=lambda i: vl[i])  # same rule as the checkpoint: lowest val loss

    train_c, val_c = "#0072B2", "#D55E00"  # colour-blind-safe pair
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), dpi=300)

    ax[0].plot(ep, tl, "-o", ms=3, color=train_c, label="Training")
    ax[0].plot(ep, vl, "-s", ms=3, color=val_c, label="Validation")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Weighted cross-entropy loss")
    ax[0].set_title("(a) Loss")

    ax[1].plot(ep, ta, "-o", ms=3, color=train_c, label="Training")
    ax[1].plot(ep, va, "-s", ms=3, color=val_c, label="Validation")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title("(b) Accuracy")

    for x in ax:
        x.axvline(ep[best], color="gray", ls="--", lw=0.8, label=f"Best checkpoint (epoch {ep[best]})")
        x.grid(alpha=0.3)
        x.set_xticks(ep)
        x.legend(frameon=False, fontsize=8)
        for s in ("top", "right"):
            x.spines[s].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", format="pdf")
    print(f"Saved {a.out} | best epoch (lowest val loss) = {ep[best]}")


if __name__ == "__main__":
    main()
