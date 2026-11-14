"""
train_with_history.py -- training with per-epoch logging (V4 of the NCMCS 2026 paper).

Model, loss, optimizer, scheduler and checkpoint rule are the same as in V3:
  ResNet-18 (model.py), class-weighted cross-entropy, AdamW (lr 1e-4, weight decay 1e-3),
  ReduceLROnPlateau (factor 0.5, patience 2) on the validation loss, 15 epochs, batch 32,
  checkpoint with the lowest validation loss. Seed 42.

V4 changes (no change to the method):
  * reads the 60/20/20 split from the folder "split" (made by split_dataset.py)
  * the "lr" column is now the learning rate USED during the epoch
    ("lr_after" is the value after the scheduler step), so halvings are easy to read
  * writes results/train_summary.txt with every training number the paper needs

Outputs:
  models/best_model.pth      best checkpoint (used by evaluate.py)
  results/history.csv        one row per epoch (used by plot_curves.py)
  results/train_summary.txt  best epoch, best val loss, LR halvings, accuracies, time
"""
import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from dataset import get_data_loaders, get_class_weights
from model import get_model

SEED = 42
DATA_DIR = "split"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(num_epochs=15, batch_size=32, lr=1e-4):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Training initialized on device: {device} (seed={SEED})")

    train_loader, val_loader, _ = get_data_loaders(data_dir=DATA_DIR, batch_size=batch_size)
    class_weights = get_class_weights(train_loader).to(device)
    print(f"--> Using Class-Weighted Loss: {class_weights}")

    model = get_model(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_loss = float("inf")
    best_epoch = 0
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    best_model_path = "models/best_model.pth"

    fields = ["epoch", "lr", "lr_after", "train_loss", "train_acc", "val_loss", "val_acc",
              "val_precision_diseased", "val_recall_diseased", "val_f1_diseased",
              "val_precision_healthy", "val_recall_healthy", "val_f1_healthy", "seconds"]
    history = []
    t_start = time.time()

    print(f"\n--- Starting Training for {num_epochs} Epochs ---")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        lr_used = optimizer.param_groups[0]["lr"]

        # ---- train ----
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        train_loss, train_acc = run_loss / total, correct / total

        # ---- validate ----
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        v_true, v_pred = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                v_loss += criterion(outputs, labels).item() * images.size(0)
                preds = outputs.argmax(1)
                v_correct += preds.eq(labels).sum().item()
                v_total += labels.size(0)
                v_true.extend(labels.cpu().numpy())
                v_pred.extend(preds.cpu().numpy())
        val_loss, val_acc = v_loss / v_total, v_correct / v_total
        p, r, f, _ = precision_recall_fscore_support(v_true, v_pred, labels=[0, 1], zero_division=0)

        scheduler.step(val_loss)
        lr_after = optimizer.param_groups[0]["lr"]
        secs = time.time() - t0

        history.append({
            "epoch": epoch, "lr": lr_used, "lr_after": lr_after,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_precision_diseased": p[0], "val_recall_diseased": r[0], "val_f1_diseased": f[0],
            "val_precision_healthy": p[1], "val_recall_healthy": r[1], "val_f1_healthy": f[1],
            "seconds": secs,
        })
        print(f"Epoch [{epoch}/{num_epochs}] ({secs:.1f}s) | LR: {lr_used:.2e} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"Val F1 (D/H): {f[0]:.4f}/{f[1]:.4f}")
        if lr_after < lr_used:
            print(f"   [lr] learning rate halved after epoch {epoch}: {lr_used:.2e} -> {lr_after:.2e}")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"   [*] Saving Best Model Checkpoint to {best_model_path}")

    minutes = (time.time() - t_start) / 60

    with open("results/history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(history)

    # ---- summary for the paper ----
    b = history[best_epoch - 1]
    cuts = [h["epoch"] for h in history if h["lr_after"] < h["lr"]]
    lines = [
        f"Seed: {SEED} | data: {DATA_DIR} | epochs: {num_epochs} | batch: {batch_size}",
        f"Class weights (Diseased, Healthy): {[round(x, 4) for x in class_weights.tolist()]}",
        f"Learning rate halved after epoch(s): {cuts if cuts else 'never'}",
        f"Best epoch (lowest val loss): {best_epoch} | best val loss: {best_val_loss:.4f}",
        f"At best epoch: val acc {b['val_acc']*100:.2f}% | val F1 healthy {b['val_f1_healthy']:.3f} "
        f"| train acc {b['train_acc']*100:.2f}%",
        f"Train loss: epoch 1 {history[0]['train_loss']:.4f} -> epoch {num_epochs} {history[-1]['train_loss']:.4f}",
        f"Epoch {num_epochs}: train acc {history[-1]['train_acc']*100:.2f}% | val acc {history[-1]['val_acc']*100:.2f}%",
        f"Total training time: {minutes:.1f} min",
    ]
    with open("results/train_summary.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n--- Training Complete! ---")
    print("\n".join(lines))
    print("History saved to results/history.csv | summary saved to results/train_summary.txt")


if __name__ == "__main__":
    train_model()
