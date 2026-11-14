"""
evaluate.py -- test-set evaluation of the best checkpoint (V4 of the NCMCS 2026 paper).

V4 changes:
  * reads the test set from the folder "split" (made by split_dataset.py)
  * reports the metrics used in the paper: accuracy and per-class precision,
    recall and F1-score (+ macro average); ROC-AUC removed
  * confusion matrix printed with the correct labels (Diseased = positive class)
  * writes results/test_metrics.txt
  * confusion-matrix figure without an internal title (the paper caption describes it)

Outputs:
  results/test_metrics.txt
  figures/confusion_matrix.pdf
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataset import get_data_loaders
from model import get_model

DATA_DIR = "split"
CLASS_NAMES = ["Diseased", "Healthy"]   # index 0 = Diseased (positive class), 1 = Healthy


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Evaluating model on device: {device}")

    _, _, test_loader = get_data_loaders(data_dir=DATA_DIR, batch_size=32)

    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print(f"[!] Error: Model checkpoint not found at {model_path}. Please run train_with_history.py first.")
        return

    model = get_model(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    print("\n--- Running Inference on Test Set ---")
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    y, p = np.array(all_labels), np.array(all_preds)

    cm = confusion_matrix(y, p, labels=[0, 1])
    acc = accuracy_score(y, p)
    report = classification_report(y, p, labels=[0, 1], target_names=CLASS_NAMES, digits=4)

    lines = [
        "========================================",
        "        TEST-SET METRICS (paper)        ",
        "========================================",
        f"Test images: {len(y)} (Diseased: {(y == 0).sum()}, Healthy: {(y == 1).sum()})",
        f"Accuracy: {acc*100:.2f}% ({(y == p).sum()} of {len(y)} correct)",
        "",
        "Classification report (precision, recall, F1-score):",
        report,
        "Confusion matrix (rows = true label, columns = predicted label):",
        f"                 pred Diseased   pred Healthy",
        f"true Diseased    {cm[0, 0]:13d}   {cm[0, 1]:12d}",
        f"true Healthy     {cm[1, 0]:13d}   {cm[1, 1]:12d}",
        "",
        f"Diseased fish missed (predicted Healthy): {cm[0, 1]} of {cm[0].sum()}",
        f"Healthy fish labelled Diseased:           {cm[1, 0]} of {cm[1].sum()}",
        "========================================",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs("results", exist_ok=True)
    with open("results/test_metrics.txt", "w") as fh:
        fh.write(text + "\n")
    print("[*] Metrics saved to results/test_metrics.txt")

    plot_and_save_confusion_matrix(cm)


def plot_and_save_confusion_matrix(cm):
    """Saves a vector PDF confusion matrix for LaTeX (no title: the caption describes it)."""
    plt.figure(figsize=(4.5, 3.8), dpi=300)
    sns.set_theme(style="white", font_scale=1.1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=1, linecolor="gray", annot_kws={"size": 14})
    plt.xlabel("Predicted label", fontsize=11, labelpad=8)
    plt.ylabel("True label", fontsize=11, labelpad=8)
    os.makedirs("figures", exist_ok=True)
    fig_path = "figures/confusion_matrix.pdf"
    plt.savefig(fig_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"[*] Confusion matrix figure saved to {fig_path}")


if __name__ == "__main__":
    evaluate_model()
