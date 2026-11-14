# fish-disease-binary-screening
Binary (Diseased vs Healthy) fish disease screening with a class-weighted ResNet-18 — NCMCS 2026

# Binary Fish Disease Screening with a Class-Weighted ResNet-18

Code and results for the paper:

> **A Binary Deep Learning Framework for Automated Fish Disease Detection via Class Aggregation**
> Abdelkrim Bouannane, Boutkhil Sidaoui, Amine Dahane, Bedr-Eddine Benaissa, and Djelloul Bouchiha.
> *National Conference on Mathematics and Computer Science (NCMCS 2026)*, University of Naâma, Algeria.

Instead of telling several fish diseases apart, the model answers a simpler screening question: **is this fish diseased or healthy?** The six disease classes of a public freshwater fish dataset are merged into one *Diseased* class, and an ImageNet-pretrained ResNet-18 is fine-tuned with a class-weighted cross-entropy loss to handle the 6:1 class imbalance.

---

## Results (test set, 350 images)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Diseased | 99.66% | 99.00% | 0.9933 | 300 |
| Healthy | 94.23% | 98.00% | 0.9608 | 50 |
| **Macro average** | **96.95%** | **98.50%** | **0.9770** | 350 |

**Accuracy: 98.86%** (346 of 350 test images correct). The model missed 3 of 300 diseased fish and labelled 1 of 50 healthy fish as diseased.

| | Predicted Diseased | Predicted Healthy |
|---|---|---|
| **True Diseased** | 297 | 3 |
| **True Healthy** | 1 | 49 |

Figures: [`figures/confusion_matrix.pdf`](figures/confusion_matrix.pdf) · [`figures/training_curves.pdf`](figures/training_curves.pdf)
Full numbers: [`results/test_metrics.txt`](results/test_metrics.txt) · [`results/train_summary.txt`](results/train_summary.txt) · [`results/history.csv`](results/history.csv)

These results come from a single training run (seed 42) on one dataset of curated images; see the paper for the limitations.

---

## Dataset

**Freshwater Fish Disease Aquaculture in South Asia** (Biswas et al., 2024), available on Kaggle under a **CC0: Public Domain** licence:
https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia

We use the 1,750 images of its `Train` folder (seven classes, 250 images each):

| Original class | Binary label |
|---|---|
| Bacterial Red Disease, Bacterial Aeromoniasis, Bacterial Gill Disease, Fungal Saprolegniasis, Parasitic Diseases, Viral White Tail Disease | **Diseased** (1,500) |
| Healthy Fish | **Healthy** (250) |

The images are **not** included in this repository; download them from Kaggle.

### Split (60/20/20, stratified, seed 42)

| Split | Diseased | Healthy | Total |
|---|---|---|---|
| Training | 900 | 150 | 1,050 |
| Validation | 300 | 50 | 350 |
| Test | 300 | 50 | 350 |
| **Total** | **1,500** | **250** | **1,750** |

[`split/split_files.csv`](split/split_files.csv) lists the set (train / val / test) of every image, so the exact split can be checked or rebuilt.

---

## Repository structure

```
├── split_dataset.py        # 60/20/20 stratified split (seed 42)
├── dataset.py              # data loaders, augmentation, class weights
├── model.py                # ResNet-18 with Dropout(0.4) + 2-class head
├── train_with_history.py   # training with per-epoch logging
├── evaluate.py             # test-set metrics and confusion matrix
├── plot_curves.py          # training curves figure
├── requirements.txt
├── split/split_files.csv   # file list of the three sets
├── results/                # history.csv, train_summary.txt, test_metrics.txt
└── figures/                # confusion_matrix.pdf, training_curves.pdf
```

---

## How to reproduce

### 1. Install

Python 3.14 was used (see versions below).

```bash
pip install -r requirements.txt
```

For GPU training, install PyTorch for your CUDA version first, following https://pytorch.org/get-started/locally/ (we used CUDA 12.6).

### 2. Prepare the data

Download the dataset from Kaggle and, from its `Train` folder, put the images in:

```
data/
├── Diseased fish/   # all images of the six disease classes (1,500)
└── Healthy fish/    # all images of Healthy Fish (250)
```

### 3. Split, train, evaluate

Run these commands from the project folder:

```bash
python split_dataset.py --src data --out split
python train_with_history.py
python evaluate.py
python plot_curves.py
```

| Step | Output |
|---|---|
| `split_dataset.py` | `split/{train,val,test}/{Diseased,Healthy}/`, `split/split_files.csv` |
| `train_with_history.py` | `models/best_model.pth`, `results/history.csv`, `results/train_summary.txt` |
| `evaluate.py` | `results/test_metrics.txt`, `figures/confusion_matrix.pdf` |
| `plot_curves.py` | `figures/training_curves.pdf` |

On an NVIDIA RTX 4050 laptop GPU, training takes about 4 minutes. cuDNN deterministic mode is not enabled, so a rerun can differ slightly.

---

## Training settings

| Setting | Value |
|---|---|
| Backbone | ResNet-18, ImageNet weights, all layers fine-tuned |
| Head | Dropout (p = 0.4) + Linear (512 → 2) |
| Input | 224 × 224, ImageNet normalisation |
| Augmentation (training only) | horizontal flip (p = 0.5), rotation ±15°, brightness and contrast jitter 0.1 |
| Loss | cross-entropy with class weights w_c = N / (2 N_c): Diseased 0.5833, Healthy 3.5 |
| Optimiser | AdamW, learning rate 1e-4, weight decay 1e-3 |
| Scheduler | ReduceLROnPlateau on validation loss (factor 0.5, patience 2) |
| Epochs / batch size | 15 / 32, no early stopping |
| Checkpoint | lowest validation loss |
| Seed | 42 |

Software used: Python 3.14.4, PyTorch 2.14.0, torchvision 0.29.0, scikit-learn 1.9.1, NumPy 2.5.2, Windows 11.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{bouannane2026binary,
  title     = {A Binary Deep Learning Framework for Automated Fish Disease Detection via Class Aggregation},
  author    = {Bouannane, Abdelkrim and Sidaoui, Boutkhil and Dahane, Amine and Benaissa, Bedr-Eddine and Bouchiha, Djelloul},
  booktitle = {National Conference on Mathematics and Computer Science (NCMCS 2026)},
  year      = {2026}
}
```

Related work by the same authors:

```bibtex
@inproceedings{bouannane2026review,
  title     = {Fish Disease Detection in Smart Aquaculture: A Review of Deep Learning, Federated Learning, and IoT Integration},
  author    = {Bouannane, Abdelkrim and Sidaoui, Boutkhil and Dahane, Amine and Benaissa, Bedr-Eddine and Bouchiha, Djelloul},
  booktitle = {10th International Conference on Scientific and Academic Research},
  address   = {Konya, Turkey},
  year      = {2026}
}
```

Please also cite the dataset:

> S. Biswas et al., "Empirical evaluation of deep learning techniques for fish disease detection in aquaculture systems: A transfer learning and fusion-based approach," *IEEE Access*, vol. 12, pp. 176136–176154, 2024. doi: [10.1109/ACCESS.2024.3504283](https://doi.org/10.1109/ACCESS.2024.3504283)

---

## Licence

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The dataset belongs to its authors and is distributed on Kaggle under CC0: Public Domain.

## Contact

Abdelkrim Bouannane — Department of Computer Science, University of Naâma – SALHI Ahmed, Algeria
bouannane.abdelkrim@cuniv-naama.dz
