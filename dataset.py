import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_data_loaders(data_dir="split", batch_size=32):
    """
    Sets up PyTorch DataLoaders for the binary classification dataset 
    ('Diseased' vs 'Healthy') with robust data augmentations and normalization.
    """
    
    # 1. Training Transforms with Data Augmentation
    # Augmentations improve model generalization and combat overfitting.
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),               # Standardize dimensions for CNN backbones
        transforms.RandomHorizontalFlip(p=0.5),      # Spatial invariance
        transforms.RandomRotation(degrees=15),       # Orientation robustness
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Lighting robustness
        transforms.ToTensor(),                       # Scale pixels to [0.0, 1.0]
        transforms.Normalize(                        # ImageNet normalization stats
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. Validation & Test Transforms (Deterministic)
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 3. Load Datasets using ImageFolder (Automatically maps subfolder names to labels)
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_path, transform=eval_transforms)
    test_dataset = datasets.ImageFolder(root=test_path, transform=eval_transforms)
    for ds in (train_dataset, val_dataset, test_dataset):
        assert ds.classes == ["Diseased", "Healthy"], f"unexpected class folders: {ds.classes}"

    # 4. Construct DataLoaders optimized for your RTX 4050 GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    # 5. Log Dataset Metrics for Paper Documentation (counted, not hard-coded)
    #    Expected with the 60/20/20 split: train 900/150, val 300/50, test 300/50
    print(f"--- Dataset Loading Summary ---")
    print(f"Detected Classes & Indices: {train_dataset.class_to_idx}")
    for name, ds in (("Training", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)):
        counts = np.bincount(np.array(ds.targets), minlength=len(ds.classes))
        detail = ", ".join(f"{c}: {n}" for c, n in zip(ds.classes, counts))
        print(f"{name + ' Samples:':20s}{len(ds)} ({detail})")

    return train_loader, val_loader, test_loader

def get_class_weights(train_loader):
    """
    Computes class weights to handle the 6:1 class imbalance between 
    Diseased and Healthy samples, preventing model bias during training.
    """
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
        
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    
    return torch.tensor(class_weights, dtype=torch.float)

if __name__ == "__main__":
    # Test script execution
    train_loader, val_loader, test_loader = get_data_loaders()
    weights = get_class_weights(train_loader)
    print(f"Computed Loss Class Weights (to handle imbalance): {weights}")