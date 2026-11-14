import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, pretrained=True):
    """
    Constructs a binary classification model using a pre-trained ResNet-18 backbone.
    Modified the final classification head to output 2 classes (Diseased, Healthy).
    """
    # Load ResNet-18 with modern PyTorch weights syntax
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze early feature extraction layers if desired (optional transfer learning strategy)
    # For full fine-tuning over your year of experiments, you can leave them unfrozen.
    for param in model.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer (fc) to match your binary task (2 output classes)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),  # Dropout regularization to prevent overfitting
        nn.Linear(in_features, num_classes)
    )

    return model

if __name__ == "__main__":
    # Test model instantiation and forward pass on GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    
    # Create a dummy batch (Batch Size: 4, Channels: 3, Height: 224, Width: 224)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"Model successfully initialized on device: {device}")
    print(f"Dummy batch output shape (Batch Size, Classes): {output.shape}")