import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int):
    """Build ResNet50 model for food classification"""
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model

def save_checkpoint(path: str, model: nn.Module, class_names):
    ckpt = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "arch": "resnet50",
    }
    torch.save(ckpt, path)

def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    return ckpt["state_dict"], ckpt["class_names"]
