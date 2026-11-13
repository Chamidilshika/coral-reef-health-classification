import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B0_Weights
)
import timm

def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

def load_resnet50():
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Remove the final classification layer
    model.fc = nn.Identity()

    return model

def load_efficientnet_b0():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # Remove the final classification layer
    model.classifier = nn.Identity()

    return model

def load_vit_small():
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=0
    )

    return model

def load_all_models():
    device = get_device()

    models_dict = {
        "resnet50": load_resnet50(),
        "efficientnet_b0": load_efficientnet_b0(),
        "vit_small": load_vit_small()
    }

    for name in models_dict:
        models_dict[name] = models_dict[name].to(device)
        models_dict[name].eval()

    return models_dict, device