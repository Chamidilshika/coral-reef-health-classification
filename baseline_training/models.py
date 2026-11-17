import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B0_Weights
)
import timm


def build_model(model_name):

    if model_name == "resnet50":

        model = models.resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            2
        )

    elif model_name == "efficientnet_b0":

        model = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            2
        )

    elif model_name == "vit_small":

        model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=2
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return model


def freeze_backbone(model, model_name):

    for parameter in model.parameters():
        parameter.requires_grad = False

    # Keep the classification head trainable
    if model_name == "resnet50":

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif model_name == "efficientnet_b0":

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    elif model_name == "vit_small":

        for parameter in model.head.parameters():
            parameter.requires_grad = True


def unfreeze_selected_layers(model, model_name):

    # First freeze everything
    for parameter in model.parameters():
        parameter.requires_grad = False

    if model_name == "resnet50":

        # Fine-tune the final residual block
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif model_name == "efficientnet_b0":

        # Fine-tune the final feature blocks
        for parameter in model.features[-2:].parameters():
            parameter.requires_grad = True

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    elif model_name == "vit_small":

        # Fine-tune the final transformer block
        for parameter in model.blocks[-1].parameters():
            parameter.requires_grad = True

        for parameter in model.norm.parameters():
            parameter.requires_grad = True

        for parameter in model.head.parameters():
            parameter.requires_grad = True


def count_trainable_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )