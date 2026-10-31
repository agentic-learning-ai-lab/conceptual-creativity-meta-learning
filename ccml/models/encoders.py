"""Encoder models for CCML Appraiser."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models


class MNISTEncoder(nn.Module):
    """
    LeNet-style CNN encoder for MNIST images.
    Handles 32x32 grayscale images.
    """

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        self.main = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(self, x):
        features = self.main(x)
        logits = self.classifier(features)
        return features, logits

    def encode(self, x):
        """Return only features without classification head."""
        return self.main(x)


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder that extracts intermediate features.
    Uses pretrained ResNet and exposes layer2, layer3, and layer4 features.

    Args:
        resnet_variant: Which ResNet to use ("resnet18", "resnet34", "resnet50")
        pretrained: Whether to use ImageNet pretrained weights
        modify_first_conv: If True, change conv1 stride from 2 to 1 (reduces downsampling)
        remove_maxpool: If True, replace maxpool with identity (reduces downsampling)
    """

    def __init__(
        self,
        resnet_variant: str = "resnet50",
        pretrained: bool = True,
        modify_first_conv: bool = False,
        remove_maxpool: bool = False,
    ):
        super().__init__()

        self.resnet_variant = resnet_variant
        self.modify_first_conv = modify_first_conv
        self.remove_maxpool = remove_maxpool

        # Load pretrained ResNet
        if resnet_variant == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.layer2_channels = 512
            self.layer3_channels = 1024
            self.layer4_channels = 2048
        elif resnet_variant == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            self.layer2_channels = 128
            self.layer3_channels = 256
            self.layer4_channels = 512
        elif resnet_variant == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.layer2_channels = 128
            self.layer3_channels = 256
            self.layer4_channels = 512
        else:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")

        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Optionally modify first conv stride
        if modify_first_conv:
            old_conv = self.conv1
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=1, padding=3, bias=False
            )
            if pretrained:
                self.conv1.weight.data = old_conv.weight.data

        # Optionally remove maxpool
        if remove_maxpool:
            self.maxpool = nn.Identity()

        # For classification (optional)
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x, return_intermediates: bool = False):
        """
        Forward pass through ResNet.

        Args:
            x: Input image tensor [B, 3, H, W]
            return_intermediates: If True, return layer2, layer3, and layer4 features

        Returns:
            If return_intermediates=False: layer4_features
            If return_intermediates=True: (layer2_features, layer3_features, layer4_features)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        layer2_features = self.layer2(x)
        layer3_features = self.layer3(layer2_features)
        layer4_features = self.layer4(layer3_features)

        if return_intermediates:
            return layer2_features, layer3_features, layer4_features
        else:
            return layer4_features

    def forward_classifier(self, x):
        """Forward pass with classification head."""
        layer4_features = self.forward(x, return_intermediates=False)
        x = self.avgpool(layer4_features)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

    def get_spatial_size(self, img_size: int, layer_name: str) -> int:
        """Get spatial size for any layer given input image size."""
        if layer_name == "pixels":
            return img_size

        size = img_size

        # conv1: stride 2 (unless modified)
        if not self.modify_first_conv:
            size = size // 2

        # maxpool: stride 2 (unless removed)
        if not self.remove_maxpool:
            size = size // 2

        # layer1: no stride
        # layer2: stride 2
        size = size // 2

        if layer_name == "layer2":
            return size

        # layer3: stride 2
        size = size // 2

        if layer_name == "layer3":
            return size

        # layer4: stride 2
        size = size // 2

        if layer_name == "layer4":
            return size

        raise ValueError(f"Unknown layer: {layer_name}")

    def get_channels(self, layer_name: str) -> int:
        """Get number of channels for a layer."""
        if layer_name == "layer2":
            return self.layer2_channels
        elif layer_name == "layer3":
            return self.layer3_channels
        elif layer_name == "layer4":
            return self.layer4_channels
        elif layer_name == "pixels":
            return 3
        else:
            raise ValueError(f"Unknown layer: {layer_name}")


