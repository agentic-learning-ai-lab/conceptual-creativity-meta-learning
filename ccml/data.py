"""Data loading utilities for CCML.

All loaders output images in [0, 1] range by default for consistency.
Use normalize=True to get [-1, 1] range (for training diffusion models, etc.).
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


def get_mnist_loaders(
    batch_size: int = 128,
    img_size: int = 32,
    use_augmentation: bool = True,
    shuffle_train: bool = True,
    num_workers: int = 4,
    normalize: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset with appropriate transforms.

    Args:
        batch_size: Batch size for data loaders
        img_size: Target image size
        use_augmentation: Whether to use data augmentation for training
        shuffle_train: Whether to shuffle training data
        num_workers: Number of data loading workers
        normalize: If True, output [-1, 1]. If False, output [0, 1] (default)

    Returns:
        (train_loader, test_loader) with images in [0, 1] or [-1, 1] range
    """
    train_transform_list = [
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
    ]

    if use_augmentation:
        train_transform_list.append(
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
        )
        train_transform_list.append(transforms.RandomCrop(img_size, padding=4))

    if normalize:
        train_transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    train_transform = transforms.Compose(train_transform_list)

    test_transform_list = [
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
    ]
    if normalize:
        test_transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    test_transform = transforms.Compose(test_transform_list)

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


# Denormalization utilities for visualization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor from ImageNet normalization to [0, 1]."""
    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(tensor.device)
    std = IMAGENET_STD.view(1, 3, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)


def denormalize_mnist(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2
