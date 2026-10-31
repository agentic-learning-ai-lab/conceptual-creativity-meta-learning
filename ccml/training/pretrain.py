"""Pretraining functions for encoder and decoder."""

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ccml.models.encoders import MNISTEncoder, ResNetEncoder
from ccml.models.decoders import MNISTDecoder, ResNetDecoder
from ccml.models.appraisers import Appraiser


def pretrain_encoder(
    encoder: MNISTEncoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
    device: str = "cuda",
) -> MNISTEncoder:
    """
    Pretrain MNIST encoder on classification task.

    Args:
        encoder: MNISTEncoder to train
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save weights (optional)
        device: Device to train on

    Returns:
        Trained encoder
    """
    print("--- Pre-training Appraiser Encoder ---")
    encoder.to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        encoder.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=0):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            _, logits = encoder(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Average Training Loss: {avg_loss:.4f}")

        # Evaluate
        train_acc = evaluate_encoder_accuracy(encoder, train_loader, "Train", device)
        test_acc = evaluate_encoder_accuracy(encoder, test_loader, "Test", device)

    if save_path:
        torch.save(encoder.state_dict(), save_path)
        print(f"Encoder weights saved to {save_path}")

    return encoder


def pretrain_decoder(
    encoder: nn.Module,
    decoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
    device: str = "cuda",
    reconstruction_target: str = "pixels",
    reconstruction_from: str = "layer4",
) -> nn.Module:
    """
    Pretrain decoder to reconstruct from encoder features.

    Args:
        encoder: Encoder model (frozen during training)
        decoder: Decoder model to train
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save weights (optional)
        device: Device to train on
        reconstruction_target: What to reconstruct ("pixels", "layer2", "layer3")
        reconstruction_from: Which layer to use as bottleneck

    Returns:
        Trained decoder
    """
    print(f"--- Pre-training Appraiser Decoder ({reconstruction_from} -> {reconstruction_target}) ---")
    encoder.to(device).eval()
    decoder.to(device).train()

    # Use lower learning rate for pixel reconstruction
    actual_lr = lr * 0.1 if reconstruction_target == "pixels" else lr
    optimizer = optim.Adam(decoder.parameters(), lr=actual_lr)
    criterion = nn.MSELoss()

    if reconstruction_target == "pixels":
        print(f"  Using reduced learning rate for pixel reconstruction: {actual_lr}")

    for epoch in range(epochs):
        total_loss = 0

        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=0):
            images = images.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                # Get bottleneck features
                if isinstance(encoder, MNISTEncoder):
                    features, _ = encoder(images)
                    bottleneck = features
                    target = images
                elif isinstance(encoder, ResNetEncoder):
                    layer2, layer3, layer4 = encoder(images, return_intermediates=True)

                    if reconstruction_from == "layer4":
                        bottleneck = layer4
                    elif reconstruction_from == "layer3":
                        bottleneck = layer3
                    else:
                        raise ValueError(f"Unknown reconstruction_from: {reconstruction_from}")

                    if reconstruction_target == "pixels":
                        target = images
                    elif reconstruction_target == "layer2":
                        target = layer2
                    elif reconstruction_target == "layer3":
                        target = layer3
                    else:
                        raise ValueError(f"Unknown target: {reconstruction_target}")
                else:
                    raise ValueError(f"Unknown encoder type: {type(encoder)}")

            # Decode and compute loss
            reconstructed = decoder(bottleneck)
            loss = criterion(reconstructed, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Average Training Loss: {avg_loss:.6f}")

        # Evaluate
        evaluate_decoder_mse(
            encoder, decoder, test_loader, "Test", device,
            reconstruction_target, reconstruction_from
        )

    if save_path:
        torch.save(decoder.state_dict(), save_path)
        print(f"Decoder weights saved to {save_path}")

    return decoder


def evaluate_encoder_accuracy(
    encoder: MNISTEncoder,
    dataloader: DataLoader,
    dataset_name: str = "",
    device: str = "cuda",
) -> float:
    """Evaluate encoder classification accuracy."""
    encoder.to(device).eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            _, logits = encoder(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"  Accuracy on {dataset_name} set: {accuracy:.2f}%")
    return accuracy


def evaluate_decoder_mse(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    dataset_name: str = "",
    device: str = "cuda",
    reconstruction_target: str = "pixels",
    reconstruction_from: str = "layer4",
) -> float:
    """Evaluate decoder reconstruction MSE."""
    encoder.to(device).eval()
    decoder.to(device).eval()
    criterion = nn.MSELoss()

    total_loss = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # Get bottleneck and target
            if isinstance(encoder, MNISTEncoder):
                features, _ = encoder(images)
                bottleneck = features
                target = images
            elif isinstance(encoder, ResNetEncoder):
                layer2, layer3, layer4 = encoder(images, return_intermediates=True)

                if reconstruction_from == "layer4":
                    bottleneck = layer4
                elif reconstruction_from == "layer3":
                    bottleneck = layer3
                else:
                    raise ValueError(f"Unknown reconstruction_from: {reconstruction_from}")

                if reconstruction_target == "pixels":
                    target = images
                elif reconstruction_target == "layer2":
                    target = layer2
                elif reconstruction_target == "layer3":
                    target = layer3
                else:
                    raise ValueError(f"Unknown target: {reconstruction_target}")
            else:
                raise ValueError(f"Unknown encoder type: {type(encoder)}")

            reconstructed = decoder(bottleneck)
            loss = criterion(reconstructed, target)
            total_loss += loss.item()

    avg_mse = total_loss / len(dataloader)
    print(f"  Average MSE ({reconstruction_from}->{reconstruction_target}) on {dataset_name} set: {avg_mse:.6f}")
    return avg_mse
