"""Visualization and utility functions for CCML."""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from ccml.data import denormalize_imagenet, denormalize_mnist


def visualize_reconstructions(
    encoder,
    decoder,
    dataloader,
    num_images: int = 8,
    filename: str = "reconstructions.png",
    is_grayscale: bool = True,
):
    """
    Visualize original vs reconstructed images.

    Args:
        encoder: Encoder model
        decoder: Decoder model
        dataloader: Data loader to get images from
        num_images: Number of images to visualize
        filename: Output filename
        is_grayscale: Whether images are grayscale (MNIST) or RGB
    """
    device = next(encoder.parameters()).device
    encoder.eval()
    decoder.eval()

    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)

    with torch.no_grad():
        features, _ = encoder(images)
        reconstructions = decoder(features)

    if is_grayscale:
        comparison = torch.cat([images.cpu(), reconstructions.cpu()])
        comparison = denormalize_mnist(comparison)
    else:
        comparison = torch.cat([images.cpu(), reconstructions.cpu()])
        comparison = denormalize_imagenet(comparison)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3.5))

    for i in range(num_images):
        if is_grayscale:
            axes[0, i].imshow(comparison[i].squeeze(), cmap="gray")
            axes[1, i].imshow(comparison[i + num_images].squeeze(), cmap="gray")
        else:
            axes[0, i].imshow(comparison[i].permute(1, 2, 0))
            axes[1, i].imshow(comparison[i + num_images].permute(1, 2, 0))

        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        axes[1, i].set_title("Recon")
        axes[1, i].axis("off")

    plt.suptitle("Original vs. Reconstructed Images")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Reconstructions saved to {filename}")


def visualize_encoder_filters(encoder, filename: str = "encoder_filters.png"):
    """Visualize first layer encoder filters."""
    filters = encoder.main[0].weight.data.clone().cpu()
    filters = (filters - filters.min()) / (filters.max() - filters.min())

    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < len(filters):
            ax.imshow(filters[i, 0], cmap="gray")
        ax.axis("off")

    plt.suptitle("First Layer Encoder Filters")
    plt.savefig(filename)
    plt.close()
    print(f"Encoder filters saved to {filename}")


def visualize_inner_loop_progress(
    original,
    initial_recon,
    final_recon,
    initial_loss: float,
    final_loss: float,
    filename: str,
    is_grayscale: bool = True,
):
    """Visualize inner loop learning progress."""
    original = original.detach().cpu().squeeze()
    initial_recon = initial_recon.detach().cpu().squeeze()
    final_recon = final_recon.detach().cpu().squeeze()

    if is_grayscale:
        original = denormalize_mnist(original)
        initial_recon = denormalize_mnist(initial_recon)
        final_recon = denormalize_mnist(final_recon)
        cmap = "gray"
    else:
        original = denormalize_imagenet(original.unsqueeze(0)).squeeze().permute(1, 2, 0)
        initial_recon = denormalize_imagenet(initial_recon.unsqueeze(0)).squeeze().permute(1, 2, 0)
        final_recon = denormalize_imagenet(final_recon.unsqueeze(0)).squeeze().permute(1, 2, 0)
        cmap = None

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(original, cmap=cmap)
    axes[0].set_title("Original Sample")
    axes[0].axis("off")

    axes[1].imshow(initial_recon, cmap=cmap)
    axes[1].set_title(f"Initial Recon\nLoss: {initial_loss:.4f}")
    axes[1].axis("off")

    axes[2].imshow(final_recon, cmap=cmap)
    axes[2].set_title(f"Final Recon\nLoss: {final_loss:.4f}")
    axes[2].axis("off")

    plt.suptitle("Appraiser's Inner Loop Learning")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Inner loop progress saved to {filename}")


def save_individual_image(
    image_tensor: torch.Tensor,
    image_idx: int,
    output_dir: str,
    initial_loss: float,
    final_loss: float,
    reward: float,
    is_grayscale: bool = False,
) -> str:
    """
    Save individual generated image with measurements in filename.

    Returns:
        Path to saved file
    """
    img = image_tensor.detach().cpu()
    if img.dim() == 4:
        img = img.squeeze(0)

    if is_grayscale:
        img = denormalize_mnist(img)
        if img.shape[0] == 3:
            img = img.mean(dim=0)
        elif img.shape[0] == 1:
            img = img.squeeze(0)
    else:
        img = denormalize_imagenet(img.unsqueeze(0)).squeeze()
        img = img.permute(1, 2, 0)

    img = img.numpy()

    filename = os.path.join(
        output_dir,
        f"image_{image_idx:03d}_R{reward:.4f}_Li{initial_loss:.4f}_Lf{final_loss:.4f}.png",
    )

    if is_grayscale and len(img.shape) == 2:
        plt.imsave(filename, img, cmap="gray")
    else:
        plt.imsave(filename, img)

    return filename


def visualize_generation_grid(
    candidates: List[Dict],
    filename: str,
    max_rows: int = 4,
    max_cols: int = 5,
    is_grayscale: bool = False,
):
    """
    Visualize generated images in a grid sorted by reward.

    Args:
        candidates: List of dicts with 'image', 'reward', 'initial_loss', 'final_loss'
        filename: Output filename
        max_rows: Maximum rows per page
        max_cols: Maximum columns per page
        is_grayscale: Whether images are grayscale
    """
    num_total = len(candidates)
    images_per_page = max_rows * max_cols
    num_pages = (num_total + images_per_page - 1) // images_per_page

    for page_idx in range(num_pages):
        start_idx = page_idx * images_per_page
        end_idx = min(start_idx + images_per_page, num_total)

        current_candidates = candidates[start_idx:end_idx]
        num_images_in_page = len(current_candidates)

        actual_rows = min(max_rows, (num_images_in_page + max_cols - 1) // max_cols)
        actual_cols = min(max_cols, num_images_in_page)

        fig, axes = plt.subplots(actual_rows, actual_cols, figsize=(actual_cols * 3, actual_rows * 3))

        if num_images_in_page == 1:
            axes = np.array([axes])
        axes = axes.flatten() if num_images_in_page > 1 else axes

        for i, candidate in enumerate(current_candidates):
            rank = start_idx + i + 1
            reward = candidate["reward"]

            img = candidate["image"].detach().cpu()
            if img.dim() == 4:
                img = img.squeeze(0)

            if is_grayscale:
                img = denormalize_mnist(img)
                if img.shape[0] == 3:
                    img = img.mean(dim=0)
                elif img.shape[0] == 1:
                    img = img.squeeze(0)
                axes[i].imshow(img, cmap="gray")
            else:
                img = denormalize_imagenet(img.unsqueeze(0)).squeeze()
                img = img.permute(1, 2, 0).numpy()
                axes[i].imshow(img)

            axes[i].set_title(
                f"Rank {rank}\nR: {reward:.4f}\nL_i: {candidate['initial_loss']:.4f}\nL_f: {candidate['final_loss']:.4f}",
                fontsize=9,
            )
            axes[i].axis("off")

        # Hide unused subplots
        for i in range(num_images_in_page, len(axes)):
            axes[i].axis("off")

        page_info = f"Page {page_idx+1}/{num_pages}" if num_pages > 1 else "All Images"
        fig.suptitle(f"{page_info} - Images Sorted by Reward", fontsize=14, fontweight="bold")
        fig.tight_layout()

        if num_pages > 1:
            grid_filename = f"{filename.replace('.png', '')}_page_{page_idx+1}.png"
        else:
            grid_filename = filename

        plt.savefig(grid_filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved grid to {grid_filename}")


def find_nearest_neighbors(
    generated_images: List[torch.Tensor],
    reference_dataset,
    device: str = "cuda",
) -> List[torch.Tensor]:
    """
    Find nearest neighbors in reference dataset for generated images.

    Args:
        generated_images: List of generated image tensors
        reference_dataset: Dataset to search in
        device: Device to run on

    Returns:
        List of nearest neighbor tensors
    """
    # Load all reference images
    all_ref_images = torch.stack([img for img, _ in reference_dataset]).to(device)

    nearest_neighbors = []

    for gen_img in generated_images:
        # Convert to grayscale if needed
        gen_img_gray = gen_img.mean(dim=1, keepdim=True).to(device)

        # Compute MSE to all reference images
        distances = torch.mean((gen_img_gray - all_ref_images) ** 2, dim=(1, 2, 3))

        # Find minimum
        min_index = torch.argmin(distances)
        best_match = all_ref_images[min_index].unsqueeze(0).cpu()

        nearest_neighbors.append(best_match)

    return nearest_neighbors
