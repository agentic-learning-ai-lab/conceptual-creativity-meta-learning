#!/usr/bin/env python3
"""MNIST experiment for CCML.

Run from project root:
    python -m experiments.mnist
    python -m experiments.mnist --config configs/mnist.yaml
    python -m experiments.mnist --config configs/mnist.yaml --num_images 20

With meta-gradient guidance:
    python -m experiments.mnist --use_meta_gradient --num_images 10
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import Omniglot
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from ccml.data import denormalize_mnist, get_mnist_loaders
from ccml.models.appraisers import Appraiser
from ccml.models.creators import create_mnist_creator
from ccml.models.decoders import MNISTDecoder
from ccml.models.encoders import MNISTEncoder
from ccml.rewards import calculate_creator_reward
from ccml.training.evaluation import (
    run_inner_loop,
    compute_fisher_information,
    get_original_params,
)
from ccml.training.guidance import generate_with_guidance, generate_with_meta_guidance
from ccml.training.pretrain import pretrain_decoder, pretrain_encoder
from ccml.utils import load_config, merge_config_with_args
from ccml.utils import (
    find_nearest_neighbors,
    save_individual_image,
    visualize_encoder_filters,
    visualize_generation_grid,
    visualize_reconstructions,
)


def setup_logging(output_dir: str, experiment_name: str = "mnist"):
    """Setup logging to both file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{experiment_name}_{timestamp}.log")

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Setup logger
    logger = logging.getLogger("ccml")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


# --- MNIST-specific visualization functions ---


def visualize_poc_category(category_name, results, base_filename):
    """Helper function to create a grid for a single PoC category."""
    if not results:
        print(f"  [Warning] No results to visualize for category: {category_name}")
        return

    fig, axes = plt.subplots(len(results), 3, figsize=(9, len(results) * 1.5))
    if len(results) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Proof-of-Concept: {category_name}", fontsize=16)

    for i, res in enumerate(results):
        original = denormalize_mnist(res["original"].squeeze())
        initial_recon = denormalize_mnist(res["initial_recon"].squeeze())
        final_recon = denormalize_mnist(res["final_recon"].squeeze())

        axes[i, 0].imshow(original, cmap="gray")
        axes[i, 1].imshow(initial_recon, cmap="gray")
        axes[i, 2].imshow(final_recon, cmap="gray")

        if i == 0:
            axes[i, 0].set_title("Original Sample")
            axes[i, 1].set_title("Initial Recon")
            axes[i, 2].set_title("Final Recon")

        axes[i, 0].set_ylabel(f"Sample {i+1}", rotation=0, labelpad=30, ha="right")
        axes[i, 1].set_xlabel(f"Loss: {res['initial_loss']:.4f}", fontsize=8)
        axes[i, 2].set_xlabel(
            f"Loss: {res['final_loss']:.4f}\nReward: {res['shaped_reward']:.4f}",
            fontsize=8,
        )

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    safe_category_name = (
        category_name.replace(" ", "_").replace("(", "").replace(")", "")
    )
    filename = base_filename.replace(".png", f"_{safe_category_name}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Proof-of-concept visualization for {category_name} saved to {filename}")


def proof_of_concept_test(
    appraiser, filename, finetune_layers, device, inner_loop_steps, inner_loop_lr, img_size, reward_config
):
    """Run proof-of-concept test comparing different image categories."""
    print(f"\n--- Running Proof-of-Concept Test (finetuning: {finetune_layers}) ---")

    omniglot_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img: TF.invert(img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    omniglot_dataset = Omniglot(root="./data", background=True, download=True)

    greek_images = [
        omniglot_dataset[i][0]
        for i, (_, c) in enumerate(omniglot_dataset._flat_character_images)
        if "Greek" in omniglot_dataset._characters[c]
    ][:200:20]
    latin_images = [
        omniglot_dataset[i][0]
        for i, (_, c) in enumerate(omniglot_dataset._flat_character_images)
        if "Latin" in omniglot_dataset._characters[c]
    ][:200:20]
    korean_images = [
        omniglot_dataset[i][0]
        for i, (_, c) in enumerate(omniglot_dataset._flat_character_images)
        if "Korean" in omniglot_dataset._characters[c]
    ][:200:20]

    mnist_test_loader = get_mnist_loaders(1, use_augmentation=False, normalize=True)[1]

    test_categories = {
        "Greek": [omniglot_transform(img) for img in greek_images],
        "English (Latin)": [omniglot_transform(img) for img in latin_images],
        "Korean": [omniglot_transform(img) for img in korean_images],
        "White Noise": [torch.randn(1, img_size, img_size) * 0.5 for _ in range(10)],
        "MNIST": [img.squeeze(0) for img, _ in list(iter(mnist_test_loader))[:10]],
    }

    for category_name, images in test_categories.items():
        results = []
        print(f"\n  Testing category: {category_name}")

        for image in tqdm(images, desc=f"Processing {category_name}"):
            image = image.unsqueeze(0).to(device)

            # Clone appraiser for fine-tuning
            appraiser_ft = appraiser.clone().to(device)

            # Get initial reconstruction and loss
            with torch.no_grad():
                _, initial_recon = appraiser_ft(image)
                initial_loss = appraiser_ft.compute_reconstruction_loss(image).item()

            # Run inner loop optimization
            run_inner_loop(
                appraiser_ft,
                image,
                finetune_layers,
                inner_loop_steps,
                inner_loop_lr,
                retain_graph=False,
            )

            # Get final reconstruction and loss
            with torch.no_grad():
                _, final_recon = appraiser_ft(image)
                final_loss = appraiser_ft.compute_reconstruction_loss(image).item()

            shaped_reward, _, _, _ = calculate_creator_reward(
                torch.tensor(initial_loss), torch.tensor(final_loss), **reward_config
            )

            results.append(
                {
                    "original": image.cpu(),
                    "initial_recon": initial_recon.cpu(),
                    "final_recon": final_recon.cpu(),
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "shaped_reward": shaped_reward.item(),
                }
            )

        visualize_poc_category(category_name, results, filename)


def save_individual_image_with_reconstructions(
    generated_image,
    initial_recon,
    final_recon,
    image_idx,
    output_dir,
    initial_loss,
    final_loss,
    reward,
):
    """Save individual generated image with reconstructions in a comparison grid."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def denormalize_and_plot(ax, tensor, title):
        img = tensor.detach().cpu()
        if img.dim() == 4:
            img = img.squeeze(0)
        if img.shape[0] == 3:
            img = img.mean(dim=0)
        elif img.shape[0] == 1:
            img = img.squeeze(0)
        img = denormalize_mnist(img)
        img = img.clamp(0, 1)
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    denormalize_and_plot(
        axes[0], generated_image, f"Generated Image #{image_idx}\nReward: {reward:.4f}"
    )
    denormalize_and_plot(
        axes[1], initial_recon, f"Initial Reconstruction\nLoss: {initial_loss:.4f}"
    )
    denormalize_and_plot(
        axes[2], final_recon, f"Final Reconstruction\nLoss: {final_loss:.4f}"
    )

    fig.suptitle(
        f"Image {image_idx} - Reward: {reward:.4f}", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()

    filename = os.path.join(
        output_dir,
        f"image_{image_idx:03d}_R{reward:.4f}_Li{initial_loss:.4f}_Lf{final_loss:.4f}.png",
    )
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    return filename


def visualize_guided_vs_nn_grid(
    all_candidates, nearest_neighbors, base_filename, num_per_grid=5
):
    """Generate grids comparing guided images with nearest neighbors."""
    print("\n--- Generating Guided vs. Nearest Neighbor Grids ---")

    num_total = len(all_candidates)
    num_grids = (num_total + num_per_grid - 1) // num_per_grid

    for grid_idx in range(num_grids):
        start_idx = grid_idx * num_per_grid
        end_idx = min(start_idx + num_per_grid, num_total)

        current_candidates = all_candidates[start_idx:end_idx]
        current_nns = nearest_neighbors[start_idx:end_idx]
        num_images_in_grid = len(current_candidates)

        fig, axes = plt.subplots(4, num_images_in_grid, figsize=(num_images_in_grid * 2.5, 9))
        if num_images_in_grid == 1:
            axes = axes.reshape(-1, 1)

        def plot_image(ax, tensor, title):
            tensor = tensor.detach().cpu()
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            if tensor.shape[0] == 3:
                tensor = tensor.mean(dim=0)
            elif tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            tensor = denormalize_mnist(tensor)
            tensor = tensor.clamp(0, 1)
            ax.imshow(tensor, cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        for i, candidate in enumerate(current_candidates):
            rank = start_idx + i + 1
            reward = candidate["reward"]
            plot_image(
                axes[0, i], candidate["image"], f"Rank {rank} (R: {reward:.4f})"
            )
            plot_image(
                axes[1, i],
                candidate["initial_recon"],
                f"Initial Recon\nLoss: {candidate['initial_loss']:.4f}",
            )
            plot_image(
                axes[2, i],
                candidate["final_recon"],
                f"Final Recon\nLoss: {candidate['final_loss']:.4f}",
            )
            plot_image(axes[3, i], current_nns[i], "Nearest Neighbor")

        fig.tight_layout()
        grid_filename = f"{base_filename.replace('.png', '')}_part_{grid_idx+1}.png"
        plt.savefig(grid_filename)
        plt.close()
        print(f"Saved guided grid to {grid_filename}")


def visualize_unguided_grid(unguided_images, base_filename, num_per_grid=5):
    """Generate grids of unguided images."""
    print("\n--- Generating Unguided Results Grids ---")
    num_total = len(unguided_images)
    num_grids = (num_total + num_per_grid - 1) // num_per_grid

    for grid_idx in range(num_grids):
        start_idx = grid_idx * num_per_grid
        end_idx = min(start_idx + num_per_grid, num_total)

        current_images = unguided_images[start_idx:end_idx]
        num_images_in_grid = len(current_images)

        fig, axes = plt.subplots(1, num_images_in_grid, figsize=(num_images_in_grid * 2, 2.5))
        if num_images_in_grid == 1:
            axes = [axes]

        for i, img in enumerate(current_images):
            axes[i].imshow(img.convert("L"), cmap="gray")
            axes[i].set_title(f"Unguided {start_idx + i + 1}")
            axes[i].axis("off")

        fig.tight_layout()
        grid_filename = f"{base_filename.replace('.png', '')}_part_{grid_idx+1}.png"
        plt.savefig(grid_filename)
        plt.close()
        print(f"Saved unguided grid to {grid_filename}")


def visualize_optimization_curves(candidates, output_path, guidance_type="regular", reward_type="original"):
    """Visualize reward and loss curves during guided generation."""
    # Filter candidates that have reward history
    candidates_with_history = [c for c in candidates if "reward_history" in c and c["reward_history"]]

    if not candidates_with_history:
        print("No reward history available for visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Optimization Curves ({guidance_type.capitalize()} Guidance, {reward_type} reward)", fontsize=14)

    colors = plt.cm.viridis(np.linspace(0, 1, len(candidates_with_history)))

    # Plot 1: Reward over timesteps
    ax = axes[0, 0]
    for i, candidate in enumerate(candidates_with_history):
        history = candidate["reward_history"]
        timesteps = [h["timestep"] for h in history]
        rewards = [h["reward"] for h in history]
        ax.plot(timesteps, rewards, color=colors[i], alpha=0.7, linewidth=1.5,
                label=f"Image {i+1} (R={candidate['reward']:.3f})")
    ax.set_xlabel("Diffusion Timestep")
    ax.set_ylabel("Reward")
    ax.set_title("Reward During Generation")
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    if reward_type == "layer_diff":
        # Plot 2: L_first over timesteps
        ax = axes[0, 1]
        for i, candidate in enumerate(candidates_with_history):
            history = candidate["reward_history"]
            timesteps = [h["timestep"] for h in history]
            L_first = [h["L_first_final"] for h in history]
            ax.plot(timesteps, L_first, color=colors[i], alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Diffusion Timestep")
        ax.set_ylabel("L_first (decoder_first)")
        ax.set_title("Decoder First Loss During Generation")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

        # Plot 3: L_last over timesteps
        ax = axes[1, 0]
        for i, candidate in enumerate(candidates_with_history):
            history = candidate["reward_history"]
            timesteps = [h["timestep"] for h in history]
            L_last = [h["L_last_final"] for h in history]
            ax.plot(timesteps, L_last, color=colors[i], alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Diffusion Timestep")
        ax.set_ylabel("L_last (decoder_last)")
        ax.set_title("Decoder Last Loss During Generation")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

        # Plot 4: Layer diff over timesteps
        ax = axes[1, 1]
        for i, candidate in enumerate(candidates_with_history):
            history = candidate["reward_history"]
            timesteps = [h["timestep"] for h in history]
            layer_diff = [h["layer_diff"] for h in history]
            ax.plot(timesteps, layer_diff, color=colors[i], alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Diffusion Timestep")
        ax.set_ylabel("Layer Diff (L_last - L_first)")
        ax.set_title("Layer Difference During Generation")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
    else:
        # Plot 2: Initial loss over timesteps
        ax = axes[0, 1]
        for i, candidate in enumerate(candidates_with_history):
            history = candidate["reward_history"]
            timesteps = [h["timestep"] for h in history]
            init_losses = [h["initial_loss"] for h in history]
            ax.plot(timesteps, init_losses, color=colors[i], alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Diffusion Timestep")
        ax.set_ylabel("Initial Loss (before fine-tuning)")
        ax.set_title("Initial Loss During Generation")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

        # Plot 3: Final loss over timesteps
        ax = axes[1, 0]
        for i, candidate in enumerate(candidates_with_history):
            history = candidate["reward_history"]
            timesteps = [h["timestep"] for h in history]
            final_losses = [h["final_loss"] for h in history]
            ax.plot(timesteps, final_losses, color=colors[i], alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Diffusion Timestep")
        ax.set_ylabel("Final Loss (after fine-tuning)")
        ax.set_title("Final Loss During Generation")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

        # Plot 4: Improvement over timesteps
        ax = axes[1, 1]
        for i, candidate in enumerate(candidates_with_history):
            history = candidate["reward_history"]
            timesteps = [h["timestep"] for h in history]
            improvements = [h["improvement"] for h in history]
            ax.plot(timesteps, improvements, color=colors[i], alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Diffusion Timestep")
        ax.set_ylabel("Improvement (init - final)")
        ax.set_title("Improvement During Generation")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved optimization curves to {output_path}")


# --- Main execution ---


def main():
    parser = argparse.ArgumentParser(description="Run CCML on MNIST")

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config)",
    )

    # Paths
    parser.add_argument("--output_dir", type=str, default="outputs/mnist")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/mnist")

    # Model
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--feature_dim", type=int, default=512)

    # Pretraining
    parser.add_argument("--encoder_pretrain_epochs", type=int, default=5)
    parser.add_argument("--decoder_pretrain_epochs", type=int, default=5)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_batch_size", type=int, default=128)

    # Bi-level optimization (inner loop)
    parser.add_argument("--inner_loop_steps", type=int, default=10)
    parser.add_argument("--inner_loop_lr", type=float, default=1e-4)
    parser.add_argument("--finetune_layers", type=str, default="all")

    # Guidance (outer loop)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--guidance_steps", type=int, default=200)
    parser.add_argument("--guidance_scale", type=float, default=500.0)

    # Reward shaping
    parser.add_argument("--ideal_initial_loss", type=float, default=0.15)
    parser.add_argument("--initial_loss_sigma", type=float, default=0.2)
    parser.add_argument("--final_loss_temp", type=float, default=0.1)

    # Flags
    parser.add_argument("--skip_pretraining", action="store_true")
    parser.add_argument("--pretrain_only", action="store_true",
                        help="Stop after pretraining (or loading) the encoder and decoder")
    parser.add_argument("--run_poc", action="store_true", help="Run proof-of-concept test")
    parser.add_argument(
        "--use_meta_gradient",
        action="store_true",
        help="Use meta-gradient (MAML-style) guidance instead of regular guidance"
    )
    parser.add_argument(
        "--use_layer_diff",
        action="store_true",
        help="Use layer-difference reward (decoder_first vs decoder_last)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Regularization
    parser.add_argument(
        "--reg_type",
        type=str,
        default=None,
        choices=[None, "l2", "ewc"],
        help="Regularization type for inner loop (l2 or ewc)"
    )
    parser.add_argument(
        "--reg_weight",
        type=float,
        default=0.0,
        help="Regularization weight (lambda)"
    )
    parser.add_argument(
        "--fisher_samples",
        type=int,
        default=200,
        help="Number of samples for computing Fisher information (EWC)"
    )

    args = parser.parse_args()

    # Load and merge config
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args, parser)
        print(f"Loaded config from: {args.config}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Auto-suffix output directory with guidance type
    if args.use_layer_diff:
        guidance_type = "layer_diff"
    elif args.use_meta_gradient:
        guidance_type = "meta"
    else:
        guidance_type = "regular"
    if not args.output_dir.endswith(f"_{guidance_type}"):
        args.output_dir = f"{args.output_dir}_{guidance_type}"

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Setup logging
    logger, log_file = setup_logging(args.output_dir, f"mnist_{guidance_type}")

    logger.info("=" * 60)
    logger.info("CCML MNIST Experiment")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Log file: {log_file}")

    # Build reward config from args
    reward_config = {
        "ideal_initial_loss": args.ideal_initial_loss,
        "initial_loss_sigma": args.initial_loss_sigma,
        "final_loss_temp": args.final_loss_temp,
    }

    # Log configuration
    logger.info("")
    logger.info("=== Configuration ===")
    logger.info(f"Image Size: {args.img_size}")
    logger.info(f"Inner Loop: {args.inner_loop_steps} steps @ lr={args.inner_loop_lr}")
    logger.info(f"Guidance: {args.guidance_steps} steps @ scale={args.guidance_scale}")
    logger.info(f"Finetune Layers: {args.finetune_layers}")
    logger.info(f"Meta-Gradient: {args.use_meta_gradient}")
    logger.info(f"Reward Config: ideal_init={args.ideal_initial_loss}, sigma={args.initial_loss_sigma}, temp={args.final_loss_temp}")
    logger.info(f"Regularization: type={args.reg_type}, weight={args.reg_weight}")
    logger.info("=====================")

    # Paths
    encoder_path = os.path.join(args.checkpoint_dir, "encoder.pth")
    decoder_path = os.path.join(args.checkpoint_dir, "decoder.pth")

    # Load data (normalize=True for [-1, 1] range expected by appraiser)
    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.pretrain_batch_size, normalize=True
    )
    nn_train_loader, _ = get_mnist_loaders(
        batch_size=args.pretrain_batch_size, use_augmentation=False,
        shuffle_train=False, normalize=True
    )

    # Create appraiser components
    encoder = MNISTEncoder(feature_dim=args.feature_dim)
    decoder = MNISTDecoder(feature_dim=args.feature_dim)

    # Pretrain or load encoder
    if not args.skip_pretraining and not os.path.exists(encoder_path):
        pretrain_encoder(
            encoder,
            train_loader,
            test_loader,
            epochs=args.encoder_pretrain_epochs,
            lr=args.pretrain_lr,
            save_path=encoder_path,
            device=device,
        )
        visualize_encoder_filters(
            encoder, os.path.join(args.output_dir, "encoder_filters.png")
        )
    else:
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"Loaded encoder from {encoder_path}")

    encoder.to(device)

    # Pretrain or load decoder
    if not args.skip_pretraining and not os.path.exists(decoder_path):
        pretrain_decoder(
            encoder,
            decoder,
            train_loader,
            test_loader,
            epochs=args.decoder_pretrain_epochs,
            lr=args.pretrain_lr,
            save_path=decoder_path,
            device=device,
        )
        visualize_reconstructions(
            encoder,
            decoder,
            test_loader,
            filename=os.path.join(args.output_dir, "reconstructions.png"),
        )
    else:
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        print(f"Loaded decoder from {decoder_path}")

    decoder.to(device)
    if args.pretrain_only:
        return

    # Create appraiser
    appraiser = Appraiser(
        encoder=encoder,
        decoder=decoder,
        reconstruction_target="pixels",
        reconstruction_from="features",
    )

    # Setup regularization if enabled
    original_params_dict = None
    fisher_dict = None
    if args.reg_type is not None and args.reg_weight > 0:
        logger.info("")
        logger.info("--- Setting up regularization ---")
        # For layer_diff mode, get params for full decoder (includes first and last)
        reg_layers = "decoder" if args.use_layer_diff else args.finetune_layers
        original_params_dict = get_original_params(appraiser, reg_layers)
        logger.info(f"Stored original params for {reg_layers} layers ({len(original_params_dict)} tensors)")

        if args.reg_type == "ewc":
            logger.info(f"Computing Fisher information ({args.fisher_samples} samples)...")
            fisher_dict = compute_fisher_information(
                appraiser, train_loader, reg_layers,
                device=device, num_samples=args.fisher_samples
            )
            logger.info("Fisher information computed.")

    # Run proof-of-concept test if requested
    if args.run_poc:
        proof_of_concept_test(
            appraiser=appraiser,
            filename=os.path.join(args.output_dir, "proof_of_concept.png"),
            finetune_layers=args.finetune_layers,
            device=device,
            inner_loop_steps=args.inner_loop_steps,
            inner_loop_lr=args.inner_loop_lr,
            img_size=args.img_size,
            reward_config=reward_config,
        )

    # Create creator
    logger.info("")
    logger.info("--- Loading Creator (Diffusion Model) ---")
    creator = create_mnist_creator(device=device)
    logger.info("Creator loaded.")

    # Select guidance function and reward type
    reward_type = "layer_diff" if args.use_layer_diff else "original"
    if args.use_meta_gradient:
        guidance_fn = generate_with_meta_guidance
        logger.info("Using META-GRADIENT guidance (MAML-style differentiable inner loop)")
    else:
        guidance_fn = generate_with_guidance
        logger.info("Using REGULAR guidance (non-differentiable inner loop)")
    logger.info(f"Reward type: {reward_type}")

    # Generate creative samples
    logger.info("")
    logger.info(f"--- Generating {args.num_images} Creative Samples ---")
    candidates = []
    generation_times = []

    # Print header
    header = f"{'#':<4} {'Init Loss':<12} {'Final Loss':<12} {'Improve%':<10} {'Reward':<10} {'Time(s)':<10}"
    logger.info("")
    logger.info(header)
    logger.info("=" * len(header))

    total_start_time = time.time()

    for i in range(args.num_images):
        image_start_time = time.time()

        # Set seed for reproducibility (each image gets a different seed)
        torch.manual_seed(args.seed + i)

        logger.debug(f"Generating image {i+1}/{args.num_images} (seed={args.seed + i})")

        generated, final_step_data = guidance_fn(
            creator=creator,
            appraiser=appraiser,
            finetune_layers=args.finetune_layers,
            num_steps=args.guidance_steps,
            guidance_scale=args.guidance_scale,
            inner_loop_steps=args.inner_loop_steps,
            inner_loop_lr=args.inner_loop_lr,
            device=device,
            visualize_final_loop=(i == 0),
            reward_config=reward_config,
            reward_type=reward_type,
            reg_type=args.reg_type,
            reg_weight=args.reg_weight,
            original_params_dict=original_params_dict,
            fisher_dict=fisher_dict,
        )

        # Extract metrics based on reward type
        if reward_type == "layer_diff":
            from ccml.rewards import calculate_layer_difference_reward
            L_first = final_step_data["L_first_final"]
            L_last = final_step_data["L_last_final"]
            initial_loss = final_step_data["initial_loss"]
            reward, _, _, _ = calculate_layer_difference_reward(
                torch.tensor(L_first), torch.tensor(L_last), args.final_loss_temp,
                args.ideal_initial_loss, args.initial_loss_sigma,
                L_initial=torch.tensor(initial_loss)
            )
            final_loss = L_first  # Use decoder_first loss as the "final" for display
            improvement = L_last - L_first  # Layer diff
            improvement_pct = (improvement / L_last * 100) if L_last > 0 else 0
        else:
            reward, _, _, _ = calculate_creator_reward(
                torch.tensor(final_step_data["initial_loss"]),
                torch.tensor(final_step_data["final_loss"]),
                **reward_config,
            )
            initial_loss = final_step_data["initial_loss"]
            final_loss = final_step_data["final_loss"]
            improvement = initial_loss - final_loss
            improvement_pct = (improvement / initial_loss * 100) if initial_loss > 0 else 0

        image_time = time.time() - image_start_time
        generation_times.append(image_time)

        # Log result
        logger.info(
            f"{i+1:<4} {initial_loss:<12.4f} {final_loss:<12.4f} {improvement_pct:<10.1f} {reward.item():<10.4f} {image_time:<10.1f}"
        )

        # Save individual image
        saved_path = save_individual_image(
            generated,
            i + 1,
            args.output_dir,
            initial_loss,
            final_loss,
            reward.item(),
            is_grayscale=True,
        )
        logger.debug(f"Saved: {saved_path}")

        # Build candidate dict based on reward type
        candidate = {
            "image": generated,
            "initial_recon": final_step_data["initial_recon"],
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "reward": reward.item(),
            "time": image_time,
            "reward_history": final_step_data.get("reward_history", []),
        }
        if reward_type == "layer_diff":
            candidate["first_recon"] = final_step_data["first_recon"]
            candidate["last_recon"] = final_step_data["last_recon"]
            candidate["final_recon"] = final_step_data["first_recon"]  # Use first for grid
            candidate["L_first_final"] = final_step_data["L_first_final"]
            candidate["L_last_final"] = final_step_data["L_last_final"]
            candidate["layer_diff"] = final_step_data["layer_diff"]
        else:
            candidate["final_recon"] = final_step_data["final_recon"]
        candidates.append(candidate)

    total_time = time.time() - total_start_time
    logger.info("=" * len(header))

    # Summary statistics
    rewards = [c["reward"] for c in candidates]
    init_losses = [c["initial_loss"] for c in candidates]
    final_losses = [c["final_loss"] for c in candidates]
    improvements = [(c["initial_loss"] - c["final_loss"]) / c["initial_loss"] * 100 for c in candidates]

    logger.info("")
    logger.info("=== Generation Summary ===")
    logger.info(f"Guidance Type: {'Meta-Gradient' if args.use_meta_gradient else 'Regular'}")
    logger.info(f"Reward Type: {reward_type}")
    logger.info(f"Total Images: {len(candidates)}")
    logger.info(f"Total Time: {total_time:.1f}s ({total_time/len(candidates):.1f}s per image)")
    logger.info("")
    logger.info(f"Reward:      mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}, min={np.min(rewards):.4f}, max={np.max(rewards):.4f}")
    logger.info(f"Init Loss:   mean={np.mean(init_losses):.4f}, std={np.std(init_losses):.4f}")
    logger.info(f"Final Loss:  mean={np.mean(final_losses):.4f}, std={np.std(final_losses):.4f}")
    logger.info(f"Improve %:   mean={np.mean(improvements):.1f}%, std={np.std(improvements):.1f}%")
    logger.info("==========================")

    # Sort and visualize
    sorted_candidates = sorted(candidates, key=lambda x: x["reward"], reverse=True)

    visualize_generation_grid(
        sorted_candidates,
        os.path.join(args.output_dir, "generation_grid.png"),
        is_grayscale=True,
    )

    # Visualize optimization curves
    visualize_optimization_curves(
        candidates,
        os.path.join(args.output_dir, "optimization_curves.png"),
        guidance_type=guidance_type,
        reward_type=reward_type,
    )

    # Find nearest neighbors and visualize
    guided_images = [c["image"] for c in sorted_candidates]
    nearest_neighbors = find_nearest_neighbors(
        guided_images, nn_train_loader.dataset, device=device
    )

    visualize_guided_vs_nn_grid(
        sorted_candidates,
        nearest_neighbors,
        os.path.join(args.output_dir, "guided_vs_nn.png"),
    )

    logger.info("")
    logger.info("--- Experiment Complete ---")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
