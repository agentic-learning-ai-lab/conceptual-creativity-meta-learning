#!/usr/bin/env python3
"""MNIST baselines for comparison with CCML guidance.

Baselines:
1. Random perturbation: replace learnability gradient with random noise
2. Repulsive guidance: push away from MNIST digit embeddings (ProCreate-style)

Run from project root:
    CUDA_VISIBLE_DEVICES=0 python -m experiments.baselines.mnist_baselines \
        --method random --output_dir outputs/baseline_random
    CUDA_VISIBLE_DEVICES=0 python -m experiments.baselines.mnist_baselines \
        --method repulsive --output_dir outputs/baseline_repulsive
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm

from ccml.data import get_mnist_loaders
from ccml.models.appraisers import Appraiser
from ccml.models.creators import create_mnist_creator
from ccml.models.decoders import MNISTDecoder
from ccml.models.encoders import MNISTEncoder
from ccml.training.evaluation import run_inner_loop
from experiments.mnist_frontier import compute_learnability, evaluate_category


def generate_random_perturbation(creator, num_steps=1000, perturbation_scale=2.0,
                                 device="cuda"):
    """Generate with random perturbations instead of learnability gradient."""
    img_size = creator.get_img_size()
    channels = creator.get_channels()
    latents = torch.randn((1, channels, img_size, img_size), device=device)
    scheduler = creator.get_scheduler(num_steps)

    for t in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
        with torch.no_grad():
            noise_pred = creator.unet(latents, t).sample
            step_out = scheduler.step(noise_pred, t, latents)
            latents = step_out.prev_sample

        # Random perturbation (same magnitude structure as guidance)
        perturbation = torch.randn_like(latents) * perturbation_scale / num_steps
        latents = latents + perturbation

    # Get predicted clean image from final step
    with torch.no_grad():
        noise_pred = creator.unet(latents, scheduler.timesteps[-1]).sample
        pred_x0 = scheduler.step(noise_pred, scheduler.timesteps[-1],
                                  latents).pred_original_sample
    return pred_x0


def generate_repulsive(creator, digit_embeddings, encoder, num_steps=1000,
                       guidance_scale=5.0, device="cuda"):
    """Generate by pushing away from MNIST digit embeddings.

    At each step, compute the predicted clean image, encode it, and compute
    gradient to maximize distance from all 10 digit prototypes.
    """
    img_size = creator.get_img_size()
    channels = creator.get_channels()
    latents = torch.randn((1, channels, img_size, img_size), device=device)
    scheduler = creator.get_scheduler(num_steps)

    for t in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
        latents.requires_grad_(True)

        # Noise prediction — no grad through UNet (channels_last issue)
        with torch.no_grad():
            noise_pred = creator.unet(latents, t).sample

        # Predicted clean image (gradient flows through scheduler step)
        pred_x0 = scheduler.step(noise_pred, t, latents).pred_original_sample

        # Convert to grayscale if needed
        if pred_x0.shape[1] == 3:
            pred_gray = pred_x0.mean(dim=1, keepdim=True)
        else:
            pred_gray = pred_x0

        # Encode current prediction
        out = encoder(pred_gray)
        feat = out[0] if isinstance(out, tuple) else out

        # Maximize distance from all digit prototypes (repulsive loss)
        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(digit_embeddings, dim=-1)
        similarities = torch.mm(feat_norm, proto_norm.t())  # [1, 10]
        repulsive_loss = similarities.mean()  # minimize similarity

        repulsive_loss.backward()
        grad = latents.grad.detach()

        with torch.no_grad():
            step_out = scheduler.step(noise_pred, t, latents.detach())
            latents = step_out.prev_sample
            # Apply repulsive guidance (subtract grad to move away)
            latents = latents - guidance_scale * grad

    # Get predicted clean image from final step
    with torch.no_grad():
        noise_pred = creator.unet(latents, scheduler.timesteps[-1]).sample
        pred_x0 = scheduler.step(noise_pred, scheduler.timesteps[-1],
                                  latents).pred_original_sample
    return pred_x0


def compute_digit_embeddings(encoder, device="cuda", num_per_digit=50):
    """Compute prototype embedding for each digit 0-9."""
    _, test_loader = get_mnist_loaders(1, use_augmentation=False, normalize=True)

    digit_features = {i: [] for i in range(10)}
    for img, label in test_loader:
        label_val = label.item()
        if len(digit_features[label_val]) >= num_per_digit:
            if all(len(v) >= num_per_digit for v in digit_features.values()):
                break
            continue
        img = img.to(device)
        with torch.no_grad():
            out = encoder(img)
            feat = out[0] if isinstance(out, tuple) else out
        digit_features[label_val].append(feat)

    # Average each digit's features to get prototype
    prototypes = []
    for i in range(10):
        proto = torch.cat(digit_features[i], dim=0).mean(dim=0)
        prototypes.append(proto)

    return torch.stack(prototypes).to(device)  # [10, feat_dim]


def main():
    parser = argparse.ArgumentParser(description="MNIST baselines")
    parser.add_argument("--method", type=str, required=True,
                        choices=["random", "repulsive"],
                        help="Baseline method")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/mnist")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--num_generated", type=int, default=10)
    parser.add_argument("--generation_steps", type=int, default=1000)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--perturbation_scale", type=float, default=2.0)
    parser.add_argument("--finetune_layers", type=str, default="decoder_first")
    parser.add_argument("--inner_loop_steps", type=int, default=10)
    parser.add_argument("--inner_loop_lr", type=float, default=1e-4)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--skip_references", action="store_true",
                        help="Skip reference category evaluation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load appraiser
    encoder = MNISTEncoder(feature_dim=args.feature_dim)
    decoder = MNISTDecoder(feature_dim=args.feature_dim)
    encoder.load_state_dict(torch.load(
        os.path.join(args.checkpoint_dir, "encoder.pth"), map_location=device))
    decoder.load_state_dict(torch.load(
        os.path.join(args.checkpoint_dir, "decoder.pth"), map_location=device))
    encoder.to(device).eval()
    decoder.to(device).eval()

    appraiser = Appraiser(
        encoder=encoder, decoder=decoder,
        reconstruction_target="pixels", reconstruction_from="features",
    )

    all_results = {}

    # Reference categories (reuse from mnist_frontier)
    if not args.skip_references:
        from torchvision.datasets import Omniglot
        from torchvision.transforms import functional as TF

        _, test_loader = get_mnist_loaders(1, use_augmentation=False, normalize=True)
        mnist_images = []
        for img, _ in test_loader:
            mnist_images.append(img.to(device))
            if len(mnist_images) >= args.num_samples:
                break
        all_results["MNIST"] = evaluate_category(
            appraiser, mnist_images, "MNIST",
            args.finetune_layers, args.inner_loop_steps, args.inner_loop_lr)

        omniglot_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda img: TF.invert(img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        omniglot_dataset = Omniglot(root="./data", background=True, download=True)

        for script_name in ["Greek", "Korean"]:
            images = []
            for i, (_, c) in enumerate(omniglot_dataset._flat_character_images):
                if script_name in omniglot_dataset._characters[c]:
                    img = omniglot_transform(omniglot_dataset[i][0]).to(device)
                    images.append(img)
                    if len(images) >= args.num_samples:
                        break
            all_results[script_name] = evaluate_category(
                appraiser, images, script_name,
                args.finetune_layers, args.inner_loop_steps, args.inner_loop_lr)

        noise_images = [
            torch.randn(1, 1, 32, 32).to(device) * 0.5
            for _ in range(args.num_samples)
        ]
        all_results["Noise"] = evaluate_category(
            appraiser, noise_images, "Noise",
            args.finetune_layers, args.inner_loop_steps, args.inner_loop_lr)

        for alpha in [0.3, 0.5, 0.7]:
            cat_name = f"Blend (noise={alpha:.0%})"
            blended = [(1 - alpha) * img + alpha * torch.randn_like(img) * 0.5
                       for img in mnist_images[:args.num_samples]]
            all_results[cat_name] = evaluate_category(
                appraiser, blended, cat_name,
                args.finetune_layers, args.inner_loop_steps, args.inner_loop_lr)

    # Generate baseline images
    creator = create_mnist_creator(device=device)
    method_label = f"Baseline ({args.method})"
    print(f"\n=== Generating {method_label}: {args.num_generated} images ===")

    if args.method == "repulsive":
        print("Computing digit prototypes...")
        digit_embeddings = compute_digit_embeddings(encoder, device)
        print(f"Digit embeddings shape: {digit_embeddings.shape}")

    img_dir = os.path.join(args.output_dir, "images", args.method)
    os.makedirs(img_dir, exist_ok=True)

    baseline_results = []
    for i in range(args.num_generated):
        if args.method == "random":
            generated = generate_random_perturbation(
                creator, num_steps=args.generation_steps,
                perturbation_scale=args.perturbation_scale, device=device)
        elif args.method == "repulsive":
            generated = generate_repulsive(
                creator, digit_embeddings, encoder,
                num_steps=args.generation_steps,
                guidance_scale=args.guidance_scale, device=device)

        if generated.shape[1] == 3:
            generated = generated.mean(dim=1, keepdim=True)

        # Save raw image
        img_np = generated.squeeze().detach().cpu().numpy()
        img_np = np.clip((img_np + 1) / 2, 0, 1)
        plt.imsave(os.path.join(img_dir, f"{i:03d}.png"), img_np, cmap="gray")

        # Evaluate learnability
        scores = compute_learnability(
            appraiser, generated, args.finetune_layers,
            args.inner_loop_steps, args.inner_loop_lr)
        scores["category"] = method_label
        scores["index"] = i
        baseline_results.append(scores)

        print(f"  [{i+1}/{args.num_generated}] L_0={scores['L_0']:.4f} "
              f"L_T={scores['L_T']:.4f} imp={scores['improvement']:.4f}")

    all_results[method_label] = baseline_results

    # Save
    scores_path = os.path.join(args.output_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved scores to {scores_path}")


if __name__ == "__main__":
    main()
