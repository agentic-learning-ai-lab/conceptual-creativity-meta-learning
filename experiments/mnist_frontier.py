#!/usr/bin/env python3
"""MNIST unfamiliarity--learnability frontier evaluation.

Computes (L_0, L_0 - L_T) for reference categories and CCML-generated images,
then produces the frontier scatter plot.

Run from project root:
    CUDA_VISIBLE_DEVICES=0 python -m experiments.mnist_frontier
    CUDA_VISIBLE_DEVICES=0 python -m experiments.mnist_frontier --num_samples 50
    CUDA_VISIBLE_DEVICES=0 python -m experiments.mnist_frontier --skip_generation
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import Omniglot
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from ccml.data import get_mnist_loaders
from ccml.models.appraisers import Appraiser
from ccml.models.creators import create_mnist_creator
from ccml.models.decoders import MNISTDecoder
from ccml.models.encoders import MNISTEncoder
from ccml.rewards import calculate_creator_reward
from ccml.training.evaluation import run_inner_loop
from ccml.training.guidance import generate_with_guidance, generate_with_meta_guidance


def compute_learnability(appraiser, image, finetune_layers="decoder",
                         inner_loop_steps=10, inner_loop_lr=1e-4):
    """Compute (L_0, L_T, improvement) for a single image."""
    with torch.no_grad():
        L_0 = appraiser.compute_reconstruction_loss(image).item()

    appraiser_clone = appraiser.clone()
    run_inner_loop(
        appraiser_clone, image, finetune_layers,
        inner_loop_steps, inner_loop_lr, retain_graph=False
    )
    with torch.no_grad():
        L_T = appraiser_clone.compute_reconstruction_loss(image).item()

    return {"L_0": L_0, "L_T": L_T, "improvement": L_0 - L_T}


def evaluate_category(appraiser, images, category_name, finetune_layers,
                      inner_loop_steps, inner_loop_lr):
    """Evaluate learnability for a batch of images."""
    results = []
    for i, img in enumerate(tqdm(images, desc=category_name, leave=False)):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        scores = compute_learnability(
            appraiser, img, finetune_layers, inner_loop_steps, inner_loop_lr
        )
        scores["category"] = category_name
        scores["index"] = i
        results.append(scores)
    return results


def compute_shaped_reward(L_0, L_T, ideal_initial_loss=0.15,
                          initial_loss_sigma=0.2, final_loss_temp=0.1):
    """Compute the shaped reward from raw learnability scores."""
    improvement = L_0 - L_T
    complexity = np.exp(-((L_0 - ideal_initial_loss) ** 2)
                        / (2 * initial_loss_sigma ** 2))
    success = np.exp(-L_T / final_loss_temp)
    return improvement * complexity * success


def plot_frontier(all_results, output_path, reward_config=None):
    """Plot the unfamiliarity--learnability frontier (two panels)."""
    if reward_config is None:
        reward_config = {
            "ideal_initial_loss": 0.15,
            "initial_loss_sigma": 0.2,
            "final_loss_temp": 0.1,
        }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color/marker scheme
    style = {
        "MNIST": {"color": "#4CAF50", "marker": "o", "alpha": 0.6},
        "Greek": {"color": "#2196F3", "marker": "s", "alpha": 0.6},
        "Korean": {"color": "#9C27B0", "marker": "^", "alpha": 0.6},
        "Noise": {"color": "#F44336", "marker": "x", "alpha": 0.6},
        "Blend (noise=30%)": {"color": "#8BC34A", "marker": "v", "alpha": 0.6},
        "Blend (noise=50%)": {"color": "#FF9800", "marker": "v", "alpha": 0.6},
        "Blend (noise=70%)": {"color": "#E91E63", "marker": "v", "alpha": 0.6},
    }
    # CCML runs get distinct colors by guidance scale
    ccml_colors = ["#00BCD4", "#795548", "#607D8B", "#3F51B5", "#009688"]

    for cat_name, results in all_results.items():
        L_0 = np.array([r["L_0"] for r in results])
        L_T = np.array([r["L_T"] for r in results])
        imp = L_0 - L_T
        shaped = compute_shaped_reward(L_0, L_T, **reward_config)

        if cat_name in style:
            s = style[cat_name]
            kw = dict(label=cat_name, c=s["color"], marker=s["marker"],
                      alpha=s["alpha"], s=50, zorder=3)
        elif cat_name.startswith("CCML"):
            idx = len([k for k in all_results if k.startswith("CCML")
                       and list(all_results.keys()).index(k) <
                       list(all_results.keys()).index(cat_name)])
            color = ccml_colors[idx % len(ccml_colors)]
            kw = dict(label=cat_name, c=color, marker="D", alpha=0.7,
                      s=60, edgecolors="black", linewidths=0.5, zorder=4)
        else:
            continue

        axes[0].scatter(L_0, imp, **kw)
        axes[1].scatter(L_0, shaped, **kw)

    # Left panel: raw improvement
    axes[0].set_xlabel("Initial Loss $L_0$ (Unfamiliarity)", fontsize=12)
    axes[0].set_ylabel("Improvement $L_0 - L_T$", fontsize=12)
    axes[0].set_title("Raw Improvement", fontsize=13)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right panel: shaped reward
    axes[1].set_xlabel("Initial Loss $L_0$ (Unfamiliarity)", fontsize=12)
    axes[1].set_ylabel("Shaped Reward (Learnability)", fontsize=12)
    axes[1].set_title("Shaped Reward: improvement $\\times$ complexity $\\times$ success",
                       fontsize=11)
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved frontier plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MNIST frontier evaluation")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/mnist")
    parser.add_argument("--output_dir", type=str, default="outputs/mnist_frontier")
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Samples per reference category")
    parser.add_argument("--num_generated", type=int, default=10,
                        help="CCML images per guidance scale")
    parser.add_argument("--finetune_layers", type=str, default="decoder_first")
    parser.add_argument("--inner_loop_steps", type=int, default=10)
    parser.add_argument("--inner_loop_lr", type=float, default=1e-4)
    parser.add_argument("--guidance_scales", type=str, default="10,50,100,500",
                        help="Comma-separated guidance scales to sweep")
    parser.add_argument("--generation_steps", type=int, default=1000)
    parser.add_argument("--use_meta_gradient", action="store_true",
                        help="Use MAML-style differentiable inner loop")
    parser.add_argument("--include_unguided", action="store_true",
                        help="Also generate unguided (scale=0) images")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Only evaluate reference categories, skip CCML generation")
    parser.add_argument("--ideal_initial_loss", type=float, default=0.15,
                        help="Center of complexity gate (mu)")
    parser.add_argument("--initial_loss_sigma", type=float, default=0.2,
                        help="Width of complexity gate (sigma)")
    parser.add_argument("--final_loss_temp", type=float, default=0.1,
                        help="Success gate temperature (tau)")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--img_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained appraiser
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
    print(f"Loaded appraiser from {args.checkpoint_dir}")

    all_results = {}

    # --- Reference categories ---
    print("\n=== Evaluating reference categories ===")

    # MNIST
    _, test_loader = get_mnist_loaders(1, use_augmentation=False, normalize=True)
    mnist_images = []
    for img, _ in test_loader:
        mnist_images.append(img.to(device))
        if len(mnist_images) >= args.num_samples:
            break
    all_results["MNIST"] = evaluate_category(
        appraiser, mnist_images, "MNIST",
        args.finetune_layers, args.inner_loop_steps, args.inner_loop_lr)

    # Omniglot
    omniglot_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Lambda(lambda img: TF.invert(img)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    omniglot_dataset = Omniglot(root="./data", background=True, download=True)

    # Greek
    greek_images = []
    for i, (_, c) in enumerate(omniglot_dataset._flat_character_images):
        if "Greek" in omniglot_dataset._characters[c]:
            img = omniglot_transform(omniglot_dataset[i][0]).to(device)
            greek_images.append(img)
            if len(greek_images) >= args.num_samples:
                break
    all_results["Greek"] = evaluate_category(
        appraiser, greek_images, "Greek",
        args.finetune_layers, args.inner_loop_steps, args.inner_loop_lr)

    # Korean
    korean_images = []
    for i, (_, c) in enumerate(omniglot_dataset._flat_character_images):
        if "Korean" in omniglot_dataset._characters[c]:
            img = omniglot_transform(omniglot_dataset[i][0]).to(device)
            korean_images.append(img)
            if len(korean_images) >= args.num_samples:
                break
    all_results["Korean"] = evaluate_category(
        appraiser, korean_images, "Korean",
        args.finetune_layers, args.inner_loop_steps, args.inner_loop_lr)

    # Noise
    noise_images = [
        torch.randn(1, 1, args.img_size, args.img_size).to(device) * 0.5
        for _ in range(args.num_samples)
    ]
    all_results["Noise"] = evaluate_category(
        appraiser, noise_images, "Noise",
        args.finetune_layers, args.inner_loop_steps, args.inner_loop_lr)

    # Noise-blended synthetics: mix MNIST digits with noise at different ratios
    # This creates images with similar L_0 but varying structure
    blend_ratios = [0.3, 0.5, 0.7]
    for alpha in blend_ratios:
        cat_name = f"Blend (noise={alpha:.0%})"
        blended_images = []
        for img in mnist_images[:args.num_samples]:
            noise = torch.randn_like(img) * 0.5
            blended = (1 - alpha) * img + alpha * noise
            blended_images.append(blended)
        all_results[cat_name] = evaluate_category(
            appraiser, blended_images, cat_name,
            args.finetune_layers, args.inner_loop_steps, args.inner_loop_lr)

    # Print reference summary
    print("\n=== Reference Category Summary ===")
    print(f"{'Category':<12} {'n':<5} {'L_0 mean':<12} {'L_0 std':<12} "
          f"{'Improv mean':<12} {'Improv std':<12}")
    print("-" * 65)
    for cat, res in all_results.items():
        L0 = [r["L_0"] for r in res]
        imp = [r["improvement"] for r in res]
        print(f"{cat:<12} {len(res):<5} {np.mean(L0):<12.4f} {np.std(L0):<12.4f} "
              f"{np.mean(imp):<12.4f} {np.std(imp):<12.4f}")

    # --- CCML generation at different guidance scales ---
    if not args.skip_generation:
        print("\n=== Generating CCML images ===")
        guidance_scales = [float(s) for s in args.guidance_scales.split(",")
                          if s.strip()]

        reward_config = {
            "ideal_initial_loss": args.ideal_initial_loss,
            "initial_loss_sigma": args.initial_loss_sigma,
            "final_loss_temp": args.final_loss_temp,
        }

        creator = create_mnist_creator(device=device)
        guidance_fn = generate_with_meta_guidance if args.use_meta_gradient else generate_with_guidance
        print(f"Creator loaded. Guidance: {'meta-gradient' if args.use_meta_gradient else 'regular'}")

        # Unguided generation (scale=0) for baseline
        if args.include_unguided:
            cat_name = "CCML (unguided)"
            print(f"\n--- {cat_name}: generating {args.num_generated} images ---")
            unguided_results = []
            img_dir = os.path.join(args.output_dir, "images", "unguided")
            os.makedirs(img_dir, exist_ok=True)
            for i in range(args.num_generated):
                imgs = creator.generate_unguided(batch_size=1,
                                                  num_inference_steps=1000)
                # Convert PIL to tensor
                generated = transforms.ToTensor()(imgs[0]).unsqueeze(0).to(device)
                generated = generated * 2 - 1  # [0,1] -> [-1,1]
                if generated.shape[1] == 3:
                    generated = generated.mean(dim=1, keepdim=True)
                # Save raw image
                img_np = generated.squeeze().detach().cpu().numpy()
                img_np = np.clip((img_np + 1) / 2, 0, 1)
                plt.imsave(os.path.join(img_dir, f"{i:03d}.png"),
                           img_np, cmap="gray")
                scores = compute_learnability(
                    appraiser, generated, args.finetune_layers,
                    args.inner_loop_steps, args.inner_loop_lr)
                scores["category"] = cat_name
                scores["index"] = i
                scores["guidance_scale"] = 0
                unguided_results.append(scores)
                print(f"  [{i+1}/{args.num_generated}] L_0={scores['L_0']:.4f} "
                      f"improvement={scores['improvement']:.4f}")
            all_results[cat_name] = unguided_results

        for scale in guidance_scales:
            cat_name = f"CCML (scale={int(scale)})"
            print(f"\n--- {cat_name}: generating {args.num_generated} images ---")
            ccml_results = []

            for i in range(args.num_generated):
                generated, final_step_data = guidance_fn(
                    creator=creator,
                    appraiser=appraiser,
                    finetune_layers=args.finetune_layers,
                    num_steps=args.generation_steps,
                    guidance_scale=scale,
                    inner_loop_steps=args.inner_loop_steps,
                    inner_loop_lr=args.inner_loop_lr,
                    device=device,
                    reward_config=reward_config,
                )

                # Convert to grayscale if needed (DDPM returns 3-channel)
                if generated.shape[1] == 3:
                    generated = generated.mean(dim=1, keepdim=True)

                # Save raw image
                img_dir = os.path.join(args.output_dir, "images",
                                       f"scale_{int(scale)}")
                os.makedirs(img_dir, exist_ok=True)
                img_np = generated.squeeze().detach().cpu().numpy()
                img_np = np.clip((img_np + 1) / 2, 0, 1)  # [-1,1] -> [0,1]
                plt.imsave(os.path.join(img_dir, f"{i:03d}.png"),
                           img_np, cmap="gray")

                # Re-evaluate learnability on the final generated image
                scores = compute_learnability(
                    appraiser, generated, args.finetune_layers,
                    args.inner_loop_steps, args.inner_loop_lr)
                scores["category"] = cat_name
                scores["index"] = i
                scores["guidance_scale"] = scale
                # Also store the generation-time metrics
                scores["gen_initial_loss"] = final_step_data["initial_loss"]
                scores["gen_final_loss"] = final_step_data["final_loss"]
                ccml_results.append(scores)

                print(f"  [{i+1}/{args.num_generated}] L_0={scores['L_0']:.4f} "
                      f"improvement={scores['improvement']:.4f}")

            all_results[cat_name] = ccml_results

    # Save all results
    scores_path = os.path.join(args.output_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved scores to {scores_path}")

    # Plot frontier
    reward_config = {
        "ideal_initial_loss": 0.15,
        "initial_loss_sigma": 0.2,
        "final_loss_temp": 0.1,
    }
    plot_frontier(all_results, os.path.join(args.output_dir, "frontier.png"),
                  reward_config=reward_config)

    # Also save a summary table
    print("\n=== Full Summary ===")
    print(f"{'Category':<25} {'n':<5} {'L_0 mean':<12} {'Improv mean':<12}")
    print("-" * 55)
    for cat, res in all_results.items():
        L0 = [r["L_0"] for r in res]
        imp = [r["improvement"] for r in res]
        print(f"{cat:<25} {len(res):<5} {np.mean(L0):<12.4f} {np.mean(imp):<12.4f}")


if __name__ == "__main__":
    main()
