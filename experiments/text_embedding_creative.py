#!/usr/bin/env python3
"""
Creator-Appraiser generation on natural images with the CLIP Appraiser.

The Creator is a frozen Stable Diffusion model with a learnable text embedding.
At each denoising step, the Appraiser fine-tunes a low-rank adapter on CLIP's
text projection for a few steps to match the predicted clean image; the
similarity improvement is the reward, and its meta-gradient updates the text
embedding.

Run from project root (see README.md for the paper's settings):
    python -m experiments.text_embedding_creative --lora_rank 8 --class_label 291
"""

import argparse
import re
import copy
import json
import os
import random
import sys
import time
import urllib.request
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm.auto import tqdm

import higher  # For meta-gradient (MAML-style differentiable optimization)

from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel, CLIPTokenizer
from torchvision import datasets, transforms

from ccml.training.clip_similarity import CLIPSimilarity


# ImageNet class labels
IMAGENET_LABELS = None


def get_imagenet_label(class_id: int) -> str:
    """Get ImageNet class name from class ID."""
    global IMAGENET_LABELS
    if IMAGENET_LABELS is None:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                IMAGENET_LABELS = json.loads(response.read().decode())
        except Exception:
            IMAGENET_LABELS = []

    if 0 <= class_id < len(IMAGENET_LABELS):
        return IMAGENET_LABELS[class_id]
    return f"class_{class_id}"


class LearnableTextEmbedding(nn.Module):
    """Learnable text embedding for creativity guidance."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_tokens: int = 8,
        init_strategy: str = "from_text",
        init_text: str = None,
        tokenizer=None,
        text_encoder=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        if init_strategy == "random":
            self.embedding = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.01)
        elif init_strategy == "zeros":
            self.embedding = nn.Parameter(torch.zeros(num_tokens, embed_dim))
        elif init_strategy == "from_text" and init_text and tokenizer and text_encoder:
            with torch.no_grad():
                tokens = tokenizer(
                    init_text,
                    padding=False,
                    truncation=True,
                    max_length=num_tokens,
                    return_tensors="pt",
                )
                text_emb = text_encoder(tokens.input_ids.to(text_encoder.device))[0]
                init_emb = text_emb[0, 1:num_tokens+1, :]
            self.embedding = nn.Parameter(init_emb.clone())
        elif init_strategy == "from_full_prompt" and init_text and tokenizer and text_encoder:
            with torch.no_grad():
                tokens = tokenizer(
                    init_text,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                text_emb = text_encoder(tokens.input_ids.to(text_encoder.device))[0]
                init_emb = text_emb[0]
            self.num_tokens = init_emb.size(0)
            self.embedding = nn.Parameter(init_emb.clone())
        else:
            self.embedding = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.01)

    def forward(self) -> torch.Tensor:
        """Return embedding with batch dimension: (1, num_tokens, embed_dim)"""
        return self.embedding.unsqueeze(0)


class TextProjLoRA(nn.Module):
    """Low-rank adapter for CLIP's text_projection layer.

    Re-initialized to zero (no perturbation) at the start of each inner loop.
    The K-step inner optimization fine-tunes A, B so that the LoRA-adapted
    projection of the prompt matches the positive image's features.

    Capacity is bounded by rank r (typical r=8). With r << 768 the inner
    optimization cannot trivially memorize any image's direction in K steps.
    """
    def __init__(self, dim: int = 768, rank: int = 8):
        super().__init__()
        # delta_W = A @ B, A: (dim, r), B: (r, dim).  init A=0 -> delta_W = 0
        self.A = nn.Parameter(torch.zeros(dim, rank))
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.01)

    def delta_W(self) -> torch.Tensor:
        return self.A @ self.B    # (dim, dim), rank <= r

    def forward(self, e_text_post_proj: torch.Tensor, e_text_pre_proj: torch.Tensor) -> torch.Tensor:
        """Adapted text features = original text_projection output + delta_W @ pre-projection input.

        e_text_post_proj: (1, dim) — clip.text_projection(e_text_pre_proj), frozen output
        e_text_pre_proj:  (1, dim) — pooled output of clip.text_model, frozen
        """
        delta = e_text_pre_proj @ self.delta_W().t()  # (1, dim)
        return F.normalize(e_text_post_proj + delta, dim=-1)


class CLIPLearnabilityAppraiser(nn.Module):
    """
    Frozen CLIP. The inner loop adapts a low-rank LoRA on text_projection
    to better match the positive image's features. Reward = sim_T - sim_0.
    """

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-large-patch14",
        lora_rank: int = 8,
        device: str = "cuda",
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        self.device = device
        self.proj_dim = self.clip.config.projection_dim
        self.lora_rank = lora_rank
        self._text_cache = {}

    def get_text_features_pre_post(self, text: str):
        """Returns (pre_proj_pooled, post_proj_unnormalized) — both (1, dim)."""
        if text not in self._text_cache:
            tokens = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                txt_outputs = self.clip.text_model(**tokens)
                pre = txt_outputs.pooler_output                   # (1, hidden_size)
                post = self.clip.text_projection(pre)             # (1, projection_dim)
            self._text_cache[text] = (pre, post)
        return self._text_cache[text]

    def encode_image_pixels(self, pixel_values: torch.Tensor, with_grad: bool = False) -> torch.Tensor:
        def _forward():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output                 # (B, vision_hidden_size)
            return self.clip.visual_projection(pooled)            # (B, projection_dim)
        if with_grad:
            feats = _forward()
        else:
            with torch.no_grad():
                feats = _forward()
        return F.normalize(feats, dim=-1)

    def preprocess_tensor(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        return (images - mean) / std


def run_clip_learnability_inner_loop_meta(
    appraiser: CLIPLearnabilityAppraiser,
    positive_images: torch.Tensor,
    target_text: str,
    inner_loop_steps: int = 10,
    inner_loop_lr: float = 1e-3,
    log_inner: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    CLIP-learnability inner loop with meta-gradient.

    1. Encode positive image (with grad to text emb) and prompt text (frozen, cached).
    2. Initialize learnable LoRA on text_projection (A=0).
    3. K-step SGD on (A, B) to maximize cos_sim(image_pos, lora_adapted_text).
    4. Reward = sim_T - sim_0.

    Returns (initial_loss, final_loss, initial_sim, final_sim) where
    initial_loss = -sim_0, final_loss = -sim_T (so L_0 - L_T = sim_T - sim_0 = improvement).
    """
    e_text_pre, e_text_post = appraiser.get_text_features_pre_post(target_text)
    e_pos = appraiser.encode_image_pixels(positive_images, with_grad=True)  # (B_pos, D), normalized

    lora = TextProjLoRA(dim=appraiser.proj_dim, rank=appraiser.lora_rank).to(appraiser.device)
    optimizer = optim.Adam(lora.parameters(), lr=inner_loop_lr)

    def sim_loss(flora_module):
        e_text_adapted = flora_module(e_text_post, e_text_pre)     # (1, D)
        sim = (e_pos @ e_text_adapted.t()).squeeze(-1)             # (B_pos,)
        return -sim.mean()

    initial_loss = sim_loss(lora)
    initial_sim = (-initial_loss).item()
    if log_inner:
        print(f"  inner step 0 (init): sim={initial_sim:.4f}", flush=True)

    appraiser.train()
    with higher.innerloop_ctx(lora, optimizer, copy_initial_weights=True) as (flora, diffopt):
        for step in range(inner_loop_steps):
            loss = sim_loss(flora)
            diffopt.step(loss)
            if log_inner:
                with torch.no_grad():
                    s = -loss.item()
                print(f"  inner step {step+1}: sim={s:.4f}", flush=True)
        final_loss = sim_loss(flora)
        final_sim = (-final_loss).item()
    return initial_loss, final_loss, initial_sim, final_sim


def generate_with_text_embedding_guidance(
    pipe: StableDiffusionPipeline,
    appraiser: CLIPLearnabilityAppraiser,
    learnable_embedding: LearnableTextEmbedding,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    creativity_scale: float = 100.0,
    inner_loop_steps: int = 10,
    inner_loop_lr: float = 1e-3,
    max_grad_norm: float = 1.0,
    clip_similarity: CLIPSimilarity = None,
    clip_weight: float = 0.0,
    target_text: str = None,
    use_novelty_weight: bool = False,
    seed: int = None,
    device: str = "cuda",
    visualize: bool = True,
) -> dict:
    """
    Generate an image with Creator-Appraiser guidance on the text embedding.

    At each denoising step:
    1. Run one denoising step with current text embedding
    2. Decode the predicted clean image
    3. Fine-tune the Appraiser's text-projection LoRA to match the image
    4. Compute the reward as the similarity improvement
    5. Backprop gradient to text embedding
    6. Apply creativity guidance to text embedding
    """
    if seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    vae = pipe.vae
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    uncond_tokens = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    uncond_embedding = text_encoder(uncond_tokens.input_ids)[0]

    latent_shape = (1, unet.config.in_channels, 64, 64)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=pipe.unet.dtype)
    latents = latents * scheduler.init_noise_sigma

    reward_history = []

    pbar = tqdm(enumerate(timesteps), total=len(timesteps), desc="Generating", disable=not visualize)
    for i, t in pbar:
        learnable_embedding.requires_grad_(True)

        text_emb = learnable_embedding()
        max_len = tokenizer.model_max_length

        if text_emb.size(1) < max_len:
            padding = torch.zeros(
                1, max_len - text_emb.size(1), text_emb.size(2),
                device=device, dtype=text_emb.dtype
            )
            text_emb_padded = torch.cat([text_emb, padding], dim=1)
        else:
            text_emb_padded = text_emb[:, :max_len, :]

        text_emb_padded = text_emb_padded.to(dtype=uncond_embedding.dtype)
        text_embeddings = torch.cat([uncond_embedding, text_emb_padded])

        latent_model_input = torch.cat([latents, latents])
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict noise - need gradients for text embedding guidance
        # Run unconditional without grad, conditional with grad
        with torch.no_grad():
            noise_pred_uncond = unet(
                latent_model_input[:1],
                t,
                encoder_hidden_states=uncond_embedding,
            ).sample

        # Conditional prediction WITH gradients (connects text_embeddings to output)
        noise_pred_cond = unet(
            latent_model_input[1:],
            t,
            encoder_hidden_states=text_emb_padded,
        ).sample

        # CFG combination
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Compute predicted x0 for appraiser (keeps gradient connection to text embedding)
        # x0 = (x_t - sqrt(1-alpha) * noise) / sqrt(alpha)
        alpha_prod_t = scheduler.alphas_cumprod[t.long().cpu()].to(device)
        pred_x0 = (latents - (1 - alpha_prod_t) ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

        # Decode pred_x0 (gradient connected to text embedding via noise_pred_cond)
        decoded = vae.decode(pred_x0 / vae.config.scaling_factor).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)

        # Debug: check gradient flow on first step
        if i == 0 and visualize:
            print(f"\n[DEBUG] Gradient flow check:", flush=True)
            print(f"  text_emb_padded.requires_grad: {text_emb_padded.requires_grad}", flush=True)
            print(f"  noise_pred_cond.requires_grad: {noise_pred_cond.requires_grad}", flush=True)
            print(f"  noise_pred.requires_grad: {noise_pred.requires_grad}", flush=True)
            print(f"  pred_x0.requires_grad: {pred_x0.requires_grad}", flush=True)
            print(f"  decoded.requires_grad: {decoded.requires_grad}", flush=True)

        decoded_clip = appraiser.preprocess_tensor(decoded)

        # Log per-step inner-loop progress at the first denoising step (one image only)
        log_inner = visualize and (i == 0)
        if log_inner:
            print(f"\n[INNER LOOP TRACE — denoising step {i}]", flush=True)

        # CLIP-learnability inner loop: text-direction offset fine-tune
        initial_loss, final_loss, initial_acc, final_acc = run_clip_learnability_inner_loop_meta(
            appraiser,
            decoded_clip,
            target_text or "",
            inner_loop_steps,
            inner_loop_lr,
            log_inner=log_inner,
        )

        # Reward = contrastive learnability (L_0 - L_T). Optional curriculum
        # weighting (1 + (1 - acc_0)) is gated; acc_0 comes from a non-
        # differentiable argmax and contributes only a per-image scalar
        # multiplier (1x or 2x), not gradient direction.
        improvement = initial_loss - final_loss
        novelty = 1.0 - initial_acc
        if use_novelty_weight:
            base_reward = improvement * (1.0 + novelty)
        else:
            base_reward = improvement

        # Add CLIP-text similarity for semantic regularization
        clip_sim = 0.0
        if clip_similarity is not None and clip_weight > 0 and target_text:
            clip_sim = clip_similarity(decoded, target_text, with_grad=True)
            reward = base_reward + clip_weight * clip_sim
        else:
            reward = base_reward

        reward_history.append({
            "step": i,
            "timestep": t.item(),
            "initial_loss": initial_loss.item(),
            "final_loss": final_loss.item(),
            "initial_acc": initial_acc,
            "final_acc": final_acc,
            "improvement": improvement.item(),
            "novelty": novelty,
            "clip_sim": clip_sim.item() if isinstance(clip_sim, torch.Tensor) else clip_sim,
            "reward": reward.item(),
        })

        if visualize:
            pbar.set_postfix({
                "reward": f"{reward.item():.4f}",
                "improv": f"{improvement.item():.4f}",
                "acc": f"{initial_acc:.2f}→{final_acc:.2f}",
            })

        if creativity_scale > 0 and i < len(timesteps) - 1:
            if torch.isnan(reward) or torch.isinf(reward):
                if visualize:
                    print(f"  [WARN] Step {i}: NaN/Inf reward, skipping guidance")
            else:
                grad = torch.autograd.grad(reward, learnable_embedding.embedding, retain_graph=False)[0]

                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    if visualize:
                        print(f"  [WARN] Step {i}: NaN/Inf gradient, skipping guidance")
                else:
                    grad_norm = grad.norm()

                    # Debug: print gradient info on first step
                    if i == 0 and visualize:
                        print(f"  grad.norm(): {grad_norm.item():.6f}", flush=True)
                        print(f"  embedding.norm(): {learnable_embedding.embedding.norm().item():.4f}", flush=True)

                    if grad_norm > max_grad_norm:
                        grad = grad * (max_grad_norm / grad_norm)

                    with torch.no_grad():
                        learnable_embedding.embedding.add_(creativity_scale * grad)

        with torch.no_grad():
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    with torch.no_grad():
        final_image = vae.decode(latents / vae.config.scaling_factor).sample
        final_image = (final_image / 2 + 0.5).clamp(0, 1)

    final_image_pil = Image.fromarray(
        (final_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )

    return {
        "image": final_image.cpu(),
        "image_pil": final_image_pil,
        "text_embedding": learnable_embedding.embedding.detach().cpu(),
        "reward_history": reward_history,
        "final_reward": reward_history[-1]["reward"] if reward_history else 0.0,
    }


def visualize_result(result, class_name, output_path):
    """Visualize generated image with reward curve."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    img = result["image"][0].float().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    axes[0].imshow(img)
    init_text = result.get("init_text", class_name)
    axes[0].set_title(f"Prompt: {init_text}", fontsize=10)
    axes[0].axis("off")

    rewards = [r["reward"] for r in result["reward_history"]]
    axes[1].plot(rewards)
    axes[1].set_title(f"Final Reward: {result['final_reward']:.4f}")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_ranked_grid(results, output_path, num_per_row=5):
    """Create ranked grid of all generated images sorted by reward."""
    sorted_results = sorted(results, key=lambda x: x["final_reward"], reverse=True)

    num_images = len(sorted_results)
    num_rows = (num_images + num_per_row - 1) // num_per_row

    fig, axes = plt.subplots(num_rows, num_per_row, figsize=(num_per_row * 2.5, num_rows * 3))

    if num_rows == 1 and num_per_row == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_per_row == 1:
        axes = axes.reshape(-1, 1)

    for idx, result in enumerate(sorted_results):
        row = idx // num_per_row
        col = idx % num_per_row

        ax = axes[row, col]
        img = result["image"][0].float().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        title = result.get("init_text") or result.get("class_name", "unknown")
        if len(title) > 28:
            title = title[:25] + "..."
        ax.set_title(f"{title}\nR:{result['final_reward']:.3f}", fontsize=8)
        ax.axis("off")

    for idx in range(num_images, num_rows * num_per_row):
        row = idx // num_per_row
        col = idx % num_per_row
        axes[row, col].axis("off")

    plt.suptitle("Generated Images - Ranked by Reward", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ranked grid to: {output_path}")


class TeeLogger:
    """Write to both stdout and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def isatty(self):
        return self.stdout.isatty()

    def close(self):
        self.file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Creator-Appraiser generation with the CLIP Appraiser")

    # Models
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14",
                        help="CLIP model for the Appraiser")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="Rank of the Appraiser's LoRA on CLIP's text projection")

    # Class conditioning
    parser.add_argument("--class_label", type=int, default=None,
                        help="ImageNet class ID (0-999). Random if not specified.")

    # Text embedding
    parser.add_argument("--num_tokens", type=int, default=8,
                        help="Number of learnable tokens")
    parser.add_argument("--init_strategy", type=str, default="from_text",
                        choices=["random", "zeros", "from_text", "from_full_prompt"],
                        help="Text embedding initialization")
    parser.add_argument("--init_text", type=str, default=None,
                        help="Text for initialization (default: auto from class label)")

    # Generation
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of images to generate")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="CFG scale")
    parser.add_argument("--creativity_scale", type=float, default=50.0,
                        help="Creativity guidance scale")
    parser.add_argument("--clip_weight", type=float, default=0.0,
                        help="Weight for CLIP-text similarity (semantic regularization)")

    # Inner loop
    parser.add_argument("--inner_loop_steps", type=int, default=10,
                        help="Classifier fine-tuning steps")
    parser.add_argument("--inner_loop_lr", type=float, default=1e-3,
                        help="Classifier fine-tuning learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping")
    parser.add_argument("--use_novelty_weight", action="store_true",
                        help="Apply (1+(1-acc_0)) per-image curriculum factor "
                             "to the reward (default off; acc_0 is a coin-flip "
                             "binary 1x/2x scalar, not a differentiable signal)")


    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/text_embedding_creative",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Class selection moved into generation loop for random class per image
    fixed_class_label = args.class_label  # None means random per image

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    suffix = f"{timestamp}_{job_id}" if job_id else timestamp
    # Run dirs are named clipLora_R{rank}_{prompt}_..., the pattern the scoring and
    # analysis scripts look for.
    if args.init_text:
        words = re.sub(r"[^a-z0-9 -]", "", args.init_text.lower()).split()
        if words[:3] == ["a", "photo", "of"]:
            slug = "_".join(words[4:] if len(words) > 3 and words[3] in ("a", "an") else words[3:])
        else:
            slug = "oe_" + "_".join(words[:3])
    elif fixed_class_label is not None:
        slug = get_imagenet_label(fixed_class_label).lower().replace(" ", "_")
    else:
        slug = "random"
    run_name = f"clipLora_R{args.lora_rank:03d}_{slug}_{suffix}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Set up logging
    log_path = os.path.join(run_dir, "log.txt")
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print("=" * 70)
    print("Creator-Appraiser generation with the CLIP Appraiser")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"SD Model: {args.sd_model}")
    print(f"CLIP Model: {args.clip_model}")
    print(f"Class: {fixed_class_label if fixed_class_label is not None else 'random per image'}")
    print(f"Num tokens: {args.num_tokens}")
    print(f"Creativity scale: {args.creativity_scale}")
    print(f"CLIP weight: {args.clip_weight}")
    print(f"Inner loop: {args.inner_loop_steps} steps, lr={args.inner_loop_lr}")
    print("=" * 70)

    # Save config
    config = vars(args).copy()
    config["device"] = device
    config["run_dir"] = run_dir
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load Stable Diffusion
    print("\nLoading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    print(f"Loading CLIP Appraiser (LoRA rank={args.lora_rank})...")
    appraiser = CLIPLearnabilityAppraiser(
        clip_model=args.clip_model,
        lora_rank=args.lora_rank,
        device=device,
    )

    # Load CLIP similarity for semantic regularization
    clip_similarity = None
    if args.clip_weight > 0:
        print(f"Loading CLIP similarity (weight={args.clip_weight})...")
        clip_similarity = CLIPSimilarity(
            model_name=args.clip_model,
            device=device,
        )

    # Generate images
    print(f"\n{'='*70}")
    print(f"Generating {args.num_images} creative images...")
    print(f"{'='*70}")
    print("-" * 70)
    header = f"{'#':<4} {'Reward':<12} {'Improvement':<12} {'Init Acc':<10} {'Final Acc':<10} {'Time(s)':<10}"
    print(header)
    print("-" * 70)

    results = []
    total_start = time.time()

    for img_idx in range(args.num_images):
        start_time = time.time()

        # Select class for this image (random if not fixed)
        if fixed_class_label is not None:
            class_label = fixed_class_label
        else:
            class_label = random.randint(0, 999)
        class_name = get_imagenet_label(class_label)

        # Determine init text
        if args.init_text is not None:
            init_text = args.init_text
        else:
            init_text = f"a photo of a {class_name}"

        print(f"\n[Image {img_idx+1}] Class: {class_label} ({class_name}), Init: '{init_text}'")

        embed_dim = pipe.text_encoder.config.hidden_size
        learnable_embedding = LearnableTextEmbedding(
            embed_dim=embed_dim,
            num_tokens=args.num_tokens,
            init_strategy=args.init_strategy,
            init_text=init_text,
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
        ).to(device)

        result = generate_with_text_embedding_guidance(
            pipe=pipe,
            appraiser=appraiser,
            learnable_embedding=learnable_embedding,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            creativity_scale=args.creativity_scale,
            inner_loop_steps=args.inner_loop_steps,
            inner_loop_lr=args.inner_loop_lr,
            max_grad_norm=args.max_grad_norm,
            clip_similarity=clip_similarity,
            clip_weight=args.clip_weight,
            target_text=init_text,
            use_novelty_weight=args.use_novelty_weight,
            seed=args.seed + img_idx,
            device=device,
            visualize=(img_idx == 0),
        )

        # Store class info in result
        result["class_label"] = class_label
        result["class_name"] = class_name
        result["init_text"] = init_text

        elapsed = time.time() - start_time
        results.append(result)

        last_step = result["reward_history"][-1]
        print(f"{img_idx+1:<4} {result['final_reward']:<12.4f} {last_step['improvement']:<12.4f} "
              f"{last_step['initial_acc']:<10.2f} {last_step['final_acc']:<10.2f} {elapsed:<10.1f}")

        # Save visualization
        vis_path = os.path.join(run_dir, f"image_{img_idx:03d}.png")
        visualize_result(result, class_name, vis_path)

        # Save raw image
        result["image_pil"].save(os.path.join(run_dir, f"raw_{img_idx:03d}.png"))

    print("-" * 70)
    total_time = time.time() - total_start

    # Summary
    avg_reward = np.mean([r["final_reward"] for r in results])

    print(f"\nSummary:")
    print(f"  Total time: {total_time:.1f}s ({total_time/args.num_images:.1f}s per image)")
    print(f"  Average reward: {avg_reward:.4f}")

    # Save results summary
    summary = {
        "args": vars(args),
        "fixed_class_label": fixed_class_label,
        "num_images": len(results),
        "total_time_seconds": total_time,
        "avg_reward": avg_reward,
        "per_image": [
            {
                "index": i,
                "class_label": r["class_label"],
                "class_name": r["class_name"],
                "init_text": r["init_text"],
                "final_reward": r["final_reward"],
                "reward_history": r["reward_history"],
            }
            for i, r in enumerate(results)
        ],
    }

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Generate ranked grid
    ranked_grid_path = os.path.join(run_dir, "ranked_grid.png")
    visualize_ranked_grid(results, ranked_grid_path)

    print(f"\nResults saved to: {run_dir}")

    # Close logger
    sys.stdout = tee.stdout
    tee.close()
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
