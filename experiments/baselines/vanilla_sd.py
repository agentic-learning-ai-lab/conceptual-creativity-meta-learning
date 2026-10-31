#!/usr/bin/env python3
"""
Vanilla Stable Diffusion baseline + truncation-artifact ablation.

Two modes (selected via --mode):
  * full        : standard SD pipeline using the full 77-position prompt embedding
                  (the true "original creator" baseline).
  * trunc_pad   : mimics the embedding path used in text_embedding_creative.py
                  at creativity_scale=0 — tokenize with max_length=N,
                  re-encode the truncated tokens, take positions [1:N+1],
                  zero-pad to 77. Isolates the tokenization/zero-pad artifact
                  from any meta-learning contribution.

Output layout matches text_embedding_creative.py.
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def build_text_embeddings_full(pipe, prompt, device):
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        cond_emb = text_encoder(tokens.input_ids)[0]
    uncond_tokens = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        uncond_emb = text_encoder(uncond_tokens.input_ids)[0]
    return uncond_emb, cond_emb


def build_text_embeddings_trunc_pad(pipe, prompt, num_tokens, device):
    """Mimic text_embedding_creative.py:99 — truncated re-encoding then zero pad."""
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    max_len = tokenizer.model_max_length

    with torch.no_grad():
        tokens = tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=num_tokens,
            return_tensors="pt",
        ).to(device)
        text_emb = text_encoder(tokens.input_ids)[0]
        init_emb = text_emb[0, 1:num_tokens + 1, :].unsqueeze(0)
        # zero-pad up to max_len
        pad = torch.zeros(
            1, max_len - init_emb.size(1), init_emb.size(2),
            device=device, dtype=init_emb.dtype,
        )
        cond_emb = torch.cat([init_emb, pad], dim=1)

        uncond_tokens = tokenizer(
            "",
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        uncond_emb = text_encoder(uncond_tokens.input_ids)[0]

    return uncond_emb, cond_emb


def generate_with_embeddings(pipe, uncond_emb, cond_emb, num_inference_steps, guidance_scale, generator, device):
    """Run SD denoising loop with pre-built text embeddings (no high-level pipe call)."""
    scheduler = pipe.scheduler
    unet = pipe.unet
    vae = pipe.vae

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    latent_shape = (1, unet.config.in_channels, 64, 64)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=unet.dtype)
    latents = latents * scheduler.init_noise_sigma

    text_embeddings = torch.cat([uncond_emb, cond_emb]).to(unet.dtype)

    for t in timesteps:
        latent_model_input = torch.cat([latents, latents])
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    with torch.no_grad():
        image = vae.decode(latents / vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--prompt", required=True)
    p.add_argument("--mode", choices=["full", "trunc_pad"], default="full")
    p.add_argument("--num_tokens", type=int, default=8,
                   help="Used only in trunc_pad mode; matches num_tokens in text_embedding_creative.")
    p.add_argument("--num_images", type=int, default=20)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--output_dir", default="outputs/text_embedding_creative")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default=None,
                   help="Run dir name under output_dir; default vanilla_full_cfg{scale}_{timestamp}")
    args = p.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    suffix = f"{timestamp}_{job_id}" if job_id else f"{timestamp}_{os.getpid()}"
    if args.mode == "full":
        run_name = f"vanilla_full_cfg{args.guidance_scale}_{suffix}"
    else:
        run_name = f"vanilla_truncpad{args.num_tokens}_cfg{args.guidance_scale}_{suffix}"
    run_name = args.run_name or run_name
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config = vars(args).copy()
    config["run_dir"] = run_dir
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model, torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    if args.mode == "full":
        uncond_emb, cond_emb = build_text_embeddings_full(pipe, args.prompt, device)
    else:
        uncond_emb, cond_emb = build_text_embeddings_trunc_pad(pipe, args.prompt, args.num_tokens, device)

    print(f"Mode: {args.mode}")
    print(f"Prompt: {args.prompt}")
    print(f"Generating {args.num_images} images @ cfg={args.guidance_scale}, steps={args.num_inference_steps}")

    t0 = time.time()
    for i in range(args.num_images):
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        image = generate_with_embeddings(
            pipe, uncond_emb, cond_emb,
            args.num_inference_steps, args.guidance_scale, gen, device,
        )
        img_pil = Image.fromarray(
            (image[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        )
        img_pil.save(os.path.join(run_dir, f"image_{i:03d}.png"))
        print(f"  [{i+1}/{args.num_images}] saved", flush=True)

    print(f"Done in {time.time()-t0:.1f}s -> {run_dir}")


if __name__ == "__main__":
    main()
