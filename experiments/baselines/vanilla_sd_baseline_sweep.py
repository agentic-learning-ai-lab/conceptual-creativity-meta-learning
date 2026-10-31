#!/usr/bin/env python3
"""
Vanilla Stable Diffusion baseline with CFG-scale and negative-prompt support.

Baseline runner for the diversity baselines; it does not overwrite the
vanilla_sd.py output layout. Two experiments call this script:

  * CFG-scale sweep       — one cell per (prompt, cfg) with negative_prompt="".
  * Negative-prompt sweep — one cell per (prompt, negative_prompt) at cfg=7.5.

Each invocation processes ONE (prompt, cfg, negative_prompt) cell and writes
its images into a run directory named by --run_name.
Model + text embeddings are built once; generation loops for num_images.

Output layout:
    {output_dir}/{run_name}/
        config.json
        image_000.png ... image_{N-1:03d}.png
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


def build_text_embeddings(pipe, prompt, negative_prompt, device):
    """Return (uncond_emb, cond_emb) for classifier-free guidance.

    uncond_emb is the embedding of `negative_prompt` (empty string reproduces
    the standard unconditional path). cond_emb is the embedding of `prompt`.
    Matches the diffusers convention: with CFG, uncond acts as the *negative*.
    """
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    max_len = tokenizer.model_max_length

    def encode(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            emb = text_encoder(tokens.input_ids)[0]
        return emb

    cond_emb = encode(prompt)
    uncond_emb = encode(negative_prompt if negative_prompt else "")
    return uncond_emb, cond_emb


def generate_one(pipe, uncond_emb, cond_emb, num_inference_steps, guidance_scale,
                 generator, device):
    scheduler = pipe.scheduler
    unet = pipe.unet
    vae = pipe.vae

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    latent_shape = (1, unet.config.in_channels, 64, 64)
    latents = torch.randn(latent_shape, generator=generator, device=device,
                          dtype=unet.dtype)
    latents = latents * scheduler.init_noise_sigma

    text_embeddings = torch.cat([uncond_emb, cond_emb]).to(unet.dtype)

    for t in timesteps:
        latent_model_input = torch.cat([latents, latents])
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t,
                              encoder_hidden_states=text_embeddings).sample
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
    p.add_argument("--negative_prompt", default="",
                   help="Empty string uses the standard unconditional path.")
    p.add_argument("--num_images", type=int, default=100)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--output_dir", required=True,
                   help="Parent directory; each cell writes a run_name subdir here.")
    p.add_argument("--run_name", default="",
                   help="Subdirectory name for this cell. Auto-generated if empty.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.run_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        jid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        neg_tag = "neg" if args.negative_prompt else "noneg"
        args.run_name = f"vanilla_cfg{args.guidance_scale}_{neg_tag}_{ts}_{jid}"

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    config = vars(args).copy()
    config["run_dir"] = run_dir
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.sd_model} on {device} ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model, torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    uncond_emb, cond_emb = build_text_embeddings(
        pipe, args.prompt, args.negative_prompt, device,
    )

    print(f"Prompt          : {args.prompt}")
    print(f"Negative prompt : {args.negative_prompt!r}")
    print(f"CFG             : {args.guidance_scale}")
    print(f"Num images      : {args.num_images}")
    print(f"Inference steps : {args.num_inference_steps}")
    print(f"Run dir         : {run_dir}")

    t0 = time.time()
    for i in range(args.num_images):
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        image = generate_one(
            pipe, uncond_emb, cond_emb,
            args.num_inference_steps, args.guidance_scale, gen, device,
        )
        img_pil = Image.fromarray(
            (image[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        )
        img_pil.save(os.path.join(run_dir, f"image_{i:03d}.png"))
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{args.num_images}] "
                  f"({(time.time()-t0)/(i+1):.1f}s/img)", flush=True)

    print(f"Done in {time.time()-t0:.1f}s -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
