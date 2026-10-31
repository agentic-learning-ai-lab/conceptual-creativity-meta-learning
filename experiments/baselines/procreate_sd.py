#!/usr/bin/env python3
"""ProCreate baseline for CCML.

Faithful port of ProCreate's propulsive-energy guidance
    Lu et al., "ProCreate, Don't Reproduce! Propulsive Energy Diffusion
    for Creative Generation", ECCV 2024.
    https://github.com/agentic-learning-ai-lab/procreate-diffusion

Original uses DreamSim as the reference-image encoder; we substitute **OpenAI CLIP ViT-L/14** (the
same image tower CCML already uses). This changes the metric identity of
the baseline but preserves the algorithm: at each denoising step we
estimate pred_x0 via a DDIM step, decode via the VAE, embed with the
image tower, and take a gradient step on the latent to *minimize* cosine
similarity to the nearest reference embedding.

Reference set: vanilla SD-v1.5 (cfg=7.5, 100+ steps) samples for the same
prompt, taken from an existing directory of image_*.png files. Both
originals and horizontal mirrors are embedded (ProCreate does this too).

Hyperparameters follow the ProCreate paper except `dreamsim_w` (called
`gamma` here) is swept. See
`experiments/prompts/procreate_baseline_v1.json` for the 20-prompt
manifest.

Example
-------
    python -m experiments.baselines.procreate_sd \
        --prompt "a photo of a peacock" \
        --ref_dir outputs/text_embedding_creative/vanilla_full_cfg7.5_.../ \
        --gamma 10 --num_images 100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor, CLIPModel


# -----------------------------------------------------------------------------
# Reference embedding
# -----------------------------------------------------------------------------
class ReferenceEncoder(torch.nn.Module):
    """CLIP image tower used in place of DreamSim.

    Exposes:
      - `embed_pil(pil_images)`: for pre-computing reference embeddings
      - `embed_tensor(images)`:  for embedding decoded pred_x0 with grad

    All outputs are L2-normalised so cosine similarity is a dot product.
    """

    def __init__(self, clip_model_name: str, device: str, dtype=torch.float32):
        super().__init__()
        model = CLIPModel.from_pretrained(clip_model_name)
        self.vision_model = model.vision_model.to(device=device, dtype=dtype)
        self.visual_projection = model.visual_projection.to(device=device, dtype=dtype)
        self.processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.device = device
        self.dtype = dtype

        # CLIP normalisation stats (image processor defaults for ViT-L/14).
        mean = torch.tensor(self.processor.image_mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=device, dtype=dtype).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)
        self.target_size = self.processor.crop_size["height"] \
            if isinstance(self.processor.crop_size, dict) else self.processor.crop_size

    def _project(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs[1]
        emb = self.visual_projection(pooled)
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def embed_pil(self, pil_images: List[Image.Image]) -> torch.Tensor:
        proc = self.processor(images=pil_images, return_tensors="pt")
        px = proc["pixel_values"].to(device=self.device, dtype=self.dtype)
        return self._project(px)

    def embed_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) in [0, 1]. Grad flows."""
        x = F.interpolate(
            images, size=(self.target_size, self.target_size),
            mode="bilinear", align_corners=False, antialias=True,
        )
        x = (x - self._mean) / self._std
        return self._project(x.to(self.dtype))


class DreamSimReferenceEncoder(torch.nn.Module):
    """DreamSim ensemble encoder (paper-faithful).

    Wraps `dreamsim.dreamsim(pretrained=True, dreamsim_type='ensemble')` and
    exposes the same (embed_pil, embed_tensor) API as ReferenceEncoder.
    Ensemble concatenates DINO ViT-B/16 + CLIP ViT-B/16 + OpenCLIP ViT-B/16
    features, so this is the actual perceptual similarity ProCreate paper uses.
    """

    def __init__(self, cache_dir: str, device: str, dtype=torch.float32):
        super().__init__()
        from dreamsim import dreamsim as _load_dreamsim
        model, prep = _load_dreamsim(
            pretrained=True, device=device, cache_dir=cache_dir,
            dreamsim_type="ensemble", normalize_embeds=True,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.prep = prep
        self.device = device
        self.dtype = dtype
        self.target_size = 224

    @torch.no_grad()
    def embed_pil(self, pil_images):
        px = torch.cat([self.prep(im) for im in pil_images], dim=0).to(self.device)
        emb = self.model.embed(px)
        return F.normalize(emb, dim=-1)

    def embed_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) in [0, 1]. Grad flows."""
        x = F.interpolate(
            images, size=(self.target_size, self.target_size),
            mode="bilinear", align_corners=False, antialias=True,
        )
        # DreamSim's DINO/CLIP backbones are float32; VAE decodes in fp16.
        emb = self.model.embed(x.float())
        return F.normalize(emb, dim=-1)


# -----------------------------------------------------------------------------
# ProCreate DDIM sampler
# -----------------------------------------------------------------------------
def build_text_embeddings(pipe, prompt: str, device: str):
    tk = pipe.tokenizer
    te = pipe.text_encoder
    cond = tk(prompt, padding="max_length", max_length=tk.model_max_length,
              truncation=True, return_tensors="pt").to(device)
    uncond = tk("", padding="max_length", max_length=tk.model_max_length,
                truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        cond_emb = te(cond.input_ids)[0]
        uncond_emb = te(uncond.input_ids)[0]
    return uncond_emb, cond_emb


def _decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to pixel space in [0,1]. Grad flows for guidance."""
    x = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor).sample
    return (x / 2 + 0.5).clamp(0, 1)


def _cfg_noise_pred(unet, latents, t, text_embeddings, guidance_scale, scheduler):
    latent_in = torch.cat([latents, latents])
    latent_in = scheduler.scale_model_input(latent_in, t)
    noise = unet(latent_in, t, encoder_hidden_states=text_embeddings).sample
    noise_u, noise_c = noise.chunk(2)
    return noise_u + guidance_scale * (noise_c - noise_u)


def _pred_x0_from_noise(scheduler, noise_pred, t, latents):
    """DDIM predicted clean sample from a single noise prediction."""
    alpha_prod_t = scheduler.alphas_cumprod[t].to(latents.device)
    beta_prod_t = 1 - alpha_prod_t
    return (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()


def procreate_generate(
    pipe,
    ref_encoder: ReferenceEncoder,
    ref_embeddings: torch.Tensor,
    uncond_emb: torch.Tensor,
    cond_emb: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    gamma: float,
    max_grad_norm: float,
    generator: torch.Generator,
    device: str,
) -> torch.Tensor:
    """Run one ProCreate-guided DDIM sampling trajectory. Returns image in [0,1]."""
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
        if gamma != 0.0:
            latents = latents.detach().requires_grad_(True)

        noise_pred = _cfg_noise_pred(
            unet, latents, t, text_embeddings, guidance_scale, scheduler
        )

        if gamma == 0.0:
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            continue

        # Propulsive gradient on latents from nearest-reference cosine sim.
        pred_x0 = _pred_x0_from_noise(scheduler, noise_pred, t, latents)
        pred_img = _decode_latents(vae, pred_x0)             # (1,3,512,512)
        pred_emb = ref_encoder.embed_tensor(pred_img).float()  # (1, D)

        with torch.no_grad():
            sims_all = pred_emb @ ref_embeddings.T           # (1, N_ref)
            closest_idx = sims_all.argmax(dim=1)             # (1,)
        target_emb = ref_embeddings[closest_idx]             # (1, D)
        loss = F.cosine_similarity(pred_emb, target_emb, dim=-1).mean() * gamma

        grad = torch.autograd.grad(loss, latents)[0]
        # ProCreate's elementwise clamp + grad-norm cap.
        grad = torch.clamp(grad, min=-gamma, max=gamma)
        gnorm = grad.norm()
        cap = gamma * max_grad_norm
        if gnorm > cap:
            grad = grad * (cap / gnorm)

        # Apply guidance to noise prediction, scaled by (1 - alpha_bar_t)^0.5.
        alpha_prod_t = scheduler.alphas_cumprod[t].to(latents.device)
        noise_pred_g = noise_pred.detach() + (1.0 - alpha_prod_t).sqrt() * grad

        latents = scheduler.step(noise_pred_g, t, latents.detach()).prev_sample
        latents = latents.detach()

    with torch.no_grad():
        image = _decode_latents(vae, latents)
    return image


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def _load_reference_images(ref_dir: str, num_ref: int, seed: int) -> List[Image.Image]:
    paths = sorted(glob.glob(os.path.join(ref_dir, "image_*.png")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(ref_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNGs found in {ref_dir}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(num_ref, len(paths)), replace=False)
    return [Image.open(paths[int(i)]).convert("RGB") for i in idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--prompt", required=True)
    p.add_argument("--ref_dir", required=True,
                   help="Directory of vanilla-SD reference PNGs for this prompt.")
    p.add_argument("--num_ref", type=int, default=100)
    p.add_argument("--gamma", type=float, required=True,
                   help="Propulsion scale (ProCreate's dreamsim_w).")
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--num_images", type=int, default=100)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--output_dir", default="outputs/procreate_sd")
    p.add_argument("--run_name", default=None,
                   help="Override auto-generated run directory name.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder", choices=("clip", "dreamsim"), default="clip",
                   help="Reference encoder: 'clip' (CLIP ViT-L/14 substitute) or "
                        "'dreamsim' (paper-faithful ensemble; requires "
                        "dreamsim package + DreamSim cache).")
    p.add_argument("--dreamsim_cache", default=os.path.expanduser("~/.cache/dreamsim"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    suffix = "_".join(x for x in [timestamp, job_id, array_id, str(os.getpid())] if x)
    run_name = args.run_name or f"procreate_gamma{args.gamma:g}_{suffix}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config = vars(args).copy()
    config["run_dir"] = run_dir
    if args.encoder == "clip":
        config["note"] = ("CLIP ViT-L/14 substituted for DreamSim in this run. "
                          "Algorithm otherwise faithful to ProCreate ECCV 2024.")
    else:
        config["note"] = ("DreamSim ensemble encoder (paper-faithful) used as "
                          "reference similarity. ProCreate ECCV 2024.")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[procreate] run_dir = {run_dir}", flush=True)
    print(f"[procreate] prompt  = {args.prompt!r}", flush=True)
    print(f"[procreate] gamma   = {args.gamma}   max_grad_norm = {args.max_grad_norm}", flush=True)
    print(f"[procreate] ref_dir = {args.ref_dir}", flush=True)

    # SD pipeline with DDIM scheduler (matches ProCreate).
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    # Freeze; we backprop only through pred_x0 -> CLIP.
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Reference encoder + reference embeddings (originals + horizontal mirrors).
    if args.encoder == "clip":
        ref_encoder = ReferenceEncoder(args.clip_model, device=device, dtype=torch.float32)
    else:
        ref_encoder = DreamSimReferenceEncoder(args.dreamsim_cache, device=device, dtype=torch.float32)
    ref_pil = _load_reference_images(args.ref_dir, args.num_ref, seed=args.seed)
    ref_pil_aug = ref_pil + [ImageOps.mirror(im) for im in ref_pil]
    print(f"[procreate] embedding {len(ref_pil_aug)} reference images...", flush=True)
    ref_embeddings = ref_encoder.embed_pil(ref_pil_aug).float()
    print(f"[procreate] ref_embeddings shape: {tuple(ref_embeddings.shape)}", flush=True)

    uncond_emb, cond_emb = build_text_embeddings(pipe, args.prompt, device)

    t0 = time.time()
    for i in range(args.num_images):
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        image = procreate_generate(
            pipe=pipe,
            ref_encoder=ref_encoder,
            ref_embeddings=ref_embeddings,
            uncond_emb=uncond_emb,
            cond_emb=cond_emb,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            gamma=args.gamma,
            max_grad_norm=args.max_grad_norm,
            generator=gen,
            device=device,
        )
        arr = (image[0].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(run_dir, f"image_{i:03d}.png"))
        if (i + 1) % 5 == 0 or i == args.num_images - 1:
            print(f"  [{i+1}/{args.num_images}] {time.time()-t0:.1f}s elapsed", flush=True)

    print(f"[procreate] done -> {run_dir} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
