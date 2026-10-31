"""Per-image scoring for best-of-N ranking of Creator-Appraiser natural-image runs.

For each image_*.png in a sweep run dir, compute:
  - init_sim: frozen CLIP image-text cosine (the inner-loop initial similarity).
  - delta_sim: K-step LoRA inner loop on the saved image, sim_T - sim_0.
  - vlm_relevance: Qwen2.5-VL judgment of whether the prompt's concept is
    present in the image (0-10). Used as a relevance gate, NOT a creativity
    signal — VLM creativity scores are unreliable.

Writes <run_dir>/scores.json with one entry per image.
Idempotent: re-runs skip images already in scores.json unless --force.

Usage
-----
    python -m experiments.score_samples \
        --runs outputs/text_embedding_creative/clipLora_R008_lion_20261030_120000 \
        --lora_rank 8

    python -m experiments.score_samples \
        --glob 'outputs/text_embedding_creative/clipLora_R*' \
        --skip-vlm
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

# Qwen2.5-VL modeling code uses torch.compiler.is_compiling(), which only exists
# in torch >= 2.3. Some environments pin torch 2.2.2 for CUDA 12.1 compatibility.
# Stub it as a no-op so the model loads and runs in eager mode.
if not hasattr(torch, "compiler") or not hasattr(torch.compiler, "is_compiling"):
    if not hasattr(torch, "compiler"):
        class _CompilerStub: pass
        torch.compiler = _CompilerStub()
    torch.compiler.is_compiling = lambda: False

# Make package imports work regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.text_embedding_creative import (
    CLIPLearnabilityAppraiser,
    get_imagenet_label,
    run_clip_learnability_inner_loop_meta,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def derive_prompt(config: dict) -> str:
    """Recover the text prompt used to generate this run."""
    if config.get("init_text"):
        return config["init_text"]
    if config.get("prompt"):
        return config["prompt"]
    cls = config.get("class_label")
    if cls is not None:
        return f"a photo of a {get_imagenet_label(int(cls))}"
    raise ValueError(f"Cannot derive prompt: neither init_text, prompt, nor class_label set")


def load_image_tensor(path: str, device: str) -> torch.Tensor:
    """Load PNG -> raw [0,1] (1, 3, H, W) tensor on device.

    No resize here: appraiser.preprocess_tensor handles 224x224 interpolation
    + CLIP normalization. This matches the training-time forward (raw decoded
    SD output -> preprocess_tensor) so init_sim numerics are consistent.
    """
    import numpy as np
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return arr.to(device)


# --------------------------------------------------------------------------
# CLIP scores
# --------------------------------------------------------------------------

@torch.no_grad()
def init_sim(appraiser: CLIPLearnabilityAppraiser, img: torch.Tensor, prompt: str) -> float:
    """Frozen CLIP image-text cosine. Equivalent to inner-loop sim_0."""
    img_pp = appraiser.preprocess_tensor(img)
    img_feat = appraiser.encode_image_pixels(img_pp, with_grad=False)  # normalized
    _, e_text_post = appraiser.get_text_features_pre_post(prompt)
    text_feat = F.normalize(e_text_post, dim=-1)
    return (img_feat @ text_feat.t()).squeeze().item()


def delta_sim(
    appraiser: CLIPLearnabilityAppraiser,
    img: torch.Tensor,
    prompt: str,
    inner_loop_steps: int = 10,
    inner_loop_lr: float = 1e-3,
) -> float:
    """Run a fresh K-step LoRA inner loop on this image. Returns sim_T - sim_0."""
    img_pp = appraiser.preprocess_tensor(img)
    _, _, sim_0, sim_T = run_clip_learnability_inner_loop_meta(
        appraiser=appraiser,
        positive_images=img_pp,
        target_text=prompt,
        inner_loop_steps=inner_loop_steps,
        inner_loop_lr=inner_loop_lr,
        log_inner=False,
    )
    return float(sim_T - sim_0)


# --------------------------------------------------------------------------
# Qwen2.5-VL relevance
# --------------------------------------------------------------------------

class QwenVLRelevanceScorer:
    """Wraps Qwen2.5-VL-7B-Instruct to judge concept-presence (0-10).

    Prompt is intentionally narrow: rate concept presence regardless of
    style/unusualness. Style judgment is NOT solicited.
    """

    JSON_RE = re.compile(r"\{[^{}]*\"score\"\s*:\s*\d+[^{}]*\}", re.DOTALL)

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda", load_in_8bit: bool = True):
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            BitsAndBytesConfig,
        )
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        if load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device

    def score(self, image_path: str, prompt: str) -> dict:
        user_text = (
            f"The intended concept is \"{prompt}\". Looking at this image, "
            f"how clearly is the concept of \"{prompt}\" present? "
            f"0 = not present at all, 10 = unambiguously and prominently present. "
            f"Style or unusualness should NOT lower the score as long as the "
            f"concept is recognizable. "
            f"Output ONLY JSON: {{\"score\": <int 0-10>, \"reason\": \"<one short sentence>\"}}."
        )
        pil_img = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text], images=[pil_img],
            padding=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return self._parse(decoded)

    def _parse(self, text: str) -> dict:
        m = self.JSON_RE.search(text)
        if m:
            try:
                return {"raw": text.strip(), **json.loads(m.group(0))}
            except json.JSONDecodeError:
                pass
        # fallback: extract first integer
        m2 = re.search(r"\b(\d{1,2})\b", text)
        score = int(m2.group(1)) if m2 else -1
        return {"raw": text.strip(), "score": score, "reason": "(parse fallback)"}


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def score_run(
    run_dir: Path,
    appraiser: Optional[CLIPLearnabilityAppraiser],
    vlm: Optional[QwenVLRelevanceScorer],
    inner_loop_steps: int,
    inner_loop_lr: float,
    force: bool = False,
    image_glob: str = "auto",
    scores_name: str = "scores.json",
):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"  skip {run_dir.name}: no config.json")
        return
    with open(config_path) as f:
        config = json.load(f)
    try:
        prompt = derive_prompt(config)
    except ValueError as e:
        print(f"  skip {run_dir.name}: {e}")
        return

    # Use the inner-loop LoRA rank that this run was generated with, so the
    # re-scored delta_sim matches training-time semantics.
    if appraiser is not None:
        cfg_rank = config.get("lora_rank")
        if cfg_rank is not None and cfg_rank != appraiser.lora_rank:
            appraiser.lora_rank = int(cfg_rank)

    if image_glob == "auto":
        # Creator-Appraiser runs save clean samples as raw_*.png and annotated
        # figures as image_*.png; baseline runs save clean image_*.png only.
        image_glob = "raw_*.png" if any(run_dir.glob("raw_*.png")) else "image_*.png"
    images = sorted(run_dir.glob(image_glob))
    if not images:
        print(f"  skip {run_dir.name}: no {image_glob}")
        return

    scores_path = run_dir / scores_name
    if scores_path.exists() and not force:
        with open(scores_path) as f:
            existing = json.load(f)
    else:
        existing = {"prompt": prompt, "config": {
            "lora_rank": config.get("lora_rank"),
            "creativity_scale": config.get("creativity_scale"),
        }, "images": {}}

    existing.setdefault("images", {})
    n_done = sum(1 for k in existing["images"] if "init_sim" in existing["images"][k])
    print(f"  {run_dir.name}: prompt={prompt!r}  images={len(images)}  done={n_done}")

    for img_path in images:
        # Scores are keyed by image_NNN.png, also for raw_NNN.png samples.
        key = img_path.name.replace("raw_", "image_", 1)
        # When --force, ignore any existing record for this image.
        rec = {} if force else existing["images"].get(key, {})

        need_clip = appraiser is not None and (
            "init_sim" not in rec or "delta_sim" not in rec
        )
        need_vlm = vlm is not None and "vlm_relevance" not in rec

        if not need_clip and not need_vlm:
            continue

        if need_clip:
            try:
                img = load_image_tensor(str(img_path), appraiser.device)
                rec["init_sim"] = init_sim(appraiser, img, prompt)
                rec["delta_sim"] = delta_sim(
                    appraiser, img, prompt,
                    inner_loop_steps=inner_loop_steps,
                    inner_loop_lr=inner_loop_lr,
                )
            except Exception as e:
                rec["init_sim"] = rec.get("init_sim", -1)
                rec["delta_sim"] = rec.get("delta_sim", -1)
                rec["clip_error"] = str(e)

        if need_vlm:
            try:
                v = vlm.score(str(img_path), prompt)
                rec["vlm_relevance"] = v.get("score", -1)
                rec["vlm_reason"] = v.get("reason", "")
            except Exception as e:
                rec["vlm_relevance"] = -1
                rec["vlm_reason"] = f"VLM error: {e}"

        existing["images"][key] = rec

        # Checkpoint after each image so a crash doesn't lose progress.
        with open(scores_path, "w") as f:
            json.dump(existing, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[], help="Specific run dirs to score")
    p.add_argument("--glob", type=str, default=None, help="Glob pattern for run dirs")
    p.add_argument("--lora_rank", type=int, default=8, help="LoRA rank for the inner loop")
    p.add_argument("--inner_loop_steps", type=int, default=10)
    p.add_argument("--inner_loop_lr", type=float, default=1e-3)
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip-vlm", action="store_true",
                   help="CLIP-only pass. Run separately from --skip-clip on "
                        "small GPUs to avoid co-resident OOM.")
    p.add_argument("--skip-clip", action="store_true",
                   help="VLM-only pass (assumes a prior --skip-vlm pass wrote "
                        "init_sim and delta_sim).")
    p.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="The paper uses the 7B model in fp16. On small GPUs, try "
                        "Qwen/Qwen2.5-VL-3B-Instruct with --vlm_8bit.")
    p.add_argument("--vlm_8bit", action="store_true",
                   help="Load the VLM in int8 (needs bitsandbytes).")
    p.add_argument("--no_vlm_8bit", action="store_true",
                   help="Load the VLM in fp16 (the default; kept for compatibility).")
    p.add_argument("--force", action="store_true", help="Re-score even if entries exist")
    p.add_argument("--image_glob", type=str, default="auto",
                   help="Images to score; 'auto' uses raw_*.png when present, else image_*.png")
    p.add_argument("--scores_name", type=str, default="scores.json")
    args = p.parse_args()

    runs: list[Path] = []
    runs.extend(Path(r) for r in args.runs)
    if args.glob:
        runs.extend(Path(p) for p in glob.glob(args.glob))
    runs = [r for r in runs if r.is_dir()]
    if not runs:
        print("No run dirs given. Use --runs or --glob.", file=sys.stderr)
        sys.exit(1)

    appraiser = None
    if not args.skip_clip:
        print(f"Loading CLIP appraiser (rank={args.lora_rank}) on {args.device}...")
        appraiser = CLIPLearnabilityAppraiser(
            clip_model=args.clip_model,
            lora_rank=args.lora_rank,
            device=args.device,
        )

    vlm = None
    if not args.skip_vlm:
        print(f"Loading VLM {args.vlm_model} (8bit={args.vlm_8bit})...")
        vlm = QwenVLRelevanceScorer(
            model_id=args.vlm_model,
            device=args.device,
            load_in_8bit=args.vlm_8bit,
        )

    print(f"\nScoring {len(runs)} run dir(s)...")
    for run_dir in runs:
        score_run(
            run_dir, appraiser, vlm,
            inner_loop_steps=args.inner_loop_steps,
            inner_loop_lr=args.inner_loop_lr,
            force=args.force,
            image_glob=args.image_glob,
            scores_name=args.scores_name,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
