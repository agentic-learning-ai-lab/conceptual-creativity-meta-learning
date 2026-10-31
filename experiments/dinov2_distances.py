"""DINOv2 reachability metric: embed every image with a frozen vision encoder
not used in our pipeline, then per prompt compute

    d_V(x) = min_{x' in V}  || f(x) - f(x') ||         # NN distance into vanilla cloud
    d_C(x) = min_{x' in C}  || f(x) - f(x') ||         # NN distance into CA cloud

where V is the set of vanilla SD samples for the prompt and C is the set of
CA samples. Computed on:
  - x in V    -> intra-vanilla NN baseline
  - x in C    -> the reachability metric ("how far from any vanilla sample")

Two-stage CLI:
  embed       embed PNGs in --runs dirs to <run>/dinov2.npz
  distances   group runs by prompt, compute d_V/d_C, write distances.json

Frozen DINOv2 ViT-L/14 from HuggingFace (facebook/dinov2-large). Cosine
distance on L2-normalized features.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_IMAGENET_LABELS = None
def get_imagenet_label(class_id):
    """ImageNet class id -> label. Standalone (no torch dependency)."""
    global _IMAGENET_LABELS
    if _IMAGENET_LABELS is None:
        # Try cached local copy first, then fetch from upstream.
        for cache in [Path("/tmp/imagenet_labels.json"),
                      Path.home() / ".cache" / "imagenet_labels.json"]:
            if cache.exists():
                _IMAGENET_LABELS = json.load(open(cache))
                break
        if _IMAGENET_LABELS is None:
            import urllib.request
            url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
            try:
                _IMAGENET_LABELS = json.loads(urllib.request.urlopen(url, timeout=5).read())
                cache = Path.home() / ".cache" / "imagenet_labels.json"
                cache.parent.mkdir(parents=True, exist_ok=True)
                json.dump(_IMAGENET_LABELS, open(cache, "w"))
            except Exception:
                _IMAGENET_LABELS = []
    if 0 <= class_id < len(_IMAGENET_LABELS):
        return _IMAGENET_LABELS[class_id]
    return f"class_{class_id}"


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

def load_dinov2(model_id: str = "facebook/dinov2-large", device: str = "cuda"):
    import torch
    from transformers import AutoModel, AutoImageProcessor
    print(f"Loading DINOv2 ({model_id}) on {device} ...")
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_id)
    return model, processor


def embed_dir(run_dir: Path, model, processor, device: str, batch: int = 16,
              pattern: str = "*.png", force: bool = False):
    """Embed all PNGs matching pattern in run_dir to run_dir/dinov2.npz."""
    out_path = run_dir / "dinov2.npz"
    if out_path.exists() and not force:
        return  # already done

    pngs = sorted(run_dir.glob(pattern))
    # Skip if dir has no images
    if not pngs:
        return

    import torch
    import torch.nn.functional as F
    feats = {}
    with torch.no_grad():
        for i in range(0, len(pngs), batch):
            chunk = pngs[i : i + batch]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            inputs = processor(imgs, return_tensors="pt").to(device, dtype=torch.float16)
            out = model(**inputs)
            emb = out.pooler_output.float()  # (B, D)
            emb = F.normalize(emb, dim=-1)
            emb = emb.cpu().numpy()
            for p, e in zip(chunk, emb):
                feats[p.name] = e

    names = list(feats.keys())
    arr = np.stack([feats[n] for n in names], axis=0)
    np.savez(out_path, names=np.array(names), embeddings=arr)
    print(f"  {run_dir.name}: {len(names)} embeddings -> {out_path.name}")


def cmd_embed(args):
    import glob as _g
    runs = []
    for r in args.runs or []:
        runs.append(Path(r))
    for pattern in args.glob or []:
        runs.extend(Path(p) for p in _g.glob(pattern))
    runs = [r for r in runs if r.is_dir()]
    if not runs:
        print("No run dirs given.", file=sys.stderr)
        sys.exit(1)

    model, processor = load_dinov2(args.model_id, args.device)

    # Vanilla dirs save raw images as image_NNN.png (not raw_NNN.png).
    # Creator-Appraiser dirs have both image_NNN.png (annotated) and raw_NNN.png (clean);
    # raw_*.png is what we want to score for reachability.
    for run_dir in runs:
        cfg = run_dir / "config.json"
        if not cfg.exists():
            continue
        # Route by what the dir actually contains, not by config key (which
        # has been ambiguous in older runs). Creator-Appraiser dirs always have raw_*.png
        # (clean) plus image_*.png (annotated with reward plot — not for
        # scoring); vanilla dirs only have image_*.png (clean, no plot).
        has_raw = bool(list(run_dir.glob("raw_*.png")))
        if has_raw:
            embed_dir(run_dir, model, processor, args.device,
                      batch=args.batch, pattern="raw_*.png", force=args.force)
        else:
            embed_dir(run_dir, model, processor, args.device,
                      batch=args.batch, pattern="image_*.png", force=args.force)


# --------------------------------------------------------------------------
# Prompt grouping + distances
# --------------------------------------------------------------------------

def derive_prompt(config: dict) -> str:
    """Recover the text prompt for a run dir."""
    if config.get("prompt"):
        return config["prompt"]
    if config.get("init_text"):
        return config["init_text"]
    cls = config.get("class_label")
    if cls is not None:
        return f"a photo of a {get_imagenet_label(int(cls))}"
    return None


def _normalize_prompt(p: str) -> str:
    """Normalize for grouping: lowercase, collapse whitespace, ASCII-fy curly quotes."""
    p = p.replace("\u2019", "'").replace("\u2018", "'")
    p = p.replace("\u201c", '"').replace("\u201d", '"')
    p = " ".join(p.split())
    return p.strip().lower()


def discover(globs: list[str]) -> dict[str, dict]:
    """Group dirs by prompt. Returns {prompt: {"vanilla": [Path, ...], "ca": [Path, ...]}}.

    Routes by config: CA dirs have `lora_rank` set, vanilla dirs do not.
    Works whether or not the source PNGs have been synced locally.
    """
    import glob as _g
    groups: dict[str, dict[str, list[Path]]] = {}
    seen = set()
    for g in globs:
        for d in _g.glob(g):
            d = Path(d).resolve()
            if not d.is_dir() or d in seen:
                continue
            seen.add(d)
            cfg = d / "config.json"
            if not cfg.exists():
                continue
            c = json.load(open(cfg))
            prompt = derive_prompt(c)
            if not prompt:
                continue
            prompt = _normalize_prompt(prompt)
            kind = "ca" if c.get("lora_rank") is not None else "vanilla"
            groups.setdefault(prompt, {"vanilla": [], "ca": []})[kind].append(d)
    return groups


def load_embeddings(run_dir: Path) -> tuple[list[str], np.ndarray]:
    p = run_dir / "dinov2.npz"
    if not p.exists():
        return [], np.zeros((0,))
    d = np.load(p)
    names = [str(x) for x in d["names"]]   # numpy.str_ -> str
    emb = d["embeddings"]
    if len(names) != emb.shape[0]:
        print(f"  WARN: {run_dir.name}: name/emb length mismatch "
              f"({len(names)} vs {emb.shape[0]}); skipping.")
        return [], np.zeros((0,))
    return names, emb


def load_embeddings_vlm_filtered(run_dir: Path, min_vlm_rel: int
                                 ) -> tuple[list[str], np.ndarray, int, int]:
    """Load CA embeddings filtered by VLM relevance.

    raw_NNN.png embeddings are keyed in scores.json by image_NNN.png.
    Returns (names, emb, n_total, n_kept). If scores.json is missing,
    drops the whole dir (returns empty) — at high LoRA ranks unfiltered
    samples include off-class drift, which would inflate d_V_for_C.
    """
    names, emb = load_embeddings(run_dir)
    if not names:
        return names, emb, 0, 0
    if min_vlm_rel <= 0:
        return names, emb, len(names), len(names)
    sp = run_dir / "scores.json"
    if not sp.exists():
        return [], np.zeros((0,)), len(names), 0
    img_scores = json.load(open(sp)).get("images", {})
    keep_idx = []
    for i, n in enumerate(names):
        key = n.replace("raw_", "image_", 1) if n.startswith("raw_") else n
        rel = img_scores.get(key, {}).get("vlm_relevance", -1)
        if rel >= min_vlm_rel:
            keep_idx.append(i)
    if not keep_idx:
        return [], np.zeros((0,)), len(names), 0
    keep_arr = np.array(keep_idx)
    return [names[i] for i in keep_idx], emb[keep_arr], len(names), len(keep_idx)


def cosine_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance between L2-normalized rows of A and B."""
    return 1.0 - A @ B.T


def cmd_distances(args):
    globs = args.glob or []
    if not globs:
        globs = [
            "outputs/text_embedding_creative/clipLora_R*",
            "outputs/text_embedding_creative/vanilla_full_cfg7.5_*",
        ]
    groups = discover(globs)
    print(f"Found {len(groups)} prompts:")
    for p, dirs in sorted(groups.items()):
        # Count per-dir image files for sanity
        n_v = sum(len(load_embeddings(d)[0]) for d in dirs['vanilla'])
        n_c = sum(len(load_embeddings(d)[0]) for d in dirs['ca'])
        print(f"  {p[:55]:<55s}  V_dirs={len(dirs['vanilla'])}  V_imgs={n_v:>5d}  "
              f"C_dirs={len(dirs['ca'])}  C_imgs={n_c:>5d}")

    # min_V_size guards against |V|-bias: nearest-neighbor distances shrink
    # monotonically as |V| grows, so prompts with too-small vanilla envelopes
    # would inflate d_V_for_C and corrupt the cross-prompt aggregate.
    if args.min_v_size and args.min_v_size > 0:
        print(f"\nFiltering: skip prompts with |V| < {args.min_v_size}")

    out = {}
    rng = np.random.RandomState(args.seed)
    for prompt, dirs in sorted(groups.items()):
        # Aggregate embeddings + filenames per kind
        v_names, v_dirs, v_embs = [], [], []
        for d in dirs["vanilla"]:
            n, e = load_embeddings(d)
            for nn in n:
                v_names.append(nn); v_dirs.append(str(d))
            if e.size: v_embs.append(e)
        c_names, c_dirs, c_embs = [], [], []
        n_total_ca, n_kept_ca = 0, 0
        for d in dirs["ca"]:
            n, e, t, k = load_embeddings_vlm_filtered(d, args.min_vlm_rel)
            n_total_ca += t; n_kept_ca += k
            for nn in n:
                c_names.append(nn); c_dirs.append(str(d))
            if e.size: c_embs.append(e)
        if not v_embs or not c_embs:
            print(f"  skip {prompt[:40]}: missing embeddings (V={bool(v_embs)} C={bool(c_embs)})")
            continue
        V = np.concatenate(v_embs)
        C = np.concatenate(c_embs)
        if args.min_v_size and len(V) < args.min_v_size:
            print(f"  skip {prompt[:40]}: |V|={len(V)} < {args.min_v_size}")
            continue
        # Optional: subsample V to a fixed size for cross-prompt comparability.
        if args.v_size and len(V) > args.v_size:
            idx = rng.choice(len(V), args.v_size, replace=False)
            V = V[idx]
            v_names = [v_names[i] for i in idx]
            v_dirs = [v_dirs[i] for i in idx]

        # CA -> V  (the reachability metric)
        D_cv = cosine_dist(C, V)            # (n_ca, n_v)
        d_V_for_C = D_cv.min(axis=1)
        nn_idx_for_C = D_cv.argmin(axis=1)   # for each CA sample, idx of nearest vanilla

        # V -> V (intra-vanilla baseline). Self-distance is 0; mask diagonal.
        D_vv = cosine_dist(V, V)
        np.fill_diagonal(D_vv, np.inf)
        d_V_for_V = D_vv.min(axis=1)

        # C -> C (intra-CA spread). Same mask trick for self-distance.
        if len(C) > 1:
            D_cc = cosine_dist(C, C)
            np.fill_diagonal(D_cc, np.inf)
            d_C_for_C = D_cc.min(axis=1)
        else:
            d_C_for_C = np.array([0.0])

        out[prompt] = {
            "n_vanilla": len(v_names),
            "n_ca": len(c_names),
            "n_ca_total_pre_vlm": n_total_ca,
            "n_ca_kept_post_vlm": n_kept_ca,
            "min_vlm_rel": args.min_vlm_rel,
            "d_V_for_C_mean": float(d_V_for_C.mean()),
            "d_V_for_C_std": float(d_V_for_C.std()),
            "d_V_for_C_median": float(np.median(d_V_for_C)),
            "d_V_for_V_mean": float(d_V_for_V.mean()),
            "d_V_for_V_std": float(d_V_for_V.std()),
            "d_V_for_V_median": float(np.median(d_V_for_V)),
            "d_C_for_C_mean": float(d_C_for_C.mean()),
            "d_C_for_C_std": float(d_C_for_C.std()),
            "d_C_for_C_median": float(np.median(d_C_for_C)),
            "ratio_C_over_V": float(d_V_for_C.mean() / max(d_V_for_V.mean(), 1e-12)),
            "per_ca": [
                {
                    "ca_dir": c_dirs[i], "ca_name": c_names[i],
                    "d_V": float(d_V_for_C[i]),
                    "nn_vanilla_dir": v_dirs[nn_idx_for_C[i]],
                    "nn_vanilla_name": v_names[nn_idx_for_C[i]],
                }
                for i in range(len(c_names))
            ],
        }

    out_path = Path(args.output)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")

    print("\nPer-prompt summary:")
    print(f"  {'prompt':<55s}  d_V(C)  d_V(V)  d_C(C)  ratio  kept")
    for prompt, r in out.items():
        kept = f"{r.get('n_ca_kept_post_vlm', 0)}/{r.get('n_ca_total_pre_vlm', 0)}"
        print(f"  {prompt[:55]:<55s}  "
              f"{r['d_V_for_C_mean']:.3f}  {r['d_V_for_V_mean']:.3f}  "
              f"{r['d_C_for_C_mean']:.3f}  {r['ratio_C_over_V']:.2f}x  {kept}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="cmd", required=True)

    p1 = sp.add_parser("embed")
    p1.add_argument("--runs", nargs="*", default=[])
    p1.add_argument("--glob", action="append", default=[],
                    help="Glob pattern (repeatable). Python expands the pattern; "
                         "always quote at the shell so the host doesn't pre-expand.")
    p1.add_argument("--model_id", type=str, default="facebook/dinov2-large")
    p1.add_argument("--device", type=str, default="cuda")
    p1.add_argument("--batch", type=int, default=16)
    p1.add_argument("--force", action="store_true")
    p1.set_defaults(func=cmd_embed)

    p2 = sp.add_parser("distances")
    p2.add_argument("--glob", action="append", default=[])
    p2.add_argument("--output", type=str,
                    default="outputs/dinov2_distances.json")
    p2.add_argument("--min_v_size", type=int, default=200,
                    help="Skip prompts whose vanilla envelope has < this many "
                         "samples — nearest-neighbor distances are |V|-biased "
                         "so undersized envelopes inflate d_V_for_C.")
    p2.add_argument("--v_size", type=int, default=None,
                    help="If set, subsample V to this fixed size per prompt for "
                         "cross-prompt comparability (default: use all).")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--min_vlm_rel", type=int, default=5,
                    help="Drop CA samples with vlm_relevance < this. Default 5 "
                         "removes off-class drift (especially at high LoRA "
                         "ranks). 0 disables filtering.")
    p2.set_defaults(func=cmd_distances)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
