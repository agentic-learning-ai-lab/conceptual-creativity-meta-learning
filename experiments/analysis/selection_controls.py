"""Selection controls for the DINOv2 reachability analysis (Appendix C).

Three tests:
  1. Weirdest-vanilla test: for each prompt, max intra-vanilla NN distance
     (i.e. the distance of the most-outlier vanilla sample to its nearest
     other vanilla sample). Compare against Creator-Appraiser NN distances.
  2. Unconditional NN distance: for Creator-Appraiser rank sweep, NN distance
     computed over ALL samples (no Qwen filter). Shows the trend is not
     driven by gate-induced subpopulation selection.
  3. Matched-acceptance NN distance: take top-K by Qwen score for each method
     at a matched acceptance rate (K = 13 per prompt, matching r=32's 13%);
     compute NN distance on that fixed-size subset.
"""
import json
import os
import numpy as np
from collections import defaultdict
from pathlib import Path

# Root of the run directories; override with CCML_OUTPUTS.
ROOT = Path(os.environ.get("CCML_OUTPUTS", "outputs"))

PROMPTS = [
    "a photo of a lion", "a photo of a golden retriever", "a photo of a cheeseburger",
    "a photo of a daisy", "a photo of a volcano", "a photo of a jellyfish",
    "a photo of a peacock", "a photo of a balloon", "a photo of a brown bear",
    "a photo of a castle", "a photo of a espresso", "a photo of a flamingo",
    "a photo of a lighthouse",
    "stained-glass cathedral interior at night", "neon samurai in rain",
    "ancient library lit by fireflies", "an alien botanical garden",
    "a fortune teller's tent at twilight",
    "macro photograph of bioluminescent jellyfish in the deep ocean",
    "translucent koi swimming in clouds",
]


def load_npz(p):
    d = np.load(p, allow_pickle=True)
    return list(d["names"]), d["embeddings"]


def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def nn_to_pool(C, V):
    """For each row of C, min cosine distance to any row of V. Returns (|C|,)."""
    D = 1.0 - norm(C) @ norm(V).T
    return D.min(axis=1)


def intra_max(V):
    """For each row of V, its nearest-neighbor distance to V \\ {self}. Then max."""
    Vn = norm(V)
    D = 1.0 - Vn @ Vn.T
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)  # per-row NN dist, size (|V|,)


def config_prompt(dp):
    for name in ("config.json", "scores.json"):
        p = dp / name
        if p.exists():
            try:
                d = json.load(open(p))
                pr = d.get("prompt")
                if pr:
                    return pr
            except Exception:
                pass
    return None


def vanilla_dirs_for(prompt, manifest_lookup):
    """Return list of vanilla_full_cfg7.5_* dirs for a prompt."""
    tec = ROOT / "text_embedding_creative"
    out = []
    for d in sorted(tec.iterdir()):
        if not d.name.startswith("vanilla_full_cfg7.5"):
            continue
        p = config_prompt(d)
        if p == prompt:
            out.append(d)
    return out


def ccml_dirs_for(prompt, rank_tag):
    """rank_tag in {R004, R008, R016, R032}."""
    tec = ROOT / "text_embedding_creative"
    out = []
    for d in sorted(tec.iterdir()):
        if not d.name.startswith(f"clipLora_{rank_tag}_"):
            continue
        p = config_prompt(d)
        if p == prompt:
            out.append(d)
    return out


def load_embs_and_scores(cell):
    """Return (embeddings, vlm_scores) — arrays aligned by image_XXX.png key.
    Falls back to raw_XXX.png-style names by mapping to image_XXX.png."""
    npz = cell / "dinov2.npz"
    sj = cell / "scores.json"
    if not npz.exists():
        return None, None
    names, embs = load_npz(npz)
    scores = {}
    if sj.exists():
        try:
            imgs = json.load(open(sj)).get("images", {})
            for k, v in imgs.items():
                scores[k] = v.get("vlm_relevance", 0)
        except Exception:
            pass
    vlm = np.zeros(len(names))
    for i, n in enumerate(names):
        key = n.replace("raw_", "image_")
        vlm[i] = scores.get(key, 0)
    return embs, vlm


def main():
    print("=== TEST 1: Weirdest-vanilla vs Creator-Appraiser ===")
    per_prompt_intra_max = []
    per_prompt_intra_p95 = []
    per_prompt_intra_p90 = []
    per_prompt_intra_median = []
    per_prompt_ca = defaultdict(list)  # rank -> [per-prompt medians]

    for prompt in PROMPTS:
        vdirs = vanilla_dirs_for(prompt, None)
        V_parts = []
        for d in vdirs:
            e, _ = load_embs_and_scores(d)
            if e is not None:
                V_parts.append(e)
        if not V_parts:
            continue
        V = np.concatenate(V_parts, 0)
        if len(V) < 100:
            continue

        # Intra-V per-sample NN distance
        d_intra = intra_max(V)   # (|V|,)
        v_max = float(d_intra.max())
        v_p95 = float(np.percentile(d_intra, 95))
        v_p90 = float(np.percentile(d_intra, 90))
        v_med = float(np.mean(d_intra))
        per_prompt_intra_max.append(v_max)
        per_prompt_intra_p95.append(v_p95)
        per_prompt_intra_p90.append(v_p90)
        per_prompt_intra_median.append(v_med)

        # CCML across ranks
        for rank_tag, label in [("R004", "r=4"), ("R008", "r=8"),
                                ("R016", "r=16"), ("R032", "r=32")]:
            ce_parts = []
            for d in ccml_dirs_for(prompt, rank_tag):
                e, s = load_embs_and_scores(d)
                if e is not None:
                    # Gated subset
                    keep = s >= 5
                    if keep.any():
                        ce_parts.append(e[keep])
            if not ce_parts:
                continue
            C = np.concatenate(ce_parts, 0)
            d_ca = nn_to_pool(C, V)
            per_prompt_ca[label].append(float(np.mean(d_ca)))

    print(f"Across {len(per_prompt_intra_max)} prompts:")
    print(f"  Weirdest vanilla (max intra-V NN dist), median across prompts: "
          f"{np.median(per_prompt_intra_max):.4f}")
    print(f"  Vanilla top-5%   (95th %ile intra-V), median across prompts:   "
          f"{np.median(per_prompt_intra_p95):.4f}")
    print(f"  Vanilla top-10%  (90th %ile intra-V), median across prompts:   "
          f"{np.median(per_prompt_intra_p90):.4f}")
    print(f"  Vanilla typical  (mean intra-V),      median across prompts:     "
          f"{np.median(per_prompt_intra_median):.4f}")
    for label in ("r=4", "r=8", "r=16", "r=32"):
        if per_prompt_ca[label]:
            m = float(np.median(per_prompt_ca[label]))
            print(f"  Creator-Appraiser {label} (gated, mean NN), median across prompts: {m:.4f}")

    print()
    print("=== TEST 2: Unconditional CCML NN distance (no Qwen gate) ===")
    unc_ca = defaultdict(list)
    for prompt in PROMPTS:
        vdirs = vanilla_dirs_for(prompt, None)
        V_parts = []
        for d in vdirs:
            e, _ = load_embs_and_scores(d)
            if e is not None:
                V_parts.append(e)
        if not V_parts:
            continue
        V = np.concatenate(V_parts, 0)

        for rank_tag, label in [("R004", "r=4"), ("R008", "r=8"),
                                ("R016", "r=16"), ("R032", "r=32")]:
            ce_parts = []
            for d in ccml_dirs_for(prompt, rank_tag):
                e, _ = load_embs_and_scores(d)
                if e is not None:
                    ce_parts.append(e)  # NO gating
            if not ce_parts:
                continue
            C = np.concatenate(ce_parts, 0)
            d_ca = nn_to_pool(C, V)
            unc_ca[label].append(float(np.mean(d_ca)))

    for label in ("r=4", "r=8", "r=16", "r=32"):
        if unc_ca[label]:
            m = float(np.median(unc_ca[label]))
            print(f"  Creator-Appraiser {label} (UNGATED, mean NN), median across prompts: {m:.4f}")

    print()
    print("=== TEST 3: Matched-acceptance CCML NN distance (top-13 by Qwen per prompt) ===")
    K = 13
    matched = defaultdict(list)
    for prompt in PROMPTS:
        vdirs = vanilla_dirs_for(prompt, None)
        V_parts = []
        for d in vdirs:
            e, _ = load_embs_and_scores(d)
            if e is not None:
                V_parts.append(e)
        if not V_parts:
            continue
        V = np.concatenate(V_parts, 0)

        for rank_tag, label in [("R004", "r=4"), ("R008", "r=8"),
                                ("R016", "r=16"), ("R032", "r=32")]:
            ce_parts = []
            sc_parts = []
            for d in ccml_dirs_for(prompt, rank_tag):
                e, s = load_embs_and_scores(d)
                if e is not None:
                    ce_parts.append(e)
                    sc_parts.append(s)
            if not ce_parts:
                continue
            C = np.concatenate(ce_parts, 0)
            S = np.concatenate(sc_parts, 0)
            if len(C) < K:
                continue
            idx = np.argsort(-S)[:K]
            C_top = C[idx]
            d_ca = nn_to_pool(C_top, V)
            matched[label].append(float(np.mean(d_ca)))

    for label in ("r=4", "r=8", "r=16", "r=32"):
        if matched[label]:
            m = float(np.median(matched[label]))
            print(f"  Creator-Appraiser {label} (top-{K} by Qwen, mean NN), median across prompts: {m:.4f}")


if __name__ == "__main__":
    main()
