"""Compute DINOv2 NN-distance to vanilla + Vendi score per method.

Reads the DINOv2 embeddings (.npz) and scores.json files under ROOT.

For each method (CFG=1, CFG=3, ..., CCML r=8):
  - Group dirs by prompt via config.json
  - For each prompt:
      * Load vanilla_full pool embeddings (V)
      * Load method's embeddings (C) filtered by VLM >= 5
      * Compute mean nearest-neighbor cosine distance from C to V
      * Compute Vendi score on C
  - Aggregate across prompts: mean of both metrics

Output: /tmp/baseline_dinov2_analysis.json
"""
import argparse
import os
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def load_npz(path: Path) -> tuple:
    d = np.load(path, allow_pickle=True)
    return list(d["names"]), d["embeddings"]


def cosine_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def nn_distance(C: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Nearest-neighbor cosine distance from each row of C to any row of V."""
    Cn = cosine_normalize(C)
    Vn = cosine_normalize(V)
    D = 1.0 - Cn @ Vn.T   # (|C|, |V|) cosine distance
    return D.min(axis=1)


def vendi_score(C: np.ndarray) -> float:
    """Vendi score = exp(entropy of eigenvalues of K/n)."""
    if len(C) < 2:
        return float("nan")
    Cn = cosine_normalize(C)
    K = Cn @ Cn.T
    n = len(K)
    eigs = np.linalg.eigvalsh(K / n)
    eigs = np.clip(eigs, 1e-12, None)
    return float(np.exp(-np.sum(eigs * np.log(eigs))))


def config_prompt(dir_path: Path) -> str:
    """Prompt for a cell dir. Try config.json first; fall back to scores.json
    (CCML configs use class_label instead of prompt; scores.json records the
    resolved prompt string from the scoring script)."""
    cf = dir_path / "config.json"
    if cf.exists():
        try:
            p = json.load(open(cf)).get("prompt")
            if p:
                return p
        except Exception:
            pass
    sj = dir_path / "scores.json"
    if sj.exists():
        try:
            return json.load(open(sj)).get("prompt")
        except Exception:
            pass
    return None


def load_filtered_embeddings(dir_path: Path, min_vlm: int = 5) -> np.ndarray:
    """Load dinov2.npz, filter to samples with VLM >= min_vlm via scores.json.

    dinov2.npz has 'names' like image_000.png (or raw_000.png for CCML).
    scores.json has entries keyed by image_XXX.png.
    """
    npz = dir_path / "dinov2.npz"
    scores_p = dir_path / "scores.json"
    if not npz.exists() or not scores_p.exists():
        return None
    names, embs = load_npz(npz)
    scores = json.load(open(scores_p)).get("images", {})

    keep_idx = []
    for i, name in enumerate(names):
        # dinov2.npz names might be "image_007.png" or "raw_007.png"
        key = name.replace("raw_", "image_")
        e = scores.get(key)
        if e is not None and e.get("vlm_relevance", 0) >= min_vlm:
            keep_idx.append(i)
    if not keep_idx:
        return None
    return embs[keep_idx]


def load_all(dir_path: Path) -> np.ndarray:
    """Load dinov2.npz without VLM filter (used for vanilla pool)."""
    npz = dir_path / "dinov2.npz"
    if not npz.exists():
        return None
    _, embs = load_npz(npz)
    return embs


# --- prompt → method → cell-dir grouping ---
def index_all_dirs():
    """Return {prompt: {method_key: [dir, ...]}}. Method key is a canonical label."""
    index = defaultdict(lambda: defaultdict(list))

    dataset_specs = [
        ("baseline_cfg_sweep",         lambda cell: ("CFG=" + cell.split("_cfg")[-1],)),
        ("baseline_cfg_sweep_dense",   lambda cell: ("CFG=" + cell.split("_cfg")[-1],)),
        ("baseline_cfg_ultralow",      lambda cell: ("CFG=" + cell.split("_cfg")[-1],)),
        ("baseline_negative_prompt",   lambda cell: ("Neg(weak/" + cell.rsplit("_", 1)[-1] + ")",)),
        ("baseline_negative_prompt_strong", lambda cell: ("Neg(strong/" + cell.rsplit("_", 1)[-1] + ")",)),
    ]
    for dataset, keyfn in dataset_specs:
        base = ROOT / dataset
        if not base.exists():
            continue
        for cell in sorted(base.iterdir()):
            if not cell.is_dir():
                continue
            p = config_prompt(cell)
            if p is None:
                continue
            method_key = keyfn(cell.name)[0]
            index[p][method_key].append(cell)

    # ProCreate: gamma*/prompt*
    for proc_root_name, label_prefix in [
        ("procreate_sd_high_gamma", "ProCreate γ="),
        ("procreate_dreamsim_sanity", "ProCreate-DS γ="),
    ]:
        proc_root = ROOT / proc_root_name
        if not proc_root.exists():
            continue
        for gamma_dir in sorted(proc_root.iterdir()):
            if not gamma_dir.is_dir() or not gamma_dir.name.startswith("gamma"):
                continue
            gamma = gamma_dir.name.replace("gamma", "")
            for cell in sorted(gamma_dir.iterdir()):
                if not cell.is_dir():
                    continue
                p = config_prompt(cell)
                if p is None:
                    continue
                index[p][f"{label_prefix}{gamma}"].append(cell)

    # CCML rank sweep: text_embedding_creative/clipLora_R{004,008,016,032}_*
    ccml_root = ROOT / "text_embedding_creative"
    ccml_ranks = {"R004": "CCML r=4", "R008": "CCML r=8",
                  "R016": "CCML r=16", "R032": "CCML r=32"}
    if ccml_root.exists():
        for cell in sorted(ccml_root.iterdir()):
            for tag, label in ccml_ranks.items():
                if cell.name.startswith(f"clipLora_{tag}_"):
                    p = config_prompt(cell)
                    if p is None:
                        break
                    index[p][label].append(cell)
                    break

    # Vanilla pool: text_embedding_creative/vanilla_full_cfg7.5_*
    for cell in sorted((ROOT / "text_embedding_creative").iterdir()):
        if not cell.name.startswith("vanilla_full_cfg7.5"):
            continue
        p = config_prompt(cell)
        if p is None:
            continue
        index[p]["__vanilla_pool__"].append(cell)

    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/tmp/baseline_dinov2_analysis.json")
    ap.add_argument("--min_v_size", type=int, default=100,
                    help="Skip prompts whose vanilla pool has < this many samples.")
    args = ap.parse_args()

    index = index_all_dirs()
    print(f"Indexed {len(index)} prompts.")

    # Per method: {method: {prompt: {nn_dist, vendi, n_samples, n_pass, n_total}}}
    per_method_prompt = defaultdict(dict)

    prompts_used = 0
    for prompt in PROMPTS:
        if prompt not in index:
            print(f"[skip] prompt not found: {prompt}")
            continue
        methods = index[prompt]
        vanilla_dirs = methods.pop("__vanilla_pool__", [])
        if not vanilla_dirs:
            print(f"[skip] no vanilla pool for {prompt}")
            continue

        V_parts = [load_all(d) for d in vanilla_dirs]
        V_parts = [v for v in V_parts if v is not None]
        if not V_parts:
            continue
        V = np.concatenate(V_parts, axis=0)
        if len(V) < args.min_v_size:
            print(f"[skip] vanilla pool too small for {prompt}: {len(V)}")
            continue

        print(f"  {prompt[:50]:<50}  |V|={len(V)}  methods={len(methods)}")
        prompts_used += 1

        for method, dirs in methods.items():
            # Combine multiple dirs for this method+prompt (e.g., multiple runs)
            C_parts = []
            gated_sim0 = []  # frozen-CLIP sim_0 on samples that pass Qwen>=5
            n_total = 0
            n_pass = 0
            for d in dirs:
                sj = d / "scores.json"
                if sj.exists():
                    scores = json.load(open(sj)).get("images", {})
                    n_total += len(scores)
                    for e in scores.values():
                        if e.get("vlm_relevance", 0) >= 5:
                            n_pass += 1
                            # scores.json stores sim_0 as 'init_sim'
                            s0 = e.get("init_sim", e.get("sim_0", e.get("clip_sim")))
                            if s0 is not None:
                                gated_sim0.append(float(s0))
                C = load_filtered_embeddings(d, min_vlm=5)
                if C is not None:
                    C_parts.append(C)
            if not C_parts or n_total == 0:
                continue
            C = np.concatenate(C_parts, axis=0)
            per_method_prompt[method][prompt] = {
                "n_samples": len(C),
                "n_total": n_total,
                "n_pass": n_pass,
                "pass_rate": n_pass / n_total if n_total else float("nan"),
                "nn_dist_mean": float(nn_distance(C, V).mean()),
                "sim0_mean": float(np.mean(gated_sim0)) if gated_sim0 else float("nan"),
                "vendi": vendi_score(C),
            }

    print(f"\n=== Aggregate per method (across {prompts_used} prompts) ===")
    print(f"{'method':<24} {'n':>4} {'pass':>6} {'NN mean':>9} {'NN med':>9} {'sim0':>7} {'Vendi':>7}")
    out = {}
    for method, per_prompt in sorted(per_method_prompt.items()):
        n = len(per_prompt)
        pass_rate = np.mean([v["pass_rate"] for v in per_prompt.values()])
        nn_vals = [v["nn_dist_mean"] for v in per_prompt.values()]
        nn_mean = np.mean(nn_vals)
        nn_median = np.median(nn_vals)
        sim0_vals = [v["sim0_mean"] for v in per_prompt.values() if v["sim0_mean"] == v["sim0_mean"]]
        sim0_median = np.median(sim0_vals) if sim0_vals else float("nan")
        vendi = np.mean([v["vendi"] for v in per_prompt.values() if v["vendi"] == v["vendi"]])
        print(f"  {method:<22} {n:>4d} {pass_rate:>6.3f} {nn_mean:>9.4f} {nn_median:>9.4f} {sim0_median:>7.3f} {vendi:>7.2f}")
        out[method] = {
            "n_prompts": n,
            "mean_pass_rate": float(pass_rate),
            "mean_nn_dist_gated": float(nn_mean),
            "median_nn_dist_gated": float(nn_median),
            "median_sim0_gated": float(sim0_median),
            "mean_vendi_gated": float(vendi),
            "per_prompt": per_prompt,
        }

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
