#!/usr/bin/env python3
"""Best-of-N ranker for Creator-Appraiser natural-image runs.

Reads <run_dir>/scores.json (produced by experiments/score_samples.py),
groups images by prompt across all sweep ranks, applies a relevance gate,
and ranks survivors by delta_sim.

Default gate: init_sim > 0.20 AND vlm_relevance >= 5.
Default rank: delta_sim descending.

Usage
-----
    python scripts/bon_rank.py \
        --glob 'outputs/text_embedding_creative/clipLora_R*' \
        --topk 4 \
        --output outputs/picks_bon

    # Validate auto-picks against manual picks_manual/
    python scripts/bon_rank.py \
        --glob 'outputs/text_embedding_creative/clipLora_R*' \
        --validate outputs/picks_manual
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------------------------------
# Discovery and grouping
# --------------------------------------------------------------------------

def load_scored_runs(globs: list[str]) -> list[dict]:
    """Discover sweep dirs that have a scores.json. One record per run."""
    runs = []
    seen = set()
    for g in globs:
        for d in glob.glob(g):
            d = Path(d).resolve()
            if not d.is_dir() or d in seen:
                continue
            seen.add(d)
            sp = d / "scores.json"
            if not sp.exists():
                continue
            with open(sp) as f:
                data = json.load(f)
            runs.append({
                "run_dir": d,
                "prompt": data.get("prompt"),
                "lora_rank": (data.get("config") or {}).get("lora_rank"),
                "images": data.get("images", {}),
            })
    return runs


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:60]


def group_by_prompt(runs: list[dict]) -> dict[str, list[dict]]:
    """Pool runs that share the same text prompt (across ranks)."""
    groups = defaultdict(list)
    for r in runs:
        if r["prompt"]:
            groups[r["prompt"]].append(r)
    return dict(groups)


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

def rank_prompt(
    runs: list[dict],
    init_sim_min: float,
    vlm_min: int,
    use_vlm: bool,
    vlm_only_gate: bool = False,
) -> list[dict]:
    """Return all images for a prompt, gated and ranked."""
    items = []
    for r in runs:
        for fname, rec in r["images"].items():
            init_sim = rec.get("init_sim")
            delta_sim = rec.get("delta_sim")
            vlm = rec.get("vlm_relevance", -1)
            if init_sim is None or delta_sim is None:
                continue
            gate_clip = True if vlm_only_gate else (init_sim >= init_sim_min)
            gate_vlm = (vlm >= vlm_min) if use_vlm else True
            items.append({
                "path": str(r["run_dir"] / fname),
                "fname": fname,
                "run": r["run_dir"].name,
                "lora_rank": r["lora_rank"],
                "init_sim": init_sim,
                "delta_sim": delta_sim,
                "vlm_relevance": vlm,
                "vlm_reason": rec.get("vlm_reason", ""),
                "passed_gate": gate_clip and gate_vlm,
            })
    items.sort(
        key=lambda x: (x["passed_gate"], x["delta_sim"], x["init_sim"]),
        reverse=True,
    )
    return items


def write_picks(prompt: str, ranked: list[dict], topk: int, out_root: Path) -> Path:
    out_dir = out_root / slugify(prompt)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe previous BoN picks for this prompt (idempotent, but keep manual files
    # alone — we never write to picks_manual).
    for old in out_dir.glob("B*.png"):
        old.unlink()

    survivors = [it for it in ranked if it["passed_gate"]]
    picks = survivors[:topk]
    # Job id is the trailing "J<digits>" in the sweep dirname; encode it in
    # the copy name so it can be matched against manual picks (which have the
    # same J<id>_r<idx> structure).
    jid_re = re.compile(r"_J(\d+)$")
    for i, it in enumerate(picks):
        rank_lbl = f"R{it['lora_rank']:03d}" if it["lora_rank"] is not None else "Rxxx"
        m = jid_re.search(it["run"])
        jid = f"J{m.group(1)}" if m else "Jxxx"
        # B<rank>_R<rank>_J<id>_<srcstem>.png — encodes provenance.
        dst = out_dir / f"B{i:02d}_{rank_lbl}_{jid}_{Path(it['path']).stem}.png"
        shutil.copy2(it["path"], dst)

    # Also write a JSON record of the full ranking for inspection.
    with open(out_dir / "ranking.json", "w") as f:
        json.dump({
            "prompt": prompt,
            "n_total": len(ranked),
            "n_passed_gate": sum(1 for it in ranked if it["passed_gate"]),
            "topk_picks": [it["fname"] for it in picks],
            "ranked": ranked[: max(topk * 3, 20)],  # save top 3*K for review
        }, f, indent=2, default=str)
    return out_dir


# --------------------------------------------------------------------------
# Validation against manual picks
# --------------------------------------------------------------------------

def validate(picks_bon_root: Path, manual_root: Path, prompts: list[str]):
    """For each prompt, compare top-K BoN picks against R*.png files in
    manual_root/<class>/. Reports overlap (by source filename stem)."""
    print("\n" + "=" * 72)
    print("VALIDATION: BoN top-K vs manual picks")
    print("=" * 72)
    if not manual_root.exists():
        print(f"  manual root not found: {manual_root}")
        return

    # Manual dirs are named by class slug, BoN by slugify(prompt). Try to match.
    manual_dirs = {d.name: d for d in manual_root.iterdir() if d.is_dir()}

    overlaps = []
    for prompt in prompts:
        bon_dir = picks_bon_root / slugify(prompt)
        if not bon_dir.exists():
            continue
        bon_picks = list(bon_dir.glob("B*.png"))
        if not bon_picks:
            continue

        # Try to find matching manual dir. Manual dirs are usually a class slug
        # ("lion", "stained_glass_cathedral"); the BoN prompt is the full text
        # ("a photo of a lion"). Match if the manual slug is a substring of
        # the prompt slug, ignoring the "a_photo_of_a_" prefix.
        slug = slugify(prompt)
        slug_clean = re.sub(r"^a_photo_of_a_", "", slug)
        manual = None
        for mname, mdir in manual_dirs.items():
            if mname == slug or mname == slug_clean or mname in slug_clean or slug_clean in mname:
                manual = mdir
                break
        if manual is None:
            continue

        manual_picks = list(manual.glob("R*.png"))
        if not manual_picks:
            continue

        # Match on (J<jobid>, image-index): each manual pick filename embeds
        # J<jobid>_r<idx> referring back to the sweep dir + image_<idx>.png;
        # each BoN copy name embeds J<jobid>_image_<idx>. Reduce both to the
        # (jid, idx) tuple and compare.
        def provenance(p):
            n = p.stem
            jm = re.search(r"J(\d+)", n)
            im = re.search(r"(?:image_|_r)(\d+)", n)
            return (jm.group(1) if jm else None,
                    int(im.group(1)) if im else None)

        bon_set = {provenance(p) for p in bon_picks if all(provenance(p))}
        manual_set = {provenance(p) for p in manual_picks if all(provenance(p))}
        common = bon_set & manual_set
        overlaps.append((prompt, len(common), len(manual_set), len(bon_set)))
        print(f"  {prompt[:50]:<50s}  overlap {len(common)}/{len(manual_set)} "
              f"manual  ({len(bon_set)} BoN top-K)")

    if overlaps:
        total_common = sum(o[1] for o in overlaps)
        total_manual = sum(o[2] for o in overlaps)
        print(f"\n  Overall: {total_common}/{total_manual} "
              f"({100 * total_common / max(1, total_manual):.0f}%)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", action="append", default=[],
                   help="Glob for sweep dirs (repeatable).")
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--init_sim_min", type=float, default=0.12,
                   help="Soft floor on frozen-CLIP cosine. Default 0.12 only "
                        "rejects obvious noise; the VLM is the meaningful gate.")
    p.add_argument("--vlm_min", type=int, default=5,
                   help="Qwen2.5-VL relevance gate (0-10).")
    p.add_argument("--no_vlm_gate", action="store_true",
                   help="Skip the VLM gate (use init_sim only).")
    p.add_argument("--vlm_only_gate", action="store_true",
                   help="Skip the init_sim floor; trust the VLM alone.")
    p.add_argument("--output", type=str, default="outputs/picks_bon")
    p.add_argument("--validate", type=str, default=None,
                   help="Compare top-K to manual picks under this dir.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if not args.glob:
        args.glob = ["outputs/text_embedding_creative/clipLora_R*"]

    runs = load_scored_runs(args.glob)
    print(f"Loaded {len(runs)} scored run(s).")
    if not runs:
        sys.exit(0)

    groups = group_by_prompt(runs)
    print(f"Grouped into {len(groups)} prompt(s).")

    out_root = Path(args.output)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    use_vlm = not args.no_vlm_gate
    summary = []
    for prompt, prompt_runs in sorted(groups.items()):
        ranked = rank_prompt(
            prompt_runs,
            init_sim_min=args.init_sim_min,
            vlm_min=args.vlm_min,
            use_vlm=use_vlm,
            vlm_only_gate=args.vlm_only_gate,
        )
        n_pass = sum(1 for it in ranked if it["passed_gate"])
        n_total = len(ranked)
        summary.append((prompt, n_total, n_pass))
        print(f"  {prompt[:50]:<50s}  {n_pass:>3d}/{n_total:<4d} pass gate")
        if not args.dry_run and n_pass > 0:
            write_picks(prompt, ranked, args.topk, out_root)

    print(f"\nTotal: {sum(s[1] for s in summary)} images, "
          f"{sum(s[2] for s in summary)} pass gate.")

    if args.validate:
        validate(out_root, Path(args.validate), [s[0] for s in summary])


if __name__ == "__main__":
    main()
