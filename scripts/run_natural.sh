#!/bin/bash
# All natural-image runs behind Section 4.3 and Appendices B-C: vanilla samples,
# Creator-Appraiser at four LoRA ranks, and the diversity baselines, for the 20
# prompts in experiments/prompts/prompts_20.txt; then scoring, DINOv2 embedding and
# the analyses. Output folder names match what experiments/analysis/ reads.
# Run from the repo root. Each prompt takes several GPU-hours; split the loop
# across GPUs or cluster jobs as needed.
set -euo pipefail
O=outputs
i=0
while IFS= read -r P; do
  id=$(printf "p%02d" $i)

  # Vanilla Stable Diffusion: 1000 samples, the reference set for reachability and ProCreate
  python -m experiments.baselines.vanilla_sd --prompt "$P" --num_images 1000 \
      --num_inference_steps 100 --guidance_scale 7.5 --run_name vanilla_full_cfg7.5_$id

  # Creator-Appraiser at LoRA rank 4, 8, 16, 32
  for r in 4 8 16 32; do
    python -m experiments.text_embedding_creative --init_text "$P" --lora_rank $r \
        --init_strategy from_full_prompt --num_tokens 8 --num_inference_steps 50 \
        --guidance_scale 7.5 --creativity_scale 10.0 --inner_loop_steps 10 \
        --inner_loop_lr 1e-3 --max_grad_norm 1.0 --num_images 100
  done

  # CFG-scale baseline
  for s in 0.5 1 3 5 10; do
    python -m experiments.baselines.vanilla_sd_baseline_sweep --prompt "$P" --guidance_scale $s \
        --negative_prompt "" --num_images 100 --num_inference_steps 50 \
        --output_dir $O/baseline_cfg_sweep --run_name ${id}_cfg$s
  done

  # Negative-prompt baseline (weak and strong wording, two variants each)
  python -m experiments.baselines.vanilla_sd_baseline_sweep --prompt "$P" --guidance_scale 7.5 --num_images 100 --num_inference_steps 50 \
      --negative_prompt "typical, standard, common, photorealistic, plain" \
      --output_dir $O/baseline_negative_prompt --run_name ${id}_neg_typical
  python -m experiments.baselines.vanilla_sd_baseline_sweep --prompt "$P" --guidance_scale 7.5 --num_images 100 --num_inference_steps 50 \
      --negative_prompt "boring, generic" \
      --output_dir $O/baseline_negative_prompt --run_name ${id}_neg_boring
  python -m experiments.baselines.vanilla_sd_baseline_sweep --prompt "$P" --guidance_scale 7.5 --num_images 100 --num_inference_steps 50 \
      --negative_prompt "photograph, realistic, HD, professional, dslr, natural lighting" \
      --output_dir $O/baseline_negative_prompt_strong --run_name ${id}_neg_typical
  python -m experiments.baselines.vanilla_sd_baseline_sweep --prompt "$P" --guidance_scale 7.5 --num_images 100 --num_inference_steps 50 \
      --negative_prompt "sharp focus, centered composition, stock photo, well-composed" \
      --output_dir $O/baseline_negative_prompt_strong --run_name ${id}_neg_boring

  # ProCreate with CLIP features, propelled away from the vanilla samples
  for g in 250 500 1000; do
    python -m experiments.baselines.procreate_sd --prompt "$P" --gamma $g \
        --ref_dir $O/text_embedding_creative/vanilla_full_cfg7.5_$id \
        --output_dir $O/procreate_sd_high_gamma/gamma$g --run_name $id
  done
  i=$((i + 1))
done < experiments/prompts/prompts_20.txt

# Scoring: frozen-CLIP sim_0, inner-loop delta-sim, Qwen2.5-VL-7B relevance (clean samples)
python -m experiments.score_samples \
    --glob "$O/text_embedding_creative/clipLora_R*"
for d in baseline_cfg_sweep baseline_negative_prompt baseline_negative_prompt_strong; do
  python -m experiments.score_samples --glob "$O/$d/*"
done
python -m experiments.score_samples \
    --glob "$O/procreate_sd_high_gamma/gamma*/*"

# DINOv2 embeddings for every run
python -m experiments.dinov2_distances embed \
    --glob "$O/text_embedding_creative/*" --glob "$O/baseline_*/*" --glob "$O/procreate_sd_high_gamma/gamma*/*"

# Tables: baseline comparison, selection controls, per-prompt baselines, reachability
CCML_OUTPUTS=$O python -m experiments.analysis.baselines_dinov2 --out $O/baselines_dinov2.json
CCML_OUTPUTS=$O python -m experiments.analysis.selection_controls
python -m experiments.analysis.per_prompt_baselines_table $O/baselines_dinov2.json
python -m experiments.dinov2_distances distances --glob "$O/text_embedding_creative/*"

# Automatic best-of-N picks (Appendix B)
python scripts/bon_rank.py --glob "$O/text_embedding_creative/clipLora_R*" --topk 4 --output $O/picks_bon
