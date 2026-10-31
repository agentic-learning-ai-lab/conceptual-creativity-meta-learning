#!/bin/bash
# All MNIST runs behind the paper's MNIST figures (Section 4.2, Appendix A).
# Output folder names match the paths in experiments/plot_frontier.py.
# Run from the repo root; set CUDA_VISIBLE_DEVICES to pick a GPU.
set -euo pipefail
F="python -m experiments.mnist_frontier --use_meta_gradient"

# Reference stimuli, unguided samples, and guidance scales 10/50/100/500 (mu = 0.15)
$F --include_unguided --output_dir outputs/mnist_frontier

# Guidance-scale sweep at mu = 0.15 and at mu = 0.08
$F --guidance_scales 10 --output_dir outputs/mnist_frontier_s10
for g in 1 2 5 50; do
  $F --guidance_scales $g --output_dir outputs/sweep_g$g
done
for g in 2 5 50; do
  $F --guidance_scales $g --ideal_initial_loss 0.08 --output_dir outputs/sweep_g${g}_mu008
done

# Ideal-initial-loss sweep at guidance scale 10 (mu = 0.15 is outputs/mnist_frontier_s10)
for mu in 0.02 0.04 0.06 0.08 0.10 0.12 0.20; do
  tag=$(printf "%03d" $(python -c "print(round($mu*100))"))
  $F --guidance_scales 10 --ideal_initial_loss $mu --output_dir outputs/sweep_mu$tag
done

# Baselines: random latent perturbation and repulsive guidance
B="python -m experiments.baselines.mnist_baselines"
$B --method random --perturbation_scale 2 --output_dir outputs/baseline_random
for p in 5 10 20; do
  $B --method random --perturbation_scale $p --output_dir outputs/baseline_random_p$p
done
$B --method repulsive --guidance_scale 5 --output_dir outputs/baseline_repulsive
for g in 1 2 10 20 50 100; do
  $B --method repulsive --guidance_scale $g --output_dir outputs/baseline_repulsive_g$g
done

python -m experiments.plot_frontier --all
