# Seeking the Unfamiliar but Memorable: Conceptual Creativity as Meta-Learning

Code for the NeurIPS 2026 paper by Mengye Ren.
[Project page](https://agenticlearning.ai/conceptual-creativity-meta-learning/) · [Paper](https://arxiv.org/abs/2605.16477)

We hypothesize that an artifact is creative for an observer if it is unfamiliar at first sight but quickly learnable through a brief adaptation. A *Creator* generates a candidate, an *Appraiser* adapts to it for a few inner-loop steps, and the Appraiser's improvement is the reward the Creator follows through a meta-gradient. The Creator is a frozen diffusion model. The Appraiser is an autoencoder on MNIST and CLIP with a low-rank text adapter on natural images.

## Setup

```bash
pip install torch torchvision          # pick the build for your CUDA version
pip install -r requirements.txt
pip install -e .
```

The MNIST Appraiser weights (LeNet encoder and decoder) are attached to the [v1.0 release](../../releases/tag/v1.0):

```bash
R=https://github.com/agentic-learning-ai-lab/conceptual-creativity-meta-learning/releases/download/v1.0
mkdir -p checkpoints/mnist
curl -L -o checkpoints/mnist/encoder.pth $R/encoder.pth
curl -L -o checkpoints/mnist/decoder.pth $R/decoder.pth
```

To retrain them instead, run `python -m experiments.mnist --config configs/mnist.yaml --pretrain_only`. The MNIST diffusion model (`dimpo/ddpm-mnist`), Stable Diffusion v1.5, CLIP ViT-L/14, DINOv2 and Qwen2.5-VL-7B download from the Hugging Face Hub on first use.

## MNIST (Section 4.2, Appendix A)

`scripts/run_mnist.sh` runs every MNIST setting in the paper (guidance-scale and μ sweeps, noise-perturbation and repulsive baselines) and then draws the figures into `figures/`. A single run looks like this:

```bash
python -m experiments.mnist_frontier --use_meta_gradient --guidance_scales 2 \
    --ideal_initial_loss 0.08 --output_dir outputs/sweep_g2_mu008
python -m experiments.plot_frontier --all    # also prints the overlap-zone rank correlations
```

The layer-ablation figure (fine-tuning individual decoder layers) needs `outputs/mnist_frontier/scores_layers.json`, which this release does not produce; `plot_frontier` skips it when the file is missing.

## Natural images (Section 4.3, Appendices B and C)

`scripts/run_natural.sh` runs everything for the 20 prompts in `experiments/prompts/prompts_20.txt`: 1000 vanilla samples per prompt, Creator-Appraiser at LoRA rank 4, 8, 16 and 32, the CFG-scale, negative-prompt and ProCreate baselines, scoring, DINOv2 embedding, and the tables. Each prompt takes several GPU-hours, so split the loop across GPUs for a full reproduction. A single Creator-Appraiser run:

```bash
python -m experiments.text_embedding_creative --init_text "a photo of a lion" --lora_rank 8 \
    --init_strategy from_full_prompt --num_tokens 8 --num_inference_steps 50 \
    --guidance_scale 7.5 --creativity_scale 10.0 --inner_loop_steps 10 \
    --inner_loop_lr 1e-3 --max_grad_norm 1.0 --num_images 100
```

Each run writes `outputs/text_embedding_creative/clipLora_R008_<prompt>_<time>/` with the clean samples (`raw_*.png`) and annotated figures with the reward curve (`image_*.png`). `experiments/score_samples.py` scores the clean samples: frozen-CLIP sim₀, inner-loop Δsim, and Qwen2.5-VL relevance. Scoring defaults to Qwen2.5-VL-7B in fp16 as in the paper; `--vlm_8bit` loads it in int8 and needs `bitsandbytes`.

## Citation

```bibtex
@inproceedings{ren2026seeking,
  title     = {Seeking the Unfamiliar but Memorable: Conceptual Creativity as Meta-Learning},
  author    = {Ren, Mengye},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT. Pretrained models used by the code keep their own licenses.
