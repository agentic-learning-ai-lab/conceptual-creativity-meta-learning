#!/usr/bin/env python3
"""Generate all paper-ready figures from precomputed scores and images.

Usage:
    python -m experiments.plot_frontier --all           # all figures
    python -m experiments.plot_frontier                  # main figures only
    python -m experiments.plot_frontier --baselines      # baseline comparison
    python -m experiments.plot_frontier --analysis       # scale/mu analysis
    python -m experiments.plot_frontier --grids          # all image grids
"""

import argparse
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------------
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
}

# ---------------------------------------------------------------------------
# Unified notation (matches paper)
#   L_0  = unfamiliarity
#   L_T  = residual unfamiliarity
#   eta  = guidance scale (CCML and repulsive)
#   sigma_p = perturbation scale (random baseline)
#   mu   = ideal initial loss target
# ---------------------------------------------------------------------------
XLABEL_L0 = r"Unfamiliarity $L_0$"
YLABEL_LT = r"Residual Unfamiliarity $L_T$"

# Category visual identity for reference stimuli
CATEGORIES = {
    "MNIST":              {"color": "#4CAF50", "marker": "o",  "label": "MNIST",     "ms": 30},
    "Greek":              {"color": "#2196F3", "marker": "s",  "label": "Greek",     "ms": 30},
    "Korean":             {"color": "#9C27B0", "marker": "D",  "label": "Korean",    "ms": 30},
    "Blend (noise=30%)":  {"color": "#FFB74D", "marker": "v",  "label": "Blend 30%", "ms": 35},
    "Blend (noise=50%)":  {"color": "#FF7043", "marker": "v",  "label": "Blend 50%", "ms": 35},
    "Blend (noise=70%)":  {"color": "#E53935", "marker": "v",  "label": "Blend 70%", "ms": 35},
    "Noise":              {"color": "#B71C1C", "marker": "v",  "label": "Noise",     "ms": 35},
    "Vanilla Diffusion":  {"color": "#607D8B", "marker": "X",  "label": "Vanilla Diffusion",  "ms": 40},
}

CCML_COLORS = ["#008080", "#E8A317", "#1A5276"]

# ---------------------------------------------------------------------------
# Output directories — edit these if your layout changes
# ---------------------------------------------------------------------------
BASELINE_RANDOM_PATHS = {
    2:  "outputs/baseline_random/scores.json",
    5:  "outputs/baseline_random_p5/scores.json",
    10: "outputs/baseline_random_p10/scores.json",
    20: "outputs/baseline_random_p20/scores.json",
}
BASELINE_REPULSIVE_PATHS = {
    1:   "outputs/baseline_repulsive_g1/scores.json",
    2:   "outputs/baseline_repulsive_g2/scores.json",
    5:   "outputs/baseline_repulsive/scores.json",
    10:  "outputs/baseline_repulsive_g10/scores.json",
    20:  "outputs/baseline_repulsive_g20/scores.json",
    50:  "outputs/baseline_repulsive_g50/scores.json",
    100: "outputs/baseline_repulsive_g100/scores.json",
}
CCML_PATHS = {
    1:  "outputs/sweep_g1/scores.json",
    2:  "outputs/sweep_g2_mu008/scores.json",
    5:  "outputs/sweep_g5_mu008/scores.json",
    10: "outputs/mnist_frontier_s10/scores.json",
    50: "outputs/sweep_g50/scores.json",
}
SCALE_DIRS = {
    1:  "outputs/sweep_g1",
    2:  "outputs/sweep_g2",
    5:  "outputs/sweep_g5",
    10: "outputs/mnist_frontier_s10",
    50: "outputs/sweep_g50",
}
MU_VALUES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]

def _mu_dir(m):
    if m == 0.15:
        return "outputs/mnist_frontier_s10"
    return f"outputs/sweep_mu{int(m * 100):03d}"


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------
def improvement(L0, LT):
    return L0 - LT


def complexity_gate(L0, mu=0.15, sigma=0.2):
    return np.exp(-((L0 - mu) ** 2) / (2 * sigma ** 2))


def success_gate(LT, temp=0.1):
    return np.exp(-LT / temp)


def shaped_reward(L0, LT, mu=0.15, sigma=0.2, temp=0.1):
    return improvement(L0, LT) * complexity_gate(L0, mu, sigma) * success_gate(LT, temp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract(results):
    """Extract L0 and LT arrays from a list of score dicts."""
    L0 = np.array([r["L_0"] for r in results])
    LT = np.array([r["L_T"] for r in results])
    return L0, LT


def _scatter_category(ax, L0, y, cat_name, ccml_idx=None, add_label=True,
                      ccml_label_override=None):
    """Plot one category on an axis."""
    if cat_name in CATEGORIES:
        s = CATEGORIES[cat_name]
        ax.scatter(L0, y, c=s["color"], marker=s["marker"], s=s["ms"],
                   alpha=0.65, label=s["label"] if add_label else None,
                   edgecolors="white", linewidths=0.3, zorder=3)
    elif cat_name.startswith("CCML") or cat_name.startswith("CAMEL") or cat_name.startswith("Ours"):
        color = CCML_COLORS[(ccml_idx or 0) % len(CCML_COLORS)]
        lbl = ccml_label_override if ccml_label_override else cat_name
        ax.scatter(L0, y, c=color, marker="*", s=100, alpha=0.85,
                   label=lbl if add_label else None,
                   edgecolors="black", linewidths=0.3, zorder=4)


def _shared_legend(fig, axes, ncol=7, **kwargs):
    """Add a single shared legend below the figure."""
    if hasattr(axes, 'flat'):
        handles, labels = axes.flat[0].get_legend_handles_labels()
    else:
        handles, labels = axes.get_legend_handles_labels()
    kw = dict(framealpha=0.9, edgecolor="0.8", handletextpad=0.3,
              borderpad=0.3, columnspacing=1.0, fontsize=10)
    kw.update(kwargs)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, **kw)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _get_first_ccml(path):
    """Load scores and return the first CCML category's (L0, LT)."""
    data = _load_json(path)
    for k, v in data.items():
        if k.startswith("CCML"):
            return _extract(v)
    return None, None


def _get_first_baseline(path):
    """Load scores and return the first baseline category's (L0, LT)."""
    data = _load_json(path)
    for k, v in data.items():
        if not k.startswith("CCML") and k not in CATEGORIES:
            return _extract(v)
    return None, None


def _find_image_dir(base_dir):
    """Find the first image subdirectory under base_dir/images/."""
    img_base = os.path.join(base_dir, "images")
    if os.path.isdir(img_base):
        for sub in sorted(os.listdir(img_base)):
            candidate = os.path.join(img_base, sub)
            if os.path.isdir(candidate):
                return candidate
    return None


def _mean_pixel_intensity(img_dir, n=10):
    """Compute mean pixel intensity from first n PNGs in a directory."""
    from PIL import Image
    pngs = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:n]
    if not pngs:
        return 0.0, 0.0
    vals = [np.mean(np.array(Image.open(p).convert("L"))) / 255.0 for p in pngs]
    return np.mean(vals), np.std(vals)


# ---------------------------------------------------------------------------
# Figure: Main frontier (two panels)
# ---------------------------------------------------------------------------
def plot_main_frontier(scores, output_path, subsample_ref=15, seed=42):
    rng = np.random.RandomState(seed)
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

        ccml_guided, ccml_unguided = [], []
        for k in scores:
            if not k.startswith("CCML"):
                continue
            if "unguided" in k:
                ccml_unguided.extend(scores[k])
            elif k != "CCML (scale=500)":
                ccml_guided.extend(scores[k])

        merged = {k: v for k, v in scores.items() if not k.startswith("CCML")}
        if ccml_unguided:
            merged["Vanilla Diffusion"] = ccml_unguided
        if ccml_guided:
            merged["Ours"] = ccml_guided

        ccml_idx = 0
        for cat_name, results in merged.items():
            max_pts = subsample_ref if not cat_name.startswith("Ours") else 10
            if len(results) > max_pts:
                idx = rng.choice(len(results), max_pts, replace=False)
            else:
                idx = np.arange(len(results))
            L0_full, LT_full = _extract(results)
            L0, LT = L0_full[idx], LT_full[idx]
            ci = ccml_idx if cat_name.startswith("Ours") else None
            if cat_name.startswith("Ours"):
                ccml_idx += 1
            _scatter_category(ax1, L0, LT, cat_name, ci, add_label=True)
            _scatter_category(ax2, L0, shaped_reward(L0, LT), cat_name, ci,
                              add_label=False)

        ax1.set_xlabel(XLABEL_L0)
        ax1.set_ylabel(YLABEL_LT)
        ax1.set_title("(a) Unfamiliarity vs. Learnability")
        ax2.set_xlabel(XLABEL_L0)
        ax2.set_ylabel("Shaped Reward")
        ax2.set_title("(b) Learnability Reward")

        fig.tight_layout(rect=[0, 0.13, 1, 1], w_pad=2.5)
        _shared_legend(fig, ax1, ncol=5)
        plt.savefig(output_path)
        plt.close()
    print(f"Saved main frontier → {output_path}")


# ---------------------------------------------------------------------------
# Figure: Reward component ablation
# ---------------------------------------------------------------------------
def _rank(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r


def print_overlap_correlation(scores, lo=0.05, hi=0.15):
    """Spearman rho of gate and of raw improvement against the shaped reward,
    over samples with L0 in [lo, hi] (the overlap zone in the paper's
    "Why the improvement term matters" paragraph)."""
    pools = {
        "stimuli only": {k: v for k, v in scores.items() if not k.startswith("CCML")},
        "stimuli + ours": {k: v for k, v in scores.items() if "500" not in k},
    }
    for name, pool in pools.items():
        L0 = np.concatenate([_extract(v)[0] for v in pool.values()])
        LT = np.concatenate([_extract(v)[1] for v in pool.values()])
        m = (L0 >= lo) & (L0 <= hi)
        R = _rank(shaped_reward(L0[m], LT[m]))
        rho = lambda v: np.corrcoef(_rank(v), R)[0, 1]
        print(f"Overlap zone L0 in [{lo}, {hi}], {name} (n={m.sum()}): "
              f"rho(gate, R) = {rho(complexity_gate(L0[m])):.2f}, "
              f"rho(L0-LT, R) = {rho(improvement(L0[m], LT[m])):.2f}")


def plot_reward_ablation(scores, output_path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))

        titles = [
            "(a) Gated Unfamiliarity\n" + r"$\exp(-(L_0-\mu)^2\,/\,2\sigma^2)$",
            "(b) + Improvement\n" + r"$(L_0-L_T) \times$ gate",
            "(c) Full Shaped Reward\n" + r"imp $\times$ gate $\times$ success",
            "(d) Relative Improvement\n" + r"$(L_0-L_T)\,/\,L_0$",
        ]
        ylabels = ["Score", None, None, None]

        # Merge all CCML into one group (excluding scale=500 outlier)
        ccml_merged = []
        for cat_name, results in scores.items():
            if cat_name.startswith("CCML") and "500" not in cat_name:
                ccml_merged.extend(results)

        plot_scores = {k: v for k, v in scores.items() if not k.startswith("CCML")}
        if ccml_merged:
            plot_scores["Ours"] = ccml_merged

        for cat_name, results in plot_scores.items():
            L0, LT = _extract(results)
            vals = [
                complexity_gate(L0),
                improvement(L0, LT) * complexity_gate(L0),
                shaped_reward(L0, LT),
                improvement(L0, LT) / (L0 + 1e-6),
            ]
            for i, (ax, v) in enumerate(zip(axes, vals)):
                _scatter_category(ax, L0, v, cat_name, add_label=(i == 0))

        for i, (ax, title) in enumerate(zip(axes, titles)):
            ax.set_xlabel(XLABEL_L0)
            ax.set_title(title, fontsize=11)
            if ylabels[i]:
                ax.set_ylabel(ylabels[i])

        fig.tight_layout(rect=[0, 0.10, 1, 1], w_pad=1.2)
        _shared_legend(fig, axes, ncol=7)
        plt.savefig(output_path)
        plt.close()
    print(f"Saved reward ablation → {output_path}")


# ---------------------------------------------------------------------------
# Figure: Layer ablation (appendix)
# ---------------------------------------------------------------------------
def plot_layer_ablation(layer_scores, output_path):
    ft_layers = ["decoder_first", "decoder_last", "decoder"]
    ft_labels = ["Decoder First Layer", "Decoder Last Layer", "Full Decoder"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.8))

        for col, (ft, ft_label) in enumerate(zip(ft_layers, ft_labels)):
            if ft not in layer_scores:
                continue
            scores = layer_scores[ft]
            for cat_name, results in scores.items():
                L0, LT = _extract(results)
                _scatter_category(axes[0, col], L0, improvement(L0, LT),
                                  cat_name, add_label=(col == 0))
                _scatter_category(axes[1, col], L0, shaped_reward(L0, LT),
                                  cat_name, add_label=False)
            axes[0, col].set_title(ft_label, fontweight="bold")
            for row in range(2):
                axes[row, col].set_xlabel(XLABEL_L0)

        axes[0, 0].set_ylabel(r"Raw Improvement $L_0 - L_T$")
        axes[1, 0].set_ylabel("Shaped Reward")

        fig.tight_layout(rect=[0, 0.07, 1, 1], h_pad=1.5, w_pad=1.5)
        _shared_legend(fig, axes, ncol=7)
        plt.savefig(output_path)
        plt.close()
    print(f"Saved layer ablation → {output_path}")


# ---------------------------------------------------------------------------
# Figure: Sweep comparison
# ---------------------------------------------------------------------------
def plot_sweep(sweep_dirs, output_path, subsample_ref=15, seed=42):
    rng = np.random.RandomState(seed)
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

        ref_scores = {}
        ccml_groups = []
        for label, path in sweep_dirs:
            data = _load_json(path)
            for cat, results in data.items():
                if cat.startswith("CCML"):
                    ccml_groups.append((label, results))
                elif cat not in ref_scores:
                    ref_scores[cat] = results

        for cat_name, results in ref_scores.items():
            if len(results) > subsample_ref:
                idx = rng.choice(len(results), subsample_ref, replace=False)
            else:
                idx = np.arange(len(results))
            L0, LT = _extract(results)
            L0, LT = L0[idx], LT[idx]
            _scatter_category(ax1, L0, LT, cat_name, add_label=True)
            _scatter_category(ax2, L0, shaped_reward(L0, LT), cat_name,
                              add_label=False)

        sweep_palette = [
            "#008080", "#E8A317", "#D63384", "#2E86C1", "#28B463",
            "#8E44AD", "#E74C3C", "#F39C12", "#1ABC9C", "#6C3483",
        ]
        for i, (label, results) in enumerate(ccml_groups):
            L0, LT = _extract(results)
            color = sweep_palette[i % len(sweep_palette)]
            ax1.scatter(L0, LT, c=color, marker="*", s=100, alpha=0.85,
                        label=label, edgecolors="black", linewidths=0.3, zorder=4)
            ax2.scatter(L0, shaped_reward(L0, LT), c=color, marker="*", s=100,
                        alpha=0.85, edgecolors="black", linewidths=0.3, zorder=4)

        ax1.set_xlabel(XLABEL_L0)
        ax1.set_ylabel(YLABEL_LT)
        ax1.set_title("(a) Unfamiliarity vs. Learnability")
        ax2.set_xlabel(XLABEL_L0)
        ax2.set_ylabel("Shaped Reward")
        ax2.set_title("(b) Learnability Reward")

        fig.tight_layout(rect=[0, 0.18, 1, 1], w_pad=2.5)
        _shared_legend(fig, ax1, ncol=4)
        plt.savefig(output_path)
        plt.close()
    print(f"Saved sweep figure → {output_path}")


# ---------------------------------------------------------------------------
# Figure: Baseline comparison scatter (color gradient per eta)
# ---------------------------------------------------------------------------
def plot_baseline_comparison(output_path, etas=None):
    if etas is None:
        etas = [1, 2, 5, 10]
    eta_norm = {e: 0.35 + 0.55 * i / (len(etas) - 1) for i, e in enumerate(etas)}

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.plot([0, 0.35], [0, 0.35], 'k--', alpha=0.3,
                label="$L_T = L_0$ (no learning)")

        # Plot grouped by method so legend is sorted by type
        for eta in etas:
            path = BASELINE_RANDOM_PATHS.get(eta)
            if path and os.path.exists(path):
                L0, LT = _get_first_baseline(path)
                if L0 is not None:
                    ax.scatter(L0, LT, c=[cm.Greens(eta_norm[eta])], marker="^",
                               s=45, alpha=0.8,
                               label=f"Random ($\\sigma_p={eta}$)",
                               edgecolors="white", linewidths=0.3, zorder=3)

        for eta in etas:
            path = BASELINE_REPULSIVE_PATHS.get(eta)
            if path and os.path.exists(path):
                L0, LT = _get_first_baseline(path)
                if L0 is not None:
                    ax.scatter(L0, LT, c=[cm.Reds(eta_norm[eta])], marker="v",
                               s=45, alpha=0.8,
                               label=f"Repulsive ($\\eta={eta}$)",
                               edgecolors="white", linewidths=0.3, zorder=3)

        for eta in etas:
            path = CCML_PATHS.get(eta)
            if path and os.path.exists(path):
                L0, LT = _get_first_ccml(path)
                if L0 is not None:
                    ax.scatter(L0, LT, c=[cm.Blues(eta_norm[eta])], marker="*",
                               s=120, alpha=0.85,
                               label=f"Ours ($\\eta={eta}$)",
                               edgecolors="black", linewidths=0.3, zorder=4)

        ax.set_xlabel(XLABEL_L0)
        ax.set_ylabel(YLABEL_LT)
        ax.set_xlim(-0.02, 0.35)
        ax.set_ylim(-0.02, 0.35)
        ax.legend(fontsize=6.5, loc="upper left", ncol=1)
        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()
    print(f"Saved baseline comparison → {output_path}")


# ---------------------------------------------------------------------------
# Figure: Scale analysis (L0, LT, thickness vs eta)
# ---------------------------------------------------------------------------
def plot_scale_analysis(output_path, scales=None):
    if scales is None:
        scales = [1, 2, 5, 10, 50]

    L0_means, L0_stds = [], []
    LT_means, LT_stds = [], []
    valid_scales = []

    for s in scales:
        base = SCALE_DIRS.get(s)
        if not base:
            continue
        sp = os.path.join(base, "scores.json")
        if not os.path.exists(sp):
            continue
        data = _load_json(sp)
        for k, v in data.items():
            if k.startswith("CCML"):
                L0, LT = _extract(v)
                L0_means.append(np.mean(L0)); L0_stds.append(np.std(L0))
                LT_means.append(np.mean(LT)); LT_stds.append(np.std(LT))
                valid_scales.append(s)
                break

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        x = np.arange(len(valid_scales))
        xlabels = [str(s) for s in valid_scales]

        axes[0].errorbar(x, L0_means, yerr=L0_stds, fmt='o-', color="#1E88E5",
                         capsize=3)
        axes[0].set_xticks(x); axes[0].set_xticklabels(xlabels)
        axes[0].set_xlabel(r"Guidance Scale $\eta$")
        axes[0].set_ylabel(XLABEL_L0)
        axes[0].set_title(r"(a) Unfamiliarity vs. $\eta$")

        axes[1].errorbar(x, LT_means, yerr=LT_stds, fmt='o-', color="#E53935",
                         capsize=3)
        axes[1].set_xticks(x); axes[1].set_xticklabels(xlabels)
        axes[1].set_xlabel(r"Guidance Scale $\eta$")
        axes[1].set_ylabel(YLABEL_LT)
        axes[1].set_title(r"(b) Residual Unfamiliarity vs. $\eta$")

        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()
    print(f"Saved scale analysis → {output_path}")


# ---------------------------------------------------------------------------
# Figure: Mu analysis (L0, LT, thickness vs mu)
# ---------------------------------------------------------------------------
def plot_mu_analysis(output_path, mus=None):
    if mus is None:
        mus = MU_VALUES

    L0m, L0s, LTm, LTs = [], [], [], []
    valid_mus = []

    for m in mus:
        base = _mu_dir(m)
        sp = os.path.join(base, "scores.json")
        if not os.path.exists(sp):
            continue
        data = _load_json(sp)
        for k, v in data.items():
            if k.startswith("CCML"):
                L0, LT = _extract(v)
                L0m.append(np.mean(L0)); L0s.append(np.std(L0))
                LTm.append(np.mean(LT)); LTs.append(np.std(LT))
                valid_mus.append(m)
                break

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        x = np.arange(len(valid_mus))
        xlabels = [f"{m:.2f}" for m in valid_mus]

        axes[0].errorbar(x, L0m, yerr=L0s, fmt='o-', color="#1E88E5", capsize=3)
        axes[0].set_xticks(x); axes[0].set_xticklabels(xlabels, rotation=45)
        axes[0].set_xlabel(r"Ideal Initial Loss $\mu$")
        axes[0].set_ylabel(XLABEL_L0)
        axes[0].set_title(r"(a) Unfamiliarity vs. $\mu$")

        axes[1].errorbar(x, LTm, yerr=LTs, fmt='o-', color="#E53935", capsize=3)
        axes[1].set_xticks(x); axes[1].set_xticklabels(xlabels, rotation=45)
        axes[1].set_xlabel(r"Ideal Initial Loss $\mu$")
        axes[1].set_ylabel(YLABEL_LT)
        axes[1].set_title(r"(b) Residual Unfamiliarity vs. $\mu$")

        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()
    print(f"Saved mu analysis → {output_path}")


# ---------------------------------------------------------------------------
# Figure: Image grid (generic, one row per condition)
# ---------------------------------------------------------------------------
def plot_image_grid(image_dirs, output_path, ncols=10, labelpad=60):
    """Grid of generated images, one row per condition.

    Args:
        image_dirs: list of (label, directory_path) tuples
        output_path: where to save the figure
        ncols: number of images per row
        labelpad: padding for row labels
    """
    from PIL import Image

    image_dirs = [(l, d) for l, d in image_dirs if os.path.isdir(d)]
    if not image_dirs:
        print(f"Warning: no valid image dirs, skipping {output_path}")
        return

    with plt.rc_context(STYLE):
        nrows = len(image_dirs)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 0.8, nrows * 0.95))
        if nrows == 1:
            axes = axes[np.newaxis, :]

        for row, (label, img_dir) in enumerate(image_dirs):
            pngs = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
            for col in range(ncols):
                ax = axes[row, col]
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if col < len(pngs):
                    img = Image.open(os.path.join(img_dir, pngs[col]))
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.set_visible(False)
            axes[row, 0].set_ylabel(label, fontsize=9, rotation=0,
                                     labelpad=labelpad, va="center",
                                     fontfamily="sans-serif",
                                     fontweight="bold")

        fig.tight_layout(h_pad=0.3, w_pad=0.3)
        plt.savefig(output_path)
        plt.close()
    print(f"Saved image grid → {output_path}")


# ---------------------------------------------------------------------------
# Concrete image grid configurations
# ---------------------------------------------------------------------------
def plot_grid_guidance_scale(output_dir):
    dirs = [
        ("Unguided", "outputs/mnist_frontier/images/unguided"),
        (r"$\eta=2$", "outputs/sweep_g2_mu008/images/scale_2"),
        (r"$\eta=5$", "outputs/sweep_g5_mu008/images/scale_5"),
        (r"$\eta=10$", "outputs/mnist_frontier_s10/images/scale_10"),
        (r"$\eta=50$", "outputs/sweep_g50_mu008/images/scale_50"),
    ]
    plot_image_grid(dirs, os.path.join(output_dir, "grid_guidance_scale.pdf"),
                    labelpad=40)


def plot_grid_mu_sweep(output_dir):
    dirs = []
    for m in MU_VALUES:
        base = _mu_dir(m)
        img_dir = _find_image_dir(base)
        if img_dir:
            dirs.append((f"$\\mu={m:.2f}$", img_dir))
    plot_image_grid(dirs, os.path.join(output_dir, "grid_mu_sweep.pdf"),
                    labelpad=45)


def plot_meta_ablation_grid(output_dir, etas=(1, 2, 5, 10, 50), ncols=6,
                            labelpad=60):
    """Side-by-side grid of meta-grad vs no-meta-grad CCML samples across
    guidance scales.

    Reads from outputs/mnist_frontier_{meta,nometa}_s{eta}/images/scale_{eta}/
    and writes meta_ablation_grid.pdf to output_dir. Two rows per eta
    (meta on top, no-meta below) so visual differences pop within each pair.
    """
    settings = []
    for eta in etas:
        for variant in ("meta", "nometa"):
            label = ("Meta " if variant == "meta" else "No-meta ") + f"$\\eta={eta}$"
            d = f"outputs/mnist_frontier_{variant}_s{eta}/images/scale_{eta}"
            settings.append((label, d))
    plot_image_grid(settings, os.path.join(output_dir, "meta_ablation_grid.pdf"),
                    ncols=ncols, labelpad=labelpad)


def plot_natural_image_grid(output_path,
                            selection_dir="outputs/natural_selection",
                            classes=None, max_per_row=5,
                            min_per_row=3, labelpad=20,
                            labels=None,
                            column_headers=None):
    """Tile per-class rows from outputs/natural_selection/{class}/ for the
    natural-image figure.

    Each subfolder of selection_dir is one class; its PNGs (sorted by name)
    populate a row, capped at max_per_row. Classes with fewer than
    min_per_row picks are skipped.

    `labels`: optional dict mapping subdir name -> display label. Falls back
    to subdir.replace("_"," ").title() when not provided.
    """
    from PIL import Image

    if not os.path.isdir(selection_dir):
        print(f"Warning: {selection_dir} not found, skipping {output_path}")
        return

    if classes is None:
        classes = sorted(d for d in os.listdir(selection_dir)
                         if os.path.isdir(os.path.join(selection_dir, d)))

    rows = []
    for c in classes:
        cdir = os.path.join(selection_dir, c)
        pngs = sorted(f for f in os.listdir(cdir) if f.endswith(".png"))
        if len(pngs) < min_per_row:
            print(f"  skipping {c}: {len(pngs)} picks < min {min_per_row}")
            continue
        if labels and c in labels:
            label = labels[c]
        else:
            label = c.replace("_", " ").title()
        rows.append((label, cdir, pngs[:max_per_row]))

    if not rows:
        print(f"Warning: no eligible class rows under {selection_dir}")
        return

    nrows, ncols = len(rows), max_per_row
    # If column_headers define multiple groups, insert a spacer column between
    # consecutive groups so the gap is visually wider than the regular wspace.
    spacer_after = set()
    if column_headers:
        for i in range(len(column_headers) - 1):
            _, end_left, _ = column_headers[i]
            start_right, _, _ = column_headers[i + 1]
            if end_left == start_right:
                spacer_after.add(end_left - 1)  # spacer column AFTER this index
    with plt.rc_context(STYLE):
        # Build width_ratios: 1.0 for image columns, 0.35 for spacer columns.
        from matplotlib.gridspec import GridSpec
        n_grid_cols = ncols + len(spacer_after)
        width_ratios = []
        col_to_grid = {}   # logical col index -> grid col index
        gi = 0
        for c in range(ncols):
            col_to_grid[c] = gi
            width_ratios.append(1.0)
            gi += 1
            if c in spacer_after:
                width_ratios.append(0.35)
                gi += 1
        fig_width = sum(width_ratios) * 1.8
        fig = plt.figure(figsize=(fig_width, nrows * 1.8))
        gs = GridSpec(nrows, n_grid_cols, figure=fig, width_ratios=width_ratios)
        # axes[r, c] uses logical col index c (0..ncols-1)
        axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c] = fig.add_subplot(gs[r, col_to_grid[c]])
        for r, (label, cdir, pngs) in enumerate(rows):
            for c in range(ncols):
                ax = axes[r, c]
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if c < len(pngs):
                    img = Image.open(os.path.join(cdir, pngs[c])).convert("RGB")
                    ax.imshow(img)
                else:
                    ax.set_visible(False)
            axes[r, 0].set_title(label, loc="left", fontsize=8,
                                 fontweight="bold", fontfamily="sans-serif",
                                 pad=2)
        # Optional column-group headers above the top row.
        # column_headers: list of (col_start, col_end_exclusive, label) — places a
        # fig-level text centered over the indicated column range, above the
        # first row's per-row label.
        if column_headers:
            # Reserve a small top margin for the headers (don't clear titles).
            plt.subplots_adjust(top=0.95)
            for col_start, col_end, header in column_headers:
                left = axes[0, col_start].get_position().x0
                right = axes[0, col_end - 1].get_position().x1
                xc = (left + right) / 2
                # Place header just above the first row.
                top = axes[0, 0].get_position().y1
                fig.text(xc, top + 0.008, header, ha="center", va="bottom",
                         fontsize=10, fontweight="bold", fontfamily="sans-serif")
            # Dashed vertical separator between consecutive column groups.
            from matplotlib.lines import Line2D
            for i in range(len(column_headers) - 1):
                _, end_left, _ = column_headers[i]
                start_right, _, _ = column_headers[i + 1]
                if end_left == start_right:
                    # Place line midway between the right edge of the last col
                    # of group i and the left edge of the first col of group i+1.
                    x_left = axes[0, end_left - 1].get_position().x1
                    x_right = axes[0, start_right].get_position().x0
                    x_mid = (x_left + x_right) / 2
                    y_top = axes[0, 0].get_position().y1
                    y_bot = axes[-1, 0].get_position().y0
                    fig.add_artist(Line2D([x_mid, x_mid], [y_bot, y_top],
                                          color="gray", linestyle="--",
                                          linewidth=0.8, transform=fig.transFigure))
        plt.subplots_adjust(wspace=0.08, hspace=0.18)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05)
        plt.close()
    print(f"Saved natural image grid → {output_path} ({nrows} classes)")


def plot_rank_tradeoff(output_path,
                       roots=("outputs/text_embedding_creative/clipLora_R*",),
                       vanilla_root="outputs/text_embedding_creative/vanilla_full_cfg7.5_*",
                       prompts=None, ranks=(4, 8, 16, 32), vlm_min=5):
    """Rank-tradeoff plot: $\\text{sim}_0$, $\\Delta\\text{sim}$, and the VLM
    acceptance rate as a function of LoRA rank. Reads scores.json from each
    clipLora_R* dir (CA) and each vanilla_full_cfg7.5_* dir (vanilla baseline).

    Per-prompt aggregates use VLM-relevance >= vlm_min for $\\text{sim}_0$ and
    $\\Delta\\text{sim}$ (consistent with Tables 2/3 in the paper); the VLM
    acceptance rate is the fraction of CA samples with VLM >= vlm_min over
    all CA samples per (prompt, rank). Final per-rank value is the mean
    across prompts; std bands use the same.
    """
    import glob, json, sys
    from collections import defaultdict
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from dinov2_distances import get_imagenet_label

    if prompts is None:
        prompts = [
            "a photo of a balloon", "a photo of a brown bear",
            "a photo of a castle", "a photo of a cheeseburger",
            "a photo of a daisy", "a photo of a espresso",
            "a photo of a flamingo", "a photo of a golden retriever",
            "a photo of a jellyfish", "a photo of a lighthouse",
            "a photo of a lion", "a photo of a peacock",
            "a photo of a volcano",
            "an alien botanical garden", "ancient library lit by fireflies",
            "macro photograph of bioluminescent jellyfish in the deep ocean",
            "a fortune teller's tent at twilight", "neon samurai in rain",
            "stained-glass cathedral interior at night",
            "translucent koi swimming in clouds",
        ]
    prompts = set(p.lower() for p in prompts)

    # CA: per-(prompt, rank) lists
    ca_sim0  = defaultdict(list)  # (prompt, rank) -> list of mean sim_0 (one per dir, VLM-filtered)
    ca_dsim  = defaultdict(list)
    ca_acc   = defaultdict(list)  # acceptance rate per dir

    for d in sorted(sum((glob.glob(g) for g in roots), [])):
        try:
            cfg = json.load(open(os.path.join(d, "config.json")))
            sc  = json.load(open(os.path.join(d, "scores.json")))
        except Exception:
            continue
        r = cfg.get("lora_rank")
        if r not in ranks:
            continue
        p = cfg.get("init_text") or cfg.get("prompt")
        if not p and cfg.get("class_label") is not None:
            p = f"a photo of a {get_imagenet_label(int(cfg['class_label']))}"
        if not p or p.lower() not in prompts:
            continue
        records = list(sc.get("images", {}).values())
        passed = [v for v in records
                  if v.get("vlm_relevance", -1) >= vlm_min
                  and v.get("init_sim", -1) >= 0
                  and v.get("delta_sim", -1) >= 0]
        if not records or not passed:
            continue
        ca_acc[(p.lower(), r)].append(len(passed) / len(records))
        ca_sim0[(p.lower(), r)].append(np.mean([v["init_sim"]   for v in passed]))
        ca_dsim[(p.lower(), r)].append(np.mean([v["delta_sim"] for v in passed]))

    # Vanilla per-prompt sim_0 (no inner loop)
    van_sim0 = defaultdict(list)
    for d in sorted(glob.glob(vanilla_root)):
        try:
            cfg = json.load(open(os.path.join(d, "config.json")))
            sc  = json.load(open(os.path.join(d, "scores.json")))
        except Exception:
            continue
        p = (cfg.get("prompt") or "").lower()
        if p not in prompts:
            continue
        sims = [v.get("init_sim", -1) for v in sc.get("images", {}).values()
                if v.get("init_sim", -1) >= 0]
        if sims:
            van_sim0[p].append(np.mean(sims))

    # Per-rank aggregates: mean across prompts (each prompt contributes its
    # mean across the CA dirs we found for it).
    def per_rank(d):
        out = {"rank": [], "mean": [], "std": []}
        for r in ranks:
            vals = [np.mean(d[(p, r)]) for p in prompts if (p, r) in d and len(d[(p, r)]) > 0]
            if not vals:
                continue
            out["rank"].append(r); out["mean"].append(np.mean(vals)); out["std"].append(np.std(vals))
        return out

    sim0 = per_rank(ca_sim0)
    dsim = per_rank(ca_dsim)
    acc  = per_rank(ca_acc)

    van_vals = [np.mean(van_sim0[p]) for p in prompts if p in van_sim0]
    van_mean = float(np.mean(van_vals)) if van_vals else None

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        c_sim0 = "#1A5276"; c_dsim = "#E8A317"; c_acc  = "#C0392B"

        for series, label, color, marker in [
            (sim0, r"$\mathrm{sim}_0$ (concept fidelity)", c_sim0, "o"),
            (dsim, r"$\Delta\mathrm{sim}$ (inner-loop reward)", c_dsim, "s"),
            (acc,  "VLM acceptance rate", c_acc,  "^"),
        ]:
            ax.plot(series["rank"], series["mean"], "-", color=color, lw=1.6,
                    marker=marker, ms=6, label=label)
            ax.fill_between(series["rank"],
                            np.array(series["mean"]) - np.array(series["std"]),
                            np.array(series["mean"]) + np.array(series["std"]),
                            color=color, alpha=0.12, lw=0)

        if van_mean is not None:
            ax.axhline(van_mean, ls="--", lw=1.0, color=c_sim0, alpha=0.7)
            ax.text(ranks[-1], van_mean + 0.012, f"vanilla SD $\\mathrm{{sim}}_0={van_mean:.2f}$",
                    ha="right", va="bottom", fontsize=9, color=c_sim0)

        ax.set_xscale("log", base=2)
        ax.set_xticks(list(ranks)); ax.set_xticklabels([str(r) for r in ranks])
        ax.set_xlabel("LoRA rank $r$")
        ax.set_ylabel("similarity / rate")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper right", frameon=False)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
    print(f"Saved rank tradeoff → {output_path}")


def plot_grid_baselines(output_dir):
    dirs = [
        ("Unguided", "outputs/mnist_frontier/images/unguided"),
        (r"Random $\sigma_p=2$", "outputs/baseline_random/images/random"),
        (r"Random $\sigma_p=5$", "outputs/baseline_random_p5/images/random"),
        (r"Repulsive $\eta=2$", "outputs/baseline_repulsive_g2/images/repulsive"),
        (r"Repulsive $\eta=5$", "outputs/baseline_repulsive/images/repulsive"),
        (r"Ours $\eta=2$", "outputs/sweep_g2_mu008/images/scale_2"),
        (r"Ours $\eta=5$", "outputs/sweep_g5_mu008/images/scale_5"),
    ]
    plot_image_grid(dirs, os.path.join(output_dir, "grid_baselines.pdf"),
                    labelpad=70)


def plot_baseline_combined(output_path, etas=None, grid_ncols=6):
    """Combined figure: scatter plot (left) + image grid (right)."""
    from PIL import Image
    import matplotlib.gridspec as gridspec

    if etas is None:
        etas = [1, 2, 5, 10]
    eta_norm = {e: 0.35 + 0.55 * i / (len(etas) - 1) for i, e in enumerate(etas)}

    # Selected rows for the grid (one per method at eta=5)
    # Cherry-picked indices per row
    grid_rows = [
        ("Unguided", "outputs/mnist_frontier/images/unguided",
         [0, 1, 2, 3, 4, 5]),
        (r"Random $\sigma_p=5$", "outputs/baseline_random_p5/images/random",
         [0, 1, 2, 3, 4, 5]),
        (r"Repulsive $\eta=5$", "outputs/baseline_repulsive/images/repulsive",
         [0, 1, 2, 3, 4, 5]),
        (r"Ours $\eta=5$", "outputs/sweep_g5_mu008/images/scale_5",
         [0, 2, 5, 6, 7, 9]),
    ]
    grid_rows = [(l, d, idx) for l, d, idx in grid_rows if os.path.isdir(d)]
    nrows = len(grid_rows)

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(13, 4.0))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.6], wspace=0.3)

        # --- Left: scatter ---
        ax = fig.add_subplot(gs[0])
        ax.plot([0, 0.35], [0, 0.35], 'k--', alpha=0.3,
                label="$L_T = L_0$ (no learning)")

        for eta in etas:
            path = BASELINE_RANDOM_PATHS.get(eta)
            if path and os.path.exists(path):
                L0, LT = _get_first_baseline(path)
                if L0 is not None:
                    ax.scatter(L0, LT, c=[cm.Greens(eta_norm[eta])], marker="^",
                               s=45, alpha=0.8,
                               label=f"Random ($\\sigma_p={eta}$)",
                               edgecolors="white", linewidths=0.3, zorder=3)

        for eta in etas:
            path = BASELINE_REPULSIVE_PATHS.get(eta)
            if path and os.path.exists(path):
                L0, LT = _get_first_baseline(path)
                if L0 is not None:
                    ax.scatter(L0, LT, c=[cm.Reds(eta_norm[eta])], marker="v",
                               s=45, alpha=0.8,
                               label=f"Repulsive ($\\eta={eta}$)",
                               edgecolors="white", linewidths=0.3, zorder=3)

        for eta in etas:
            path = CCML_PATHS.get(eta)
            if path and os.path.exists(path):
                L0, LT = _get_first_ccml(path)
                if L0 is not None:
                    ax.scatter(L0, LT, c=[cm.Blues(eta_norm[eta])], marker="*",
                               s=120, alpha=0.85,
                               label=f"Ours ($\\eta={eta}$)",
                               edgecolors="black", linewidths=0.3, zorder=4)

        ax.set_xlabel(XLABEL_L0)
        ax.set_ylabel(YLABEL_LT)
        ax.set_xlim(-0.02, 0.35)
        ax.set_ylim(-0.02, 0.35)
        ax.legend(fontsize=8, loc="upper left", ncol=1)

        # --- Right: image grid ---
        gs_right = gridspec.GridSpecFromSubplotSpec(
            nrows, grid_ncols, subplot_spec=gs[1],
            hspace=0.45, wspace=0.05)

        for row, (label, img_dir, pick_idx) in enumerate(grid_rows):
            pngs = sorted(f for f in os.listdir(img_dir)
                          if f.endswith(".png"))
            selected = [pngs[i] for i in pick_idx if i < len(pngs)]
            for col in range(grid_ncols):
                ax_img = fig.add_subplot(gs_right[row, col])
                ax_img.set_xticks([]); ax_img.set_yticks([])
                for spine in ax_img.spines.values():
                    spine.set_visible(False)
                if col < len(selected):
                    img = Image.open(os.path.join(img_dir, selected[col]))
                    ax_img.imshow(img, cmap="gray", vmin=0, vmax=255)
                else:
                    ax_img.set_visible(False)
                if col == 0:
                    ax_img.set_title(label, loc="left", fontsize=10,
                                     fontweight="bold",
                                     fontfamily="sans-serif", pad=6)

        # Aligned titles using fig.text at same y
        title_y = 0.97
        # Center of left panel
        left_pos = gs[0].get_position(fig)
        fig.text((left_pos.x0 + left_pos.x1) / 2, title_y,
                 "(a) Baseline Comparison",
                 ha="center", va="top", fontsize=11, fontfamily="serif")
        # Center of right panel
        right_pos = gs[1].get_position(fig)
        fig.text((right_pos.x0 + right_pos.x1) / 2, title_y,
                 "(b) Representative Samples",
                 ha="center", va="top", fontsize=11, fontfamily="serif")

        plt.savefig(output_path)
        plt.close()
    print(f"Saved baseline combined → {output_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(scores, label=""):
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
    print(f"{'Category':>22s}  {'n':>3s}  {'L_0':>7s}  {'imp':>7s}  "
          f"{'gate':>7s}  {'shaped':>9s}  {'rel':>7s}")
    print("-" * 72)
    for cat_name, results in scores.items():
        L0, LT = _extract(results)
        imp = np.mean(improvement(L0, LT))
        gate = np.mean(complexity_gate(L0))
        sr = np.mean(shaped_reward(L0, LT))
        rel = np.mean(improvement(L0, LT) / (L0 + 1e-6))
        print(f"{cat_name:>22s}  {len(results):3d}  {np.mean(L0):7.4f}  "
              f"{imp:7.4f}  {gate:7.4f}  {sr:9.6f}  {rel:7.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--scores", type=str,
                        default="outputs/mnist_frontier/scores.json")
    parser.add_argument("--layer-scores", type=str,
                        default="outputs/mnist_frontier/scores_layers.json")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--mirror-to", type=str, action="append",
                        default=[],
                        help="Additional dir to mirror output PDFs into "
                             "(repeatable; silently skipped if missing)")
    # Figure groups
    parser.add_argument("--all", action="store_true", help="Generate all")
    parser.add_argument("--baselines", action="store_true",
                        help="Baseline comparison scatter")
    parser.add_argument("--analysis", action="store_true",
                        help="Scale and mu analysis plots")
    parser.add_argument("--grids", action="store_true",
                        help="All image grids")
    parser.add_argument("--reward-ablation", action="store_true")
    parser.add_argument("--appendix", action="store_true",
                        help="Layer ablation (appendix)")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-dirs", type=str, default="")
    parser.add_argument("--natural", action="store_true",
                        help="Natural-image grid from per-class selections in outputs/natural_selection")
    parser.add_argument("--meta-ablation", action="store_true",
                        help="Meta vs no-meta CCML grid across guidance scales")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out = args.output_dir

    # Load main scores lazily — natural-image grid does not need them, so
    # plot_frontier.py can run on the laptop where MNIST scores aren't synced.
    scores = None
    if os.path.exists(args.scores):
        with open(args.scores) as f:
            scores = json.load(f)
    elif not args.natural:
        print(f"Warning: {args.scores} not found; skipping score-based figures")

    # --- Main figures (always generated when scores available) ---
    if scores is not None:
        plot_main_frontier(scores, os.path.join(out, "frontier.pdf"))
        print_summary(scores, "Main Frontier")

    # --- Reward ablation ---
    if (args.reward_ablation or args.all) and scores is not None:
        plot_reward_ablation(scores, os.path.join(out, "reward_ablation.pdf"))
        print_overlap_correlation(scores)

    # --- Layer ablation (appendix) ---
    if args.appendix or args.all:
        if os.path.exists(args.layer_scores):
            layer_scores = _load_json(args.layer_scores)
            plot_layer_ablation(layer_scores,
                                os.path.join(out, "layer_ablation.pdf"))
            for ft, sc in layer_scores.items():
                print_summary(sc, f"Layer ablation: {ft}")
        else:
            print(f"Warning: {args.layer_scores} not found")

    # --- Sweep comparison ---
    if args.sweep or args.all:
        if args.sweep_dirs:
            sweep_dirs = []
            for entry in args.sweep_dirs.split(","):
                if ":" in entry:
                    label, path = entry.split(":", 1)
                    sweep_dirs.append((label, path))
        else:
            sweep_dirs = []
            for d in sorted(os.listdir("outputs")):
                sp = os.path.join("outputs", d, "scores.json")
                if d.startswith("sweep_") and os.path.exists(sp):
                    sweep_dirs.append((d.replace("sweep_", ""), sp))
        if sweep_dirs:
            plot_sweep(sweep_dirs, os.path.join(out, "sweep.pdf"))

    # --- Baseline comparison ---
    if args.baselines or args.all:
        plot_baseline_comparison(os.path.join(out, "baseline_comparison.pdf"))
        plot_baseline_combined(os.path.join(out, "baseline_combined.pdf"))

    # --- Scale and mu analysis ---
    if args.analysis or args.all:
        plot_scale_analysis(os.path.join(out, "scale_analysis.pdf"))
        plot_mu_analysis(os.path.join(out, "mu_analysis.pdf"))

    # --- Image grids ---
    if args.grids or args.all:
        plot_grid_guidance_scale(out)
        plot_grid_mu_sweep(out)
        plot_grid_baselines(out)

    # --- Natural-image grid ---
    if args.natural or args.all:
        plot_natural_image_grid(os.path.join(out, "natural_image_grid.pdf"),
                                max_per_row=4)

    # --- Meta vs no-meta MNIST ablation grid (preview, not paper figure) ---
    if args.meta_ablation or args.all:
        plot_meta_ablation_grid(out)

    # Optionally copy the PDFs into extra directories (--mirror-to).
    for mirror in args.mirror_to:
        if not os.path.isdir(mirror) or os.path.abspath(mirror) == os.path.abspath(out):
            continue
        for name in os.listdir(out):
            if name.endswith(".pdf"):
                shutil.copy2(os.path.join(out, name), os.path.join(mirror, name))
        print(f"Mirrored PDFs: {out} -> {mirror}")


if __name__ == "__main__":
    main()
