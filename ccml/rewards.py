"""Reward functions for CCML."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def color_normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE after normalizing colors per-image.

    This makes the loss focus on structure rather than absolute colors.
    A uniform color image will have near-zero loss after normalization,
    preventing the creativity objective from favoring trivial uniform colors.

    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)

    Returns:
        Color-normalized MSE loss
    """
    # Subtract per-image mean color (B, C, 1, 1)
    pred_mean = pred.mean(dim=(2, 3), keepdim=True)
    target_mean = target.mean(dim=(2, 3), keepdim=True)

    pred_norm = pred - pred_mean
    target_norm = target - target_mean

    return F.mse_loss(pred_norm, target_norm)


def calculate_creator_reward(
    L_initial: torch.Tensor,
    L_final: torch.Tensor,
    ideal_initial_loss: float = 0.15,
    initial_loss_sigma: float = 0.2,
    final_loss_temp: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the shaped reward for the Creator based on learnability criteria.

    The reward encourages generating samples that are:
    1. Complex enough to be interesting (not trivially easy)
    2. Learnable (loss decreases with fine-tuning)
    3. Successfully reconstructed after learning

    Args:
        L_initial: Initial reconstruction loss (before fine-tuning)
        L_final: Final reconstruction loss (after fine-tuning)
        ideal_initial_loss: Target initial loss for "interesting" complexity
        initial_loss_sigma: Width of the complexity bell curve
        final_loss_temp: Temperature for final loss decay

    Returns:
        (total_reward, improvement_score, complexity_score, success_score)
    """
    # Improvement: how much did loss decrease?
    improvement_score = L_initial - L_final

    # Complexity: is initial loss in the "sweet spot"?
    # Not too easy (low loss) or too hard (high loss/noise)
    complexity_score = torch.exp(
        -((L_initial - ideal_initial_loss) ** 2) / (2 * initial_loss_sigma ** 2)
    )

    # Success: was the final loss low? (exponential decay)
    success_score = torch.exp(-L_final / final_loss_temp)

    # Total reward is the product of all three factors
    total_reward = improvement_score * complexity_score * success_score

    return total_reward, improvement_score, complexity_score, success_score


# Default reward configuration
DEFAULT_REWARD_CONFIG = {
    "ideal_initial_loss": 0.15,
    "initial_loss_sigma": 0.2,
    "final_loss_temp": 0.1,
}

# MNIST-specific configuration
MNIST_REWARD_CONFIG = {
    "ideal_initial_loss": 0.15,
    "initial_loss_sigma": 0.2,
    "final_loss_temp": 0.1,
}

# CIFAR-specific configuration (features have different scale)
CIFAR_REWARD_CONFIG = {
    "ideal_initial_loss": 0.1,
    "initial_loss_sigma": 0.1,
    "final_loss_temp": 0.1,
}


def calculate_layer_difference_reward(
    L_first_final: torch.Tensor,
    L_last_final: torch.Tensor,
    final_loss_temp: float = 0.1,
    ideal_initial_loss: float = 0.15,
    initial_loss_sigma: float = 0.2,
    L_initial: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate reward using the difference between decoder layer final losses.

    The idea:
    - decoder_last (output layer) captures low-level noise/texture
    - decoder_first (input layer) captures high-level semantic features

    If decoder_first achieves lower loss than decoder_last, the image has
    learnable semantic content (not just noise).

    Args:
        L_first_final: Final loss after fine-tuning decoder_first only
        L_last_final: Final loss after fine-tuning decoder_last only
        final_loss_temp: Temperature for success score
        ideal_initial_loss: Target initial loss for complexity sweet spot
        initial_loss_sigma: Width of the complexity bell curve
        L_initial: Initial loss (before any fine-tuning) for complexity score

    Returns:
        (total_reward, layer_diff_score, complexity_score, success_score)
    """
    # Layer difference: how much better is semantic (first) vs noise (last)?
    # Positive = semantic layer learns better = good learnability
    layer_diff_score = L_last_final - L_first_final

    # Complexity: is initial loss in the "sweet spot"?
    if L_initial is None:
        raise ValueError("L_initial is required for complexity score")
    complexity_score = torch.exp(
        -((L_initial - ideal_initial_loss) ** 2) / (2 * initial_loss_sigma ** 2)
    )

    # Success: was the semantic layer's final loss low?
    success_score = torch.exp(-L_first_final / final_loss_temp)

    # Total reward: layer_diff * complexity * success
    total_reward = layer_diff_score * complexity_score * success_score

    return total_reward, layer_diff_score, complexity_score, success_score
