"""Evaluation utilities for computing rewards and running sanity checks."""

from typing import Callable, Dict, List, Optional, Tuple

import higher
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from ccml.models.appraisers import Appraiser
from ccml.rewards import calculate_creator_reward


def compute_l2_regularization(
    current_params: List[torch.Tensor],
    original_params_dict: Dict[str, torch.Tensor],
    param_names: List[str],
) -> torch.Tensor:
    """Compute L2 distance from original weights.

    Args:
        current_params: List of current parameter tensors
        original_params_dict: Dict mapping param names to original values
        param_names: Names of the current params (same order as current_params)
    """
    reg_loss = 0.0
    for curr, name in zip(current_params, param_names):
        if name in original_params_dict:
            orig = original_params_dict[name]
            reg_loss = reg_loss + torch.sum((curr - orig) ** 2)
    return reg_loss


def compute_ewc_regularization(
    current_params: List[torch.Tensor],
    original_params_dict: Dict[str, torch.Tensor],
    fisher_dict: Dict[str, torch.Tensor],
    param_names: List[str],
) -> torch.Tensor:
    """Compute EWC regularization weighted by Fisher information.

    Args:
        current_params: List of current parameter tensors
        original_params_dict: Dict mapping param names to original values
        fisher_dict: Dict mapping param names to Fisher diagonals
        param_names: Names of the current params (same order as current_params)
    """
    reg_loss = 0.0
    for curr, name in zip(current_params, param_names):
        if name in original_params_dict and name in fisher_dict:
            orig = original_params_dict[name]
            fisher = fisher_dict[name]
            reg_loss = reg_loss + torch.sum(fisher * (curr - orig) ** 2)
    return reg_loss


def run_inner_loop(
    appraiser_ft: Appraiser,
    image: torch.Tensor,
    finetune_layers: str,
    inner_loop_steps: int,
    inner_loop_lr: float,
    retain_graph: bool = False,
    step_callback: Optional[Callable[[int, Dict], None]] = None,
    reg_type: Optional[str] = None,
    reg_weight: float = 0.0,
    original_params_dict: Optional[Dict[str, torch.Tensor]] = None,
    fisher_dict: Optional[Dict[str, torch.Tensor]] = None,
    **loss_kwargs,
) -> None:
    """
    Run inner loop optimization on a cloned appraiser.

    This modifies appraiser_ft in-place.

    Args:
        appraiser_ft: Cloned appraiser to fine-tune (modified in-place)
        image: Input image tensor (B, C, H, W)
        finetune_layers: Which layers to fine-tune (e.g., "decoder", "all")
        inner_loop_steps: Number of optimization steps
        inner_loop_lr: Learning rate for inner loop
        retain_graph: Whether to retain computation graph (needed for guidance)
        step_callback: Optional callback(step, info_dict) called after each step.
                      info_dict contains: total_loss, recon_loss, reg_loss (if regularization enabled)
        reg_type: Regularization type ("l2", "ewc", or None)
        reg_weight: Regularization weight (lambda)
        original_params_dict: Dict mapping param names to original values
        fisher_dict: Dict mapping param names to Fisher diagonals (for EWC)
        **loss_kwargs: Additional kwargs passed to compute_reconstruction_loss
                      (e.g., timestep, loss_type for DDPM appraiser)
    """
    params_to_tune = appraiser_ft.get_finetune_params(finetune_layers)
    param_names = appraiser_ft.get_finetune_param_names(finetune_layers)

    if not params_to_tune:
        return

    inner_optimizer = optim.Adam(params_to_tune, lr=inner_loop_lr)

    # Set to train mode but keep BatchNorm in eval for stability
    appraiser_ft.train()
    for module in appraiser_ft.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    for step in range(inner_loop_steps):
        inner_optimizer.zero_grad()
        recon_loss = appraiser_ft.compute_reconstruction_loss(image, **loss_kwargs)

        # Build info dict for callback
        info = {"recon_loss": recon_loss.item()}

        # Add regularization if specified
        if reg_type is not None and reg_weight > 0 and original_params_dict is not None:
            if reg_type == "l2":
                reg_loss = compute_l2_regularization(params_to_tune, original_params_dict, param_names)
                info["reg_loss"] = reg_loss.item()
                loss = recon_loss + reg_weight * reg_loss
            elif reg_type == "ewc" and fisher_dict is not None:
                reg_loss = compute_ewc_regularization(
                    params_to_tune, original_params_dict, fisher_dict, param_names
                )
                info["reg_loss"] = reg_loss.item()
                loss = recon_loss + reg_weight * reg_loss
            else:
                loss = recon_loss
        else:
            loss = recon_loss

        info["total_loss"] = loss.item()

        loss.backward(retain_graph=retain_graph)
        inner_optimizer.step()

        if step_callback is not None:
            step_callback(step, info)

    appraiser_ft.eval()


def run_inner_loop_meta(
    appraiser: Appraiser,
    image: torch.Tensor,
    finetune_layers: str,
    inner_loop_steps: int,
    inner_loop_lr: float = 1e-4,
    step_callback: Optional[Callable[[int, Dict], None]] = None,
    reg_type: Optional[str] = None,
    reg_weight: float = 0.0,
    original_params_dict: Optional[Dict[str, torch.Tensor]] = None,
    fisher_dict: Optional[Dict[str, torch.Tensor]] = None,
    with_grad: bool = False,
    score_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run MAML-style differentiable inner loop for meta-gradient computation.

    Uses the `higher` library for differentiable optimization. This allows
    computing gradients that capture how changing the input image affects
    what the model learns during fine-tuning (not just the final loss).

    The gradient of final_loss w.r.t. image captures BOTH:
    - Direct effect: how image affects final evaluation
    - Indirect effect: how image affects what fine-tuning learns

    Args:
        appraiser: Appraiser model (will be wrapped by higher, not modified)
        image: Input image tensor (B, C, H, W) with requires_grad=True
        finetune_layers: Which layers to fine-tune (e.g., "decoder", "all")
        inner_loop_steps: Number of optimization steps
        inner_loop_lr: Learning rate for inner loop (default 1e-4 for Adam)
        step_callback: Optional callback(step, info_dict) called after each step.
                      info_dict contains: total_loss, recon_loss, reg_loss (if regularization enabled)
        reg_type: Regularization type ("l2", "ewc", or None)
        reg_weight: Regularization weight (lambda)
        original_params_dict: Dict mapping param names to original values
        fisher_dict: Dict mapping param names to Fisher diagonals (for EWC)
        with_grad: If True, pass with_grad=True to appraiser forward (needed for
                   frozen encoders like DINOv2 to allow gradients through)
        score_loss_fn: Optional custom loss function(reconstruction, target) -> loss.
                      If None, uses F.mse_loss. Can be color_normalized_mse etc.

    Returns:
        (final_loss, initial_loss) where both have meta-gradients to image.
        Use these directly for reward computation.
    """
    # Default to MSE if no custom loss function
    if score_loss_fn is None:
        score_loss_fn = F.mse_loss

    # Get parameters to tune and their names
    params_to_tune = appraiser.get_finetune_params(finetune_layers)
    param_names = appraiser.get_finetune_param_names(finetune_layers)

    if not params_to_tune:
        # No parameters to tune - just compute loss
        loss = appraiser.compute_reconstruction_loss(image)
        return loss, loss

    # Create optimizer for the parameters we want to tune
    inner_opt = optim.Adam(params_to_tune, lr=inner_loop_lr)

    # Use higher for differentiable inner loop
    with higher.innerloop_ctx(
        appraiser,
        inner_opt,
        track_higher_grads=True,
        copy_initial_weights=True,
    ) as (fmodel, diffopt):
        # Compute initial loss (before any fine-tuning) using score_loss_fn
        if with_grad:
            _, initial_reconstruction = fmodel(image)
        else:
            _, initial_reconstruction = fmodel(image)
        target = appraiser.get_target(image)
        initial_loss = score_loss_fn(initial_reconstruction, target)

        # Inner loop optimization (always uses MSE for training stability)
        for step in range(inner_loop_steps):
            # Pass with_grad to enable gradients through frozen encoder (e.g., DINOv2)
            if with_grad:
                _, reconstruction = fmodel(image)
            else:
                _, reconstruction = fmodel(image)
            recon_loss = F.mse_loss(reconstruction, target)

            # Build info dict for callback
            info = {"recon_loss": recon_loss.item()}

            # Add regularization if specified
            if reg_type is not None and reg_weight > 0 and original_params_dict is not None:
                # Get current params from functional model (in same order as param_names)
                fmodel_params_dict = dict(fmodel.named_parameters())
                current_tune_params = [fmodel_params_dict[name] for name in param_names if name in fmodel_params_dict]

                if reg_type == "l2":
                    reg_loss = compute_l2_regularization(current_tune_params, original_params_dict, param_names)
                    info["reg_loss"] = reg_loss.item()
                    loss = recon_loss + reg_weight * reg_loss
                elif reg_type == "ewc" and fisher_dict is not None:
                    reg_loss = compute_ewc_regularization(
                        current_tune_params, original_params_dict, fisher_dict, param_names
                    )
                    info["reg_loss"] = reg_loss.item()
                    loss = recon_loss + reg_weight * reg_loss
                else:
                    loss = recon_loss
            else:
                loss = recon_loss

            info["total_loss"] = loss.item()

            # Differentiable optimizer step
            diffopt.step(loss)

            if step_callback is not None:
                step_callback(step, info)

        # Final evaluation with updated parameters using score_loss_fn
        if with_grad:
            _, final_reconstruction = fmodel(image)
        else:
            _, final_reconstruction = fmodel(image)
        final_loss = score_loss_fn(final_reconstruction, target)

    return final_loss, initial_loss


def compute_reward_for_image(
    image: torch.Tensor,
    appraiser: Appraiser,
    finetune_layers: str,
    inner_loop_steps: int,
    inner_loop_lr: float,
    reward_config: Dict,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute reward metrics for a single image using bi-level optimization.

    Args:
        image: Input image tensor (B, C, H, W)
        appraiser: Appraiser model (will be cloned for fine-tuning)
        finetune_layers: Which layers to fine-tune (e.g., "decoder", "all")
        inner_loop_steps: Number of inner loop optimization steps
        inner_loop_lr: Learning rate for inner loop
        reward_config: Configuration for reward calculation
        device: Device to run on

    Returns:
        Dict with keys: initial_loss, final_loss, improvement, reward
    """
    appraiser.eval()

    # Clone appraiser for fine-tuning
    appraiser_ft = appraiser.clone().to(device)

    # Get initial loss (before fine-tuning)
    with torch.no_grad():
        initial_loss = appraiser_ft.compute_reconstruction_loss(image)

    # Run inner loop optimization
    run_inner_loop(
        appraiser_ft,
        image,
        finetune_layers,
        inner_loop_steps,
        inner_loop_lr,
        retain_graph=False,
    )

    # Get final loss (after fine-tuning)
    with torch.no_grad():
        final_loss = appraiser_ft.compute_reconstruction_loss(image)

    # Calculate reward
    reward, improvement_score, _, _ = calculate_creator_reward(
        initial_loss, final_loss, **reward_config)

    return {
        "initial_loss": initial_loss.item(),
        "final_loss": final_loss.item(),
        "improvement": improvement_score.item(),
        "reward": reward.item(),
    }


def run_sanity_check(
    appraiser: Appraiser,
    test_loader,
    creator,
    finetune_layers: str,
    inner_loop_steps: int,
    inner_loop_lr: float,
    reward_config: Dict,
    device: str = "cuda",
    num_samples: int = 5,
    art_loader=None,
) -> List[Dict]:
    """
    Run sanity check comparing rewards across different image types.

    Args:
        appraiser: Appraiser model
        test_loader: DataLoader for test images (e.g., CIFAR-10, MNIST)
        creator: Creator model for generating images
        finetune_layers: Which layers to fine-tune
        inner_loop_steps: Number of inner loop steps
        inner_loop_lr: Inner loop learning rate
        reward_config: Reward configuration
        device: Device to run on
        num_samples: Number of samples per category
        art_loader: Optional DataLoader for art images (e.g., WikiArt)

    Returns:
        List of result dicts with type, index, image, and metrics
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK: Comparing Rewards Across Image Types")
    print("=" * 80)

    results = []

    # Get test images
    test_iter = iter(test_loader)
    test_images, _ = next(test_iter)
    test_images = test_images[:num_samples].to(device)

    # Get art images if available
    if art_loader is not None:
        try:
            art_iter = iter(art_loader)
            art_images, _ = next(art_iter)
            art_images = art_images[:num_samples].to(device)
        except Exception as e:
            print(f"Warning: Could not load art images: {e}")
            art_images = None
    else:
        art_images = None

    # Generate images
    print(
        f"\nGenerating {num_samples} images for comparison (1000 steps each)..."
    )
    generated_images = []
    for i in range(num_samples):
        print(f"  Generating image {i+1}/{num_samples}...")
        with torch.no_grad():
            generated = creator.pipeline(batch_size=1,
                                         num_inference_steps=1000).images[0]
            generated_tensor = transforms.ToTensor()(generated).unsqueeze(
                0).to(device)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            generated_tensor = normalize(
                generated_tensor.squeeze(0)).unsqueeze(0)
            generated_images.append(generated_tensor)
    generated_images = torch.cat(generated_images, dim=0)

    # Print header
    print(
        f"\n{'Image Type':<15} {'Sample':<8} {'L_initial':<12} {'L_final':<12} "
        f"{'Improvement':<12} {'Reward':<12}")
    print("-" * 80)

    # Evaluate each sample
    for idx in range(num_samples):
        # Test images (CIFAR/MNIST)
        test_metrics = compute_reward_for_image(
            test_images[idx:idx + 1],
            appraiser,
            finetune_layers,
            inner_loop_steps,
            inner_loop_lr,
            reward_config,
            device,
        )
        results.append({
            "type": "Test",
            "index": idx,
            "image": test_images[idx:idx + 1],
            **test_metrics,
        })
        print(
            f"{'Test':<15} {idx+1:<8} {test_metrics['initial_loss']:<12.6f} "
            f"{test_metrics['final_loss']:<12.6f} {test_metrics['improvement']:<12.6f} "
            f"{test_metrics['reward']:<12.6f}")

        # Generated images
        gen_metrics = compute_reward_for_image(
            generated_images[idx:idx + 1],
            appraiser,
            finetune_layers,
            inner_loop_steps,
            inner_loop_lr,
            reward_config,
            device,
        )
        results.append({
            "type": "Generated",
            "index": idx,
            "image": generated_images[idx:idx + 1],
            **gen_metrics,
        })
        print(
            f"{'Generated':<15} {idx+1:<8} {gen_metrics['initial_loss']:<12.6f} "
            f"{gen_metrics['final_loss']:<12.6f} {gen_metrics['improvement']:<12.6f} "
            f"{gen_metrics['reward']:<12.6f}")

        # Art images
        if art_images is not None:
            art_metrics = compute_reward_for_image(
                art_images[idx:idx + 1],
                appraiser,
                finetune_layers,
                inner_loop_steps,
                inner_loop_lr,
                reward_config,
                device,
            )
            results.append({
                "type": "Art",
                "index": idx,
                "image": art_images[idx:idx + 1],
                **art_metrics,
            })
            print(
                f"{'Art':<15} {idx+1:<8} {art_metrics['initial_loss']:<12.6f} "
                f"{art_metrics['final_loss']:<12.6f} {art_metrics['improvement']:<12.6f} "
                f"{art_metrics['reward']:<12.6f}")
        print()

    print("=" * 80)

    # Print averages
    print("\nAverage Metrics by Image Type:")
    print(
        f"{'Type':<15} {'L_initial':<12} {'L_final':<12} {'Improvement':<12} {'Reward':<12}"
    )
    print("-" * 80)

    for img_type in ["Test", "Generated", "Art"]:
        type_results = [r for r in results if r["type"] == img_type]
        if type_results:
            avg_initial = np.mean([r["initial_loss"] for r in type_results])
            avg_final = np.mean([r["final_loss"] for r in type_results])
            avg_improvement = np.mean([r["improvement"] for r in type_results])
            avg_reward = np.mean([r["reward"] for r in type_results])
            print(f"{img_type:<15} {avg_initial:<12.6f} {avg_final:<12.6f} "
                  f"{avg_improvement:<12.6f} {avg_reward:<12.6f}")

    print("=" * 80 + "\n")

    return results


def compute_fisher_information(
    appraiser: Appraiser,
    data_loader,
    finetune_layers: str,
    device: str = "cuda",
    num_samples: int = 200,
) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal Fisher information matrix for EWC regularization.

    The Fisher information measures how sensitive the loss is to each parameter,
    indicating which parameters are "important" for the current task.

    Args:
        appraiser: Appraiser model
        data_loader: DataLoader with training data
        finetune_layers: Which layers to compute Fisher for
        device: Device to run on
        num_samples: Number of samples to use for estimation

    Returns:
        Dict mapping param names to Fisher diagonal tensors
    """
    appraiser.eval()
    param_names = appraiser.get_finetune_param_names(finetune_layers)
    params_to_tune = appraiser.get_finetune_params(finetune_layers)

    if not params_to_tune:
        return {}

    # Initialize Fisher diagonals with zeros
    fisher_dict = {name: torch.zeros_like(p) for name, p in zip(param_names, params_to_tune)}

    # Compute Fisher information from gradients
    sample_count = 0
    for images, _ in data_loader:
        if sample_count >= num_samples:
            break

        images = images.to(device)
        batch_size = images.size(0)

        for i in range(batch_size):
            if sample_count >= num_samples:
                break

            appraiser.zero_grad()
            loss = appraiser.compute_reconstruction_loss(images[i:i+1])
            loss.backward()

            # Accumulate squared gradients
            for name, p in zip(param_names, params_to_tune):
                if p.grad is not None:
                    fisher_dict[name] += p.grad.detach() ** 2

            sample_count += 1

    # Average
    for name in fisher_dict:
        fisher_dict[name] = fisher_dict[name] / sample_count

    return fisher_dict


def get_original_params(
    appraiser: Appraiser,
    finetune_layers: str,
) -> Dict[str, torch.Tensor]:
    """
    Get a detached copy of the original parameters for regularization.

    Args:
        appraiser: Appraiser model
        finetune_layers: Which layers to get params for

    Returns:
        Dict mapping param names to detached parameter tensors
    """
    param_names = appraiser.get_finetune_param_names(finetune_layers)
    params = appraiser.get_finetune_params(finetune_layers)
    return {name: p.detach().clone() for name, p in zip(param_names, params)}
