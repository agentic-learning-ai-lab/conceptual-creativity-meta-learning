"""Guided generation with bi-level optimization."""

from typing import Dict, Optional, Tuple

import torch
from tqdm.auto import tqdm

from ccml.models.appraisers import Appraiser
from ccml.models.creators import DiffusionCreator
from ccml.rewards import calculate_creator_reward, calculate_layer_difference_reward, color_normalized_mse
from ccml.training.evaluation import run_inner_loop, run_inner_loop_meta


def generate_with_guidance(
    creator: DiffusionCreator,
    appraiser: Appraiser,
    finetune_layers: str = "decoder",
    num_steps: int = 1000,
    guidance_scale: float = 500.0,
    inner_loop_steps: int = 10,
    inner_loop_lr: float = 1e-4,
    device: str = "cuda",
    visualize_final_loop: bool = False,
    reward_config: Optional[Dict] = None,
    reward_type: str = "original",
    score_loss_type: str = "mse",
    reg_type: Optional[str] = None,
    reg_weight: float = 0.0,
    original_params_dict: Optional[Dict[str, torch.Tensor]] = None,
    fisher_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Generate an image using guided diffusion with bi-level optimization.

    Args:
        creator: DiffusionCreator model
        appraiser: Appraiser model (encoder + decoder)
        finetune_layers: Which layers to finetune in inner loop (for original reward)
        num_steps: Number of denoising steps
        guidance_scale: How strongly to apply guidance
        inner_loop_steps: Number of inner loop optimization steps
        inner_loop_lr: Learning rate for inner loop
        device: Device to run on
        visualize_final_loop: Whether to print detailed info for final step
        reward_config: Configuration for reward calculation
        reward_type: "original" (ideal_initial_loss) or "layer_diff" (decoder_first vs last)
        score_loss_type: Loss for reward calculation: "mse" or "color_normalized"
            (inner loop always uses MSE, this only affects reward computation)
        reg_type: Regularization type ("l2", "ewc", or None)
        reg_weight: Regularization weight (lambda)
        original_params_dict: Dict mapping param names to original values
        fisher_dict: Dict mapping param names to Fisher diagonals (for EWC)

    Returns:
        (generated_image, final_step_data)
    """
    if reward_config is None:
        reward_config = {}

    final_loss_temp = reward_config.get("final_loss_temp", 0.1)

    img_size = creator.get_img_size()
    channels = creator.get_channels()

    # Initialize latents
    latents = torch.randn((1, channels, img_size, img_size), device=device)

    # Get scheduler
    scheduler = creator.get_scheduler(num_steps)

    # Track data for the final step
    final_step_data = {}
    reward_history = []

    # Determine if appraiser expects grayscale by checking encoder input channels
    try:
        first_conv = next(m for m in appraiser.encoder.modules() if isinstance(m, torch.nn.Conv2d))
        is_grayscale = first_conv.in_channels == 1
    except StopIteration:
        is_grayscale = "mnist" in creator.model_name.lower()

    for t in tqdm(scheduler.timesteps, desc="Guided Diffusion", ncols=0):
        latents.requires_grad_(True)

        # Get noise prediction
        with torch.no_grad():
            noise_pred = creator.unet(latents, t).sample

        # Get predicted original sample
        pred_original_sample = scheduler.step(noise_pred, t, latents).pred_original_sample

        # Convert to grayscale if needed (for MNIST appraiser)
        if is_grayscale:
            pred_rgb = pred_original_sample.mean(dim=1, keepdim=True)
        else:
            pred_rgb = pred_original_sample

        is_final_step = (t == scheduler.timesteps[-1])

        if reward_type == "layer_diff":
            # Layer-difference reward: compare decoder_first vs decoder_last
            ideal_init = reward_config.get("ideal_initial_loss", 0.15)
            init_sigma = reward_config.get("initial_loss_sigma", 0.2)
            reward, L_initial, L_first, L_last, layer_diff, complexity, app_first, app_last = compute_layer_diff_reward(
                appraiser, pred_rgb, inner_loop_steps, inner_loop_lr, final_loss_temp, device,
                ideal_init, init_sigma,
                reg_type=reg_type, reg_weight=reg_weight,
                original_params_dict=original_params_dict, fisher_dict=fisher_dict,
            )

            if is_final_step:
                final_step_data["initial_loss"] = L_initial.item()
                final_step_data["L_first_final"] = L_first.item()
                final_step_data["L_last_final"] = L_last.item()
                final_step_data["layer_diff"] = layer_diff.item()
                # Capture reconstructions
                with torch.no_grad():
                    _, initial_recon = appraiser(pred_rgb)
                    _, first_recon = app_first(pred_rgb)
                    _, last_recon = app_last(pred_rgb)
                    final_step_data["initial_recon"] = initial_recon.detach().cpu()
                    final_step_data["first_recon"] = first_recon.detach().cpu()
                    final_step_data["last_recon"] = last_recon.detach().cpu()

                if visualize_final_loop:
                    print(f"\n  === Layer-Difference Reward Summary ===")
                    print(f"    Initial Loss (no finetune): {L_initial.item():.6f}")
                    print(f"    L_first (decoder_first):    {L_first.item():.6f}")
                    print(f"    L_last (decoder_last):      {L_last.item():.6f}")
                    print(f"    Layer Diff (last - first):  {layer_diff.item():.6f}")
                    print(f"    Complexity Score:           {complexity.item():.6f}")
                    print(f"    Final Reward:               {reward.item():.6f}")
                    print(f"  ========================================\n")

            reward_history.append({
                "timestep": t.item(),
                "L_first_final": L_first.item(),
                "L_last_final": L_last.item(),
                "layer_diff": layer_diff.item(),
                "complexity": complexity.item(),
                "reward": reward.item(),
            })

        else:
            # Original reward: ideal_initial_loss based
            appraiser_ft = appraiser.clone().to(device)
            initial_loss = appraiser.compute_reconstruction_loss(pred_rgb)

            step_callback = None
            if is_final_step and visualize_final_loop:
                print(f"\n  === Inner Loop Optimization (Final Denoising Timestep) ===")

                def step_callback(step, info):
                    if step == 0 or step == inner_loop_steps // 2 or step == inner_loop_steps - 1:
                        if "reg_loss" in info:
                            print(f"      Step {step+1}/{inner_loop_steps}, Total: {info['total_loss']:.6f}, Recon: {info['recon_loss']:.6f}, Reg: {info['reg_loss']:.6f}")
                        else:
                            print(f"      Step {step+1}/{inner_loop_steps}, Loss: {info['total_loss']:.6f}")

            run_inner_loop(
                appraiser_ft, pred_rgb, finetune_layers,
                inner_loop_steps, inner_loop_lr, retain_graph=True, step_callback=step_callback,
                reg_type=reg_type, reg_weight=reg_weight,
                original_params_dict=original_params_dict, fisher_dict=fisher_dict,
            )

            final_loss = appraiser_ft.compute_reconstruction_loss(pred_rgb)

            # Get reconstructions for scoring or final step viz
            if score_loss_type == "color_normalized" or is_final_step:
                with torch.no_grad():
                    _, initial_recon = appraiser(pred_rgb)
                    _, final_recon = appraiser_ft(pred_rgb)
                if is_final_step:
                    final_step_data["initial_loss"] = initial_loss.item()
                    final_step_data["final_loss"] = final_loss.item()
                    final_step_data["initial_recon"] = initial_recon.detach().cpu()
                    final_step_data["final_recon"] = final_recon.detach().cpu()

            # Compute score losses for reward (may differ from inner loop MSE)
            if score_loss_type == "color_normalized":
                L_initial = color_normalized_mse(initial_recon, pred_rgb)
                L_final = color_normalized_mse(final_recon, pred_rgb)
            else:
                L_initial = initial_loss
                L_final = final_loss

            reward, improvement, complexity, success = calculate_creator_reward(
                L_initial, L_final, **reward_config
            )

            reward_history.append({
                "timestep": t.item(),
                "initial_loss": initial_loss.item(),
                "final_loss": final_loss.item(),
                "reward": reward.item(),
                "improvement": improvement.item(),
                "complexity": complexity.item(),
                "success": success.item(),
            })

            if is_final_step and visualize_final_loop:
                _print_reward_summary(
                    initial_loss.item(), final_loss.item(), improvement.item(),
                    complexity.item(), success.item(), reward.item(), reward_config,
                )

        # Compute gradient for guidance
        grad = torch.autograd.grad(reward, latents)[0]

        # Update latents with guidance
        with torch.no_grad():
            noise_pred = creator.unet(latents, t).sample
            latents_update = scheduler.step(noise_pred, t, latents).prev_sample
            guided_latents = latents_update + guidance_scale * grad

        latents = guided_latents.detach()

    final_step_data["reward_history"] = reward_history
    return latents, final_step_data


def _print_reward_summary(
    initial_loss: float,
    final_loss: float,
    improvement: float,
    complexity: float,
    success: float,
    reward: float,
    reward_config: Dict,
):
    """Print detailed reward breakdown."""
    ideal_initial = reward_config.get("ideal_initial_loss", 0.15)

    print(f"\n  === Reconstruction Loss Summary ===")
    print(f"    Before Fine-tuning (Initial): {initial_loss:.6f}")
    print(f"    After Fine-tuning (Final):    {final_loss:.6f}")
    print(f"    Improvement (Δ):              {improvement:.6f}")
    print(f"    Improvement %:                {100 * improvement / (initial_loss + 1e-8):.1f}%")

    print(f"\n  === Creativity Reward Breakdown ===")
    print(f"    1. Initial Complexity Score:  {complexity:.4f}")
    print(f"       (Is initial loss ~{ideal_initial}? Actual: {initial_loss:.3f})")
    print(f"    2. Final Success Score:       {success:.4f}")
    print(f"       (How low is final loss? Lower is better)")
    print(f"    3. Improvement Score:         {improvement:.4f}")
    print(f"       (How much did the Appraiser learn?)")
    print(f"    ----------------------------------------")
    print(f"    Final Creativity Reward:      {reward:.4f}")
    print(f"    ========================================\n")


def generate_with_meta_guidance(
    creator: DiffusionCreator,
    appraiser: Appraiser,
    finetune_layers: str = "decoder",
    num_steps: int = 1000,
    guidance_scale: float = 500.0,
    inner_loop_steps: int = 5,
    inner_loop_lr: float = 1e-4,
    device: str = "cuda",
    visualize_final_loop: bool = False,
    reward_config: Optional[Dict] = None,
    reward_type: str = "original",
    score_loss_type: str = "mse",
    reg_type: Optional[str] = None,
    reg_weight: float = 0.0,
    original_params_dict: Optional[Dict[str, torch.Tensor]] = None,
    fisher_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Generate an image using meta-gradient guided diffusion.

    Unlike generate_with_guidance, this version uses MAML-style differentiable
    fine-tuning. The gradient captures BOTH:
    - Direct effect: how the image affects final evaluation
    - Indirect effect: how the image affects what fine-tuning learns

    This should produce better guidance because it optimizes for images that
    cause useful learning, not just images that reconstruct well after
    independent fine-tuning.

    Args:
        creator: DiffusionCreator model
        appraiser: Appraiser model (encoder + decoder)
        finetune_layers: Which layers to finetune in inner loop
        num_steps: Number of denoising steps
        guidance_scale: How strongly to apply guidance
        inner_loop_steps: Number of inner loop optimization steps (keep small, 3-5)
        inner_loop_lr: Learning rate for inner loop
        device: Device to run on
        visualize_final_loop: Whether to print detailed info for final step
        reward_config: Configuration for reward calculation
        reward_type: "original" (ideal_initial_loss) or "layer_diff" (decoder_first vs last)
        reg_type: Regularization type ("l2", "ewc", or None)
        reg_weight: Regularization weight (lambda)
        original_params_dict: Dict mapping param names to original values
        fisher_dict: Dict mapping param names to Fisher diagonals (for EWC)

    Returns:
        (generated_image, final_step_data)
    """
    if reward_config is None:
        reward_config = {}

    final_loss_temp = reward_config.get("final_loss_temp", 0.1)

    img_size = creator.get_img_size()
    channels = creator.get_channels()

    # Initialize latents
    latents = torch.randn((1, channels, img_size, img_size), device=device)

    # Get scheduler
    scheduler = creator.get_scheduler(num_steps)

    # Track data for the final step
    final_step_data = {}
    reward_history = []

    # Determine if appraiser expects grayscale by checking encoder input channels
    try:
        first_conv = next(m for m in appraiser.encoder.modules() if isinstance(m, torch.nn.Conv2d))
        is_grayscale = first_conv.in_channels == 1
    except StopIteration:
        is_grayscale = "mnist" in creator.model_name.lower()

    for t in tqdm(scheduler.timesteps, desc="Meta-Guided Diffusion", ncols=0):
        latents.requires_grad_(True)

        # Get noise prediction
        with torch.no_grad():
            noise_pred = creator.unet(latents, t).sample

        # Get predicted original sample
        pred_original_sample = scheduler.step(noise_pred, t, latents).pred_original_sample

        # Convert to grayscale if needed (for MNIST appraiser)
        if is_grayscale:
            pred_rgb = pred_original_sample.mean(dim=1, keepdim=True)
        else:
            pred_rgb = pred_original_sample

        is_final_step = (t == scheduler.timesteps[-1])

        if reward_type == "layer_diff":
            # Layer-difference reward with meta-gradients
            ideal_init = reward_config.get("ideal_initial_loss", 0.15)
            init_sigma = reward_config.get("initial_loss_sigma", 0.2)
            reward, L_initial, L_first, L_last, layer_diff, complexity = compute_layer_diff_reward_meta(
                appraiser, pred_rgb, inner_loop_steps, inner_loop_lr, final_loss_temp,
                ideal_init, init_sigma,
                reg_type=reg_type, reg_weight=reg_weight,
                original_params_dict=original_params_dict, fisher_dict=fisher_dict,
            )

            if is_final_step:
                final_step_data["initial_loss"] = L_initial.item()
                final_step_data["L_first_final"] = L_first.item()
                final_step_data["L_last_final"] = L_last.item()
                final_step_data["layer_diff"] = layer_diff.item()

                # Capture reconstructions (run regular inner loops for visualization)
                with torch.no_grad():
                    _, initial_recon = appraiser(pred_rgb)
                    final_step_data["initial_recon"] = initial_recon.detach().cpu()

                appraiser_first = appraiser.clone().to(device)
                run_inner_loop(appraiser_first, pred_rgb.detach(), "decoder_first",
                              inner_loop_steps, inner_loop_lr, retain_graph=False)
                appraiser_last = appraiser.clone().to(device)
                run_inner_loop(appraiser_last, pred_rgb.detach(), "decoder_last",
                              inner_loop_steps, inner_loop_lr, retain_graph=False)
                with torch.no_grad():
                    _, first_recon = appraiser_first(pred_rgb)
                    _, last_recon = appraiser_last(pred_rgb)
                    final_step_data["first_recon"] = first_recon.detach().cpu()
                    final_step_data["last_recon"] = last_recon.detach().cpu()

                if visualize_final_loop:
                    print(f"\n  === Meta Layer-Difference Reward Summary ===")
                    print(f"    Initial Loss (no finetune): {L_initial.item():.6f}")
                    print(f"    L_first (decoder_first):    {L_first.item():.6f}")
                    print(f"    L_last (decoder_last):      {L_last.item():.6f}")
                    print(f"    Layer Diff (last - first):  {layer_diff.item():.6f}")
                    print(f"    Complexity Score:           {complexity.item():.6f}")
                    print(f"    Final Reward:               {reward.item():.6f}")
                    print(f"  =============================================\n")

            reward_history.append({
                "timestep": t.item(),
                "L_first_final": L_first.item(),
                "L_last_final": L_last.item(),
                "layer_diff": layer_diff.item(),
                "complexity": complexity.item(),
                "reward": reward.item(),
            })

        else:
            # Original reward with meta-gradients
            initial_loss = appraiser.compute_reconstruction_loss(pred_rgb)

            step_callback = None
            if is_final_step and visualize_final_loop:
                print(f"\n  === Meta Inner Loop Optimization (Final Denoising Timestep) ===")

                def step_callback(step, info):
                    if step == 0 or step == inner_loop_steps // 2 or step == inner_loop_steps - 1:
                        if "reg_loss" in info:
                            print(f"      Step {step+1}/{inner_loop_steps}, Total: {info['total_loss']:.6f}, Recon: {info['recon_loss']:.6f}, Reg: {info['reg_loss']:.6f}")
                        else:
                            print(f"      Step {step+1}/{inner_loop_steps}, Loss: {info['total_loss']:.6f}")

            final_loss, _ = run_inner_loop_meta(
                appraiser, pred_rgb, finetune_layers,
                inner_loop_steps, inner_loop_lr, step_callback=step_callback,
                reg_type=reg_type, reg_weight=reg_weight,
                original_params_dict=original_params_dict, fisher_dict=fisher_dict,
            )

            if is_final_step:
                final_step_data["initial_loss"] = initial_loss.item()
                final_step_data["final_loss"] = final_loss.item()
                with torch.no_grad():
                    _, initial_recon = appraiser(pred_rgb)
                    final_step_data["initial_recon"] = initial_recon.detach().cpu()

                appraiser_viz = appraiser.clone().to(device)
                run_inner_loop(
                    appraiser_viz, pred_rgb.detach(), finetune_layers,
                    inner_loop_steps, inner_loop_lr, retain_graph=False,
                )
                with torch.no_grad():
                    _, final_recon = appraiser_viz(pred_rgb)
                    final_step_data["final_recon"] = final_recon.detach().cpu()

            reward, improvement, complexity, success = calculate_creator_reward(
                initial_loss, final_loss, **reward_config
            )

            reward_history.append({
                "timestep": t.item(),
                "initial_loss": initial_loss.item(),
                "final_loss": final_loss.item(),
                "reward": reward.item(),
                "improvement": improvement.item(),
                "complexity": complexity.item(),
                "success": success.item(),
            })

            if is_final_step and visualize_final_loop:
                _print_reward_summary(
                    initial_loss.item(), final_loss.item(), improvement.item(),
                    complexity.item(), success.item(), reward.item(), reward_config,
                )

        # Compute gradient for guidance (meta-gradient captures full effect)
        grad = torch.autograd.grad(reward, latents)[0]

        # Update latents with guidance
        with torch.no_grad():
            noise_pred = creator.unet(latents, t).sample
            latents_update = scheduler.step(noise_pred, t, latents).prev_sample
            guided_latents = latents_update + guidance_scale * grad

        latents = guided_latents.detach()

    final_step_data["reward_history"] = reward_history
    return latents, final_step_data


def compute_layer_diff_reward(
    appraiser: Appraiser,
    pred_rgb: torch.Tensor,
    inner_loop_steps: int,
    inner_loop_lr: float,
    final_loss_temp: float,
    device: str,
    ideal_initial_loss: float = 0.15,
    initial_loss_sigma: float = 0.2,
    reg_type: Optional[str] = None,
    reg_weight: float = 0.0,
    original_params_dict: Optional[Dict[str, torch.Tensor]] = None,
    fisher_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Appraiser, Appraiser]:
    """
    Compute layer-difference reward by fine-tuning decoder_first vs decoder_last.

    Returns:
        (reward, L_initial, L_first_final, L_last_final, layer_diff, complexity, appraiser_first, appraiser_last)
    """
    # Get initial loss (before any fine-tuning)
    L_initial = appraiser.compute_reconstruction_loss(pred_rgb)

    # Fine-tune decoder_first only
    appraiser_first = appraiser.clone().to(device)
    run_inner_loop(
        appraiser_first, pred_rgb, "decoder_first",
        inner_loop_steps, inner_loop_lr, retain_graph=True,
        reg_type=reg_type, reg_weight=reg_weight,
        original_params_dict=original_params_dict, fisher_dict=fisher_dict,
    )
    L_first_final = appraiser_first.compute_reconstruction_loss(pred_rgb)

    # Fine-tune decoder_last only
    appraiser_last = appraiser.clone().to(device)
    run_inner_loop(
        appraiser_last, pred_rgb, "decoder_last",
        inner_loop_steps, inner_loop_lr, retain_graph=True,
        reg_type=reg_type, reg_weight=reg_weight,
        original_params_dict=original_params_dict, fisher_dict=fisher_dict,
    )
    L_last_final = appraiser_last.compute_reconstruction_loss(pred_rgb)

    # Calculate layer-difference reward
    reward, layer_diff, complexity, _ = calculate_layer_difference_reward(
        L_first_final, L_last_final, final_loss_temp,
        ideal_initial_loss, initial_loss_sigma, L_initial
    )

    return reward, L_initial, L_first_final, L_last_final, layer_diff, complexity, appraiser_first, appraiser_last


def compute_layer_diff_reward_meta(
    appraiser: Appraiser,
    pred_rgb: torch.Tensor,
    inner_loop_steps: int,
    inner_loop_lr: float,
    final_loss_temp: float,
    ideal_initial_loss: float = 0.15,
    initial_loss_sigma: float = 0.2,
    reg_type: Optional[str] = None,
    reg_weight: float = 0.0,
    original_params_dict: Optional[Dict[str, torch.Tensor]] = None,
    fisher_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute layer-difference reward using meta-gradients (MAML-style).

    Both L_first and L_last are computed with differentiable inner loops,
    so the gradient captures how the image affects what each layer learns.

    Returns:
        (reward, L_initial, L_first_final, L_last_final, layer_diff, complexity)
    """
    # Get initial loss (before any fine-tuning) - needs grad for meta
    L_initial = appraiser.compute_reconstruction_loss(pred_rgb)

    # Run differentiable inner loop for decoder_first
    L_first_final, _ = run_inner_loop_meta(
        appraiser, pred_rgb, "decoder_first",
        inner_loop_steps, inner_loop_lr,
        reg_type=reg_type, reg_weight=reg_weight,
        original_params_dict=original_params_dict, fisher_dict=fisher_dict,
    )

    # Run differentiable inner loop for decoder_last
    L_last_final, _ = run_inner_loop_meta(
        appraiser, pred_rgb, "decoder_last",
        inner_loop_steps, inner_loop_lr,
        reg_type=reg_type, reg_weight=reg_weight,
        original_params_dict=original_params_dict, fisher_dict=fisher_dict,
    )

    # Calculate layer-difference reward
    reward, layer_diff, complexity, _ = calculate_layer_difference_reward(
        L_first_final, L_last_final, final_loss_temp,
        ideal_initial_loss, initial_loss_sigma, L_initial
    )

    return reward, L_initial, L_first_final, L_last_final, layer_diff, complexity
