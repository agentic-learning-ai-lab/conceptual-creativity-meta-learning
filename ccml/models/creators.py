"""Creator models (diffusion models) for CCML."""

from typing import Optional

import torch
from diffusers import DDPMPipeline, DDPMScheduler


class DiffusionCreator:
    """
    Wrapper around diffusion models for creative generation.

    Args:
        model_name: HuggingFace model name or path
        device: Device to run on
    """

    def __init__(
        self,
        model_name: str = "google/ddpm-cifar10-32",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self.pipeline = DDPMPipeline.from_pretrained(model_name)
        self.pipeline.to(device)

    @property
    def unet(self):
        return self.pipeline.unet

    @property
    def scheduler(self):
        return self.pipeline.scheduler

    def get_scheduler(self, num_steps: int = 1000) -> DDPMScheduler:
        """Get a scheduler configured for the specified number of steps."""
        scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        scheduler.set_timesteps(num_steps)
        scheduler.timesteps = scheduler.timesteps.to(self.device)
        return scheduler

    def generate_unguided(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
    ):
        """Generate images without guidance."""
        return self.pipeline(
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
        ).images

    def get_img_size(self) -> int:
        """Get the image size this model generates."""
        # Infer from model name or config
        if "mnist" in self.model_name.lower():
            return 32
        elif "cifar10-32" in self.model_name.lower():
            return 32
        elif "imagenet64" in self.model_name.lower():
            return 64
        else:
            # Default
            return 32

    def get_channels(self) -> int:
        """Get the number of channels this model generates."""
        if "mnist" in self.model_name.lower():
            return 3  # DDPM outputs RGB even for MNIST
        else:
            return 3


# Pre-configured creators for common experiments
MNIST_CREATOR = "dimpo/ddpm-mnist"


def create_mnist_creator(device: str = "cuda") -> DiffusionCreator:
    """Create a creator for MNIST experiments."""
    return DiffusionCreator(model_name=MNIST_CREATOR, device=device)


