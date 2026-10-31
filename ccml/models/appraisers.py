"""Appraiser: Encoder + Decoder combination for learnability assessment."""

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from ccml.models.encoders import MNISTEncoder, ResNetEncoder
from ccml.models.decoders import MNISTDecoder, ResNetDecoder


class Appraiser(nn.Module):
    """
    Appraiser model that combines an encoder and decoder for learnability assessment.

    The appraiser measures how "learnable" a generated sample is by:
    1. Computing initial reconstruction loss (before fine-tuning)
    2. Fine-tuning on the sample for T steps
    3. Computing final reconstruction loss (after fine-tuning)

    Args:
        encoder: Encoder model
        decoder: Decoder model
        reconstruction_target: What to reconstruct ("pixels", "layer2", "layer3")
        reconstruction_from: Which layer to use as bottleneck ("layer3", "layer4")
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reconstruction_target: str = "pixels",
        reconstruction_from: str = "layer4",
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_target = reconstruction_target
        self.reconstruction_from = reconstruction_from

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.

        Returns:
            (features, reconstruction)
        """
        if isinstance(self.encoder, MNISTEncoder):
            features, _ = self.encoder(x)
            reconstruction = self.decoder(features)
            return features, reconstruction
        elif isinstance(self.encoder, ResNetEncoder):
            layer2, layer3, layer4 = self.encoder(x, return_intermediates=True)

            # Select bottleneck
            if self.reconstruction_from == "layer4":
                bottleneck = layer4
            elif self.reconstruction_from == "layer3":
                bottleneck = layer3
            else:
                raise ValueError(f"Unknown reconstruction_from: {self.reconstruction_from}")

            reconstruction = self.decoder(bottleneck)
            return bottleneck, reconstruction
        else:
            raise ValueError(f"Unknown encoder type: {type(self.encoder)}")

    def get_target(self, x) -> torch.Tensor:
        """Get the reconstruction target for input x."""
        if self.reconstruction_target == "pixels":
            return x
        elif isinstance(self.encoder, ResNetEncoder):
            layer2, layer3, layer4 = self.encoder(x, return_intermediates=True)
            if self.reconstruction_target == "layer2":
                return layer2
            elif self.reconstruction_target == "layer3":
                return layer3
            else:
                raise ValueError(f"Unknown target: {self.reconstruction_target}")
        else:
            # MNIST encoder always reconstructs pixels
            return x

    def compute_reconstruction_loss(self, x, criterion: nn.Module = None) -> torch.Tensor:
        """Compute reconstruction loss for input x."""
        if criterion is None:
            criterion = nn.MSELoss()

        _, reconstruction = self.forward(x)
        target = self.get_target(x)
        return criterion(reconstruction, target)

    def clone(self) -> "Appraiser":
        """Create a deep copy of this appraiser for fine-tuning."""
        return copy.deepcopy(self)

    def get_finetune_param_names(self, finetune_spec: str) -> List[str]:
        """
        Get names of parameters to fine-tune based on a specification string.

        Args:
            finetune_spec: Same as get_finetune_params

        Returns:
            List of parameter names to fine-tune
        """
        params = self.get_finetune_params(finetune_spec)
        param_ids = {id(p) for p in params}
        return [name for name, p in self.named_parameters() if id(p) in param_ids]

    def get_finetune_params(self, finetune_spec: str) -> List[nn.Parameter]:
        """
        Get parameters to fine-tune based on a specification string.

        Args:
            finetune_spec: Comma-separated list of what to fine-tune:
                - "all": All parameters
                - "none": No parameters
                - "encoder": All encoder parameters
                - "decoder": All decoder parameters
                - "encoder_first": First conv layer of encoder
                - "encoder_last": Last layer of encoder
                - "decoder_first": First layer of decoder
                - "decoder_last": Last layer of decoder

        Returns:
            List of parameters to fine-tune
        """
        params = []
        specs = finetune_spec.split(",")

        if "none" in specs:
            return []

        if "all" in specs:
            return list(self.parameters())

        added_encoder_all = "encoder" in specs
        added_decoder_all = "decoder" in specs

        if added_encoder_all:
            params.extend(self.encoder.parameters())
        if added_decoder_all:
            params.extend(self.decoder.parameters())

        # Fine-grained control for encoder
        if not added_encoder_all:
            if "encoder_first" in specs:
                if isinstance(self.encoder, MNISTEncoder):
                    params.extend(self.encoder.main[0].parameters())
                elif isinstance(self.encoder, ResNetEncoder):
                    params.extend(self.encoder.conv1.parameters())

            if "encoder_last" in specs:
                if isinstance(self.encoder, MNISTEncoder):
                    params.extend(self.encoder.main[-1].parameters())
                elif isinstance(self.encoder, ResNetEncoder):
                    params.extend(self.encoder.layer4.parameters())

            if "encoder_layer4" in specs and isinstance(self.encoder, ResNetEncoder):
                params.extend(self.encoder.layer4.parameters())

            if "encoder_layer3" in specs and isinstance(self.encoder, ResNetEncoder):
                params.extend(self.encoder.layer3.parameters())

            if "encoder_layer2" in specs and isinstance(self.encoder, ResNetEncoder):
                params.extend(self.encoder.layer2.parameters())

        # Fine-grained control for decoder
        if not added_decoder_all:
            if "decoder_first" in specs:
                if isinstance(self.decoder, MNISTDecoder):
                    params.extend(self.decoder.main[0].parameters())
                elif isinstance(self.decoder, ResNetDecoder):
                    # Find first Conv2d
                    for module in self.decoder.decoder:
                        if isinstance(module, nn.Conv2d):
                            params.extend(module.parameters())
                            break

            if "decoder_last" in specs:
                if isinstance(self.decoder, MNISTDecoder):
                    params.extend(self.decoder.main[-2].parameters())
                elif isinstance(self.decoder, ResNetDecoder):
                    # Find last Conv2d
                    for module in reversed(list(self.decoder.decoder)):
                        if isinstance(module, nn.Conv2d):
                            params.extend(module.parameters())
                            break

        # Return unique parameters
        return list(dict.fromkeys(params))


