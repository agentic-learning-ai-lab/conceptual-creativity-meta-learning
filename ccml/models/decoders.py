"""Decoder models for CCML Appraiser."""

import torch
import torch.nn as nn


class MNISTDecoder(nn.Module):
    """
    Deconvolutional network to reconstruct MNIST images from features.
    Outputs 32x32 grayscale images.
    """

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        self.main = nn.Sequential(
            nn.Linear(feature_dim, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.Tanh(),  # Match [-1, 1] normalization
        )

    def forward(self, x):
        return self.main(x)


class ResNetDecoder(nn.Module):
    """
    Decoder that reconstructs features or pixels from a bottleneck layer.
    Uses transposed convolutions for upsampling.

    Args:
        layer4_channels: Number of channels in layer4 features
        layer3_channels: Number of channels in layer3 features
        layer2_channels: Number of channels in layer2 features
        reconstruction_target: What to reconstruct ("layer2", "layer3", or "pixels")
        reconstruction_from: Which layer to use as input ("layer3" or "layer4")
        img_size: Original image size
        bottleneck_spatial_size: Spatial size of the bottleneck layer
        target_spatial_size: Spatial size of the target layer
    """

    def __init__(
        self,
        layer4_channels: int = 2048,
        layer3_channels: int = 1024,
        layer2_channels: int = 512,
        reconstruction_target: str = "layer2",
        reconstruction_from: str = "layer4",
        img_size: int = 32,
        bottleneck_spatial_size: int = None,
        target_spatial_size: int = None,
    ):
        super().__init__()

        self.layer4_channels = layer4_channels
        self.layer3_channels = layer3_channels
        self.layer2_channels = layer2_channels
        self.reconstruction_target = reconstruction_target
        self.reconstruction_from = reconstruction_from
        self.img_size = img_size
        self.bottleneck_spatial_size = bottleneck_spatial_size
        self.target_spatial_size = target_spatial_size

        # Map layer names to their properties
        self.layer_info = {
            "layer4": {"channels": layer4_channels, "level": 4},
            "layer3": {"channels": layer3_channels, "level": 3},
            "layer2": {"channels": layer2_channels, "level": 2},
            "pixels": {"channels": 3, "level": 0},
        }

        # Validate configuration
        from_level = self.layer_info[reconstruction_from]["level"]
        to_level = self.layer_info[reconstruction_target]["level"]

        if from_level <= to_level and reconstruction_target != "pixels":
            raise ValueError(
                f"Cannot reconstruct {reconstruction_target} from {reconstruction_from}: "
                f"source must be deeper than target"
            )

        # Build decoder
        self.input_channels = self.layer_info[reconstruction_from]["channels"]
        self.output_channels = self.layer_info[reconstruction_target]["channels"]
        self.decoder = self._build_decoder()

    def _build_decoder(self):
        """Build decoder that upsamples from bottleneck to target."""
        import math

        # Calculate spatial dimensions
        if self.bottleneck_spatial_size is not None:
            bottleneck_size = self.bottleneck_spatial_size
        else:
            # Fallback: estimate based on img_size
            if self.reconstruction_from == "layer4":
                bottleneck_size = self.img_size // 32
            elif self.reconstruction_from == "layer3":
                bottleneck_size = self.img_size // 16
            else:
                bottleneck_size = self.img_size // 8

        # Determine target spatial size
        if self.target_spatial_size is not None:
            target_size = self.target_spatial_size
        else:
            if self.reconstruction_target == "pixels":
                target_size = self.img_size
            elif self.reconstruction_target == "layer2":
                target_size = self.img_size // 8
            elif self.reconstruction_target == "layer3":
                target_size = self.img_size // 16
            else:
                target_size = self.img_size // 32

        # Calculate number of 2x upsampling stages
        num_stages = int(math.log2(target_size / bottleneck_size))

        # Build channel progression
        channel_progression = self._get_channel_progression(num_stages)

        # Build the decoder layers
        layers = []
        in_channels = self.input_channels

        for i, out_channels in enumerate(channel_progression):
            layers.extend([
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels

        # Final output layer
        if self.reconstruction_target == "pixels":
            layers.append(nn.Conv2d(in_channels, 3, kernel_size=3, padding=1))
        else:
            layers.extend([
                nn.Conv2d(in_channels, self.output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, padding=1),
            ])

        return nn.Sequential(*layers)

    def _get_channel_progression(self, num_stages: int):
        """Determine channel counts for each upsampling stage."""
        if num_stages == 0:
            return []

        channel_progression = []
        for i in range(num_stages):
            if self.reconstruction_from == "layer4":
                if i == 0:
                    channels = self.layer3_channels
                elif i == 1:
                    channels = self.layer2_channels
                elif i < num_stages - 1:
                    channels = max(64, 256 // (2 ** (i - 1)))
                else:
                    channels = 64
            elif self.reconstruction_from == "layer3":
                if i == 0:
                    channels = self.layer2_channels
                elif i < num_stages - 1:
                    channels = max(64, 256 // (2**i))
                else:
                    channels = 64
            else:
                channels = max(64, self.input_channels // (2 ** (i + 1)))

            channel_progression.append(channels)

        return channel_progression

    def forward(self, bottleneck_features):
        """Decode bottleneck features to target representation."""
        return self.decoder(bottleneck_features)


