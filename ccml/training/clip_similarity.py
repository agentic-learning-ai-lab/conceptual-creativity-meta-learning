"""
CLIP-based semantic similarity for creativity guidance.

Provides a reward term that keeps generated images semantically related
to the target class/prompt, preventing drift towards abstract patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from PIL import Image


class CLIPSimilarity(nn.Module):
    """
    Computes CLIP similarity between images and text prompts.

    Used as a semantic regularization term in creativity guidance:
    - Prevents text embeddings from drifting to meaningless values
    - Keeps generated images related to the target class

    Usage:
        clip_sim = CLIPSimilarity(device="cuda")
        similarity = clip_sim(images, "a photo of a golden retriever")
        reward = base_reward + clip_weight * similarity
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device

        # Freeze CLIP
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Cache for text embeddings (avoid recomputing)
        self._text_cache = {}

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Get CLIP text embedding (cached)."""
        if text not in self._text_cache:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "pixel_values"}
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = F.normalize(text_features, dim=-1)
            self._text_cache[text] = text_features
        return self._text_cache[text]

    def get_image_embedding(
        self,
        images: torch.Tensor,
        with_grad: bool = True,
    ) -> torch.Tensor:
        """
        Get CLIP image embedding.

        Args:
            images: Tensor of shape (B, C, H, W) in [0, 1] range
            with_grad: Whether to allow gradients through vision encoder
        """
        # Preprocess for CLIP
        images = self._preprocess_tensor(images)

        if with_grad:
            image_features = self.model.get_image_features(pixel_values=images)
        else:
            with torch.no_grad():
                image_features = self.model.get_image_features(pixel_values=images)

        return F.normalize(image_features, dim=-1)

    def _preprocess_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess tensor images for CLIP."""
        # CLIP expects 224x224 images
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize with CLIP stats
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std

        return images

    def forward(
        self,
        images: torch.Tensor,
        text: str,
        with_grad: bool = True,
    ) -> torch.Tensor:
        """
        Compute CLIP similarity between images and text.

        Args:
            images: Tensor of shape (B, C, H, W) in [0, 1] range
            text: Target text prompt (e.g., "a photo of a golden retriever")
            with_grad: Whether to allow gradients through vision encoder

        Returns:
            Scalar tensor with mean similarity (higher = more similar)
        """
        text_features = self.get_text_embedding(text)
        image_features = self.get_image_embedding(images, with_grad=with_grad)

        # Cosine similarity (both are normalized)
        similarity = (image_features @ text_features.T).mean()

        return similarity

    def compute_similarity_batch(
        self,
        images: torch.Tensor,
        texts: List[str],
        with_grad: bool = True,
    ) -> torch.Tensor:
        """
        Compute pairwise similarities for batch of images and texts.

        Args:
            images: Tensor of shape (B, C, H, W)
            texts: List of B text prompts
            with_grad: Whether to allow gradients

        Returns:
            Tensor of shape (B,) with per-image similarities
        """
        image_features = self.get_image_embedding(images, with_grad=with_grad)

        similarities = []
        for i, text in enumerate(texts):
            text_features = self.get_text_embedding(text)
            sim = (image_features[i:i+1] @ text_features.T).squeeze()
            similarities.append(sim)

        return torch.stack(similarities)
