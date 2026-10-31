"""Minimal LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning."""

import math
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a Linear layer.

    output = original(x) + (x @ A @ B) * scaling
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = None):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank

        # Freeze original
        for p in original.parameters():
            p.requires_grad = False

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(original.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))

        # Init: A with kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


