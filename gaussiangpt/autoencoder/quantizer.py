"""Lookup-Free Quantization (LFQ) for discrete latent codes.

Based on: Magvit-v2 / LFQ concept — discretize by sign of encoder output.
Codebook indices are binary codes (0/1 per dimension), giving 2^d codes.
For codebook_size=4096, we need d=12 bits.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LookupFreeQuantizer(nn.Module):
    """Lookup-Free Quantization (LFQ).

    The encoder output z is discretized to {-1, +1} per dimension based on sign.
    Codebook indices are the binary representation of the sign pattern.
    Entropy loss encourages uniform codebook usage.
    """

    def __init__(self, codebook_size: int = 4096):
        super().__init__()
        # codebook_size must be a power of 2
        assert (codebook_size & (codebook_size - 1)) == 0, "codebook_size must be power of 2"
        self.num_bits = int(math.log2(codebook_size))
        self.codebook_size = codebook_size

        # Register bit masks for index <-> binary conversion
        bits = torch.arange(self.num_bits)
        self.register_buffer("bit_masks", 2 ** bits)  # (num_bits,)

    @property
    def latent_dim(self) -> int:
        return self.num_bits

    def encode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (..., num_bits) continuous encoder output
        Returns:
            z_q: (..., num_bits) quantized in {-1, +1}
            indices: (...,) integer codebook indices
        """
        # Straight-through: quantize by sign
        z_q = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # Convert binary {0,1} to integer indices
        bits = (z_q > 0).long()  # (..., num_bits)
        indices = (bits * self.bit_masks).sum(dim=-1)  # (...,)

        return z_q_st, indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (...,) integer codebook indices
        Returns:
            z_q: (..., num_bits) quantized values in {-1, +1}
        """
        bits = (indices.unsqueeze(-1) & self.bit_masks) > 0  # (..., num_bits)
        return bits.float() * 2 - 1  # map {0,1} -> {-1,+1}

    def entropy_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Entropy loss to encourage uniform codebook usage.

        Maximizes entropy of the per-bit marginal distributions.
        L_LFQ = -H(bits) = sum_i [p_i * log(p_i) + (1-p_i) * log(1-p_i)]
        """
        # Per-bit probability of being +1
        p = torch.sigmoid(z)  # soft approximation
        # Binary entropy per bit, averaged
        eps = 1e-6
        entropy = -(p * (p + eps).log() + (1 - p) * (1 - p + eps).log())
        return -entropy.mean()  # negative because we want to maximize entropy

    def forward(self, z: torch.Tensor):
        z_q, indices = self.encode(z)
        loss = self.entropy_loss(z)
        return z_q, indices, loss
