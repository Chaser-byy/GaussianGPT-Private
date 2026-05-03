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

    def entropy_loss(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """LFQ entropy loss in the Magvit-v2 form.

        L_LFQ = E_x[H(p_x)] - H(E_x[p_x])
              = (mean per-sample bit entropy)  -  (marginal bit entropy)

        Minimising the first term pushes each sample's per-bit probability
        towards 0 or 1 (i.e. the encoder commits confidently to a code),
        while maximising the second term keeps the marginal usage close to
        50/50 per bit, encouraging full codebook coverage.

        The previous implementation only used `-mean(H(p))`, which actually
        drives p -> 0.5 for every sample (i.e. the encoder output collapses
        to ~0 and quantisation becomes random).
        """
        # Sharper sigmoid so gradients reflect quantisation by sign.
        p = torch.sigmoid(z * (2.0 / max(temperature, 1e-6)))  # (..., num_bits)
        eps = 1e-6

        # Per-sample binary entropy of each bit, averaged over samples.
        H_per = -(p * (p + eps).log() + (1.0 - p) * (1.0 - p + eps).log())
        per_sample_term = H_per.mean()

        # Marginal entropy of each bit across the batch (flatten leading dims).
        p_marginal = p.reshape(-1, p.shape[-1]).mean(dim=0)  # (num_bits,)
        H_marg = -(p_marginal * (p_marginal + eps).log() +
                   (1.0 - p_marginal) * (1.0 - p_marginal + eps).log())
        marginal_term = H_marg.mean()

        return per_sample_term - marginal_term

    def forward(self, z: torch.Tensor):
        z_q, indices = self.encode(z)
        loss = self.entropy_loss(z)
        return z_q, indices, loss
