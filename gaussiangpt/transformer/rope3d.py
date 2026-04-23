"""3D Rotary Positional Embedding (3D RoPE) for GaussianGPT.

Extends standard 1D RoPE to 3D spatial coordinates + token-type dimension.
Each attention head dimension is split into 4 groups: x, y, z, token_type.
The attention score becomes a function of relative spatial offset.

Reference: RoFormer (Su et al., 2021), extended to 3D following GaussianGPT paper.
"""
import torch
import torch.nn as nn
import math
from typing import Tuple


def build_rope_freqs(head_dim: int, n_dims: int = 4) -> torch.Tensor:
    """Build inverse frequency tensor for RoPE.

    Args:
        head_dim: dimension of each attention head (must be divisible by 2*n_dims)
        n_dims: number of spatial dimensions (4 for x,y,z,token_type)
    Returns:
        freqs: (head_dim // (2*n_dims),) inverse frequencies per group
    """
    assert head_dim % (2 * n_dims) == 0, f"head_dim {head_dim} must be divisible by {2*n_dims}"
    dim_per_group = head_dim // (2 * n_dims)
    # Standard RoPE frequencies: theta_i = 1 / (10000^(2i/d))
    freqs = 1.0 / (10000 ** (torch.arange(0, dim_per_group, dtype=torch.float32) / dim_per_group))
    return freqs  # (dim_per_group,)


def apply_rope_1d(x: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply 1D RoPE rotation to a slice of the head dimension.

    Standard RoPE splits the dim into two halves [x1 | x2] (not interleaved).

    Args:
        x: (..., seq_len, dim_per_group * 2) — the slice to rotate
        freqs: (dim_per_group,) inverse frequencies
        positions: (..., seq_len) integer positions
    Returns:
        rotated x with same shape
    """
    d = x.shape[-1] // 2
    # Compute angles: (..., seq_len, dim_per_group)
    angles = positions.unsqueeze(-1).float() * freqs.to(x.device)
    cos = angles.cos()
    sin = angles.sin()

    # Split into two halves (standard RoPE convention)
    x1, x2 = x[..., :d], x[..., d:]  # (..., seq_len, d) each

    # Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPE3D(nn.Module):
    """3D Rotary Positional Embedding with token-type dimension.

    Splits head_dim into 4 equal groups for (x, y, z, token_type).
    Applied to both queries and keys before dot-product attention.
    """

    def __init__(self, head_dim: int, n_spatial_dims: int = 3):
        super().__init__()
        # 4 groups: x, y, z, token_type
        self.n_groups = n_spatial_dims + 1
        self.head_dim = head_dim
        freqs = build_rope_freqs(head_dim, n_dims=self.n_groups)
        self.register_buffer("freqs", freqs)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords: torch.Tensor,
        token_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (batch, n_heads, seq_len, head_dim)
            k: (batch, n_heads, seq_len, head_dim)
            coords: (batch, seq_len, 3) integer voxel coordinates (x, y, z)
            token_type: (batch, seq_len) integer token type (0=position, 1=feature)
        Returns:
            q_rot, k_rot: rotated queries and keys
        """
        dim_per_group = self.head_dim // (2 * self.n_groups)
        group_size = dim_per_group * 2

        # Positions for each group: (batch, seq_len)
        positions = [coords[..., 0], coords[..., 1], coords[..., 2], token_type]

        q_parts, k_parts = [], []
        for i, pos in enumerate(positions):
            start = i * group_size
            end = start + group_size
            q_slice = q[..., start:end]  # (batch, n_heads, seq_len, group_size)
            k_slice = k[..., start:end]

            # pos: (batch, seq_len) -> broadcast over n_heads
            pos_expanded = pos.unsqueeze(1)  # (batch, 1, seq_len)

            q_parts.append(apply_rope_1d(q_slice, self.freqs, pos_expanded))
            k_parts.append(apply_rope_1d(k_slice, self.freqs, pos_expanded))

        q_rot = torch.cat(q_parts, dim=-1)
        k_rot = torch.cat(k_parts, dim=-1)
        return q_rot, k_rot
