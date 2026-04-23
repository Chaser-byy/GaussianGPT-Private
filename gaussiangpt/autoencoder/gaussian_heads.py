"""Gaussian attribute encoder/decoder heads and feature representations."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ResidualMLP(nn.Module):
    """Residual MLP block with fixed channel width."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GaussianEncoderHead(nn.Module):
    """Encodes a single Gaussian attribute into a feature embedding.

    Architecture: Linear(dim -> dim*16) -> ResidualMLP(dim*16)
    """

    def __init__(self, in_dim: int, expand: int = 16):
        super().__init__()
        hidden = in_dim * expand
        self.expand = nn.Linear(in_dim, hidden)
        self.residual = ResidualMLP(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.expand(x))


class GaussianDecoderHead(nn.Module):
    """Decodes a feature vector back to a Gaussian attribute.

    Architecture: Linear(in_dim -> 64) -> ResidualMLP(64) -> ResidualMLP(64) -> Linear(64 -> out_dim)
    Zero-initialized final projection weights; biases set for reasonable initial visibility.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, attr_name: str = ""):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden)
        self.res1 = ResidualMLP(hidden)
        self.res2 = ResidualMLP(hidden)
        self.proj_out = nn.Linear(hidden, out_dim)
        # Zero-init weights; set biases for reasonable initial visibility (paper Appendix C)
        nn.init.zeros_(self.proj_out.weight)
        with torch.no_grad():
            if attr_name == "opacity":
                # Start with moderate opacity (logit ~0 -> sigmoid ~0.5)
                self.proj_out.bias.fill_(0.0)
            elif attr_name == "scale":
                # Start with small scale (softplus(0) = log(2) ≈ 0.693)
                self.proj_out.bias.fill_(0.0)
            elif attr_name == "color":
                # Start with mid-gray (0.5 after clamp)
                self.proj_out.bias.fill_(0.5)
            elif attr_name == "rotation":
                # Start with identity quaternion (w=1, x=y=z=0)
                if out_dim == 4:
                    self.proj_out.bias.fill_(0.0)
                    self.proj_out.bias[0] = 1.0  # w component
            # offset and sh: zero bias is fine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.proj_out(x)


# Gaussian attribute dimensions
# position offset (3), scale (3), opacity (1), rotation quaternion (4), color RGB (3)
ATTR_DIMS = {
    "offset": 3,
    "scale": 3,
    "opacity": 1,
    "rotation": 4,
    "color": 3,
}

# Optional: spherical harmonics (degree 1 = 4 coefficients per channel × 3 channels = 12)
# degree d SH has (d+1)^2 coefficients; degree 1 -> 4 per channel
SH_DIM = 4 * 3  # first-order SH, 3 color channels


class GaussianAttributeEncoder(nn.Module):
    """Encodes all Gaussian attributes into a concatenated feature vector."""

    def __init__(self, use_sh: bool = False, expand: int = 16):
        super().__init__()
        self.use_sh = use_sh
        attrs = dict(ATTR_DIMS)
        if use_sh:
            attrs["sh"] = SH_DIM

        self.heads = nn.ModuleDict(
            {name: GaussianEncoderHead(dim, expand) for name, dim in attrs.items()}
        )
        self.out_dim = sum(dim * expand for dim in attrs.values())

    def forward(self, gaussians: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            gaussians: dict of attribute tensors, each (N, attr_dim)
        Returns:
            features: (N, out_dim)
        """
        parts = []
        for name, head in self.heads.items():
            attr = gaussians[name]
            attr = self._preprocess(name, attr)
            parts.append(head(attr))
        return torch.cat(parts, dim=-1)

    def _preprocess(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Normalize inputs to similar magnitudes to stabilize training (paper Appendix C)."""
        if name == "rotation":
            # Standardize quaternion to unit norm
            return F.normalize(x, dim=-1)
        # Offsets, scales, opacities, colors: pass through as-is
        # (the autoencoder learns to handle the raw representation)
        return x


class GaussianAttributeDecoder(nn.Module):
    """Decodes a feature vector back to all Gaussian attributes."""

    def __init__(self, in_dim: int, use_sh: bool = False, hidden: int = 64):
        super().__init__()
        self.use_sh = use_sh
        attrs = dict(ATTR_DIMS)
        if use_sh:
            attrs["sh"] = SH_DIM

        self.heads = nn.ModuleDict(
            {name: GaussianDecoderHead(in_dim, dim, hidden, attr_name=name) for name, dim in attrs.items()}
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (N, in_dim)
        Returns:
            dict of decoded attribute tensors
        """
        out = {}
        for name, head in self.heads.items():
            raw = head(features)
            out[name] = self._postprocess(name, raw)
        return out

    def _postprocess(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Apply output activations matching the paper's feature representations (Appendix C).

        - scales: softplus (world-space size, always positive)
        - opacity: clamp to [-10, 10] (stored in logit space)
        - rotation: normalize to unit quaternion
        - color: clamp to [0, 1]
        - offset: unbounded world-space values (no activation)
        """
        if name == "scale":
            return F.softplus(x)
        elif name == "opacity":
            return x.clamp(-10, 10)  # logit space, sigmoid applied at render time
        elif name == "rotation":
            return F.normalize(x, dim=-1)
        elif name == "color":
            return x.clamp(0, 1)
        return x  # offset: unbounded
