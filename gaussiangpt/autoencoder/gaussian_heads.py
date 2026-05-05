"""Per-attribute Gaussian encoder/decoder heads.

Strict implementation of GaussianGPT (von Lützow et al., 2026), Sec. 3.1
and Appendix C.

Encoder head (per attribute):
    Linear(dim -> dim*16) -> ResidualMLP(dim*16)

Decoder head (per attribute):
    [Linear(in_dim -> 64)] -> ResidualMLP(64) -> ResidualMLP(64) -> Linear(64 -> dim)
    (the leading Linear is a small structural addition because the CNN's
     output channel count differs from 64; the two ResidualMLPs and the
     zero-init final projection are exactly as described in the paper.)

Feature representations (Appendix C, "Feature Representations"):
    * scales:     world-space size, predicted via *softplus* activation
    * opacities:  logit space, clamped to [-10, 10]
    * colors:     clamped to [0, 1]
    * rotations:  standardized (unit) quaternions
    * offsets:    UNBOUNDED world-space values
    * inputs are linearly scaled to similar magnitudes for training stability

Note: this differs from L3DG, which uses tanh-bounded offsets and
exp/log scale. GaussianGPT explicitly removes both restrictions.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


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

    Architecture: Linear(dim -> dim*expand) -> ResidualMLP(dim*expand)
    """

    def __init__(self, in_dim: int, expand: int = 16):
        super().__init__()
        hidden = in_dim * expand
        self.expand = nn.Linear(in_dim, hidden)
        self.residual = ResidualMLP(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.expand(x))


class GaussianDecoderHead(nn.Module):
    """Decodes a feature vector back to a single Gaussian attribute.

    Architecture: Linear(in_dim -> hidden) -> ResidualMLP(hidden) ->
                  ResidualMLP(hidden) -> Linear(hidden -> out_dim).

    The final projection is zero-initialized; biases are chosen so that
    the *post-processed* output starts at a sensible default (see
    `GaussianAttributeDecoder` for the per-attribute bias values).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 64,
        attr_name: str = "",
        init_bias: float = 0.0,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden)
        self.res1 = ResidualMLP(hidden)
        self.res2 = ResidualMLP(hidden)
        self.proj_out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.proj_out.weight)
        with torch.no_grad():
            if attr_name == "rotation" and out_dim == 4:
                # Identity quaternion (w=1, x=y=z=0); after F.normalize -> [1,0,0,0]
                self.proj_out.bias.fill_(0.0)
                self.proj_out.bias[0] = 1.0
            else:
                self.proj_out.bias.fill_(float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.proj_out(x)


# Per-attribute dimensionalities.
ATTR_DIMS = {
    "offset": 3,
    "scale": 3,
    "opacity": 1,
    "rotation": 4,
    "color": 3,
}

# Optional: first-order spherical harmonics (degree 1 -> 4 coeffs per channel x 3).
SH_DIM = 4 * 3


def _softplus_inverse(y: float) -> float:
    """Return x such that softplus(x) = y, for y > 0.

    Used to set the scale-head bias so the initial *post-softplus* scale
    is approximately the voxel size (a sensible starting size for a
    Gaussian centred in a voxel).
    """
    if y <= 0:
        raise ValueError(f"softplus_inverse requires y > 0, got {y}")
    # Numerically stable: softplus^{-1}(y) = log(exp(y) - 1) = y + log(1 - exp(-y))
    return y + math.log1p(-math.exp(-y))


class GaussianAttributeEncoder(nn.Module):
    """Encodes all Gaussian attributes into a concatenated feature vector.

    Per Appendix C, inputs are linearly scaled to bring the different
    attributes (offsets ~ a few cm, opacity-logits ~ ±10, colors ~ [0, 1],
    etc.) to comparable magnitudes before the per-attribute encoder heads.
    """

    def __init__(
        self,
        use_sh: bool = False,
        expand: int = 16,
        voxel_size: float = 0.025,
        opacity_scale: float = 0.1,
    ):
        super().__init__()
        self.use_sh = use_sh
        self.voxel_size = float(voxel_size)
        self.opacity_scale = float(opacity_scale)

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
            gaussians: dict of attribute tensors, each (N, attr_dim).
                Expected representations (matching Appendix C):
                  * offset   : world-space offsets (any magnitude)
                  * scale    : world-space positive sizes
                  * opacity  : logit space (any magnitude; will be clamped)
                  * rotation : raw quaternions (will be unit-normalised)
                  * color    : RGB in [0, 1]
        Returns:
            features: (N, out_dim)
        """
        parts = []
        for name, head in self.heads.items():
            parts.append(head(self._preprocess(name, gaussians[name])))
        return torch.cat(parts, dim=-1)

    def _preprocess(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Linear, paper-faithful normalisation to balance attribute magnitudes.

        We deliberately avoid non-linear transforms (no log on scale, no
        sigmoid on opacity) so the encoder sees the same representations
        the decoder predicts, keeping the auto-encoder roundtrip simple.
        """
        if name == "offset":
            # World-space offsets: typically O(voxel_size). Bring to ~unit.
            return x / self.voxel_size
        if name == "scale":
            # World-space positive sizes: typically O(voxel_size). Bring to ~unit.
            return x / self.voxel_size
        if name == "opacity":
            # Cap a few extreme outliers, then bring logit range ~[-1, 1].
            return x.clamp(-10.0, 10.0) * self.opacity_scale
        if name == "rotation":
            # Standardised (unit) quaternions, per Appendix C.
            return F.normalize(x, dim=-1)
        if name == "color":
            # Colours are in [0, 1] (Appendix C). Range magnitude ~0.5 is
            # already comparable to the other attributes after the scalings
            # above, so we only enforce the clamp without further shifting.
            return x.clamp(0.0, 1.0)
        # SH coefficients are already small; pass through.
        return x


class GaussianAttributeDecoder(nn.Module):
    """Decodes a feature vector back to all Gaussian attributes.

    Output activations follow Appendix C strictly:
      * offset   -> identity (unbounded world-space values)
      * scale    -> softplus  (positive world-space sizes)
      * opacity  -> clamp(-10, 10)  (logit space)
      * rotation -> F.normalize  (unit quaternion)
      * color    -> clamp(0, 1) by default; can be switched to ``sigmoid``
                    via ``color_activation='sigmoid'`` for diagnostic
                    purposes (the hard clamp kills gradients for any
                    prediction outside [0, 1]).
    """

    def __init__(
        self,
        in_dim: int,
        use_sh: bool = False,
        hidden: int = 64,
        voxel_size: float = 0.025,
        color_activation: str = "clamp",
    ):
        super().__init__()
        self.use_sh = use_sh
        self.voxel_size = float(voxel_size)
        self.color_activation = (color_activation or "clamp").lower()
        if self.color_activation not in ("clamp", "sigmoid"):
            raise ValueError(
                f"color_activation must be 'clamp' or 'sigmoid', got {color_activation!r}"
            )
        attrs = dict(ATTR_DIMS)
        if use_sh:
            attrs["sh"] = SH_DIM

        # Initial-bias presets: chosen so the *post-processed* output
        # starts at a reasonable default and the network has good initial
        # visibility (per Appendix C).
        # Note: when ``color_activation='sigmoid'``, bias=0.0 gives
        # sigmoid(0)=0.5 (mid grey), matching the clamp-mode init.
        color_bias = 0.5 if self.color_activation == "clamp" else 0.0
        init_biases = {
            "offset": 0.0,                                  # identity -> centred at voxel
            "scale": _softplus_inverse(self.voxel_size),    # softplus(bias) ~= voxel_size
            "opacity": 0.0,                                 # logit 0 -> sigmoid 0.5 occupancy
            "color": color_bias,
            "sh": 0.0,
            # rotation handled inside the head (identity quaternion).
        }

        self.heads = nn.ModuleDict({
            name: GaussianDecoderHead(
                in_dim, dim, hidden, attr_name=name,
                init_bias=init_biases.get(name, 0.0),
            )
            for name, dim in attrs.items()
        })

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (N, in_dim)
        Returns:
            dict of decoded attribute tensors, each in its natural range
            (see header).
        """
        out = {}
        for name, head in self.heads.items():
            out[name] = self._postprocess(name, head(features))
        return out

    def _postprocess(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name == "offset":
            # GaussianGPT: offsets are unbounded world-space values
            # (Appendix C). No tanh / no voxel-size clamp.
            return x
        if name == "scale":
            # World-space positive sizes via softplus (Appendix C).
            return F.softplus(x)
        if name == "opacity":
            return x.clamp(-10.0, 10.0)
        if name == "rotation":
            return F.normalize(x, dim=-1)
        if name == "color":
            if self.color_activation == "sigmoid":
                return torch.sigmoid(x)
            return x.clamp(0.0, 1.0)
        return x
