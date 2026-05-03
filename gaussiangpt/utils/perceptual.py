"""VGG19 perceptual loss matching L3DG (Sec. 3.2.2).

Quote from the paper:
    "ΦVGG is the vectorized concatenation of the first 5 feature layers
     before the max pooling operation of a VGG19 network, where each layer
     is normalized by the square root of the number of elements."

Implementation:
  * Use torchvision's pretrained VGG19 features.
  * Pick the conv-block outputs *before* each of the 5 max-pool layers (the
    standard "block1..block5 conv stack outputs").
  * Normalise each feature map by `1 / sqrt(numel)` so that L2 distance is
    comparable across layers (otherwise early high-resolution layers would
    completely dominate).
  * Inputs are RGB in [0, 1]; we apply the standard ImageNet normalisation
    expected by torchvision VGG.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet mean/std as used by torchvision pretrained models.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGPerceptualLoss(nn.Module):
    """L2 perceptual loss on the first five VGG19 conv-block outputs.

    The VGG backbone is frozen (eval mode, no grads on its parameters).
    Forward returns a scalar loss; the input tensors are expected to be
    `(B, 3, H, W)` in [0, 1]. The class is robust to small inputs (it
    pads to a minimum of 32x32 because tiny feature maps make the loss
    very noisy).
    """

    # In torchvision's VGG19, these indices correspond to the conv layers
    # that immediately precede each MaxPool (block1..block5):
    #   conv1_2 = idx 3,  conv2_2 = idx 8,  conv3_4 = idx 17,
    #   conv4_4 = idx 26, conv5_4 = idx 35.
    _LAYER_INDICES = (3, 8, 17, 26, 35)

    _MIN_SIZE = 32  # pad inputs to at least this many pixels per side

    def __init__(
        self,
        device: Optional[torch.device] = None,
        weights: str = "DEFAULT",
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        # Lazily import torchvision to avoid forcing the dep when perceptual
        # loss is disabled.
        try:
            from torchvision.models import vgg19, VGG19_Weights
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "torchvision is required for VGGPerceptualLoss"
            ) from e

        if weights_path is not None:
            model = vgg19(weights=None)
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {
                key.removeprefix("module."): value
                for key, value in state.items()
            }
            model.load_state_dict(state)
            backbone = model.features
        else:
            try:
                w = VGG19_Weights.DEFAULT if weights == "DEFAULT" else None
                backbone = vgg19(weights=w).features
            except Exception:
                # Older torchvision: fall back to deprecated `pretrained` flag.
                backbone = vgg19(pretrained=(weights is not None)).features

        # Keep only up to the deepest layer we need.
        max_idx = self._LAYER_INDICES[-1]
        self.features = backbone[: max_idx + 1].eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        if device is not None:
            self.to(device)

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self._MIN_SIZE or x.shape[-2] < self._MIN_SIZE:
            x = F.interpolate(
                x, size=(self._MIN_SIZE, self._MIN_SIZE),
                mode="bilinear", align_corners=False,
            )
        return (x - self._mean) / self._std

    def _extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        target_set = set(self._LAYER_INDICES)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in target_set:
                feats.append(x)
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L2 distance between VGG features of pred and target.

        Each feature map is divided by sqrt(numel) so that all 5 layers
        contribute on the same scale to the final sum (this matches the
        L3DG normalisation).
        """
        if pred.dim() != 4 or target.dim() != 4:
            raise ValueError(
                f"Expected (B, 3, H, W) tensors, got {pred.shape} vs {target.shape}"
            )
        pred = self._normalise(pred)
        target = self._normalise(target)
        feats_pred = self._extract(pred)
        with torch.no_grad():
            feats_tgt = self._extract(target)

        loss = pred.new_zeros(())
        for fp, ft in zip(feats_pred, feats_tgt):
            n = float(fp.numel())
            inv_sqrt_n = 1.0 / (n ** 0.5)
            loss = loss + ((fp - ft) * inv_sqrt_n).pow(2).sum()
        # Normalise by batch size so the magnitude is independent of B.
        return loss / pred.shape[0]
