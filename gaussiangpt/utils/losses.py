"""Training losses for GaussianGPT autoencoder.

L = lambda_RGB * L_RGB + lambda_perc * L_perc
  + lambda_occ * L_occ
  + lambda_LFQ * softplus(L_LFQ + 5)

Paper hyperparameters (scenes):
  lambda_RGB = 7.5, lambda_perc = 0.3, 12 images
  lambda_occ = 1.0, lambda_LFQ = 0.1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class VGGPerceptualLoss(nn.Module):
    """VGG19-based perceptual loss using relu3_3 features."""

    def __init__(self):
        super().__init__()
        if not HAS_TORCHVISION:
            raise ImportError("torchvision required for perceptual loss")
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        # Use features up to relu3_3 (index 18)
        self.features = nn.Sequential(*list(vgg.features.children())[:18])
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: (B, 3, H, W) images in [0, 1]
        """
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.l1_loss(self.features(pred), self.features(target))


class AutoencoderLoss(nn.Module):
    """Combined autoencoder training loss."""

    def __init__(
        self,
        lambda_rgb: float = 7.5,
        lambda_perc: float = 0.3,
        lambda_occ: float = 1.0,
        lambda_lfq: float = 0.1,
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_perc = lambda_perc
        self.lambda_occ = lambda_occ
        self.lambda_lfq = lambda_lfq

        self.perceptual = VGGPerceptualLoss() if (use_perceptual and HAS_TORCHVISION) else None

    def forward(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        occ_logits_list: List[torch.Tensor],
        occ_targets_list: List[torch.Tensor],
        lfq_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_rgb: (B, 3, H, W) rendered images
            gt_rgb: (B, 3, H, W) ground-truth images
            occ_logits_list: list of occupancy logits per decoder stage
            occ_targets_list: list of binary occupancy targets per stage
            lfq_loss: LFQ entropy loss scalar
        Returns:
            total_loss, loss_dict
        """
        # L1 color loss
        l_rgb = F.l1_loss(pred_rgb, gt_rgb)

        # Perceptual loss
        l_perc = torch.tensor(0.0, device=pred_rgb.device)
        if self.perceptual is not None:
            l_perc = self.perceptual(pred_rgb, gt_rgb)

        # Occupancy loss (BCE at each upsampling stage)
        l_occ = torch.tensor(0.0, device=pred_rgb.device)
        for logits, targets in zip(occ_logits_list, occ_targets_list):
            l_occ = l_occ + F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), targets.float()
            )
        if len(occ_logits_list) > 0:
            l_occ = l_occ / len(occ_logits_list)

        # LFQ entropy loss with softplus offset
        l_lfq = F.softplus(lfq_loss + 5.0)

        total = (
            self.lambda_rgb * l_rgb
            + self.lambda_perc * l_perc
            + self.lambda_occ * l_occ
            + self.lambda_lfq * l_lfq
        )

        return total, {
            "loss": total.item(),
            "l_rgb": l_rgb.item(),
            "l_perc": l_perc.item(),
            "l_occ": l_occ.item(),
            "l_lfq": l_lfq.item(),
        }
