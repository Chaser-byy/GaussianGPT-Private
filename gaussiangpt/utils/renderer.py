"""Gaussian Splatting renderer for computing re-rendering losses.

Implements alpha-compositing based 3D Gaussian rasterization.
For training, we use a differentiable renderer to compute RGB images
from predicted Gaussian parameters.

Note: For production use, consider using the official 3DGS CUDA rasterizer
(diff-gaussian-rasterization). This module provides a pure-PyTorch fallback
for development and testing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


def build_covariance_3d(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Build 3D covariance matrices from scales and quaternions.

    Args:
        scales: (N, 3) scale values (world-space sizes)
        quats: (N, 4) unit quaternions (w, x, y, z)
    Returns:
        cov3d: (N, 3, 3) covariance matrices
    """
    # Build rotation matrix from quaternion
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y),
        2*(x*y + w*z),   1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),   2*(y*z + w*x),   1 - 2*(x*x + y*y),
    ], dim=-1).view(-1, 3, 3)  # (N, 3, 3)

    # Scale matrix
    S = torch.diag_embed(scales)  # (N, 3, 3)

    # Covariance = R @ S @ S^T @ R^T
    RS = R @ S
    return RS @ RS.transpose(-1, -2)


def project_gaussians(
    means3d: torch.Tensor,
    cov3d: torch.Tensor,
    viewmat: torch.Tensor,
    projmat: torch.Tensor,
    img_h: int,
    img_w: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D Gaussians to 2D image plane.

    Args:
        means3d: (N, 3) Gaussian centers in world space
        cov3d: (N, 3, 3) 3D covariance matrices
        viewmat: (4, 4) world-to-camera matrix
        projmat: (4, 4) camera projection matrix
        img_h, img_w: image dimensions
    Returns:
        means2d: (N, 2) projected centers in pixel space
        cov2d: (N, 2, 2) projected 2D covariance matrices
        depths: (N,) depth values
    """
    N = means3d.shape[0]
    device = means3d.device

    # Transform to camera space
    ones = torch.ones(N, 1, device=device)
    means_h = torch.cat([means3d, ones], dim=-1)  # (N, 4)
    means_cam = (viewmat @ means_h.T).T  # (N, 4)
    depths = means_cam[:, 2].clamp(min=0.01)

    # Project to NDC
    means_ndc = (projmat @ means_cam.T).T  # (N, 4)
    means_ndc = means_ndc[:, :2] / means_ndc[:, 3:4]

    # Convert to pixel coordinates
    means2d = torch.stack([
        (means_ndc[:, 0] + 1) * 0.5 * img_w,
        (1 - means_ndc[:, 1]) * 0.5 * img_h,
    ], dim=-1)

    # Project covariance: J @ W @ cov3d @ W^T @ J^T
    # Jacobian of projection (approximate)
    fx = projmat[0, 0] * img_w / 2
    fy = projmat[1, 1] * img_h / 2
    tx, ty = means_cam[:, 0], means_cam[:, 1]
    tz = depths

    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -fx * tx / (tz * tz)
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -fy * ty / (tz * tz)

    W = viewmat[:3, :3].unsqueeze(0)  # (1, 3, 3)
    JW = J @ W  # (N, 2, 3)
    cov2d = JW @ cov3d @ JW.transpose(-1, -2)  # (N, 2, 2)

    # Add small regularization
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    return means2d, cov2d, depths


class GaussianRenderer(nn.Module):
    """Differentiable Gaussian splatting renderer (pure PyTorch).

    For production training, replace with the CUDA-accelerated
    diff-gaussian-rasterization package.
    """

    def __init__(self, img_h: int = 256, img_w: int = 256, bg_color: float = 1.0):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.bg_color = bg_color

    def forward(
        self,
        gaussians: Dict[str, torch.Tensor],
        viewmat: torch.Tensor,
        projmat: torch.Tensor,
    ) -> torch.Tensor:
        """Render Gaussians to an image.

        Args:
            gaussians: dict with keys 'offset' (N,3), 'scale' (N,3),
                       'opacity' (N,1), 'rotation' (N,4), 'color' (N,3)
            viewmat: (4,4) world-to-camera
            projmat: (4,4) projection matrix
        Returns:
            image: (3, H, W) rendered RGB image
        """
        means3d = gaussians["offset"]  # (N, 3) world-space positions
        scales = gaussians["scale"]    # (N, 3)
        opacities = torch.sigmoid(gaussians["opacity"].squeeze(-1))  # (N,) — opacity stored in logit space
        quats = gaussians["rotation"]  # (N, 4)
        colors = gaussians["color"]    # (N, 3)

        N = means3d.shape[0]
        device = means3d.device

        cov3d = build_covariance_3d(scales, quats)
        means2d, cov2d, depths = project_gaussians(
            means3d, cov3d, viewmat, projmat, self.img_h, self.img_w
        )

        # Sort by depth (front to back: ascending depth)
        order = torch.argsort(depths, descending=False)
        means2d = means2d[order]
        cov2d = cov2d[order]
        opacities = opacities[order]
        colors = colors[order]

        # Rasterize (simplified tile-based alpha compositing)
        image = self._rasterize(means2d, cov2d, opacities, colors)
        return image

    def _rasterize(
        self,
        means2d: torch.Tensor,
        cov2d: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
    ) -> torch.Tensor:
        """Simple per-pixel alpha compositing (slow, for reference only)."""
        H, W = self.img_h, self.img_w
        device = means2d.device

        # Create pixel grid
        ys = torch.arange(H, device=device).float() + 0.5
        xs = torch.arange(W, device=device).float() + 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pixels = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

        # Compute Gaussian contributions
        # For efficiency, only process Gaussians within a bounding box
        image = torch.zeros(3, H, W, device=device)
        alpha_acc = torch.zeros(H, W, device=device)

        # Inverse covariance
        det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
        det = det.clamp(min=1e-6)
        inv_cov = torch.stack([
            cov2d[:, 1, 1] / det, -cov2d[:, 0, 1] / det,
            -cov2d[:, 1, 0] / det, cov2d[:, 0, 0] / det,
        ], dim=-1).view(-1, 2, 2)  # (N, 2, 2)

        # Process in batches to avoid OOM
        batch_size = 512
        for i in range(0, len(means2d), batch_size):
            m = means2d[i:i+batch_size]   # (B, 2)
            ic = inv_cov[i:i+batch_size]  # (B, 2, 2)
            a = opacities[i:i+batch_size] # (B,)
            c = colors[i:i+batch_size]    # (B, 3)

            # Pixel - mean: (H, W, B, 2)
            diff = pixels.unsqueeze(2) - m.unsqueeze(0).unsqueeze(0)  # (H, W, B, 2)

            # Mahalanobis distance: diff^T @ inv_cov @ diff
            # (H, W, B, 1, 2) @ (B, 2, 2) -> (H, W, B, 1, 2) @ (H, W, B, 2, 1)
            d_ic = (diff.unsqueeze(-2) @ ic.unsqueeze(0).unsqueeze(0))  # (H, W, B, 1, 2)
            maha = (d_ic @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (H, W, B)

            gauss = torch.exp(-0.5 * maha.clamp(max=20))  # (H, W, B)
            alpha = a.unsqueeze(0).unsqueeze(0) * gauss   # (H, W, B)

            # Alpha compositing (front to back already sorted)
            for j in range(alpha.shape[2]):
                contrib = alpha[:, :, j] * (1 - alpha_acc)
                image += contrib.unsqueeze(0) * c[j].view(3, 1, 1)
                alpha_acc = alpha_acc + contrib

        # Composite background: remaining transmittance gets background color
        image = image + (1 - alpha_acc).unsqueeze(0) * self.bg_color
        return image.clamp(0, 1)
