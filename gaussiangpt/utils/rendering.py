"""3D Gaussian Splatting rendering utilities for autoencoder training.

This wraps the CUDA `diff-gaussian-rasterization` rasterizer with a small
PyTorch-friendly API (no Camera class, no Pipe, no GaussianModel needed)
and provides a simple "look-at" camera sampler that places N viewpoints on
a sphere around a scene's bounding box.

Used by `scripts/train_autoencoder.py` to compute the L3DG-style rendering
loss `L_RGB` and the perceptual loss `L_perc` between renderings of the
predicted Gaussians and renderings of the ground-truth Gaussians from the
same camera. The "GT renderings" are detached so the renderer is only used
as a fixed photo-realistic supervision signal.

Convention notes (matching official 3DGS / COLMAP):
  * Camera looks down +Z in camera space (NOT OpenGL -Z).
  * `world_view_transform` is the **transposed** world-to-camera 4x4
    (column-major form expected by the CUDA rasterizer).
  * `full_proj_transform = world_view_transform @ projection_matrix`,
    both in column-major form.
  * `camera_center = world_view_transform.inverse()[3, :3]`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    HAS_RASTERIZER = True
except Exception as _err:  # pragma: no cover - environment dependent
    HAS_RASTERIZER = False
    _RASTERIZER_IMPORT_ERR = _err


# --------------------------------------------------------------------------- #
# Camera math (numpy-free; everything in torch so that augmentations stay
# differentiable if we ever need them).
# --------------------------------------------------------------------------- #

def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm() + eps)


def look_at_RT(
    eye: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (R, T) for a 3DGS / COLMAP-style camera.

    R is the camera-to-world rotation (3x3) whose columns are the world-frame
    directions of the camera's [right, down, forward] axes (camera looks
    down +Z, image y points down).

    T is the world translation expressed *in the camera frame*, i.e.
    T = -R^T @ eye, which is the convention `getWorld2View(R, T)` expects.
    """
    eye = eye.to(torch.float32)
    target = target.to(torch.float32)
    up = up.to(torch.float32)

    forward = _safe_normalize(target - eye)              # camera +Z in world
    right = _safe_normalize(torch.linalg.cross(forward, up))  # camera +X
    down = torch.linalg.cross(forward, right)            # camera +Y (image down)

    R = torch.stack([right, down, forward], dim=1)       # 3x3 cam-to-world
    T = -R.t() @ eye                                     # 3-vec in camera frame
    return R, T


def world2view_matrix(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Build the 4x4 world-to-view matrix in the *transposed* layout the
    rasterizer expects (so it can be used directly as `viewmatrix`)."""
    Rt = torch.zeros(4, 4, dtype=torch.float32, device=R.device)
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    return Rt.t().contiguous()  # transposed for the CUDA rasterizer


def projection_matrix(znear: float, zfar: float, fovx: float, fovy: float,
                      device=None) -> torch.Tensor:
    """3DGS projection matrix in the transposed layout."""
    th_x = math.tan(fovx * 0.5)
    th_y = math.tan(fovy * 0.5)
    P = torch.zeros(4, 4, dtype=torch.float32, device=device)
    P[0, 0] = 1.0 / th_x
    P[1, 1] = 1.0 / th_y
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P.t().contiguous()


@dataclass
class MiniCam:
    """Lightweight stand-in for the official 3DGS Camera object.

    Holds only what the rasterizer + render() actually need.
    """
    image_height: int
    image_width: int
    fovx: float
    fovy: float
    znear: float
    zfar: float
    world_view_transform: torch.Tensor   # (4,4) transposed W2V
    full_proj_transform: torch.Tensor    # (4,4) transposed full proj
    camera_center: torch.Tensor          # (3,) world-space camera position

    @property
    def tanfovx(self) -> float:
        return math.tan(self.fovx * 0.5)

    @property
    def tanfovy(self) -> float:
        return math.tan(self.fovy * 0.5)


def make_camera(
    eye: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor,
    fovx: float,
    fovy: float,
    image_height: int,
    image_width: int,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> MiniCam:
    R, T = look_at_RT(eye, target, up)
    w2v_t = world2view_matrix(R, T)
    proj_t = projection_matrix(znear, zfar, fovx, fovy, device=R.device)
    full_t = w2v_t @ proj_t
    cam_center = torch.linalg.inv(w2v_t)[3, :3]
    return MiniCam(
        image_height=int(image_height),
        image_width=int(image_width),
        fovx=float(fovx), fovy=float(fovy),
        znear=float(znear), zfar=float(zfar),
        world_view_transform=w2v_t,
        full_proj_transform=full_t,
        camera_center=cam_center,
    )


# --------------------------------------------------------------------------- #
# Camera sampling around a scene bounding box.
# --------------------------------------------------------------------------- #

def _fibonacci_sphere(n: int, device) -> torch.Tensor:
    """Approximately uniform points on the unit sphere using a Fibonacci
    spiral. Returns (n, 3)."""
    i = torch.arange(n, dtype=torch.float32, device=device)
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    z = 1.0 - (2.0 * i + 1.0) / n        # in (-1, 1)
    r = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    theta = 2.0 * math.pi * (i / phi)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)


def sample_cameras_around_bbox(
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    n_views: int,
    image_height: int,
    image_width: int,
    fov_deg: float = 60.0,
    radius_factor: float = 1.5,
    upper_hemisphere_only: bool = False,
    jitter: float = 0.0,
    rng: Optional[torch.Generator] = None,
) -> List[MiniCam]:
    """Place `n_views` cameras on a sphere around the given bounding box.

    The radius is `radius_factor * 0.5 * bbox_diagonal`. Each camera looks at
    the bbox centre. Image is square (`image_height == image_width`) by
    convention; FOV is the same in both axes (`fovx == fovy`).
    """
    device = bbox_min.device
    centre = 0.5 * (bbox_min + bbox_max)
    diag = (bbox_max - bbox_min).norm().clamp_min(1e-3)
    radius = (radius_factor * 0.5 * diag).item()

    dirs = _fibonacci_sphere(n_views, device=device)
    if upper_hemisphere_only:
        dirs[:, 2] = dirs[:, 2].abs()
    if jitter > 0.0:
        if rng is None:
            noise = torch.randn_like(dirs) * jitter
        else:
            noise = torch.randn(dirs.shape, generator=rng, device=device) * jitter
        dirs = F.normalize(dirs + noise, dim=-1)

    fov = math.radians(fov_deg)
    up_vec = torch.tensor([0.0, 0.0, 1.0], device=device)
    cams: List[MiniCam] = []
    for k in range(n_views):
        eye = centre + radius * dirs[k]
        cams.append(
            make_camera(
                eye=eye, target=centre, up=up_vec,
                fovx=fov, fovy=fov,
                image_height=image_height, image_width=image_width,
            )
        )
    return cams


# --------------------------------------------------------------------------- #
# Rasterization wrapper.
# --------------------------------------------------------------------------- #

def render_gaussians(
    gaussians: Dict[str, torch.Tensor],
    camera: MiniCam,
    bg_color: Optional[torch.Tensor] = None,
    sh_degree: int = 0,
    scale_modifier: float = 1.0,
) -> torch.Tensor:
    """Render a single view with the CUDA 3DGS rasterizer.

    Args:
        gaussians: dict with the keys
            position  : (N, 3) world-space centres
            scale     : (N, 3) positive linear scales
            rotation  : (N, 4) unit quaternions (w, x, y, z)
            opacity   : (N, 1) **logits** (sigmoid will be applied here)
            color     : (N, 3) linear RGB in (0, 1)
        camera: a MiniCam built by `make_camera`.
        bg_color: 3-vec on the same device. Defaults to black (zeros).
        sh_degree: spherical harmonics degree to use. We pass colours via
            `colors_precomp`, so this is just metadata for the rasterizer
            and should be 0.
        scale_modifier: optional global multiplier for scales.

    Returns:
        rendered_image: (3, H, W) float32 in [0, 1].
    """
    if not HAS_RASTERIZER:
        raise ImportError(
            "diff-gaussian-rasterization is not importable: "
            f"{_RASTERIZER_IMPORT_ERR!r}"
        )

    device = gaussians["position"].device
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device=device)
    else:
        bg_color = bg_color.to(device=device, dtype=torch.float32)

    # The rasterizer requires its tensors on the same CUDA device as itself.
    if not device.type == "cuda":
        raise RuntimeError(
            "render_gaussians requires CUDA tensors (the rasterizer is CUDA-only). "
            f"Got device={device}."
        )

    # Move the camera tensors to the right device (they were built on the
    # bbox's device which should already be CUDA, but be defensive).
    viewmat = camera.world_view_transform.to(device)
    projmat = camera.full_proj_transform.to(device)
    campos = camera.camera_center.to(device)

    raster_settings = GaussianRasterizationSettings(
        image_height=camera.image_height,
        image_width=camera.image_width,
        tanfovx=camera.tanfovx,
        tanfovy=camera.tanfovy,
        bg=bg_color,
        scale_modifier=float(scale_modifier),
        viewmatrix=viewmat,
        projmatrix=projmat,
        sh_degree=int(sh_degree),
        campos=campos,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians["position"]
    means2D = torch.zeros_like(means3D, requires_grad=False)

    # The rasterizer expects activated opacity in (0, 1).
    opacity = torch.sigmoid(gaussians["opacity"])

    out = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=gaussians["color"],
        opacities=opacity,
        scales=gaussians["scale"],
        rotations=gaussians["rotation"],
        cov3D_precomp=None,
    )
    # The CUDA op returns 3 or 5 things depending on the build; the first
    # element is always the rendered image.
    rendered = out[0]
    return rendered.clamp(0.0, 1.0)


def render_gaussians_multiview(
    gaussians: Dict[str, torch.Tensor],
    cameras: List[MiniCam],
    bg_color: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Render the same Gaussians from N cameras. Returns (N, 3, H, W)."""
    imgs = [render_gaussians(gaussians, cam, bg_color=bg_color) for cam in cameras]
    return torch.stack(imgs, dim=0)
