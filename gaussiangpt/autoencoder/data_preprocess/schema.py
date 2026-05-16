"""Shared Gaussian scene schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GaussianScene:
    """Standard in-memory representation of one Gaussian scene."""

    scene_id: str
    xyz: np.ndarray
    color: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray
    metadata: dict = field(default_factory=dict)


def validate_gaussian_scene(scene: GaussianScene) -> None:
    """Validate that a GaussianScene follows the project data contract."""

    if not isinstance(scene.scene_id, str):
        raise ValueError("scene_id must be str")

    arrays = {
        "xyz": (scene.xyz, 3),
        "color": (scene.color, 3),
        "opacity": (scene.opacity, 1),
        "scale": (scene.scale, 3),
        "rotation": (scene.rotation, 4),
    }

    num_gaussians: Optional[int] = None
    for name, (array, width) in arrays.items():
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{name} must be np.ndarray")
        if array.dtype != np.float32:
            raise ValueError(f"{name} dtype must be np.float32")
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"{name} shape must be [N, {width}], got {array.shape}")
        if num_gaussians is None:
            num_gaussians = array.shape[0]
        elif array.shape[0] != num_gaussians:
            raise ValueError(
                f"{name} has N={array.shape[0]}, expected N={num_gaussians}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must not contain NaN or Inf")

    if num_gaussians is None or num_gaussians <= 0:
        raise ValueError("N must be > 0")


def compute_gaussian_scene_stats(scene: GaussianScene) -> dict:
    """Compute lightweight, JSON-serializable stats for a GaussianScene."""

    validate_gaussian_scene(scene)
    xyz_min = scene.xyz.min(axis=0)
    xyz_max = scene.xyz.max(axis=0)
    xyz_mean = scene.xyz.mean(axis=0)

    return {
        "scene_id": scene.scene_id,
        "num_gaussians": int(scene.xyz.shape[0]),
        "xyz_min": xyz_min.astype(float).tolist(),
        "xyz_max": xyz_max.astype(float).tolist(),
        "xyz_mean": xyz_mean.astype(float).tolist(),
        "opacity_min": float(scene.opacity.min()),
        "opacity_max": float(scene.opacity.max()),
        "opacity_mean": float(scene.opacity.mean()),
        "scale_min": scene.scale.min(axis=0).astype(float).tolist(),
        "scale_max": scene.scale.max(axis=0).astype(float).tolist(),
    }
