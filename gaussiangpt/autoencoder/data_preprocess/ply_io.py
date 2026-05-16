"""PLY readers for standard 3D Gaussian Splatting attributes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from plyfile import PlyData

from .schema import (
    GaussianScene,
    compute_gaussian_scene_stats as compute_gaussian_scene_stats,
    validate_gaussian_scene,
)

GAUSSIAN_PLY_FIELDS = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def gaussian_ply_missing_fields(path: Union[str, Path]) -> list[str]:
    """Return missing standard Gaussian PLY fields without materializing arrays."""

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    present = set(vertex.data.dtype.names or ())
    return [field for field in GAUSSIAN_PLY_FIELDS if field not in present]


def is_gaussian_ply(path: Union[str, Path]) -> bool:
    """Return True when a PLY contains the standard Gaussian fields."""

    try:
        return not gaussian_ply_missing_fields(path)
    except Exception:
        return False


def get_gaussian_ply_num_vertices(path: Union[str, Path]) -> int:
    """Return the number of vertex rows in a PLY without building Gaussian arrays."""

    ply = PlyData.read(str(path))
    return int(len(ply["vertex"].data))


def _stack_fields(vertex: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def read_gaussian_ply(
    path: Union[str, Path],
    scene_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> GaussianScene:
    """Read one standard 3DGS Gaussian PLY into a GaussianScene."""

    path = Path(path)
    missing = gaussian_ply_missing_fields(path)
    if missing:
        raise ValueError(f"missing fields: {missing}")

    vertex = PlyData.read(str(path))["vertex"].data
    scene = GaussianScene(
        scene_id=scene_id if scene_id is not None else path.stem,
        xyz=_stack_fields(vertex, ("x", "y", "z")),
        color=_stack_fields(vertex, ("f_dc_0", "f_dc_1", "f_dc_2")),
        opacity=_stack_fields(vertex, ("opacity",)),
        scale=_stack_fields(vertex, ("scale_0", "scale_1", "scale_2")),
        rotation=_stack_fields(vertex, ("rot_0", "rot_1", "rot_2", "rot_3")),
        metadata=dict(metadata or {}),
    )
    validate_gaussian_scene(scene)
    return scene
