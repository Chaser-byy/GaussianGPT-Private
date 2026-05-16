"""ASE scene, stats, and camera readers."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ply_io import get_gaussian_ply_num_vertices, read_gaussian_ply
from schema import GaussianScene


@dataclass
class ASECameras:
    """ASE camera metadata parsed from transforms_train.json."""

    scene_id: Optional[str]
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    share_intrinsics: bool
    zipped: bool
    crop_edge: int
    frames_num: int
    frames: list[dict]
    transform_device_camera: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    c2w: Optional[np.ndarray] = None
    w2c: Optional[np.ndarray] = None
    camera_centers: Optional[np.ndarray] = None
    forward_dirs: Optional[np.ndarray] = None
    frame_indices: Optional[np.ndarray] = None
    frame_ids: Optional[np.ndarray] = None
    file_paths: Optional[np.ndarray] = None
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None
    pose_convention: str = "c2w=frame_transform_matrix"
    uses_transform_device_camera: bool = False
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ASESceneRecord:
    """Filesystem record for one fixed-layout ASE scene."""

    scene_id: str
    scene_dir: Path
    ckpts_dir: Path
    stats_dir: Path
    ply_path: Optional[Path]
    transforms_path: Path
    valid: bool
    warnings: list[str] = field(default_factory=list)


def _as_matrix4(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4, 4], got {array.shape}")
    return array


def _as_vec3_optional(value, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape [3], got {array.shape}")
    return array


def _camera_id_from_file_path(file_path: str, fallback: str) -> str:
    if not file_path:
        return fallback
    return Path(file_path).stem


def _frame_id_from_file_path(file_path: str, fallback: int) -> int:
    match = re.search(r"(\d+)(?!.*\d)", file_path)
    return int(match.group(1)) if match else int(fallback)


def read_ase_cameras(
    transforms_path: Union[str, Path],
    scene_id: Optional[str] = None,
) -> ASECameras:
    """Read ASE transforms_train.json into typed camera metadata."""

    transforms_path = Path(transforms_path)
    raw = json.loads(transforms_path.read_text(encoding="utf-8"))
    warnings: list[str] = []

    transform_device_camera = None
    if raw.get("transform_device_camera") is not None:
        transform_device_camera = _as_matrix4(
            raw["transform_device_camera"], "transform_device_camera"
        )

    K = np.asarray(
        [
            [float(raw["fx"]), 0.0, float(raw["cx"])],
            [0.0, float(raw["fy"]), float(raw["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    bbox_min = _as_vec3_optional(raw.get("bbox_min"), "bbox_min")
    bbox_max = _as_vec3_optional(raw.get("bbox_max"), "bbox_max")

    frames = []
    c2w_list = []
    w2c_list = []
    frame_indices = []
    frame_ids = []
    file_paths = []
    for index, frame in enumerate(raw.get("frames", [])):
        file_path = str(frame.get("file_path", ""))
        camera_id = _camera_id_from_file_path(file_path, f"frame{index:07d}")
        frame_id = _frame_id_from_file_path(file_path, index)
        c2w = _as_matrix4(
            frame.get("transform_matrix"), f"frames[{index}].transform_matrix"
        )
        w2c = np.linalg.inv(c2w).astype(np.float32)
        c2w_list.append(c2w)
        w2c_list.append(w2c)
        frame_indices.append(index)
        frame_ids.append(frame_id)
        file_paths.append(file_path)
        frames.append(
            {
                "frame_index": index,
                "frame_id": frame_id,
                "camera_id": camera_id,
                "file_path": file_path,
                # ASE RGB camera pose convention: frame transform is c2w directly.
                # transform_device_camera is metadata/debug only for this pipeline.
                "transform_matrix": c2w,
                "c2w": c2w,
                "w2c": w2c,
            }
        )

    frames_num = int(raw.get("frames_num", len(frames)))
    if frames_num != len(frames):
        warnings.append(f"frames_num={frames_num} but found {len(frames)} frames")

    if c2w_list:
        c2w_array = np.stack(c2w_list, axis=0).astype(np.float32)
        w2c_array = np.stack(w2c_list, axis=0).astype(np.float32)
        camera_centers = c2w_array[:, :3, 3].astype(np.float32)
        forward_dirs = c2w_array[:, :3, 2].astype(np.float32)
        forward_dirs = forward_dirs / (
            np.linalg.norm(forward_dirs, axis=1, keepdims=True) + 1e-8
        )
    else:
        c2w_array = np.zeros((0, 4, 4), dtype=np.float32)
        w2c_array = np.zeros((0, 4, 4), dtype=np.float32)
        camera_centers = np.zeros((0, 3), dtype=np.float32)
        forward_dirs = np.zeros((0, 3), dtype=np.float32)

    return ASECameras(
        scene_id=scene_id,
        width=int(raw["width"]),
        height=int(raw["height"]),
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        cx=float(raw["cx"]),
        cy=float(raw["cy"]),
        share_intrinsics=bool(raw.get("share_intrinsics", True)),
        zipped=bool(raw.get("zipped", False)),
        crop_edge=int(raw.get("crop_edge", 0)),
        frames_num=frames_num,
        frames=frames,
        transform_device_camera=transform_device_camera,
        K=K,
        c2w=c2w_array,
        w2c=w2c_array,
        camera_centers=camera_centers,
        forward_dirs=forward_dirs,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        file_paths=np.asarray(file_paths, dtype=np.str_),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        pose_convention="c2w=frame_transform_matrix",
        uses_transform_device_camera=False,
        warnings=warnings,
        metadata={
            "transforms_path": str(transforms_path),
            "json_bbox_min": bbox_min.tolist() if bbox_min is not None else None,
            "json_bbox_max": bbox_max.tolist() if bbox_max is not None else None,
        },
    )


def save_ase_camera_cache(cameras: ASECameras, output_path: Union[str, Path]) -> None:
    """Save preprocessed ASE camera data for chunk-dependent camera selection."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transform_device_camera = (
        cameras.transform_device_camera
        if cameras.transform_device_camera is not None
        else np.eye(4, dtype=np.float32)
    )
    bbox_min = (
        cameras.bbox_min
        if cameras.bbox_min is not None
        else np.full(3, np.nan, dtype=np.float32)
    )
    bbox_max = (
        cameras.bbox_max
        if cameras.bbox_max is not None
        else np.full(3, np.nan, dtype=np.float32)
    )
    np.savez_compressed(
        output_path,
        K=np.asarray(cameras.K, dtype=np.float32),
        c2w=np.asarray(cameras.c2w, dtype=np.float32),
        w2c=np.asarray(cameras.w2c, dtype=np.float32),
        camera_centers=np.asarray(cameras.camera_centers, dtype=np.float32),
        forward_dirs=np.asarray(cameras.forward_dirs, dtype=np.float32),
        frame_indices=np.asarray(cameras.frame_indices, dtype=np.int64),
        frame_ids=np.asarray(cameras.frame_ids, dtype=np.int64),
        file_paths=np.asarray(cameras.file_paths, dtype=np.str_),
        width=np.asarray(cameras.width, dtype=np.int64),
        height=np.asarray(cameras.height, dtype=np.int64),
        fx=np.asarray(cameras.fx, dtype=np.float32),
        fy=np.asarray(cameras.fy, dtype=np.float32),
        cx=np.asarray(cameras.cx, dtype=np.float32),
        cy=np.asarray(cameras.cy, dtype=np.float32),
        bbox_min=np.asarray(bbox_min, dtype=np.float32),
        bbox_max=np.asarray(bbox_max, dtype=np.float32),
        transform_device_camera=np.asarray(transform_device_camera, dtype=np.float32),
        pose_convention=np.asarray(cameras.pose_convention, dtype=np.str_),
        uses_transform_device_camera=np.asarray(
            cameras.uses_transform_device_camera, dtype=np.bool_
        ),
    )


def load_ase_camera_cache(
    camera_cache_path: Union[str, Path],
    scene_id: Optional[str] = None,
) -> ASECameras:
    """Load preprocessed ASE camera npz cache."""

    camera_cache_path = Path(camera_cache_path)
    with np.load(camera_cache_path, allow_pickle=False) as data:
        c2w = np.asarray(data["c2w"], dtype=np.float32)
        w2c = np.asarray(data["w2c"], dtype=np.float32)
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        file_paths = np.asarray(data["file_paths"], dtype=np.str_)
        frames = []
        for row in range(c2w.shape[0]):
            file_path = str(file_paths[row])
            camera_id = f"frame{int(frame_ids[row]):07d}"
            if file_path:
                camera_id = _camera_id_from_file_path(file_path, camera_id)
            frames.append(
                {
                    "frame_index": int(frame_indices[row]),
                    "frame_id": int(frame_ids[row]),
                    "camera_id": camera_id,
                    "file_path": file_path,
                    "transform_matrix": c2w[row],
                    "c2w": c2w[row],
                    "w2c": w2c[row],
                }
            )

        bbox_min = np.asarray(data["bbox_min"], dtype=np.float32)
        bbox_max = np.asarray(data["bbox_max"], dtype=np.float32)
        if np.any(~np.isfinite(bbox_min)):
            bbox_min_value = None
        else:
            bbox_min_value = bbox_min
        if np.any(~np.isfinite(bbox_max)):
            bbox_max_value = None
        else:
            bbox_max_value = bbox_max

        return ASECameras(
            scene_id=scene_id,
            width=int(np.asarray(data["width"]).item()),
            height=int(np.asarray(data["height"]).item()),
            fx=float(np.asarray(data["fx"]).item()),
            fy=float(np.asarray(data["fy"]).item()),
            cx=float(np.asarray(data["cx"]).item()),
            cy=float(np.asarray(data["cy"]).item()),
            share_intrinsics=True,
            zipped=False,
            crop_edge=0,
            frames_num=int(c2w.shape[0]),
            frames=frames,
            transform_device_camera=np.asarray(
                data["transform_device_camera"], dtype=np.float32
            ),
            K=np.asarray(data["K"], dtype=np.float32),
            c2w=c2w,
            w2c=w2c,
            camera_centers=np.asarray(data["camera_centers"], dtype=np.float32),
            forward_dirs=np.asarray(data["forward_dirs"], dtype=np.float32),
            frame_indices=frame_indices,
            frame_ids=frame_ids,
            file_paths=file_paths,
            bbox_min=bbox_min_value,
            bbox_max=bbox_max_value,
            pose_convention=str(np.asarray(data["pose_convention"]).item()),
            uses_transform_device_camera=bool(
                np.asarray(data["uses_transform_device_camera"]).item()
            ),
            metadata={"camera_cache_path": str(camera_cache_path)},
        )


def camera_summary(cameras: ASECameras) -> dict:
    """Return a compact JSON-serializable camera summary."""

    return {
        "scene_id": cameras.scene_id,
        "width": cameras.width,
        "height": cameras.height,
        "fx": cameras.fx,
        "fy": cameras.fy,
        "cx": cameras.cx,
        "cy": cameras.cy,
        "num_frames": len(cameras.frames),
        "frames_num": cameras.frames_num,
        "share_intrinsics": cameras.share_intrinsics,
        "zipped": cameras.zipped,
        "crop_edge": cameras.crop_edge,
        "has_transform_device_camera": cameras.transform_device_camera is not None,
        "pose_convention": cameras.pose_convention,
        "uses_transform_device_camera": cameras.uses_transform_device_camera,
        "bbox_min": cameras.bbox_min.tolist() if cameras.bbox_min is not None else None,
        "bbox_max": cameras.bbox_max.tolist() if cameras.bbox_max is not None else None,
        "warnings": list(cameras.warnings),
    }


def _parse_scalar(value: str):
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_text_stats(text: str) -> dict:
    parsed: dict = {}
    for line in text.splitlines():
        line = line.strip().strip(",")
        if not line or line.startswith("#"):
            continue
        separator = ":" if ":" in line else "=" if "=" in line else None
        if separator is None:
            continue
        key, value = line.split(separator, 1)
        key = key.strip().strip('"').strip("'")
        if key:
            parsed[key] = _parse_scalar(value)
    if not parsed:
        raise ValueError("no key/value stats found")
    return parsed


def _parse_stats_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
        raise ValueError("JSON root is not an object")
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    return _parse_text_stats(text)


def read_ase_stats(stats_dir: Union[str, Path]) -> dict:
    """Read and merge all parseable ASE stats files from a stats directory."""

    stats_dir = Path(stats_dir)
    merged: dict = {"_source_files": [], "_parse_errors": {}}
    if not stats_dir.exists() or not stats_dir.is_dir():
        merged["_parse_errors"][str(stats_dir)] = "stats directory does not exist"
        return merged

    for path in sorted(child for child in stats_dir.iterdir() if child.is_file()):
        try:
            parsed = _parse_stats_file(path)
        except Exception as exc:
            merged["_parse_errors"][path.name] = str(exc)
            continue
        merged.update(parsed)
        merged["_source_files"].append(path.name)

    return merged


def _iteration_from_point_cloud_name(path: Path) -> int:
    match = re.fullmatch(r"point_cloud_(\d+)\.ply", path.name)
    return int(match.group(1)) if match else -1


def find_ase_ply(ckpts_dir: Union[str, Path]) -> Optional[Path]:
    """Select the Gaussian PLY in an ASE ckpts directory."""

    ckpts_dir = Path(ckpts_dir)
    preferred = ckpts_dir / "point_cloud_30000.ply"
    if preferred.is_file():
        return preferred

    point_cloud_candidates = sorted(ckpts_dir.glob("point_cloud_*.ply"))
    if point_cloud_candidates:
        return max(
            point_cloud_candidates,
            key=lambda path: (_iteration_from_point_cloud_name(path), path.name),
        )

    ply_candidates = sorted(ckpts_dir.glob("*.ply"))
    return ply_candidates[0] if ply_candidates else None


def _record_for_scene_dir(scene_dir: Path) -> ASESceneRecord:
    scene_id = scene_dir.name
    ckpts_dir = scene_dir / "ckpts"
    stats_dir = scene_dir / "stats"
    transforms_path = scene_dir / "transforms_train.json"
    warnings: list[str] = []

    if not ckpts_dir.is_dir():
        warnings.append("missing ckpts directory")
    if not stats_dir.is_dir():
        warnings.append("missing stats directory")
    if not transforms_path.is_file():
        warnings.append("missing transforms_train.json")

    ply_path = find_ase_ply(ckpts_dir) if ckpts_dir.is_dir() else None
    if ply_path is None:
        warnings.append("missing Gaussian PLY in ckpts")

    return ASESceneRecord(
        scene_id=scene_id,
        scene_dir=scene_dir,
        ckpts_dir=ckpts_dir,
        stats_dir=stats_dir,
        ply_path=ply_path,
        transforms_path=transforms_path,
        valid=not warnings,
        warnings=warnings,
    )


def discover_ase_scenes(root: Union[str, Path]) -> list[ASESceneRecord]:
    """Discover fixed-layout ASE scenes from direct child directories of root."""

    root = Path(root)
    if not root.exists():
        return []
    return [
        _record_for_scene_dir(scene_dir)
        for scene_dir in sorted(child for child in root.iterdir() if child.is_dir())
        if not scene_dir.name.startswith(".")
    ]


def _safe_num_gaussians(record: ASESceneRecord) -> tuple[Optional[int], Optional[str]]:
    if record.ply_path is None:
        return None, "missing PLY"
    try:
        return get_gaussian_ply_num_vertices(record.ply_path), None
    except Exception as exc:
        return None, str(exc)


def build_ase_scene_manifest(root: Union[str, Path]) -> list[dict]:
    """Build a JSON-serializable manifest for fixed-layout ASE scenes."""

    manifest: list[dict] = []
    for record in discover_ase_scenes(root):
        warnings = list(record.warnings)
        num_gaussians, num_error = _safe_num_gaussians(record)
        if num_error is not None:
            warnings.append(f"failed to count gaussians: {num_error}")

        stats = read_ase_stats(record.stats_dir)

        camera_info = None
        if record.transforms_path.is_file():
            try:
                camera_info = camera_summary(
                    read_ase_cameras(record.transforms_path, scene_id=record.scene_id)
                )
                warnings.extend(camera_info.get("warnings", []))
            except Exception as exc:
                camera_info = {"scene_id": record.scene_id, "warnings": [str(exc)]}
                warnings.append(f"failed to read cameras: {exc}")

        manifest.append(
            {
                "scene_id": record.scene_id,
                "scene_dir": str(record.scene_dir),
                "ckpts_dir": str(record.ckpts_dir),
                "stats_dir": str(record.stats_dir),
                "ply_path": str(record.ply_path) if record.ply_path is not None else None,
                "transforms_path": str(record.transforms_path),
                "num_gaussians": num_gaussians,
                "stats": stats,
                "camera_summary": camera_info,
                "valid": record.valid and num_error is None and camera_info is not None,
                "warnings": warnings,
            }
        )

    return manifest


def load_ase_scene_gaussians(record: ASESceneRecord) -> GaussianScene:
    """Load the selected Gaussian PLY for one ASE scene record."""

    if record.ply_path is None:
        raise ValueError(f"scene {record.scene_id} has no Gaussian PLY")
    return read_gaussian_ply(
        record.ply_path,
        scene_id=record.scene_id,
        metadata={
            "dataset": "ASE",
            "scene_dir": str(record.scene_dir),
            "ckpts_dir": str(record.ckpts_dir),
            "stats_dir": str(record.stats_dir),
            "ply_path": str(record.ply_path),
            "transforms_path": str(record.transforms_path),
        },
    )
