#!/usr/bin/env python3
"""Minimal 3D-FRONT render data preparation utilities.

This is intentionally standalone: it does not import GaussianGPT modules and
does not assume a fixed project data directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEPTH_EXTS = {".png", ".tif", ".tiff", ".exr", ".npy", ".npz"}
CAMERA_EXTS = {".json", ".txt", ".npy", ".npz"}
RGB_WORDS = {"rgb", "color", "image", "images", "render", "renders", "rerender"}
DEPTH_WORDS = {"depth", "depths"}
BAD_RGB_WORDS = DEPTH_WORDS | {"normal", "normals", "mask", "seg", "semantic", "camera", "cameras", "pose"}
CAMERA_WORDS = {"camera", "cameras", "cam", "pose", "poses", "intrinsic", "intrinsics", "extrinsic", "extrinsics", "transforms"}
DEFAULT_INVALID_DEPTH_MAX = 1.0e4
EXR_FLOAT16_MAX = 65504.0


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray
    source: str


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _tokens(path: Path) -> set:
    text = " ".join(path.parts).lower()
    return set(re.split(r"[^a-z0-9]+", text))


def _is_rgb(path: Path, image_prefix: str = "render") -> bool:
    if path.suffix.lower() not in IMAGE_EXTS:
        return False
    words = _tokens(path)
    if words & BAD_RGB_WORDS:
        return False
    if image_prefix:
        return path.name.lower().startswith(f"{image_prefix.lower()}_")
    return bool(words & RGB_WORDS)


def _is_depth(path: Path) -> bool:
    return path.suffix.lower() in DEPTH_EXTS and bool(_tokens(path) & DEPTH_WORDS)


def _is_camera(path: Path) -> bool:
    return path.suffix.lower() in CAMERA_EXTS and bool(_tokens(path) & CAMERA_WORDS)


def _frame_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"^(rgb|color|image|render|depth|camera|cam|pose)[_-]*", "", stem)
    stem = re.sub(r"[_-]*(rgb|color|image|render|depth|camera|cam|pose)$", "", stem)
    numbers = re.findall(r"\d+", stem)
    if numbers:
        return str(int(numbers[-1]))
    return stem


def _room_root_for_file(path: Path, render_root: Path) -> Path:
    rel = path.relative_to(render_root)
    parts = rel.parts
    if len(parts) >= 3:
        return render_root / parts[0] / parts[1]
    if path.parent.name.lower() in {"rgb", "color", "images", "image", "depth", "camera", "cameras"}:
        return path.parent.parent
    return path.parent


def _find_render_root(data_root: Path) -> Path:
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    preferred = [
        data_root / "3D-FRONT-TEST-RENDER",
        data_root / "3D-FRONT-RENDER",
    ]
    for path in preferred:
        if path.is_dir():
            return path
    candidates = [
        path
        for path in data_root.iterdir()
        if path.is_dir() and "render" in path.name.lower()
    ]
    if candidates:
        return sorted(candidates)[0]
    return data_root


def _looks_like_room_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "meta.json").is_file():
        return True
    has_rgb = any(_is_rgb(child) for child in path.iterdir() if child.is_file())
    has_depth = any(_is_depth(child) for child in path.iterdir() if child.is_file())
    return has_rgb and has_depth


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_matrix(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr
        return out
    if arr.size == 16:
        return arr.reshape(4, 4)
    if arr.size == 12:
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr.reshape(3, 4)
        return out
    return None


def _camera_entries_from_json(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    frames = data.get("frames")
    if isinstance(frames, list):
        entries = []
        for frame in frames:
            if isinstance(frame, dict):
                merged = dict(data)
                merged.pop("frames", None)
                merged.update(frame)
                entries.append(merged)
        return entries
    return [data]


def _camera_entry_from_frame(frame: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    camera = frame.get("camera")
    return camera if isinstance(camera, dict) else None


def _entry_matches_frame(entry: Dict[str, Any], frame: Dict[str, Any]) -> bool:
    frame_key = str(frame.get("frame_id", ""))
    candidates = []
    for key in ("file_path", "image_path", "rgb_path", "name", "id", "frame_id", "frame_index"):
        value = entry.get(key)
        if value is not None:
            candidates.append(_frame_key(Path(str(value))))
    return frame_key in candidates or _frame_key(Path(frame.get("image_path", ""))) in candidates


def _matrix_from_entry(entry: Dict[str, Any], matrix_is: str) -> Tuple[np.ndarray, str]:
    explicit_c2w_keys = ("c2w", "camera_to_world", "camera_to_world_matrix", "transform_matrix", "pose")
    explicit_w2c_keys = ("w2c", "world_to_camera", "world_to_camera_matrix", "extrinsic", "extrinsics")
    for key in explicit_c2w_keys:
        matrix = _to_matrix(entry.get(key))
        if matrix is not None:
            return matrix, key
    for key in explicit_w2c_keys:
        matrix = _to_matrix(entry.get(key))
        if matrix is not None:
            return np.linalg.inv(matrix), key

    for key in ("matrix", "camera", "data"):
        matrix = _to_matrix(entry.get(key))
        if matrix is not None:
            if matrix_is == "w2c":
                return np.linalg.inv(matrix), f"{key}(as_w2c)"
            return matrix, f"{key}(as_c2w)"
    raise ValueError(f"could not find a 3x4/4x4 camera pose in keys={sorted(entry.keys())}")


def _source_camera_convention(entry: Dict[str, Any], args: argparse.Namespace) -> str:
    convention = str(args.camera_convention)
    if convention != "auto":
        return convention
    return str(entry.get("source_convention", "opencv")).lower()


def _convert_c2w_to_opencv(c2w: np.ndarray, convention: str) -> np.ndarray:
    if convention in {"opencv", "colmap", "3dgs"}:
        return c2w
    if convention in {"blender", "opengl"}:
        axis_flip = np.diag([1.0, -1.0, -1.0, 1.0])
        return c2w @ axis_flip
    raise ValueError(
        "camera convention must be auto/opencv/colmap/3dgs/blender/opengl, "
        f"got {convention!r}"
    )


def _intrinsics_from_entry(
    entry: Dict[str, Any],
    width: int,
    height: int,
    fallback_fov_deg: Optional[float],
) -> Tuple[float, float, float, float, str]:
    for key in ("K", "intrinsic", "intrinsics", "camera_intrinsic"):
        if entry.get(key) is not None:
            k = np.asarray(entry[key], dtype=np.float64)
            if k.shape == (3, 3):
                return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), key

    fx = entry.get("fx", entry.get("fl_x", entry.get("focal_x")))
    fy = entry.get("fy", entry.get("fl_y", entry.get("focal_y", fx)))
    cx = entry.get("cx", entry.get("principal_x", width * 0.5))
    cy = entry.get("cy", entry.get("principal_y", height * 0.5))
    if fx is not None and fy is not None:
        return float(fx), float(fy), float(cx), float(cy), "fx/fy"

    angle_x = entry.get("camera_angle_x", entry.get("fov_x"))
    angle_y = entry.get("camera_angle_y", entry.get("fov_y"))
    if angle_x is not None or angle_y is not None:
        if angle_x is None:
            angle_x = angle_y
        if angle_y is None:
            angle_y = angle_x
        fx_val = width / (2.0 * math.tan(float(angle_x) * 0.5))
        fy_val = height / (2.0 * math.tan(float(angle_y) * 0.5))
        return fx_val, fy_val, width * 0.5, height * 0.5, "fov"

    if fallback_fov_deg is None:
        raise ValueError("camera intrinsics are missing; pass --fov-deg or --fx/--fy")
    fov = math.radians(float(fallback_fov_deg))
    focal = width / (2.0 * math.tan(fov * 0.5))
    return focal, focal, width * 0.5, height * 0.5, f"fallback_fov_{fallback_fov_deg:g}"


def _apply_intrinsic_overrides(
    entry: Dict[str, Any],
    args: argparse.Namespace,
    width: int,
    height: int,
) -> Dict[str, Any]:
    if args.fx is None and args.fy is None and args.cx is None and args.cy is None:
        return entry
    out = dict(entry)
    fx = args.fx if args.fx is not None else args.fy
    fy = args.fy if args.fy is not None else args.fx
    if fx is not None:
        out["fx"] = float(fx)
    if fy is not None:
        out["fy"] = float(fy)
    out["cx"] = float(args.cx) if args.cx is not None else float(width) * 0.5
    out["cy"] = float(args.cy) if args.cy is not None else float(height) * 0.5
    return out


def _load_camera_from_path(
    camera_path: Path,
    frame: Dict[str, Any],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> Camera:
    suffix = camera_path.suffix.lower()
    source = camera_path.as_posix()
    if suffix == ".json":
        data = _load_json(camera_path)
        entries = _camera_entries_from_json(data)
        if not entries:
            raise ValueError(f"{camera_path} does not contain camera entries")
        entry = None
        for candidate in entries:
            if _entry_matches_frame(candidate, frame):
                entry = candidate
                break
        if entry is None:
            entry = entries[0]
        entry = _apply_intrinsic_overrides(entry, args, width, height)
        c2w, pose_source = _matrix_from_entry(entry, args.matrix_is)
        c2w = _convert_c2w_to_opencv(c2w, _source_camera_convention(entry, args))
        fx, fy, cx, cy, intr_source = _intrinsics_from_entry(
            entry, width, height, args.fov_deg
        )
        return Camera(fx, fy, cx, cy, c2w, f"{source}:{pose_source}:{intr_source}")

    if suffix in {".npy", ".npz"}:
        loaded = np.load(camera_path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            entry = {key: loaded[key] for key in loaded.files}
        else:
            arr = loaded
            if arr.shape == () and isinstance(arr.item(), dict):
                entry = arr.item()
            else:
                entry = {"matrix": arr}
        entry = _apply_intrinsic_overrides(entry, args, width, height)
        c2w, pose_source = _matrix_from_entry(entry, args.matrix_is)
        c2w = _convert_c2w_to_opencv(c2w, _source_camera_convention(entry, args))
        fx, fy, cx, cy, intr_source = _intrinsics_from_entry(
            entry, width, height, args.fov_deg
        )
        return Camera(fx, fy, cx, cy, c2w, f"{source}:{pose_source}:{intr_source}")

    if suffix == ".txt":
        numbers = np.loadtxt(camera_path, dtype=np.float64)
        entry = _apply_intrinsic_overrides({"matrix": numbers}, args, width, height)
        c2w, pose_source = _matrix_from_entry(entry, args.matrix_is)
        c2w = _convert_c2w_to_opencv(c2w, _source_camera_convention(entry, args))
        fx, fy, cx, cy, intr_source = _intrinsics_from_entry(
            entry, width, height, args.fov_deg
        )
        return Camera(fx, fy, cx, cy, c2w, f"{source}:{pose_source}:{intr_source}")

    raise ValueError(f"unsupported camera file type: {camera_path}")


def _load_camera_for_frame(
    camera_path: Optional[Path],
    frame: Dict[str, Any],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> Camera:
    entry = _camera_entry_from_frame(frame)
    if entry is not None:
        entry = _apply_intrinsic_overrides(entry, args, width, height)
        c2w, pose_source = _matrix_from_entry(entry, args.matrix_is)
        c2w = _convert_c2w_to_opencv(c2w, _source_camera_convention(entry, args))
        fx, fy, cx, cy, intr_source = _intrinsics_from_entry(
            entry, width, height, args.fov_deg
        )
        return Camera(fx, fy, cx, cy, c2w, f"manifest:{pose_source}:{intr_source}")
    if camera_path is None:
        raise ValueError(f"frame has no camera_path or embedded camera: {frame}")
    return _load_camera_from_path(camera_path, frame, width, height, args)


def _read_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required: python -m pip install pillow") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _read_depth(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if suffix == ".npz":
        data = np.load(path)
        key = data.files[0]
        return np.asarray(data[key], dtype=np.float32)
    if suffix == ".exr":
        try:
            import cv2

            depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if depth is not None:
                depth = np.asarray(depth, dtype=np.float32)
            else:
                raise RuntimeError("cv2.imread returned None")
        except Exception:
            depth = None
        if depth is None:
            try:
                import imageio.v3 as iio
            except ImportError as exc:
                raise RuntimeError(
                    "EXR depth requires OpenCV or imageio. Install one of:\n"
                    "  python -m pip install opencv-python\n"
                    "  python -m pip install imageio imageio[ffmpeg]\n"
                ) from exc
            depth = np.asarray(iio.imread(path), dtype=np.float32)
    elif suffix in {".tif", ".tiff"}:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("TIFF depth requires imageio: python -m pip install imageio") from exc
        depth = np.asarray(iio.imread(path), dtype=np.float32)
    else:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required: python -m pip install pillow") from exc
        with Image.open(path) as image:
            depth = np.asarray(image)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32, copy=False)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _finite_min_max(values: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return float(finite.min()), float(finite.max())


def _array_stats(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "median": None,
            "p01": None,
            "p05": None,
            "p95": None,
            "p99": None,
        }
    percentiles = np.percentile(finite, [1, 5, 50, 95, 99])
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "median": float(percentiles[2]),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
    }


def _clean_depth_before_scale(depth_raw: np.ndarray, invalid_depth_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """Remove EXR sentinel/background depths before unit scaling and near/far."""

    depth_raw = np.asarray(depth_raw, dtype=np.float32)
    finite = np.isfinite(depth_raw)
    invalid_large = depth_raw >= float(invalid_depth_max)
    invalid_exr_sentinel = depth_raw >= (EXR_FLOAT16_MAX - 1.0)
    valid_clean = finite & (depth_raw > 0.0) & ~invalid_large & ~invalid_exr_sentinel
    depth_clean = depth_raw.astype(np.float32, copy=True)
    depth_clean[~valid_clean] = np.nan
    return depth_clean, valid_clean


def _depth_masks_and_stats(
    depth_raw: np.ndarray,
    depth_scale: float,
    near: float,
    far: float,
    invalid_depth_max: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return cleaned metric depth, final valid mask, and rich debug stats."""

    depth_clean_raw, clean_mask = _clean_depth_before_scale(
        depth_raw, invalid_depth_max=invalid_depth_max
    )
    depth = depth_clean_raw / float(depth_scale)
    range_mask = np.isfinite(depth) & (depth >= float(near)) & (depth <= float(far))
    cleaned_values = depth[np.isfinite(depth)]
    ranged_values = depth[range_mask]
    raw_min, raw_max = _finite_min_max(depth_raw)
    clean_ratio = float(np.count_nonzero(clean_mask) / clean_mask.size) if clean_mask.size else 0.0
    range_ratio = float(np.count_nonzero(range_mask) / range_mask.size) if range_mask.size else 0.0
    cleaned_stats = _array_stats(cleaned_values)
    range_stats = _array_stats(ranged_values)
    median = cleaned_stats.get("median")
    p95 = cleaned_stats.get("p95")
    looks_like_mm = bool(
        median is not None
        and median > 100.0
        and p95 is not None
        and p95 > 100.0
    )
    stats = {
        "raw": {
            "shape": list(np.asarray(depth_raw).shape),
            "dtype": str(np.asarray(depth_raw).dtype),
            "min": raw_min,
            "max": raw_max,
            "finite_count": int(np.count_nonzero(np.isfinite(depth_raw))),
            "nonpositive_count": int(np.count_nonzero(np.asarray(depth_raw) <= 0.0)),
            "invalid_large_count": int(
                np.count_nonzero(np.asarray(depth_raw, dtype=np.float32) >= float(invalid_depth_max))
            ),
            "exr_sentinel_count": int(
                np.count_nonzero(np.asarray(depth_raw, dtype=np.float32) >= (EXR_FLOAT16_MAX - 1.0))
            ),
        },
        "cleaned_before_near_far": cleaned_stats,
        "near_far_filtered": range_stats,
        "clean_valid_ratio": clean_ratio,
        "valid_ratio": range_ratio,
        "depth_scale": float(depth_scale),
        "near": float(near),
        "far": float(far),
        "invalid_depth_max": float(invalid_depth_max),
        "looks_like_millimeters": looks_like_mm,
    }
    return depth.astype(np.float32, copy=False), range_mask, stats


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(points, colors):
            f.write(
                f"{xyz[0]:.7g} {xyz[1]:.7g} {xyz[2]:.7g} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def _meta_frame_by_type(location: Dict[str, Any], frame_type: str) -> Optional[Dict[str, Any]]:
    for frame in location.get("frames", []):
        if frame.get("type") == frame_type:
            return frame
    return None


def _intrinsics_from_meta(meta: Dict[str, Any], width: int, height: int) -> Dict[str, float]:
    angle_x = meta.get("camera_angle_x")
    if angle_x is not None:
        fx = float(width) / (2.0 * math.tan(float(angle_x) * 0.5))
        return {
            "fx": fx,
            "fy": fx,
            "cx": float(width) * 0.5,
            "cy": float(height) * 0.5,
            "camera_angle_x": float(angle_x),
        }
    sensor_width = meta.get("sensor_width")
    camera_lens = meta.get("camera_lens")
    if sensor_width is not None and camera_lens is not None:
        fx = float(camera_lens) / float(sensor_width) * float(width)
        return {
            "fx": fx,
            "fy": fx,
            "cx": float(width) * 0.5,
            "cy": float(height) * 0.5,
        }
    return {}


def _scan_room_from_meta(
    room_path: Path,
    data_root: Path,
    image_prefix: str,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    meta_path = room_path / "meta.json"
    meta = _load_json(meta_path)
    locations = meta.get("locations")
    if not isinstance(locations, list):
        raise ValueError(f"{meta_path} is missing a list-valued 'locations' field")

    frames = []
    for loc in locations:
        if not isinstance(loc, dict):
            warnings.append(f"{room_path}: skipping non-dict location entry")
            continue
        image_info = _meta_frame_by_type(loc, image_prefix)
        if image_info is None and image_prefix == "render":
            image_info = _meta_frame_by_type(loc, "render")
        depth_info = _meta_frame_by_type(loc, "depth")
        if image_info is None or depth_info is None:
            warnings.append(
                f"{room_path}: location {loc.get('index')} missing "
                f"{image_prefix}/depth frame in meta.json"
            )
            continue
        image_path = room_path / image_info["name"]
        depth_path = room_path / depth_info["name"]
        if not image_path.is_file() or not depth_path.is_file():
            warnings.append(
                f"{room_path}: location {loc.get('index')} references missing files "
                f"image={image_info.get('name')} exists={image_path.is_file()} "
                f"depth={depth_info.get('name')} exists={depth_path.is_file()}"
            )
            continue
        width = int(image_info.get("width") or depth_info.get("width") or 0)
        height = int(image_info.get("height") or depth_info.get("height") or 0)
        camera = {
            "transform_matrix": loc.get("transform_matrix"),
            "source_convention": "blender",
            "position": loc.get("position"),
            "camera_type": loc.get("camera_type"),
            **_intrinsics_from_meta(meta, width, height),
        }
        frame_id = str(loc.get("index") or _frame_key(image_path))
        frames.append(
            {
                "frame_id": frame_id,
                "image_path": _rel(image_path, data_root),
                "depth_path": _rel(depth_path, data_root),
                "camera_path": _rel(meta_path, data_root),
                "camera": camera,
                "width": width,
                "height": height,
                "source_index": loc.get("index"),
                "source_position": loc.get("position"),
            }
        )

    if not frames:
        return None, warnings
    rel_room = _rel(room_path, data_root)
    room_id = rel_room.replace("/", "__")
    return (
        {
            "room_id": room_id,
            "room_path": rel_room,
            "meta_path": _rel(meta_path, data_root),
            "num_frames": len(frames),
            "frames": frames,
        },
        warnings,
    )


def scan_render_data(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    render_root = _find_render_root(data_root)
    if not render_root.is_dir():
        raise SystemExit(f"render root does not exist: {render_root}")

    print(f"[scan] data_root={data_root}")
    print(f"[scan] render_root={render_root}")
    meta_room_paths = sorted({path.parent for path in render_root.rglob("meta.json")})
    rooms = []
    warnings = []
    if meta_room_paths and args.layout in {"auto", "meta"}:
        for room_path in meta_room_paths[: args.max_rooms if args.max_rooms is not None else None]:
            room, room_warnings = _scan_room_from_meta(
                room_path, data_root, args.image_prefix
            )
            warnings.extend(room_warnings)
            if room is not None:
                rooms.append(room)
    elif args.layout == "meta":
        raise SystemExit(f"no meta.json room files found under {render_root}")

    if rooms:
        total_frames = sum(int(room["num_frames"]) for room in rooms)
        manifest = {
            "data_root": data_root.as_posix(),
            "render_root": _rel(render_root, data_root),
            "layout": "meta",
            "image_prefix": args.image_prefix,
            "num_rooms": len(rooms),
            "num_frames": total_frames,
            "rooms": rooms,
            "warnings": warnings[:200],
        }
        _write_json(args.output, manifest)
        print(f"[scan] wrote {args.output}")
        print(f"[scan] layout=meta rooms={len(rooms)} frames={total_frames} warnings={len(warnings)}")
        for warning in warnings[:20]:
            print(f"[scan warning] {warning}")
        return

    all_files = [path for path in render_root.rglob("*") if path.is_file()]
    rgb_files = [path for path in all_files if _is_rgb(path, args.image_prefix)]
    depth_files = [path for path in all_files if _is_depth(path)]
    camera_files = [path for path in all_files if _is_camera(path)]
    if not rgb_files:
        raise SystemExit(f"no RGB images found under {render_root}")
    if not depth_files:
        raise SystemExit(f"no depth files found under {render_root}")
    if not camera_files:
        raise SystemExit(f"no camera files found under {render_root}")

    room_paths = sorted({_room_root_for_file(path, render_root) for path in rgb_files})
    if args.max_rooms is not None:
        room_paths = room_paths[: int(args.max_rooms)]

    total_frames = 0
    for room_path in room_paths:
        room_rgb = sorted([p for p in rgb_files if _under(p, room_path)])
        room_depth = sorted([p for p in depth_files if _under(p, room_path)])
        room_cam = sorted([p for p in camera_files if _under(p, room_path)])
        if not room_depth:
            warnings.append(f"{room_path}: no depth files")
            continue
        if not room_cam:
            warnings.append(f"{room_path}: no camera files")

        depth_by_key = {_frame_key(path): path for path in room_depth}
        cam_by_key = {_frame_key(path): path for path in room_cam}
        room_level_cameras = [
            path for path in room_cam if path.suffix.lower() == ".json" and "transforms" in path.name.lower()
        ]
        default_camera = room_level_cameras[0] if room_level_cameras else (room_cam[0] if len(room_cam) == 1 else None)

        frames = []
        for image_path in room_rgb:
            key = _frame_key(image_path)
            depth_path = depth_by_key.get(key)
            if depth_path is None and len(room_depth) == len(room_rgb):
                depth_path = room_depth[room_rgb.index(image_path)]
            if depth_path is None:
                warnings.append(f"{room_path}: missing depth for image {image_path.name}")
                continue
            camera_path = cam_by_key.get(key, default_camera)
            if camera_path is None:
                warnings.append(f"{room_path}: missing camera for image {image_path.name}")
            frame = {
                "frame_id": key,
                "image_path": _rel(image_path, data_root),
                "depth_path": _rel(depth_path, data_root),
                "camera_path": _rel(camera_path, data_root) if camera_path else None,
            }
            frames.append(frame)

        if frames:
            rel_room = _rel(room_path, data_root)
            room_id = rel_room.replace("/", "__")
            rooms.append(
                {
                    "room_id": room_id,
                    "room_path": rel_room,
                    "num_frames": len(frames),
                    "frames": frames,
                }
            )
            total_frames += len(frames)

    if not rooms:
        raise SystemExit("scan found no rooms with RGB/depth pairs")

    manifest = {
        "data_root": data_root.as_posix(),
        "render_root": _rel(render_root, data_root),
        "num_rooms": len(rooms),
        "num_frames": total_frames,
        "rooms": rooms,
        "warnings": warnings[:200],
    }
    _write_json(args.output, manifest)
    print(f"[scan] wrote {args.output}")
    print(f"[scan] rooms={len(rooms)} frames={total_frames} warnings={len(warnings)}")
    for warning in warnings[:20]:
        print(f"[scan warning] {warning}")


def _load_manifest(path: Path) -> Dict[str, Any]:
    manifest = _load_json(path)
    if "rooms" not in manifest:
        raise ValueError(f"manifest missing 'rooms': {path}")
    return manifest


def _select_room(manifest: Dict[str, Any], room_id: Optional[str]) -> Dict[str, Any]:
    rooms = manifest["rooms"]
    if not rooms:
        raise ValueError("manifest contains no rooms")
    if room_id is None:
        return rooms[0]
    for room in rooms:
        if room["room_id"] == room_id or room.get("room_path") == room_id:
            return room
    raise ValueError(f"room not found: {room_id}")


def _frame_paths(data_root: Path, frame: Dict[str, Any]) -> Tuple[Path, Path, Optional[Path]]:
    image_path = data_root / frame["image_path"]
    depth_path = data_root / frame["depth_path"]
    camera_path = data_root / frame["camera_path"] if frame.get("camera_path") else None
    return image_path, depth_path, camera_path


def _depth_stats(depth_raw: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, Any]]:
    _, valid, stats = _depth_masks_and_stats(
        depth_raw=depth_raw,
        depth_scale=float(args.depth_scale),
        near=float(args.near),
        far=float(args.far),
        invalid_depth_max=float(args.invalid_depth_max),
    )
    return valid, stats


def debug_room(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest)
    data_root = Path(manifest.get("data_root", args.manifest.parent)).resolve()
    room = _select_room(manifest, args.room_id)
    if not room["frames"]:
        raise SystemExit(f"room has no frames: {room['room_id']}")
    frame = room["frames"][min(int(args.frame_index), len(room["frames"]) - 1)]
    image_path, depth_path, camera_path = _frame_paths(data_root, frame)
    rgb = _read_rgb(image_path)
    depth_raw = _read_depth(depth_path)
    if rgb.shape[:2] != depth_raw.shape[:2]:
        raise SystemExit(
            f"RGB/depth shape mismatch: rgb={rgb.shape} depth={depth_raw.shape} "
            f"image={image_path} depth={depth_path}"
        )
    valid, stats = _depth_stats(depth_raw, args)
    camera = _load_camera_for_frame(camera_path, frame, rgb.shape[1], rgb.shape[0], args)
    print(f"[debug] room_id={room['room_id']} frames={len(room['frames'])}")
    print(f"[debug] image={image_path} shape={rgb.shape} dtype={rgb.dtype}")
    print(f"[debug] depth={depth_path} shape={depth_raw.shape} dtype={depth_raw.dtype}")
    print(f"[debug] raw_depth={json.dumps(stats['raw'], sort_keys=True)}")
    print(
        "[debug] cleaned_depth="
        f"{json.dumps(stats['cleaned_before_near_far'], sort_keys=True)} "
        f"clean_valid_ratio={stats['clean_valid_ratio']:.4f}"
    )
    print(
        "[debug] near_far_depth="
        f"{json.dumps(stats['near_far_filtered'], sort_keys=True)} "
        f"valid_ratio={stats['valid_ratio']:.4f}"
    )
    if stats["looks_like_millimeters"] and float(args.depth_scale) == 1.0:
        print(
            "[debug warning] cleaned depth median/p95 are >100; this may be "
            "millimeters. Try --depth-scale 1000."
        )
    print(
        "[debug] camera "
        f"fx={camera.fx:.4g} fy={camera.fy:.4g} cx={camera.cx:.4g} cy={camera.cy:.4g} "
        f"source={camera.source}"
    )
    print(f"[debug] c2w=\n{camera.c2w}")
    print(f"[debug] valid_depth_pixels={int(np.count_nonzero(valid))}/{valid.size}")


def _backproject(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, camera: Camera, points_per_image: int, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    if points_per_image > 0 and xs.size > points_per_image:
        indices = rng.sample(range(xs.size), points_per_image)
        xs = xs[indices]
        ys = ys[indices]
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - camera.cx) / camera.fx * z
    y = (ys.astype(np.float64) - camera.cy) / camera.fy * z
    camera_points = np.stack([x, y, z, np.ones_like(z)], axis=1)
    world = (camera.c2w @ camera_points.T).T[:, :3]
    colors = rgb[ys, xs, :3]
    return world.astype(np.float32), colors.astype(np.uint8)


def _voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return points, colors
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return points[unique_indices], colors[unique_indices]


def _count_ply_vertices(path: Path) -> int:
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"could not find vertex count in PLY header: {path}")


def _read_ascii_xyzrgb_ply(path: Path, max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="ascii", errors="ignore") as f:
        header = []
        vertex_count = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"unexpected EOF while reading PLY header: {path}")
            line = line.strip()
            header.append(line)
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"missing vertex count in PLY: {path}")
        rows = []
        limit = vertex_count if max_points is None else min(vertex_count, int(max_points))
        for _ in range(limit):
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 6:
                continue
            rows.append(
                [
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    int(float(parts[3])),
                    int(float(parts[4])),
                    int(float(parts[5])),
                ]
            )
    if not rows:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    arr = np.asarray(rows)
    return arr[:, :3].astype(np.float32), np.clip(arr[:, 3:6], 0, 255).astype(np.uint8)


def _rotation_matrix_to_qvec(rot: np.ndarray) -> np.ndarray:
    """COLMAP quaternion [qw, qx, qy, qz] from a world-to-camera rotation."""

    rot = np.asarray(rot, dtype=np.float64)
    trace = np.trace(rot)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    if q[0] < 0.0:
        q *= -1.0
    return q


def _copy_or_link(src: Path, dst: Path, symlink: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def build_room(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest)
    data_root = Path(manifest.get("data_root", args.manifest.parent)).resolve()
    room = _select_room(manifest, args.room_id)
    frames = room["frames"][: args.max_views if args.max_views else None]
    if not frames:
        raise SystemExit(f"room has no frames: {room['room_id']}")

    rng = random.Random(int(args.seed))
    out_room = args.output_dir / room["room_id"]
    image_out = out_room / "images"
    image_out.mkdir(parents=True, exist_ok=True)
    all_points = []
    all_colors = []
    used_frames = []
    skipped_frames = []

    for idx, frame in enumerate(frames):
        image_path, depth_path, camera_path = _frame_paths(data_root, frame)
        try:
            rgb = _read_rgb(image_path)
        except Exception as exc:
            reason = f"rgb_read_failed: {exc}"
            skipped_frames.append({"frame_id": frame.get("frame_id"), "reason": reason})
            print(f"[build warning] frame={frame.get('frame_id')} {reason}")
            continue
        try:
            depth_raw = _read_depth(depth_path)
        except Exception as exc:
            reason = f"depth_read_failed: {exc}"
            skipped_frames.append({"frame_id": frame.get("frame_id"), "reason": reason})
            print(f"[build warning] frame={frame.get('frame_id')} {reason}")
            continue
        if rgb.shape[:2] != depth_raw.shape[:2]:
            reason = (
                f"RGB/depth shape mismatch for frame={frame['frame_id']}: "
                f"rgb={rgb.shape} depth={depth_raw.shape}"
            )
            skipped_frames.append({"frame_id": frame.get("frame_id"), "reason": reason})
            print(f"[build warning] {reason}")
            continue
        depth, valid, stats = _depth_masks_and_stats(
            depth_raw=depth_raw,
            depth_scale=float(args.depth_scale),
            near=float(args.near),
            far=float(args.far),
            invalid_depth_max=float(args.invalid_depth_max),
        )
        try:
            camera = _load_camera_for_frame(camera_path, frame, rgb.shape[1], rgb.shape[0], args)
        except Exception as exc:
            reason = f"camera_read_failed: {exc}"
            skipped_frames.append({"frame_id": frame.get("frame_id"), "reason": reason})
            print(f"[build warning] frame={frame.get('frame_id')} {reason}")
            continue
        points, colors = _backproject(rgb, depth, valid, camera, int(args.points_per_image), rng)
        if points.shape[0] == 0:
            clean_count = int(stats["cleaned_before_near_far"]["count"])
            range_count = int(stats["near_far_filtered"]["count"])
            if clean_count == 0:
                reason = "depth_all_invalid_after_cleaning"
            elif range_count == 0:
                reason = "all_depth_filtered_by_near_far"
            else:
                reason = "zero_points_after_sampling"
            skipped_frames.append(
                {
                    "frame_id": frame.get("frame_id"),
                    "reason": reason,
                    "depth_stats": stats,
                }
            )
            print(
                f"[build warning] frame={frame['frame_id']} produced zero points "
                f"reason={reason} clean_count={clean_count} near_far_count={range_count}"
            )
            continue
        all_points.append(points)
        all_colors.append(colors)
        image_name = f"{idx:05d}{image_path.suffix.lower()}"
        if args.copy_images:
            shutil.copy2(image_path, image_out / image_name)
        used_frames.append(
            {
                "frame_id": frame["frame_id"],
                "image_path": f"images/{image_name}" if args.copy_images else frame["image_path"],
                "source_image_path": frame["image_path"],
                "source_depth_path": frame["depth_path"],
                "source_camera_path": frame.get("camera_path"),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "fx": camera.fx,
                "fy": camera.fy,
                "cx": camera.cx,
                "cy": camera.cy,
                "c2w": camera.c2w.tolist(),
                "camera_source": camera.source,
                "valid_depth_ratio": stats["valid_ratio"],
                "clean_valid_ratio": stats["clean_valid_ratio"],
                "depth_stats": stats,
                "generated_points": int(points.shape[0]),
            }
        )
        print(
            f"[build] frame={frame['frame_id']} points={points.shape[0]} "
            f"clean_depth={stats['clean_valid_ratio']:.3f} "
            f"near_far_depth={stats['valid_ratio']:.3f}"
        )

    if not all_points:
        _write_json(
            out_room / "build_failed_summary.json",
            {
                "room_id": room["room_id"],
                "reason": "no_points_generated",
                "num_input_frames": len(frames),
                "skipped_frames": skipped_frames,
                "depth_scale": float(args.depth_scale),
                "near": float(args.near),
                "far": float(args.far),
                "invalid_depth_max": float(args.invalid_depth_max),
            },
        )
        raise SystemExit(
            f"no points generated for room {room['room_id']}; wrote "
            f"{out_room / 'build_failed_summary.json'}"
        )
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    points, colors = _voxel_downsample(points, colors, float(args.voxel_size))
    if args.max_points and points.shape[0] > int(args.max_points):
        indices = rng.sample(range(points.shape[0]), int(args.max_points))
        points = points[indices]
        colors = colors[indices]

    ply_path = out_room / "points3D.ply"
    _write_ply(ply_path, points, colors)
    bbox_min = points.min(axis=0).tolist()
    bbox_max = points.max(axis=0).tolist()
    transforms = {
        "format": "minimal_3dfront_depth_unproject_v1",
        "room_id": room["room_id"],
        "source_room_path": room.get("room_path"),
        "points3D": "points3D.ply",
        "num_views": len(used_frames),
        "num_points": int(points.shape[0]),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "camera_convention": "opencv_colmap_c2w",
        "depth_scale": float(args.depth_scale),
        "near": float(args.near),
        "far": float(args.far),
        "invalid_depth_max": float(args.invalid_depth_max),
        "skipped_frames": skipped_frames,
        "frames": used_frames,
    }
    _write_json(out_room / "transforms.json", transforms)
    print(f"[build] room={room['room_id']} views={len(used_frames)} points={points.shape[0]}")
    print(f"[build] bbox_min={bbox_min} bbox_max={bbox_max}")
    print(f"[build] wrote {ply_path}")
    print(f"[build] wrote {out_room / 'transforms.json'}")


def _room_package_dir(args: argparse.Namespace, room: Dict[str, Any]) -> Path:
    if getattr(args, "room_package_dir", None) is not None:
        return Path(args.room_package_dir)
    built_root = Path(args.built_rooms_dir)
    return built_root / room["room_id"]


def _write_colmap_text_dataset(
    room_package: Path,
    scaffold_dir: Path,
    symlink_images: bool,
    max_colmap_points: Optional[int],
) -> Dict[str, Any]:
    transforms_path = room_package / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"missing room transforms.json: {transforms_path}")
    transforms = _load_json(transforms_path)
    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError(f"room transforms.json has no frames: {transforms_path}")

    points_path = room_package / transforms.get("points3D", "points3D.ply")
    if not points_path.is_file():
        raise FileNotFoundError(f"missing points3D.ply: {points_path}")
    point_count = _count_ply_vertices(points_path)
    if point_count <= 0:
        raise ValueError(f"points3D.ply has no vertices: {points_path}")

    image_dir = scaffold_dir / "images"
    sparse_dir = scaffold_dir / "sparse" / "0"
    image_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    camera_keys: Dict[Tuple[int, int, float, float, float, float], int] = {}
    cameras = []
    image_entries = []
    copied_images = []
    for image_id, frame in enumerate(frames, start=1):
        src_image = room_package / frame["image_path"]
        if not src_image.is_file():
            raise FileNotFoundError(f"missing frame image: {src_image}")
        image_name = Path(frame["image_path"]).name
        dst_image = image_dir / image_name
        _copy_or_link(src_image, dst_image, symlink=symlink_images)
        copied_images.append(dst_image)

        width = int(frame["width"])
        height = int(frame["height"])
        fx = float(frame["fx"])
        fy = float(frame["fy"])
        cx = float(frame["cx"])
        cy = float(frame["cy"])
        cam_key = (width, height, round(fx, 8), round(fy, 8), round(cx, 8), round(cy, 8))
        if cam_key not in camera_keys:
            camera_id = len(camera_keys) + 1
            camera_keys[cam_key] = camera_id
            cameras.append((camera_id, width, height, fx, fy, cx, cy))
        camera_id = camera_keys[cam_key]

        c2w = np.asarray(frame["c2w"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        qvec = _rotation_matrix_to_qvec(w2c[:3, :3])
        tvec = w2c[:3, 3]
        image_entries.append((image_id, qvec, tvec, camera_id, image_name))

    with (sparse_dir / "cameras.txt").open("w", encoding="ascii") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for camera_id, width, height, fx, fy, cx, cy in cameras:
            f.write(f"{camera_id} PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n")

    with (sparse_dir / "images.txt").open("w", encoding="ascii") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, qvec, tvec, camera_id, image_name in image_entries:
            values = " ".join(f"{v:.17g}" for v in [*qvec.tolist(), *tvec.tolist()])
            f.write(f"{image_id} {values} {camera_id} {image_name}\n\n")

    points, colors = _read_ascii_xyzrgb_ply(points_path, max_points=max_colmap_points)
    with (sparse_dir / "points3D.txt").open("w", encoding="ascii") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for idx, (xyz, rgb) in enumerate(zip(points, colors), start=1):
            f.write(
                f"{idx} {xyz[0]:.12g} {xyz[1]:.12g} {xyz[2]:.12g} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0\n"
            )

    shutil.copy2(points_path, scaffold_dir / "points3D.ply")
    summary = {
        "format": "gaussiangpt_3dfront_scaffold_room_v1",
        "room_id": transforms.get("room_id", room_package.name),
        "source_room_package": room_package.as_posix(),
        "dataset_dir": scaffold_dir.as_posix(),
        "images_dir": "images",
        "sparse_dir": "sparse/0",
        "num_images": len(image_entries),
        "num_cameras": len(cameras),
        "num_points3D_ply": int(point_count),
        "num_points3D_colmap": int(points.shape[0]),
        "bbox_min": transforms.get("bbox_min"),
        "bbox_max": transforms.get("bbox_max"),
        "camera_convention": "COLMAP world-to-camera in images.txt; OpenCV/COLMAP c2w in source transforms.json",
        "anchor_aligned_to_voxel_centers": False,
        "one_gaussian_per_anchor": False,
        "voxel_size": None,
        "gaussiangpt_constraints_note": (
            "Depth-initialized COLMAP/3DGS package is prepared. Scaffold-GS "
            "anchor voxel alignment and one-Gaussian-per-anchor require a "
            "Scaffold-GS training code path or post-processing adapter."
        ),
    }
    _write_json(scaffold_dir / "summary.json", summary)
    return summary


def prepare_scaffold_room(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest)
    room = _select_room(manifest, args.room_id)
    room_package = _room_package_dir(args, room)
    if not room_package.is_dir():
        raise SystemExit(
            f"room package not found: {room_package}. Run build-room first or pass --room-package-dir."
        )
    scaffold_dir = args.output_dir / room["room_id"]
    scaffold_dir.mkdir(parents=True, exist_ok=True)
    summary = _write_colmap_text_dataset(
        room_package=room_package,
        scaffold_dir=scaffold_dir,
        symlink_images=bool(args.symlink_images),
        max_colmap_points=args.max_colmap_points,
    )
    if args.scaffold_repo is not None:
        summary["scaffold_repo"] = str(args.scaffold_repo)
        _write_json(scaffold_dir / "summary.json", summary)
    print(f"[prepare-scaffold] dataset={scaffold_dir}")
    print(
        f"[prepare-scaffold] images={summary['num_images']} cameras={summary['num_cameras']} "
        f"points_ply={summary['num_points3D_ply']} points_colmap={summary['num_points3D_colmap']}"
    )
    print(f"[prepare-scaffold] wrote {scaffold_dir / 'summary.json'}")


def _find_training_script(repo: Path) -> Path:
    candidates = [repo / "train.py", repo / "train_scaffold.py"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(repo.rglob("train.py"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"could not find train.py under Scaffold-GS/3DGS repo: {repo}")


def _find_render_script(repo: Path) -> Optional[Path]:
    for name in ("render.py", "render_scaffold.py"):
        candidate = repo / name
        if candidate.is_file():
            return candidate
    matches = sorted(repo.rglob("render.py"))
    return matches[0] if matches else None


def _split_extra_args(value: Optional[str]) -> List[str]:
    if not value:
        return []
    import shlex

    return shlex.split(value)


def _find_training_outputs(output_dir: Path) -> Dict[str, Any]:
    ply_files = sorted(output_dir.rglob("*.ply")) if output_dir.exists() else []
    ckpt_files = sorted(
        list(output_dir.rglob("*.pth")) + list(output_dir.rglob("*.pt"))
    ) if output_dir.exists() else []
    return {
        "ply_files": [path.as_posix() for path in ply_files],
        "checkpoint_files": [path.as_posix() for path in ckpt_files],
        "has_gaussian_ply": bool(ply_files),
        "has_checkpoint": bool(ckpt_files),
    }


def _append_stable_debug_args(cmd: List[str]) -> Dict[str, float]:
    preset = {
        "offset_lr_init": 0.001,
        "offset_lr_final": 0.00001,
        "feature_lr": 0.00075,
        "opacity_lr": 0.002,
        "scaling_lr": 0.0007,
        "rotation_lr": 0.0002,
        "mlp_opacity_lr_init": 0.0002,
        "mlp_opacity_lr_final": 0.000002,
        "mlp_cov_lr_init": 0.0004,
        "mlp_cov_lr_final": 0.0004,
        "mlp_color_lr_init": 0.0008,
        "mlp_color_lr_final": 0.000005,
        "appearance_lr_init": 0.005,
        "appearance_lr_final": 0.00005,
    }
    for key, value in preset.items():
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(["--debug_from", "0", "--nan_debug_interval", "10"])
    return preset


def _read_ply_vertex(path: Path) -> Tuple[np.ndarray, List[str], str]:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError(
            "Reading Scaffold-GS binary PLY files requires plyfile: "
            "python -m pip install plyfile"
        ) from exc
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = list(vertex.dtype.names or [])
    fmt = "ascii" if ply.text else f"binary_{ply.byte_order or 'native'}"
    return vertex, names, fmt


def _stack_ply_fields(vertex: np.ndarray, names: Sequence[str]) -> np.ndarray:
    missing = [name for name in names if name not in (vertex.dtype.names or ())]
    if missing:
        raise ValueError(f"missing PLY fields: {missing}")
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def _sorted_prefixed(names: Iterable[str], prefix: str) -> List[str]:
    out = [name for name in names if name.startswith(prefix)]
    return sorted(out, key=lambda name: int(name.rsplit("_", 1)[-1]))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def _logit_np(x: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32)


def _normalize_np(x: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return (x / norm).astype(np.float32)


def _ply_field_summary(path: Path) -> Dict[str, Any]:
    vertex, names, fmt = _read_ply_vertex(path)
    name_set = set(names)
    offset_names = _sorted_prefixed(names, "f_offset_")
    anchor_feat_names = _sorted_prefixed(names, "f_anchor_feat_")
    scale_names = _sorted_prefixed(names, "scale_")
    rot_names = _sorted_prefixed(names, "rot_")
    has_xyz = all(name in name_set for name in ("x", "y", "z"))
    has_rgb = all(name in name_set for name in ("red", "green", "blue"))
    has_f_dc = all(name in name_set for name in ("f_dc_0", "f_dc_1", "f_dc_2"))
    has_opacity = "opacity" in name_set
    has_scale = len(scale_names) >= 3
    has_rotation = len(rot_names) >= 4
    has_anchor_features = bool(anchor_feat_names)
    has_offsets = bool(offset_names)
    if has_anchor_features or has_offsets:
        rep_type = "scaffold_gs_anchor_representation"
    elif has_f_dc and has_opacity and has_scale and has_rotation:
        rep_type = "vanilla_3dgs_ply"
    elif has_xyz and has_rgb and not has_opacity:
        rep_type = "ordinary_rgb_point_cloud"
    else:
        rep_type = "unknown_ply"
    offsets_per_anchor = len(offset_names) // 3 if len(offset_names) % 3 == 0 else None
    return {
        "path": path.as_posix(),
        "format": fmt,
        "num_vertices": int(len(vertex)),
        "properties": names,
        "output_representation_type": rep_type,
        "has_xyz": has_xyz,
        "has_rgb": has_rgb,
        "has_f_dc": has_f_dc,
        "has_opacity": has_opacity,
        "has_scale": has_scale,
        "num_scale_fields": len(scale_names),
        "has_rotation": has_rotation,
        "num_rotation_fields": len(rot_names),
        "has_anchor_features": has_anchor_features,
        "num_anchor_feature_fields": len(anchor_feat_names),
        "has_offsets": has_offsets,
        "num_offset_fields": len(offset_names),
        "offsets_per_anchor": offsets_per_anchor,
        "directly_usable_by_gaussiangpt_ae": bool(
            rep_type == "vanilla_3dgs_ply" and has_xyz and has_f_dc and has_opacity and has_scale and has_rotation
        ),
        "directly_viewable_by_web_3dgs_viewer": bool(
            rep_type == "vanilla_3dgs_ply" and has_xyz and has_f_dc and has_opacity and has_scale and has_rotation
        ),
    }


def _ply_finite_summary(path: Path) -> Dict[str, Any]:
    vertex, names, _ = _read_ply_vertex(path)
    out = {"path": path.as_posix(), "valid": True, "fields": {}}
    for name in names:
        values = np.asarray(vertex[name])
        if not np.issubdtype(values.dtype, np.number):
            continue
        finite = np.isfinite(values)
        field = {
            "count": int(values.size),
            "finite_count": int(np.count_nonzero(finite)),
            "nan_count": int(np.count_nonzero(np.isnan(values))) if np.issubdtype(values.dtype, np.floating) else 0,
            "inf_count": int(np.count_nonzero(np.isinf(values))) if np.issubdtype(values.dtype, np.floating) else 0,
        }
        if values.size and np.count_nonzero(finite):
            finite_values = values[finite]
            field["min"] = float(np.min(finite_values))
            field["max"] = float(np.max(finite_values))
        else:
            field["min"] = None
            field["max"] = None
        if field["finite_count"] != field["count"]:
            out["valid"] = False
        out["fields"][name] = field
    return out


def _latest_scaffold_iteration(model_dir: Path) -> Optional[int]:
    point_cloud_dir = model_dir / "point_cloud"
    if not point_cloud_dir.is_dir():
        return None
    iterations = []
    for child in point_cloud_dir.iterdir():
        match = re.fullmatch(r"iteration_(\d+)", child.name)
        if match and (child / "point_cloud.ply").is_file():
            iterations.append(int(match.group(1)))
    return max(iterations) if iterations else None


def _resolve_scaffold_ply(model_dir: Optional[Path], point_cloud_ply: Optional[Path], iteration: int) -> Path:
    if point_cloud_ply is not None:
        return point_cloud_ply.resolve()
    if model_dir is None:
        raise SystemExit("pass either --model-dir or --point-cloud-ply")
    model_dir = model_dir.resolve()
    if iteration < 0:
        latest = _latest_scaffold_iteration(model_dir)
        if latest is None:
            raise SystemExit(f"could not find point_cloud/iteration_*/point_cloud.ply under {model_dir}")
        iteration = latest
    ply = model_dir / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    if not ply.is_file():
        raise SystemExit(f"missing Scaffold-GS point cloud: {ply}")
    return ply.resolve()


def _resolve_scaffold_iteration_from_ply(path: Path) -> Optional[int]:
    match = re.search(r"iteration_(\d+)", path.as_posix())
    return int(match.group(1)) if match else None


def _find_training_checkpoint_summary(model_dir: Path) -> Dict[str, Any]:
    ckpts = sorted(model_dir.glob("chkpnt*.pth"))
    out: Dict[str, Any] = {"checkpoint_files": [p.as_posix() for p in ckpts], "has_checkpoint": bool(ckpts)}
    if not ckpts:
        return out
    latest = ckpts[-1]
    out["inspected_checkpoint"] = latest.as_posix()
    try:
        import torch

        obj = torch.load(str(latest), map_location="cpu")
        out["loadable_by_torch"] = True
        out["top_level_type"] = type(obj).__name__
        if isinstance(obj, tuple):
            out["tuple_len"] = len(obj)
            if len(obj) >= 2:
                out["reported_iteration"] = int(obj[1]) if isinstance(obj[1], (int, np.integer)) else str(obj[1])
            if obj:
                tensors = []
                first = obj[0]
                if isinstance(first, tuple):
                    for idx, item in enumerate(first):
                        tensors.append(
                            {
                                "index": idx,
                                "type": type(item).__name__,
                                "shape": list(item.shape) if hasattr(item, "shape") else None,
                            }
                        )
                out["model_tuple_fields"] = tensors
    except Exception as exc:
        out["loadable_by_torch"] = False
        out["error"] = str(exc)
    return out


def inspect_scaffold_output(args: argparse.Namespace) -> None:
    ply = _resolve_scaffold_ply(args.model_dir, args.point_cloud_ply, args.iteration)
    summary = _ply_field_summary(ply)
    vertex, _, _ = _read_ply_vertex(ply)
    anchor_alignment = _voxel_alignment_stats(
        _stack_ply_fields(vertex, ("x", "y", "z")),
        args.anchor_voxel_size,
        args.alignment_tolerance,
    )
    summary["voxel_size"] = float(args.anchor_voxel_size)
    summary["anchor_aligned_to_voxel_centers"] = bool(anchor_alignment["aligned"])
    summary["anchor_alignment"] = anchor_alignment
    summary["one_gaussian_per_anchor"] = bool(summary.get("offsets_per_anchor") == 1)
    model_dir = args.model_dir.resolve() if args.model_dir is not None else ply.parents[2]
    train_summary_path = model_dir / "train_summary.json"
    if train_summary_path.is_file():
        train_summary = _load_json(train_summary_path)
        summary["train_summary"] = {
            "path": train_summary_path.as_posix(),
            "iterations": train_summary.get("iterations"),
            "returncode": train_summary.get("returncode"),
            "used_scaffold_gs": bool("Scaffold-GS" in str(train_summary.get("scaffold_repo", "")) or (model_dir / "cfg_args").is_file()),
            "gaussiangpt_mode_effective": bool(train_summary.get("gaussiangpt_mode_effective", False)),
        }
    summary["checkpoint"] = _find_training_checkpoint_summary(model_dir)
    summary["source_training_iterations"] = _resolve_scaffold_iteration_from_ply(ply)
    output = args.output or (model_dir / "inspect_summary.json")
    _write_json(output, summary)
    print(f"[inspect-scaffold] representation={summary['output_representation_type']}")
    print(f"[inspect-scaffold] ae_usable={summary['directly_usable_by_gaussiangpt_ae']}")
    print(f"[inspect-scaffold] viewer_usable={summary['directly_viewable_by_web_3dgs_viewer']}")
    print(f"[inspect-scaffold] wrote {output}")


def _write_gaussian_ply(
    path: Path,
    xyz: np.ndarray,
    color: np.ndarray,
    opacity_logit: np.ndarray,
    scale_world: np.ndarray,
    rotation: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    color = np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0)
    opacity_logit = np.asarray(opacity_logit, dtype=np.float32)
    scale_log = np.log(np.maximum(np.asarray(scale_world, dtype=np.float32), 1.0e-8))
    rotation = _normalize_np(rotation)
    c0 = 0.28209479177387814
    f_dc = (color - 0.5) / c0
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        for name in (
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
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
        ):
            f.write(f"property float {name}\n")
        f.write("end_header\n")
        normals = np.zeros_like(xyz)
        rows = np.concatenate([xyz, normals, f_dc, opacity_logit, scale_log, rotation], axis=1)
        for row in rows:
            f.write(" ".join(f"{float(v):.9g}" for v in row))
            f.write("\n")


def _decode_scaffold_with_mlp(
    ply: Path,
    canonical_view: Sequence[float],
    canonical_distance: float,
    chunk_size: int,
    no_opacity_filter: bool = False,
    export_all_anchors: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    import torch

    vertex, names, _ = _read_ply_vertex(ply)
    offsets_names = _sorted_prefixed(names, "f_offset_")
    feat_names = _sorted_prefixed(names, "f_anchor_feat_")
    scale_names = _sorted_prefixed(names, "scale_")
    rot_names = _sorted_prefixed(names, "rot_")
    anchor = _stack_ply_fields(vertex, ("x", "y", "z"))
    offsets_flat = _stack_ply_fields(vertex, offsets_names)
    offsets = offsets_flat.reshape(anchor.shape[0], 3, -1).transpose(0, 2, 1).astype(np.float32)
    feat = _stack_ply_fields(vertex, feat_names)
    scaling_repeat = np.exp(_stack_ply_fields(vertex, scale_names))
    stored_opacity = _stack_ply_fields(vertex, ("opacity",))
    stored_rotation = _stack_ply_fields(vertex, rot_names)
    if export_all_anchors:
        scale_world = scaling_repeat[:, 3:6] if scaling_repeat.shape[1] >= 6 else scaling_repeat[:, :3]
        return (
            {
                "xyz": anchor,
                "color": np.full((anchor.shape[0], 3), 0.5, dtype=np.float32),
                "opacity": stored_opacity,
                "scale": scale_world.astype(np.float32),
                "rotation": _normalize_np(stored_rotation[:, :4]),
                "anchor_index": np.arange(anchor.shape[0], dtype=np.int64),
            },
            {
                "export_method": "scaffold_anchor_debug_export_all_anchors",
                "limitations": [
                    "Debug export bypasses Scaffold-GS opacity/color/covariance MLP decode and writes gray anchor Gaussians.",
                    "This output is not a faithful GaussianGPT GT Gaussian scene.",
                ],
            },
        )
    checkpoint_dir = ply.parent
    opacity_path = checkpoint_dir / "opacity_mlp.pt"
    cov_path = checkpoint_dir / "cov_mlp.pt"
    color_path = checkpoint_dir / "color_mlp.pt"
    if not opacity_path.is_file() or not cov_path.is_file() or not color_path.is_file():
        scale_world = scaling_repeat[:, 3:6] if scaling_repeat.shape[1] >= 6 else scaling_repeat[:, :3]
        return (
            {
                "xyz": anchor,
                "color": np.full((anchor.shape[0], 3), 0.5, dtype=np.float32),
                "opacity": stored_opacity,
                "scale": scale_world.astype(np.float32),
                "rotation": _normalize_np(stored_rotation[:, :4]),
                "anchor_index": np.arange(anchor.shape[0], dtype=np.int64),
            },
            {
                "export_method": "scaffold_anchor_fallback_no_mlp",
                "limitations": [
                    "MLP checkpoints were not found; exported stored anchor opacity/scale/rotation and gray color.",
                    "This is not a faithful explicit decode of Scaffold-GS neural Gaussians.",
                ],
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opacity_mlp = torch.jit.load(str(opacity_path), map_location=device).eval()
    cov_mlp = torch.jit.load(str(cov_path), map_location=device).eval()
    color_mlp = torch.jit.load(str(color_path), map_location=device).eval()
    feature_bank_mlp = None
    feature_bank_path = checkpoint_dir / "feature_bank_mlp.pt"
    if feature_bank_path.is_file():
        feature_bank_mlp = torch.jit.load(str(feature_bank_path), map_location=device).eval()
    appearance = None
    appearance_path = checkpoint_dir / "embedding_appearance.pt"
    if appearance_path.is_file():
        appearance = torch.jit.load(str(appearance_path), map_location=device).eval()

    canonical = np.asarray(canonical_view, dtype=np.float32)
    norm = float(np.linalg.norm(canonical))
    if norm <= 0.0:
        raise ValueError("--canonical-view must be non-zero")
    canonical = canonical / norm
    xyz_list = []
    color_list = []
    opacity_list = []
    scale_list = []
    rot_list = []
    anchor_index_list = []

    def _first_working(module: Any, candidates: List[Any]) -> Any:
        last_exc = None
        for candidate in candidates:
            try:
                return module(candidate)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"could not find compatible MLP input shape: {last_exc}")

    with torch.no_grad():
        for start in range(0, anchor.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), anchor.shape[0])
            feat_t = torch.from_numpy(feat[start:end]).float().to(device)
            anchor_t = torch.from_numpy(anchor[start:end]).float().to(device)
            offsets_t = torch.from_numpy(offsets[start:end]).float().to(device)
            scaling_t = torch.from_numpy(scaling_repeat[start:end]).float().to(device)
            view_t = torch.from_numpy(np.repeat(canonical[None, :], end - start, axis=0)).float().to(device)
            dist_t = torch.full((end - start, 1), float(canonical_distance), dtype=torch.float32, device=device)
            if feature_bank_mlp is not None:
                weights = feature_bank_mlp(torch.cat([view_t, dist_t], dim=1)).unsqueeze(dim=1)
                feat_fb = (
                    feat_t[:, ::4, None].repeat(1, 4, 1) * weights[:, :, :1]
                    + feat_t[:, ::2, None].repeat(1, 2, 1) * weights[:, :, 1:2]
                    + feat_t[:, ::1, None] * weights[:, :, 2:]
                )
                feat_t = feat_fb.squeeze(dim=-1)
            local = torch.cat([feat_t, view_t], dim=1)
            local_dist = torch.cat([feat_t, view_t, dist_t], dim=1)
            neural_opacity = _first_working(opacity_mlp, [local, local_dist]).reshape(-1, 1)
            scale_rot = _first_working(cov_mlp, [local, local_dist]).reshape((end - start) * offsets.shape[1], 7)
            color_candidates = [local, local_dist]
            if appearance is not None:
                app = appearance(torch.zeros(end - start, dtype=torch.long, device=device))
                color_candidates = [torch.cat([local, app], dim=1), torch.cat([local_dist, app], dim=1)]
            color = _first_working(color_mlp, color_candidates).reshape((end - start) * offsets.shape[1], 3)
            if no_opacity_filter:
                mask = torch.ones_like(neural_opacity, dtype=torch.bool).view(-1)
            else:
                mask = (neural_opacity > 0.0).view(-1)
            repeated_scaling = scaling_t[:, None, :].repeat(1, offsets.shape[1], 1).reshape(-1, scaling_t.shape[1])
            repeated_anchor = anchor_t[:, None, :].repeat(1, offsets.shape[1], 1).reshape(-1, 3)
            offsets_all = offsets_t.reshape(-1, 3)
            xyz = repeated_anchor + offsets_all * repeated_scaling[:, :3]
            scale = repeated_scaling[:, 3:6] * torch.sigmoid(scale_rot[:, :3])
            rot = torch.nn.functional.normalize(scale_rot[:, 3:7], dim=1)
            anchor_indices = torch.arange(start, end, dtype=torch.long, device=device)[:, None].repeat(1, offsets.shape[1]).reshape(-1)
            xyz_list.append(xyz[mask].cpu().numpy().astype(np.float32))
            color_list.append(color[mask].cpu().numpy().astype(np.float32))
            opacity_list.append(_logit_np(neural_opacity[mask].cpu().numpy()))
            scale_list.append(scale[mask].cpu().numpy().astype(np.float32))
            rot_list.append(rot[mask].cpu().numpy().astype(np.float32))
            anchor_index_list.append(anchor_indices[mask].cpu().numpy().astype(np.int64))

    arrays = {
        "xyz": np.concatenate(xyz_list, axis=0),
        "color": np.concatenate(color_list, axis=0),
        "opacity": np.concatenate(opacity_list, axis=0),
        "scale": np.concatenate(scale_list, axis=0),
        "rotation": np.concatenate(rot_list, axis=0),
        "anchor_index": np.concatenate(anchor_index_list, axis=0),
    }
    meta = {
        "export_method": "scaffold_gs_mlp_canonical_view_decode",
        "canonical_view": canonical.astype(float).tolist(),
        "canonical_distance": float(canonical_distance),
        "limitations": [
            "Scaffold-GS predicts opacity/color/covariance from viewing direction and optional appearance embeddings.",
            "Export decodes a static explicit Gaussian scene using the chosen canonical view/distance.",
        ],
    }
    if no_opacity_filter:
        meta["limitations"].append("Opacity filter was disabled; exported all decoded offsets regardless of neural opacity sign.")
    if appearance is not None:
        meta["limitations"].append("Appearance embedding is decoded with camera index 0.")
    return arrays, meta


def _voxel_alignment_stats(xyz: np.ndarray, voxel_size: float, tolerance: float) -> Dict[str, Any]:
    xyz = np.asarray(xyz, dtype=np.float32)
    if voxel_size <= 0.0 or xyz.size == 0:
        return {"voxel_size": float(voxel_size), "aligned": False, "max_abs_error": None}
    snapped = np.round(xyz / float(voxel_size)) * float(voxel_size)
    err = np.abs(xyz - snapped)
    max_err = float(err.max()) if err.size else 0.0
    return {
        "voxel_size": float(voxel_size),
        "aligned": bool(max_err <= float(tolerance)),
        "max_abs_error": max_err,
        "tolerance": float(tolerance),
    }


def export_gt_gaussian(args: argparse.Namespace) -> None:
    ply = _resolve_scaffold_ply(args.model_dir, args.point_cloud_ply, args.iteration)
    model_dir = args.model_dir.resolve() if args.model_dir is not None else ply.parents[2]
    train_summary_path = model_dir / "train_summary.json"
    nan_debug_path = model_dir / "nan_debug.json"
    if nan_debug_path.is_file():
        raise SystemExit(f"training produced NaN/invalid render; export invalid: {nan_debug_path}")
    if train_summary_path.is_file():
        train_summary = _load_json(train_summary_path)
        if train_summary.get("nan_detected"):
            raise SystemExit("training produced NaN, export invalid")
        if train_summary.get("returncode") not in (None, 0):
            raise SystemExit(f"training failed with returncode={train_summary.get('returncode')}; export invalid")
    finite_summary = _ply_finite_summary(ply)
    if not finite_summary["valid"]:
        raise SystemExit(f"source Scaffold-GS PLY contains NaN/Inf; export invalid: {ply}")
    output_ply = args.output_ply.resolve()
    arrays, export_meta = _decode_scaffold_with_mlp(
        ply,
        canonical_view=args.canonical_view,
        canonical_distance=args.canonical_distance,
        chunk_size=args.chunk_size,
        no_opacity_filter=bool(args.no_opacity_filter),
        export_all_anchors=bool(args.export_all_anchors),
    )
    if arrays["xyz"].shape[0] <= 0:
        raise SystemExit(
            "export produced zero Gaussians after opacity filtering; training may be unstable. "
            "Inspect nan_debug.json/train logs first, or use --no-opacity-filter only for debug export."
        )
    for name in ("xyz", "color", "opacity", "scale", "rotation"):
        if not np.isfinite(arrays[name]).all():
            raise SystemExit(f"decoded Gaussian attribute {name} contains NaN/Inf; export invalid")
    _write_gaussian_ply(output_ply, arrays["xyz"], arrays["color"], arrays["opacity"], arrays["scale"], arrays["rotation"])
    output_pt = args.output_pt.resolve() if args.output_pt is not None else output_ply.with_suffix(".pt")
    try:
        import torch

        torch.save(
            {
                "positions": torch.from_numpy(arrays["xyz"]),
                "colors": torch.from_numpy(np.clip(arrays["color"], 0.0, 1.0)),
                "opacities": torch.from_numpy(arrays["opacity"]),
                "scales": torch.from_numpy(arrays["scale"]),
                "rotations": torch.from_numpy(_normalize_np(arrays["rotation"])),
                "source_anchor_indices": torch.from_numpy(arrays["anchor_index"]),
            },
            str(output_pt),
        )
    except Exception as exc:
        output_pt = None
        export_meta.setdefault("limitations", []).append(f"failed to write PT export: {exc}")
    anchor_summary = _ply_field_summary(ply)
    voxel_stats = _voxel_alignment_stats(
        _stack_ply_fields(_read_ply_vertex(ply)[0], ("x", "y", "z")),
        args.anchor_voxel_size,
        args.alignment_tolerance,
    )
    per_anchor_counts = np.bincount(arrays["anchor_index"], minlength=anchor_summary["num_vertices"])
    summary = {
        "format": "gaussiangpt_gt_gaussian_export_v1",
        "source_point_cloud_ply": ply.as_posix(),
        "source_checkpoint": ply.parent.as_posix(),
        "source_training_iterations": _resolve_scaffold_iteration_from_ply(ply),
        "output_ply": output_ply.as_posix(),
        "output_pt": output_pt.as_posix() if output_pt is not None else None,
        "voxel_size": float(args.anchor_voxel_size),
        "anchor_aligned_to_voxel_centers": bool(voxel_stats["aligned"]),
        "anchor_alignment": voxel_stats,
        "one_gaussian_per_anchor": bool(anchor_summary.get("offsets_per_anchor") == 1 and int(per_anchor_counts.max()) <= 1),
        "offsets_per_anchor": anchor_summary.get("offsets_per_anchor"),
        "num_anchors": int(anchor_summary["num_vertices"]),
        "num_gaussians": int(arrays["xyz"].shape[0]),
        "has_explicit_gaussian_attributes": True,
        "directly_usable_by_gaussiangpt_ae": True,
        "directly_viewable_by_web_3dgs_viewer": True,
        "output_representation_type": "vanilla_3dgs_ply",
        "source_ply_finite": bool(finite_summary["valid"]),
        "no_opacity_filter": bool(args.no_opacity_filter),
        "export_all_anchors": bool(args.export_all_anchors),
        **export_meta,
    }
    summary_path = args.summary.resolve() if args.summary is not None else output_ply.with_name(output_ply.stem + "_summary.json")
    _write_json(summary_path, summary)
    print(f"[export-gt-gaussian] wrote {output_ply}")
    if output_pt is not None:
        print(f"[export-gt-gaussian] wrote {output_pt}")
    print(f"[export-gt-gaussian] summary={summary_path}")


def _load_exported_gaussian(path: Path) -> Dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        vertex, names, _ = _read_ply_vertex(path)
        colors = _stack_ply_fields(vertex, ("f_dc_0", "f_dc_1", "f_dc_2"))
        c0 = 0.28209479177387814
        return {
            "position": _stack_ply_fields(vertex, ("x", "y", "z")),
            "color": np.clip(colors * c0 + 0.5, 0.0, 1.0),
            "opacity": _stack_ply_fields(vertex, ("opacity",)),
            "scale": np.exp(_stack_ply_fields(vertex, ("scale_0", "scale_1", "scale_2"))),
            "rotation": _normalize_np(_stack_ply_fields(vertex, ("rot_0", "rot_1", "rot_2", "rot_3"))),
        }
    if suffix in {".pt", ".pth"}:
        import torch

        data = torch.load(str(path), map_location="cpu")
        def _as_np(*keys: str) -> np.ndarray:
            value = None
            for key in keys:
                if key in data:
                    value = data[key]
                    break
            if value is None:
                return value
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        return {
            "position": _as_np("positions", "position"),
            "color": _as_np("colors", "color"),
            "opacity": _as_np("opacities", "opacity"),
            "scale": _as_np("scales", "scale"),
            "rotation": _as_np("rotations", "rotation"),
        }
    if suffix == ".npz":
        data = np.load(path)
        def _npz_value(*keys: str) -> np.ndarray:
            for key in keys:
                if key in data:
                    return np.asarray(data[key], dtype=np.float32)
            return None

        return {
            "position": _npz_value("positions", "position"),
            "color": _npz_value("colors", "color"),
            "opacity": _npz_value("opacities", "opacity"),
            "scale": _npz_value("scales", "scale"),
            "rotation": _npz_value("rotations", "rotation"),
        }
    raise ValueError(f"unsupported GT Gaussian extension: {path}")


def validate_gt_gaussian(args: argparse.Namespace) -> None:
    path = args.input.resolve()
    arrays = _load_exported_gaussian(path)
    required = ("position", "opacity", "scale", "rotation", "color")
    errors = []
    n = None
    dims = {"position": 3, "opacity": 1, "scale": 3, "rotation": 4, "color": 3}
    for name in required:
        arr = arrays.get(name)
        if arr is None:
            errors.append(f"missing {name}")
            continue
        if arr.dtype != np.float32:
            arrays[name] = arr.astype(np.float32)
            arr = arrays[name]
        if arr.ndim != 2 or arr.shape[1] != dims[name]:
            errors.append(f"{name} shape should be [N,{dims[name]}], got {arr.shape}")
        if not np.isfinite(arr).all():
            errors.append(f"{name} contains NaN/Inf")
        n = arr.shape[0] if n is None else n
        if arr.shape[0] != n:
            errors.append(f"{name} N={arr.shape[0]} differs from expected {n}")
    alignment = _voxel_alignment_stats(arrays["position"], args.voxel_size, args.alignment_tolerance)
    if args.require_voxel_aligned and not alignment["aligned"]:
        errors.append("positions are not aligned to requested voxel centers")
    unique_voxels = None
    one_per_voxel = False
    if args.voxel_size > 0 and n:
        coords = np.round(arrays["position"] / float(args.voxel_size)).astype(np.int64)
        unique_voxels = np.unique(coords, axis=0).shape[0]
        one_per_voxel = unique_voxels == n
        if args.require_one_gaussian_per_anchor and not one_per_voxel:
            errors.append("more than one Gaussian maps to at least one voxel center")
    summary = {
        "input": path.as_posix(),
        "valid": not errors,
        "errors": errors,
        "has_position": "position" in arrays,
        "has_opacity": "opacity" in arrays,
        "has_scale": "scale" in arrays,
        "has_rotation": "rotation" in arrays,
        "has_color": "color" in arrays,
        "num_gaussians": int(n or 0),
        "voxel_size": float(args.voxel_size),
        "anchor_aligned_to_voxel_centers": bool(alignment["aligned"]),
        "alignment": alignment,
        "one_gaussian_per_anchor": bool(one_per_voxel),
        "num_unique_voxel_centers": int(unique_voxels) if unique_voxels is not None else None,
        "directly_usable_by_gaussiangpt_ae": bool(not errors),
    }
    output = args.output or path.with_name(path.stem + "_validate_summary.json")
    _write_json(output, summary)
    print(f"[validate-gt-gaussian] valid={summary['valid']}")
    print(f"[validate-gt-gaussian] gaussians={summary['num_gaussians']}")
    print(f"[validate-gt-gaussian] wrote {output}")
    if errors:
        raise SystemExit("; ".join(errors))


def train_scaffold_room(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.resolve()
    repo = args.scaffold_repo.resolve()
    output_dir = args.output_dir.resolve()
    summary_path = dataset_dir / "summary.json"
    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset dir does not exist: {dataset_dir}")
    if not (dataset_dir / "images").is_dir():
        raise SystemExit(f"dataset dir is missing images/: {dataset_dir}")
    if not (dataset_dir / "sparse" / "0" / "cameras.txt").is_file():
        raise SystemExit(f"dataset dir is missing COLMAP sparse/0/cameras.txt: {dataset_dir}")
    if not (dataset_dir / "points3D.ply").is_file():
        raise SystemExit(f"dataset dir is missing points3D.ply: {dataset_dir}")
    points_count = _count_ply_vertices(dataset_dir / "points3D.ply")
    if points_count <= 0:
        raise SystemExit(f"points3D.ply has no vertices: {dataset_dir / 'points3D.ply'}")
    train_script = _find_training_script(repo)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        str(train_script),
        "-s",
        str(dataset_dir),
        "-m",
        str(output_dir),
        "--iterations",
        str(int(args.iterations)),
    ]
    if args.gaussiangpt_mode:
        cmd.append("--gaussiangpt_mode")
    if args.one_gaussian_per_anchor:
        cmd.append("--one_gaussian_per_anchor")
    if args.align_anchor_to_voxel_center:
        cmd.append("--align_anchor_to_voxel_center")
    if args.anchor_voxel_size is not None:
        cmd.extend(["--anchor_voxel_size", str(float(args.anchor_voxel_size))])
    stable_debug_preset = _append_stable_debug_args(cmd) if args.stable_debug else None
    if args.detect_anomaly:
        cmd.append("--detect_anomaly")
    if args.debug:
        cmd.append("--debug")
    if args.nan_debug_interval is not None:
        cmd.extend(["--nan_debug_interval", str(int(args.nan_debug_interval))])
    cmd.extend(_split_extra_args(args.extra_args))
    gaussiangpt_mode_effective = bool(args.gaussiangpt_mode)
    anchor_voxel_size_effective = float(args.anchor_voxel_size) if args.anchor_voxel_size is not None else None
    one_gaussian_effective = bool(args.gaussiangpt_mode and args.one_gaussian_per_anchor)
    anchor_alignment_effective = bool(args.gaussiangpt_mode and args.align_anchor_to_voxel_center and anchor_voxel_size_effective is not None)
    train_summary = {
        "dataset_dir": dataset_dir.as_posix(),
        "scaffold_repo": repo.as_posix(),
        "train_script": train_script.as_posix(),
        "output_dir": output_dir.as_posix(),
        "iterations": int(args.iterations),
        "command": cmd,
        "dry_run": bool(args.dry_run),
        "input_checks": {
            "num_images": len(list((dataset_dir / "images").iterdir())),
            "points3D_ply_vertices": int(points_count),
            "has_summary_json": summary_path.is_file(),
        },
        "used_scaffold_gs": True,
        "depth_initialized_point_cloud": True,
        "gaussiangpt_mode_requested": bool(args.gaussiangpt_mode),
        "gaussiangpt_mode_effective": gaussiangpt_mode_effective,
        "anchor_aligned_to_voxel_centers": anchor_alignment_effective,
        "align_anchor_to_voxel_center_effective": anchor_alignment_effective,
        "one_gaussian_per_anchor": one_gaussian_effective,
        "one_gaussian_per_anchor_effective": one_gaussian_effective,
        "voxel_size": anchor_voxel_size_effective,
        "anchor_voxel_size": anchor_voxel_size_effective,
        "anchor_growing_pruning_disabled": bool(gaussiangpt_mode_effective),
        "stable_debug": bool(args.stable_debug),
        "stable_debug_preset": stable_debug_preset,
        "detect_anomaly": bool(args.detect_anomaly),
        "debug": bool(args.debug),
        "nan_debug_interval": int(args.nan_debug_interval) if args.nan_debug_interval is not None else (10 if args.stable_debug else None),
        "paper_faithful_training_note": (
            "GaussianGPT mode requests Scaffold-GS anchors aligned to voxel centers and n_offsets=1. "
            "Final explicit Gaussian attributes still require export-gt-gaussian because Scaffold-GS stores "
            "anchor features plus MLPs."
        )
        if gaussiangpt_mode_effective
        else "Plain Scaffold-GS training; use GaussianGPT flags for constrained GT Gaussian preparation.",
    }
    _write_json(output_dir / "train_summary.json", train_summary)
    print("[train-scaffold] command:")
    print("  " + " ".join(cmd))
    if args.dry_run:
        print(f"[train-scaffold] dry-run summary={output_dir / 'train_summary.json'}")
        return

    start = time.time()
    with (output_dir / "train_stdout_stderr.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    outputs = _find_training_outputs(output_dir)
    nan_debug_path = output_dir / "nan_debug.json"
    representation_summary = None
    try:
        trained_ply = _resolve_scaffold_ply(output_dir, None, int(args.iterations))
        representation_summary = _ply_field_summary(trained_ply)
        vertex, _, _ = _read_ply_vertex(trained_ply)
        output_alignment = _voxel_alignment_stats(
            _stack_ply_fields(vertex, ("x", "y", "z")),
            float(args.anchor_voxel_size),
            1.0e-4,
        )
        representation_summary["anchor_aligned_to_voxel_centers"] = bool(output_alignment["aligned"])
        representation_summary["anchor_alignment"] = output_alignment
        representation_summary["one_gaussian_per_anchor"] = bool(representation_summary.get("offsets_per_anchor") == 1)
    except (Exception, SystemExit) as exc:
        representation_summary = {"error": str(exc)}
    train_summary.update(
        {
            "returncode": int(proc.returncode),
            "elapsed_seconds": float(elapsed),
            "outputs": outputs,
            "output_representation": representation_summary,
            "nan_detected": nan_debug_path.is_file(),
            "nan_debug_path": nan_debug_path.as_posix() if nan_debug_path.is_file() else None,
        }
    )
    if representation_summary and "output_representation_type" in representation_summary:
        for key in (
            "output_representation_type",
            "has_xyz",
            "has_rgb",
            "has_f_dc",
            "has_opacity",
            "has_scale",
            "has_rotation",
            "has_anchor_features",
            "anchor_aligned_to_voxel_centers",
            "one_gaussian_per_anchor",
            "directly_usable_by_gaussiangpt_ae",
            "directly_viewable_by_web_3dgs_viewer",
        ):
            train_summary[key] = representation_summary[key]

    render_script = _find_render_script(repo) if args.render_preview else None
    if proc.returncode == 0 and render_script is not None:
        render_cmd = [
            args.python,
            str(render_script),
            "-s",
            str(dataset_dir),
            "-m",
            str(output_dir),
            "--iteration",
            str(int(args.iterations)),
        ]
        render_cmd.extend(_split_extra_args(args.render_extra_args))
        train_summary["render_command"] = render_cmd
        with (output_dir / "render_stdout_stderr.log").open("w", encoding="utf-8") as log:
            render_proc = subprocess.run(render_cmd, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT)
        train_summary["render_returncode"] = int(render_proc.returncode)
        train_summary["render_outputs"] = _find_training_outputs(output_dir)

    _write_json(output_dir / "train_summary.json", train_summary)
    print(f"[train-scaffold] returncode={proc.returncode} elapsed={elapsed:.1f}s")
    print(f"[train-scaffold] summary={output_dir / 'train_summary.json'}")
    if proc.returncode != 0:
        detail = f"see {output_dir / 'train_stdout_stderr.log'}"
        if nan_debug_path.is_file():
            detail += f" and {nan_debug_path}"
        raise SystemExit(
            f"Scaffold/3DGS training failed with returncode={proc.returncode}; {detail}"
        )


def add_common_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Raw depth divisor, e.g. 1000 for millimeters.")
    parser.add_argument("--near", type=float, default=1.0e-4, help="Minimum valid depth after scaling.")
    parser.add_argument("--far", type=float, default=20.0, help="Maximum valid depth after scaling.")
    parser.add_argument(
        "--invalid-depth-max",
        type=float,
        default=DEFAULT_INVALID_DEPTH_MAX,
        help="Discard raw depth >= this before scaling; filters EXR background/sentinel values.",
    )
    parser.add_argument("--fov-deg", type=float, default=60.0, help="Fallback horizontal FOV if camera intrinsics are missing.")
    parser.add_argument("--fx", type=float, default=None, help="Manual focal length override in pixels.")
    parser.add_argument("--fy", type=float, default=None, help="Manual focal length override in pixels.")
    parser.add_argument("--cx", type=float, default=None, help="Manual principal point x override in pixels.")
    parser.add_argument("--cy", type=float, default=None, help="Manual principal point y override in pixels.")
    parser.add_argument(
        "--camera-convention",
        choices=["auto", "opencv", "colmap", "3dgs", "blender", "opengl"],
        default="auto",
        help="Source pose convention. auto treats 3D-FRONT meta.json as blender and other files as opencv.",
    )
    parser.add_argument("--matrix-is", choices=["c2w", "w2c"], default="c2w", help="Convention for unlabeled 4x4 camera matrices.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan/debug/build minimal 3D-FRONT render room packages.")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan rendered data and write a manifest.json.")
    scan.add_argument("--data-root", required=True, type=Path, help="Extracted 3D-FRONT render root or parent directory.")
    scan.add_argument("--output", required=True, type=Path, help="Output manifest JSON path.")
    scan.add_argument("--max-rooms", type=int, default=None, help="Optional limit for a small first pass.")
    scan.add_argument(
        "--layout",
        choices=["auto", "meta", "glob"],
        default="auto",
        help="Use room meta.json if available, or fallback to filename glob pairing.",
    )
    scan.add_argument(
        "--image-prefix",
        default="render",
        help="RGB image filename/type prefix to use, usually render or rerender.",
    )
    scan.set_defaults(func=scan_render_data)

    debug = sub.add_parser("debug-room", help="Print sanity checks for one room/frame.")
    debug.add_argument("--manifest", required=True, type=Path)
    debug.add_argument("--room-id", default=None, help="Room id or room_path from manifest. Defaults to first room.")
    debug.add_argument("--frame-index", type=int, default=0)
    add_common_camera_args(debug)
    debug.set_defaults(func=debug_room)

    build = sub.add_parser("build-room", help="Generate points3D.ply and a minimal transforms.json for one room.")
    build.add_argument("--manifest", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--room-id", default=None, help="Room id or room_path from manifest. Defaults to first room.")
    build.add_argument("--max-views", type=int, default=8)
    build.add_argument("--points-per-image", type=int, default=20000)
    build.add_argument("--max-points", type=int, default=300000)
    build.add_argument("--voxel-size", type=float, default=0.02)
    build.add_argument("--seed", type=int, default=1234)
    build.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    add_common_camera_args(build)
    build.set_defaults(func=build_room)

    scaffold = sub.add_parser("prepare-scaffold-room", help="Write a room-local COLMAP text package for Scaffold-GS/3DGS.")
    scaffold.add_argument("--manifest", required=True, type=Path)
    scaffold.add_argument("--built-rooms-dir", required=True, type=Path, help="Directory passed to build-room --output-dir.")
    scaffold.add_argument("--output-dir", required=True, type=Path, help="Directory for Scaffold/3DGS-ready room datasets.")
    scaffold.add_argument("--room-id", default=None, help="Room id or room_path from manifest. Defaults to first room.")
    scaffold.add_argument("--room-package-dir", type=Path, default=None, help="Explicit built room package directory.")
    scaffold.add_argument("--scaffold-repo", type=Path, default=None, help="Optional Scaffold-GS/3DGS repo path recorded in summary.")
    scaffold.add_argument("--symlink-images", action=argparse.BooleanOptionalAction, default=False)
    scaffold.add_argument("--max-colmap-points", type=int, default=None, help="Optional cap for sparse/0/points3D.txt; points3D.ply is copied in full.")
    scaffold.set_defaults(func=prepare_scaffold_room)

    train_scaffold = sub.add_parser("train-scaffold-room", help="Launch external Scaffold-GS/3DGS training for one prepared room.")
    train_scaffold.add_argument("--dataset-dir", required=True, type=Path, help="Output of prepare-scaffold-room for one room.")
    train_scaffold.add_argument("--scaffold-repo", required=True, type=Path, help="Path to Scaffold-GS or compatible 3DGS repository.")
    train_scaffold.add_argument("--output-dir", required=True, type=Path, help="Training output/model directory.")
    train_scaffold.add_argument("--iterations", type=int, default=60000)
    train_scaffold.add_argument("--python", default=sys.executable)
    train_scaffold.add_argument("--extra-args", default="", help="Extra args passed to train.py, shell-quoted string.")
    train_scaffold.add_argument("--dry-run", action="store_true")
    train_scaffold.add_argument("--render-preview", action=argparse.BooleanOptionalAction, default=True)
    train_scaffold.add_argument("--render-extra-args", default="", help="Extra args passed to render.py if present.")
    train_scaffold.add_argument("--gaussiangpt-mode", action="store_true", help="Enable the constrained Scaffold-GS GaussianGPT preparation mode.")
    train_scaffold.add_argument("--anchor-voxel-size", type=float, default=0.025, help="Voxel size used to align Scaffold-GS anchors.")
    train_scaffold.add_argument("--one-gaussian-per-anchor", action="store_true", help="Force Scaffold-GS n_offsets=1 in GaussianGPT mode.")
    train_scaffold.add_argument("--align-anchor-to-voxel-center", action="store_true", help="Snap anchors to round(x/voxel_size)*voxel_size.")
    train_scaffold.add_argument("--stable-debug", "--stable_debug", dest="stable_debug", action="store_true", help="Use conservative learning rates and frequent NaN/loss logging.")
    train_scaffold.add_argument("--detect-anomaly", "--detect_anomaly", dest="detect_anomaly", action="store_true", help="Pass --detect_anomaly to Scaffold-GS train.py.")
    train_scaffold.add_argument("--debug", action="store_true", help="Pass --debug to Scaffold-GS pipeline args.")
    train_scaffold.add_argument("--nan-debug-interval", "--nan_debug_interval", dest="nan_debug_interval", type=int, default=None, help="Print loss component stats every N iterations.")
    train_scaffold.set_defaults(func=train_scaffold_room)

    inspect = sub.add_parser("inspect-scaffold-output", help="Inspect Scaffold-GS point_cloud.ply/checkpoint fields.")
    inspect.add_argument("--model-dir", type=Path, default=None, help="Scaffold-GS training output/model directory.")
    inspect.add_argument("--point-cloud-ply", type=Path, default=None, help="Explicit point_cloud.ply path.")
    inspect.add_argument("--iteration", type=int, default=-1, help="-1 means latest iteration under --model-dir.")
    inspect.add_argument("--output", type=Path, default=None, help="Output inspect summary JSON.")
    inspect.add_argument("--anchor-voxel-size", type=float, default=0.025)
    inspect.add_argument("--alignment-tolerance", type=float, default=1.0e-4)
    inspect.set_defaults(func=inspect_scaffold_output)

    export = sub.add_parser("export-gt-gaussian", help="Export explicit GaussianGPT-compatible GT Gaussian attributes.")
    export.add_argument("--model-dir", type=Path, default=None, help="Scaffold-GS training output/model directory.")
    export.add_argument("--point-cloud-ply", type=Path, default=None, help="Explicit Scaffold-GS point_cloud.ply path.")
    export.add_argument("--iteration", type=int, default=-1, help="-1 means latest iteration under --model-dir.")
    export.add_argument("--output-ply", required=True, type=Path, help="Output standard 3DGS/GaussianGPT-compatible PLY.")
    export.add_argument("--output-pt", type=Path, default=None, help="Optional tensor checkpoint output; defaults next to PLY.")
    export.add_argument("--summary", type=Path, default=None, help="Output export summary JSON.")
    export.add_argument("--anchor-voxel-size", type=float, default=0.025)
    export.add_argument("--alignment-tolerance", type=float, default=1.0e-4)
    export.add_argument("--canonical-view", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    export.add_argument("--canonical-distance", type=float, default=1.0)
    export.add_argument("--chunk-size", type=int, default=65536)
    export.add_argument("--no-opacity-filter", action="store_true", help="Debug only: export decoded Gaussians even when neural opacity <= 0.")
    export.add_argument("--export-all-anchors", action="store_true", help="Debug only: bypass MLP decode and export one gray Gaussian per anchor.")
    export.set_defaults(func=export_gt_gaussian)

    validate = sub.add_parser("validate-gt-gaussian", help="Validate exported GT Gaussian PLY/PT/NPZ attributes.")
    validate.add_argument("--input", required=True, type=Path, help="Exported PLY/PT/NPZ file.")
    validate.add_argument("--output", type=Path, default=None, help="Output validation summary JSON.")
    validate.add_argument("--voxel-size", type=float, default=0.025)
    validate.add_argument("--alignment-tolerance", type=float, default=1.0e-4)
    validate.add_argument("--require-voxel-aligned", action="store_true")
    validate.add_argument("--require-one-gaussian-per-anchor", action="store_true")
    validate.set_defaults(func=validate_gt_gaussian)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
