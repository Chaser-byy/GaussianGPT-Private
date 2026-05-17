"""Voxelization, feature encoding, and ASE voxel cache utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .ase import (
    discover_ase_scenes,
    load_ase_scene_gaussians,
    read_ase_cameras,
    save_ase_camera_cache,
)
from .schema import GaussianScene, validate_gaussian_scene

SH_C0 = 0.28209479177387814
DEFAULT_FEATURE_SCALES = {
    "offset": 1.0,
    "color": 1.0,
    "opacity": 1.0,
    "scale": 1.0,
    "rotation": 1.0,
}


def _import_minkowski_engine():
    try:
        import MinkowskiEngine as ME
    except ImportError as exc:
        raise ImportError(
            "MinkowskiEngine is required for voxelization in the AutoDL environment."
        ) from exc
    return ME


def _extract_sparse_quantize_result(result, shuffled_coords: np.ndarray) -> tuple:
    if isinstance(result, tuple):
        index = None
        coords = None
        for item in result:
            array = np.asarray(item)
            if array.ndim == 1 and np.issubdtype(array.dtype, np.integer):
                index = array.astype(np.int64, copy=False)
            elif array.ndim == 2 and array.shape[1] == 3:
                coords = array.astype(np.int32, copy=False)
        if index is None:
            raise ValueError("MinkowskiEngine sparse_quantize did not return unique indices")
        if coords is None:
            coords = shuffled_coords[index]
        return coords, index

    index = np.asarray(result, dtype=np.int64)
    if index.ndim != 1:
        raise ValueError("MinkowskiEngine sparse_quantize return value is not a 1D index array")
    return shuffled_coords[index], index


def quantize_scene_with_minkowski(
    scene: GaussianScene,
    voxel_size: float = 0.025,
    seed: int = 42,
) -> Dict:
    """Quantize a Gaussian scene with MinkowskiEngine sparse_quantize."""

    validate_gaussian_scene(scene)
    ME = _import_minkowski_engine()

    scene_origin = scene.xyz.min(axis=0).astype(np.float32)
    scene_coords_all = np.floor((scene.xyz - scene_origin) / voxel_size).astype(np.int32)

    rng = np.random.RandomState(seed)
    shuffled_original_indices = rng.permutation(scene.xyz.shape[0]).astype(np.int64)
    shuffled_coords = scene_coords_all[shuffled_original_indices]

    result = ME.utils.sparse_quantize(shuffled_coords, return_index=True)
    quantized_coords, selected_shuffled_indices = _extract_sparse_quantize_result(
        result, shuffled_coords
    )
    selected_indices = shuffled_original_indices[selected_shuffled_indices].astype(np.int64)

    return {
        "scene_origin": scene_origin,
        "scene_coords": quantized_coords.astype(np.int32, copy=False),
        "selected_indices": selected_indices,
    }


def _feature_scales(feature_scales: Dict = None) -> Dict:
    scales = dict(DEFAULT_FEATURE_SCALES)
    if feature_scales is not None:
        scales.update(feature_scales)
    return scales


def encode_gaussian_features_for_ae(
    relative_xyz: np.ndarray,
    color_raw: np.ndarray,
    opacity_raw: np.ndarray,
    scale_raw: np.ndarray,
    rotation_raw: np.ndarray,
    voxel_size: float,
    feature_scales: Dict = None,
) -> np.ndarray:
    """Encode Gaussian attributes using the GaussianGPT AE feature representation."""

    del voxel_size
    scales = _feature_scales(feature_scales)

    relative_xyz = np.asarray(relative_xyz, dtype=np.float32) * float(scales["offset"])
    rgb = np.asarray(color_raw, dtype=np.float32) * SH_C0 + 0.5
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32) * float(scales["color"])

    opacity = np.clip(np.asarray(opacity_raw, dtype=np.float32), -10.0, 10.0)
    opacity = opacity.astype(np.float32) * float(scales["opacity"])

    # Standard 3DGS PLY files store scale_* in log-space. GaussianGPT's AE
    # trains on positive world-space sizes, while the decoder predicts those
    # positive sizes via softplus. Therefore PLY input is decoded with exp()
    # here; softplus belongs on the model output side, not on the PLY reader.
    scale = np.exp(np.asarray(scale_raw, dtype=np.float32))
    scale = np.maximum(scale, 1e-8).astype(np.float32) * float(scales["scale"])

    quat = np.asarray(rotation_raw, dtype=np.float32)
    quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-8)
    negative_w = quat[:, 0] < 0
    quat[negative_w] *= -1.0
    quat = quat.astype(np.float32) * float(scales["rotation"])

    return np.concatenate([relative_xyz, rgb, opacity, scale, quat], axis=1).astype(
        np.float32, copy=False
    )


def build_scene_voxel_features(
    scene: GaussianScene,
    scene_origin: np.ndarray,
    scene_coords: np.ndarray,
    selected_indices: np.ndarray,
    voxel_size: float = 0.025,
    feature_scales: Dict = None,
) -> Dict:
    """Build whole-scene 14D voxel features after sparse quantization."""

    validate_gaussian_scene(scene)
    scene_origin = np.asarray(scene_origin, dtype=np.float32)
    scene_coords = np.asarray(scene_coords, dtype=np.int32)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)

    voxel_center_world = scene_origin + (scene_coords.astype(np.float32) + 0.5) * float(
        voxel_size
    )
    xyz_selected = scene.xyz[selected_indices]
    relative_xyz = (xyz_selected - voxel_center_world).astype(np.float32)
    feats = encode_gaussian_features_for_ae(
        relative_xyz=relative_xyz,
        color_raw=scene.color[selected_indices],
        opacity_raw=scene.opacity[selected_indices],
        scale_raw=scene.scale[selected_indices],
        rotation_raw=scene.rotation[selected_indices],
        voxel_size=voxel_size,
        feature_scales=feature_scales,
    )

    return {
        "coords": scene_coords.astype(np.int32, copy=False),
        "feats": feats,
        "selected_global_indices": selected_indices.astype(np.int64, copy=False),
    }


def _metadata_json(metadata: Dict) -> np.ndarray:
    return np.asarray(json.dumps(metadata), dtype=np.str_)


def _array_to_list_or_none(value: Optional[np.ndarray]):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).tolist()


def _bbox_sanity_warnings(
    json_bbox_min: Optional[np.ndarray],
    json_bbox_max: Optional[np.ndarray],
    ply_xyz_min: np.ndarray,
    ply_xyz_max: np.ndarray,
    abs_threshold: float = 0.5,
    rel_threshold: float = 0.25,
) -> List[str]:
    """Check scene-level JSON bbox against PLY xyz range without rejecting scenes."""

    if json_bbox_min is None or json_bbox_max is None:
        return ["missing json bbox_min/bbox_max in transforms_train.json"]

    json_bbox_min = np.asarray(json_bbox_min, dtype=np.float32)
    json_bbox_max = np.asarray(json_bbox_max, dtype=np.float32)
    ply_xyz_min = np.asarray(ply_xyz_min, dtype=np.float32)
    ply_xyz_max = np.asarray(ply_xyz_max, dtype=np.float32)
    edge = np.maximum(json_bbox_max - json_bbox_min, 1e-6)
    diff = np.maximum(np.abs(json_bbox_min - ply_xyz_min), np.abs(json_bbox_max - ply_xyz_max))
    rel = diff / edge
    if np.any(diff > abs_threshold) or np.any(rel > rel_threshold):
        return [
            "json bbox differs from PLY xyz range; json bbox is scene-level metadata, "
            "not the sampled chunk bbox"
        ]
    return []


def _cache_world_bbox(
    scene_origin: np.ndarray,
    scene_coords: np.ndarray,
    voxel_size: float,
) -> tuple:
    if scene_coords.shape[0] == 0:
        origin = np.asarray(scene_origin, dtype=np.float32)
        return origin.copy(), origin.copy()
    coord_min = scene_coords.min(axis=0).astype(np.float32)
    coord_max = scene_coords.max(axis=0).astype(np.float32) + 1.0
    origin = np.asarray(scene_origin, dtype=np.float32)
    return (
        (origin + coord_min * float(voxel_size)).astype(np.float32),
        (origin + coord_max * float(voxel_size)).astype(np.float32),
    )


def build_ase_voxel_cache(
    root: Union[str, Path],
    output_root: Union[str, Path],
    voxel_size: float = 0.025,
    overwrite: bool = False,
    seed: int = 42,
    scene_ids: Optional[List[str]] = None,
) -> Dict:
    """Voxelize ASE scenes once and save compact scene-level npz caches."""

    root = Path(root)
    output_root = Path(output_root)
    scenes_dir = output_root / "scenes"
    cameras_dir = output_root / "cameras"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    cameras_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    for record in discover_ase_scenes(root):
        if scene_ids is not None and record.scene_id not in scene_ids:
            continue
        cache_path = scenes_dir / f"{record.scene_id}.npz"
        camera_cache_path = cameras_dir / f"{record.scene_id}_cameras.npz"
        item = {
            "scene_id": record.scene_id,
            "cache_path": str(cache_path),
            "camera_cache_path": str(camera_cache_path),
            "written": False,
            "camera_cache_written": False,
            "skipped": False,
            "error": None,
            "warnings": [],
        }
        if not record.valid:
            item["error"] = "; ".join(record.warnings)
            results.append(item)
            continue
        if (
            cache_path.exists()
            and camera_cache_path.exists()
            and not overwrite
        ):
            item["skipped"] = True
            results.append(item)
            continue

        try:
            scene = load_ase_scene_gaussians(record)
            cameras = read_ase_cameras(record.transforms_path, scene_id=record.scene_id)
            if overwrite or not camera_cache_path.exists():
                save_ase_camera_cache(cameras, camera_cache_path)
                item["camera_cache_written"] = True
            quantized = quantize_scene_with_minkowski(
                scene, voxel_size=voxel_size, seed=seed
            )
            voxel_features = build_scene_voxel_features(
                scene,
                quantized["scene_origin"],
                quantized["scene_coords"],
                quantized["selected_indices"],
                voxel_size=voxel_size,
            )
            ply_xyz_min = scene.xyz.min(axis=0).astype(np.float32)
            ply_xyz_max = scene.xyz.max(axis=0).astype(np.float32)
            cache_world_bbox_min, cache_world_bbox_max = _cache_world_bbox(
                quantized["scene_origin"],
                voxel_features["coords"],
                voxel_size,
            )
            bbox_warnings = _bbox_sanity_warnings(
                cameras.bbox_min,
                cameras.bbox_max,
                ply_xyz_min,
                ply_xyz_max,
            )
            item["warnings"].extend(bbox_warnings)
            for warning in bbox_warnings:
                print(f"WARNING scene {record.scene_id}: {warning}")
            metadata = {
                "scene_id": record.scene_id,
                "scene_dir": str(record.scene_dir),
                "ply_path": str(record.ply_path),
                "transforms_path": str(record.transforms_path),
                "camera_cache_path": str(camera_cache_path),
                "num_input_gaussians": int(scene.xyz.shape[0]),
                "num_voxels": int(voxel_features["coords"].shape[0]),
                "voxel_size": float(voxel_size),
                "feature_representation": "gaussiangpt_ae_v1_exp_ply_scale",
                "color_representation": "rgb_from_sh_dc_clamped_0_1",
                "opacity_representation": "logit_clamped_-10_10",
                "scale_representation": "world_size_from_exp_ply_log_scale",
                "rotation_representation": "unit_quaternion_w_positive",
                "offset_representation": "world_offset_from_voxel_center",
                "feature_scales": dict(DEFAULT_FEATURE_SCALES),
                "json_bbox_min": _array_to_list_or_none(cameras.bbox_min),
                "json_bbox_max": _array_to_list_or_none(cameras.bbox_max),
                "ply_xyz_min": ply_xyz_min.tolist(),
                "ply_xyz_max": ply_xyz_max.tolist(),
                "cache_world_bbox_min": cache_world_bbox_min.tolist(),
                "cache_world_bbox_max": cache_world_bbox_max.tolist(),
                # JSON bbox is the ASE scene-level world bbox. Sampled chunk bboxes
                # must come from sampled voxel coords, scene_origin, and voxel_size.
                "bbox_semantics": "json bbox is scene-level world bbox, not sampled chunk bbox",
                "bbox_warnings": bbox_warnings,
            }
            if overwrite or not cache_path.exists():
                np.savez_compressed(
                    cache_path,
                    scene_coords=voxel_features["coords"],
                    scene_feats=voxel_features["feats"],
                    selected_global_indices=voxel_features["selected_global_indices"],
                    scene_origin=quantized["scene_origin"],
                    voxel_size=np.asarray(float(voxel_size), dtype=np.float32),
                    metadata_json=_metadata_json(metadata),
                )
                item["written"] = True
        except Exception as exc:
            item["error"] = str(exc)
        results.append(item)

    summary = build_ase_voxel_cache_summary(output_root, scene_ids=scene_ids)
    summary["failed_scenes"] = sum(1 for item in results if item["error"] is not None)
    summary["requested_scenes"] = scene_ids
    summary["processed_scenes"] = len(results)
    summary["newly_written_scenes"] = sum(1 for item in results if item["written"])
    summary["newly_written_camera_caches"] = sum(
        1 for item in results if item["camera_cache_written"]
    )
    summary["skipped_scenes"] = sum(1 for item in results if item["skipped"])
    summary["build_results"] = results
    return summary


def load_ase_voxel_cache(scene_cache_path: Union[str, Path]) -> Dict:
    """Load one ASE scene-level voxel cache npz."""

    scene_cache_path = Path(scene_cache_path)
    with np.load(scene_cache_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return {
            "scene_id": metadata["scene_id"],
            "scene_origin": np.asarray(data["scene_origin"], dtype=np.float32),
            "scene_coords": np.asarray(data["scene_coords"], dtype=np.int32),
            "scene_feats": np.asarray(data["scene_feats"], dtype=np.float32),
            "selected_global_indices": np.asarray(
                data["selected_global_indices"], dtype=np.int64
            ),
            "voxel_size": float(np.asarray(data["voxel_size"]).item()),
            "metadata": metadata,
        }


def build_ase_voxel_cache_summary(
    output_root: Union[str, Path],
    scene_ids: Optional[List[str]] = None,
) -> Dict:
    """Summarize an ASE voxel cache output directory."""

    output_root = Path(output_root)
    cache_paths = sorted((output_root / "scenes").glob("*.npz"))
    if scene_ids is not None:
        wanted = set(scene_ids)
        cache_paths = [path for path in cache_paths if path.stem in wanted]
    num_voxels: List[int] = []
    voxel_sizes: List[float] = []
    for path in cache_paths:
        try:
            cache = load_ase_voxel_cache(path)
        except Exception:
            continue
        num_voxels.append(int(cache["scene_coords"].shape[0]))
        voxel_sizes.append(float(cache["voxel_size"]))

    return {
        "total_scenes": len(cache_paths),
        "written_scenes": len(num_voxels),
        "failed_scenes": 0,
        "voxel_size": voxel_sizes[0] if voxel_sizes else None,
        "min_num_voxels": int(min(num_voxels)) if num_voxels else None,
        "max_num_voxels": int(max(num_voxels)) if num_voxels else None,
        "mean_num_voxels": float(sum(num_voxels) / len(num_voxels))
        if num_voxels
        else None,
    }
