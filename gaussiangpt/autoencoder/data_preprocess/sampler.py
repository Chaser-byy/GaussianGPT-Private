"""Online ASE chunk sampler backed by precomputed voxel caches."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .ase import ASECameras, load_ase_camera_cache
from .voxelize import load_ase_voxel_cache


def _bbox_corners(chunk_world_min: np.ndarray, chunk_world_max: np.ndarray) -> np.ndarray:
    lo = np.asarray(chunk_world_min, dtype=np.float32)
    hi = np.asarray(chunk_world_max, dtype=np.float32)
    corners = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                corners.append([x, y, z])
    return np.asarray(corners, dtype=np.float32)


def _zero_camera_score(frame: Dict) -> Dict:
    return {
        "camera_id": frame.get("camera_id"),
        "frame_index": frame.get("frame_index"),
        "frame_id": frame.get("frame_id"),
        "file_path": frame.get("file_path"),
        "chunk_coverage": 0.0,
        "image_coverage": 0.0,
        "visible_ratio": 0.0,
        "visible_corners": 0,
        "total_corners": 8,
        "valid_projection": False,
        "projected_bbox": None,
        "projected_bbox_area": 0.0,
        "intersection_area": 0.0,
        "depth_min": None,
        "depth_max": None,
        "near_plane_crossing": False,
        "selection_mode": None,
    }


def _attach_camera_pose(score: Dict, frame: Dict, cameras: ASECameras) -> Dict:
    """Attach pose/intrinsics when training will render from scored ASE views."""

    c2w = frame.get("c2w", frame.get("transform_matrix"))
    if c2w is None:
        return score
    c2w = np.asarray(c2w, dtype=np.float32)
    w2c = frame.get("w2c")
    if w2c is None:
        w2c = np.linalg.inv(c2w).astype(np.float32)
    score.update(
        {
            "c2w": c2w.astype(np.float32, copy=True),
            "w2c": np.asarray(w2c, dtype=np.float32).astype(np.float32, copy=True),
            "width": int(cameras.width),
            "height": int(cameras.height),
            "fx": float(cameras.fx),
            "fy": float(cameras.fy),
            "cx": float(cameras.cx),
            "cy": float(cameras.cy),
        }
    )
    return score


def score_cameras_for_chunk(
    cameras: ASECameras,
    chunk_world_min: np.ndarray,
    chunk_world_max: np.ndarray,
    top_k: int = 12,
    include_camera_matrices: bool = False,
) -> List[Dict]:
    """Score cameras by projected chunk bbox overlap with the image plane.

    chunk_coverage follows the GaussianGPT ASE heuristic: it is the fraction of
    the projected chunk bbox area contained by the camera image window. The
    projected bbox is built from the chunk corners that are in front of the
    camera and finite after projection. This is geometric scoring only; no
    renderer visibility is implied.
    """

    corners = _bbox_corners(chunk_world_min, chunk_world_max)
    corners_h = np.concatenate([corners, np.ones((8, 1), dtype=np.float32)], axis=1)
    image_area = float(cameras.width * cameras.height)
    scores: List[Dict] = []

    for frame in cameras.frames:
        try:
            if frame.get("w2c") is not None:
                world_to_camera = np.asarray(frame["w2c"], dtype=np.float32)
            else:
                world_to_camera = np.linalg.inv(
                    np.asarray(frame["transform_matrix"], dtype=np.float32)
                ).astype(np.float32)
            camera_points = (world_to_camera @ corners_h.T).T[:, :3]
            depth = camera_points[:, 2]
            depth_min = float(np.min(depth))
            depth_max = float(np.max(depth))
            near_plane_crossing = bool(depth_min <= 1e-6 < depth_max)
            front = depth > 1e-6
            if not np.any(front):
                zero_score = _zero_camera_score(frame)
                zero_score["depth_min"] = depth_min
                zero_score["depth_max"] = depth_max
                if include_camera_matrices:
                    zero_score = _attach_camera_pose(zero_score, frame, cameras)
                scores.append(zero_score)
                continue

            visible_points = camera_points[front]
            u = cameras.fx * (visible_points[:, 0] / visible_points[:, 2]) + cameras.cx
            v = cameras.fy * (visible_points[:, 1] / visible_points[:, 2]) + cameras.cy
            finite = np.isfinite(u) & np.isfinite(v)
            if not np.any(finite):
                zero_score = _zero_camera_score(frame)
                zero_score["depth_min"] = depth_min
                zero_score["depth_max"] = depth_max
                zero_score["near_plane_crossing"] = near_plane_crossing
                if include_camera_matrices:
                    zero_score = _attach_camera_pose(zero_score, frame, cameras)
                scores.append(zero_score)
                continue
            u = u[finite]
            v = v[finite]
            inside_image = (u >= 0.0) & (u <= float(cameras.width)) & (v >= 0.0) & (
                v <= float(cameras.height)
            )
            visible_corners = int(np.count_nonzero(inside_image))
            total_corners = int(corners.shape[0])
            visible_ratio = float(visible_corners / total_corners)

            x_min = float(np.min(u))
            x_max = float(np.max(u))
            y_min = float(np.min(v))
            y_max = float(np.max(v))
            projected_bbox = [x_min, y_min, x_max, y_max]
            projected_bbox_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
            if projected_bbox_area <= 0.0:
                zero_score = _zero_camera_score(frame)
                zero_score["visible_ratio"] = visible_ratio
                zero_score["visible_corners"] = visible_corners
                zero_score["total_corners"] = total_corners
                zero_score["projected_bbox"] = projected_bbox
                zero_score["depth_min"] = depth_min
                zero_score["depth_max"] = depth_max
                zero_score["near_plane_crossing"] = near_plane_crossing
                if include_camera_matrices:
                    zero_score = _attach_camera_pose(zero_score, frame, cameras)
                scores.append(zero_score)
                continue

            inter_x_min = max(0.0, x_min)
            inter_x_max = min(float(cameras.width), x_max)
            inter_y_min = max(0.0, y_min)
            inter_y_max = min(float(cameras.height), y_max)
            intersection_area = max(0.0, inter_x_max - inter_x_min) * max(
                0.0, inter_y_max - inter_y_min
            )
            chunk_coverage = float(intersection_area / projected_bbox_area)
            image_coverage = (
                float(intersection_area / image_area) if image_area > 0.0 else 0.0
            )

            score = {
                "camera_id": frame.get("camera_id"),
                "frame_index": frame.get("frame_index"),
                "frame_id": frame.get("frame_id"),
                "file_path": frame.get("file_path"),
                "chunk_coverage": chunk_coverage,
                "image_coverage": image_coverage,
                "visible_ratio": visible_ratio,
                "visible_corners": visible_corners,
                "total_corners": total_corners,
                "valid_projection": True,
                "projected_bbox": projected_bbox,
                "projected_bbox_area": float(projected_bbox_area),
                "intersection_area": float(intersection_area),
                "depth_min": depth_min,
                "depth_max": depth_max,
                "near_plane_crossing": near_plane_crossing,
                "selection_mode": None,
            }
            if include_camera_matrices:
                score = _attach_camera_pose(score, frame, cameras)
            scores.append(score)
        except Exception:
            zero_score = _zero_camera_score(frame)
            if include_camera_matrices:
                zero_score = _attach_camera_pose(zero_score, frame, cameras)
            scores.append(zero_score)

    return sorted(
        scores,
        key=lambda item: (
            item["chunk_coverage"],
            item["intersection_area"],
            item["image_coverage"],
        ),
        reverse=True,
    )[:top_k]


def select_cameras_for_chunk(
    cameras: ASECameras,
    chunk_world_min: np.ndarray,
    chunk_world_max: np.ndarray,
    top_k: int = 12,
    preferred_coverage: float = 0.4,
    include_camera_matrices: bool = False,
) -> List[Dict]:
    """Select cameras for a chunk and annotate why each camera was selected."""

    scores = score_cameras_for_chunk(
        cameras,
        chunk_world_min,
        chunk_world_max,
        top_k=len(cameras.frames),
        include_camera_matrices=include_camera_matrices,
    )
    preferred = [
        dict(score, selection_mode="preferred")
        for score in scores
        if score["valid_projection"] and score["chunk_coverage"] >= preferred_coverage
    ]
    fallback_overlap = [
        dict(score, selection_mode="fallback_overlap")
        for score in scores
        if score["valid_projection"]
        and score["intersection_area"] > 0.0
        and score["frame_index"] not in {item["frame_index"] for item in preferred}
    ]
    selected = (preferred + fallback_overlap)[:top_k]
    if selected:
        return selected

    return [
        dict(score, selection_mode="no_overlap_topk")
        for score in scores[:top_k]
    ]


def build_chunk_from_scene_cache(
    scene_cache: Dict,
    chunk_min_voxel: np.ndarray,
    chunk_shape_voxels: np.ndarray,
) -> Dict:
    """Build chunk-local sparse tensors from a scene-level voxel cache."""

    scene_coords = np.asarray(scene_cache["scene_coords"], dtype=np.int32)
    scene_feats = np.asarray(scene_cache["scene_feats"], dtype=np.float32)
    selected_global_indices = np.asarray(
        scene_cache["selected_global_indices"], dtype=np.int64
    )
    chunk_min_voxel = np.asarray(chunk_min_voxel, dtype=np.int32)
    chunk_shape_voxels = np.asarray(chunk_shape_voxels, dtype=np.int32)
    chunk_max_voxel = chunk_min_voxel + chunk_shape_voxels
    inside = np.all(
        (scene_coords >= chunk_min_voxel) & (scene_coords < chunk_max_voxel),
        axis=1,
    )

    coords = (scene_coords[inside] - chunk_min_voxel).astype(np.int32, copy=False)
    feats = scene_feats[inside].astype(np.float32, copy=False)
    selected = selected_global_indices[inside].astype(np.int64, copy=False)
    return {
        "coords": coords,
        "feats": feats,
        "target_feats": feats.copy(),
        "selected_global_indices": selected,
        "num_occupied_voxels": int(coords.shape[0]),
    }


class ASEOnlineChunkSampler:
    """Randomly sample occupied chunks from scene-level voxel caches."""

    def __init__(
        self,
        cache_root: Union[str, Path],
        chunk_size: float = 4.0,
        occupancy_threshold: float = 0.2,
        max_candidate_chunks: int = 10,
        top_k_cameras: int = 12,
        seed: int = 42,
        z_mode: str = "fixed_160",
        preferred_coverage: float = 0.4,
        include_camera_matrices: bool = False,
        scene_id: Optional[str] = None,
        scene_ids: Optional[List[str]] = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.scene_cache_paths = sorted((self.cache_root / "scenes").glob("*.npz"))
        if scene_id is not None and scene_ids is not None:
            raise ValueError("pass either scene_id or scene_ids, not both")
        selected_scene_ids = [scene_id] if scene_id is not None else scene_ids
        if selected_scene_ids is not None:
            wanted = set(selected_scene_ids)
            self.scene_cache_paths = [
                path for path in self.scene_cache_paths if path.stem in wanted
            ]
        if not self.scene_cache_paths:
            detail = (
                f" for scene_ids={selected_scene_ids}"
                if selected_scene_ids is not None
                else ""
            )
            raise ValueError(
                f"no ASE voxel cache files found under {self.cache_root}/scenes{detail}"
            )

        first_cache = load_ase_voxel_cache(self.scene_cache_paths[0])
        self.voxel_size = float(first_cache["voxel_size"])
        self.chunk_size = float(chunk_size)
        self.occupancy_threshold = float(occupancy_threshold)
        self.max_candidate_chunks = int(max_candidate_chunks)
        self.top_k_cameras = int(top_k_cameras)
        self.seed = int(seed)
        self.z_mode = z_mode
        self.scene_ids = selected_scene_ids
        self.preferred_coverage = float(preferred_coverage)
        self.include_camera_matrices = bool(include_camera_matrices)
        self.rng = np.random.RandomState(self.seed)
        if self.z_mode not in {"fixed_160", "full_height"}:
            raise ValueError("z_mode must be 'fixed_160' or 'full_height'")

        self.xy_voxels = int(round(self.chunk_size / self.voxel_size))
        if self.xy_voxels <= 0:
            raise ValueError("chunk_size / voxel_size must round to a positive integer")
        self.chunk_shape_voxels = np.asarray(
            [self.xy_voxels, self.xy_voxels, self.xy_voxels],
            dtype=np.int32,
        )
        self._scene_cache: Dict[str, Dict] = {str(self.scene_cache_paths[0]): first_cache}
        self._camera_cache: Dict[str, object] = {}

    def set_seed(self, seed: int) -> None:
        """Reset the sampler RNG.

        DataLoader workers inherit the parent process state under fork, so each
        worker must get its own RNG stream before online chunk sampling starts.
        """
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

    def _load_scene_cache(self, path: Path) -> Dict:
        key = str(path)
        if key not in self._scene_cache:
            self._scene_cache[key] = load_ase_voxel_cache(path)
        return self._scene_cache[key]

    def _camera_cache_path(self, metadata: Dict) -> Path:
        if metadata.get("camera_cache_path"):
            return Path(metadata["camera_cache_path"])
        return self.cache_root / "cameras" / f"{metadata['scene_id']}_cameras.npz"

    def _load_cameras(self, metadata: Dict):
        camera_cache_path = self._camera_cache_path(metadata)
        key = str(camera_cache_path)
        if key not in self._camera_cache:
            if not camera_cache_path.is_file():
                raise FileNotFoundError(
                    f"missing ASE camera cache {camera_cache_path}; rebuild the ASE cache"
                )
            self._camera_cache[key] = load_ase_camera_cache(
                camera_cache_path, scene_id=metadata["scene_id"]
            )
        return self._camera_cache[key]

    def _scene_z_info(self, scene_coords: np.ndarray) -> Dict:
        z_min = int(scene_coords[:, 2].min())
        z_max = int(scene_coords[:, 2].max()) + 1
        z_voxels = self.xy_voxels if self.z_mode == "fixed_160" else z_max - z_min
        return {
            "scene_z_min_voxel": z_min,
            "scene_z_max_voxel": z_max,
            "scene_z_voxels": z_max - z_min,
            "chunk_shape_voxels": np.asarray(
                [self.xy_voxels, self.xy_voxels, z_voxels], dtype=np.int32
            ),
        }

    def _sample_chunk_min(
        self, scene_coords: np.ndarray, chunk_shape_voxels: np.ndarray
    ) -> np.ndarray:
        coord_min = scene_coords.min(axis=0).astype(np.int32)
        coord_max = scene_coords.max(axis=0).astype(np.int32)
        chunk_min = np.zeros(3, dtype=np.int32)
        for axis in (0, 1):
            low = int(coord_min[axis])
            high = int(coord_max[axis] - chunk_shape_voxels[axis] + 1)
            if high < low:
                high = low
            chunk_min[axis] = int(self.rng.randint(low, high + 1))
        chunk_min[2] = int(coord_min[2])
        return chunk_min

    def _chunk_occupancy(
        self,
        scene_coords: np.ndarray,
        chunk_min_voxel: np.ndarray,
        chunk_shape_voxels: np.ndarray,
    ) -> float:
        chunk_max_voxel = chunk_min_voxel + chunk_shape_voxels
        inside = np.all(
            (scene_coords >= chunk_min_voxel) & (scene_coords < chunk_max_voxel),
            axis=1,
        )
        total_voxels = int(np.prod(chunk_shape_voxels))
        return float(np.count_nonzero(inside) / total_voxels) if total_voxels > 0 else 0.0

    def _choose_chunk(self, scene_coords: np.ndarray, chunk_shape_voxels: np.ndarray) -> tuple:
        best_chunk_min: Optional[np.ndarray] = None
        best_occupancy = -1.0
        candidate_occupancies = []
        for _ in range(self.max_candidate_chunks):
            candidate = self._sample_chunk_min(scene_coords, chunk_shape_voxels)
            occupancy = self._chunk_occupancy(scene_coords, candidate, chunk_shape_voxels)
            candidate_occupancies.append(float(occupancy))
            if occupancy > best_occupancy:
                best_chunk_min = candidate
                best_occupancy = occupancy
            if occupancy >= self.occupancy_threshold:
                return candidate, occupancy, True, candidate_occupancies, best_occupancy
        if best_chunk_min is None:
            best_chunk_min = scene_coords.min(axis=0).astype(np.int32)
            best_occupancy = self._chunk_occupancy(
                scene_coords, best_chunk_min, chunk_shape_voxels
            )
        return best_chunk_min, best_occupancy, False, candidate_occupancies, best_occupancy

    def sample(self) -> Dict:
        """Sample one random chunk from a random scene cache."""

        cache_path = self.scene_cache_paths[int(self.rng.randint(0, len(self.scene_cache_paths)))]
        scene_cache = self._load_scene_cache(cache_path)
        scene_coords = scene_cache["scene_coords"]
        scene_origin = scene_cache["scene_origin"]
        metadata = scene_cache["metadata"]

        z_info = self._scene_z_info(scene_coords)
        chunk_shape_voxels = z_info["chunk_shape_voxels"]
        (
            chunk_min_voxel,
            occupancy,
            accepted_by_threshold,
            candidate_occupancies,
            best_candidate_occupancy,
        ) = self._choose_chunk(scene_coords, chunk_shape_voxels)
        chunk_max_voxel = chunk_min_voxel + chunk_shape_voxels
        chunk = build_chunk_from_scene_cache(
            scene_cache, chunk_min_voxel, chunk_shape_voxels
        )

        voxel_size = float(scene_cache["voxel_size"])
        chunk_world_min = scene_origin + chunk_min_voxel.astype(np.float32) * voxel_size
        chunk_world_max = scene_origin + chunk_max_voxel.astype(np.float32) * voxel_size

        transforms_path = metadata["transforms_path"]
        cameras = self._load_cameras(metadata)
        top_cameras = select_cameras_for_chunk(
            cameras,
            chunk_world_min,
            chunk_world_max,
            top_k=self.top_k_cameras,
            preferred_coverage=self.preferred_coverage,
            include_camera_matrices=self.include_camera_matrices,
        )

        return {
            "scene_id": metadata["scene_id"],
            "ply_path": metadata["ply_path"],
            "transforms_path": transforms_path,
            "coords": chunk["coords"],
            "feats": chunk["feats"],
            "target_feats": chunk["target_feats"],
            "selected_global_indices": chunk["selected_global_indices"],
            "scene_origin": scene_origin,
            "chunk_min_voxel": chunk_min_voxel.astype(np.int32, copy=False),
            "chunk_max_voxel": chunk_max_voxel.astype(np.int32, copy=False),
            "chunk_world_min": chunk_world_min.astype(np.float32, copy=False),
            "chunk_world_max": chunk_world_max.astype(np.float32, copy=False),
            "voxel_size": voxel_size,
            "chunk_shape_voxels": chunk_shape_voxels.copy(),
            "occupancy": float(occupancy),
            "num_occupied_voxels": int(chunk["num_occupied_voxels"]),
            "num_gaussians_after_voxel_dedup": int(scene_coords.shape[0]),
            "top_cameras": top_cameras,
            "camera_debug": {
                "camera_cache_path": str(self._camera_cache_path(metadata)),
                "pose_convention": cameras.pose_convention,
                "uses_transform_device_camera": cameras.uses_transform_device_camera,
                "top_cameras": top_cameras,
            },
            "z_mode": self.z_mode,
            "scene_z_min_voxel": int(z_info["scene_z_min_voxel"]),
            "scene_z_max_voxel": int(z_info["scene_z_max_voxel"]),
            "scene_z_voxels": int(z_info["scene_z_voxels"]),
            "accepted_by_threshold": bool(accepted_by_threshold),
            "candidate_occupancies": candidate_occupancies,
            "best_candidate_occupancy": float(best_candidate_occupancy),
        }
