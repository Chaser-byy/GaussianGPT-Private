"""Online ASE chunk sampler backed by precomputed voxel caches."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .ase import ASECameras, load_ase_camera_cache
from .voxelize import load_ase_voxel_cache


FRUSTUM_SCORE_KEY = "frustum_image_coverage"


def _zero_camera_score(frame: Dict) -> Dict:
    return {
        "camera_id": frame.get("camera_id"),
        "frame_index": frame.get("frame_index"),
        "frame_id": frame.get("frame_id"),
        "file_path": frame.get("file_path"),
        "frustum_image_coverage": 0.0,
        # Backward-compatible aliases for older configs/log readers.
        "chunk_coverage": 0.0,
        "image_coverage": 0.0,
        "projected_area": 0.0,
        "image_area": 0.0,
        "num_intersection_vertices": 0,
        "has_overlap": False,
        "valid_projection": False,
        "selection_mode": "frustum_image_coverage",
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


def _aabb_world_planes_to_camera(
    chunk_world_min: np.ndarray,
    chunk_world_max: np.ndarray,
    w2c: np.ndarray,
) -> np.ndarray:
    """Return camera-space AABB halfspaces as rows [a, b, c, d].

    Each plane is ``normal dot point_cam + offset >= 0``.
    """

    c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    lo = np.asarray(chunk_world_min, dtype=np.float64)
    hi = np.asarray(chunk_world_max, dtype=np.float64)
    world_planes = [
        ([1.0, 0.0, 0.0], -lo[0]),
        ([-1.0, 0.0, 0.0], hi[0]),
        ([0.0, 1.0, 0.0], -lo[1]),
        ([0.0, -1.0, 0.0], hi[1]),
        ([0.0, 0.0, 1.0], -lo[2]),
        ([0.0, 0.0, -1.0], hi[2]),
    ]
    planes = []
    for normal_world, offset_world in world_planes:
        normal_world = np.asarray(normal_world, dtype=np.float64)
        normal_cam = rotation.T @ normal_world
        offset_cam = float(normal_world @ translation + offset_world)
        planes.append(np.concatenate([normal_cam, [offset_cam]]))
    return np.asarray(planes, dtype=np.float64)


def _camera_frustum_planes(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float = 1e-4,
    far: Optional[float] = None,
) -> np.ndarray:
    planes = [
        [0.0, 0.0, 1.0, -float(near)],
        [float(fx), 0.0, float(cx), 0.0],
        [-float(fx), 0.0, float(width) - float(cx), 0.0],
        [0.0, float(fy), float(cy), 0.0],
        [0.0, -float(fy), float(height) - float(cy), 0.0],
    ]
    if far is not None:
        planes.append([0.0, 0.0, -1.0, float(far)])
    return np.asarray(planes, dtype=np.float64)


def _intersect_convex_polyhedron_vertices(
    planes: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    vertices = []
    for indices in combinations(range(int(planes.shape[0])), 3):
        subset = planes[list(indices)]
        A = subset[:, :3]
        b = subset[:, 3]
        try:
            point = np.linalg.solve(A, -b)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(point)):
            continue
        if np.all((planes[:, :3] @ point + planes[:, 3]) >= -eps):
            vertices.append(point)
    if not vertices:
        return np.zeros((0, 3), dtype=np.float64)

    unique = []
    for point in vertices:
        if not any(np.linalg.norm(point - existing) <= 1e-5 for existing in unique):
            unique.append(point)
    return np.asarray(unique, dtype=np.float64)


def _project_cam_points(
    points_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float = 1e-8,
) -> np.ndarray:
    points_cam = np.asarray(points_cam, dtype=np.float64)
    valid = points_cam[:, 2] > float(near)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float64)
    points_cam = points_cam[valid]
    u = float(fx) * points_cam[:, 0] / points_cam[:, 2] + float(cx)
    v = float(fy) * points_cam[:, 1] / points_cam[:, 2] + float(cy)
    uv = np.stack([u, v], axis=1)
    return uv[np.all(np.isfinite(uv), axis=1)]


def _convex_hull_area_2d(points_uv: np.ndarray) -> float:
    points = sorted({(float(p[0]), float(p[1])) for p in np.asarray(points_uv)})
    if len(points) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-9:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-9:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0

    area = 0.0
    for idx, point in enumerate(hull):
        nxt = hull[(idx + 1) % len(hull)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(area) * 0.5


def score_camera_for_chunk_frustum_image_coverage(
    frame: Dict,
    cameras: ASECameras,
    chunk_world_min: np.ndarray,
    chunk_world_max: np.ndarray,
    near: float = 1e-4,
) -> Dict:
    """Score the projected visible chunk-frustum intersection area."""

    score = _zero_camera_score(frame)
    image_area = float(cameras.width * cameras.height)
    score["image_area"] = image_area
    try:
        if frame.get("w2c") is not None:
            world_to_camera = np.asarray(frame["w2c"], dtype=np.float64)
        else:
            world_to_camera = np.linalg.inv(
                np.asarray(frame["transform_matrix"], dtype=np.float64)
            )
        aabb_planes = _aabb_world_planes_to_camera(
            chunk_world_min, chunk_world_max, world_to_camera
        )
        frustum_planes = _camera_frustum_planes(
            cameras.fx,
            cameras.fy,
            cameras.cx,
            cameras.cy,
            cameras.width,
            cameras.height,
            near=near,
        )
        planes = np.concatenate([aabb_planes, frustum_planes], axis=0)
        vertices_cam = _intersect_convex_polyhedron_vertices(planes)
        score["num_intersection_vertices"] = int(vertices_cam.shape[0])
        if vertices_cam.shape[0] < 3:
            return score

        uv = _project_cam_points(
            vertices_cam,
            cameras.fx,
            cameras.fy,
            cameras.cx,
            cameras.cy,
            near=near,
        )
        if uv.shape[0] < 3:
            return score

        projected_area = _convex_hull_area_2d(uv)
        coverage = projected_area / image_area if image_area > 0.0 else 0.0
        coverage = float(np.clip(coverage, 0.0, 1.0))
        score.update(
            {
                "frustum_image_coverage": coverage,
                "chunk_coverage": coverage,
                "image_coverage": coverage,
                "projected_area": float(projected_area),
                "has_overlap": coverage > 0.0,
                "valid_projection": coverage > 0.0,
                "selection_mode": "frustum_image_coverage",
            }
        )
        return score
    except Exception as exc:
        score["error"] = str(exc)
        return score


def score_cameras_for_chunk(
    cameras: ASECameras,
    chunk_world_min: np.ndarray,
    chunk_world_max: np.ndarray,
    top_k: int = 0,
    include_camera_matrices: bool = False,
) -> List[Dict]:
    """Score all ASE cameras by chunk-frustum projected image coverage."""

    del top_k
    scores: List[Dict] = []
    for frame in cameras.frames:
        score = score_camera_for_chunk_frustum_image_coverage(
            frame, cameras, chunk_world_min, chunk_world_max
        )
        if include_camera_matrices:
            score = _attach_camera_pose(score, frame, cameras)
        scores.append(score)

    return sorted(
        scores,
        key=lambda item: (
            item[FRUSTUM_SCORE_KEY],
            item["projected_area"],
            item["num_intersection_vertices"],
        ),
        reverse=True,
    )


def select_cameras_for_chunk(
    cameras: ASECameras,
    chunk_world_min: np.ndarray,
    chunk_world_max: np.ndarray,
    top_k: int = 12,
    preferred_coverage: float = 0.4,
    include_camera_matrices: bool = False,
) -> List[Dict]:
    """Return frustum-image coverage scores for every ASE camera.

    `top_k` and `preferred_coverage` are accepted for compatibility with older
    configs, but final preferred/fallback selection happens in training.
    """

    del top_k, preferred_coverage
    return score_cameras_for_chunk(
        cameras,
        chunk_world_min,
        chunk_world_max,
        top_k=0,
        include_camera_matrices=include_camera_matrices,
    )


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
                "scoring_dataset_type": "ase",
                "score_key": FRUSTUM_SCORE_KEY,
                "selection_mode": "frustum_image_coverage",
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
