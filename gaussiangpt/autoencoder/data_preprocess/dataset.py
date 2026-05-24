"""PyTorch Dataset wrapper for online ASE chunk sampling."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Union

from .sampler import ASEOnlineChunkSampler


class ASEChunkDataset:
    """Map-style dataset that samples ASE chunks online from voxel caches."""

    def __init__(
        self,
        cache_root: Union[str, Path],
        num_samples_per_epoch: int = 1000,
        chunk_size: float = 4.0,
        occupancy_threshold: float = 0.2,
        max_candidate_chunks: int = 10,
        top_k_cameras: int = 12,
        seed: int = 42,
        z_mode: str = "fixed_160",
        preferred_coverage: float = 0.4,
        include_camera_matrices: bool = False,
        scene_ids: Optional[List[str]] = None,
        fixed_chunk: bool = False,
        fixed_sample: Optional[Dict] = None,
    ) -> None:
        self.num_samples_per_epoch = int(num_samples_per_epoch)
        self._worker_seed: Optional[int] = None
        self.fixed_chunk = bool(fixed_chunk or fixed_sample is not None)
        self._fixed_sample: Optional[Dict] = (
            copy.deepcopy(fixed_sample) if fixed_sample is not None else None
        )
        self.sampler = None
        if self._fixed_sample is None:
            self.sampler = ASEOnlineChunkSampler(
                cache_root=cache_root,
                chunk_size=chunk_size,
                occupancy_threshold=occupancy_threshold,
                max_candidate_chunks=max_candidate_chunks,
                top_k_cameras=top_k_cameras,
                seed=seed,
                z_mode=z_mode,
                preferred_coverage=preferred_coverage,
                include_camera_matrices=include_camera_matrices,
                scene_ids=scene_ids,
            )
        if self.fixed_chunk and self._fixed_sample is None:
            self._fixed_sample = self.sampler.sample()

    def __len__(self) -> int:
        return self.num_samples_per_epoch

    def _maybe_seed_worker(self) -> None:
        try:
            from torch.utils.data import get_worker_info
        except ImportError:
            return

        worker_info = get_worker_info()
        if worker_info is None:
            return

        if self.sampler is None:
            return

        worker_seed = int(worker_info.seed % (2**32))
        if self._worker_seed != worker_seed:
            self.sampler.set_seed(worker_seed)
            self._worker_seed = worker_seed

    def __getitem__(self, idx: int) -> Dict:
        del idx
        if self._fixed_sample is None:
            self._maybe_seed_worker()
            sample = self.sampler.sample()
        else:
            sample = copy.deepcopy(self._fixed_sample)
        sample["metadata"] = {
            "scene_id": sample["scene_id"],
            "ply_path": sample["ply_path"],
            "transforms_path": sample["transforms_path"],
            "chunk_min_voxel": sample["chunk_min_voxel"],
            "chunk_max_voxel": sample["chunk_max_voxel"],
            "chunk_shape_voxels": sample["chunk_shape_voxels"],
            "chunk_world_min": sample["chunk_world_min"],
            "chunk_world_max": sample["chunk_world_max"],
            "voxel_size": sample["voxel_size"],
            "occupancy": sample["occupancy"],
            "top_cameras": sample["top_cameras"],
            "camera_debug": sample["camera_debug"],
            "z_mode": sample["z_mode"],
            "accepted_by_threshold": sample["accepted_by_threshold"],
            "candidate_occupancies": sample["candidate_occupancies"],
            "best_candidate_occupancy": sample["best_candidate_occupancy"],
        }
        return sample

    def fixed_sample(self) -> Optional[Dict]:
        """Return the cached fixed chunk sample, if fixed chunk training is enabled."""

        if self._fixed_sample is None:
            return None
        return copy.deepcopy(self._fixed_sample)

    def fixed_chunk_summary(self) -> Optional[str]:
        """Human-readable description of the fixed chunk used by this dataset."""

        if self._fixed_sample is None:
            return None
        sample = self._fixed_sample
        return (
            f"scene_id={sample['scene_id']} "
            f"chunk_min={sample['chunk_min_voxel'].tolist()} "
            f"chunk_max={sample['chunk_max_voxel'].tolist()} "
            f"shape={sample['chunk_shape_voxels'].tolist()} "
            f"occupancy={float(sample['occupancy']):.6f} "
            f"accepted={bool(sample['accepted_by_threshold'])}"
        )
