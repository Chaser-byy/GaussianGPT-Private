"""Dataset classes for GaussianGPT training.

Supports:
  - GaussianSceneDataset: loads pre-optimized 3D Gaussian scenes
  - PhotoShapeDataset: loads PhotoShape chair dataset (for object experiments)

Data format expected:
  Each scene is stored as a .pt file containing:
    {
      'positions': (N, 3) float32 world-space Gaussian centers,
      'scales': (N, 3) float32 scale values,
      'opacities': (N, 1) float32 opacity values,
      'rotations': (N, 4) float32 quaternions (w, x, y, z),
      'colors': (N, 3) float32 RGB colors [0, 1],
      'sh': (N, 12) float32 spherical harmonics (optional, degree-1: 4 coeffs x 3 channels),
      'images': (M, 3, H, W) float32 rendered images (optional),
      'cameras': list of camera dicts with 'viewmat' and 'projmat' (optional),
    }
"""
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import random

from gaussiangpt.utils.serialization import ChunkSampler


class GaussianSceneDataset(Dataset):
    """Dataset of pre-optimized 3D Gaussian scenes.

    Voxelizes Gaussians at base_voxel_size and returns chunks.
    Applies 8x augmentation via rotations (0/90/180/270) and reflections.
    """

    def __init__(
        self,
        data_dir: str,
        base_voxel_size: float = 0.025,
        n_down: int = 3,
        chunk_size: Tuple[int, int, int] = (16, 16, 16),
        min_occupancy: float = 0.2,
        augment: bool = True,
        split: str = "train",
        split_ratio: float = 0.9,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.base_voxel_size = base_voxel_size
        self.latent_voxel_size = base_voxel_size * (2 ** n_down)
        self.chunk_size = chunk_size
        self.augment = augment
        self.chunk_sampler = ChunkSampler(chunk_size, min_occupancy=min_occupancy)

        # Find all scene files
        all_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pt") or f.endswith(".pth")
        ])
        n_train = int(len(all_files) * split_ratio)
        if split == "train":
            self.files = all_files[:n_train]
        else:
            self.files = all_files[n_train:]

        # 8x augmentation: 4 rotations x 2 reflections
        self.aug_factor = 8 if augment else 1
        self.n_scenes = len(self.files)

    def __len__(self) -> int:
        return self.n_scenes * self.aug_factor

    def __getitem__(self, idx: int) -> Dict:
        scene_idx = idx % self.n_scenes
        aug_idx = idx // self.n_scenes if self.augment else 0

        data = torch.load(self.files[scene_idx], map_location="cpu")
        gaussians = {
            "offset": data["positions"],
            "scale": data["scales"],
            "opacity": data["opacities"],
            "rotation": data["rotations"],
            "color": data["colors"],
        }
        if "sh" in data:
            gaussians["sh"] = data["sh"]

        # Apply augmentation
        if self.augment:
            gaussians = self._augment(gaussians, aug_idx)

        # Voxelize at latent resolution
        voxel_coords, voxel_gaussians = self._voxelize(gaussians)

        # Sample a chunk
        chunk_result = self.chunk_sampler.sample_chunk(voxel_coords)
        if chunk_result is None:
            # Fallback: return empty chunk
            return self.__getitem__((idx + 1) % len(self))

        chunk_coords, chunk_origin = chunk_result

        # Get Gaussians in this chunk
        flat_in_chunk = self._get_chunk_gaussians(voxel_coords, chunk_coords, chunk_origin, voxel_gaussians)

        result = {
            "voxel_coords": chunk_coords,
            "chunk_origin": chunk_origin,
            **flat_in_chunk,
        }

        if "images" in data and "cameras" in data:
            result["images"] = data["images"]
            result["cameras"] = data["cameras"]

        return result

    def _augment(self, gaussians: Dict, aug_idx: int) -> Dict:
        """Apply rotation/reflection augmentation in the horizontal plane."""
        rot_idx = aug_idx % 4
        flip = aug_idx >= 4

        positions = gaussians["offset"].clone()
        rotations = gaussians["rotation"].clone()

        # Rotate around vertical (z) axis
        angle = rot_idx * math.pi / 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1],
        ], dtype=torch.float32)

        positions = positions @ R.T

        # Rotate quaternions
        # Rotation around z by angle: q_rot = (cos(a/2), 0, 0, sin(a/2))
        half = angle / 2
        q_rot = torch.tensor([math.cos(half), 0, 0, math.sin(half)])
        rotations = self._quat_multiply(q_rot.unsqueeze(0), rotations)

        if flip:
            positions[:, 0] = -positions[:, 0]
            # Flip quaternion x component
            rotations[:, 1] = -rotations[:, 1]
            rotations[:, 3] = -rotations[:, 3]

        gaussians = dict(gaussians)
        gaussians["offset"] = positions
        gaussians["rotation"] = F.normalize(rotations, dim=-1)
        return gaussians

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication (w, x, y, z) convention."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    def _voxelize(
        self, gaussians: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """Assign Gaussians to voxels at base voxel resolution.

        Returns voxel coordinates and per-voxel Gaussian attributes
        (one Gaussian per voxel — first occurrence kept).
        """
        positions = gaussians["offset"]
        voxel_coords = torch.floor(positions / self.base_voxel_size).long()

        # Deduplicate: keep first occurrence per voxel.
        unique_coords, inverse = torch.unique(voxel_coords, dim=0, return_inverse=True)
        n_voxels = unique_coords.shape[0]

        # Vectorized first-occurrence: scatter source indices into first_occ.
        # scatter_ with reduce='amin' picks the minimum (= first) source index per voxel.
        src = torch.arange(len(positions), dtype=torch.long)
        first_occ = torch.full((n_voxels,), len(positions), dtype=torch.long)
        first_occ.scatter_reduce_(0, inverse, src, reduce="amin", include_self=True)

        voxel_gaussians = {key: val[first_occ] for key, val in gaussians.items()}
        return unique_coords, voxel_gaussians

    def _get_chunk_gaussians(
        self,
        voxel_coords: torch.Tensor,
        chunk_coords: torch.Tensor,
        chunk_origin: torch.Tensor,
        voxel_gaussians: Dict,
    ) -> Dict:
        """Extract Gaussians for voxels in the chunk.

        chunk_coords: (M, 3) relative coords within chunk (0..chunk_size-1)
        chunk_origin: (3,) global offset of chunk
        voxel_coords: (N, 3) global voxel coordinates
        """
        abs_chunk_coords = chunk_coords + chunk_origin  # (M, 3) global coords in chunk

        # Build a set of chunk coord tuples for O(N) lookup
        chunk_set = set(map(tuple, abs_chunk_coords.tolist()))

        # Find which global voxels fall inside the chunk
        global_idx = [
            i for i, c in enumerate(voxel_coords.tolist())
            if tuple(c) in chunk_set
        ]
        global_idx = torch.tensor(global_idx, dtype=torch.long)

        return {key: val[global_idx] for key, val in voxel_gaussians.items()}


class TokenizedSceneDataset(Dataset):
    """Dataset that tokenizes scenes using a trained autoencoder.

    Used for GPT training: returns (tokens, coords, token_type) sequences.
    """

    def __init__(
        self,
        data_dir: str,
        autoencoder,
        base_voxel_size: float = 0.025,
        n_down: int = 3,
        chunk_size: Tuple[int, int, int] = (16, 16, 16),
        min_occupancy: float = 0.3,
        augment: bool = True,
        split: str = "train",
        split_ratio: float = 0.9,
        device: str = "cpu",
    ):
        super().__init__()
        self.scene_dataset = GaussianSceneDataset(
            data_dir=data_dir,
            base_voxel_size=base_voxel_size,
            n_down=n_down,
            chunk_size=chunk_size,
            min_occupancy=min_occupancy,
            augment=augment,
            split=split,
            split_ratio=split_ratio,
        )
        self.autoencoder = autoencoder
        self.device = device
        self.chunk_size = chunk_size  # base voxel chunk size
        self.n_down = n_down

        # Latent chunk size: encoder downsamples by 2^n_down in each spatial dim
        self.latent_chunk_size = tuple(c // (2 ** n_down) for c in chunk_size)

        # BOS/EOS match GaussianGPT vocabulary: latent_chunk_vol and latent_chunk_vol+1
        latent_chunk_vol = self.latent_chunk_size[0] * self.latent_chunk_size[1] * self.latent_chunk_size[2]
        self.BOS = latent_chunk_vol
        self.EOS = latent_chunk_vol + 1

    def __len__(self) -> int:
        return len(self.scene_dataset)

    def __getitem__(self, idx: int) -> Dict:
        from gaussiangpt.utils.serialization import serialize_latent_grid
        from gaussiangpt.autoencoder.sparse_cnn import HAS_MINKOWSKI

        batch = self.scene_dataset[idx]
        voxel_coords = batch["voxel_coords"]
        gaussians = {k: v for k, v in batch.items()
                     if k in ("offset", "scale", "opacity", "rotation", "color", "sh")}

        # Encode to latent codes
        with torch.no_grad():
            gaussians_dev = {k: v.to(self.device) for k, v in gaussians.items()}
            voxel_coords_dev = voxel_coords.to(self.device)
            voxel_features = self.autoencoder.attr_encoder(gaussians_dev)

            if HAS_MINKOWSKI:
                import MinkowskiEngine as ME
                batch_idx = torch.zeros(len(voxel_coords_dev), 1, dtype=torch.int, device=self.device)
                coords_me = torch.cat([batch_idx, voxel_coords_dev.int()], dim=1)
                sparse_input = ME.SparseTensor(features=voxel_features, coordinates=coords_me)
                z_sparse = self.autoencoder.encoder(sparse_input)
                _, indices, _ = self.autoencoder.quantizer(z_sparse.F)
                # Use latent-space coordinates for serialization
                latent_coords = z_sparse.C[:, 1:].cpu()  # drop batch dim
            else:
                # Dense fallback: place features in a grid, encode, read back at voxel positions
                cx, cy, cz = self.chunk_size
                C = voxel_features.shape[-1]
                grid = torch.zeros(1, C, cx, cy, cz, device=self.device)
                vc = voxel_coords_dev
                grid[0, :, vc[:, 0], vc[:, 1], vc[:, 2]] = voxel_features.T
                z_grid = self.autoencoder.encoder(grid)  # (1, num_bits, cx', cy', cz')
                # Downsample voxel coords to latent resolution
                scale = 2 ** self.n_down
                latent_vc = (vc // scale).clamp(0, z_grid.shape[2] - 1)
                z_at_voxels = z_grid[0, :, latent_vc[:, 0], latent_vc[:, 1], latent_vc[:, 2]].T
                _, indices, _ = self.autoencoder.quantizer(z_at_voxels)
                latent_coords = latent_vc.cpu()

        # Serialize to token sequence using latent chunk dimensions
        tokens, coords, token_type = serialize_latent_grid(
            voxel_coords=latent_coords,
            voxel_codes=indices.cpu(),
            chunk_size=self.latent_chunk_size,
            BOS=self.BOS,
            EOS=self.EOS,
        )

        return {
            "tokens": tokens,
            "coords": coords,
            "token_type": token_type,
        }


class PhotoShapeDataset(Dataset):
    """PhotoShape chairs dataset for object-level experiments.

    Expects data_dir to contain .pt files with normalized chair Gaussians.
    Objects are discretized on a 128^3 grid with 2 downsampling stages.
    """

    def __init__(
        self,
        data_dir: str,
        grid_size: int = 128,
        split: str = "train",
        split_ratio: float = 0.9,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.grid_size = grid_size

        all_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pt") or f.endswith(".pth")
        ])
        n_train = int(len(all_files) * split_ratio)
        self.files = all_files[:n_train] if split == "train" else all_files[n_train:]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        data = torch.load(self.files[idx], map_location="cpu")
        return {
            "offset": data["positions"],
            "scale": data["scales"],
            "opacity": data["opacities"],
            "rotation": data["rotations"],
            "color": data["colors"],
            "sh": data.get("sh", None),
            "images": data.get("images", None),
            "cameras": data.get("cameras", None),
        }
