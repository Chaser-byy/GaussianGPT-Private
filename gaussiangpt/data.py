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
import numpy as np
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
    SCENE_EXTENSIONS = (".pt", ".pth", ".ply")

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
        fixed_chunk: bool = False,
        voxel_dedup: str = "random",
    ):
        data_dir = data_dir + "/" + split
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.base_voxel_size = base_voxel_size
        self.latent_voxel_size = base_voxel_size * (2 ** n_down)
        self.chunk_size = chunk_size
        self.augment = augment
        # ``fixed_chunk=True`` makes the chunk sampler deterministic per
        # scene -- handy for verifying the model can overfit a static
        # target. With ``augment`` off, the dataset becomes a pure
        # memorisation task.
        self.chunk_sampler = ChunkSampler(
            chunk_size, min_occupancy=min_occupancy, deterministic=fixed_chunk,
        )
        # ``voxel_dedup`` controls which Gaussian survives per voxel when
        # multiple Gaussians fall in the same voxel:
        #   * "random"  -- paper-faithful (subsample uniformly).
        #   * "first"   -- deterministic; lowest input index per voxel.
        #   * "nearest" -- deterministic; pick the Gaussian whose centre
        #                  is closest to the voxel centre (cleanest GT).
        self.voxel_dedup = (voxel_dedup or "random").lower()
        if self.voxel_dedup not in ("random", "first", "nearest"):
            raise ValueError(
                f"voxel_dedup must be 'random'|'first'|'nearest', got {voxel_dedup!r}"
            )

        # Find all scene files
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(self.SCENE_EXTENSIONS)
        ])
        # n_train = int(len(all_files) * split_ratio)
        # if split == "train":
        #     self.files = all_files[:n_train]
        # else:
        #     self.files = all_files[n_train:]

        # 8x augmentation: 4 rotations x 2 reflections
        self.aug_factor = 8 if augment else 1
        self.n_scenes = len(self.files)

    def __len__(self) -> int:
        return self.n_scenes * self.aug_factor

    def __getitem__(self, idx: int) -> Dict:
        scene_idx = idx % self.n_scenes
        aug_idx = idx // self.n_scenes if self.augment else 0

        data = self._load_scene_file(self.files[scene_idx])
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

        # Voxelize at base resolution.
        voxel_coords, voxel_gaussians = self._voxelize(gaussians)   #每个 occupied voxel 的全局整数坐标。每个 occupied voxel 对应随机/确定性选出的一个 Gaussian 属性。

        # Sample a chunk and apply the same mask to the aligned Gaussian attributes.
        # In val/test mode we want validation reconstructions to be exactly
        # comparable across runs and across training steps, so seed the
        # chunk-sampling RNG deterministically by (split, scene_idx, aug_idx).
        # This was previously random even with augment=False.
        if self.split != "train":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(hash(("ae_val_chunk", scene_idx, aug_idx)) & 0x7FFFFFFF)
                chunk_result = self.chunk_sampler.sample_chunk(voxel_coords, voxel_gaussians)
        else:
            chunk_result = self.chunk_sampler.sample_chunk(voxel_coords, voxel_gaussians)
        if chunk_result is None:
            # Fallback: return empty chunk
            return self.__getitem__((idx + 1) % len(self))

        chunk_coords, chunk_origin, flat_in_chunk = chunk_result

        result = {
            "voxel_coords": chunk_coords,
            "chunk_origin": chunk_origin,
            **flat_in_chunk,
        }

        if "images" in data and "cameras" in data:
            result["images"] = data["images"]
            result["cameras"] = data["cameras"]

        return result

    def _load_scene_file(self, path: str) -> Dict:
        """Load either a tensor checkpoint scene or a standard 3DGS PLY scene."""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".pt", ".pth"):
            return torch.load(path, map_location="cpu")
        if ext == ".ply":
            return self._load_ply_scene(path)
        raise ValueError(f"Unsupported scene file extension: {path}")

    def _load_ply_scene(self, path: str) -> Dict[str, torch.Tensor]:
        """Parse a 3D Gaussian Splatting .ply file into the tensor scene format."""
        vertex_count, properties, fmt, header_end, header_lines = self._read_ply_header(path)
        if vertex_count <= 0:
            raise ValueError(f"PLY file has no vertex data: {path}")

        dtype = self._ply_dtype(properties, fmt)
        if fmt == "ascii":
            vertex_data = np.loadtxt(path, dtype=dtype, skiprows=header_lines, max_rows=vertex_count)
        else:
            vertex_data = np.fromfile(path, dtype=dtype, count=vertex_count, offset=header_end)

        def tensor_from_props(names: List[str]) -> torch.Tensor:
            missing = [name for name in names if name not in vertex_data.dtype.names]
            if missing:
                raise KeyError(f"Missing PLY properties {missing} in {path}")
            arr = np.stack([vertex_data[name] for name in names], axis=1).astype(np.float32, copy=False)
            return torch.from_numpy(arr.copy())

        positions = tensor_from_props(["x", "y", "z"])
        opacities = tensor_from_props(["opacity"])
        scales = torch.exp(tensor_from_props(self._sorted_prefixed_props(vertex_data, "scale_")))
        # 3DGS PLYs store quaternions un-normalised (the renderer normalises
        # at use time). The autoencoder decoder predicts unit quaternions,
        # so normalise the targets here once for a fair L1/MSE comparison.
        # 由于 gaussian_heads.py:195 也进行了归一化，所以这里这个归一化功能上是冗余的，但留着更鲁棒
        rotations = F.normalize(
            tensor_from_props(self._sorted_prefixed_props(vertex_data, "rot_")),
            dim=-1,
        )

        C0 = 0.28209479177387814  # SH degree-0 normalization constant.
        f_dc_names = [f"f_dc_{i}" for i in range(3)]
        if all(name in vertex_data.dtype.names for name in f_dc_names):
            f_dc = tensor_from_props(f_dc_names)
            colors = (f_dc * C0 + 0.5).clamp(0, 1)
        elif all(name in vertex_data.dtype.names for name in ("red", "green", "blue")):
            colors = tensor_from_props(["red", "green", "blue"]) / 255.0
        else:
            raise KeyError(f"PLY file needs either f_dc_* or red/green/blue color properties: {path}")

        data = {
            "positions": positions,
            "scales": scales,
            "opacities": opacities,
            "rotations": rotations,
            "colors": colors,
        }

        f_rest_names = self._sorted_prefixed_props(vertex_data, "f_rest_")
        if all(name in vertex_data.dtype.names for name in f_dc_names) and len(f_rest_names) >= 9:
            f_rest = tensor_from_props(f_rest_names[:9])
            data["sh"] = torch.cat([tensor_from_props(f_dc_names), f_rest], dim=1)

        return data

    def _read_ply_header(self, path: str) -> Tuple[int, List[Tuple[str, str]], str, int, int]:
        vertex_count = 0
        properties = []
        fmt = None
        in_vertex = False
        header_lines = 0

        with open(path, "rb") as f:
            first_line = f.readline()
            header_lines += 1
            if first_line.strip() != b"ply":
                raise ValueError(f"Not a PLY file: {path}")

            while True:
                line_bytes = f.readline()
                if not line_bytes:
                    raise ValueError(f"PLY header missing end_header: {path}")
                header_lines += 1
                line = line_bytes.decode("ascii").strip()
                parts = line.split()

                if not parts or parts[0] == "comment":
                    continue
                if parts[0] == "format":
                    fmt = parts[1]
                elif parts[0] == "element":
                    in_vertex = parts[1] == "vertex"
                    if in_vertex:
                        vertex_count = int(parts[2])
                elif parts[0] == "property" and in_vertex:
                    if parts[1] == "list":
                        raise ValueError(f"List properties are not supported for vertex data in {path}")
                    properties.append((parts[2], parts[1]))
                elif parts[0] == "end_header":
                    break

            header_end = f.tell()

        if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY format {fmt!r} in {path}")
        if not properties:
            raise ValueError(f"PLY file has no vertex properties: {path}")

        return vertex_count, properties, fmt, header_end, header_lines

    def _ply_dtype(self, properties: List[Tuple[str, str]], fmt: str) -> np.dtype:
        endian = "<" if fmt == "binary_little_endian" else ">" if fmt == "binary_big_endian" else "="
        type_map = {
            "char": "i1",
            "int8": "i1",
            "uchar": "u1",
            "uint8": "u1",
            "short": "i2",
            "int16": "i2",
            "ushort": "u2",
            "uint16": "u2",
            "int": "i4",
            "int32": "i4",
            "uint": "u4",
            "uint32": "u4",
            "float": "f4",
            "float32": "f4",
            "double": "f8",
            "float64": "f8",
        }
        fields = []
        for name, ply_type in properties:
            if ply_type not in type_map:
                raise ValueError(f"Unsupported PLY property type {ply_type!r}")
            np_type = type_map[ply_type]
            fields.append((name, np_type if np_type.endswith("1") else endian + np_type))
        return np.dtype(fields)

    def _sorted_prefixed_props(self, vertex_data: np.ndarray, prefix: str) -> List[str]:
        names = [name for name in vertex_data.dtype.names if name.startswith(prefix)]
        return sorted(names, key=lambda name: int(name[len(prefix):]))

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
        (one Gaussian per voxel, selected per ``self.voxel_dedup``).
        """
        positions = gaussians["offset"]
        voxel_coords = torch.floor(positions / self.base_voxel_size).long()

        # Choose the per-voxel survivor index according to the dedup
        # strategy. ``selected_idx`` will end up as a (n_voxels,) tensor
        # of indices into the original ``positions`` array.
        if self.voxel_dedup == "nearest":
            # Compute squared distance from each Gaussian to its voxel's
            # centre, then pick the closest one per voxel.
            voxel_centers = (voxel_coords.to(positions.dtype) + 0.5) * self.base_voxel_size
            dist_sq = ((positions - voxel_centers) ** 2).sum(dim=-1)

            unique_coords, inverse = torch.unique(voxel_coords, dim=0, return_inverse=True)
            n_voxels = unique_coords.shape[0]

            # # 1. 计算每个体素的点数
            # voxel_counts = torch.bincount(inverse).cpu().numpy()

            # # 2. 绘制并保存统计图
            # import matplotlib.pyplot as plt
            
            # plt.figure(figsize=(10, 6))
            # plt.hist(voxel_counts, bins=range(1, voxel_counts.max() + 2), edgecolor='black', alpha=0.7)
            # plt.title(f"Distribution of Gaussian Points per Voxel (Total Voxels: {n_voxels})")
            # plt.xlabel("Number of Points in a Voxel")
            # plt.ylabel("Frequency (Number of Voxels)")
            # plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # # 保存到本地文件
            # save_path = "/home/chengtianle/GaussianGPT/output/voxel_distribution.png"
            # plt.savefig(save_path)
            # plt.close()
            
            # print(f"统计图已保存至: {save_path}")
            # assert False

            # For each unique voxel, find the index with minimum dist_sq.
            # Encode (dist, original_idx) into a single int64 sort key so
            # ``scatter_reduce_(amin)`` returns the index of the minimum.
            INF = torch.tensor(float("inf"), device=positions.device)
            min_dist = torch.full((n_voxels,), float("inf"), device=positions.device)
            min_dist.scatter_reduce_(0, inverse, dist_sq, reduce="amin", include_self=True)
            # Mark the winners: any input row whose dist matches its
            # voxel's min. Ties are broken by lowest original index.
            is_winner = dist_sq == min_dist[inverse]
            big = positions.shape[0]
            tie_score = torch.where(
                is_winner,
                torch.arange(big, dtype=torch.long, device=positions.device),
                torch.full((big,), big, dtype=torch.long, device=positions.device),
            )
            selected_idx = torch.full(
                (n_voxels,), big, dtype=torch.long, device=positions.device,
            )
            selected_idx.scatter_reduce_(0, inverse, tie_score, reduce="amin", include_self=True)
        else:
            if self.voxel_dedup == "first" or self.split != "train":
                # Deterministic: lowest input index per voxel.
                perm = torch.arange(len(positions), device=positions.device)
            else:
                # Paper-faithful random subsampling.
                perm = torch.randperm(len(positions), device=positions.device)

            permuted_coords = voxel_coords[perm]
            unique_coords, inverse = torch.unique(permuted_coords, dim=0, return_inverse=True)
            n_voxels = unique_coords.shape[0]

            src = torch.arange(len(positions), dtype=torch.long, device=positions.device)
            first_occ = torch.full(
                (n_voxels,), len(positions), dtype=torch.long, device=positions.device,
            )
            first_occ.scatter_reduce_(0, inverse, src, reduce="amin", include_self=True)
            selected_idx = perm[first_occ]

        voxel_gaussians = {key: val[selected_idx] for key, val in gaussians.items()}
        voxel_centers = (unique_coords.to(positions.dtype) + 0.5) * self.base_voxel_size
        voxel_gaussians["offset"] = positions[selected_idx] - voxel_centers
        return unique_coords, voxel_gaussians

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
        data_dir = data_dir + "/" + split
        self.data_dir = data_dir
        self.grid_size = grid_size

        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pt") or f.endswith(".pth")
        ])
        # n_train = int(len(all_files) * split_ratio)
        # self.files = all_files[:n_train] if split == "train" else all_files[n_train:]

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
