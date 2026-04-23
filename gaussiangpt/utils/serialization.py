"""3D grid serialization: convert sparse voxel grid to/from token sequences.

Serialization order: xyz traversal (x most significant, z least significant).
This means we iterate a scene-height column at every (x,y) position before
jumping to the next one.

Sequence format: [BOS, pos_0, feat_0, pos_1, feat_1, ..., EOS]
  - pos_i: voxel index within the current chunk (relative coordinates)
  - feat_i: LFQ codebook index for that voxel
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def xyz_order_indices(chunk_size: Tuple[int, int, int]) -> torch.Tensor:
    """Generate voxel indices in xyz traversal order.

    Args:
        chunk_size: (cx, cy, cz)
    Returns:
        indices: (cx*cy*cz,) flat indices in xyz order
    """
    cx, cy, cz = chunk_size
    # xyz order: x is most significant, z is least significant
    # flat index = x*(cy*cz) + y*cz + z
    xs = torch.arange(cx)
    ys = torch.arange(cy)
    zs = torch.arange(cz)
    # meshgrid in xyz order
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
    flat = grid_x * (cy * cz) + grid_y * cz + grid_z
    return flat.reshape(-1)  # (cx*cy*cz,)


def coord_to_flat_idx(coords: torch.Tensor, chunk_size: Tuple[int, int, int]) -> torch.Tensor:
    """Convert (N, 3) integer coordinates to flat indices.

    Args:
        coords: (N, 3) integer coordinates (x, y, z) within chunk
        chunk_size: (cx, cy, cz)
    Returns:
        indices: (N,) flat indices
    """
    cx, cy, cz = chunk_size
    return coords[:, 0] * (cy * cz) + coords[:, 1] * cz + coords[:, 2]


def flat_idx_to_coord(indices: torch.Tensor, chunk_size: Tuple[int, int, int]) -> torch.Tensor:
    """Convert flat indices to (N, 3) integer coordinates.

    Args:
        indices: (N,) flat indices
        chunk_size: (cx, cy, cz)
    Returns:
        coords: (N, 3) integer coordinates (x, y, z)
    """
    cx, cy, cz = chunk_size
    x = indices // (cy * cz)
    y = (indices // cz) % cy
    z = indices % cz
    return torch.stack([x, y, z], dim=-1)


def serialize_latent_grid(
    voxel_coords: torch.Tensor,
    voxel_codes: torch.Tensor,
    chunk_size: Tuple[int, int, int],
    BOS: int,
    EOS: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Serialize a sparse latent grid into an interleaved token sequence.

    Args:
        voxel_coords: (N, 3) integer coordinates of occupied voxels (relative to chunk)
        voxel_codes: (N,) LFQ codebook indices for each voxel
        chunk_size: (cx, cy, cz) chunk dimensions
        BOS: begin-of-sequence token index
        EOS: end-of-sequence token index
    Returns:
        tokens: (2*N+2,) interleaved [BOS, pos_0, feat_0, ..., EOS]
        coords: (2*N+2, 3) spatial coordinates for each token
        token_type: (2*N+2,) 0=position token, 1=feature token
    """
    N = voxel_coords.shape[0]
    device = voxel_coords.device

    # Sort voxels by xyz order
    flat_idx = coord_to_flat_idx(voxel_coords, chunk_size)
    order = torch.argsort(flat_idx)
    voxel_coords = voxel_coords[order]
    voxel_codes = voxel_codes[order]
    flat_idx = flat_idx[order]

    # Build interleaved sequence vectorized
    # Layout: [BOS, pos_0, feat_0, pos_1, feat_1, ..., EOS]
    seq_len = 2 * N + 2
    tokens = torch.zeros(seq_len, dtype=torch.long, device=device)
    coords = torch.zeros(seq_len, 3, dtype=torch.long, device=device)
    token_type = torch.zeros(seq_len, dtype=torch.long, device=device)

    # BOS
    tokens[0] = BOS

    # Position tokens at odd indices 1, 3, 5, ...
    pos_indices = torch.arange(1, 2 * N, 2, device=device)
    tokens[pos_indices] = flat_idx
    coords[pos_indices] = voxel_coords
    token_type[pos_indices] = 0

    # Feature tokens at even indices 2, 4, 6, ...
    feat_indices = torch.arange(2, 2 * N + 1, 2, device=device)
    tokens[feat_indices] = voxel_codes
    coords[feat_indices] = voxel_coords  # same coord as preceding position token
    token_type[feat_indices] = 1

    # EOS
    tokens[-1] = EOS

    return tokens, coords, token_type


def deserialize_token_sequence(
    tokens: torch.Tensor,
    chunk_size: Tuple[int, int, int],
    BOS: int,
    EOS: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deserialize a token sequence back to voxel coordinates and codes.

    Args:
        tokens: (T,) token sequence
        chunk_size: (cx, cy, cz)
        BOS: BOS token index
        EOS: EOS token index
    Returns:
        voxel_coords: (N, 3) integer coordinates
        voxel_codes: (N,) codebook indices
    """
    # Strip BOS/EOS
    mask = (tokens != BOS) & (tokens != EOS)
    tokens = tokens[mask]

    # Tokens alternate: pos, feat, pos, feat, ...
    if len(tokens) % 2 != 0:
        tokens = tokens[:-1]  # drop incomplete pair

    pos_tokens = tokens[0::2]  # position tokens
    feat_tokens = tokens[1::2]  # feature tokens

    voxel_coords = flat_idx_to_coord(pos_tokens, chunk_size)
    return voxel_coords, feat_tokens


class ChunkSampler:
    """Samples spatial chunks from a scene for training.

    Paper: chunks have fixed vertical positions to align floor heights.
    Minimum occupancy threshold: 0.2 for autoencoder, 0.3 for GPT.
    Up to 10 candidate chunks are tried; if none meet threshold, use highest.
    """

    def __init__(
        self,
        chunk_size: Tuple[int, int, int],
        min_occupancy: float = 0.2,
        max_tries: int = 10,
    ):
        self.chunk_size = chunk_size
        self.min_occupancy = min_occupancy
        self.max_tries = max_tries

    def sample_chunk(
        self,
        voxel_coords: torch.Tensor,
        scene_bounds: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a chunk from the scene.

        Args:
            voxel_coords: (N, 3) all occupied voxel coordinates in the scene
            scene_bounds: optional (min_xyz, max_xyz) bounds
        Returns:
            chunk_coords: (M, 3) voxel coordinates within the chunk (relative)
            chunk_origin: (3,) origin of the chunk in scene coordinates
        """
        cx, cy, cz = self.chunk_size
        chunk_vol = cx * cy * cz

        if scene_bounds is None:
            min_xyz = voxel_coords.min(0).values
            max_xyz = voxel_coords.max(0).values
        else:
            min_xyz = torch.tensor(scene_bounds[0])
            max_xyz = torch.tensor(scene_bounds[1])

        best_chunk = None
        best_occ = 0.0

        for _ in range(self.max_tries):
            # Random chunk origin (fixed z=0 for floor alignment)
            ox_lo = int(min_xyz[0].item())
            ox_hi = max(ox_lo + 1, int(max_xyz[0].item()) - cx + 2)
            oy_lo = int(min_xyz[1].item())
            oy_hi = max(oy_lo + 1, int(max_xyz[1].item()) - cy + 2)
            ox = torch.randint(ox_lo, ox_hi, (1,)).item()
            oy = torch.randint(oy_lo, oy_hi, (1,)).item()
            oz = int(min_xyz[2].item())  # fixed vertical position

            origin = torch.tensor([ox, oy, oz], dtype=torch.long)

            # Find voxels within chunk
            rel = voxel_coords - origin
            mask = (
                (rel[:, 0] >= 0) & (rel[:, 0] < cx) &
                (rel[:, 1] >= 0) & (rel[:, 1] < cy) &
                (rel[:, 2] >= 0) & (rel[:, 2] < cz)
            )
            chunk_voxels = rel[mask]
            occ = len(chunk_voxels) / chunk_vol

            if occ >= self.min_occupancy:
                return chunk_voxels, origin

            if occ > best_occ:
                best_occ = occ
                best_chunk = (chunk_voxels, origin)

        return best_chunk
