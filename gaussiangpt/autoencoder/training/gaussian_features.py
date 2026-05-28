"""Gaussian feature layout helpers used by AE training.

ASE voxel caches store each Gaussian as a fixed 14D feature row. Centralizing
the layout here keeps slicing readable without changing the tensor views.
"""

from typing import Optional

import torch


GAUSSIAN_FEATURE_SLICES = {
    "offset": slice(0, 3),
    "color": slice(3, 6),
    "opacity": slice(6, 7),
    "scale": slice(7, 10),
    "rotation": slice(10, 14),
}


def gaussian_features_to_attrs(feats: torch.Tensor) -> dict:
    """View ASE 14D feature rows as the attribute dict expected by the AE heads.

    Feature layout is fixed by the ASE voxel cache:
      offset[0:3], color[3:6], opacity[6:7], scale[7:10], rotation[10:14].
    The slices are views into ``feats``; no data is copied or transformed here.
    """

    return {
        name: feats[:, feature_slice]
        for name, feature_slice in GAUSSIAN_FEATURE_SLICES.items()
    }


def sample_scene_origin(
    sample: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    value = sample.get("scene_origin")
    if value is None:
        value = (sample.get("metadata", {}) or {}).get("scene_origin")
    if value is None:
        return None
    return torch.as_tensor(value, dtype=dtype, device=device).reshape(3)


def build_world_positions(
    sample: dict,
    pred_offset: torch.Tensor,
    base_voxel_size: float,
    device: torch.device,
) -> torch.Tensor:
    """Convert per-voxel offsets into absolute world-space positions.

    `voxel_coords` are chunk-local; `chunk_origin` (if present) shifts them
    back into the scene's voxel frame. ASE voxel caches keep the world-space
    scene origin separately, so add it back before rendering from absolute
    camera poses.
    """
    voxel_coords = sample["voxel_coords"].to(device)
    if "chunk_origin" in sample:
        abs_voxel_coords = voxel_coords + sample["chunk_origin"].to(device)
    else:
        abs_voxel_coords = voxel_coords
    voxel_centers = (abs_voxel_coords.to(pred_offset.dtype) + 0.5) * base_voxel_size
    scene_origin = sample_scene_origin(sample, device, pred_offset.dtype)
    if scene_origin is not None:
        voxel_centers = voxel_centers + scene_origin
    return voxel_centers + pred_offset


def ase_batch_sample_to_legacy_sample(batch: dict, sample_index: int = 0) -> dict:
    """Extract one collated ASE sample in the legacy per-sample format.

    Validation reconstruction still runs through the older single-sample helper,
    so this keeps that path isolated from the batched training representation.
    """
    coords = batch["coords"]
    feats = batch.get("target_feats", batch["feats"])
    mask = coords[:, 0] == int(sample_index)
    if not bool(mask.any()):
        raise ValueError(f"ASE batch does not contain sample_index={sample_index}")

    sample_feats = feats[mask]
    sample_attrs = gaussian_features_to_attrs(sample_feats)
    meta = batch["metas"][sample_index]
    return {
        "voxel_coords": coords[mask, 1:4],
        "chunk_origin": torch.as_tensor(meta["chunk_min_voxel"], dtype=torch.long),
        "scene_origin": meta.get("scene_origin"),
        **sample_attrs,
        "metadata": meta,
    }
