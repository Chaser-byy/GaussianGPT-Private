"""Train the GaussianGPT autoencoder.

Usage:
    python scripts/train_autoencoder.py --config configs/autoencoder_scene.yaml

Paper training details:
  - Adam optimizer, lr=1e-4, cosine decay to 10%
  - 4 RTX A6000 GPUs, effective batch size 8 (scenes) / 24 (objects)
  - ~4 days for scenes, ~2 days for PhotoShape
"""
import os
import argparse
import math
import yaml
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from plyfile import PlyData, PlyElement

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussiangpt.autoencoder import GaussianAutoencoder
from gaussiangpt.autoencoder.diagnostics import ColorClampDiagnostics
from gaussiangpt.data import GaussianSceneDataset
from gaussiangpt.utils.rendering import (
    HAS_RASTERIZER,
    make_camera_from_world_to_camera,
    sample_cameras_around_bbox,
    render_gaussians,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/autoencoder_scene.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/autoencoder")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val_every_steps", type=int, default=None)
    parser.add_argument("--cache_root", type=str, default=None)
    return parser.parse_args()


def sparse_collate(batch):
    """Collate function for variable-size sparse Gaussian data.

    Returns a list of per-sample dicts rather than stacking tensors,
    since the number of voxels N differs across samples.
    """
    return batch


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _sample_scene_origin(
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


def _build_world_positions(
    sample: dict,
    pred_offset: torch.Tensor,
    base_voxel_size: float,
    device: torch.device,
) -> torch.Tensor:
    """Convert per-voxel offsets into absolute world-space positions.

    `voxel_coords` are chunk-local; `chunk_origin` (if present) shifts them
    back into the scene's voxel frame. ASE voxel caches keep the world-space
    scene origin separately, so add it back when present before rendering from
    absolute camera poses.
    """
    voxel_coords = sample["voxel_coords"].to(device)
    if "chunk_origin" in sample:
        abs_voxel_coords = voxel_coords + sample["chunk_origin"].to(device)
    else:
        abs_voxel_coords = voxel_coords
    voxel_centers = (abs_voxel_coords.to(pred_offset.dtype) + 0.5) * base_voxel_size
    scene_origin = _sample_scene_origin(sample, device, pred_offset.dtype)
    if scene_origin is not None:
        voxel_centers = voxel_centers + scene_origin
    return voxel_centers + pred_offset


def _camera_sampling_config(cfg: dict) -> tuple:
    camera_cfg = cfg.get("camera_sampling", {}) or {}
    mode = str(camera_cfg.get("mode", "orbit")).lower()
    if mode in {"default", "orbit", "around_bbox", "bbox"}:
        mode = "orbit"
    elif mode in {"score", "scored", "scoring", "gaussiangpt"}:
        mode = "scoring"
    else:
        raise ValueError(
            "camera_sampling.mode must be 'orbit' or 'scoring', "
            f"got {mode!r}"
        )
    return camera_cfg, mode


def _camera_dataset_type(cfg: dict) -> str:
    camera_cfg, _mode = _camera_sampling_config(cfg)
    dataset_type = str(camera_cfg.get("dataset_type", "ase")).lower()
    aliases = {
        "ase": "ase",
        "3dfront": "3dfront",
        "3d-front": "3dfront",
        "3d_front": "3dfront",
        "front": "3dfront",
    }
    if dataset_type not in aliases:
        raise ValueError(
            "camera_sampling.dataset_type must be 'ase' or '3dfront', "
            f"got {dataset_type!r}"
        )
    return aliases[dataset_type]


def _validate_camera_sampling_config(cfg: dict) -> None:
    camera_cfg, mode = _camera_sampling_config(cfg)
    if mode != "scoring":
        return

    dataset_type = _camera_dataset_type(cfg)
    if dataset_type == "ase":
        score_key = str(camera_cfg.get("score_key", "chunk_coverage"))
        if score_key != "chunk_coverage":
            raise ValueError(
                "GaussianGPT ASE camera scoring uses projected chunk bbox "
                "coverage. Set camera_sampling.score_key='chunk_coverage'; "
                f"got {score_key!r}."
            )
        return

    raise NotImplementedError(
        "camera_sampling.dataset_type='3dfront' requires the GaussianGPT "
        "3D-FRONT path with pre-rendered images and depth maps to compute "
        "visible chunk area. This repository path currently trains from ASE "
        "camera caches only, so 3D-FRONT depth-map scoring is intentionally "
        "not emulated."
    )


def _effective_render_view_count(cfg: dict, loss_cfg: Optional[dict] = None) -> int:
    loss_cfg = cfg.get("loss", {}) if loss_cfg is None else loss_cfg
    n_views = int(loss_cfg.get("n_images", 0))
    camera_cfg, mode = _camera_sampling_config(cfg)
    if mode == "scoring" and camera_cfg.get("num_views") is not None:
        n_views = int(camera_cfg["num_views"])
    return n_views


def _validation_pruning_config(cfg: dict) -> dict:
    validation_cfg = cfg.get("validation", {}) or {}
    occ_threshold = validation_cfg.get(
        "occ_threshold",
        validation_cfg.get("occupancy_threshold", 0.5),
    )
    prune_min_keep = validation_cfg.get(
        "prune_min_keep",
        validation_cfg.get("min_keep", 1),
    )
    return {
        "prune": bool(validation_cfg.get("prune", False)),
        "occ_threshold": float(occ_threshold),
        "prune_min_keep": int(prune_min_keep),
    }


def _count_coords_by_sample(coords: Optional[torch.Tensor], n_samples: int) -> list:
    if coords is None or coords.numel() == 0:
        return [0 for _ in range(n_samples)]
    if coords.dim() == 2 and coords.shape[1] >= 4:
        batch_index = coords[:, 0].detach().long().cpu()
        counts = torch.bincount(batch_index, minlength=n_samples)
        return [int(v) for v in counts[:n_samples].tolist()]
    return [int(coords.shape[0])] + [0 for _ in range(max(n_samples - 1, 0))]


def _decoder_pre_prune_counts(occ_list: list, fallback_counts: list, n_samples: int) -> list:
    if not occ_list:
        return fallback_counts
    last_occ = occ_list[-1]
    if hasattr(last_occ, "C"):
        return _count_coords_by_sample(last_occ.C, n_samples)
    if torch.is_tensor(last_occ) and last_occ.dim() >= 5:
        batch_count = min(int(last_occ.shape[0]), n_samples)
        counts = [int(last_occ[i].numel()) for i in range(batch_count)]
        counts.extend([0 for _ in range(n_samples - batch_count)])
        return counts
    return fallback_counts


def _log_validation_pruning_counts(
    batch: dict,
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    occ_list: list,
    prune: bool,
    log_prefix: str,
) -> None:
    n_samples = len(batch.get("metas", [])) or 1
    gt_counts = _count_coords_by_sample(gt_coords, n_samples)
    post_counts = _count_coords_by_sample(pred_coords, n_samples)
    pre_counts = _decoder_pre_prune_counts(occ_list, post_counts, n_samples) if prune else post_counts
    mode = "pruned" if prune else "unpruned"
    for sample_idx in range(n_samples):
        print(
            f"  [{log_prefix} pruning] sample={sample_idx} "
            f"mode={mode} gt_voxels={gt_counts[sample_idx]} "
            f"pre_voxels={pre_counts[sample_idx]} post_voxels={post_counts[sample_idx]} "
            f"pre_gaussians={pre_counts[sample_idx]} post_gaussians={post_counts[sample_idx]}"
        )


def _camera_candidate_count(cfg: dict, n_views: int) -> Optional[int]:
    camera_cfg, mode = _camera_sampling_config(cfg)
    if mode != "scoring":
        return None
    if _camera_dataset_type(cfg) == "ase":
        return None
    if camera_cfg.get("num_candidates") is not None:
        value = int(camera_cfg["num_candidates"])
        return value if value > 0 else None
    return max(1, 8 * max(1, int(n_views)))


def _camera_score_key(cfg: dict) -> str:
    camera_cfg, _mode = _camera_sampling_config(cfg)
    return str(camera_cfg.get("score_key", "chunk_coverage"))


def _camera_candidates_from_sample(sample: dict) -> list:
    meta = sample.get("metadata", {}) or {}
    candidates = meta.get("top_cameras")
    if candidates is None:
        candidates = (meta.get("camera_debug", {}) or {}).get("top_cameras", [])
    return list(candidates or [])


def _camera_candidate_signature(candidates: list, score_key: str) -> tuple:
    signature = []
    for item in candidates:
        score = item.get(score_key, item.get("chunk_coverage", 0.0))
        try:
            score_value = round(float(score), 6)
        except (TypeError, ValueError):
            score_value = 0.0
        signature.append(
            (
                str(item.get("camera_id", "")),
                int(item.get("frame_index", item.get("frame_id", -1)) or -1),
                score_value,
            )
        )
    return tuple(signature)


def _scored_camera_to_minicam(
    camera_info: dict,
    image_size: int,
    fallback_fov_deg: float,
    device: torch.device,
):
    world_to_camera = torch.as_tensor(
        np.asarray(camera_info["w2c"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    width = float(camera_info.get("width", image_size))
    height = float(camera_info.get("height", image_size))
    fx = float(camera_info.get("fx", 0.0))
    fy = float(camera_info.get("fy", 0.0))
    fallback_fov = math.radians(float(fallback_fov_deg))
    fovx = 2.0 * math.atan(width / max(2.0 * fx, 1e-8)) if fx > 0 else fallback_fov
    fovy = 2.0 * math.atan(height / max(2.0 * fy, 1e-8)) if fy > 0 else fallback_fov
    return make_camera_from_world_to_camera(
        world_to_camera=world_to_camera,
        fovx=fovx,
        fovy=fovy,
        image_height=image_size,
        image_width=image_size,
    )


def _camera_sampling_generator(
    cfg: dict,
    sample: dict,
    device: torch.device,
    rng: Optional[torch.Generator],
    global_step: Optional[int],
) -> Optional[torch.Generator]:
    camera_cfg, _mode = _camera_sampling_config(cfg)
    seed_value = camera_cfg.get("seed")
    if seed_value is None:
        return rng

    meta = sample.get("metadata", {}) or {}
    seed = int(seed_value) + int(global_step or 0) * 1_000_003
    for char in str(meta.get("scene_id", "")):
        seed = (seed * 33 + ord(char)) % (2**63 - 1)
    chunk = meta.get("chunk_min_voxel", sample.get("chunk_origin", [0, 0, 0]))
    if torch.is_tensor(chunk):
        chunk = chunk.detach().cpu().tolist()
    for idx, value in enumerate(chunk):
        seed = (seed * 1009 + (idx + 1) * int(value)) % (2**63 - 1)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _sample_scored_cameras(
    sample: dict,
    cfg: dict,
    n_views: int,
    image_size: int,
    fallback_fov_deg: float,
    device: torch.device,
    rng: Optional[torch.Generator],
    global_step: Optional[int],
) -> tuple:
    """Sample cameras with the dataset-specific GaussianGPT Appendix C rule."""

    _validate_camera_sampling_config(cfg)
    camera_cfg, _mode = _camera_sampling_config(cfg)
    dataset_type = _camera_dataset_type(cfg)
    score_key = _camera_score_key(cfg)
    temperature = max(float(camera_cfg.get("temperature", 1.0)), 1e-6)
    raw_candidates = _camera_candidates_from_sample(sample)
    candidates = [item for item in raw_candidates if "w2c" in item]
    debug = {
        "source": "scoring",
        "dataset_type": dataset_type,
        "score_key": score_key,
        "candidate_count": len(raw_candidates),
        "usable_candidate_count": len(candidates),
        "selection_mode": "score_only",
        "sampled_count": 0,
        "fallback": None,
    }
    if not candidates:
        raise RuntimeError(
            "camera_sampling.mode=scoring,dataset_type=ase requires scored ASE "
            "camera candidates with poses in sample metadata. Rebuild the ASE "
            "camera cache or run the dataset with include_camera_matrices=True."
        )

    score_values = []
    for item in candidates:
        score = item.get(score_key, item.get("chunk_coverage", 0.0))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score) or score < 0.0:
            score = 0.0
        score_values.append(score)

    scores = torch.tensor(score_values, dtype=torch.float32, device=device)
    score_min = float(scores.min().detach().item())
    score_max = float(scores.max().detach().item())
    score_mean = float(scores.mean().detach().item())
    score_sum = float(scores.sum().detach().item())

    if score_sum <= 0.0:
        probs = torch.full_like(scores, 1.0 / float(scores.numel()))
        positive_probability_count = int(scores.numel())
        debug["fallback"] = "all_zero_scores_uniform"
    else:
        # GaussianGPT samples proportionally to the chunk-view score. The
        # exponent form preserves exactly-zero probabilities for zero-score
        # cameras while allowing flatter or sharper distributions.
        weights = scores.pow(1.0 / temperature)
        probs = weights / weights.sum().clamp_min(1e-12)
        positive_probability_count = int(
            torch.count_nonzero(weights > 0.0).detach().item()
        )

    replacement = int(n_views) > positive_probability_count
    if replacement:
        debug["fallback"] = (
            f"{debug['fallback']}+replacement"
            if debug["fallback"]
            else "positive_score_count_lt_num_views_replacement"
        )
    generator = _camera_sampling_generator(cfg, sample, device, rng, global_step)
    sampled = torch.multinomial(
        probs,
        int(n_views),
        replacement=replacement,
        generator=generator,
    )
    sampled_indices = [int(value) for value in sampled.detach().cpu().tolist()]
    selected = [candidates[index] for index in sampled_indices]
    cameras = [
        _scored_camera_to_minicam(item, image_size, fallback_fov_deg, device)
        for item in selected
    ]

    prob_min = float(probs.min().detach().item())
    prob_max = float(probs.max().detach().item())
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).detach().item())
    debug.update(
        {
            "sampled_count": len(cameras),
            "sampled_indices": sampled_indices,
            "sampled_frame_indices": [
                int(item.get("frame_index", item.get("frame_id", -1)) or -1)
                for item in selected
            ],
            "sampled_camera_ids": [str(item.get("camera_id", "")) for item in selected],
            "score_min": score_min,
            "score_max": score_max,
            "score_mean": score_mean,
            "score_sum": score_sum,
            "positive_probability_count": positive_probability_count,
            "prob_min": prob_min,
            "prob_max": prob_max,
            "prob_entropy": entropy,
            "temperature": temperature,
        }
    )
    return cameras, debug


def _should_log_camera_sampling(cfg: dict, global_step: Optional[int]) -> bool:
    camera_cfg, mode = _camera_sampling_config(cfg)
    if mode != "scoring":
        return False
    if global_step is None:
        return False
    every = int(camera_cfg.get("log_every", 100))
    return int(global_step) == 1 or (every > 0 and int(global_step) % every == 0)


def _log_camera_sampling(sample: dict, global_step: int, debug: dict) -> None:
    meta = sample.get("metadata", {}) or {}
    chunk = meta.get("chunk_min_voxel", sample.get("chunk_origin", None))
    if torch.is_tensor(chunk):
        chunk = chunk.detach().cpu().tolist()
    print(
        "  [camera sampling] "
        f"step={global_step} mode=scoring "
        f"dataset={debug.get('dataset_type', 'ase')} "
        f"scene={meta.get('scene_id', '')} "
        f"chunk_min={chunk} source={debug.get('source')} "
        f"candidates={debug.get('usable_candidate_count', 0)}/"
        f"{debug.get('candidate_count', 0)} "
        f"sampled={debug.get('sampled_count', 0)} "
        f"score_key={debug.get('score_key')} "
        f"selection={debug.get('selection_mode', 'score_only')} "
        f"score={debug.get('score_min', 0.0):.4g}/"
        f"{debug.get('score_mean', 0.0):.4g}/"
        f"{debug.get('score_max', 0.0):.4g} "
        f"prob={debug.get('prob_min', 0.0):.4g}/"
        f"{debug.get('prob_max', 0.0):.4g} "
        f"entropy={debug.get('prob_entropy', 0.0):.4g} "
        f"indices={debug.get('sampled_indices', [])} "
        f"frames={debug.get('sampled_frame_indices', [])} "
        f"fallback={debug.get('fallback')}"
    )


def _render_loss_for_sample(
    sample: dict,
    pred_gaussians: dict,
    gt_gaussians: dict,
    cfg: dict,
    device: torch.device,
    perceptual: nn.Module = None,
    rng: torch.Generator = None,
    gt_render_cache: Optional[dict] = None,
    global_step: Optional[int] = None,
) -> tuple:
    """Render N views of GT and predicted Gaussians; return (l_rgb, l_perc).

    The GT renderings are detached so the renderer is only used as a fixed
    photo-realistic supervision signal. Returns scalar tensors on `device`.
    """
    if not HAS_RASTERIZER or device.type != "cuda":
        zero = torch.zeros((), device=device)
        return zero, zero

    loss_cfg = cfg.get("loss", {})
    camera_cfg, camera_mode = _camera_sampling_config(cfg)
    n_views = _effective_render_view_count(cfg, loss_cfg)
    img_size = int(loss_cfg.get("render_size", 128))
    if n_views <= 0:
        zero = torch.zeros((), device=device)
        return zero, zero

    base_voxel_size = float(cfg["data"]["base_voxel_size"])

    # Build absolute world positions for both GT and reconstruction.
    # GaussianGPT predicts offsets as unbounded world-space values
    # (Appendix C), so we always size the camera sphere from the GT bbox
    # rather than relying on any prediction-side bound.
    gt_position = _build_world_positions(
        sample, gt_gaussians["offset"].detach(), base_voxel_size, device
    )
    pred_sample = sample
    if "pred_voxel_coords" in sample:
        pred_sample = dict(sample)
        pred_sample["voxel_coords"] = sample["pred_voxel_coords"]
    pred_position = _build_world_positions(
        pred_sample, pred_gaussians["offset"], base_voxel_size, device
    )

    bg = torch.zeros(3, device=device)
    gt_pack = {
        "position": gt_position,
        "scale": gt_gaussians["scale"].detach(),
        "rotation": gt_gaussians["rotation"].detach(),
        "opacity": gt_gaussians["opacity"].detach(),
        "color": gt_gaussians["color"].detach(),
    }
    pred_pack = {
        "position": pred_position,
        "scale": pred_gaussians["scale"],
        "rotation": pred_gaussians["rotation"],
        "opacity": pred_gaussians["opacity"],
        "color": pred_gaussians["color"],
    }

    fov_deg = float(loss_cfg.get("fov_deg", 60.0))
    radius_factor = float(loss_cfg.get("radius_factor", 1.5))
    upper_hemisphere_only = bool(loss_cfg.get("upper_hemisphere_only", False))
    camera_jitter = float(loss_cfg.get("camera_jitter", 0.0))
    score_key = _camera_score_key(cfg)
    candidates = _camera_candidates_from_sample(sample)
    camera_debug = None
    cache_key = None
    cameras = None
    cached_gt_imgs = None
    cacheable_cameras = camera_mode == "scoring" or camera_jitter == 0.0
    if gt_render_cache is not None and cacheable_cameras:
        meta = sample.get("metadata", {})
        chunk_origin_value = meta.get("chunk_min_voxel", sample.get("chunk_origin"))
        if torch.is_tensor(chunk_origin_value):
            chunk_origin_value = chunk_origin_value.detach().cpu().tolist()
        chunk_origin_key = tuple(int(v) for v in chunk_origin_value)
        candidate_count_for_cache = _camera_candidate_count(cfg, n_views)
        cache_key = (
            str(meta.get("scene_id", "")),
            chunk_origin_key,
            int(sample["voxel_coords"].shape[0]),
            str(device),
            int(n_views),
            int(img_size),
            float(fov_deg),
            float(radius_factor),
            bool(upper_hemisphere_only),
            str(camera_mode),
            (candidate_count_for_cache if candidate_count_for_cache is not None else "all")
            if camera_mode == "scoring" else 0,
            float(camera_cfg.get("temperature", 1.0)) if camera_mode == "scoring" else 0.0,
            str(score_key) if camera_mode == "scoring" else "",
            _camera_candidate_signature(candidates, score_key)
            if camera_mode == "scoring"
            else (),
        )
        cached = gt_render_cache.get(cache_key)
        if cached is not None:
            cameras = cached["cameras"]
            cached_gt_imgs = cached["gt_imgs"]
            camera_debug = dict(cached.get("camera_debug", {}))
            if camera_debug:
                camera_debug["source"] = "gt_render_cache"

    if cameras is None:
        bbox_min = gt_position.min(dim=0).values.detach()
        bbox_max = gt_position.max(dim=0).values.detach()
        if camera_mode == "scoring":
            cameras, camera_debug = _sample_scored_cameras(
                sample=sample,
                cfg=cfg,
                n_views=n_views,
                image_size=img_size,
                fallback_fov_deg=fov_deg,
                device=device,
                rng=rng,
                global_step=global_step,
            )
        if not cameras:
            if camera_mode == "scoring":
                if camera_debug is None:
                    camera_debug = {}
                camera_debug["source"] = "orbit_fallback"
            # Camera sphere is sized from the GT bbox so the views naturally
            # cover the scene as it gets reconstructed. This is the original
            # orbit path and remains the default behavior.
            cameras = sample_cameras_around_bbox(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                n_views=n_views,
                image_height=img_size,
                image_width=img_size,
                fov_deg=fov_deg,
                radius_factor=radius_factor,
                upper_hemisphere_only=upper_hemisphere_only,
                jitter=camera_jitter,
                rng=rng,
            )

    if (
        camera_mode == "scoring"
        and camera_debug is not None
        and _should_log_camera_sampling(cfg, global_step)
    ):
        _log_camera_sampling(sample, int(global_step), camera_debug)

    l_rgb = torch.zeros((), device=device)
    l_perc = torch.zeros((), device=device)
    perc_buf_pred, perc_buf_gt = [], []
    perc_batch_size = max(1, int(loss_cfg.get("perceptual_batch_size", 1)))
    gt_imgs_to_cache = [] if cache_key is not None and cached_gt_imgs is None else None
    for view_idx, cam in enumerate(cameras):
        if cached_gt_imgs is not None:
            img_gt = cached_gt_imgs[view_idx]
        else:
            with torch.no_grad():
                img_gt = render_gaussians(gt_pack, cam, bg_color=bg).detach()
            if gt_imgs_to_cache is not None:
                gt_imgs_to_cache.append(img_gt)
        img_pred = render_gaussians(pred_pack, cam, bg_color=bg)
        l_rgb = l_rgb + torch.nn.functional.l1_loss(img_pred, img_gt)
        if perceptual is not None and float(loss_cfg.get("lambda_perc", 0.0)) > 0:
            perc_buf_pred.append(img_pred)
            perc_buf_gt.append(img_gt)
    if gt_imgs_to_cache is not None:
        gt_render_cache[cache_key] = {
            "cameras": cameras,
            "gt_imgs": tuple(gt_imgs_to_cache),
            "camera_debug": camera_debug,
        }

    l_rgb = l_rgb / float(n_views)
    if perc_buf_pred:
        # VGG at 512px is memory-heavy. Run it in small chunks so the peak
        # activation footprint is bounded by perceptual_batch_size rather than
        # n_views, while preserving the same average-over-views loss.
        for start in range(0, len(perc_buf_pred), perc_batch_size):
            end = start + perc_batch_size
            pred_chunk = torch.stack(perc_buf_pred[start:end], dim=0)
            gt_chunk = torch.stack(perc_buf_gt[start:end], dim=0)
            chunk_loss = perceptual(pred_chunk, gt_chunk)
            l_perc = l_perc + chunk_loss * float(pred_chunk.shape[0])
        l_perc = l_perc / float(len(perc_buf_pred))
    return l_rgb, l_perc


def _coord_hash(coords: torch.Tensor) -> torch.Tensor:
    """Build collision-free integer hashes for a small set of 4D sparse coords."""

    coords = coords.to(torch.long)
    mins = coords.min(dim=0).values
    shifted = coords - mins
    dims = shifted.max(dim=0).values + 1
    key = shifted[:, 0]
    for dim in range(1, shifted.shape[1]):
        key = key * dims[dim] + shifted[:, dim]
    return key


def _sparse_occupancy_targets(
    occ,
    gt_coords: torch.Tensor,
    stage_idx: int,
    n_stages: int,
    device: torch.device,
) -> tuple:
    """Align a decoder occupancy SparseTensor with the current chunk occupancy.

    ``gt_coords`` are batched base-voxel coordinates ``[b, x, y, z]`` from the
    dataloader. Each decoder occupancy head lives at its own tensor stride, so
    GT occupied voxels are downsampled to that stage before matching against
    ``occ.C``. The returned target has exactly one value per occupancy logit.
    """

    occ_coords = occ.C.to(device=device, dtype=torch.long)
    occ_logits = occ.F.squeeze(-1)
    tensor_stride = getattr(occ, "tensor_stride", None)
    if tensor_stride is None:
        stride = 2 ** max(n_stages - stage_idx - 1, 0)
    elif isinstance(tensor_stride, (list, tuple)):
        stride = int(tensor_stride[0])
    else:
        stride = int(tensor_stride)
    stride = max(stride, 1)

    if occ_logits.numel() == 0:
        return occ_logits, torch.empty_like(occ_logits), stride

    gt_stage_coords = gt_coords.to(device=device, dtype=torch.long).clone()
    gt_stage_coords[:, 1:] = torch.div(
        gt_stage_coords[:, 1:], stride, rounding_mode="floor"
    )
    gt_stage_coords = torch.unique(gt_stage_coords, dim=0)

    all_coords = torch.cat([occ_coords, gt_stage_coords], dim=0)
    all_hash = _coord_hash(all_coords)
    occ_hash = all_hash[: occ_coords.shape[0]]
    gt_hash = torch.unique(all_hash[occ_coords.shape[0]:])
    targets = torch.isin(occ_hash, gt_hash).to(dtype=occ_logits.dtype)
    return occ_logits, targets, stride


# def compute_batch_loss(
#     raw_model,
#     batch_list,
#     cfg: dict,
#     device: torch.device,
#     backward: bool = False,
#     perceptual: nn.Module = None,
#     rng: torch.Generator = None,
# ):
#     """Compute one sparse batch loss, optionally backpropagating per-sample losses.

#     GaussianGPT Eq. (1):
#         L = lambda_rgb  * L1(renderings)         # L3DG L_RGB
#           + lambda_perc * VGG19(renderings)      # L3DG L_perc
#           + lambda_occ  * BCE(occupancy)         # L3DG L_occ
#           + lambda_lfq  * softplus(L_LFQ + 5)    # LFQ codebook entropy
#     """
#     loss_cfg = cfg.get("loss", {})
#     lambda_rgb = float(loss_cfg.get("lambda_rgb", 0.0))
#     lambda_perc = float(loss_cfg.get("lambda_perc", 0.0))
#     lambda_occ = float(loss_cfg.get("lambda_occ", 0.0))
#     lambda_lfq = float(loss_cfg.get("lambda_lfq", 0.0))

#     batch_loss = torch.tensor(0.0, device=device)
#     batch_occ = batch_lfq = 0.0
#     batch_rgb = batch_perc = 0.0
#     n = max(len(batch_list), 1)

#     for sample in batch_list:
#         voxel_coords = sample["voxel_coords"].to(device)
#         gaussians = {k: v.to(device) for k, v in sample.items()
#                      if k in ("offset", "scale", "opacity", "rotation", "color", "sh")}

#         pred_gaussians, occ_list, lfq_loss, indices = raw_model(gaussians, voxel_coords)

#         # ---- L_occ: BCE on the per-stage occupancy logits ----
#         from gaussiangpt.autoencoder.sparse_cnn import HAS_MINKOWSKI
#         l_occ = torch.tensor(0.0, device=device)
#         if occ_list:
#             for stage_idx, occ in enumerate(occ_list):
#                 if HAS_MINKOWSKI:
#                     occ_feat = occ.F  # (M, 1)
#                     targets = torch.ones(occ_feat.shape[0], device=device)
#                 else:
#                     # Dense: occ is (1, 1, X', Y', Z')
#                     scale = 2 ** (len(occ_list) - stage_idx)
#                     occ_flat = occ.view(-1)
#                     gt_grid = torch.zeros_like(occ_flat)
#                     stage_vc = (voxel_coords // scale).clamp(0, occ.shape[2] - 1)
#                     flat_idx = (
#                         stage_vc[:, 0] * occ.shape[3] * occ.shape[4]
#                         + stage_vc[:, 1] * occ.shape[4]
#                         + stage_vc[:, 2]
#                     )
#                     gt_grid.scatter_(0, flat_idx, 1.0)
#                     occ_feat = occ_flat.unsqueeze(-1)
#                     targets = gt_grid
#                 l_occ = l_occ + torch.nn.functional.binary_cross_entropy_with_logits(
#                     occ_feat.squeeze(-1), targets
#                 )
#             l_occ = l_occ / len(occ_list)

#         # ---- L_RGB / L_perc: rendering supervision (L3DG-style) ----
#         if (lambda_rgb > 0 or lambda_perc > 0) and "position" not in pred_gaussians:
#             # The decoder doesn't output absolute positions; we attach them
#             # here so the renderer can use them. The same is done in
#             # `_render_loss_for_sample` for the GT.
#             pred_gaussians["position"] = _build_world_positions(
#                 sample, pred_gaussians["offset"], cfg["data"]["base_voxel_size"], device,
#             )
#         if lambda_rgb > 0 or lambda_perc > 0:
#             l_rgb, l_perc = _render_loss_for_sample(
#                 sample, pred_gaussians, gaussians, cfg, device,
#                 perceptual=perceptual, rng=rng,
#             )
#         else:
#             l_rgb = torch.zeros((), device=device)
#             l_perc = torch.zeros((), device=device)

#         # GaussianGPT Eq. (1): the LFQ entropy term is wrapped in
#         #     λ_LFQ · softplus(L_LFQ + 5)
#         # The purpose of the offset + softplus is purely cosmetic --- it
#         # keeps the displayed loss positive (since L_LFQ ∈ [−log 2, +log 2]
#         # can dip below zero) without meaningfully changing the gradient
#         # (sigmoid(L_LFQ + 5) ≈ 1 throughout the operating range).
#         l_lfq = torch.nn.functional.softplus(lfq_loss + 5.0)

#         sample_loss = (
#             lambda_rgb * l_rgb
#             + lambda_perc * l_perc
#             + lambda_occ * l_occ
#             + lambda_lfq * l_lfq
#         ) / n

#         if backward:
#             sample_loss.backward()

#         batch_loss = batch_loss + sample_loss.detach()
#         batch_occ += l_occ.item() / n
#         # Log the actual term that enters the loss (softplus-wrapped), so
#         # the printed value matches the gradient that backprops.
#         batch_lfq += l_lfq.item() / n
#         batch_rgb += float(l_rgb.detach().item()) / n
#         batch_perc += float(l_perc.detach().item()) / n

#     return batch_loss, batch_occ, batch_lfq, batch_rgb, batch_perc

def compute_batch_loss(
    raw_model,
    batch,             # 关键：此时传入的是由 ase_sparse_collate 打包好的大字典，不再是原版的 list
    cfg: dict,
    device: torch.device,
    backward: bool = False,
    perceptual: nn.Module = None,
    rng: torch.Generator = None,
    gt_render_cache: Optional[dict] = None,
    global_step: Optional[int] = None,
    decoder_prune: bool = False,
    occupancy_threshold: float = 0.5,
    prune_min_keep: int = 1,
    log_pruning: bool = False,
    log_prefix: str = "validation",
):
    from gaussiangpt.autoencoder.sparse_cnn import HAS_MINKOWSKI

    loss_cfg = cfg.get("loss", {})
    lambda_rgb = float(loss_cfg.get("lambda_rgb", 0.0))
    lambda_perc = float(loss_cfg.get("lambda_perc", 0.0))
    lambda_occ = float(loss_cfg.get("lambda_occ", 0.0))
    lambda_lfq = float(loss_cfg.get("lambda_lfq", 0.0))

    # 1. 提取全批次拼接后的 4D 坐标与 14D 原始高斯特征
    coords = batch["coords"].to(device)       # 形状: (N_total, 4) -> [b, x, y, z]
    feats = batch["feats"].to(device)         # 形状: (N_total, 14)
    if not HAS_MINKOWSKI and coords.shape[1] == 4:
        raise RuntimeError(
            "The ASE dataloader produces batched sparse coordinates and requires "
            "MinkowskiEngine. Dense fallback only supports the legacy single-sample "
            "3D-coordinate path."
        )
    
    # 2. 核心衔接机制：将 14 维扁平特征解包切片，对齐师兄原模型 heads 期待的输入字典
    gaussians = {
        "offset": feats[:, 0:3],
        "color": feats[:, 3:6],
        "opacity": feats[:, 6:7],
        "scale": feats[:, 7:10],
        "rotation": feats[:, 10:14]
    }

    # 3. 彻底告别 for 循环！整个 Batch 放入 Sparse CNN 一把梭完成前向推理
    pred_gaussians, occ_list, lfq_loss, indices = raw_model(
        gaussians,
        coords,
        prune=decoder_prune,
        occupancy_threshold=occupancy_threshold,
        min_keep=prune_min_keep,
    )
    pred_coords = pred_gaussians.pop("_coords", None)
    if pred_coords is None:
        pred_coords = coords
    if log_pruning:
        _log_validation_pruning_counts(
            batch,
            coords,
            pred_coords,
            occ_list,
            decoder_prune,
            log_prefix,
        )

    # ---- L_occ 占位损失计算 ----
    l_occ = torch.tensor(0.0, device=device)
    occ_debug = []
    occ_stage_count = 0
    if occ_list:
        for stage_idx, occ in enumerate(occ_list):
            occ_logits, targets, stride = _sparse_occupancy_targets(
                occ, coords, stage_idx, len(occ_list), device
            )
            if occ_logits.numel() == 0:
                continue
            stage_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                occ_logits, targets
            )
            l_occ = l_occ + stage_loss
            occ_stage_count += 1
            positives = int(targets.sum().detach().item())
            total = int(targets.numel())
            occ_debug.append(
                {
                    "stage": stage_idx,
                    "stride": stride,
                    "logits": total,
                    "positives": positives,
                    "ratio": float(positives / max(total, 1)),
                    "loss": float(stage_loss.detach().item()),
                }
            )
        if occ_stage_count > 0:
            l_occ = l_occ / occ_stage_count
    occ_log_every = int(cfg.get("debug", {}).get("occ_check_every", 0))
    if occ_log_every > 0 and global_step is not None and occ_debug:
        if global_step == 1 or global_step % occ_log_every == 0:
            pieces = [
                (
                    f"s{item['stage']}:stride={item['stride']} "
                    f"logits={item['logits']} pos={item['positives']} "
                    f"ratio={item['ratio']:.3f} loss={item['loss']:.4f}"
                )
                for item in occ_debug
            ]
            print(
                f"  [occ target] step={global_step} "
                f"gt_voxels={int(coords.shape[0])} "
                + " | ".join(pieces)
            )

    # ---- L_LFQ 量化离散损失 ----
    l_lfq = torch.nn.functional.softplus(lfq_loss + 5.0)

    # ---- L_RGB / L_perc 渲染损失（前向并行，渲染串行） ----
    l_rgb = torch.tensor(0.0, device=device)
    l_perc = torch.tensor(0.0, device=device)
    
    if lambda_rgb > 0 or lambda_perc > 0:
        # 由于每个场景块有自己独立的相机视角，渲染必须通过 batch_index 拆开分别渲染
        gt_batch_indices = coords[:, 0]  # 提取第 0 列的 batch 标签
        pred_batch_indices = pred_coords[:, 0]

        render_count = 0
        for meta_idx, _meta in enumerate(batch["metas"]):
            gt_mask = gt_batch_indices == meta_idx
            pred_mask = pred_batch_indices == meta_idx
            if not bool(gt_mask.any()) or not bool(pred_mask.any()):
                continue
            render_count += 1
            
            # 剥离出当前单样本的预测值与真值
            sample_pred = {k: v[pred_mask] for k, v in pred_gaussians.items()}
            sample_gt = {k: v[gt_mask] for k, v in gaussians.items()}
            
            # 重新组装元数据供渲染器定位
            meta_sample = {
                "voxel_coords": coords[gt_mask, 1:4],  # 去掉开头的 batch 维，还原成 3D 坐标
                "pred_voxel_coords": pred_coords[pred_mask, 1:4],
                "chunk_origin": torch.tensor(batch["metas"][meta_idx]["chunk_min_voxel"], device=device),
                "metadata": batch["metas"][meta_idx],
            }
            
            # 复用原版的单视图渲染损失函数
            # 注意：传入单样本的相机或场景参数，可以从 batch["metas"][meta_idx] 中灵活读取
            single_rgb, single_perc = _render_loss_for_sample(
                meta_sample, sample_pred, sample_gt, cfg, device,
                perceptual=perceptual, rng=rng, gt_render_cache=gt_render_cache,
                global_step=global_step,
            )
            l_rgb = l_rgb + single_rgb
            l_perc = l_perc + single_perc
        if render_count > 0:
            l_rgb = l_rgb / render_count
            l_perc = l_perc / render_count

    # 4. 综合总 Loss
    total_loss = (
        lambda_rgb * l_rgb
        + lambda_perc * l_perc
        + lambda_occ * l_occ
        + lambda_lfq * l_lfq
    )

    if backward:
        total_loss.backward()

    return (
        total_loss, 
        l_occ.item(), 
        l_lfq.item(), 
        float(l_rgb.detach().item()), 
        float(l_perc.detach().item())
    )


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
    meta = batch["metas"][sample_index]
    return {
        "voxel_coords": coords[mask, 1:4],
        "chunk_origin": torch.as_tensor(meta["chunk_min_voxel"], dtype=torch.long),
        "scene_origin": meta.get("scene_origin"),
        "offset": sample_feats[:, 0:3],
        "color": sample_feats[:, 3:6],
        "opacity": sample_feats[:, 6:7],
        "scale": sample_feats[:, 7:10],
        "rotation": sample_feats[:, 10:14],
        "metadata": meta,
    }

def save_gaussians_as_ply(gaussians: dict, path: str):
    """Write reconstructed Gaussians as a 3DGS-compatible Binary PLY file.

    The autoencoder decoder already returns attributes in their natural
    representation (see gaussian_heads.GaussianAttributeDecoder._postprocess):
      * color   ∈ (0, 1)    -- linear RGB
      * opacity ∈ [-10, 10] -- already in logit space (3DGS stores it this way)
      * scale   ∈ R+        -- linear positive scale (3DGS stores log-scale)
      * rotation            -- unit quaternion
    """
    positions = gaussians["position"].detach().cpu().numpy()

    n_pts = positions.shape[0]

    # 1) Colour / SH -> 3DGS f_dc (degree 0) + f_rest (higher orders).
    #    3DGS rendering convention: rendered_dc = f_dc * C0 + 0.5,  C0 = 1/(2*sqrt(pi)).
    #    Loader (data._load_ply_scene) stores SH as [f_dc(3), f_rest[:9]] = 12 dims,
    #    so we round-trip the same layout here when "sh" is present.
    C0 = 0.28209479177387814
    f_rest = np.zeros((n_pts, 45), dtype=np.float32)
    if "sh" in gaussians and gaussians["sh"] is not None:
        sh = gaussians["sh"].detach().cpu().numpy()
        f_dc = sh[:, :3].astype(np.float32, copy=False)
        n_rest = min(sh.shape[1] - 3, 45)
        if n_rest > 0:
            f_rest[:, :n_rest] = sh[:, 3:3 + n_rest]
    else:
        colors = gaussians.get("color", torch.full((n_pts, 3), 0.5))
        colors = colors.detach().cpu().clamp(0.001, 0.999)
        f_dc = ((colors - 0.5) / C0).numpy()

    # 2) Opacity is ALREADY a logit out of the decoder; the 3DGS PLY field
    #    is also a logit (renderer applies sigmoid). Just write it through.
    #    The previous code re-applied `logit(...)`, which treats the stored
    #    logit as a probability and produced near-±inf values (so the
    #    reconstructed scene rendered as either fully transparent or fully
    #    opaque garbage).
    opacity = gaussians.get(
        "opacity", torch.full((n_pts, 1), 2.2)  # ~sigmoid(2.2)≈0.9
    ).detach().cpu().clamp(-10.0, 10.0).numpy()

    # 3) Scale -> log-scale (3DGS PLY stores log-space scale).
    scale = gaussians.get("scale", torch.ones(n_pts, 3)).detach().cpu().clamp_min(1e-8)
    log_scale = torch.log(scale).numpy()

    # 4) Rotation -> unit quaternion (w, x, y, z); 3DGS renderer normalises again at use.
    if "rotation" in gaussians and gaussians["rotation"].numel() > 0:
        rotation = torch.nn.functional.normalize(
            gaussians["rotation"].detach().cpu(), dim=-1
        ).numpy()
    else:
        # Fallback: identity quaternion (w=1) so the renderer doesn't blow up.
        rotation = np.zeros((n_pts, 4), dtype=np.float32)
        rotation[:, 0] = 1.0

    # 法线补零
    normals = np.zeros_like(positions)

    # 6. 构建 NumPy 结构化数组 (定义各个属性的数据类型)
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    
    for i in range(3):
        dtype_full.append((f'f_dc_{i}', 'f4'))
    for i in range(45):
        dtype_full.append((f'f_rest_{i}', 'f4'))
        
    dtype_full.append(('opacity', 'f4'))
    
    for i in range(3):
        dtype_full.append((f'scale_{i}', 'f4'))
    for i in range(4):
        dtype_full.append((f'rot_{i}', 'f4'))

    elements = np.empty(positions.shape[0], dtype=dtype_full)
    
    # 7. 填入数据
    elements['x'] = positions[:, 0]
    elements['y'] = positions[:, 1]
    elements['z'] = positions[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    
    for i in range(3):
        elements[f'f_dc_{i}'] = f_dc[:, i]
    for i in range(45):
        elements[f'f_rest_{i}'] = f_rest[:, i]
        
    elements['opacity'] = opacity[:, 0]
    
    for i in range(3):
        elements[f'scale_{i}'] = log_scale[:, i]
    for i in range(4):
        elements[f'rot_{i}'] = rotation[:, i]

    # 8. 保存为 Binary Little Endian PLY
    os.makedirs(os.path.dirname(path), exist_ok=True)
    el = PlyElement.describe(elements, 'vertex')
    # text=False 保证输出的是 binary 格式
    PlyData([el], text=False).write(path)

def save_validation_reconstruction(
    raw_model,
    sample: dict,
    cfg: dict,
    device: torch.device,
    ply_path: str,
    image_path: Optional[str] = None,
    prune: bool = False,
    occupancy_threshold: float = 0.5,
    prune_min_keep: int = 1,
):
    """Save a validation reconstruction.

    Always writes the predicted Gaussians as a 3DGS-compatible .ply at
    `ply_path`. If `image_path` is given, additionally renders N views of
    GT and predicted Gaussians and saves them as a 2-row PNG (top row =
    GT, bottom row = predicted) for quick eyeballing.
    """
    voxel_coords = sample["voxel_coords"].to(device)
    gaussians = {k: v.to(device) for k, v in sample.items()
                 if k in ("offset", "scale", "opacity", "rotation", "color", "sh")}

    pred_gaussians, _, _, _ = raw_model(
        gaussians,
        voxel_coords,
        prune=prune,
        occupancy_threshold=occupancy_threshold,
        min_keep=prune_min_keep,
    )
    pred_coords = pred_gaussians.pop("_coords", None)
    if pred_coords is not None:
        pred_voxel_coords = pred_coords[:, 1:] if pred_coords.shape[1] == 4 else pred_coords
    else:
        pred_voxel_coords = voxel_coords
    base_voxel_size = float(cfg["data"]["base_voxel_size"])
    pred_sample = dict(sample)
    pred_sample["voxel_coords"] = pred_voxel_coords
    pred_gaussians["position"] = _build_world_positions(
        pred_sample, pred_gaussians["offset"], base_voxel_size, device
    )
    save_gaussians_as_ply(pred_gaussians, ply_path)

    # ---- Optional: render GT vs. Pred views and save as a side-by-side PNG ----
    if image_path is None or not HAS_RASTERIZER or device.type != "cuda":
        return

    loss_cfg = cfg.get("loss", {})
    camera_cfg, camera_mode = _camera_sampling_config(cfg)
    n_views = _effective_render_view_count(cfg, loss_cfg)
    img_size = int(loss_cfg.get("render_size", 128))
    if n_views <= 0:
        return

    gt_position = _build_world_positions(
        sample, gaussians["offset"], base_voxel_size, device
    )
    bbox_min = gt_position.min(dim=0).values
    bbox_max = gt_position.max(dim=0).values
    cameras = []
    if camera_mode == "scoring":
        cameras, _camera_debug = _sample_scored_cameras(
            sample=sample,
            cfg=cfg,
            n_views=n_views,
            image_size=img_size,
            fallback_fov_deg=float(loss_cfg.get("fov_deg", 60.0)),
            device=device,
            rng=None,
            global_step=None,
        )
    if not cameras:
        # Validation orbit views are deterministic (no jitter) so renderings
        # are directly comparable across val runs.
        cameras = sample_cameras_around_bbox(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            n_views=n_views,
            image_height=img_size,
            image_width=img_size,
            fov_deg=float(loss_cfg.get("fov_deg", 60.0)),
            radius_factor=float(loss_cfg.get("radius_factor", 1.5)),
            upper_hemisphere_only=bool(loss_cfg.get("upper_hemisphere_only", False)),
            jitter=0.0,
        )

    bg = torch.zeros(3, device=device)
    gt_pack = {
        "position": gt_position,
        "scale": gaussians["scale"],
        "rotation": gaussians["rotation"],
        "opacity": gaussians["opacity"],
        "color": gaussians["color"],
    }
    pred_pack = {
        "position": pred_gaussians["position"],
        "scale": pred_gaussians["scale"],
        "rotation": pred_gaussians["rotation"],
        "opacity": pred_gaussians["opacity"],
        "color": pred_gaussians["color"],
    }

    gt_imgs, pred_imgs = [], []
    for cam in cameras:
        gt_imgs.append(render_gaussians(gt_pack, cam, bg_color=bg))
        pred_imgs.append(render_gaussians(pred_pack, cam, bg_color=bg))

    # 2-row grid: first row GT, second row Pred (each row has n_views images).
    from torchvision.utils import make_grid, save_image
    stacked = torch.stack(gt_imgs + pred_imgs, dim=0)  # (2N, 3, H, W)
    grid = make_grid(stacked, nrow=n_views, pad_value=1.0)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    save_image(grid, image_path)


def validate(
    raw_model,
    val_loader,
    cfg: dict,
    device: torch.device,
    epoch: int,
    global_step: int,
    output_dir: str,
    perceptual: nn.Module = None,
    gt_render_cache: Optional[dict] = None,
):
    """Run validation over the full validation loader and print average losses."""
    if len(val_loader) == 0:
        print(f"Validation skipped at step {global_step}: val_loader is empty")
        return None

    raw_model.eval()
    total_loss = total_occ = total_lfq = 0.0
    total_rgb = total_perc = 0.0
    val_prune_cfg = _validation_pruning_config(cfg)
    val_prune = bool(val_prune_cfg["prune"])
    val_occ_threshold = float(val_prune_cfg["occ_threshold"])
    val_prune_min_keep = int(val_prune_cfg["prune_min_keep"])
    prediction_mode = "pruned" if val_prune else "unpruned"
    print(
        f"[validation pruning] enabled={val_prune} "
        f"occ_threshold={val_occ_threshold} prune_min_keep={val_prune_min_keep} "
        f"render_prediction={prediction_mode} saved_reconstruction={prediction_mode}"
    )
    recon_tag = f"epoch_{epoch:04d}_step_{global_step:08d}"
    recon_path = os.path.join(output_dir, "val_reconstructions", f"{recon_tag}.ply")
    image_path = os.path.join(output_dir, "val_renderings", f"{recon_tag}.png")
    saved_reconstruction = False
    # Validation uses a deterministic camera-jitter RNG so reconstruction
    # quality is comparable across val runs. The generator must live on
    # the same device as the tensors it samples (camera jitter is CUDA
    # when rendering happens on GPU).
    val_rng = torch.Generator(device=device)
    val_rng.manual_seed(int(global_step))
    with torch.no_grad():
        for val_batch_idx, batch in enumerate(val_loader):
            (batch_loss, batch_occ, batch_lfq,
             batch_rgb, batch_perc) = compute_batch_loss(
                raw_model, batch, cfg, device, backward=False,
                perceptual=perceptual, rng=val_rng,
                gt_render_cache=gt_render_cache,
                global_step=global_step,
                decoder_prune=val_prune,
                occupancy_threshold=val_occ_threshold,
                prune_min_keep=val_prune_min_keep,
                log_pruning=True,
                log_prefix=f"validation batch={val_batch_idx}",
            )
            total_loss += batch_loss.item()
            total_occ += batch_occ
            total_lfq += batch_lfq
            total_rgb += batch_rgb
            total_perc += batch_perc
            if not saved_reconstruction and batch["coords"].numel() > 0:
                save_validation_reconstruction(
                    raw_model, ase_batch_sample_to_legacy_sample(batch, 0), cfg, device,
                    ply_path=recon_path, image_path=image_path,
                    prune=val_prune,
                    occupancy_threshold=val_occ_threshold,
                    prune_min_keep=val_prune_min_keep,
                )
                saved_reconstruction = True

    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    print(
        f"Validation Epoch {epoch} Step {global_step} "
        f"Loss: {avg_loss:.4f} "
        f"rgb: {total_rgb / n_batches:.4f} perc: {total_perc / n_batches:.4f} "
        f"occ: {total_occ / n_batches:.4f} lfq: {total_lfq / n_batches:.4f}"
    )
    if saved_reconstruction:
        print(f"Saved validation reconstruction ({prediction_mode}): {recon_path}")
        if (
            HAS_RASTERIZER
            and device.type == "cuda"
            and _effective_render_view_count(cfg, cfg.get("loss", {})) > 0
        ):
            print(f"Saved validation renderings ({prediction_mode}):     {image_path}")
    return avg_loss


def train(cfg: dict, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPU(s), device: {device}")

    # ---- Debug toggles (all OFF by default = paper-faithful) ----
    # See README / configs/autoencoder_scene.yaml for the full list.
    debug_cfg = cfg.get("debug", {}) or {}
    data_debug_cfg = cfg.get("data", {}) or {}
    norm_kind = str(debug_cfg.get("norm", "bn")).lower()
    color_act = str(debug_cfg.get("color_activation", "clamp")).lower()
    fixed_chunk = bool(data_debug_cfg.get("fixed_chunk", debug_cfg.get("fixed_chunk", False)))
    no_augment = bool(debug_cfg.get("no_augment", False))
    voxel_dedup = str(debug_cfg.get("voxel_dedup", "random")).lower()
    grad_clip = float(debug_cfg.get("grad_clip", 1.0))
    log_grad_norm_every = int(debug_cfg.get("log_grad_norm_every", 0))
    if any([
        norm_kind != "bn",
        color_act != "clamp",
        fixed_chunk,
        no_augment,
        voxel_dedup != "random",
        grad_clip != 1.0,
        log_grad_norm_every > 0,
    ]):
        print(
            "[debug] norm=" + norm_kind
            + f" color_activation={color_act}"
            + f" fixed_chunk={fixed_chunk} no_augment={no_augment}"
            + f" voxel_dedup={voxel_dedup} grad_clip={grad_clip}"
            + f" log_grad_norm_every={log_grad_norm_every}"
        )

    # Model
    model = GaussianAutoencoder(
        base_ch=cfg["model"]["base_ch"],
        n_down=cfg["model"]["n_down"],
        codebook_size=cfg["model"]["codebook_size"],
        use_sh=cfg["model"].get("use_sh", False),
        voxel_size=cfg["data"]["base_voxel_size"],
        norm=norm_kind,
        color_activation=color_act,
    ).to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)

    # Dataset
    from gaussiangpt.autoencoder.data_preprocess.dataset import ASEChunkDataset
    from gaussiangpt.autoencoder.data_preprocess.collate import ase_sparse_collate
    data_cfg = cfg["data"]
    cache_root = args.cache_root or data_cfg.get("cache_root") or args.data_dir
    if not cache_root:
        raise ValueError(
            "ASE dataloader requires data.cache_root or --cache_root "
            "(--data_dir is also accepted as a compatibility fallback)."
        )
    base_voxel_size = float(data_cfg["base_voxel_size"])
    chunk_size_voxels = tuple(data_cfg.get("chunk_size", [160, 160, 160]))
    chunk_size_world = float(
        data_cfg.get("ase_chunk_size", chunk_size_voxels[0] * base_voxel_size)
    )
    train_samples_per_epoch = int(data_cfg.get("train_samples_per_epoch", 1000))
    val_samples_per_epoch = int(data_cfg.get("val_samples_per_epoch", 200))
    occupancy_threshold = float(data_cfg.get("min_occupancy_ae", 0.2))
    max_candidate_chunks = int(data_cfg.get("max_candidate_chunks", 10))
    top_k_cameras = int(data_cfg.get("top_k_cameras", 12))
    z_mode = str(data_cfg.get("z_mode", "fixed_160"))
    preferred_coverage = float(data_cfg.get("preferred_coverage", 0.4))
    train_scene_ids = data_cfg.get("train_scene_ids")
    val_scene_ids = data_cfg.get("val_scene_ids")
    loss_cfg = cfg.get("loss", {})
    camera_cfg, camera_mode = _camera_sampling_config(cfg)
    _validate_camera_sampling_config(cfg)
    camera_dataset_type = _camera_dataset_type(cfg) if camera_mode == "scoring" else "n/a"
    render_n_views = _effective_render_view_count(cfg, loss_cfg)
    camera_num_candidates = _camera_candidate_count(cfg, render_n_views)
    include_camera_matrices = camera_mode == "scoring"
    sampler_top_k_cameras = top_k_cameras
    if camera_mode == "scoring":
        if camera_dataset_type == "ase":
            sampler_top_k_cameras = int(camera_num_candidates or 0)
        else:
            sampler_top_k_cameras = max(int(camera_num_candidates), int(render_n_views))
    candidate_label = "all" if sampler_top_k_cameras <= 0 else str(sampler_top_k_cameras)
    print(
        "[camera sampling] "
        f"mode={camera_mode} dataset_type={camera_dataset_type} "
        f"render_views={render_n_views} "
        f"candidate_cameras={candidate_label} "
        f"score_key={_camera_score_key(cfg)} "
        f"temperature={camera_cfg.get('temperature', 1.0)} "
        f"selection=score_only"
    )

    train_dataset = ASEChunkDataset(
        cache_root=cache_root,
        num_samples_per_epoch=train_samples_per_epoch,
        chunk_size=chunk_size_world,
        occupancy_threshold=occupancy_threshold,
        max_candidate_chunks=max_candidate_chunks,
        top_k_cameras=sampler_top_k_cameras,
        seed=int(data_cfg.get("seed", 42)),
        z_mode=z_mode,
        preferred_coverage=preferred_coverage,
        include_camera_matrices=include_camera_matrices,
        scene_ids=train_scene_ids,
        fixed_chunk=fixed_chunk,
    )
    fixed_sample = train_dataset.fixed_sample() if fixed_chunk else None
    
    # 替换验证集实例化（可以共享同一个类，指定不同的场景 id 列表）
    val_dataset = ASEChunkDataset(
        cache_root=cache_root,
        num_samples_per_epoch=val_samples_per_epoch,
        chunk_size=chunk_size_world,
        occupancy_threshold=occupancy_threshold,
        max_candidate_chunks=max_candidate_chunks,
        top_k_cameras=sampler_top_k_cameras,
        seed=int(data_cfg.get("val_seed", 4242)),
        z_mode=z_mode,
        preferred_coverage=preferred_coverage,
        include_camera_matrices=include_camera_matrices,
        scene_ids=val_scene_ids,
        fixed_chunk=fixed_chunk,
        fixed_sample=fixed_sample,
    )
    if fixed_chunk:
        print("[fixed_chunk] enabled")
        summary = train_dataset.fixed_chunk_summary()
        if summary is not None:
            print(f"[fixed_chunk] using {summary}")
        print("[fixed_chunk] validation dataset reuses the training fixed chunk")
    else:
        print("[fixed_chunk] disabled; using online chunk sampling")

    batch_size = args.batch_size or cfg["training"]["batch_size"]
    
    # 关键：将 collate_fn 替换为你写的真正的并行化组装函数 ase_sparse_collate
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last=True,
        collate_fn=ase_sparse_collate,  # 替换这里！
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=int(data_cfg.get("val_num_workers", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        collate_fn=ase_sparse_collate,  # 替换这里！
    )
    # data_dir = args.data_dir or cfg["data"]["data_dir"]
    # train_dataset = GaussianSceneDataset(
    #     data_dir=data_dir,
    #     base_voxel_size=cfg["data"]["base_voxel_size"],
    #     n_down=cfg["model"]["n_down"],
    #     chunk_size=tuple(cfg["data"]["chunk_size"]),
    #     min_occupancy=cfg["data"].get("min_occupancy_ae", 0.2),
    #     augment=(not no_augment),
    #     split="train",
    #     fixed_chunk=fixed_chunk,
    #     voxel_dedup=voxel_dedup,
    # )
    # val_dataset = GaussianSceneDataset(
    #     data_dir=data_dir,
    #     base_voxel_size=cfg["data"]["base_voxel_size"],
    #     n_down=cfg["model"]["n_down"],
    #     chunk_size=tuple(cfg["data"]["chunk_size"]),
    #     min_occupancy=cfg["data"].get("min_occupancy_ae", 0.2),
    #     augment=False,
    #     split="val",
    #     # Validation always uses deterministic chunks; voxel dedup tracks
    #     # the train setting so the GT representation stays consistent.
    #     fixed_chunk=fixed_chunk,
    #     voxel_dedup=voxel_dedup,
    # )

    # batch_size = args.batch_size or cfg["training"]["batch_size"]
    # train_loader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True,
    #     num_workers=4, pin_memory=False, drop_last=True,
    #     collate_fn=sparse_collate,
    # )
    # val_loader = DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=False,
    #     num_workers=2, pin_memory=False,
    #     collate_fn=sparse_collate,
    # )

    # Optimizer
    lr = args.lr or cfg["training"]["lr"]
    optimizer = Adam(model.parameters(), lr=lr)
    epochs = args.epochs or cfg["training"]["epochs"]
    val_every_steps = args.val_every_steps
    if val_every_steps is None:
        val_every_steps = cfg["training"].get("val_every_steps", 0)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    # Resume: 从断点继续训练
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(args.output_dir, exist_ok=True)
    config_save_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}")

    # Note: MinkowskiEngine is incompatible with DataParallel's scatter mechanism.
    # Use the unwrapped model for forward; DataParallel is only safe for dense fallback.
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    global_step = 0

    # ---- Per-head decoder diagnostics ----
    # Multi-line readout per `color_check_every` steps:
    #   * colour head pre-activation distribution (clamp/sigmoid aware)
    #   * opacity distribution + decisive fraction (alpha near 0 or 1) --
    #     a low decisive fraction is the classic "soft averaging"
    #     failure mode that turns vivid colours into grey mush.
    #   * scale distribution vs voxel size -- "ballooning" Gaussians
    #     also cause desaturation and detail loss.
    # Set `diagnostics.color_check_every <= 0` in the config to disable.
    diag_cfg = cfg.get("diagnostics", {})
    color_diag = ColorClampDiagnostics(
        raw_model.attr_decoder,
        every=int(diag_cfg.get("color_check_every", 200)),
        voxel_size=float(cfg["data"]["base_voxel_size"]),
    )

    # ---- Optional rendering / perceptual supervision ----
    # Lazily build the VGG perceptual loss only when actually requested,
    # so that disabling it (lambda_perc=0 or n_images=0) avoids loading
    # ~80MB of weights and an extra GPU-side module.
    use_render = (
        float(loss_cfg.get("lambda_rgb", 0.0)) > 0.0
        or float(loss_cfg.get("lambda_perc", 0.0)) > 0.0
    ) and render_n_views > 0
    perceptual = None
    if use_render:
        if not HAS_RASTERIZER:
            print("WARNING: lambda_rgb/perc > 0 but diff-gaussian-rasterization "
                  "is not importable; rendering losses will be skipped.")
        elif device.type != "cuda":
            print("WARNING: rendering losses require CUDA; skipping on CPU.")
        else:
            print(
                f"Rendering supervision enabled: "
                f"n_views={render_n_views} "
                f"img={loss_cfg.get('render_size', 128)}^2 "
                f"lambda_rgb={loss_cfg.get('lambda_rgb', 0.0)} "
                f"lambda_perc={loss_cfg.get('lambda_perc', 0.0)}"
            )
            if float(loss_cfg.get("lambda_perc", 0.0)) > 0:
                from gaussiangpt.utils.perceptual import VGGPerceptualLoss
                perceptual = VGGPerceptualLoss(
                    device=device,
                    weights_path=loss_cfg.get("vgg19_weights_path"),
                )

    cache_fixed_gt_render = (
        fixed_chunk
        and use_render
        and HAS_RASTERIZER
        and device.type == "cuda"
        and (
            camera_mode == "scoring"
            or float(loss_cfg.get("camera_jitter", 0.0)) == 0.0
        )
        and bool(loss_cfg.get("cache_fixed_gt_render", True))
    )
    gt_render_cache = {} if cache_fixed_gt_render else None
    if cache_fixed_gt_render:
        print(f"[fixed_chunk] GT render cache enabled (camera_sampling={camera_mode})")
    elif fixed_chunk and use_render:
        print("[fixed_chunk] GT render cache disabled")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for step, batch_list in enumerate(train_loader):
            # batch_list is a list of per-sample dicts (sparse_collate)
            # Accumulate gradients over the batch manually
            optimizer.zero_grad()
            (batch_loss, batch_occ, batch_lfq,
             batch_rgb, batch_perc) = compute_batch_loss(
                raw_model, batch_list, cfg, device, backward=True,
                perceptual=perceptual, gt_render_cache=gt_render_cache,
                global_step=global_step + 1,
            )

            # ``clip_grad_norm_`` returns the *pre-clip* total gradient
            # norm; logging it occasionally is one of the cheapest ways
            # to spot the "everything is being clipped" pathology.
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip,
            )
            optimizer.step()
            global_step += 1

            total_loss += batch_loss.item()
            if step % 100 == 0:
            # if True:
                print(
                    f"Epoch {epoch} Step {step}/{len(train_loader)} "
                    f"Loss: {batch_loss.item():.4f} "
                    f"rgb: {batch_rgb:.4f} perc: {batch_perc:.4f} "
                    f"occ: {batch_occ:.4f} lfq: {batch_lfq:.4f}"
                )
            if log_grad_norm_every > 0 and global_step % log_grad_norm_every == 0:
                clipped = float(pre_clip_norm) > grad_clip
                print(
                    f"  [grad] step={global_step} "
                    f"pre_clip_norm={float(pre_clip_norm):.3f} "
                    f"clip={grad_clip:.2f} "
                    f"{'(clipped)' if clipped else ''}"
                )

            # Colour-clamp dead-gradient check. Reads .grad (still alive
            # after optimizer.step), then drops the captured features so
            # the next step doesn't see stale data.
            color_diag.maybe_log(global_step)
            color_diag.clear()

            if val_every_steps > 0 and global_step % val_every_steps == 0:
            # if True:
                validate(
                    raw_model, val_loader, cfg, device, epoch,
                    global_step, args.output_dir, perceptual=perceptual,
                    gt_render_cache=gt_render_cache,
                )
                model.train()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f} lr: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (epoch + 1) % cfg["training"].get("save_every", 10) == 0:
        # if True:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final
    torch.save({
        "epoch": epochs - 1,
        "model": model.state_dict(),
        "config": cfg,
    }, os.path.join(args.output_dir, "final.pt"))
    color_diag.close()
    print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, args)
